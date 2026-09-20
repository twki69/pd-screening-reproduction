"""Model comparison statistics.

The review objected that a Wilcoxon test on five outer-fold values has
negligible power and that fold estimates are not independent, so a
non-significant p-value cannot be read as evidence of equivalence. This
module therefore reports, for every model pair:

* the paired difference in subject-level AUC across all repeated outer folds,
* the matched-pairs rank-biserial correlation as an effect size,
* a bias-corrected bootstrap confidence interval for the mean difference,
  resampled at the level of repeats to respect the nesting,
* Holm-adjusted p-values across all pairwise comparisons.

None of these establish equivalence on their own; the confidence interval
is what licenses a statement about how large a difference the data can
still hide.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats


def rank_biserial(diff: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation for the Wilcoxon test."""
    d = diff[diff != 0]
    if d.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    r_plus = ranks[d > 0].sum()
    r_minus = ranks[d < 0].sum()
    total = ranks.sum()
    return float((r_plus - r_minus) / total)


def cluster_bootstrap_ci(df: pd.DataFrame, col: str, *, n_boot: int = 10000,
                         alpha: float = 0.05, seed: int = 0) -> tuple:
    """Percentile CI for a mean difference, resampling whole repeats.

    Outer folds within one repeat partition the same subjects and are not
    independent, so the resampling unit is the repeat, not the fold.
    """
    rng = np.random.default_rng(seed)
    repeats = df["repeat"].unique()
    by_repeat = {r: df.loc[df["repeat"] == r, col].to_numpy() for r in repeats}
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(repeats, size=repeats.size, replace=True)
        means[b] = np.concatenate([by_repeat[r] for r in pick]).mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def holm(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = p.size
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def summarise_models(fold_scores: pd.DataFrame,
                     metric: str = "auc_subject") -> pd.DataFrame:
    """Mean, SD and repeat-level bootstrap CI of each model's score."""
    rows = []
    for name, g in fold_scores.groupby("model"):
        lo, hi = cluster_bootstrap_ci(g, metric)
        per_repeat = g.groupby("repeat")[metric].mean()
        rows.append({
            "model": name,
            "mean": g[metric].mean(),
            "sd_fold": g[metric].std(ddof=1),
            "sd_repeat": per_repeat.std(ddof=1),
            "ci_lo": lo, "ci_hi": hi,
            "min_fold": g[metric].min(), "max_fold": g[metric].max(),
            "n_folds": int(g[metric].notna().sum()),
        })
    return (pd.DataFrame(rows).sort_values("mean", ascending=False)
            .reset_index(drop=True))


def pairwise_comparisons(fold_scores: pd.DataFrame,
                         metric: str = "auc_subject",
                         seed: int = 0) -> pd.DataFrame:
    """All pairwise paired comparisons with effect size, CI and Holm p."""
    wide = fold_scores.pivot_table(index=["repeat", "fold"], columns="model",
                                   values=metric).dropna()
    models = sorted(wide.columns)
    rows = []
    for a, b in itertools.combinations(models, 2):
        diff = (wide[a] - wide[b]).to_numpy()
        if np.allclose(diff, 0):
            p = 1.0
        else:
            p = stats.wilcoxon(wide[a], wide[b],
                               zero_method="wilcox").pvalue
        d = pd.DataFrame({"repeat": wide.index.get_level_values("repeat"),
                          "diff": diff})
        lo, hi = cluster_bootstrap_ci(d, "diff", seed=seed)
        rows.append({"model_a": a, "model_b": b,
                     "mean_diff": diff.mean(),
                     "ci_lo": lo, "ci_hi": hi,
                     "rank_biserial": rank_biserial(diff),
                     "p_raw": p, "n_pairs": diff.size})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_raw"].to_numpy())
    return out.sort_values("p_holm").reset_index(drop=True)


def equivalence_statement(row: pd.Series, margin: float = 0.05) -> str:
    """Plain-language reading of one comparison against a stated margin."""
    lo, hi = row["ci_lo"], row["ci_hi"]
    if lo > 0 or hi < 0:
        return "difference detected"
    if abs(lo) < margin and abs(hi) < margin:
        return f"practically equivalent within +/-{margin:g} AUC"
    return "inconclusive: CI admits a difference larger than the margin"
