"""Out-of-fold explainability aggregated across all outer folds.

The review objected that the previous explanations were computed on a single
seven-subject split on which the classifier predicted one class for every
recording, and that explanations from a degenerate classifier cannot support
claims about biomarkers.

This module instead recomputes explanations on every held-out outer fold of
the repeated grouped nested design, using the hyperparameters that fold
actually selected, and reports the spread across folds rather than a single
ranking. Two stability measures accompany every mean importance:

* a percentile interval across folds, and
* selection frequency -- the proportion of folds in which the feature lands
  in the top five.

Sign convention for permutation importance: values are the DROP in
subject-level AUC caused by shuffling the feature, so a positive value means
the feature helped and a NEGATIVE value means the model scored better without
it, which for a small fold is evidence of noise rather than of a protective
effect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from .cv import subject_aggregate
from .data import Cohort


_INT_PARAMS = ("depth", "leaf", "neighbors", "estimators", "leaves",
               "child_samples", "iterations")


def _coerce(name: str, value):
    """Restore integer hyperparameters lost to CSV round-tripping."""
    if isinstance(value, float) and any(k in name for k in _INT_PARAMS):
        if float(value).is_integer():
            return int(value)
    return value


def _subject_auc(est, X, y, groups) -> float:
    prob = est.predict_proba(X)[:, 1]
    sub = subject_aggregate(prob, groups, y)
    if sub["y"].nunique() < 2:
        return np.nan
    return roc_auc_score(sub["y"], sub["prob"])


def permutation_importance_subject(est, X, y, groups, *, n_repeats: int = 10,
                                   seed: int = 0) -> np.ndarray:
    """AUC drop per feature, scored after subject-level aggregation."""
    rng = np.random.default_rng(seed)
    base = _subject_auc(est, X, y, groups)
    drops = np.zeros(X.shape[1])
    if not np.isfinite(base):
        return np.full(X.shape[1], np.nan)
    for j in range(X.shape[1]):
        vals = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp.iloc[:, j] = rng.permutation(Xp.iloc[:, j].to_numpy())
            vals.append(base - _subject_auc(est, Xp, y, groups))
        drops[j] = np.mean(vals)
    return drops


def _shap_values(est, X_train, X_eval) -> np.ndarray | None:
    """Mean |SHAP| per feature, or None when no tree explainer applies."""
    import shap
    clf = est.named_steps["clf"]
    scaler = est.named_steps["scaler"]
    Xs = pd.DataFrame(scaler.transform(X_eval), columns=X_eval.columns)
    try:
        expl = shap.TreeExplainer(clf)
        vals = expl.shap_values(Xs)
    except Exception:
        return None
    if isinstance(vals, list):
        vals = vals[-1]
    vals = np.asarray(vals)
    if vals.ndim == 3:
        vals = vals[:, :, -1]
    return np.abs(vals).mean(axis=0)


def run_explanations(cohort: Cohort, models: dict, best_params: pd.DataFrame,
                     *, model_names: list[str], n_repeats: int = 5,
                     n_outer: int = 5, base_seed: int = 20260917,
                     perm_repeats: int = 5) -> pd.DataFrame:
    """Recompute explanations on every held-out outer fold."""
    X, y, groups = cohort.X, cohort.y, cohort.groups
    rows = []
    for rep in range(n_repeats):
        seed = base_seed + rep
        outer = StratifiedGroupKFold(n_splits=n_outer, shuffle=True,
                                     random_state=seed)
        for fold, (tr, te) in enumerate(outer.split(X, y, groups)):
            for name in model_names:
                pipe, _ = models[name]
                sel = best_params[(best_params.model == name)
                                  & (best_params["repeat"] == rep)
                                  & (best_params.fold == fold)]
                params = {c: _coerce(c, sel.iloc[0][c])
                          for c in sel.columns
                          if c.startswith("clf__")
                          and pd.notna(sel.iloc[0][c])}
                est = clone(pipe).set_params(**params)
                est.fit(X.iloc[tr], y[tr])

                auc = _subject_auc(est, X.iloc[te], y[te], groups[te])
                pred = est.predict(X.iloc[te])
                degenerate = len(np.unique(pred)) == 1

                perm = permutation_importance_subject(
                    est, X.iloc[te], y[te], groups[te],
                    n_repeats=perm_repeats, seed=seed + fold)
                shap_mean = _shap_values(est, X.iloc[tr], X.iloc[te])

                for j, feat in enumerate(cohort.features):
                    rows.append({
                        "model": name, "repeat": rep, "fold": fold,
                        "feature": feat,
                        "perm_auc_drop": perm[j],
                        "mean_abs_shap": (np.nan if shap_mean is None
                                          else shap_mean[j]),
                        "fold_auc": auc,
                        "fold_degenerate": degenerate,
                    })
    return pd.DataFrame(rows)


def stability_table(expl: pd.DataFrame, model: str,
                    value: str = "mean_abs_shap", top_k: int = 5) -> pd.DataFrame:
    """Mean importance, fold interval, and top-k selection frequency."""
    d = expl[(expl.model == model) & expl[value].notna()].copy()
    if d.empty:
        return pd.DataFrame()
    d["rank"] = (d.groupby(["repeat", "fold"])[value]
                 .rank(ascending=False, method="min"))
    n_folds = d.groupby(["repeat", "fold"]).ngroups
    g = d.groupby("feature")[value]
    out = pd.DataFrame({
        "mean": g.mean(),
        "sd": g.std(ddof=1),
        "p2.5": g.quantile(0.025),
        "p97.5": g.quantile(0.975),
        "median_rank": d.groupby("feature")["rank"].median(),
        "top_k_frequency": (d[d["rank"] <= top_k].groupby("feature").size()
                            / n_folds).reindex(g.mean().index).fillna(0.0),
    })
    return out.sort_values("mean", ascending=False)
