"""Repeated stratified group nested cross-validation with subject-level scoring.

Design decisions that answer the review:

* Grouping unit is the subject; no subject contributes recordings to both
  the training and the evaluation side of any split, in either loop.
* The outer loop is repeated with different seeds, so split-to-split
  variability is measured rather than assumed.
* Predictions are aggregated to one probability per subject before any
  metric is computed, so participants contributing more recordings cannot
  be over-weighted. Recording-level scores are retained separately as a
  supplementary view.
* Fold composition (PD / control subject counts) is recorded for every
  outer and inner split.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

from .splits import DeterministicStratifiedGroupKFold as StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

from .data import Cohort


@dataclass
class CVResult:
    fold_scores: pd.DataFrame          # one row per (model, repeat, fold)
    oof_subject: pd.DataFrame          # one row per (model, repeat, subject)
    oof_recording: pd.DataFrame        # one row per (model, repeat, recording)
    fold_composition: pd.DataFrame     # subject counts per split
    best_params: pd.DataFrame          # selected hyperparameters per fold
    meta: dict = field(default_factory=dict)


def subject_aggregate(probs: np.ndarray, groups: np.ndarray,
                      y: np.ndarray) -> pd.DataFrame:
    """Mean predicted probability per subject, with that subject's label."""
    df = pd.DataFrame({"subject": groups, "prob": probs, "y": y})
    out = df.groupby("subject").agg(prob=("prob", "mean"),
                                    y=("y", "first"),
                                    n_recordings=("prob", "size"))
    return out.reset_index()


def _composition(y: np.ndarray, groups: np.ndarray, idx: np.ndarray) -> dict:
    sub = pd.DataFrame({"g": groups[idx], "y": y[idx]}).groupby("g")["y"].first()
    return {"n_subjects": int(sub.size),
            "n_pd_subjects": int((sub == 1).sum()),
            "n_hc_subjects": int((sub == 0).sum()),
            "n_recordings": int(idx.size)}


def run_nested_cv(cohort: Cohort, models: dict, *, n_repeats: int = 10,
                  n_outer: int = 5, n_inner: int = 3, base_seed: int = 20260917,
                  scoring: str = "roc_auc", n_jobs: int = 1,
                  verbose: bool = True) -> CVResult:
    X, y, groups = cohort.X, cohort.y, cohort.groups
    fold_rows, oof_sub_rows, oof_rec_rows, comp_rows, param_rows = [], [], [], [], []

    for rep in range(n_repeats):
        seed = base_seed + rep
        outer = StratifiedGroupKFold(n_splits=n_outer, shuffle=True,
                                     random_state=seed)
        for fold, (tr, te) in enumerate(outer.split(X, y, groups)):
            comp_rows.append({"repeat": rep, "fold": fold, "seed": seed,
                              "split": "outer_train", **_composition(y, groups, tr)})
            comp_rows.append({"repeat": rep, "fold": fold, "seed": seed,
                              "split": "outer_test", **_composition(y, groups, te)})

            inner = StratifiedGroupKFold(n_splits=n_inner, shuffle=True,
                                         random_state=seed)
            for i, (itr, ite) in enumerate(
                    inner.split(X.iloc[tr], y[tr], groups[tr])):
                comp_rows.append({"repeat": rep, "fold": fold, "seed": seed,
                                  "split": f"inner{i}_val",
                                  **_composition(y[tr], groups[tr], ite)})

            for name, (pipe, grid) in models.items():
                gs = GridSearchCV(clone(pipe), grid, scoring=scoring,
                                  cv=list(inner.split(X.iloc[tr], y[tr],
                                                      groups[tr])),
                                  n_jobs=n_jobs, refit=True, error_score="raise")
                gs.fit(X.iloc[tr], y[tr], groups=groups[tr])

                prob = gs.predict_proba(X.iloc[te])[:, 1]
                sub = subject_aggregate(prob, groups[te], y[te])

                auc_sub = (roc_auc_score(sub["y"], sub["prob"])
                           if sub["y"].nunique() > 1 else np.nan)
                auc_rec = (roc_auc_score(y[te], prob)
                           if len(np.unique(y[te])) > 1 else np.nan)

                fold_rows.append({
                    "model": name, "repeat": rep, "fold": fold, "seed": seed,
                    "auc_subject": auc_sub, "auc_recording": auc_rec,
                    "n_test_subjects": int(sub.shape[0]),
                    "n_test_hc_subjects": int((sub["y"] == 0).sum()),
                })
                param_rows.append({"model": name, "repeat": rep, "fold": fold,
                                   **gs.best_params_,
                                   "inner_best_score": gs.best_score_})
                for _, r in sub.iterrows():
                    oof_sub_rows.append({"model": name, "repeat": rep,
                                         "fold": fold, "subject": r["subject"],
                                         "prob": r["prob"], "y": int(r["y"]),
                                         "n_recordings": int(r["n_recordings"])})
                for j, k in enumerate(te):
                    oof_rec_rows.append({"model": name, "repeat": rep,
                                         "fold": fold, "row": int(k),
                                         "subject": groups[k],
                                         "prob": float(prob[j]), "y": int(y[k])})
        if verbose:
            print(f"  repeat {rep + 1}/{n_repeats} done", flush=True)

    return CVResult(
        fold_scores=pd.DataFrame(fold_rows),
        oof_subject=pd.DataFrame(oof_sub_rows),
        oof_recording=pd.DataFrame(oof_rec_rows),
        fold_composition=pd.DataFrame(comp_rows).drop_duplicates(),
        best_params=pd.DataFrame(param_rows),
        meta={"cohort": cohort.name, "n_features": len(cohort.features),
              "n_repeats": n_repeats, "n_outer": n_outer, "n_inner": n_inner,
              "base_seed": base_seed, "scoring": scoring},
    )
