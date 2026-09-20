"""External validation on the Istanbul cohort.

The review objected that the previously reported external results selected a
Youden-J threshold using labels from the same cohort on which the
threshold-dependent metrics were then reported, which makes those metrics
optimistically biased rather than blind.

This module therefore evaluates three clearly separated threshold policies:

A. ``oxford_fixed``   -- threshold chosen from Oxford out-of-fold predictions
                         only. No Istanbul label is touched before scoring.
                         This is the genuinely blind transfer result.
B. ``istanbul_split`` -- Istanbul is divided at subject level into a
                         calibration half and an untouched test half. The
                         threshold comes from calibration; every metric is
                         reported on the test half only.
C. ``in_sample``      -- threshold chosen on the whole cohort and scored on the
                         whole cohort. Reported only to quantify the size of
                         the optimism it introduces; never presented as a
                         validation result.

Hyperparameters are re-selected on the Oxford cohort using only the 16 shared
features, rather than reused from the 22-feature search.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from .data import Cohort, HARMONISED_MAP
from .cv import subject_aggregate


def youden_threshold(y: np.ndarray, prob: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y, prob)
    return float(thr[np.argmax(tpr - fpr)])


def refit_on_oxford(cohort: Cohort, models: dict, *, n_inner: int = 5,
                    seed: int = 20260917) -> dict:
    """Re-select hyperparameters on the 16 shared features, subject-grouped."""
    fitted = {}
    inner = StratifiedGroupKFold(n_splits=n_inner, shuffle=True,
                                 random_state=seed)
    splits = list(inner.split(cohort.X, cohort.y, cohort.groups))
    for name, (pipe, grid) in models.items():
        gs = GridSearchCV(clone(pipe), grid, scoring="roc_auc", cv=splits,
                          n_jobs=1, refit=True, error_score="raise")
        gs.fit(cohort.X, cohort.y)
        fitted[name] = {"estimator": gs.best_estimator_,
                        "params": gs.best_params_,
                        "oxford_inner_auc": float(gs.best_score_)}
    return fitted


def subject_split(sub: pd.DataFrame, *, frac_cal: float = 0.5,
                  seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Class-stratified subject-level split into calibration and test."""
    rng = np.random.default_rng(seed)
    cal, test = [], []
    for label, g in sub.groupby("y"):
        ids = g["subject"].to_numpy()
        rng.shuffle(ids)
        k = int(round(len(ids) * frac_cal))
        cal.extend(ids[:k])
        test.extend(ids[k:])
    return np.array(cal), np.array(test)


def _metrics(y: np.ndarray, prob: np.ndarray, thr: float) -> dict:
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": thr,
        "auc": roc_auc_score(y, prob) if len(np.unique(y)) > 1 else np.nan,
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "sensitivity": recall_score(y, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "mcc": matthews_corrcoef(y, pred),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_subjects": int(len(y)),
    }


def bootstrap_metric_ci(y: np.ndarray, prob: np.ndarray, thr: float,
                        metric: str = "balanced_accuracy", *,
                        n_boot: int = 2000, seed: int = 3) -> tuple:
    """Percentile CI over resampled subjects (vectorised balanced accuracy)."""
    rng = np.random.default_rng(seed)
    pred = (prob >= thr).astype(int)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    yb, pb = y[idx], pred[idx]
    tp = ((yb == 1) & (pb == 1)).sum(1)
    fn = ((yb == 1) & (pb == 0)).sum(1)
    tn = ((yb == 0) & (pb == 0)).sum(1)
    fp = ((yb == 0) & (pb == 1)).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
    vals = (sens + spec) / 2
    vals = vals[np.isfinite(vals)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def run_external(ox16: Cohort, ist: Cohort, models: dict,
                 oxford_oof: pd.DataFrame, *, seed: int = 11) -> dict:
    """Full external-validation protocol for the selected models."""
    fitted = refit_on_oxford(ox16, models)
    rows, subject_frames = [], {}

    for name, info in fitted.items():
        est = info["estimator"]
        prob = est.predict_proba(ist.X)[:, 1]
        sub = subject_aggregate(prob, ist.groups, ist.y)
        subject_frames[name] = sub

        # Policy A: threshold from Oxford out-of-fold predictions only.
        ox_oof = oxford_oof[oxford_oof["model"] == name]
        ox_sub = (ox_oof.groupby("subject")
                  .agg(prob=("prob", "mean"), y=("y", "first")).reset_index())
        thr_a = youden_threshold(ox_sub["y"].to_numpy(), ox_sub["prob"].to_numpy())

        cal_ids, test_ids = subject_split(sub, seed=seed)
        cal = sub[sub["subject"].isin(cal_ids)]
        test = sub[sub["subject"].isin(test_ids)]

        # Policy B: threshold from the Istanbul calibration half only.
        thr_b = youden_threshold(cal["y"].to_numpy(), cal["prob"].to_numpy())
        # Policy C: in-sample threshold (reported as a bias illustration).
        thr_c = youden_threshold(sub["y"].to_numpy(), sub["prob"].to_numpy())

        yt, pt = test["y"].to_numpy(), test["prob"].to_numpy()
        ya, pa = sub["y"].to_numpy(), sub["prob"].to_numpy()

        for policy, thr, (yy, pp), scope in [
            ("A_oxford_fixed", thr_a, (yt, pt), "istanbul_test_half"),
            ("A_oxford_fixed_full", thr_a, (ya, pa), "istanbul_all"),
            ("B_istanbul_calibrated", thr_b, (yt, pt), "istanbul_test_half"),
            ("C_in_sample_biased", thr_c, (ya, pa), "istanbul_all"),
        ]:
            m = _metrics(yy, pp, thr)
            lo, hi = bootstrap_metric_ci(yy, pp, thr)
            rows.append({"model": name, "policy": policy, "scope": scope,
                         **m, "bal_acc_ci_lo": lo, "bal_acc_ci_hi": hi,
                         "oxford_inner_auc": info["oxford_inner_auc"],
                         "params": str(info["params"])})

    return {"table": pd.DataFrame(rows),
            "subject_probs": subject_frames,
            "fitted": fitted}


def distribution_shift(ox16: Cohort, ist: Cohort) -> pd.DataFrame:
    """Per-feature cross-cohort comparison, controls-only and overall.

    Restricting to control subjects removes disease prevalence as a
    confound, so a large remaining shift indicates a measurement or
    extraction difference rather than a clinical one.
    """
    rows = []
    for i, f in enumerate(ox16.features):
        a, b = ox16.X.iloc[:, i], ist.X.iloc[:, i]
        ac, bc = a[ox16.y == 0], b[ist.y == 0]
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        pooled_c = np.sqrt((ac.var(ddof=1) + bc.var(ddof=1)) / 2)
        rows.append({
            "feature": f,
            "istanbul_column": HARMONISED_MAP[f][0],
            "oxford_mean": a.mean(), "istanbul_mean": b.mean(),
            "oxford_sd": a.std(ddof=1), "istanbul_sd": b.std(ddof=1),
            "smd_all": (a.mean() - b.mean()) / pooled if pooled else np.nan,
            "smd_controls": ((ac.mean() - bc.mean()) / pooled_c
                             if pooled_c else np.nan),
            "oxford_missing": int(a.isna().sum()),
            "istanbul_missing": int(b.isna().sum()),
        })
    out = pd.DataFrame(rows)
    out["abs_smd_controls"] = out["smd_controls"].abs()
    return out.sort_values("abs_smd_controls", ascending=False)
