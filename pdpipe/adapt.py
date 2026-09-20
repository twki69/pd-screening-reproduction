"""Label-free domain adaptation for cross-cohort transfer.

The cross-cohort comparison shows that 12 of the 16 nominally shared
features differ by more than one standard deviation between cohorts *among
control subjects alone*, which points to differences in extraction software
and measurement convention rather than to clinical differences.

The mitigation applied here is the simplest one that does not use any
Istanbul label: each cohort is standardised with its own mean and standard
deviation before the Oxford-trained classifier is applied. This is
transductive -- it requires the target feature matrix, though never its
labels -- and that limitation is stated explicitly rather than hidden.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from .cv import subject_aggregate
from .data import Cohort
from .external import (_metrics, bootstrap_metric_ci, subject_split,
                       youden_threshold)


def fit_adapted(ox: Cohort, models: dict, seed_params: dict) -> dict:
    """Fit each classifier on cohort-standardised Oxford features."""
    out = {}
    for name, (pipe, _) in models.items():
        clf = clone(pipe.named_steps["clf"])
        params = {k.replace("clf__", ""): v
                  for k, v in seed_params[name]["params"].items()}
        clf.set_params(**params)
        sc = StandardScaler().fit(ox.X)
        clf.fit(pd.DataFrame(sc.transform(ox.X), columns=ox.X.columns), ox.y)
        out[name] = {"clf": clf, "oxford_scaler": sc, "params": params}
    return out


def transfer_adapted(fitted: dict, ox: Cohort, ist: Cohort,
                     oxford_oof: pd.DataFrame, *, seed: int = 11) -> pd.DataFrame:
    """Apply adapted models to Istanbul under the same threshold policies."""
    rows = []
    target_scaler = StandardScaler().fit(ist.X)   # labels never touched
    Xt = pd.DataFrame(target_scaler.transform(ist.X), columns=ist.X.columns)

    for name, info in fitted.items():
        prob = info["clf"].predict_proba(Xt)[:, 1]
        sub = subject_aggregate(prob, ist.groups, ist.y)

        ox_oof = oxford_oof[oxford_oof["model"] == name]
        ox_sub = (ox_oof.groupby("subject")
                  .agg(prob=("prob", "mean"), y=("y", "first")).reset_index())
        thr_a = youden_threshold(ox_sub["y"].to_numpy(), ox_sub["prob"].to_numpy())

        cal_ids, test_ids = subject_split(sub, seed=seed)
        cal = sub[sub["subject"].isin(cal_ids)]
        test = sub[sub["subject"].isin(test_ids)]
        thr_b = youden_threshold(cal["y"].to_numpy(), cal["prob"].to_numpy())

        for policy, thr, frame, scope in [
            ("A_oxford_fixed", thr_a, test, "istanbul_test_half"),
            ("A_oxford_fixed_full", thr_a, sub, "istanbul_all"),
            ("B_istanbul_calibrated", thr_b, test, "istanbul_test_half"),
        ]:
            y, p = frame["y"].to_numpy(), frame["prob"].to_numpy()
            m = _metrics(y, p, thr)
            lo, hi = bootstrap_metric_ci(y, p, thr)
            rows.append({"model": name, "adaptation": "per_cohort_zscore",
                         "policy": policy, "scope": scope, **m,
                         "bal_acc_ci_lo": lo, "bal_acc_ci_hi": hi})
    return pd.DataFrame(rows)
