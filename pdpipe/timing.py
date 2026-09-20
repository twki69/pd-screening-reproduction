"""Reproducible computational-efficiency benchmark.

The review objected that the previous timing came from a single split, did not
include the nested hyperparameter search or preprocessing, and was compared
against GPU deep-learning studies run on different data, tasks and hardware.

This module therefore reports, for every model and on a single declared
machine:

* environment: processor, core count, memory, operating system, Python and
  library versions, and the BLAS/OpenMP thread settings in force;
* end-to-end tuning time -- the complete inner grid search plus refit, which
  is what a practitioner actually pays to deploy a model;
* fit time and inference time for a stated batch size, each repeated a
  declared number of times and reported as mean and standard deviation;
* peak resident memory during the measured block.

Timings from different machines are not comparable, so the environment block
is reported alongside every table rather than in a footnote. No comparison is
made against figures taken from other papers' hardware.
"""
from __future__ import annotations

import gc
import os
import platform
import resource
import statistics
import sys
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from .data import Cohort

THREAD_VARS = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]


def pin_threads(n: int = 1) -> None:
    """Pin BLAS/OpenMP threads so timings do not depend on host core count.

    Must be called before numpy or the boosting libraries are imported for the
    settings to take full effect; the benchmark scripts call it first.
    """
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(n)


def environment_report() -> dict:
    """Everything needed to make the timing table interpretable."""
    import sklearn
    try:
        import catboost, lightgbm, xgboost
        boost = {"catboost": catboost.__version__,
                 "lightgbm": lightgbm.__version__,
                 "xgboost": xgboost.__version__}
    except Exception:
        boost = {}

    mem_total = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_total = round(int(line.split()[1]) / 1024 ** 2, 2)
                    break
    except Exception:
        pass

    cpu_model = platform.processor() or None
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass

    return {
        "cpu_model": cpu_model,
        "logical_cores": os.cpu_count(),
        "memory_gb": mem_total,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        **boost,
        "thread_env": {v: os.environ.get(v, "unset") for v in THREAD_VARS},
    }


def _peak_memory_mb() -> float:
    """Peak resident set size of this process, in MB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1024 if sys.platform != "darwin" else peak / 1024 ** 2


def benchmark(cohort: Cohort, models: dict, *, n_repetitions: int = 5,
              inference_batch: int = 195, n_inner: int = 3,
              seed: int = 20260917) -> tuple[pd.DataFrame, dict]:
    """Time tuning, fitting and inference for each model on one machine."""
    X, y, groups = cohort.X, cohort.y, cohort.groups
    inner = StratifiedGroupKFold(n_splits=n_inner, shuffle=True,
                                 random_state=seed)
    splits = list(inner.split(X, y, groups))
    Xb = X.sample(n=inference_batch, replace=inference_batch > len(X),
                  random_state=seed)

    rows = []
    for name, (pipe, grid) in models.items():
        n_configs = int(np.prod([len(v) for v in grid.values()])) if grid else 1

        gc.collect()
        t0 = time.perf_counter()
        gs = GridSearchCV(clone(pipe), grid, scoring="roc_auc", cv=splits,
                          n_jobs=1, refit=True)
        gs.fit(X, y)
        tune_s = time.perf_counter() - t0

        fit_times, infer_times = [], []
        for r in range(n_repetitions):
            est = clone(gs.best_estimator_)
            gc.collect()
            t0 = time.perf_counter()
            est.fit(X, y)
            fit_times.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            est.predict_proba(Xb)
            infer_times.append(time.perf_counter() - t0)

        rows.append({
            "model": name,
            "n_grid_configs": n_configs,
            "n_inner_folds": n_inner,
            "end_to_end_tuning_s": round(tune_s, 4),
            "fit_mean_s": round(statistics.mean(fit_times), 4),
            "fit_sd_s": round(statistics.pstdev(fit_times)
                              if len(fit_times) > 1 else 0.0, 4),
            "inference_mean_ms": round(statistics.mean(infer_times) * 1000, 3),
            "inference_sd_ms": round((statistics.pstdev(infer_times) * 1000)
                                     if len(infer_times) > 1 else 0.0, 3),
            "inference_batch_size": inference_batch,
            "inference_per_sample_ms": round(
                statistics.mean(infer_times) * 1000 / inference_batch, 4),
            "n_repetitions": n_repetitions,
            "peak_rss_mb": round(_peak_memory_mb(), 1),
            "seed": seed,
            "best_params": str(gs.best_params_),
        })
    return pd.DataFrame(rows), environment_report()
