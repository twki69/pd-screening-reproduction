"""Reproduce every result reported in the paper, end to end.

    python run_all.py              # full run (10 repeats)
    python run_all.py --quick      # 3 repeats, for a fast check

Outputs are written to results/. With the seeds fixed below, the full run
reproduces the published figures exactly on any machine.
"""
import argparse, json, warnings
import os
os.makedirs("results", exist_ok=True)
warnings.filterwarnings("ignore")

import pandas as pd

from pdpipe.data import load_oxford, load_oxford_meta, load_istanbul, harmonised_oxford
from pdpipe.models import build_models
from pdpipe.cv import run_nested_cv
from pdpipe.stats import summarise_models, pairwise_comparisons
from pdpipe.external import run_external, distribution_shift
from pdpipe.explain import run_explanations, stability_table

SEED = 20260917
EXTERNAL_MODELS = ("LogisticRegression", "RandomForest", "CatBoost")


def main(quick: bool = False):
    n_rep = 3 if quick else 10
    ox, ist = load_oxford(), load_istanbul()
    print(f"Oxford: {ox.summary()}")
    print(f"Istanbul: {ist.summary()}")
    load_oxford_meta().to_csv("results/subject_mapping.csv", index=False)

    print(f"\n[1/5] Nested CV, 22 features, {n_rep} repeats ...")
    r22 = run_nested_cv(ox, build_models(), n_repeats=n_rep, base_seed=SEED)
    for k in ("fold_scores", "oof_subject", "fold_composition", "best_params"):
        getattr(r22, k).to_csv(f"results/main22_{k}.csv", index=False)
    print(summarise_models(r22.fold_scores).round(3).to_string(index=False))

    print("\n[2/5] Pairwise statistical comparison ...")
    pairwise_comparisons(r22.fold_scores).to_csv("results/main22_pairwise.csv", index=False)

    print(f"\n[3/5] Nested CV, 16 shared features ...")
    ox16 = harmonised_oxford(ox)
    r16 = run_nested_cv(ox16, build_models(), n_repeats=n_rep, base_seed=SEED)
    r16.fold_scores.to_csv("results/main16_fold_scores.csv", index=False)
    r16.oof_subject.to_csv("results/main16_oof_subject.csv", index=False)

    print("\n[4/5] External validation and harmonisation ...")
    sel = {k: v for k, v in build_models().items() if k in EXTERNAL_MODELS}
    ext = run_external(ox16, ist, sel, r16.oof_subject)
    ext["table"].to_csv("results/external.csv", index=False)
    distribution_shift(ox16, ist).to_csv("results/harmonisation_shift.csv", index=False)

    print("\n[5/5] Out-of-fold explainability ...")
    e = run_explanations(ox, build_models(), r22.best_params,
                         model_names=list(EXTERNAL_MODELS),
                         n_repeats=min(5, n_rep))
    e.to_csv("results/explanations.csv", index=False)
    for m in EXTERNAL_MODELS:
        t = stability_table(e, m, "mean_abs_shap")
        if not t.empty:
            t.to_csv(f"results/xai_{m}.csv")

    print("\nDone. See results/. Run benchmark_timing.py separately for timings.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(**vars(ap.parse_args()))
