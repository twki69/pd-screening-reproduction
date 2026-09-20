# Leakage-Aware Evaluation of Machine Learning Models for Voice-Based Parkinson's Disease Screening

Reproduction code for the manuscript submitted to *Engineering Reports*.

Both datasets are downloaded automatically from the UCI Machine Learning
Repository, so no data redistribution is required.

## What this code establishes

Published results on the UCI Oxford dysphonia benchmark commonly exceed 95%
accuracy. That dataset holds roughly six recordings per participant, so a
random train/test split places recordings of the same speaker on both sides.
This pipeline enforces the participant as the unit of analysis throughout and
reports what remains: **subject-level AUC 0.870** for logistic regression.

## Quick start

```bash
git clone <repository-url> && cd <repository>
pip install -r requirements.txt
python run_all.py              # full run, ~25 min on one CPU core
python benchmark_timing.py     # timings for your machine
```

For a fast check (3 repeats instead of 10): `python run_all.py --quick`.

On Colab, open `notebooks/reproduce.ipynb` and run all cells.
Extracting the Istanbul archive requires `apt-get install -y unrar-free`,
which the notebook handles.

## What each module does

| Module | Purpose |
|---|---|
| `pdpipe/data.py` | Dataset download, subject parsing, 16-feature harmonisation map |
| `pdpipe/models.py` | Nine classifiers with their search spaces |
| `pdpipe/cv.py` | Repeated stratified group nested CV; subject-level aggregation |
| `pdpipe/stats.py` | Effect sizes, repeat-level bootstrap CIs, Holm correction |
| `pdpipe/external.py` | Three threshold policies; cross-cohort distribution shift |
| `pdpipe/adapt.py` | Label-free domain adaptation (both strategies fail; see paper) |
| `pdpipe/explain.py` | Out-of-fold SHAP and permutation importance with stability |
| `pdpipe/timing.py` | Environment capture and repeated timing measurement |

## Design decisions that matter

**Grouping.** The subject is the grouping unit in *both* loops of the nested
design. No participant contributes recordings to both sides of any split.

**Repetition.** Ten repeats give 50 outer folds. Single-run estimates on this
cohort are unstable: individual fold AUCs span 0.00 to 1.00, and 20 of 50
outer test folds contain a single control subject.

**Aggregation.** Predictions are averaged per participant before any metric is
computed, so participants with more recordings are not over-weighted and
internal results stay comparable with external ones.

**No oversampling.** SMOTE is deliberately absent. Applied before splitting it
propagates speaker identity across the partition; applied within folds it does
not address the leakage this study is about.

**Threshold policies.** External results separate a threshold fixed from the
source cohort (blind) from one calibrated on a held-out half of the target
cohort. These give very different answers and must not be conflated.

## Reproducibility

All seeds derive from `SEED = 20260917`. Library versions are pinned in
`requirements.txt`; different versions may shift figures slightly. Timings are
hardware-dependent by nature — run `benchmark_timing.py` yourself rather than
quoting numbers measured elsewhere. It writes a ready-to-paste LaTeX block to
`results/timing_table.tex`.

## Citation

Twki S.M., Fahim M.R.I., Masum H.M. *A Leakage-Aware Comparative Evaluation of
Machine Learning Models for Voice-Based Parkinson's Disease Screening, with
External Validation on an Independent Cohort.* Engineering Reports (under
review).

## Data sources

- Little M.A. et al. (2009). *IEEE Trans. Biomed. Eng.* 56(4):1015–1022.
  [UCI dataset 174](https://archive.ics.uci.edu/dataset/174/parkinsons)
- Sakar C.O. et al. (2019). *Applied Soft Computing* 74:255–263.
  [UCI dataset 470](https://archive.ics.uci.edu/dataset/470/parkinson+s+disease+classification)

## Licence

MIT (code). The datasets remain under the terms of the UCI repository.
