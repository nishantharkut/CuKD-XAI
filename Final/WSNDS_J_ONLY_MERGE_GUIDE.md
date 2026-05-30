# WSN-DS J-only Merge Guide

This route is for the time-efficient final path: keep the completed existing 10-seed v2.3 results, run only Config J (`J_CoDistill_RF_CL`) over the same seeds, then merge and recompute aggregates/statistics/figures.

## Files

- `cukd_xai_wsnds_j_only_merge.py`: source file.
- `cukd_xai_wsnds_j_only_merge.ipynb`: notebook generated from the source.
- `WSNDS_J_ONLY_MERGE_GUIDE.md`: this guide.

## Required Inputs

- Existing 10-seed JSON: `../Results - 10 Seed Run 30 may/cukd_xai_results.json`
- WSN-DS CSV: `WSN-DS.csv` in the notebook folder, or set `WSNDS_PATH`.

The notebook validates that the existing 10-seed result has the same seeds, classes, feature names, and required baseline configs before merging. It refuses to merge into an already-merged JSON containing `J_CoDistill_RF_CL`.

## What It Runs

It does not rerun A/B/C/D/E/E2/F/G/I.

For each seed it trains only the support components needed for J:

- calibrated RF soft labels
- fair-budget CL MLP teacher
- Config J student for `student_A_32_16`
- Config J student for `student_B_64_32`

Then it writes a new folder:

`../Results - 10 Seed Run 30 may + J`

## Outputs

- `cukd_xai_results_with_J.json`
- `j_only_results.json`
- `merge_report.json`
- `wsnds_results_student_A.csv`
- `wsnds_results_student_B.csv`
- updated per-class F1, Pareto, J confusion matrix, and J loss-curve figures

## Time Estimate

Expected runtime is roughly 6-16 hours for the full J-only 10-seed merge, depending on CPU/RAM/GPU and RF calibration speed. It is much cheaper than rerunning the full 10-seed v2.3 route, but J still needs RF calibration and a CL teacher per seed.

## How To Run

Open `cukd_xai_wsnds_j_only_merge.ipynb`, clear outputs, and click Run All.

Keep `J_ONLY_QUICK_MODE = False` for final evidence. Use quick mode only to sanity-check paths and syntax.

## Publication Use

Use this merged output only if the merge report confirms the same seeds and compatibility checks passed. For Wilcoxon/significance claims, use `wilcoxon_results_with_J` in `cukd_xai_results_with_J.json`.
