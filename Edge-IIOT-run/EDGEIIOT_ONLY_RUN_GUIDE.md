# Edge-IIoT-only Run Guide

Created: 2026-05-29

Use these files when you want the Edge-IIoT secondary-dataset evidence without rerunning the completed main WSN-DS experiment:

- `cukd_xai_edgeiiot_only.py`
- `cukd_xai_edgeiiot_only.ipynb`

This Edge-IIoT-only route does not rerun WSN-DS and does not depend on the v2.3 notebook cells.

## Before Run All

Set the CSV path in Cell 1:

```python
EDGEIIOT_ML_PATH = "/home/ubuntu/nishn_workspce/oig-exclusion-testing/.cukd_xai_secret/datasets/edgeiiot/extracted/Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv"
EDGEIIOT_TARGET_COL = "Attack_type"
EDGEIIOT_RUN_MODE = "edgeiiot_final"
```

Use the selected ML CSV, not the larger DNN CSV, unless you intentionally want a much heavier run.

## What It Runs

For each seed, it trains one calibrated RF teacher and reuses it for both students:

- `A_RF_calibrated`
- `D_student_scratch`
- `E_KD_from_RF`

Students:

- `student_A_32_16`
- `student_B_64_32`

Default final seeds:

```python
[42, 123, 456, 789, 1001]
```

## Safety Choices

- Drops known leakage/source columns and auxiliary target columns before training. Drops only missing/empty target rows; missing feature values are handled by numeric imputation or categorical missing tokens.
- Uses `Attack_type` for multiclass evaluation.
- Learns categorical caps only on the train split.
- Scales only continuous numeric columns.
- Keeps one-hot dummy columns as 0/1.
- Uses sigmoid RF calibration with class-count-safe CV.
- Uses CPU training for MLP students to avoid CUDA memory failure on large one-hot matrices.

## Outputs

The notebook writes:

- `edgeiiot_only_outputs/edgeiiot_generalization_summary.csv`
- `edgeiiot_only_outputs/edgeiiot_generalization_results.json`

## Runtime

Expected time for 5 seeds on the i9-13950HX / RTX 1000 Ada system is roughly 12-30 hours depending on RAM pressure and CSV size. If memory is very low, close other apps first.

## Validation Status

The actual `ML-EdgeIIoT-dataset.csv` is present in this workspace and was audited: 157,800 rows, 63 columns, 15 `Attack_type` classes, no malformed rows, and no invalid targets. Runtime validation still depends on the Jupyter environment having numpy, pandas, scikit-learn, and torch installed.
