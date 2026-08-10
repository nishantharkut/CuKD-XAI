# WSN-DS train-only-scaler confirmation workspace

This additive workspace does not replace the archived WSN-DS experiments or
their results. Despite the historical directory name, the active random-row
rerun must not be described as fully leakage-free: the scaler is corrected,
but exact feature duplicates can still cross its fixed row-level partitions.

## Evidence routes

### Active v1 statistical rerun

The process recorded in
`results/wsnds/leakage_free_rerun/main_10seed/executed_source_snapshot/`
uses:

1. the archived seed-42 stratified 70/15/15 raw-row split;
2. `StandardScaler.fit` on the training partition only;
3. the ten optimization seeds `42, 123, 456, 789, 1001, 2024, 3141, 5678,
   8192, 9999` on that one fixed split; and
4. `T=4`, `alpha=0.7` selected by its preliminary v1 grid procedure.

The v1 grid did not hold initialization and shuffle streams constant across
all candidates. Therefore, the chosen pair is a shared preliminary setting,
not evidence of an independently controlled RF-KD optimum.

The active-v1 configuration names containing `_fair` are historical labels,
not a verified compute-matched curriculum experiment. The staged schedule
processes about `26.97` full-dataset epoch equivalents (`3x0.33 + 3x0.66 +
24x1.0`) before any global early stopping, while the baseline schedule uses
full-dataset epochs. Global early stopping can also terminate before the full
stage. Do not use active-v1 curriculum comparisons as causal or compute-fair
evidence.

The active-v1 SHAP audit compares a curriculum-teacher student with an
independently fitted RF. It is a cross-model attribution comparison, not an
attribution-transfer test for the headline RF-KD student.

### Exact-duplicate sensitivity route

`run_tier15_confirmation.py --mode duplicate-sensitivity` groups rows by all
17 raw model features before splitting. It keeps every feature group within
one partition, fits scaling on grouped training only, freezes `T=4` and
`alpha=0.7`, and runs five optimization seeds for only:

- calibrated RF teacher;
- scratch Student A and RF-KD Student A; and
- scratch Student B and RF-KD Student B.

Scratch and RF-KD use the same initial state within each student/seed. The
output includes per-seed metrics, mean and sample standard deviation,
per-class F1, and paired RF-KD-minus-scratch differences. Five-seed summaries
are descriptive; the route makes no statistical-significance claim.
It is a duplicate-sensitivity experiment, not a matched causal ablation against
the random-row route; those outputs must not be compared as if duplicate
handling were the only changed factor.

The passed data-only preflight is at
`results/wsnds/confirmation_runs_v2/preflight/preflight_report.json`. It records
13,505 feature-duplicate rows after first occurrence, three mixed-label
feature groups, nonzero feature-group overlap under the archived split, and
zero feature-group overlap under the sensitivity split.

### Focused deployment route

`run_tier15_confirmation.py --mode deployment` trains only seed-42 RF-KD
Student A and Student B. It binds the existing seed-42 calibrated-RF soft
targets to the active executed-source hash, dataset hash, raw split, and exact
train-only scaler. It does not refit the RF teacher or run unrelated routes.

The corrected fixed-point exporter and hardware tools are:

- `deployment/firmware_export/wsnds_rfkd_hil/export_train_only_deployment.py`
- `deployment/hardware_hil/host/prepare_strict_firmware_bundle.py`
- `deployment/hardware_hil/host/stream_vectors_strict.py`
- `deployment/hardware_hil/host/verify_results_strict.py`
- `deployment/hardware_hil/host/record_compile_evidence.py`
- `deployment/hardware_hil/host/generate_strict_report.py`
- `deployment/hardware_hil/docs/11_TRAIN_ONLY_SCALER_HIL_RUNBOOK.md`

The historical exporter still reproduces the archived global-scaler lineage;
do not use it for corrected deployment models.

## Safe commands

Data-only preflight, with no model training:

```powershell
& "experiments\wsnds\leakage_free_rerun\.venv\Scripts\python.exe" `
  "experiments\wsnds\leakage_free_rerun\run_tier15_confirmation.py" `
  --mode preflight
```

Training modes require `--confirm-training` and are blocked on Windows while
the preserved active-v1 worker PID is alive. Every output route refuses a
nonempty destination unless its documented, hash-verifying resume path is
used.

If the active process completes all 20 seed checkpoints but fails later during
SHAP, plotting, or report generation, run
`recover_active_v1_results.py`. Use its validated tables and Holm-adjusted
paired tests as the canonical active-v1 metric curation even if the broad
script also finishes its own reports. The recovery tool recomputes all neural
metrics from confusion matrices, validates aliases and parameter counts, and
accepts the SMOTE route only when it exists for every required comparison.
The RF baseline checkpoints contain summary metrics but no confusion matrices,
so those RF summaries cannot be independently reconstructed. The tool does not
reconstruct XAI, quantization, runtime, or deployment artifacts.

All ten active-v1 runs vary optimizer seeds on one fixed archived split. They
do not estimate uncertainty across independently sampled train/test splits.

## Claim boundary

The WSN-DS hardware route replays already extracted 17-feature records over
USB serial. It does not establish live packet capture, packet-to-feature
extraction on an MCU, board energy, radio integration, or TelosB execution.
