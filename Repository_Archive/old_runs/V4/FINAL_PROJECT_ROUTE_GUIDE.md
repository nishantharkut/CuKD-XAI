# CuKD-XAI Final Project Route Guide

Created: 2026-05-27

This guide is the current final route for turning the v2.3 CuKD-XAI work into a paper-ready project. The root v2.3 source is still the stable base:

- `cukd_xai_colab.py`
- `cukd_xai_colab.ipynb`

The v3 file is useful for ideas, but should not be used wholesale.

## Current Evidence

Best verified evidence remains the May 6 v2.3 WSN-DS run:

- RF teacher: macro-F1 `0.978967 +/- 0.000289`
- Student A `(32, 16)` KD from RF: macro-F1 `0.920490 +/- 0.002759`, 1,189 params
- Student B `(64, 32)` KD from RF: macro-F1 `0.934540 +/- 0.007716`, 3,397 params
- CL fixed from catastrophic failure to neutral, but CL is not a win
- SHAP teacher/student rank agreement is weak: rho `0.117647`, bootstrap mean `0.166667`
- Dynamic INT8 reduced size only modestly and hurt F1

## Final Narrative

Do not frame the paper as "curriculum learning wins."

Frame it as:

> KB-scale RF/tree-ensemble-to-MLP knowledge distillation for WSN/edge IDS, with cross-dataset validation, deployment-feasibility profiling, and SHAP-based teacher-student explanation alignment analysis.

Safe claims:

- compact student IDS models
- RF-to-student KD as the practical winner
- explicit model size and latency profiling
- explanation-alignment audit
- WSN-DS primary plus Edge-IIoTset generalization

Unsafe claims unless proven later:

- highest accuracy
- first XAI on WSN-DS
- actual TelosB/MSP430 deployment
- hardware energy measurements

## Dataset Plan

Primary dataset:

- WSN-DS, 5-class multiclass, already validated

Secondary/generalization dataset:

- Edge-IIoTset selected ML CSV, 15-class multiclass via `Attack_type`

Do not use CICIoT for this route unless explicitly revived.

Edge-IIoTset audit results:

- `ML-EdgeIIoT-dataset.csv`: 157,800 rows, 63 columns, manageable
- `DNN-EdgeIIoT-dataset.csv`: 2,219,201 rows, 63 columns, optional only after ML path is stable
- Use ML CSV first because available RAM can be low

Edge-IIoTset preprocessing must:

1. Drop leakage/source columns listed in the README:
   - `frame.time`
   - `ip.src_host`
   - `ip.dst_host`
   - `arp.src.proto_ipv4`
   - `arp.dst.proto_ipv4`
   - `http.file_data`
   - `http.request.full_uri`
   - `icmp.transmit_timestamp`
   - `http.request.uri.query`
   - `tcp.options`
   - `tcp.payload`
   - `tcp.srcport`
   - `tcp.dstport`
   - `udp.port`
   - `mqtt.msg`
2. Drop missing rows.
3. Drop duplicates after leakage-column removal.
4. Drop zero-variance columns.
5. One-hot encode remaining categorical columns.
6. Fit scaler on training split only for final paper-grade experiments.
7. Report rare classes separately, especially `MITM` and `Fingerprinting`.

## Code Changes Added

The root v2.3 Python source now includes disabled-by-default final-route controls:

- `FINAL_RUN_MODE`
- `RUN_PUBLICATION_10_SEEDS`
- `RUN_EDGEIIOT_ML_SMOKE`
- `RUN_EDGEIIOT_ML_5SEED`
- `RUN_QAT_FOR_BEST_STUDENTS`
- `RUN_DEPLOYMENT_PROFILE`
- `RESUME_FINAL_RUNS`
- `FINAL_RUN_ROOT`
- `EDGEIIOT_ML_PATH`

New final-route cells appended to `cukd_xai_colab.py`:

- final leftover-work roadmap
- crash-resume helpers
- Edge-IIoTset leakage-safe preprocessing
- compact KD runner for Edge/generalization
- focused INT8/QAT deployment-feasibility utilities
- Edge-IIoTset ML run entrypoint

The notebook was regenerated from source with:

```bash
python3 make_notebook_preserve_banner.py
```

This preserves the top orientation banner and writes a backup:

- `cukd_xai_colab.ipynb.bak`

## How To Run Safely In Jupyter

Use Cell 2 as the notebook switchboard. Set EDGEIIOT_ML_PATH first if you are using Edge-IIoTset, keep RUN_LEGACY_V23_EXPERIMENTS = False, choose exactly one final mode, then click Jupyter Run All.

Valid modes:

~~~python
FINAL_RUN_MODE = "none"            # load/setup only
FINAL_RUN_MODE = "edge_smoke"      # quick Edge-IIoTset preprocessing/runtime check
FINAL_RUN_MODE = "edge_final"      # Edge-IIoTset 5-seed final
FINAL_RUN_MODE = "wsnds_final"     # WSN-DS 10-seed resumable final
FINAL_RUN_MODE = "qat_profile"     # QAT + dynamic INT8 + latency/profile after checkpoints exist
FINAL_RUN_MODE = "shap_alignment"  # RF/student SHAP alignment after checkpoints exist
~~~

Leave the older RUN_* final booleans False unless you intentionally want the backwards-compatible manual style. Do not combine multiple modes/guards in one Run All pass.

### 1. Edge-IIoTset Smoke Test

Set in Cell 2:

~~~python
FINAL_RUN_MODE = "edge_smoke"
EDGEIIOT_ML_PATH = "path/to/ML-EdgeIIoT-dataset.csv"
~~~

Smoke mode uses:

- 1 seed
- 100 RF trees
- Student A and Student B
- default KD settings to keep the check fast
- no full final claim

Purpose:

- verify preprocessing
- verify all 15 classes
- estimate runtime
- catch memory issues

### 2. Edge-IIoTset 5-Seed Final

After smoke passes, set:

~~~python
FINAL_RUN_MODE = "edge_final"
RESUME_FINAL_RUNS = True
EDGEIIOT_ML_PATH = "path/to/ML-EdgeIIoT-dataset.csv"
~~~

Outputs go to:

~~~text
final_runs/edgeiiot_ml_5seed/
~~~

The final Edge run caches an RF-KD validation grid, writes one metrics file per seed/student, and skips completed seeds on rerun when the saved run config matches.

### 3. WSN-DS 10-Seed Final

Set in Cell 2:

~~~python
FINAL_RUN_MODE = "wsnds_final"
RESUME_FINAL_RUNS = True
RUN_LEGACY_V23_EXPERIMENTS = False
~~~

This runs the v2.3 WSN-DS core through the resumable final route with 10 publication seeds. It is expensive. Expect roughly the same order of runtime as the historical 5-seed run, scaled by seed count and current CPU thermals.

### 4. QAT / INT8 / Latency Profile

After WSN-DS and/or Edge final checkpoints exist, set:

~~~python
FINAL_RUN_MODE = "qat_profile"
~~~

This runs dynamic INT8, focused QAT, and CPU deployment profiling only for the final best RF-KD students:

- WSN-DS Student A KD from RF
- WSN-DS Student B KD from RF
- Edge-IIoTset Student A KD from RF
- Edge-IIoTset Student B KD from RF

Report:

- fp32 F1
- dynamic INT8 F1
- QAT INT8 F1
- F1 drop
- on-disk size
- theoretical weights-only int8 size
- p50/p95 repeated-run CPU latency

### 5. SHAP Alignment

After the final checkpoints exist, set:

~~~python
FINAL_RUN_MODE = "shap_alignment"
~~~

This computes RF-teacher versus RF-KD-student SHAP rank alignment for WSN-DS and Edge-IIoTset.

## Crash Resume Design

Final Edge runs save per seed:

```text
final_runs/<run_name>/seed_<seed>/<student_name>/metrics.json
final_runs/<run_name>/seed_<seed>/<student_name>/models/*.pt
final_runs/<run_name>/<student_name>_aggregate_so_far.csv
final_runs/<run_name>/preprocessing_metadata.json
```

If Jupyter dies, restart and rerun with:

```python
RESUME_FINAL_RUNS = True
```

Completed seeds will be loaded from disk instead of rerun.

## Remaining Work Checklist

1. Run Edge-IIoTset smoke test.
2. Inspect smoke metrics and runtime.
3. Run Edge-IIoTset 5-seed final.
4. Run WSN-DS 10-seed final if compute allows.
5. Run focused INT8 + QAT for best students.
6. Run latency/FLOPs/size/memory profiling.
7. Run SHAP alignment on both datasets.
8. Create final result tables and figures.
9. Live-verify novelty claims before paper writing.
10. Write paper with honest limitations.

## Current Advice

Use WSN-DS as the primary dataset because the method is already deeply validated there. Use Edge-IIoTset as the generalization dataset because it strengthens the edge/IIoT relevance without losing the original WSN story.

If Edge-IIoTset results are weak, do not hide them. Diagnose teacher strength, student compression gap, rare-class failures, and leakage-preprocessing effects.


## Final-Route Code Audit Update

Added after the leftover-work audit:

- `RUN_WSNDS_RESUME_FINAL`
- `RUN_SHAP_ALIGNMENT_FINAL`
- `run_wsnds_final_with_resume()`
- `run_wsnds_qat_and_profile_entrypoint()`
- `run_edgeiiot_qat_and_profile_entrypoint()`
- `run_wsnds_rf_kd_shap_alignment_entrypoint()`
- `run_edgeiiot_shap_alignment_entrypoint()`
- `write_final_summary_tables()`

These additions close the earlier gaps where QAT, deployment profiling, WSN-DS resume, and final RF-to-KD SHAP alignment were only planned or helper-level scaffolded.


### Legacy v2.3 Guard

The notebook now has:

```python
RUN_LEGACY_V23_EXPERIMENTS = False
```

Keep this `False` for the final project route. It prevents a top-to-bottom notebook run from launching the old unresumable v2.3 multi-seed sweep before the final resumable entrypoints are available.

Set it to `True` only if you intentionally want to reproduce the historical v2.3 notebook outputs.

Edge-IIoTset ML categorical audit: the remaining object columns expand to only 35 one-hot columns on the selected ML CSV, so the current one-hot preprocessing is not a RAM blow-up risk for the ML file.

### Run All Mode Selector

The notebook is now set up for your normal Jupyter `Run All` workflow:

1. In Cell 2, set exactly one `FINAL_RUN_MODE`.
2. Keep `RUN_LEGACY_V23_EXPERIMENTS = False`.
3. Keep the old final `RUN_*` booleans `False` unless you are intentionally using the backwards-compatible flag path.
4. Click `Run All`.

Mode map:

```python
FINAL_RUN_MODE = "none"            # setup only
FINAL_RUN_MODE = "edge_smoke"      # Edge-IIoTset smoke check
FINAL_RUN_MODE = "edge_final"      # Edge-IIoTset 5-seed final
FINAL_RUN_MODE = "wsnds_final"     # WSN-DS 10-seed final
FINAL_RUN_MODE = "qat_profile"     # QAT + INT8 + CPU profile
FINAL_RUN_MODE = "shap_alignment"  # final SHAP alignment
```

`qat_profile` and `shap_alignment` require final checkpoints from `wsnds_final` and/or `edge_final` first.

### KD Hyperparameter Protection

The final route now protects against silent KD degradation:

- WSN-DS final calls/caches `run_wsnds_kd_grid_search_final()` before publication seeds unless explicit `kd_T` / `kd_alpha` values are passed.
- Edge-IIoTset smoke uses default KD settings to stay fast.
- Edge-IIoTset 5-seed final calls/caches `resolve_edgeiiot_kd_hyperparams()`, which runs an RF-to-student validation grid before the final seeds.

Do not report smoke metrics as final paper metrics. Smoke is only for runtime/preprocessing validation.

### New Output Locations

WSN-DS resumable final:

```text
final_runs/wsnds_10seed_resume/
```

QAT/profile outputs:

```text
final_runs/wsnds_qat_profile/
final_runs/edgeiiot_qat_profile/
```

SHAP alignment outputs:

```text
final_runs/shap_alignment/
```

### Current Remaining Work After Code Append

The code now includes the missing final-route mechanisms, but the project still needs the actual long experiments to be run and inspected:

1. Run Edge-IIoTset smoke.
2. Run Edge-IIoTset 5-seed final if smoke is stable.
3. Run WSN-DS resumable 10-seed final.
4. Run final QAT/profile entrypoints after checkpoints exist.
5. Run final SHAP alignment entrypoints after checkpoints exist.
6. Build final paper tables/figures from the saved CSV/JSON outputs.
7. Re-check novelty claims against current literature before writing/submission.
