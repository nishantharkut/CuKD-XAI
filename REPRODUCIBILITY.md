# Reproducibility Guide

This guide separates evidence inspection, software verification, model
regeneration, and physical hardware replay. These activities have different
data, compute, and equipment requirements. The current claim boundary is
defined by registry `cukd_fgds_evidence_registry_20260814_v3` in
`results/evidence_registry/fgds_20260814_current/`.

## Reproduction Levels

| Level | Purpose | Main requirements |
|---|---|---|
| A | Inspect reported values and claim boundaries | Git checkout; no training |
| B | Run tracked-source and CLI smoke checks | Python 3.11 and repository dependencies |
| C | Verify the sealed current registry and manifests | Hydrated Git LFS objects |
| D | Regenerate the primary ten-seed WSN-DS result | WSN-DS CSV, CUDA-capable PyTorch recommended |
| E | Regenerate extended WSN-DS analyses | Completed level-D run; substantial additional compute |
| F | Repeat conversion and fixed-point software audits | Model artifacts, C compiler, ONNX/OpenVINO stack |
| G | Repeat physical HIL | ESP32-C3, Arduino UNO R4 WiFi, Arduino CLI/toolchains, serial host |
| H | Repeat Edge-IIoTset analysis | Full Edge-IIoTset data; high RAM, disk, and compute |

The repository status is `passed_with_open_planned_work`. The ten-seed
scratch-controlled XAI experiment and final-ten-seed-lineage Wi-Fi campaign
are not completed evidence and must not be represented as reproduced results.

## 1. Clone And Git LFS

The repository uses Git LFS for datasets, packet captures, paper PDFs, and
selected model/result artifacts. Install Git LFS before cloning:

```text
git lfs install
git clone https://github.com/nishantharkut/TinyRF-KD.git
cd TinyRF-KD
git lfs fsck
```

A normal clone downloads all available LFS objects and can be large. For a
metadata-only review, skip automatic LFS download and inspect the tracked JSON,
CSV, Markdown, and source files first.

PowerShell:

```powershell
$env:GIT_LFS_SKIP_SMUDGE = "1"
git clone https://github.com/nishantharkut/TinyRF-KD.git
Remove-Item Env:GIT_LFS_SKIP_SMUDGE
Set-Location TinyRF-KD
```

Bash:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/nishantharkut/TinyRF-KD.git
cd TinyRF-KD
```

Hydrate only the primary WSN-DS dataset when regenerating the main result:

```text
git lfs pull --include="data/wsnds/WSN-DS.csv"
```

Hydrate the current WSN-DS model evidence when deep manifest verification is
required:

```text
git lfs pull --include="results/wsnds/confirmation_runs_v2/local_feature_group_10seed_20260811/**,results/wsnds/evidence_completion_20260811/**,results/wsnds/evidence_completion_20260812/**,results/wsnds/evidence_completion_20260813/**"
```

The Edge-IIoTset corpus is much larger. Pull `data/edge_iiot/**` only for the
secondary Edge analysis. Access to the remote LFS objects and compliance with
the original dataset licenses remain reviewer prerequisites.

## 2. Python Environment

The supported baseline is Python 3.11. The root requirements file uses bounded
version ranges so the repository remains installable across CPU and CUDA
hosts.

```text
python -m venv .venv
```

PowerShell activation:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Bash activation:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a CUDA rerun close to the recorded training environment, install the
PyTorch 2.5.1 CUDA 12.1 build from the official PyTorch package index before
installing `requirements.txt`.

The current primary execution contracts record this reference stack:

| Component | Recorded version |
|---|---|
| Python | 3.11.9 |
| NumPy | 2.4.6 |
| pandas | 3.0.3 |
| SciPy | 1.17.1 in the controlled full-route run |
| scikit-learn | 1.8.0 |
| imbalanced-learn | 0.14.2 in the controlled full-route run |
| joblib | 1.5.3 in the controlled full-route run |
| PyTorch | 2.5.1+cu121 |
| ONNX | 1.22.0 in the runtime run |
| ONNX Runtime | 1.28.0 in the runtime run |
| OpenVINO | 2026.3.0 build recorded in the runtime report |
| Training GPU | NVIDIA GeForce RTX 4050 Laptop GPU |

The authoritative records are the `environment` objects in:

- `results/wsnds/confirmation_runs_v2/local_feature_group_10seed_20260811/feature_group_10seed/execution_contract.json`
- `results/wsnds/evidence_completion_20260811/fgds_controlled_full_routes_10seed_v2/execution_contract.json`
- `results/runtime/onnx_openvino/wsnds/fgds_seed42_exact/runtime_report.json`

Deterministic PyTorch algorithms were enabled in the primary run. Exact model
bytes are not promised across different operating systems, CUDA libraries, or
GPU architectures. Protocol identity, split/scaler hashes, row counts,
per-seed metrics, and statistical conclusions are the cross-host comparison
targets.

## 3. Fast Reviewer Checks

These checks do not train models or access boards:

```text
python -m pytest tests/repository/test_active_cli_smoke.py tests/repository/test_repository_structure_smoke.py -q
python -m pytest tests/hardware tests/hardware_deployment_run -q
python -m compileall -q experiments deployment tests
python experiments/wsnds/leakage_free_rerun/verify_protected_sources.py verify
```

The first test validates current reviewer-facing entrypoints and current
evidence anchors. The hardware test set validates host-side contracts and
parsers; it does not replace physical MCU replay.

The complete test suite performs expensive integrity checks over large
artifacts and can run for many minutes. Run it only after all required LFS
objects have been hydrated:

```text
python -m pytest -q
```

An unsmudged LFS pointer is text metadata, not the underlying CSV, checkpoint,
or NumPy archive. Data-dependent failures in an LFS-skipped checkout do not
test the research logic.

## 4. Verify Current Sealed Evidence

Start with:

- `results/evidence_registry/fgds_20260814_current/EVIDENCE_REGISTRY.md`
- `results/evidence_registry/fgds_20260814_current/claim_boundaries.csv`
- `results/evidence_registry/fgds_20260814_current/evidence_registry.json`
- `results/evidence_registry/fgds_20260814_current/artifact_manifest.json`

After hydrating the referenced LFS objects, perform a read-only deep registry
verification:

```text
python -m experiments.evidence.build_fgds_evidence_registry --output-dir results/evidence_registry/fgds_20260814_current --verify-existing
```

The command must exit successfully without regenerating the registry.

## 5. Regenerate The Primary WSN-DS Result

The primary protocol identifier is
`wsnds_feature_group_split_train_only_scaler_10seed_v2`. It keeps exact raw
17-feature groups within one partition, fits `StandardScaler` on training
only, and evaluates seeds `42, 123, 456, 789, 1001, 2024, 3141, 5678, 8192,
9999` on one fixed split.

The recorded input identity is:

| Item | SHA-256 |
|---|---|
| WSN-DS CSV | `c65d05b983a85753bd62b6f76c5739fc52fe0c14cbb7644255cee4742f5ff7c9` |
| Split indices | `3d4061aa020122d4c5c5b2f7722de71e0c223c533869d3fdfa1f10784a0a0473` |
| Fitted scaler | `5303fb570aeb82ffaf88e2d4cceda94a7611762f67c86761990e6a4f09af5dd6` |

Run the non-training preflight first. Use a new, empty output root:

```text
python experiments/wsnds/leakage_free_rerun/run_feature_group_10seed_confirmation.py --mode preflight --dataset-csv data/wsnds/WSN-DS.csv --output-root results/reproductions/wsnds_fgds_review
```

Start the complete primary run only after preflight passes:

```text
python experiments/wsnds/leakage_free_rerun/run_feature_group_10seed_confirmation.py --mode duplicate-sensitivity --dataset-csv data/wsnds/WSN-DS.csv --output-root results/reproductions/wsnds_fgds_review --device cuda --confirm-training
```

`duplicate-sensitivity` is the preserved CLI name for the finalized ten-seed
feature-group-disjoint mode. The run writes
`results/reproductions/wsnds_fgds_review/feature_group_10seed/`. Existing
nonempty evidence is refused unless the strict `--resume` path verifies it.

Generate the paired analysis into another new directory:

```text
python experiments/wsnds/leakage_free_rerun/analyze_feature_group_confirmation.py --run-dir results/reproductions/wsnds_fgds_review/feature_group_10seed --output-dir results/reproductions/wsnds_fgds_review/feature_group_10seed_analysis
```

Compare the regenerated aggregate with the current values in the evidence
registry. The current split contains 262,197 training, 56,163 validation, and
56,301 test rows, with zero cross-partition exact feature-group overlap.

## 6. Extended WSN-DS Analyses

These are separate analyses. They do not replace the primary ten-seed result.

### Controlled full-route matrix

```text
python -m experiments.wsnds.evidence_completion.run_fgds_full_routes --dataset-csv data/wsnds/WSN-DS.csv --base-root results/reproductions/wsnds_fgds_review/feature_group_10seed --output-dir results/reproductions/wsnds_fgds_review/controlled_full_routes --device cuda --confirm-training
```

Finalize and seal the completed route matrix before any downstream analysis.
The finalizer corrects the stored-probability ECE representation, creates the
exact executed-source snapshot, and verifies every completed seed:

```text
python -m experiments.wsnds.evidence_completion.finalize_fgds_full_routes --dataset-csv data/wsnds/WSN-DS.csv --base-root results/reproductions/wsnds_fgds_review/feature_group_10seed --output-root results/reproductions/wsnds_fgds_review/controlled_full_routes --confirm-finalization
```

### Repeated-pattern sensitivity

```text
python -m experiments.wsnds.evidence_completion.analyze_fgds_group_balanced_routes --dataset-csv data/wsnds/WSN-DS.csv --base-root results/reproductions/wsnds_fgds_review/feature_group_10seed --full-route-root results/reproductions/wsnds_fgds_review/controlled_full_routes --full-route-executed-source results/reproductions/wsnds_fgds_review/controlled_full_routes/executed_full_routes_source.py --output-dir results/reproductions/wsnds_fgds_review/group_balanced_routes --confirm-analysis
```

### Behavioral response transfer

```text
python -m experiments.wsnds.evidence_completion.analyze_fgds_behavioral_transfer_logits --run-root results/reproductions/wsnds_fgds_review/feature_group_10seed --dataset data/wsnds/WSN-DS.csv --output-dir results/reproductions/wsnds_fgds_review/behavioral_transfer
```

### Additional declared analyses

Use `--help` before execution. Each command defaults to preflight or requires
an explicit confirmation flag:

```text
python -m experiments.wsnds.evidence_completion.run_fgds_multisplit_core_confirmation --help
python -m experiments.wsnds.evidence_completion.run_fgds_rfkd_hyperparameter_sensitivity --help
python -m experiments.wsnds.evidence_completion.run_fgds_exact_teacher_shap --help
python -m experiments.wsnds.evidence_completion.run_fgds_fixed_point_refinement --help
```

The tracked `run_fgds_controlled_xai_transfer` CLI is a planned experiment.
It is not part of the current completed registry.

## 7. Runtime And Fixed-Point Software

The current software evidence is under:

- `results/runtime/onnx_openvino/wsnds/fgds_seed42_exact/`
- `results/wsnds/evidence_completion_20260813/fgds_all_seed_fixed_point_audit_v1/`
- `deployment/firmware_export/wsnds_final_hil/`

Inspect the exact CLI contracts before writing new output:

```text
python deployment/firmware_export/wsnds_rfkd_hil/export_fgds_runtime.py --help
python -m deployment.firmware_export.wsnds_final_hil.export_final_seed42 --help
python -m deployment.firmware_export.wsnds_final_hil.audit_all_seeds --help
```

Deeply verify the completed all-seed fixed-point audit:

```text
python -m deployment.firmware_export.wsnds_final_hil.audit_all_seeds --output-dir results/wsnds/evidence_completion_20260813/fgds_all_seed_fixed_point_audit_v1 --verify-only
```

The audit contains 40 model-seed instances, with 26 complete gate passes and
14 retained gate failures. A failed gate is a result and must not be removed.

## 8. Physical HIL

The final USB campaign is under
`results/hardware_hil/final_fgds_seed42_v1/`. Its contract, common cohort,
bundles, session evidence, blocked-route record, and final report are tracked.
The campaign contains six gate-eligible model-board sessions and 337,806
full-test replay rows. Student B scratch was not executed because its
fixed-point quality gates failed.

The final HIL controller is:

```text
python -m deployment.final_hil --help
```

Its lifecycle is `contract`, `cohort`, `bundle`, `build-upload`, staged replay,
stage verification, session completion, campaign completion, and archive
verification. Every subcommand exposes its exact required inputs:

```text
python -m deployment.final_hil contract --help
python -m deployment.final_hil run-usb-stage --help
python -m deployment.final_hil verify-stage --help
python -m deployment.final_hil complete-session --help
python -m deployment.final_hil complete-campaign --help
```

Physical repetition additionally requires:

| Component | Recorded target |
|---|---|
| Boards | ESP32-C3 and Arduino UNO R4 WiFi |
| Host transport | USB serial for the final campaign |
| Stages | warmup 10, smoke 10, three timing cohorts of 1,000, full 56,301 |
| Model specimens | Seed-42 gate-eligible Student A scratch, Student A RF-KD, Student B RF-KD |

The tracked report supports exact fixed-reference replay for the recorded
board specimens. It does not establish multi-unit variability, energy use,
live packet capture, packet-to-feature extraction, or physical TelosB
execution.

The portable campaign archive is retained outside Git. Its recorded SHA-256
is `0361f70877b00a27df5e7c559d178a9f4fbdd37136c05dc7d68fdce0b4c79561`.
The repository alone can verify the tracked campaign records but cannot fetch
that external archive.

## 9. Edge-IIoTset

The current Edge result is secondary protocol-sensitivity evidence. It uses
40 inputs and 15 classes with 1,556,588/332,240/330,373 train/validation/test
rows. It records zero pre-encode group overlap and nonzero encoded exact-row
overlap after train-fitted representation processing.

The preserved runner is:

```text
python experiments/wsnds/leakage_free_rerun/run_leftover_e2e_closure.py --help
```

The current Edge stage is compute-intensive and uses the full DNN Edge-IIoTset
CSV. Its evidence is under
`results/leftover_e2e_closure/04_edge_group_aware/`. The protocol changes the
split, representation, RF teacher size, and training configuration relative
to the literature-style run. Differences between the two Edge protocols
cannot be attributed to grouping alone.

## 10. Reproduction Report

A reviewer should record:

1. Git commit and whether LFS objects were hydrated.
2. Python, package, operating-system, CPU, GPU, CUDA, compiler, and board-core versions.
3. Dataset SHA-256 values.
4. Output directory and execution-contract SHA-256.
5. Which level from this guide was completed.
6. Deviations from the recorded environment or physical hardware.
7. Any retained gate failures, timeouts, or incomplete stages.

Do not write regenerated output into a sealed evidence directory. Use a new
directory under `results/reproductions/` or outside the repository and compare
it with the tracked evidence after completion.
