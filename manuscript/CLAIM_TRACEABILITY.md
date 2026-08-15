# CuKD-XAI Manuscript Claim Traceability

> **Earlier manuscript ledger:** This file records the evidence used by the
> current historical `main.tex` draft. It is not the current research claim
> ledger. Use `results/evidence_registry/fgds_20260814_current/` until the
> manuscript is rebuilt.

This ledger records the evidence used by `main.tex`. Generated table fragments are reproducible views; the CSV, JSON, source code, and raw logs listed here remain authoritative only for that earlier manuscript lineage.

## WSN-DS Dataset and Protocol

| Manuscript item | Repository evidence |
|---|---|
| 374,661 records, 19 original columns, 17 model features, five labels | `data/wsnds/WSN-DS.csv`; loading and column-selection code in `experiments/wsnds/main/cukd_xai_colab.py` |
| Class counts: 10,049 Blackhole, 3,312 Flooding, 14,596 Grayhole, 340,066 Normal, 6,638 TDMA | `data/wsnds/WSN-DS.csv` |
| Fixed split: 262,252 train, 56,209 validation, 56,200 test | `deployment/firmware_export/wsnds_rfkd_hil/generated_student_a_rfkd_hil_full/equivalence_report.json`; Student B equivalent |
| Ten seeds and fixed split | `results/wsnds/final_results/2026-05-30-10seed-plus-j/cukd_xai_results_with_J.json` |
| Scaler fitted before splitting | execution order in `experiments/wsnds/main/cukd_xai_colab.py` (`fit_transform` precedes `train_test_split`) |
| RF, full-MLP, Student A/B architectures, loss functions, training schedule, and curriculum definitions | `experiments/wsnds/main/cukd_xai_colab.py` |

## WSN-DS Predictive Results

| Result family | Authoritative artifact | Manuscript rendering |
|---|---|---|
| All 15 Student A configurations | `results/wsnds/final_results/2026-05-30-10seed-plus-j/wsnds_results_student_A.csv` | `generated/wsn_student_a_all_rows.tex` |
| All 15 Student B configurations | `results/wsnds/final_results/2026-05-30-10seed-plus-j/wsnds_results_student_B.csv` | `generated/wsn_student_b_all_rows.tex` |
| Accuracy, macro-F1, weighted-F1, per-class F1, parameters, and size | same Student A/B CSV files | main WSN tables and appendices |
| Co-distillation paired Wilcoxon comparisons | `results/wsnds/final_results/2026-05-30-10seed-plus-j/cukd_xai_results_with_J.json`, key `wilcoxon_results_with_J`; duplicate alias rows removed and Holm correction derived across ten unique comparisons | `generated/wsn_codistill_wilcoxon_rows.tex` |
| Size-performance plot | Student A/B CSV files | `generated/fig_wsn_tradeoff.tex` |

The primary reported WSN-DS points are Student A RF-KD (1,189 parameters, 4.64 KB, macro-F1 0.919971 +/- 0.003124) and Student B co-distillation (3,397 parameters, 13.27 KB, macro-F1 0.933526 +/- 0.011361). The appendices retain every stored route, including the archived alias rows.

## Explanation Audit

| Item | Repository evidence | Manuscript rendering |
|---|---|---|
| Audited pair | final-seed Student A `F_KD_from_CL_MLP` and RF reference in `cukd_xai_results_with_J.json`; this student was distilled from the curriculum MLP, not the RF |
| Global rank agreement | `shap_results.ranking_agreement_spearman` and `ranking_agreement_p` | macros in `generated/shap_summary_macros.tex` |
| Repeated-subsample sensitivity | `shap_results.bootstrap_spearman_mean` and `bootstrap_spearman_std` | same generated macro file |
| All 17 teacher and student ranks | `student_global_importance` and `teacher_global_importance` | `generated/shap_rank_rows.tex` |

This evidence applies to one preserved Student A curriculum-KD/RF-reference pair. It measures cross-model explanation alignment, not attribution transfer from that student's actual teacher, and is not evidence that every route or deployed RF-KD model has the same relationship.

## Software Conversion and Runtime

| Result family | Repository evidence | Rows used |
|---|---|---:|
| PyTorch FP32 and QAT accuracy/macro-F1 | `results/runtime/onnx_openvino/wsnds/wsnds_qat_summary.csv` | 6 |
| ONNX FP32 and dynamic-INT8 accuracy, macro-F1, artifact size, and latency | `results/runtime/onnx_openvino/wsnds/runtime_from_existing_outputs/wsnds_existing_artifact_runtime_summary.csv` | 12 of 18 |
| OpenVINO accuracy, macro-F1, size, latency, and ONNX agreement | same runtime summary | 6 of 18 |

The runtime table is artifact-level evidence, not a ten-seed aggregate. Dynamic INT8 reduces serialized size in the archived runs but does not provide a measured host-latency advantage. OpenVINO predictions agree with their ONNX sources in the stored comparison.

Train-only seed-42 RF-KD ONNX Runtime evidence (separate from the archived multi-route table) is under `results/runtime/onnx_openvino/wsnds/train_only_seed42_copy/train_only_runtime_results.json`. FP32 ONNX matches PyTorch predictions exactly; dynamic INT8 is size-oriented and incurs macro-F1 loss. OpenVINO was not re-executed for the train-only copy package.

## Fixed-Point Export and Hardware

| Item | Repository evidence |
|---|---|
| Student A/B integer-reference agreement, FP32 agreement, accuracy, F1, and latency | four `full_56200_metrics.json` files under `results/hardware_hil/board_replay/` |
| Complete sequence integrity | four corresponding `full_56200_sequence.json` files; each records 56,200 expected, 56,200 completed, no missing/duplicate/unexpected IDs, and 56,200 `OK` statuses |
| HIL model and replay seeds | `results/runtime/onnx_openvino/wsnds/wsnds_deployment_results.json` records proof seed 9999; both firmware `export_summary.json` files record test-vector seed 42 and ordered indices 0--56,199; both `preprocess_metadata.json` files record the split contract with random state 42 |
| Model-only fixed-point bytes and MAC counts | `results/hardware_hil/reports/final_postprocessing/model_only_footprint.csv` |
| Inference cycles, cycles/MAC, and throughput | `results/hardware_hil/reports/final_postprocessing/cycles_per_mac.csv` |
| Full compile footprint and serial-baseline delta | raw files in `results/hardware_hil/compile_logs/`; derived `compile_framework_baseline.csv` |
| Quantization drift and per-class drift | `quantization_drift_summary.csv` and `quantization_drift_by_class.csv` in the final post-processing directory |
| Final combined hardware report | `results/hardware_hil/reports/final_hardware_hil_results_table.md` |
| MSP430F1611 text/data/BSS and stack estimates | `deployment/msp430/MSP430_CROSS_COMPILE_REPORT.md` |

The four board/model pairs are Student A RF-KD and Student B RF-KD on ESP32-C3 and Arduino UNO R4 WiFi. All four reproduce the generated fixed-point reference on every replayed record. The experiment starts from already extracted 17-feature WSN-DS records and measures model-core preprocessing and inference, not live radio traffic or energy.

## Train-Only Seed-42 Confirmation (Tier 1.5)

This package closes the train-fitted-scaler gap for **deployment and conversion evidence**. It does **not** replace the archived ten-seed predictive tables.

| Item | Repository evidence |
|---|---|
| Train-only RF-KD deployment seed 42 (A/B FP32) | `results/wsnds/confirmation_runs_v2/deployment_seed_42/aggregate_results.json`; `results/wsnds/confirmation_runs_v2/deployment_seed_42/seed_42/student_A_KD_from_RF_fp32.pt`; `results/wsnds/confirmation_runs_v2/deployment_seed_42/seed_42/student_B_KD_from_RF_fp32.pt` |
| Preprocessing / split contracts | `results/wsnds/confirmation_runs_v2/deployment_seed_42/preprocessing_contract.json`; `results/wsnds/confirmation_runs_v2/deployment_seed_42/split_indices.npz`; `results/wsnds/confirmation_runs_v2/deployment_seed_42/scaler_parameters.npz` |
| QAT refine probe (not selected for HIL) | `results/wsnds/confirmation_runs_v2/deployment_seed_42_qat/qat_refinement_report.json` |
| Fixed-point export A/B (copy pipeline) | `deployment/firmware_export/wsnds_rfkd_hil/generated_train_only_student_A_seed42_copy/equivalence_report.json`; `deployment/firmware_export/wsnds_rfkd_hil/generated_train_only_student_B_seed42_copy/equivalence_report.json` |
| Host ONNX FP32 + dynamic INT8 | `results/runtime/onnx_openvino/wsnds/train_only_seed42_copy/train_only_runtime_summary.csv`; `results/runtime/onnx_openvino/wsnds/train_only_seed42_copy/deployable_runtime_artifacts/E_student_A_KD_from_RF_train_only.onnx`; `results/runtime/onnx_openvino/wsnds/train_only_seed42_copy/deployable_runtime_artifacts/E_student_B_KD_from_RF_train_only.onnx` |
| Host OpenVINO FP32 from ONNX | `results/runtime/onnx_openvino/wsnds/train_only_seed42_copy/train_only_openvino_results.json`; `results/runtime/onnx_openvino/wsnds/train_only_seed42_copy/train_only_openvino_summary.csv` |
| Master tier-1.5 report | `results/hardware_hil/train_only_scaler_copy/TIER15_MASTER_REPORT.json`; `results/hardware_hil/train_only_scaler_copy/TIER15_MASTER_REPORT.md` |
| Four-pair HIL full 56,200 | `results/hardware_hil/train_only_scaler_copy/four_pair_summary.json`; `results/hardware_hil/train_only_scaler_copy/pi5_arduino_r4_student_A/full_56200_metrics.json`; `results/hardware_hil/train_only_scaler_copy/pi5_arduino_r4_student_B/full_56200_metrics.json`; `results/hardware_hil/train_only_scaler_copy/pi5_esp32c3_student_A/full_56200_metrics.json`; `results/hardware_hil/train_only_scaler_copy/pi5_esp32c3_student_B/full_56200_metrics.json` |
| Completeness ledger | `results/runtime/onnx_openvino/wsnds/train_only_seed42_copy/train_only_tier15_completeness.json` |
| Compile footprints (arduino-cli on Pi) | `results/hardware_hil/train_only_scaler_copy/compile_evidence/compile_footprint_summary.json`; per-pair `results/hardware_hil/train_only_scaler_copy/compile_evidence/esp32c3_student_A_footprint.json`; `results/hardware_hil/train_only_scaler_copy/compile_evidence/esp32c3_student_B_footprint.json`; `results/hardware_hil/train_only_scaler_copy/compile_evidence/arduino_r4_student_A_footprint.json`; `results/hardware_hil/train_only_scaler_copy/compile_evidence/arduino_r4_student_B_footprint.json` |
| Smoke reconfirm after reflash (10-row, all OK) | `results/hardware_hil/train_only_scaler_copy/compile_evidence/smoke_esp32c3_student_A/smoke_10_sequence.json`; `results/hardware_hil/train_only_scaler_copy/compile_evidence/smoke_esp32c3_student_B/smoke_10_sequence.json`; `results/hardware_hil/train_only_scaler_copy/compile_evidence/smoke_arduino_r4_student_A/smoke_10_sequence.json`; `results/hardware_hil/train_only_scaler_copy/compile_evidence/smoke_arduino_r4_student_B/smoke_10_sequence.json` |

Measured anchors (train-only seed 42):

- FP32 test macro-F1: A 0.9485, B 0.9449 (`results/wsnds/confirmation_runs_v2/deployment_seed_42/aggregate_results.json`)
- ONNX FP32 agreement vs PyTorch: 1.0 for A and B
- Dynamic INT8 ONNX macro-F1: A 0.8938, B 0.9066
- Fixed-point vs FP32 prediction agreement at export: A 0.9919, B 0.9905
- MCU vs fixed reference on all four pairs: 1.0; n=56200; all statuses OK
- MCU macro-F1 (fixed path): A 0.9244, B 0.9180
- Mean total latency (µs): ESP32-C3 A 116.5 / B 320.3; R4 A 301.5 / B 791.4
- Compile flash (bytes): ESP32-C3 A 281776 / B 284132; R4 A 56384 / B 58736
- Compile RAM globals (bytes): ESP32-C3 A/B 13592; R4 A/B 7128
- Smoke reconfirm after reflash: 10/10 OK on all four model--board pairs

Copy-only helpers used for export/HIL/compile do not modify original strict tools or archived board_replay rows.

## Edge-IIoTset Robustness

| Protocol | Metadata | Complete result matrix |
|---|---|---|
| Strict: 157,800 rows, 43 features, 15 classes, five seeds | `results/edge_iiot/strict_generalization/edgeiiot_v23_metadata.json` | `results/edge_iiot/strict_generalization/edgeiiot_v23_config_rankings.csv` |
| Literature-oriented: 2,219,201 rows, 49 features, 15 classes, five seeds | `results/edge_iiot/literature_comparable/edgeiiot_v23_metadata.json` | `results/edge_iiot/literature_comparable/edgeiiot_v23_config_rankings.csv` |
| Protocol-aware literature comparison | `results/edge_iiot/literature_metric_gap/edgeiiot_literature_metric_comparison.md` | comparison and discussion sections |

The best compact macro-F1 is 0.711114 in the strict protocol and 0.824382 in the literature-oriented protocol. Because the source file, retained fields, row count, and input dimension differ, the manuscript treats the gap as protocol sensitivity rather than a training-method improvement.

## Prior-Work Evidence

Bibliographic metadata comes from local paper PDFs where available and primary DOI, publisher, arXiv, or official project records. The comparison table preserves each paper's stated metric label; an unspecified F1 is not reclassified as macro-F1. WSN-DS and Edge-IIoTset scores are not presented as directly rank-comparable when split, balancing, retained features, or task construction differ.

Local supporting papers are indexed under `docs/literature/`. The cited primary records and exact citation keys are in `references.bib`.

## Automated Integrity Checks

`scripts/validate_manuscript.py` runs `scripts/build_evidence.py` and verifies:

- all expected result-family row counts;
- all four full HIL sequence documents;
- all six compiler logs against the four-row derived footprint table;
- all 17 SHAP features;
- ten unique co-distillation comparisons and selected Holm-adjusted probability anchors;
- required manuscript sections and generated table inclusions;
- bibliography coverage for every citation key;
- source-derived numerical anchors for WSN-DS, HIL, and both Edge-IIoTset protocols.

The evidence manifest stores SHA-256 digests for every consumed artifact. A changed source is therefore visible even when its filename and row count remain unchanged.

## Feature-Group Sensitivity (5 seeds)

| Item | Repository evidence |
|---|---|
| Aggregate metrics | `results/wsnds/confirmation_runs_v2/remote_winterfell_feature_group_5seed_20260805/feature_group_5seed/aggregate_results.json` |
| Compact summary | `results/wsnds/confirmation_runs_v2/feature_group_5seed_summary_copy.json` |

Anchors: Student A RF-KD macro-F1 $0.91411\pm0.00687$; Student B RF-KD $0.92815\pm0.00741$; mean KD-minus-scratch macro-F1 deltas near zero. Boundary: descriptive sensitivity under exact-feature-group disjoint partitions; not a matched significance test against the archived random-row route.

## Unresolved Evidence Limits

- Multi-seed WSN predictive tables still use the archived pre-split scaler; the seed-42 RF-KD deployment/HIL/ONNX/OpenVINO chain and the five-seed feature-group package use train-only scaling but do not rewrite the archived ten-seed tables.
- Train-only confirmation is single-seed and is not a replacement ten-seed confidence interval for the random-row protocol.
- Seed variability uses one fixed data split (per protocol) and does not estimate multi-split uncertainty.
- Wilcoxon comparisons are exploratory, Holm-adjusted across ten unique co-distillation tests, and based on only ten paired seeds from one split.
- The SHAP audit covers one Student A pair, not every seed and route.
- HIL uses USB replay of extracted tabular features; no packet-to-feature radio pipeline or board-level energy trace exists.
- MSP430F1611 evidence is cross-compile memory feasibility, not a physical TelosB run.


## Train-Only Leftover Closure (primary predictive + J)

| Item | Authoritative artifact |
|---|---|
| Train-only 10-seed aggregates (no J) | `results/wsnds/leakage_free_rerun/main_10seed_full_aggregate_copy/` |
| Train-only 10-seed + J merge | `results/wsnds/leakage_free_rerun/main_10seed_train_only_plus_j/` |
| Reconstructed J base package | `results/wsnds/leakage_free_rerun/main_10seed_v2_reconstructed/` |
| Seed-5678 CL-ext re-run | `results/leftover_e2e_closure/02_seed5678_clext/` |
| Per-route set_seed D/E | `results/leftover_e2e_closure/03_per_route_set_seed/` |
| Edge group-aware literature | `results/leftover_e2e_closure/04_edge_group_aware/` |
| Claim freeze (updated) | `results/paper_strength_e2e/06_claim_freeze.json` |
| Deployment SHAP (RF-KD) | `results/paper_strength_e2e/shap_train_only_deployment/` |
| Train-only four-pair HIL | `results/hardware_hil/train_only_scaler_copy/` |

Primary Student A RF-KD train-only: macro-F1 0.92034 ± 0.00618.
Primary Student B RF-KD train-only: macro-F1 0.93917 ± 0.01225.
