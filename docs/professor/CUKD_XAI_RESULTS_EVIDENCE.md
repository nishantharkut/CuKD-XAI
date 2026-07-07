# CuKD-XAI Results and Evidence Ledger

This file records the current project results for professor discussion. Tables are rounded for readability where appropriate. Use the cited source files for exact unrounded values.

## Source Snapshot

| Item | Value |
|---|---|
| Repo branch checked | `main` |
| Repo sync status when prepared | `HEAD...origin/main = 0 0` |
| Main WSN-DS result source | `results/wsnds/final_results/2026-05-30-10seed-plus-j/wsnds_results_student_A.csv`, `results/wsnds/final_results/2026-05-30-10seed-plus-j/wsnds_results_student_B.csv` |
| Hardware result source | `results/hardware_hil/reports/final_postprocessing/final_postprocessing_analysis.md` and companion CSV files in the same folder |
| Edge-IIoT strict source | `results/edge_iiot/strict_generalization/edgeiiot_v23_config_rankings.csv` |
| Edge-IIoT literature-comparable source | `results/edge_iiot/literature_comparable/edgeiiot_v23_config_rankings.csv` |

## Evidence Index

| Evidence area | Source |
|---|---|
| WSN-DS Student A 10-seed table | `results/wsnds/final_results/2026-05-30-10seed-plus-j/wsnds_results_student_A.csv` |
| WSN-DS Student B 10-seed table | `results/wsnds/final_results/2026-05-30-10seed-plus-j/wsnds_results_student_B.csv` |
| WSN-DS combined J/co-distill output | `results/wsnds/final_results/2026-05-30-10seed-plus-j/cukd_xai_results_with_J.json`, `results/wsnds/final_results/2026-05-30-10seed-plus-j/j_only_results.json` |
| SHAP explanation alignment summary | `docs/professor/PROFESSOR_RESULTS_COMPARISON.md`, `results/wsnds/legacy_runs/2026-05-30-10seed/cukd_xai_results.json` |
| ONNX/OpenVINO runtime | `results/runtime/onnx_openvino/wsnds/runtime_from_existing_outputs/wsnds_existing_artifact_runtime_summary.csv` |
| HIL fidelity | `results/hardware_hil/reports/final_postprocessing/hil_fidelity.csv` |
| HIL cycles/MAC | `results/hardware_hil/reports/final_postprocessing/cycles_per_mac.csv` |
| Fixed-point footprint | `results/hardware_hil/reports/final_postprocessing/model_only_footprint.csv` |
| Compile footprint | `results/hardware_hil/reports/final_postprocessing/compile_framework_baseline.csv` |
| Quantization drift | `results/hardware_hil/reports/final_postprocessing/quantization_drift_summary.csv` |
| HIL source traceability | `results/hardware_hil/reports/final_postprocessing/evidence_traceability.csv` |
| MSP430 cross-compile footprint | `deployment/msp430/MSP430_CROSS_COMPILE_REPORT.md` |
| Edge-IIoT strict rankings | `results/edge_iiot/strict_generalization/edgeiiot_v23_config_rankings.csv` |
| Edge-IIoT literature-comparable rankings | `results/edge_iiot/literature_comparable/edgeiiot_v23_config_rankings.csv` |
| Edge-IIoT literature comparison | `docs/literature/comparison_tables/EDGEIIOT_LITERATURE_COMPARISON_FOR_PROFESSOR.md` |

## WSN-DS 10-Seed Results

Source: `results/wsnds/final_results/2026-05-30-10seed-plus-j/wsnds_results_student_A.csv` and `results/wsnds/final_results/2026-05-30-10seed-plus-j/wsnds_results_student_B.csv`.

### Student A Capacity: 17-32-16-5

| Config | Accuracy mean | Accuracy std | Macro-F1 mean | Macro-F1 std | Params | Size KB |
|---|---:|---:|---:|---:|---:|---:|
| `A_RF_500` | 0.996600 | 0.000051 | 0.978889 | 0.000251 | n/a | 85064.54 |
| `B_Full_MLP` | 0.987322 | 0.000399 | 0.923195 | 0.002375 | 69893 | 273.02 |
| `C2_CL_MLP_domain` | 0.985731 | 0.000481 | 0.913808 | 0.003158 | 69893 | 273.02 |
| `C_CL_MLP_loss` | 0.985886 | 0.000649 | 0.914730 | 0.003903 | 69893 | 273.02 |
| `C_CL_MLP_loss_ext` | 0.978667 | 0.023790 | 0.867257 | 0.154820 | 69893 | 273.02 |
| `C_CL_MLP_loss_fair` | 0.985886 | 0.000649 | 0.914730 | 0.003903 | 69893 | 273.02 |
| `D_Small_MLP` | 0.984708 | 0.002125 | 0.912303 | 0.006223 | 1189 | 4.64 |
| `E2_KD_from_MLP` | 0.986126 | 0.000426 | 0.916355 | 0.002759 | 1189 | 4.64 |
| `E_KD_from_RF` | 0.986875 | 0.000517 | 0.919971 | 0.003124 | 1189 | 4.64 |
| `F_KD_from_CL_MLP` | 0.985509 | 0.000334 | 0.912912 | 0.002126 | 1189 | 4.64 |
| `F_KD_from_CL_MLP_ext` | 0.983607 | 0.006659 | 0.906372 | 0.025109 | 1189 | 4.64 |
| `F_KD_from_CL_MLP_fair` | 0.985509 | 0.000334 | 0.912912 | 0.002126 | 1189 | 4.64 |
| `G_KD_random_pacing` | 0.985680 | 0.000463 | 0.913696 | 0.002806 | 1189 | 4.64 |
| `I_KD_from_SMOTE_MLP` | 0.985600 | 0.000322 | 0.913603 | 0.001918 | 1189 | 4.64 |
| `J_CoDistill_RF_CL` | 0.986514 | 0.000697 | 0.918062 | 0.004100 | 1189 | 4.64 |

Interpretation: Student A `E_KD_from_RF` is the strongest ultra-small RF-KD result: `0.986875` accuracy, `0.919971` macro-F1, `4.64 KB`.

### Student B Capacity: 17-64-32-5

| Config | Accuracy mean | Accuracy std | Macro-F1 mean | Macro-F1 std | Params | Size KB |
|---|---:|---:|---:|---:|---:|---:|
| `A_RF_500` | 0.996600 | 0.000051 | 0.978889 | 0.000251 | n/a | 85064.54 |
| `B_Full_MLP` | 0.987322 | 0.000399 | 0.923195 | 0.002375 | 69893 | 273.02 |
| `C2_CL_MLP_domain` | 0.985731 | 0.000481 | 0.913808 | 0.003158 | 69893 | 273.02 |
| `C_CL_MLP_loss` | 0.985886 | 0.000649 | 0.914730 | 0.003903 | 69893 | 273.02 |
| `C_CL_MLP_loss_ext` | 0.978667 | 0.023790 | 0.867257 | 0.154820 | 69893 | 273.02 |
| `C_CL_MLP_loss_fair` | 0.985886 | 0.000649 | 0.914730 | 0.003903 | 69893 | 273.02 |
| `D_Small_MLP` | 0.988835 | 0.000637 | 0.932169 | 0.003750 | 3397 | 13.27 |
| `E2_KD_from_MLP` | 0.988068 | 0.001078 | 0.927548 | 0.006821 | 3397 | 13.27 |
| `E_KD_from_RF` | 0.989114 | 0.001248 | 0.932808 | 0.007590 | 3397 | 13.27 |
| `F_KD_from_CL_MLP` | 0.987749 | 0.000583 | 0.925816 | 0.003423 | 3397 | 13.27 |
| `F_KD_from_CL_MLP_ext` | 0.985528 | 0.006661 | 0.917207 | 0.025905 | 3397 | 13.27 |
| `F_KD_from_CL_MLP_fair` | 0.987749 | 0.000583 | 0.925816 | 0.003423 | 3397 | 13.27 |
| `G_KD_random_pacing` | 0.987521 | 0.000915 | 0.924603 | 0.005503 | 3397 | 13.27 |
| `I_KD_from_SMOTE_MLP` | 0.988302 | 0.001364 | 0.929978 | 0.008311 | 3397 | 13.27 |
| `J_CoDistill_RF_CL` | 0.989133 | 0.001849 | 0.933526 | 0.011361 | 3397 | 13.27 |

Interpretation: Student B `J_CoDistill_RF_CL` has the best Student B mean macro-F1 (`0.933526`), while Student B `E_KD_from_RF` is nearly identical and simpler (`0.932808` macro-F1).

## Compression Results

Sources: WSN-DS CSVs above and `results/hardware_hil/reports/final_postprocessing/model_only_footprint.csv`.

| Model/artifact | Size basis | Size |
|---|---|---:|
| RF teacher | serialized model | 85064.54 KB |
| Full MLP | FP32 params | 273.02 KB |
| Student A | FP32 params | 4.64 KB |
| Student B | FP32 params | 13.27 KB |
| Student A RF-KD | fixed-point model-only params | 1348 B |
| Student B RF-KD | fixed-point model-only params | 3700 B |

| Compression comparison | Ratio |
|---|---:|
| Student A FP32 vs RF teacher | 18315x smaller |
| Student B FP32 vs RF teacher | 6411x smaller |
| Student A FP32 vs full MLP | 58.8x smaller |
| Student B FP32 vs full MLP | 20.6x smaller |
| Student A fixed-point params vs RF teacher | 64619x smaller |
| Student B fixed-point params vs RF teacher | 23542x smaller |
| Student A fixed-point params vs full MLP | 207.4x smaller |
| Student B fixed-point params vs full MLP | 75.6x smaller |

## SHAP Explanation Alignment

Source: `docs/professor/PROFESSOR_RESULTS_COMPARISON.md` and raw SHAP data in `results/wsnds/legacy_runs/2026-05-30-10seed/cukd_xai_results.json`.

| Metric | Value |
|---|---:|
| Teacher-student global SHAP rank Spearman rho | 0.0466 |
| Spearman p-value | 0.8591 |
| Bootstrap Spearman mean | 0.0015 |
| Bootstrap Spearman std | 0.1068 |

Interpretation: the student can preserve useful predictive performance without preserving the teacher's global feature-importance ordering. This is the explanation-faithfulness contribution.

## Deployment Runtime Results

Source: `results/runtime/onnx_openvino/wsnds/runtime_from_existing_outputs/wsnds_existing_artifact_runtime_summary.csv`.

### Deployment Term Definitions

| Term | Evidence role | Safe interpretation |
|---|---|---|
| ONNX | Standard exported neural-network artifact used for software inference outside the original PyTorch training script. | Shows exportability and runtime reproducibility of the student models; not MCU firmware by itself. |
| OpenVINO | Independent optimized software inference runtime used after ONNX export. | OpenVINO FP32 agreement of `1.0` vs ONNX means the converted runtime preserved the ONNX predictions in the checked runs. |
| Dynamic INT8 | Runtime/exporter-level post-training 8-bit quantization attempt. | It reduced some artifact sizes, but lowered macro-F1 and does not justify an INT8 speedup claim. |
| Fixed-point C | Separate integer firmware path generated for MCU-class replay. | This is the evidence path for ESP32-C3, Arduino R4, and MSP430F1611 memory feasibility. |
| HIL replay | Host-to-board serial replay of saved WSN-DS test vectors. | Validates fixed-point firmware execution against generated references; does not claim live packet capture or energy measurement. |

| Model | Variant | Accuracy | Macro-F1 | Size KB | Batch-1 p50 ms | Batch-1 p95 ms | OpenVINO agreement vs ONNX |
|---|---|---:|---:|---:|---:|---:|---:|
| `D_student_A_scratch` | `onnx_fp32` | 0.984555 | 0.906946 | 5.44 | 0.0267 | 0.0336 | n/a |
| `D_student_A_scratch` | `onnx_dynamic_int8` | 0.981174 | 0.891391 | 5.07 | 0.0393 | 0.0513 | n/a |
| `D_student_A_scratch` | `openvino_fp32_from_onnx` | 0.984555 | 0.906946 | 5.44 | 0.1289 | 0.2800 | 1.0 |
| `E_student_A_KD_from_RF` | `onnx_fp32` | 0.986370 | 0.917478 | 5.44 | 0.0275 | 0.0689 | n/a |
| `E_student_A_KD_from_RF` | `onnx_dynamic_int8` | 0.983665 | 0.900602 | 5.07 | 0.0395 | 0.1013 | n/a |
| `E_student_A_KD_from_RF` | `openvino_fp32_from_onnx` | 0.986370 | 0.917478 | 5.44 | 0.1307 | 0.4292 | 1.0 |
| `J_student_A_CoDistill_RF_CL` | `onnx_fp32` | 0.987776 | 0.925356 | 5.44 | 0.0279 | 0.0850 | n/a |
| `J_student_A_CoDistill_RF_CL` | `onnx_dynamic_int8` | 0.983238 | 0.899330 | 5.07 | 0.0395 | 0.0602 | n/a |
| `J_student_A_CoDistill_RF_CL` | `openvino_fp32_from_onnx` | 0.987776 | 0.925356 | 5.44 | 0.1318 | 0.4457 | 1.0 |
| `D_student_B_scratch` | `onnx_fp32` | 0.988505 | 0.930173 | 14.07 | 0.0289 | 0.0308 | n/a |
| `D_student_B_scratch` | `onnx_dynamic_int8` | 0.984075 | 0.904071 | 7.37 | 0.0392 | 0.0544 | n/a |
| `D_student_B_scratch` | `openvino_fp32_from_onnx` | 0.988505 | 0.930173 | 14.07 | 0.1313 | 0.4624 | 1.0 |
| `E_student_B_KD_from_RF` | `onnx_fp32` | 0.991050 | 0.944707 | 14.07 | 0.0285 | 0.0713 | n/a |
| `E_student_B_KD_from_RF` | `onnx_dynamic_int8` | 0.987491 | 0.923095 | 7.37 | 0.0396 | 0.0549 | n/a |
| `E_student_B_KD_from_RF` | `openvino_fp32_from_onnx` | 0.991050 | 0.944707 | 14.07 | 0.1304 | 0.4743 | 1.0 |
| `J_student_B_CoDistill_RF_CL` | `onnx_fp32` | 0.988932 | 0.931830 | 14.07 | 0.0284 | 0.0594 | n/a |
| `J_student_B_CoDistill_RF_CL` | `onnx_dynamic_int8` | 0.987224 | 0.921311 | 7.37 | 0.0401 | 0.1055 | n/a |
| `J_student_B_CoDistill_RF_CL` | `openvino_fp32_from_onnx` | 0.988932 | 0.931830 | 14.07 | 0.1302 | 0.2785 | 1.0 |

Interpretation: ONNX FP32 gives very small software-runtime artifacts and fast batch-1 p50 timings. Dynamic INT8 reduces size but lowers macro-F1 and does not support an INT8 speedup claim.

## Hardware HIL Results

Source: `results/hardware_hil/reports/final_postprocessing/hil_fidelity.csv`.

| Board | Model | Vectors | Accuracy | Macro-F1 | MCU vs fixed | MCU vs FP32 | Mean total us | P99 total us |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ESP32-C3 DevKitM-1 | Student A RF-KD | 56200 | 0.985623 | 0.914014 | 1.0 | 0.995000 | 118.40 | 125 |
| Arduino R4 WiFi | Student A RF-KD | 56200 | 0.985623 | 0.914014 | 1.0 | 0.995000 | 301.63 | 305 |
| ESP32-C3 DevKitM-1 | Student B RF-KD | 56200 | 0.986957 | 0.918099 | 1.0 | 0.993897 | 332.33 | 338 |
| Arduino R4 WiFi | Student B RF-KD | 56200 | 0.986957 | 0.918099 | 1.0 | 0.993897 | 791.57 | 795 |

Boundary: these are MCU-class USB serial replay tests using already extracted WSN-DS records. They are not live WSN packet capture, energy measurement, or physical TelosB deployment.

### HIL Cycles per MAC

Source: `results/hardware_hil/reports/final_postprocessing/cycles_per_mac.csv`.

| Board | Model | Clock MHz | MACs | Mean inference us | Inference cycles | Cycles/MAC | Total throughput ceiling/s |
|---|---|---:|---:|---:|---:|---:|---:|
| ESP32-C3 DevKitM-1 | Student A RF-KD | 160 | 1136 | 112.31 | 17970 | 15.82 | 8445.7 |
| Arduino R4 WiFi | Student A RF-KD | 48 | 1136 | 280.36 | 13457 | 11.85 | 3315.3 |
| ESP32-C3 DevKitM-1 | Student B RF-KD | 160 | 3296 | 325.08 | 52013 | 15.78 | 3009.1 |
| Arduino R4 WiFi | Student B RF-KD | 48 | 3296 | 770.68 | 36993 | 11.22 | 1263.3 |

The throughput ceiling is computed from measured on-device total processing time only. It is not radio/network throughput.

### Fixed-Point Model-Only Footprint

Source: `results/hardware_hil/reports/final_postprocessing/model_only_footprint.csv`.

| Model | Architecture | MACs | Weight bytes | Bias bytes | Param bytes | Numeric format |
|---|---|---:|---:|---:|---:|---|
| Student A RF-KD | 17-32-16-5 | 1136 | 1136 | 212 | 1348 | int8 weights + int32 biases + int16 activations |
| Student B RF-KD | 17-64-32-5 | 3296 | 3296 | 404 | 3700 | int8 weights + int32 biases + int16 activations |

### Compile Footprint

Source: `results/hardware_hil/reports/final_postprocessing/compile_framework_baseline.csv` and compile logs under `results/hardware_hil/compile_logs/`.

| Board | Model | Program bytes | Global bytes | Serial baseline program | Serial baseline globals | Program delta | Global delta |
|---|---|---:|---:|---:|---:|---:|---:|
| ESP32-C3 DevKitM-1 | Student A RF-KD | 278836 | 13556 | 274156 | 13164 | 4680 | 392 |
| Arduino R4 WiFi | Student A RF-KD | 56104 | 7128 | 51856 | 6740 | 4248 | 388 |
| ESP32-C3 DevKitM-1 | Student B RF-KD | 281192 | 13556 | 274156 | 13164 | 7036 | 392 |
| Arduino R4 WiFi | Student B RF-KD | 58440 | 7128 | 51856 | 6740 | 6584 | 388 |

### Quantization Drift

Source: `results/hardware_hil/reports/final_postprocessing/quantization_drift_summary.csv`.

| Model | Reference rows | Drift count | Drift fraction |
|---|---:|---:|---:|
| Student A RF-KD | 56200 | 281 | 0.005000 |
| Student B RF-KD | 56200 | 343 | 0.006103 |

Drift means fixed-point reference prediction differs from FP32 prediction. Because MCU-vs-fixed agreement is `1.0`, this is fixed-point quantization drift, not serial transport failure.

## MSP430F1611 Cross-Compile Evidence

Source: `deployment/msp430/MSP430_CROSS_COMPILE_REPORT.md`.

| Item | Evidence |
|---|---|
| Target | `msp430f1611`, TelosB/Tmote Sky-style MSP430 WSN mote class |
| Nominal target memory | 48 KB Flash, 10 KB RAM |
| Compiler | Mitto Systems MSP430-GCC `9.3.1.11`, support files `1.212`, `-Os` |
| Inference object `.text` / `.rodata` | `494 B` / `1,348 B` |
| Preprocess object `.text` / `.rodata` | `412 B` / `136 B` |
| Linked smoke firmware `text` | `2,842 B` |
| Linked smoke firmware `.data` / `.bss` | `0 B` / `6 B` |
| Conservative project-function stack estimate | about `248 B` during prediction |

Safe interpretation: this supports MSP430/TelosB-class memory feasibility of the model core. It does not prove real TelosB latency, energy, radio integration, or live WSN feature extraction.

## Edge-IIoT Strict Generalization Results

Source: `results/edge_iiot/strict_generalization/edgeiiot_v23_config_rankings.csv`.

Protocol note: strict route removes leakage/identifier/source/payload-style columns, uses multiclass `Attack_type`, train-only preprocessing, continuous-only scaling, and 5 seeds.

| Student | Config | Accuracy mean | Macro-F1 mean | Macro-F1 std | Params | Size KB |
|---|---|---:|---:|---:|---:|---:|
| `student_A_32_16` | `A_RF_500` | 0.883295 | 0.872913 | 0.000229 | n/a | 109879.57 |
| `student_B_64_32` | `A_RF_500` | 0.883295 | 0.872913 | 0.000229 | n/a | 109879.57 |
| `student_B_64_32` | `E_KD_from_RF` | 0.730900 | 0.711114 | 0.002519 | 5391 | 21.06 |
| `student_B_64_32` | `J_CoDistill_RF_CL` | 0.730351 | 0.710017 | 0.002997 | 5391 | 21.06 |
| `student_A_32_16` | `J_CoDistill_RF_CL` | 0.724284 | 0.700965 | 0.006508 | 2191 | 8.56 |
| `student_A_32_16` | `B_Full_MLP` | 0.724309 | 0.688005 | 0.003668 | 74511 | 291.06 |
| `student_B_64_32` | `B_Full_MLP` | 0.724309 | 0.688005 | 0.003668 | 74511 | 291.06 |
| `student_B_64_32` | `D_Small_MLP` | 0.723642 | 0.687790 | 0.001751 | 5391 | 21.06 |
| `student_A_32_16` | `E_KD_from_RF` | 0.710891 | 0.686257 | 0.010816 | 2191 | 8.56 |
| `student_B_64_32` | `F_KD_from_CL_MLP` | 0.719839 | 0.686026 | 0.003333 | 5391 | 21.06 |
| `student_B_64_32` | `E2_KD_from_MLP` | 0.718944 | 0.684991 | 0.002803 | 5391 | 21.06 |
| `student_A_32_16` | `C_CL_MLP_loss_fair` | 0.721377 | 0.682523 | 0.012110 | 74511 | 291.06 |
| `student_B_64_32` | `C_CL_MLP_loss_fair` | 0.721377 | 0.682523 | 0.012110 | 74511 | 291.06 |
| `student_A_32_16` | `E2_KD_from_MLP` | 0.712134 | 0.677202 | 0.004407 | 2191 | 8.56 |
| `student_A_32_16` | `F_KD_from_CL_MLP` | 0.707698 | 0.671344 | 0.016670 | 2191 | 8.56 |
| `student_A_32_16` | `D_Small_MLP` | 0.700059 | 0.664589 | 0.020008 | 2191 | 8.56 |

Interpretation: strict Edge-IIoT is a stress test and exposes a capacity-complexity bottleneck. Do not compare these numbers directly against binary or leakage-prone Edge-IIoT papers.

## Edge-IIoT Literature-Comparable Selected-Capacity Results

Source: `results/edge_iiot/literature_comparable/edgeiiot_v23_config_rankings.csv`.

Protocol note: completed 5-seed selected-capacity route.

| Student | Config | Accuracy mean | Macro-F1 mean | Macro-F1 std | Params | Size KB |
|---|---|---:|---:|---:|---:|---:|
| `student_A_32_16` | `A_RF_500` | 0.982513 | 0.889193 | 0.001087 | n/a | 101114.94 |
| `student_B_64_32` | `A_RF_500` | 0.982513 | 0.889193 | 0.001087 | n/a | 101114.94 |
| `student_C_128_64` | `A_RF_500` | 0.982513 | 0.889193 | 0.001087 | n/a | 101114.94 |
| `student_A_32_16` | `A_LightGBM` | 0.986699 | 0.886116 | 0.000330 | n/a | 15224.05 |
| `student_B_64_32` | `A_LightGBM` | 0.986699 | 0.886116 | 0.000330 | n/a | 15224.05 |
| `student_C_128_64` | `A_LightGBM` | 0.986699 | 0.886116 | 0.000330 | n/a | 15224.05 |
| `student_C_128_64` | `E_KD_from_RF` | 0.968509 | 0.824382 | 0.000906 | 15631 | 61.06 |
| `student_C_128_64` | `E3_KD_from_LightGBM` | 0.968457 | 0.824192 | 0.003131 | 15631 | 61.06 |
| `student_C_128_64` | `D_Small_MLP` | 0.968681 | 0.823916 | 0.000334 | 15631 | 61.06 |
| `student_B_64_32` | `E3_KD_from_LightGBM` | 0.967974 | 0.821965 | 0.000996 | 5775 | 22.56 |
| `student_C_128_64` | `J_CoDistill_RF_CL` | 0.967752 | 0.820980 | 0.002759 | 15631 | 61.06 |
| `student_B_64_32` | `E_KD_from_RF` | 0.967217 | 0.819664 | 0.002634 | 5775 | 22.56 |
| `student_B_64_32` | `D_Small_MLP` | 0.967889 | 0.818949 | 0.004886 | 5775 | 22.56 |
| `student_B_64_32` | `J_CoDistill_RF_CL` | 0.966568 | 0.816876 | 0.003451 | 5775 | 22.56 |
| `student_A_32_16` | `E3_KD_from_LightGBM` | 0.966091 | 0.814329 | 0.003869 | 2383 | 9.31 |
| `student_A_32_16` | `J_CoDistill_RF_CL` | 0.965785 | 0.812653 | 0.002589 | 2383 | 9.31 |
| `student_A_32_16` | `E_KD_from_RF` | 0.965615 | 0.811837 | 0.003168 | 2383 | 9.31 |
| `student_A_32_16` | `D_Small_MLP` | 0.965471 | 0.809107 | 0.003892 | 2383 | 9.31 |

Interpretation: Student C RF-KD is the best current selected-capacity student by macro-F1 (`0.824382`, `61.06 KB`). Student B LightGBM-KD is the smaller strong point (`0.821965`, `22.56 KB`). Student A remains the smallest (`9.31 KB`) but lower in macro-F1.

## Verified Base and Related-Paper Comparison

Source rule for this table: paper-side numbers are taken from local PDFs in `docs/literature/papers/`, primary publisher pages, arXiv pages, or the already generated Edge-IIoT comparison file. CuKD-XAI-side numbers are taken from the result files cited earlier in this document. `NR` means the paper did not report that item in the checked source.

Important comparison caution: many IDS papers report a plain `F1-score` without saying whether it is macro, weighted, or micro. For this project, macro-F1 is central because WSN-DS and Edge-IIoT are imbalanced. Do not compare our macro-F1 directly with an unspecified paper F1 unless the averaging basis is known.

### WSN-DS and WSN/IoT IDS Papers

| Paper / work | Dataset and method | Reported paper result | Paper resource evidence | Closest CuKD-XAI comparison | Safe professor interpretation | Source |
|---|---|---:|---|---|---|---|
| Almomani et al. 2016, WSN-DS dataset paper | Original WSN-DS dataset and baseline IDS models | ANN baseline around `96.6%` accuracy in older repo context; exact macro-F1 not used here | Model footprint/latency NR | Student B `J`: accuracy `98.913%`, macro-F1 `93.353%`, `13.27 KB`; Student A `E`: accuracy `98.688%`, macro-F1 `91.997%`, `4.64 KB` | Use this as the dataset origin and baseline context, not as modern SOTA. | https://doi.org/10.1155/2016/4731953 |
| Ghadi et al. 2024, IEEE Access review | WSN security review | No direct benchmark table for CuKD-XAI comparison | Review motivates the conflict between energy efficiency and security complexity | Use as motivation only | This is the base/background paper, not a result baseline. It supports the resource-constrained WSN framing. | `docs/literature/papers/base_paper.pdf`, https://doi.org/10.1109/ACCESS.2024.3355312 |
| Talukder et al. 2024, MLSTL-WSN | WSN-DS, SMOTE-Tomek, RF/ML models | Binary accuracy `99.78%`; multiclass accuracy `99.92%`; WiSTL RF accuracy/precision/recall/F1 `99.92%` | Time complexity discussed; absolute model size/MCU latency NR | Student B `J`: accuracy `98.913%`, macro-F1 `93.353%`, `13.27 KB`; Student A `E`: accuracy `98.688%`, macro-F1 `91.997%`, `4.64 KB` | They are stronger for raw WSN-DS accuracy. They also show SHAP/RF-style WSN-DS work exists, so our novelty must be compression, deployment evidence, and explanation-faithfulness audit. | https://doi.org/10.1007/s10207-024-00833-z |
| Talukder et al. 2025, Scientific Reports KMS+PCA+RFC | WSN-DS and TON-IoT, KMeans-SMOTE + PCA + RF | WSN-DS accuracy `99.94%`, F1 `99.94%`; TON-IoT accuracy/F1 `99.97%` | Complexity analysis claims reduced training/prediction time; absolute model size/MCU evidence NR | Student B `J`: accuracy `98.913%`, macro-F1 `93.353%`, `13.27 KB` | This is a clear WSN-DS accuracy-SOTA comparator. CuKD-XAI should not claim better accuracy; it should claim a much smaller deployable student. | `docs/literature/papers/sota_wsn_ds_2025.pdf`, https://doi.org/10.1038/s41598-025-87028-1 |
| Birahim et al. 2025, IEEE Access PSO explainable ensemble | WSN-DS, PSO + RF/DT/KNN ensemble, SMOTE-Tomek, LIME/SHAP | Accuracy `99.73%`; precision/recall/F1 `99.72%` | Size/latency NR in checked source | Student B `J`: accuracy `98.913%`, macro-F1 `93.353%`, `13.27 KB` | They are stronger on raw detection and already use SHAP/LIME. Our safe XAI novelty is not "first XAI"; it is teacher-student explanation-faithfulness after compression. | https://doi.org/10.1109/ACCESS.2025.3528341 |
| Pandey et al. 2025, Scientific Reports TS-RF | WSN-DS, CICIDS2017, CIC-IoT 2023; Tabu Search optimized RF | WSN-DS optimized RF accuracy improves from `99.42%` to `99.67%`; optimized WSN-DS macro average about `0.98`, weighted average about `1.00` | Google Colab/PC setup reported; model size/MCU evidence NR | Student B `J`: accuracy `98.913%`, macro-F1 `93.353%`, `13.27 KB` | Again, stronger pure detection. It helps show the WSN-DS leaderboard is saturated by optimized tree models, so the paper should focus on compression and deployability. | https://doi.org/10.1038/s41598-025-03498-3 |
| Nguyen et al. 2024, Sensors GSWO-CatBoost | WSN-DS, WSNBFSF, NSL-KDD, CICIDS2017; GSWO feature selection + CatBoost | WSN-DS headline accuracy `99.65%`; comparison table reports accuracy `99.65%`, precision `97.27%`, recall `97.78%`, F1 `97.47%` | Inference time `16 ms` for WSN-DS in the paper comparison table | Student B `J`: accuracy `98.913%`, macro-F1 `93.353%`, `13.27 KB`; Student A `E`: `4.64 KB` | This paper is both accuracy and speed oriented. Our advantage is explicit tiny model storage and firmware/HIL proof, not raw accuracy. | https://doi.org/10.3390/s24113339 |
| Alfarra and AbuSamra 2025 hybrid IDS | WSN-DS plus ns-3 50-node LoRa simulation; integer rule filter + pruned INT8 CNN-LSTM gateway | Accuracy `98%`, macro-F1 `0.93`, minority recalls `0.84-0.95` | Sensor rule filter about `0.05 mJ` per packet; INT8 CNN-LSTM `28 mJ` per 32-window batch; average latency `42 ms`, p95 `55 ms`; T50 `69 days` | Student B `J`: accuracy `98.913%`, macro-F1 `93.353%`, `13.27 KB`; HIL replay is not energy measurement | This is close to our macro-F1 range and stronger on energy/lifetime realism. We should not imply we have battery evidence; we have stronger fixed-point/HIL reproducibility. | `docs/literature/papers/alfarra_2025.pdf`, https://doi.org/10.37936/ecti-cit.2025194.263081 |
| Xiao and Duan 2025 metaheuristic soft-voting ensemble | WSN-style IDS, DNN + CatBoost soft voting, QIO/OOA tuning | QIO-enabled DCQI test accuracy `95.62%`, test F1-score `96.07%` | QIO-enhanced ensemble training about `175 min`; estimated storage about `125 MB` | Student B `J`: accuracy `98.913%`, macro-F1 `93.353%`, `13.27 KB` | Our accuracy and storage are favorable, but their F1 averaging is not the same as our macro-F1 unless explicitly confirmed. Use this as a resource-aware ensemble comparator. | `docs/literature/papers/Metaheuristically optimized deep soft-voting ensemble for explainable and resource-aware signal processing in wireless sensor network intrusion detection.pdf`, https://doi.org/10.1007/s11760-025-04880-4 |
| Vidhya and Varunadevi 2026 binarized simplicial CNN | WSN-DS 5-class DoS/IDS, binarized simplicial CNN with 1-bit weight direction | Abstract reports relative accuracy gains of `4.05%`, `7.52%`, and `2.91%` over named baselines; absolute accuracy and size not verified from full PDF | Full model size/latency not verified; full PDF not available in repo context | Not enough verified numeric detail for direct comparison | This matters because it invalidates any "first compression on WSN-DS" claim. Use only as a compression-related competitor unless the full PDF is obtained. | https://doi.org/10.1002/dac.70277 |
| Rana et al. 2024 WSN-DS ML benchmarking | WSN-DS case-study benchmarking with standard ML models | Exact metrics not retained in current evidence file | Resource evidence NR | Not enough verified numeric detail for direct comparison | Use as a general WSN-DS benchmarking citation only if the chapter is checked before manuscript use. | https://doi.org/10.1007/978-981-99-8129-8_15 |
| Salmi and Oughdir 2022/2023 CNN-LSTM WSN DoS | WSN-DS / WSN DoS deep-learning baseline | Older repo notes cite around `97%` accuracy | Resource evidence NR | Student B `J`: `98.913%` accuracy and `13.27 KB` | Use as an earlier deep-learning WSN-DS/DoS baseline, but verify the exact venue/year/metric before manuscript use. | Listed in `docs/archive/updates/2026-04-12/CuKD_XAI_EXECUTION_PLAN.md`; source URL noted in repo: https://journalofbigdata.springeropen.com/articles/10.1186/s40537-023-00692-w |
| Benaddi et al. 2025, SHAP-guided Kronecker KD | TON-IoT, not WSN-DS; SHAP feature pruning + KD + Kronecker student | Table II: student FP32 accuracy `0.9968`, macro-F1 `0.9863`; INT8 accuracy `0.9969`, macro-F1 `0.9867` | Table II: teacher `769,922` params / `3021.53 KB`; student `3,042` params / `22.29 KB`; FP32 mean latency `1.29 ms` | CuKD-XAI WSN-DS Student B `13.27 KB`, Student A `4.64 KB`; Edge-IIoT Student C selected-capacity `61.06 KB` | This is the closest KD/compression/XAI-style related work, but on TON-IoT and with Kronecker layers. It helps justify that compression+XAI IDS is active; our distinction is WSN-DS RF-to-student compression plus HIL/MSP430 evidence and SHAP faithfulness audit. | `docs/literature/papers/benaddi_2025.pdf`, https://arxiv.org/abs/2512.19488 |

Note on Benaddi et al.: the checked arXiv HTML/PDF table reports `3,042` student parameters, while a method paragraph in the same paper text states `1,282` parameters. Use the table values for comparison unless the professor asks specifically about that inconsistency.

### Broader XAI/KD/IoT IDS Landscape

This table is for discussion breadth. Most rows are not direct WSN-DS competitors. Use it to answer "what else exists in the field?" and to show that SHAP, LIME, KD, attention, rule induction, and concept drift are already active directions. Do not use these rows as direct metric competitors unless the dataset, class protocol, and F1 averaging basis match.

Primary local source: `docs/archive/updates/2026-04-12/XAI_IDS_WSN_IoT_Literature_Map_2023_2026.md`. Verification levels follow that file and `docs/archive/updates/2026-04-12/CuKD_XAI_EXECUTION_PLAN.md`: **PDF/local** means a local PDF or extracted text exists; **primary-link** means the row is backed by a DOI/publisher/arXiv URL in the repo; **context-only** means useful for positioning but not enough for a numeric claim.

| Work | Main idea | Dataset(s) | Reported result/resource in local literature map | Relation to CuKD-XAI | Use in professor discussion |
|---|---|---|---|---|---|
| Hossain and Islam 2025, federated SHAP-KD | Federated IoT botnet IDS with SHAP-based feature knowledge sharing | IoT botnet/N-BaIoT context | Accuracy `99.99%` across botnet types in local map | Shows SHAP+KD exists for IoT, but not WSN-DS | Use to avoid overclaiming "first SHAP+KD IDS"; claim WSN-DS-specific compression/evidence instead. |
| Okey et al. 2026 RAID-KL | 1D-CNN teacher-student KD with adaptive KL-JS loss and SHAP | CICIoT2023, CICIoMT2024, NIMSLABIoT2025 | `91.24%` compression, `11.3%` CPU reduction, `64.33%` memory reduction | KD/compression IDS related work, different datasets | Shows resource-aware KD is active; CuKD-XAI must differentiate via WSN-DS + HIL/MSP430. |
| AL-Nomasy et al. 2025 DistillGuard | Transformer teacher to lightweight student with gradient-based explanation | IDS datasets not fully specified in local map | Superior accuracy/efficiency vs SOTA in local map | KD+XAI context, not WSN-DS | Mention only as transformer-KD IDS context unless full paper is checked. |
| IEEE TCE 2025 lightweight explainable KD | Teacher-student KD for resource-constrained consumer devices | N-BaIoT, CIC-IDS2023 | Student accuracy `99.87%` on N-BaIoT, `98.71%` on CIC-IDS2023; `263k -> 120k` params, `1.00 MB -> 471.67 KB` | KD compression related work, not WSN-DS | Useful to show our KB-scale WSN-DS students are much smaller than many KD IDS students. |
| Benaddi et al. 2025 | SHAP feature pruning + Kronecker KD | TON-IoT | Student `0.9968` accuracy, `0.9863` macro-F1, `22.29 KB` | Closest methodological analogue | Already in direct table; keep as must-cite. |
| Self-attention XAI framework 2025 | SA-DNN with learnable feature gating plus SHAP/LIME | BoT-IoT, N-BaIoT, UNSW-NB15 | `99.3%` BoT-IoT accuracy, `99.6%` N-BaIoT accuracy | XAI/attention IDS context | Shows attention/XAI is active; not a compression comparator. |
| Dong et al. 2026 SHAP-NGBoost | SHAP-enhanced natural-gradient boosting | UNSW-NB15, CICIDS2017, N-BaIoT | Outperforms mainstream baselines in local map | Tree/boosting XAI context | Supports claim that tree/boosting methods dominate raw metrics. |
| Rajkumar and Shalinie 2025 QNN+SHAP | Quantum neural IDS interpreted by SHAP | CIC-IoT2022, SDN-DDoS24 | `0.98` expectation value; `113 ms` latency | Emerging XAI method, not directly comparable | Mention only if professor asks about non-classical IDS/XAI trends. |
| Versatile XAI framework 2025 | ANOVA + SHAP + LIME, XGBoost detector | CIC-DDoS2019, CICIoT2023, 5G PFCP | F1 `>=99%`; LIME time `36s -> 4.9s`; about `70%` dimensionality reduction | XAI efficiency context | Shows XAI runtime/feature reduction is studied outside WSN-DS. |
| Alabbadi and Bajaber 2025 | CNN/DNN/TabNet with SHAP/LIME over IoT streams | TON-IoT sub-datasets | Network `99.24%`; IoT average `99.96%` in local map | XAI on TON-IoT, not WSN-DS | Good context for SHAP/LIME saturation in IoT IDS. |
| Bin Hulayyil et al. 2025 | Explainable AI-based IDS in IoT systems | CUSmartHome, IoT23 | Efficiency demonstrated; exact metric not retained | General IoT XAI context | Do not use numerically unless source is reopened. |
| GA-optimized LSTM/GRU + LIME 2025 | Genetic algorithm optimized recurrent IDS with LIME | IoT network datasets | `99.84%` accuracy; model size `108.42 KB` after dynamic quantization | Lightweight XAI IDS, different datasets | Shows compression/quantization exists; not WSN-DS/KD comparable. |
| Mohale and Obagbuwa 2025 | ML IDS evaluated with LIME/SHAP/ELI5 | UNSW-NB15 | Best accuracy `87%` with XGBoost/CatBoost | XAI evaluation context | Useful reminder that dataset difficulty changes absolute scores. |
| Hermosilla et al. 2025 | SHAP vs LIME forensic IDS comparison | UNSW-NB15 | XGBoost `97.8%` accuracy; SHAP stability/coherence findings | Explanation-quality context | Supports need for explanation evaluation beyond accuracy. |
| Gaspar and Silva 2024 | SHAP/LIME applicability on MLP IDS | IoT IDS datasets | Demonstrates SHAP/LIME applicability on MLP | Directly relevant to our student MLP explanations | Shows SHAP on MLP is not new; our audit is teacher-student faithfulness. |
| Samout et al. 2025 | XAI models for big-data WSN IDS | KDD Cup 99, agricultural/WSN framing | High accuracy/precision/recall/F1 in local map | WSN framing but not WSN-DS | Context only; not a direct WSN-DS result. |
| Rule-induction IoT IDS 2025 | Inherently interpretable rule induction with ensembles | CIC-IDS2017, CICIoT2023 | XGBoost `99.91%` on CIC-IDS2017; `98.54%` on CICIoT2023 | Alternative explainability path | Helps explain why SHAP is only one XAI route. |
| Trustworthy adaptive AI IIoT 2025 | Online/adaptive ensemble with SHAP | Industrial IoT traffic | `96.4%` accuracy, `2.1%` FPR, `35 ms` edge detection time | IIoT real-time/XAI context | Useful if professor asks about online/adaptive IDS gap. |
| Khan et al. 2025 VANET SHAP transformer | Transformer + SHAP for VANET IDS | VeReMi extension | Multiclass `96.15%`; binary `98.28%` | Non-WSN XAI transformer context | Not directly comparable; shows SHAP/transformers in IDS are active. |
| L-XAIDS 2025 | LIME + ELI5 framework | Network IDS datasets | Higher detection rate and lower FPR vs three approaches | LIME-focused XAI context | Supports that LIME is co-dominant with SHAP. |
| CNN + SHAP/LIME IoT 2025 | Lightweight 1D-CNN with SHAP/LIME | TON-IoT | Resource-efficient; SHAP used for feature selection | Lightweight CNN XAI context | Not a KD/WSN-DS comparator. |
| XAI-IDS Industry 5.0 survey 2025 | Survey of XAI IDS and adversarial XAI | 135 studies | Documents XAI as both IDS aid and attack surface | Survey context | Use to justify why explanation auditing matters. |
| Explainable DL IDS IoT survey 2026 | PRISMA review of explainable DL IDS | 129 studies | Proposes detection-performance/resource-efficiency/explanation-quality trilemma | Survey context | Very useful framing for CuKD-XAI's three-way trade-off. |
| XAI+IDS systematic review 2025 | Systematic review of XAI integration in IDS | General IDS | Rule/tree XAI preferred; trade-offs remain challenging | Survey context | Use in related work, not as metric comparator. |

Broad-field conclusion from the local literature map: **SHAP/XAI for IDS is common, KD/compression for IoT IDS is active, and WSN-DS raw accuracy is saturated. The safest CuKD-XAI novelty is the WSN-DS-specific combination of KB-scale RF-to-neural-student compression, SHAP teacher-student faithfulness audit, and deployment evidence.**

### Edge-IIoT Papers

The Edge-IIoT comparison is supporting evidence, not the main WSN-DS paper claim. Source table: `results/edge_iiot/literature_metric_gap/edgeiiot_literature_metric_comparison.md`.

| Paper / work | Reported paper result | Our closest selected-capacity result | Safe professor interpretation | Source |
|---|---|---|---|---|
| Ferrag et al. 2022 Edge-IIoTset | Original 15-class DNN accuracy `94.67%`; RF `80.83%`, SVM `77.61%`, KNN `79.18%`, DT `67.11%` | Student C RF-KD accuracy `96.85%`, macro-F1 `82.44%`, `61.06 KB` | Our selected-capacity route beats the original Edge-IIoTset DNN accuracy, but F1 was not reported in that original baseline table. | https://doi.org/10.1109/ACCESS.2022.3165809 |
| Diab et al. 2025 hardware-aware Edge-IIoT | LightGBM accuracy `95.25%`, F1 `94.74%`, `74.93 KB` flash; HW-NAS 1D-CNN accuracy `96.73%`, F1 `97.24%`, `190.34 KB` flash | Student C RF-KD accuracy `96.85%`, macro-F1 `82.44%`, `61.06 KB` | Our accuracy/storage are competitive, but their F1 basis is unspecified in the generated comparison, so avoid claiming macro-F1 superiority. | https://arxiv.org/abs/2512.02272 |
| Gao et al. 2026 lightweight TCN | TCN with 22 features: accuracy `93.79%`, F1 `93.13%`, `16.25 KB`; all features: accuracy `94.24%`, F1 `93.71%`, `25.37 KB` | Student A LightGBM-KD accuracy `96.61%`, `9.31 KB`; Student B LightGBM-KD accuracy `96.80%`, `22.56 KB` | Our selected-capacity Edge-IIoT accuracy/size are strong against this lightweight comparator, but F1 averaging still needs cautious wording. | https://doi.org/10.3390/electronics15050938 |

### Comparison Takeaway

| Area | Current position |
|---|---|
| WSN-DS accuracy SOTA | CuKD-XAI does not beat top WSN-DS RF/CatBoost/ensemble papers reporting about `99.65-99.94%` accuracy. |
| WSN-DS compression | CuKD-XAI gives a strong size/performance trade-off: `4.64 KB` and `13.27 KB` FP32 students against an `85064.54 KB` RF teacher, plus fixed-point model-only footprints of `1348 B` and `3700 B`. |
| WSN-DS XAI | SHAP/LIME already appear in related WSN IDS papers. The safe novelty is explanation-faithfulness auditing after compression, not first-use XAI. |
| Hardware evidence | Current evidence supports software export, fixed-point C, HIL replay, and MSP430 memory feasibility. It does not support live packet capture or energy claims. |
| Edge-IIoT | Selected-capacity Edge-IIoT results are promising, but they are supporting/generalization evidence and must be compared with protocol and F1-basis caveats. |

## Final Paper-Safe Result Claims

Use these exact claim boundaries:

1. **Main WSN-DS compression claim:** Student A RF-KD reaches `0.986875` accuracy and `0.919971` macro-F1 at `4.64 KB`; Student B co-distillation reaches `0.989133` accuracy and `0.933526` macro-F1 at `13.27 KB`.
2. **Teacher comparison claim:** The RF teacher remains stronger (`0.996600` accuracy, `0.978889` macro-F1) but is about `85064.54 KB`.
3. **Fixed-point/HIL claim:** RF-KD fixed-point firmware replay exactly matches the generated fixed-point reference for all 56,200 vectors on ESP32-C3 and Arduino R4.
4. **MSP430 claim:** The Student A fixed-point core cross-compiles for MSP430F1611 with a linked smoke firmware footprint of `2,842 B` text, `0 B` data, and `6 B` bss.
5. **XAI claim:** SHAP rank agreement between teacher and student is near zero, so predictive compression does not imply explanation-faithfulness.
6. **Edge-IIoT claim:** Edge-IIoT validates protocol sensitivity and capacity limits; selected-capacity results are promising but should be framed separately from WSN-DS.

## Claims Not Supported by Current Evidence

- Best WSN-DS accuracy.
- First SHAP use on WSN-DS.
- Physical TelosB deployment.
- Live WSN packet capture.
- On-mote packet-to-feature extraction.
- Energy or battery-life measurement.
- INT8 speedup.
- Co-distillation universally improving over RF-KD.


