# CuKD-XAI Results and Literature Comparison

Prepared: 2026-05-31  
Project: CuKD-XAI for lightweight explainable intrusion detection

This document summarizes the results currently available in the repository and compares them against the closest WSN-DS, IoT IDS, XAI, KD, and Edge-IIoT literature. The goal is to present the work honestly: this is not an accuracy-leaderboard paper. The strongest contribution is compression, deployment portability, and explanation-faithfulness analysis for a full multiclass WSN-DS setting, with Edge-IIoT used as a stress/generalization check.

## 1. Executive Summary

The best primary WSN-DS story is:

| Claim | Current evidence |
|---|---:|
| Full WSN-DS multiclass evaluation | 5 classes, 10 publication seeds |
| Best ultra-small Student A `(32,16)` | `E_KD_from_RF`: accuracy `0.9869`, macro-F1 `0.9200`, `1,189` params, `4.64 KB` FP32 |
| Best balanced Student B `(64,32)` | `J_CoDistill_RF_CL`: accuracy `0.9891`, macro-F1 `0.9335`, `3,397` params, `13.27 KB` FP32 |
| Best reliable Student B alternative | `E_KD_from_RF`: accuracy `0.9891`, macro-F1 `0.9328`, `3,397` params, `13.27 KB` FP32 |
| RF teacher | accuracy `0.9966`, macro-F1 `0.9789`, serialized size `85,064.54 KB` |
| Compression vs RF teacher | Student A: about `18,315x` smaller; Student B: about `6,410x` smaller |
| Deployment proof | ONNX FP32 artifacts run with p50 batch-1 latency about `0.027-0.029 ms`; OpenVINO outputs match ONNX exactly |
| Explanation audit | SHAP teacher-student rank agreement is near zero: Spearman rho `0.0466`, p `0.8591` |

The safest headline:

> CuKD-XAI compresses a high-performing WSN-DS intrusion detector into a KB-scale neural student while preserving useful multiclass accuracy, and it shows that predictive distillation does not automatically preserve teacher feature-importance reasoning.

Avoid saying:

> "We beat WSN-DS SOTA accuracy."

That is false. WSN-DS accuracy is already saturated near `99.7-99.94%` in RF/tree/ensemble papers.

## 2. Current Result Inventory

| Result group | Status | Main source artifact |
|---|---|---|
| WSN-DS v2.3 10-seed baseline | Complete | `results/wsnds/legacy_runs/2026-05-30-10seed/wsnds_results_student_A.csv`, `wsnds_results_student_B.csv` |
| WSN-DS v2.3 10-seed with J merge | Complete on `origin/main` | `results/wsnds/final_results/2026-05-30-10seed-plus-j/wsnds_results_student_A.csv`, `wsnds_results_student_B.csv` |
| WSN-DS deployment/runtime proof | Complete on `origin/main` | `results/runtime/onnx_openvino/wsnds/runtime_from_existing_outputs/wsnds_existing_artifact_runtime_summary.csv` |
| WSN-DS fixed-point C export path | Added, reproducible exporter | `deployment/msp430/export_wsnds_student_a_rfkd_int8.py`, `deployment/msp430/wsnds_student_a_rfkd_int8_inference.c` |
| Edge-IIoT strict v2.3 stress test | Complete, 5 seeds | `results/edge_iiot/strict_generalization/edgeiiot_v23_results_student_A.csv`, `student_B.csv` |
| Edge-IIoT literature-comparable selected-capacity run | Preliminary, seeds 42, 123, and 456 checkpoints available | `results/edge_iiot/literature_comparable/edgeiiot_v23_seed_42_checkpoint.json`, `edgeiiot_v23_seed_123_checkpoint.json`, `edgeiiot_v23_seed_456_checkpoint.json` on `origin/main` |
| SHAP teacher-student explanation alignment | Complete for WSN-DS v2.3 | `results/wsnds/legacy_runs/2026-05-30-10seed/cukd_xai_results.json` |

## 3. WSN-DS Primary Results: 10 Seeds

Source: J-inclusive result files from `results/wsnds/final_results/2026-05-30-10seed-plus-j/`.

### 3.1 Student A: Ultra-Small `(32,16)`, 1,189 Params, 4.64 KB FP32

| Config | Accuracy mean | Macro-F1 mean | Macro-F1 std | Size | Interpretation |
|---|---:|---:|---:|---:|---|
| `A_RF_500` | `0.9966` | `0.9789` | `0.0003` | `85,064.54 KB` | High-capacity teacher |
| `B_Full_MLP` | `0.9873` | `0.9232` | `0.0024` | `273.02 KB` | Full neural teacher |
| `D_Small_MLP` | `0.9847` | `0.9123` | `0.0062` | `4.64 KB` | Scratch tiny baseline |
| `E2_KD_from_MLP` | `0.9861` | `0.9164` | `0.0028` | `4.64 KB` | KD from MLP teacher |
| `E_KD_from_RF` | `0.9869` | `0.9200` | `0.0031` | `4.64 KB` | Best Student A |
| `F_KD_from_CL_MLP` | `0.9855` | `0.9129` | `0.0021` | `4.64 KB` | CL-KD did not help here |
| `J_CoDistill_RF_CL` | `0.9865` | `0.9181` | `0.0041` | `4.64 KB` | Good, but below RF-KD for Student A |

Key Student A deltas:

| Comparison | Macro-F1 delta |
|---|---:|
| `E_KD_from_RF` vs scratch `D` | `+0.0077` |
| `E_KD_from_RF` vs full MLP `B` | `-0.0032` |
| `E_KD_from_RF` vs RF teacher | `-0.0589` |
| `J_CoDistill_RF_CL` vs `E_KD_from_RF` | `-0.0019` |

Interpretation: for the smallest student, simple calibrated RF distillation is stronger than co-distillation. Student A is the best "tiny footprint" result: `4.64 KB` FP32, `1.16 KB` raw INT8-weight equivalent, about `18,315x` smaller than the serialized RF teacher.

### 3.2 Student B: Balanced `(64,32)`, 3,397 Params, 13.27 KB FP32

| Config | Accuracy mean | Macro-F1 mean | Macro-F1 std | Size | Interpretation |
|---|---:|---:|---:|---:|---|
| `A_RF_500` | `0.9966` | `0.9789` | `0.0003` | `85,064.54 KB` | High-capacity teacher |
| `B_Full_MLP` | `0.9873` | `0.9232` | `0.0024` | `273.02 KB` | Full neural teacher |
| `D_Small_MLP` | `0.9888` | `0.9322` | `0.0038` | `13.27 KB` | Scratch compact baseline |
| `E2_KD_from_MLP` | `0.9881` | `0.9275` | `0.0068` | `13.27 KB` | MLP-KD underperforms RF-KD |
| `E_KD_from_RF` | `0.9891` | `0.9328` | `0.0076` | `13.27 KB` | Strong reliable KD result |
| `F_KD_from_CL_MLP` | `0.9877` | `0.9258` | `0.0034` | `13.27 KB` | CL-KD not competitive |
| `J_CoDistill_RF_CL` | `0.9891` | `0.9335` | `0.0114` | `13.27 KB` | Best mean, but close to RF-KD |

Key Student B deltas:

| Comparison | Macro-F1 delta |
|---|---:|
| `J_CoDistill_RF_CL` vs scratch `D` | `+0.0014` |
| `J_CoDistill_RF_CL` vs RF-KD `E` | `+0.0007` |
| `J_CoDistill_RF_CL` vs full MLP `B` | `+0.0103` |
| `J_CoDistill_RF_CL` vs RF teacher | `-0.0454` |
| `E_KD_from_RF` vs full MLP `B` | `+0.0096` |

Interpretation: Student B is the strongest publishable accuracy-compression point. Co-distillation has the best mean, but the improvement over RF-KD is very small relative to its std. The conservative claim is:

> RF-KD is robust and reproducible; co-distillation is capacity-dependent and marginally improves the balanced student in this run.

## 4. Deployment and Runtime Proof

Source: `results/runtime/onnx_openvino/wsnds/runtime_from_existing_outputs/wsnds_existing_artifact_runtime_summary.csv`.

Important caution: these are artifact-level deployment results from existing trained artifacts, not 10-seed means.

| Model artifact | Runtime | Variant | Accuracy | Macro-F1 | Serialized size | p50 latency batch-1 | OpenVINO agreement |
|---|---|---|---:|---:|---:|---:|---:|
| `D_student_A_scratch` | ONNX Runtime | FP32 | `0.9846` | `0.9069` | `5.44 KB` | `0.0267 ms` | n/a |
| `E_student_A_KD_from_RF` | ONNX Runtime | FP32 | `0.9864` | `0.9175` | `5.44 KB` | `0.0275 ms` | n/a |
| `J_student_A_CoDistill` | ONNX Runtime | FP32 | `0.9878` | `0.9254` | `5.44 KB` | `0.0279 ms` | n/a |
| `D_student_B_scratch` | ONNX Runtime | FP32 | `0.9885` | `0.9302` | `14.07 KB` | `0.0289 ms` | n/a |
| `E_student_B_KD_from_RF` | ONNX Runtime | FP32 | `0.9910` | `0.9447` | `14.07 KB` | `0.0285 ms` | n/a |
| `J_student_B_CoDistill` | ONNX Runtime | FP32 | `0.9889` | `0.9318` | `14.07 KB` | `0.0284 ms` | n/a |
| All above | OpenVINO | FP32 from ONNX | same as ONNX | same as ONNX | same as ONNX | about `0.130 ms` | `1.0000` |

Deployment interpretation:

- ONNX FP32 is the cleanest deployability result: tiny artifacts, very low CPU inference latency, and no accuracy conversion loss.
- OpenVINO is useful as portability proof, not speed proof in this run. It exactly matches ONNX predictions but is slower for batch-1 on this CPU setup.
- Dynamic INT8 ONNX reduced size slightly but lowered macro-F1 and was not faster in this CPU measurement. Do not claim INT8 speedup from the current evidence.
- This is still not a real WSN mote or microcontroller experiment. Present it as software deployment proof, not hardware deployment proof.

### 4.1 Fixed-Point C Export Path

After the hardware-focused review, the embedded headline should use Student A `E_KD_from_RF`, not Config J. Student A RF-KD is the best ultra-small WSN-DS result: macro-F1 `0.9200`, `1,189` params, and `4.64 KB` FP32. Student A `J_CoDistill_RF_CL` is lower at macro-F1 `0.9181`, so J belongs in the ablation/capacity discussion, not the embedded headline.

The `deployment/firmware_export/wsnds_rfkd_hil/` path is now an end-to-end software export proof, not just a small weight dump. The runner loads the trained Student A RF-KD state dict, reproduces the v2.3 WSN-DS preprocessing metadata, exports int8/calibrated-int16 C headers, exports integer StandardScaler metadata, generates held-out representative test vectors, compiles the dependency-free C preprocessing and inference sources, runs a generated C self-test, and writes an equivalence report comparing fixed-point behavior against the FP32 tensor forward pass.

Expected generated parameter storage remains about `1,348 bytes` before compiler/code overhead. The additional StandardScaler proof represents normalization as `17` integer subtracts, `17` integer multiplies, `17` shifts, and `17` saturations per sample after the raw WSN-DS features already exist. The full 56,200-vector software export on `origin/main` showed fixed-vs-FP32 agreement `0.9947`, fixed accuracy `0.98635` vs FP32 accuracy `0.98637`, zero input saturation (`0 / 955400`), and final fixed logits inside `[-19507, 9228]`, safely within signed int16 range. The correct claim is stronger but still bounded: this proves a reproducible software path to a calibrated integer C model core plus integer normalization metadata. It is still not a physical TelosB deployment, and WSN-DS feature extraction on the mote remains outside the current artifact.

### 4.2 MSP430F1611 Cross-Compile Evidence

The fixed-point preprocessing and inference core was cross-compiled for `msp430f1611` using Mitto Systems MSP430-GCC `9.3.1.11`, MSP430 support files `1.212`, and `-Os`. This is target-toolchain footprint evidence for TelosB/Tmote Sky-class constraints, not a physical hardware deployment.

Object-level footprint:

| Object | `.text` | `.rodata` | `.data` | `.bss` |
|---|---:|---:|---:|---:|
| `wsnds_preprocess_int16_msp430.o` | `412 B` | `136 B` | `0 B` | `0 B` |
| `wsnds_student_a_rfkd_int8_inference_msp430.o` | `494 B` | `1,348 B` | `0 B` | `0 B` |

Linked smoke firmware footprint:

| Metric | Value |
|---|---:|
| `msp430-elf-size` text | `2,842 B` |
| `.data` | `0 B` |
| `.bss` | `6 B` |
| total `text + data + bss` | `2,848 B` |
| `.rodata` section | `1,484 B` |
| `.text` section | `1,356 B` |

Compiler-reported stack usage with `-fstack-usage`:

| Function | Stack usage |
|---|---:|
| `main` | `104 B` |
| `cukd_standardize_raw_q` | `26 B` |
| `cukd_dense_i8_q15` | `26 B` |
| `cukd_forward_q15` | `106 B` |
| `cukd_predict_q15` | `12 B` |

Interpretation: the linked smoke firmware uses about `2.8 KB` Flash-class storage and `6 B` static RAM/heap before stack, which is comfortably inside an MSP430F1611/TelosB-class 48 KB Flash and 10 KB RAM budget. A conservative project-function call-chain estimate is about `248 B` stack during prediction, excluding ABI helper internals, interrupt nesting, and OS/network-stack pressure.

Disassembly confirmed wider-arithmetic helper routines: `__mulhisi2`, `__mulsi2`, `__mspabi_mpyll`, `__mspabi_srai`, `__mspabi_sral`, and `__mspabi_srall`. This is an important limitation: MSP430 memory feasibility is now supported, but latency and energy still require physical mote measurement.

Paper-safe wording:

> We additionally cross-compiled the fixed-point preprocessing and inference core for MSP430F1611. A linked smoke firmware required `2,842 B` Flash-class `text` storage, `0 B` `.data`, and `6 B` `.bss`, with bounded compiler-reported project-function stack usage. Disassembly confirms helper routines for wider integer arithmetic, so this supports target-toolchain memory feasibility but not final latency or energy claims.

## 5. Edge-IIoT Strict Generalization Stress Test

Source: `results/edge_iiot/strict_generalization/edgeiiot_v23_results_student_A.csv` and `student_B.csv`.

This run uses a stricter Edge-IIoT adapter: leakage/identifier/source/payload columns removed, multiclass `Attack_type`, train-only preprocessing fit, continuous-only scaling, and 5 seeds. It is intentionally harder than many literature protocols.

### 5.1 Strict Edge-IIoT Student A

| Config | Accuracy mean | Macro-F1 mean | Macro-F1 std | Params | Size |
|---|---:|---:|---:|---:|---:|
| `A_RF_500` | `0.8833` | `0.8729` | `0.0002` | n/a | `109,879.57 KB` |
| `D_Small_MLP` | `0.7001` | `0.6646` | `0.0200` | `2,191` | `8.56 KB` |
| `E_KD_from_RF` | `0.7109` | `0.6863` | `0.0108` | `2,191` | `8.56 KB` |
| `J_CoDistill_RF_CL` | `0.7243` | `0.7010` | `0.0065` | `2,191` | `8.56 KB` |

Student A strict Edge interpretation:

- Best Student A is `J_CoDistill_RF_CL`.
- `J` improves over scratch by `+0.0364` macro-F1.
- `J` remains `0.1719` macro-F1 behind the RF teacher.
- The model is about `12,838x` smaller than the serialized RF teacher.

### 5.2 Strict Edge-IIoT Student B

| Config | Accuracy mean | Macro-F1 mean | Macro-F1 std | Params | Size |
|---|---:|---:|---:|---:|---:|
| `A_RF_500` | `0.8833` | `0.8729` | `0.0002` | n/a | `109,879.57 KB` |
| `D_Small_MLP` | `0.7236` | `0.6878` | `0.0018` | `5,391` | `21.06 KB` |
| `E_KD_from_RF` | `0.7309` | `0.7111` | `0.0025` | `5,391` | `21.06 KB` |
| `J_CoDistill_RF_CL` | `0.7304` | `0.7100` | `0.0030` | `5,391` | `21.06 KB` |

Student B strict Edge interpretation:

- Best Student B is `E_KD_from_RF`, very slightly above `J`.
- RF-KD improves over scratch by `+0.0233` macro-F1.
- RF-KD remains `0.1618` macro-F1 behind the RF teacher.
- The model is about `5,218x` smaller than the serialized RF teacher.

Strict Edge conclusion:

The strict Edge-IIoT result should be framed as a stress test. It demonstrates that the compact KD students still improve over scratch under a harder 15-class benchmark, but also exposes a capacity-complexity limit. Do not compare the `0.70-0.71` strict Edge macro-F1 directly against papers that use binary labels, leakage-prone features, identity fields, payload fields, or accuracy-only reporting.

## 6. Edge-IIoT Literature-Comparable Selected-Capacity Result

Source: `results/edge_iiot/literature_comparable/edgeiiot_v23_seed_42_checkpoint.json`, `edgeiiot_v23_seed_123_checkpoint.json`, and `edgeiiot_v23_seed_456_checkpoint.json`.

Status: preliminary. Seeds `42`, `123`, and `456` are currently confirmed in the available artifacts. Treat this as a promising pilot until the full 5-seed run finishes.

### 6.1 Seed 456 Snapshot

| Model | Accuracy | Macro-F1 | Params | Size | Interpretation |
|---|---:|---:|---:|---:|---|
| `A_RF_500` | `0.9822` | `0.8881` | n/a | `102,459.08 KB` | Strong tree teacher |
| `A_LightGBM` | `0.9867` | `0.8861` | n/a | `15,365.30 KB` | Highest accuracy, macro-F1 close to RF |
| `C_CL_MLP_loss_fair` | `0.9652` | `0.7968` | `75,279` | `294.06 KB` | Full MLP/CL neural teacher |
| Student A best, `E3_KD_from_LightGBM` | `0.9671` | `0.8183` | `2,383` | `9.31 KB` | Best tiny student for seed 456 |
| Student B best, `E_KD_from_RF` | `0.9676` | `0.8218` | `5,775` | `22.56 KB` | Best balanced student for seed 456 |
| Student C best, `E3_KD_from_LightGBM` | `0.9688` | `0.8248` | `15,631` | `61.06 KB` | Best selected-capacity student for seed 456 |
| Student C `J_CoDistill_RF_CL` | `0.9660` | `0.8168` | `15,631` | `61.06 KB` | Co-distillation below RF/LightGBM KD here |

### 6.2 Three-Seed Preliminary Summary: Seeds 42, 123, and 456

| Group | Best/selected config | Macro-F1 mean | Macro-F1 std | Seed values | Interpretation |
|---|---|---:|---:|---|---|
| Teacher | `A_RF_500` | `0.8888` | `0.0011` | `[0.8880, 0.8903, 0.8881]` | Strongest macro-F1 teacher |
| Teacher | `A_LightGBM` | `0.8863` | `0.0003` | `[0.8861, 0.8867, 0.8861]` | Highest accuracy teacher, macro-F1 close to RF |
| Student A `(32,16)` | `E3_KD_from_LightGBM` | `0.8129` | `0.0038` | `[0.8102, 0.8102, 0.8183]` | Best 3-seed tiny student so far |
| Student A `(32,16)` | `J_CoDistill_RF_CL` | `0.8117` | `0.0024` | `[0.8129, 0.8138, 0.8084]` | Close, but no longer best after seed 456 |
| Student B `(64,32)` | `E3_KD_from_LightGBM` | `0.8214` | `0.0007` | `[0.8222, 0.8215, 0.8205]` | Best 3-seed balanced student so far |
| Student B `(64,32)` | `D_Small_MLP` | `0.8212` | `0.0004` | `[0.8217, 0.8210, 0.8209]` | Scratch is nearly tied; KD gain is small here |
| Student B `(64,32)` | `E_KD_from_RF` | `0.8210` | `0.0011` | `[0.8218, 0.8194, 0.8218]` | Essentially tied with E3 and scratch |
| Student C `(128,64)` | `E3_KD_from_LightGBM` | `0.8250` | `0.0006` | `[0.8258, 0.8244, 0.8248]` | Best 3-seed selected-capacity student |
| Student C `(128,64)` | `E_KD_from_RF` | `0.8247` | `0.0007` | `[0.8257, 0.8243, 0.8242]` | Essentially tied with LightGBM-KD |
| Student C `(128,64)` | `J_CoDistill_RF_CL` | `0.8194` | `0.0020` | `[0.8198, 0.8216, 0.8168]` | Consistently below RF-KD and LightGBM-KD |

Interpretation:

- The literature-comparable Edge route improves student macro-F1 from the strict `0.70-0.71` range to about `0.81-0.825` in the available three-seed checkpoints.
- LightGBM-KD (`E3`) and RF-KD (`E`) are currently more promising than co-distillation on Edge-IIoT selected-capacity.
- Student B/C results are very stable, but the KD gains over scratch are small under this protocol.
- This is not yet a final Edge claim. The full 5-seed result is still needed before publication-level comparison.

## 7. SHAP Explanation Alignment Result

Source: `results/wsnds/legacy_runs/2026-05-30-10seed/cukd_xai_results.json`.

| Metric | Value |
|---|---:|
| Teacher-student global SHAP rank Spearman rho | `0.0466` |
| Spearman p-value | `0.8591` |
| Bootstrap Spearman mean | `0.0015` |
| Bootstrap Spearman std | `0.1068` |

Interpretation:

- The student can preserve useful predictive performance without preserving the teacher's feature-importance ordering.
- This is a stronger XAI contribution than simply "we used SHAP."
- Several papers already use SHAP on WSN-DS. The better claim is explanation-faithfulness auditing after compression, not first use of SHAP.

## 8. Extended Paper-by-Paper Metric Comparison

Use this section when presenting the research or writing the related-work/results comparison. The table separates comparable metrics from missing or non-comparable metrics. If a field is not available in the current repository context, it is explicitly marked instead of inferred.

### 8.1 WSN-DS Papers

| Paper | Dataset and protocol | Reported detection metric | Reported size/latency/resource metric | Our closest comparable result | How to position CuKD-XAI |
|---|---|---:|---:|---|---|
| Almomani et al. 2016, WSN-DS dataset paper | WSN-DS original dataset and baseline models | ANN around `96.6%` accuracy in repo context; exact macro-F1 not available here | Not reported in available context | Student B `J`: acc `98.91%`, macro-F1 `93.35%`; Student A `E`: acc `98.69%`, macro-F1 `92.00%` | We use the same dataset family but target compression and explanation audit, not dataset introduction. |
| Talukder et al. 2025, Scientific Reports, KMS+PCA+RFC | WSN-DS, KMeans-SMOTE balancing, PCA, 5-fold CV | Acc `99.94%`, F1 `99.94%`; per-class F1 around `99.87-99.99%` | Size/latency not reported in available context; model is RF-based and expected MB-scale | Best Student B `J`: acc `98.91%`, macro-F1 `93.35%`, `13.27 KB`; Student A `E`: acc `98.69%`, macro-F1 `92.00%`, `4.64 KB` | They are accuracy SOTA. We are not beating them. Our angle is `6,410x-18,315x` smaller than our RF teacher plus deployment and explanation-faithfulness analysis. |
| MLSTL-WSN, Talukder et al. 2024, IJIS | WSN-DS, SMOTE-Tomek, LightGBM, SHAP/RFE feature selection | Binary acc `99.78%`; multiclass acc `99.92%`; exact macro-F1 not available in current summary | Modeling-time reduction `46%`; model size/latency not available in current summary | Student B `J`: acc `98.91%`, macro-F1 `93.35%`, `13.27 KB` | They already cover SHAP feature selection on WSN-DS. Our differentiator is KD compression into a neural student plus SHAP teacher-student alignment audit. |
| Birahim et al. 2025, IEEE Access, PSO ensemble + SHAP/LIME | WSN-DS, PSO, SMOTE-Tomek, ensemble of RF/DT/KNN | Acc `99.73%`, precision `99.72%`, recall `99.72%`, F1 `99.72%` | Model size/latency not reported in available context | Student B `J`: acc `98.91%`, macro-F1 `93.35%`, `13.27 KB`; deployment ONNX p50 about `0.028 ms` | They prove high WSN-DS accuracy with XAI. We should not claim first XAI. Our claim is compressed deployable student and faithfulness gap. |
| Pandey et al. 2025, Scientific Reports, Tabu Search RF | WSN-DS and other IDS datasets, optimized RF | Acc/F1 about `99.67%` on WSN-DS per repo literature map | Compression/deployment/XAI not reported in available context | Student B `J`: acc `98.91%`, macro-F1 `93.35%`; Student B `E`: acc `98.91%`, macro-F1 `93.28%` | They are stronger on pure detection, but still tree/ensemble-focused. Our model is KB-scale neural deployment. |
| GSWO-CatBoost, Sensors 2024 | WSN-DS, CatBoost optimized by Grey Sunflower Whale Optimization | Acc about `99.65%`; F1 not available in repo context | Size/latency not available in repo context | Student B `J`: acc `98.91%`, macro-F1 `93.35%`, `13.27 KB` | Another accuracy-SOTA style comparator. Use it to show WSN-DS is saturated and why compression is the real gap. |
| Alfarra and AbuSamra 2025, pruned INT8 CNN-LSTM | WSN-DS + ns-3, but classifier covers 4 classes; TDMA/Scheduling is treated outside the classifier/open-set path | Acc about `98%`; macro-F1 about `0.93` | 50% pruned INT8 CNN-LSTM; about `42 ms`, `28 mJ` per 32-window batch, `69-day` T50 lifetime | Student B `J`: 5-class acc `98.91%`, macro-F1 `93.35%`, `13.27 KB`, ONNX p50 `0.028 ms`; Student A `E`: 5-class acc `98.69%`, macro-F1 `92.00%`, `4.64 KB` | This is one of the closest deployability papers. Our advantage is retaining all 5 WSN-DS classes and much smaller software artifact. Their advantage is real energy/lifetime simulation evidence. |
| Xiao and Duan 2025, metaheuristic soft-voting ensemble | WSN-DS, DNN + CatBoost soft voting, metaheuristic optimization, CAM-style sensitivity | Test acc `95.62%`; F1 about `96.07%` in repo context, but exact averaging definition should be verified | Resource-aware framing; exact deployment size/latency not available in current summary | Student B `J`: acc `98.91%`, macro-F1 `93.35%`; Student A `E`: acc `98.69%`, macro-F1 `92.00%` | Our accuracy is higher but macro-F1 comparison may not be apples-to-apples unless their F1 averaging is confirmed. We are also much more explicit on model size. |
| Vidhya and Varunadevi 2026, binarized simplicial CNN | WSN-DS 5-class according to repo note | Exact metrics not verified in available local context | Exact size/latency not verified in available local context | Not enough verified information for a numeric comparison | Do not cite metric claims until the PDF/details are verified. Mention only if final paper access confirms numbers. |

### 8.2 KD, XAI, and Lightweight IoT IDS Papers Outside WSN-DS

| Paper | Dataset and protocol | Reported detection metric | Reported size/latency/resource metric | Our closest comparable result | How to position CuKD-XAI |
|---|---|---:|---:|---|---|
| Benaddi et al. 2025, SHAP-guided KD + Kronecker IDS | TON_IoT, 9 attack types, SHAP feature pruning, DNN teacher to Kronecker student | Teacher acc `0.9989`, macro-F1 `0.9955`; student FP32 acc `0.9968`, macro-F1 `0.9863`; student INT8 macro-F1 `0.9867` | Teacher `769,922` params, `3021.53 KB`; student `3,042` params, `22.29 KB`; about `253x` param reduction; latency values in repo context need final verification because notes include both batch and per-sample forms | WSN Student B `J`: `3,397` params, `13.27 KB`, acc `0.9891`, macro-F1 `0.9335`; WSN Student A `E`: `1,189` params, `4.64 KB`, acc `0.9869`, macro-F1 `0.9200` | They are the closest KD+SHAP lightweight IDS paper, but not WSN-DS. Our novelty is applying KD/compression and explanation alignment to WSN-DS LEACH attack classes. |
| Narkedimilli et al. 2025a, adaptive curriculum learning + XAI | CIC-IoV-2024, CIC-APT-IIoT, Edge-IIoT | Acc about `97-98%` in repo context | Architecture about `94,051` params; deployment size/latency not available here | Our CL-only branches are not the best; RF-KD and J are stronger on WSN-DS | Use this to show CL exists in IDS, but CuKD-XAI is not mainly a CL success story. |
| Narkedimilli et al. 2025b, CL + image transformation + XAI | CIC-APT-IIoT, Edge-IIoT, CIC-IoV-2024 | Exact numeric metrics not available in current repo summary | Size/latency not available in current repo summary | Our Edge-IIoT selected-capacity pilot is the closest dataset overlap, but seeds 42, 123, and 456 are currently available | Cite as CL/XAI context, not direct metric comparison unless full paper metrics are verified. |
| Hossain and Islam 2025, federated SHAP-based KD | IoT botnet datasets | Acc `99.99%` across botnet types in repo literature map | Compression/latency not available in current summary | Not directly comparable to WSN-DS or Edge-IIoT multiclass runs | Shows SHAP+KD exists in IoT IDS, so our novelty must be WSN-DS compression plus explanation-faithfulness audit. |
| Okey et al. 2026, RAID-KL | CICIoT2023, CICIoMT2024, NIMSLABIoT2025 | Accuracy maintained, exact per-dataset metrics not available in current summary | `91.24%` model compression, `11.3%` CPU reduction, `64.33%` memory reduction | Our WSN compression is much larger vs RF teacher: `6,410x-18,315x` size reduction; deployment proof is ONNX/OpenVINO software-level | Use as evidence that KD/compression for IDS is active, but not on WSN-DS. |
| Alabbadi and Bajaber 2025, XAI over IoT data streams | TON_IoT sub-datasets | Network acc `99.24%`; IoT average `99.96%` in repo map | Size/latency not available in current summary | Not directly comparable | XAI/IoT context only, not a compression comparator. |
| Versatile XAI framework 2025 | CIC-DDoS2019, CICIoT2023, 5G PFCP | F1 `>=99%` across datasets in repo map | LIME explanation time reduced from `36s` to `4.9s`; about `70%` dimensionality reduction | Not directly comparable | Useful for XAI literature context; not WSN-DS or KB-scale student deployment. |
| Mohale and Obagbuwa 2025, ML IDS with XAI | UNSW-NB15 | Best acc `87%` with XGBoost/CatBoost in repo map | Size/latency not available | Not directly comparable | Shows that lower absolute IDS numbers can be acceptable on harder datasets when analysis is rigorous. |

### 8.3 Edge-IIoT Literature Claims That Need Final Verification

The following metrics were discussed during planning, but the exact paper PDFs/DOIs are not fully verified inside the currently inspected repo context. They should not be used in a manuscript until the source papers are opened and checked.

| Claimed paper/result | User-provided metric | How our current Edge result compares | Handling recommendation |
|---|---:|---|---|
| IIoT-TinyDNN 2025 | `2,255` params, acc `92.99%` | Our strict Edge best is macro-F1 `0.7111`; literature-comparable three-seed students reach macro-F1 about `0.813-0.825` | Verify exact dataset split, binary/multiclass target, leakage columns, and metric definition before comparing. |
| LightGBM / TCN Edge-IIoT papers 2025/2026 | LightGBM acc `95.3%`, TCN acc `97.2%`, `75-190 KB`, `30 ms` on Raspberry Pi 3 B+ | Our three-seed LightGBM has macro-F1 about `0.8863`; Student C has three-seed macro-F1 about `0.8250`, `61.06 KB` | Compare accuracy only with caution because macro-F1 shows class imbalance hardness. Need exact class protocol. |
| CNN-GRU hybrid 2026 | Acc `96.88%` | Our Edge selected-capacity students are below this in macro-F1; teacher/LightGBM accuracy is comparable or higher | Verify if binary, multiclass, or selected attack classes. Do not overclaim. |
| Heavy DL / ensemble Edge-IIoT baselines | Acc `>=99%` up to `99.99%` | Our strict protocol is much lower; literature-comparable teacher accuracy is `98.23-98.67%` but macro-F1 about `0.886-0.888` | If those papers keep leakage/identity columns or use binary labels, they are not fair direct comparators. |

## 9. Recommended Presentation Sequence

### 9.1 Main Technical Message

1. WSN-DS accuracy is saturated in the literature, with tree/ensemble papers reporting about `99.7-99.94%`.
2. CuKD-XAI does not beat that leaderboard, but it changes the deployment trade-off: `4.64-13.27 KB` students retain `0.9200-0.9335` macro-F1 across 10 seeds.
3. The strongest balanced WSN-DS student is Student B with either:
   - `J_CoDistill_RF_CL`: acc `0.9891`, macro-F1 `0.9335`, `13.27 KB`, best mean.
   - `E_KD_from_RF`: acc `0.9891`, macro-F1 `0.9328`, `13.27 KB`, nearly identical and more conservative.
4. The strongest ultra-small WSN-DS student is Student A `E_KD_from_RF`: acc `0.9869`, macro-F1 `0.9200`, `4.64 KB`.
5. ONNX/OpenVINO deployment proof shows the student artifacts are portable and extremely fast in software runtime testing.
6. The SHAP result is novel and scientifically useful: distillation transfers task performance but not teacher explanation ranking.

### 9.2 Suggested Comparison Statement

> Compared with WSN-DS SOTA papers that report `99.7-99.94%` accuracy/F1 using RF, LightGBM, CatBoost, or ensemble methods, our method sacrifices about `4.5-5.9` macro-F1 points against the RF teacher but reduces serialized model size by about `6,410x-18,315x`. Unlike prior WSN-DS SHAP papers, we do not only explain the final model; we quantify whether the compressed student preserves the teacher's feature-importance ranking, and we find that it does not. This gives the paper a compression-deployment contribution and an explanation-faithfulness contribution.

### 9.3 Best Tables to Show

Use these three tables in a meeting:

1. WSN-DS Student A/B 10-seed table from Sections 3.1 and 3.2.
2. Literature comparison table from Section 8.1.
3. Deployment table from Section 4.

For Edge-IIoT, show it as ongoing robustness evidence:

- Strict stress test: complete, conservative, shows capacity limit.
- Literature-comparable route: promising three-seed result, but needs multi-seed completion before final claims.

## 10. Novelty, Limitations, and Safe Claims

### 10.1 Defensible Novelty

| Novelty axis | Defensible claim |
|---|---|
| WSN-DS compression | KD-based compression of WSN-DS multiclass IDS into KB-scale MLP students. |
| Deployment | ONNX/OpenVINO software deployment proof plus fixed-point C export with integer StandardScaler metadata, generated self-test vectors, and MSP430F1611 target-toolchain footprint evidence. |
| Explanation audit | Quantitative SHAP teacher-student rank-alignment analysis after compression. |
| Multi-seed evidence | 10-seed WSN-DS evaluation with Student A/B capacity comparison and multiple KD baselines. |
| Generalization stress | Edge-IIoT strict and literature-comparable routes expose capacity and protocol sensitivity. |

### 10.2 Limitations to Admit

| Limitation | Honest wording |
|---|---|
| Not WSN-DS accuracy SOTA | "Our method is not designed to beat oversampled tree-ensemble SOTA accuracy; it targets KB-scale deployment." |
| RF teacher still much better in macro-F1 | "The compressed student keeps useful accuracy but does not close the full teacher-student gap." |
| Edge-IIoT strict result is modest | "Strict 15-class Edge-IIoT exposes a capacity-complexity bottleneck." |
| No physical WSN mote yet | "Deployment is currently software-runtime, fixed-point C, and MSP430 cross-compile footprint proof; real WSN mote flashing, latency, and energy measurements remain future work." |
| INT8 not beneficial in current runtime test | "Dynamic INT8 reduced artifact size but did not improve latency or F1 in this CPU runtime." |
| SHAP alignment low | "This is not a failure of prediction; it is an explanation-faithfulness warning." |

### 10.3 Claims to Avoid

- Do not claim "first SHAP on WSN-DS." Birahim 2025 and MLSTL-WSN already use SHAP on WSN-DS.
- Do not claim "best WSN-DS accuracy." SOTA papers report around `99.7-99.94%`.
- Do not claim "hardware deployment" unless real WSN mote measurements are added. MSP430 cross-compilation supports memory feasibility only.
- Do not claim full on-mote WSN-DS feature extraction; the current artifact covers integer StandardScaler normalization after raw features exist.
- Do not claim "INT8 speedup" from the current deployment result.
- Do not claim co-distillation always improves. It helps some settings but RF-KD is often as good or better.

## 11. References and Source Links from Repo Context

Primary result files:

- `results/wsnds/legacy_runs/2026-05-30-10seed/wsnds_results_student_A.csv`
- `results/wsnds/legacy_runs/2026-05-30-10seed/wsnds_results_student_B.csv`
- `results/wsnds/final_results/2026-05-30-10seed-plus-j/wsnds_results_student_A.csv`
- `results/wsnds/final_results/2026-05-30-10seed-plus-j/wsnds_results_student_B.csv`
- `results/runtime/onnx_openvino/wsnds/runtime_from_existing_outputs/wsnds_existing_artifact_runtime_summary.csv`
- `deployment/msp430/export_wsnds_student_a_rfkd_int8.py`
- `deployment/msp430/wsnds_student_a_rfkd_int8_inference.c`
- `deployment/msp430/wsnds_student_a_rfkd_self_test.c`
- `deployment/msp430/run_wsnds_student_a_rfkd_e2e.py`
- `deployment/msp430/README.md`
- `results/edge_iiot/strict_generalization/edgeiiot_v23_results_student_A.csv`
- `results/edge_iiot/strict_generalization/edgeiiot_v23_results_student_B.csv`
- `results/edge_iiot/literature_comparable/edgeiiot_v23_seed_42_checkpoint.json`
- `results/edge_iiot/literature_comparable/edgeiiot_v23_seed_123_checkpoint.json`
- `results/edge_iiot/literature_comparable/edgeiiot_v23_seed_456_checkpoint.json`
- `results/wsnds/legacy_runs/2026-05-30-10seed/cukd_xai_results.json`

Local literature/context files:

- `research_history/documentation_snapshots/updates/2026-04-12/CuKD_XAI_MASTER_DOCUMENT.md`
- `research_history/documentation_snapshots/updates/2026-04-12/XAI_IDS_WSN_IoT_Literature_Map_2023_2026.md`
- `research_history/documentation_snapshots/updates/2026-04-12/MONDAY_PRESENTATION_PACK.md`
- `docs/literature/papers/base_paper.pdf`
- `docs/literature/papers/sota_wsn_ds_2025.pdf`
- `docs/literature/papers/alfarra_2025.pdf`
- `docs/literature/papers/benaddi_2025.pdf`
- `docs/literature/papers/Metaheuristically optimized deep soft-voting ensemble for explainable and resource-aware signal processing in wireless sensor network intrusion detection.pdf`

Paper links recorded in repo context:

- Almomani et al. 2016 WSN-DS: https://doi.org/10.1155/2016/4731953
- Ghadi et al. 2024 IEEE Access review: https://doi.org/10.1109/ACCESS.2024.3355312
- Talukder et al. 2025 Scientific Reports WSN-DS SOTA: https://doi.org/10.1038/s41598-025-87028-1
- MLSTL-WSN: https://doi.org/10.1007/s10207-024-00833-z
- Birahim et al. 2025 IEEE Access: https://doi.org/10.1109/ACCESS.2025.3528341
- Pandey et al. 2025 Tabu RF: https://doi.org/10.1038/s41598-025-03498-3
- GSWO-CatBoost Sensors 2024: https://doi.org/10.3390/s24113339
- Alfarra and AbuSamra 2025: https://doi.org/10.37936/ecti-cit.2025194.263081
- Benaddi et al. 2025 SHAP+KD Kronecker IDS: https://arxiv.org/abs/2512.19488
- Xiao and Duan 2025: https://doi.org/10.1007/s11760-025-04880-4




