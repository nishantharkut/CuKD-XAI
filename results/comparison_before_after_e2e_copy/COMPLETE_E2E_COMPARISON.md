# Complete E2E Comparison: Manuscript-era (BEFORE) vs New Research Evidence (AFTER)

**Package path:** `results/comparison_before_after_e2e_copy/`  
**Machine-readable tables:** `01_*.csv` … `10_*.csv` + `00_MASTER_COMPARISON_INDEX.json`

## Definitions (read first)

| Label | Meaning |
|---|---|
| **BEFORE** | Evidence used when the manuscript draft was prepared: archived pre-split-scaler 10-seed WSN tables (`2026-05-30-10seed-plus-j`), archived HIL (`board_replay`), archived ONNX/QAT/runtime package |
| **AFTER** | Post-leakage train-only research package: full 10-seed train-only aggregates, seed-42 deployment, feature-group 5-seed, train-only export/HIL/runtime |

**Not comparable as one number without labels:**

1. Pre-split scaler ≠ train-only scaler  
2. Archived HIL weights ≠ train-only seed-42 exports  
3. Feature-group split ≠ random-row split  
4. Seed-42 single-seed ≠ 10-seed mean  
5. Incomplete `main_10seed/*.csv` with **n=2** is **not** AFTER primary  

---

## 0) Protocol / preprocessing

| Item | BEFORE | AFTER |
|---|---|---|
| Scaler fit | Full matrix **before** split | **Train partition only** after split |
| Split | Stratified random-row, seed 42 | Same indices / shapes for random-row path |
| Train / val / test | 262252 / 56209 / 56200 | Same |
| Scaler mean shift vs global (diag) | — | max ~0.003 SD; relative scale change max ~2.5% |
| Dataset SHA-256 | same CSV family | `c65d05b9…f7c9` |

---

## 1) Multi-seed predictive — **all Student A configs**

Macro-F1 mean ± std (n=10 both sides unless noted). Δ = AFTER − BEFORE.

| Config | BEFORE F1 | AFTER F1 (train-only full 10) | Δ F1 | BEFORE Acc | AFTER Acc |
|---|---:|---:|---:|---:|---:|
| A_RF_500 | 0.978889 ± 0.000251 | 0.978850 ± 0.000267 | −0.000039 | 0.996600 | 0.996591 |
| B_Full_MLP | 0.923195 ± 0.002375 | 0.922689 ± 0.002006 | −0.000506 | 0.987322 | 0.987263 |
| C2_CL_MLP_domain | 0.913808 ± 0.003158 | 0.913137 ± 0.004541 | −0.000670 | 0.985731 | 0.985655 |
| C_CL_MLP_loss | 0.914730 ± 0.003903 | 0.913586 ± 0.003855 | −0.001143 | 0.985886 | 0.985726 |
| C_CL_MLP_loss_fair | 0.914730 ± 0.003903 | 0.913586 ± 0.003855 | −0.001143 | 0.985886 | 0.985726 |
| C_CL_MLP_loss_ext | 0.867257 ± 0.154820 | 0.867420 ± 0.161681 | +0.000164 | 0.978667 | 0.978591 |
| D_Small_MLP (scratch) | 0.912303 ± 0.006223 | 0.910903 ± 0.006843 | −0.001400 | 0.984708 | 0.984365 |
| E2_KD_from_MLP | 0.916355 ± 0.002759 | 0.912039 ± 0.002335 | −0.004315 | 0.986126 | 0.985391 |
| **E_KD_from_RF (RF-KD)** | **0.919971 ± 0.003124** | **0.920344 ± 0.006513** | **+0.000373** | 0.986875 | 0.986943 |
| F_KD_from_CL_MLP | 0.912912 ± 0.002126 | 0.910490 ± 0.003104 | −0.002423 | 0.985509 | 0.985066 |
| F_KD_from_CL_MLP_fair | 0.912912 ± 0.002126 | 0.910490 ± 0.003104 | −0.002423 | 0.985509 | 0.985066 |
| F_KD_from_CL_MLP_ext | 0.906372 ± 0.025109 | 0.893658 ± 0.055421 | −0.012714 | 0.983607 | 0.981174 |
| G_KD_random_pacing | 0.913696 ± 0.002806 | 0.910809 ± 0.003754 | −0.002888 | 0.985680 | 0.985181 |
| I_KD_from_SMOTE_MLP | 0.913603 ± 0.001918 | 0.908499 ± 0.004963 | −0.005104 | 0.985600 | 0.984708 |
| **J_CoDistill_RF_CL** | **0.918062 ± 0.004100** | **MISSING** | — | 0.986514 | — |

## 2) Multi-seed predictive — **all Student B configs**

| Config | BEFORE F1 | AFTER F1 | Δ F1 | BEFORE Acc | AFTER Acc |
|---|---:|---:|---:|---:|---:|
| A_RF_500 | 0.978889 ± 0.000251 | 0.978850 ± 0.000267 | −0.000039 | 0.996600 | 0.996591 |
| B_Full_MLP | 0.923195 ± 0.002375 | 0.922689 ± 0.002006 | −0.000506 | 0.987322 | 0.987263 |
| C2_CL_MLP_domain | 0.913808 ± 0.003158 | 0.913137 ± 0.004541 | −0.000670 | 0.985731 | 0.985655 |
| C_CL_MLP_loss | 0.914730 ± 0.003903 | 0.913586 ± 0.003855 | −0.001143 | 0.985886 | 0.985726 |
| C_CL_MLP_loss_fair | 0.914730 ± 0.003903 | 0.913586 ± 0.003855 | −0.001143 | 0.985886 | 0.985726 |
| C_CL_MLP_loss_ext | 0.867257 ± 0.154820 | 0.867420 ± 0.161681 | +0.000164 | 0.978667 | 0.978591 |
| D_Small_MLP (scratch) | 0.932169 ± 0.003750 | 0.931341 ± 0.006575 | −0.000828 | 0.988835 | 0.988730 |
| E2_KD_from_MLP | 0.927548 ± 0.006821 | 0.917737 ± 0.004575 | −0.009810 | 0.988068 | 0.986432 |
| **E_KD_from_RF (RF-KD)** | **0.932808 ± 0.007590** | **0.939173 ± 0.012915** | **+0.006365** | 0.989114 | 0.990189 |
| F_KD_from_CL_MLP | 0.925816 ± 0.003423 | 0.916580 ± 0.005070 | −0.009236 | 0.987749 | 0.986210 |
| F_KD_from_CL_MLP_fair | 0.925816 ± 0.003423 | 0.916580 ± 0.005070 | −0.009236 | 0.987749 | 0.986210 |
| F_KD_from_CL_MLP_ext | 0.917207 ± 0.025905 | 0.904538 ± 0.042078 | −0.012668 | 0.985528 | 0.983093 |
| G_KD_random_pacing | 0.924603 ± 0.005503 | 0.916609 ± 0.004897 | −0.007994 | 0.987521 | 0.986139 |
| I_KD_from_SMOTE_MLP | 0.929978 ± 0.008311 | 0.917561 ± 0.008772 | −0.012417 | 0.988302 | 0.986354 |
| **J_CoDistill_RF_CL** | **0.933526 ± 0.011361** (paper best B) | **MISSING** | — | 0.989133 | — |

**Anomaly (AFTER only, disclosed):** seed **5678**, `C_CL_MLP_loss_ext`, macro-F1 **0.4075** (both student files share teacher metrics) — inflates std on curriculum-ext routes.

---

## 3) Feature-group 5-seed (NEW; not in original manuscript tables)

| Model | FG F1 (5-seed) | Archived random-row 10-seed F1 | Train-only random-row 10-seed F1 | FG − archived | FG − trainonly10 |
|---|---:|---:|---:|---:|---:|
| A RF-KD | 0.914112 ± 0.006865 | 0.919971 | 0.920344 | −0.00586 | −0.00623 |
| A Scratch | 0.913878 ± 0.004570 | 0.912303 | 0.910903 | +0.00158 | +0.00298 |
| B RF-KD | 0.928145 ± 0.007414 | 0.932808 | 0.939173 | −0.00466 | −0.01103 |
| B Scratch | 0.929795 ± 0.005500 | 0.932169 | 0.931341 | −0.00237 | −0.00155 |

**Paired KD − scratch (FG only):** A F1 Δ mean **+0.00023**; B F1 Δ mean **−0.00165**.

---

## 4) Deployment seed-42 only (NEW artifact route)

| Student | Seed-42 F1 | Seed-42 Acc | vs archived 10-seed F1 | vs train-only 10-seed F1 |
|---|---:|---:|---:|---:|
| A RF-KD | **0.948509** | 0.991423 | +0.0285 | +0.0282 |
| B RF-KD | **0.944930** | 0.991121 | +0.0121 | +0.0058 |

Per-class F1 seed-42 A: Blackhole 0.916, Flooding 0.954, Grayhole 0.923, Normal 0.998, TDMA 0.953.  
Per-class F1 seed-42 B: Blackhole 0.903, Flooding 0.954, Grayhole 0.915, Normal 0.998, TDMA 0.954.

---

## 5) Fixed-point export

### Equivalence (vector-level)

| Tag | fixed vs FP32 agreement | FP32 acc on vectors | Fixed acc on vectors |
|---|---:|---:|---:|
| Archived Student A | ~0.995 (HIL metrics) | — | — |
| Archived Student B | ~0.994 (HIL metrics) | — | — |
| Train-only A export | **0.991940** | 0.991423 | 0.987491 |
| Train-only B export | **0.990463** | 0.991121 | 0.987278 |

### Macro-F1 drop gate (train-only measured)

| Student | FP32 F1 | Fixed F1 | Drop | Pass 0.01 strict? | Pass 0.03 copy? |
|---|---:|---:|---:|---|---|
| A | 0.948509 | 0.924416 | **0.024093** | **No** | Yes |
| B | 0.944930 | 0.917988 | **0.026942** | **No** | Yes |

Saturation audits: zero parameter/activation saturation on audited paths (train-only reports).

---

## 6) Hardware HIL full 56,200

| Board | Student | BEFORE F1 | AFTER F1 | Δ F1 | BEFORE MCU/fixed | AFTER MCU/fixed | BEFORE MCU/FP32 | AFTER MCU/FP32 | BEFORE mean µs | AFTER mean µs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ESP32-C3 | A | 0.914014 | **0.924416** | +0.0104 | 1.0 | 1.0 | **0.9950** | 0.9919 | **118.4** | **116.5** |
| Arduino R4 | A | 0.914014 | **0.924416** | +0.0104 | 1.0 | 1.0 | 0.9950 | 0.9919 | **301.6** | **301.5** |
| ESP32-C3 | B | 0.918099 | 0.917988 | −0.0001 | 1.0 | 1.0 | **0.9939** | 0.9905 | **332.3** | **320.3** |
| Arduino R4 | B | 0.918099 | 0.917988 | −0.0001 | 1.0 | 1.0 | 0.9939 | 0.9905 | **791.6** | **791.4** |

All AFTER pairs: n=56200, status OK=56200.  
Archived cycles/MAC (before only table): ESP32 ~15.8 cyc/MAC; R4 ~11.2–11.8 cyc/MAC (same architecture class).

---

## 7) Compile footprints

| Board | Student | BEFORE flash used | AFTER flash used | Δ flash | BEFORE RAM used | AFTER RAM used | Δ RAM |
|---|---|---:|---:|---:|---:|---:|---:|
| ESP32-C3 | A | (see compile log) | 281776 | — | (log) | 13592 | — |
| ESP32-C3 | B | (log) | 284132 | — | (log) | 13592 | — |
| R4 | A | (log) | 56384 | — | (log) | 7128 | — |
| R4 | B | (log) | 58736 | — | (log) | 7128 | — |

Exact before/after pairs in `08_compile_footprint_before_after.csv` (parsed from archived short logs + new verbose logs).

---

## 8) Host runtime (RF-KD focus)

### Archived RF-KD (BEFORE)

| Model | Variant | Acc | F1 | size KB | p50 ms |
|---|---|---:|---:|---:|---:|
| E A RF-KD | ONNX FP32 | 0.986370 | 0.917478 | 5.44 | 0.0275 |
| E A RF-KD | ONNX dyn INT8 | 0.983665 | 0.900602 | 5.07 | 0.0395 |
| E A RF-KD | OpenVINO FP32 | 0.986370 | 0.917478 | 5.44 | 0.1307 (agree 1.0 vs ONNX) |
| E B RF-KD | ONNX FP32 | 0.991050 | 0.944707 | 14.07 | 0.0285 |
| E B RF-KD | ONNX dyn INT8 | 0.987491 | 0.923095 | 7.37 | 0.0396 |
| E B RF-KD | OpenVINO FP32 | 0.991050 | 0.944707 | 14.07 | 0.1304 |

Archived package total: **18 runtime rows** (scratch/RF-KD/co-distill × variants).

### Train-only RF-KD seed-42 (AFTER)

| Model | Variant | Acc | F1 | agree PT | p50 ms |
|---|---|---:|---:|---:|---:|
| A | ONNX FP32 | 0.991423 | 0.948509 | **1.0** | 0.0271 |
| A | ONNX dyn INT8 | 0.982206 | 0.893790 | 0.9861 | 0.0369 |
| A | OpenVINO FP32 | 0.991423 | 0.948509 | **1.0** (vs ORT 1.0) | 0.1098 |
| B | ONNX FP32 | 0.991121 | 0.944930 | **1.0** | 0.0214 |
| B | ONNX dyn INT8 | 0.984555 | 0.906592 | 0.9898 | 0.0380 |
| B | OpenVINO FP32 | 0.991121 | 0.944930 | **1.0** | 0.1138 |

---

## 9) QAT

### Archived QAT (BEFORE, multi-route package, proof seed 9999)
Includes D/E/J A/B; e.g. E_student_A_KD_from_RF QAT F1 0.8996 (Δ vs FP32 −0.0178); E_student_B −0.0234.

### Train-only seed-42 QAT probe (AFTER)

| Student | Baseline PTQ drop | After-QAT FP32 F1 | After-QAT fixed drop | Used for HIL? |
|---|---:|---:|---:|---|
| A | 0.0264 | **0.8784** (worse absolute) | 0.0128 | **No** |
| B | (see report) | improved drop possible | — | **No** (A absolute F1 harm drove non-selection) |

---

## 10) SHAP / co-distill / Edge / MSP430

| Item | BEFORE | AFTER |
|---|---|---|
| SHAP audit | Curriculum-KD A vs RF, ρ≈0.047, p≈0.86; bootstrap mean ≈0.0015 | **No SHAP on train-only RF-KD deploy student** |
| Co-distill J | Full 10-seed results (B best mean 0.9335) | **Not trained** under train-only |
| Edge strict/lit matrices | Present (manuscript tables) | Unchanged |
| Edge duplicate audit | Not in original manuscript evidence pack | **NEW**: strict 5954 dup rows; lit 236006; cross-partition groups 743 / 95600 |
| MSP430 cross-compile | Present | Unchanged (feasibility only) |

---

## 11) What the incomplete n=2 file falsely showed (trap)

| | n | A RF-KD F1 | B RF-KD F1 |
|--|--:|---:|---:|
| Incomplete merge | 2 | 0.9221 | **0.9557** (over-optimistic) |
| Full train-only 10 | 10 | 0.9203 | 0.9392 |
| Archived 10 | 10 | 0.9200 | 0.9328 |

---

## 12) Bottom-line map

| Layer | BEFORE (manuscript base) | AFTER (new research) | Change type |
|---|---|---|---|
| Scaler protocol | Pre-split | Train-only | **Method fix** |
| 10-seed RF-KD A | 0.9200 | 0.9203 | ~same |
| 10-seed RF-KD B | 0.9328 | 0.9392 | modest ↑ mean, ↑ variance |
| Co-distill J | Present | Missing | **gap** |
| Feature-group | Absent | Present | **new** |
| Deploy seed-42 F1 | N/A | 0.9485 / 0.9449 | **new** |
| Fixed drop gate | Strict narrative | Measured 2.4–2.7% | **new constraint** |
| HIL MCU/fixed | 1.0 | 1.0 | same strength |
| HIL MCU/FP32 | 0.995 / 0.994 | 0.992 / 0.991 | slightly worse agreement |
| HIL F1 A | 0.914 | 0.924 | ↑ (different model) |
| HIL F1 B | 0.918 | 0.918 | ~same |
| ONNX/OV train-only | N/A for this lineage | Agree 1.0 FP32 | **new** |
| Edge dup audit | N/A | Present | **new** |

---

## Machine-readable files

All under `results/comparison_before_after_e2e_copy/`:

1. `01_wsn_multiseed_all_configs.csv` — every config A/B  
2. `02_feature_group_vs_randomrow.csv`  
3. `03_deployment_seed42_vs_multiseed.csv`  
4. `04_qat_before_after.csv`  
5. `05_runtime_onnx_openvino_before_after.csv`  
6. `06_hil_four_pair_before_after.csv`  
7. `07_export_fixedpoint_before_after.csv`  
8. `08_compile_footprint_before_after.csv`  
9. `09_preprocessing_protocol_before_after.csv`  
10. `10_shap_j_edge_msp430_anomalies_presence.csv`  
11. `00_MASTER_COMPARISON_INDEX.json`  
12. This file: `COMPLETE_E2E_COMPARISON.md`
