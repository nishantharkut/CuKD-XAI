# CuKD-XAI FG-DS Evidence Registry

Status: `passed_with_open_planned_work`
Registry: `cukd_fgds_evidence_registry_20260814_v3`

## Primary WSN-DS Evidence

The primary predictive result is the ten-seed feature-group-disjoint run with a train-only scaler. It uses one fixed split and ten optimizer seeds.

| Route | Macro-F1 mean | Sample SD |
|---|---:|---:|
| Student A scratch | 0.914792 | 0.005658 |
| Student A RF-KD | 0.913781 | 0.004546 |
| Student B scratch | 0.932867 | 0.005727 |
| Student B RF-KD | 0.932142 | 0.010930 |

Student A RF-KD minus scratch: -0.001012, exact Wilcoxon p=0.556641, Holm-adjusted p=1.000000.

Student B RF-KD minus scratch: -0.000725, exact Wilcoxon p=0.845703, Holm-adjusted p=1.000000.

## Controlled Full-Route Evidence

The complete teacher and student route matrix was rerun on the same clean split for ten seeds. The table below reports the student routes. Exact signed-rank inference and sample SD are preserved in the sealed aggregate.

| Student | Route | Macro-F1 mean | Sample SD |
|---|---|---:|---:|
| student_A | D_Small_MLP | 0.914792 | 0.005658 |
| student_A | E_KD_from_RF | 0.913781 | 0.004546 |
| student_A | E2_KD_from_MLP | 0.914321 | 0.003204 |
| student_A | F_KD_from_CL_MLP_fair | 0.915357 | 0.003084 |
| student_A | F_KD_from_CL_MLP_ext | 0.912296 | 0.003330 |
| student_A | G_KD_random_pacing | 0.914970 | 0.003614 |
| student_A | I_KD_from_SMOTE_MLP | 0.911577 | 0.006552 |
| student_A | J_CoDistill_RF_CL | 0.917463 | 0.006553 |
| student_B | D_Small_MLP | 0.932867 | 0.005727 |
| student_B | E_KD_from_RF | 0.932142 | 0.010930 |
| student_B | E2_KD_from_MLP | 0.922492 | 0.003731 |
| student_B | F_KD_from_CL_MLP_fair | 0.920344 | 0.005458 |
| student_B | F_KD_from_CL_MLP_ext | 0.922113 | 0.005211 |
| student_B | G_KD_random_pacing | 0.921007 | 0.003971 |
| student_B | I_KD_from_SMOTE_MLP | 0.919619 | 0.005092 |
| student_B | J_CoDistill_RF_CL | 0.927575 | 0.007727 |

This is a controlled FG-DS reimplementation of the complete route matrix. Route-level RNG resets and shared initial states differ from the archived sequential execution, so archived-to-current changes cannot be attributed to split correction alone.

## RF-KD Hyperparameter Sensitivity

The 3 x 3 temperature-alpha surface contains 180 training jobs across two students and ten seeds. It is descriptive only; no winning cell was selected and the primary result was not replaced.

| Comparison | Mean macro-F1 difference | Exact Wilcoxon p | Within-student Holm p | Global Holm p |
|---|---:|---:|---:|---:|
| student_A:T1_alpha03_minus_persisted_scratch_test_macro_f1 | 0.009276 | 0.005859 | 0.052734 | 0.105469 |
| student_A:T4_alpha07_minus_persisted_scratch_test_macro_f1 | -0.001012 | 0.556641 | 1.000000 | 1.000000 |
| student_B:T4_alpha07_minus_persisted_scratch_test_macro_f1 | -0.000725 | 0.845703 | 1.000000 | 1.000000 |

This response surface evaluates sensitivity after the clean primary run. The same fixed validation partition is used for early stopping in every cell and the same fixed test partition is evaluated repeatedly. The surface must not be used for post-hoc model selection, a new primary claim, or an unbiased estimate of a selected cell's generalization.

## Repeated-Pattern Sensitivity

The 56,301 test records form 54,174 exact feature groups. There are 0 mixed-label groups in the test partition.

| View | Student | RF-KD minus scratch | Exact Wilcoxon p | Holm p |
|---|---|---:|---:|---:|
| row_level | student_A | -0.001012 | 0.556641 | 1.000000 |
| row_level | student_B | -0.000725 | 0.845703 | 1.000000 |
| inverse_test_group_size | student_A | 0.001209 | 0.431641 | 1.000000 |
| inverse_test_group_size | student_B | -0.000204 | 1.000000 | 1.000000 |
| pure_group_representative | student_A | 0.001209 | 0.431641 | 1.000000 |
| pure_group_representative | student_B | -0.000204 | 1.000000 | 1.000000 |

Row-level metrics remain primary because they preserve the benchmark record distribution. Inverse-group-size and one-representative-per-pure-group views are sensitivity analyses for repeated exact test patterns, not replacement test sets.

## Behavioral-Transfer Evidence

The post-hoc ten-seed analysis compares each RF-KD student with its matched scratch model under the same held-out, exact-group-balanced T=4 response-distribution contract.

| Student | Mean scratch-minus-RF-KD KL | Sample SD | Positive seeds | Holm-adjusted p |
|---|---:|---:|---:|---:|
| student_A | 0.191178 | 0.045466 | 10 | 0.003906 |
| student_B | 0.193702 | 0.021476 | 10 | 0.003906 |

This analysis tests held-out in-distribution response-distribution transfer from the calibrated RF to each checkpoint-reconstructed student and compares RF-KD with matched scratch under the same T=4 output contract used by KD and XAI. It does not establish causal mechanism transfer, off-manifold decision-boundary equivalence, explanation transfer, or deployment fidelity.

## Split-Sensitivity Confirmation

The core scratch-versus-RF-KD comparison was repeated across 10 exact-feature-group splits with two paired optimizer seeds per split (80 student training jobs). Student A had a positive split-level mean RF-KD effect on all 10 splits; Student B was positive on 5 and negative on 5.

This confirmation estimates sensitivity to ten exact-feature-group split seeds for the core scratch versus RF-KD comparison. Each split-level value averages two paired optimizer seeds. The repeated holdouts overlap and are therefore reported descriptively, not as independent replications. This does not replace the finalized ten-optimizer-seed result on the fixed primary split and does not cover the full route matrix, deployment, XAI, or Edge-IIoTset.

## Current XAI Evidence

Permutation SHAP explains a fixed stratified subset of 500 of 56,301 test records for one seed-42 deployment specimen. The reconstructed calibrated RF teacher passed train and test output validation. Exact equivalence on synthetic masked inputs is not claimed.

| Output contract | Student | Global rank rho mean | Sample SD | Mean top-5 overlap |
|---|---|---:|---:|---:|
| fp32_deployment_source_probabilities_T1 | student_A | 0.411765 | 0.028897 | 3.333 |
| fp32_deployment_source_probabilities_T1 | student_B | 0.446078 | 0.016072 | 4.000 |
| kd_softened_probabilities_T4 | student_A | 0.403595 | 0.012336 | 3.000 |
| kd_softened_probabilities_T4 | student_B | 0.553922 | 0.021226 | 3.667 |

Maximum local-accuracy residual across all 18 SHAP artifacts: 4.441e-16 (gate 1.0e-06).

## Runtime and Hardware Evidence

One fixed seed-42 deployment specimen was replayed over USB and Wi-Fi UDP on four board-model pairs. Each transport executed 225,204 full-test predictions. All pairs matched the fixed-point reference predictions and logits exactly.

| Pair | MCU macro-F1 | MCU vs FP32 | USB compute mean (us) | Wi-Fi compute mean (us) | Wi-Fi retransmissions |
|---|---:|---:|---:|---:|---:|
| esp32c3_student_A | 0.905694 | 0.995009 | 116.413 | 162.449 | 101 |
| esp32c3_student_B | 0.914564 | 0.994743 | 320.237 | 401.424 | 0 |
| arduino_r4_student_A | 0.905694 | 0.995009 | 301.080 | 300.199 | 1 |
| arduino_r4_student_B | 0.914564 | 0.994743 | 791.019 | 789.977 | 3 |

USB and Wi-Fi use the same model artifacts and test records. They are execution replications, not independent predictive samples. Wi-Fi retransmissions are application-level retries after response timeouts.

## All-Seed Software Fixed-Point Audit

The fixed-point exporter audited 40 model-seed instances. 26 passed every software quality and exact C/Python equivalence gate; 14 were retained as gate failures rather than omitted.

The 40 models are training-run/model-seed instances sharing one fixed feature-group-disjoint split and one train-only-fitted scaler; they are not 40 independent data splits.

## Final USB Hardware Campaign

The final campaign executed 6 gate-eligible model-board sessions, each with 56,301 full-test rows. Across 337,806 reported replay rows, every MCU prediction and fixed-point logit matched the fixed reference exactly.

| Model | ESP32-C3 | Arduino R4 | Deployment status |
|---|---|---|---|
| Student A scratch | exact | exact | passed |
| Student A RF-KD | exact | exact | passed |
| Student B RF-KD | exact | exact | passed |
| Student B scratch | not executed | not executed | blocked by fixed-point gates |

The portable archive contains 416 files and is retained outside Git with SHA-256 `0361f70877b00a27df5e7c559d178a9f4fbdd37136c05dc7d68fdce0b4c79561`.

One seed-42 model specimen per fixed-point-gate-eligible route on one physical specimen of each board type. Gate-failed routes remain explicit non-deployment results. Exact replay and timing evidence do not establish multi-seed or multi-unit hardware variability, energy, or secure attestation.

## Software-Only Fixed-Point Refinement

| Student | Source PTQ fixed macro-F1 | Refined fixed macro-F1 | Delta | Fixed vs float agreement |
|---|---:|---:|---:|---:|
| student_A | 0.905694 | 0.913642 | 0.007948 | 0.997265 |
| student_B | 0.914564 | 0.921566 | 0.007002 | 0.996341 |

The refinement changes the two model states after validation-based epoch selection. It has not been strictly exported or replayed on either board, so it cannot replace the preserved PTQ USB or Wi-Fi results.

## Current MSP430F1611 Static Evidence

| Student | Static flash (bytes) | Static RAM lower bound (bytes) | Maximum single-function stack (bytes) |
|---|---:|---:|---:|
| student_A | 2846 | 8 | 106 |
| student_B | 5196 | 8 | 202 |

Static MSP430F1611 cross-compile and memory-footprint evidence only. No physical TelosB execution, latency, energy, radio integration, or live WSN feature-extraction claim is supported.

## Secondary Edge-IIoTset Evidence

The group-aware Edge run uses 40 inputs and 1,556,588/332,240/330,373 train/validation/test rows. Pre-encode group overlap is zero; encoded exact-row overlaps remain 163, 157, and 26 for train-test, train-validation, and validation-test.

## Open Planned Work

Two planned experiments are not part of this registry: the ten-seed scratch-controlled XAI audit and the six final-lineage Wi-Fi sessions. The completed seed-42 SHAP specimen and older five-seed Wi-Fi campaign remain valid only within their recorded lineages.

The available SHAP result is a completed seed-42 specimen audit. It is not a substitute for the planned ten-seed scratch-controlled XAI experiment.

The completed Wi-Fi campaign belongs to the distinct five-seed deployment lineage. The final ten-seed seed-42 lineage currently has USB evidence only.

## Boundaries

The incomplete SHAP-v2 attempt is explicitly excluded. Historical 56,200-row QAT and SHAP artifacts, the legacy MSP430 report, and archived random-row extensions remain preserved but are not current FG-DS evidence. See `evidence_registry.json` for machine-readable lineage rules, exact hashes, and claim boundaries.

