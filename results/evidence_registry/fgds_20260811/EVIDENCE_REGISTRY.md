# CuKD-XAI FG-DS Evidence Registry

Status: `passed`

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

## Runtime and Hardware Evidence

One fixed seed-42 deployment specimen was replayed over USB and Wi-Fi UDP on four board-model pairs. Each transport executed 225,204 full-test predictions. All pairs matched the fixed-point reference predictions and logits exactly.

| Pair | MCU macro-F1 | MCU vs FP32 | USB compute mean (us) | Wi-Fi compute mean (us) | Wi-Fi retransmissions |
|---|---:|---:|---:|---:|---:|
| esp32c3_student_A | 0.905694 | 0.995009 | 116.413 | 162.449 | 101 |
| esp32c3_student_B | 0.914564 | 0.994743 | 320.237 | 401.424 | 0 |
| arduino_r4_student_A | 0.905694 | 0.995009 | 301.080 | 300.199 | 1 |
| arduino_r4_student_B | 0.914564 | 0.994743 | 791.019 | 789.977 | 3 |

USB and Wi-Fi use the same model artifacts and test records. They are execution replications, not independent predictive samples. Wi-Fi retransmissions are application-level retries after response timeouts.

## Secondary Edge-IIoTset Evidence

The group-aware Edge run uses 40 inputs and 1,556,588/332,240/330,373 train/validation/test rows. Pre-encode group overlap is zero; encoded exact-row overlaps remain 163, 157, and 26 for train-test, train-validation, and validation-test.

## Boundaries

Historical 56,200-row QAT and SHAP artifacts, the legacy MSP430 static report, and the random-row full-route/curriculum/co-distillation evidence are retained but are not primary FG-DS evidence. See `evidence_registry.json` for the machine-readable lineage rules and source hashes.
