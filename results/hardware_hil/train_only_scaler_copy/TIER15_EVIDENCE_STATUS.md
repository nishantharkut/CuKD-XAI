# Tier 1.5 Evidence Status (Train-Only Seed 42)

**Deployment/HIL package status: collected**  
**Overall research status: NOT complete** (see `results/RESEARCH_COMPLETION_STATUS.md`)  
**Manuscript status: draft only — finalize after research sign-off**  
**Master HIL/runtime report:** `TIER15_MASTER_REPORT.json` / `TIER15_MASTER_REPORT.md`  
Pi HIL host: `192.168.137.234`

## Full checklist

| Layer | Status | Primary path |
|---|---|---|
| Deployment seed-42 train-only RF-KD A/B | complete | `results/wsnds/confirmation_runs_v2/deployment_seed_42/` |
| Feature-group 5-seed sensitivity | complete | `.../remote_winterfell_feature_group_5seed_20260805/feature_group_5seed/` |
| QAT probe (not selected for HIL) | complete | `.../deployment_seed_42_qat/` |
| Fixed-point export A/B | complete | `generated_train_only_student_{A,B}_seed42_copy/` |
| Host C self-test binaries | complete (exit 0) | same export dirs (`cukd_train_only_self_test.exe`) |
| ONNX FP32 + dynamic INT8 | complete | `results/runtime/.../train_only_seed42_copy/` |
| OpenVINO FP32 from ONNX | complete (agree 1.0) | `train_only_openvino_results.json` |
| HIL four-pair full 56,200 | complete (MCU/fixed=1.0) | `four_pair_summary.json` |
| MCU vs FP32 agreement | complete (A 0.9919, B 0.9905) | metrics in each `pi5_*` folder |
| Compile footprints (4) | complete | `compile_evidence/compile_footprint_summary.json` |
| Smoke reconfirm (4x10 OK) | complete | `compile_evidence/smoke_*` |
| Manuscript claims | updated | `manuscript/main.tex`, `CLAIM_TRACEABILITY.md` |
| Manuscript validate | passed | `manuscript/generated/validation_report.json` |

## Headline numbers

### Predictive
- Train-only seed-42 RF-KD macro-F1: **A 0.9485**, **B 0.9449**
- Feature-group 5-seed RF-KD macro-F1: **A 0.9141+/-0.0069**, **B 0.9281+/-0.0074**
- Feature-group KD-minus-scratch mean delta: **A +0.0002**, **B -0.0017** (descriptive)

### Host conversion
- ONNX FP32 vs PyTorch agreement: **1.0**
- OpenVINO vs PyTorch / ORT agreement: **1.0**
- Dynamic INT8: size probe only (macro-F1 drop)

### Hardware
| Board | Student | MCU/fixed | MCU/FP32 | macro-F1 | mean us |
|---|---|---:|---:|---:|---:|
| ESP32-C3 | A | 1.0 | 0.9919 | 0.9244 | 116.5 |
| ESP32-C3 | B | 1.0 | 0.9905 | 0.9180 | 320.3 |
| Arduino R4 | A | 1.0 | 0.9919 | 0.9244 | 301.5 |
| Arduino R4 | B | 1.0 | 0.9905 | 0.9180 | 791.4 |

### Compile flash / RAM
| Board | Student | Flash used | RAM used |
|---|---|---:|---:|
| ESP32-C3 | A | 281776 | 13592 |
| ESP32-C3 | B | 284132 | 13592 |
| R4 | A | 56384 | 7128 |
| R4 | B | 58736 | 7128 |

## Claim boundary (do not over-claim)

1. **Archived 10-seed tables** = primary multi-seed predictive evidence (pre-split scaler lineage).
2. **Train-only seed 42** = deployment + conversion + HIL under correct scaler.
3. **Feature-group 5-seed** = leakage-control sensitivity; not a significance test vs archived route.
4. **Still out of scope:** live radio/energy/packet pipeline; physical MSP430; full 10-seed train-only random-row re-distribution.
