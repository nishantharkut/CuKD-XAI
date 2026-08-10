# Leftover E2E Closure Master Report

**Status:** complete  
**Device:** CUDA (RTX 4050)  
**Hardware re-HIL:** **Not required** (deployment RF-KD subject weights and 0.01/0.03 PTQ path unchanged)

## Stage checklist

| Leftover | Status | Artifact |
|---|---|---|
| Co-distill J under train-only | **Done** | `results/wsnds/leakage_free_rerun/main_10seed_train_only_plus_j/` |
| Seed-5678 CL-ext collapse | **Done** | `results/leftover_e2e_closure/02_seed5678_clext/` |
| Edge lit group-aware | **Done** | `results/leftover_e2e_closure/04_edge_group_aware/` |
| Per-route set_seed retrain | **Done** | `results/leftover_e2e_closure/03_per_route_set_seed/` |
| Manuscript rewrite from freeze | **Done** | `manuscript/main.tex` + `manuscript/scripts/rewrite_from_freeze.py` |
| Hardware re-HIL | **N/A** | only if subject weights / 0.01 PTQ change |

## 1. Co-distill J (train-only, 10 seeds)

| Student | J macro-F1 | RF-KD macro-F1 | Scratch | J−E | J−D |
|---|---:|---:|---:|---:|---:|
| A | 0.9158 ± 0.0026 | 0.9203 ± 0.0062 | 0.9109 | **−0.0046** | +0.0049 |
| B | 0.9323 ± 0.0141 | 0.9392 ± 0.0123 | 0.9313 | **−0.0069** | +0.0009 |

**Claim:** Train-only co-distillation is **not** superior to RF-KD. Retire unqualified J-superiority (X4).

## 2. Seed-5678 CL-ext

| Condition | macro-F1 |
|---|---:|
| Original multi-config checkpoint | **0.4075** (collapse; BH/GH F1=0) |
| Isolated set_seed(5678) CL-ext re-run (5 trials) | **0.9145** (collapse rate 0.00) |
| Isolated fair-schedule re-run | 0.9195 |

**Finding:** Collapse is **multi-config RNG-path contingent**, not an inherent seed-5678 curriculum failure under clean seeding. Report original collapse with disclosure + recovery note.

## 3. Per-route set_seed (D + E, 10 seeds)

| Student/route | Per-route mean | Pipeline mean | Δ | paired t p |
|---|---:|---:|---:|---:|
| A D | 0.9131 | 0.9109 | +0.0022 | 0.478 |
| A E | 0.9138 | 0.9203 | **−0.0066** | **0.037** |
| B D | 0.9307 | 0.9313 | −0.0006 | 0.796 |
| B E | 0.9312 | 0.9392 | −0.0079 | 0.182 |

Seed-42 Student A E: per-route 0.9148 | pipeline 0.9249 | deployment-clean ref **0.9485**  
(Note: reseed recalibrated RF soft targets; deployment uses cached `rf_soft_seed_42.npy`.)

**Claim:** Dual identity is measured. Per-route set_seed alone does **not** recover deployment 0.9485 when soft targets are re-fit.

## 4. Edge literature group-aware (5 seeds)

| Protocol | Test cross-partition exposure | A RF-KD macro-F1 | A KD−scratch |
|---|---:|---:|---:|
| Literature random-row (prior) | ~17% test rows | ~0.812 | ~+0.003 |
| **Group-aware (this run)** | **0% pre-encode groups** (163 post-encode row overlaps) | **0.7676 ± 0.0079** | **+0.0340** |
| Group-aware B RF-KD | same | 0.8129 ± 0.0068 | +0.0073 |

Split sizes: train 1,556,588 / val 332,240 / test 330,373; input dim 40.

**Claim:** Leakage-safe group-aware evaluation lowers absolute A RF-KD vs random-row literature scores while preserving a positive KD−scratch margin for A.

## 5. Manuscript

- Abstract + main WSN table rewritten to **train-only primary** numbers including J.
- Claim freeze updated: C7–C10; X4 revised.
- Traceability: `manuscript/CLAIM_TRACEABILITY.md` leftover section.

## 6. Hardware re-HIL

**Not required.** No change to deployment subject weights or 0.01-strict PTQ path.
