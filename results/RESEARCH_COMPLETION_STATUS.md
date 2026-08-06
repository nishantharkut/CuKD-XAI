# CuKD-XAI Research Completion Status (Tier-1 bar)

**Date:** 2026-08-06  
**Manuscript policy:** **NOT complete. Do not submit.** Finish research first, then rewrite manuscript against frozen evidence.

This document is the research gate, not a paper draft.

---

## 1. What “tier-1 / no-rejection-scope” means here

A reviewer can reject if any of these fail:

1. **Primary multi-seed numbers** come from a disclosed, leakage-safe protocol with complete aggregation.
2. **Deployment chain** is consistent with that protocol (scaler, seed, model, fixed-point, board).
3. **Negative controls / limitations** are measured, not only narrated (feature-group, duplicates, QAT fail, gate policy).
4. **Artifacts are reproducible** (hashes, checkpoints, one command path, integrity tests).
5. **Claims never mix lineages** (archived pre-split vs train-only vs feature-group).

Hardware unavailability does **not** block software research completion. It only blocks new board replays.

---

## 2. Evidence lineages (do not mix)

| Lineage | Protocol | Multi-seed? | Deployment HIL? | Use |
|---|---|---|---|---|
| **L0 Archived** | Pre-split global scaler, random-row split | Yes (10 seeds, includes J) | Yes (archived board_replay) | Historical / compare-only until replaced |
| **L1 Train-only 10-seed** | Train-only scaler, same random-row split | **Yes (10 seeds complete in checkpoints)** | No (different unit from deploy) | **Primary multi-seed predictive** |
| **L1b Deployment clean seed-42** | Train-only scaler; `set_seed(42)` then RF-KD only | Single seed | **Yes four-pair HIL** | **Primary deployment / HIL / fixed-point** |
| **L2 Feature-group 5-seed** | Train-only scaler, exact feature-group disjoint split | Yes (5 seeds; RF-KD + scratch only) | No | Leakage-control **sensitivity** |
| **L3 Edge-IIoT** | Train-only preprocess; two CSVs | Yes (5 seeds) | No | Robustness / protocol sensitivity |

---

## 3. Critical integrity finding (fixed in this session)

### Problem
`results/wsnds/leakage_free_rerun/main_10seed/wsnds_results_student_{A,B}.csv` and `cukd_xai_results.json` report **`n_seeds = 2`** (only 8192/9999).

### Reality
All **20 checkpoints** exist with full route metrics for seeds  
`{42,123,456,789,1001,2024,3141,5678,8192,9999}`.

### Fix (software, done)
Full re-aggregation (copy outputs only):

- `results/wsnds/leakage_free_rerun/main_10seed_full_aggregate_copy/wsnds_results_student_A_10seed.csv`
- `results/wsnds/leakage_free_rerun/main_10seed_full_aggregate_copy/wsnds_results_student_B_10seed.csv`
- `results/wsnds/leakage_free_rerun/main_10seed_full_aggregate_copy/full_10seed_aggregate_report.json`

### Headline train-only 10-seed RF-KD (ddof=1)
| Student | Macro-F1 mean ± std | Accuracy mean |
|---|---:|---:|
| A `E_KD_from_RF` | **0.9203 ± 0.0065** | 0.9869 |
| B `E_KD_from_RF` | **0.9392 ± 0.0129** | 0.9902 |

(Exact values in CSV/report; re-run builder if regenerating.)

### Incomplete file policy
**Do not cite** `main_10seed/wsnds_results_student_*.csv` as primary. Treat as incomplete merge residue.

---

## 4. What is research-complete (software / already-collected HIL)

| Work package | Status | Notes |
|---|---|---|
| Train-only preprocessing contract | **Complete** | `leakage_free_preprocessing.json` |
| 10×2 student checkpoints | **Complete** | 20 files |
| Full 10-seed aggregation | **Complete (copy)** | see §3 |
| Recovered Holm/Wilcoxon package | **Complete** | `recovered_main_10seed_v1/` |
| Seed-42 RF-KD deployment weights | **Complete** | `deployment_seed_42/` |
| QAT probe (negative for A absolute F1) | **Complete** | documented non-selection |
| Fixed-point export A/B (copy gate 0.03) | **Complete** | drop A≈0.024, B≈0.027; gate 0.03 |
| Host C self-test binaries | **Complete** | exit 0 observed |
| ONNX FP32 + dynamic INT8 | **Complete** | FP32 agree 1.0 vs PyTorch |
| OpenVINO FP32 from ONNX | **Complete** | agree 1.0 vs PT/ORT |
| Four-pair full 56,200 HIL (train-only copy) | **Complete** | MCU/fixed 1.0 all pairs |
| Compile footprints + smoke reconfirm | **Complete** | collected while hardware available |
| Feature-group 5-seed RF-KD/scratch | **Complete** | descriptive sensitivity |
| Edge duplicate audit | **Complete as audit** | not yet elevated into final claim matrix |
| Master HIL/runtime package | **Complete** | `TIER15_MASTER_REPORT.*` |

---

## 5. What is **not** research-complete (blocks “final paper”)

### P0 — must close before manuscript freeze

1. **Promote L1 (train-only 10-seed) as primary statistical evidence**  
   - Rebuild paper tables from `main_10seed_full_aggregate_copy/`, not archived L0, **or**  
   - Explicit dual-table design with L0 as “historical pre-split” only.  
   - **Manuscript must not be finished until this choice is frozen.**

2. **Gate policy for fixed-point export (0.01 vs 0.03)**  
   - Strict exporter still enforces **0.01**; train-only PTQ needs **≈0.024–0.027**.  
   - Tier-1 options (pick one, document scientifically):  
     a) Improve PTQ/QAT until **0.01** passes without absolute F1 collapse, **or**  
     b) Publish **measured** float→fixed drop bound (0.03) with class-wise drift tables and refuse to call it “0.01-strict”.  
   - Current copy pipeline is (b) de facto; not yet a formal publication decision.

3. **Co-distillation (J) under train-only**  
   - L1 checkpoints **do not include** `J_CoDistill_RF_CL`.  
   - Any final claim that co-distillation is/ isn’t better **under train-only** is currently unsupported.  
   - Options: retrain J under L1, or permanently restrict co-distill claims to L0 with explicit lineage labels.

4. **Curriculum-ext seed-5678 collapse**  
   - At least one curriculum-ext run collapses to macro-F1 ≈ 0.41.  
   - Must remain visible; decide reporting rule (include with note vs primary-route exclusion with reason).

5. **SHAP / explanation audit on deployed RF-KD train-only model** — **DONE**  
   - Results: `results/paper_strength_e2e/shap_train_only_deployment/shap_results.json`  
   - Deployment A/B global Spearman vs RF: **ρ≈0.24 / 0.23**, both **non-significant**; bootstrap means ~0.18 / 0.16.  
   - Claim freeze: **C6** allowed, **X5** forbids “preserves RF ranks.”  

### P1 — strong for tier-1, software-doable without boards

6. **Side-by-side lineage comparison package** (L0 vs L1 vs L2) for RF-KD A/B only.  
7. **Integrity tests** that fail CI if aggregates regress to n_seeds≠10 or HIL agree≠1.0.  
8. **Edge-IIoT duplicate findings** integrated into robustness conclusions (already audited).  
9. **Formal strict HIL report generator** on train-only evidence (env JSON, stage report) — can synthesize from existing files without re-running boards if sequences are intact.

### P2 — requires hardware again (user will notify)

10. Optional: re-flash + env_check + formal `generate_strict_report` under non-`_copy` paths if gate 0.01 becomes achievable.  
11. Optional: energy / radio — **out of current scope**; state as future work, never imply measured.

### Explicitly out of scope (do not block software completion)

- Live packet-to-feature pipeline  
- Physical MSP430/TelosB execution  
- Adversarial robustness / drift  
- Full OpenVINO multi-route archive redo (train-only RF-KD already done)

---

## 6. Deployment / HIL readiness (already collected)

Four-pair train-only seed-42 RF-KD:

| Board | Student | MCU/fixed | MCU/FP32 | macro-F1 | mean latency |
|---|---|---:|---:|---:|---:|
| ESP32-C3 | A | 1.0 | 0.9919 | 0.9244 | 116.5 µs |
| ESP32-C3 | B | 1.0 | 0.9905 | 0.9180 | 320.3 µs |
| Arduino R4 | A | 1.0 | 0.9919 | 0.9244 | 301.5 µs |
| Arduino R4 | B | 1.0 | 0.9905 | 0.9180 | 791.4 µs |

These support **numerical deployment fidelity** under train-only seed-42 RF-KD. They do **not** by themselves justify multi-seed predictive claims (use L1 tables for that).

---

## 7. Manuscript status (explicit)

| Item | Status |
|---|---|
| Draft `main.tex` exists | Yes — **draft only** |
| Claim text partially updated | Yes — **provisional** |
| PDF rebuilt after partial edits | Yes — **still not final** |
| Frozen to complete research evidence | **No** |
| Ready for submission | **No** |

**Rule:** When research P0 items are closed, rewrite manuscript tables/abstract/threats once from frozen L1/L2/HIL packages. Avoid further piecemeal manuscript polishing until then.

---

## 8. Recommended completion order (no hardware)

1. Freeze L1 10-seed tables (`main_10seed_full_aggregate_copy`) as primary predictive evidence.  
2. Decide gate policy 0.01 vs measured 0.03 (document decision record).  
3. Decide J/co-distill: retrain under train-only **or** lineage-limit claims.  
4. SHAP decision for RF-KD train-only (run or disclaim).  
5. Lineage comparison pack + integrity tests.  
6. **Only then** complete manuscript (tables, claims, threats, reproducibility).  
7. When hardware returns: only if a P0 decision requires new board evidence.

---

## 9. Sign-off checklist (research complete)

- [x] L1 10-seed full aggregates built (`main_10seed_full_aggregate_copy/`)  
- [x] Incomplete n=2 CSVs quarantined  
- [x] Dual identity frozen (multi-seed seed-42 ≠ deployment seed-42) — `results/paper_strength_e2e/01_dual_identity_freeze.json`  
- [x] Protocol ladder + per-class Δ tables — `results/paper_strength_e2e/02_*`, `03_*`  
- [x] Gate policy **B** frozen for deployment unit (measured 0.03) — `04_gate_policy_freeze.json`  
- [x] Claim freeze (allowed / forbidden) — `06_claim_freeze.json`  
- [x] FG aggregate verified consistent with per-seed CSVs  
- [x] Integrity tests green (`test_paper_strength_e2e_copy` + `test_train_only_research_integrity_copy`, 15 passed)  
- [x] Co-distill J under train-only 10-seed — `main_10seed_train_only_plus_j/` (A 0.9158 / B 0.9323; underperforms RF-KD)  
- [x] Seed-5678 CL-ext: original collapse 0.41 + isolated recovery 0.9145 — `leftover_e2e_closure/02_seed5678_clext/`  
- [x] XAI: SHAP on deployment RF-KD A/B vs RF teacher (`paper_strength_e2e/shap_train_only_deployment/`)  
- [x] Per-route `set_seed` multi-seed retrain D+E — `leftover_e2e_closure/03_per_route_set_seed/`  
- [x] Edge literature group-aware 5-seed — `leftover_e2e_closure/04_edge_group_aware/`  
- [x] Manuscript rewrite from claim freeze (train-only primary + J) — `manuscript/scripts/rewrite_from_freeze.py`  
- [x] Hardware re-HIL not required (subject weights / 0.01 PTQ unchanged)

### Active E2E packages

1. **`results/paper_strength_e2e/`** — dual identity, ladder, gate, SHAP, claim freeze  
2. **`results/leftover_e2e_closure/`** — J, 5678, reseed, edge group-aware, master report  

**Research leftover table is complete.** Manuscript text rewritten from freeze; human review of freeze + PDF rebuild still recommended before submission.
