# Critical investigation: FG aggregate + seed-42 deployment (blocking)

**Date:** 2026-08-06  
**Status:** Explained. Downstream hardware claims on deployment seed-42 remain valid only as **deployment-route artifacts**, not as “the seed-42 member of the train-only 10-seed table.”

---

## Bug #1 — Feature-group Student A RF-KD mean “wrong”?

### Verdict: **Not a source aggregation bug**

Source file:

`results/wsnds/confirmation_runs_v2/remote_winterfell_feature_group_5seed_20260805/feature_group_5seed/aggregate_results.json`

| Seed | A RF-KD macro-F1 (from seed_completion **and** recompute from predictions CSV) |
|-----:|----------------------------------------------------------------------------------:|
| 42 | 0.9160761302322431 |
| 123 | 0.9210575900826801 |
| 456 | 0.9101979257019259 |
| 789 | 0.9042445625698784 |
| 1001 | 0.918985647518111 |

- Numpy mean of these five values = **0.9141123712209677**
- Reported `macro_f1.mean` = **0.9141123712209677** (exact match)
- Recomputed from each `student_A_KD_from_RF_test_predictions.csv` matches seed_completion metrics to numerical precision
- Test set size under feature-group split: **56,301** rows (not 56,200) — expected for group-disjoint partitioning

### Where the false alarm came from

The per-seed list `[0.9103, 0.9234, 0.9113, 0.9124, 0.9235]` (mean ≈ 0.91618) **does not appear in the FG package**. It is not the content of `aggregate_results.json` or the prediction CSVs.  
`02_feature_group_vs_randomrow.csv` correctly copied the JSON mean; there is nothing to “un-patch” in the aggregator for this number.

### Action taken

- Regenerated FG summary with **explicit per-seed values** (no hand-patched mean).
- Added integrity check: `mean(values) == reported mean` and recompute-from-CSV.

---

## Bug #2 — Seed-42 deployment F1 0.9485 vs train-only 10-seed mean 0.9203

### Verdict: **Real protocol / training-path mismatch — not a wrong test-set size**

| Source | Student A RF-KD macro-F1 | Acc | n test |
|--------|-------------------------:|----:|-------:|
| **main_10seed checkpoint seed 42** | **0.924859** | 0.987740 | (evaluated on 56,200 test in that run) |
| **deployment_seed_42** (predictions recompute) | **0.948509** | 0.991423 | **56,200** |
| Train-only 10-seed mean (A RF-KD) | 0.920344 ± 0.006513 | 0.986943 | 10 seeds |

Z-scores vs the 10-seed A RF-KD distribution:

| Point | z |
|-------|--:|
| Checkpoint seed 42 (0.92486) | **+0.69** (normal) |
| Deployment seed 42 (0.94851) | **+4.32** (cannot be “that same 10-seed table’s seed 42”) |

### Test set is **not** the leak

Deployment:

- `test_indices` length **56,200**
- Predictions CSV **n=56,200**
- Split hash test = `64256808…` (same as train-only preprocessing report)
- Scaler fit partition = train only  
- **RF soft targets** `rf_train_probabilities.npy` **byte-identical** to `main_10seed/rf_soft_seed_42.npy`

So evaluation is on the **same archived random-row test partition** and the **same bound teacher soft targets**. The discrepancy is **not** “validation mixed into test” in the deployment metrics CSV.

### Root cause: different **student training trajectory** (RNG / pipeline)

**main_10seed** (`run_leakage_free_wsnds.executed.py` → `run_all_configs`):

1. `set_seed(seed)` **once** at the start of the seed.
2. Trains many models in order: RF teacher path, Full MLP, curriculum teachers, **scratch D**, **then** E RF-KD, then E2/F/G/I…
3. `StudentMLP(...)` for E is constructed **without** a fresh `set_seed(seed)`.
4. Therefore E’s initialization and mini-batch RNG depend on all prior training for that seed.

**deployment** (`run_tier15_confirmation` mode=deployment):

1. `set_seed(42)`.
2. Trains **only** RF-KD (no scratch/teacher chain in front).
3. Same KD_T=4, KD_ALPHA=0.7, TRAIN_CONFIG, same soft targets.

**Conclusion:** Deployment seed-42 RF-KD is a **clean seed-42-initialized KD student** on bound soft targets.  
The 10-seed table’s “seed 42” RF-KD is a **post-pipeline-RNG KD student**.  
They share protocol labels (train-only scaler, same split, same soft targets) but are **not the same trained weights** and must not be treated as the same statistical unit.

This matches the deployment contract note already stored:

> “One-seed deployment training is an artifact-generation route, not a multi-seed statistical estimate or an exact recovery of active-v1 weights.”

### Implications for hardware / fixed-point (already run)

| Claim | OK? |
|-------|-----|
| HIL measures the **deployment** integer model faithfully (MCU/fixed = 1.0) | **Yes** |
| Deployment FP32 test F1 = 0.9485 on the 56,200 test set | **Yes** (recomputed) |
| “This is the seed-42 point from the train-only 10-seed table” | **No — false** |
| Fixed-point drop ~0.024–0.027 on that deployment model | **Yes** — real PTQ property of **that** model |
| Need to redo HIL only if we switch to **checkpoint** weights | Only if research decision requires HIL of the multi-seed-pipeline seed-42 student |

**Gates “not fixed” because:**  
The 0.01 macro-F1 drop gate fails for the **actual deployment PTQ** (measured drop A≈0.024, B≈0.027). That is not a mis-read of F1 aggregation; it is the conversion loss of the high-performing clean-seed deployment student. Options remain:

- **A)** Improve PTQ/QAT until drop ≤ 0.01 without destroying absolute F1, re-export, re-HIL  
- **B)** Publish measured bound (0.03) for this deployment route and never call it 0.01-strict  
- **C)** Re-export/HIL from **checkpoint** seed-42 weights (F1≈0.925) if the paper’s “seed 42 of the 10-seed study” must be the hardware subject  

---

## Fixed-point export / gates — why not “fixed” earlier

There was no silent arithmetic bug in the gate numbers. What blocked “fixing gates” is a **policy + model-identity** issue:

1. Strict code still encodes **0.01** drop.  
2. Train-only **deployment** PTQ **cannot** pass 0.01 with current PTQ (drops ~2.4–2.7%).  
3. Relaxing to **0.03** in `*_copy` tools was operational, not a completed research decision.  
4. Until Bug #2 is labeled correctly, fixing gates on the wrong conceptual model wastes work.

**No further export/HIL should proceed** until the research choice among A/B/C above is made.

---

## Actions completed in this investigation

1. Recomputed FG metrics from all five prediction CSVs → confirms aggregate JSON.  
2. Recomputed deployment metrics from prediction CSVs → confirms 0.9485 / 0.9449.  
3. Compared soft-target arrays → identical.  
4. Located RNG/order mismatch in main_10seed `run_all_configs`.  
5. Wrote this report; regenerate FG comparison CSV with per-seed columns (no hand-patched mean).

## Required next decisions (human)

1. For **statistics / paper multi-seed tables:** use main_10seed full aggregates only; seed 42 row = **0.9249**, not 0.9485.  
2. For **hardware subject:** keep deployment weights (already HIL’d) **or** re-export from checkpoint weights.  
3. For **gates:** choose A, B, or C above explicitly.
