# FG Student A RF-KD mean: 0.9141 vs “0.9162” — final reconciliation

**Date:** 2026-08-06 (re-verified from prediction CSVs)  
**Question:** Why does the report use **0.9141** when some earlier note listed seeds that average to **~0.9162**?

## Direct answer

| Claim | Verdict |
|---|---|
| Official FG mean 0.914112… is wrong | **False** |
| There is a second FG training run with mean 0.9162 | **No evidence in the repo** |
| Authoritative mean | **0.9141123712209677** from the only FG package that exists |

## Authoritative source

`results/wsnds/confirmation_runs_v2/remote_winterfell_feature_group_5seed_20260805/feature_group_5seed/`

Recomputed **today** from each `student_A_KD_from_RF_test_predictions.csv` (`true_label` / `predicted_label`, sklearn macro-F1):

| Seed | A RF-KD macro-F1 (from CSV) |
|-----:|----------------------------:|
| 42 | 0.9160761302322431 |
| 123 | 0.9210575900826801 |
| 456 | 0.9101979257019259 |
| 789 | 0.9042445625698784 |
| 1001 | 0.9189856475181110 |

- **Mean** = **0.9141123712209677**
- **Sample std (ddof=1)** = **0.006865452782879801**
- Matches `aggregate_results.json` and `feature_group_5seed_summary_copy.json` **exactly**

Same package, scratch (for paired Δ):

| Seed | A scratch | B RF-KD | B scratch |
|-----:|----------:|--------:|----------:|
| 42 | 0.910523 | 0.922381 | 0.925109 |
| 123 | 0.912210 | 0.934310 | 0.923900 |
| 456 | 0.911065 | 0.922832 | 0.937521 |
| 789 | 0.913867 | 0.923240 | 0.930715 |
| 1001 | 0.921726 | 0.937960 | 0.931729 |

Means: A scratch **0.913878**, B RF-KD **0.928145**, B scratch **0.929795**.

## Where “0.9162” came from

The list that averages to ≈0.91618:

`[0.9103, 0.9234, 0.9113, 0.9124, 0.9235]`

**does not appear** in:

- `aggregate_results.json`
- any `seed_*/seed_completion.json` student_A_rf_kd metrics
- any prediction CSV recompute
- the protocol ladder FG cells

It is a **false-alarm / mis-transcribed list** (likely confused with another protocol’s seed cells or a hand-written summary). It is **not** an alternate FG checkpoint set.

## What is the “real” training run?

**One FG run only** in this repo:

`remote_winterfell_feature_group_5seed_20260805/feature_group_5seed/`  
protocol_id: `wsnds_feature_group_split_train_only_scaler_5seed_v1`  
seeds: `{42,123,456,789,1001}`  
test size under FG split: **56,301** rows

There is nothing to un-patch or re-average. **Use 0.9141 ± 0.0069 for Student A RF-KD under FG.**

## Paired KD−scratch (confirmed)

mean(A RF-KD − A scratch) over the five seeds above = **+0.000234** (not significant).  
That conclusion is independent of the false 0.9162 list.
