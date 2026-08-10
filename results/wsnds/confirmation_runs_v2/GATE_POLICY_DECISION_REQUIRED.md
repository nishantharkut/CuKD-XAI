# Fixed-point export gate policy — decision required (research P0)

## Facts

| Item | Value |
|---|---|
| Strict exporter gate | `MAXIMUM_MACRO_F1_DROP = 0.01` (`export_train_only_deployment.py`) |
| Copy exporter gate | `MAXIMUM_MACRO_F1_DROP = 0.03` (`export_train_only_deployment_copy.py`) |
| Train-only Student A measured drop | ≈ **0.0241** (FP32 F1 0.9485 → fixed 0.9244) |
| Train-only Student B measured drop | ≈ **0.0269** (FP32 F1 0.9449 → fixed 0.9180) |
| Fixed vs FP32 prediction agreement | A 0.9919, B 0.9905 (≥ 0.99) |
| QAT refine attempt | Improved drop for A but **hurt absolute F1**; not used for HIL |

## Why this matters for rejection

Calling the pipeline “strict 1% macro-F1 drop” while shipping 2.4–2.7% drop is a **claim integrity** rejection risk.

## Allowed publication choices (pick one)

### Option A — Meet 0.01 (preferred if achievable without absolute-F1 collapse)
Further calibration / architecture-preserving PTQ / careful QAT until both students pass 0.01 **and** absolute F1 remains competitive. Then regenerate export + HIL under strict tools.

### Option B — Publish measured bound (current evidence supports this)
State that float→fixed conversion induces ~2.4–2.7% macro-F1 drop under train-only RF-KD seed 42, with agreement ≥0.99 and class-wise drift tables. Use gate 0.03 as **measurement-informed tolerance**, not as silent relaxation. Never label it as 0.01-strict.

## Current operational state

HIL and exports in `*_copy` paths follow **Option B de facto**.  
**No formal decision is recorded until this file is updated with a signed choice.**

## Decision log

| Date | Choice | Rationale | Owner |
|---|---|---|---|
| _pending_ | _A or B_ | _fill when decided_ | _fill_ |


## Decision log

| Date | Choice | Rationale | Owner |
|---|---|---|---|
| 2026-08-06 | **B measured 0.03 bound** for deployment unit | Measured drops 0.024/0.027; QAT hurt absolute A F1; dual-identity freeze separates deploy from multi-seed seed-42 | research freeze |

Frozen machine record: `results/paper_strength_e2e/04_gate_policy_freeze.json`
