# Paper-strength E2E package (frozen research objects)

Built by `build_paper_strength_package.py`. No retraining, no HIL re-run.

## Files

| File | Purpose |
|---|---|
| `01_dual_identity_freeze.json` | Multi-seed seed-42 vs deployment seed-42 |
| `02_protocol_ladder.csv/.json` | Absolute F1 + KD−scratch across protocols |
| `03_per_class_kd_minus_scratch.csv` | Minority-class tradeoffs |
| `04_gate_policy_freeze.json` | Option B measured 0.03 bound |
| `05_edge_duplicate_disclosure.csv` | Edge leakage framing |
| `06_claim_freeze.json` | Allowed vs forbidden claims |

## Headline ladder (KD − scratch macro-F1)

| Student | Archived RR | Train-only RR (paired) | FG disjoint (paired) |
|---|---:|---:|---:|
| A | +0.0077 | +0.0094 (t p=0.048) | +0.0002 (t p=0.946) |
| B | +0.0006 | +0.0078 (t p=0.097) | -0.0017 (t p=0.735) |

## Dual identity (Student A RF-KD)

- Multi-seed pipeline seed 42: **0.9249** (z=+0.69)
- Deployment clean seed 42: **0.9485** (z=+4.32)
- Soft targets identical: **True**

## Next (optional HW)

- Keep HIL on deployment unit (already complete) **or** re-HIL checkpoint weights if required.
- Manuscript rewrite uses `06_claim_freeze.json` only after review.
\n## SHAP (deployment RF-KD)\n\n| Student | rho | p | boot mean |\n|---|---:|---:|---:|\n| A | 0.2377 | 0.3582 | 0.1784 |\n| B | 0.2255 | 0.3842 | 0.1623 |\n\nDetails: shap_train_only_deployment/\n