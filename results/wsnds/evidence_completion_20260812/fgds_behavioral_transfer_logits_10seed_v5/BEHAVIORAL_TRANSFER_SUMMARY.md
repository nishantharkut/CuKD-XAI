# FG-DS Checkpoint-Logit Behavioral-Transfer Audit

This additive analysis reconstructs the finalized ten-seed student checkpoints on the fixed feature-group-disjoint WSN-DS test partition. It does not retrain a model.

## Primary Metric

The primary metric is exact-feature-group-balanced KL divergence from the calibrated RF distribution to direct checkpoint-logit student outputs at T=4. Positive transfer gain means scratch KL minus RF-KD KL is positive.

| Student | Scratch KL | RF-KD KL | Transfer gain | 95% t interval | Exact p | Holm p | Positive seeds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Student A | 0.207357 | 0.016179 | 0.191178 | [0.158654, 0.223702] | 0.001953 | 0.003906 | 10/10 |
| Student B | 0.206961 | 0.013259 | 0.193702 | [0.178339, 0.209065] | 0.001953 | 0.003906 | 10/10 |

## Scope

This analysis tests held-out in-distribution response-distribution transfer from the calibrated RF to each checkpoint-reconstructed student and compares RF-KD with matched scratch under the same T=4 output contract used by KD and XAI. It does not establish causal mechanism transfer, off-manifold decision-boundary equivalence, explanation transfer, or deployment fidelity.

The inference unit is the paired training-run/model seed on one fixed clean split. Per-class outputs and row-weighted values are descriptive sensitivities.
