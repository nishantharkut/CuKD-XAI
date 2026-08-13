# FG-DS Multi-Split Core Confirmation

Ten seeded exact-feature-group holdout assignments were evaluated. Each split-level comparison averages two paired optimizer seeds. Because the repeated holdouts overlap, this is descriptive sensitivity evidence.

| Student | Mean RF-KD minus scratch macro-F1 | Sample SD | Range | Positive splits |
|---|---:|---:|---:|---:|
| Student A | 0.007126 | 0.003730 | [0.003743, 0.016149] | 10/10 |
| Student B | 0.001163 | 0.004799 | [-0.006576, 0.007646] | 5/10 |

## Scope

This confirmation estimates sensitivity to ten exact-feature-group split seeds for the core scratch versus RF-KD comparison. Each split-level value averages two paired optimizer seeds. The repeated holdouts overlap and are therefore reported descriptively, not as independent replications. This does not replace the finalized ten-optimizer-seed result on the fixed primary split and does not cover the full route matrix, deployment, XAI, or Edge-IIoTset.
