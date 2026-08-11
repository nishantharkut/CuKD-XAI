# WSN-DS Feature-Group 10-Seed Confirmation

All values below were recomputed from the manifest-bound prediction CSV files.

| Route | Macro-F1 mean | Sample SD | Teacher gap | Retention |
|---|---:|---:|---:|---:|
| Calibrated RF teacher | 0.979500 | 0.000489 | 0.000000 | 1.000000 |
| Student A scratch | 0.914792 | 0.005658 | 0.064708 | 0.933938 |
| Student A RF-KD | 0.913781 | 0.004546 | 0.065719 | 0.932905 |
| Student B scratch | 0.932867 | 0.005727 | 0.046633 | 0.952391 |
| Student B RF-KD | 0.932142 | 0.010930 | 0.047358 | 0.951651 |

## Paired RF-KD versus scratch

| Student | Mean difference | Exact p | Holm p | Reject at 0.05 |
|---|---:|---:|---:|---:|
| Student A | -0.001012 | 0.556641 | 1.000000 | false |
| Student B | -0.000725 | 0.845703 | 1.000000 | false |

## Evaluation boundary

Ten paired optimizer seeds on one fixed feature-group-disjoint split; the analysis does not estimate variation across independently sampled splits.

Two-sided paired Wilcoxon signed-rank permutation test obtained by enumerating all sign assignments after removing zero differences; Holm correction is applied across the Student A and Student B tests.
