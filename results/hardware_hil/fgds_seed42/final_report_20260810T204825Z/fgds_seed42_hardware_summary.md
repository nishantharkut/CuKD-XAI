# Feature-group-disjoint WSN-DS HIL summary

| Run | Vectors | MCU/fixed | Exact logits | MCU/FP32 | Accuracy | Macro-F1 | Mean compute (us) | Compute p95 (us) | Compute p99 (us) | Flash bytes | RAM bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| arduino_r4_student_A | 56301 | 1.000000 | 1.000000 | 0.995009 | 0.984814 | 0.905694 | 301.080 | 304 | 305 | 56416 | 7128 |
| arduino_r4_student_B | 56301 | 1.000000 | 1.000000 | 0.994743 | 0.986004 | 0.914564 | 791.019 | 794 | 796 | 58768 | 7128 |
| esp32c3_student_A | 56301 | 1.000000 | 1.000000 | 0.995009 | 0.984814 | 0.905694 | 116.413 | 122 | 122 | 281792 | 13592 |
| esp32c3_student_B | 56301 | 1.000000 | 1.000000 | 0.994743 | 0.986004 | 0.914564 | 320.237 | 326 | 327 | 284148 | 13592 |

## Claim boundary

Operator-selected serial-endpoint replay of extracted WSN-DS feature records with firmware identity verification; no cryptographic board attestation, live packet capture, on-board feature extraction, energy measurement, or TelosB execution. The replay uses the preserved seed-42 RF-KD models and the 56,301-row feature-group-disjoint test partition. It establishes execution fidelity for that seed, not five-seed hardware replication.

Flash and RAM values are Arduino frontend reports associated with a binary that embeds the strict export and bundle IDs. The sketch/log/binary association detects accidental mixing but is not cryptographic Arduino-build attestation. FQBN, board-core, and toolchain strings are checked in the verbose log; the frontend version is operator-recorded. These values are not measured energy or runtime heap usage.

Board latency is preprocessing plus inference compute time measured by firmware micros(); it excludes USB serial transport and host overhead.
