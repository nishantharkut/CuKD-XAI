# FG-DS Wi-Fi UDP Hardware-in-the-Loop Report

| Board | Student | Vectors | Fixed pred. | Exact logits | FP32 agree. | Macro-F1 | Compute mean (us) | Host RTT mean (us) | Retries | Flash (B) | RAM (B) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| esp32c3 | student_A | 56301 | 1.000000 | 1.000000 | 0.995009 | 0.905694 | 162.449 | 48830.028 | 101 | 956091 | 40236 |
| esp32c3 | student_B | 56301 | 1.000000 | 1.000000 | 0.994743 | 0.914564 | 401.424 | 9570.774 | 0 | 958447 | 40236 |
| arduino_r4 | student_A | 56301 | 1.000000 | 1.000000 | 0.995009 | 0.905694 | 300.199 | 98486.718 | 1 | 71548 | 12568 |
| arduino_r4 | student_B | 56301 | 1.000000 | 1.000000 | 0.994743 | 0.914564 | 789.977 | 148823.039 | 3 | 73900 | 12568 |

## Measurement Boundary

The MCU timed code region and host-observed datagram RTT are separate measurements. MCU wall-clock timing may include interrupt preemption. Their difference is not reported as pure wireless latency.

## Claim Boundary

Controlled-LAN Wi-Fi UDP replay of extracted FG-DS features into exact fixed-point MCU inference. The evidence does not establish live WSN capture, on-device feature extraction, transport security, energy efficiency, BLE, or physical WSN-radio deployment.
