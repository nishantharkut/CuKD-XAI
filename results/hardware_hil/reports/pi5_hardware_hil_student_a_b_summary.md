# CuKD-XAI Hardware HIL Summary

This report covers MCU-class replay of WSN-DS 17-feature records. It does not claim live WSN packet capture, energy measurement, or physical TelosB deployment.

## Fidelity

| Board | Vectors | MCU vs Fixed Ref | MCU vs FP32 | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|---:|
| pi5_esp32c3_student_a | 56200 | 1.000000 | 0.995000 | 0.985623 | 0.914014 | 0.985732 |
| pi5_arduino_r4_student_a | 56200 | 1.000000 | 0.995000 | 0.985623 | 0.914014 | 0.985732 |
| pi5_esp32c3_student_b | 56200 | 1.000000 | 0.993897 | 0.986957 | 0.918099 | 0.986946 |
| pi5_arduino_r4_student_b | 56200 | 1.000000 | 0.993897 | 0.986957 | 0.918099 | 0.986946 |

## Claim Boundary

The hardware experiments validate firmware-level fixed-point execution on available MCU-class development boards using replayed WSN-DS records. They exclude live packet-to-feature extraction and energy profiling.

