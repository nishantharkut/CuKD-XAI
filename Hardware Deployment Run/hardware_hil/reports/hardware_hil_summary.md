# CuKD-XAI Hardware HIL Summary

This report covers MCU-class replay of WSN-DS 17-feature records. It does not claim live WSN packet capture, energy measurement, or physical TelosB deployment.

## Fidelity

| Board | Vectors | MCU vs Fixed Ref | MCU vs FP32 | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|---:|
| esp32c3 | 56200 | 1.000000 | 0.995000 | 0.985623 | 0.914014 | 0.985732 |
| arduino_r4 | 56200 | 1.000000 | 0.995000 | 0.985623 | 0.914014 | 0.985732 |

## Claim Boundary

The hardware experiments validate firmware-level fixed-point execution on available MCU-class development boards using replayed WSN-DS records. They exclude live packet-to-feature extraction and energy profiling.
