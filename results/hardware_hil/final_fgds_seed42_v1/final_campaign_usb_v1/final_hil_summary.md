# Final FGDS Seed-42 Hardware HIL Results

This report is derived from a deeply verified portable evidence archive. It reports USB serial replay only.

## Fidelity and task metrics

| Model | Board | Rows | Accuracy | Macro-F1 | Weighted-F1 | MCU vs fixed | MCU vs FP32 | Exact logits |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Student A scratch | ESP32-C3 | 56301 | 0.979876 | 0.903667 | 0.980667 | 1.000000 | 0.995879 | 1.000000 |
| Student A RF-KD | ESP32-C3 | 56301 | 0.986377 | 0.916959 | 0.986443 | 1.000000 | 0.997727 | 1.000000 |
| Student B RF-KD | ESP32-C3 | 56301 | 0.986288 | 0.917018 | 0.986395 | 1.000000 | 0.996998 | 1.000000 |
| Student B RF-KD | Arduino UNO R4 WiFi | 56301 | 0.986288 | 0.917018 | 0.986395 | 1.000000 | 0.996998 | 1.000000 |
| Student A RF-KD | Arduino UNO R4 WiFi | 56301 | 0.986377 | 0.916959 | 0.986443 | 1.000000 | 0.997727 | 1.000000 |
| Student A scratch | Arduino UNO R4 WiFi | 56301 | 0.979876 | 0.903667 | 0.980667 | 1.000000 | 0.995879 | 1.000000 |

## Fixed-point quality by model

The signed delta is fixed-point macro-F1 minus FP32 macro-F1. The gate uses the absolute delta.

| Model | FP32 macro-F1 | Fixed macro-F1 | Signed delta | Absolute gate delta | Fixed vs FP32 |
|---|---:|---:|---:|---:|---:|
| Student A scratch | 0.910288 | 0.903667 | -0.006621 | 0.006621 | 0.995879 |
| Student A RF-KD | 0.914624 | 0.916959 | 0.002335 | 0.002335 | 0.997727 |
| Student B RF-KD | 0.921481 | 0.917018 | -0.004462 | 0.004462 | 0.996998 |

## Device timing

Each entry uses three ordered 1,000-row repetitions on one physical board. The reported SD is the sample SD of the three repeat means. P95 and p99 are descriptive percentiles across the combined 3,000 device timings.

| Model | Board | Preprocess mean (us) | Inference mean (us) | Total mean (us) | Total repeat SD (us) | Total p95 (us) | Total p99 (us) |
|---|---|---:|---:|---:|---:|---:|---:|
| Student A scratch | ESP32-C3 | 6.612 | 111.243 | 117.855 | 0.017 | 124 | 125 |
| Student A RF-KD | ESP32-C3 | 6.645 | 111.287 | 117.932 | 0.076 | 124 | 125 |
| Student B RF-KD | ESP32-C3 | 7.781 | 313.335 | 321.116 | 0.082 | 328 | 329 |
| Student B RF-KD | Arduino UNO R4 WiFi | 20.730 | 772.726 | 793.455 | 0.028 | 797 | 797 |
| Student A RF-KD | Arduino UNO R4 WiFi | 20.759 | 281.585 | 302.344 | 0.036 | 306 | 306 |
| Student A scratch | Arduino UNO R4 WiFi | 21.051 | 281.225 | 302.276 | 0.038 | 305 | 306 |

## Host-observed transaction timing

These values are descriptive. They include serial transfer, operating-system overhead, response handling, and device computation; they are not pure transport latency.

| Model | Board | Host RTT mean (us) | Host RTT p95 (us) | Transaction mean (us) | Transaction p95 (us) |
|---|---|---:|---:|---:|---:|
| Student A scratch | ESP32-C3 | 15822.892 | 16932 | 15822.892 | 16932 |
| Student A RF-KD | ESP32-C3 | 16173.188 | 17133 | 16173.188 | 17133 |
| Student B RF-KD | ESP32-C3 | 16635.353 | 17748 | 16635.353 | 17748 |
| Student B RF-KD | Arduino UNO R4 WiFi | 15728.752 | 16822 | 15728.752 | 16822 |
| Student A RF-KD | Arduino UNO R4 WiFi | 14844.833 | 15816 | 14844.833 | 15816 |
| Student A scratch | Arduino UNO R4 WiFi | 14723.629 | 15820 | 14723.629 | 15820 |

## Compile and model footprint

| Model | Board | Program bytes | Program capacity | Global bytes | Dynamic-memory capacity | Parameter bytes | MACs/inference |
|---|---|---:|---:|---:|---:|---:|---:|
| Student A scratch | ESP32-C3 | 281882 | 1310720 | 13592 | 327680 | 1348 | 1136 |
| Student A RF-KD | ESP32-C3 | 281874 | 1310720 | 13592 | 327680 | 1348 | 1136 |
| Student B RF-KD | ESP32-C3 | 284230 | 1310720 | 13592 | 327680 | 3700 | 3296 |
| Student B RF-KD | Arduino UNO R4 WiFi | 58864 | 262144 | 7128 | 32768 | 3700 | 3296 |
| Student A RF-KD | Arduino UNO R4 WiFi | 56528 | 262144 | 7128 | 32768 | 1348 | 1136 |
| Student A scratch | Arduino UNO R4 WiFi | 56512 | 262144 | 7128 | 32768 | 1348 | 1136 |

## Per-class fixed-point F1

| Model | Blackhole | Flooding | Grayhole | Normal | TDMA |
|---|---:|---:|---:|---:|---:|
| Student A scratch | 0.859915 | 0.928302 | 0.800000 | 0.993395 | 0.936721 |
| Student A RF-KD | 0.837319 | 0.941738 | 0.846659 | 0.997894 | 0.961185 |
| Student B RF-KD | 0.842680 | 0.941738 | 0.845605 | 0.997795 | 0.957274 |

## Blocked route

- Student B scratch: Fixed-point quality gates failed: fixed/FP32 agreement 0.989574 is below 0.99; absolute macro-F1 drop 0.020193 exceeds 0.015. It was blocked before firmware generation on both boards.

## Evidence identity

- Report ID: `dda909ee24262574264b2515308636ed1aedecc410c3d6d7b87391406dc78b23`
- Archive ID: `79969dc146a3889983ddd28120578a7f74daa5b78977733985d3ae6a2a5e6fce`
- Campaign evidence ID: `922f46573eded7b63c605990b6434c3bbc8c89dfe00cdd5274b19eb2cf4fcad5`
- Contract ID: `aa123a4d668b9fd7f7c0c182f9811979e6291437c5ebd7f75cf7f25d8afc6c7a`

## Claim boundary

One seed-42 model specimen per fixed-point-gate-eligible route on one physical specimen of each board type. Gate-failed routes remain explicit non-deployment results. Exact replay and timing evidence do not establish multi-seed or multi-unit hardware variability, energy, or secure attestation.

The replay inputs are already extracted 17-feature WSN-DS records. The campaign does not measure live packet-to-feature extraction, energy, transport latency, multi-unit variation, or multi-seed hardware variation.

Compiled program storage includes the firmware and platform runtime. Static global memory is not a measurement of peak RAM usage.
