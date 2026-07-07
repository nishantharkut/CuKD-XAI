# Final HIL Post-Processing Analysis

Boundary: this report post-processes existing replay logs, reference CSVs, and optional compile logs. It does not run new hardware, measure energy, perform live WSN packet capture, or validate packet-to-feature extraction.

## HIL Fidelity
| Model | Board | Vectors | Accuracy | Macro-F1 | MCU vs Fixed | MCU vs FP32 | Mean Total us | P99 Total us |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Student A RF-KD | ESP32-C3 DevKitM-1 | 56200 | 0.98562 | 0.91401 | 1.00000 | 0.99500 | 118.40 | 125 |
| Student A RF-KD | Arduino R4 WiFi | 56200 | 0.98562 | 0.91401 | 1.00000 | 0.99500 | 301.63 | 305 |
| Student B RF-KD | ESP32-C3 DevKitM-1 | 56200 | 0.98696 | 0.91810 | 1.00000 | 0.99390 | 332.33 | 338 |
| Student B RF-KD | Arduino R4 WiFi | 56200 | 0.98696 | 0.91810 | 1.00000 | 0.99390 | 791.57 | 795 |

## Cycles Per MAC
| Model | Board | Clock MHz | MACs | Mean Inference us | Inference Cycles | Cycles/MAC | Total Throughput Ceiling/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Student A RF-KD | ESP32-C3 DevKitM-1 | 160 | 1136 | 112.31 | 17970 | 15.82 | 8445.7 |
| Student A RF-KD | Arduino R4 WiFi | 48 | 1136 | 280.36 | 13457 | 11.85 | 3315.3 |
| Student B RF-KD | ESP32-C3 DevKitM-1 | 160 | 3296 | 325.08 | 52013 | 15.78 | 3009.1 |
| Student B RF-KD | Arduino R4 WiFi | 48 | 3296 | 770.68 | 36993 | 11.22 | 1263.3 |

Throughput ceiling is computed from on-device measured total processing time only. It is not a claim about serial, radio, or live network packet throughput.

## Model-Only Fixed-Point Footprint
| Model | Architecture | MACs | Weight Bytes | Bias Bytes | Param Bytes | Format |
| --- | --- | --- | --- | --- | --- | --- |
| Student A RF-KD | 17-32-16-5 | 1136 | 1136 | 212 | 1348 | int8 weights + int32 biases + int16 activations |
| Student B RF-KD | 17-64-32-5 | 3296 | 3296 | 404 | 3700 | int8 weights + int32 biases + int16 activations |

## Compile And Framework Baseline
| Model | Board | Program Bytes | Global Bytes | Serial Baseline Program | Serial Baseline Globals | Program Delta | Global Delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Student A RF-KD | ESP32-C3 DevKitM-1 | 278836 | 13556 | 274156 | 13164 | 4680 | 392 |
| Student A RF-KD | Arduino R4 WiFi | 56104 | 7128 | 51856 | 6740 | 4248 | 388 |
| Student B RF-KD | ESP32-C3 DevKitM-1 | 281192 | 13556 | 274156 | 13164 | 7036 | 392 |
| Student B RF-KD | Arduino R4 WiFi | 58440 | 7128 | 51856 | 6740 | 6584 | 388 |

If baseline columns show `NR`, compile the serial-baseline sketches and paste the Arduino IDE output into `results/hardware_hil/compile_logs/*_serial_baseline_compile.txt`, then rerun this script.

## Quantization Drift Summary
| Model | Reference Rows | Drift Count | Drift Fraction | Reference |
| --- | --- | --- | --- | --- |
| Student A RF-KD | 56200 | 281 | 0.005000 | deployment\firmware_export\wsnds_rfkd_hil\generated_student_a_rfkd_hil_full\hil_reference_predictions.csv |
| Student B RF-KD | 56200 | 343 | 0.006103 | deployment\firmware_export\wsnds_rfkd_hil\generated_student_b_rfkd_hil_full\hil_reference_predictions.csv |

## Quantization Drift By True Class
| Model | True Class | Drift Count | Fraction Of All Rows | Reference |
| --- | --- | --- | --- | --- |
| Student A RF-KD | Blackhole | 107 | 0.001904 | deployment\firmware_export\wsnds_rfkd_hil\generated_student_a_rfkd_hil_full\hil_reference_predictions.csv |
| Student A RF-KD | Grayhole | 150 | 0.002669 | deployment\firmware_export\wsnds_rfkd_hil\generated_student_a_rfkd_hil_full\hil_reference_predictions.csv |
| Student A RF-KD | Normal | 23 | 0.000409 | deployment\firmware_export\wsnds_rfkd_hil\generated_student_a_rfkd_hil_full\hil_reference_predictions.csv |
| Student A RF-KD | TDMA | 1 | 0.000018 | deployment\firmware_export\wsnds_rfkd_hil\generated_student_a_rfkd_hil_full\hil_reference_predictions.csv |
| Student B RF-KD | Blackhole | 261 | 0.004644 | deployment\firmware_export\wsnds_rfkd_hil\generated_student_b_rfkd_hil_full\hil_reference_predictions.csv |
| Student B RF-KD | Grayhole | 65 | 0.001157 | deployment\firmware_export\wsnds_rfkd_hil\generated_student_b_rfkd_hil_full\hil_reference_predictions.csv |
| Student B RF-KD | Normal | 16 | 0.000285 | deployment\firmware_export\wsnds_rfkd_hil\generated_student_b_rfkd_hil_full\hil_reference_predictions.csv |
| Student B RF-KD | TDMA | 1 | 0.000018 | deployment\firmware_export\wsnds_rfkd_hil\generated_student_b_rfkd_hil_full\hil_reference_predictions.csv |

Drift means `fixed_pred != fp32_pred` in the generated reference CSV. Because MCU-vs-fixed agreement is expected to be 1.00000, this is the fixed-point quantization drift profile, not a serial transport error.

## Evidence Traceability
| Claim | Source | Status |
| --- | --- | --- |
| Student A RF-KD ESP32-C3 DevKitM-1 HIL metrics | results\hardware_hil\board_replay\pi5_esp32c3\full_56200_metrics.json | present |
| Student A RF-KD ESP32-C3 DevKitM-1 raw MCU replay CSV | results/hardware_hil/board_replay/pi5_esp32c3/full_56200_mcu.csv OR results/hardware_hil/board_replay/esp32c3/full_56200_mcu.csv | present |
| Student A RF-KD ESP32-C3 DevKitM-1 compile summary | results/hardware_hil/compile_logs/esp32c3_student_a_compile.txt | present |
| Student A RF-KD FP32-to-fixed drift profile | deployment\firmware_export\wsnds_rfkd_hil\generated_student_a_rfkd_hil_full\hil_reference_predictions.csv | present |
| Student A RF-KD Arduino R4 WiFi HIL metrics | results\hardware_hil\board_replay\pi5_arduino_r4\full_56200_metrics.json | present |
| Student A RF-KD Arduino R4 WiFi raw MCU replay CSV | results/hardware_hil/board_replay/pi5_arduino_r4/full_56200_mcu.csv OR results/hardware_hil/board_replay/arduino_r4/full_56200_mcu.csv | present |
| Student A RF-KD Arduino R4 WiFi compile summary | results/hardware_hil/compile_logs/arduino_r4_student_a_compile.txt | present |
| Student B RF-KD ESP32-C3 DevKitM-1 HIL metrics | results\hardware_hil\board_replay\pi5_esp32c3_student_b\full_56200_metrics.json | present |
| Student B RF-KD ESP32-C3 DevKitM-1 raw MCU replay CSV | results/hardware_hil/board_replay/pi5_esp32c3_student_b/full_56200_mcu.csv | present |
| Student B RF-KD ESP32-C3 DevKitM-1 compile summary | results/hardware_hil/compile_logs/esp32c3_student_b_compile.txt | present |
| Student B RF-KD FP32-to-fixed drift profile | deployment\firmware_export\wsnds_rfkd_hil\generated_student_b_rfkd_hil_full\hil_reference_predictions.csv | present |
| Student B RF-KD Arduino R4 WiFi HIL metrics | results\hardware_hil\board_replay\pi5_arduino_r4_student_b\full_56200_metrics.json | present |
| Student B RF-KD Arduino R4 WiFi raw MCU replay CSV | results/hardware_hil/board_replay/pi5_arduino_r4_student_b/full_56200_mcu.csv | present |
| Student B RF-KD Arduino R4 WiFi compile summary | results/hardware_hil/compile_logs/arduino_r4_student_b_compile.txt | present |
| ESP32-C3 DevKitM-1 serial baseline compile summary | results/hardware_hil/compile_logs/esp32c3_serial_baseline_compile.txt | present |
| Arduino R4 WiFi serial baseline compile summary | results/hardware_hil/compile_logs/arduino_r4_serial_baseline_compile.txt | present |
