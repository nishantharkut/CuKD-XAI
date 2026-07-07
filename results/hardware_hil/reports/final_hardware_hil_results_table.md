# Final Hardware HIL Results

Boundary: USB serial replay of already extracted WSN-DS 17-feature records. Metrics are copied from the committed HIL JSON outputs and Arduino IDE upload summaries listed below. These results do not claim live packet capture, energy measurement, or physical TelosB deployment.

| Model | Board | Vectors | Accuracy | Macro-F1 | Weighted-F1 | MCU vs Fixed | MCU vs FP32 | Mean Total Latency | P99 Total Latency | Arduino IDE Program Storage | Arduino IDE Global Variables |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Student A RF-KD | ESP32-C3 DevKitM-1 | 56,200 | 0.98562 | 0.91401 | 0.98573 | 1.00000 | 0.99500 | 118.40 us | 125 us | 278,836 B | 13,556 B |
| Student A RF-KD | Arduino R4 WiFi | 56,200 | 0.98562 | 0.91401 | 0.98573 | 1.00000 | 0.99500 | 301.63 us | 305 us | 56,104 B | 7,128 B |
| Student B RF-KD | ESP32-C3 DevKitM-1 | 56,200 | 0.98696 | 0.91810 | 0.98695 | 1.00000 | 0.99390 | 332.33 us | 338 us | 281,192 B | 13,556 B |
| Student B RF-KD | Arduino R4 WiFi | 56,200 | 0.98696 | 0.91810 | 0.98695 | 1.00000 | 0.99390 | 791.57 us | 795 us | 58,440 B | 7,128 B |

## Evidence Sources

- Student A ESP32-C3 metrics: `results/hardware_hil/board_replay/pi5_esp32c3/full_56200_metrics.json`
- Student A Arduino R4 metrics: `results/hardware_hil/board_replay/pi5_arduino_r4/full_56200_metrics.json`
- Student B ESP32-C3 metrics: `results/hardware_hil/board_replay/pi5_esp32c3_student_b/full_56200_metrics.json`
- Student B Arduino R4 metrics: `results/hardware_hil/board_replay/pi5_arduino_r4_student_b/full_56200_metrics.json`
- Compile summaries: `results/hardware_hil/compile_logs/*.txt`

## Numeric Comparison

Relative to Student A, Student B changes the full-replay metrics as follows:

- Accuracy: 0.98562 to 0.98696, an absolute change of +0.00133.
- Macro-F1: 0.91401 to 0.91810, an absolute change of +0.00409.
- MCU-vs-fixed-reference agreement: 1.00000 for both Student A and Student B on both boards.
- ESP32-C3 mean total latency: 118.40 us to 332.33 us, an absolute change of +213.93 us.
- Arduino R4 mean total latency: 301.63 us to 791.57 us, an absolute change of +489.94 us.

The Arduino IDE program-storage and global-variable values include the board firmware framework and serial replay harness; they are not pure model-only memory measurements.

## Claim Boundary

The hardware-in-the-loop evidence validates fixed-point inference and integer preprocessing against the generated fixed-point reference over replayed WSN-DS tabular records. It does not validate live WSN packet-to-feature extraction, energy consumption, or execution on a physical TelosB mote.

