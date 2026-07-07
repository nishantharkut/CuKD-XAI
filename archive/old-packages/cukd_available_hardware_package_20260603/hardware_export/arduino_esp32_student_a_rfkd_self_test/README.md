# Arduino / ESP32 Student A RF-KD Self-Test

This sketch runs the generated fixed-point CuKD-XAI Student A RF-KD inference core on Arduino-style hardware and reports whether on-device outputs match the generated fixed-point reference vectors.

## Recommended Board

Use ESP32 first. It has much more flash/RAM than small Arduino boards and is a better available microcontroller proxy for this project.

Use small Arduino boards only with a small test-vector header, such as 256 vectors.

## Required Files In The Sketch Folder

The Arduino sketch folder must contain:

- `arduino_esp32_student_a_rfkd_self_test.ino`
- `wsnds_student_a_rfkd_int8_inference.c`
- `model_weights.h`
- `test_vectors.h`

Use the packager from the repo root:

```bash
python3 hardware_export/prepare_arduino_esp32_package.py \
  --generated-dir hardware_export/generated_student_a_rfkd_hw_256 \
  --output-dir hardware_export/arduino_esp32_student_a_rfkd_package
```

On Windows PowerShell:

```powershell
python .\hardware_export\prepare_arduino_esp32_package.py `
  --generated-dir hardware_export\generated_student_a_rfkd_hw_256 `
  --output-dir hardware_export\arduino_esp32_student_a_rfkd_package
```

Open the generated `.ino` in Arduino IDE.

## Arduino IDE Steps

1. Select your board, preferably ESP32.
2. Select the correct serial port.
3. Compile and upload.
4. Open Serial Monitor at `115200`.
5. Record the printed values.

Expected pass condition:

```text
prediction_failures = 0
logit_failures = 0
predict_wrapper_failures = 0
status = passed
```

## What To Report

Record:

- board model
- Arduino core / compiler version if visible
- number of vectors
- `prediction_failures`
- `logit_failures`
- `predict_wrapper_failures`
- `elapsed_us`
- `avg_us_per_vector`

This is an available-hardware microcontroller proxy test. It should not be described as live WSN mote deployment unless the same core is tested on an actual WSN mote or target-compatible WSN board.
