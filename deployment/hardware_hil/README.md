# CuKD-XAI Hardware HIL Package

This folder contains the dedicated hardware-in-the-loop validation package for
the CuKD-XAI WSN-DS RF-KD student models.

## Scope

This package validates MCU-class test-vector replay for the WSN-DS RF-KD
fixed-point student models:

- dataset: WSN-DS
- feature contract: 17-feature records
- Student A model: `E_KD_from_RF`, architecture `17 -> 32 -> 16 -> 5`
- Student B model: `E_KD_from_RF`, architecture `17 -> 64 -> 32 -> 5`
- host: Raspberry Pi 5
- primary MCU: ESP32-C3
- secondary MCU: Arduino R4

This is test-vector replay, not live packet capture, and not energy measurement. It does not claim TelosB/TinyOS/Contiki physical deployment.
Raspberry Pi AI HAT+ is future gateway work, not the core MCU result.

## Evidence Boundaries

The hardware replay validates fixed-point firmware execution and device-side
latency on available MCU boards. It must be reported separately from:

- Python/FP32 model metrics,
- host fixed-point export metrics,
- MSP430F1611 cross-compiled memory evidence.

Do not mix ESP32-C3 or Arduino R4 latency with MSP430 memory evidence as if
they came from the same physical device.

## Reused Artifacts

This package reuses:

- `deployment/msp430/run_wsnds_student_a_rfkd_e2e.py`
- `deployment/msp430/wsnds_student_a_rfkd_int8_inference.c`
- `deployment/msp430/wsnds_preprocess_int16.c`
- `deployment/msp430/MSP430_CROSS_COMPILE_REPORT.md`
- trained artifact path `origin/main:results/runtime/onnx_openvino/wsnds/tmp/E_student_A_KD_from_RF_fp32.pt`
- trained artifact path `origin/main:results/runtime/onnx_openvino/wsnds/tmp/E_student_B_KD_from_RF_fp32.pt`

Generated headers such as `model_weights.h` and `preprocess_int_metadata.h`
must be copied or included in the board firmware build.

## Execution Order

1. Generate fixed-point model and reference artifacts from `deployment/firmware_export/wsnds_rfkd_hil`.
2. Build the ESP32-C3 firmware.
3. Run a 10-row smoke replay from Raspberry Pi 5.
4. Run a 1,000-row validation replay.
5. Run the full 56,200-row replay if stable.
6. Repeat on Arduino R4 where memory and runtime permit.
7. Verify MCU logs against fixed-point reference predictions.
8. Generate report tables.

## Paper-Safe Claim

The hardware experiments validate firmware-level fixed-point execution of the
compressed WSN-DS Student A and Student B IDS cores on available MCU-class
development boards using replayed 17-feature WSN-DS records. They do not claim
live WSN packet capture, raw packet-to-feature extraction, energy profiling, or
physical TelosB deployment.



## Beginner Runbook

Start with `deployment/hardware_hil/docs/00_READ_THIS_FIRST.md`, then follow:

1. `deployment/hardware_hil/docs/01_RASPBERRY_PI5_HOST_SETUP.md`
2. `deployment/hardware_hil/docs/02_GENERATE_REPLAY_ASSETS.md`
3. `deployment/hardware_hil/docs/03_BUILD_FIRMWARE_BUNDLES.md`
4. `deployment/hardware_hil/docs/04_FLASH_ESP32C3.md`
5. `deployment/hardware_hil/docs/06_RUN_REPLAY_AND_VERIFY.md`
6. `deployment/hardware_hil/docs/05_FLASH_ARDUINO_R4.md` for the secondary board
7. `deployment/hardware_hil/docs/07_TROUBLESHOOTING.md` if serial or agreement issues appear

Official hardware documentation links are collected in `deployment/hardware_hil/docs/OFFICIAL_REFERENCES.md`.

Student B capacity runbook: `deployment/hardware_hil/docs/10_STUDENT_B_HIL_RUNBOOK.md`.

Final compact evidence table: `results/hardware_hil/reports/final_hardware_hil_results_table.md`.



