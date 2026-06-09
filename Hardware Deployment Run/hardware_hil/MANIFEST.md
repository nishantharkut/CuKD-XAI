# Hardware HIL Manifest

This manifest lists the files expected in a complete CuKD-XAI hardware HIL
evidence package.

## Static Files

- `README.md`
- `MANIFEST.md`
- `protocol/serial_protocol.md`
- `docs/00_READ_THIS_FIRST.md`
- `docs/01_RASPBERRY_PI5_HOST_SETUP.md`
- `docs/02_GENERATE_REPLAY_ASSETS.md`
- `docs/03_BUILD_FIRMWARE_BUNDLES.md`
- `docs/04_FLASH_ESP32C3.md`
- `docs/05_FLASH_ARDUINO_R4.md`
- `docs/06_RUN_REPLAY_AND_VERIFY.md`
- `docs/07_TROUBLESHOOTING.md`
- `docs/08_PHYSICAL_CONNECTIONS_AND_CABLES.md`
- `docs/09_OPTIONAL_BREADBOARD_AND_JUMPER_GUIDE.md`
- `docs/10_STUDENT_B_HIL_RUNBOOK.md`
- `docs/OFFICIAL_REFERENCES.md`
- `firmware/common/cukd_model.h`
- `firmware/common/cukd_model.c`
- `firmware/common/cukd_preprocess.h`
- `firmware/common/cukd_preprocess.c`
- `firmware/common/cukd_protocol.h`
- `firmware/common/cukd_protocol.c`
- `firmware/esp32c3/src/main.cpp`
- `firmware/arduino_r4/cukd_hil_r4/cukd_hil_r4.ino`
- `host/hil_common.py`
- `host/env_check.py`
- `host/stream_vectors.py`
- `host/verify_results.py`
- `host/generate_report.py`
- `host/prepare_firmware_bundle.py`
- `host/requirements.txt`
- `compile_logs/esp32c3_student_a_compile.txt`
- `compile_logs/arduino_r4_student_a_compile.txt`
- `compile_logs/esp32c3_student_b_compile.txt`
- `compile_logs/arduino_r4_student_b_compile.txt`
- `reports/final_hardware_hil_results_table.md`

## Generated Evidence

The following files are produced during real runs and should be preserved:

- dataset hash
- test split hash
- model artifact hash
- quantization metadata hash
- raw serial logs
- decoded MCU response CSVs
- board environment JSON
- toolchain build logs
- normalized memory reports
- metrics JSON
- report Markdown and CSV tables

## Claim Boundary

This package is for MCU test-vector replay. It is not a live WSN deployment,
not packet capture, not energy measurement, and not an AI HAT+ result.

