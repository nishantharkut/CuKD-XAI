# Student B HIL Runbook

This runbook adds the WSN-DS Student B RF-KD hardware replay path without overwriting the completed Student A evidence.

## Scope

- Model: `E_student_B_KD_from_RF`
- Architecture: `17 -> 64 -> 32 -> 5`
- Dataset contract: WSN-DS 17-feature replay records
- Purpose: capacity/accuracy trade-off evidence beside Student A
- Boundary: USB serial replay only; not live WSN packet capture, not energy measurement, not Raspberry Pi AI HAT+ inference, and not physical TelosB deployment.

Do not overwrite Student A folders or Student A results. Student B must use separate generated, build, and result folders.

## 1. Generate Student B Export Artifacts On The Windows Laptop

Run this inside the existing `cukd_env`, from the `repository root` folder:

```powershell
cd "C:\N Drive\Acads\6th SEM\WCT\CuKD-XAI"

python .\deployment\firmware_export\wsnds_rfkd_hil\run_wsnds_student_a_rfkd_e2e.py `
  --state-dict results\runtime\onnx_openvino\wsnds\tmp\E_student_B_KD_from_RF_fp32.pt `
  --dataset-csv data\wsnds\WSN-DS.csv `
  --output-dir deployment\firmware_export\wsnds_rfkd_hil\generated_student_b_rfkd_hil_full `
  --num-test-vectors 56200 `
  --model-label "WSN-DS Student B E_KD_from_RF" `
  --self-test-name cukd_student_b_rfkd_self_test
```

Expected output folder:

```text
deployment/firmware_export/wsnds_rfkd_hil/generated_student_b_rfkd_hil_full
```

Minimum files that must exist before continuing:

- `model_weights.h`
- `preprocess_int_metadata.h`
- `hil_replay_vectors.csv`
- `hil_reference_predictions.csv`
- `equivalence_report.json`
- `export_summary.json`
- `e2e_run_report.json`

Check `export_summary.json` before flashing. It should report:

- `model`: `WSN-DS Student B E_KD_from_RF`
- `dims`: `[17, 64, 32, 5]`
- `macs_per_inference`: `3296`

## 2. Create Student B Firmware Bundles

Still from the repository root:

```powershell
python -m hardware_hil.host.prepare_firmware_bundle `
  --board esp32c3 `
  --generated-dir deployment/firmware_export/wsnds_rfkd_hil/generated_student_b_rfkd_hil_full `
  --output-dir deployment/hardware_hil/build/cukd_hil_esp32c3_student_b

python -m hardware_hil.host.prepare_firmware_bundle `
  --board arduino_r4 `
  --generated-dir deployment/firmware_export/wsnds_rfkd_hil/generated_student_b_rfkd_hil_full `
  --output-dir deployment/hardware_hil/build/cukd_hil_arduino_r4_student_b
```

Open and upload these sketches in Arduino IDE:

```text
deployment/hardware_hil/build/cukd_hil_esp32c3_student_b/cukd_hil_esp32c3_student_b.ino
deployment/hardware_hil/build/cukd_hil_arduino_r4_student_b/cukd_hil_arduino_r4_student_b.ino
```

Preserve the Arduino IDE compile memory output for each board. Report it separately from MSP430 static cross-compile evidence.

## 2.5. Make Student B Artifacts Available On The Raspberry Pi

Before running replay from the Pi, make sure this folder exists on the Pi at the same relative path:

```text
deployment/firmware_export/wsnds_rfkd_hil/generated_student_b_rfkd_hil_full
```

Use one of these two routes:

- Preferred route: commit/push the generated Student B artifacts and firmware bundle folders from the laptop, then pull them on the Pi.
- Manual route: copy `deployment/firmware_export/wsnds_rfkd_hil/generated_student_b_rfkd_hil_full` into the Pi checkout under `deployment/firmware_export/wsnds_rfkd_hil/`.

Do not reuse `generated_student_a_rfkd_hil_full`; the replay vectors and fixed-point reference predictions must come from the Student B export.

## 3. Run ESP32-C3 Student B Replay On Raspberry Pi 5

Use the ESP32-C3 DevKitM-1 as the primary MCU target. On the Raspberry Pi, from the repository root:

```bash
source .venv-hil/bin/activate

PORT=/dev/ttyUSB0
GEN=deployment/firmware_export/wsnds_rfkd_hil/generated_student_b_rfkd_hil_full
BOARD=pi5_esp32c3_student_b
mkdir -p results/hardware_hil/board_replay/${BOARD}
```

Change `PORT` only if `python -m hardware_hil.host.env_check --output results/hardware_hil/board_replay/pi5_environment_esp32c3_student_b_connected.json` shows a different serial device.

Smoke run:

```bash
python -m hardware_hil.host.stream_vectors \
  --port ${PORT} \
  --vectors-csv ${GEN}/hil_replay_vectors.csv \
  --output-csv results/hardware_hil/board_replay/${BOARD}/smoke_mcu.csv \
  --summary-json results/hardware_hil/board_replay/${BOARD}/smoke_sequence.json \
  --limit 10 \
  --timeout 2.0

python -m hardware_hil.host.verify_results \
  --mcu-csv results/hardware_hil/board_replay/${BOARD}/smoke_mcu.csv \
  --reference-csv ${GEN}/hil_reference_predictions.csv \
  --output-json results/hardware_hil/board_replay/${BOARD}/smoke_metrics.json
```

Validation run:

```bash
python -m hardware_hil.host.stream_vectors \
  --port ${PORT} \
  --vectors-csv ${GEN}/hil_replay_vectors.csv \
  --output-csv results/hardware_hil/board_replay/${BOARD}/validation_1000_mcu.csv \
  --summary-json results/hardware_hil/board_replay/${BOARD}/validation_1000_sequence.json \
  --limit 1000 \
  --timeout 2.0

python -m hardware_hil.host.verify_results \
  --mcu-csv results/hardware_hil/board_replay/${BOARD}/validation_1000_mcu.csv \
  --reference-csv ${GEN}/hil_reference_predictions.csv \
  --output-json results/hardware_hil/board_replay/${BOARD}/validation_1000_metrics.json
```

Full run:

```bash
python -m hardware_hil.host.stream_vectors \
  --port ${PORT} \
  --vectors-csv ${GEN}/hil_replay_vectors.csv \
  --output-csv results/hardware_hil/board_replay/${BOARD}/full_56200_mcu.csv \
  --summary-json results/hardware_hil/board_replay/${BOARD}/full_56200_sequence.json \
  --timeout 2.0

python -m hardware_hil.host.verify_results \
  --mcu-csv results/hardware_hil/board_replay/${BOARD}/full_56200_mcu.csv \
  --reference-csv ${GEN}/hil_reference_predictions.csv \
  --output-json results/hardware_hil/board_replay/${BOARD}/full_56200_metrics.json
```

## 4. Run Arduino R4 Student B Replay On Raspberry Pi 5

After flashing `cukd_hil_arduino_r4_student_b`, connect only the Arduino R4 by USB to the Pi.

```bash
source .venv-hil/bin/activate

PORT=/dev/ttyACM0
GEN=deployment/firmware_export/wsnds_rfkd_hil/generated_student_b_rfkd_hil_full
BOARD=pi5_arduino_r4_student_b
mkdir -p results/hardware_hil/board_replay/${BOARD}
```

Use the same smoke, validation, full, and verify commands from the ESP32-C3 section. Only `PORT` and `BOARD` change.

## 5. Generate Combined Report After Both Student B Runs

After Student A and Student B full metrics exist:

```bash
python -m hardware_hil.host.generate_report \
  --metric pi5_esp32c3_student_a=results/hardware_hil/board_replay/pi5_esp32c3/full_56200_metrics.json \
  --metric pi5_arduino_r4_student_a=results/hardware_hil/board_replay/pi5_arduino_r4/full_56200_metrics.json \
  --metric pi5_esp32c3_student_b=results/hardware_hil/board_replay/pi5_esp32c3_student_b/full_56200_metrics.json \
  --metric pi5_arduino_r4_student_b=results/hardware_hil/board_replay/pi5_arduino_r4_student_b/full_56200_metrics.json \
  --output-md results/hardware_hil/reports/pi5_hardware_hil_student_a_b_summary.md \
  --output-csv results/hardware_hil/reports/pi5_hardware_hil_student_a_b_tables.csv
```

## Pass Criteria

For each Student B board, preserve:

- `full_56200_sequence.json` with `completed = 56200`, no missing rows, and no non-OK status rows.
- `full_56200_metrics.json` with `mcu_vs_fixed_reference_agreement = 1.0` or an explained discrepancy.
- Latency percentiles reported per board.
- Arduino IDE memory output for the exact Student B sketch upload.

Student B is an additional capacity/accuracy trade-off result. Student A remains the ultra-small primary MCU compression evidence.


