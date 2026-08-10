# FG-DS Seed-42 Hardware Replay Runbook

## 1. Experiment Contract

This run is the hardware confirmation for the WSN-DS feature-group-disjoint
(FG-DS), train-only-scaler, seed-42 RF-KD Student A and Student B models.

The FG-DS test partition contains 56,301 records. The complete experiment is:

```text
2 models x 2 boards x 56,301 test records = 225,204 full-test MCU predictions
```

The host replays already extracted WSN-DS 17-feature records over USB serial.
The device times integer preprocessing and fixed-point inference. This does not
measure live packet capture, packet-to-feature extraction, radio operation, or
energy consumption.

The archived 56,200-record random-row HIL results are historical evidence. Do
not relabel them as FG-DS results or combine them with this report.

## 2. Frozen Assets

Exports:

```text
deployment/firmware_export/wsnds_rfkd_hil/generated_fgds_student_A_seed42
deployment/firmware_export/wsnds_rfkd_hil/generated_fgds_student_B_seed42
```

Firmware bundles:

```text
deployment/hardware_hil/build/fgds_student_A_seed42_esp32c3
deployment/hardware_hil/build/fgds_student_B_seed42_esp32c3
deployment/hardware_hil/build/fgds_student_A_seed42_arduino_r4
deployment/hardware_hil/build/fgds_student_B_seed42_arduino_r4
```

Verified compile evidence:

| Target | FQBN | Core | Compiler | Flash | RAM |
|---|---|---:|---:|---:|---:|
| ESP32-C3, Student A | `esp32:esp32:esp32c3` | 3.3.11 | 14.2.0 | 281,792 / 1,310,720 B | 13,592 / 327,680 B |
| ESP32-C3, Student B | `esp32:esp32:esp32c3` | 3.3.11 | 14.2.0 | 284,148 / 1,310,720 B | 13,592 / 327,680 B |
| Arduino R4 WiFi, Student A | `arduino:renesas_uno:unor4wifi` | 1.6.0 | 7.2.1 | 56,416 / 262,144 B | 7,128 / 32,768 B |
| Arduino R4 WiFi, Student B | `arduino:renesas_uno:unor4wifi` | 1.6.0 | 7.2.1 | 58,768 / 262,144 B | 7,128 / 32,768 B |

Each preserved binary contains its expected export ID and bundle ID. The
strict serial handshake checks the same IDs after flashing, before accepting
any prediction.

## 3. Update And Verify The Raspberry Pi Checkout

The frozen FG-DS files are committed with exact-byte Git attributes because
their manifests contain SHA-256 hashes of generated text and binary artifacts.
On the Raspberry Pi:

```bash
cd ~/cukd-xai/CuKD-XAI

git fetch origin

git pull --ff-only origin main
```

Create or activate the Pi host environment:

```bash
if [ -f .venv-hil/bin/activate ]; then
  source .venv-hil/bin/activate
else
  python3 -m venv .venv-hil
  source .venv-hil/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -r deployment/hardware_hil/host/requirements.txt
fi
```

Run host-only preflight after the pull:

```bash
python deployment/hardware_hil/host/preflight_fgds_hil.py \
  --output-json results/hardware_hil/fgds_seed42/pi5_preflight.json

cat results/hardware_hil/fgds_seed42/pi5_preflight.json
```

Required preflight status:

```text
ready_for_compile_flash_and_hil
```

Preflight validates files, hashes, split identity, model identity, and bundle
identity. It is not physical execution evidence.

## 4. ESP32-C3 Student A

On the laptop, open and upload this exact sketch without editing the bundle:

```text
C:\N Drive\Research\Cukd-XAI\CuKD-XAI\deployment\hardware_hil\build\fgds_student_A_seed42_esp32c3\fgds_student_A_seed42_esp32c3.ino
```

Connect only the flashed ESP32-C3 to the Raspberry Pi. On the Pi:

```bash
cd ~/cukd-xai/CuKD-XAI
source .venv-hil/bin/activate

python deployment/hardware_hil/host/env_check.py \
  --output results/hardware_hil/fgds_seed42/pi5_environment_esp32c3_student_A.json

ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null

PORT=/dev/ttyUSB0
GEN=deployment/firmware_export/wsnds_rfkd_hil/generated_fgds_student_A_seed42
BUNDLE=deployment/hardware_hil/build/fgds_student_A_seed42_esp32c3
RESULT=results/hardware_hil/fgds_seed42/pi5_esp32c3_student_A
HOST_ENV=results/hardware_hil/fgds_seed42/pi5_environment_esp32c3_student_A.json

bash deployment/hardware_hil/scripts/run_fgds_hil.sh \
  "${PORT}" \
  "${GEN}" \
  "${BUNDLE}" \
  "${RESULT}" \
  "${HOST_ENV}"
```

Use the port printed by `env_check.py`; `/dev/ttyUSB0` is only the previously
observed ESP32-C3 port.

## 5. ESP32-C3 Student B

Upload this exact sketch:

```text
C:\N Drive\Research\Cukd-XAI\CuKD-XAI\deployment\hardware_hil\build\fgds_student_B_seed42_esp32c3\fgds_student_B_seed42_esp32c3.ino
```

Then run on the Pi:

```bash
cd ~/cukd-xai/CuKD-XAI
source .venv-hil/bin/activate

python deployment/hardware_hil/host/env_check.py \
  --output results/hardware_hil/fgds_seed42/pi5_environment_esp32c3_student_B.json

PORT=/dev/ttyUSB0
GEN=deployment/firmware_export/wsnds_rfkd_hil/generated_fgds_student_B_seed42
BUNDLE=deployment/hardware_hil/build/fgds_student_B_seed42_esp32c3
RESULT=results/hardware_hil/fgds_seed42/pi5_esp32c3_student_B
HOST_ENV=results/hardware_hil/fgds_seed42/pi5_environment_esp32c3_student_B.json

bash deployment/hardware_hil/scripts/run_fgds_hil.sh \
  "${PORT}" \
  "${GEN}" \
  "${BUNDLE}" \
  "${RESULT}" \
  "${HOST_ENV}"
```

## 6. Arduino R4 WiFi Student A

Upload this exact sketch:

```text
C:\N Drive\Research\Cukd-XAI\CuKD-XAI\deployment\hardware_hil\build\fgds_student_A_seed42_arduino_r4\fgds_student_A_seed42_arduino_r4.ino
```

Connect only the flashed Arduino R4 to the Pi, then run:

```bash
cd ~/cukd-xai/CuKD-XAI
source .venv-hil/bin/activate

python deployment/hardware_hil/host/env_check.py \
  --output results/hardware_hil/fgds_seed42/pi5_environment_arduino_r4_student_A.json

ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null

PORT=/dev/ttyACM0
GEN=deployment/firmware_export/wsnds_rfkd_hil/generated_fgds_student_A_seed42
BUNDLE=deployment/hardware_hil/build/fgds_student_A_seed42_arduino_r4
RESULT=results/hardware_hil/fgds_seed42/pi5_arduino_r4_student_A
HOST_ENV=results/hardware_hil/fgds_seed42/pi5_environment_arduino_r4_student_A.json

bash deployment/hardware_hil/scripts/run_fgds_hil.sh \
  "${PORT}" \
  "${GEN}" \
  "${BUNDLE}" \
  "${RESULT}" \
  "${HOST_ENV}"
```

Use the port printed by `env_check.py`; `/dev/ttyACM0` is only the previously
observed Arduino R4 port.

## 7. Arduino R4 WiFi Student B

Upload this exact sketch:

```text
C:\N Drive\Research\Cukd-XAI\CuKD-XAI\deployment\hardware_hil\build\fgds_student_B_seed42_arduino_r4\fgds_student_B_seed42_arduino_r4.ino
```

Then run on the Pi:

```bash
cd ~/cukd-xai/CuKD-XAI
source .venv-hil/bin/activate

python deployment/hardware_hil/host/env_check.py \
  --output results/hardware_hil/fgds_seed42/pi5_environment_arduino_r4_student_B.json

PORT=/dev/ttyACM0
GEN=deployment/firmware_export/wsnds_rfkd_hil/generated_fgds_student_B_seed42
BUNDLE=deployment/hardware_hil/build/fgds_student_B_seed42_arduino_r4
RESULT=results/hardware_hil/fgds_seed42/pi5_arduino_r4_student_B
HOST_ENV=results/hardware_hil/fgds_seed42/pi5_environment_arduino_r4_student_B.json

bash deployment/hardware_hil/scripts/run_fgds_hil.sh \
  "${PORT}" \
  "${GEN}" \
  "${BUNDLE}" \
  "${RESULT}" \
  "${HOST_ENV}"
```

## 8. Acceptance Checks

Each invocation runs 10, 1,000, and 56,301 records in order. It refuses to
overwrite a completed or in-progress result directory. A failed partial run is
preserved under a timestamped `.failed` directory.

For each pair, inspect:

```bash
cat "${RESULT}/full_56301_sequence.json"

cat "${RESULT}/full_56301_metrics.json"
```

Required full-run conditions:

- `expected = 56301`
- `completed = 56301`
- no missing, duplicate, unexpected, or non-OK rows
- `mcu_vs_fixed_reference_agreement = 1.0`
- `exact_logit_agreement = 1.0`

Do not continue to the final report if any condition fails.

## 9. Final Four-Pair Report

After all four full runs pass:

```bash
cd ~/cukd-xai/CuKD-XAI
source .venv-hil/bin/activate

REPORT="results/hardware_hil/fgds_seed42/final_report_$(date -u +%Y%m%dT%H%M%SZ)"

python deployment/hardware_hil/host/generate_fgds_report.py \
  --run esp32c3_student_A=results/hardware_hil/fgds_seed42/pi5_esp32c3_student_A/full_56301_metrics.json \
  --run esp32c3_student_B=results/hardware_hil/fgds_seed42/pi5_esp32c3_student_B/full_56301_metrics.json \
  --run arduino_r4_student_A=results/hardware_hil/fgds_seed42/pi5_arduino_r4_student_A/full_56301_metrics.json \
  --run arduino_r4_student_B=results/hardware_hil/fgds_seed42/pi5_arduino_r4_student_B/full_56301_metrics.json \
  --compile esp32c3_student_A=results/hardware_hil/fgds_seed42/compile_evidence/esp32c3_student_A.json \
  --compile esp32c3_student_B=results/hardware_hil/fgds_seed42/compile_evidence/esp32c3_student_B.json \
  --compile arduino_r4_student_A=results/hardware_hil/fgds_seed42/compile_evidence/arduino_r4_student_A.json \
  --compile arduino_r4_student_B=results/hardware_hil/fgds_seed42/compile_evidence/arduino_r4_student_B.json \
  --output-json "${REPORT}/fgds_seed42_hardware_summary.json" \
  --output-csv "${REPORT}/fgds_seed42_hardware_table.csv" \
  --output-md "${REPORT}/fgds_seed42_hardware_summary.md"

cat "${REPORT}/fgds_seed42_hardware_summary.md"

cat "${REPORT}/final_report_manifest.json"
```

The report generator rejects mixed export IDs, mixed bundle IDs, duplicate
evidence files, incomplete stage inventories, changed source evidence,
non-exact fixed-point or logit agreement, and any full-test count other than
56,301.
