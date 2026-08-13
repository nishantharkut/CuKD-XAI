# FG-DS Wi-Fi UDP HIL Runbook

These commands run the four required board/model pairs on the Raspberry Pi. The
verified repository location is `/home/project/Desktop/CuKD-XAI`. Both boards
may remain connected by USB, but compile, flash, provision, and replay one pair
at a time.

## 1. Network and repository checks

Use a 2.4 GHz WPA/WPA2 network shared by the Raspberry Pi and both boards.
Disable wireless client isolation. Internet access is not required.

```bash
cd /home/project/Desktop/CuKD-XAI
git status --short
git pull --ff-only
source .venv-hil/bin/activate

ARDUINO_CLI="$HOME/.local/bin/arduino-cli"
"$ARDUINO_CLI" version
"$ARDUINO_CLI" core list
"$ARDUINO_CLI" board list
ls -l /dev/ttyUSB* /dev/ttyACM*
```

Expected tool versions for the prepared bundles:

- Arduino CLI 1.5.1.
- ESP32 core 3.3.11.
- Arduino Renesas UNO core 1.6.0.

At the time of preparation, ESP32-C3 was `/dev/ttyUSB0` and UNO R4 WiFi was
`/dev/ttyACM0`. Confirm with `board list`; do not assume names after reconnecting.

## 2. Select one board/model pair

Run exactly one of the following variable blocks.

ESP32-C3, Student A:

```bash
LABEL=esp32c3_student_A
GEN=deployment/firmware_export/wsnds_rfkd_hil/generated_fgds_student_A_seed42
BUNDLE=deployment/wireless_hil/build/fgds_student_A_seed42_esp32c3_wifi_udp
FQBN=esp32:esp32:esp32c3
PORT=/dev/ttyUSB0
CORE_ID=esp32:esp32
EXPECTED_CORE_VERSION=3.3.11
COMPILER="$HOME/.arduino15/packages/esp32/tools/esp-rv32/2601/bin/riscv32-esp-elf-g++"
EXPECTED_TOOLCHAIN_VERSION=14.2.0
```

ESP32-C3, Student B:

```bash
LABEL=esp32c3_student_B
GEN=deployment/firmware_export/wsnds_rfkd_hil/generated_fgds_student_B_seed42
BUNDLE=deployment/wireless_hil/build/fgds_student_B_seed42_esp32c3_wifi_udp
FQBN=esp32:esp32:esp32c3
PORT=/dev/ttyUSB0
CORE_ID=esp32:esp32
EXPECTED_CORE_VERSION=3.3.11
COMPILER="$HOME/.arduino15/packages/esp32/tools/esp-rv32/2601/bin/riscv32-esp-elf-g++"
EXPECTED_TOOLCHAIN_VERSION=14.2.0
```

UNO R4 WiFi, Student A:

```bash
LABEL=arduino_r4_student_A
GEN=deployment/firmware_export/wsnds_rfkd_hil/generated_fgds_student_A_seed42
BUNDLE=deployment/wireless_hil/build/fgds_student_A_seed42_arduino_r4_wifi_udp
FQBN=arduino:renesas_uno:unor4wifi
PORT=/dev/ttyACM0
CORE_ID=arduino:renesas_uno
EXPECTED_CORE_VERSION=1.6.0
COMPILER="$HOME/.arduino15/packages/arduino/tools/arm-none-eabi-gcc/7-2017q4/bin/arm-none-eabi-g++"
EXPECTED_TOOLCHAIN_VERSION=7.2.1
```

UNO R4 WiFi, Student B:

```bash
LABEL=arduino_r4_student_B
GEN=deployment/firmware_export/wsnds_rfkd_hil/generated_fgds_student_B_seed42
BUNDLE=deployment/wireless_hil/build/fgds_student_B_seed42_arduino_r4_wifi_udp
FQBN=arduino:renesas_uno:unor4wifi
PORT=/dev/ttyACM0
CORE_ID=arduino:renesas_uno
EXPECTED_CORE_VERSION=1.6.0
COMPILER="$HOME/.arduino15/packages/arduino/tools/arm-none-eabi-gcc/7-2017q4/bin/arm-none-eabi-g++"
EXPECTED_TOOLCHAIN_VERSION=7.2.1
```

## 3. Verify, compile, and preserve compile evidence

The build cache must be outside the immutable bundle. Do not use `--output-dir`.
With ESP32 core 3.3.11, that option exports a `build/` tree into the sketch even
when `--build-path` is external, invalidating the bundle manifest.

```bash
export GEN BUNDLE

python - <<'PY'
import os
from pathlib import Path
from deployment.wireless_hil.host.wireless_common import (
    verify_export_for_wireless,
    verify_wireless_bundle,
)

generated = Path(os.environ["GEN"])
bundle = Path(os.environ["BUNDLE"])
export = verify_export_for_wireless(generated)
verified = verify_wireless_bundle(bundle, export)
print(verified["board"], verified["student"], verified["wireless_bundle_id"])
PY

test -x "$COMPILER"
CORE_VERSION="$("$ARDUINO_CLI" core list | awk -v id="$CORE_ID" '$1 == id {print $2}')"
FRONTEND_VERSION="$("$ARDUINO_CLI" version | sed -n 's/^arduino-cli[[:space:]]*Version:[[:space:]]*\([^[:space:]]*\).*/arduino-cli \1/p')"
TOOLCHAIN_VERSION="$("$COMPILER" -dumpfullversion -dumpversion | tr -d '\r\n')"
test "$CORE_VERSION" = "$EXPECTED_CORE_VERSION"
test -n "$FRONTEND_VERSION"
test "$TOOLCHAIN_VERSION" = "$EXPECTED_TOOLCHAIN_VERSION"

RAW="results/wireless_hil/fgds_seed42/compile_raw/${LABEL}"
EVIDENCE="results/wireless_hil/fgds_seed42/compile_evidence/${LABEL}.json"
test ! -e "$RAW"
test ! -e "$EVIDENCE"
mkdir -p "$RAW/build"

set -o pipefail
{
  printf 'CUKD_FQBN=%s\n' "$FQBN"
  printf 'CUKD_BOARD_CORE_VERSION=%s\n' "$CORE_VERSION"
  printf 'CUKD_FRONTEND_VERSION=%s\n' "$FRONTEND_VERSION"
  printf 'CUKD_TOOLCHAIN_VERSION=%s\n' "$TOOLCHAIN_VERSION"
  "$COMPILER" --version
  "$ARDUINO_CLI" --no-color compile --verbose --warnings all \
    --fqbn "$FQBN" \
    --build-path "$RAW/build" \
    "$BUNDLE"
} 2>&1 | tee "$RAW/compile.log"

BINARY="$RAW/build/$(basename "$BUNDLE").ino.bin"
test -s "$BINARY"

python -m deployment.wireless_hil.host.record_wireless_compile_evidence \
  --generated-dir "$GEN" \
  --bundle-dir "$BUNDLE" \
  --compile-log "$RAW/compile.log" \
  --binary "$BINARY" \
  --fqbn "$FQBN" \
  --board-core-version "$CORE_VERSION" \
  --frontend-version "$FRONTEND_VERSION" \
  --toolchain-version "$TOOLCHAIN_VERSION" \
  --output-json "$EVIDENCE"
```

Do not rerun a failed compile into the same `RAW` directory. Preserve it and use
a new explicitly named directory for the diagnostic attempt.

## 4. Upload the compiled firmware

Close every serial monitor first.

```bash
"$ARDUINO_CLI" upload \
  --port "$PORT" \
  --fqbn "$FQBN" \
  --build-path "$RAW/build" \
  "$BUNDLE"
```

The uploaded binary is the same file whose identity and footprint were sealed
in compile evidence.

## 5. Provision Wi-Fi interactively

```bash
mkdir -p results/wireless_hil/fgds_seed42/connections
CONNECTION="results/wireless_hil/fgds_seed42/connections/${LABEL}.json"
test ! -e "$CONNECTION"

python -m deployment.wireless_hil.host.configure_wifi_serial \
  --port "$PORT" \
  --generated-dir "$GEN" \
  --bundle-dir "$BUNDLE" \
  --output-json "$CONNECTION"
```

Enter the SSID and password only at the prompts. They are not echoed into the
command or connection record. Provisioning closes serial before returning.

## 6. Run smoke, validation, and full replay

```bash
FINAL="results/wireless_hil/fgds_seed42/pi5_${LABEL}"
test ! -e "$FINAL"
test ! -e "${FINAL}.in_progress"

bash deployment/wireless_hil/scripts/run_fgds_wireless_hil.sh \
  "$GEN" \
  "$BUNDLE" \
  "$CONNECTION" \
  "$FINAL"
```

The runner performs preflight, `smoke_10`, `validation_1000`, `full_56301`,
independent verification after every stage, and final inventory sealing. A
successful directory contains `wireless_hil_completion_manifest.json` with
status `complete`.

If the run fails, the script preserves a timestamped `.failed.*` directory. Do
not reuse the old connection session. Run provisioning again with a new
connection filename. If the canonical final path was never created, it may be
reused; never overwrite a completed or in-progress path. Reprovisioning is
required because stage ordinals are monotonic within one session.

## 7. Repeat all four pairs

For Student B on the same board, select the Student B variable block, compile,
flash, provision, and replay again. Then repeat Student A and B on the other
board. Do not copy a connection record or compile evidence between pairs; both
are bound to the exact board, student export, and bundle ID.

## 8. Generate the four-pair report

Choose a new output directory name. The report generator refuses overwrite.

```bash
REPORT=results/wireless_hil/fgds_seed42/final_report_$(date -u +%Y%m%dT%H%M%SZ)

python -m deployment.wireless_hil.host.generate_wireless_report \
  --run esp32c3_student_A=results/wireless_hil/fgds_seed42/pi5_esp32c3_student_A/full_56301_metrics.json \
  --run esp32c3_student_B=results/wireless_hil/fgds_seed42/pi5_esp32c3_student_B/full_56301_metrics.json \
  --run arduino_r4_student_A=results/wireless_hil/fgds_seed42/pi5_arduino_r4_student_A/full_56301_metrics.json \
  --run arduino_r4_student_B=results/wireless_hil/fgds_seed42/pi5_arduino_r4_student_B/full_56301_metrics.json \
  --compile esp32c3_student_A=results/wireless_hil/fgds_seed42/compile_evidence/esp32c3_student_A.json \
  --compile esp32c3_student_B=results/wireless_hil/fgds_seed42/compile_evidence/esp32c3_student_B.json \
  --compile arduino_r4_student_A=results/wireless_hil/fgds_seed42/compile_evidence/arduino_r4_student_A_dhcp_v2.json \
  --compile arduino_r4_student_B=results/wireless_hil/fgds_seed42/compile_evidence/arduino_r4_student_B_dhcp_v2.json \
  --output-dir "$REPORT"

cat "$REPORT/wireless_hil_final_report.md"
```

The report is accepted only if every full stage contains 56,301 rows, predictions
and logits match the fixed reference exactly, all four compile records reconcile
with their preserved logs and binaries, and Student A/B classification outputs
are identical across the two boards.
