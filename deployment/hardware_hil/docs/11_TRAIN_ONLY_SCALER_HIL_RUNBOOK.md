# Train-only-scaler WSN-DS HIL runbook

> **Historical scaler-only correction:** This lineage corrected scaler fitting
> but retained a random-row split. It is superseded for current claims by the
> ten-seed feature-group-disjoint lineage and final USB campaign.

This is the train-only-scaler WSN-DS deployment lineage. It creates new artifacts and
does not modify or relabel the archived global-scaler models or HIL evidence.

## Experimental boundary

- Deployment seed: `42`.
- Models: RF-KD Student A `(32,16)` and Student B `(64,32)`.
- Fixed preliminary KD setting: `T=4`, `alpha=0.7`, selected once by the
  active-v1 MLP-teacher validation grid. Candidate RNG streams were not held
  constant, so this is not claimed as an RF-KD optimum.
- Boards: ESP32-C3 and Arduino UNO R4 WiFi.
- Four physical board/model pairs, each replaying all `56,200` test records.
- The scaler and fixed-point calibration are fit on training data only.

The deployment students are fresh deterministic seed-42 retrainings under the
corrected protocol using the hash-bound preserved seed-42 RF soft targets. They
are not claimed to be the exact active-v1 checkpoint or an old hardware model.
Replay starts from already extracted WSN-DS features. It does not establish
live packet capture, on-board packet-to-feature extraction, energy, radio, or
TelosB behavior.
The archive-compatible random-row split contains exact feature groups that
cross partitions. This route is retained for direct comparison and deployment
fidelity; it is not evidence of duplicate-free generalization.

## 1. Confirm that the active statistical run is finished

On Windows, from the repository root:

```powershell
$ActiveManifest = Get-Content `
  "results\wsnds\leakage_free_rerun\main_10seed\executed_source_snapshot\execution_manifest.json" `
  | ConvertFrom-Json
Get-Process -Id $ActiveManifest.worker_pid -ErrorAction SilentlyContinue
```

Do not start confirmation training while this prints a process. After it exits,
confirm that all 20 checkpoint files exist. Process exit alone is not evidence
of successful completion.

```powershell
(Get-ChildItem `
  "results\wsnds\leakage_free_rerun\main_10seed\checkpoint_student_*_seed_*.json").Count
```

The expected count is `20`.

## 2. Train only the two deployment students

```powershell
& "experiments\wsnds\leakage_free_rerun\.venv\Scripts\python.exe" `
  "experiments\wsnds\leakage_free_rerun\run_tier15_confirmation.py" `
  --mode deployment `
  --device cuda `
  --confirm-training
```

The command verifies the exact dataset, archived split, train-only scaler,
executed-source evidence, execution contract, and preserved seed-42 RF cache.
It refuses to overwrite a nonempty destination.

## 3. Export both fixed-point models

Do not use `--skip-host-compile` for publication evidence. The host test builds
and executes the same preprocessing and inference C kernels used by the board
bundles.

```powershell
$PY = "experiments\wsnds\leakage_free_rerun\.venv\Scripts\python.exe"

& $PY "deployment\firmware_export\wsnds_rfkd_hil\export_train_only_deployment.py" `
  --student A `
  --output-dir "deployment\firmware_export\wsnds_rfkd_hil\generated_train_only_student_A_seed42"

& $PY "deployment\firmware_export\wsnds_rfkd_hil\export_train_only_deployment.py" `
  --student B `
  --output-dir "deployment\firmware_export\wsnds_rfkd_hil\generated_train_only_student_B_seed42"
```

A passed export requires full-test fixed/FP32 agreement of at least `0.99`,
macro-F1 drop no greater than `0.01`, zero audited saturation at every numeric
clipping point, and exact host C equivalence. The quality thresholds are fixed
in code and cannot be weakened from the command line.

## 4. Create the four firmware bundles

```powershell
& $PY "deployment\hardware_hil\host\prepare_strict_firmware_bundle.py" `
  --board esp32c3 `
  --generated-dir "deployment\firmware_export\wsnds_rfkd_hil\generated_train_only_student_A_seed42" `
  --output-dir "deployment\hardware_hil\build\train_only_student_A_seed42_esp32c3"

& $PY "deployment\hardware_hil\host\prepare_strict_firmware_bundle.py" `
  --board arduino_r4 `
  --generated-dir "deployment\firmware_export\wsnds_rfkd_hil\generated_train_only_student_A_seed42" `
  --output-dir "deployment\hardware_hil\build\train_only_student_A_seed42_arduino_r4"

& $PY "deployment\hardware_hil\host\prepare_strict_firmware_bundle.py" `
  --board esp32c3 `
  --generated-dir "deployment\firmware_export\wsnds_rfkd_hil\generated_train_only_student_B_seed42" `
  --output-dir "deployment\hardware_hil\build\train_only_student_B_seed42_esp32c3"

& $PY "deployment\hardware_hil\host\prepare_strict_firmware_bundle.py" `
  --board arduino_r4 `
  --generated-dir "deployment\firmware_export\wsnds_rfkd_hil\generated_train_only_student_B_seed42" `
  --output-dir "deployment\hardware_hil\build\train_only_student_B_seed42_arduino_r4"
```

Each sketch responds to `CUKDID?` with its student, export ID, and bundle ID.
The Pi replay refuses a device carrying any other firmware identity.

## 5. Compile all four bundles and record evidence

For each bundle in Arduino IDE:

1. Open the `.ino` whose filename equals its containing directory.
2. Select the exact physical board/FQBN.
3. Enable verbose compilation output.
4. Compile and use **Sketch > Export Compiled Binary**.
5. Save the complete compile output unchanged.
6. Keep the binary whose filename includes the strict sketch filename.

Keep both the compile log and exported binary outside the firmware bundle and
strict export directories. Those directories are immutable inputs whose exact
inventories are checked by later tools.

Record each compile with `record_compile_evidence.py`. Supply the actual FQBN,
board-core version, Arduino IDE/CLI version, toolchain version, log path, and
binary path. The recorder rejects placeholders, wrong board/student identity,
inconsistent footprint arithmetic, a log naming another sketch, and a binary
without both strict IDs. The exact FQBN, board-core version, and toolchain
version must appear in the verbose log. The frontend version is retained as an
operator-recorded field because Arduino does not embed it as an authenticated
binary claim.

Use this PowerShell helper from the repository root. It prompts for observed
values so the evidence cannot silently inherit example version strings:

```powershell
$PY = "experiments\wsnds\leakage_free_rerun\.venv\Scripts\python.exe"

function Record-StrictCompile {
  param(
    [Parameter(Mandatory=$true)][ValidateSet("A", "B")][string]$Student,
    [Parameter(Mandatory=$true)][string]$Generated,
    [Parameter(Mandatory=$true)][string]$Bundle,
    [Parameter(Mandatory=$true)][string]$Fqbn,
    [Parameter(Mandatory=$true)][string]$Output
  )

  $Log = Read-Host "Full unchanged verbose compile-log path for $Student / $Fqbn"
  $Binary = Read-Host "Exported strict firmware-binary path for $Student / $Fqbn"
  $Core = Read-Host "Exact board-core version shown in this verbose log"
  $Frontend = Read-Host "Exact Arduino IDE or Arduino CLI version used"
  $Toolchain = Read-Host "Exact compiler toolchain/version shown by the compile environment"

  & $PY "deployment\hardware_hil\host\record_compile_evidence.py" `
    --student $Student `
    --generated-dir $Generated `
    --bundle-dir $Bundle `
    --compile-log $Log `
    --binary $Binary `
    --fqbn $Fqbn `
    --board-core-version $Core `
    --frontend-version $Frontend `
    --toolchain-version $Toolchain `
    --output-json $Output
}

Record-StrictCompile `
  -Student A `
  -Generated "deployment\firmware_export\wsnds_rfkd_hil\generated_train_only_student_A_seed42" `
  -Bundle "deployment\hardware_hil\build\train_only_student_A_seed42_esp32c3" `
  -Fqbn "esp32:esp32:esp32c3" `
  -Output "results\hardware_hil\train_only_scaler\compile_evidence\esp32c3_student_A.json"

Record-StrictCompile `
  -Student B `
  -Generated "deployment\firmware_export\wsnds_rfkd_hil\generated_train_only_student_B_seed42" `
  -Bundle "deployment\hardware_hil\build\train_only_student_B_seed42_esp32c3" `
  -Fqbn "esp32:esp32:esp32c3" `
  -Output "results\hardware_hil\train_only_scaler\compile_evidence\esp32c3_student_B.json"

Record-StrictCompile `
  -Student A `
  -Generated "deployment\firmware_export\wsnds_rfkd_hil\generated_train_only_student_A_seed42" `
  -Bundle "deployment\hardware_hil\build\train_only_student_A_seed42_arduino_r4" `
  -Fqbn "arduino:renesas_uno:unor4wifi" `
  -Output "results\hardware_hil\train_only_scaler\compile_evidence\arduino_r4_student_A.json"

Record-StrictCompile `
  -Student B `
  -Generated "deployment\firmware_export\wsnds_rfkd_hil\generated_train_only_student_B_seed42" `
  -Bundle "deployment\hardware_hil\build\train_only_student_B_seed42_arduino_r4" `
  -Fqbn "arduino:renesas_uno:unor4wifi" `
  -Output "results\hardware_hil\train_only_scaler\compile_evidence\arduino_r4_student_B.json"
```

Use these four output records:

```text
results/hardware_hil/train_only_scaler/compile_evidence/esp32c3_student_A.json
results/hardware_hil/train_only_scaler/compile_evidence/arduino_r4_student_A.json
results/hardware_hil/train_only_scaler/compile_evidence/esp32c3_student_B.json
results/hardware_hil/train_only_scaler/compile_evidence/arduino_r4_student_B.json
```

Each JSON has a sibling `_artifacts` directory containing the log, binary,
strict export report, full fixed-point reference CSV, and manifests. Keep those
directories with the JSONs.
The identity/sketch checks detect accidental artifact mixing; they are not a
cryptographic attestation from Arduino that a particular log produced a
particular binary. This boundary is retained in the final report.

## 6. Prepare the Raspberry Pi once

Transfer the repository additions through the private transfer workflow,
including both generated exports, all four bundles, all four compile-evidence
JSON files, and their four `_artifacts` directories. Preserve repository-relative
paths. On the Pi:

```bash
cd "$HOME/cukd-xai/CuKD-XAI"

python3 -m venv .venv-hil

source .venv-hil/bin/activate

python -m pip install --upgrade pip

python -m pip install \
  -r deployment/hardware_hil/host/requirements.txt

chmod +x deployment/hardware_hil/scripts/run_strict_hil.sh
```

If `.venv-hil` already exists, do not recreate it; activate it and run the
requirements command.

## 7. Flash and replay each pair sequentially

There is one ESP32-C3 and one R4, so do not flash all four variants first. For
each pair below, upload that exact bundle on the laptop, unplug the board,
connect it to the Pi, verify its serial port, and immediately execute its replay.
Only then move to the next firmware.

Recommended order:

1. Student A on ESP32-C3.
2. Student B on ESP32-C3.
3. Student A on Arduino R4.
4. Student B on Arduino R4.

After connecting a board to the Pi, set `PAIR` to the exact pair about to run.
Use one of `pi5_esp32c3_student_A`, `pi5_esp32c3_student_B`,
`pi5_arduino_r4_student_A`, or `pi5_arduino_r4_student_B`:

```bash
PAIR=pi5_esp32c3_student_A

test ! -e results/hardware_hil/train_only_scaler/${PAIR}_environment_connected.json

python -m deployment.hardware_hil.host.env_check \
  --output results/hardware_hil/train_only_scaler/${PAIR}_environment_connected.json
```

Use a new filename for every pair; do not overwrite another board's environment
record. Confirm that the listed serial port appears only after that board is
connected.

Use the reported `/dev/ttyUSB*` or `/dev/ttyACM*` path. The four replay commands
are:

```bash
deployment/hardware_hil/scripts/run_strict_hil.sh \
  /dev/ttyUSB0 \
  deployment/firmware_export/wsnds_rfkd_hil/generated_train_only_student_A_seed42 \
  deployment/hardware_hil/build/train_only_student_A_seed42_esp32c3 \
  results/hardware_hil/train_only_scaler/pi5_esp32c3_student_A \
  results/hardware_hil/train_only_scaler/pi5_esp32c3_student_A_environment_connected.json

deployment/hardware_hil/scripts/run_strict_hil.sh \
  /dev/ttyUSB0 \
  deployment/firmware_export/wsnds_rfkd_hil/generated_train_only_student_B_seed42 \
  deployment/hardware_hil/build/train_only_student_B_seed42_esp32c3 \
  results/hardware_hil/train_only_scaler/pi5_esp32c3_student_B \
  results/hardware_hil/train_only_scaler/pi5_esp32c3_student_B_environment_connected.json

deployment/hardware_hil/scripts/run_strict_hil.sh \
  /dev/ttyACM0 \
  deployment/firmware_export/wsnds_rfkd_hil/generated_train_only_student_A_seed42 \
  deployment/hardware_hil/build/train_only_student_A_seed42_arduino_r4 \
  results/hardware_hil/train_only_scaler/pi5_arduino_r4_student_A \
  results/hardware_hil/train_only_scaler/pi5_arduino_r4_student_A_environment_connected.json

deployment/hardware_hil/scripts/run_strict_hil.sh \
  /dev/ttyACM0 \
  deployment/firmware_export/wsnds_rfkd_hil/generated_train_only_student_B_seed42 \
  deployment/hardware_hil/build/train_only_student_B_seed42_arduino_r4 \
  results/hardware_hil/train_only_scaler/pi5_arduino_r4_student_B \
  results/hardware_hil/train_only_scaler/pi5_arduino_r4_student_B_environment_connected.json
```

The script performs 10-row, 1,000-row, and full-test checks. Result directories
must not already exist. It writes to a sibling `.in_progress` directory, creates
a hash manifest only after all three stages pass, and then renames the directory
to its final name. A failed attempt is preserved with a timestamped `.failed`
suffix and is never accepted by the final report. Board timing is preprocessing
plus inference compute time from firmware `micros()`; it excludes request
parsing, response formatting, USB serial, and host overhead.
The stream stages have generous wall-clock limits, and interrupt/termination
signals preserve partial evidence under the same `.failed` convention.

## 8. Generate the final four-pair report

Run this on the machine that has both the four full HIL metrics files and the
four portable compile-evidence records:

```bash
python deployment/hardware_hil/host/generate_strict_report.py \
  --run esp32c3_student_A=results/hardware_hil/train_only_scaler/pi5_esp32c3_student_A/full_56200_metrics.json \
  --run arduino_r4_student_A=results/hardware_hil/train_only_scaler/pi5_arduino_r4_student_A/full_56200_metrics.json \
  --run esp32c3_student_B=results/hardware_hil/train_only_scaler/pi5_esp32c3_student_B/full_56200_metrics.json \
  --run arduino_r4_student_B=results/hardware_hil/train_only_scaler/pi5_arduino_r4_student_B/full_56200_metrics.json \
  --compile esp32c3_student_A=results/hardware_hil/train_only_scaler/compile_evidence/esp32c3_student_A.json \
  --compile arduino_r4_student_A=results/hardware_hil/train_only_scaler/compile_evidence/arduino_r4_student_A.json \
  --compile esp32c3_student_B=results/hardware_hil/train_only_scaler/compile_evidence/esp32c3_student_B.json \
  --compile arduino_r4_student_B=results/hardware_hil/train_only_scaler/compile_evidence/arduino_r4_student_B.json \
  --output-json results/hardware_hil/train_only_scaler/reports/final_hil_summary.json \
  --output-csv results/hardware_hil/train_only_scaler/reports/final_hil_table.csv \
  --output-md results/hardware_hil/train_only_scaler/reports/final_hil_summary.md
```

The generator accepts exactly four distinct board/student metrics records and
four distinct compile records. It rehashes portable evidence, checks the strict
export and bundle identities again, requires unique board bundles and raw MCU
CSVs, requires one common export
per student across both boards, and requires distinct Student A/B exports. It
requires the exact smoke, 1,000-row, and full-test stage inventory, then reopens
each full MCU CSV, sequence summary, and portable reference CSV;
recomputes exact logits, predictions, classification metrics, and timing
summaries; and publishes the three outputs plus a hash manifest by one directory
rename. The `reports` directory must not already exist.
