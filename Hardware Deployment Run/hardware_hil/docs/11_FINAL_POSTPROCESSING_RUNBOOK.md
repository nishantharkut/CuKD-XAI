# Final Post-Processing Runbook

This is the final analysis layer after the Student A and Student B HIL runs.
It does not run new models and does not change firmware behavior.

## What This Adds

- Framework-overhead baseline logs for ESP32-C3 and Arduino R4.
- Cycles-per-MAC analysis from existing latency metrics.
- Model-only fixed-point footprint table.
- Quantization drift analysis: rows where fixed-point predictions differ from FP32 predictions.
- Evidence traceability table mapping claims to result files.

## Claim Boundary

These outputs are post-processing evidence only. They do not prove live WSN
packet capture, packet-to-feature extraction, energy consumption, or physical
TelosB deployment.

## 1. Copy These Files Into The Repo

Copy this package into your repository root so the paths land under:

```text
Hardware Deployment Run/hardware_hil/host/analyze_final_hil_evidence.py
Hardware Deployment Run/hardware_hil/docs/11_FINAL_POSTPROCESSING_RUNBOOK.md
Hardware Deployment Run/hardware_hil/empty_baselines/esp32c3_serial_baseline/esp32c3_serial_baseline.ino
Hardware Deployment Run/hardware_hil/empty_baselines/arduino_r4_serial_baseline/arduino_r4_serial_baseline.ino
Hardware Deployment Run/hardware_hil/compile_logs/README.md
Hardware Deployment Run/tests/test_final_hil_postprocessing.py
```

Do not delete existing HIL results.

## 2. Compile The ESP32-C3 Serial Baseline

Use the same Arduino IDE board package and board selection used for the ESP32-C3
Student A/B uploads.

Open:

```text
Hardware Deployment Run/hardware_hil/empty_baselines/esp32c3_serial_baseline/esp32c3_serial_baseline.ino
```

Click **Verify**. Upload is not required.

Copy the full compile summary lines into:

```text
Hardware Deployment Run/hardware_hil/compile_logs/esp32c3_serial_baseline_compile.txt
```

The file should contain lines like:

```text
Sketch uses XXXXX bytes (YY%) of program storage space. Maximum is 1310720 bytes.
Global variables use XXXXX bytes (YY%) of dynamic memory, leaving XXXXX bytes for local variables. Maximum is 327680 bytes.
Board: ESP32-C3 DevKitM-1
Source: Arduino IDE Verify output for serial-only baseline.
```

## 3. Compile The Arduino R4 Serial Baseline

Use the same Arduino IDE board package and board selection used for the Arduino
R4 Student A/B uploads.

Open:

```text
Hardware Deployment Run/hardware_hil/empty_baselines/arduino_r4_serial_baseline/arduino_r4_serial_baseline.ino
```

Click **Verify**. Upload is not required.

Copy the full compile summary lines into:

```text
Hardware Deployment Run/hardware_hil/compile_logs/arduino_r4_serial_baseline_compile.txt
```

The file should contain lines like:

```text
Sketch uses XXXXX bytes (YY%) of program storage space. Maximum is 262144 bytes.
Global variables use XXXXX bytes (YY%) of dynamic memory, leaving XXXXX bytes for local variables. Maximum is 32768 bytes.
Board: Arduino R4 WiFi
Source: Arduino IDE Verify output for serial-only baseline.
```

## 4. Run The Final Analysis

From this directory:

```powershell
cd "C:\N Drive\Acads\6th SEM\WCT\CuKD-XAI\Hardware Deployment Run"
```

Run:

```powershell
python -m unittest .\tests\test_final_hil_postprocessing.py
```

Then:

```powershell
python .\hardware_hil\host\analyze_final_hil_evidence.py `
  --project-root . `
  --output-dir hardware_hil\reports\final_postprocessing
```

## 5. Files To Preserve

After the script finishes, preserve:

```text
Hardware Deployment Run/hardware_hil/reports/final_postprocessing/final_postprocessing_analysis.md
Hardware Deployment Run/hardware_hil/reports/final_postprocessing/final_postprocessing_analysis.json
Hardware Deployment Run/hardware_hil/reports/final_postprocessing/hil_fidelity.csv
Hardware Deployment Run/hardware_hil/reports/final_postprocessing/cycles_per_mac.csv
Hardware Deployment Run/hardware_hil/reports/final_postprocessing/model_only_footprint.csv
Hardware Deployment Run/hardware_hil/reports/final_postprocessing/compile_framework_baseline.csv
Hardware Deployment Run/hardware_hil/reports/final_postprocessing/quantization_drift_summary.csv
Hardware Deployment Run/hardware_hil/reports/final_postprocessing/quantization_drift_by_true_class.csv
Hardware Deployment Run/hardware_hil/reports/final_postprocessing/quantization_drift_pairs.csv
Hardware Deployment Run/hardware_hil/reports/final_postprocessing/evidence_traceability.csv
```

## 6. How To Use The Results In The Paper

- Use `cycles_per_mac.csv` for the architecture-efficiency table.
- Use `model_only_footprint.csv` to separate the model core from vendor framework
  overhead.
- Use `compile_framework_baseline.csv` to defend the ESP32-C3 and Arduino R4
  binary sizes.
- Use the quantization drift CSVs to discuss where fixed-point behavior differs
  from FP32.
- Use `evidence_traceability.csv` to keep every manuscript claim tied to a file.

