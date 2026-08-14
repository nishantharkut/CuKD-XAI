# Run Replay And Verify

Run these commands on the Raspberry Pi 5 host from the repository root.

Set variables first:

```bash
PORT=/dev/ttyACM0
GEN=hardware_export/generated_student_a_rfkd_hil_full
BOARD=esp32c3
mkdir -p hardware_hil/results/${BOARD}
```

Change `PORT` if your board appears as `/dev/ttyUSB0` or another path.

## 1. Smoke Run: 10 Rows

```bash
python -m hardware_hil.host.stream_vectors \
  --port ${PORT} \
  --vectors-csv ${GEN}/hil_replay_vectors.csv \
  --output-csv hardware_hil/results/${BOARD}/smoke_mcu.csv \
  --summary-json hardware_hil/results/${BOARD}/smoke_sequence.json \
  --limit 10 \
  --timeout 2.0
```

Verify:

```bash
python -m hardware_hil.host.verify_results \
  --mcu-csv hardware_hil/results/${BOARD}/smoke_mcu.csv \
  --reference-csv ${GEN}/hil_reference_predictions.csv \
  --output-json hardware_hil/results/${BOARD}/smoke_metrics.json
```

Do not run the full set until smoke passes with zero non-OK rows.

## 2. Validation Run: 1,000 Rows

```bash
python -m hardware_hil.host.stream_vectors \
  --port ${PORT} \
  --vectors-csv ${GEN}/hil_replay_vectors.csv \
  --output-csv hardware_hil/results/${BOARD}/validation_1000_mcu.csv \
  --summary-json hardware_hil/results/${BOARD}/validation_1000_sequence.json \
  --limit 1000 \
  --timeout 2.0

python -m hardware_hil.host.verify_results \
  --mcu-csv hardware_hil/results/${BOARD}/validation_1000_mcu.csv \
  --reference-csv ${GEN}/hil_reference_predictions.csv \
  --output-json hardware_hil/results/${BOARD}/validation_1000_metrics.json
```

## 3. Full Run: 56,200 Rows

```bash
python -m hardware_hil.host.stream_vectors \
  --port ${PORT} \
  --vectors-csv ${GEN}/hil_replay_vectors.csv \
  --output-csv hardware_hil/results/${BOARD}/full_56200_mcu.csv \
  --summary-json hardware_hil/results/${BOARD}/full_56200_sequence.json \
  --timeout 2.0

python -m hardware_hil.host.verify_results \
  --mcu-csv hardware_hil/results/${BOARD}/full_56200_mcu.csv \
  --reference-csv ${GEN}/hil_reference_predictions.csv \
  --output-json hardware_hil/results/${BOARD}/full_56200_metrics.json
```

## 4. Report

After one or both boards have full metrics:

```bash
python -m hardware_hil.host.generate_report \
  --metric esp32c3=hardware_hil/results/esp32c3/full_56200_metrics.json \
  --metric arduino_r4=hardware_hil/results/arduino_r4/full_56200_metrics.json \
  --output-md hardware_hil/reports/hardware_hil_summary.md \
  --output-csv hardware_hil/reports/hardware_hil_tables.csv
```

If only ESP32-C3 is complete, omit the Arduino metric until that run exists.
