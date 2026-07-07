# Generate Replay Assets

This step creates the model headers, integer preprocessing metadata, replay CSV, and reference CSV used by the hardware run.

## Full 56,200-Vector Export

From the repository root:

```bash
python deployment/firmware_export/wsnds_rfkd_hil/run_wsnds_student_a_rfkd_e2e.py \
  --state-dict results/runtime/onnx_openvino/wsnds/tmp/E_student_A_KD_from_RF_fp32.pt \
  --dataset-csv data/wsnds/WSN-DS.csv \
  --output-dir deployment/firmware_export/wsnds_rfkd_hil/generated_student_a_rfkd_hil_full \
  --num-test-vectors 56200
```

Expected generated files include:

- `model_weights.h`
- `preprocess_int_metadata.h`
- `test_vectors.h`
- `hil_replay_vectors.csv`
- `hil_reference_predictions.csv`
- `equivalence_report.json`
- `export_summary.json`

## Why The CSVs Matter

The MCU streamer uses `hil_replay_vectors.csv`, not the C header. The verifier uses `hil_reference_predictions.csv`.

The replay CSV contains already extracted WSN-DS feature values encoded as fixed-point integers. The MCU still runs integer StandardScaler preprocessing before inference.

## Optional Smaller Export For Early Testing

If you only want a small generated directory before the full export:

```bash
python deployment/firmware_export/wsnds_rfkd_hil/run_wsnds_student_a_rfkd_e2e.py \
  --state-dict results/runtime/onnx_openvino/wsnds/tmp/E_student_A_KD_from_RF_fp32.pt \
  --dataset-csv data/wsnds/WSN-DS.csv \
  --output-dir deployment/firmware_export/wsnds_rfkd_hil/generated_student_a_rfkd_hil_1000 \
  --num-test-vectors 1000
```

For the final paper evidence, use the full 56,200-row export if the board replay is stable.


