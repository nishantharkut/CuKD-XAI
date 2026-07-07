# WSN-DS Runtime Benchmarks From Existing Artifacts

This notebook does not retrain any model. It loads existing .pt FP32 student artifacts from the completed deployment/QAT output folder and only runs ONNX Runtime / OpenVINO export and latency benchmarking.

## Inputs

Default existing output folder:

`results/runtime/onnx_openvino/wsnds/`

It must contain:

`tmp/D_student_A_scratch_fp32.pt`
`tmp/E_student_A_KD_from_RF_fp32.pt`
`tmp/J_student_A_CoDistill_RF_CL_fp32.pt`
`tmp/D_student_B_scratch_fp32.pt`
`tmp/E_student_B_KD_from_RF_fp32.pt`
`tmp/J_student_B_CoDistill_RF_CL_fp32.pt`

Set `EXISTING_DEPLOYMENT_OUTPUT_DIR` if the folder is somewhere else.

The notebook also needs `data/wsnds/WSN-DS.csv` to recreate the exact v2.3 test split for accuracy/F1 checks.

If any required .pt file is missing, the notebook stops before benchmarking instead of writing a misleading all-skipped CSV.

## Outputs

Written under:

`<existing deployment output>/runtime_from_existing_outputs/`

Key files:

- `wsnds_existing_artifact_runtime_summary.csv`
- `wsnds_existing_artifact_runtime_results.json`
- `deployable_runtime_results/*.onnx`

## Runtime Interpretation

Use only rows where `status = ok`.

ONNX Runtime rows compare:

- `onnx_fp32`
- `onnx_dynamic_int8`

OpenVINO rows report:

- `openvino_fp32_from_onnx`

Do not claim INT8 speedup unless `onnx_dynamic_int8` is actually faster than `onnx_fp32` in the CSV.


