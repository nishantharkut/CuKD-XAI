# WSN-DS Student A RF-KD End-to-End Fixed-Point Export

This folder is the hardware-facing export path for the mote-scale CuKD-XAI result. It does not modify the training notebooks or the existing WSN-DS/Edge-IIoT results.

## Correct Research Framing

For the ultra-small Student A model, the WSN-DS 10-seed result is:

| Config | Macro-F1 | Params | FP32 size | Role |
|---|---:|---:|---:|---|
| `E_KD_from_RF` | `0.9200` | `1,189` | `4.64 KB` | Primary embedded model |
| `J_CoDistill_RF_CL` | `0.9181` | `1,189` | `4.64 KB` | Ablation/capacity analysis |

So the TelosB/MSP430-style hardware narrative should use Student A `E_KD_from_RF` as the main model. Config J should not be marketed as the embedded headline.

## Full E2E Command

Run this from the repository root in the same Jupyter/Python environment where PyTorch, NumPy, Pandas, scikit-learn, and the trained artifact are available:

```bash
python3 hardware_export/run_wsnds_student_a_rfkd_e2e.py \
  --state-dict origin/main:Final/wsnds_deployment_qat_outputs/tmp/E_student_A_KD_from_RF_fp32.pt \
  --dataset-csv WSN-DS.csv \
  --output-dir hardware_export/generated_student_a_rfkd_e2e \
  --num-test-vectors 256
```

If the `.pt` file is already local, use:

```bash
python3 hardware_export/run_wsnds_student_a_rfkd_e2e.py \
  --state-dict Final/wsnds_deployment_qat_outputs/tmp/E_student_A_KD_from_RF_fp32.pt \
  --dataset-csv WSN-DS.csv \
  --output-dir hardware_export/generated_student_a_rfkd_e2e \
  --num-test-vectors 256
```

The runner performs the complete software proof path:

1. Loads the trained Student A RF-KD FP32 state dict.
2. Exports `model_weights.h` with int8 weights, int32 biases, and calibrated int16 activation metadata.
3. Reproduces the v2.3 WSN-DS preprocessing contract: strip columns, drop `id`, label-encode classes alphabetically, fit `StandardScaler` on all rows, and use the seed-42 stratified 70/15/15 split.
4. Writes `preprocess_metadata.h` and `preprocess_metadata.json` with feature order, class order, scaler mean, and scaler scale.
5. Selects deterministic class-covered test vectors from the WSN-DS test split.
6. Runs the FP32 Student MLP forward pass directly from the saved tensors.
7. Quantizes the standardized test vectors to calibrated signed int16 and runs the Python reference fixed-point simulator.
8. Writes `test_vectors.h` and `equivalence_report.json`.
9. Compiles `wsnds_student_a_rfkd_int8_inference.c` plus `wsnds_student_a_rfkd_self_test.c` with `gcc`.
10. Runs the generated C self-test and writes `e2e_run_report.json`.

## Generated Files

The default output directory is `hardware_export/generated_student_a_rfkd_e2e/`.

| File | Purpose |
|---|---|
| `model_weights.h` | int8 weights, int32 biases, model dimensions, parameter byte counts |
| `preprocess_metadata.h` | C-readable feature order, class names, scaler mean/scale constants |
| `preprocess_metadata.json` | audit-friendly preprocessing contract and scaler metadata |
| `test_vectors.h` | standardized calibrated int16 test vectors, expected fixed-point logits, expected fixed predictions |
| `equivalence_report.json` | FP32-vs-fixed agreement, fixed accuracy on vectors, input saturation statistics |
| `export_summary.json` | model export summary plus E2E artifact summary |
| `cukd_student_a_rfkd_self_test` | host-compiled self-test binary, if `gcc` is available |
| `e2e_run_report.json` | export/compile/self-test commands, return codes, stdout/stderr |

## Manual Compile Check

If you want to compile manually after generating the headers:

```bash
gcc -std=c99 -Wall -Wextra -Os \
  -Ihardware_export/generated_student_a_rfkd_e2e \
  hardware_export/wsnds_student_a_rfkd_int8_inference.c \
  hardware_export/wsnds_student_a_rfkd_self_test.c \
  -o hardware_export/generated_student_a_rfkd_e2e/cukd_student_a_rfkd_self_test

hardware_export/generated_student_a_rfkd_e2e/cukd_student_a_rfkd_self_test
```

For MSP430, if the toolchain is installed:

```bash
msp430-elf-gcc -mmcu=msp430f1611 -std=c99 -Wall -Wextra -Os \
  -Ihardware_export/generated_student_a_rfkd_e2e \
  -c hardware_export/wsnds_student_a_rfkd_int8_inference.c \
  -o /tmp/wsnds_student_a_rfkd_int8_msp430.o

msp430-elf-size /tmp/wsnds_student_a_rfkd_int8_msp430.o
```

## What Counts as Evidence

Use these fields in the paper/report:

- `export_summary.json`: static parameter bytes, activation estimate, MACs per inference.
- `equivalence_report.json`: whether the fixed-point export agrees with FP32 on representative held-out WSN-DS vectors.
- `equivalence_report.json -> input_quantization`: whether standardized features saturate after calibrated int16 encoding.
- `e2e_run_report.json`: whether the integer C kernel compiled and matched the generated fixed-point vectors.

The C self-test passing means the generated C implementation matches the generated fixed-point reference. It does not by itself prove the fixed-point model preserves the FP32 macro-F1. That claim requires the agreement/accuracy fields in `equivalence_report.json`, and ideally a larger vector count or full-test export.

## Static Memory Estimate

Student A architecture: `17 -> 32 -> 16 -> 5`

| Item | Estimate |
|---|---:|
| int8 weights | `1,136 bytes` |
| int32 biases | `212 bytes` |
| total generated parameters | about `1,348 bytes` before compiler/code overhead |
| Q15 activations estimate | about `140 bytes` if all layer buffers are counted |
| MACs per inference | `1,136` multiply-accumulates |

This is stronger than a desktop-only PyTorch/ONNX latency claim because it gives a dependency-free integer C kernel plus generated reproducibility vectors.

## Honest Limitations

- This is an E2E software export proof, not a physical TelosB deployment.
- WSN-DS feature extraction on the mote is not implemented here.
- `preprocess_metadata.h` contains float scaler constants for reproducibility/host preprocessing; the no-FPU C inference core consumes standardized calibrated-int16 vectors.
- Even calibrated int16 encoding can saturate if future data exceeds calibration ranges; check `equivalence_report.json` before claiming fixed-point accuracy preservation.
- MSP430 `.text`, `.data`, and `.bss` numbers require `msp430-elf-gcc` or another MSP430 toolchain.

## Paper-Safe Claim

Safe:

> We provide an end-to-end software export path for the best ultra-small WSN-DS student, including v2.3 preprocessing metadata, fixed-point integer C inference, generated held-out test vectors, and a host self-test that verifies the C kernel against the Python fixed-point reference.

Unsafe:

> We deployed the complete IDS pipeline on TelosB hardware.
