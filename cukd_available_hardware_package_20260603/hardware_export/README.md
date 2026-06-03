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
5. Writes `preprocess_int_metadata.h` and `preprocess_int_metadata.json` with integer StandardScaler constants for subtract/multiply/shift normalization.
6. Selects deterministic class-covered test vectors from the WSN-DS test split.
7. Runs the FP32 Student MLP forward pass directly from the saved tensors.
8. Quantizes the standardized test vectors to calibrated signed int16 and runs the Python reference fixed-point simulator.
9. Writes `test_vectors.h` and `equivalence_report.json`.
10. Compiles `wsnds_preprocess_int16.c`, `wsnds_student_a_rfkd_int8_inference.c`, and `wsnds_student_a_rfkd_self_test.c` with `gcc`.
11. Runs the generated C self-test and writes `e2e_run_report.json`.

## Generated Files

The default output directory is `hardware_export/generated_student_a_rfkd_e2e/`. The final full-test output used for reporting is `hardware_export/generated_student_a_rfkd_e2e_v2/`.

| File | Purpose |
|---|---|
| `model_weights.h` | int8 weights, int32 biases, model dimensions, parameter byte counts |
| `preprocess_metadata.h` | C-readable feature order, class names, scaler mean/scale constants |
| `preprocess_metadata.json` | audit-friendly preprocessing contract and scaler metadata |
| `preprocess_int_metadata.h` | no-FPU integer scaler constants for StandardScaler normalization |
| `preprocess_int_metadata.json` | audit-friendly integer preprocessing formula, scales, shifts, and operation counts |
| `test_vectors.h` | standardized calibrated int16 test vectors, expected fixed-point logits, expected fixed predictions |
| `equivalence_report.json` | FP32-vs-fixed agreement, fixed accuracy on vectors, input saturation statistics |
| `export_summary.json` | model export summary plus E2E artifact summary |
| `cukd_student_a_rfkd_self_test` | host-compiled self-test binary, if `gcc` is available |
| `e2e_run_report.json` | export/compile/self-test commands, return codes, stdout/stderr |

## Integer StandardScaler Proof

The exported inference core consumes already standardized calibrated-int16 inputs. To address the normalization-cost question, the exporter now also emits an integer StandardScaler contract:

```c
standardized_q = ((raw_q - scaler_mean_q) * scaler_inv_scale_q) >> right_shift
```

For WSN-DS this is `17` subtracts, `17` multiplies, `17` shifts, and `17` saturations per sample after the raw 17 features already exist. The generated files are `preprocess_int_metadata.h`, `preprocess_int_metadata.json`, and `wsnds_preprocess_int16.c`.

This strengthens the no-FPU argument for normalization, but it still does not implement the original WSN-DS feature extraction process on a mote. Keep that distinction clear in the paper.

## Numerical Precision Evidence

From the full 56,200-vector WSN-DS software export already recorded in `origin/main:hardware_export/generated_student_a_rfkd_e2e_v2/equivalence_report.json`:

| Field | Value | Interpretation |
|---|---:|---|
| fixed-vs-FP32 agreement | `0.9946975088967972` | fixed-point predictions match the FP32 student for about 99.47% of held-out samples |
| fixed accuracy on vectors | `0.9863523131672598` | fixed-point model accuracy over the full WSN-DS test split |
| FP32 accuracy on vectors | `0.9863701067615659` | FP32 student accuracy over the same split |
| fixed logit range | `[-19507, 9228]` | final logits remain inside signed int16 bounds |
| input saturation | `0 / 955400` | calibrated Q8 standardized inputs did not saturate on the full test split |

The generated layer metadata uses calibrated fractional widths: input Q8, layer output Q9/Q8/Q7, and shifts 4/7/7. These numbers are evidence for numerical stability of the generated software artifact, not a substitute for real board timing or energy measurements.

## MSP430F1611 Target-Toolchain Evidence

The final fixed-point model core was also cross-compiled for `msp430f1611` using Mitto Systems MSP430-GCC `9.3.1.11`, MSP430 support files `1.212`, and `-Os`. This directly addresses the host-vs-target compiler objection, but it still does not measure physical mote latency or energy.

Object-level results:

| Object | `.text` | `.rodata` | `.data` | `.bss` |
|---|---:|---:|---:|---:|
| `wsnds_preprocess_int16_msp430.o` | `412 B` | `136 B` | `0 B` | `0 B` |
| `wsnds_student_a_rfkd_int8_inference_msp430.o` | `494 B` | `1,348 B` | `0 B` | `0 B` |

Linked MSP430 smoke firmware results:

| Metric | Value |
|---|---:|
| `msp430-elf-size` text | `2,842 B` |
| `.data` | `0 B` |
| `.bss` | `6 B` |
| total `text + data + bss` | `2,848 B` |
| `.rodata` section | `1,484 B` |
| `.text` section | `1,356 B` |

Compiler-reported stack usage with `-fstack-usage`:

| Function | Stack usage |
|---|---:|
| `main` | `104 B` |
| `cukd_standardize_raw_q` | `26 B` |
| `cukd_dense_i8_q15` | `26 B` |
| `cukd_forward_q15` | `106 B` |
| `cukd_predict_q15` | `12 B` |

Disassembly confirms MSP430 helper routines for wider arithmetic: `__mulhisi2`, `__mulsi2`, `__mspabi_mpyll`, `__mspabi_srai`, `__mspabi_sral`, and `__mspabi_srall`. This is expected because MSP430 is a 16-bit architecture. The wider arithmetic is retained to avoid overflow and preserve the observed fixed-point stability; therefore the evidence supports memory feasibility, not final cycle latency.

See `hardware_export/MSP430_CROSS_COMPILE_REPORT.md` for the full MSP430 report.

## WSN Mote Next Step

Use `hardware_export/WSN_MOTE_HARDWARE_REQUEST.md` when asking for lab hardware. The immediate experiment should be compile-size plus serial self-test on an actual WSN-class board, not Raspberry Pi benchmarking.

## Manual Compile Check

If you want to compile manually after generating the headers:

```bash
gcc -std=c99 -Wall -Wextra -Os \
  -Ihardware_export/generated_student_a_rfkd_e2e_v2 \
  hardware_export/wsnds_preprocess_int16.c \
  hardware_export/wsnds_student_a_rfkd_int8_inference.c \
  hardware_export/wsnds_student_a_rfkd_self_test.c \
  -o hardware_export/generated_student_a_rfkd_e2e_v2/cukd_student_a_rfkd_self_test

hardware_export/generated_student_a_rfkd_e2e_v2/cukd_student_a_rfkd_self_test
```

For MSP430F1611, if the MSP430-GCC toolchain and support files are installed:

```bash
msp430-elf-gcc -mmcu=msp430f1611 -std=c99 -Wall -Wextra -Os \
  -Ihardware_export/generated_student_a_rfkd_e2e_v2 \
  -c hardware_export/wsnds_preprocess_int16.c \
  -o hardware_export/msp430_build_v2/wsnds_preprocess_int16_msp430.o

msp430-elf-gcc -mmcu=msp430f1611 -std=c99 -Wall -Wextra -Os \
  -Ihardware_export/generated_student_a_rfkd_e2e_v2 \
  -c hardware_export/wsnds_student_a_rfkd_int8_inference.c \
  -o hardware_export/msp430_build_v2/wsnds_student_a_rfkd_int8_inference_msp430.o

msp430-elf-size -A hardware_export/msp430_build_v2/wsnds_preprocess_int16_msp430.o
msp430-elf-size -A hardware_export/msp430_build_v2/wsnds_student_a_rfkd_int8_inference_msp430.o
```

## What Counts as Evidence

Use these fields in the paper/report:

- `export_summary.json`: static parameter bytes, activation estimate, MACs per inference.
- `equivalence_report.json`: whether the fixed-point export agrees with FP32 on representative held-out WSN-DS vectors.
- `equivalence_report.json -> input_quantization`: whether standardized features saturate after calibrated int16 encoding.
- `e2e_run_report.json`: whether the integer C kernel compiled and matched the generated fixed-point vectors.
- `hardware_export/msp430_build_v2/`: MSP430F1611 cross-compiled smoke firmware, object files, and disassembly evidence.
- `*.su`: compiler-reported stack-usage files for the MSP430 smoke build.

The C self-test passing means the generated C implementation matches the generated fixed-point reference. It does not by itself prove the fixed-point model preserves the FP32 macro-F1. That claim requires the agreement/accuracy fields in `equivalence_report.json`, and ideally a larger vector count or full-test export.

## Static Memory Estimate

Student A architecture: `17 -> 32 -> 16 -> 5`

| Item | Estimate |
|---|---:|
| int8 weights | `1,136 bytes` |
| int32 biases | `212 bytes` |
| total generated parameters | about `1,348 bytes` before compiler/code overhead |
| int16 activations estimate | about `140 bytes` if all layer buffers are counted |
| MACs per inference | `1,136` multiply-accumulates |

This is stronger than a desktop-only PyTorch/ONNX latency claim because it gives a dependency-free integer C kernel plus generated reproducibility vectors.

## Honest Limitations

- This is an E2E software export proof, not a physical TelosB deployment.
- WSN-DS feature extraction on the mote is not implemented here.
- `preprocess_metadata.h` contains float scaler constants for reproducibility/host preprocessing; `preprocess_int_metadata.h` contains the no-FPU integer StandardScaler constants.
- The generated integer preprocessing covers normalization only; WSN-DS raw feature extraction on the mote remains outside this artifact.
- Even calibrated int16 encoding can saturate if future data exceeds calibration ranges; check `equivalence_report.json` before claiming fixed-point accuracy preservation.
- MSP430F1611 cross-compilation shows memory feasibility, but does not provide physical latency or energy.
- MSP430 disassembly includes helper routines for 32-bit/64-bit arithmetic; cycle cost must be measured on hardware.

## Paper-Safe Claim

Safe:

> We provide an end-to-end software export path for the best ultra-small WSN-DS student, including v2.3 preprocessing metadata, fixed-point integer C inference, generated held-out test vectors, host C self-test, and MSP430F1611 target-toolchain footprint evidence for the preprocessing and inference core.

Unsafe:

> We deployed the complete IDS pipeline on TelosB hardware.
