# MSP430F1611 Cross-Compile Report

This report records the target-toolchain footprint evidence for the CuKD-XAI WSN-DS Student A RF-KD fixed-point C export.

## Scope

This is a cross-compilation and static-footprint proof for an MSP430F1611-class target. It is not a physical TelosB/Tmote Sky deployment and does not measure real latency, energy, radio behavior, or live WSN feature extraction.

## Target and Toolchain

| Item | Value |
|---|---|
| Target MCU | `msp430f1611` |
| Target class | TelosB/Tmote Sky-style MSP430 WSN mote |
| Nominal target memory | 48 KB Flash, 10 KB RAM |
| Compiler | `msp430-elf-gcc` from Mitto Systems MSP430-GCC `9.3.1.11` |
| Binutils | `msp430-elf-size` / `msp430-elf-objdump` `2.34` |
| Support files | MSP430 GCC support files `1.212` |
| Optimization | `-Os` |
| Stack reporting | `-fstack-usage` |

## Object-Level Cross-Compile Footprint

| Object | `.text` | `.rodata` | `.data` | `.bss` | Notes |
|---|---:|---:|---:|---:|---|
| `wsnds_preprocess_int16_msp430.o` | `412 B` | `136 B` | `0 B` | `0 B` | Integer StandardScaler normalization helper and constants |
| `wsnds_student_a_rfkd_int8_inference_msp430.o` | `494 B` | `1,348 B` | `0 B` | `0 B` | Fixed-point MLP inference; model weights/biases remain in read-only storage |

## Linked Smoke Firmware Footprint

The smoke firmware links integer StandardScaler normalization, fixed-point inference, and a minimal `main` that calls both paths.

`msp430-elf-size -A` reported:

| Section | Size |
|---|---:|
| `__reset_vector` | `2 B` |
| `.rodata` | `1,484 B` |
| `.text` | `1,356 B` |
| `.data` | `0 B` |
| `.bss` | `2 B` |
| `.heap` | `4 B` |

`msp430-elf-size` summary:

| Field | Size |
|---|---:|
| `text` | `2,842 B` |
| `data` | `0 B` |
| `bss` | `6 B` |
| `dec` | `2,848 B` |

Interpretation: the smoke firmware uses about `2.8 KB` of Flash-class storage and `6 B` of static RAM/heap before runtime stack. This is well below the MSP430F1611/TelosB-class 48 KB Flash and 10 KB RAM budget.

## Compiler-Reported Stack Usage

Generated with `-fstack-usage`:

| Function | Stack usage |
|---|---:|
| `main` | `104 B` |
| `cukd_standardize_raw_q` | `26 B` |
| `cukd_dense_i8_q15` | `26 B` |
| `cukd_forward_q15` | `106 B` |
| `cukd_predict_q15` | `12 B` |

A conservative project-function call-chain estimate is about `248 B` during prediction (`main + cukd_predict_q15 + cukd_forward_q15 + cukd_dense_i8_q15`) and about `130 B` during preprocessing (`main + cukd_standardize_raw_q`). This excludes interrupt nesting, OS/network stack pressure, and any ABI helper internals.

## Arithmetic Helper Evidence

Disassembly confirmed that wider arithmetic is lowered into MSP430 helper routines, including:

- `__mspabi_srai`
- `__mspabi_sral`
- `__mspabi_srall`
- `__mulhisi2`
- `__mulsi2`
- `__mspabi_mpyll`

This is expected on a 16-bit MSP430 target. The design intentionally uses wider accumulators to preserve fixed-point numerical stability, but this means the cross-compile result supports memory feasibility, not final cycle latency or energy.

## Safe Claim

Safe:

> The fixed-point preprocessing and inference core cross-compiles for MSP430F1611. A linked smoke firmware requires `2,842 B` Flash-class `text` storage, `0 B` `.data`, and `6 B` `.bss`, with compiler-reported bounded project-function stack usage. This supports MSP430/TelosB-class memory-feasibility of the model core.

Unsafe:

> The complete IDS has been deployed on TelosB hardware with measured real-time latency and energy.

## Remaining Hardware Work

- Flash on an actual WSN-class board.
- Run serial self-tests on-device.
- Measure cycle latency and energy.
- Integrate with TinyOS/Contiki/RIOT or the lab's mote firmware stack.
- Implement or account for live WSN feature extraction cost.
