# Current FGDS MSP430F1611 Static Evidence

This directory contains the reproducible target-toolchain check for the current feature-group-disjoint (FGDS) seed-42 Student A and Student B fixed-point exports. It does not reuse the archived random-row Student A footprint as evidence for the current models.

These exports are the preserved seed-42 deployment models from the five-seed FGDS confirmation lineage. They are separate trained states from the later ten-seed aggregate experiment; the runner does not substitute an aggregate seed or imply ten hardware/compiler replications.

## Evidence Boundary

The runner establishes only that the preserved integer preprocessing and inference cores compile and link for `msp430f1611`, together with compiler-reported section sizes and per-function stack-usage records. It does not establish physical TelosB execution, latency, energy, radio integration, or live WSN feature extraction.

The source artifacts are consumed in place and never modified. Every generated object, ELF, map, disassembly, stack file, and JSON report is written below this directory.

## Frozen Inputs

The runner verifies all files listed by each strict export manifest and then checks the contract fields used by this build:

- FGDS protocol ID and seed `42`
- dataset, split/scaler lineage, execution contract, and RF soft-target content identity
- Student A and Student B model artifact and generated-header identities
- export ID recomputed from the canonical strict-report identity payload
- current integer preprocessing header and current common C source hashes
- exact dimensions, parameter bytes, activation estimate, and MAC count

The expected identities are recorded in `expected_contracts.json`. Any mismatch stops the build.

## Accepted Toolchain Inputs

No download is implemented. Supply either:

1. extracted roots for TI MSP430 GCC `9.3.1.11` and support files `1.212`; or
2. local copies of the official archives named `msp430-gcc-9.3.1.11_win64.zip` and `msp430-gcc-support-files-1.212.zip`.

Archive mode computes and records each archive SHA-256 before extracting it into `toolchain_cache/`. Optional expected hashes can be passed to make archive verification externally anchored. Root mode hashes the compiler, size tool, objdump tool, and selected `msp430f1611.ld` file. Both modes verify the expected release markers and compiler version output.

## Input-Only Check

This command performs no toolchain discovery and no compilation:

```powershell
python deployment/msp430/current_fgds_static/run_static_cross_compile.py --verify-inputs-only
```

## Build from Extracted Roots

```powershell
python deployment/msp430/current_fgds_static/run_static_cross_compile.py `
  --toolchain-root "C:\path\to\msp430-gcc-9.3.1.11_win64" `
  --support-root "C:\path\to\msp430-gcc-support-files-1.212"
```

## Build from Local Official Archives

```powershell
python deployment/msp430/current_fgds_static/run_static_cross_compile.py `
  --toolchain-archive "C:\path\to\msp430-gcc-9.3.1.11_win64.zip" `
  --support-archive "C:\path\to\msp430-gcc-support-files-1.212.zip" `
  --toolchain-archive-sha256 "<independently recorded SHA-256>" `
  --support-archive-sha256 "<independently recorded SHA-256>"
```

The default output is:

```text
deployment/msp430/current_fgds_static/artifacts/
  student_A/msp430_static_evidence.json
  student_B/msp430_static_evidence.json
  msp430_static_summary.json
```

Each per-model report records the exact command argument arrays and printable commands, compiler/binutils versions, source and input hashes, ELF and object section sizes, compiler stack entries, generated artifact hashes, and wide-arithmetic helper symbols found in the disassembly. A failed tool or build gate produces a failure JSON instead of a successful footprint claim.

Existing per-student output is left untouched unless `--overwrite` is supplied. During a new build, progress is persisted after every command so a failure report retains the exact successful and failed invocations plus any partial artifacts.

## Interpretation

The reported static flash load is the Berkeley `text + data` total: executable code and read-only constants in `text`, plus initialized writable data that must also be stored in flash. The reported static RAM value is only the `data + bss` lower bound. Runtime call-chain stack, interrupt nesting, and any operating-system or network-stack state are excluded. The `.su` records are retained as per-function compiler evidence and are not presented as a proven whole-program peak stack value.
