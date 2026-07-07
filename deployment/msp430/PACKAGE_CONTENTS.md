# CuKD-XAI Hardware Export MSP430 Docs Package

This package contains the updated Markdown documentation, the fixed-point export source files, the final v2 summary artifacts, and the MSP430F1611 cross-compile evidence.

Start with:

- `docs/professor/PROFESSOR_RESULTS_COMPARISON.md`
- `deployment/msp430/README.md`
- `deployment/msp430/MSP430_CROSS_COMPILE_REPORT.md`
- `deployment/msp430/WSN_MOTE_HARDWARE_REQUEST.md`

Key evidence:

- v2 full-test fixed-point agreement: `0.9946975088967972`
- v2 fixed accuracy: `0.9863523131672598`
- input saturation: `0 / 955400`
- MSP430F1611 linked smoke firmware: `text=2842`, `data=0`, `bss=6`
- compiler-reported project-function stack: `main=104 B`, `cukd_forward_q15=106 B`, `cukd_standardize_raw_q=26 B`

Boundary: this is software fixed-point plus MSP430 target-toolchain footprint evidence, not physical mote latency/energy deployment.


