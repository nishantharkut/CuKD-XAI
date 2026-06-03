# WSN Mote Hardware Request for CuKD-XAI

Use this when asking for lab hardware. The software artifact now cross-compiles for MSP430F1611 and fits the target memory budget. The remaining hardware experiment is to flash an actual WSN-class mote, run serial self-tests, and measure latency/energy under the lab firmware stack.

## Best Hardware to Ask For

Preferred WSN-class boards:

| Board family | Why it is useful |
|---|---|
| TelosB / Tmote Sky | Strong match to the paper narrative: MSP430-class mote, very tight RAM/Flash budget. |
| Zolertia Z1 | MSP430-class WSN board with Contiki support. |
| MICAz / Mica2 | Classic WSN mote family; useful if the lab already has programming tools. |
| OpenMote / CC2538 | Newer WSN/6LoWPAN board; easier toolchain than very old motes. |
| nRF52840 sensor/802.15.4 board | Practical modern fallback if classic motes are unavailable. |

Ask for the board, programmer/debugger, serial cable/access, and whatever OS/toolchain the lab already uses.

## Information Needed From Professor/Lab

Please ask for these details before changing the code for a specific board:

| Needed detail | Why it matters |
|---|---|
| Exact mote model and MCU | Determines compiler, word size, RAM, Flash, and cycle counter availability. |
| RAM and Flash limits | Needed for `.text`, `.data`, `.bss`, stack, and model parameter accounting. |
| Toolchain | Examples: `msp430-gcc`, `msp430-elf-gcc`, TinyOS `nescc`, Contiki/RIOT build system. |
| Operating system | TinyOS, Contiki, RIOT, or bare-metal changes integration style. |
| Programming/debug adapter | Needed for flashing and serial output. |
| Serial console access | Needed to run self-test and print pass/fail/latency. |
| Power/energy measurement option | Optional but valuable for publication: Monsoon, Joulescope, oscilloscope shunt, or lab power analyzer. |

## First Hardware Experiment

1. Start from the MSP430F1611 cross-compiled smoke result in `hardware_export/msp430_build_v2/`.
2. Rebuild for the exact board/MCU if the lab mote is not MSP430F1611.
3. Include a small subset of generated `test_vectors.h`, not the full 56,200 vectors, to avoid Flash pressure.
4. Print serial pass/fail for the generated vectors.
5. Record compiled `.text`, `.rodata`, `.data`, `.bss`, heap, and stack usage.
6. Measure inference latency in cycles or microseconds using a hardware timer.
7. Optional: measure energy per inference with a power analyzer or shunt setup.

## Paper-Safe Claim After This Experiment

Safe if it passes:

> The fixed-point CuKD-XAI Student A RF-KD inference core and integer StandardScaler normalization helper cross-compile for MSP430F1611 and fit within TelosB-class memory. If flashed and tested on the actual mote, we can additionally report measured Flash/RAM footprint, serial self-test status, latency, and energy.

Still unsafe unless separately implemented:

> The full WSN-DS feature extraction and live IDS pipeline has been deployed on a mote.

## Current Software Artifact Boundary

Already covered:

- int8 weights and int32 biases for Student A `E_KD_from_RF`.
- calibrated int16 activation scaling.
- integer StandardScaler metadata and C normalization helper.
- generated test vectors and host C self-test.
- full-test software agreement evidence when `--num-test-vectors 56200` is used.
- MSP430F1611 cross-compiled smoke firmware: `2,842 B` Flash-class text, `0 B` `.data`, `6 B` `.bss`, and bounded project-function stack usage.

Not covered yet:

- live packet/routing feature extraction on the mote.
- mote OS integration.
- radio duty-cycle impact.
- real board latency and energy.
- interrupt/OS/network-stack stack-pressure measurement during live traffic.
