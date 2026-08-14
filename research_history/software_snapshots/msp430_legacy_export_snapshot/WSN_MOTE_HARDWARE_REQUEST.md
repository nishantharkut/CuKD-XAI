# WSN Mote Hardware Request for CuKD-XAI

Use this when asking for lab hardware. The goal is not to rebuild the full WSN-DS data-collection pipeline immediately; the first hardware experiment is to prove that the fixed-point IDS model core and integer normalization helper compile and execute within a WSN-class mote budget.

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

1. Generate the fixed-point artifacts using `hardware_export/run_wsnds_student_a_rfkd_e2e.py`.
2. Compile `wsnds_preprocess_int16.c` and `wsnds_student_a_rfkd_int8_inference.c` for the board target.
3. Include a small subset of generated `test_vectors.h`, not the full 56,200 vectors, to avoid Flash pressure.
4. Print serial pass/fail for the generated vectors.
5. Record compiled `.text`, `.data`, `.bss`, and estimated stack usage.
6. Measure inference latency in cycles or microseconds if a timer is available.
7. Optional: measure energy per inference.

## Paper-Safe Claim After This Experiment

Safe if it passes:

> The fixed-point CuKD-XAI Student A RF-KD inference core and integer StandardScaler normalization helper compile and pass generated vector tests on a WSN-class mote/toolchain, with measured Flash/RAM footprint and latency.

Still unsafe unless separately implemented:

> The full WSN-DS feature extraction and live IDS pipeline has been deployed on a mote.

## Current Software Artifact Boundary

Already covered:

- int8 weights and int32 biases for Student A `E_KD_from_RF`.
- calibrated int16 activation scaling.
- integer StandardScaler metadata and C normalization helper.
- generated test vectors and host C self-test.
- full-test software agreement evidence when `--num-test-vectors 56200` is used.

Not covered yet:

- live packet/routing feature extraction on the mote.
- mote OS integration.
- radio duty-cycle impact.
- real board latency and energy.
