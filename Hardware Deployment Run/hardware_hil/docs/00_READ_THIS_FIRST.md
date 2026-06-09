# Hardware HIL: Read This First

This folder is for the final hardware-in-the-loop validation of the CuKD-XAI WSN-DS Student A RF-KD model.

## What This Proves

- The fixed-point Student A IDS firmware runs on real MCU-class boards.
- The MCU predictions match the generated fixed-point reference.
- Device-side preprocessing, inference, and total latency are measured by the board firmware.
- The replay loop can process smoke, 1,000-row, and full 56,200-row WSN-DS test-vector sets.

## What This Does Not Prove

- No live WSN packet capture.
- No raw radio-frame-to-feature extraction.
- No energy measurement.
- No Raspberry Pi AI HAT+ result.
- No physical TelosB result unless actual motes are added later.

## Devices

- Raspberry Pi 5: host controller, serial logger, report generator.
- ESP32-C3 DevKit: primary real MCU execution target.
- Arduino R4: secondary MCU portability target.
- Raspberry Pi AI HAT+: not used in the core result.

## Order Of Work

1. Read `hardware_hil/docs/08_PHYSICAL_CONNECTIONS_AND_CABLES.md`.
2. Generate model and replay artifacts from `hardware_export`.
3. Build one firmware bundle for ESP32-C3 and one for Arduino R4.
4. Flash the ESP32-C3 first.
5. Connect only ESP32-C3 to Raspberry Pi 5 by USB.
6. Run 10-row smoke replay.
7. Run 1,000-row replay.
8. Run full 56,200-row replay only after the first two pass.
9. Repeat on Arduino R4.
10. Verify logs and generate the report.

## Pass Criteria

For each board run, preserve:

- serial response CSV,
- sequence summary JSON,
- verification metrics JSON,
- board/environment notes,
- build memory output from the Arduino build tool or IDE.

The strongest claim is high `mcu_vs_fixed_reference_agreement`, zero non-OK status rows, stable row completion, and latency percentiles reported separately for each board.

## Connection Rule

Use USB serial only. Do not wire Raspberry Pi GPIO pins to ESP32-C3 or Arduino R4 pins for this experiment. The detailed physical setup is in `hardware_hil/docs/08_PHYSICAL_CONNECTIONS_AND_CABLES.md`.


## Optional Breadboard Practice

If you want to use jumper wires and a breadboard, first read `hardware_hil/docs/09_OPTIONAL_BREADBOARD_AND_JUMPER_GUIDE.md`. Treat it as practice/debugging only; it is not part of the core HIL evidence.


## Student B Capacity Extension

After Student A evidence is preserved, use `hardware_hil/docs/10_STUDENT_B_HIL_RUNBOOK.md` for the separate Student B RF-KD run. Keep Student B generated folders, firmware bundles, and result folders separate from Student A.
