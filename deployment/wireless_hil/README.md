# FG-DS Wi-Fi UDP Hardware-in-the-Loop

This subsystem extends the frozen USB-serial FG-DS hardware evaluation with a
controlled-LAN Wi-Fi UDP transport. It uses the same seed-42 Student A and
Student B fixed-point exports and the same integer preprocessing and inference
core. The USB evidence under `results/hardware_hil/fgds_seed42` is not changed.

## Current status

The protocol-v2 firmware, host replay, verification, completion, compile
evidence, and report path are implemented. All four board/model bundles compile
against ESP32 core 3.3.11 and Arduino Renesas UNO core 1.6.0. The physical Wi-Fi
replay results have not yet been collected and must not be reported as completed
evidence.

## Experiment matrix

| Board | Model | Export | Full-test rows |
|---|---|---|---:|
| ESP32-C3 | Student A RF-KD | FG-DS seed 42 | 56,301 |
| ESP32-C3 | Student B RF-KD | FG-DS seed 42 | 56,301 |
| UNO R4 WiFi | Student A RF-KD | FG-DS seed 42 | 56,301 |
| UNO R4 WiFi | Student B RF-KD | FG-DS seed 42 | 56,301 |

Each board/model pair runs three ordered stages in one provisioned session:

1. `smoke_10`: 10 rows, stage ordinal 1.
2. `validation_1000`: 1,000 rows, stage ordinal 2.
3. `full_56301`: all 56,301 FG-DS test rows, stage ordinal 3.

The headline evaluation unit is the full 56,301-row stage. Smoke and validation
are execution gates and repeat prefixes of the same test ordering.

## Acceptance gates

Wireless transport is not allowed to change model output. Every full stage must
contain 56,301 ordered `OK` rows with no missing, duplicate, or unexpected row
IDs, 1.0 prediction agreement with the fixed reference, and 1.0 exact agreement
across all five integer logits. The verifier also reconciles the resulting
classification metrics directly with the manifest-bound strict export report.

The frozen expected full-stage values are:

| Model | Fixed accuracy | Fixed macro-F1 | Fixed vs. FP32 agreement |
|---|---:|---:|---:|
| Student A RF-KD | 0.984813768850 | 0.905693700314 | 0.995008969645 |
| Student B RF-KD | 0.986003800998 | 0.914564120068 | 0.994742544537 |

Any prediction, logit, classification-metric, row-sequence, or identity mismatch
invalidates the run. RTT, retries, response timeouts, RSSI, and footprint are
separate transport/platform measurements and are not model-quality metrics.

## What the experiment establishes

- The exact hash-bound fixed-point model executes on each MCU target.
- MCU predictions and all five integer logits can be compared exactly with the
  generated fixed reference for every row.
- The complete ordered replay can traverse a Wi-Fi UDP request/response path
  with explicit retries and auditable transport counters.
- MCU preprocessing/inference time and host-observed datagram round-trip time
  are recorded as separate measurements. The MCU value is wall-clock time for
  the bounded model code region and can include interrupt preemption.
- Flash and global RAM usage are bound to a preserved verbose compile log and
  firmware binary for every board/model pair.

## Claim boundary

The Raspberry Pi replays already extracted 17-feature FG-DS records. This path
does not perform live packet capture, packet-to-feature extraction, energy
measurement, secure application transport, Internet-scale testing, BLE, or
physical TelosB/WSN-radio deployment. Host RTT minus MCU compute time is not a
measurement of pure wireless latency because the two measurements use
independent clocks and include different software boundaries.

## Board roles

- On ESP32-C3, Wi-Fi transport and fixed-point inference execute on the same
  ESP32-C3 MCU.
- On UNO R4 WiFi, fixed-point inference executes on the RA4M1. The onboard
  ESP32-S3 connectivity module provides Wi-Fi through the `WiFiS3` stack.

## Credential and security boundary

SSID and password are entered interactively on the Raspberry Pi and sent over
local USB serial. They are not accepted through command-line arguments or
environment variables and are not written to source, manifests, connection
records, or result files. ESP32 Arduino persistence is disabled before Wi-Fi
initialization. Credential persistence by the UNO R4 WiFi connectivity
coprocessor is not asserted.

The random session ID provides run correlation and endpoint binding. It is not
cryptographic authentication. UDP payloads are protected by CRC for corruption
detection, not encrypted or authenticated against an attacker.

## Files

- `build/`: four immutable, hash-inventoried Arduino bundles.
- `firmware/`: protocol-v2 Arduino firmware and C parsers.
- `host/`: provisioning, replay, verification, completion, compile evidence,
  and final reporting tools.
- `protocol/udp_protocol.md`: exact wire and state-machine contract.
- `RUNBOOK.md`: verified Raspberry Pi commands for all four physical runs.
- `scripts/run_fgds_wireless_hil.sh`: atomic three-stage runner.

Do not compile with a build cache inside a bundle. Arduino CLI must receive an
external `--build-path`. Do not pass `--output-dir`: ESP32 core 3.3.11 exports a
`build/` tree into the sketch folder when that option is used, which correctly
invalidates the immutable bundle inventory.

See [RUNBOOK.md](RUNBOOK.md) before wiring or flashing either board.
