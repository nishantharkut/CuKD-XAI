# Troubleshooting

## Serial Port Busy

Symptom: Python cannot open the port.

Fix:

- Close Arduino Serial Monitor.
- Close any other terminal watching the port.
- Unplug/replug the board.

## Timeout On First Row

Symptom: `timeout waiting for row_id=0`.

Fix:

- Confirm the flashed sketch is the HIL sketch, not the self-test sketch.
- Check that host baud is `115200`.
- Increase `--settle-seconds 4` for boards that reset on serial open.
- Press reset once after the Python script opens the port only if the board is stuck.

## BAD_CHECKSUM

Symptom: response status is `BAD_CHECKSUM`.

Fix:

- Do not edit CSV rows manually.
- Regenerate `hil_replay_vectors.csv` from the exporter.
- Confirm the host script and firmware use the same `hardware_hil/protocol/serial_protocol.md` contract.

## BAD_LENGTH

Symptom: board reports `BAD_LENGTH`.

Fix:

- Confirm the CSV header is exactly `row_id,f0,...,f16`.
- Confirm you are using `hil_replay_vectors.csv`, not `hil_reference_predictions.csv`.

## Low MCU Agreement

Symptom: `mcu_vs_fixed_reference_agreement` is unexpectedly low.

Fix:

- Regenerate the firmware bundle from the same generated directory used for the replay CSV.
- Confirm `model_weights.h` and `preprocess_int_metadata.h` in the sketch folder match the generated directory.
- Reflash the board after bundling.
- Run the 10-row smoke test again before full replay.

## Arduino Build Cannot Find Header

Symptom: build error for `model_weights.h`, `preprocess_int_metadata.h`, or `cukd_model.h`.

Fix:

- Use `hardware_hil.host.prepare_firmware_bundle` instead of manually copying files.
- Open the `.ino` file inside the generated sketch folder, not the template folder.


## Mid-Run Serial Failure

If a long replay fails midway, the streamer still writes the collected response CSV and summary JSON. The summary includes an `error` object with the exception type, message, and completed row count. Preserve those partial files; they are useful for debugging before rerunning smoke or validation.
