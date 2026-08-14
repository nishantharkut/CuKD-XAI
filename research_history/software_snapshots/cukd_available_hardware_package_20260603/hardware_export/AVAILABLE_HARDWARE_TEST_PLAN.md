# Available-Hardware Test Plan

This plan is for the hardware you already have access to: Arduino-class boards, ESP32, and Raspberry Pi. It is designed to strengthen the deployment evidence without overclaiming that the model has already been integrated into a live WSN protocol stack.

## Priority

1. **ESP32 microcontroller proxy**
   - Best available practical target if WSN motes are not available.
   - Runs the same dependency-free fixed-point C inference core.
   - Use it to report on-device self-test pass/fail and latency from `micros()`.
   - This is a microcontroller proxy, not a TelosB deployment claim.

2. **Arduino-class board**
   - Useful only if the board has enough flash/RAM for the selected test-vector header.
   - Avoid full 56,200-vector headers on small Arduino boards.
   - Use 256 or 1,000 generated test vectors for the board demo.

3. **Raspberry Pi / Raspberry Pi AI Hat**
   - Use only as an edge-gateway comparison.
   - Do not use Raspberry Pi results to claim WSN mote feasibility; it is far more capable than a WSN mote.

4. **Actual WSN mote, if professor can arrange it**
   - Best target for the paper if available.
   - Keep the same C core and generated headers.
   - Report cross-compiled footprint, static RAM, stack, latency/cycles if measurable, and self-test agreement.

## What To Test

Use the already exported Student A RF-KD fixed-point model:

- `model_weights.h`
- `test_vectors.h`
- `wsnds_student_a_rfkd_int8_inference.c`

For hardware, generate a small test-vector set first:

```powershell
python .\hardware_export\run_wsnds_student_a_rfkd_e2e.py `
    --state-dict origin/main:Final/wsnds_deployment_qat_outputs/tmp/E_student_A_KD_from_RF_fp32.pt `
    --dataset-csv WSN-DS.csv `
    --output-dir hardware_export\generated_student_a_rfkd_hw_256 `
    --num-test-vectors 256
```

Use 1,000 vectors if the board has enough flash:

```powershell
python .\hardware_export\run_wsnds_student_a_rfkd_e2e.py `
    --state-dict origin/main:Final/wsnds_deployment_qat_outputs/tmp/E_student_A_KD_from_RF_fp32.pt `
    --dataset-csv WSN-DS.csv `
    --output-dir hardware_export\generated_student_a_rfkd_hw_1000 `
    --num-test-vectors 1000
```

Keep the full 56,200-vector result as host/MSP430-toolchain evidence. Do not force that full vector header onto small boards.

## What To Record

For ESP32/Arduino:

- board name and exact chip/module
- core clock if available
- compiler / Arduino core version
- number of test vectors
- `prediction_failures`
- `logit_failures`
- total elapsed microseconds
- average microseconds per vector
- whether the sketch uses only the fixed-point C inference core

For the paper, phrase it as:

> In the absence of a physical WSN mote, we validate the generated fixed-point inference core on an available microcontroller proxy. This demonstrates dependency-free integer execution of the compressed IDS core, while full packet-to-feature extraction and WSN stack integration remain future work.

## Claims Allowed

Allowed:

- fixed-point inference core runs on available MCU hardware
- generated C weights and test vectors compile into an Arduino/ESP32 sketch
- on-device predictions/logits match generated fixed-point references
- Raspberry Pi can be used as an edge-gateway runtime comparison

Not allowed unless actual WSN mote testing is done:

- deployed on TelosB
- integrated with TinyOS/Contiki
- live packet-level WSN intrusion detection
- end-to-end mote energy consumption

## How This Fits The Paper

This hardware work strengthens the systems evidence. It does not change the main model metrics:

- Student A RF-KD: 1,189 parameters
- fixed-point parameter payload: 1,348 bytes
- fixed-point activation estimate: 140 bytes
- inference arithmetic: 1,136 MACs
- preprocessing arithmetic: 68 integer operations after raw features exist

If WSN motes are unavailable, present the ESP32/Arduino result as a practical execution proof and keep the exact WSN mote deployment as a limitation/future work item.
