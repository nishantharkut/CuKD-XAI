# ESP32-C3 HIL Firmware

This firmware is the primary real-MCU execution target for CuKD-XAI Student A.

## Build Inputs

The include path must contain:

- `model_weights.h`
- `preprocess_int_metadata.h`
- `hardware_hil/firmware/common/*.h`

Compile these common sources with the sketch:

- `cukd_model.c`
- `cukd_preprocess.c`
- `cukd_protocol.c`

## Runtime Scope

Wi-Fi, Bluetooth, filesystems, and cloud services are intentionally unused.
The board receives one 17-feature fixed-point row, validates CRC, runs integer
preprocessing, runs fixed-point inference, and returns one response line.


