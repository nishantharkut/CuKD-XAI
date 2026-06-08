# Arduino R4 HIL Firmware

This firmware is the secondary MCU portability target. Record the exact board
variant as UNO R4 Minima or UNO R4 WiFi. If the WiFi variant is used, the
hardware claim must refer only to the Renesas RA4M1 application MCU.

## Build Inputs

The include path must contain:

- `model_weights.h`
- `preprocess_int_metadata.h`
- `hardware_hil/firmware/common/*.h`

Compile these common sources with the sketch:

- `cukd_model.c`
- `cukd_preprocess.c`
- `cukd_protocol.c`

