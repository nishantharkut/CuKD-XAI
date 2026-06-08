# Build Firmware Bundles

The firmware bundle helper creates Arduino-IDE-ready sketch folders. This avoids manual copying mistakes.

## ESP32-C3 Bundle

```bash
python -m hardware_hil.host.prepare_firmware_bundle \
  --board esp32c3 \
  --generated-dir hardware_export/generated_student_a_rfkd_hil_full \
  --output-dir hardware_hil/build/cukd_hil_esp32c3
```

Open this folder in Arduino IDE:

```text
hardware_hil/build/cukd_hil_esp32c3/cukd_hil_esp32c3.ino
```

## Arduino R4 Bundle

```bash
python -m hardware_hil.host.prepare_firmware_bundle \
  --board arduino_r4 \
  --generated-dir hardware_export/generated_student_a_rfkd_hil_full \
  --output-dir hardware_hil/build/cukd_hil_arduino_r4
```

Open this folder in Arduino IDE:

```text
hardware_hil/build/cukd_hil_arduino_r4/cukd_hil_arduino_r4.ino
```

## Bundle Contents

Each bundle contains:

- board sketch `.ino`,
- `model_weights.h`,
- `preprocess_int_metadata.h`,
- `cukd_model.c/.h`,
- `cukd_preprocess.c/.h`,
- `cukd_protocol.c/.h`,
- `bundle_manifest.json`.

Do not edit generated headers by hand. Regenerate the bundle if the model/export changes.
