# Flash Arduino R4

The Arduino R4 is the secondary portability target.

Official docs used:

- UNO R4 Minima docs: https://docs.arduino.cc/hardware/uno-r4-minima/
- UNO R4 WiFi docs: https://docs.arduino.cc/hardware/uno-r4-wifi/
- Arduino IDE documentation: https://docs.arduino.cc/software/ide-v2/

## Board Variant Rule

Record exactly which board you have:

- UNO R4 Minima: report it as Renesas RA4M1 MCU execution.
- UNO R4 WiFi: still report only the Renesas RA4M1 application MCU for this experiment; do not claim ESP32-S3 acceleration or networking.

The firmware does not use Wi-Fi.

## Arduino IDE Setup

1. Install Arduino IDE 2.x.
2. Install/select the official Arduino UNO R4 board package.
3. Connect the board by USB.
4. Select the correct UNO R4 variant and serial port.
5. Open:

```text
deployment/hardware_hil/build/cukd_hil_arduino_r4/cukd_hil_arduino_r4.ino
```

Upload it.

## Before Streaming

Close Arduino Serial Monitor before running host replay. Keep baud at `115200`.

