# Flash ESP32-C3

The ESP32-C3 is the primary real MCU target.

Official docs used:

- ESP32-C3-DevKitM-1 guide: https://docs.espressif.com/projects/esp-dev-kits/en/latest/esp32c3/esp32-c3-devkitm-1/index.html
- Arduino-ESP32 documentation: https://docs.espressif.com/projects/arduino-esp32/en/latest/
- Arduino-ESP32 ESP32-C3 board docs: https://docs.espressif.com/projects/arduino-esp32/en/latest/boards/ESP32-C3-DevKitM-1.html

## Arduino IDE Setup

1. Install Arduino IDE 2.x.
2. Install the Espressif `esp32` board package through Boards Manager, following the Arduino-ESP32 official documentation.
3. Connect the ESP32-C3 board by USB.
4. Select the ESP32-C3 DevKitM board profile if available. If the exact profile is not listed, use the matching ESP32-C3 development module profile provided by the installed Espressif core.
5. Use baud `115200` for serial monitor and host streaming.

## Open The Bundle

Open:

```text
deployment/hardware_hil/build/cukd_hil_esp32c3/cukd_hil_esp32c3.ino
```

Upload it to the board.

## Serial Check

After upload, close Arduino Serial Monitor before running the Python streamer. Only one program can own the serial port at a time.

If the serial port does not appear, unplug/replug the board, check the USB data cable, and rerun:

```bash
python -m hardware_hil.host.env_check --output results/hardware_hil/board_replay/pi5_environment_esp32c3.json
```

