# Physical Connections And Cables

This experiment uses USB serial only. There is no breadboard wiring and no GPIO jumper wiring for the ESP32-C3 or Arduino R4 HIL runs.

## 1. Safety Rule

Do not connect the Raspberry Pi 5 GPIO header to the ESP32-C3 or Arduino R4 for this experiment.

Do not connect:

- Pi `5V` pin to board `5V`,
- Pi `3V3` pin to board `3V3`,
- Pi `GND` pin to board `GND`,
- Pi UART TX/RX pins to board TX/RX pins.

The board is powered and controlled through its USB data cable. This keeps the paper claim clean and avoids beginner wiring mistakes.

## 2. Desk Layout

Use this physical layout:

```text
Wall outlet
   |
   | Official Raspberry Pi 27 W USB-C power supply
   v
Raspberry Pi 5  <---- USB data cable ---->  one MCU board at a time
(host/logger)                              (ESP32-C3 or Arduino R4)
```

Keep only one MCU board connected during the first smoke tests. This makes the serial port easy to identify.

## 3. Raspberry Pi 5 Connections

Connect these first:

1. Active cooler installed on Raspberry Pi 5.
2. microSD / boot storage inserted.
3. Keyboard, mouse, and monitor if using desktop mode, or Ethernet/Wi-Fi/SSH if using headless mode.
4. Official 27 W USB-C power supply into the Pi 5 power port.

Raspberry Pi's official documentation recommends the official supply; for Raspberry Pi 5 the table lists `5 V at 5 A` and the 27 W USB-C supply. The official setup guide also says to connect power after the boot media and peripherals are ready.

## 4. ESP32-C3 Connection

### Cable

For an Espressif ESP32-C3-DevKitM-1-style board, the official user guide lists the Micro-USB port as the default and recommended power-supply option.

Use:

- USB-A to Micro-USB data cable for ESP32-C3-DevKitM-1 / M1-style boards, if your board has Micro-USB.
- USB-A to USB-C data cable if your particular ESP32-C3 Mini board has USB-C.

The cable must be a data cable, not a charge-only cable.

### Physical Steps

1. Keep the Pi powered on.
2. Plug the USB-A end into a Raspberry Pi USB-A port.
3. Plug the Micro-USB or USB-C end into the ESP32-C3 board's USB connector.
4. The board power LED should turn on.
5. Run the environment check:

```bash
python -m hardware_hil.host.env_check \
  --output results/hardware_hil/board_replay/pi5_environment_esp32c3_connected.json
```

Expected serial device names on Linux commonly look like:

- `/dev/ttyACM0`
- `/dev/ttyUSB0`

Use the port that appears after plugging in the board.

### If Upload Fails

Some ESP32-C3 boards may need bootloader mode during upload. Espressif's Arduino-ESP32 install guide notes that some boards may require holding the BOOT button while uploading.

Use this sequence only if normal upload fails:

1. Hold `BOOT`.
2. Press and release `RESET` if the board has a reset button.
3. Start upload.
4. Release `BOOT` after upload starts.

Do not use this during normal serial replay unless the board is stuck.

## 5. Arduino R4 Connection

### Cable

The Arduino UNO R4 Minima hardware page states that UNO R4 Minima uses a USB-C connector. The getting-started documentation also expects the board to be connected through USB-C.

Use:

- USB-A to USB-C data cable from Raspberry Pi 5 to Arduino R4.

The cable must support data.

### Physical Steps

1. Keep the Pi powered on.
2. Plug USB-A into the Raspberry Pi 5.
3. Plug USB-C into the Arduino R4.
4. The board power LED should turn on.
5. Run:

```bash
python -m hardware_hil.host.env_check \
  --output results/hardware_hil/board_replay/pi5_environment_arduino_r4_connected.json
```

Record whether the board is UNO R4 Minima or UNO R4 WiFi.

If it is UNO R4 WiFi, the paper claim must still refer only to the Renesas RA4M1 application MCU. This HIL firmware does not use the Wi-Fi module.

## 6. Flashing Computer Choices

You have two valid workflows.

### Option A: Flash From Your Laptop, Replay From Pi

Use this if Arduino IDE is easier on your Windows laptop.

1. Connect board to laptop.
2. Upload the generated HIL sketch from Arduino IDE.
3. Close Arduino IDE Serial Monitor.
4. Unplug board from laptop.
5. Plug board into Raspberry Pi 5.
6. Run the Pi replay commands.

### Option B: Flash And Replay From Raspberry Pi

Use this if Arduino IDE or Arduino CLI is installed on the Pi.

1. Connect board to Pi.
2. Upload the generated HIL sketch.
3. Close Serial Monitor.
4. Run the Pi replay commands.

Do not keep Serial Monitor open while the Python streamer is running. Only one program can use the serial port at a time.

## 7. What To Photograph For The Professor

Take clear photos of:

1. Raspberry Pi 5 powered by official USB-C supply.
2. ESP32-C3 connected to Pi by USB cable only.
3. Arduino R4 connected to Pi by USB cable only.
4. Terminal showing the smoke run command and output files.
5. Terminal showing verification metrics JSON path.

These photos support the experimental setup. They are not a substitute for the CSV/JSON logs.

## 8. Common Beginner Mistakes

- Using a charge-only USB cable.
- Keeping Arduino Serial Monitor open during Python replay.
- Connecting two MCU boards at once and using the wrong serial port.
- Opening the template sketch instead of the generated bundle sketch.
- Editing `model_weights.h` manually.
- Connecting Pi GPIO pins to MCU pins even though USB is enough.

## 9. Minimum First-Day Goal

Do not start with the full 56,200-row run.

Your first hardware goal is only:

1. Flash ESP32-C3.
2. Identify its serial port.
3. Run 10-row smoke replay.
4. Verify zero non-OK statuses.
5. Save the smoke CSV and JSON.


## 10. About Jumper Wires And Breadboards

You can use jumper wires and a breadboard for optional beginner LED tests, but not for the first CuKD-XAI HIL run. See `deployment/hardware_hil/docs/09_OPTIONAL_BREADBOARD_AND_JUMPER_GUIDE.md`.

The main experiment remains USB serial only because it avoids voltage-level mistakes and gives the cleanest reviewer-safe evidence.

