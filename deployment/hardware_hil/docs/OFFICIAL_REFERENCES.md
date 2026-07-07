# Official Hardware References Used

These links are the hardware/software documentation sources used for the HIL instructions. Project-specific protocol and model details are defined inside this repository.

## Raspberry Pi 5 Host

- Raspberry Pi documentation: https://www.raspberrypi.com/documentation/
- Raspberry Pi computer documentation: https://www.raspberrypi.com/documentation/computers/raspberry-pi.html

Use in this project: host controller, serial logging, report generation. The Raspberry Pi does not provide the MCU result.

## ESP32-C3

- ESP32-C3-DevKitM-1 user guide: https://docs.espressif.com/projects/esp-dev-kits/en/latest/esp32c3/esp32-c3-devkitm-1/index.html
- Arduino-ESP32 documentation: https://docs.espressif.com/projects/arduino-esp32/en/latest/
- ESP32-C3-DevKitM-1 Arduino-ESP32 board page: https://docs.espressif.com/projects/arduino-esp32/en/latest/boards/ESP32-C3-DevKitM-1.html

Use in this project: primary USB-serial MCU replay target using Arduino-compatible firmware.

## Arduino UNO R4

- UNO R4 Minima docs: https://docs.arduino.cc/hardware/uno-r4-minima/
- UNO R4 WiFi docs: https://docs.arduino.cc/hardware/uno-r4-wifi/
- Arduino IDE 2 docs: https://docs.arduino.cc/software/ide-v2/

Use in this project: secondary portability target. If using UNO R4 WiFi, the claim remains about the Renesas RA4M1 application MCU only.

## Claim Boundary

The official docs describe hardware setup, board selection, flashing, and serial usage. They do not validate CuKD-XAI accuracy. Accuracy, fixed-point agreement, and latency evidence must come from this repository's generated artifacts and hardware logs.

## Connection Facts Used In The Beginner Guide

- Raspberry Pi documentation lists Raspberry Pi 5 with `5 V at 5 A` and the 27 W USB-C power supply recommendation. It also instructs users to connect power after boot media and peripherals are prepared.
- Espressif's ESP32-C3-DevKitM-1 user guide lists the Micro-USB port as the default and recommended power-supply option for that board family.
- Arduino's UNO R4 Minima hardware page identifies the board as Renesas RA4M1-based and states that the board uses USB-C.
- Arduino-ESP32 documentation explains installing ESP32 support through Arduino IDE Boards Manager and selecting the board/COM port before upload.

Project-specific decision: even though these boards expose GPIO pins, CuKD-XAI HIL uses USB serial only. No GPIO wiring is required for this validation.

