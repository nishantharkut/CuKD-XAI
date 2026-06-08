# Optional Breadboard And Jumper Guide

This guide is for learning and safe hardware handling. It is not required for the CuKD-XAI paper-critical HIL run.

## 1. Most Important Rule

For the actual CuKD-XAI HIL experiment, use USB serial only:

```text
Raspberry Pi 5 USB port  <---- USB cable ---->  ESP32-C3 or Arduino R4
```

Do not connect Raspberry Pi GPIO pins to the MCU board during the main HIL run.

Why:

- Raspberry Pi GPIO inputs are 3.3 V tolerant according to Raspberry Pi documentation.
- Arduino UNO R4 Minima is a 5 V board according to the Arduino datasheet.
- Direct Arduino R4 TX/output into Raspberry Pi GPIO can damage the Pi.
- USB serial already provides power, data, and ground reference safely through the board's USB interface.

## 2. What The Breadboard Is Useful For

Use the breadboard only for optional learning checks, such as:

- blinking an external LED,
- checking that jumper wires and resistors are working,
- learning pin naming,
- adding a non-paper status LED later.

Do not use the breadboard for the first HIL smoke replay.

## 3. Parts Needed For Optional LED Test

- Breadboard.
- Jumper wires.
- LED.
- 220 ohm or 330 ohm resistor.
- One MCU board: either Arduino R4 or ESP32-C3.

Do not connect both MCU boards to the same breadboard for the first test.

## 4. LED Polarity

An LED has two legs:

- Long leg: anode, positive side.
- Short leg: cathode, ground side.

Basic connection:

```text
GPIO pin ---- resistor ---- LED long leg
LED short leg ---- GND
```

The resistor is required. Do not connect LED directly to a GPIO pin.

## 5. Optional Arduino R4 LED Wiring

Arduino UNO R4 Minima is a 5 V board. Its datasheet says the board operates at 5 V and its pins are 5 V except the 3.3 V pin.

For a safe external LED test:

```text
Arduino R4 D8  ---- 220R/330R resistor ---- LED long leg
LED short leg  ---- Arduino R4 GND
```

Steps:

1. Keep Arduino connected by USB.
2. Put the LED across two different breadboard rows.
3. Connect D8 to one side of the resistor.
4. Connect the resistor's other side to the LED long leg row.
5. Connect LED short leg row to Arduino GND.
6. Upload a simple Blink-style sketch using D8.

This is only a board-learning test. It does not need to be included in the paper.

## 6. Optional ESP32-C3 LED Wiring

ESP32-C3 boards use 3.3 V GPIO logic. Espressif's ESP32-C3-DevKitM-1 documentation says GPIO pins are broken out to headers for interfacing, and the Micro-USB port powers and communicates with the board.

Use a general-purpose GPIO pin from the exact board pinout. If your board has a pin labeled `GPIO4` / `IO4`, it is usually a reasonable beginner LED pin. If your board silkscreen differs, send a photo before wiring.

```text
ESP32-C3 GPIO4/IO4 ---- 220R/330R resistor ---- LED long leg
LED short leg          ---- ESP32-C3 GND
```

Steps:

1. Keep ESP32-C3 connected by USB.
2. Put the LED across two different breadboard rows.
3. Connect GPIO4/IO4 to one side of the resistor.
4. Connect the resistor's other side to the LED long leg row.
5. Connect LED short leg row to ESP32-C3 GND.
6. Upload a simple Blink-style sketch using GPIO4.

Avoid pins labeled BOOT, EN, RST, 5V, 3V3, GND, TX, RX for beginner LED wiring unless the exact board documentation says to use them.

## 7. Do Not Do This

Do not make these connections for CuKD-XAI HIL:

```text
Arduino R4 TX  ---- Raspberry Pi RX      # unsafe without level shifting
Arduino R4 D-pin ---- Raspberry Pi GPIO  # unsafe without level shifting
ESP32 5V       ---- Raspberry Pi 5V       # unnecessary power backfeed risk
ESP32 3V3      ---- Raspberry Pi 3V3      # unnecessary power rail coupling
Any board VIN  ---- Raspberry Pi GPIO     # wrong and unsafe
```

## 8. If You Really Want Pi GPIO UART Later

This is not recommended for the paper-critical run. Use USB serial instead.

If a later debugging experiment requires direct UART:

### ESP32-C3 To Raspberry Pi UART

Only possible if both sides use 3.3 V logic and the board is still powered safely.

Minimum wiring would be:

```text
Pi GND   ---- ESP32-C3 GND
Pi TXD   ---- ESP32-C3 RX GPIO
Pi RXD   ---- ESP32-C3 TX GPIO
```

But you must first confirm the exact ESP32-C3 board UART pins from the board documentation. Also, Raspberry Pi serial console settings must be changed in Raspberry Pi OS. This is extra risk and not needed.

### Arduino R4 To Raspberry Pi UART

Do not connect directly. Arduino R4 is 5 V logic and Raspberry Pi GPIO is 3.3 V tolerant. You would need a proper bidirectional logic level shifter or at least correct level shifting for Arduino TX to Pi RX.

This is unnecessary for CuKD-XAI because USB serial already works.

## 9. Paper Position

Breadboard/jumper tests can be mentioned only as setup practice or optional debugging. The paper's hardware evidence should remain:

- ESP32-C3 USB serial replay,
- Arduino R4 USB serial replay,
- no live WSN feature extraction,
- no GPIO wiring claim,
- no energy claim.
