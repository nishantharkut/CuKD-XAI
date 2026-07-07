# Raspberry Pi 5 Host Setup

The Raspberry Pi 5 is only the host/logger. It does not run the IDS model for the main hardware result.

## 1. Prepare The Pi

Use the official Raspberry Pi power supply or an equivalent 5 V / 5 A USB-C supply. Keep the active cooler attached for long full-split serial runs.

Official docs:

- Raspberry Pi documentation: https://www.raspberrypi.com/documentation/
- Raspberry Pi 5 product documentation: https://www.raspberrypi.com/documentation/computers/raspberry-pi.html

## 2. Copy Or Clone The Repo

On the Pi, enter the repository root:

```bash
cd CuKD-XAI
```

## 3. Create A Small Host Python Environment

```bash
python3 -m venv .venv-hil
source .venv-hil/bin/activate
python -m pip install --upgrade pip
python -m pip install -r hardware_hil/host/requirements.txt
```

The host scripts only need `pyserial` plus the Python standard library.

## 4. Record Host Environment

Run this before hardware replay:

```bash
python -m hardware_hil.host.env_check \
  --output results/hardware_hil/board_replay/pi5_environment.json
```

This records Python version, platform, and available serial ports.

## 5. Find The Serial Port

After plugging in a board, rerun:

```bash
python -m hardware_hil.host.env_check \
  --output results/hardware_hil/board_replay/pi5_environment_after_board.json
```

Common Linux port names are:

- `/dev/ttyACM0`
- `/dev/ttyUSB0`

Use the port that appears after connecting the board.

## 6. Physical Connection Rule

For the HIL experiment, connect the ESP32-C3 or Arduino R4 to the Raspberry Pi 5 using the board USB cable only. Do not use Pi GPIO pins, UART pins, 5V pins, 3V3 pins, or GND jumpers. Follow `deployment/hardware_hil/docs/08_PHYSICAL_CONNECTIONS_AND_CABLES.md` before flashing or replaying vectors.

