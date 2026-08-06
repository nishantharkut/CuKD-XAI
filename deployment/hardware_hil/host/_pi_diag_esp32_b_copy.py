"""Diagnose stuck ESP32 B stream on Pi. Copy-only helper."""
from __future__ import annotations

import os
import sys

import paramiko

HOST = os.environ.get("CUKD_PI_HOST", "192.168.137.234")
USER = os.environ.get("CUKD_PI_USER", "project")
PASSWORD = os.environ.get("CUKD_PI_PASSWORD", "")
if not PASSWORD:
    raise SystemExit("Set CUKD_PI_PASSWORD")

CMD = r"""
set -x
PID=$(pgrep -f 'stream_vectors.*student_B' | head -n1)
echo PID=$PID
if [ -n "$PID" ]; then
  echo ---cwd---
  readlink -f /proc/$PID/cwd
  echo ---cmdline---
  tr '\0' ' ' < /proc/$PID/cmdline; echo
  echo ---fds---
  ls -la /proc/$PID/fd 2>&1 | head -n 40
  echo ---environ snippet---
  tr '\0' '\n' < /proc/$PID/environ | grep -E 'PWD|HOME|PATH' | head
fi
echo ---find csv---
find $HOME/Desktop/CuKD-XAI -name 'full_56200_mcu.csv' 2>/dev/null
echo ---lsof port---
lsof /dev/ttyUSB0 2>&1 || fuser /dev/ttyUSB0 2>&1 || true
echo ---nohup logs---
ls -la $HOME/Desktop/CuKD-XAI/results/hardware_hil/train_only_scaler_copy/pi5_esp32c3_student_B/ 2>&1
ls -la $HOME/nohup* $HOME/Desktop/CuKD-XAI/*nohup* $HOME/Desktop/CuKD-XAI/results/hardware_hil/train_only_scaler_copy/*nohup* 2>&1 | head
find $HOME/Desktop/CuKD-XAI/results -name '*esp32*b*' -o -name '*nohup*' 2>/dev/null | head -n 30
echo ---recent logs---
find $HOME/Desktop/CuKD-XAI/results/hardware_hil -type f -mmin -30 2>/dev/null | head -n 40
"""


def main() -> int:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        HOST,
        username=USER,
        password=PASSWORD,
        timeout=25,
        allow_agent=False,
        look_for_keys=False,
    )
    _, stdout, stderr = client.exec_command(CMD, timeout=90)
    print(stdout.read().decode(errors="replace"))
    err = stderr.read().decode(errors="replace")
    if err.strip():
        print("STDERR:", err[-3000:])
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
