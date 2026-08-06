"""Poll ESP32 Student B full stream progress on Pi. Copy-only helper."""
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
ps aux | grep stream_vectors | grep -v grep || echo NO_STREAM
B=$HOME/Desktop/CuKD-XAI/results/hardware_hil/train_only_scaler_copy/pi5_esp32c3_student_B
ls -la "$B"/full_56200* 2>&1 || true
if test -f "$B/full_56200_mcu.csv"; then
  echo -n B_FULL_LINES=
  wc -l < "$B/full_56200_mcu.csv"
  echo -n B_FULL_BYTES=
  wc -c < "$B/full_56200_mcu.csv"
  tail -n 1 "$B/full_56200_mcu.csv"
else
  echo NO_FULL_CSV_YET
fi
if test -f "$B/full_56200_sequence.json"; then
  echo B_SEQ=YES
  cat "$B/full_56200_sequence.json"
else
  echo B_SEQ=NO
fi
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
    _, stdout, stderr = client.exec_command(CMD, timeout=60)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    print(out)
    if err.strip():
        print(err, file=sys.stderr)
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
