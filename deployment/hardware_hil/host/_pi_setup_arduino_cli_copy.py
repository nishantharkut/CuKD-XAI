"""One-shot helper (copy/new file): install arduino-cli on Pi and flash R4 Student A.

Does not modify existing project sources.
"""
from __future__ import annotations

import os
import sys

import paramiko

HOST = os.environ.get("CUKD_PI_HOST", "10.94.138.123")
USER = os.environ.get("CUKD_PI_USER", "project")
PASSWORD = os.environ.get("CUKD_PI_PASSWORD", "")
if not PASSWORD:
    raise SystemExit("Set CUKD_PI_PASSWORD in the environment (do not hardcode).")


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 900) -> tuple[int, str, str]:
    print(">>>", cmd[:200])
    _stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    code = stdout.channel.recv_exit_status()
    if out:
        print(out[-4000:] if len(out) > 4000 else out)
    if err.strip():
        print("ERR", err[-2000:])
    print("exit", code)
    return code, out, err


def main() -> int:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        HOST,
        username=USER,
        password=PASSWORD,
        timeout=30,
        allow_agent=False,
        look_for_keys=False,
    )
    try:
        run(client, "mkdir -p $HOME/.local/bin")
        run(
            client,
            "curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh "
            "| BINDIR=$HOME/.local/bin sh",
        )
        run(client, "export PATH=$HOME/.local/bin:$PATH; arduino-cli version")
        run(
            client,
            "export PATH=$HOME/.local/bin:$PATH; "
            "arduino-cli config init || true; arduino-cli core update-index",
        )
        run(
            client,
            "export PATH=$HOME/.local/bin:$PATH; arduino-cli core install arduino:renesas_uno",
            timeout=1200,
        )
        # Compile Student A train-only copy sketch
        sketch = (
            "$HOME/Desktop/CuKD-XAI/deployment/hardware_hil/build/"
            "train_only_student_A_seed42_arduino_r4_copy"
        )
        run(
            client,
            "export PATH=$HOME/.local/bin:$PATH; "
            f"arduino-cli compile --fqbn arduino:renesas_uno:unor4wifi {sketch}",
            timeout=1200,
        )
        run(client, "ls -la /dev/ttyACM* 2>&1")
        run(
            client,
            "export PATH=$HOME/.local/bin:$PATH; "
            f"arduino-cli upload -p /dev/ttyACM0 --fqbn arduino:renesas_uno:unor4wifi {sketch}",
            timeout=600,
        )
        # Identity probe after flash
        run(
            client,
            "cd $HOME/Desktop/CuKD-XAI && . .venv-hil/bin/activate && python - <<'PY'\n"
            "import serial, time\n"
            "ser=serial.Serial('/dev/ttyACM0', 115200, timeout=1)\n"
            "time.sleep(1.5)\n"
            "ser.reset_input_buffer()\n"
            "ser.write(b'CUKDID?\\n'); ser.flush(); time.sleep(0.5)\n"
            "print(ser.read(1024))\n"
            "ser.close()\n"
            "PY",
        )
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
