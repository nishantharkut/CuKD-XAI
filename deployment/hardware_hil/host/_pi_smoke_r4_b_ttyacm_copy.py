"""Retry R4 Student B upload via /dev/ttyACM0; fetch ESP32 B smoke either way."""
from __future__ import annotations

import os
from pathlib import Path

import paramiko

HOST = os.environ.get("CUKD_PI_HOST", "192.168.137.234")
PASSWORD = os.environ["CUKD_PI_PASSWORD"]
REMOTE = "/home/project/Desktop/CuKD-XAI"
LOCAL = Path(r"C:\N Drive\Research\Cukd-XAI\CuKD-XAI")
OUT_LOCAL = LOCAL / "results/hardware_hil/train_only_scaler_copy/compile_evidence"


def run(client, cmd, timeout=600, check=True):
    print(">>>", cmd[:220])
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    code = stdout.channel.recv_exit_status()
    if out:
        print(out[-2500:] if len(out) > 2500 else out)
    if err.strip():
        print("ERR", err[-1200:])
    print("exit", code)
    if check and code != 0:
        raise RuntimeError(f"failed ({code})")
    return code, out


def main() -> int:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        HOST,
        username="project",
        password=PASSWORD,
        timeout=30,
        allow_agent=False,
        look_for_keys=False,
    )
    try:
        run(
            client,
            "fuser -v /dev/ttyACM0 /dev/ttyUSB0 2>&1 || true; "
            "ls -la /dev/ttyACM* /dev/serial/by-id 2>&1",
            timeout=30,
            check=False,
        )
        sketch = f"{REMOTE}/deployment/hardware_hil/build/train_only_student_B_seed42_arduino_r4_copy"
        vectors = (
            f"{REMOTE}/deployment/firmware_export/wsnds_rfkd_hil/"
            "generated_train_only_student_B_seed42_copy/hil_replay_vectors.csv"
        )
        smoke_dir = (
            f"{REMOTE}/results/hardware_hil/train_only_scaler_copy/"
            "compile_evidence/smoke_arduino_r4_student_B"
        )
        # try ttyACM0 first, then by-id
        code, _ = run(
            client,
            "export PATH=$HOME/.local/bin:$PATH; "
            f"arduino-cli upload -p /dev/ttyACM0 --fqbn arduino:renesas_uno:unor4wifi {sketch}",
            timeout=600,
            check=False,
        )
        if code != 0:
            code, _ = run(
                client,
                "export PATH=$HOME/.local/bin:$PATH; "
                "arduino-cli board list; "
                f"arduino-cli upload -p /dev/serial/by-id/usb-Arduino_UNO_WiFi_R4_CMSIS-DAP_48CA43BD5890-if01 "
                f"--fqbn arduino:renesas_uno:unor4wifi {sketch}",
                timeout=600,
                check=False,
            )
        if code == 0:
            run(client, f"mkdir -p {smoke_dir}")
            run(
                client,
                f"cd {REMOTE} && . .venv-hil/bin/activate && "
                f"python -m deployment.hardware_hil.host.stream_vectors "
                f"--port /dev/ttyACM0 --vectors-csv {vectors} "
                f"--output-csv {smoke_dir}/smoke_10_mcu.csv "
                f"--summary-json {smoke_dir}/smoke_10_sequence.json "
                f"--baud 115200 --timeout 3 --settle-seconds 2 --limit 10",
                timeout=180,
            )
        else:
            print("R4 upload still failed; keeping prior full HIL B evidence")

        # identity probe either way
        run(
            client,
            f"cd {REMOTE} && . .venv-hil/bin/activate && python - <<'PY'\n"
            "import serial,time\n"
            "for p in ['/dev/ttyACM0','/dev/ttyUSB0']:\n"
            "  try:\n"
            "    ser=serial.Serial(p,115200,timeout=1.5); time.sleep(1.0); ser.reset_input_buffer()\n"
            "    ser.write(b'CUKDID?\\n'); ser.flush(); time.sleep(0.5); print(p, ser.read(400)); ser.close()\n"
            "  except Exception as e:\n"
            "    print(p, e)\n"
            "PY",
            timeout=60,
            check=False,
        )

        sftp = client.open_sftp()
        try:
            for key in ["smoke_arduino_r4_student_B", "smoke_esp32c3_student_B"]:
                rd = (
                    f"{REMOTE}/results/hardware_hil/train_only_scaler_copy/"
                    f"compile_evidence/{key}"
                )
                ld = OUT_LOCAL / key
                ld.mkdir(parents=True, exist_ok=True)
                for name in ["smoke_10_mcu.csv", "smoke_10_sequence.json"]:
                    try:
                        sftp.get(f"{rd}/{name}", str(ld / name))
                        print("fetched", key, name)
                    except OSError as exc:
                        print("missing", key, name, exc)
        finally:
            sftp.close()
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
