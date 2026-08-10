"""Flash + smoke train-only Student B on Arduino R4 only; also fetch ESP32 B smoke."""
from __future__ import annotations

import os
from pathlib import Path

import paramiko

HOST = os.environ.get("CUKD_PI_HOST", "192.168.137.234")
PASSWORD = os.environ["CUKD_PI_PASSWORD"]
REMOTE = "/home/project/Desktop/CuKD-XAI"
LOCAL = Path(r"C:\N Drive\Research\Cukd-XAI\CuKD-XAI")
OUT_LOCAL = LOCAL / "results/hardware_hil/train_only_scaler_copy/compile_evidence"

PORT = "/dev/serial/by-id/usb-Arduino_UNO_WiFi_R4_CMSIS-DAP_48CA43BD5890-if01"
FQBN = "arduino:renesas_uno:unor4wifi"
SKETCH = f"{REMOTE}/deployment/hardware_hil/build/train_only_student_B_seed42_arduino_r4_copy"
VECTORS = (
    f"{REMOTE}/deployment/firmware_export/wsnds_rfkd_hil/"
    "generated_train_only_student_B_seed42_copy/hil_replay_vectors.csv"
)


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 600) -> str:
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
    if code != 0:
        raise RuntimeError(f"failed ({code})")
    return out


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
        # wait briefly if USB re-enumerating
        run(
            client,
            "for i in 1 2 3 4 5 6 7 8 9 10; do "
            "test -e /dev/ttyACM0 && break; sleep 1; done; "
            "ls -la /dev/ttyACM0 /dev/serial/by-id/ 2>&1",
            timeout=60,
        )
        run(
            client,
            "export PATH=$HOME/.local/bin:$PATH; "
            f"arduino-cli upload -p {PORT} --fqbn {FQBN} {SKETCH}",
            timeout=600,
        )
        smoke_dir = (
            f"{REMOTE}/results/hardware_hil/train_only_scaler_copy/"
            "compile_evidence/smoke_arduino_r4_student_B"
        )
        run(client, f"mkdir -p {smoke_dir}")
        run(
            client,
            f"cd {REMOTE} && . .venv-hil/bin/activate && "
            f"python -m deployment.hardware_hil.host.stream_vectors "
            f"--port {PORT} --vectors-csv {VECTORS} "
            f"--output-csv {smoke_dir}/smoke_10_mcu.csv "
            f"--summary-json {smoke_dir}/smoke_10_sequence.json "
            f"--baud 115200 --timeout 3 --settle-seconds 2 --limit 10",
            timeout=180,
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
    print("R4_B_SMOKE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
