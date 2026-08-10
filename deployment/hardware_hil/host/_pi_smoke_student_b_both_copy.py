"""Flash train-only Student B on ESP32-C3 and Arduino R4, run smoke-10, fetch."""
from __future__ import annotations

import os
from pathlib import Path

import paramiko

HOST = os.environ.get("CUKD_PI_HOST", "192.168.137.234")
PASSWORD = os.environ["CUKD_PI_PASSWORD"]
REMOTE = "/home/project/Desktop/CuKD-XAI"
LOCAL = Path(r"C:\N Drive\Research\Cukd-XAI\CuKD-XAI")
OUT_LOCAL = LOCAL / "results/hardware_hil/train_only_scaler_copy/compile_evidence"

ITEMS = [
    (
        "esp32c3_student_B",
        "esp32:esp32:esp32c3",
        "/dev/serial/by-id/usb-Silicon_Labs_CP2102N_USB_to_UART_Bridge_Controller_"
        "f8df315e2a6def11a9bec5a7c169b110-if00-port0",
        f"{REMOTE}/deployment/hardware_hil/build/train_only_student_B_seed42_esp32c3_copy",
        f"{REMOTE}/deployment/firmware_export/wsnds_rfkd_hil/"
        "generated_train_only_student_B_seed42_copy/hil_replay_vectors.csv",
    ),
    (
        "arduino_r4_student_B",
        "arduino:renesas_uno:unor4wifi",
        "/dev/serial/by-id/usb-Arduino_UNO_WiFi_R4_CMSIS-DAP_48CA43BD5890-if01",
        f"{REMOTE}/deployment/hardware_hil/build/train_only_student_B_seed42_arduino_r4_copy",
        f"{REMOTE}/deployment/firmware_export/wsnds_rfkd_hil/"
        "generated_train_only_student_B_seed42_copy/hil_replay_vectors.csv",
    ),
]


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 600) -> str:
    print(">>>", cmd[:220])
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    code = stdout.channel.recv_exit_status()
    if out:
        print(out[-2500:] if len(out) > 2500 else out)
    if err.strip():
        print("ERR", err[-1000:])
    print("exit", code)
    if code != 0:
        raise RuntimeError(f"command failed ({code}): {cmd[:120]}")
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
        for key, fqbn, port, sketch, vectors in ITEMS:
            run(
                client,
                "export PATH=$HOME/.local/bin:$PATH; "
                f"arduino-cli upload -p {port} --fqbn {fqbn} {sketch}",
                timeout=600,
            )
            smoke_dir = (
                f"{REMOTE}/results/hardware_hil/train_only_scaler_copy/"
                f"compile_evidence/smoke_{key}"
            )
            run(client, f"mkdir -p {smoke_dir}")
            run(
                client,
                f"cd {REMOTE} && . .venv-hil/bin/activate && "
                f"python -m deployment.hardware_hil.host.stream_vectors "
                f"--port {port} --vectors-csv {vectors} "
                f"--output-csv {smoke_dir}/smoke_10_mcu.csv "
                f"--summary-json {smoke_dir}/smoke_10_sequence.json "
                f"--baud 115200 --timeout 3 --settle-seconds 2 --limit 10",
                timeout=180,
            )

        run(
            client,
            f"cd {REMOTE} && . .venv-hil/bin/activate && python - <<'PY'\n"
            "import serial, time\n"
            "ports=[\n"
            "'/dev/serial/by-id/usb-Arduino_UNO_WiFi_R4_CMSIS-DAP_48CA43BD5890-if01',\n"
            "'/dev/serial/by-id/usb-Silicon_Labs_CP2102N_USB_to_UART_Bridge_Controller_"
            "f8df315e2a6def11a9bec5a7c169b110-if00-port0',\n"
            "]\n"
            "for p in ports:\n"
            "  ser=serial.Serial(p,115200,timeout=1.5)\n"
            "  time.sleep(1.2); ser.reset_input_buffer()\n"
            "  ser.write(b'CUKDID?\\n'); ser.flush(); time.sleep(0.5)\n"
            "  print(p.split('/')[-1], ser.read(400)); ser.close()\n"
            "PY",
            timeout=60,
        )

        sftp = client.open_sftp()
        try:
            for key, *_ in ITEMS:
                rd = (
                    f"{REMOTE}/results/hardware_hil/train_only_scaler_copy/"
                    f"compile_evidence/smoke_{key}"
                )
                ld = OUT_LOCAL / f"smoke_{key}"
                ld.mkdir(parents=True, exist_ok=True)
                for name in ["smoke_10_mcu.csv", "smoke_10_sequence.json"]:
                    sftp.get(f"{rd}/{name}", str(ld / name))
                    print("fetched", key, name)
        finally:
            sftp.close()
    finally:
        client.close()
    print("SMOKE_B_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
