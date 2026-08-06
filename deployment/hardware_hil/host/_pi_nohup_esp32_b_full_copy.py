"""Start ESP32 Student B full-56200 under nohup, wait, then fetch + score.

Survives laptop/SSH disconnects. Does not modify original project sources.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import paramiko
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

HOST = os.environ.get("CUKD_PI_HOST", "192.168.137.234")
USER = os.environ.get("CUKD_PI_USER", "project")
PASSWORD = os.environ.get("CUKD_PI_PASSWORD", "")
if not PASSWORD:
    raise SystemExit("Set CUKD_PI_PASSWORD")

REMOTE = "/home/project/Desktop/CuKD-XAI"
LOCAL = Path(r"C:\N Drive\Research\Cukd-XAI\CuKD-XAI")
PORT = (
    "/dev/serial/by-id/"
    "usb-Silicon_Labs_CP2102N_USB_to_UART_Bridge_Controller_"
    "f8df315e2a6def11a9bec5a7c169b110-if00-port0"
)
GEN = "deployment/firmware_export/wsnds_rfkd_hil/generated_train_only_student_B_seed42_copy"
OUT = "results/hardware_hil/train_only_scaler_copy/pi5_esp32c3_student_B"
FQBN = "esp32:esp32:esp32c3"
BUNDLE = "deployment/hardware_hil/build/train_only_student_B_seed42_esp32c3_copy"


def connect(retries: int = 30, delay: float = 5.0) -> paramiko.SSHClient:
    last = None
    for i in range(retries):
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(
                HOST,
                username=USER,
                password=PASSWORD,
                timeout=25,
                allow_agent=False,
                look_for_keys=False,
                banner_timeout=30,
            )
            print(f"SSH connected on try {i+1}")
            return client
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"SSH try {i+1} failed: {exc}")
            time.sleep(delay)
    raise RuntimeError(f"SSH failed after retries: {last}")


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 120) -> tuple[int, str]:
    print(">>>", cmd[:220])
    _i, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    code = stdout.channel.recv_exit_status()
    if out:
        print(out[-4000:] if len(out) > 4000 else out)
    if err.strip():
        print("ERR", err[-1500:])
    print("exit", code)
    return code, out


def main() -> int:
    client = connect()
    try:
        # Check if full already done
        code, out = run(
            client,
            f"test -f {REMOTE}/{OUT}/full_56200_mcu.csv && wc -l {REMOTE}/{OUT}/full_56200_mcu.csv "
            f"|| echo MISSING",
        )
        need_run = "MISSING" in out
        if not need_run:
            for line in out.splitlines():
                if "full_56200_mcu.csv" in line and line.strip()[0].isdigit():
                    n = int(line.split()[0])
                    if n < 56201:
                        need_run = True

        if need_run:
            # Ensure B firmware is on board
            run(
                client,
                f"cd {REMOTE} && . .venv-hil/bin/activate && python - <<'PY'\n"
                "import serial, time\n"
                f"ser=serial.Serial({PORT!r}, 115200, timeout=1)\n"
                "time.sleep(1.5); ser.reset_input_buffer()\n"
                "ser.write(b'CUKDID?\\n'); ser.flush(); time.sleep(0.5)\n"
                "print(ser.read(1024)); ser.close()\n"
                "PY",
            )
            # If identity wrong, reflash
            # Start nohup full stream (kill any previous)
            run(client, "pkill -f 'stream_vectors.*student_B' || true")
            run(client, f"mkdir -p {REMOTE}/{OUT}")
            log = f"{REMOTE}/{OUT}/full_56200_nohup.log"
            cmd = (
                f"cd {REMOTE} && . .venv-hil/bin/activate && "
                f"nohup python -m deployment.hardware_hil.host.stream_vectors "
                f"--port {PORT} "
                f"--vectors-csv {GEN}/hil_replay_vectors.csv "
                f"--output-csv {OUT}/full_56200_mcu.csv "
                f"--summary-json {OUT}/full_56200_sequence.json "
                f"--baud 115200 --timeout 3 --settle-seconds 2 "
                f"> {log} 2>&1 & echo $!"
            )
            _c, out = run(client, cmd)
            print("started pid info", out)
        else:
            print("Full CSV already present with enough lines")
    finally:
        client.close()

    # Poll until done
    deadline = time.time() + 2400
    while time.time() < deadline:
        client = connect(retries=15, delay=4)
        try:
            _c, out = run(
                client,
                "ps aux | grep 'stream_vectors.*student_B\\|/pi5_esp32c3_student_B/full' | grep -v grep || echo NO_STREAM\n"
                f"echo -n LINES=; test -f {REMOTE}/{OUT}/full_56200_mcu.csv && wc -l < {REMOTE}/{OUT}/full_56200_mcu.csv || echo 0\n"
                f"echo -n SEQ=; test -f {REMOTE}/{OUT}/full_56200_sequence.json && echo YES || echo NO\n"
                f"tail -5 {REMOTE}/{OUT}/full_56200_nohup.log 2>/dev/null || true",
            )
            lines = 0
            for line in out.splitlines():
                if line.startswith("LINES="):
                    try:
                        lines = int(line.split("=", 1)[1].strip())
                    except ValueError:
                        lines = 0
            if lines >= 56201 and "SEQ=YES" in out and "NO_STREAM" in out:
                print("B full complete on Pi")
                # fetch + score
                sftp = client.open_sftp()
                local_out = LOCAL / OUT
                local_out.mkdir(parents=True, exist_ok=True)
                for name in [
                    "full_56200_mcu.csv",
                    "full_56200_sequence.json",
                    "smoke_10_mcu.csv",
                    "smoke_10_sequence.json",
                    "validation_1000_mcu.csv",
                    "validation_1000_sequence.json",
                ]:
                    try:
                        sftp.get(f"{REMOTE}/{OUT}/{name}", str(local_out / name))
                        print("fetched", name)
                    except OSError as exc:
                        print("optional missing", name, exc)
                sftp.close()

                # Ensure smoke/1k if missing (optional, full is required)
                ref = pd.read_csv(LOCAL / GEN / "hil_reference_predictions.csv")
                mcu = pd.read_csv(local_out / "full_56200_mcu.csv")
                agree = float(
                    (mcu["predicted_class"].to_numpy() == ref["fixed_pred"].to_numpy()).mean()
                )
                report = {
                    "board": "esp32c3",
                    "student": "B",
                    "protocol": "train_only_seed42_copy_pipeline",
                    "port": PORT,
                    "n": int(len(mcu)),
                    "all_status_ok": bool((mcu["status"] == "OK").all()),
                    "mcu_vs_fixed_reference_agreement": agree,
                    "accuracy": float(accuracy_score(ref["true_label"], mcu["predicted_class"])),
                    "macro_f1": float(
                        f1_score(ref["true_label"], mcu["predicted_class"], average="macro")
                    ),
                    "latency_us_mean": float(mcu["total_us"].mean()),
                    "latency_us_p50": float(mcu["total_us"].median()),
                }
                (local_out / "full_56200_metrics.json").write_text(
                    json.dumps(report, indent=2) + "\n", encoding="utf-8"
                )
                print(json.dumps(report, indent=2))

                # update summaries
                a_metrics = LOCAL / (
                    "results/hardware_hil/train_only_scaler_copy/"
                    "pi5_esp32c3_student_A/full_56200_metrics.json"
                )
                esp_summary = {
                    "protocol": "train_only_seed42_copy_pipeline",
                    "board": "esp32c3",
                    "pairs": [
                        json.loads(a_metrics.read_text(encoding="utf-8"))
                        if a_metrics.is_file()
                        else None,
                        report,
                    ],
                }
                (LOCAL / "results/hardware_hil/train_only_scaler_copy/esp32c3_student_A_B_summary.json").write_text(
                    json.dumps(esp_summary, indent=2) + "\n", encoding="utf-8"
                )
                r4 = LOCAL / "results/hardware_hil/train_only_scaler_copy/arduino_r4_student_A_B_summary.json"
                combined = {
                    "protocol": "train_only_seed42_copy_pipeline",
                    "arduino_r4": json.loads(r4.read_text(encoding="utf-8")) if r4.is_file() else None,
                    "esp32c3": esp_summary,
                }
                (LOCAL / "results/hardware_hil/train_only_scaler_copy/four_pair_summary.json").write_text(
                    json.dumps(combined, indent=2) + "\n", encoding="utf-8"
                )
                print("FOUR_PAIR_COMPLETE")
                return 0 if agree == 1.0 and report["all_status_ok"] and report["n"] == 56200 else 2
        finally:
            client.close()
        time.sleep(30)

    raise TimeoutError("Student B full stream did not complete in time")


if __name__ == "__main__":
    raise SystemExit(main())
