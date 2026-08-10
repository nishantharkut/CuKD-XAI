"""Fetch ESP32 A results; wait for B full stream; score both; write summaries."""
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


def connect(retries: int = 20) -> paramiko.SSHClient:
    last = None
    for i in range(retries):
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(
                HOST,
                username=USER,
                password=PASSWORD,
                timeout=25,
                allow_agent=False,
                look_for_keys=False,
                banner_timeout=30,
            )
            print(f"SSH ok try {i+1} host={HOST}")
            return c
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"SSH fail try {i+1}: {exc}")
            time.sleep(3)
    raise RuntimeError(last)


def run(c: paramiko.SSHClient, cmd: str, timeout: int = 60) -> str:
    _i, o, e = c.exec_command(cmd, timeout=timeout)
    out = o.read().decode(errors="replace")
    err = e.read().decode(errors="replace")
    code = o.channel.recv_exit_status()
    if out:
        print(out[-3000:] if len(out) > 3000 else out)
    if err.strip():
        print("ERR", err[:1000])
    if code != 0:
        print("exit", code)
    return out


def fetch_student(c: paramiko.SSHClient, letter: str) -> dict:
    out_rel = f"results/hardware_hil/train_only_scaler_copy/pi5_esp32c3_student_{letter}"
    gen = f"deployment/firmware_export/wsnds_rfkd_hil/generated_train_only_student_{letter}_seed42_copy"
    local_out = LOCAL / out_rel
    local_out.mkdir(parents=True, exist_ok=True)
    sftp = c.open_sftp()
    for name in [
        "smoke_10_mcu.csv",
        "smoke_10_sequence.json",
        "validation_1000_mcu.csv",
        "validation_1000_sequence.json",
        "full_56200_mcu.csv",
        "full_56200_sequence.json",
    ]:
        try:
            sftp.get(f"{REMOTE}/{out_rel}/{name}", str(local_out / name))
            print("fetched", letter, name)
        except OSError as exc:
            print("missing", letter, name, exc)
    sftp.close()

    ref = pd.read_csv(LOCAL / gen / "hil_reference_predictions.csv")
    mcu = pd.read_csv(local_out / "full_56200_mcu.csv")
    agree = float((mcu["predicted_class"].to_numpy() == ref["fixed_pred"].to_numpy()).mean())
    report = {
        "board": "esp32c3",
        "student": letter,
        "protocol": "train_only_seed42_copy_pipeline",
        "port": PORT,
        "n": int(len(mcu)),
        "all_status_ok": bool((mcu["status"] == "OK").all()),
        "mcu_vs_fixed_reference_agreement": agree,
        "accuracy": float(accuracy_score(ref["true_label"], mcu["predicted_class"])),
        "macro_f1": float(f1_score(ref["true_label"], mcu["predicted_class"], average="macro")),
        "latency_us_mean": float(mcu["total_us"].mean()),
        "latency_us_p50": float(mcu["total_us"].median()),
    }
    (local_out / "full_56200_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if report["n"] != 56200 or not report["all_status_ok"] or agree != 1.0:
        raise RuntimeError(f"quality fail {letter}: {report}")
    return report


def main() -> int:
    # Fetch A now (already complete on Pi)
    c = connect()
    try:
        a_report = fetch_student(c, "A")
    finally:
        c.close()

    # Wait for B full
    deadline = time.time() + 2400
    while time.time() < deadline:
        c = connect()
        try:
            out = run(
                c,
                "ps aux | grep stream_vectors | grep -v grep || echo NO_STREAM; "
                "B=$HOME/Desktop/CuKD-XAI/results/hardware_hil/train_only_scaler_copy/pi5_esp32c3_student_B; "
                "echo -n LINES=; if test -f $B/full_56200_mcu.csv; then wc -l < $B/full_56200_mcu.csv; else echo 0; fi; "
                "if test -f $B/full_56200_sequence.json; then echo SEQ=YES; else echo SEQ=NO; fi",
            )
            lines = 0
            for line in out.splitlines():
                if line.startswith("LINES="):
                    try:
                        lines = int(line.split("=", 1)[1].strip())
                    except ValueError:
                        lines = 0
            streaming = "NO_STREAM" not in out and "stream_vectors" in out
            if lines >= 56201 and "SEQ=YES" in out and not streaming:
                b_report = fetch_student(c, "B")
                esp = {"protocol": "train_only_seed42_copy_pipeline", "board": "esp32c3", "pairs": [a_report, b_report]}
                (LOCAL / "results/hardware_hil/train_only_scaler_copy/esp32c3_student_A_B_summary.json").write_text(
                    json.dumps(esp, indent=2) + "\n", encoding="utf-8"
                )
                r4p = LOCAL / "results/hardware_hil/train_only_scaler_copy/arduino_r4_student_A_B_summary.json"
                combined = {
                    "protocol": "train_only_seed42_copy_pipeline",
                    "arduino_r4": json.loads(r4p.read_text(encoding="utf-8")) if r4p.is_file() else None,
                    "esp32c3": esp,
                }
                (LOCAL / "results/hardware_hil/train_only_scaler_copy/four_pair_summary.json").write_text(
                    json.dumps(combined, indent=2) + "\n", encoding="utf-8"
                )
                print("FOUR_PAIR_COMPLETE")
                return 0
            # If no stream and incomplete, start nohup B full
            if "NO_STREAM" in out and lines < 56201:
                print("Starting nohup B full stream")
                PORT = (
                    "/dev/serial/by-id/"
                    "usb-Silicon_Labs_CP2102N_USB_to_UART_Bridge_Controller_"
                    "f8df315e2a6def11a9bec5a7c169b110-if00-port0"
                )
                GEN = "deployment/firmware_export/wsnds_rfkd_hil/generated_train_only_student_B_seed42_copy"
                OUT = "results/hardware_hil/train_only_scaler_copy/pi5_esp32c3_student_B"
                # verify identity
                run_cmd = (
                    f"cd {REMOTE} && . .venv-hil/bin/activate && python - <<'PY'\n"
                    "import serial, time\n"
                    f"ser=serial.Serial({PORT!r}, 115200, timeout=1)\n"
                    "time.sleep(1.5); ser.reset_input_buffer()\n"
                    "ser.write(b'CUKDID?\\n'); ser.flush(); time.sleep(0.5)\n"
                    "print(ser.read(1024)); ser.close()\n"
                    "PY"
                )
                _i, o, e = c.exec_command(run_cmd, timeout=30)
                print(o.read().decode(errors="replace"))
                id_out = o.read().decode(errors="replace") if False else ""
                # re-read properly
                _i, o, e = c.exec_command(run_cmd, timeout=30)
                id_out = o.read().decode(errors="replace")
                print(id_out)
                if "student_B" not in id_out:
                    # reflash B
                    sketch = f"$HOME/Desktop/CuKD-XAI/deployment/hardware_hil/build/train_only_student_B_seed42_esp32c3_copy"
                    for cmd in [
                        f"export PATH=$HOME/.local/bin:$PATH; arduino-cli compile --fqbn esp32:esp32:esp32c3 {sketch}",
                        f"export PATH=$HOME/.local/bin:$PATH; arduino-cli upload -p {PORT} --fqbn esp32:esp32:esp32c3 {sketch}",
                    ]:
                        _i, o, e = c.exec_command(cmd, timeout=1800)
                        print(o.read().decode(errors="replace")[-2000:])
                        print(e.read().decode(errors="replace")[-1000:])
                        if o.channel.recv_exit_status() != 0 and e.channel.recv_exit_status() != 0:
                            pass
                # start nohup
                start = (
                    f"cd {REMOTE} && . .venv-hil/bin/activate && mkdir -p {OUT} && "
                    f"nohup python -m deployment.hardware_hil.host.stream_vectors "
                    f"--port {PORT} "
                    f"--vectors-csv {GEN}/hil_replay_vectors.csv "
                    f"--output-csv {OUT}/full_56200_mcu.csv "
                    f"--summary-json {OUT}/full_56200_sequence.json "
                    f"--baud 115200 --timeout 3 --settle-seconds 2 "
                    f"> {OUT}/full_56200_nohup.log 2>&1 & echo STARTED_$!"
                )
                _i, o, e = c.exec_command(start, timeout=30)
                print(o.read().decode(errors="replace"))
        finally:
            c.close()
        time.sleep(30)

    raise TimeoutError("B full did not finish")


if __name__ == "__main__":
    raise SystemExit(main())
