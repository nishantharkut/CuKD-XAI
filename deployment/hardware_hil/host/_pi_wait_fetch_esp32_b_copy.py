"""Wait for ESP32 Student B full stream on Pi, then fetch + score.

Does not kill an in-progress stream. Only starts a new nohup job if none is
running and the full CSV is missing. Survives SSH drops on the poll loop.
Copy-only helper; does not modify original project sources.
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
POLL_SEC = 30
DEADLINE_SEC = 3600


def connect(retries: int = 20, delay: float = 4.0) -> paramiko.SSHClient:
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
            return client
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"SSH try {i+1} failed: {exc}")
            time.sleep(delay)
    raise RuntimeError(f"SSH failed: {last}")


def exec_out(client: paramiko.SSHClient, cmd: str, timeout: int = 90) -> str:
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    code = stdout.channel.recv_exit_status()
    if err.strip():
        print("ERR", err[-800:])
    print(out[-2500:] if len(out) > 2500 else out)
    print("exit", code)
    return out


def status(client: paramiko.SSHClient) -> dict:
    cmd = f"""
ps aux | grep 'stream_vectors.*student_B\\|/pi5_esp32c3_student_B/full' | grep -v grep || echo NO_STREAM
echo -n LINES=
test -f {REMOTE}/{OUT}/full_56200_mcu.csv && wc -l < {REMOTE}/{OUT}/full_56200_mcu.csv || echo 0
echo -n SEQ=
test -f {REMOTE}/{OUT}/full_56200_sequence.json && echo YES || echo NO
echo -n BYTES=
test -f {REMOTE}/{OUT}/full_56200_mcu.csv && wc -c < {REMOTE}/{OUT}/full_56200_mcu.csv || echo 0
tail -n 5 {REMOTE}/{OUT}/full_56200_nohup.log 2>/dev/null || true
"""
    out = exec_out(client, cmd, timeout=60)
    lines = 0
    for line in out.splitlines():
        if line.startswith("LINES="):
            try:
                lines = int(line.split("=", 1)[1].strip())
            except ValueError:
                lines = 0
    return {
        "raw": out,
        "streaming": "NO_STREAM" not in out,
        "lines": lines,
        "seq": "SEQ=YES" in out,
        "done": lines >= 56201 and "SEQ=YES" in out,
    }


def start_if_needed(client: paramiko.SSHClient) -> None:
    st = status(client)
    if st["done"]:
        print("Already complete on Pi")
        return
    if st["streaming"]:
        print("Stream already running; will poll only")
        return

    print("Starting detached nohup full stream")
    log = f"{REMOTE}/{OUT}/full_56200_nohup.log"
    # setsid + stdin/out/err redirect so SSH channel can close immediately
    start_cmd = (
        f"mkdir -p {REMOTE}/{OUT} && "
        f"cd {REMOTE} && . .venv-hil/bin/activate && "
        f"setsid nohup python -m deployment.hardware_hil.host.stream_vectors "
        f"--port {PORT} "
        f"--vectors-csv {GEN}/hil_replay_vectors.csv "
        f"--output-csv {OUT}/full_56200_mcu.csv "
        f"--summary-json {OUT}/full_56200_sequence.json "
        f"--baud 115200 --timeout 3 --settle-seconds 2 "
        f"> {log} 2>&1 < /dev/null & echo STARTED_PID=$!"
    )
    # Use a short-lived channel; do not hang on job control
    transport = client.get_transport()
    assert transport is not None
    chan = transport.open_session()
    chan.settimeout(30)
    chan.exec_command(start_cmd)
    # Read briefly then close; process is detached via setsid/nohup
    time.sleep(2)
    try:
        data = chan.recv(4096).decode(errors="replace")
        print(data)
    except Exception as exc:  # noqa: BLE001
        print("start channel read:", exc)
    try:
        chan.close()
    except Exception:
        pass
    time.sleep(2)
    status(client)


def fetch_and_score(client: paramiko.SSHClient) -> int:
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
        "full_56200_nohup.log",
    ]:
        try:
            sftp.get(f"{REMOTE}/{OUT}/{name}", str(local_out / name))
            print("fetched", name)
        except OSError as exc:
            print("optional missing", name, exc)
    sftp.close()

    ref = pd.read_csv(LOCAL / GEN / "hil_reference_predictions.csv")
    mcu = pd.read_csv(local_out / "full_56200_mcu.csv")
    agree = float((mcu["predicted_class"].to_numpy() == ref["fixed_pred"].to_numpy()).mean())
    report = {
        "board": "esp32c3",
        "student": "B",
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

    a_metrics = LOCAL / (
        "results/hardware_hil/train_only_scaler_copy/pi5_esp32c3_student_A/full_56200_metrics.json"
    )
    esp_summary = {
        "protocol": "train_only_seed42_copy_pipeline",
        "board": "esp32c3",
        "pairs": [
            json.loads(a_metrics.read_text(encoding="utf-8")) if a_metrics.is_file() else None,
            report,
        ],
    }
    (
        LOCAL / "results/hardware_hil/train_only_scaler_copy/esp32c3_student_A_B_summary.json"
    ).write_text(json.dumps(esp_summary, indent=2) + "\n", encoding="utf-8")

    r4 = LOCAL / "results/hardware_hil/train_only_scaler_copy/arduino_r4_student_A_B_summary.json"
    combined = {
        "protocol": "train_only_seed42_copy_pipeline",
        "arduino_r4": json.loads(r4.read_text(encoding="utf-8")) if r4.is_file() else None,
        "esp32c3": esp_summary,
    }
    (
        LOCAL / "results/hardware_hil/train_only_scaler_copy/four_pair_summary.json"
    ).write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    print("FOUR_PAIR_COMPLETE")
    ok = agree == 1.0 and report["all_status_ok"] and report["n"] == 56200
    return 0 if ok else 2


def main() -> int:
    client = connect()
    try:
        start_if_needed(client)
    finally:
        client.close()

    deadline = time.time() + DEADLINE_SEC
    while time.time() < deadline:
        client = connect()
        try:
            st = status(client)
            print(
                f"progress streaming={st['streaming']} lines={st['lines']} seq={st['seq']}"
            )
            if st["done"] and not st["streaming"]:
                return fetch_and_score(client)
            if st["done"] and st["streaming"]:
                # rare race: files present while process exiting
                time.sleep(5)
                st2 = status(client)
                if st2["done"]:
                    return fetch_and_score(client)
        finally:
            client.close()
        time.sleep(POLL_SEC)

    raise TimeoutError("ESP32 B full stream did not complete in time")


if __name__ == "__main__":
    raise SystemExit(main())
