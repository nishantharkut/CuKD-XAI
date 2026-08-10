"""Wait for in-flight Student A full stream; if parent died, finish A metrics + run B."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import paramiko
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

HOST = os.environ.get("CUKD_PI_HOST", "10.94.138.123")
USER = os.environ.get("CUKD_PI_USER", "project")
PASSWORD = os.environ.get("CUKD_PI_PASSWORD", "")
REMOTE = "/home/project/Desktop/CuKD-XAI"
LOCAL = Path(r"C:\N Drive\Research\Cukd-XAI\CuKD-XAI")
PORT = (
    "/dev/serial/by-id/"
    "usb-Silicon_Labs_CP2102N_USB_to_UART_Bridge_Controller_"
    "f8df315e2a6def11a9bec5a7c169b110-if00-port0"
)
FQBN = "esp32:esp32:esp32c3"


def run(client, cmd, timeout=7200):
    print(">>>", cmd[:220])
    _i, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    code = stdout.channel.recv_exit_status()
    if out:
        print(out[-5000:] if len(out) > 5000 else out)
    if err.strip():
        print("ERR", err[-2000:])
    print("exit", code)
    return code, out


def wait_for_a_full(client, max_wait_s=2400):
    out_csv = (
        f"{REMOTE}/results/hardware_hil/train_only_scaler_copy/"
        "pi5_esp32c3_student_A/full_56200_mcu.csv"
    )
    start = time.time()
    while time.time() - start < max_wait_s:
        code, out = run(
            client,
            "ps aux | grep stream_vectors | grep -v grep; "
            f"test -f {out_csv} && wc -l {out_csv} || echo 'NO_FULL_CSV_YET'",
            timeout=30,
        )
        if "NO_FULL_CSV_YET" not in out and "full_56200_mcu.csv" in out:
            # check line count
            for line in out.splitlines():
                if "full_56200_mcu.csv" in line and line.strip()[0].isdigit():
                    n = int(line.strip().split()[0])
                    print("full csv lines", n)
                    if n >= 56201:
                        return True
        # still streaming?
        if "stream_vectors" not in out and "NO_FULL_CSV_YET" in out:
            print("stream died without full csv — will restart full for A")
            return False
        time.sleep(30)
    raise TimeoutError("timeout waiting for Student A full stream")


def stream(client, letter, limit, tag):
    gen = f"deployment/firmware_export/wsnds_rfkd_hil/generated_train_only_student_{letter}_seed42_copy"
    out = f"results/hardware_hil/train_only_scaler_copy/pi5_esp32c3_student_{letter}"
    limit_arg = f"--limit {limit}" if limit is not None else ""
    return run(
        client,
        f"cd {REMOTE} && . .venv-hil/bin/activate && "
        f"python -m deployment.hardware_hil.host.stream_vectors "
        f"--port {PORT} "
        f"--vectors-csv {gen}/hil_replay_vectors.csv "
        f"--output-csv {out}/{tag}_mcu.csv "
        f"--summary-json {out}/{tag}_sequence.json "
        f"{limit_arg} --baud 115200 --timeout 3 --settle-seconds 2",
        timeout=7200 if limit is None else 1800,
    )[0]


def fetch_and_score(client, letter):
    gen = f"deployment/firmware_export/wsnds_rfkd_hil/generated_train_only_student_{letter}_seed42_copy"
    out = f"results/hardware_hil/train_only_scaler_copy/pi5_esp32c3_student_{letter}"
    local_out = LOCAL / out
    local_out.mkdir(parents=True, exist_ok=True)
    sftp = client.open_sftp()
    for name in [
        "smoke_10_mcu.csv",
        "smoke_10_sequence.json",
        "validation_1000_mcu.csv",
        "validation_1000_sequence.json",
        "full_56200_mcu.csv",
        "full_56200_sequence.json",
    ]:
        try:
            sftp.get(f"{REMOTE}/{out}/{name}", str(local_out / name))
            print("fetched", letter, name)
        except OSError as exc:
            print("skip", name, exc)
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
    if agree != 1.0 or not report["all_status_ok"] or report["n"] != 56200:
        raise RuntimeError(f"quality fail {letter}: {report}")
    return report


def run_b(client):
    letter = "B"
    gen = f"deployment/firmware_export/wsnds_rfkd_hil/generated_train_only_student_{letter}_seed42_copy"
    bundle = f"deployment/hardware_hil/build/train_only_student_{letter}_seed42_esp32c3_copy"
    out = f"results/hardware_hil/train_only_scaler_copy/pi5_esp32c3_student_{letter}"
    sketch = f"$HOME/Desktop/CuKD-XAI/{bundle}"
    run(client, f"mkdir -p {REMOTE}/{out}")
    code, _ = run(
        client,
        f"export PATH=$HOME/.local/bin:$PATH; "
        f"arduino-cli compile --fqbn {FQBN} {sketch}",
        timeout=1800,
    )
    if code:
        raise RuntimeError("B compile failed")
    code, _ = run(
        client,
        f"export PATH=$HOME/.local/bin:$PATH; "
        f"arduino-cli upload -p {PORT} --fqbn {FQBN} {sketch}",
        timeout=900,
    )
    if code:
        raise RuntimeError("B upload failed")
    run(
        client,
        f"cd {REMOTE} && . .venv-hil/bin/activate && python - <<'PY'\n"
        "import serial, time\n"
        f"ser=serial.Serial({PORT!r}, 115200, timeout=1)\n"
        "time.sleep(2); ser.reset_input_buffer()\n"
        "ser.write(b'CUKDID?\\n'); ser.flush(); time.sleep(0.6)\n"
        "print(ser.read(1024)); ser.close()\n"
        "PY",
    )
    for limit, tag in [(10, "smoke_10"), (1000, "validation_1000"), (None, "full_56200")]:
        if stream(client, letter, limit, tag):
            raise RuntimeError(f"B stream {tag} failed")
    return fetch_and_score(client, "B")


def main():
    if not PASSWORD:
        raise SystemExit("Set CUKD_PI_PASSWORD")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        HOST, username=USER, password=PASSWORD, timeout=30, allow_agent=False, look_for_keys=False
    )
    reports = []
    try:
        ok = wait_for_a_full(client)
        if not ok:
            # restart only full for A (smoke/1k already exist)
            run(
                client,
                f"mkdir -p {REMOTE}/results/hardware_hil/train_only_scaler_copy/pi5_esp32c3_student_A",
            )
            if stream(client, "A", None, "full_56200"):
                raise RuntimeError("A full restart failed")
        reports.append(fetch_and_score(client, "A"))
        reports.append(run_b(client))
        summary = {
            "protocol": "train_only_seed42_copy_pipeline",
            "board": "esp32c3",
            "port": PORT,
            "pairs": reports,
        }
        path = LOCAL / "results/hardware_hil/train_only_scaler_copy/esp32c3_student_A_B_summary.json"
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        # combined four-pair summary
        r4_path = LOCAL / "results/hardware_hil/train_only_scaler_copy/arduino_r4_student_A_B_summary.json"
        combined = {
            "protocol": "train_only_seed42_copy_pipeline",
            "arduino_r4": json.loads(r4_path.read_text(encoding="utf-8")) if r4_path.is_file() else None,
            "esp32c3": summary,
        }
        (LOCAL / "results/hardware_hil/train_only_scaler_copy/four_pair_summary.json").write_text(
            json.dumps(combined, indent=2) + "\n", encoding="utf-8"
        )
        print("COMPLETE four-pair summary written")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
