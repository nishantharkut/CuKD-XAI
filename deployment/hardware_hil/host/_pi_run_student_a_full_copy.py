"""Run Student A R4 validation-1000 and full-56200 HIL (copy pipeline)."""
from __future__ import annotations

import os
from pathlib import Path

import paramiko
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

HOST = os.environ.get("CUKD_PI_HOST", "10.94.138.123")
USER = os.environ.get("CUKD_PI_USER", "project")
PASSWORD = os.environ.get("CUKD_PI_PASSWORD", "")
if not PASSWORD:
    raise SystemExit("Set CUKD_PI_PASSWORD")

REMOTE = "/home/project/Desktop/CuKD-XAI"
GEN = "deployment/firmware_export/wsnds_rfkd_hil/generated_train_only_student_A_seed42_copy"
OUT = "results/hardware_hil/train_only_scaler_copy/pi5_arduino_r4_student_A"
LOCAL = Path(r"C:\N Drive\Research\Cukd-XAI\CuKD-XAI")


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 7200) -> int:
    print(">>>", cmd[:240])
    _i, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    code = stdout.channel.recv_exit_status()
    if out:
        print(out[-4000:] if len(out) > 4000 else out)
    if err.strip():
        print("ERR", err[-2000:])
    print("exit", code)
    return code


def stream(client: paramiko.SSHClient, limit: int | None, tag: str) -> int:
    limit_arg = f"--limit {limit}" if limit is not None else ""
    return run(
        client,
        f"cd {REMOTE} && . .venv-hil/bin/activate && "
        f"python -m deployment.hardware_hil.host.stream_vectors "
        f"--port /dev/ttyACM0 "
        f"--vectors-csv {GEN}/hil_replay_vectors.csv "
        f"--output-csv {OUT}/{tag}_mcu.csv "
        f"--summary-json {OUT}/{tag}_sequence.json "
        f"{limit_arg} --baud 115200 --timeout 3 --settle-seconds 2",
        timeout=7200 if limit is None else 1800,
    )


def main() -> int:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        HOST, username=USER, password=PASSWORD, timeout=30, allow_agent=False, look_for_keys=False
    )
    try:
        run(client, f"mkdir -p {REMOTE}/{OUT}")
        if stream(client, 1000, "validation_1000") != 0:
            return 1
        if stream(client, None, "full_56200") != 0:
            return 1

        # Pull results back for local verification
        sftp = client.open_sftp()
        local_out = LOCAL / OUT
        local_out.mkdir(parents=True, exist_ok=True)
        for name in [
            "smoke_10_mcu.csv",
            "smoke_10_sequence.json",
            "validation_1000_mcu.csv",
            "validation_1000_sequence.json",
            "full_56200_mcu.csv",
            "full_56200_sequence.json",
        ]:
            remote = f"{REMOTE}/{OUT}/{name}"
            try:
                sftp.get(remote, str(local_out / name))
                print("fetched", name)
            except OSError as exc:
                print("missing", name, exc)
        sftp.close()
    finally:
        client.close()

    # Local agreement metrics vs fixed reference
    ref = pd.read_csv(LOCAL / GEN / "hil_reference_predictions.csv")
    mcu = pd.read_csv(local_out / "full_56200_mcu.csv")
    assert len(mcu) == 56200
    assert (mcu["status"] == "OK").all()
    agree = float((mcu["predicted_class"].to_numpy() == ref["fixed_pred"].to_numpy()).mean())
    acc = float(accuracy_score(ref["true_label"], mcu["predicted_class"]))
    macro = float(f1_score(ref["true_label"], mcu["predicted_class"], average="macro"))
    report = {
        "board": "arduino_r4",
        "student": "A",
        "protocol": "train_only_seed42_copy_pipeline",
        "n": 56200,
        "all_status_ok": True,
        "mcu_vs_fixed_reference_agreement": agree,
        "accuracy": acc,
        "macro_f1": macro,
        "latency_us_mean": float(mcu["total_us"].mean()),
        "latency_us_p50": float(mcu["total_us"].median()),
    }
    import json

    (local_out / "full_56200_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0 if agree == 1.0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
