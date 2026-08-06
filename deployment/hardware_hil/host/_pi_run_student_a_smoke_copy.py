"""Upload stream helper if needed and run Student A R4 smoke-10 (copy pipeline)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import paramiko

HOST = os.environ.get("CUKD_PI_HOST", "10.94.138.123")
USER = os.environ.get("CUKD_PI_USER", "project")
PASSWORD = os.environ.get("CUKD_PI_PASSWORD", "")
if not PASSWORD:
    raise SystemExit("Set CUKD_PI_PASSWORD")

LOCAL = Path(r"C:\N Drive\Research\Cukd-XAI\CuKD-XAI")
REMOTE = "/home/project/Desktop/CuKD-XAI"


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 7200) -> int:
    print(">>>", cmd[:220])
    _i, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    code = stdout.channel.recv_exit_status()
    if out:
        print(out[-6000:] if len(out) > 6000 else out)
    if err.strip():
        print("ERR", err[-2500:])
    print("exit", code)
    return code


def main() -> int:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        HOST, username=USER, password=PASSWORD, timeout=30, allow_agent=False, look_for_keys=False
    )
    try:
        sftp = client.open_sftp()
        for rel in [
            "deployment/hardware_hil/host/stream_vectors_strict_copy.py",
            "deployment/hardware_hil/host/stream_vectors_strict.py",
        ]:
            lp = LOCAL / rel
            if lp.is_file():
                sftp.put(str(lp), f"{REMOTE}/{rel}")
                print("uploaded", rel)
        sftp.close()

        gen = (
            "deployment/firmware_export/wsnds_rfkd_hil/"
            "generated_train_only_student_A_seed42_copy"
        )
        out = "results/hardware_hil/train_only_scaler_copy/pi5_arduino_r4_student_A"
        run(client, f"mkdir -p {REMOTE}/{out}")

        # Smoke 10
        code = run(
            client,
            f"cd {REMOTE} && . .venv-hil/bin/activate && "
            f"python -m deployment.hardware_hil.host.stream_vectors "
            f"--port /dev/ttyACM0 "
            f"--vectors-csv {gen}/hil_replay_vectors.csv "
            f"--output-csv {out}/smoke_10_mcu.csv "
            f"--summary-json {out}/smoke_10_sequence.json "
            f"--limit 10 --baud 115200 --timeout 3",
            timeout=120,
        )
        if code != 0:
            return code

        # Verify smoke vs reference (first 10)
        code = run(
            client,
            f"cd {REMOTE} && . .venv-hil/bin/activate && python - <<'PY'\n"
            "import pandas as pd, json\n"
            f"mcu=pd.read_csv('{out}/smoke_10_mcu.csv')\n"
            f"ref=pd.read_csv('{gen}/hil_reference_predictions.csv').head(10)\n"
            "print('mcu cols', list(mcu.columns))\n"
            "print(mcu.head())\n"
            "print('seq', open('" + out + "/smoke_10_sequence.json').read()[:800])\n"
            "if 'predicted_class' in mcu.columns and 'fixed_pred' in ref.columns:\n"
            "    agree=(mcu['predicted_class'].to_numpy()==ref['fixed_pred'].to_numpy()).mean()\n"
            "    print('smoke_vs_fixed_agree', float(agree))\n"
            "PY",
            timeout=60,
        )
        return code
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
