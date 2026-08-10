"""Upload ESP32-C3 train-only copy bundles, install esp32 core if needed, flash A then B, full HIL."""
from __future__ import annotations

import json
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
LOCAL = Path(r"C:\N Drive\Research\Cukd-XAI\CuKD-XAI")
FQBN = "esp32:esp32:esp32c3"


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 7200) -> tuple[int, str]:
    print(">>>", cmd[:240])
    _i, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    code = stdout.channel.recv_exit_status()
    if out:
        print(out[-5000:] if len(out) > 5000 else out)
    if err.strip():
        print("ERR", err[-2500:])
    print("exit", code)
    return code, out


def ensure_dir(sftp: paramiko.SFTPClient, remote: str) -> None:
    parts = remote.strip("/").split("/")
    cur = ""
    for part in parts:
        cur += "/" + part
        try:
            sftp.stat(cur)
        except FileNotFoundError:
            sftp.mkdir(cur)


def upload_tree(sftp: paramiko.SFTPClient, local_dir: Path, remote_dir: str) -> None:
    ensure_dir(sftp, remote_dir)
    for path in local_dir.iterdir():
        if path.is_file():
            sftp.put(str(path), f"{remote_dir}/{path.name}")
            print("put", path.name)


def detect_esp_port(client: paramiko.SSHClient) -> str:
    _code, out = run(
        client,
        "ls -la /dev/ttyACM* /dev/ttyUSB* /dev/serial/by-id 2>&1; dmesg | tail -30",
    )
    # Prefer by-id naming for ESP
    _c2, byid = run(client, "ls -1 /dev/serial/by-id 2>/dev/null || true")
    ports = []
    for line in byid.splitlines():
        line = line.strip()
        if not line:
            continue
        lower = line.lower()
        if "arduino" in lower or "uno" in lower:
            continue
        if "esp" in lower or "silicon" in lower or "cp210" in lower or "ch340" in lower or "usbserial" in lower:
            ports.append(f"/dev/serial/by-id/{line}")
    if ports:
        return ports[0]
    # Fallback: ttyUSB0 preferred for ESP; if only one ACM and not responding as Arduino identity later
    _c3, ls = run(client, "ls -1 /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || true")
    candidates = [p.strip() for p in ls.splitlines() if p.strip().startswith("/dev/")]
    if not candidates:
        raise RuntimeError("No serial ports found for ESP32")
    # Prefer ttyUSB*
    for p in candidates:
        if "ttyUSB" in p:
            return p
    return candidates[0]


def stream(
    client: paramiko.SSHClient, port: str, gen: str, out: str, limit: int | None, tag: str
) -> int:
    limit_arg = f"--limit {limit}" if limit is not None else ""
    code, _ = run(
        client,
        f"cd {REMOTE} && . .venv-hil/bin/activate && "
        f"python -m deployment.hardware_hil.host.stream_vectors "
        f"--port {port} "
        f"--vectors-csv {gen}/hil_replay_vectors.csv "
        f"--output-csv {out}/{tag}_mcu.csv "
        f"--summary-json {out}/{tag}_sequence.json "
        f"{limit_arg} --baud 115200 --timeout 3 --settle-seconds 2",
        timeout=7200 if limit is None else 1800,
    )
    return code


def run_student(client: paramiko.SSHClient, letter: str, port: str) -> dict:
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
        raise RuntimeError(f"compile failed student {letter}")
    code, _ = run(
        client,
        f"export PATH=$HOME/.local/bin:$PATH; "
        f"arduino-cli upload -p {port} --fqbn {FQBN} {sketch}",
        timeout=900,
    )
    if code:
        raise RuntimeError(f"upload failed student {letter}")

    run(
        client,
        f"cd {REMOTE} && . .venv-hil/bin/activate && python - <<'PY'\n"
        "import serial, time, sys\n"
        f"port={port!r}\n"
        "ser=serial.Serial(port, 115200, timeout=1)\n"
        "time.sleep(2.0); ser.reset_input_buffer()\n"
        "ser.write(b'CUKDID?\\n'); ser.flush(); time.sleep(0.6)\n"
        "print(ser.read(1024)); ser.close()\n"
        "PY",
    )

    for limit, tag in [(10, "smoke_10"), (1000, "validation_1000"), (None, "full_56200")]:
        if stream(client, port, gen, out, limit, tag):
            raise RuntimeError(f"stream failed {letter} {tag}")

    # fetch
    sftp = client.open_sftp()
    local_out = LOCAL / out
    local_out.mkdir(parents=True, exist_ok=True)
    for name in [
        "smoke_10_mcu.csv",
        "smoke_10_sequence.json",
        "validation_1000_mcu.csv",
        "validation_1000_sequence.json",
        "full_56200_mcu.csv",
        "full_56200_sequence.json",
    ]:
        sftp.get(f"{REMOTE}/{out}/{name}", str(local_out / name))
        print("fetched", letter, name)
    sftp.close()

    ref = pd.read_csv(LOCAL / gen / "hil_reference_predictions.csv")
    mcu = pd.read_csv(local_out / "full_56200_mcu.csv")
    agree = float((mcu["predicted_class"].to_numpy() == ref["fixed_pred"].to_numpy()).mean())
    report = {
        "board": "esp32c3",
        "student": letter,
        "protocol": "train_only_seed42_copy_pipeline",
        "port": port,
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
    if agree != 1.0 or not report["all_status_ok"]:
        raise RuntimeError(f"Student {letter} HIL quality failed: {report}")
    return report


def main() -> int:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        HOST, username=USER, password=PASSWORD, timeout=30, allow_agent=False, look_for_keys=False
    )
    try:
        # Upload ESP32 bundles
        sftp = client.open_sftp()
        for letter in ["A", "B"]:
            upload_tree(
                sftp,
                LOCAL
                / f"deployment/hardware_hil/build/train_only_student_{letter}_seed42_esp32c3_copy",
                f"{REMOTE}/deployment/hardware_hil/build/train_only_student_{letter}_seed42_esp32c3_copy",
            )
        sftp.close()

        # Install esp32 core if missing
        run(client, "export PATH=$HOME/.local/bin:$PATH; arduino-cli version")
        run(
            client,
            "export PATH=$HOME/.local/bin:$PATH; "
            "arduino-cli config dump | head -40; "
            "grep -q 'esp32' $HOME/.arduino15/arduino-cli.yaml 2>/dev/null || "
            "arduino-cli config add board_manager.additional_urls "
            "https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json",
        )
        run(client, "export PATH=$HOME/.local/bin:$PATH; arduino-cli core update-index", timeout=600)
        run(
            client,
            "export PATH=$HOME/.local/bin:$PATH; arduino-cli core install esp32:esp32",
            timeout=1800,
        )

        port = detect_esp_port(client)
        print("Using ESP port:", port)

        reports = []
        for letter in ["A", "B"]:
            reports.append(run_student(client, letter, port))

        summary = {
            "protocol": "train_only_seed42_copy_pipeline",
            "board": "esp32c3",
            "port": port,
            "pairs": reports,
        }
        out = LOCAL / "results/hardware_hil/train_only_scaler_copy/esp32c3_student_A_B_summary.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print("Wrote", out)
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
