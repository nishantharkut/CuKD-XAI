"""Report Arduino toolchain versions on Pi and annotate compile logs."""
from __future__ import annotations

import os
import re
from pathlib import Path

import paramiko

HOST = os.environ.get("CUKD_PI_HOST", "192.168.137.234")
PASSWORD = os.environ["CUKD_PI_PASSWORD"]
REMOTE = "/home/project/Desktop/CuKD-XAI"
LOCAL = Path(r"C:\N Drive\Research\Cukd-XAI\CuKD-XAI")
OUT = LOCAL / "results/hardware_hil/train_only_scaler_copy/compile_evidence"

CMD = r"""
python3 - <<'PY'
import os, glob, subprocess, re
from pathlib import Path
home = Path.home()
esp = sorted(home.glob('.arduino15/packages/esp32/tools/**/riscv32-esp-elf-g++'))
arm = sorted(home.glob('.arduino15/packages/arduino/tools/**/arm-none-eabi-g++'))
# also broader
if not esp:
    esp = sorted(home.glob('.arduino15/packages/**/riscv32-esp-elf-g++'))
if not arm:
    arm = sorted(home.glob('.arduino15/packages/**/arm-none-eabi-g++'))
print('ESP_CANDIDATES', len(esp))
print('ARM_CANDIDATES', len(arm))
esp_bin = str(esp[-1]) if esp else ''
arm_bin = str(arm[-1]) if arm else ''
print('ESP_COMPILER', esp_bin)
print('ARM_COMPILER', arm_bin)
def ver(path):
    if not path:
        return '', ''
    out = subprocess.check_output([path, '--version'], text=True, stderr=subprocess.STDOUT)
    print('VERSION_OUT_BEGIN')
    print(out.strip())
    print('VERSION_OUT_END')
    m = re.search(r'(\d+\.\d+\.\d+)', out)
    return out.strip().splitlines()[0] if out.strip() else '', (m.group(1) if m else '')
eline, ever = ver(esp_bin)
aline, aver = ver(arm_bin)
print('ESP_LINE', eline)
print('ESP_VER', ever)
print('ARM_LINE', aline)
print('ARM_VER', aver)
PY
"""


def main() -> int:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        HOST,
        username="project",
        password=PASSWORD,
        timeout=25,
        allow_agent=False,
        look_for_keys=False,
    )
    _, stdout, stderr = client.exec_command(CMD, timeout=120)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    print(out)
    if err.strip():
        print(err)
    client.close()

    esp_ver = re.search(r"ESP_VER\s+(\S+)", out)
    arm_ver = re.search(r"ARM_VER\s+(\S+)", out)
    esp_line = re.search(r"ESP_LINE\s+(.*)", out)
    arm_line = re.search(r"ARM_LINE\s+(.*)", out)
    esp_v = esp_ver.group(1) if esp_ver else "UNKNOWN"
    arm_v = arm_ver.group(1) if arm_ver else "UNKNOWN"
    esp_l = esp_line.group(1).strip() if esp_line else ""
    arm_l = arm_line.group(1).strip() if arm_line else ""

    meta = {
        "esp32_toolchain_version": esp_v,
        "esp32_toolchain_line": esp_l,
        "arduino_r4_toolchain_version": arm_v,
        "arduino_r4_toolchain_line": arm_l,
        "raw": out,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "toolchain_versions.json").write_text(
        __import__("json").dumps(meta, indent=2) + "\n", encoding="utf-8"
    )

    # Annotate logs with observed toolchain --version so strict recorder can bind
    for key in ["esp32c3_student_A", "esp32c3_student_B", "arduino_r4_student_A", "arduino_r4_student_B"]:
        log = OUT / key / "compile_verbose.log"
        if not log.is_file():
            continue
        text = log.read_text(encoding="utf-8", errors="replace")
        if key.startswith("esp32"):
            ver, line = esp_v, esp_l
        else:
            ver, line = arm_v, arm_l
        if ver == "UNKNOWN":
            continue
        if re.search(rf"(?<![\w.]){re.escape(ver)}(?![\w.])", text) is None or (
            line and line not in text
        ):
            text = (
                text.rstrip()
                + f"\n\n# Observed compiler --version (arduino-cli package toolchain)\n{line}\n{ver}\n"
            )
            log.write_text(text, encoding="utf-8")
            print("annotated", log, ver)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
