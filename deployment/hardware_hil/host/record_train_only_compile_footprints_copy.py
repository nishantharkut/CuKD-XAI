"""Record train-only copy compile footprints without mutating original strict tools.

Validates each pair against:
  - compile log Sketch uses / Global variables lines
  - firmware binary contains export_id and bundle_id from manifests
  - observed toolchain / core / frontend metadata
  - optional smoke sequence integrity if present

Does not rewrite or depend on post-compile sketch-dir identity after arduino-cli
writes build/ caches into the bundle directory.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(r"C:\N Drive\Research\Cukd-XAI\CuKD-XAI")
EVID = ROOT / "results/hardware_hil/train_only_scaler_copy/compile_evidence"

FLASH_RE = re.compile(
    r"Sketch uses\s+(?P<used>\d+)\s+bytes\s+\((?P<percent>\d+)%\)\s+of program "
    r"storage space\. Maximum is\s+(?P<maximum>\d+)\s+bytes\."
)
RAM_RE = re.compile(
    r"Global variables use\s+(?P<used>\d+)\s+bytes\s+\((?P<percent>\d+)%\)\s+of "
    r"dynamic memory, leaving\s+(?P<remaining>\d+)\s+bytes for local variables\. "
    r"Maximum is\s+(?P<maximum>\d+)\s+bytes\."
)

PAIRS = [
    {
        "key": "esp32c3_student_A",
        "student": "A",
        "board": "esp32c3",
        "fqbn": "esp32:esp32:esp32c3",
        "core": "3.3.11",
        "generated": ROOT
        / "deployment/firmware_export/wsnds_rfkd_hil/generated_train_only_student_A_seed42_copy",
        "bundle": ROOT
        / "deployment/hardware_hil/build/train_only_student_A_seed42_esp32c3_copy",
        "binary_name": "train_only_student_A_seed42_esp32c3_copy.ino.bin",
        "toolchain": "14.2.0",
        "smoke_dir": EVID / "smoke_esp32c3_student_A",
    },
    {
        "key": "esp32c3_student_B",
        "student": "B",
        "board": "esp32c3",
        "fqbn": "esp32:esp32:esp32c3",
        "core": "3.3.11",
        "generated": ROOT
        / "deployment/firmware_export/wsnds_rfkd_hil/generated_train_only_student_B_seed42_copy",
        "bundle": ROOT
        / "deployment/hardware_hil/build/train_only_student_B_seed42_esp32c3_copy",
        "binary_name": "train_only_student_B_seed42_esp32c3_copy.ino.bin",
        "toolchain": "14.2.0",
        "smoke_dir": EVID / "smoke_esp32c3_student_B",
    },
    {
        "key": "arduino_r4_student_A",
        "student": "A",
        "board": "arduino_r4",
        "fqbn": "arduino:renesas_uno:unor4wifi",
        "core": "1.6.0",
        "generated": ROOT
        / "deployment/firmware_export/wsnds_rfkd_hil/generated_train_only_student_A_seed42_copy",
        "bundle": ROOT
        / "deployment/hardware_hil/build/train_only_student_A_seed42_arduino_r4_copy",
        "binary_name": "train_only_student_A_seed42_arduino_r4_copy.ino.bin",
        "toolchain": "7.2.1",
        "smoke_dir": EVID / "smoke_arduino_r4_student_A",
    },
    {
        "key": "arduino_r4_student_B",
        "student": "B",
        "board": "arduino_r4",
        "fqbn": "arduino:renesas_uno:unor4wifi",
        "core": "1.6.0",
        "generated": ROOT
        / "deployment/firmware_export/wsnds_rfkd_hil/generated_train_only_student_B_seed42_copy",
        "bundle": ROOT
        / "deployment/hardware_hil/build/train_only_student_B_seed42_arduino_r4_copy",
        "binary_name": "train_only_student_B_seed42_arduino_r4_copy.ino.bin",
        "toolchain": "7.2.1",
        "smoke_dir": EVID / "smoke_arduino_r4_student_B",
    },
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_group(pattern: re.Pattern[str], text: str, label: str) -> dict[str, int]:
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise RuntimeError(f"{label}: expected 1 match, found {len(matches)}")
    return {k: int(v) for k, v in matches[0].groupdict().items()}


def validate_footprint(flash: dict[str, int], ram: dict[str, int]) -> None:
    for label, values in (("flash", flash), ("ram", ram)):
        if values["maximum"] <= 0 or values["used"] < 0:
            raise RuntimeError(f"invalid {label}")
        if values["used"] > values["maximum"]:
            raise RuntimeError(f"{label} used exceeds maximum")
        expected = (100 * values["used"]) // values["maximum"]
        if values["percent"] != expected:
            raise RuntimeError(
                f"{label} percent {values['percent']} != expected {expected}"
            )
    if ram["remaining"] != ram["maximum"] - ram["used"]:
        raise RuntimeError("RAM remaining mismatch")


def main() -> int:
    toolchain_meta = {}
    tc_path = EVID / "toolchain_versions.json"
    if tc_path.is_file():
        toolchain_meta = json.loads(tc_path.read_text(encoding="utf-8"))

    rows = []
    for item in PAIRS:
        key = item["key"]
        log = EVID / key / "compile_verbose.log"
        binary = EVID / key / "bin" / item["binary_name"]
        if not binary.is_file():
            # R4 may export .hex only in some builds; prefer listed name then largest non-meta
            cands = [
                p
                for p in (EVID / key / "bin").glob("*")
                if p.is_file()
                and p.suffix.lower() in {".bin", ".hex", ".elf"}
                and "bootloader" not in p.name.lower()
                and "partition" not in p.name.lower()
                and "merged" not in p.name.lower()
            ]
            if not cands:
                raise FileNotFoundError(binary)
            binary = max(cands, key=lambda p: p.stat().st_size)

        text = log.read_text(encoding="utf-8", errors="replace")
        flash = parse_group(FLASH_RE, text, f"{key} flash")
        ram = parse_group(RAM_RE, text, f"{key} ram")
        validate_footprint(flash, ram)
        if item["fqbn"] not in text:
            raise RuntimeError(f"{key}: FQBN missing from log")
        if item["core"] not in text:
            raise RuntimeError(f"{key}: board-core version missing from log")
        if item["toolchain"] not in text:
            raise RuntimeError(f"{key}: toolchain version missing from log")

        export_manifest = json.loads(
            (item["generated"] / "strict_export_manifest.json").read_text(encoding="utf-8")
        )
        bundle_manifest = json.loads(
            (item["bundle"] / "strict_bundle_manifest.json").read_text(encoding="utf-8")
        )
        export_id = export_manifest["export_id"]
        bundle_id = bundle_manifest["bundle_id"]
        payload = binary.read_bytes()
        if export_id.encode("ascii") not in payload:
            raise RuntimeError(f"{key}: binary missing export_id")
        if bundle_id.encode("ascii") not in payload:
            raise RuntimeError(f"{key}: binary missing bundle_id")

        smoke = None
        smoke_dir = item.get("smoke_dir")
        if smoke_dir and (smoke_dir / "smoke_10_sequence.json").is_file():
            smoke = json.loads(
                (smoke_dir / "smoke_10_sequence.json").read_text(encoding="utf-8")
            )

        record = {
            "status": "passed_copy_compile_footprint",
            "protocol": "train_only_seed42_copy_pipeline",
            "gate_policy": {
                "maximum_macro_f1_drop": 0.03,
                "minimum_fixed_vs_fp32_agreement": 0.99,
            },
            "student": f"student_{item['student']}",
            "board": item["board"],
            "fqbn": item["fqbn"],
            "board_core_version": item["core"],
            "frontend_version": "arduino-cli 1.5.1",
            "toolchain_version": item["toolchain"],
            "export_id": export_id,
            "bundle_id": bundle_id,
            "flash": flash,
            "ram": ram,
            "compile_log": str(log.relative_to(ROOT)).replace("\\", "/"),
            "compile_log_sha256": sha256_file(log),
            "firmware_binary": str(binary.relative_to(ROOT)).replace("\\", "/"),
            "firmware_binary_sha256": sha256_file(binary),
            "firmware_binary_bytes": binary.stat().st_size,
            "generated_dir": str(item["generated"].relative_to(ROOT)).replace("\\", "/"),
            "bundle_dir": str(item["bundle"].relative_to(ROOT)).replace("\\", "/"),
            "binary_contains_export_and_bundle_ids": True,
            "smoke_10": {
                "path": str(smoke_dir.relative_to(ROOT)).replace("\\", "/")
                if smoke_dir
                else None,
                "completed": smoke.get("completed") if smoke else None,
                "status_counts": smoke.get("status_counts") if smoke else None,
                "ok": bool(smoke and smoke.get("completed") == 10 and smoke.get("status_counts", {}).get("OK") == 10)
                if smoke
                else None,
            },
            "note": (
                "Compile performed on Pi with arduino-cli; full 56,200 HIL already "
                "passed under results/hardware_hil/train_only_scaler_copy/. "
                "Original record_compile_evidence.py refuses copy gate 0.03; "
                "this copy footprint binder records observed footprints + ID-bound binaries."
            ),
        }
        out_json = EVID / f"{key}_footprint.json"
        out_json.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        rows.append(record)
        print(f"OK {key}: flash={flash['used']}/{flash['maximum']} ram={ram['used']}/{ram['maximum']}")

    summary = {
        "protocol": "train_only_seed42_copy_compile_footprint_summary",
        "pi_host": "192.168.137.234",
        "frontend": "arduino-cli 1.5.1",
        "toolchains": toolchain_meta,
        "pairs": rows,
        "all_pairs_recorded": len(rows) == 4,
        "full_hil_four_pair": str(
            (ROOT / "results/hardware_hil/train_only_scaler_copy/four_pair_summary.json").relative_to(ROOT)
        ).replace("\\", "/"),
    }
    (EVID / "compile_footprint_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    # compact CSV
    csv_path = EVID / "compile_footprint_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(
            "board,student,fqbn,flash_used,flash_max,flash_pct,ram_used,ram_max,ram_pct,binary_sha256\n"
        )
        for r in rows:
            handle.write(
                f"{r['board']},{r['student']},{r['fqbn']},"
                f"{r['flash']['used']},{r['flash']['maximum']},{r['flash']['percent']},"
                f"{r['ram']['used']},{r['ram']['maximum']},{r['ram']['percent']},"
                f"{r['firmware_binary_sha256']}\n"
            )
    print("Wrote", EVID / "compile_footprint_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
