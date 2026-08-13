"""Build a hash-bound Wi-Fi UDP firmware bundle from a passed FG-DS export."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

try:
    from .wireless_common import (
        DEFAULT_DEVICE_UDP_PORT,
        GENERATED_FILES,
        MAX_DATAGRAM_BYTES,
        MODEL_COMMON_FILES,
        SERIAL_IDENTITY_QUERY,
        WIRELESS_COMMON_FILES,
        WIRELESS_PROTOCOL_ID,
        canonical_json_sha256,
        sha256_file,
        verify_export_for_wireless,
    )
except ImportError:
    from wireless_common import (  # type: ignore
        DEFAULT_DEVICE_UDP_PORT,
        GENERATED_FILES,
        MAX_DATAGRAM_BYTES,
        MODEL_COMMON_FILES,
        SERIAL_IDENTITY_QUERY,
        WIRELESS_COMMON_FILES,
        WIRELESS_PROTOCOL_ID,
        canonical_json_sha256,
        sha256_file,
        verify_export_for_wireless,
    )


WIRELESS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WIRELESS_ROOT.parents[1]
HARDWARE_HIL_ROOT = REPO_ROOT / "deployment" / "hardware_hil"
TEMPLATE = WIRELESS_ROOT / "firmware" / "cukd_wireless_fgds" / "cukd_wireless_fgds.ino"
WIRELESS_COMMON = WIRELESS_ROOT / "firmware" / "common"
MODEL_COMMON = HARDWARE_HIL_ROOT / "firmware" / "common"
def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def prepare_staging(output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite wireless bundle: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.tmp.{os.getpid()}.{time.time_ns()}"
    if staging.exists():
        raise FileExistsError(f"Stale wireless bundle staging path exists: {staging}")
    staging.mkdir()
    return staging


def discard_staging(staging: Path, parent: Path) -> None:
    resolved = staging.resolve()
    if (
        resolved.parent != parent.resolve()
        or not resolved.name.startswith(".")
        or ".tmp." not in resolved.name
    ):
        raise RuntimeError(f"Refusing to remove unsafe staging path: {resolved}")
    shutil.rmtree(resolved)


def read_export_report(generated_dir: Path) -> dict[str, Any]:
    report = json.loads(
        (generated_dir / "strict_export_report.json").read_text(encoding="utf-8")
    )
    if not isinstance(report, dict) or report.get("status") != "passed":
        raise RuntimeError("FG-DS export report is not passed")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", choices=["arduino_r4", "esp32c3"], required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    generated_dir = resolve(args.generated_dir)
    output_dir = resolve(args.output_dir)
    try:
        output_dir.relative_to(generated_dir)
    except ValueError:
        pass
    else:
        raise RuntimeError("Wireless bundle output cannot be inside its strict export")
    export_manifest = verify_export_for_wireless(generated_dir)
    export_report = read_export_report(generated_dir)
    tested_common = export_report.get("provenance", {}).get("firmware_common_files")
    expected_tested = {
        "cukd_model.c",
        "cukd_model.h",
        "cukd_preprocess.c",
        "cukd_preprocess.h",
    }
    if not isinstance(tested_common, dict) or set(tested_common) != expected_tested:
        raise RuntimeError("FG-DS export lacks complete host-tested model-core hashes")
    for name, expected_hash in tested_common.items():
        if sha256_file(MODEL_COMMON / name) != expected_hash:
            raise RuntimeError(f"Host-tested model core changed after export: {name}")

    source_files = [
        TEMPLATE,
        *[MODEL_COMMON / name for name in sorted(MODEL_COMMON_FILES)],
        *[WIRELESS_COMMON / name for name in sorted(WIRELESS_COMMON_FILES)],
        *[generated_dir / name for name in sorted(GENERATED_FILES)],
    ]
    for path in source_files:
        if not path.is_file():
            raise FileNotFoundError(path)
    snapshots = {path.name: path.read_bytes() for path in source_files}
    if len(snapshots) != len(source_files):
        raise RuntimeError("Wireless bundle source filenames are not unique")
    export_hashes = {
        item["path"]: item["sha256"] for item in export_manifest.get("files", [])
    }
    for name in GENERATED_FILES:
        if hashlib.sha256(snapshots[name]).hexdigest() != export_hashes.get(name):
            raise RuntimeError(f"Wireless bundle export input changed: {name}")

    identity_payload = {
        "protocol_id": WIRELESS_PROTOCOL_ID,
        "board": args.board,
        "student": export_manifest["student"],
        "export_id": export_manifest["export_id"],
        "strict_export_manifest_sha256": sha256_file(
            generated_dir / "strict_export_manifest.json"
        ),
        "device_udp_port": DEFAULT_DEVICE_UDP_PORT,
        "max_datagram_bytes": MAX_DATAGRAM_BYTES,
        "source_files": [
            {
                "name": path.name,
                "size_bytes": len(snapshots[path.name]),
                "sha256": hashlib.sha256(snapshots[path.name]).hexdigest(),
            }
            for path in source_files
        ],
        "bundler_sha256": sha256_file(Path(__file__).resolve()),
    }
    wireless_bundle_id = canonical_json_sha256(identity_payload)
    board_macro = (
        "CUKD_WIRELESS_BOARD_ESP32C3"
        if args.board == "esp32c3"
        else "CUKD_WIRELESS_BOARD_ARDUINO_R4"
    )
    identity_header = (
        "#ifndef CUKD_WIRELESS_BUNDLE_IDENTITY_H\n"
        "#define CUKD_WIRELESS_BUNDLE_IDENTITY_H\n"
        f'#define CUKD_WIRELESS_BUNDLE_ID "{wireless_bundle_id}"\n'
        f'#define CUKD_WIRELESS_PROTOCOL_ID "{WIRELESS_PROTOCOL_ID}"\n'
        f'#define CUKD_WIRELESS_BOARD_ID "{args.board}"\n'
        f"#define CUKD_WIRELESS_UDP_PORT {DEFAULT_DEVICE_UDP_PORT}u\n"
        f"#define CUKD_WIRELESS_MAX_DATAGRAM {MAX_DATAGRAM_BYTES}u\n"
        f"#define {board_macro} 1\n"
        "#endif\n"
    )

    staging = prepare_staging(output_dir)
    sketch_name = f"{output_dir.name}.ino"
    try:
        for name in [
            *sorted(MODEL_COMMON_FILES),
            *sorted(WIRELESS_COMMON_FILES),
            *sorted(GENERATED_FILES),
        ]:
            (staging / name).write_bytes(snapshots[name])
        (staging / sketch_name).write_bytes(snapshots[TEMPLATE.name])
        (staging / "cukd_wireless_bundle_identity.h").write_text(
            identity_header,
            encoding="ascii",
        )
        manifest_path = staging / "wireless_bundle_manifest.json"
        files = [
            {
                "path": path.relative_to(staging).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(staging.iterdir())
            if path.is_file() and path != manifest_path
        ]
        manifest = {
            "status": "passed",
            "protocol_id": WIRELESS_PROTOCOL_ID,
            "transport": "IEEE 802.11 Wi-Fi UDP",
            "board": args.board,
            "student": export_manifest["student"],
            "export_id": export_manifest["export_id"],
            "wireless_bundle_id": wireless_bundle_id,
            "wireless_bundle_identity_payload": identity_payload,
            "strict_export_manifest_sha256": identity_payload[
                "strict_export_manifest_sha256"
            ],
            "device_udp_port": DEFAULT_DEVICE_UDP_PORT,
            "max_datagram_bytes": MAX_DATAGRAM_BYTES,
            "bundler_sha256": identity_payload["bundler_sha256"],
            "generated_dir_recorded": str(generated_dir),
            "sketch_file": sketch_name,
            "serial_identity_query": SERIAL_IDENTITY_QUERY,
            "device_identity_response": (
                f"CUKDWBUILD,{export_manifest['student']},"
                f"{export_manifest['export_id']},{wireless_bundle_id},"
                f"{args.board},{WIRELESS_PROTOCOL_ID}"
            ),
            "credential_boundary": (
                "SSID and password are provisioned over local USB serial at runtime "
                "and are excluded from source, manifests, and result artifacts. ESP32 "
                "firmware disables Arduino Wi-Fi persistence before initialization; "
                "WiFiS3 coprocessor persistence is not asserted for UNO R4 WiFi."
            ),
            "file_count_excluding_manifest": len(files),
            "files": files,
            "claim_boundary": (
                "Controlled-LAN Wi-Fi UDP replay of already extracted 17-feature "
                "FG-DS records into the fixed-point MCU inference path. It does not "
                "perform live packet capture, packet-to-feature extraction, secure "
                "application transport, energy measurement, or BLE/WSN-radio testing."
            ),
        }
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, manifest_path)
        os.replace(staging, output_dir)
    except Exception:
        if staging.exists():
            discard_staging(staging, output_dir.parent)
        raise
    print(output_dir / "wireless_bundle_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
