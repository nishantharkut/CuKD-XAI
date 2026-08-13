"""Bind a verbose Arduino compile and firmware binary to one wireless bundle."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from deployment.hardware_hil.host.record_fgds_compile_evidence import (
    BOARD_FQBN_PREFIXES,
    FLASH_PATTERN,
    RAM_PATTERN,
    parsed_match,
    require_observed_metadata,
    validate_footprint,
)

try:
    from .wireless_common import (
        WIRELESS_PROTOCOL_ID,
        read_compile_log_text,
        sha256_file,
        validate_compile_log_metadata,
        verify_export_for_wireless,
        verify_wireless_bundle,
    )
except ImportError:
    from wireless_common import (  # type: ignore
        WIRELESS_PROTOCOL_ID,
        read_compile_log_text,
        sha256_file,
        validate_compile_log_metadata,
        verify_export_for_wireless,
        verify_wireless_bundle,
    )


def verify_binary_identity(binary: Path, required_values: dict[str, str]) -> None:
    payload = binary.read_bytes()
    for label, value in required_values.items():
        if value.encode("ascii") not in payload:
            raise RuntimeError(f"Firmware binary does not contain {label}: {value}")


def artifact_record(output: Path, artifact: Path, *, include_size: bool = True) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": artifact.relative_to(output.parent).as_posix(),
        "sha256": sha256_file(artifact),
    }
    if include_size:
        record["size_bytes"] = artifact.stat().st_size
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--compile-log", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--fqbn", required=True)
    parser.add_argument("--board-core-version", required=True)
    parser.add_argument("--frontend-version", required=True)
    parser.add_argument("--toolchain-version", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    generated_dir = args.generated_dir.resolve()
    bundle_dir = args.bundle_dir.resolve()
    compile_log = args.compile_log.resolve()
    binary = args.binary.resolve()
    output = args.output_json.resolve()
    artifact_dir = output.parent / f"{output.stem}_artifacts"
    artifact_staging = output.parent / f".{output.stem}_artifacts.tmp"
    output_temporary = output.with_suffix(output.suffix + ".tmp")
    if any(path.exists() for path in [output, artifact_dir, artifact_staging, output_temporary]):
        raise FileExistsError(f"Refusing to overwrite wireless compile evidence: {output}")
    if (
        artifact_staging.resolve().parent != output.parent.resolve()
        or artifact_staging.name != f".{output.stem}_artifacts.tmp"
    ):
        raise RuntimeError("Unsafe compile-evidence staging path")
    for protected in [generated_dir, bundle_dir]:
        for candidate in [output, artifact_dir.resolve()]:
            try:
                candidate.relative_to(protected)
            except ValueError:
                continue
            raise RuntimeError("Compile evidence cannot be written inside an input bundle")
    if not compile_log.is_file() or not binary.is_file() or binary.stat().st_size <= 0:
        raise FileNotFoundError(compile_log if not compile_log.is_file() else binary)
    if compile_log == binary or sha256_file(compile_log) == sha256_file(binary):
        raise RuntimeError("Compile log and firmware binary must be distinct artifacts")

    export = verify_export_for_wireless(generated_dir)
    bundle = verify_wireless_bundle(bundle_dir, export)
    verify_binary_identity(
        binary,
        {
            "export_id": export["export_id"],
            "wireless_bundle_id": bundle["wireless_bundle_id"],
            "wireless_protocol_id": WIRELESS_PROTOCOL_ID,
        },
    )
    text = read_compile_log_text(compile_log)
    flash = parsed_match(FLASH_PATTERN, text, "flash")
    ram = parsed_match(RAM_PATTERN, text, "RAM")
    validate_footprint(flash, ram)
    sketch_file = bundle["sketch_file"]
    if sketch_file not in text or sketch_file not in binary.name:
        raise RuntimeError("Compile log/binary does not identify the wireless sketch")

    fqbn = require_observed_metadata("FQBN", args.fqbn)
    if not any(
        fqbn == prefix or fqbn.startswith(prefix + ":")
        for prefix in BOARD_FQBN_PREFIXES[bundle["board"]]
    ):
        raise RuntimeError("FQBN does not match the wireless bundle board")
    board_core_version = require_observed_metadata(
        "board core version", args.board_core_version
    )
    frontend_version = require_observed_metadata(
        "frontend version", args.frontend_version
    )
    toolchain_version = require_observed_metadata(
        "toolchain version", args.toolchain_version
    )
    validate_compile_log_metadata(
        text,
        fqbn=fqbn,
        board_core_version=board_core_version,
        frontend_version=frontend_version,
        toolchain_version=toolchain_version,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    artifact_staging.mkdir()
    copied = {
        "compile_log": artifact_staging / "compile_log.txt",
        "firmware_binary": artifact_staging / "firmware.bin",
        "strict_export_manifest": artifact_staging / "strict_export_manifest.json",
        "strict_export_report": artifact_staging / "strict_export_report.json",
        "wireless_bundle_manifest": artifact_staging / "wireless_bundle_manifest.json",
        "hil_reference_predictions": artifact_staging / "hil_reference_predictions.csv",
    }
    sources = {
        "compile_log": compile_log,
        "firmware_binary": binary,
        "strict_export_manifest": generated_dir / "strict_export_manifest.json",
        "strict_export_report": generated_dir / "strict_export_report.json",
        "wireless_bundle_manifest": bundle_dir / "wireless_bundle_manifest.json",
        "hil_reference_predictions": generated_dir / "hil_reference_predictions.csv",
    }
    try:
        for name, source in sources.items():
            shutil.copy2(source, copied[name])
            if sha256_file(copied[name]) != sha256_file(source):
                raise RuntimeError(f"Copied compile artifact changed: {name}")
        os.replace(artifact_staging, artifact_dir)
        copied = {name: artifact_dir / path.name for name, path in copied.items()}
        evidence = {
            "status": "passed",
            "protocol_id": WIRELESS_PROTOCOL_ID,
            "student": bundle["student"],
            "board": bundle["board"],
            "export_id": bundle["export_id"],
            "wireless_bundle_id": bundle["wireless_bundle_id"],
            "sketch_file": sketch_file,
            "fqbn": fqbn,
            "board_core_version": board_core_version,
            "frontend_version": frontend_version,
            "toolchain_version": toolchain_version,
            "flash": flash,
            "ram": ram,
            "compile_log_sha256": sha256_file(compile_log),
            "firmware_binary_size_bytes": binary.stat().st_size,
            "firmware_binary_sha256": sha256_file(binary),
            "strict_export_manifest_sha256": sha256_file(
                generated_dir / "strict_export_manifest.json"
            ),
            "wireless_bundle_manifest_sha256": bundle["_manifest_sha256"],
            "compile_evidence_script_sha256": sha256_file(Path(__file__).resolve()),
            "portable_artifacts": {
                name: artifact_record(
                    output,
                    path,
                    include_size=name
                    in {"compile_log", "firmware_binary", "hil_reference_predictions"},
                )
                for name, path in copied.items()
            },
            "credential_boundary": (
                "The compiled bundle contains no configured SSID or password. Runtime "
                "credentials are supplied only after flashing through serial provisioning."
            ),
            "compile_association_boundary": (
                "The preserved binary embeds the strict export ID, wireless bundle ID, "
                "and wireless protocol ID. The verbose log and binary filename identify "
                "the same sketch. This is provenance binding, not secure boot attestation."
            ),
        }
        output_temporary.write_text(
            json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(output_temporary, output)
    except Exception:
        if artifact_staging.exists():
            shutil.rmtree(artifact_staging)
        if artifact_dir.exists():
            if (
                artifact_dir.resolve().parent != output.parent.resolve()
                or artifact_dir.name != f"{output.stem}_artifacts"
            ):
                raise RuntimeError("Refusing to remove unsafe compile artifact path")
            shutil.rmtree(artifact_dir)
        if output_temporary.exists():
            output_temporary.unlink()
        raise
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
