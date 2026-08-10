"""COPY of record_compile_evidence.py for train-only copy gate (macro_drop 0.03).

Original record_compile_evidence.py is untouched. This variant imports
stream_vectors_strict_copy so export manifests with maximum_macro_f1_drop=0.03
are accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

try:
    from .stream_vectors_strict_copy import verify_bundle, verify_export
except ImportError:
    from stream_vectors_strict_copy import verify_bundle, verify_export


FLASH_PATTERN = re.compile(
    r"Sketch uses\s+(?P<used>\d+)\s+bytes\s+\((?P<percent>\d+)%\)\s+of program "
    r"storage space\. Maximum is\s+(?P<maximum>\d+)\s+bytes\."
)
RAM_PATTERN = re.compile(
    r"Global variables use\s+(?P<used>\d+)\s+bytes\s+\((?P<percent>\d+)%\)\s+of "
    r"dynamic memory, leaving\s+(?P<remaining>\d+)\s+bytes for local variables\. "
    r"Maximum is\s+(?P<maximum>\d+)\s+bytes\."
)
BOARD_FQBN_PREFIXES = {
    "esp32c3": ("esp32:esp32:esp32c3",),
    "arduino_r4": ("arduino:renesas_uno:unor4wifi",),
}
PLACEHOLDER_TOKENS = {"OBSERVED", "UNKNOWN", "PLACEHOLDER", "TODO"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parsed_match(pattern: re.Pattern[str], text: str, label: str) -> dict[str, int]:
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {label} footprint line, found {len(matches)}")
    return {key: int(value) for key, value in matches[0].groupdict().items()}


def verify_binary_identity(binary: Path, export_id: str, bundle_id: str) -> None:
    payload = binary.read_bytes()
    for label, value in (("export_id", export_id), ("bundle_id", bundle_id)):
        if value.encode("ascii") not in payload:
            raise RuntimeError(
                f"Firmware binary does not contain the expected {label}: {value}"
            )


def require_observed_metadata(label: str, value: str) -> str:
    cleaned = value.strip()
    if not cleaned or any(token in cleaned.upper() for token in PLACEHOLDER_TOKENS):
        raise RuntimeError(f"{label} is empty or still contains a placeholder")
    return cleaned


def validate_footprint(flash: dict[str, int], ram: dict[str, int]) -> None:
    for label, values in (("flash", flash), ("RAM", ram)):
        if values["maximum"] <= 0 or values["used"] < 0:
            raise RuntimeError(f"{label} footprint contains an invalid capacity")
        if values["used"] > values["maximum"]:
            raise RuntimeError(f"{label} footprint exceeds declared capacity")
        expected_percent = (100 * values["used"]) // values["maximum"]
        if values["percent"] != expected_percent:
            raise RuntimeError(
                f"{label} percentage is {values['percent']}, expected {expected_percent}"
            )
    if ram["remaining"] < 0 or ram["remaining"] != ram["maximum"] - ram["used"]:
        raise RuntimeError("RAM remaining bytes do not equal maximum minus used")


def relative_artifact_path(output: Path, artifact: Path) -> str:
    return artifact.relative_to(output.parent).as_posix()


def preserve_orphan(path: Path) -> Path:
    preserved = path.parent / f"{path.name}.failed.{time.time_ns()}"
    os.replace(path, preserved)
    return preserved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", choices=["A", "B"], required=True)
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
    artifact_temp = output.parent / f".{artifact_dir.name}.tmp"
    for protected_root in [generated_dir, bundle_dir]:
        for candidate in [output, artifact_dir.resolve()]:
            try:
                candidate.relative_to(protected_root)
            except ValueError:
                continue
            raise RuntimeError("Compile evidence cannot be written inside a protected input")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite compile evidence: {output}")
    output_temporary = output.with_suffix(output.suffix + ".tmp")
    for orphan in [artifact_dir, artifact_temp, output_temporary]:
        if orphan.exists():
            preserve_orphan(orphan)
    if not compile_log.is_file() or not binary.is_file():
        raise FileNotFoundError(compile_log if not compile_log.is_file() else binary)
    compile_log_sha256 = sha256_file(compile_log)
    firmware_binary_sha256 = sha256_file(binary)
    if compile_log == binary or compile_log_sha256 == firmware_binary_sha256:
        raise RuntimeError("Compile log and firmware binary must be distinct artifacts")
    if binary.stat().st_size <= 0:
        raise RuntimeError("Firmware binary is empty")

    export_manifest = verify_export(generated_dir)
    bundle_manifest = verify_bundle(bundle_dir, export_manifest)
    expected_student = export_manifest["student"]
    if expected_student != f"student_{args.student}":
        raise RuntimeError(
            f"--student {args.student} conflicts with export identity {expected_student}"
        )
    verify_binary_identity(
        binary,
        export_manifest["export_id"],
        bundle_manifest["bundle_id"],
    )
    text = compile_log.read_text(encoding="utf-8", errors="replace")
    flash = parsed_match(FLASH_PATTERN, text, "flash")
    ram = parsed_match(RAM_PATTERN, text, "RAM")
    validate_footprint(flash, ram)
    sketch_file = bundle_manifest["sketch_file"]
    if sketch_file not in text:
        raise RuntimeError(
            f"Verbose compile log does not mention the strict bundle sketch {sketch_file}"
        )
    if sketch_file not in binary.name:
        raise RuntimeError(
            f"Firmware binary name {binary.name!r} is not derived from {sketch_file!r}"
        )
    fqbn = require_observed_metadata("FQBN", args.fqbn)
    valid_fqbn = any(
        fqbn == prefix or fqbn.startswith(prefix + ":")
        for prefix in BOARD_FQBN_PREFIXES[bundle_manifest["board"]]
    )
    if not valid_fqbn:
        raise RuntimeError(
            f"FQBN {fqbn!r} does not match bundle board {bundle_manifest['board']}"
        )
    board_core_version = require_observed_metadata(
        "board core version", args.board_core_version
    )
    frontend_version = require_observed_metadata("frontend version", args.frontend_version)
    toolchain_version = require_observed_metadata(
        "toolchain version", args.toolchain_version
    )
    if re.search(
        rf"(?<![\w.]){re.escape(board_core_version)}(?![\w.])", text
    ) is None:
        raise RuntimeError("Verbose compile log does not contain the board-core version")
    if fqbn not in text:
        raise RuntimeError("Verbose compile log does not contain the exact FQBN")
    if re.search(
        rf"(?<![\w.]){re.escape(toolchain_version)}(?![\w.])", text
    ) is None:
        raise RuntimeError("Verbose compile log does not contain the toolchain version")

    output.parent.mkdir(parents=True, exist_ok=True)
    artifact_temp.mkdir()
    copied_log = artifact_temp / "compile_log.txt"
    copied_binary = artifact_temp / f"firmware{binary.suffix.lower()}"
    copied_export_manifest = artifact_temp / "strict_export_manifest.json"
    copied_export_report = artifact_temp / "strict_export_report.json"
    copied_bundle_manifest = artifact_temp / "strict_bundle_manifest.json"
    copied_reference = artifact_temp / "hil_reference_predictions.csv"
    portable_sources = {
        "strict_export_manifest": generated_dir / "strict_export_manifest.json",
        "strict_export_report": generated_dir / "strict_export_report.json",
        "strict_bundle_manifest": bundle_dir / "strict_bundle_manifest.json",
        "hil_reference_predictions": generated_dir / "hil_reference_predictions.csv",
    }
    portable_source_hashes = {
        name: sha256_file(path) for name, path in portable_sources.items()
    }
    shutil.copy2(compile_log, copied_log)
    shutil.copy2(binary, copied_binary)
    shutil.copy2(portable_sources["strict_export_manifest"], copied_export_manifest)
    shutil.copy2(portable_sources["strict_export_report"], copied_export_report)
    shutil.copy2(portable_sources["strict_bundle_manifest"], copied_bundle_manifest)
    shutil.copy2(portable_sources["hil_reference_predictions"], copied_reference)
    if sha256_file(copied_log) != compile_log_sha256:
        raise RuntimeError("Copied compile log changed after validation")
    if sha256_file(copied_binary) != firmware_binary_sha256:
        raise RuntimeError("Copied firmware binary changed after validation")
    copied_portable = {
        "strict_export_manifest": copied_export_manifest,
        "strict_export_report": copied_export_report,
        "strict_bundle_manifest": copied_bundle_manifest,
        "hil_reference_predictions": copied_reference,
    }
    for name, copied_path in copied_portable.items():
        if sha256_file(copied_path) != portable_source_hashes[name]:
            raise RuntimeError(f"Copied {name} changed after validation")
    os.replace(artifact_temp, artifact_dir)
    copied_log = artifact_dir / copied_log.name
    copied_binary = artifact_dir / copied_binary.name
    copied_export_manifest = artifact_dir / copied_export_manifest.name
    copied_export_report = artifact_dir / copied_export_report.name
    copied_bundle_manifest = artifact_dir / copied_bundle_manifest.name
    copied_reference = artifact_dir / copied_reference.name

    evidence = {
        "status": "passed",
        "student": expected_student,
        "board": bundle_manifest["board"],
        "export_id": export_manifest["export_id"],
        "bundle_id": bundle_manifest["bundle_id"],
        "sketch_file": sketch_file,
        "fqbn": fqbn,
        "board_core_version": board_core_version,
        "frontend_version": frontend_version,
        "toolchain_version": toolchain_version,
        "compile_evidence_script_sha256": sha256_file(Path(__file__).resolve()),
        "metadata_source": (
            "FQBN and version strings are operator-recorded from the compile "
            "environment. The exact FQBN, board-core version, and toolchain "
            "version must also appear in the preserved verbose compile log; the "
            "frontend version remains operator-recorded."
        ),
        "compile_association_boundary": (
            "The binary embeds the strict export and bundle IDs; its filename and "
            "the verbose log both identify the same strict sketch. This detects "
            "accidental cross-model/cross-board mixing but is not a cryptographic "
            "attestation by the Arduino frontend that the log produced the binary."
        ),
        "compile_log_path_recorded": str(compile_log),
        "compile_log_sha256": compile_log_sha256,
        "firmware_binary_path_recorded": str(binary),
        "firmware_binary_size_bytes": binary.stat().st_size,
        "firmware_binary_sha256": firmware_binary_sha256,
        "flash": flash,
        "ram": ram,
        "strict_export_manifest_sha256": portable_source_hashes["strict_export_manifest"],
        "strict_bundle_manifest_sha256": portable_source_hashes["strict_bundle_manifest"],
        "portable_artifacts": {
            "compile_log": {
                "path": relative_artifact_path(output, copied_log),
                "size_bytes": copied_log.stat().st_size,
                "sha256": sha256_file(copied_log),
            },
            "firmware_binary": {
                "path": relative_artifact_path(output, copied_binary),
                "size_bytes": copied_binary.stat().st_size,
                "sha256": sha256_file(copied_binary),
            },
            "strict_export_manifest": {
                "path": relative_artifact_path(output, copied_export_manifest),
                "sha256": sha256_file(copied_export_manifest),
            },
            "strict_export_report": {
                "path": relative_artifact_path(output, copied_export_report),
                "sha256": sha256_file(copied_export_report),
            },
            "strict_bundle_manifest": {
                "path": relative_artifact_path(output, copied_bundle_manifest),
                "sha256": sha256_file(copied_bundle_manifest),
            },
            "hil_reference_predictions": {
                "path": relative_artifact_path(output, copied_reference),
                "size_bytes": copied_reference.stat().st_size,
                "sha256": sha256_file(copied_reference),
            },
        },
    }
    output_temporary.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    os.replace(output_temporary, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
