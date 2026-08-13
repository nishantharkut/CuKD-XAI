#!/usr/bin/env python3
"""Validate and seal the existing MSP430 static-evidence artifact tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[2]
DEFAULT_ARTIFACT_ROOT = MODULE_DIR / "artifacts"
EXPECTED_CONTRACT_PATH = MODULE_DIR / "expected_contracts.json"
MANIFEST_NAME = "msp430_static_root_manifest.json"
EXPECTED_STUDENTS = ("student_A", "student_B")
ROOT_SEAL_SCOPE = "root seal for preserved MSP430F1611 static cross-compile evidence"
TOOLCHAIN_SCOPE_BOUNDARY = (
    "This root seal validates the toolchain files recorded by each preserved build "
    "report. It does not expand that historical inventory to every transitive "
    "compiler, linker, archive, startup object, or system-header input."
)


class SealError(RuntimeError):
    """Raised when the preserved evidence cannot be validated exactly."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SealError(f"Cannot read JSON evidence {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SealError(f"Expected a JSON object: {path}")
    return value


def verify_record(
    record: dict[str, Any], label: str, *, relative_to: Path | None = None
) -> Path:
    path_value = record.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise SealError(f"{label} has no path")
    path = Path(path_value)
    if not path.is_absolute() and relative_to is not None:
        path = relative_to / path
    path = path.resolve()
    if not path.is_file():
        raise SealError(f"{label} is missing: {path}")
    if path.stat().st_size != record.get("size_bytes"):
        raise SealError(f"{label} size differs: {path}")
    if sha256_file(path) != record.get("sha256"):
        raise SealError(f"{label} hash differs: {path}")
    return path


def validate_strict_export_manifest(
    manifest_record: dict[str, Any], student: str, expected_contract: dict[str, Any]
) -> None:
    path_value = manifest_record.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise SealError(f"Strict-export manifest path is absent: {student}")
    manifest_path = Path(path_value)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    manifest_path = manifest_path.resolve()
    try:
        manifest_path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise SealError(f"Strict-export manifest escapes repository root: {manifest_path}") from exc
    if not manifest_path.is_file():
        raise SealError(f"Strict-export manifest is absent: {manifest_path}")
    if sha256_file(manifest_path) != manifest_record.get("sha256"):
        raise SealError(f"Strict-export manifest hash differs: {student}")
    strict_manifest = read_json(manifest_path)
    records = manifest_record.get("files")
    if not isinstance(records, list) or len(records) != 12:
        raise SealError(f"Strict-export nested inventory is absent: {student}")
    expected_common = expected_contract.get("shared_contract", {})
    expected_student = expected_contract.get("students", {}).get(student, {})
    if (
        strict_manifest.get("status") != "passed"
        or strict_manifest.get("student") != student
        or strict_manifest.get("protocol_id") != expected_common.get("protocol_id")
        or strict_manifest.get("export_id") != expected_student.get("export_id")
        or strict_manifest.get("file_count_excluding_manifest") != 12
        or strict_manifest.get("files") != records
    ):
        raise SealError(f"Strict-export manifest semantics differ: {student}")
    declared: set[Path] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise SealError(f"Strict-export record {index} is invalid: {student}")
        path = verify_record(
            record,
            f"{student} strict-export file {index}",
            relative_to=manifest_path.parent,
        )
        try:
            path.relative_to(manifest_path.parent)
        except ValueError as exc:
            raise SealError(f"Strict-export file escapes its directory: {path}") from exc
        if path == manifest_path or path in declared:
            raise SealError(f"Duplicate or self-referential strict-export file: {path}")
        declared.add(path)
    actual = {
        path.resolve()
        for path in manifest_path.parent.iterdir()
        if path.is_file() and path.resolve() != manifest_path
    }
    if actual != declared:
        raise SealError(f"Strict-export on-disk inventory differs: {student}")


def validate_student(
    root: Path, student: str, expected_contract: dict[str, Any]
) -> dict[str, Any]:
    student_dir = (root / student).resolve()
    try:
        student_dir.relative_to(root)
    except ValueError as exc:
        raise SealError(f"Student directory escapes artifact root: {student_dir}") from exc
    if not student_dir.is_dir():
        raise SealError(f"Student artifact directory is absent: {student_dir}")
    report_path = student_dir / "msp430_static_evidence.json"
    report = read_json(report_path)
    if (
        report.get("schema_version") != 1
        or report.get("status") != "success"
        or report.get("student") != student
        or report.get("target") != expected_contract.get("target")
        or report.get("evidence_scope") != expected_contract.get("evidence_scope")
        or report.get("claim_boundary") != expected_contract.get("claim_boundary")
    ):
        raise SealError(f"Invalid completed report identity: {student}")

    declared: set[Path] = set()
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 9:
        raise SealError(f"Artifact inventory is absent: {student}")
    for index, record in enumerate(artifacts):
        if not isinstance(record, dict):
            raise SealError(f"Artifact record {index} is invalid: {student}")
        path = verify_record(record, f"{student} artifact {index}")
        try:
            path.relative_to(student_dir)
        except ValueError as exc:
            raise SealError(f"Artifact escapes the student directory: {path}") from exc
        if path == report_path or path in declared:
            raise SealError(f"Duplicate or self-referential artifact: {path}")
        declared.add(path)
    actual = {
        path.resolve()
        for path in student_dir.iterdir()
        if path.is_file() and path.resolve() != report_path.resolve()
    }
    if actual != declared:
        raise SealError(f"On-disk artifact inventory differs: {student}")

    toolchain = report.get("toolchain")
    if (
        not isinstance(toolchain, dict)
        or not isinstance(toolchain.get("files"), list)
        or len(toolchain["files"]) != 5
    ):
        raise SealError(f"Toolchain inventory is absent: {student}")
    toolchain_paths: set[Path] = set()
    for index, record in enumerate(toolchain["files"]):
        if not isinstance(record, dict):
            raise SealError(f"Toolchain record {index} is invalid: {student}")
        path = verify_record(record, f"{student} recorded toolchain input {index}")
        if path in toolchain_paths:
            raise SealError(f"Duplicate recorded toolchain input: {student}/{path}")
        toolchain_paths.add(path)

    input_validation = report.get("input_validation")
    if (
        not isinstance(input_validation, dict)
        or input_validation.get("passed") is not True
        or not isinstance(input_validation.get("files"), list)
        or len(input_validation["files"]) != 12
        or not isinstance(input_validation.get("manifest"), dict)
    ):
        raise SealError(f"Input-validation evidence is absent: {student}")
    input_paths: set[Path] = set()
    for index, record in enumerate(input_validation["files"]):
        if not isinstance(record, dict):
            raise SealError(f"Input record {index} is invalid: {student}")
        path = verify_record(
            record, f"{student} source/export input {index}", relative_to=REPO_ROOT
        )
        try:
            path.relative_to(REPO_ROOT)
        except ValueError as exc:
            raise SealError(f"Source/export input escapes repository root: {path}") from exc
        if path in input_paths:
            raise SealError(f"Duplicate source/export input: {student}/{path}")
        input_paths.add(path)
    validate_strict_export_manifest(
        input_validation["manifest"], student, expected_contract
    )

    contract = report.get("contract_identity")
    if not isinstance(contract, dict):
        raise SealError(f"Contract identity is absent: {student}")
    expected_common = expected_contract.get("shared_contract", {})
    for field in (
        "protocol_id",
        "seed",
        "dataset_sha256",
        "split_hashes",
        "scaler_sha256",
        "execution_contract_sha256",
        "rf_train_probability_content_sha256",
        "feature_overlap_audit",
    ):
        if contract.get(field) != expected_common.get(field):
            raise SealError(f"Frozen common contract differs for {student}/{field}")
    expected_student = expected_contract.get("students", {}).get(student, {})
    for field in (
        "export_id",
        "model_file_sha256",
        "model_artifact_sha256",
        "dimensions",
        "parameter_bytes",
        "activation_bytes_estimate",
        "macs_per_inference",
    ):
        if contract.get(field) != expected_student.get(field):
            raise SealError(f"Frozen student contract differs for {student}/{field}")
    return {
        "report": report,
        "report_path": report_path,
        "artifact_count": len(artifacts),
        "recorded_toolchain_file_count": len(toolchain["files"]),
        "export_id": contract.get("export_id"),
        "seed": contract.get("seed"),
        "dataset_sha256": contract.get("dataset_sha256"),
        "split_hashes": contract.get("split_hashes"),
        "scaler_sha256": contract.get("scaler_sha256"),
        "target": report.get("target"),
        "evidence_scope": report.get("evidence_scope"),
        "claim_boundary": report.get("claim_boundary"),
        "protocol_id": contract.get("protocol_id"),
        "execution_contract_sha256": contract.get("execution_contract_sha256"),
        "rf_train_probability_content_sha256": contract.get(
            "rf_train_probability_content_sha256"
        ),
        "feature_overlap_audit": contract.get("feature_overlap_audit"),
        "toolchain_inventory": [
            {
                "role": record.get("role"),
                "size_bytes": record.get("size_bytes"),
                "sha256": record.get("sha256"),
            }
            for record in toolchain["files"]
        ],
    }


def file_entry(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def semantic_payload(
    summary: dict[str, Any], students: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    return {
        "scope": ROOT_SEAL_SCOPE,
        "claim_boundary": summary.get("claim_boundary"),
        "evidence_scope": students["student_A"]["evidence_scope"],
        "target": students["student_A"]["target"],
        "students": {
            student: {
                "export_id": students[student]["export_id"],
                "seed": students[student]["seed"],
                "artifact_count_excluding_report": students[student][
                    "artifact_count"
                ],
                "recorded_toolchain_file_count": students[student][
                    "recorded_toolchain_file_count"
                ],
                "report_sha256": sha256_file(students[student]["report_path"]),
            }
            for student in EXPECTED_STUDENTS
        },
        "common_contract": {
            "protocol_id": students["student_A"]["protocol_id"],
            "dataset_sha256": students["student_A"]["dataset_sha256"],
            "split_hashes": students["student_A"]["split_hashes"],
            "scaler_sha256": students["student_A"]["scaler_sha256"],
            "execution_contract_sha256": students["student_A"][
                "execution_contract_sha256"
            ],
            "rf_train_probability_content_sha256": students["student_A"][
                "rf_train_probability_content_sha256"
            ],
            "feature_overlap_audit": students["student_A"][
                "feature_overlap_audit"
            ],
        },
        "expected_contract_sha256": sha256_file(EXPECTED_CONTRACT_PATH),
        "sealer_source_sha256": sha256_file(Path(__file__).resolve()),
        "toolchain_scope_boundary": TOOLCHAIN_SCOPE_BOUNDARY,
    }


def verify_existing_manifest(
    root: Path,
    manifest_path: Path,
    expected_semantics: dict[str, Any],
) -> None:
    manifest = read_json(manifest_path)
    files = manifest.get("files")
    expected_keys = {
        "schema_version",
        "generated_at_utc",
        "status",
        "file_count_excluding_manifest",
        "files",
        *expected_semantics,
    }
    if (
        set(manifest) != expected_keys
        or manifest.get("schema_version") != 1
        or manifest.get("status") != "passed"
        or not isinstance(manifest.get("generated_at_utc"), str)
        or not manifest["generated_at_utc"]
        or not isinstance(files, list)
        or manifest.get("file_count_excluding_manifest") != len(files)
        or any(manifest.get(key) != value for key, value in expected_semantics.items())
    ):
        raise SealError("Existing root manifest identity is invalid")
    declared: set[str] = set()
    for index, record in enumerate(files):
        if not isinstance(record, dict):
            raise SealError(f"Root inventory record {index} is invalid")
        if set(record) != {"path", "size_bytes", "sha256"}:
            raise SealError(f"Root inventory record {index} has unexpected fields")
        relative = record.get("path")
        if not isinstance(relative, str) or not relative or relative in declared:
            raise SealError(f"Root inventory path is invalid: {relative!r}")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise SealError(f"Root inventory path escapes: {relative}")
        path = verify_record(record, f"root artifact {index}", relative_to=root)
        try:
            canonical = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise SealError(f"Root inventory file escapes artifact root: {path}") from exc
        if canonical != relative_path.as_posix() or canonical in declared:
            raise SealError(f"Root inventory path is not canonical: {relative}")
        declared.add(canonical)
    actual: set[str] = set()
    for candidate in root.rglob("*"):
        if not candidate.is_file() or candidate.resolve() == manifest_path.resolve():
            continue
        resolved = candidate.resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise SealError(f"On-disk artifact escapes root: {candidate}") from exc
        actual.add(relative)
    if actual != declared:
        raise SealError("Existing root manifest inventory differs from disk")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    try:
        root.relative_to(MODULE_DIR)
    except ValueError as exc:
        raise SealError(f"Artifact root must remain under {MODULE_DIR}: {root}") from exc
    if not root.is_dir():
        raise SealError(f"Artifact root is absent: {root}")
    manifest_path = root / MANIFEST_NAME
    expected_contract = read_json(EXPECTED_CONTRACT_PATH)
    summary_path = root / "msp430_static_summary.json"
    summary = read_json(summary_path)
    if summary.get("status") != "success" or summary.get("students") != {
        "student_A": "success",
        "student_B": "success",
    } or summary.get("claim_boundary") != expected_contract.get("claim_boundary"):
        raise SealError("The MSP430 root summary is not a complete two-student success")

    students = {
        student: validate_student(root, student, expected_contract)
        for student in EXPECTED_STUDENTS
    }
    for field in (
        "seed",
        "dataset_sha256",
        "split_hashes",
        "scaler_sha256",
        "target",
        "evidence_scope",
        "claim_boundary",
        "protocol_id",
        "execution_contract_sha256",
        "rf_train_probability_content_sha256",
        "feature_overlap_audit",
        "toolchain_inventory",
    ):
        if students["student_A"][field] != students["student_B"][field]:
            raise SealError(f"Cross-student contract differs for {field}")

    semantics = semantic_payload(summary, students)

    if args.verify_existing:
        if not manifest_path.is_file():
            raise SealError(f"Root manifest is absent: {manifest_path}")
        verify_existing_manifest(root, manifest_path, semantics)
        print(manifest_path)
        return 0
    if manifest_path.exists():
        raise SealError(f"Refusing to overwrite existing root manifest: {manifest_path}")

    files = sorted(
        (
            file_entry(root, path)
            for path in root.rglob("*")
            if path.is_file() and path.resolve() != manifest_path.resolve()
        ),
        key=lambda item: item["path"],
    )
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "file_count_excluding_manifest": len(files),
        "files": files,
        **semantics,
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, manifest_path)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SealError as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
