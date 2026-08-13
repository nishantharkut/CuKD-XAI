#!/usr/bin/env python3
"""Build hash-bound FGDS Student A/B cores for MSP430F1611.

The script performs no download and no physical-device operation. It accepts
either extracted TI MSP430 GCC/support roots or local copies of the official
ZIP archives. All extraction and build output is constrained to this module's
directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[2]
CONTRACT_PATH = MODULE_DIR / "expected_contracts.json"
HARNESS_PATH = MODULE_DIR / "msp430_smoke_main.c"
DEFAULT_OUTPUT_DIR = MODULE_DIR / "artifacts"
DEFAULT_CACHE_DIR = MODULE_DIR / "toolchain_cache"

CLAIM_BOUNDARY = (
    "Static MSP430F1611 cross-compile and memory-footprint evidence only. "
    "No physical TelosB execution, latency, energy, radio integration, or "
    "live WSN feature-extraction claim is supported."
)


class EvidenceError(RuntimeError):
    """Raised when an evidence contract or build gate fails."""


class CommandLog(list[dict[str, Any]]):
    """Command list that persists build progress after every invocation."""

    def __init__(self, report: dict[str, Any], report_path: Path) -> None:
        super().__init__()
        self.report = report
        self.report_path = report_path

    def append(self, item: dict[str, Any]) -> None:
        super().append(item)
        write_json(self.report_path, self.report)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, role: str, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    display_path = str(resolved)
    if relative_to is not None:
        try:
            display_path = resolved.relative_to(relative_to.resolve()).as_posix()
        except ValueError:
            pass
    return {
        "role": role,
        "path": display_path,
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise EvidenceError(f"Missing {label}: {path}")
    return path


def require_directory(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_dir():
        raise EvidenceError(f"Missing {label}: {path}")
    return path


def require_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(require_file(path, label))
    if actual.lower() != expected.lower():
        raise EvidenceError(
            f"SHA-256 mismatch for {label}: expected {expected.lower()}, got {actual.lower()}"
        )
    return actual.lower()


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise EvidenceError(f"Contract mismatch for {label}: expected {expected!r}, got {actual!r}")


def ensure_within_module(path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(MODULE_DIR)
    except ValueError as exc:
        raise EvidenceError(f"{label} must remain under {MODULE_DIR}: {resolved}") from exc
    return resolved


def module_output_path(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = MODULE_DIR / path
    return ensure_within_module(path, label)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(require_file(path, "JSON file").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"Cannot parse JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"Expected a JSON object in {path}")
    return value


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_defines(path: Path) -> dict[str, str]:
    defines: dict[str, str] = {}
    pattern = re.compile(r"^\s*#define\s+([A-Za-z_]\w*)\s+(.+?)\s*$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            defines[match.group(1)] = match.group(2)
    return defines


def parse_int_define(defines: dict[str, str], name: str) -> int:
    if name not in defines:
        raise EvidenceError(f"Required C macro is absent: {name}")
    value = defines[name].split("/*", 1)[0].strip()
    try:
        return int(value, 0)
    except ValueError as exc:
        raise EvidenceError(f"C macro {name} is not an integer literal: {value!r}") from exc


def parse_string_define(defines: dict[str, str], name: str) -> str:
    if name not in defines:
        raise EvidenceError(f"Required C macro is absent: {name}")
    value = defines[name].strip()
    if len(value) < 2 or value[0] != '"' or value[-1] != '"':
        raise EvidenceError(f"C macro {name} is not a string literal: {value!r}")
    return value[1:-1]


def validate_manifest(generated_dir: Path, expected_hash: str) -> dict[str, Any]:
    manifest_path = generated_dir / "strict_export_manifest.json"
    require_hash(manifest_path, expected_hash, "strict export manifest")
    manifest = load_json(manifest_path)
    require_equal(manifest.get("status"), "passed", "manifest status")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise EvidenceError("Manifest files field is not a list")
    if manifest.get("file_count_excluding_manifest") != len(files):
        raise EvidenceError("Manifest file count does not equal the number of file entries")
    verified: list[dict[str, Any]] = []
    for entry in files:
        if not isinstance(entry, dict):
            raise EvidenceError("Manifest contains a non-object file entry")
        name = entry.get("path")
        if not isinstance(name, str) or Path(name).name != name:
            raise EvidenceError(f"Unsafe or nested manifest path: {name!r}")
        path = generated_dir / name
        require_hash(path, str(entry.get("sha256", "")), f"manifest file {name}")
        require_equal(path.stat().st_size, entry.get("size_bytes"), f"manifest size for {name}")
        verified.append({"path": name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"path": str(manifest_path), "sha256": sha256_file(manifest_path), "files": verified}


def validate_common_sources(contract: dict[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    common_contract = contract["common_sources"]
    common_dir = require_directory(REPO_ROOT / common_contract["directory"], "common source directory")
    records: list[dict[str, Any]] = []
    for name, expected_hash in sorted(common_contract["files"].items()):
        path = common_dir / name
        require_hash(path, expected_hash, f"common source {name}")
        records.append(file_record(path, f"common_source:{name}", relative_to=REPO_ROOT))
    return common_dir, records


def validate_student(
    student: str,
    contract: dict[str, Any],
    common_records: list[dict[str, Any]],
) -> dict[str, Any]:
    expected = contract["students"][student]
    shared = contract["shared_contract"]
    generated_dir = require_directory(
        REPO_ROOT / expected["generated_directory"], f"{student} generated directory"
    )
    manifest_evidence = validate_manifest(generated_dir, expected["manifest_sha256"])

    report_path = generated_dir / "strict_export_report.json"
    identity_path = generated_dir / "cukd_export_identity.h"
    model_path = generated_dir / "model_weights.h"
    preprocess_path = generated_dir / "preprocess_int_metadata.h"
    require_hash(report_path, expected["strict_report_sha256"], f"{student} strict report")
    require_hash(identity_path, expected["identity_header_sha256"], f"{student} identity header")
    require_hash(model_path, expected["model_header_sha256"], f"{student} model header")
    require_hash(preprocess_path, shared["preprocess_header_sha256"], f"{student} preprocess header")

    report = load_json(report_path)
    provenance = report.get("provenance", {})
    gates = report.get("gates", {})
    teacher = provenance.get("teacher_soft_target_provenance", {})
    require_equal(report.get("status"), "passed", f"{student} strict report status")
    require_equal(report.get("export_id"), expected["export_id"], f"{student} export ID")
    require_equal(
        canonical_json_sha256(report.get("export_identity_payload")),
        expected["export_id"],
        f"{student} canonical export identity hash",
    )
    require_equal(provenance.get("student"), student, f"{student} provenance student")
    require_equal(provenance.get("protocol_id"), shared["protocol_id"], f"{student} protocol")
    require_equal(provenance.get("seed"), shared["seed"], f"{student} seed")
    require_equal(provenance.get("dataset_sha256"), shared["dataset_sha256"], f"{student} dataset")
    require_equal(provenance.get("split_hashes"), shared["split_hashes"], f"{student} split hashes")
    require_equal(provenance.get("scaler_sha256"), shared["scaler_sha256"], f"{student} scaler")
    require_equal(
        provenance.get("execution_contract_sha256"),
        shared["execution_contract_sha256"],
        f"{student} execution contract",
    )
    require_equal(provenance.get("model_file_sha256"), expected["model_file_sha256"], f"{student} model file")
    require_equal(
        provenance.get("model_artifact_sha256"),
        expected["model_artifact_sha256"],
        f"{student} model artifact",
    )
    require_equal(
        teacher.get("train_probability_content_sha256"),
        shared["rf_train_probability_content_sha256"],
        f"{student} RF train probability content",
    )
    require_equal(gates.get("full_test_rows"), shared["full_test_rows"], f"{student} full test rows")
    require_equal(
        provenance.get("calibration_partition"),
        shared["calibration_partition"],
        f"{student} calibration partition",
    )
    require_equal(
        provenance.get("feature_overlap_audit"),
        shared["feature_overlap_audit"],
        f"{student} feature-overlap audit",
    )
    require_equal(
        provenance.get("firmware_common_files"),
        contract["common_sources"]["files"],
        f"{student} recorded common source hashes",
    )

    identity_defines = parse_defines(identity_path)
    require_equal(parse_string_define(identity_defines, "CUKD_EXPORT_ID"), expected["export_id"], f"{student} identity macro")
    require_equal(parse_string_define(identity_defines, "CUKD_STUDENT_ID"), student, f"{student} student macro")

    model_defines = parse_defines(model_path)
    dimensions = [
        parse_int_define(model_defines, "CUKD_INPUT_DIM"),
        parse_int_define(model_defines, "CUKD_H1_DIM"),
        parse_int_define(model_defines, "CUKD_H2_DIM"),
        parse_int_define(model_defines, "CUKD_OUTPUT_DIM"),
    ]
    require_equal(dimensions, expected["dimensions"], f"{student} model dimensions")
    require_equal(parse_int_define(model_defines, "CUKD_PARAM_BYTES"), expected["parameter_bytes"], f"{student} parameter bytes")
    require_equal(
        parse_int_define(model_defines, "CUKD_ACTIVATION_BYTES_EST"),
        expected["activation_bytes_estimate"],
        f"{student} activation estimate",
    )
    require_equal(parse_int_define(model_defines, "CUKD_MACS_PER_INFERENCE"), expected["macs_per_inference"], f"{student} MAC count")

    preprocess_defines = parse_defines(preprocess_path)
    require_equal(parse_int_define(preprocess_defines, "CUKD_PREPROCESS_INPUT_DIM"), shared["input_dim"], f"{student} preprocess input dimension")
    require_equal(parse_int_define(preprocess_defines, "CUKD_PREPROCESS_RAW_Q_FRAC"), shared["preprocess_raw_q_frac"], f"{student} raw Q fraction")
    require_equal(parse_int_define(preprocess_defines, "CUKD_PREPROCESS_OUTPUT_Q_FRAC"), shared["preprocess_output_q_frac"], f"{student} output Q fraction")
    require_equal(parse_int_define(preprocess_defines, "CUKD_PREPROCESS_RIGHT_SHIFT"), shared["preprocess_right_shift"], f"{student} preprocess shift")

    inputs = list(common_records)
    inputs.extend(
        [
            file_record(CONTRACT_PATH, "frozen_expected_contract", relative_to=REPO_ROOT),
            file_record(Path(__file__), "orchestration_script", relative_to=REPO_ROOT),
            file_record(HARNESS_PATH, "static_link_harness", relative_to=REPO_ROOT),
            file_record(generated_dir / "strict_export_manifest.json", "strict_export_manifest", relative_to=REPO_ROOT),
            file_record(report_path, "strict_export_report", relative_to=REPO_ROOT),
            file_record(identity_path, "export_identity_header", relative_to=REPO_ROOT),
            file_record(model_path, "model_weights_header", relative_to=REPO_ROOT),
            file_record(preprocess_path, "integer_preprocess_header", relative_to=REPO_ROOT),
        ]
    )
    return {
        "student": student,
        "generated_dir": generated_dir,
        "export_id": expected["export_id"],
        "protocol_id": shared["protocol_id"],
        "seed": shared["seed"],
        "dataset_sha256": shared["dataset_sha256"],
        "split_hashes": shared["split_hashes"],
        "scaler_sha256": shared["scaler_sha256"],
        "execution_contract_sha256": shared["execution_contract_sha256"],
        "rf_train_probability_content_sha256": shared["rf_train_probability_content_sha256"],
        "model_file_sha256": expected["model_file_sha256"],
        "model_artifact_sha256": expected["model_artifact_sha256"],
        "feature_overlap_audit": shared["feature_overlap_audit"],
        "dimensions": dimensions,
        "parameter_bytes": expected["parameter_bytes"],
        "activation_bytes_estimate": expected["activation_bytes_estimate"],
        "macs_per_inference": expected["macs_per_inference"],
        "manifest_evidence": manifest_evidence,
        "input_files": inputs,
    }


def safe_extract_zip(archive: Path, destination: Path) -> None:
    destination = ensure_within_module(destination, "archive extraction directory")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            member_path = (destination / member.filename).resolve()
            try:
                member_path.relative_to(destination)
            except ValueError as exc:
                raise EvidenceError(f"ZIP member escapes extraction directory: {member.filename}") from exc
        handle.extractall(destination)


def extracted_archive_root(archive: Path, cache_dir: Path, label: str, expected_hash: str | None) -> tuple[Path, dict[str, Any]]:
    archive = require_file(archive, f"{label} archive")
    actual_hash = sha256_file(archive)
    if expected_hash and actual_hash.lower() != expected_hash.lower():
        raise EvidenceError(
            f"SHA-256 mismatch for {label} archive: expected {expected_hash.lower()}, got {actual_hash.lower()}"
        )
    destination = ensure_within_module(cache_dir / f"{label}_{actual_hash[:16]}", f"{label} cache")
    marker = destination / ".archive_sha256"
    if destination.exists():
        if not marker.is_file() or marker.read_text(encoding="ascii").strip() != actual_hash:
            raise EvidenceError(f"Existing {label} cache is not bound to archive {actual_hash}: {destination}")
    else:
        safe_extract_zip(archive, destination)
        marker.write_text(actual_hash + "\n", encoding="ascii")
    return destination, file_record(archive, f"{label}_archive")


def find_tool(root: Path, basename: str) -> Path:
    candidates: list[Path] = []
    for name in (basename, basename + ".exe"):
        candidates.extend(path.resolve() for path in root.rglob(name) if path.is_file())
    candidates = sorted(set(candidates), key=lambda path: (len(path.parts), str(path).lower()))
    if not candidates:
        raise EvidenceError(f"Cannot find {basename} under {root}")
    best_depth = len(candidates[0].parts)
    best = [candidate for candidate in candidates if len(candidate.parts) == best_depth]
    if len(best) != 1:
        raise EvidenceError(f"Ambiguous {basename} candidates at the same depth: {best}")
    return best[0]


def find_linker_script(support_root: Path) -> tuple[Path, list[Path]]:
    candidates = sorted(
        (path.resolve() for path in support_root.rglob("msp430f1611.ld") if path.is_file()),
        key=lambda path: (len(path.parts), str(path).lower()),
    )
    if not candidates:
        raise EvidenceError(f"Cannot find msp430f1611.ld under {support_root}")
    hashes = {sha256_file(path) for path in candidates}
    if len(hashes) != 1:
        raise EvidenceError("Multiple msp430f1611.ld files with different content were found")
    return candidates[0], candidates


def path_evidence_text(paths: Iterable[Path], extra: str = "") -> str:
    return " ".join([extra, *(str(path) for path in paths)]).lower()


def find_support_release_record(support_root: Path, release_marker: str) -> Path:
    candidates = sorted(
        (path.resolve() for path in support_root.rglob("Revisions_Header.txt") if path.is_file()),
        key=lambda path: (len(path.parts), str(path).lower()),
    )
    matching = [
        path
        for path in candidates
        if release_marker.lower() in path.read_text(encoding="utf-8", errors="replace").lower()
    ]
    if not matching:
        raise EvidenceError(
            f"Cannot verify support-files release marker {release_marker} from Revisions_Header.txt"
        )
    hashes = {sha256_file(path) for path in matching}
    if len(hashes) != 1:
        raise EvidenceError("Multiple matching support release records have different content")
    return matching[0]


def command_display(argv: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(argv)
    return shlex.join(argv)


def run_command(
    name: str,
    argv: list[str],
    cwd: Path,
    command_records: list[dict[str, Any]],
    *,
    stdout_file: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update({"LC_ALL": "C", "LANG": "C", "TZ": "UTC", "SOURCE_DATE_EPOCH": "0"})
    completed = subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    record: dict[str, Any] = {
        "name": name,
        "argv": argv,
        "display": command_display(argv),
        "cwd": str(cwd),
        "returncode": completed.returncode,
        "stderr": completed.stderr,
    }
    if stdout_file is None:
        record["stdout"] = completed.stdout
    else:
        stdout_file.write_text(completed.stdout, encoding="utf-8", newline="\n")
        record["stdout_artifact"] = file_record(stdout_file, f"command_stdout:{name}")
    command_records.append(record)
    if completed.returncode != 0:
        raise EvidenceError(f"Command {name!r} failed with exit code {completed.returncode}")
    return completed


def resolve_toolchain(args: argparse.Namespace, contract: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cache_dir = module_output_path(args.cache_dir, "toolchain cache directory")
    archive_records: list[dict[str, Any]] = []
    if args.toolchain_archive:
        toolchain_root, record = extracted_archive_root(
            Path(args.toolchain_archive), cache_dir, "toolchain", args.toolchain_archive_sha256
        )
        archive_records.append(record)
    else:
        toolchain_root = require_directory(Path(args.toolchain_root), "toolchain root")
    if args.support_archive:
        support_root, record = extracted_archive_root(
            Path(args.support_archive), cache_dir, "support", args.support_archive_sha256
        )
        archive_records.append(record)
    else:
        support_root = require_directory(Path(args.support_root), "support root")

    gcc = find_tool(toolchain_root, "msp430-elf-gcc")
    size = find_tool(toolchain_root, "msp430-elf-size")
    objdump = find_tool(toolchain_root, "msp430-elf-objdump")
    linker_script, linker_candidates = find_linker_script(support_root)

    version_commands: list[dict[str, Any]] = []
    gcc_version = run_command("gcc_version", [str(gcc), "--version"], MODULE_DIR, version_commands)
    size_version = run_command("size_version", [str(size), "--version"], MODULE_DIR, version_commands)
    objdump_version = run_command("objdump_version", [str(objdump), "--version"], MODULE_DIR, version_commands)
    expected_toolchain = contract["toolchain"]
    version_text = gcc_version.stdout + gcc_version.stderr
    if "msp430" not in version_text.lower():
        raise EvidenceError("Compiler version output does not identify an MSP430 compiler")
    if expected_toolchain["required_gcc_version_substring"] not in version_text:
        raise EvidenceError(
            "Compiler version does not match the frozen TI MSP430 GCC release: "
            + version_text.strip()
        )
    size_version_text = size_version.stdout + size_version.stderr
    objdump_version_text = objdump_version.stdout + objdump_version.stderr
    required_binutils = expected_toolchain["required_binutils_version_substring"]
    if required_binutils not in size_version_text or required_binutils not in objdump_version_text:
        raise EvidenceError(
            f"Binutils version does not match required release {required_binutils}: "
            f"size={size_version_text.strip()!r}, objdump={objdump_version_text.strip()!r}"
        )
    release_context = path_evidence_text(
        [toolchain_root, Path(args.toolchain_archive) if args.toolchain_archive else toolchain_root],
        version_text,
    )
    if expected_toolchain["required_toolchain_release_marker"].lower() not in release_context:
        raise EvidenceError(
            f"Cannot verify toolchain release marker {expected_toolchain['required_toolchain_release_marker']} "
            "from the archive/root path or compiler version output"
        )
    support_release_record = find_support_release_record(
        support_root,
        expected_toolchain["required_support_release_marker"],
    )

    tools = [
        file_record(gcc, "msp430_gcc"),
        file_record(size, "msp430_size"),
        file_record(objdump, "msp430_objdump"),
        file_record(linker_script, "msp430f1611_linker_script"),
        file_record(support_release_record, "support_files_release_record"),
        *archive_records,
    ]
    if args.toolchain_archive and args.support_archive:
        input_mode = "archives"
    elif args.toolchain_root and args.support_root:
        input_mode = "extracted_roots"
    else:
        input_mode = "mixed_archive_and_root"
    return {
        "mode": input_mode,
        "toolchain_root": str(toolchain_root),
        "support_root": str(support_root),
        "gcc": str(gcc),
        "size": str(size),
        "objdump": str(objdump),
        "linker_script": str(linker_script),
        "linker_script_candidates": [str(path) for path in linker_candidates],
        "support_release_record": file_record(
            support_release_record, "support_files_release_record"
        ),
        "gcc_version": version_text.strip(),
        "size_version": size_version_text.strip(),
        "objdump_version": objdump_version_text.strip(),
        "files": tools,
        "version_commands": version_commands,
    }, tools


def parse_berkeley_size(output: str) -> dict[str, Any]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    for line in reversed(lines):
        fields = line.split()
        if len(fields) >= 6 and all(re.fullmatch(r"[0-9]+", value) for value in fields[:4]):
            return {
                "text": int(fields[0]),
                "data": int(fields[1]),
                "bss": int(fields[2]),
                "dec": int(fields[3]),
                "hex": fields[4],
                "filename": " ".join(fields[5:]),
            }
    raise EvidenceError(f"Cannot parse Berkeley size output:\n{output}")


def parse_sysv_sections(output: str) -> dict[str, int]:
    sections: dict[str, int] = {}
    pattern = re.compile(r"^(\S+)\s+([0-9]+)\s+(?:0x[0-9A-Fa-f]+|[0-9]+)\s*$")
    for raw_line in output.splitlines():
        match = pattern.match(raw_line.strip())
        if match and match.group(1).lower() not in {"section", "total"}:
            sections[match.group(1)] = int(match.group(2))
    if not sections:
        raise EvidenceError(f"Cannot parse SysV section output:\n{output}")
    return sections


def parse_stack_usage(paths: list[Path]) -> dict[str, Any]:
    pattern = re.compile(r"^(.*):(\d+):(\d+):([^\t]+)\t(\d+)\t(.+)$")
    entries: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    for path in sorted(paths, key=lambda item: str(item).lower()):
        files.append(file_record(path, "compiler_stack_usage"))
        for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            match = pattern.match(line)
            if not match:
                raise EvidenceError(f"Cannot parse {path}:{line_number}: {line!r}")
            entries.append(
                {
                    "source": match.group(1),
                    "line": int(match.group(2)),
                    "column": int(match.group(3)),
                    "function": match.group(4),
                    "bytes": int(match.group(5)),
                    "qualifier": match.group(6).strip(),
                    "evidence_file": str(path),
                }
            )
    if not entries:
        raise EvidenceError("The compiler emitted no parseable stack-usage entries")
    expected_functions = [
        "main",
        "cukd_standardize_raw_q",
        "cukd_dense_i8_q15",
        "cukd_forward_q15",
        "cukd_predict_q15",
    ]
    observed_functions = sorted({entry["function"] for entry in entries})
    missing_functions = [
        function
        for function in expected_functions
        if not any(
            observed == function or observed.startswith(function + ".")
            for observed in observed_functions
        )
    ]
    if missing_functions:
        raise EvidenceError(
            "Compiler stack evidence is incomplete for project functions: "
            + ", ".join(missing_functions)
        )
    return {
        "flag": "-fstack-usage",
        "files": files,
        "entries": entries,
        "expected_project_functions": expected_functions,
        "observed_functions": observed_functions,
        "missing_expected_functions": [],
        "maximum_single_function_bytes": max(entry["bytes"] for entry in entries),
        "interpretation": (
            "Compiler-reported per-function static stack usage. These entries are not summed as a "
            "whole-program peak because the report does not prove a complete target call graph. "
            "Interrupt nesting and any OS/network stack remain excluded."
        ),
    }


def helper_symbols(disassembly: str) -> list[str]:
    symbols = set(
        re.findall(r"<((?:__mspabi|__mul|__div|__mod)[A-Za-z0-9_]*)>", disassembly)
    )
    return sorted(symbols)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def base_report(student_contract: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "status": "running",
        "student": student_contract["student"],
        "target": contract["target"],
        "evidence_scope": contract["evidence_scope"],
        "claim_boundary": CLAIM_BOUNDARY,
        "contract_identity": {
            "export_id": student_contract["export_id"],
            "protocol_id": student_contract["protocol_id"],
            "seed": student_contract["seed"],
            "dataset_sha256": student_contract["dataset_sha256"],
            "split_hashes": student_contract["split_hashes"],
            "scaler_sha256": student_contract["scaler_sha256"],
            "execution_contract_sha256": student_contract["execution_contract_sha256"],
            "rf_train_probability_content_sha256": student_contract[
                "rf_train_probability_content_sha256"
            ],
            "model_file_sha256": student_contract["model_file_sha256"],
            "model_artifact_sha256": student_contract["model_artifact_sha256"],
            "feature_overlap_audit": student_contract["feature_overlap_audit"],
            "dimensions": student_contract["dimensions"],
            "parameter_bytes": student_contract["parameter_bytes"],
            "activation_bytes_estimate": student_contract["activation_bytes_estimate"],
            "macs_per_inference": student_contract["macs_per_inference"],
        },
        "input_validation": {
            "passed": True,
            "manifest": student_contract["manifest_evidence"],
            "files": student_contract["input_files"],
        },
        "commands": [],
    }


def build_student(
    student_contract: dict[str, Any],
    contract: dict[str, Any],
    toolchain: dict[str, Any],
    output_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    student = student_contract["student"]
    build_dir = ensure_within_module(output_dir / student, f"{student} build directory")
    report_path = build_dir / "msp430_static_evidence.json"
    if build_dir.exists():
        if not overwrite:
            raise EvidenceError(f"Build directory already exists; use --overwrite to replace it: {build_dir}")
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    report = base_report(student_contract, contract)
    report["toolchain"] = {key: value for key, value in toolchain.items() if key != "version_commands"}
    commands = CommandLog(report, report_path)
    report["commands"] = commands
    commands.extend(toolchain["version_commands"])
    write_json(report_path, report)
    generated_dir = student_contract["generated_dir"]
    common_dir = REPO_ROOT / contract["common_sources"]["directory"]

    gcc = toolchain["gcc"]
    size = toolchain["size"]
    objdump = toolchain["objdump"]
    compiler_flags = [
        "-mmcu=msp430f1611",
        "-std=c99",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-Os",
        "-ffunction-sections",
        "-fdata-sections",
        "-fstack-usage",
        f"-I{generated_dir}",
        f"-I{common_dir}",
    ]
    sources = [
        ("cukd_model", common_dir / "cukd_model.c"),
        ("cukd_preprocess", common_dir / "cukd_preprocess.c"),
        ("msp430_smoke_main", HARNESS_PATH),
    ]
    objects: list[Path] = []
    for name, source in sources:
        object_path = build_dir / f"{name}.o"
        run_command(
            f"compile_{name}",
            [gcc, *compiler_flags, "-c", str(source), "-o", str(object_path)],
            build_dir,
            commands,
        )
        objects.append(object_path)

    elf_path = build_dir / f"cukd_{student}_msp430f1611.elf"
    map_path = build_dir / f"cukd_{student}_msp430f1611.map"
    run_command(
        "link",
        [
            gcc,
            "-mmcu=msp430f1611",
            "-Os",
            "-Wl,--gc-sections",
            f"-Wl,-Map,{map_path}",
            f"-L{Path(toolchain['linker_script']).parent}",
            *(str(path) for path in objects),
            "-o",
            str(elf_path),
        ],
        build_dir,
        commands,
    )

    summary_output = run_command(
        "size_summary", [size, str(elf_path)], build_dir, commands
    ).stdout
    section_output = run_command(
        "size_sections", [size, "-A", str(elf_path)], build_dir, commands
    ).stdout
    object_sections: dict[str, dict[str, int]] = {}
    for object_path in objects:
        output = run_command(
            f"size_object_{object_path.stem}",
            [size, "-A", str(object_path)],
            build_dir,
            commands,
        ).stdout
        object_sections[object_path.name] = parse_sysv_sections(output)

    disassembly_path = build_dir / f"cukd_{student}_msp430f1611_disassembly.txt"
    run_command(
        "objdump_disassembly",
        [objdump, "-d", str(elf_path)],
        build_dir,
        commands,
        stdout_file=disassembly_path,
    )
    disassembly = disassembly_path.read_text(encoding="utf-8", errors="replace")
    stack_paths = list(build_dir.glob("*.su"))
    stack = parse_stack_usage(stack_paths)
    summary = parse_berkeley_size(summary_output)
    sections = parse_sysv_sections(section_output)
    flash_budget = int(contract["target"]["flash_budget_bytes"])
    ram_budget = int(contract["target"]["ram_budget_bytes"])

    artifacts = [
        file_record(path, "cross_compile_artifact")
        for path in sorted(build_dir.iterdir(), key=lambda item: item.name.lower())
        if path.is_file() and path != report_path
    ]
    report.update(
        {
            "status": "success",
            "completed_at_utc": utc_now(),
            "compiler_flags": compiler_flags,
            "linked_footprint": {
                "berkeley_summary": summary,
                "sections": sections,
                "object_sections": object_sections,
                "memory_budget_context": {
                    "nominal_flash_bytes": flash_budget,
                    "nominal_ram_bytes": ram_budget,
                    "static_flash_load_bytes": summary["text"] + summary["data"],
                    "static_ram_lower_bound_bytes": summary["data"] + summary["bss"],
                    "static_flash_load_fraction_of_nominal_flash": (
                        summary["text"] + summary["data"]
                    )
                    / flash_budget,
                    "static_ram_lower_bound_fraction_of_nominal_ram": (
                        summary["data"] + summary["bss"]
                    )
                    / ram_budget,
                    "memory_accounting_boundary": (
                        "Static flash load is GNU size text + data. Static RAM lower bound is "
                        "data + bss and excludes call-chain stack, interrupt nesting, operating-system "
                        "state, and network-stack state."
                    ),
                    "warning": (
                        "The Berkeley text field combines code and read-only data. Static data plus BSS "
                        "does not include runtime stack, interrupt state, or an OS/network stack."
                    ),
                },
            },
            "stack_evidence": stack,
            "wide_arithmetic_helper_symbols": helper_symbols(disassembly),
            "artifacts": artifacts,
            "claim_boundary": CLAIM_BOUNDARY,
        }
    )
    write_json(report_path, report)
    return report


def failure_report(
    student: str,
    contract: dict[str, Any],
    output_dir: Path,
    exc: BaseException,
    partial: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report_path = ensure_within_module(
        output_dir / student / "msp430_static_evidence.json", f"{student} failure report"
    )
    if partial is None and report_path.is_file():
        partial = load_json(report_path)
    report = partial or {
        "schema_version": 1,
        "student": student,
        "target": contract["target"],
        "evidence_scope": contract["evidence_scope"],
        "commands": [],
    }
    report.update(
        {
            "status": "failure",
            "completed_at_utc": utc_now(),
            "claim_boundary": CLAIM_BOUNDARY,
            "failure": {"type": type(exc).__name__, "message": str(exc)},
        }
    )
    build_dir = report_path.parent
    if build_dir.is_dir():
        report["available_partial_artifacts"] = [
            file_record(path, "partial_cross_compile_artifact")
            for path in sorted(build_dir.iterdir(), key=lambda item: item.name.lower())
            if path.is_file() and path != report_path
        ]
    write_json(report_path, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create current FGDS Student A/B MSP430F1611 static cross-compile evidence."
    )
    toolchain = parser.add_mutually_exclusive_group()
    toolchain.add_argument("--toolchain-root", help="Extracted TI MSP430 GCC 9.3.1.11 root")
    toolchain.add_argument("--toolchain-archive", help="Local official msp430-gcc-9.3.1.11_win64.zip")
    support = parser.add_mutually_exclusive_group()
    support.add_argument("--support-root", help="Extracted TI MSP430 GCC support-files 1.212 root")
    support.add_argument("--support-archive", help="Local official msp430-gcc-support-files-1.212.zip")
    parser.add_argument("--toolchain-archive-sha256", help="Optional expected SHA-256 for the local toolchain ZIP")
    parser.add_argument("--support-archive-sha256", help="Optional expected SHA-256 for the local support ZIP")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory under this module")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="Archive extraction cache under this module")
    parser.add_argument("--overwrite", action="store_true", help="Replace this module's existing per-student build directories")
    parser.add_argument(
        "--verify-inputs-only",
        action="store_true",
        help="Verify frozen FGDS/common-source contracts and exit without resolving tools or compiling",
    )
    args = parser.parse_args(argv)
    if not args.verify_inputs_only:
        if not (args.toolchain_root or args.toolchain_archive):
            parser.error("one of --toolchain-root or --toolchain-archive is required")
        if not (args.support_root or args.support_archive):
            parser.error("one of --support-root or --support-archive is required")
    if args.toolchain_archive_sha256 and not args.toolchain_archive:
        parser.error("--toolchain-archive-sha256 requires --toolchain-archive")
    if args.support_archive_sha256 and not args.support_archive:
        parser.error("--support-archive-sha256 requires --support-archive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    contract = load_json(CONTRACT_PATH)
    require_equal(contract.get("schema_version"), 1, "expected-contract schema version")
    require_equal(contract.get("claim_boundary"), CLAIM_BOUNDARY, "claim boundary")
    common_dir, common_records = validate_common_sources(contract)
    del common_dir
    validated: dict[str, dict[str, Any]] = {}
    for student in ("student_A", "student_B"):
        validated[student] = validate_student(student, contract, common_records)

    if args.verify_inputs_only:
        summary = {
            "status": "verified",
            "compilation_run": False,
            "claim_boundary": CLAIM_BOUNDARY,
            "students": {
                name: {
                    "export_id": value["export_id"],
                    "protocol_id": value["protocol_id"],
                    "seed": value["seed"],
                    "dimensions": value["dimensions"],
                    "manifest_files_verified": len(value["manifest_evidence"]["files"]),
                }
                for name, value in validated.items()
            },
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    output_dir = module_output_path(args.output_dir, "output directory")
    if not args.overwrite:
        existing = [
            output_dir / student
            for student in ("student_A", "student_B")
            if (output_dir / student).exists()
        ]
        if existing:
            raise EvidenceError(
                "Existing per-student output will not be modified without --overwrite: "
                + ", ".join(str(path) for path in existing)
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        toolchain, _ = resolve_toolchain(args, contract)
    except Exception as exc:
        for student in ("student_A", "student_B"):
            failure_report(student, contract, output_dir, exc)
        raise

    statuses: dict[str, str] = {}
    for student in ("student_A", "student_B"):
        try:
            result = build_student(validated[student], contract, toolchain, output_dir, args.overwrite)
        except Exception as exc:
            failure_report(student, contract, output_dir, exc)
            statuses[student] = "failure"
            print(f"{student}: failure: {exc}", file=sys.stderr)
        else:
            statuses[student] = result["status"]
            print(f"{student}: {result['status']}")
    summary = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "status": "success" if all(value == "success" for value in statuses.values()) else "failure",
        "students": statuses,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    write_json(output_dir / "msp430_static_summary.json", summary)
    return 0 if summary["status"] == "success" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvidenceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
