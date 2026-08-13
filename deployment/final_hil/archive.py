"""Original-path-independent archive for final HIL campaign evidence.

The source evidence records intentionally remain byte-for-byte unchanged.  Their
absolute paths are historical provenance only; ``semantic_map`` binds each such
record to an archive-local copy used by the deep verifier.
"""

from __future__ import annotations

import csv
import math
import os
import shutil
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

from .bundles import verify_final_bundle
from .contracts import (
    BOARDS,
    EXPECTED_FULL_ROWS,
    FINAL_STAGES,
    MODEL_KEYS,
    TRANSPORTS,
    FinalExportIdentity,
    Verifier,
    _is_sha256,
    atomic_write_json,
    canonical_json_sha256,
    read_json,
    sha256_file,
    validate_balanced_cohort,
    validate_blocked_audit,
    validate_campaign_contract,
    validate_final_export,
)
from .evidence import (
    CAMPAIGN_EVIDENCE_SCHEMA,
    SESSION_SCHEMA,
    TIMING_STATISTICAL_UNIT,
    validate_build_upload_provenance,
    validate_campaign_session_ledger,
    validate_session_connection_record,
    validate_session_completion,
    validate_session_stage_ledger,
    verify_complete_campaign,
)
from .runtime import (
    ATTEMPT_SCHEMA,
    _attempt_wifi_binding,
    _attempt_payload_hash,
    _bundle_source_options,
    _read_response_csv,
    _recovery_contract,
    _started_payload_hash,
    _validate_attempt_connection_set,
    _validate_attempt_payload_shape,
    load_stage_dataset,
    require_session_id,
    validate_host_environment,
    verify_response_records,
    verify_stage_attempt,
)


ARCHIVE_SCHEMA = "cukd_final_hil_portable_archive_v2"
ARCHIVE_MANIFEST = "archive_manifest.json"
REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_EXECUTION_SOURCE_PATHS = (
    "deployment/final_hil/__init__.py",
    "deployment/final_hil/__main__.py",
    "deployment/final_hil/archive.py",
    "deployment/final_hil/campaign.py",
    "deployment/final_hil/contracts.py",
    "deployment/final_hil/bundles.py",
    "deployment/final_hil/runtime.py",
    "deployment/final_hil/evidence.py",
    "deployment/firmware_export/wsnds_final_hil/export_final_seed42.py",
    "deployment/firmware_export/wsnds_rfkd_hil/export_fgds_seed42_deployment.py",
    "deployment/firmware_export/wsnds_rfkd_hil/export_wsnds_student_a_rfkd_int8.py",
    "deployment/firmware_export/wsnds_rfkd_hil/wsnds_train_only_self_test.c",
    "experiments/wsnds/leakage_free_rerun/tier15_common.py",
    "deployment/hardware_hil/host/hil_common.py",
    "deployment/hardware_hil/host/generate_fgds_balanced_timing_cohort.py",
    "deployment/hardware_hil/host/stream_vectors.py",
    "deployment/hardware_hil/host/stream_vectors_fgds_strict.py",
    "deployment/hardware_hil/firmware/common/cukd_model.h",
    "deployment/hardware_hil/firmware/common/cukd_model.c",
    "deployment/hardware_hil/firmware/common/cukd_preprocess.h",
    "deployment/hardware_hil/firmware/common/cukd_preprocess.c",
    "deployment/hardware_hil/firmware/common/cukd_protocol.h",
    "deployment/hardware_hil/firmware/common/cukd_protocol.c",
    "deployment/hardware_hil/firmware/esp32c3/src/main.cpp",
    "deployment/hardware_hil/firmware/arduino_r4/cukd_hil_r4/cukd_hil_r4.ino",
    "deployment/wireless_hil/host/udp_session.py",
    "deployment/wireless_hil/host/wireless_common.py",
    "deployment/wireless_hil/firmware/common/cukd_wifi_config.h",
    "deployment/wireless_hil/firmware/common/cukd_wifi_config.c",
    "deployment/wireless_hil/firmware/common/cukd_wifi_envelope.h",
    "deployment/wireless_hil/firmware/common/cukd_wifi_envelope.c",
    "deployment/wireless_hil/firmware/cukd_wireless_fgds/cukd_wireless_fgds.ino",
)
FINAL_EXPORT_SOURCE_SNAPSHOTS = {
    "source_snapshot/python/export_final_seed42.py",
    "source_snapshot/python/export_fgds_seed42_deployment.py",
    "source_snapshot/python/export_wsnds_student_a_rfkd_int8.py",
    "source_snapshot/python/tier15_common.py",
    "source_snapshot/c/wsnds_train_only_self_test.c",
    "source_snapshot/c/cukd_model.c",
    "source_snapshot/c/cukd_model.h",
    "source_snapshot/c/cukd_preprocess.c",
    "source_snapshot/c/cukd_preprocess.h",
}
_BLOCKED_VERIFIER_LOCK = threading.Lock()


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("Archive evidence timestamp must use a zero UTC offset")
    return parsed


def _safe_local(root: Path, relative: str) -> Path:
    path = Path(relative)
    if (
        not relative
        or "\\" in relative
        or path.as_posix() != relative
        or path.is_absolute()
        or path.drive
        or ".." in path.parts
    ):
        raise RuntimeError(f"Unsafe archive-relative path: {relative!r}")
    source_root = Path(root)
    if source_root.is_symlink():
        raise RuntimeError(f"Archive root cannot be a symlink: {source_root}")
    resolved_root = source_root.resolve()
    if not resolved_root.is_dir():
        raise NotADirectoryError(resolved_root)
    candidate = resolved_root
    for part in path.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise RuntimeError(f"Archive path traverses a symlink: {relative!r}")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError(f"Archive path escapes its root: {relative!r}") from exc
    return resolved


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _copy_file(source: Path, destination: Path) -> None:
    original = Path(source)
    if original.is_symlink():
        raise RuntimeError(f"Archive source cannot be a symlink: {original}")
    source = original.resolve()
    if not source.is_file():
        raise RuntimeError(f"Archive source is not a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Duplicate archive destination: {destination}")
    shutil.copyfile(source, destination)


def _tree_files(root: Path) -> list[Path]:
    source_root = Path(root)
    if source_root.is_symlink():
        raise RuntimeError(f"Archive tree root cannot be a symlink: {source_root}")
    root = source_root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    files: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise RuntimeError(f"Symlinks are not admissible archive evidence: {path}")
        if path.is_file():
            files.append(path)
    return files


def _copy_tree(source: Path, destination: Path) -> list[str]:
    original = Path(source)
    if original.is_symlink():
        raise RuntimeError(f"Archive tree source cannot be a symlink: {original}")
    source = original.resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Duplicate archive destination: {destination}")
    destination.mkdir(parents=True)
    copied: list[str] = []
    for member in _tree_files(source):
        relative = member.relative_to(source)
        target = destination / relative
        _copy_file(member, target)
        copied.append(relative.as_posix())
    return copied


def _inventory(root: Path) -> list[dict[str, Any]]:
    manifest = (root / ARCHIVE_MANIFEST).resolve()
    return [
        {
            "path": _relative(root, path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in _tree_files(root)
        if path.resolve() != manifest
    ]


def _semantic_file_set(semantic: Mapping[str, Any]) -> set[str]:
    owned: set[str] = set()

    def add(relative: Any) -> None:
        if not isinstance(relative, str) or not relative:
            raise RuntimeError("Archive semantic map contains a missing file path")
        path = Path(relative)
        if (
            "\\" in relative
            or path.as_posix() != relative
            or path.is_absolute()
            or path.drive
            or ".." in path.parts
        ):
            raise RuntimeError(f"Archive semantic map contains an unsafe path: {relative!r}")
        normalized = path.as_posix()
        if normalized in owned:
            raise RuntimeError(f"Archive file has multiple semantic owners: {normalized}")
        owned.add(normalized)

    campaign = semantic.get("campaign")
    if not isinstance(campaign, Mapping):
        raise RuntimeError("Archive campaign semantic map is missing")
    add(campaign.get("contract_local_path"))
    if campaign.get("evidence_local_path") is not None:
        add(campaign.get("evidence_local_path"))

    models = semantic.get("models")
    if not isinstance(models, Mapping):
        raise RuntimeError("Archive model semantic map is missing")
    for model in models.values():
        if not isinstance(model, Mapping):
            raise RuntimeError("Archive model semantic map is malformed")
        if model.get("status") == "passed":
            base = Path(str(model.get("export_local_dir")))
            members = model.get("files_relative_to_export")
            if not isinstance(members, list):
                raise RuntimeError("Archive export inventory is missing")
            for member in members:
                add((base / str(member)).as_posix())
        elif model.get("status") == "blocked" and model.get(
            "blocked_audit_local_path"
        ) is not None:
            add(model.get("blocked_audit_local_path"))

    cohort = semantic.get("cohort")
    if cohort is not None:
        if not isinstance(cohort, Mapping):
            raise RuntimeError("Archive cohort semantic map is malformed")
        base = Path(str(cohort.get("local_dir")))
        members = cohort.get("files_relative_to_cohort")
        if not isinstance(members, list):
            raise RuntimeError("Archive cohort inventory is missing")
        for member in members:
            add((base / str(member)).as_posix())

    combinations = semantic.get("combinations")
    if not isinstance(combinations, Mapping):
        raise RuntimeError("Archive combination semantic map is malformed")
    for combination in combinations.values():
        if not isinstance(combination, Mapping):
            raise RuntimeError("Archive combination semantic map is malformed")
        add(combination.get("session_local_path"))
        bundle_base = Path(str(combination.get("bundle_local_dir")))
        bundle_members = combination.get("bundle_files_relative")
        if not isinstance(bundle_members, list):
            raise RuntimeError("Archive bundle inventory is missing")
        for member in bundle_members:
            add((bundle_base / str(member)).as_posix())
        provenance = combination.get("provenance")
        if not isinstance(provenance, Mapping):
            raise RuntimeError("Archive provenance semantic map is missing")
        provenance_base = Path(str(provenance.get("local_dir")))
        provenance_members = provenance.get("files_relative_to_provenance")
        if not isinstance(provenance_members, list):
            raise RuntimeError("Archive provenance inventory is missing")
        for member in provenance_members:
            add((provenance_base / str(member)).as_posix())
        connection = combination.get("connection")
        if connection is not None:
            if not isinstance(connection, Mapping):
                raise RuntimeError("Archive connection semantic map is malformed")
            add(connection.get("local_path"))
        attempts = combination.get("attempts")
        if not isinstance(attempts, list):
            raise RuntimeError("Archive attempt semantic map is missing")
        for attempt in attempts:
            if not isinstance(attempt, Mapping):
                raise RuntimeError("Archive attempt semantic map is malformed")
            attempt_base = Path(str(attempt.get("local_dir")))
            attempt_members = attempt.get("files_relative_to_attempt")
            if not isinstance(attempt_members, list):
                raise RuntimeError("Archive attempt inventory is missing")
            for member in attempt_members:
                add((attempt_base / str(member)).as_posix())

    sources = semantic.get("host_sources")
    if not isinstance(sources, list) or any(
        not isinstance(source, Mapping) for source in sources
    ):
        raise RuntimeError("Archive host-source semantic map is malformed")

    execution_sources = semantic.get("archive_execution_sources")
    if not isinstance(execution_sources, list):
        raise RuntimeError("Archive execution-source semantic map is malformed")
    for source in execution_sources:
        if not isinstance(source, Mapping):
            raise RuntimeError("Archive execution-source entry is malformed")
        add(source.get("local_path"))

    blocked_verifier = semantic.get("blocked_verifier")
    if blocked_verifier is not None:
        if not isinstance(blocked_verifier, Mapping):
            raise RuntimeError("Archive blocked-verifier semantic map is malformed")
        base = Path(str(blocked_verifier.get("repository_local_root")))
        members = blocked_verifier.get("files_relative_to_repository")
        if not isinstance(members, list):
            raise RuntimeError("Archive blocked-verifier inventory is missing")
        for member in members:
            add((base / str(member)).as_posix())
    return owned


def _verify_seal(payload: Mapping[str, Any], field: str, label: str) -> None:
    copy = dict(payload)
    recorded = copy.pop(field, None)
    if recorded != canonical_json_sha256(copy):
        raise RuntimeError(f"{label} canonical payload ID is invalid")


def _validate_campaign_evidence_header(
    payload: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    _verify_seal(payload, "campaign_evidence_id", "Campaign evidence")
    expected_status = (
        "passed_with_blocked_routes" if contract.get("blocked_routes") else "passed"
    )
    if (
        payload.get("schema") != CAMPAIGN_EVIDENCE_SCHEMA
        or payload.get("status") != expected_status
    ):
        raise RuntimeError("Campaign evidence is not a passed final campaign")
    if payload.get("contract_id") != contract.get("contract_id"):
        raise RuntimeError("Campaign evidence belongs to another contract")


def _portable_campaign_evidence_equal(
    recorded: Mapping[str, Any], recomputed: Mapping[str, Any]
) -> bool:
    recorded_copy = dict(recorded)
    recomputed_copy = dict(recomputed)
    recorded_copy.pop("campaign_evidence_id", None)
    recomputed_copy.pop("campaign_evidence_id", None)
    recorded_sessions = recorded_copy.get("sessions")
    recomputed_sessions = recomputed_copy.get("sessions")
    if not isinstance(recorded_sessions, list) or not isinstance(
        recomputed_sessions, list
    ):
        return False
    if len(recorded_sessions) != len(recomputed_sessions):
        return False
    recorded_copy["sessions"] = []
    recomputed_copy["sessions"] = []
    for recorded_item, recomputed_item in zip(
        recorded_sessions, recomputed_sessions
    ):
        if not isinstance(recorded_item, Mapping) or not isinstance(
            recomputed_item, Mapping
        ):
            return False
        left = dict(recorded_item)
        right = dict(recomputed_item)
        if not isinstance(left.pop("session_path_recorded", None), str):
            return False
        if not isinstance(right.pop("session_path_recorded", None), str):
            return False
        recorded_copy["sessions"].append(left)
        recomputed_copy["sessions"].append(right)
    return recorded_copy == recomputed_copy


def _relocate_recorded_path(
    recorded: str,
    *,
    source_roots: Mapping[str, Path],
    expected_kind: str,
) -> Path:
    if not isinstance(recorded, str) or not recorded:
        raise RuntimeError("Historical evidence path is missing")
    normalized = recorded.replace("\\", "/").rstrip("/")
    if not (
        normalized.startswith("/")
        or (
            len(normalized) >= 3
            and normalized[0].isalpha()
            and normalized[1:3] == ":/"
        )
    ):
        raise RuntimeError(f"Historical evidence path is not absolute: {recorded}")
    if expected_kind not in {"file", "dir"}:
        raise ValueError(f"Unsupported relocated evidence kind: {expected_kind}")
    matches: list[Path] = []
    for old_root, local_root in source_roots.items():
        old = str(old_root).replace("\\", "/").rstrip("/")
        if not old or not (
            normalized == old or normalized.startswith(old + "/")
        ):
            continue
        suffix = normalized[len(old) :].lstrip("/")
        local = Path(local_root)
        candidate = local.resolve()
        if suffix:
            candidate = candidate.joinpath(*suffix.split("/"))
        relative_parts = Path(suffix).parts if suffix else ()
        probe = local.resolve()
        for part in relative_parts:
            probe = probe / part
            if probe.is_symlink():
                raise RuntimeError(
                    f"Relocated evidence path traverses a symlink: {recorded}"
                )
        resolved_candidate = candidate.resolve()
        try:
            resolved_candidate.relative_to(local.resolve())
        except ValueError as exc:
            raise RuntimeError(
                f"Relocated evidence path escapes its local root: {recorded}"
            ) from exc
        matches.append(resolved_candidate)
    if len(matches) != 1:
        raise RuntimeError(
            f"Historical path resolves through {len(matches)} source-root mappings: "
            f"{recorded}"
        )
    resolved = matches[0]
    if expected_kind == "file" and not resolved.is_file():
        raise RuntimeError(f"Relocated evidence file is unavailable: {recorded}")
    if expected_kind == "dir" and not resolved.is_dir():
        raise RuntimeError(f"Relocated evidence directory is unavailable: {recorded}")
    return resolved


def _collect_source_reference(
    references: dict[str, dict[str, Any]],
    digest: Any,
    *,
    role: str,
    record_path: Path,
    expected_relative_path: str,
) -> None:
    if not _is_sha256(digest):
        raise RuntimeError(f"Malformed {role} source hash in {record_path}")
    relative = Path(expected_relative_path)
    if (
        not expected_relative_path
        or "\\" in expected_relative_path
        or relative.as_posix() != expected_relative_path
        or relative.is_absolute()
        or relative.drive
        or ".." in relative.parts
    ):
        raise RuntimeError(f"Unsafe host-source path: {expected_relative_path!r}")
    key = f"path:{expected_relative_path}"
    item = references.setdefault(
        key,
        {
            "sha256": str(digest),
            "expected_relative_path": expected_relative_path,
            "references": [],
        },
    )
    if item["sha256"] != digest:
        raise RuntimeError(
            f"Host-source path {expected_relative_path!r} has conflicting recorded hashes"
        )
    item["references"].append(
        {"role": role, "record_path_original": str(record_path.resolve())}
    )


def _collect_environment_sources(
    references: dict[str, dict[str, Any]],
    environment: Any,
    *,
    role: str,
    record_path: Path,
) -> None:
    if not isinstance(environment, Mapping):
        return
    dependencies = environment.get("source_dependencies")
    if isinstance(dependencies, list):
        for item in dependencies:
            if not isinstance(item, Mapping):
                raise RuntimeError(f"Malformed host-source ledger in {record_path}")
            relative = item.get("path")
            digest = item.get("sha256")
            if not isinstance(relative, str):
                raise RuntimeError(f"Malformed host-source path in {record_path}")
            _collect_source_reference(
                references,
                digest,
                role=role,
                record_path=record_path,
                expected_relative_path=relative,
            )


def _copy_archive_execution_sources(
    *, root: Path, destination: Path
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for relative in ARCHIVE_EXECUTION_SOURCE_PATHS:
        source = _safe_local(root, relative)
        if not source.is_file():
            raise RuntimeError(f"Archive execution source is missing: {relative}")
        local = destination / relative
        _copy_file(source, local)
        entries.append(
            {
                "path": relative,
                "local_path": local.relative_to(destination.parent.parent).as_posix(),
                "size_bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        )
    return entries


def _verify_archive_execution_sources(
    root: Path, entries: Any
) -> Path:
    if not isinstance(entries, list):
        raise RuntimeError("Archive execution-source ledger is missing")
    expected = set(ARCHIVE_EXECUTION_SOURCE_PATHS)
    observed = {
        item.get("path") for item in entries if isinstance(item, Mapping)
    }
    if observed != expected or len(entries) != len(expected):
        raise RuntimeError("Archive execution-source ledger is incomplete or duplicated")
    local_repository: Path | None = None
    for item in entries:
        relative = str(item["path"])
        local = _safe_local(root, str(item["local_path"]))
        if (
            item.get("local_path") != f"archive_execution/repository/{relative}"
            or not local.is_file()
            or local.stat().st_size != item.get("size_bytes")
            or sha256_file(local) != item.get("sha256")
        ):
            raise RuntimeError(f"Archive execution source changed: {relative}")
        candidate = local
        for _ in Path(relative).parts:
            candidate = candidate.parent
        if local_repository is None:
            local_repository = candidate
        elif candidate != local_repository:
            raise RuntimeError("Archive execution sources do not share one local root")
    if local_repository is None:
        raise RuntimeError("Archive execution-source root is missing")
    return local_repository


def _macro_f1(labels: list[int], predictions: list[int]) -> float:
    values: list[float] = []
    for label in range(5):
        true_positive = sum(
            truth == label and prediction == label
            for truth, prediction in zip(labels, predictions)
        )
        false_positive = sum(
            truth != label and prediction == label
            for truth, prediction in zip(labels, predictions)
        )
        false_negative = sum(
            truth == label and prediction != label
            for truth, prediction in zip(labels, predictions)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return sum(values) / len(values)


def _portable_final_export_verifier(root: Path) -> dict[str, Any]:
    """Verify one archived export without dataset, split, or checkpoint paths."""

    resolved = root.resolve()
    report = read_json(resolved / "final_export_report.json")
    identity = read_json(resolved / "final_export_identity.json")
    if report.get("status") != "passed" or report.get("identity") != identity:
        raise RuntimeError("Archived final export report/identity is invalid")
    snapshots = identity.get("source_snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        raise RuntimeError("Archived final export source ledger is missing")
    observed_snapshots: set[str] = set()
    for item in snapshots:
        if not isinstance(item, Mapping):
            raise RuntimeError("Archived final export source ledger is malformed")
        relative = item.get("snapshot_path")
        if not isinstance(relative, str) or relative in observed_snapshots:
            raise RuntimeError("Archived final export source path is duplicated")
        member = _safe_local(resolved, relative)
        if (
            not member.is_file()
            or member.stat().st_size != item.get("size_bytes")
            or sha256_file(member) != item.get("sha256")
        ):
            raise RuntimeError(f"Archived final export source changed: {relative}")
        observed_snapshots.add(relative)
    if observed_snapshots != FINAL_EXPORT_SOURCE_SNAPSHOTS:
        raise RuntimeError("Archived final export source snapshot set is incomplete")
    if report.get("source_artifacts", {}).get("source_snapshots") != snapshots:
        raise RuntimeError("Archived final export source ledgers disagree")

    replay_path = resolved / "hil_replay_vectors.csv"
    reference_path = resolved / "hil_reference_predictions.csv"
    replay_sources: list[int] = []
    expected_replay_fields = [
        "row_id",
        "source_row_index",
        *[f"f{index}" for index in range(17)],
    ]
    with replay_path.open(newline="", encoding="ascii") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_replay_fields:
            raise RuntimeError("Archived full replay schema is invalid")
        for expected_row, row in enumerate(reader):
            if int(row["row_id"]) != expected_row:
                raise RuntimeError("Archived full replay row IDs are not dense")
            replay_sources.append(int(row["source_row_index"]))
            for index in range(17):
                value = int(row[f"f{index}"])
                if value < -(2**31) or value > 2**31 - 1:
                    raise RuntimeError("Archived replay input is outside signed int32")
    if (
        len(replay_sources) != EXPECTED_FULL_ROWS
        or len(set(replay_sources)) != EXPECTED_FULL_ROWS
    ):
        raise RuntimeError("Archived full replay source identities are incomplete")

    labels: list[int] = []
    fixed_predictions: list[int] = []
    fp32_predictions: list[int] = []
    expected_reference_fields = [
        "row_id",
        "source_row_index",
        "true_label",
        "fixed_pred",
        "fp32_pred",
        *[f"fixed_logit_{index}" for index in range(5)],
    ]
    with reference_path.open(newline="", encoding="ascii") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_reference_fields:
            raise RuntimeError("Archived full reference schema is invalid")
        for expected_row, row in enumerate(reader):
            if (
                expected_row >= EXPECTED_FULL_ROWS
                or int(row["row_id"]) != expected_row
                or int(row["source_row_index"]) != replay_sources[expected_row]
            ):
                raise RuntimeError("Archived full reference identity differs from replay")
            truth = int(row["true_label"])
            fixed = int(row["fixed_pred"])
            fp32 = int(row["fp32_pred"])
            if any(value not in range(5) for value in [truth, fixed, fp32]):
                raise RuntimeError("Archived full reference contains an invalid class")
            logits = [int(row[f"fixed_logit_{index}"]) for index in range(5)]
            if any(value < -(2**15) or value > 2**15 - 1 for value in logits):
                raise RuntimeError("Archived fixed logit is outside signed int16")
            if max(range(5), key=logits.__getitem__) != fixed:
                raise RuntimeError("Archived fixed prediction differs from logit argmax")
            labels.append(truth)
            fixed_predictions.append(fixed)
            fp32_predictions.append(fp32)
    if len(labels) != EXPECTED_FULL_ROWS:
        raise RuntimeError("Archived full reference row count is incomplete")

    agreement = sum(
        fixed == fp32
        for fixed, fp32 in zip(fixed_predictions, fp32_predictions)
    ) / EXPECTED_FULL_ROWS
    fp32_macro_f1 = _macro_f1(labels, fp32_predictions)
    fixed_macro_f1 = _macro_f1(labels, fixed_predictions)
    absolute_change = abs(fp32_macro_f1 - fixed_macro_f1)
    gates = report.get("gates")
    if not isinstance(gates, Mapping):
        raise RuntimeError("Archived final export gate ledger is missing")
    for key, observed in {
        "fixed_vs_fp32_agreement": agreement,
        "fp32_macro_f1": fp32_macro_f1,
        "fixed_macro_f1": fixed_macro_f1,
        "absolute_macro_f1_drop": absolute_change,
    }.items():
        try:
            recorded = float(gates[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Archived final export gate is missing: {key}") from exc
        if not math.isclose(recorded, observed, rel_tol=1e-12, abs_tol=1e-12):
            raise RuntimeError(f"Archived final export gate differs: {key}")
    if (
        gates.get("test_rows") != EXPECTED_FULL_ROWS
        or gates.get("minimum_fixed_vs_fp32_agreement") != 0.99
        or gates.get("maximum_absolute_macro_f1_drop") != 0.015
        or agreement < 0.99
        or absolute_change > 0.015
    ):
        raise RuntimeError("Archived final export fails the frozen quality thresholds")
    for key in [
        "quality_gates_passed",
        "zero_saturation_passed",
        "accumulator_bounds_passed",
        "preprocess_bounds_passed",
        "host_equivalence_passed",
        "dense_row_ids_passed",
        "source_row_ids_complete_unique_passed",
    ]:
        if gates.get(key) is not True:
            raise RuntimeError(f"Archived final export gate is not passed: {key}")
    for key in [
        "raw_input_saturation_count",
        "weight_saturation_count",
        "bias_saturation_count",
        "activation_saturation_count",
        "integer_preprocess_saturation_count",
    ]:
        if gates.get(key) != 0:
            raise RuntimeError(f"Archived final export saturation is nonzero: {key}")
    if gates.get("standardized_input_saturation_count") != {
        "train": 0,
        "validation": 0,
        "test": 0,
    }:
        raise RuntimeError("Archived standardized-input saturation is nonzero")
    accumulator_bounds = gates.get("accumulator_bounds")
    if (
        not isinstance(accumulator_bounds, list)
        or [item.get("layer") for item in accumulator_bounds] != [0, 1, 2]
        or any(
            item.get("passed") is not True
            or not isinstance(item.get("post_left_shift_absolute_bound"), int)
            or item["post_left_shift_absolute_bound"] > item.get("int32_max", -1)
            for item in accumulator_bounds
        )
    ):
        raise RuntimeError("Archived accumulator bound is not passed")
    preprocess_bounds = gates.get("preprocess_multiply_bounds")
    if (
        not isinstance(preprocess_bounds, list)
        or [item.get("feature") for item in preprocess_bounds] != list(range(17))
        or any(
            item.get("passed") is not True
            or not isinstance(item.get("maximum_product_absolute"), int)
            or item["maximum_product_absolute"] > item.get("int64_max", -1)
            for item in preprocess_bounds
        )
    ):
        raise RuntimeError("Archived preprocessing bound is not passed")
    host = report.get("host_equivalence")
    if (
        not isinstance(host, Mapping)
        or host.get("status") != "passed"
        or host.get("rows") != EXPECTED_FULL_ROWS
        or host.get("preprocessed_inputs_exact") is not True
        or host.get("fixed_logits_exact") is not True
        or host.get("fixed_predictions_exact") is not True
        or not _is_sha256(host.get("temporary_executable_sha256"))
        or host.get("temporary_executable_retained") is not False
    ):
        raise RuntimeError("Archived host-equivalence record is incomplete")
    for command_name in ["compiler_version", "compile", "self_test"]:
        command = host.get(command_name)
        if not isinstance(command, Mapping) or command.get("returncode") != 0:
            raise RuntimeError(
                f"Archived host-equivalence command did not pass: {command_name}"
            )
    return {
        "status": "passed",
        "export_id": identity.get("export_id"),
        "student": identity.get("student"),
        "route": identity.get("route"),
        "test_rows": EXPECTED_FULL_ROWS,
        "quality_gates_passed": True,
        "fixed_vs_fp32_agreement": agreement,
        "absolute_macro_f1_drop": absolute_change,
        "portable_verification": True,
    }


def _portable_blocked_audit_verifier(path: Path) -> dict[str, Any]:
    payload = read_json(path.resolve())
    copy = dict(payload)
    recorded_payload_hash = copy.pop("audit_payload_sha256", None)
    if recorded_payload_hash != canonical_json_sha256(copy):
        raise RuntimeError("Archived blocked-audit payload hash is invalid")
    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        raise RuntimeError("Archived blocked-audit identity is missing")
    identity_copy = dict(identity)
    blocked_id = identity_copy.pop("blocked_audit_id", None)
    if blocked_id != canonical_json_sha256(identity_copy):
        raise RuntimeError("Archived blocked-audit identity hash is invalid")
    try:
        agreement = float(payload.get("fixed_vs_fp32_agreement"))
        absolute_change = float(payload.get("absolute_macro_f1_drop"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Archived blocked-audit metrics are invalid") from exc
    if (
        payload.get("status") != "blocked"
        or payload.get("quality_gates_passed") is not False
        or payload.get("test_rows") != EXPECTED_FULL_ROWS
        or payload.get("zero_saturation_passed") is not True
        or payload.get("accumulator_bounds_passed") is not True
        or payload.get("preprocess_bounds_passed") is not True
        or payload.get("minimum_fixed_vs_fp32_agreement") != 0.99
        or payload.get("maximum_absolute_macro_f1_drop") != 0.015
        or not math.isfinite(agreement)
        or not math.isfinite(absolute_change)
        or not 0.0 <= agreement <= 1.0
        or not 0.0 <= absolute_change <= 1.0
        or (agreement >= 0.99 and absolute_change <= 0.015)
    ):
        raise RuntimeError("Archived blocked-audit gate evidence is invalid")
    return {
        "status": "blocked_verified",
        "blocked_audit_id": blocked_id,
        "student": identity.get("student"),
        "route": identity.get("route"),
        "fixed_vs_fp32_agreement": agreement,
        "absolute_macro_f1_drop": absolute_change,
    }


def _validate_session_source(
    session_path: Path,
    *,
    export_dir: Path,
    cohort_dir: Path,
    bundle_dir: Path,
    provenance_json: Path,
    connection_json: Path | None,
    attempt_dirs: Mapping[str, Path],
    verifier: Verifier | None,
    host_source_root: Path | None,
) -> tuple[dict[str, Any], list[tuple[Path, dict[str, Any]]]]:
    if session_path.is_symlink():
        raise RuntimeError("Session source cannot be a symlink")
    raw = read_json(session_path)
    stage_records = raw.get("stages")
    if not isinstance(stage_records, list):
        raise RuntimeError("Session stage ledger is malformed")
    by_name = {
        str(stage.get("name")): stage
        for stage in stage_records
        if isinstance(stage, Mapping)
    }
    expected_names = {stage["name"] for stage in FINAL_STAGES}
    if set(by_name) != expected_names or set(attempt_dirs) != expected_names:
        raise RuntimeError("Session attempt source set is incomplete or unexpected")
    attempt_id_dirs = {
        str(by_name[name]["attempt_id"]): Path(attempt_dirs[name])
        for name in expected_names
    }
    session = validate_session_completion(
        session_path,
        export_dir=export_dir,
        cohort_dir=cohort_dir,
        bundle_dir=bundle_dir,
        provenance_json=provenance_json,
        connection_json=connection_json,
        attempt_dirs=attempt_id_dirs,
        verifier=verifier,
        host_source_root=host_source_root,
    )
    attempts: list[tuple[Path, dict[str, Any]]] = []
    for stage in session["stages"]:
        attempt_dir = Path(attempt_dirs[stage["name"]])
        attempt = verify_stage_attempt(
            attempt_dir,
            export_dir=export_dir,
            cohort_dir=cohort_dir,
            bundle_dir=bundle_dir,
            verifier=verifier,
            host_source_root=host_source_root,
        )
        attempts.append((attempt_dir.resolve(), attempt))
    return session, attempts


def _copy_provenance(
    *,
    source_json: Path,
    destination: Path,
    validated: Mapping[str, Any],
    original_record_path: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_root = source_json.resolve().parent
    copied = _copy_tree(source_root, destination)
    return dict(validated), {
        "format": "validator_owned_recursive_inventory",
        "validated_schema": validated.get("schema"),
        "record_relative_to_provenance": source_json.name,
        "original_record_path": original_record_path,
        "original_record_sha256": sha256_file(source_json),
        "files_relative_to_provenance": copied,
    }


def create_campaign_archive(
    *,
    campaign_contract: Path,
    campaign_evidence: Path | None,
    cohort_dir: Path | None,
    export_dirs: Mapping[str, Path],
    blocked_audits: Mapping[str, Path],
    source_roots: Mapping[str, Path],
    output_dir: Path,
    host_source_root: Path | None = None,
    verifier: Verifier | None = None,
) -> Path:
    """Create and then deeply verify one non-overwriting directory archive."""

    contract_source = Path(campaign_contract)
    if contract_source.is_symlink():
        raise RuntimeError("Campaign contract cannot be a symlink")
    contract_path = contract_source.resolve()
    contract = validate_campaign_contract(contract_path)
    destination = output_dir.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite final-HIL archive: {destination}")
    expected_exports = {
        key for key in MODEL_KEYS if contract["models"][key].get("status") == "passed"
    }
    expected_blocked = {
        key for key in MODEL_KEYS if contract["models"][key].get("status") == "blocked"
    }
    if set(export_dirs) != expected_exports:
        raise RuntimeError("Archive export assignments differ from passed routes")
    if set(blocked_audits) != expected_blocked:
        raise RuntimeError("Archive blocked-audit assignments differ from blocked routes")
    if not isinstance(source_roots, Mapping):
        raise RuntimeError("Archive source-root mappings are malformed")
    local_host_source_root = Path(host_source_root) if host_source_root else REPO_ROOT
    if (
        local_host_source_root.is_symlink()
        or not local_host_source_root.resolve().is_dir()
    ):
        raise RuntimeError("Archive host-source root is invalid")
    local_host_source_root = local_host_source_root.resolve()
    source_verifier = verifier or _portable_final_export_verifier
    normalized_source_roots: dict[str, Path] = {}
    for recorded_root, local_root in source_roots.items():
        normalized_recorded_root = (
            recorded_root.replace("\\", "/").rstrip("/")
            if isinstance(recorded_root, str)
            else ""
        )
        if (
            not isinstance(recorded_root, str)
            or not recorded_root
            or normalized_recorded_root in normalized_source_roots
            or Path(local_root).is_symlink()
            or not Path(local_root).resolve().is_dir()
        ):
            raise RuntimeError("Archive source-root mapping is invalid")
        if not (
            normalized_recorded_root.startswith("/")
            or (
                len(normalized_recorded_root) >= 3
                and normalized_recorded_root[0].isalpha()
                and normalized_recorded_root[1:3] == ":/"
            )
        ):
            raise RuntimeError("Archive recorded source root is not absolute")
        normalized_source_roots[normalized_recorded_root] = Path(local_root).resolve()

    evidence_path: Path | None = None
    if campaign_evidence is not None:
        evidence_source = Path(campaign_evidence)
        if evidence_source.is_symlink():
            raise RuntimeError("Campaign evidence cannot be a symlink")
        evidence_path = evidence_source.resolve()
    evidence: dict[str, Any] | None = None
    session_paths: list[Path] = []
    executable = contract["status"] in {"ready", "ready_with_blocked_routes"}
    if executable:
        if evidence_path is None:
            raise RuntimeError("A ready campaign requires campaign evidence")
        evidence = read_json(evidence_path)
        _validate_campaign_evidence_header(evidence, contract)
        evidence_sessions = evidence.get("sessions")
        if not isinstance(evidence_sessions, list):
            raise RuntimeError("Campaign evidence session map is malformed")
        for item in evidence_sessions:
            if not isinstance(item, Mapping) or not isinstance(
                item.get("session_path_recorded"), str
            ):
                raise RuntimeError("Campaign evidence session map is malformed")
            source_session = _relocate_recorded_path(
                str(item["session_path_recorded"]),
                source_roots=normalized_source_roots,
                expected_kind="file",
            )
            if sha256_file(source_session) != item.get("session_sha256"):
                raise RuntimeError("Relocated campaign session hash differs")
            session_paths.append(source_session)
        if cohort_dir is None:
            raise RuntimeError("A ready campaign requires its balanced cohort")
        raw_sessions = [read_json(path) for path in session_paths]
        session_contexts: dict[str, dict[str, Any]] = {}
        for raw in raw_sessions:
            combination_id = str(raw.get("combination_id"))
            model_key = str(raw.get("model_key"))
            stages = raw.get("stages")
            if model_key not in export_dirs or not isinstance(stages, list):
                raise RuntimeError(
                    f"Campaign session source context is malformed: {combination_id}"
                )
            attempt_dirs = {
                str(stage["name"]): _relocate_recorded_path(
                    str(stage["attempt_path_recorded"]),
                    source_roots=normalized_source_roots,
                    expected_kind="dir",
                )
                for stage in stages
                if isinstance(stage, Mapping)
            }
            if len(attempt_dirs) != len(stages):
                raise RuntimeError(
                    f"Campaign session attempt source context is malformed: {combination_id}"
                )
            session_contexts[combination_id] = {
                "export_dir": Path(export_dirs[model_key]).resolve(),
                "cohort_dir": cohort_dir.resolve(),
                "bundle_dir": _relocate_recorded_path(
                    str(raw["bundle_path_recorded"]),
                    source_roots=normalized_source_roots,
                    expected_kind="dir",
                ),
                "provenance_json": _relocate_recorded_path(
                    str(raw["provenance_path_recorded"]),
                    source_roots=normalized_source_roots,
                    expected_kind="file",
                ),
                "connection_json": (
                    _relocate_recorded_path(
                        str(raw["connection_path_recorded"]),
                        source_roots=normalized_source_roots,
                        expected_kind="file",
                    )
                    if raw.get("transport") == "wifi_udp"
                    else None
                ),
                "attempt_dirs": attempt_dirs,
            }
        recomputed = verify_complete_campaign(
            campaign_contract=contract_path,
            session_jsons=session_paths,
            session_contexts=session_contexts,
            verifier=source_verifier,
            host_source_root=local_host_source_root,
        )
        if not _portable_campaign_evidence_equal(evidence, recomputed):
            raise RuntimeError("Campaign evidence differs from portable recomputation")
    else:
        if evidence_path is not None:
            raise RuntimeError("A blocked campaign cannot carry passed campaign evidence")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    manifest_path = temporary / ARCHIVE_MANIFEST
    source_references: dict[str, dict[str, Any]] = {}
    semantic: dict[str, Any] = {
        "campaign": {
            "contract_local_path": "campaign/campaign_contract.json",
            "contract_original_path": str(contract_path),
            "evidence_local_path": (
                "campaign/campaign_evidence.json" if evidence_path is not None else None
            ),
            "evidence_original_path": str(evidence_path) if evidence_path else None,
        },
        "models": {},
        "cohort": None,
        "combinations": {},
        "host_sources": [],
        "archive_execution_sources": [],
    }
    identities: dict[str, FinalExportIdentity] = {}
    try:
        _copy_file(contract_path, temporary / "campaign" / "campaign_contract.json")
        if evidence_path is not None:
            _copy_file(evidence_path, temporary / "campaign" / "campaign_evidence.json")

        for key in MODEL_KEYS:
            model = contract["models"][key]
            if model.get("status") == "passed":
                source = Path(export_dirs[key])
                identity = validate_final_export(source, verifier=source_verifier)
                if identity.model_key != key:
                    raise RuntimeError(f"Archive export identity differs for {key}")
                for field, value in identity.to_dict().items():
                    if field != "root" and model.get(field) != value:
                        raise RuntimeError(
                            f"Archive export differs from campaign for {key}:{field}"
                        )
                local_dir = temporary / "models" / key / "export"
                copied = _copy_tree(source, local_dir)
                identities[key] = identity
                semantic["models"][key] = {
                    "status": "passed",
                    "export_original_path": str(model.get("root")),
                    "export_local_dir": _relative(temporary, local_dir),
                    "files_relative_to_export": copied,
                }
            else:
                if model.get("status") == "blocked":
                    source = Path(blocked_audits[key])
                    blocked = validate_blocked_audit(
                        source, verifier=_portable_blocked_audit_verifier
                    )
                    if blocked["model_key"] != key:
                        raise RuntimeError(f"Archive blocked audit differs for {key}")
                    for field in [
                        "reason",
                        "blocked_audit_id",
                        "blocked_audit_sha256",
                    ]:
                        if blocked.get(field) != model.get(field):
                            raise RuntimeError(
                                f"Archive blocked audit differs for {key}:{field}"
                            )
                    local = temporary / "models" / key / "blocked_audit.json"
                    _copy_file(source, local)
                    semantic["models"][key] = {
                        "status": "blocked",
                        "blocked_audit_original_path": str(
                            model.get("blocked_audit_path_recorded")
                        ),
                        "blocked_audit_local_path": _relative(temporary, local),
                    }
                else:
                    semantic["models"][key] = {
                        "status": str(model.get("status")),
                        "reason": model.get("reason"),
                    }

        local_cohort: Path | None = None
        if cohort_dir is not None:
            source_cohort = Path(cohort_dir)
            validate_balanced_cohort(
                source_cohort,
                identities=identities or None,
                reconstruct_sources=False,
            )
            local_cohort = temporary / "cohort"
            copied = _copy_tree(source_cohort, local_cohort)
            semantic["cohort"] = {
                "original_path": str(
                    read_json(session_paths[0]).get("cohort_path_recorded")
                    if session_paths
                    else source_cohort
                ),
                "local_dir": _relative(temporary, local_cohort),
                "files_relative_to_cohort": copied,
            }

        if evidence is not None:
            if local_cohort is None:
                raise AssertionError("ready campaign has no local cohort")
            session_by_id = {
                read_json(path)["combination_id"]: path for path in session_paths
            }
            if set(session_by_id) != {
                item["combination_id"] for item in contract["eligible_combinations"]
            }:
                raise RuntimeError("Source campaign session matrix is incomplete")
            for combination in contract["eligible_combinations"]:
                combination_id = combination["combination_id"]
                source_session = session_by_id[combination_id]
                source_payload = read_json(source_session)
                model_key = combination["model_key"]
                context = session_contexts[combination_id]
                source_provenance = Path(context["provenance_json"]).resolve()
                source_bundle = Path(context["bundle_dir"]).resolve()
                source_attempts = {
                    name: Path(path).resolve()
                    for name, path in context["attempt_dirs"].items()
                }
                source_connection = (
                    Path(context["connection_json"])
                    if context["connection_json"] is not None
                    else None
                )
                validated_source_provenance = validate_build_upload_provenance(
                    source_provenance,
                    bundle_dir=source_bundle,
                    expected_export=identities[model_key],
                    host_source_root=local_host_source_root,
                )
                export_source = Path(identities[model_key].root)
                session, attempts = _validate_session_source(
                    source_session,
                    export_dir=export_source,
                    cohort_dir=cohort_dir.resolve(),
                    bundle_dir=source_bundle,
                    provenance_json=source_provenance,
                    connection_json=source_connection,
                    attempt_dirs=source_attempts,
                    verifier=source_verifier,
                    host_source_root=local_host_source_root,
                )
                prefix = (
                    temporary
                    / "combinations"
                    / f"{int(combination['execution_ordinal']):02d}_{combination_id}"
                )
                local_session = prefix / "session.json"
                _copy_file(source_session, local_session)
                local_bundle = prefix / "bundle"
                bundle_files = _copy_tree(source_bundle, local_bundle)
                bundle = verify_final_bundle(
                    source_bundle,
                    expected_export=identities[model_key],
                    **_bundle_source_options(local_host_source_root),
                )
                _collect_source_reference(
                    source_references,
                    bundle.get("bundle_identity_payload", {}).get(
                        "bundle_builder_sha256"
                    ),
                    role="bundle_builder",
                    record_path=source_bundle / "final_bundle_manifest.json",
                    expected_relative_path="deployment/final_hil/bundles.py",
                )
                local_provenance_dir = prefix / "provenance"
                validated_provenance, provenance_map = _copy_provenance(
                    source_json=source_provenance,
                    destination=local_provenance_dir,
                    validated=validated_source_provenance,
                    original_record_path=str(
                        source_payload["provenance_path_recorded"]
                    ),
                )
                _collect_environment_sources(
                    source_references,
                    validated_provenance.get("host_environment"),
                    role="provenance_runtime",
                    record_path=source_provenance,
                )
                connection_map = None
                if source_connection is not None:
                    local_connection = prefix / "connection.json"
                    _copy_file(source_connection, local_connection)
                    connection_payload = read_json(source_connection)
                    _collect_environment_sources(
                        source_references,
                        connection_payload.get("host_environment"),
                        role="wifi_connection",
                        record_path=source_connection,
                    )
                    connection_map = {
                        "original_path": source_payload["connection_path_recorded"],
                        "local_path": _relative(temporary, local_connection),
                        "original_record_sha256": sha256_file(source_connection),
                    }
                attempt_maps = []
                recorded_stage_by_name = {
                    stage["name"]: stage for stage in source_payload["stages"]
                }
                for source_attempt, attempt in attempts:
                    stage = attempt["stage"]
                    local_attempt = (
                        prefix
                        / "attempts"
                        / f"{int(stage['ordinal']):02d}_{stage['name']}"
                    )
                    copied = _copy_tree(source_attempt, local_attempt)
                    required = {"attempt_started.json", "final_attempt.json", "responses.csv"}
                    if not required <= set(copied):
                        raise RuntimeError(
                            f"Attempt {attempt['attempt_id']} lacks required raw evidence"
                        )
                    _collect_environment_sources(
                        source_references,
                        attempt.get("host_environment"),
                        role="stage_runtime",
                        record_path=source_attempt / "final_attempt.json",
                    )
                    attempt_maps.append(
                        {
                            "stage_name": stage["name"],
                            "ordinal": stage["ordinal"],
                            "attempt_id": attempt["attempt_id"],
                            "original_dir": str(
                                recorded_stage_by_name[stage["name"]][
                                    "attempt_path_recorded"
                                ]
                            ),
                            "local_dir": _relative(temporary, local_attempt),
                            "files_relative_to_attempt": copied,
                            "original_base_replay_path_recorded": attempt[
                                "input_binding"
                            ]["base_replay_path_recorded"],
                            "original_base_reference_path_recorded": attempt[
                                "input_binding"
                            ]["base_reference_path_recorded"],
                        }
                    )
                semantic["combinations"][combination_id] = {
                    "execution_ordinal": combination["execution_ordinal"],
                    "model_key": model_key,
                    "board": combination["board"],
                    "transport": combination["transport"],
                    "session_original_path": str(
                        next(
                            item["session_path_recorded"]
                            for item in evidence["sessions"]
                            if item["combination_id"] == combination_id
                        )
                    ),
                    "session_local_path": _relative(temporary, local_session),
                    "export_original_path": str(
                        source_payload["export_path_recorded"]
                    ),
                    "cohort_original_path": str(
                        source_payload["cohort_path_recorded"]
                    ),
                    "bundle_original_dir": str(
                        source_payload["bundle_path_recorded"]
                    ),
                    "bundle_local_dir": _relative(temporary, local_bundle),
                    "bundle_files_relative": bundle_files,
                    "provenance": {
                        **provenance_map,
                        "local_dir": _relative(temporary, local_provenance_dir),
                    },
                    "connection": connection_map,
                    "attempts": attempt_maps,
                    "campaign_session_id": session["campaign_session_id"],
                }

        for reference_key in sorted(source_references):
            reference = source_references[reference_key]
            digest = str(reference["sha256"])
            expected_relative = reference.get("expected_relative_path")
            if not isinstance(expected_relative, str):
                raise RuntimeError("Host-source reference lacks a canonical relative path")
            source = _safe_local(local_host_source_root, expected_relative)
            if not source.is_file() or sha256_file(source) != digest:
                raise RuntimeError(
                    f"Host source differs from recorded digest: {expected_relative}"
                )
            semantic["host_sources"].append(
                {
                    **reference,
                    "original_path": str(source),
                    "local_path": (
                        Path("archive_execution") / "repository" / expected_relative
                    ).as_posix(),
                    "size_bytes": source.stat().st_size,
                }
            )

        semantic["archive_execution_sources"] = _copy_archive_execution_sources(
            root=local_host_source_root,
            destination=temporary / "archive_execution" / "repository",
        )

        owned_files = sorted(_semantic_file_set(semantic))
        actual_files = sorted(_relative(temporary, path) for path in _tree_files(temporary))
        if owned_files != actual_files:
            raise RuntimeError("Archive construction left an unowned or missing file")
        semantic["owned_files"] = owned_files
        inventory = _inventory(temporary)
        payload: dict[str, Any] = {
            "schema": ARCHIVE_SCHEMA,
            "status": (
                evidence["status"] if executable and evidence is not None else "blocked"
            ),
            "contract_id": contract["contract_id"],
            "campaign_evidence_id": (
                evidence.get("campaign_evidence_id") if evidence is not None else None
            ),
            "semantic_map": semantic,
            "file_count_excluding_manifest": len(inventory),
            "inventory": inventory,
            "path_policy": (
                "Verification uses archive-relative semantic-map paths and archived full "
                "exports. Original absolute paths are historical strings and are never "
                "dereferenced; training/checkpoint reconstruction remains generation-time "
                "evidence rather than a portable-archive claim."
            ),
        }
        payload["archive_id"] = canonical_json_sha256(payload)
        atomic_write_json(manifest_path, payload)
        verify_campaign_archive(temporary, verifier=verifier)
        if destination.exists():
            raise FileExistsError(
                f"Refusing to overwrite final-HIL archive created concurrently: {destination}"
            )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination / ARCHIVE_MANIFEST


def _verify_inventory(root: Path, manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    inventory = manifest.get("inventory")
    if not isinstance(inventory, list):
        raise RuntimeError("Archive inventory is missing")
    listed: dict[str, dict[str, Any]] = {}
    ordered_paths: list[str] = []
    for item in inventory:
        if not isinstance(item, dict):
            raise RuntimeError("Archive inventory contains a non-object")
        relative = item.get("path")
        if not isinstance(relative, str) or relative in listed:
            raise RuntimeError("Archive inventory path is malformed or duplicated")
        path = _safe_local(root, relative)
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != item.get("size_bytes")
            or sha256_file(path) != item.get("sha256")
        ):
            raise RuntimeError(f"Archive inventory mismatch: {relative}")
        listed[relative] = item
        ordered_paths.append(relative)
    if ordered_paths != sorted(ordered_paths):
        raise RuntimeError("Archive inventory is not deterministically ordered")
    if manifest.get("file_count_excluding_manifest") != len(listed):
        raise RuntimeError("Archive inventory count is invalid")
    actual = {
        _relative(root, path)
        for path in _tree_files(root)
        if path.name != ARCHIVE_MANIFEST or path.parent.resolve() != root.resolve()
    }
    if actual != set(listed):
        raise RuntimeError("Archive contains extra or missing files")
    semantic = manifest.get("semantic_map")
    if (
        not isinstance(semantic, Mapping)
        or semantic.get("owned_files") != sorted(listed)
        or _semantic_file_set(semantic) != set(listed)
    ):
        raise RuntimeError("Archive semantic ownership map differs from inventory")
    return listed


def _verify_archived_provenance(
    *,
    root: Path,
    provenance_map: Mapping[str, Any],
    bundle: Mapping[str, Any],
    host_source_root: Path,
) -> dict[str, Any]:
    provenance_root = _safe_local(root, str(provenance_map["local_dir"]))
    expected_files = provenance_map.get("files_relative_to_provenance")
    if not isinstance(expected_files, list) or expected_files != sorted(expected_files):
        raise RuntimeError("Provenance directory inventory is malformed")
    actual_files = [
        path.relative_to(provenance_root).as_posix() for path in _tree_files(provenance_root)
    ]
    if actual_files != expected_files:
        raise RuntimeError("Provenance directory contains extra or missing files")
    record = _safe_local(
        provenance_root, str(provenance_map["record_relative_to_provenance"])
    )
    if provenance_map.get("format") != "validator_owned_recursive_inventory":
        raise RuntimeError("Unsupported archive provenance adapter")
    # The active provenance validator owns all future schemas.  A future record
    # is admissible only if that validator accepts the archive-local directory.
    payload = validate_build_upload_provenance(
        record,
        host_source_root=host_source_root,
    )
    expected = {
        "bundle_manifest_sha256": bundle["_manifest_sha256"],
        "bundle_id": bundle["bundle_id"],
        "build_contract_id": bundle["build_contract_id"],
        "board": bundle["board"],
        "transport": bundle["transport"],
        "student": bundle["student"],
        "route": bundle["route"],
        "export_id": bundle["export_id"],
        "model_sha256": bundle["model_sha256"],
        "checkpoint_file_sha256": bundle["checkpoint_file_sha256"],
        "post_reset_runtime_identity": bundle["runtime_identity_response"],
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise RuntimeError(f"Archived provenance differs from bundle for {field}")
    return payload


def _portable_input_binding_matches(
    recorded: Mapping[str, Any],
    local: Mapping[str, Any],
    attempt_map: Mapping[str, Any],
) -> bool:
    if recorded.get("base_replay_path_recorded") != attempt_map.get(
        "original_base_replay_path_recorded"
    ) or recorded.get("base_reference_path_recorded") != attempt_map.get(
        "original_base_reference_path_recorded"
    ):
        return False
    if set(recorded) != set(local):
        return False
    path_fields = {"base_replay_path_recorded", "base_reference_path_recorded"}
    return all(
        recorded.get(key) == value
        for key, value in local.items()
        if key not in path_fields
    )


def _verify_archived_attempt(
    *,
    root: Path,
    attempt_map: Mapping[str, Any],
    session: Mapping[str, Any],
    bundle: Mapping[str, Any],
    export: FinalExportIdentity,
    cohort_dir: Path,
    host_source_root: Path,
) -> dict[str, Any]:
    attempt_root = _safe_local(root, str(attempt_map["local_dir"]))
    expected_files = attempt_map.get("files_relative_to_attempt")
    if not isinstance(expected_files, list) or expected_files != sorted(expected_files):
        raise RuntimeError("Attempt directory inventory is malformed")
    actual_files = [
        path.relative_to(attempt_root).as_posix() for path in _tree_files(attempt_root)
    ]
    if actual_files != expected_files:
        raise RuntimeError("Attempt directory contains extra or missing files")
    required = {"attempt_started.json", "final_attempt.json", "responses.csv"}
    if set(actual_files) != required:
        raise RuntimeError("Attempt directory raw-evidence inventory is not exact")
    started = read_json(attempt_root / "attempt_started.json")
    attempt = read_json(attempt_root / "final_attempt.json")
    _validate_attempt_payload_shape(started, attempt)
    if started.get("attempt_started_sha256") != _started_payload_hash(started):
        raise RuntimeError("Archived attempt-start payload hash is invalid")
    if attempt.get("attempt_payload_sha256") != _attempt_payload_hash(attempt):
        raise RuntimeError("Archived final-attempt payload hash is invalid")
    validate_host_environment(
        attempt.get("host_environment", {}), source_root=host_source_root
    )
    if (
        started.get("schema") != ATTEMPT_SCHEMA
        or started.get("status") != "running"
        or attempt.get("schema") != ATTEMPT_SCHEMA
        or attempt.get("status") != "passed"
    ):
        raise RuntimeError("Archived attempt is not a passed final attempt")
    for field, value in started.items():
        if field != "status" and attempt.get(field) != value:
            raise RuntimeError(f"Archived attempt start/final field differs: {field}")
    if attempt.get("attempt_id") != attempt_map.get("attempt_id"):
        raise RuntimeError("Archived attempt ID differs from semantic map")
    if attempt.get("campaign_session_id") != session.get("campaign_session_id"):
        raise RuntimeError("Archived attempt belongs to another campaign session")
    if attempt.get("bundle_id") != bundle["bundle_id"] or attempt.get(
        "build_contract_id"
    ) != bundle["build_contract_id"]:
        raise RuntimeError("Archived attempt differs from bundle/build identity")
    if attempt.get("runtime_identity") != bundle["runtime_identity_response"]:
        raise RuntimeError("Archived attempt runtime identity differs from bundle")
    expected_combination = {
        "student": session["student"],
        "route": session["route"],
        "board": session["board"],
        "transport": session["transport"],
    }
    if attempt.get("combination") != expected_combination:
        raise RuntimeError("Archived attempt combination identity differs")
    dataset = load_stage_dataset(
        export=export,
        cohort_dir=cohort_dir,
        stage_name=str(attempt_map["stage_name"]),
    )
    if attempt.get("stage") != dataset.stage or not _portable_input_binding_matches(
        attempt.get("input_binding", {}), dataset.input_binding, attempt_map
    ):
        raise RuntimeError("Archived attempt input contract changed")
    if (
        attempt.get("responses_file") != "responses.csv"
        or attempt.get("completed_rows") != dataset.stage["rows"]
        or attempt.get("error") is not None
        or attempt.get("recovery")
        != _recovery_contract("passed", bundle["transport"])
    ):
        raise RuntimeError("Archived passed-attempt metadata is contradictory")
    if _parse_utc(str(attempt["finished_utc"])) < _parse_utc(str(attempt["started_utc"])):
        raise RuntimeError("Archived attempt UTC interval is reversed")
    responses_path = attempt_root / "responses.csv"
    if sha256_file(responses_path) != attempt.get("responses_sha256"):
        raise RuntimeError("Archived response CSV hash changed")
    responses = _read_response_csv(responses_path)
    verification = verify_response_records(
        responses=responses,
        reference_rows=dataset.reference_rows,
        transport=bundle["transport"],
        controls=attempt.get("controls"),
        session_counters=attempt.get("session_network_counters"),
    )
    if attempt.get("verification") != verification:
        raise RuntimeError("Archived stage verification differs from recomputation")
    _attempt_wifi_binding(attempt, bundle["transport"])
    return attempt


def _verify_archived_session(
    *,
    root: Path,
    combination: Mapping[str, Any],
    combination_map: Mapping[str, Any],
    contract_model: Mapping[str, Any],
    export: FinalExportIdentity,
    cohort_dir: Path,
    host_source_root: Path,
    expected_builder_sha256: str,
    canonical_source_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    session_path = _safe_local(root, str(combination_map["session_local_path"]))
    session = read_json(session_path)
    _verify_seal(session, "session_evidence_id", "Session completion")
    if session.get("schema") != SESSION_SCHEMA or session.get("status") != "passed":
        raise RuntimeError("Archived session is not passed final evidence")
    if session.get("combination_id") != combination["combination_id"]:
        raise RuntimeError("Archived session combination differs from campaign")
    expected = {
        "model_key": combination["model_key"],
        "board": combination["board"],
        "transport": combination["transport"],
        "export_id": contract_model["export_id"],
        "model_sha256": contract_model["trained_state_sha256"],
        "checkpoint_file_sha256": contract_model["checkpoint_file_sha256"],
    }
    for field, value in expected.items():
        if session.get(field) != value:
            raise RuntimeError(f"Archived session differs for {field}")
    require_session_id(str(session.get("campaign_session_id", "")))
    bundle_dir = _safe_local(root, str(combination_map["bundle_local_dir"]))
    bundle = verify_final_bundle(
        bundle_dir,
        expected_export=export,
        expected_builder_sha256=expected_builder_sha256,
        canonical_source_root=canonical_source_root,
    )
    if (
        session.get("export_path_recorded")
        != combination_map.get("export_original_path")
        or session.get("export_manifest_sha256") != export.manifest_sha256
        or session.get("cohort_path_recorded")
        != combination_map.get("cohort_original_path")
        or session.get("cohort_manifest_sha256")
        != sha256_file(cohort_dir / "final_timing_cohort_manifest.json")
        or session.get("bundle_path_recorded")
        != combination_map.get("bundle_original_dir")
        or session.get("bundle_manifest_sha256") != bundle["_manifest_sha256"]
    ):
        raise RuntimeError("Archived session input/bundle path or hash binding changed")
    for field in ["bundle_id", "build_contract_id"]:
        if session.get(field) != bundle[field]:
            raise RuntimeError(f"Archived session differs from bundle for {field}")
    provenance = _verify_archived_provenance(
        root=root,
        provenance_map=combination_map["provenance"],
        bundle=bundle,
        host_source_root=host_source_root,
    )
    provenance_map = combination_map["provenance"]
    provenance_record = _safe_local(
        _safe_local(root, str(provenance_map["local_dir"])),
        str(provenance_map["record_relative_to_provenance"]),
    )
    if (
        session.get("provenance_path_recorded")
        != provenance_map.get("original_record_path")
        or session.get("provenance_sha256") != sha256_file(provenance_record)
        or session.get("provenance_sha256")
        != provenance_map.get("original_record_sha256")
    ):
        raise RuntimeError("Archived session provenance path/hash binding changed")
    for field in [
        "provenance_id",
        "bundle_id",
        "build_contract_id",
        "board",
        "transport",
        "student",
        "route",
        "export_id",
        "model_sha256",
        "checkpoint_file_sha256",
        "physical_port_serial",
    ]:
        if session.get(field) != provenance.get(field):
            raise RuntimeError(f"Archived session/provenance differs for {field}")
    attempt_maps = combination_map.get("attempts")
    if not isinstance(attempt_maps, list) or len(attempt_maps) != len(FINAL_STAGES):
        raise RuntimeError("Archived session does not map exactly six attempts")
    by_stage = {item.get("stage_name"): item for item in attempt_maps}
    if set(by_stage) != {stage["name"] for stage in FINAL_STAGES}:
        raise RuntimeError("Archived attempt semantic map is incomplete")
    for stage in FINAL_STAGES:
        mapped = by_stage[stage["name"]]
        if mapped.get("ordinal") != stage["ordinal"]:
            raise RuntimeError("Archived attempt semantic ordinal changed")
    attempts = [
        _verify_archived_attempt(
            root=root,
            attempt_map=by_stage[stage["name"]],
            session=session,
            bundle=bundle,
            export=export,
            cohort_dir=cohort_dir,
            host_source_root=host_source_root,
        )
        for stage in FINAL_STAGES
    ]
    connection_map = combination_map.get("connection")
    connection_path = None
    if session["transport"] == "wifi_udp":
        if not isinstance(connection_map, Mapping):
            raise RuntimeError("Archived Wi-Fi session lacks its connection record")
        connection_path = _safe_local(root, str(connection_map.get("local_path")))
        if (
            session.get("connection_path_recorded")
            != connection_map.get("original_path")
            or session.get("connection_record_sha256")
            != connection_map.get("original_record_sha256")
            or sha256_file(connection_path)
            != connection_map.get("original_record_sha256")
        ):
            raise RuntimeError("Archived Wi-Fi connection map differs from its session")
    elif connection_map is not None:
        raise RuntimeError("Archived USB session carries a Wi-Fi connection record")
    connection = validate_session_connection_record(
        session=session,
        attempts=attempts,
        bundle=bundle,
        connection_json=connection_path,
        host_source_root=host_source_root,
    )
    stage_ledger = session.get("stages")
    totals = validate_session_stage_ledger(stage_ledger)
    if session.get("row_totals") != totals:
        raise RuntimeError("Archived session row totals changed")
    if session.get("warmup_excluded_from_reported_metrics") is not True or session.get(
        "smoke_excluded_from_reported_metrics"
    ) is not True:
        raise RuntimeError("Archived session reporting exclusions changed")
    if session.get("timing_statistical_unit") != TIMING_STATISTICAL_UNIT:
        raise RuntimeError("Archived session timing statistical unit changed")
    previous_finished: datetime | None = None
    for stage_record, attempt in zip(stage_ledger, attempts):
        if (
            stage_record.get("attempt_id") != attempt.get("attempt_id")
            or stage_record.get("attempt_manifest_sha256")
            != sha256_file(
                _safe_local(root, str(by_stage[stage_record["name"]]["local_dir"]))
                / "final_attempt.json"
            )
            or stage_record.get("started_utc") != attempt.get("started_utc")
            or stage_record.get("finished_utc") != attempt.get("finished_utc")
            or stage_record.get("attempt_path_recorded")
            != by_stage[stage_record["name"]].get("original_dir")
        ):
            raise RuntimeError(f"Archived session stage ledger differs: {stage_record['name']}")
        serial = attempt.get("physical_identity", {}).get("physical_port_serial")
        if serial != session.get("physical_port_serial"):
            raise RuntimeError("Archived attempts use another physical board")
        if session["transport"] == "wifi_udp" and attempt.get("physical_identity", {}).get(
            "wifi_mac_reported"
        ) != session.get("wifi_mac_reported"):
            raise RuntimeError("Archived Wi-Fi attempts use another physical radio")
        started = _parse_utc(str(attempt["started_utc"]))
        finished = _parse_utc(str(attempt["finished_utc"]))
        if previous_finished is not None and started < previous_finished:
            raise RuntimeError("Archived session stages overlap or are reordered")
        previous_finished = finished
    if session.get("started_utc") != attempts[0]["started_utc"] or session.get(
        "finished_utc"
    ) != attempts[-1]["finished_utc"]:
        raise RuntimeError("Archived session interval differs from attempts")
    if _parse_utc(str(provenance["finished_utc"])) > _parse_utc(
        str(session["started_utc"])
    ):
        raise RuntimeError("Archived build/upload overlaps stage execution")
    if connection is not None and _parse_utc(
        str(provenance["finished_utc"])
    ) > _parse_utc(str(connection["started_utc"])):
        raise RuntimeError("Archived build/upload overlaps Wi-Fi provisioning")
    return session, bundle, provenance


def verify_campaign_archive(
    archive_dir: Path,
    *,
    verifier: Verifier | None = None,
) -> dict[str, Any]:
    """Deeply verify an archive without dereferencing any original path."""

    source_root = Path(archive_dir)
    if source_root.is_symlink():
        raise RuntimeError("Final-HIL archive root cannot be a symlink")
    root = source_root.resolve()
    manifest = read_json(root / ARCHIVE_MANIFEST)
    recorded_id = manifest.get("archive_id")
    copy = dict(manifest)
    copy.pop("archive_id", None)
    if recorded_id != canonical_json_sha256(copy):
        raise RuntimeError("Final-HIL archive ID is invalid")
    if manifest.get("schema") != ARCHIVE_SCHEMA:
        raise RuntimeError("Unsupported final-HIL archive schema")
    _verify_inventory(root, manifest)
    semantic = manifest["semantic_map"]
    archive_execution_root = _verify_archive_execution_sources(
        root, semantic.get("archive_execution_sources")
    )
    if not archive_execution_root.is_dir():
        raise RuntimeError("Archive execution-source root is invalid")
    archived_bundle_builder_sha256 = sha256_file(
        archive_execution_root / "deployment" / "final_hil" / "bundles.py"
    )
    campaign_map = semantic.get("campaign")
    if not isinstance(campaign_map, Mapping):
        raise RuntimeError("Archive campaign semantic map is missing")
    contract = validate_campaign_contract(
        _safe_local(root, str(campaign_map["contract_local_path"]))
    )
    if manifest.get("contract_id") != contract["contract_id"]:
        raise RuntimeError("Archive manifest belongs to another campaign contract")

    model_map = semantic.get("models")
    if not isinstance(model_map, Mapping) or set(model_map) != set(MODEL_KEYS):
        raise RuntimeError("Archive does not retain all four intended model entries")
    identities: dict[str, FinalExportIdentity] = {}
    blocked_models: set[str] = set()
    for key in MODEL_KEYS:
        contract_model = contract["models"][key]
        archived_model = model_map[key]
        if archived_model.get("status") != contract_model.get("status"):
            raise RuntimeError(f"Archive model status differs for {key}")
        if contract_model.get("status") == "passed":
            export_dir = _safe_local(root, str(archived_model["export_local_dir"]))
            identity = validate_final_export(
                export_dir,
                verifier=verifier or _portable_final_export_verifier,
            )
            if identity.model_key != key:
                raise RuntimeError(f"Archive export identity differs for {key}")
            for field, value in identity.to_dict().items():
                if field == "root":
                    continue
                if contract_model.get(field) != value:
                    raise RuntimeError(f"Archive export differs from contract: {key}:{field}")
            identities[key] = identity
        elif contract_model.get("status") == "blocked":
            audit_path = _safe_local(root, str(archived_model["blocked_audit_local_path"]))
            blocked = validate_blocked_audit(
                audit_path,
                verifier=_portable_blocked_audit_verifier,
            )
            if (
                blocked["model_key"] != key
                or blocked["blocked_audit_sha256"]
                != contract_model.get("blocked_audit_sha256")
            ):
                raise RuntimeError(f"Archived blocked audit differs for {key}")
            blocked_models.add(key)
        else:
            blocked_models.add(key)

    executable = contract["status"] in {"ready", "ready_with_blocked_routes"}
    if not executable:
        if campaign_map.get("evidence_local_path") is not None:
            raise RuntimeError("Blocked archive contains passed campaign evidence")
        if semantic.get("combinations"):
            raise RuntimeError("Blocked archive must not imply a completed reduced matrix")
        if not blocked_models:
            raise RuntimeError("Blocked archive has no retained blocker")
        if manifest.get("status") != "blocked" or manifest.get(
            "campaign_evidence_id"
        ) is not None:
            raise RuntimeError("Blocked archive status is inconsistent")
        return {
            "schema": ARCHIVE_SCHEMA,
            "status": "blocked",
            "archive_id": recorded_id,
            "contract_id": contract["contract_id"],
            "blocked_models": sorted(blocked_models),
            "matrix_was_not_reduced": len(contract["combinations"])
            == len(MODEL_KEYS) * len(BOARDS) * len(contract["transports"]),
        }

    expected_eligible_models = {
        key for key in MODEL_KEYS if contract["models"][key].get("status") == "passed"
    }
    if set(identities) != expected_eligible_models:
        raise RuntimeError("Ready archive does not contain every eligible export")
    cohort_map = semantic.get("cohort")
    if not isinstance(cohort_map, Mapping):
        raise RuntimeError("Ready archive lacks the balanced cohort")
    cohort_dir = _safe_local(root, str(cohort_map["local_dir"]))
    validate_balanced_cohort(
        cohort_dir,
        identities=identities,
        reconstruct_sources=False,
    )
    evidence_path_text = campaign_map.get("evidence_local_path")
    if not isinstance(evidence_path_text, str):
        raise RuntimeError("Ready archive lacks campaign evidence")
    evidence = read_json(_safe_local(root, evidence_path_text))
    _validate_campaign_evidence_header(evidence, contract)
    if manifest.get("campaign_evidence_id") != evidence["campaign_evidence_id"]:
        raise RuntimeError("Archive manifest campaign evidence ID differs")

    combination_map = semantic.get("combinations")
    expected_combinations = contract["eligible_combinations"]
    expected_ids = [item["combination_id"] for item in expected_combinations]
    if not isinstance(combination_map, Mapping) or set(combination_map) != set(expected_ids):
        raise RuntimeError("Archive campaign matrix is incomplete or reduced")
    sessions: list[dict[str, Any]] = []
    bundles: dict[str, dict[str, Any]] = {}
    provenances: dict[str, dict[str, Any]] = {}
    host_source_root = archive_execution_root
    for combination in expected_combinations:
        combination_id = combination["combination_id"]
        mapped = combination_map[combination_id]
        if (
            mapped.get("execution_ordinal") != combination["execution_ordinal"]
            or mapped.get("model_key") != combination["model_key"]
            or mapped.get("board") != combination["board"]
            or mapped.get("transport") != combination["transport"]
        ):
            raise RuntimeError(f"Archive combination semantic map differs: {combination_id}")
        session, bundle, provenance = _verify_archived_session(
            root=root,
            combination=combination,
            combination_map=mapped,
            contract_model=contract["models"][combination["model_key"]],
            export=identities[combination["model_key"]],
            cohort_dir=cohort_dir,
            host_source_root=host_source_root,
            expected_builder_sha256=archived_bundle_builder_sha256,
            canonical_source_root=archive_execution_root,
        )
        sessions.append(session)
        bundles[combination_id] = bundle
        provenances[combination_id] = provenance

    totals = validate_campaign_session_ledger(contract, sessions)
    if evidence.get("totals") != totals:
        raise RuntimeError("Archived campaign totals differ from recomputation")
    evidence_sessions = evidence.get("sessions")
    if not isinstance(evidence_sessions, list) or [
        item.get("combination_id") for item in evidence_sessions
    ] != expected_ids:
        raise RuntimeError("Archived campaign evidence session order changed")
    by_id = {session["combination_id"]: session for session in sessions}
    for field in [
        "session_evidence_id",
        "campaign_session_id",
        "bundle_id",
        "provenance_id",
    ]:
        values = [session.get(field) for session in sessions]
        if len(values) != len(set(values)):
            raise RuntimeError(f"Archived campaign duplicates session identity: {field}")
    for item in evidence_sessions:
        session = by_id[item["combination_id"]]
        if item.get("session_evidence_id") != session["session_evidence_id"]:
            raise RuntimeError("Campaign/session evidence ID differs")
        local_session = _safe_local(
            root,
            str(combination_map[item["combination_id"]]["session_local_path"]),
        )
        if item.get("session_sha256") != sha256_file(local_session):
            raise RuntimeError("Campaign/session file hash differs")
        if item.get("session_path_recorded") != combination_map[
            item["combination_id"]
        ].get("session_original_path"):
            raise RuntimeError("Campaign historical session path differs from semantic map")

    specimens: dict[str, dict[str, Any]] = {}
    build_contracts: dict[str, str] = {}
    wifi_in_scope = "wifi_udp" in contract["transports"]
    for board in BOARDS:
        board_sessions = [session for session in sessions if session["board"] == board]
        serials = {session.get("physical_port_serial") for session in board_sessions}
        wifi_macs = {
            session.get("wifi_mac_reported")
            for session in board_sessions
            if session["transport"] == "wifi_udp"
        }
        board_builds = {session.get("build_contract_id") for session in board_sessions}
        if len(serials) != 1 or not next(iter(serials)):
            raise RuntimeError(f"Archive uses ambiguous physical specimens for {board}")
        if wifi_in_scope and (len(wifi_macs) != 1 or not next(iter(wifi_macs))):
            raise RuntimeError(f"Archive uses ambiguous Wi-Fi radios for {board}")
        if len(board_builds) != 1 or not next(iter(board_builds)):
            raise RuntimeError(f"Archive uses inconsistent build contracts for {board}")
        build_contracts[board] = str(next(iter(board_builds)))
        wifi_mac = next(iter(wifi_macs)) if wifi_macs else None
        specimens[board] = {
            "physical_port_serial": next(iter(serials)),
            "wifi_mac_reported": wifi_mac,
            "session_count": len(board_sessions),
        }
    if evidence.get("physical_specimens") != specimens:
        raise RuntimeError("Archived campaign physical specimen ledger changed")
    if evidence.get("boards") != list(BOARDS) or evidence.get("transports") != list(
        contract["transports"]
    ):
        raise RuntimeError("Archived campaign board/transport scope changed")
    if evidence.get("all_four_models_retained") is not True:
        raise RuntimeError("Archived campaign does not retain all four models")
    if evidence.get("all_gate_eligible_combinations_executed") is not True:
        raise RuntimeError("Archived campaign does not execute every eligible combination")
    if evidence.get("blocked_routes") != contract.get("blocked_routes") or evidence.get(
        "excluded_combinations"
    ) != contract.get("excluded_combinations"):
        raise RuntimeError("Archived blocked-route ledger differs from the contract")
    if evidence.get("board_build_contracts") != build_contracts:
        raise RuntimeError("Archived campaign board build-contract ledger changed")
    previous_finished: datetime | None = None
    for combination in expected_combinations:
        session = by_id[combination["combination_id"]]
        started = _parse_utc(str(session["started_utc"]))
        finished = _parse_utc(str(session["finished_utc"]))
        if previous_finished is not None and started < previous_finished:
            raise RuntimeError("Archived campaign execution order is invalid")
        previous_finished = finished

    host_sources = semantic.get("host_sources")
    if not isinstance(host_sources, list):
        raise RuntimeError("Archive host-source map is missing")
    source_by_key: dict[str, Mapping[str, Any]] = {}
    for item in host_sources:
        if not isinstance(item, Mapping) or not _is_sha256(item.get("sha256")):
            raise RuntimeError("Archive host-source entry is malformed")
        digest = str(item["sha256"])
        expected_relative = item.get("expected_relative_path")
        if not isinstance(expected_relative, str):
            raise RuntimeError("Archive host source lacks a canonical relative path")
        key = f"path:{expected_relative}"
        if key in source_by_key:
            raise RuntimeError("Archive host-source identity is duplicated")
        path = _safe_local(root, str(item["local_path"]))
        if (
            item.get("local_path")
            != f"archive_execution/repository/{expected_relative}"
            or not path.is_file()
            or path.stat().st_size != item.get("size_bytes")
            or sha256_file(path) != digest
        ):
            raise RuntimeError("Archived host-source snapshot changed")
        source_by_key[key] = item
    required_builder_hashes = {
        str(bundle["bundle_identity_payload"]["bundle_builder_sha256"])
        for bundle in bundles.values()
    }
    bundle_source = source_by_key.get("path:deployment/final_hil/bundles.py")
    if (
        len(required_builder_hashes) != 1
        or bundle_source is None
        or bundle_source.get("sha256") != next(iter(required_builder_hashes))
    ):
        raise RuntimeError("Archive lacks the exact bundle-builder source snapshot")
    required_relative_sources: dict[str, str] = {}
    for provenance in provenances.values():
        environment = provenance.get("host_environment", {})
        for dependency in environment.get("source_dependencies", []):
            required_relative_sources[str(dependency["path"])] = str(
                dependency["sha256"]
            )
    for session in sessions:
        for stage in session["stages"]:
            attempt_map = next(
                item
                for item in combination_map[session["combination_id"]]["attempts"]
                if item["stage_name"] == stage["name"]
            )
            attempt = read_json(
                _safe_local(root, str(attempt_map["local_dir"])) / "final_attempt.json"
            )
            environment = attempt.get("host_environment", {})
            for dependency in environment.get("source_dependencies", []):
                required_relative_sources[str(dependency["path"])] = str(
                    dependency["sha256"]
                )
    for relative, digest in required_relative_sources.items():
        item = source_by_key.get(f"path:{relative}")
        if item is None or item.get("sha256") != digest:
            raise RuntimeError("Archive lacks an exact relative host-source snapshot")
    if manifest.get("status") != evidence["status"]:
        raise RuntimeError("Ready campaign archive status is not passed")
    return {
        "schema": ARCHIVE_SCHEMA,
        "status": evidence["status"],
        "archive_id": recorded_id,
        "contract_id": contract["contract_id"],
        "campaign_evidence_id": evidence["campaign_evidence_id"],
        "combination_count": len(sessions),
        "totals": totals,
        "physical_specimens": specimens,
        "build_contracts_by_board": build_contracts,
        "portable": True,
        "original_source_paths_dereferenced": False,
        "archive_local_source_snapshots_verified": True,
        "verifier_execution_mode": "current_checkout_with_archived_source_ledger_verified",
        "portable_selection_boundary": (
            "Archive-local verification binds the cohort rows and preprocessing hashes "
            "to the archived full exports. Reconstructing deterministic group selection "
            "from the original dataset remains a generation-time check."
        ),
    }
