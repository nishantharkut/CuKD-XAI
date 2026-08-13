"""Canonical export, cohort, and final-campaign contracts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


CAMPAIGN_PROTOCOL_ID = "cukd_final_hil_campaign_seed42_v1"
COHORT_PROTOCOL_ID = "cukd_final_hil_balanced_timing_1000_v1"
SOURCE_PROTOCOL_ID = "wsnds_feature_group_split_train_only_scaler_10seed_v2"
EXPECTED_SEED = 42
EXPECTED_FULL_ROWS = 56_301
EXPECTED_SPLIT_SIZES = {"train": 262_197, "validation": 56_163, "test": EXPECTED_FULL_ROWS}
GROUPS_PER_CLASS = 200
CLASS_COUNT = 5
CLASS_NAMES = ("Blackhole", "Flooding", "Grayhole", "Normal", "TDMA")
_FINAL_EXPORT_VERIFIER_LOCK = threading.RLock()
_GENERATED_HEADER_NAMES = frozenset(
    {"model_weights.h", "preprocess_metadata.h", "preprocess_int_metadata.h"}
)
MODEL_KEYS = (
    "student_A_scratch",
    "student_A_rf_kd",
    "student_B_scratch",
    "student_B_rf_kd",
)
BOARDS = ("esp32c3", "arduino_r4")
TRANSPORTS = ("usb_serial", "wifi_udp")
FINAL_STAGES = (
    {
        "name": "warmup_10",
        "ordinal": 1,
        "rows": 10,
        "input_role": "balanced_timing_warmup",
        "include_in_reported_metrics": False,
        "include_in_timing_metrics": False,
        "include_in_fidelity_metrics": False,
    },
    {
        "name": "smoke_10",
        "ordinal": 2,
        "rows": 10,
        "input_role": "full_replay_smoke_prefix",
        "include_in_reported_metrics": False,
        "include_in_timing_metrics": False,
        "include_in_fidelity_metrics": False,
    },
    {
        "name": "timing_1000_r1",
        "ordinal": 3,
        "rows": 1000,
        "input_role": "balanced_timing",
        "include_in_reported_metrics": True,
        "include_in_timing_metrics": True,
        "include_in_fidelity_metrics": False,
        "timing_repeat": 1,
    },
    {
        "name": "timing_1000_r2",
        "ordinal": 4,
        "rows": 1000,
        "input_role": "balanced_timing",
        "include_in_reported_metrics": True,
        "include_in_timing_metrics": True,
        "include_in_fidelity_metrics": False,
        "timing_repeat": 2,
    },
    {
        "name": "timing_1000_r3",
        "ordinal": 5,
        "rows": 1000,
        "input_role": "balanced_timing",
        "include_in_reported_metrics": True,
        "include_in_timing_metrics": True,
        "include_in_fidelity_metrics": False,
        "timing_repeat": 3,
    },
    {
        "name": "full_56301",
        "ordinal": 6,
        "rows": EXPECTED_FULL_ROWS,
        "input_role": "full_replay",
        "include_in_reported_metrics": True,
        "include_in_timing_metrics": False,
        "include_in_fidelity_metrics": True,
    },
)

Verifier = Callable[[Path], Mapping[str, Any]]
BlockedVerifier = Callable[[Path], Mapping[str, Any]]


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Stale temporary evidence exists: {temporary}")
    temporary_created = False
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            temporary_created = True
            handle.write(json.dumps(payload, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"Refusing to overwrite evidence: {path}") from exc
    finally:
        if temporary_created and temporary.exists():
            temporary.unlink()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _nested(container: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = container
    for part in path:
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _first(
    containers: Iterable[Mapping[str, Any]],
    *paths: Sequence[str],
) -> Any:
    for container in containers:
        for path in paths:
            value = _nested(container, path)
            if value is not None:
                return value
    return None


def _verify_optional_payload_hash(
    payload: Mapping[str, Any], field: str, label: str
) -> None:
    if field not in payload:
        return
    copy = dict(payload)
    recorded = copy.pop(field)
    if recorded != canonical_json_sha256(copy):
        raise RuntimeError(f"{label} canonical payload hash is invalid")


def _manifest_files(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    files = manifest.get("files")
    if files is None and isinstance(manifest.get("inventory"), Mapping):
        files = manifest["inventory"].get("files")
    if not isinstance(files, list):
        raise RuntimeError("Final export manifest has no file inventory")
    result: list[dict[str, Any]] = []
    for item in files:
        if not isinstance(item, dict):
            raise RuntimeError("Final export inventory contains a non-object entry")
        result.append(item)
    return result


def _verify_inventory(root: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    manifest_path = root / "final_export_manifest.json"
    if manifest_path.is_symlink():
        raise RuntimeError("Final export manifest cannot be a symlink")
    files = _manifest_files(manifest)
    listed: dict[str, str] = {}
    for item in files:
        relative = item.get("path")
        expected_hash = item.get("sha256")
        expected_size = item.get("size_bytes")
        if not isinstance(relative, str) or not relative or not _is_sha256(expected_hash):
            raise RuntimeError("Final export inventory entry is malformed")
        relative_path = Path(relative)
        normalized_relative = relative_path.as_posix()
        if (
            "\\" in relative
            or normalized_relative != relative
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or normalized_relative in listed
        ):
            raise RuntimeError(f"Unsafe or duplicate final export path: {relative!r}")
        member_path = root / relative_path
        if member_path.is_symlink():
            raise RuntimeError(f"Final export member cannot be a symlink: {relative}")
        member = member_path.resolve()
        try:
            member.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"Final export path escapes its root: {relative!r}") from exc
        if not member.is_file():
            raise RuntimeError(f"Final export member is missing: {relative}")
        if expected_size is not None and member.stat().st_size != int(expected_size):
            raise RuntimeError(f"Final export member size changed: {relative}")
        if sha256_file(member) != expected_hash:
            raise RuntimeError(f"Final export member hash changed: {relative}")
        listed[normalized_relative] = expected_hash
    required = {
        "final_export_identity.json",
        "final_export_report.json",
        "hil_replay_vectors.csv",
        "hil_reference_predictions.csv",
        "cukd_export_identity.h",
        "model_weights.h",
        "preprocess_int_metadata.h",
    }
    if not required <= set(listed):
        raise RuntimeError(
            f"Final export inventory is missing: {sorted(required - set(listed))}"
        )
    recorded_count = manifest.get("file_count_excluding_manifest")
    if recorded_count is not None and int(recorded_count) != len(listed):
        raise RuntimeError("Final export inventory count is invalid")
    manifest_path = manifest_path.resolve()
    members = list(root.rglob("*"))
    if any(path.is_symlink() for path in members):
        raise RuntimeError("Final export contains a symlink")
    actual = {
        path.relative_to(root).as_posix()
        for path in members
        if path.is_file() and path.resolve() != manifest_path
    }
    if actual != set(listed):
        raise RuntimeError("Final export contains extra or unlisted files")
    return listed


@dataclass(frozen=True)
class FinalExportIdentity:
    root: str
    protocol: str
    seed: int
    student: str
    route: str
    checkpoint_file_sha256: str
    trained_state_sha256: str
    export_id: str
    dataset_sha256: str
    split_indices_sha256: str
    scaler_sha256: str
    manifest_sha256: str
    report_sha256: str
    full_replay_sha256: str
    full_reference_sha256: str
    test_rows: int
    gate_status: str

    @property
    def model_key(self) -> str:
        return model_key(self.student, self.route)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def model_key(student: str, route: str) -> str:
    normalized_student = str(student).upper()
    normalized_route = str(route).lower()
    key = f"student_{normalized_student}_{normalized_route}"
    if key not in MODEL_KEYS:
        raise ValueError(f"Unsupported final model identity: {student}/{route}")
    return key


def _canonical_final_verifier(root: Path) -> Mapping[str, Any]:
    from deployment.firmware_export.wsnds_final_hil.export_final_seed42 import (
        verify_final_export,
    )

    # The preserved exports were generated on Windows and their raw manifest
    # hashes correctly bind CRLF header bytes.  The exporter-owned verifier
    # regenerates those headers in a temporary directory; without this narrow
    # adapter, Linux writes LF and fails a raw-byte comparison even when every
    # generated line and numeric value is identical.  Preserve the newline
    # convention of each sealed header while the canonical verifier runs.
    with _FINAL_EXPORT_VERIFIER_LOCK:
        original_write_text = Path.write_text
        owner_thread = threading.get_ident()

        def write_text_with_sealed_newlines(
            path: Path,
            data: str,
            encoding: str | None = None,
            errors: str | None = None,
            newline: str | None = None,
        ) -> int:
            if (
                threading.get_ident() == owner_thread
                and path.name in _GENERATED_HEADER_NAMES
                and path.parent.name.startswith("cukd_lineage_headers_")
            ):
                sealed = root / path.name
                if not sealed.is_file() or sealed.is_symlink():
                    raise RuntimeError(
                        f"Sealed generated header is unavailable: {path.name}"
                    )
                raw = sealed.read_bytes()
                without_crlf = raw.replace(b"\r\n", b"\n")
                if b"\r" in without_crlf:
                    raise RuntimeError(
                        f"Sealed generated header has unsupported line endings: {path.name}"
                    )
                newline = "\r\n" if b"\r\n" in raw else "\n"
                data = data.replace("\r\n", "\n")
                if "\r" in data:
                    raise RuntimeError(
                        f"Regenerated header has unsupported line endings: {path.name}"
                    )
            return original_write_text(
                path,
                data,
                encoding=encoding,
                errors=errors,
                newline=newline,
            )

        Path.write_text = write_text_with_sealed_newlines
        try:
            return verify_final_export(root)
        finally:
            Path.write_text = original_write_text


def validate_final_export(
    root: Path,
    *,
    verifier: Verifier | None = None,
) -> FinalExportIdentity:
    """Verify and adapt one final export through a single schema boundary.

    The exporter-owned verifier remains authoritative.  The checks below bind
    the fields consumed by final HIL so a straightforward exporter schema change
    is isolated to this function.
    """

    source_root = Path(root)
    if source_root.is_symlink():
        raise RuntimeError("Final export root cannot be a symlink")
    resolved = source_root.resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)
    verification = dict((verifier or _canonical_final_verifier)(resolved))
    if verification.get("status") != "passed":
        raise RuntimeError("The exporter-owned final verification did not pass")

    manifest_path = resolved / "final_export_manifest.json"
    report_path = resolved / "final_export_report.json"
    identity_path = resolved / "final_export_identity.json"
    manifest = read_json(manifest_path)
    report = read_json(report_path)
    identity_file = read_json(identity_path)
    _verify_optional_payload_hash(manifest, "manifest_payload_sha256", "Manifest")
    _verify_optional_payload_hash(report, "report_payload_sha256", "Report")
    inventory = _verify_inventory(resolved, manifest)

    report_identity = report.get("identity")
    if report_identity is not None and report_identity != identity_file:
        raise RuntimeError("Final report and identity file disagree")
    identity = identity_file
    export_id = _first(
        [identity, report, manifest, verification],
        ("export_id",),
        ("identity", "export_id"),
    )
    if not _is_sha256(export_id):
        raise RuntimeError("Final export ID is missing or malformed")
    identity_without_id = dict(identity)
    identity_without_id.pop("export_id", None)
    if "export_id" in identity and canonical_json_sha256(identity_without_id) != export_id:
        raise RuntimeError("Final export ID does not hash its identity payload")
    if manifest.get("identity_canonical_sha256") is not None and manifest.get(
        "identity_canonical_sha256"
    ) != canonical_json_sha256(identity):
        raise RuntimeError("Final manifest identity canonical hash is invalid")
    if manifest.get("report_canonical_sha256") is not None and manifest.get(
        "report_canonical_sha256"
    ) != canonical_json_sha256(report):
        raise RuntimeError("Final manifest report canonical hash is invalid")

    containers = [identity, report, manifest, verification]
    protocol = _first(containers, ("protocol",), ("protocol_id",), ("identity", "protocol"))
    seed = _first(containers, ("seed",), ("identity", "seed"))
    student = _first(containers, ("student",), ("identity", "student"))
    route = _first(containers, ("route",), ("identity", "route"))
    checkpoint_hash = _first(
        containers,
        ("checkpoint_file_sha256",),
        ("model_file_sha256",),
        ("source_artifacts", "checkpoint_file_sha256"),
        ("identity", "checkpoint_file_sha256"),
    )
    trained_hash = _first(
        containers,
        ("trained_state_sha256",),
        ("identity", "trained_state_sha256"),
    )
    dataset_hash = _first(containers, ("dataset_sha256",), ("identity", "dataset_sha256"))
    split_hash = _first(
        containers, ("split_indices_sha256",), ("identity", "split_indices_sha256")
    )
    scaler_hash = _first(containers, ("scaler_sha256",), ("identity", "scaler_sha256"))
    test_rows = _first(
        [report, verification, manifest],
        ("gates", "test_rows"),
        ("test_rows",),
        ("full_test_rows",),
    )
    gate_passed = _first(
        [report, verification],
        ("gates", "quality_gates_passed"),
        ("quality_gates_passed",),
    )
    gate_status = _first(
        [report, verification],
        ("gates", "gate_status"),
        ("gate_status",),
    )
    if gate_passed is None and isinstance(gate_status, str):
        gate_passed = gate_status.lower() == "passed"

    if not isinstance(protocol, str) or not protocol:
        raise RuntimeError("Final export protocol is missing")
    if int(seed) != EXPECTED_SEED:
        raise RuntimeError(f"Final HIL requires seed {EXPECTED_SEED}")
    key = model_key(str(student), str(route))
    del key
    for label, value in [
        ("checkpoint", checkpoint_hash),
        ("trained state", trained_hash),
        ("dataset", dataset_hash),
        ("split indices", split_hash),
        ("scaler", scaler_hash),
    ]:
        if not _is_sha256(value):
            raise RuntimeError(f"Final export {label} hash is missing or malformed")
    if int(test_rows) != EXPECTED_FULL_ROWS:
        raise RuntimeError(
            f"Final export has {test_rows} test rows; expected {EXPECTED_FULL_ROWS}"
        )
    if manifest.get("status") != "passed" or report.get("status") != "passed":
        raise RuntimeError("Final export manifest/report status is not passed")
    if gate_passed is not True:
        raise RuntimeError("Final export fixed-point quality gates are not passed")
    for container_name, container in [("manifest", manifest), ("verification", verification)]:
        observed_id = _first([container], ("export_id",), ("identity", "export_id"))
        if observed_id is not None and observed_id != export_id:
            raise RuntimeError(f"Final {container_name} export ID disagrees")
    for field, expected in {
        "protocol_id": protocol,
        "seed": int(seed),
        "student": str(student).upper(),
        "route": str(route).lower(),
    }.items():
        observed = manifest.get(field)
        if observed is not None and (
            str(observed).lower() if field in {"student", "route"} else observed
        ) != (
            str(expected).lower() if field in {"student", "route"} else expected
        ):
            raise RuntimeError(f"Final manifest identity mismatch for {field}")

    return FinalExportIdentity(
        root=str(resolved),
        protocol=protocol,
        seed=int(seed),
        student=str(student).upper(),
        route=str(route).lower(),
        checkpoint_file_sha256=str(checkpoint_hash),
        trained_state_sha256=str(trained_hash),
        export_id=str(export_id),
        dataset_sha256=str(dataset_hash),
        split_indices_sha256=str(split_hash),
        scaler_sha256=str(scaler_hash),
        manifest_sha256=sha256_file(manifest_path),
        report_sha256=sha256_file(report_path),
        full_replay_sha256=inventory["hil_replay_vectors.csv"],
        full_reference_sha256=inventory["hil_reference_predictions.csv"],
        test_rows=int(test_rows),
        gate_status="passed",
    )


def _canonical_blocked_verifier(path: Path) -> Mapping[str, Any]:
    from deployment.firmware_export.wsnds_final_hil.export_final_seed42 import (
        verify_blocked_audit,
    )

    return verify_blocked_audit(path)


def validate_blocked_audit(
    path: Path, *, verifier: BlockedVerifier | None = None
) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink():
        raise RuntimeError("Blocked audit cannot be a symlink")
    resolved = source.resolve()
    payload = read_json(resolved)
    verification = dict((verifier or _canonical_blocked_verifier)(resolved))
    if verification.get("status") != "blocked_verified":
        raise RuntimeError("The exporter-owned blocked-audit verification did not pass")
    if "audit_payload_sha256" not in payload:
        raise RuntimeError("Blocked audit canonical payload hash is missing")
    _verify_optional_payload_hash(payload, "audit_payload_sha256", "Blocked audit")
    if payload.get("status") != "blocked":
        raise RuntimeError("A blocked route audit must have status='blocked'")
    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        identity = payload
    identity_without_id = dict(identity)
    blocked_id = identity_without_id.pop("blocked_audit_id", None)
    if blocked_id != canonical_json_sha256(identity_without_id):
        raise RuntimeError("Blocked route identity hash is invalid")
    protocol = identity.get(
        "protocol", identity.get("protocol_id", payload.get("protocol_id"))
    )
    if not isinstance(protocol, str) or not protocol:
        raise RuntimeError("Blocked route audit lacks protocol identity")
    if identity.get("seed", payload.get("seed")) != EXPECTED_SEED:
        raise RuntimeError("Blocked route audit is not for final seed 42")
    key = model_key(
        str(identity.get("student", payload.get("student"))),
        str(identity.get("route", payload.get("route"))),
    )
    if (
        verification.get("student") != str(identity.get("student")).upper()
        or verification.get("route") != str(identity.get("route")).lower()
        or verification.get("blocked_audit_id") != blocked_id
    ):
        raise RuntimeError("Blocked route identity disagrees with exporter verification")
    quality_gates_passed = _first(
        [payload],
        ("quality_gates_passed",),
        ("gates", "quality_gates_passed"),
    )
    if quality_gates_passed is not False:
        raise RuntimeError("Blocked route audit does not record a failed quality gate")
    reason = payload.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise RuntimeError("Blocked route audit lacks a reason")
    return {
        "model_key": key,
        "status": "blocked",
        "protocol": protocol,
        "seed": EXPECTED_SEED,
        "student": str(identity.get("student", payload.get("student"))).upper(),
        "route": str(identity.get("route", payload.get("route"))).lower(),
        "reason": reason,
        "blocked_audit_id": blocked_id,
        "blocked_audit_path_recorded": str(resolved),
        "blocked_audit_sha256": sha256_file(resolved),
        "fixed_vs_fp32_agreement": verification.get("fixed_vs_fp32_agreement"),
        "absolute_macro_f1_drop": verification.get("absolute_macro_f1_drop"),
    }


def _campaign_row_totals(combination_count: int) -> dict[str, int]:
    def rows_for(role: str) -> int:
        return combination_count * sum(
            int(stage["rows"])
            for stage in FINAL_STAGES
            if stage["input_role"] == role
        )

    return {
        "warmup_excluded": rows_for("balanced_timing_warmup"),
        "smoke": rows_for("full_replay_smoke_prefix"),
        "balanced_timing": rows_for("balanced_timing"),
        "full_exact_replay": rows_for("full_replay"),
        "all_device_inferences": combination_count
        * sum(int(stage["rows"]) for stage in FINAL_STAGES),
    }


def _selected_transports(transports: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(transports)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Campaign transport scope must be nonempty and unique")
    if any(transport not in TRANSPORTS for transport in selected):
        raise ValueError(f"Unsupported campaign transport scope: {selected}")
    canonical = tuple(transport for transport in TRANSPORTS if transport in selected)
    if selected != canonical:
        raise ValueError("Campaign transports must follow canonical order")
    return selected


def _derived_campaign_shape(
    transports: Sequence[str] = TRANSPORTS,
) -> dict[str, Any]:
    selected_transports = _selected_transports(transports)
    combinations: list[dict[str, Any]] = []
    for board_index, board in enumerate(BOARDS):
        ordered_models = MODEL_KEYS if board_index == 0 else tuple(reversed(MODEL_KEYS))
        for model in ordered_models:
            model_index = MODEL_KEYS.index(model)
            ordered_transports = (
                selected_transports
                if (board_index + model_index) % 2 == 0
                else tuple(reversed(selected_transports))
            )
            for transport in ordered_transports:
                combinations.append(
                    {
                        "execution_ordinal": len(combinations) + 1,
                        "combination_id": f"{model}__{board}__{transport}",
                        "model_key": model,
                        "board": board,
                        "transport": transport,
                    }
                )
    count = len(combinations)
    return {
        "combinations": combinations,
        "expected_combination_count": count,
        "expected_stage_attempts": count * len(FINAL_STAGES),
        "expected_rows": _campaign_row_totals(count),
    }


def _execution_shape(combinations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(combinations)
    return {
        "expected_eligible_combination_count": count,
        "expected_eligible_stage_attempts": count * len(FINAL_STAGES),
        "expected_eligible_rows": _campaign_row_totals(count),
    }


def build_campaign_contract(
    model_sources: Mapping[str, Mapping[str, Path] | Path],
    *,
    transports: Sequence[str] = TRANSPORTS,
    output_path: Path | None = None,
    verifier: Verifier | None = None,
    blocked_verifier: BlockedVerifier | None = None,
) -> dict[str, Any]:
    """Build an immutable intended matrix and an explicit gate-eligible subset."""

    selected_transports = _selected_transports(transports)
    unknown = set(model_sources) - set(MODEL_KEYS)
    if unknown:
        raise ValueError(f"Unknown final model keys: {sorted(unknown)}")
    models: dict[str, dict[str, Any]] = {}
    blockers: list[dict[str, str]] = []
    blocked_routes: list[dict[str, str]] = []
    for key in MODEL_KEYS:
        source = model_sources.get(key)
        if source is None:
            models[key] = {"model_key": key, "status": "missing"}
            blockers.append({"model_key": key, "reason": "model source is missing"})
            continue
        try:
            if isinstance(source, Path):
                identity = validate_final_export(source, verifier=verifier)
                if identity.model_key != key:
                    raise RuntimeError(
                        f"Source identity is {identity.model_key}, expected {key}"
                    )
                models[key] = {"status": "passed", **identity.to_dict()}
            elif "blocked_audit" in source:
                blocked = validate_blocked_audit(
                    Path(source["blocked_audit"]), verifier=blocked_verifier
                )
                if blocked["model_key"] != key:
                    raise RuntimeError(
                        f"Blocked identity is {blocked['model_key']}, expected {key}"
                    )
                models[key] = blocked
                blocked_routes.append({"model_key": key, "reason": blocked["reason"]})
            elif "export_dir" in source:
                identity = validate_final_export(
                    Path(source["export_dir"]), verifier=verifier
                )
                if identity.model_key != key:
                    raise RuntimeError(
                        f"Source identity is {identity.model_key}, expected {key}"
                    )
                models[key] = {"status": "passed", **identity.to_dict()}
            else:
                raise ValueError("Model source requires export_dir or blocked_audit")
        except Exception as exc:
            models[key] = {
                "model_key": key,
                "status": "invalid",
                "reason": f"{exc.__class__.__name__}: {exc}",
            }
            blockers.append({"model_key": key, "reason": models[key]["reason"]})

    shape = _derived_campaign_shape(selected_transports)
    eligible_combinations = [
        dict(item)
        for item in shape["combinations"]
        if models[item["model_key"]].get("status") == "passed"
    ]
    excluded_combinations = [
        {
            **dict(item),
            "model_status": models[item["model_key"]].get("status"),
            "reason": models[item["model_key"]].get("reason"),
        }
        for item in shape["combinations"]
        if models[item["model_key"]].get("status") != "passed"
    ]
    if not eligible_combinations:
        blockers.append(
            {"scope": "campaign", "reason": "no model passed the frozen export gates"}
        )
    status = (
        "blocked"
        if blockers
        else ("ready_with_blocked_routes" if blocked_routes else "ready")
    )
    payload: dict[str, Any] = {
        "schema": "cukd_final_hil_campaign_contract_v1",
        "protocol_id": CAMPAIGN_PROTOCOL_ID,
        "status": status,
        "seed": EXPECTED_SEED,
        "models": models,
        "boards": list(BOARDS),
        "transports": list(selected_transports),
        "stages": [dict(stage) for stage in FINAL_STAGES],
        **shape,
        "eligible_combinations": eligible_combinations,
        "excluded_combinations": excluded_combinations,
        **_execution_shape(eligible_combinations),
        "blockers": blockers,
        "blocked_routes": blocked_routes,
        "execution_order_policy": (
            "Boards run sequentially; model order is reversed on the second board."
            if len(selected_transports) == 1
            else (
                "Boards run sequentially; model order is reversed on the second board, "
                "and USB/Wi-Fi first position alternates by model within each board."
            )
        ),
        "measurement_boundaries": {
            "device_preprocess": "integer standardization only",
            "device_inference": "fixed-point forward pass plus argmax",
            "device_total": "preprocess plus inference",
            "host_rtt": (
                "request write/send through complete matching response receipt; "
                "reported separately and not interpreted as pure transport latency"
            ),
            "warmup": "auditable but excluded from reported metrics",
        },
    }
    payload["contract_id"] = canonical_json_sha256(payload)
    if output_path is not None:
        atomic_write_json(output_path.resolve(), payload)
    return payload


def validate_campaign_contract(payload_or_path: Mapping[str, Any] | Path) -> dict[str, Any]:
    payload = (
        read_json(payload_or_path.resolve())
        if isinstance(payload_or_path, Path)
        else dict(payload_or_path)
    )
    recorded_id = payload.get("contract_id")
    copy = dict(payload)
    copy.pop("contract_id", None)
    if recorded_id != canonical_json_sha256(copy):
        raise RuntimeError("Campaign contract ID is invalid")
    if payload.get("schema") != "cukd_final_hil_campaign_contract_v1":
        raise RuntimeError("Unsupported campaign contract schema")
    if payload.get("protocol_id") != CAMPAIGN_PROTOCOL_ID or payload.get("seed") != 42:
        raise RuntimeError("Campaign protocol or seed is invalid")
    if tuple(payload.get("boards", [])) != BOARDS:
        raise RuntimeError("Campaign board order changed")
    try:
        selected_transports = _selected_transports(payload.get("transports", []))
    except ValueError as exc:
        raise RuntimeError("Campaign transport scope is invalid") from exc
    if payload.get("stages") != [dict(stage) for stage in FINAL_STAGES]:
        raise RuntimeError("Campaign stage order or semantics changed")
    derived = _derived_campaign_shape(selected_transports)
    for key, expected in derived.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"Campaign-derived field is invalid: {key}")
    models = payload.get("models")
    if not isinstance(models, dict) or set(models) != set(MODEL_KEYS):
        raise RuntimeError("Campaign does not retain all four intended models")
    passed_models: list[Mapping[str, Any]] = []
    expected_blockers: list[dict[str, str]] = []
    expected_blocked_routes: list[dict[str, str]] = []
    for key in MODEL_KEYS:
        entry = models[key]
        if not isinstance(entry, Mapping):
            raise RuntimeError(f"Campaign model entry is malformed: {key}")
        status = entry.get("status")
        if status == "passed":
            if model_key(str(entry.get("student")), str(entry.get("route"))) != key:
                raise RuntimeError(f"Campaign model identity differs from key: {key}")
            if entry.get("seed") != EXPECTED_SEED or entry.get("gate_status") != "passed":
                raise RuntimeError(f"Campaign model did not pass final seed-42 gates: {key}")
            for field in [
                "checkpoint_file_sha256",
                "trained_state_sha256",
                "export_id",
                "dataset_sha256",
                "split_indices_sha256",
                "scaler_sha256",
                "manifest_sha256",
                "report_sha256",
                "full_replay_sha256",
                "full_reference_sha256",
            ]:
                if not _is_sha256(entry.get(field)):
                    raise RuntimeError(f"Campaign model hash is invalid: {key}:{field}")
            if entry.get("test_rows") != EXPECTED_FULL_ROWS:
                raise RuntimeError(f"Campaign model row count is invalid: {key}")
            passed_models.append(entry)
        elif status == "blocked":
            if (
                entry.get("model_key") != key
                or entry.get("seed") != EXPECTED_SEED
                or not _is_sha256(entry.get("blocked_audit_id"))
                or not _is_sha256(entry.get("blocked_audit_sha256"))
            ):
                raise RuntimeError(f"Campaign blocked-route evidence is invalid: {key}")
            expected_blocked_routes.append(
                {"model_key": key, "reason": str(entry.get("reason"))}
            )
        else:
            expected_blockers.append(
                {"model_key": key, "reason": str(entry.get("reason"))}
            )
    for field in [
        "protocol",
        "dataset_sha256",
        "split_indices_sha256",
        "scaler_sha256",
        "full_replay_sha256",
    ]:
        if len({entry.get(field) for entry in passed_models}) > 1:
            raise RuntimeError(f"Passed campaign models disagree on {field}")
    export_ids = [entry.get("export_id") for entry in passed_models]
    if len(export_ids) != len(set(export_ids)):
        raise RuntimeError("Passed campaign routes do not have distinct export IDs")
    intended = derived["combinations"]
    expected_eligible = [
        dict(item) for item in intended if models[item["model_key"]].get("status") == "passed"
    ]
    expected_excluded = [
        {
            **dict(item),
            "model_status": models[item["model_key"]].get("status"),
            "reason": models[item["model_key"]].get("reason"),
        }
        for item in intended
        if models[item["model_key"]].get("status") != "passed"
    ]
    if not expected_eligible:
        expected_blockers.append(
            {"scope": "campaign", "reason": "no model passed the frozen export gates"}
        )
    expected_status = (
        "blocked"
        if expected_blockers
        else ("ready_with_blocked_routes" if expected_blocked_routes else "ready")
    )
    if payload.get("status") != expected_status:
        raise RuntimeError("Campaign status does not reflect model gate state")
    recorded_blockers = payload.get("blockers")
    recorded_blocked_routes = payload.get("blocked_routes")
    if not isinstance(recorded_blockers, list) or not isinstance(
        recorded_blocked_routes, list
    ):
        raise RuntimeError("Campaign blocker ledger is malformed")
    if recorded_blockers != expected_blockers:
        raise RuntimeError("Campaign blocker ledger does not match model gate state")
    if recorded_blocked_routes != expected_blocked_routes:
        raise RuntimeError("Campaign blocked-route ledger does not match gate state")
    if payload.get("eligible_combinations") != expected_eligible:
        raise RuntimeError("Campaign eligible-combination ledger is invalid")
    if payload.get("excluded_combinations") != expected_excluded:
        raise RuntimeError("Campaign excluded-combination ledger is invalid")
    for key, expected in _execution_shape(expected_eligible).items():
        if payload.get(key) != expected:
            raise RuntimeError(f"Campaign eligible-derived field is invalid: {key}")
    return payload


def _csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        return fields, list(reader)


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _model_reference_name(key: str) -> str:
    if key not in MODEL_KEYS:
        raise ValueError(key)
    return f"balanced_timing_reference_{key}.csv"


def _validate_dense_rows(rows: Sequence[Mapping[str, str]], expected: int, label: str) -> None:
    observed = [int(row["row_id"]) for row in rows]
    if len(rows) != expected or observed != list(range(expected)):
        raise RuntimeError(f"{label} rows are not the dense ordered 0..{expected - 1} sequence")


def validate_cohort_selection(rows: Sequence[Mapping[str, Any]]) -> None:
    """Apply the anti-prefix, balance, and unique-group invariants."""

    expected_rows = GROUPS_PER_CLASS * CLASS_COUNT
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Balanced timing cohort must contain exactly {expected_rows:,} rows"
        )
    timing_ids = [int(row["timing_row_id"]) for row in rows]
    originals = [int(row["original_full_test_row_id"]) for row in rows]
    labels = [int(row["true_label"]) for row in rows]
    groups = [str(row["feature_group_sha256"]) for row in rows]
    sources = [int(row["source_row_index"]) for row in rows]
    class_ranks = [int(row["class_rank"]) for row in rows]
    class_names = [str(row["class_name"]) for row in rows]
    selection_ranks = [str(row["selection_rank_sha256"]) for row in rows]
    if timing_ids != list(range(expected_rows)):
        raise RuntimeError("Timing cohort IDs are not dense and ordered")
    if originals == list(range(expected_rows)):
        raise RuntimeError("Timing cohort is the first-1,000 full-replay prefix")
    if len(set(originals)) != expected_rows:
        raise RuntimeError("Timing cohort repeats a full-test row")
    if min(originals) < 0 or max(originals) >= EXPECTED_FULL_ROWS:
        raise RuntimeError("Timing cohort contains an invalid full-test row ID")
    if len(set(sources)) != expected_rows or min(sources) < 0:
        raise RuntimeError("Timing cohort source-row identities are invalid or repeated")
    if len(set(groups)) != expected_rows or not all(
        _is_sha256(value) for value in groups
    ):
        raise RuntimeError("Timing cohort feature-group identities are invalid or repeated")
    counts = {label: labels.count(label) for label in range(CLASS_COUNT)}
    if counts != {label: GROUPS_PER_CLASS for label in range(CLASS_COUNT)}:
        raise RuntimeError(f"Timing cohort is not class balanced: {counts}")
    expected_interleave = [index % CLASS_COUNT for index in range(expected_rows)]
    if labels != expected_interleave:
        raise RuntimeError("Timing cohort class interleave order changed")
    if class_ranks != [index // CLASS_COUNT for index in range(expected_rows)]:
        raise RuntimeError("Timing cohort within-class rank order changed")
    if class_names != [CLASS_NAMES[label] for label in labels]:
        raise RuntimeError("Timing cohort class names disagree with labels")
    if not all(_is_sha256(value) for value in selection_ranks):
        raise RuntimeError("Timing cohort selection-rank identities are invalid")


def _resolve_recorded_path(
    override: Path | None,
    recorded: Any,
    label: str,
) -> Path:
    if override is not None:
        return override.resolve()
    if not isinstance(recorded, str) or not recorded:
        raise RuntimeError(f"Balanced timing cohort does not record its {label} source")
    return Path(recorded).resolve()


def _validate_balanced_cohort_sources(
    manifest: Mapping[str, Any],
    replay_rows: Sequence[Mapping[str, str]],
    references: Mapping[str, Sequence[Mapping[str, str]]],
    *,
    identities: Mapping[str, FinalExportIdentity] | None,
    dataset_csv: Path | None,
    split_root: Path | None,
    export_dirs: Mapping[str, Path] | None,
) -> None:
    """Reconstruct the deterministic cohort from its immutable source evidence."""

    import numpy as np

    from deployment.hardware_hil.host import generate_fgds_balanced_timing_cohort as source

    dataset_entry = manifest.get("dataset")
    split_entry = manifest.get("split")
    model_entries = manifest.get("models")
    if not isinstance(dataset_entry, Mapping) or not isinstance(split_entry, Mapping):
        raise RuntimeError("Balanced timing cohort source ledger is malformed")
    if not isinstance(model_entries, Mapping) or not model_entries:
        raise RuntimeError("Balanced timing cohort model source ledger is malformed")

    dataset = _resolve_recorded_path(
        dataset_csv, dataset_entry.get("path_recorded"), "dataset"
    )
    split_path = _resolve_recorded_path(
        split_root, split_entry.get("path_recorded"), "split"
    )
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    if not split_path.is_dir():
        raise NotADirectoryError(split_path)

    if identities is None:
        if export_dirs is None:
            inferred: dict[str, Path] = {}
            for key, entry in model_entries.items():
                if not isinstance(entry, Mapping):
                    raise RuntimeError(f"Balanced timing cohort model entry is malformed: {key}")
                inferred[str(key)] = _resolve_recorded_path(
                    None, entry.get("path_recorded"), f"{key} final export"
                )
            export_dirs = inferred
        if set(export_dirs) != set(model_entries):
            raise RuntimeError("Balanced timing cohort source export set is incomplete")
        identities = {
            key: validate_final_export(Path(path)) for key, path in export_dirs.items()
        }
    elif set(identities) != set(model_entries):
        raise RuntimeError("Balanced timing cohort source identity set is incomplete")

    for key, identity in identities.items():
        if identity.model_key != key:
            raise RuntimeError(f"Cohort source {key} identifies itself as {identity.model_key}")
        if identity.protocol != SOURCE_PROTOCOL_ID:
            raise RuntimeError(f"Cohort source {key} has the wrong training protocol")
        entry = model_entries.get(key)
        if not isinstance(entry, Mapping):
            raise RuntimeError(f"Balanced timing cohort model entry is malformed: {key}")
        for field in [
            "export_id",
            "trained_state_sha256",
            "full_replay_sha256",
            "full_reference_sha256",
        ]:
            if entry.get(field) != getattr(identity, field):
                raise RuntimeError(f"Cohort source binding differs for {key}:{field}")
    for field in [
        "protocol",
        "dataset_sha256",
        "split_indices_sha256",
        "scaler_sha256",
        "full_replay_sha256",
    ]:
        if len({getattr(identity, field) for identity in identities.values()}) != 1:
            raise RuntimeError(f"Cohort source exports disagree on {field}")

    features, labels, target_column = source.load_dataset(dataset)
    split, execution, preprocessing, split_manifest = source.load_split(
        split_path,
        len(labels),
        EXPECTED_SPLIT_SIZES,
    )
    common_identity = next(iter(identities.values()))
    if sha256_file(dataset) != common_identity.dataset_sha256:
        raise RuntimeError("Cohort dataset differs from the final export lineage")
    if execution.get("split_indices_sha256") != common_identity.split_indices_sha256:
        raise RuntimeError("Cohort split differs from the final export lineage")
    if execution.get("scaler_sha256") != common_identity.scaler_sha256:
        raise RuntimeError("Cohort scaler differs from the final export lineage")
    if dataset_entry.get("sha256") != sha256_file(dataset):
        raise RuntimeError("Cohort dataset ledger hash is invalid")
    if dataset_entry.get("target_column") != target_column:
        raise RuntimeError("Cohort dataset target-column ledger is invalid")
    if split_entry.get("manifest_sha256") != sha256_file(
        split_path / "artifact_manifest.json"
    ):
        raise RuntimeError("Cohort split-manifest ledger hash is invalid")
    if split_entry.get("manifest_status") != split_manifest.get("status"):
        raise RuntimeError("Cohort split-manifest status is invalid")
    if split_entry.get("split_indices_sha256") != execution.get(
        "split_indices_sha256"
    ):
        raise RuntimeError("Cohort split ledger is invalid")
    if split_entry.get("scaler_sha256") != execution.get("scaler_sha256"):
        raise RuntimeError("Cohort scaler ledger is invalid")
    if split_entry.get("preprocessing_contract_sha256") != canonical_json_sha256(
        preprocessing
    ):
        raise RuntimeError("Cohort preprocessing-contract ledger is invalid")

    groups = source.build_groups(features, labels, split)
    if any(record.partition_mask & (record.partition_mask - 1) for record in groups.values()):
        raise RuntimeError("Cohort source split contains cross-partition feature groups")
    selected = source.select_groups(groups, GROUPS_PER_CLASS, EXPECTED_SEED)
    test_indices = [int(value) for value in split["test"]]
    full_id_by_source = {
        source_row_index: full_row_id
        for full_row_id, source_row_index in enumerate(test_indices)
    }
    for row in selected:
        row["original_full_test_row_id"] = full_id_by_source[int(row["source_row_index"])]
    validate_cohort_selection(selected)
    if selected != manifest.get("rows"):
        raise RuntimeError(
            "Balanced timing cohort selection does not reconstruct from dataset and split"
        )

    expected_replay_fields = [
        "row_id",
        "source_row_index",
        *[f"f{index}" for index in range(17)],
    ]
    expected_reference_fields = [
        "row_id",
        "source_row_index",
        "true_label",
        "fixed_pred",
        "fp32_pred",
        *[f"fixed_logit_{index}" for index in range(CLASS_COUNT)],
    ]
    selected_sources = [int(row["source_row_index"]) for row in selected]
    expected_replay_by_model: dict[str, dict[int, Mapping[str, str]]] = {}
    expected_reference_by_model: dict[str, dict[int, Mapping[str, str]]] = {}
    for key, identity in identities.items():
        root = Path(identity.root).resolve()
        entry = model_entries[key]
        if not isinstance(entry, Mapping):
            raise RuntimeError(f"Balanced timing cohort model entry is malformed: {key}")
        recorded_export = entry.get("path_recorded")
        if export_dirs is None and isinstance(recorded_export, str):
            if Path(recorded_export).resolve() != root:
                raise RuntimeError(f"Cohort recorded export path differs for {key}")
        replay_fields, full_replay = _csv_rows(root / "hil_replay_vectors.csv")
        reference_fields, full_reference = _csv_rows(
            root / "hil_reference_predictions.csv"
        )
        if replay_fields != expected_replay_fields:
            raise RuntimeError(f"Final export replay schema changed for {key}")
        if reference_fields != expected_reference_fields:
            raise RuntimeError(f"Final export reference schema changed for {key}")
        _validate_dense_rows(full_replay, EXPECTED_FULL_ROWS, f"{key} full replay")
        _validate_dense_rows(full_reference, EXPECTED_FULL_ROWS, f"{key} full reference")
        replay_sources = [int(row["source_row_index"]) for row in full_replay]
        reference_sources = [int(row["source_row_index"]) for row in full_reference]
        if replay_sources != test_indices or reference_sources != test_indices:
            raise RuntimeError(f"Final export source-row order differs from split for {key}")
        observed_labels = np.asarray(
            [int(row["true_label"]) for row in full_reference], dtype=np.int64
        )
        if not np.array_equal(observed_labels, labels[split["test"]]):
            raise RuntimeError(f"Final export labels differ from dataset for {key}")

        metadata = read_json(root / "preprocess_int_metadata.json")
        raw_q_frac = metadata.get("raw_q_frac")
        if (
            metadata.get("input_dim") != 17
            or not isinstance(raw_q_frac, int)
            or raw_q_frac < 0
            or raw_q_frac > 30
        ):
            raise RuntimeError(f"Final export raw-input contract is invalid for {key}")
        expected_raw = np.rint(
            np.asarray(features[split["test"]], dtype=np.float64)
            * float(1 << raw_q_frac)
        )
        int32 = np.iinfo(np.int32)
        if np.any(expected_raw < int32.min) or np.any(expected_raw > int32.max):
            raise RuntimeError("Cohort source dataset saturates the raw fixed-point contract")
        observed_raw = np.column_stack(
            [
                np.asarray([int(row[f"f{index}"]) for row in full_replay], dtype=np.int64)
                for index in range(17)
            ]
        )
        if not np.array_equal(observed_raw, expected_raw.astype(np.int64)):
            raise RuntimeError(f"Final export replay differs from dataset for {key}")
        expected_replay_by_model[key] = {
            int(row["source_row_index"]): row for row in full_replay
        }
        expected_reference_by_model[key] = {
            int(row["source_row_index"]): row for row in full_reference
        }

    first_key = next(iter(identities))
    replay_source = expected_replay_by_model[first_key]
    for timing_id, (selection_row, observed) in enumerate(zip(selected, replay_rows)):
        source_row = int(selection_row["source_row_index"])
        full = replay_source[source_row]
        expected_ids = {
            "row_id": timing_id,
            "timing_row_id": timing_id,
            "original_full_test_row_id": int(
                selection_row["original_full_test_row_id"]
            ),
            "source_row_index": source_row,
        }
        if any(int(observed[field]) != value for field, value in expected_ids.items()):
            raise RuntimeError("Balanced timing replay identity does not reconstruct")
        if any(int(observed[f"f{index}"]) != int(full[f"f{index}"]) for index in range(17)):
            raise RuntimeError("Balanced timing replay features do not reconstruct")
        for key, rows_by_source in expected_replay_by_model.items():
            if any(
                int(rows_by_source[source_row][f"f{index}"])
                != int(full[f"f{index}"])
                for index in range(17)
            ):
                raise RuntimeError(f"Final exports disagree on cohort input for {key}")

    for key, observed_rows in references.items():
        source_reference = expected_reference_by_model[key]
        for timing_id, (selection_row, observed) in enumerate(
            zip(selected, observed_rows)
        ):
            source_row = int(selection_row["source_row_index"])
            full = source_reference[source_row]
            identity_fields = {
                "row_id": timing_id,
                "timing_row_id": timing_id,
                "original_full_test_row_id": int(
                    selection_row["original_full_test_row_id"]
                ),
                "source_row_index": source_row,
            }
            if any(
                int(observed[field]) != value for field, value in identity_fields.items()
            ):
                raise RuntimeError(f"Balanced timing reference identity differs for {key}")
            for field in expected_reference_fields[2:]:
                if int(observed[field]) != int(full[field]):
                    raise RuntimeError(
                        f"Balanced timing reference does not reconstruct for {key}:{field}"
                    )


def _validate_balanced_cohort_export_binding(
    manifest: Mapping[str, Any],
    replay_rows: Sequence[Mapping[str, str]],
    references: Mapping[str, Sequence[Mapping[str, str]]],
    identities: Mapping[str, FinalExportIdentity],
) -> None:
    """Verify cohort rows solely against locally supplied full-export CSVs."""

    selection = manifest.get("rows")
    if (
        not isinstance(selection, list)
        or not identities
        or not set(identities) <= set(references)
    ):
        raise RuntimeError("Portable cohort/export model binding is incomplete")
    expected_replay_fields = [
        "row_id",
        "source_row_index",
        *[f"f{index}" for index in range(17)],
    ]
    expected_reference_fields = [
        "row_id",
        "source_row_index",
        "true_label",
        "fixed_pred",
        "fp32_pred",
        *[f"fixed_logit_{index}" for index in range(CLASS_COUNT)],
    ]
    full_replays: dict[str, list[dict[str, str]]] = {}
    full_references: dict[str, list[dict[str, str]]] = {}
    common_sources: list[int] | None = None
    for key, identity in identities.items():
        export_root = Path(identity.root).resolve()
        replay_path = export_root / "hil_replay_vectors.csv"
        reference_path = export_root / "hil_reference_predictions.csv"
        if (
            sha256_file(replay_path) != identity.full_replay_sha256
            or sha256_file(reference_path) != identity.full_reference_sha256
        ):
            raise RuntimeError(f"Portable cohort source hash differs for {key}")
        replay_fields, full_replay = _csv_rows(replay_path)
        reference_fields, full_reference = _csv_rows(reference_path)
        if replay_fields != expected_replay_fields:
            raise RuntimeError(f"Portable full replay schema changed for {key}")
        if reference_fields != expected_reference_fields:
            raise RuntimeError(f"Portable full reference schema changed for {key}")
        _validate_dense_rows(full_replay, EXPECTED_FULL_ROWS, f"{key} full replay")
        _validate_dense_rows(
            full_reference, EXPECTED_FULL_ROWS, f"{key} full reference"
        )
        replay_sources = [int(row["source_row_index"]) for row in full_replay]
        reference_sources = [int(row["source_row_index"]) for row in full_reference]
        if (
            replay_sources != reference_sources
            or len(set(replay_sources)) != EXPECTED_FULL_ROWS
        ):
            raise RuntimeError(f"Portable full-export source identity differs for {key}")
        if common_sources is None:
            common_sources = replay_sources
        elif replay_sources != common_sources:
            raise RuntimeError("Portable final exports use different full-test row orders")
        full_replays[key] = full_replay
        full_references[key] = full_reference

    first_key = next(iter(identities))
    for timing_id, (selected, observed) in enumerate(zip(selection, replay_rows)):
        full_id = int(selected["original_full_test_row_id"])
        source_id = int(selected["source_row_index"])
        if full_id < 0 or full_id >= EXPECTED_FULL_ROWS:
            raise RuntimeError("Portable cohort full-test row identity is out of range")
        canonical = full_replays[first_key][full_id]
        if int(canonical["source_row_index"]) != source_id:
            raise RuntimeError("Portable cohort source/full-row identity does not bind")
        expected_ids = {
            "row_id": timing_id,
            "timing_row_id": timing_id,
            "original_full_test_row_id": full_id,
            "source_row_index": source_id,
        }
        if any(int(observed[field]) != value for field, value in expected_ids.items()):
            raise RuntimeError("Portable cohort replay identity does not bind")
        for index in range(17):
            field = f"f{index}"
            if int(observed[field]) != int(canonical[field]):
                raise RuntimeError("Portable cohort replay features do not bind")
        for key, full_replay in full_replays.items():
            candidate = full_replay[full_id]
            if int(candidate["source_row_index"]) != source_id or any(
                int(candidate[f"f{index}"]) != int(canonical[f"f{index}"])
                for index in range(17)
            ):
                raise RuntimeError(f"Portable full exports disagree on input for {key}")

    for key in identities:
        observed_rows = references[key]
        full_reference = full_references[key]
        for timing_id, (selected, observed) in enumerate(zip(selection, observed_rows)):
            full_id = int(selected["original_full_test_row_id"])
            source_id = int(selected["source_row_index"])
            canonical = full_reference[full_id]
            expected_ids = {
                "row_id": timing_id,
                "timing_row_id": timing_id,
                "original_full_test_row_id": full_id,
                "source_row_index": source_id,
            }
            if any(int(observed[field]) != value for field, value in expected_ids.items()):
                raise RuntimeError(f"Portable cohort reference identity differs for {key}")
            if int(canonical["source_row_index"]) != source_id:
                raise RuntimeError(f"Portable full reference source differs for {key}")
            for field in expected_reference_fields[2:]:
                if int(observed[field]) != int(canonical[field]):
                    raise RuntimeError(
                        f"Portable cohort reference does not bind for {key}:{field}"
                    )


def generate_balanced_cohort(
    export_dirs: Mapping[str, Path],
    *,
    dataset_csv: Path,
    split_root: Path,
    output_dir: Path,
    verifier: Verifier | None = None,
) -> Path:
    """Select one cohort once and bind every gate-eligible model reference to it."""

    if not export_dirs or not set(export_dirs) <= set(MODEL_KEYS):
        raise ValueError("Balanced cohort generation requires a nonempty final-model subset")
    identities = {
        key: validate_final_export(path, verifier=verifier)
        for key, path in export_dirs.items()
    }
    for key, identity in identities.items():
        if identity.model_key != key:
            raise RuntimeError(f"Export {key} identifies itself as {identity.model_key}")
        if identity.protocol != SOURCE_PROTOCOL_ID:
            raise RuntimeError(f"Export {key} has the wrong training protocol")
    for field in [
        "protocol",
        "dataset_sha256",
        "split_indices_sha256",
        "scaler_sha256",
        "full_replay_sha256",
    ]:
        values = {getattr(identity, field) for identity in identities.values()}
        if len(values) != 1:
            raise RuntimeError(f"Final exports disagree on {field}")

    # Reuse the audited feature-group definition and split verifier.  No model
    # or fixed-point numeric code is duplicated here.
    from deployment.hardware_hil.host.generate_fgds_balanced_timing_cohort import (
        build_groups,
        load_dataset,
        load_split,
        select_groups,
    )

    dataset = dataset_csv.resolve()
    split_path = split_root.resolve()
    destination = output_dir.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite timing cohort: {destination}")
    features, labels, target_column = load_dataset(dataset)
    split, execution, preprocessing, split_manifest = load_split(
        split_path,
        len(labels),
        EXPECTED_SPLIT_SIZES,
    )
    if sha256_file(dataset) != next(iter(identities.values())).dataset_sha256:
        raise RuntimeError("Dataset hash differs from final export identity")
    if execution.get("split_indices_sha256") != next(
        iter(identities.values())
    ).split_indices_sha256:
        raise RuntimeError("Split hash differs from final export identity")
    if execution.get("scaler_sha256") != next(iter(identities.values())).scaler_sha256:
        raise RuntimeError("Scaler hash differs from final export identity")

    groups = build_groups(features, labels, split)
    if any(record.partition_mask & (record.partition_mask - 1) for record in groups.values()):
        raise RuntimeError("Source split contains cross-partition feature groups")
    selected = select_groups(groups, GROUPS_PER_CLASS, EXPECTED_SEED)
    test_indices = [int(value) for value in split["test"]]
    full_id_by_source = {
        source_row_index: full_row_id
        for full_row_id, source_row_index in enumerate(test_indices)
    }
    for selection in selected:
        selection["original_full_test_row_id"] = full_id_by_source[
            int(selection["source_row_index"])
        ]
    validate_cohort_selection(selected)

    replay_columns = ["row_id", "source_row_index", *[f"f{i}" for i in range(17)]]
    reference_columns = [
        "row_id",
        "source_row_index",
        "true_label",
        "fixed_pred",
        "fp32_pred",
        *[f"fixed_logit_{i}" for i in range(CLASS_COUNT)],
    ]
    full_exports: dict[str, tuple[list[dict[str, str]], list[dict[str, str]]]] = {}
    canonical_replay_by_source: dict[int, dict[str, str]] | None = None
    selected_sources = [int(row["source_row_index"]) for row in selected]
    for key, identity in identities.items():
        root = Path(identity.root)
        replay_fields, replay_rows = _csv_rows(root / "hil_replay_vectors.csv")
        reference_fields, reference_rows = _csv_rows(
            root / "hil_reference_predictions.csv"
        )
        if replay_fields != replay_columns or reference_fields != reference_columns:
            raise RuntimeError(f"Final export {key} replay/reference schema changed")
        _validate_dense_rows(replay_rows, EXPECTED_FULL_ROWS, f"{key} replay")
        _validate_dense_rows(reference_rows, EXPECTED_FULL_ROWS, f"{key} reference")
        replay_sources = [int(row["source_row_index"]) for row in replay_rows]
        reference_sources = [int(row["source_row_index"]) for row in reference_rows]
        if replay_sources != test_indices or reference_sources != test_indices:
            raise RuntimeError(f"Final export {key} source-row order differs from split")
        by_source = {int(row["source_row_index"]): row for row in replay_rows}
        if canonical_replay_by_source is None:
            canonical_replay_by_source = by_source
        elif any(
            [by_source[source][f"f{i}"] for i in range(17)]
            != [canonical_replay_by_source[source][f"f{i}"] for i in range(17)]
            for source in selected_sources
        ):
            raise RuntimeError("Final exports encode different balanced replay features")
        full_exports[key] = (replay_rows, reference_rows)

    if canonical_replay_by_source is None:
        raise RuntimeError("No canonical replay was loaded")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    manifest_path = temporary / "final_timing_cohort_manifest.json"
    try:
        common_rows: list[dict[str, Any]] = []
        selection_rows: list[dict[str, Any]] = []
        model_entries: dict[str, dict[str, Any]] = {}
        for timing_id, selection in enumerate(selected):
            source = int(selection["source_row_index"])
            original = int(selection["original_full_test_row_id"])
            source_replay = canonical_replay_by_source[source]
            common_rows.append(
                {
                    "row_id": timing_id,
                    "timing_row_id": timing_id,
                    "original_full_test_row_id": original,
                    "source_row_index": source,
                    **{f"f{i}": source_replay[f"f{i}"] for i in range(17)},
                }
            )
            selection_rows.append(dict(selection))
        validate_cohort_selection(selection_rows)
        replay_name = "balanced_timing_replay_vectors.csv"
        replay_path = temporary / replay_name
        _write_csv(
            replay_path,
            [
                "row_id",
                "timing_row_id",
                "original_full_test_row_id",
                "source_row_index",
                *[f"f{i}" for i in range(17)],
            ],
            common_rows,
        )
        for key, identity in identities.items():
            _, reference_rows = full_exports[key]
            reference_by_source = {
                int(row["source_row_index"]): row for row in reference_rows
            }
            output_rows = []
            for selection in selection_rows:
                source = int(selection["source_row_index"])
                original = int(selection["original_full_test_row_id"])
                reference = reference_by_source[source]
                output_rows.append(
                    {
                        "row_id": int(selection["timing_row_id"]),
                        "timing_row_id": int(selection["timing_row_id"]),
                        "original_full_test_row_id": original,
                        "source_row_index": source,
                        **{
                            name: reference[name]
                            for name in reference_columns
                            if name not in {"row_id", "source_row_index"}
                        },
                    }
                )
            reference_name = _model_reference_name(key)
            reference_path = temporary / reference_name
            _write_csv(
                reference_path,
                [
                    "row_id",
                    "timing_row_id",
                    "original_full_test_row_id",
                    "source_row_index",
                    *reference_columns[2:],
                ],
                output_rows,
            )
            model_entries[key] = {
                "path_recorded": str(Path(identity.root).resolve()),
                "export_id": identity.export_id,
                "trained_state_sha256": identity.trained_state_sha256,
                "full_replay_sha256": identity.full_replay_sha256,
                "full_reference_sha256": identity.full_reference_sha256,
                "reference_file": reference_name,
                "reference_sha256": sha256_file(reference_path),
            }
        files = [
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(temporary.iterdir())
            if path.is_file() and path != manifest_path
        ]
        manifest: dict[str, Any] = {
            "schema": "cukd_final_hil_balanced_cohort_v1",
            "protocol_id": COHORT_PROTOCOL_ID,
            "status": "passed",
            "source_protocol_id": next(iter(identities.values())).protocol,
            "selection": {
                "seed": EXPECTED_SEED,
                "groups_per_class": GROUPS_PER_CLASS,
                "class_count": CLASS_COUNT,
                "algorithm": "sha256_seeded_canonical_f32_rank_v1",
                "not_first_1000_full_replay_prefix": True,
            },
            "dataset": {
                "path_recorded": str(dataset),
                "sha256": sha256_file(dataset),
                "target_column": target_column,
            },
            "split": {
                "path_recorded": str(split_path),
                "manifest_sha256": sha256_file(split_path / "artifact_manifest.json"),
                "manifest_status": split_manifest["status"],
                "split_indices_sha256": execution["split_indices_sha256"],
                "scaler_sha256": execution["scaler_sha256"],
                "preprocessing_contract_sha256": canonical_json_sha256(preprocessing),
            },
            "replay_file": replay_name,
            "replay_sha256": sha256_file(replay_path),
            "models": model_entries,
            "rows": selection_rows,
            "file_count_excluding_manifest": len(files),
            "files": files,
        }
        manifest["manifest_payload_sha256"] = canonical_json_sha256(manifest)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        validate_balanced_cohort(temporary, identities=identities)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination / "final_timing_cohort_manifest.json"


def validate_balanced_cohort(
    root: Path,
    *,
    identities: Mapping[str, FinalExportIdentity] | None = None,
    dataset_csv: Path | None = None,
    split_root: Path | None = None,
    export_dirs: Mapping[str, Path] | None = None,
    reconstruct_sources: bool = True,
    allow_identity_subset: bool = False,
) -> dict[str, Any]:
    source_root = Path(root)
    if source_root.is_symlink():
        raise RuntimeError("Balanced timing cohort root cannot be a symlink")
    resolved = source_root.resolve()
    manifest_path = resolved / "final_timing_cohort_manifest.json"
    if manifest_path.is_symlink():
        raise RuntimeError("Balanced timing cohort manifest cannot be a symlink")
    manifest = read_json(manifest_path)
    if "manifest_payload_sha256" not in manifest:
        raise RuntimeError("Cohort manifest canonical payload hash is missing")
    _verify_optional_payload_hash(manifest, "manifest_payload_sha256", "Cohort manifest")
    if (
        manifest.get("schema") != "cukd_final_hil_balanced_cohort_v1"
        or manifest.get("protocol_id") != COHORT_PROTOCOL_ID
        or manifest.get("status") != "passed"
    ):
        raise RuntimeError("Balanced timing cohort contract is invalid")
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError("Balanced timing cohort row ledger is missing")
    validate_cohort_selection(rows)
    selection = manifest.get("selection", {})
    if selection != {
        "seed": EXPECTED_SEED,
        "groups_per_class": GROUPS_PER_CLASS,
        "class_count": CLASS_COUNT,
        "algorithm": "sha256_seeded_canonical_f32_rank_v1",
        "not_first_1000_full_replay_prefix": True,
    }:
        raise RuntimeError("Balanced timing cohort does not carry the anti-prefix gate")
    files = _manifest_files(manifest)
    listed: dict[str, str] = {}
    for item in files:
        relative = item.get("path")
        if (
            not isinstance(relative, str)
            or "/" in relative
            or "\\" in relative
            or Path(relative).name != relative
            or relative in listed
        ):
            raise RuntimeError("Cohort inventory path is unsafe")
        path = resolved / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != item.get("size_bytes")
            or sha256_file(path) != item.get("sha256")
        ):
            raise RuntimeError(f"Cohort inventory mismatch: {relative}")
        listed[relative] = str(item["sha256"])
    if manifest.get("file_count_excluding_manifest") != len(files):
        raise RuntimeError("Cohort inventory count is invalid")
    members = list(resolved.rglob("*"))
    if any(path.is_symlink() for path in members):
        raise RuntimeError("Cohort contains a symlink")
    actual = {
        path.relative_to(resolved).as_posix()
        for path in members
        if path.is_file() and path.resolve() != manifest_path.resolve()
    }
    if actual != set(listed):
        raise RuntimeError("Cohort contains extra or unlisted files")
    replay_name = manifest.get("replay_file")
    if replay_name not in listed or listed[replay_name] != manifest.get("replay_sha256"):
        raise RuntimeError("Cohort replay file is not hash-bound")
    replay_fields, replay_rows = _csv_rows(resolved / str(replay_name))
    expected_replay_fields = [
        "row_id",
        "timing_row_id",
        "original_full_test_row_id",
        "source_row_index",
        *[f"f{i}" for i in range(17)],
    ]
    if replay_fields != expected_replay_fields:
        raise RuntimeError("Balanced replay CSV schema is invalid")
    expected_cohort_rows = GROUPS_PER_CLASS * CLASS_COUNT
    _validate_dense_rows(replay_rows, expected_cohort_rows, "Balanced replay")
    if [int(row["timing_row_id"]) for row in replay_rows] != list(
        range(expected_cohort_rows)
    ):
        raise RuntimeError("Balanced replay timing IDs are invalid")
    for field in ["original_full_test_row_id", "source_row_index"]:
        if [int(row[field]) for row in replay_rows] != [int(row[field]) for row in rows]:
            raise RuntimeError(f"Balanced replay and cohort {field} identities disagree")
    for row in replay_rows:
        values = [int(row[f"f{index}"]) for index in range(17)]
        if any(value < -(2**31) or value > 2**31 - 1 for value in values):
            raise RuntimeError("Balanced replay feature is outside signed int32")
    if [int(row["original_full_test_row_id"]) for row in replay_rows] == list(
        range(expected_cohort_rows)
    ):
        raise RuntimeError("Balanced replay is the forbidden first-1,000 prefix")
    model_entries = manifest.get("models")
    if (
        not isinstance(model_entries, dict)
        or not model_entries
        or not set(model_entries) <= set(MODEL_KEYS)
    ):
        raise RuntimeError("Cohort does not bind a valid nonempty final-model subset")
    if allow_identity_subset and reconstruct_sources:
        raise ValueError("Identity-subset validation requires portable source binding")
    if identities is not None:
        supplied_keys = set(identities)
        cohort_keys = set(model_entries)
        if allow_identity_subset:
            if not supplied_keys or not supplied_keys <= cohort_keys:
                raise RuntimeError(
                    "Cohort does not contain every supplied final export"
                )
        elif cohort_keys != supplied_keys:
            raise RuntimeError("Cohort model set differs from supplied final exports")
    reference_rows_by_model: dict[str, list[dict[str, str]]] = {}
    for key in MODEL_KEYS:
        if key not in model_entries:
            continue
        entry = model_entries[key]
        reference_name = entry.get("reference_file")
        if (
            not isinstance(reference_name, str)
            or reference_name != _model_reference_name(key)
            or listed.get(reference_name) != entry.get("reference_sha256")
        ):
            raise RuntimeError(f"Cohort reference is not hash-bound for {key}")
        reference_fields, reference_rows = _csv_rows(resolved / reference_name)
        expected_reference_fields = [
            "row_id",
            "timing_row_id",
            "original_full_test_row_id",
            "source_row_index",
            "true_label",
            "fixed_pred",
            "fp32_pred",
            *[f"fixed_logit_{index}" for index in range(CLASS_COUNT)],
        ]
        if reference_fields != expected_reference_fields:
            raise RuntimeError(f"Cohort reference schema is invalid for {key}")
        _validate_dense_rows(
            reference_rows, expected_cohort_rows, f"{key} cohort reference"
        )
        for field in ["timing_row_id", "original_full_test_row_id", "source_row_index"]:
            if [int(row[field]) for row in reference_rows] != [
                int(row[field]) for row in rows
            ]:
                raise RuntimeError(f"Cohort reference {field} disagrees for {key}")
        labels = [int(row["true_label"]) for row in reference_rows]
        if labels != [int(row["true_label"]) for row in rows]:
            raise RuntimeError(f"Cohort reference labels disagree for {key}")
        for row in reference_rows:
            if any(int(row[field]) not in range(CLASS_COUNT) for field in [
                "true_label", "fixed_pred", "fp32_pred"
            ]):
                raise RuntimeError(f"Cohort reference class is invalid for {key}")
            for index in range(CLASS_COUNT):
                int(row[f"fixed_logit_{index}"])
        if identities is not None and key in identities:
            identity = identities[key]
            for field in [
                "export_id",
                "trained_state_sha256",
                "full_replay_sha256",
                "full_reference_sha256",
            ]:
                if entry.get(field) != getattr(identity, field):
                    raise RuntimeError(f"Cohort/export binding differs for {key}:{field}")
        elif identities is not None and not allow_identity_subset:
            raise RuntimeError(f"No final export identity supplied for {key}")
        reference_rows_by_model[key] = reference_rows
    if identities is not None:
        common_protocols = {identity.protocol for identity in identities.values()}
        common_datasets = {identity.dataset_sha256 for identity in identities.values()}
        common_splits = {identity.split_indices_sha256 for identity in identities.values()}
        common_scalers = {identity.scaler_sha256 for identity in identities.values()}
        if any(
            len(values) != 1
            for values in [common_protocols, common_datasets, common_splits, common_scalers]
        ):
            raise RuntimeError("Portable final exports disagree on preprocessing lineage")
        if manifest.get("source_protocol_id") != next(iter(common_protocols)):
            raise RuntimeError("Cohort protocol differs from final exports")
        if manifest.get("dataset", {}).get("sha256") != next(iter(common_datasets)):
            raise RuntimeError("Cohort dataset hash differs from final exports")
        if manifest.get("split", {}).get("split_indices_sha256") != next(
            iter(common_splits)
        ):
            raise RuntimeError("Cohort split hash differs from final exports")
        if manifest.get("split", {}).get("scaler_sha256") != next(iter(common_scalers)):
            raise RuntimeError("Cohort scaler hash differs from final exports")
    if reconstruct_sources:
        _validate_balanced_cohort_sources(
            manifest,
            replay_rows,
            reference_rows_by_model,
            identities=identities,
            dataset_csv=dataset_csv,
            split_root=split_root,
            export_dirs=export_dirs,
        )
    else:
        if identities is None:
            raise RuntimeError(
                "Portable cohort validation requires archive-local export identities"
            )
        _validate_balanced_cohort_export_binding(
            manifest,
            replay_rows,
            reference_rows_by_model,
            identities,
        )
    return manifest


def stage_contract(name: str) -> dict[str, Any]:
    for stage in FINAL_STAGES:
        if stage["name"] == name:
            return dict(stage)
    raise ValueError(f"Unknown final HIL stage: {name}")
