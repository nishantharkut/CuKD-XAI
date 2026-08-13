"""Final stage input resolution, transport execution, and exact verification."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import ipaddress
import json
import math
import os
import platform
import secrets
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from deployment.hardware_hil.host.hil_common import (
    decode_response_line,
    encode_request_line,
)
from deployment.wireless_hil.host.udp_session import (
    IGNORED_DATAGRAM_CATEGORIES,
    StrictUdpSession,
    parse_abort_response,
    parse_begin_response,
    parse_end_response,
)
from deployment.wireless_hil.host.wireless_common import (
    ABORT_TRANSACTION_ID,
    BEGIN_TRANSACTION_ID,
    DEFAULT_DEVICE_UDP_PORT,
    DEFAULT_HOST_UDP_PORT,
    END_TRANSACTION_ID,
    IDENTITY_TRANSACTION_ID,
    MAX_ATTEMPTS,
    decode_wifi_config_response,
    encode_wifi_config_line,
)

from .bundles import RUNTIME_IDENTITY_PREFIX, verify_final_bundle
from .contracts import (
    FinalExportIdentity,
    Verifier,
    _csv_rows,
    _is_sha256,
    _validate_dense_rows,
    atomic_write_json,
    canonical_json_sha256,
    read_json,
    sha256_file,
    stage_contract,
    validate_balanced_cohort,
    validate_final_export,
)


ATTEMPT_SCHEMA = "cukd_final_hil_stage_attempt_v2"
CONNECTION_SCHEMA = "cukd_final_hil_wifi_connection_v2"
FINAL_WIFI_PROTOCOL = "cukd_fgds_wifi_udp_session_v2"
FINAL_RESPONSE_FIELDS = [
    "row_id",
    "status",
    "predicted_class",
    *[f"fixed_logit_{index}" for index in range(5)],
    "preprocess_us",
    "inference_us",
    "total_us",
    "host_observed_rtt_us",
    "transaction_elapsed_us",
    "attempts",
    "response_timeout_count",
    "ignored_datagram_count",
    *[f"ignored_{category}_count" for category in IGNORED_DATAGRAM_CATEGORIES],
]
SERIAL_CONTROL_POLICIES = {
    "esp32c3": {"dtr": False, "rts": False},
    "arduino_r4": {"dtr": True, "rts": False},
}

HOST_SOURCE_DEPENDENCIES = (
    "deployment/final_hil/__init__.py",
    "deployment/final_hil/__main__.py",
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
    "deployment/wireless_hil/host/udp_session.py",
    "deployment/wireless_hil/host/wireless_common.py",
)
HOST_PACKAGE_DISTRIBUTIONS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "pyserial": "pyserial",
    "scikit-learn": "scikit-learn",
    "scipy": "scipy",
    "torch": "torch",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("UTC evidence timestamp must use a zero UTC offset")
    return parsed


def _absolute_path_without_symlinks(path: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(path))
    if any(component.is_symlink() for component in (absolute, *absolute.parents)):
        raise RuntimeError(f"{label} path cannot contain symlinks")
    return absolute


def _run_capture(command: list[str], cwd: Path) -> tuple[int, str]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return -1, ""
    return result.returncode, result.stdout.strip()


def collect_host_environment(repo_root: Path | None = None) -> dict[str, Any]:
    selected_root = repo_root or Path(__file__).resolve().parents[2]
    root = _absolute_path_without_symlinks(selected_root, "Final HIL host source root")
    if not root.is_dir():
        raise RuntimeError("Final HIL host source root is not a directory")
    revision_code, revision = _run_capture(["git", "rev-parse", "HEAD"], root)
    status_code, status = _run_capture(["git", "status", "--porcelain=v1"], root)
    source_dependencies = []
    for relative in HOST_SOURCE_DEPENDENCIES:
        path = _absolute_path_without_symlinks(
            root / relative, "Final HIL host dependency"
        )
        if not path.is_file():
            raise RuntimeError(f"Final HIL host dependency is missing: {relative}")
        source_dependencies.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    package_versions: dict[str, str | None] = {}
    for label, distribution in HOST_PACKAGE_DISTRIBUTIONS.items():
        try:
            package_versions[label] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            package_versions[label] = None
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "python_executable": sys.executable,
        "git_revision": revision if revision_code == 0 else None,
        "git_status_available": status_code == 0,
        "git_worktree_dirty": bool(status) if status_code == 0 else None,
        "git_status_sha256": (
            hashlib.sha256(status.encode("utf-8")).hexdigest()
            if status_code == 0
            else None
        ),
        "source_dependency_schema": "cukd_final_hil_host_sources_v1",
        "source_dependencies": source_dependencies,
        "package_versions": package_versions,
    }


def validate_host_environment(
    payload: Mapping[str, Any], *, source_root: Path | None = None
) -> dict[str, Any]:
    """Verify the exact host code that executed or verified an HIL record."""

    environment = dict(payload)
    if environment.get("source_dependency_schema") != "cukd_final_hil_host_sources_v1":
        raise RuntimeError("Final HIL host source ledger schema is invalid")
    entries = environment.get("source_dependencies")
    if not isinstance(entries, list):
        raise RuntimeError("Final HIL host source ledger is missing")
    expected = set(HOST_SOURCE_DEPENDENCIES)
    observed = {
        item.get("path") for item in entries if isinstance(item, Mapping)
    }
    if observed != expected or len(entries) != len(expected):
        raise RuntimeError("Final HIL host source ledger is incomplete or duplicated")
    source = Path(source_root) if source_root is not None else Path(__file__).resolve().parents[2]
    root = _absolute_path_without_symlinks(source, "Final HIL host source root")
    if not root.is_dir():
        raise RuntimeError("Final HIL host source root is not a directory")
    for item in entries:
        relative = item.get("path")
        if (
            not isinstance(relative, str)
            or not relative
            or "\\" in relative
            or Path(relative).as_posix() != relative
            or Path(relative).is_absolute()
            or Path(relative).drive
            or ".." in Path(relative).parts
        ):
            raise RuntimeError("Final HIL host source ledger path is unsafe")
        path = _absolute_path_without_symlinks(
            root / relative, "Final HIL host dependency"
        )
        if (
            not path.is_file()
            or path.stat().st_size != item.get("size_bytes")
            or sha256_file(path) != item.get("sha256")
        ):
            raise RuntimeError(f"Final HIL host dependency changed: {relative}")
    versions = environment.get("package_versions")
    if (
        not isinstance(versions, Mapping)
        or set(versions) != set(HOST_PACKAGE_DISTRIBUTIONS)
        or any(not isinstance(value, str) or not value for value in versions.values())
    ):
        raise RuntimeError("Final HIL host package-version ledger is missing")
    return environment


def _bundle_source_options(host_source_root: Path | None) -> dict[str, Any]:
    if host_source_root is None:
        return {}
    root = Path(host_source_root)
    if root.is_symlink() or not root.resolve().is_dir():
        raise RuntimeError("Final HIL host source root is invalid")
    resolved = root.resolve()
    builder = resolved / "deployment" / "final_hil" / "bundles.py"
    if not builder.is_file() or builder.is_symlink():
        raise RuntimeError("Final HIL bundle-builder source is unavailable")
    return {
        "expected_builder_sha256": sha256_file(builder),
        "canonical_source_root": resolved,
    }


def _canonical_serial_endpoint(port: str) -> str:
    if os.name != "posix":
        return port
    try:
        return str(Path(port).resolve(strict=True))
    except (OSError, RuntimeError):
        return port


def _detected_serial_number(port: str) -> str:
    try:
        from serial.tools import list_ports
    except ImportError as exc:  # pragma: no cover - hardware host only
        raise RuntimeError("pyserial is required for physical-port identity") from exc
    endpoints = list(list_ports.comports())
    matches = [item for item in endpoints if item.device == port]
    if not matches:
        requested_endpoint = _canonical_serial_endpoint(port)
        matches = [
            item
            for item in endpoints
            if _canonical_serial_endpoint(item.device) == requested_endpoint
        ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one enumerated serial endpoint for {port}")
    serial_number = matches[0].serial_number
    if not isinstance(serial_number, str) or not serial_number.strip():
        raise RuntimeError(f"Serial endpoint {port} has no stable physical serial number")
    return serial_number.strip()


def require_physical_port_serial(port: str, expected: str) -> str:
    if not isinstance(expected, str) or not expected.strip():
        raise ValueError("A non-empty physical port serial is required")
    observed = _detected_serial_number(port)
    if observed != expected.strip():
        raise RuntimeError(
            f"Physical serial mismatch on {port}: observed {observed!r}, "
            f"expected {expected.strip()!r}"
        )
    return observed


def require_session_id(value: str, label: str = "Campaign session ID") -> str:
    if (
        not isinstance(value, str)
        or len(value) != 32
        or any(character not in "0123456789ABCDEF" for character in value)
    ):
        raise ValueError(f"{label} must be 32 uppercase hexadecimal digits")
    return value


def _serial_control_policy(board: str) -> dict[str, bool]:
    try:
        return dict(SERIAL_CONTROL_POLICIES[board])
    except KeyError as exc:
        raise ValueError(f"Unsupported serial-control board: {board}") from exc


def _open_serial_with_board_policy(
    serial_module: Any,
    *,
    board: str,
    port: str,
    baud: int,
    timeout_seconds: float,
) -> Any:
    policy = _serial_control_policy(board)
    device = serial_module.Serial(
        port=None,
        baudrate=baud,
        timeout=timeout_seconds,
        write_timeout=timeout_seconds,
        xonxoff=False,
        rtscts=False,
        dsrdtr=False,
    )
    device.port = port
    device.dtr = policy["dtr"]
    device.rts = policy["rts"]
    device.open()
    return device


def _write_serial_all(device: Any, payload: bytes) -> None:
    written = device.write(payload)
    if written != len(payload):
        raise RuntimeError(
            f"Serial write accepted {written} of {len(payload)} bytes"
        )
    device.flush()


@dataclass(frozen=True)
class StageDataset:
    stage: dict[str, Any]
    replay_rows: list[dict[str, Any]]
    reference_rows: list[dict[str, Any]]
    input_binding: dict[str, Any]


def _reference_map(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        result.append(
            {
                "row_id": int(row["row_id"]),
                "source_row_index": int(row["source_row_index"]),
                "true_label": int(row["true_label"]),
                "fixed_pred": int(row["fixed_pred"]),
                "fp32_pred": int(row["fp32_pred"]),
                "logits": [int(row[f"fixed_logit_{index}"]) for index in range(5)],
                **(
                    {
                        "original_full_test_row_id": int(
                            row["original_full_test_row_id"]
                        )
                    }
                    if "original_full_test_row_id" in row
                    else {}
                ),
            }
        )
    return result


def _replay_map(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "row_id": int(row["row_id"]),
            "source_row_index": int(row["source_row_index"]),
            "features": [int(row[f"f{index}"]) for index in range(17)],
            **(
                {
                    "original_full_test_row_id": int(
                        row["original_full_test_row_id"]
                    )
                }
                if "original_full_test_row_id" in row
                else {}
            ),
        }
        for row in rows
    ]


def load_stage_dataset(
    *,
    export: FinalExportIdentity,
    cohort_dir: Path,
    stage_name: str,
) -> StageDataset:
    stage = stage_contract(stage_name)
    cohort = validate_balanced_cohort(
        cohort_dir,
        identities={export.model_key: export},
        reconstruct_sources=False,
        allow_identity_subset=True,
    )
    model_entry = cohort["models"].get(export.model_key)
    if not isinstance(model_entry, dict):
        raise RuntimeError("Balanced cohort lacks this final model")
    expected_export_fields = {
        "export_id": export.export_id,
        "trained_state_sha256": export.trained_state_sha256,
        "full_replay_sha256": export.full_replay_sha256,
        "full_reference_sha256": export.full_reference_sha256,
    }
    for field, expected in expected_export_fields.items():
        if model_entry.get(field) != expected:
            raise RuntimeError(f"Balanced cohort differs from export for {field}")

    if stage["input_role"] in {"balanced_timing", "balanced_timing_warmup"}:
        replay_path = cohort_dir.resolve() / cohort["replay_file"]
        reference_path = cohort_dir.resolve() / model_entry["reference_file"]
        source_kind = "balanced_timing_cohort"
        if cohort["selection"].get("not_first_1000_full_replay_prefix") is not True:
            raise RuntimeError("Timing stage resolved to an unguarded first-1,000 prefix")
    else:
        replay_path = Path(export.root) / "hil_replay_vectors.csv"
        reference_path = Path(export.root) / "hil_reference_predictions.csv"
        source_kind = "full_final_export"

    _, raw_replay = _csv_rows(replay_path)
    _, raw_reference = _csv_rows(reference_path)
    expected_rows = int(stage["rows"])
    if stage_name in {"warmup_10", "smoke_10"}:
        raw_replay = raw_replay[:expected_rows]
        raw_reference = raw_reference[:expected_rows]
    if len(raw_replay) != expected_rows or len(raw_reference) != expected_rows:
        raise RuntimeError(f"Stage {stage_name} resolved to the wrong row count")
    _validate_dense_rows(raw_replay, expected_rows, f"{stage_name} replay")
    _validate_dense_rows(raw_reference, expected_rows, f"{stage_name} reference")
    replay_rows = _replay_map(raw_replay)
    reference_rows = _reference_map(raw_reference)
    if [row["source_row_index"] for row in replay_rows] != [
        row["source_row_index"] for row in reference_rows
    ]:
        raise RuntimeError("Stage replay/reference source-row identities differ")
    if stage["input_role"] == "full_replay_smoke_prefix" and [
        row["row_id"] for row in replay_rows
    ] != list(range(10)):
        raise RuntimeError("Smoke stage is not the explicit full-replay prefix")
    if stage["input_role"] in {"balanced_timing", "balanced_timing_warmup"}:
        originals = [row["original_full_test_row_id"] for row in replay_rows]
        if len(replay_rows) == 1000 and originals == list(range(1000)):
            raise RuntimeError("Timing stage accidentally uses the full-replay prefix")

    selected_binding = canonical_json_sha256(
        {
            "replay": replay_rows,
            "reference": reference_rows,
        }
    )
    return StageDataset(
        stage=stage,
        replay_rows=replay_rows,
        reference_rows=reference_rows,
        input_binding={
            "source_kind": source_kind,
            "base_replay_path_recorded": str(replay_path),
            "base_replay_sha256": sha256_file(replay_path),
            "base_reference_path_recorded": str(reference_path),
            "base_reference_sha256": sha256_file(reference_path),
            "selected_rows_canonical_sha256": selected_binding,
            "cohort_manifest_sha256": sha256_file(
                cohort_dir.resolve() / "final_timing_cohort_manifest.json"
            ),
            "is_full_replay_prefix": stage["input_role"]
            == "full_replay_smoke_prefix",
            "is_balanced_timing_cohort": stage["input_role"]
            in {"balanced_timing", "balanced_timing_warmup"},
        },
    )


def _write_response_csv(path: Path, responses: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FINAL_RESPONSE_FIELDS)
        writer.writeheader()
        for response in responses:
            logits = list(response["logits"])
            row = {
                **{field: response.get(field, 0) for field in FINAL_RESPONSE_FIELDS},
                **{
                    f"fixed_logit_{index}": int(logits[index])
                    for index in range(5)
                },
            }
            writer.writerow(row)
    os.replace(temporary, path)


def _read_response_csv(path: Path) -> list[dict[str, Any]]:
    fields, rows = _csv_rows(path)
    if fields != FINAL_RESPONSE_FIELDS:
        raise RuntimeError("Final response CSV schema is invalid")
    result = []
    for row in rows:
        parsed = {
            "row_id": int(row["row_id"]),
            "status": row["status"],
            "predicted_class": int(row["predicted_class"]),
            "logits": [int(row[f"fixed_logit_{index}"]) for index in range(5)],
        }
        for field in [
            "preprocess_us",
            "inference_us",
            "total_us",
            "host_observed_rtt_us",
            "transaction_elapsed_us",
            "attempts",
            "response_timeout_count",
            "ignored_datagram_count",
            *[f"ignored_{category}_count" for category in IGNORED_DATAGRAM_CATEGORIES],
        ]:
            parsed[field] = int(row[field])
        result.append(parsed)
    return result


def validate_wifi_counter_reconciliation(
    *,
    rows: Sequence[Mapping[str, Any]],
    controls: Mapping[str, Any],
    session_counters: Mapping[str, Any],
    expected_rows: int,
) -> dict[str, int]:
    max_attempts = controls.get("udp_max_attempts")
    if (
        isinstance(max_attempts, bool)
        or not isinstance(max_attempts, int)
        or not 1 <= max_attempts <= MAX_ATTEMPTS
    ):
        raise RuntimeError("Wi-Fi attempt lacks a valid retry ceiling")
    end = controls.get("end")
    begin = controls.get("begin")
    if not isinstance(end, Mapping) or not isinstance(begin, Mapping):
        raise RuntimeError("Wi-Fi attempt lacks begin/end control evidence")
    device = end.get("device_stage_counters")
    if not isinstance(device, Mapping):
        raise RuntimeError("Wi-Fi attempt lacks device stage counters")
    for field in [
        "oversized_datagrams",
        "short_reads",
        "bad_envelopes",
        "wrong_sessions",
        "wrong_endpoints",
        "wrong_stages",
        "control_errors",
        "data_errors",
        "stale_transactions",
    ]:
        if int(device.get(field, -1)) != 0:
            raise RuntimeError(f"Wi-Fi non-retry device counter is nonzero: {field}")
    for field, value in session_counters.items():
        disallowed_ignored = field.startswith("ignored_") and field != (
            "ignored_wrong_attempt"
        )
        if (disallowed_ignored or field in {
            "stale_datagrams_drained_before_identity",
            "device_protocol_errors",
        }) and int(value) != 0:
            raise RuntimeError(f"Wi-Fi non-retry host counter is nonzero: {field}")
    if int(device.get("completed_rows", -1)) != expected_rows or int(
        device.get("inferences", -1)
    ) != expected_rows:
        raise RuntimeError("Wi-Fi device did not prove exact inference count")
    identity = controls.get("identity")
    end_control = controls.get("end")
    if not isinstance(identity, Mapping) or not isinstance(end_control, Mapping):
        raise RuntimeError("Wi-Fi attempt lacks identity/end exchange evidence")
    exchanges = [identity, begin, *rows, end_control]
    for exchange in exchanges:
        attempts = int(exchange.get("attempts", 0))
        timeouts = int(exchange.get("response_timeout_count", -1))
        ignored_total = int(exchange.get("ignored_datagram_count", -1))
        ignored_parts = sum(
            int(exchange.get(f"ignored_{category}_count", -1))
            for category in IGNORED_DATAGRAM_CATEGORIES
        )
        if attempts < 1 or attempts > max_attempts or timeouts != attempts - 1:
            raise RuntimeError("Wi-Fi exchange retry/timeout counts disagree")
        if ignored_total < 0 or ignored_parts != ignored_total:
            raise RuntimeError("Wi-Fi exchange ignored-datagram counts disagree")
        prior_attempt_responses = int(
            exchange.get("ignored_wrong_attempt_count", -1)
        )
        if not 0 <= prior_attempt_responses <= timeouts:
            raise RuntimeError(
                "Wi-Fi prior-attempt responses exceed recorded timeouts"
            )
    total_retransmissions = sum(int(item["attempts"]) - 1 for item in exchanges)
    total_ignored = sum(int(item["ignored_datagram_count"]) for item in exchanges)
    expected_host_counters = {
        "retransmissions": total_retransmissions,
        "response_timeouts": total_retransmissions,
        "datagrams_sent": len(exchanges) + total_retransmissions,
        "datagrams_received": len(exchanges) + total_ignored,
    }
    for field, expected in expected_host_counters.items():
        if int(session_counters.get(field, -1)) != expected:
            raise RuntimeError(f"Wi-Fi host session counter cannot be reconciled: {field}")
    for category in IGNORED_DATAGRAM_CATEGORIES:
        expected = sum(
            int(item[f"ignored_{category}_count"]) for item in exchanges
        )
        if int(session_counters.get(f"ignored_{category}", -1)) != expected:
            raise RuntimeError(
                f"Wi-Fi host ignored counter cannot be reconciled: {category}"
            )
    data_retransmissions = sum(int(row["attempts"]) - 1 for row in rows)
    prior_attempt_responses = sum(
        int(item["ignored_wrong_attempt_count"]) for item in exchanges
    )
    duplicate_replays = int(device.get("duplicate_replays", -1))
    if duplicate_replays < 0 or duplicate_replays > data_retransmissions:
        raise RuntimeError("Device duplicate replays exceed host data retransmissions")
    begin_retransmissions = int(begin.get("attempts", 0)) - 1
    received = int(device.get("received_datagrams", -1))
    begin_replays_received = received - expected_rows - 2 - duplicate_replays
    if not 0 <= begin_replays_received <= begin_retransmissions:
        raise RuntimeError("Device received-datagram count cannot be reconciled")
    return {
        "host_data_retransmissions": data_retransmissions,
        "host_ignored_prior_attempt_responses": prior_attempt_responses,
        "device_duplicate_replays": duplicate_replays,
        "host_data_retransmissions_without_device_duplicate": (
            data_retransmissions - duplicate_replays
        ),
        "host_begin_retransmissions": begin_retransmissions,
        "device_begin_replays_received": begin_replays_received,
        "host_begin_retransmissions_not_received": (
            begin_retransmissions - begin_replays_received
        ),
    }


def verify_response_records(
    *,
    responses: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    transport: str,
    controls: Mapping[str, Any] | None = None,
    session_counters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected = len(reference_rows)
    observed_ids = [int(row["row_id"]) for row in responses]
    if len(responses) != expected or observed_ids != list(range(expected)):
        raise RuntimeError("Response sequence is incomplete, duplicated, or out of order")
    if any(row.get("status") != "OK" for row in responses):
        raise RuntimeError("Response sequence contains a non-OK status")
    mismatches: list[int] = []
    for response, reference in zip(responses, reference_rows):
        row_id = int(response["row_id"])
        if (
            int(response["predicted_class"]) != int(reference["fixed_pred"])
            or list(response["logits"]) != list(reference["logits"])
        ):
            mismatches.append(row_id)
        preprocess = int(response["preprocess_us"])
        inference = int(response["inference_us"])
        total = int(response["total_us"])
        rtt = int(response["host_observed_rtt_us"])
        elapsed = int(response["transaction_elapsed_us"])
        if min(preprocess, inference, total, rtt, elapsed) <= 0 or total != (
            preprocess + inference
        ):
            raise RuntimeError(f"Invalid device/host timing at row {row_id}")
        if max(preprocess, inference, total) > 0xFFFFFFFF:
            raise RuntimeError(f"Device timing exceeds uint32 at row {row_id}")
        if rtt < total:
            raise RuntimeError(
                f"Host RTT is below the included device compute time at row {row_id}"
            )
        ignored_parts = sum(
            int(response[f"ignored_{category}_count"])
            for category in IGNORED_DATAGRAM_CATEGORIES
        )
        if ignored_parts != int(response["ignored_datagram_count"]):
            raise RuntimeError(f"Ignored-datagram counts disagree at row {row_id}")
        if elapsed < rtt:
            raise RuntimeError(f"Transaction elapsed time is below successful RTT at row {row_id}")
        if transport == "usb_serial":
            if (
                int(response["attempts"]) != 1
                or int(response["response_timeout_count"]) != 0
                or elapsed != rtt
            ):
                raise RuntimeError("USB evidence contains a retry or retry timing")
        elif transport == "wifi_udp":
            if (
                int(response["attempts"]) < 1
                or int(response["response_timeout_count"])
                != int(response["attempts"]) - 1
            ):
                raise RuntimeError("Wi-Fi response retry/timeout counts disagree")
        else:
            raise ValueError(f"Unsupported final transport: {transport}")
    if mismatches:
        raise RuntimeError(f"Exact prediction/logit mismatches: {mismatches[:20]}")
    reconciliation = None
    if transport == "wifi_udp":
        reconciliation = validate_wifi_counter_reconciliation(
            rows=responses,
            controls=controls or {},
            session_counters=session_counters or {},
            expected_rows=expected,
        )
    return {
        "status": "passed",
        "rows": expected,
        "sequence_exact": True,
        "predictions_exact": True,
        "logits_exact": True,
        "device_timing_boundary": "preprocess; forward pass plus argmax; their sum",
        "host_rtt_boundary": (
            "request write/send through complete matching response receipt; separate "
            "from device timing and not pure transport latency"
        ),
        "wifi_retry_reconciliation": reconciliation,
    }


def _attempt_payload_hash(payload: Mapping[str, Any]) -> str:
    copy = dict(payload)
    copy.pop("attempt_payload_sha256", None)
    return canonical_json_sha256(copy)


def _started_payload_hash(payload: Mapping[str, Any]) -> str:
    copy = dict(payload)
    copy.pop("attempt_started_sha256", None)
    return canonical_json_sha256(copy)


_ATTEMPT_STARTED_FIELDS = {
    "schema",
    "status",
    "attempt_id",
    "campaign_session_id",
    "started_utc",
    "combination",
    "bundle_id",
    "build_contract_id",
    "stage",
    "input_binding",
    "physical_identity",
    "attempt_started_sha256",
}
_ATTEMPT_FINAL_ONLY_FIELDS = {
    "finished_utc",
    "runtime_identity",
    "responses_file",
    "responses_sha256",
    "completed_rows",
    "controls",
    "session_network_counters",
    "verification",
    "error",
    "host_environment",
    "recovery",
    "attempt_payload_sha256",
}


def _validate_attempt_payload_shape(
    started: Mapping[str, Any], attempt: Mapping[str, Any]
) -> None:
    if set(started) != _ATTEMPT_STARTED_FIELDS:
        raise RuntimeError("Stage attempt-start field set is not canonical")
    if set(attempt) != _ATTEMPT_STARTED_FIELDS | _ATTEMPT_FINAL_ONLY_FIELDS:
        raise RuntimeError("Final stage-attempt field set is not canonical")


def verify_stage_attempt(
    attempt_dir: Path,
    *,
    export_dir: Path,
    cohort_dir: Path,
    bundle_dir: Path,
    verifier: Verifier | None = None,
    host_source_root: Path | None = None,
) -> dict[str, Any]:
    source_root = Path(attempt_dir)
    if source_root.is_symlink():
        raise RuntimeError("Stage attempt root cannot be a symlink")
    root = source_root.resolve()
    members = list(root.rglob("*"))
    if any(path.is_symlink() for path in members):
        raise RuntimeError("Stage attempt cannot contain symlinks")
    actual_files = {
        path.relative_to(root).as_posix() for path in members if path.is_file()
    }
    if actual_files != {
        "attempt_started.json",
        "final_attempt.json",
        "responses.csv",
    }:
        raise RuntimeError("Stage attempt file inventory is not exact")
    attempt = read_json(root / "final_attempt.json")
    if attempt.get("attempt_payload_sha256") != _attempt_payload_hash(attempt):
        raise RuntimeError("Stage attempt payload hash is invalid")
    if attempt.get("schema") != ATTEMPT_SCHEMA or attempt.get("status") != "passed":
        raise RuntimeError("Only a passed, finalized attempt is admissible")
    started = read_json(root / "attempt_started.json")
    _validate_attempt_payload_shape(started, attempt)
    if started.get("attempt_started_sha256") != _started_payload_hash(started):
        raise RuntimeError("Stage attempt start payload hash is invalid")
    if started.get("status") != "running" or started.get("attempt_id") != attempt.get(
        "attempt_id"
    ):
        raise RuntimeError("Stage attempt start/final identities differ")
    for field, value in started.items():
        if field != "status" and attempt.get(field) != value:
            raise RuntimeError(f"Stage start/final field differs: {field}")
    if _parse_utc(attempt["finished_utc"]) < _parse_utc(attempt["started_utc"]):
        raise RuntimeError("Stage attempt UTC interval is reversed")
    export = validate_final_export(export_dir, verifier=verifier)
    bundle = verify_final_bundle(
        bundle_dir,
        expected_export=export,
        **_bundle_source_options(host_source_root),
    )
    expected_combination = {
        "student": bundle["student"],
        "route": bundle["route"],
        "board": bundle["board"],
        "transport": bundle["transport"],
    }
    if (
        attempt.get("bundle_id") != bundle["bundle_id"]
        or attempt.get("build_contract_id") != bundle["build_contract_id"]
        or attempt.get("combination") != expected_combination
        or attempt.get("runtime_identity") != bundle["runtime_identity_response"]
    ):
        raise RuntimeError("Stage attempt is bound to another firmware bundle")
    _attempt_wifi_binding(attempt, bundle["transport"])
    dataset = load_stage_dataset(
        export=export,
        cohort_dir=cohort_dir,
        stage_name=attempt["stage"]["name"],
    )
    if attempt.get("stage") != dataset.stage or attempt.get(
        "input_binding"
    ) != dataset.input_binding:
        raise RuntimeError("Stage attempt input contract changed")
    if (
        attempt.get("responses_file") != "responses.csv"
        or attempt.get("completed_rows") != dataset.stage["rows"]
        or attempt.get("error") is not None
        or attempt.get("recovery") != _recovery_contract("passed", bundle["transport"])
    ):
        raise RuntimeError("Passed stage attempt metadata is contradictory")
    csv_path = root / "responses.csv"
    if sha256_file(csv_path) != attempt.get("responses_sha256"):
        raise RuntimeError("Stage response CSV hash changed")
    responses = _read_response_csv(csv_path)
    verification = verify_response_records(
        responses=responses,
        reference_rows=dataset.reference_rows,
        transport=bundle["transport"],
        controls=attempt.get("controls"),
        session_counters=attempt.get("session_network_counters"),
    )
    if attempt.get("verification") != verification:
        raise RuntimeError("Recorded stage verification differs from recomputation")
    validate_host_environment(
        attempt.get("host_environment", {}), source_root=host_source_root
    )
    return attempt


def _new_attempt_directory(output_root: Path, attempt_id: str | None) -> tuple[Path, str]:
    destination, identifier = _validate_attempt_destination(output_root, attempt_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    return destination, identifier


def _validate_attempt_destination(
    output_root: Path, attempt_id: str | None
) -> tuple[Path, str]:
    identifier = attempt_id or secrets.token_hex(16).upper()
    if len(identifier) != 32 or any(
        character not in "0123456789ABCDEF" for character in identifier
    ):
        raise ValueError("Attempt ID must be 32 uppercase hexadecimal digits")
    destination = output_root.resolve() / identifier
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite stage attempt: {destination}")
    return destination, identifier


def _recovery_contract(status: str, transport: str) -> dict[str, Any]:
    failed = status != "passed"
    wifi_failed = failed and transport == "wifi_udp"
    return {
        "resume_permitted": False,
        "new_attempt_required": failed,
        "new_campaign_session_required": wifi_failed,
        "new_wifi_session_required": wifi_failed,
        "rule": (
            "Failed or aborted evidence is retained but inadmissible. For USB, "
            "restart the failed stage at row zero under a new attempt. For Wi-Fi, "
            "restart the complete six-stage combination under new campaign and "
            "network session IDs so every accepted stage shares one connection."
        ),
    }


def _attempt_wifi_binding(
    attempt: Mapping[str, Any], transport: str
) -> tuple[str, str, str, str, float, int] | None:
    controls = attempt.get("controls")
    physical = attempt.get("physical_identity")
    if not isinstance(controls, Mapping) or not isinstance(physical, Mapping):
        raise RuntimeError("Stage attempt transport metadata is malformed")
    keys = {
        "wifi_network_session_id",
        "connection_path_recorded",
        "connection_record_sha256",
        "connection_payload_sha256",
        "udp_timeout_seconds",
        "udp_max_attempts",
    }
    if transport == "usb_serial":
        if any(key in controls for key in keys) or "wifi_network_session_id" in physical:
            raise RuntimeError("USB stage carries Wi-Fi connection metadata")
        combination = attempt.get("combination")
        if not isinstance(combination, Mapping):
            raise RuntimeError("USB stage lacks a valid board identity")
        expected_controls = {
            "serial_open_policy": _serial_control_policy(str(combination.get("board"))),
            "baud": 115200,
        }
        if dict(controls) != expected_controls:
            raise RuntimeError("USB stage serial-control policy differs from its board")
        return None
    if transport != "wifi_udp":
        raise RuntimeError(f"Unsupported stage-attempt transport: {transport}")
    network_session_id = controls.get("wifi_network_session_id")
    require_session_id(str(network_session_id or ""), "Wi-Fi network session ID")
    if physical.get("wifi_network_session_id") != network_session_id:
        raise RuntimeError("Wi-Fi stage carries inconsistent network session IDs")
    connection_record_sha256 = controls.get("connection_record_sha256")
    connection_payload_sha256 = controls.get("connection_payload_sha256")
    connection_path_recorded = controls.get("connection_path_recorded")
    timeout_seconds = controls.get("udp_timeout_seconds")
    max_attempts = controls.get("udp_max_attempts")
    if not _is_sha256(connection_record_sha256) or not _is_sha256(
        connection_payload_sha256
    ):
        raise RuntimeError("Wi-Fi stage connection record binding is invalid")
    if not isinstance(connection_path_recorded, str) or not connection_path_recorded:
        raise RuntimeError("Wi-Fi stage connection path binding is invalid")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0
        or isinstance(max_attempts, bool)
        or not isinstance(max_attempts, int)
        or not 1 <= max_attempts <= MAX_ATTEMPTS
    ):
        raise RuntimeError("Wi-Fi stage execution policy binding is invalid")
    return (
        str(network_session_id),
        connection_path_recorded,
        str(connection_record_sha256),
        str(connection_payload_sha256),
        float(timeout_seconds),
        max_attempts,
    )


def _validate_attempt_connection_set(
    attempts: Sequence[Mapping[str, Any]], transport: str
) -> tuple[str, str, str, str, float, int] | None:
    bindings = {
        binding
        for binding in (_attempt_wifi_binding(attempt, transport) for attempt in attempts)
        if binding is not None
    }
    if transport == "wifi_udp" and len(bindings) != 1:
        raise RuntimeError("Wi-Fi stages do not share one sealed connection record")
    if transport == "usb_serial" and bindings:
        raise RuntimeError("USB stages carry Wi-Fi connection metadata")
    return next(iter(bindings)) if bindings else None


def _write_attempt_started(
    destination: Path,
    *,
    attempt_id: str,
    campaign_session_id: str,
    bundle: Mapping[str, Any],
    dataset: StageDataset,
    physical_identity: Mapping[str, Any],
) -> dict[str, Any]:
    require_session_id(campaign_session_id)
    payload = {
        "schema": ATTEMPT_SCHEMA,
        "status": "running",
        "attempt_id": attempt_id,
        "campaign_session_id": campaign_session_id,
        "started_utc": utc_now(),
        "combination": {
            "student": bundle["student"],
            "route": bundle["route"],
            "board": bundle["board"],
            "transport": bundle["transport"],
        },
        "bundle_id": bundle["bundle_id"],
        "build_contract_id": bundle["build_contract_id"],
        "stage": dataset.stage,
        "input_binding": dataset.input_binding,
        "physical_identity": dict(physical_identity),
    }
    payload["attempt_started_sha256"] = _started_payload_hash(payload)
    atomic_write_json(destination / "attempt_started.json", payload)
    return payload


def _finalize_attempt(
    destination: Path,
    *,
    started: Mapping[str, Any],
    bundle: Mapping[str, Any],
    responses: Sequence[Mapping[str, Any]],
    dataset: StageDataset,
    observed_runtime_identity: str | None,
    controls: Mapping[str, Any],
    session_counters: Mapping[str, Any],
    error: Mapping[str, Any] | None,
    aborted: bool,
) -> Path:
    csv_path = destination / "responses.csv"
    _write_response_csv(csv_path, responses)
    verification: dict[str, Any] | None = None
    status = "aborted" if aborted else "failed"
    if error is None and observed_runtime_identity == bundle["runtime_identity_response"]:
        try:
            verification = verify_response_records(
                responses=responses,
                reference_rows=dataset.reference_rows,
                transport=bundle["transport"],
                controls=controls,
                session_counters=session_counters,
            )
            status = "passed"
        except Exception as exc:
            error = {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "phase": "post_capture_verification",
            }
            status = "failed"
    payload: dict[str, Any] = {
        **dict(started),
        "status": status,
        "finished_utc": utc_now(),
        "runtime_identity": observed_runtime_identity,
        "responses_file": "responses.csv",
        "responses_sha256": sha256_file(csv_path),
        "completed_rows": len(responses),
        "controls": dict(controls),
        "session_network_counters": dict(session_counters),
        "verification": verification,
        "error": dict(error) if error is not None else None,
        "host_environment": collect_host_environment(),
        "recovery": _recovery_contract(status, bundle["transport"]),
    }
    payload["attempt_payload_sha256"] = _attempt_payload_hash(payload)
    final_path = destination / "final_attempt.json"
    atomic_write_json(final_path, payload)
    return final_path


def execute_usb_stage(
    *,
    export_dir: Path,
    cohort_dir: Path,
    bundle_dir: Path,
    stage_name: str,
    port: str,
    physical_port_serial: str,
    output_root: Path,
    campaign_session_id: str,
    baud: int = 115200,
    timeout_seconds: float = 2.0,
    settle_seconds: float = 2.0,
    attempt_id: str | None = None,
    verifier: Verifier | None = None,
) -> Path:
    """Run one USB stage with exactly one request per row and no retry."""

    if timeout_seconds <= 0 or baud <= 0 or settle_seconds < 0:
        raise ValueError("USB timing and baud parameters are invalid")
    require_session_id(campaign_session_id)
    try:
        import serial
    except ImportError as exc:  # pragma: no cover - hardware host only
        raise RuntimeError("pyserial is required") from exc
    export = validate_final_export(export_dir, verifier=verifier)
    bundle = verify_final_bundle(bundle_dir, expected_export=export)
    if bundle["transport"] != "usb_serial":
        raise RuntimeError("USB execution requires a usb_serial final bundle")
    dataset = load_stage_dataset(export=export, cohort_dir=cohort_dir, stage_name=stage_name)
    _, identifier = _validate_attempt_destination(output_root, attempt_id)
    serial_number = require_physical_port_serial(port, physical_port_serial)
    destination, identifier = _new_attempt_directory(output_root, identifier)
    started = _write_attempt_started(
        destination,
        attempt_id=identifier,
        campaign_session_id=campaign_session_id,
        bundle=bundle,
        dataset=dataset,
        physical_identity={"port": port, "physical_port_serial": serial_number},
    )
    responses: list[dict[str, Any]] = []
    observed_identity: str | None = None
    error: dict[str, Any] | None = None
    aborted = False
    device = None
    try:
        device = _open_serial_with_board_policy(
            serial,
            board=str(bundle["board"]),
            port=port,
            baud=baud,
            timeout_seconds=timeout_seconds,
        )
        time.sleep(settle_seconds)
        device.reset_input_buffer()
        device.reset_output_buffer()
        _write_serial_all(device, b"CUKDID?\n")
        deadline = time.monotonic() + max(3.0, timeout_seconds * 5)
        while time.monotonic() < deadline:
            line = device.readline().decode("ascii", errors="strict").strip()
            if line.startswith(RUNTIME_IDENTITY_PREFIX + ","):
                observed_identity = line
                break
        if observed_identity != bundle["runtime_identity_response"]:
            raise RuntimeError("USB runtime identity differs from final bundle")
        for row in dataset.replay_rows:
            row_id = int(row["row_id"])
            request = encode_request_line(row_id, row["features"]).encode("ascii")
            started_ns = time.monotonic_ns()
            _write_serial_all(device, request)
            decoded = None
            for _ in range(25):
                line = device.readline().decode("ascii", errors="strict")
                if not line:
                    raise TimeoutError(f"USB timeout at row {row_id}; no retry permitted")
                if not line.startswith("CUKD1R,"):
                    continue
                decoded = decode_response_line(line)
                break
            if decoded is None or int(decoded["row_id"]) != row_id:
                raise RuntimeError(f"USB response sequence failed at row {row_id}")
            elapsed_us = (time.monotonic_ns() - started_ns) // 1000
            responses.append(
                {
                    **decoded,
                    "host_observed_rtt_us": elapsed_us,
                    "transaction_elapsed_us": elapsed_us,
                    "attempts": 1,
                    "response_timeout_count": 0,
                    "ignored_datagram_count": 0,
                    **{
                        f"ignored_{category}_count": 0
                        for category in IGNORED_DATAGRAM_CATEGORIES
                    },
                }
            )
    except KeyboardInterrupt as exc:  # pragma: no cover - operator path
        aborted = True
        error = {"type": exc.__class__.__name__, "message": "operator interruption"}
    except Exception as exc:  # pragma: no cover - hardware host path
        error = {"type": exc.__class__.__name__, "message": str(exc)}
    finally:
        if device is not None:
            device.close()
    return _finalize_attempt(
        destination,
        started=started,
        bundle=bundle,
        responses=responses,
        dataset=dataset,
        observed_runtime_identity=observed_identity,
        controls={
            "serial_open_policy": _serial_control_policy(str(bundle["board"])),
            "baud": baud,
        },
        session_counters={},
        error=error,
        aborted=aborted,
    )


def _read_matching_serial_line(
    device: Any, prefix: str, deadline: float, *, expose: bool
) -> str:
    observed: list[str] = []
    while time.monotonic() < deadline:
        raw = device.readline()
        try:
            line = raw.decode("ascii", errors="strict").strip()
        except UnicodeDecodeError:
            line = ""
        if line and expose:
            observed.append(line)
        if line.startswith(prefix):
            return line
    suffix = f"; observed={observed[-5:]}" if expose else ""
    raise TimeoutError(f"No serial response with prefix {prefix!r}{suffix}")


def query_runtime_identity_serial(
    *,
    export_dir: Path,
    bundle_dir: Path,
    port: str,
    physical_port_serial: str,
    baud: int = 115200,
    timeout_seconds: float = 2.0,
    settle_seconds: float = 2.0,
    verifier: Verifier | None = None,
) -> str:
    """Query the flashed board directly and require the exact bundle identity."""

    if baud <= 0 or timeout_seconds <= 0 or settle_seconds < 0:
        raise ValueError("Runtime identity serial settings are invalid")
    try:
        import serial
    except ImportError as exc:  # pragma: no cover - hardware host only
        raise RuntimeError("pyserial is required") from exc
    export = validate_final_export(export_dir, verifier=verifier)
    bundle = verify_final_bundle(bundle_dir, expected_export=export)
    require_physical_port_serial(port, physical_port_serial)
    request = b"CUKDID?\n" if bundle["transport"] == "usb_serial" else b"CUKDWID?\n"
    device = _open_serial_with_board_policy(
        serial,
        board=str(bundle["board"]),
        port=port,
        baud=baud,
        timeout_seconds=timeout_seconds,
    )
    try:
        time.sleep(settle_seconds)
        device.reset_input_buffer()
        device.reset_output_buffer()
        _write_serial_all(device, request)
        observed = _read_matching_serial_line(
            device,
            RUNTIME_IDENTITY_PREFIX + ",",
            time.monotonic() + max(5.0, timeout_seconds * 5),
            expose=True,
        )
    finally:
        device.close()
    if observed != bundle["runtime_identity_response"]:
        raise RuntimeError("Post-upload runtime identity differs from final bundle")
    return observed


def configure_final_wifi(
    *,
    export_dir: Path,
    bundle_dir: Path,
    port: str,
    physical_port_serial: str,
    ssid: str,
    password: str,
    output_json: Path,
    baud: int = 115200,
    timeout_seconds: float = 2.0,
    connect_timeout_seconds: float = 45.0,
    verifier: Verifier | None = None,
) -> Path:
    """Provision a final Wi-Fi bundle without recording either credential."""

    destination = _absolute_path_without_symlinks(output_json, "Wi-Fi connection")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite Wi-Fi connection: {destination}")
    if (
        isinstance(baud, bool)
        or not isinstance(baud, int)
        or baud <= 0
        or isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0
        or isinstance(connect_timeout_seconds, bool)
        or not isinstance(connect_timeout_seconds, (int, float))
        or not math.isfinite(float(connect_timeout_seconds))
        or float(connect_timeout_seconds) <= 0
    ):
        raise ValueError("Wi-Fi provisioning timing and baud parameters are invalid")
    export = validate_final_export(export_dir, verifier=verifier)
    bundle = verify_final_bundle(bundle_dir, expected_export=export)
    if bundle["transport"] != "wifi_udp":
        raise RuntimeError("Wi-Fi provisioning requires a wifi_udp final bundle")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary_created = False
    config = ""
    try:
        if temporary.exists():
            raise FileExistsError(f"Stale temporary evidence exists: {temporary}")
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write("")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_created = True
        session_id = secrets.token_hex(16).upper()
        config = encode_wifi_config_line(
            ssid, password, DEFAULT_DEVICE_UDP_PORT, session_id
        )
        try:
            import serial
        except ImportError as exc:  # pragma: no cover - hardware host only
            raise RuntimeError("pyserial is required") from exc
        serial_number = require_physical_port_serial(port, physical_port_serial)
        started_utc = utc_now()
        device = _open_serial_with_board_policy(
            serial,
            board=str(bundle["board"]),
            port=port,
            baud=baud,
            timeout_seconds=timeout_seconds,
        )
        observed_identity = ""
        response_line = ""
        try:
            time.sleep(2.0)
            device.reset_input_buffer()
            device.reset_output_buffer()
            _write_serial_all(device, b"CUKDWID?\n")
            observed_identity = _read_matching_serial_line(
                device,
                RUNTIME_IDENTITY_PREFIX + ",",
                time.monotonic() + max(5.0, timeout_seconds * 5),
                expose=True,
            )
            if observed_identity != bundle["runtime_identity_response"]:
                raise RuntimeError("Wi-Fi serial runtime identity differs from final bundle")
            _write_serial_all(device, config.encode("ascii"))
            response_line = _read_matching_serial_line(
                device,
                "CUKDWCFG2R,",
                time.monotonic() + connect_timeout_seconds,
                expose=False,
            )
        finally:
            device.close()
        response = decode_wifi_config_response(response_line)
        if (
            response["session_id"] != session_id
            or response["status"] != "OK"
            or response["udp_port"] != DEFAULT_DEVICE_UDP_PORT
        ):
            raise RuntimeError("Board did not establish the requested final Wi-Fi session")
        payload: dict[str, Any] = {
            "schema": CONNECTION_SCHEMA,
            "status": "connected",
            "protocol_id": FINAL_WIFI_PROTOCOL,
            "bundle_id": bundle["bundle_id"],
            "build_contract_id": bundle["build_contract_id"],
            "runtime_identity": observed_identity,
            "board": bundle["board"],
            "student": bundle["student"],
            "route": bundle["route"],
            "transport": "wifi_udp",
            "session_id": session_id,
            "device_ip": response["device_ip"],
            "device_udp_port": response["udp_port"],
            "host_udp_port": DEFAULT_HOST_UDP_PORT,
            "provisioning_port": port,
            "physical_port_serial": serial_number,
            "serial_open_policy": _serial_control_policy(str(bundle["board"])),
            "wifi_mac_reported": response["wifi_mac"],
            "rssi_dbm_at_connection": response["rssi_dbm"],
            "connectivity_firmware_reported": response["connectivity_firmware"],
            "started_utc": started_utc,
            "finished_utc": utc_now(),
            "credentials_recorded": False,
            "host_environment": collect_host_environment(),
            "security_boundary": (
                "Session and endpoint binding provide experiment correlation, not "
                "cryptographic authentication or payload confidentiality."
            ),
        }
        payload["connection_payload_sha256"] = canonical_json_sha256(payload)
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Refusing to overwrite Wi-Fi connection: {destination}"
            ) from exc
        temporary.unlink()
        temporary_created = False
        return destination
    finally:
        config = ""
        password = ""
        ssid = ""
        if temporary_created and temporary.exists():
            temporary.unlink()


def validate_wifi_connection(
    path: Path,
    *,
    bundle: Mapping[str, Any],
    host_source_root: Path | None = None,
) -> dict[str, Any]:
    source = _absolute_path_without_symlinks(path, "Wi-Fi connection")
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = read_json(source)
    copy = dict(payload)
    recorded = copy.pop("connection_payload_sha256", None)
    if recorded != canonical_json_sha256(copy):
        raise RuntimeError("Wi-Fi connection payload hash is invalid")
    if payload.get("schema") != CONNECTION_SCHEMA or payload.get("status") != "connected":
        raise RuntimeError("Wi-Fi connection record is not connected final evidence")
    for field, expected in {
        "protocol_id": FINAL_WIFI_PROTOCOL,
        "bundle_id": bundle["bundle_id"],
        "build_contract_id": bundle["build_contract_id"],
        "runtime_identity": bundle["runtime_identity_response"],
        "board": bundle["board"],
        "student": bundle["student"],
        "route": bundle["route"],
        "transport": "wifi_udp",
        "device_udp_port": DEFAULT_DEVICE_UDP_PORT,
        "host_udp_port": DEFAULT_HOST_UDP_PORT,
        "credentials_recorded": False,
        "serial_open_policy": _serial_control_policy(str(bundle["board"])),
    }.items():
        if payload.get(field) != expected:
            raise RuntimeError(f"Wi-Fi connection differs from bundle for {field}")
    if (
        not isinstance(payload.get("physical_port_serial"), str)
        or not payload["physical_port_serial"]
        or not isinstance(payload.get("wifi_mac_reported"), str)
        or not payload["wifi_mac_reported"]
    ):
        raise RuntimeError("Wi-Fi connection lacks physical board identity")
    try:
        endpoint = ipaddress.ip_address(str(payload.get("device_ip", "")))
    except ValueError as exc:
        raise RuntimeError("Wi-Fi connection has an invalid device endpoint") from exc
    if endpoint.version != 4 or endpoint.is_unspecified or endpoint.is_multicast:
        raise RuntimeError("Wi-Fi connection has an invalid device endpoint")
    rssi = payload.get("rssi_dbm_at_connection")
    if isinstance(rssi, bool) or not isinstance(rssi, int) or not -127 <= rssi <= 0:
        raise RuntimeError("Wi-Fi connection has an invalid RSSI")
    require_session_id(str(payload.get("session_id", "")), "Wi-Fi network session ID")
    if _parse_utc(payload["finished_utc"]) < _parse_utc(payload["started_utc"]):
        raise RuntimeError("Wi-Fi connection UTC interval is reversed")
    validate_host_environment(
        payload.get("host_environment", {}), source_root=host_source_root
    )
    return payload


def execute_wifi_stage(
    *,
    export_dir: Path,
    cohort_dir: Path,
    bundle_dir: Path,
    connection_json: Path,
    stage_name: str,
    output_root: Path,
    campaign_session_id: str,
    timeout_seconds: float = 1.0,
    max_attempts: int = 3,
    attempt_id: str | None = None,
    verifier: Verifier | None = None,
) -> Path:
    """Run one stop-and-wait Wi-Fi stage with idempotent transaction retries."""

    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0
        or isinstance(max_attempts, bool)
        or not isinstance(max_attempts, int)
        or not 1 <= max_attempts <= MAX_ATTEMPTS
    ):
        raise ValueError("Wi-Fi timeout/max-attempts contract is invalid")
    require_session_id(campaign_session_id)
    stage_contract(stage_name)
    export = validate_final_export(export_dir, verifier=verifier)
    bundle = verify_final_bundle(bundle_dir, expected_export=export)
    if bundle["transport"] != "wifi_udp":
        raise RuntimeError("Wi-Fi execution requires a wifi_udp final bundle")
    connection_path = _absolute_path_without_symlinks(
        connection_json, "Wi-Fi connection"
    )
    if not connection_path.is_file():
        raise FileNotFoundError(connection_path)
    connection_record_sha256 = sha256_file(connection_path)
    connection = validate_wifi_connection(connection_path, bundle=bundle)
    if sha256_file(connection_path) != connection_record_sha256:
        raise RuntimeError("Wi-Fi connection record changed while it was validated")
    dataset = load_stage_dataset(export=export, cohort_dir=cohort_dir, stage_name=stage_name)
    prepared = [
        (
            int(row["row_id"]),
            encode_request_line(int(row["row_id"]), row["features"]).rstrip("\n"),
        )
        for row in dataset.replay_rows
    ]
    _, identifier = _validate_attempt_destination(output_root, attempt_id)
    require_physical_port_serial(
        str(connection["provisioning_port"]),
        str(connection["physical_port_serial"]),
    )
    destination, identifier = _new_attempt_directory(output_root, identifier)
    started = _write_attempt_started(
        destination,
        attempt_id=identifier,
        campaign_session_id=campaign_session_id,
        bundle=bundle,
        dataset=dataset,
        physical_identity={
            "provisioning_port": connection["provisioning_port"],
            "physical_port_serial": connection["physical_port_serial"],
            "wifi_mac_reported": connection["wifi_mac_reported"],
            "wifi_network_session_id": connection["session_id"],
        },
    )
    stage_id = secrets.token_hex(8).upper()
    responses: list[dict[str, Any]] = []
    controls: dict[str, Any] = {
        "wifi_network_session_id": connection["session_id"],
        "connection_path_recorded": str(connection_path),
        "connection_record_sha256": connection_record_sha256,
        "connection_payload_sha256": connection["connection_payload_sha256"],
        "udp_timeout_seconds": float(timeout_seconds),
        "udp_max_attempts": max_attempts,
    }
    observed_identity: str | None = None
    error: dict[str, Any] | None = None
    aborted = False
    begin_attempted = False
    stage_closed = False
    session: StrictUdpSession | None = None
    session_counters: dict[str, Any] = {}
    try:
        session = StrictUdpSession(
            device_ip=connection["device_ip"],
            device_port=DEFAULT_DEVICE_UDP_PORT,
            host_port=DEFAULT_HOST_UDP_PORT,
            session_id=connection["session_id"],
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
        )
        controls["stale_datagrams_drained_before_identity"] = (
            session.drain_stale_datagrams()
        )
        identity = session.exchange(
            stage_id=stage_id,
            transaction_id=IDENTITY_TRANSACTION_ID,
            inner_text="CUKDWID?",
        )
        observed_identity = identity.inner_text
        controls["identity"] = identity.evidence()
        if observed_identity != bundle["runtime_identity_response"]:
            raise RuntimeError("Wi-Fi UDP runtime identity differs from final bundle")
        begin_attempted = True
        begin = session.exchange(
            stage_id=stage_id,
            transaction_id=BEGIN_TRANSACTION_ID,
            inner_text=(
                f"CUKDWBEGIN,{stage_id},{dataset.stage['ordinal']},"
                f"{dataset.stage['rows']}"
            ),
        )
        controls["begin"] = {
            **begin.evidence(),
            "device_acknowledgement": parse_begin_response(
                begin.inner_text,
                stage_id=stage_id,
                ordinal=int(dataset.stage["ordinal"]),
                expected_rows=int(dataset.stage["rows"]),
            ),
        }
        for row_id, inner in prepared:
            exchange = session.exchange(
                stage_id=stage_id,
                transaction_id=f"{row_id:016X}",
                inner_text=inner,
            )
            decoded = decode_response_line(exchange.inner_text)
            if int(decoded["row_id"]) != row_id:
                raise RuntimeError("Wi-Fi response row differs from transaction ID")
            evidence = exchange.evidence()
            responses.append(
                {
                    **decoded,
                    "host_observed_rtt_us": evidence[
                        "host_observed_datagram_rtt_us"
                    ],
                    "transaction_elapsed_us": evidence["transaction_elapsed_us"],
                    "attempts": evidence["attempts"],
                    "response_timeout_count": evidence["response_timeout_count"],
                    "ignored_datagram_count": evidence["ignored_datagram_count"],
                    **{
                        f"ignored_{category}_count": evidence[
                            f"ignored_{category}_count"
                        ]
                        for category in IGNORED_DATAGRAM_CATEGORIES
                    },
                }
            )
        end = session.exchange(
            stage_id=stage_id,
            transaction_id=END_TRANSACTION_ID,
            inner_text=f"CUKDWEND,{stage_id},{len(responses)}",
        )
        controls["end"] = {
            **end.evidence(),
            "device_stage_counters": parse_end_response(
                end.inner_text,
                stage_id=stage_id,
                ordinal=int(dataset.stage["ordinal"]),
                expected_rows=int(dataset.stage["rows"]),
            ),
        }
        stage_closed = True
    except KeyboardInterrupt as exc:  # pragma: no cover - operator path
        aborted = True
        error = {"type": exc.__class__.__name__, "message": "operator interruption"}
    except Exception as exc:  # pragma: no cover - hardware path
        error = {"type": exc.__class__.__name__, "message": str(exc)}
    finally:
        if session is not None and begin_attempted and not stage_closed:
            abort_errors = []
            candidates = [len(responses)]
            if len(responses) < int(dataset.stage["rows"]):
                candidates.append(len(responses) + 1)
            for completed in candidates:
                try:
                    abort = session.exchange(
                        stage_id=stage_id,
                        transaction_id=ABORT_TRANSACTION_ID,
                        inner_text=f"CUKDWABORT,{stage_id},{completed}",
                    )
                    controls["abort"] = {
                        **abort.evidence(),
                        "device_acknowledgement": parse_abort_response(
                            abort.inner_text,
                            stage_id=stage_id,
                            completed_rows=completed,
                            expected_rows=int(dataset.stage["rows"]),
                        ),
                    }
                    stage_closed = True
                    break
                except Exception as abort_exc:  # pragma: no cover - hardware path
                    abort_errors.append(
                        {
                            "completed_candidate": completed,
                            "type": abort_exc.__class__.__name__,
                            "message": str(abort_exc),
                        }
                    )
            if not stage_closed and error is not None:
                error["abort_errors"] = abort_errors
        if session is not None:
            session_counters = session.counter_evidence()
            session.close()
    controls["stage_id"] = stage_id
    return _finalize_attempt(
        destination,
        started=started,
        bundle=bundle,
        responses=responses,
        dataset=dataset,
        observed_runtime_identity=observed_identity,
        controls=controls,
        session_counters=session_counters,
        error=error,
        aborted=aborted,
    )
