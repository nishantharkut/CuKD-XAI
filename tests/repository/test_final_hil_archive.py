from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

import deployment.final_hil.archive as archive
from deployment.final_hil.bundles import runtime_identity
from deployment.final_hil.contracts import (
    FinalExportIdentity,
    canonical_json_sha256,
    read_json,
    sha256_file,
)
from deployment.final_hil.evidence import PROVENANCE_SCHEMA, TIMING_STATISTICAL_UNIT
from deployment.final_hil.runtime import (
    ATTEMPT_SCHEMA,
    CONNECTION_SCHEMA,
    FINAL_RESPONSE_FIELDS,
    FINAL_WIFI_PROTOCOL,
    _attempt_payload_hash,
    _recovery_contract,
    _started_payload_hash,
    collect_host_environment,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_KEY = "student_A_rf_kd"
BOARD = "esp32c3"
TRANSPORT = "wifi_udp"
COMBINATION_ID = f"{MODEL_KEY}__{BOARD}__{TRANSPORT}"
STAGES = tuple(
    {
        "name": name,
        "ordinal": ordinal,
        "rows": 1,
        "include_in_reported_metrics": name not in {"warmup_10", "smoke_10"},
        "include_in_timing_metrics": name.startswith("timing_1000_"),
        "include_in_fidelity_metrics": name == "full_56301",
    }
    for ordinal, name in enumerate(
        (
            "warmup_10",
            "smoke_10",
            "timing_1000_r1",
            "timing_1000_r2",
            "timing_1000_r3",
            "full_56301",
        ),
        start=1,
    )
)
HASHES = {
    "checkpoint": "1" * 64,
    "trained": "2" * 64,
    "dataset": "3" * 64,
    "split": "4" * 64,
    "scaler": "5" * 64,
    "export": "6" * 64,
    "report": "7" * 64,
    "bundle": "a" * 64,
    "build": "b" * 64,
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _seal(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    result.pop(field, None)
    result[field] = canonical_json_sha256(result)
    return result


def _portable_blocked_payload() -> dict[str, Any]:
    identity: dict[str, Any] = {
        "protocol": "wsnds_feature_group_split_train_only_scaler_10seed_v2",
        "seed": 42,
        "student": "B",
        "route": "scratch",
    }
    identity["blocked_audit_id"] = canonical_json_sha256(identity)
    return _seal(
        {
            "status": "blocked",
            "identity": identity,
            "quality_gates_passed": False,
            "test_rows": 56_301,
            "zero_saturation_passed": True,
            "accumulator_bounds_passed": True,
            "preprocess_bounds_passed": True,
            "fixed_vs_fp32_agreement": 0.989,
            "minimum_fixed_vs_fp32_agreement": 0.99,
            "absolute_macro_f1_drop": 0.02,
            "maximum_absolute_macro_f1_drop": 0.015,
        },
        "audit_payload_sha256",
    )


def test_portable_blocked_audit_rejects_wrong_cardinality_and_nonfinite_metrics(
    tmp_path: Path,
) -> None:
    path = tmp_path / "blocked.json"
    payload = _portable_blocked_payload()
    _write_json(path, payload)
    assert archive._portable_blocked_audit_verifier(path)["status"] == (
        "blocked_verified"
    )

    for field, value in [
        ("test_rows", 56_300),
        ("fixed_vs_fp32_agreement", float("nan")),
        ("absolute_macro_f1_drop", float("inf")),
    ]:
        tampered = dict(payload)
        tampered[field] = value
        tampered = _seal(tampered, "audit_payload_sha256")
        _write_json(path, tampered)
        with pytest.raises(RuntimeError, match="blocked-audit gate evidence"):
            archive._portable_blocked_audit_verifier(path)


def _identity(root: Path) -> FinalExportIdentity:
    return FinalExportIdentity(
        root=str(root.resolve()),
        protocol="test_fgds_protocol",
        seed=42,
        student="A",
        route="rf_kd",
        checkpoint_file_sha256=HASHES["checkpoint"],
        trained_state_sha256=HASHES["trained"],
        export_id=HASHES["export"],
        dataset_sha256=HASHES["dataset"],
        split_indices_sha256=HASHES["split"],
        scaler_sha256=HASHES["scaler"],
        manifest_sha256=sha256_file(root / "final_export_manifest.json"),
        report_sha256=HASHES["report"],
        full_replay_sha256=sha256_file(root / "hil_replay_vectors.csv"),
        full_reference_sha256=sha256_file(
            root / "hil_reference_predictions.csv"
        ),
        test_rows=1,
        gate_status="passed",
    )


def _patch_compact_validators(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(archive, "MODEL_KEYS", (MODEL_KEY,))
    monkeypatch.setattr(archive, "BOARDS", (BOARD,))
    monkeypatch.setattr(archive, "TRANSPORTS", (TRANSPORT,))
    monkeypatch.setattr(archive, "FINAL_STAGES", STAGES)

    def validate_contract(value: Path | Mapping[str, Any]) -> dict[str, Any]:
        payload = read_json(value) if isinstance(value, Path) else dict(value)
        copy = dict(payload)
        recorded = copy.pop("contract_id")
        if recorded != canonical_json_sha256(copy):
            raise RuntimeError("Campaign contract ID is invalid")
        return payload

    def validate_export(root: Path, **_: Any) -> FinalExportIdentity:
        if (root / "export_marker.txt").read_text(encoding="ascii") != "export\n":
            raise RuntimeError("Export changed")
        return _identity(root)

    def validate_cohort(root: Path, **_: Any) -> dict[str, Any]:
        payload = read_json(root / "final_timing_cohort_manifest.json")
        if payload.get("status") != "passed":
            raise RuntimeError("Cohort changed")
        return payload

    def validate_bundle(
        root: Path,
        *,
        expected_export: FinalExportIdentity | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        payload = read_json(root / "final_bundle_manifest.json")
        _validate_payload_seal(payload, "manifest_payload_sha256")
        if expected_export is not None and payload["export_id"] != expected_export.export_id:
            raise RuntimeError("Bundle/export mismatch")
        result = dict(payload)
        result["_manifest_sha256"] = sha256_file(
            root / "final_bundle_manifest.json"
        )
        return result

    def validate_provenance(
        path: Path,
        *,
        bundle_dir: Path | None = None,
        expected_export: FinalExportIdentity | None = None,
        artifact_root: Path | None = None,
        host_source_root: Path | None = None,
    ) -> dict[str, Any]:
        payload = read_json(path)
        _validate_payload_seal(payload, "provenance_id")
        if payload.get("schema") != PROVENANCE_SCHEMA:
            raise RuntimeError("Unsupported build/upload provenance schema")
        if "artifacts" not in payload:
            for field in ["binary", "verbose_compile_log", "verbose_upload_log"]:
                item = payload[field]
                member = Path(item["path_recorded"])
                if (
                    member.stat().st_size != item["size_bytes"]
                    or sha256_file(member) != item["sha256"]
                ):
                    raise RuntimeError("Legacy provenance artifact changed")
            return payload
        root = (artifact_root or path.parent).resolve()
        listed = payload["artifacts"]
        actual = {
            member.relative_to(root).as_posix()
            for member in root.rglob("*")
            if member.is_file() and member.resolve() != path.resolve()
        }
        if actual != {item["path"] for item in listed}:
            raise RuntimeError("Build provenance contains extra or missing artifacts")
        for item in listed:
            member = root / item["path"]
            if (
                member.stat().st_size != item["size_bytes"]
                or sha256_file(member) != item["sha256"]
            ):
                raise RuntimeError("Build provenance artifact changed")
        if host_source_root is not None:
            archive.validate_host_environment(
                payload["host_environment"], source_root=host_source_root
            )
        if bundle_dir is not None:
            bundle = validate_bundle(bundle_dir)
            if payload["bundle_id"] != bundle["bundle_id"]:
                raise RuntimeError("Build provenance differs from bundle")
        return payload

    def validate_session(path: Path, **_: Any) -> dict[str, Any]:
        payload = read_json(path)
        _validate_payload_seal(payload, "session_evidence_id")
        return payload

    def verify_attempt(path: Path, **_: Any) -> dict[str, Any]:
        payload = read_json(path / "final_attempt.json")
        if payload["attempt_payload_sha256"] != _attempt_payload_hash(payload):
            raise RuntimeError("Attempt seal changed")
        return payload

    def stage_dataset(
        *, export: FinalExportIdentity, cohort_dir: Path, stage_name: str, **_: Any
    ) -> SimpleNamespace:
        stage = next(item for item in STAGES if item["name"] == stage_name)
        return SimpleNamespace(
            stage=dict(stage),
            input_binding={
                "base_replay_path_recorded": str(
                    Path(export.root) / "hil_replay_vectors.csv"
                ),
                "base_reference_path_recorded": str(
                    Path(export.root) / "hil_reference_predictions.csv"
                ),
                "base_replay_sha256": sha256_file(
                    Path(export.root) / "hil_replay_vectors.csv"
                ),
                "base_reference_sha256": sha256_file(
                    Path(export.root) / "hil_reference_predictions.csv"
                ),
                "cohort_manifest_sha256": sha256_file(
                    cohort_dir / "final_timing_cohort_manifest.json"
                ),
            },
            reference_rows=[
                {
                    "row_id": 0,
                    "fixed_pred": 3,
                    **{f"fixed_logit_{index}": int(index == 3) for index in range(5)},
                }
            ],
        )

    def verify_responses(
        *, responses: list[dict[str, Any]], reference_rows: list[dict[str, Any]], **_: Any
    ) -> dict[str, Any]:
        if len(responses) != 1 or int(responses[0]["row_id"]) != 0:
            raise RuntimeError("Response sequence differs")
        reference = reference_rows[0]
        if int(responses[0]["predicted_class"]) != reference["fixed_pred"]:
            raise RuntimeError("Response prediction differs")
        for index, logit in enumerate(responses[0]["logits"]):
            if int(logit) != reference[f"fixed_logit_{index}"]:
                raise RuntimeError("Response logits differ")
        return {
            "status": "passed",
            "sequence_exact": True,
            "predictions_exact": True,
            "logits_exact": True,
            "completed_rows": 1,
            "wifi_retry_reconciliation": {"status": "passed"},
        }

    def stage_ledger(stages: list[dict[str, Any]]) -> dict[str, int]:
        if [item["name"] for item in stages] != [item["name"] for item in STAGES]:
            raise RuntimeError("Stage sequence changed")
        return {
            "all_rows": 6,
            "timing_rows": 3,
            "full_rows": 1,
            "warmup_rows": 1,
            "smoke_rows": 1,
        }

    def campaign_ledger(
        contract: Mapping[str, Any], sessions: list[Mapping[str, Any]]
    ) -> dict[str, int]:
        if [item["combination_id"] for item in sessions] != [COMBINATION_ID]:
            raise RuntimeError("Campaign matrix is incomplete")
        return {
            "session_count": 1,
            "all_rows": 6,
            "balanced_timing_rows": 3,
            "full_exact_replay_rows": 1,
            "warmup_rows": 1,
            "smoke_rows": 1,
        }

    monkeypatch.setattr(archive, "validate_campaign_contract", validate_contract)
    monkeypatch.setattr(archive, "validate_final_export", validate_export)
    monkeypatch.setattr(archive, "validate_balanced_cohort", validate_cohort)
    monkeypatch.setattr(archive, "verify_final_bundle", validate_bundle)
    monkeypatch.setattr(
        archive, "validate_build_upload_provenance", validate_provenance
    )
    monkeypatch.setattr(archive, "validate_session_completion", validate_session)
    monkeypatch.setattr(archive, "verify_stage_attempt", verify_attempt)
    monkeypatch.setattr(archive, "load_stage_dataset", stage_dataset)
    monkeypatch.setattr(archive, "verify_response_records", verify_responses)
    monkeypatch.setattr(archive, "validate_session_stage_ledger", stage_ledger)
    monkeypatch.setattr(archive, "validate_campaign_session_ledger", campaign_ledger)
    monkeypatch.setattr(
        archive,
        "verify_complete_campaign",
        lambda **kwargs: (
            read_json(kwargs["campaign_contract"].parent / "evidence.json")
            if kwargs.get("session_contexts")
            and set(kwargs["session_contexts"]) == {COMBINATION_ID}
            and set(kwargs["session_contexts"][COMBINATION_ID])
            == {
                "export_dir",
                "cohort_dir",
                "bundle_dir",
                "provenance_json",
                "connection_json",
                "attempt_dirs",
            }
            and set(kwargs["session_contexts"][COMBINATION_ID]["attempt_dirs"])
            == {stage["name"] for stage in STAGES}
            else (_ for _ in ()).throw(
                AssertionError("archive campaign recomputation lacks explicit sources")
            )
        ),
    )


def _validate_payload_seal(payload: Mapping[str, Any], field: str) -> None:
    copy = dict(payload)
    recorded = copy.pop(field, None)
    if recorded != canonical_json_sha256(copy):
        raise RuntimeError(f"{field} is invalid")


def _make_source_campaign(root: Path) -> tuple[Path, Path, Path]:
    export = root / "export"
    export.mkdir(parents=True)
    (export / "export_marker.txt").write_text("export\n", encoding="ascii")
    (export / "hil_replay_vectors.csv").write_text("row_id\n0\n", encoding="ascii")
    (export / "hil_reference_predictions.csv").write_text(
        "row_id,fixed_pred\n0,3\n", encoding="ascii"
    )
    (export / "final_export_manifest.json").write_text("{}\n", encoding="ascii")
    identity = _identity(export)

    cohort = root / "cohort"
    cohort.mkdir()
    cohort_manifest = {"status": "passed", "model_key": MODEL_KEY}
    _write_json(cohort / "final_timing_cohort_manifest.json", cohort_manifest)
    (cohort / "timing_rows.csv").write_text("row_id\n0\n", encoding="ascii")

    bundle = root / "bundle"
    bundle.mkdir()
    runtime_identity_line = runtime_identity(
        identity,
        bundle_id=HASHES["bundle"],
        board=BOARD,
        transport=TRANSPORT,
        build_contract_id=HASHES["build"],
    )
    bundle_payload = {
        "status": "passed",
        "bundle_id": HASHES["bundle"],
        "build_contract_id": HASHES["build"],
        "board": BOARD,
        "transport": TRANSPORT,
        "student": "A",
        "route": "rf_kd",
        "export_id": identity.export_id,
        "model_sha256": identity.trained_state_sha256,
        "checkpoint_file_sha256": identity.checkpoint_file_sha256,
        "runtime_identity_response": runtime_identity_line,
        "bundle_identity_payload": {
            "bundle_builder_sha256": sha256_file(
                REPO_ROOT / "deployment" / "final_hil" / "bundles.py"
            )
        },
    }
    bundle_payload = _seal(bundle_payload, "manifest_payload_sha256")
    _write_json(bundle / "final_bundle_manifest.json", bundle_payload)
    (bundle / "firmware.ino").write_text("void setup() {}\n", encoding="ascii")
    bundle_manifest_sha = sha256_file(bundle / "final_bundle_manifest.json")

    provenance = root / "provenance"
    for relative, contents in {
        "build_artifacts/firmware.bin": b"binary",
        "logs/compile.log": b"verbose compile\n",
        "logs/upload.log": b"verbose upload\n",
    }.items():
        path = provenance / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    artifacts = [
        {
            "path": path.relative_to(provenance).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(provenance.rglob("*"))
        if path.is_file()
    ]
    base_time = datetime(2026, 8, 13, tzinfo=timezone.utc)
    environment = collect_host_environment(REPO_ROOT)
    provenance_payload = {
        "schema": PROVENANCE_SCHEMA,
        "status": "passed",
        "bundle_path_recorded": str(bundle.resolve()),
        "bundle_manifest_sha256": bundle_manifest_sha,
        "bundle_id": HASHES["bundle"],
        "build_contract_id": HASHES["build"],
        "board": BOARD,
        "transport": TRANSPORT,
        "student": "A",
        "route": "rf_kd",
        "export_id": identity.export_id,
        "model_sha256": identity.trained_state_sha256,
        "checkpoint_file_sha256": identity.checkpoint_file_sha256,
        "physical_port_serial": "SERIAL-ESP32",
        "started_utc": base_time.isoformat(),
        "finished_utc": (base_time + timedelta(minutes=1)).isoformat(),
        "post_reset_runtime_identity": runtime_identity_line,
        "host_environment": environment,
        "secure_attestation": False,
        "artifact_count_excluding_manifest": len(artifacts),
        "artifacts": artifacts,
    }
    provenance_payload = _seal(provenance_payload, "provenance_id")
    provenance_json = provenance / "build_upload_provenance.json"
    _write_json(provenance_json, provenance_payload)

    session_id = "A" * 32
    connection_payload = {
        "schema": CONNECTION_SCHEMA,
        "status": "connected",
        "protocol_id": FINAL_WIFI_PROTOCOL,
        "bundle_id": HASHES["bundle"],
        "build_contract_id": HASHES["build"],
        "runtime_identity": runtime_identity_line,
        "board": BOARD,
        "student": "A",
        "route": "rf_kd",
        "transport": TRANSPORT,
        "session_id": "B" * 32,
        "device_ip": "192.0.2.20",
        "device_udp_port": 42101,
        "host_udp_port": 42102,
        "provisioning_port": "COM_TEST",
        "physical_port_serial": "SERIAL-ESP32",
        "serial_open_policy": {"dtr": False, "rts": False},
        "wifi_mac_reported": "AA:BB:CC:DD:EE:FF",
        "rssi_dbm_at_connection": -50,
        "connectivity_firmware_reported": "test-firmware",
        "started_utc": (base_time + timedelta(minutes=1, seconds=5)).isoformat(),
        "finished_utc": (base_time + timedelta(minutes=1, seconds=30)).isoformat(),
        "credentials_recorded": False,
        "host_environment": environment,
        "security_boundary": (
            "Session and endpoint binding provide experiment correlation, not "
            "cryptographic authentication or payload confidentiality."
        ),
    }
    connection_payload["connection_payload_sha256"] = canonical_json_sha256(
        connection_payload
    )
    connection_path = root / "connection.json"
    _write_json(connection_path, connection_payload)
    connection_record_sha256 = sha256_file(connection_path)
    stage_ledger = []
    verification = {
        "status": "passed",
        "sequence_exact": True,
        "predictions_exact": True,
        "logits_exact": True,
        "completed_rows": 1,
        "wifi_retry_reconciliation": {"status": "passed"},
    }
    for index, stage in enumerate(STAGES):
        attempt_dir = root / "attempts" / f"{stage['ordinal']:02d}_{stage['name']}"
        attempt_dir.mkdir(parents=True)
        response = {
            "row_id": 0,
            "status": "OK",
            "predicted_class": 3,
            **{f"fixed_logit_{item}": int(item == 3) for item in range(5)},
            "preprocess_us": 1,
            "inference_us": 2,
            "total_us": 3,
            "host_observed_rtt_us": 4,
            "transaction_elapsed_us": 5,
            "attempts": 1,
            "response_timeout_count": 0,
            "ignored_datagram_count": 0,
        }
        for field in FINAL_RESPONSE_FIELDS:
            response.setdefault(field, 0)
        with (attempt_dir / "responses.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=FINAL_RESPONSE_FIELDS)
            writer.writeheader()
            writer.writerow(response)
        started_time = base_time + timedelta(minutes=2 + index * 2)
        attempt_id = f"{index + 1:032X}"
        dataset = {
            "base_replay_path_recorded": str(
                (export / "hil_replay_vectors.csv").resolve()
            ),
            "base_reference_path_recorded": str(
                (export / "hil_reference_predictions.csv").resolve()
            ),
            "base_replay_sha256": sha256_file(export / "hil_replay_vectors.csv"),
            "base_reference_sha256": sha256_file(
                export / "hil_reference_predictions.csv"
            ),
            "cohort_manifest_sha256": sha256_file(
                cohort / "final_timing_cohort_manifest.json"
            ),
        }
        started_payload = {
            "schema": ATTEMPT_SCHEMA,
            "status": "running",
            "attempt_id": attempt_id,
            "campaign_session_id": session_id,
            "bundle_id": HASHES["bundle"],
            "build_contract_id": HASHES["build"],
            "combination": {
                "student": "A",
                "route": "rf_kd",
                "board": BOARD,
                "transport": TRANSPORT,
            },
            "stage": dict(stage),
            "input_binding": dataset,
            "started_utc": started_time.isoformat(),
            "physical_identity": {
                "physical_port_serial": "SERIAL-ESP32",
                "wifi_mac_reported": "AA:BB:CC:DD:EE:FF",
                "wifi_network_session_id": "B" * 32,
            },
        }
        started_payload["attempt_started_sha256"] = _started_payload_hash(
            started_payload
        )
        _write_json(attempt_dir / "attempt_started.json", started_payload)
        final_payload = {
            **started_payload,
            "status": "passed",
            "finished_utc": (started_time + timedelta(minutes=1)).isoformat(),
            "runtime_identity": runtime_identity_line,
            "responses_file": "responses.csv",
            "completed_rows": 1,
            "responses_sha256": sha256_file(attempt_dir / "responses.csv"),
            "verification": verification,
            "controls": {
                "wifi_network_session_id": "B" * 32,
                "connection_path_recorded": str(connection_path.resolve()),
                "connection_record_sha256": connection_record_sha256,
                "connection_payload_sha256": connection_payload[
                    "connection_payload_sha256"
                ],
                "udp_timeout_seconds": 1.0,
                "udp_max_attempts": 3,
                "identity": {},
                "begin": {},
                "end": {},
            },
            "session_network_counters": {},
            "error": None,
            "host_environment": environment,
            "recovery": _recovery_contract("passed", TRANSPORT),
        }
        final_payload["attempt_payload_sha256"] = _attempt_payload_hash(
            final_payload
        )
        _write_json(attempt_dir / "final_attempt.json", final_payload)
        stage_ledger.append(
            {
                **dict(stage),
                "status": "passed",
                "attempt_id": attempt_id,
                "attempt_path_recorded": str(attempt_dir.resolve()),
                "attempt_manifest_sha256": sha256_file(
                    attempt_dir / "final_attempt.json"
                ),
                "started_utc": final_payload["started_utc"],
                "finished_utc": final_payload["finished_utc"],
            }
        )

    row_totals = {
        "all_rows": 6,
        "timing_rows": 3,
        "full_rows": 1,
        "warmup_rows": 1,
        "smoke_rows": 1,
    }
    session = {
        "schema": archive.SESSION_SCHEMA,
        "status": "passed",
        "campaign_session_id": session_id,
        "started_utc": stage_ledger[0]["started_utc"],
        "finished_utc": stage_ledger[-1]["finished_utc"],
        "combination_id": COMBINATION_ID,
        "model_key": MODEL_KEY,
        "student": "A",
        "route": "rf_kd",
        "board": BOARD,
        "transport": TRANSPORT,
        "export_id": identity.export_id,
        "model_sha256": identity.trained_state_sha256,
        "checkpoint_file_sha256": identity.checkpoint_file_sha256,
        "bundle_id": HASHES["bundle"],
        "build_contract_id": HASHES["build"],
        "export_path_recorded": str(export.resolve()),
        "export_manifest_sha256": identity.manifest_sha256,
        "cohort_path_recorded": str(cohort.resolve()),
        "cohort_manifest_sha256": sha256_file(
            cohort / "final_timing_cohort_manifest.json"
        ),
        "bundle_path_recorded": str(bundle.resolve()),
        "bundle_manifest_sha256": bundle_manifest_sha,
        "physical_port_serial": "SERIAL-ESP32",
        "wifi_mac_reported": "AA:BB:CC:DD:EE:FF",
        "provenance_id": provenance_payload["provenance_id"],
        "provenance_path_recorded": str(provenance_json.resolve()),
        "provenance_sha256": sha256_file(provenance_json),
        "stages": stage_ledger,
        "row_totals": row_totals,
        "warmup_excluded_from_reported_metrics": True,
        "smoke_excluded_from_reported_metrics": True,
        "timing_statistical_unit": TIMING_STATISTICAL_UNIT,
        "wifi_network_session_id": "B" * 32,
        "connection_path_recorded": str(connection_path.resolve()),
        "connection_record_sha256": connection_record_sha256,
        "connection_payload_sha256": connection_payload[
            "connection_payload_sha256"
        ],
        "udp_timeout_seconds": 1.0,
        "udp_max_attempts": 3,
    }
    session = _seal(session, "session_evidence_id")
    session_path = root / "session.json"
    _write_json(session_path, session)

    model = identity.to_dict()
    model.pop("root")
    model["root"] = str(export.resolve())
    model["status"] = "passed"
    contract = {
        "schema": "test_contract",
        "status": "ready",
        "transports": [TRANSPORT],
        "models": {MODEL_KEY: model},
        "combinations": [
            {
                "combination_id": COMBINATION_ID,
                "execution_ordinal": 1,
                "model_key": MODEL_KEY,
                "board": BOARD,
                "transport": TRANSPORT,
            }
        ],
        "eligible_combinations": [
            {
                "combination_id": COMBINATION_ID,
                "execution_ordinal": 1,
                "model_key": MODEL_KEY,
                "board": BOARD,
                "transport": TRANSPORT,
            }
        ],
        "excluded_combinations": [],
        "blocked_routes": [],
    }
    contract = _seal(contract, "contract_id")
    contract_path = root / "contract.json"
    _write_json(contract_path, contract)
    evidence = {
        "schema": archive.CAMPAIGN_EVIDENCE_SCHEMA,
        "status": "passed",
        "contract_id": contract["contract_id"],
        "sessions": [
            {
                "combination_id": COMBINATION_ID,
                "session_evidence_id": session["session_evidence_id"],
                "session_path_recorded": str(session_path.resolve()),
                "session_sha256": sha256_file(session_path),
            }
        ],
        "totals": {
            "session_count": 1,
            "all_rows": 6,
            "balanced_timing_rows": 3,
            "full_exact_replay_rows": 1,
            "warmup_rows": 1,
            "smoke_rows": 1,
        },
        "physical_specimens": {
            BOARD: {
                "physical_port_serial": "SERIAL-ESP32",
                "wifi_mac_reported": "AA:BB:CC:DD:EE:FF",
                "session_count": 1,
            }
        },
        "boards": [BOARD],
        "transports": [TRANSPORT],
        "all_four_models_retained": True,
        "all_gate_eligible_combinations_executed": True,
        "blocked_routes": [],
        "excluded_combinations": [],
        "board_build_contracts": {BOARD: HASHES["build"]},
    }
    evidence = _seal(evidence, "campaign_evidence_id")
    evidence_path = root / "evidence.json"
    _write_json(evidence_path, evidence)
    return contract_path, evidence_path, cohort


def _refresh_archive_envelope(root: Path) -> None:
    path = root / archive.ARCHIVE_MANIFEST
    payload = read_json(path)
    payload["inventory"] = archive._inventory(root)
    payload["file_count_excluding_manifest"] = len(payload["inventory"])
    payload["semantic_map"]["owned_files"] = sorted(
        item["path"] for item in payload["inventory"]
    )
    payload = _seal(payload, "archive_id")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _archive_source_arguments(source: Path) -> dict[str, Any]:
    return {
        "source_roots": {str(source.resolve()): source},
    }


def _create_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    _patch_compact_validators(monkeypatch)
    source = tmp_path / "source"
    contract, evidence, cohort = _make_source_campaign(source)
    contract_payload = read_json(contract)
    contract_payload["models"][MODEL_KEY]["root"] = "Z:/unavailable/original/export"
    contract_payload = _seal(contract_payload, "contract_id")
    _write_json(contract, contract_payload)
    evidence_payload = read_json(evidence)
    evidence_payload["contract_id"] = contract_payload["contract_id"]
    evidence_payload = _seal(evidence_payload, "campaign_evidence_id")
    _write_json(evidence, evidence_payload)
    destination = tmp_path / "archive"
    archive.create_campaign_archive(
        campaign_contract=contract,
        campaign_evidence=evidence,
        cohort_dir=cohort,
        export_dirs={MODEL_KEY: source / "export"},
        blocked_audits={},
        **_archive_source_arguments(source),
        output_dir=destination,
    )
    return source, destination


def test_archive_is_portable_complete_and_non_overwriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, destination = _create_archive(tmp_path, monkeypatch)
    manifest = read_json(destination / archive.ARCHIVE_MANIFEST)
    mapped = manifest["semantic_map"]["combinations"][COMBINATION_ID]
    assert len(mapped["attempts"]) == 6
    assert (destination / mapped["connection"]["local_path"]).is_file()
    provenance_root = destination / mapped["provenance"]["local_dir"]
    assert (provenance_root / "build_artifacts" / "firmware.bin").is_file()
    assert (provenance_root / "logs" / "compile.log").is_file()
    assert (provenance_root / "logs" / "upload.log").is_file()
    shutil.rmtree(source)
    monkeypatch.setattr(archive, "REPO_ROOT", tmp_path / "missing_repository")
    compact_cohort_validator = archive.validate_balanced_cohort

    def validate_portable_cohort(root: Path, **kwargs: Any) -> dict[str, Any]:
        assert kwargs.get("reconstruct_sources") is False
        return compact_cohort_validator(root, **kwargs)

    monkeypatch.setattr(
        archive, "validate_balanced_cohort", validate_portable_cohort
    )
    result = archive.verify_campaign_archive(destination)
    assert result["status"] == "passed"
    assert result["portable"] is True
    assert result["original_source_paths_dereferenced"] is False
    assert result["archive_local_source_snapshots_verified"] is True
    assert (
        result["verifier_execution_mode"]
        == "current_checkout_with_archived_source_ledger_verified"
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        archive.create_campaign_archive(
            campaign_contract=destination / "campaign" / "campaign_contract.json",
            campaign_evidence=destination / "campaign" / "campaign_evidence.json",
            cohort_dir=destination / "cohort",
            export_dirs={MODEL_KEY: destination / "models" / MODEL_KEY / "export"},
            blocked_audits={},
            source_roots={},
            output_dir=destination,
        )


def test_archive_creation_relocates_all_sealed_session_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_compact_validators(monkeypatch)
    original = tmp_path / "original_source"
    contract, evidence, cohort = _make_source_campaign(original)
    relocated = tmp_path / "relocated_source"
    shutil.copytree(original, relocated)
    shutil.rmtree(original)

    archive.create_campaign_archive(
        campaign_contract=relocated / contract.relative_to(original),
        campaign_evidence=relocated / evidence.relative_to(original),
        cohort_dir=relocated / cohort.relative_to(original),
        export_dirs={MODEL_KEY: relocated / "export"},
        blocked_audits={},
        source_roots={str(original.resolve()): relocated},
        output_dir=tmp_path / "archive_relocated",
    )
    assert archive.verify_campaign_archive(tmp_path / "archive_relocated")[
        "portable"
    ] is True


def test_archive_creation_rejects_incomplete_relocation_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_compact_validators(monkeypatch)
    original = tmp_path / "original_source"
    contract, evidence, cohort = _make_source_campaign(original)
    relocated = tmp_path / "relocated_source"
    shutil.copytree(original, relocated)
    shutil.rmtree(original)

    with pytest.raises(RuntimeError, match="Historical path resolves through 0"):
        archive.create_campaign_archive(
            campaign_contract=relocated / contract.relative_to(original),
            campaign_evidence=relocated / evidence.relative_to(original),
            cohort_dir=relocated / cohort.relative_to(original),
            export_dirs={MODEL_KEY: relocated / "export"},
            blocked_audits={},
            source_roots={},
            output_dir=tmp_path / "archive_missing_mapping",
        )


def test_relocation_mapping_cannot_escape_its_local_root(tmp_path: Path) -> None:
    local_root = tmp_path / "local"
    local_root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="escapes its local root"):
        archive._relocate_recorded_path(
            "/recorded/root/../outside.json",
            source_roots={"/recorded/root": local_root},
            expected_kind="file",
        )


def test_relocation_mapping_rejects_normalized_duplicate_roots(
    tmp_path: Path,
) -> None:
    local_root = tmp_path / "local"
    local_root.mkdir()
    evidence = local_root / "evidence.json"
    evidence.write_text("{}\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="through 2 source-root mappings"):
        archive._relocate_recorded_path(
            "C:/recorded/root/evidence.json",
            source_roots={
                "C:/recorded/root": local_root,
                "C:\\recorded\\root\\": local_root,
            },
            expected_kind="file",
        )


def test_archive_retains_blocked_route_without_executing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_compact_validators(monkeypatch)
    blocked_key = "student_B_rf_kd"
    monkeypatch.setattr(archive, "MODEL_KEYS", (MODEL_KEY, blocked_key))
    source = tmp_path / "source"
    contract_path, evidence_path, cohort = _make_source_campaign(source)
    blocked_path = source / "blocked_student_B.json"
    blocked_payload = {
        "model_key": blocked_key,
        "reason": "fixed-point quality gate failed",
        "blocked_audit_sha256": "c" * 64,
    }
    _write_json(blocked_path, blocked_payload)
    monkeypatch.setattr(
        archive,
        "validate_blocked_audit",
        lambda path, **_: read_json(path),
    )

    blocked_combination = {
        "combination_id": f"{blocked_key}__{BOARD}__{TRANSPORT}",
        "execution_ordinal": 2,
        "model_key": blocked_key,
        "board": BOARD,
        "transport": TRANSPORT,
        "model_status": "blocked",
        "reason": "fixed-point quality gate failed",
    }
    contract = read_json(contract_path)
    contract["status"] = "ready_with_blocked_routes"
    contract["models"][blocked_key] = {
        "model_key": blocked_key,
        "status": "blocked",
        "reason": "fixed-point quality gate failed",
        "blocked_audit_path_recorded": "Z:/unavailable/original/blocked.json",
        "blocked_audit_sha256": blocked_payload["blocked_audit_sha256"],
    }
    contract["combinations"].append(
        {
            key: value
            for key, value in blocked_combination.items()
            if key not in {"model_status", "reason"}
        }
    )
    contract["excluded_combinations"] = [blocked_combination]
    contract["blocked_routes"] = [
        {"model_key": blocked_key, "reason": "fixed-point quality gate failed"}
    ]
    contract = _seal(contract, "contract_id")
    _write_json(contract_path, contract)

    evidence = read_json(evidence_path)
    evidence["status"] = "passed_with_blocked_routes"
    evidence["contract_id"] = contract["contract_id"]
    evidence["blocked_routes"] = contract["blocked_routes"]
    evidence["excluded_combinations"] = contract["excluded_combinations"]
    evidence = _seal(evidence, "campaign_evidence_id")
    _write_json(evidence_path, evidence)

    destination = tmp_path / "archive"
    archive.create_campaign_archive(
        campaign_contract=contract_path,
        campaign_evidence=evidence_path,
        cohort_dir=cohort,
        export_dirs={MODEL_KEY: source / "export"},
        blocked_audits={blocked_key: blocked_path},
        **_archive_source_arguments(source),
        output_dir=destination,
    )
    result = archive.verify_campaign_archive(destination)
    manifest = read_json(destination / archive.ARCHIVE_MANIFEST)
    assert result["status"] == "passed_with_blocked_routes"
    assert set(manifest["semantic_map"]["combinations"]) == {COMBINATION_ID}
    assert manifest["semantic_map"]["models"][blocked_key]["status"] == "blocked"


def test_archive_rejects_legacy_binary_log_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_compact_validators(monkeypatch)
    source = tmp_path / "source"
    contract_path, evidence_path, cohort = _make_source_campaign(source)
    provenance_path = source / "provenance" / "build_upload_provenance.json"
    current = read_json(provenance_path)
    legacy = {
        key: value
        for key, value in current.items()
        if key not in {"artifact_count_excluding_manifest", "artifacts", "provenance_id"}
    }
    legacy["schema"] = "test_legacy_build_upload_provenance_v1"
    legacy["bundle_manifest_path_recorded"] = str(
        (source / "bundle" / "final_bundle_manifest.json").resolve()
    )
    legacy.pop("bundle_path_recorded", None)
    for field, relative, verbose in [
        ("binary", "build_artifacts/firmware.bin", None),
        ("verbose_compile_log", "logs/compile.log", True),
        ("verbose_upload_log", "logs/upload.log", True),
    ]:
        path = source / "provenance" / relative
        legacy[field] = {
            "path_recorded": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if verbose is not None:
            legacy[field]["verbose"] = verbose
    legacy = _seal(legacy, "provenance_id")
    _write_json(provenance_path, legacy)

    session_path = source / "session.json"
    session = read_json(session_path)
    session["provenance_id"] = legacy["provenance_id"]
    session["provenance_sha256"] = sha256_file(provenance_path)
    session = _seal(session, "session_evidence_id")
    _write_json(session_path, session)
    evidence = read_json(evidence_path)
    evidence["sessions"][0]["session_evidence_id"] = session[
        "session_evidence_id"
    ]
    evidence["sessions"][0]["session_sha256"] = sha256_file(session_path)
    evidence = _seal(evidence, "campaign_evidence_id")
    _write_json(evidence_path, evidence)

    with pytest.raises(RuntimeError):
        archive.create_campaign_archive(
            campaign_contract=contract_path,
            campaign_evidence=evidence_path,
            cohort_dir=cohort,
            export_dirs={MODEL_KEY: source / "export"},
            blocked_audits={},
            **_archive_source_arguments(source),
            output_dir=tmp_path / "archive",
        )


@pytest.mark.parametrize(
    "target_kind",
    ["responses", "binary", "compile_log", "upload_log", "connection", "session"],
)
def test_archive_rejects_tampered_evidence_after_outer_inventory_is_resealed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
) -> None:
    _, destination = _create_archive(tmp_path, monkeypatch)
    manifest = read_json(destination / archive.ARCHIVE_MANIFEST)
    mapped = manifest["semantic_map"]["combinations"][COMBINATION_ID]
    provenance_root = destination / mapped["provenance"]["local_dir"]
    targets = {
        "responses": destination / mapped["attempts"][0]["local_dir"] / "responses.csv",
        "binary": provenance_root / "build_artifacts" / "firmware.bin",
        "compile_log": provenance_root / "logs" / "compile.log",
        "upload_log": provenance_root / "logs" / "upload.log",
        "connection": destination / mapped["connection"]["local_path"],
        "session": destination / mapped["session_local_path"],
    }
    if target_kind == "session":
        session = read_json(targets[target_kind])
        session["physical_port_serial"] = "SUBSTITUTED"
        targets[target_kind].write_text(
            json.dumps(session, indent=2) + "\n", encoding="utf-8"
        )
    else:
        with targets[target_kind].open("ab") as handle:
            handle.write(b"tamper")
    _refresh_archive_envelope(destination)
    with pytest.raises(RuntimeError):
        archive.verify_campaign_archive(destination)


def test_archive_rejects_inventory_and_semantically_unowned_file_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, destination = _create_archive(tmp_path, monkeypatch)
    manifest_path = destination / archive.ARCHIVE_MANIFEST
    payload = read_json(manifest_path)
    payload["inventory"][0]["sha256"] = "f" * 64
    payload = _seal(payload, "archive_id")
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="inventory mismatch"):
        archive.verify_campaign_archive(destination)

    payload = read_json(manifest_path)
    payload["inventory"][0]["sha256"] = sha256_file(
        destination / payload["inventory"][0]["path"]
    )
    extra = destination / "unowned.txt"
    extra.write_text("extra\n", encoding="ascii")
    _refresh_archive_envelope(destination)
    with pytest.raises(RuntimeError, match="semantic ownership"):
        archive.verify_campaign_archive(destination)
