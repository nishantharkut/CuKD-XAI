from __future__ import annotations

import csv
import json
import shutil
import threading
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deployment.wireless_hil.host.wireless_common import (
    IDENTITY_TRANSACTION_ID,
    MAX_ATTEMPTS,
    MAX_DATAGRAM_BYTES,
    RESPONSE_ENVELOPE_PREFIX,
    encode_wireless_envelope,
)
from deployment.wireless_hil.host.udp_session import IGNORED_DATAGRAM_CATEGORIES

from deployment.final_hil.bundles import (
    RUNTIME_IDENTITY_PREFIX,
    parse_runtime_identity,
    prepare_final_bundle,
    runtime_identity,
    validate_build_contract,
    verify_final_bundle,
)
from deployment.final_hil import bundles as final_bundles
from deployment.final_hil import (
    campaign as final_campaign,
    contracts,
    evidence as final_evidence,
    runtime as final_runtime,
)
from deployment.final_hil.contracts import (
    FINAL_STAGES,
    MODEL_KEYS,
    build_campaign_contract,
    canonical_json_sha256,
    generate_balanced_cohort,
    sha256_file,
    validate_balanced_cohort,
    validate_campaign_contract,
    validate_cohort_selection,
    validate_final_export,
)
from deployment.final_hil.evidence import (
    TIMING_STATISTICAL_UNIT,
    _validate_upload_artifact_delta,
    preflight_campaign,
    record_build_upload_provenance,
    validate_build_upload_provenance,
    validate_campaign_session_ledger,
    validate_common_board_build_contracts,
    validate_session_completion,
    validate_session_stage_ledger,
)
from deployment.final_hil.runtime import (
    _open_serial_with_board_policy,
    collect_host_environment,
    validate_host_environment,
    validate_wifi_counter_reconciliation,
    verify_response_records,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_COMMON = REPO_ROOT / "deployment" / "hardware_hil" / "firmware" / "common"
HASHES = {
    "checkpoint": "1" * 64,
    "trained": "2" * 64,
    "dataset": "3" * 64,
    "split": "4" * 64,
    "scaler": "5" * 64,
}


@pytest.mark.parametrize(
    ("board", "expected_dtr", "expected_rts"),
    [
        ("esp32c3", False, False),
        ("arduino_r4", True, False),
    ],
)
def test_serial_open_uses_explicit_board_control_policy(
    board: str, expected_dtr: bool, expected_rts: bool
) -> None:
    class FakeDevice:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.port = None
            self.dtr = None
            self.rts = None
            self.opened_with = None

        def open(self) -> None:
            self.opened_with = (self.port, self.dtr, self.rts)

    created: list[FakeDevice] = []

    def construct(**kwargs: object) -> FakeDevice:
        device = FakeDevice(**kwargs)
        created.append(device)
        return device

    serial_module = SimpleNamespace(Serial=construct)

    observed = _open_serial_with_board_policy(
        serial_module,
        board=board,
        port="/dev/example",
        baud=115200,
        timeout_seconds=2.0,
    )

    assert observed is created[0]
    assert created[0].opened_with == ("/dev/example", expected_dtr, expected_rts)
    assert created[0].kwargs == {
        "port": None,
        "baudrate": 115200,
        "timeout": 2.0,
        "write_timeout": 2.0,
        "xonxoff": False,
        "rtscts": False,
        "dsrdtr": False,
    }


def test_serial_open_rejects_unknown_board_before_open() -> None:
    serial_module = SimpleNamespace(
        Serial=lambda **kwargs: pytest.fail("serial device must not be constructed")
    )
    with pytest.raises(ValueError, match="Unsupported serial-control board"):
        _open_serial_with_board_policy(
            serial_module,
            board="unknown",
            port="/dev/example",
            baud=115200,
            timeout_seconds=2.0,
        )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _fake_verifier(root: Path) -> dict:
    identity = json.loads((root / "final_export_identity.json").read_text())
    return {
        "status": "passed",
        "export_id": identity["export_id"],
        "student": identity["student"],
        "route": identity["route"],
        "test_rows": 56_301,
    }


def _fake_blocked_verifier(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    identity = payload["identity"]
    return {
        "status": "blocked_verified",
        "blocked_audit_id": identity["blocked_audit_id"],
        "student": identity["student"],
        "route": identity["route"],
        "fixed_vs_fp32_agreement": 0.98,
        "absolute_macro_f1_drop": 0.02,
    }


def test_canonical_final_verifier_preserves_sealed_header_newlines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export_root = tmp_path / "sealed_export"
    export_root.mkdir()
    expected = b"#define FIRST 1\r\n#define SECOND 2\r\n"
    for name in [
        "model_weights.h",
        "preprocess_metadata.h",
        "preprocess_int_metadata.h",
    ]:
        (export_root / name).write_bytes(expected)
    native_control = tmp_path / "native_control.txt"
    native_control.write_text(
        "#define FIRST 1\n#define SECOND 2\n", encoding="ascii"
    )
    native_bytes = native_control.read_bytes()

    from deployment.firmware_export.wsnds_final_hil import export_final_seed42

    def fake_verify_final_export(root: Path) -> dict:
        assert root == export_root
        unrelated_root = tmp_path / "cukd_lineage_headers_unrelated_thread"
        unrelated_root.mkdir()
        unrelated = unrelated_root / "model_weights.h"
        writer = threading.Thread(
            target=lambda: unrelated.write_text(
                "#define FIRST 1\n#define SECOND 2\n", encoding="ascii"
            )
        )
        writer.start()
        writer.join()
        assert unrelated.read_bytes() == native_bytes

        generated_root = tmp_path / "cukd_lineage_headers_regression"
        generated_root.mkdir()
        for name in [
            "model_weights.h",
            "preprocess_metadata.h",
            "preprocess_int_metadata.h",
        ]:
            generated = generated_root / name
            generated.write_text(
                "#define FIRST 1\n#define SECOND 2\n", encoding="ascii"
            )
            assert generated.read_bytes() == expected
        return {"status": "passed"}

    monkeypatch.setattr(
        export_final_seed42, "verify_final_export", fake_verify_final_export
    )

    verifier = final_evidence._memoized_export_verifier(None)
    assert verifier(export_root) == {"status": "passed"}
    assert verifier(export_root) == {"status": "passed"}


def test_campaign_resume_requires_an_ordered_session_prefix(tmp_path: Path) -> None:
    expected = ["first", "second", "third"]
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / "second.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="campaign-order prefix"):
        final_campaign._existing_session_prefix(tmp_path, sessions, expected)

    (sessions / "second.json").unlink()
    first = sessions / "first.json"
    first.write_text("{}\n", encoding="utf-8")
    assert final_campaign._existing_session_prefix(tmp_path, sessions, expected) == [first]


def test_campaign_credentials_are_private_and_consumed_once(tmp_path: Path) -> None:
    credentials = tmp_path / "wifi.json"
    credentials.write_text(
        json.dumps({"ssid": "test-network", "password": "test-password"}) + "\n",
        encoding="utf-8",
    )
    credentials.chmod(0o600)

    assert final_campaign._load_wifi_credentials(
        credentials, allowed_root=tmp_path
    ) == (
        "test-network",
        "test-password",
    )
    assert not credentials.exists()


def _make_export(
    root: Path,
    student: str,
    route: str,
    *,
    hashes: dict[str, str] | None = None,
    test_rows: int = 56_301,
) -> Path:
    bound_hashes = hashes or HASHES
    root.mkdir()
    (root / "model_weights.h").write_text(
        "#define CUKD_INPUT_DIM 17\n", encoding="ascii"
    )
    (root / "preprocess_int_metadata.h").write_text(
        "#define CUKD_PREPROCESS_INPUT_DIM 17\n", encoding="ascii"
    )
    source_snapshots = []
    for name in ["cukd_model.c", "cukd_model.h", "cukd_preprocess.c", "cukd_preprocess.h"]:
        source_snapshots.append(
            {
                "origin_relative_path": (
                    f"deployment/hardware_hil/firmware/common/{name}"
                ),
                "snapshot_path": f"source_snapshot/c/{name}",
                "size_bytes": (MODEL_COMMON / name).stat().st_size,
                "sha256": sha256_file(MODEL_COMMON / name),
            }
        )
    identity = {
        "protocol": "wsnds_feature_group_split_train_only_scaler_10seed_v2",
        "seed": 42,
        "student": student,
        "route": route,
        "checkpoint_file_sha256": bound_hashes["checkpoint"],
        "trained_state_sha256": bound_hashes["trained"],
        "dataset_sha256": bound_hashes["dataset"],
        "split_indices_sha256": bound_hashes["split"],
        "scaler_sha256": bound_hashes["scaler"],
        "source_snapshots": source_snapshots,
        "core_files": [
            {
                "path": name,
                "size_bytes": (root / name).stat().st_size,
                "sha256": sha256_file(root / name),
            }
            for name in ["model_weights.h", "preprocess_int_metadata.h"]
        ],
    }
    identity["export_id"] = canonical_json_sha256(identity)
    _write_json(root / "final_export_identity.json", identity)
    (root / "cukd_export_identity.h").write_text(
        "#ifndef CUKD_FINAL_EXPORT_IDENTITY_H\n"
        "#define CUKD_FINAL_EXPORT_IDENTITY_H\n"
        "#define CUKD_EXPORT_ID \"%s\"\n"
        "#define CUKD_PROTOCOL_ID \"%s\"\n"
        "#define CUKD_EXPORT_SEED 42\n"
        "#define CUKD_STUDENT_ID \"%s\"\n"
        "#define CUKD_ROUTE_ID \"%s\"\n"
        "#define CUKD_CHECKPOINT_FILE_SHA256 \"%s\"\n"
        "#define CUKD_TRAINED_STATE_SHA256 \"%s\"\n"
        "#endif\n"
        % (
            identity["export_id"],
            identity["protocol"],
            student,
            route,
            identity["checkpoint_file_sha256"],
            identity["trained_state_sha256"],
        ),
        encoding="ascii",
    )
    (root / "hil_replay_vectors.csv").write_text(
        "row_id,source_row_index," + ",".join(f"f{i}" for i in range(17)) + "\n"
        "0,123," + ",".join("0" for _ in range(17)) + "\n",
        encoding="ascii",
    )
    (root / "hil_reference_predictions.csv").write_text(
        "row_id,source_row_index,true_label,fixed_pred,fp32_pred,"
        + ",".join(f"fixed_logit_{i}" for i in range(5))
        + "\n0,123,3,3,3,0,0,0,1,0\n",
        encoding="ascii",
    )
    report = {
        "status": "passed",
        "identity": identity,
        "gates": {"test_rows": test_rows, "quality_gates_passed": True},
    }
    report["report_payload_sha256"] = canonical_json_sha256(report)
    _write_json(root / "final_export_report.json", report)
    files = [
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.iterdir())
        if path.is_file()
    ]
    manifest = {
        "status": "passed",
        "protocol_id": identity["protocol"],
        "seed": 42,
        "student": student,
        "route": route,
        "export_id": identity["export_id"],
        "files": files,
        "file_count_excluding_manifest": len(files),
    }
    manifest["manifest_payload_sha256"] = canonical_json_sha256(manifest)
    _write_json(root / "final_export_manifest.json", manifest)
    return root


def _export_identity(root: Path):
    return validate_final_export(root, verifier=_fake_verifier)


def _build_contract(board: str) -> dict:
    targets = {
        "esp32c3": ("esp32:esp32:esp32c3", "esp32:esp32"),
        "arduino_r4": (
            "arduino:renesas_uno:unor4wifi",
            "arduino:renesas_uno",
        ),
    }
    fqbn, platform_id = targets[board]
    return {
        "schema": "cukd_final_hil_build_contract_v1",
        "board": board,
        "fqbn": fqbn,
        "platform_id": platform_id,
        "board_core_version": "1.2.3",
        "frontend_version": "arduino-cli 1.5.1",
        "toolchain_version": "test-toolchain-1",
        "compile_command": [
            "arduino-cli",
            "compile",
            "--fqbn",
            "{fqbn}",
            "--verbose",
            "--output-dir",
            "{build_dir}",
            "{bundle}",
        ],
        "upload_command": [
            "arduino-cli",
            "upload",
            "--fqbn",
            "{fqbn}",
            "--port",
            "{port}",
            "--verbose",
            "--input-dir",
            "{build_dir}",
            "{bundle}",
        ],
    }


def _four_exports(tmp_path: Path) -> dict[str, Path]:
    result = {}
    for key in MODEL_KEYS:
        _, student, *route_parts = key.split("_")
        route = "_".join(route_parts)
        result[key] = _make_export(tmp_path / key, student, route)
    return result


def test_build_contract_rejects_a_second_timed_inference_pass() -> None:
    contract = _build_contract("esp32c3")
    validate_build_contract(contract)
    contract["compile_command"].append("-DCUKD_HIL_VERIFY_PREDICT_WRAPPER")
    with pytest.raises(RuntimeError, match="second-pass predict wrapper"):
        validate_build_contract(contract)


def test_build_contract_rejects_unapproved_compile_flags() -> None:
    contract = _build_contract("esp32c3")
    contract["compile_command"].insert(-1, "--build-property")
    contract["compile_command"].insert(-1, "compiler.c.extra_flags=-DUNAUDITED")
    with pytest.raises(RuntimeError, match="unapproved argument"):
        validate_build_contract(contract)


def test_host_source_ledger_rejects_dependency_substitution() -> None:
    environment = collect_host_environment(REPO_ROOT)
    validate_host_environment(environment, source_root=REPO_ROOT)
    tampered = json.loads(json.dumps(environment))
    tampered["source_dependencies"][0]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="dependency changed"):
        validate_host_environment(tampered, source_root=REPO_ROOT)


def test_one_build_contract_is_required_per_board() -> None:
    records = [
        {"board": board, "build_contract_id": character * 64}
        for board, character in [("esp32c3", "a"), ("arduino_r4", "b")]
        for _ in range(2)
    ]
    assert validate_common_board_build_contracts(records) == {
        "esp32c3": "a" * 64,
        "arduino_r4": "b" * 64,
    }
    records[1]["build_contract_id"] = "c" * 64
    with pytest.raises(RuntimeError, match="different build contracts"):
        validate_common_board_build_contracts(records)


def test_build_upload_provenance_executes_and_seals_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export_dir = _make_export(tmp_path / "export", "A", "rf_kd")
    bundle_dir = tmp_path / "bundle"
    prepare_final_bundle(
        export_dir=export_dir,
        output_dir=bundle_dir,
        board="esp32c3",
        transport="usb_serial",
        build_contract=_build_contract("esp32c3"),
        verifier=_fake_verifier,
    )
    export = _export_identity(export_dir)
    bundle = verify_final_bundle(bundle_dir, expected_export=export)

    def fake_runner(command: list[str], **_: object) -> SimpleNamespace:
        if len(command) > 1 and command[1] == "compile":
            build_dir = Path(command[command.index("--output-dir") + 1])
            (build_dir / "firmware.bin").write_bytes(b"compiled-firmware")
            sketch_build = Path(command[-1]) / "build" / "test.platform.board"
            sketch_build.mkdir(parents=True)
            (sketch_build / "intermediate.o").write_bytes(b"compiler-intermediate")
            (sketch_build / "boot_app0.bin").write_bytes(b"core-boot-image")
        elif len(command) > 1 and command[1] == "upload":
            build_dir = Path(command[command.index("--input-dir") + 1])
            (build_dir / "firmware_flashed.bin").write_bytes(
                (build_dir / "firmware.bin").read_bytes()
            )
            (build_dir / "boot_app0_flashed.bin").write_bytes(b"core-boot-image")
        if len(command) > 1 and command[1] == "version":
            output = json.dumps(
                {"Application": "arduino-cli", "VersionString": "1.5.1"}
            )
        elif len(command) > 2 and command[1:3] == ["core", "list"]:
            output = json.dumps(
                {
                    "platforms": [
                        {"id": "esp32:esp32", "installed_version": "1.2.3"}
                    ]
                }
            )
        elif len(command) > 2 and command[1:3] == ["board", "details"]:
            output = json.dumps(
                {"fqbn": "esp32:esp32:esp32c3", "version": "1.2.3"}
            )
        else:
            output = "test-toolchain-1 verbose command completed\n"
        return SimpleNamespace(returncode=0, stdout=output)

    monkeypatch.setattr(
        "deployment.final_hil.evidence.require_physical_port_serial",
        lambda port, expected: expected,
    )
    provenance = record_build_upload_provenance(
        export_dir=export_dir,
        bundle_dir=bundle_dir,
        physical_port="COM_TEST",
        physical_port_serial="SERIAL_TEST",
        output_dir=tmp_path / "provenance",
        runner=fake_runner,
        identity_query=lambda **_: bundle["runtime_identity_response"],
        verifier=_fake_verifier,
    )
    payload = validate_build_upload_provenance(
        provenance,
        bundle_dir=bundle_dir,
        expected_export=export,
    )
    assert payload["status"] == "passed"
    assert payload["commands"]["compile"]["returncode"] == 0
    assert payload["build_artifacts_before_upload"] != payload[
        "build_artifacts_after_upload"
    ]
    assert [item["path"] for item in payload["uploader_added_artifacts"]] == [
        "boot_app0_flashed.bin",
        "firmware_flashed.bin",
    ]
    assert [
        item["path"] for item in payload["bundle_tool_side_effect_artifacts"]
    ] == [
        "arduino_cli_compile_build_side_effect/test.platform.board/boot_app0.bin",
        "arduino_cli_compile_build_side_effect/test.platform.board/intermediate.o",
    ]
    assert not (bundle_dir / "build").exists()
    verify_final_bundle(bundle_dir, expected_export=export)

    old_bundle = payload["bundle_path_recorded"]
    old_build = payload["build_dir_path_recorded"]
    payload["bundle_path_recorded"] = "/home/project/final_hil/bundle"
    payload["build_dir_path_recorded"] = "/tmp/final_hil/build"
    for name in ["compile", "upload"]:
        payload["commands"][name]["command"] = [
            payload["bundle_path_recorded"]
            if token == old_bundle
            else payload["build_dir_path_recorded"]
            if token == old_build
            else token
            for token in payload["commands"][name]["command"]
        ]
    unsealed = dict(payload)
    unsealed.pop("provenance_id")
    payload["provenance_id"] = canonical_json_sha256(unsealed)
    _write_json(provenance, payload)
    validate_build_upload_provenance(
        provenance,
        bundle_dir=bundle_dir,
        expected_export=export,
    )

    valid_cross_host_payload = json.loads(json.dumps(payload))
    payload["commands"]["compile"]["command"][-1] = "unexpected-token"
    unsealed = dict(payload)
    unsealed.pop("provenance_id")
    payload["provenance_id"] = canonical_json_sha256(unsealed)
    _write_json(provenance, payload)
    with pytest.raises(RuntimeError, match="compile command binding differs"):
        validate_build_upload_provenance(
            provenance,
            bundle_dir=bundle_dir,
            expected_export=export,
        )

    _write_json(provenance, valid_cross_host_payload)
    (provenance.parent / "build_artifacts" / "firmware.bin").write_bytes(b"tamper")
    with pytest.raises(RuntimeError, match="artifact changed"):
        validate_build_upload_provenance(
            provenance,
            bundle_dir=bundle_dir,
            expected_export=export,
        )


def test_upload_artifact_delta_rejects_mutation_and_unapproved_additions() -> None:
    firmware = {
        "path": "firmware.bin",
        "size_bytes": 8,
        "sha256": "1" * 64,
    }
    flashed = {
        "path": "firmware_flashed.bin",
        "size_bytes": 8,
        "sha256": "1" * 64,
    }
    boot = {
        "path": "boot_app0_flashed.bin",
        "size_bytes": 4,
        "sha256": "2" * 64,
    }
    side_effect = {
        "path": "arduino_cli_compile_build_side_effect/test/boot_app0.bin",
        "size_bytes": 4,
        "sha256": "2" * 64,
    }
    assert _validate_upload_artifact_delta(
        [firmware],
        [firmware, boot, flashed],
        board="esp32c3",
        bundle_side_effects=[side_effect],
    ) == [boot, flashed]

    changed = {**firmware, "sha256": "3" * 64}
    with pytest.raises(RuntimeError, match="changed or removed"):
        _validate_upload_artifact_delta([firmware], [changed], board="esp32c3")
    with pytest.raises(RuntimeError, match="changed or removed"):
        _validate_upload_artifact_delta([firmware], [], board="esp32c3")
    with pytest.raises(RuntimeError, match="differs from its compile output"):
        _validate_upload_artifact_delta(
            [firmware], [firmware, {**flashed, "sha256": "4" * 64}], board="esp32c3"
        )
    with pytest.raises(RuntimeError, match="unexpected build artifact"):
        _validate_upload_artifact_delta(
            [firmware],
            [firmware, {"path": "extra.txt", "size_bytes": 1, "sha256": "5" * 64}],
            board="esp32c3",
        )
    with pytest.raises(RuntimeError, match="no matching source"):
        _validate_upload_artifact_delta(
            [firmware], [firmware, boot], board="arduino_r4"
        )
    with pytest.raises(RuntimeError, match="no matching source"):
        _validate_upload_artifact_delta(
            [firmware], [firmware, boot], board="esp32c3"
        )


def test_arduino_inspection_requires_exact_structured_identities(tmp_path: Path) -> None:
    contract = _build_contract("esp32c3")
    frontend = tmp_path / "frontend.json"
    cores = tmp_path / "cores.json"
    board = tmp_path / "board.json"
    _write_json(frontend, {"Application": "arduino-cli", "VersionString": "1.5.1"})
    _write_json(
        cores,
        {"platforms": [{"id": "esp32:esp32", "installed_version": "1.2.3"}]},
    )
    _write_json(board, {"fqbn": contract["fqbn"], "version": "1.2.3"})

    final_evidence._validate_arduino_inspection(
        contract=contract,
        frontend_log=frontend,
        cores_log=cores,
        board_log=board,
    )

    _write_json(board, {"fqbn": "esp32:esp32:wrong", "version": "1.2.3"})
    with pytest.raises(RuntimeError, match="board FQBN differs"):
        final_evidence._validate_arduino_inspection(
            contract=contract,
            frontend_log=frontend,
            cores_log=cores,
            board_log=board,
        )


def test_build_upload_precondition_failure_is_recorded_before_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export_dir = _make_export(tmp_path / "export", "A", "rf_kd")
    bundle_dir = tmp_path / "bundle"
    prepare_final_bundle(
        export_dir=export_dir,
        output_dir=bundle_dir,
        board="esp32c3",
        transport="usb_serial",
        build_contract=_build_contract("esp32c3"),
        verifier=_fake_verifier,
    )
    runner_calls: list[list[str]] = []

    def forbidden_runner(command: list[str], **_: object) -> SimpleNamespace:
        runner_calls.append(command)
        return SimpleNamespace(returncode=0, stdout="unexpected command\n")

    monkeypatch.setattr(
        "deployment.final_hil.evidence.require_physical_port_serial",
        lambda *_: (_ for _ in ()).throw(RuntimeError("physical serial mismatch")),
    )
    output_dir = tmp_path / "failed_provenance"
    with pytest.raises(RuntimeError, match="physical serial mismatch"):
        record_build_upload_provenance(
            export_dir=export_dir,
            bundle_dir=bundle_dir,
            physical_port="COM_TEST",
            physical_port_serial="SERIAL_TEST",
            output_dir=output_dir,
            runner=forbidden_runner,
            verifier=_fake_verifier,
        )
    failure = json.loads(
        (output_dir / "failed_build_upload.json").read_text(encoding="utf-8")
    )
    assert runner_calls == []
    assert failure["status"] == "failed"
    assert failure["commands"] == {}
    assert failure["physical_port_serial"] is None
    assert failure["physical_port_serial_expected"] == "SERIAL_TEST"


def test_build_upload_host_precondition_is_checked_before_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export_dir = _make_export(tmp_path / "export", "A", "rf_kd")
    bundle_dir = tmp_path / "bundle"
    prepare_final_bundle(
        export_dir=export_dir,
        output_dir=bundle_dir,
        board="esp32c3",
        transport="usb_serial",
        build_contract=_build_contract("esp32c3"),
        verifier=_fake_verifier,
    )
    runner_calls: list[list[str]] = []

    def forbidden_runner(command: list[str], **_: object) -> SimpleNamespace:
        runner_calls.append(command)
        return SimpleNamespace(returncode=0, stdout="unexpected command\n")

    monkeypatch.setattr(
        "deployment.final_hil.evidence.require_physical_port_serial",
        lambda _port, expected: expected,
    )
    monkeypatch.setattr(
        "deployment.final_hil.evidence.collect_host_environment",
        lambda: {"git_revision": None},
    )
    output_dir = tmp_path / "failed_host_provenance"
    with pytest.raises(RuntimeError, match="exact host Git revision"):
        record_build_upload_provenance(
            export_dir=export_dir,
            bundle_dir=bundle_dir,
            physical_port="COM_TEST",
            physical_port_serial="SERIAL_TEST",
            output_dir=output_dir,
            runner=forbidden_runner,
            verifier=_fake_verifier,
        )
    failure = json.loads(
        (output_dir / "failed_build_upload.json").read_text(encoding="utf-8")
    )
    assert runner_calls == []
    assert failure["status"] == "failed"
    assert failure["commands"] == {}
    assert failure["physical_port_serial"] == "SERIAL_TEST"


def test_final_export_identity_and_file_tamper_are_rejected(tmp_path: Path) -> None:
    export_dir = _make_export(tmp_path / "export", "A", "scratch")
    identity = validate_final_export(export_dir, verifier=_fake_verifier)
    assert identity.model_key == "student_A_scratch"
    assert identity.gate_status == "passed"
    assert identity.full_replay_sha256 == sha256_file(
        export_dir / "hil_replay_vectors.csv"
    )

    with (export_dir / "hil_replay_vectors.csv").open("a", encoding="ascii") as handle:
        handle.write("tamper\n")
    with pytest.raises(RuntimeError, match="changed"):
        validate_final_export(export_dir, verifier=_fake_verifier)


def test_final_export_rejects_cross_platform_path_aliases(tmp_path: Path) -> None:
    export_dir = _make_export(tmp_path / "export", "A", "scratch")
    manifest_path = export_dir / "final_export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    aliased = dict(manifest["files"][0])
    aliased["path"] = ".\\" + str(aliased["path"])
    manifest["files"].append(aliased)
    manifest["file_count_excluding_manifest"] += 1
    manifest.pop("manifest_payload_sha256")
    manifest["manifest_payload_sha256"] = canonical_json_sha256(manifest)
    _write_json(manifest_path, manifest)
    with pytest.raises(RuntimeError, match="Unsafe or duplicate final export path"):
        validate_final_export(export_dir, verifier=_fake_verifier)


@pytest.mark.parametrize("board", ["esp32c3", "arduino_r4"])
@pytest.mark.parametrize("transport", ["usb_serial", "wifi_udp"])
def test_final_bundle_embeds_complete_identity_without_changing_timer(
    tmp_path: Path, board: str, transport: str
) -> None:
    export_dir = _make_export(tmp_path / "export", "A", "rf_kd")
    expected_export = _export_identity(export_dir)
    output = tmp_path / f"bundle_{transport}"
    manifest_path = prepare_final_bundle(
        export_dir=export_dir,
        output_dir=output,
        board=board,
        transport=transport,
        build_contract=_build_contract(board),
        verifier=_fake_verifier,
    )
    with pytest.raises(TypeError, match="expected_export"):
        verify_final_bundle(output)  # type: ignore[call-arg]
    manifest = verify_final_bundle(output, expected_export=expected_export)
    parsed = parse_runtime_identity(manifest["runtime_identity_response"])
    assert parsed == {
        "prefix": RUNTIME_IDENTITY_PREFIX,
        "protocol": "wsnds_feature_group_split_train_only_scaler_10seed_v2",
        "seed": 42,
        "student": "A",
        "route": "rf_kd",
        "model_sha256": HASHES["trained"],
        "export_id": manifest["export_id"],
        "bundle_id": manifest["bundle_id"],
        "board": board,
        "transport": transport,
        "build_contract_id": manifest["build_contract_id"],
    }
    assert sha256_file(manifest_path) == manifest["_manifest_sha256"]
    if transport == "wifi_udp":
        envelope = encode_wireless_envelope(
            prefix=RESPONSE_ENVELOPE_PREFIX,
            session_id="F" * 32,
            stage_id="F" * 16,
            transaction_id=IDENTITY_TRANSACTION_ID,
            attempt=MAX_ATTEMPTS,
            inner_text=manifest["runtime_identity_response"],
        )
        assert len(envelope) <= MAX_DATAGRAM_BYTES
    sketch = (output / manifest["sketch_file"]).read_text(encoding="utf-8")
    assert sketch.index("argmax_logits(logits)") < sketch.index("inference_end =")

    bundle_manifest_path = output / "final_bundle_manifest.json"
    original_manifest = bundle_manifest_path.read_text(encoding="utf-8")
    tampered_manifest = json.loads(original_manifest)
    tampered_manifest.pop("manifest_payload_sha256")
    tampered_manifest["runtime_identity_response"] = tampered_manifest[
        "runtime_identity_response"
    ].replace(HASHES["trained"], "9" * 64)
    tampered_manifest["manifest_payload_sha256"] = canonical_json_sha256(
        tampered_manifest
    )
    _write_json(bundle_manifest_path, tampered_manifest)
    with pytest.raises(RuntimeError, match="Runtime identity differs"):
        verify_final_bundle(output, expected_export=expected_export)
    bundle_manifest_path.write_text(original_manifest, encoding="utf-8")

    with (output / manifest["sketch_file"]).open("a", encoding="utf-8") as handle:
        handle.write("// tamper\n")
    with pytest.raises(RuntimeError, match="inventory mismatch"):
        verify_final_bundle(output, expected_export=expected_export)


def test_final_bundle_rejects_unmanifested_nested_build_output(
    tmp_path: Path,
) -> None:
    export_dir = _make_export(tmp_path / "export", "A", "rf_kd")
    expected_export = _export_identity(export_dir)
    output = tmp_path / "bundle"
    prepare_final_bundle(
        export_dir=export_dir,
        output_dir=output,
        board="esp32c3",
        transport="usb_serial",
        build_contract=_build_contract("esp32c3"),
        verifier=_fake_verifier,
    )
    generated = output / "build"
    generated.mkdir()
    (generated / "firmware.bin").write_bytes(b"generated")
    with pytest.raises(RuntimeError, match="unlisted directory"):
        verify_final_bundle(output, expected_export=expected_export)


@pytest.mark.parametrize(
    ("source_name", "expected_error", "canonical_drift"),
    [
        ("model_weights.h", "differs from export identity", False),
        ("cukd_protocol.c", "differs from the canonical source", False),
        ("cukd_model.c", "differs from host-tested export snapshot", True),
    ],
)
def test_bundle_rejects_coherently_resealed_compiled_source(
    tmp_path: Path, source_name: str, expected_error: str, canonical_drift: bool
) -> None:
    export_dir = _make_export(tmp_path / "export", "A", "rf_kd")
    expected_export = _export_identity(export_dir)
    output = tmp_path / "bundle"
    prepare_final_bundle(
        export_dir=export_dir,
        output_dir=output,
        board="esp32c3",
        transport="usb_serial",
        build_contract=_build_contract("esp32c3"),
        verifier=_fake_verifier,
    )
    manifest_path = output / "final_bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_bundle_id = manifest["bundle_id"]
    source_path = output / source_name
    source_path.write_bytes(source_path.read_bytes() + b"\n/* resealed */\n")
    for item in manifest["bundle_identity_payload"]["source_files"]:
        if item["name"] == source_name:
            item["size_bytes"] = source_path.stat().st_size
            item["sha256"] = sha256_file(source_path)
            break
    else:
        raise AssertionError(source_name)
    new_bundle_id = canonical_json_sha256(manifest["bundle_identity_payload"])
    manifest["bundle_id"] = new_bundle_id
    manifest["runtime_identity_response"] = manifest[
        "runtime_identity_response"
    ].replace(old_bundle_id, new_bundle_id)
    header_path = output / "cukd_final_bundle_identity.h"
    header_path.write_text(
        header_path.read_text(encoding="ascii").replace(old_bundle_id, new_bundle_id),
        encoding="ascii",
        newline="\n",
    )
    for item in manifest["files"]:
        member = output / item["path"]
        item["size_bytes"] = member.stat().st_size
        item["sha256"] = sha256_file(member)
    manifest.pop("manifest_payload_sha256")
    manifest["manifest_payload_sha256"] = canonical_json_sha256(manifest)
    _write_json(manifest_path, manifest)
    canonical_root = None
    if canonical_drift:
        canonical_root = tmp_path / "changed_checkout"
        canonical_relatives = [
            final_bundles.USB_TEMPLATE_RELATIVES["esp32c3"],
            *[
                final_bundles.MODEL_COMMON_RELATIVE / name
                for name in final_bundles.MODEL_COMMON_FILES
            ],
        ]
        for relative in canonical_relatives:
            destination = canonical_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO_ROOT / relative, destination)
        (canonical_root / final_bundles.MODEL_COMMON_RELATIVE / source_name).write_bytes(
            source_path.read_bytes()
        )
    with pytest.raises(RuntimeError, match=expected_error):
        verify_final_bundle(
            output,
            expected_export=expected_export,
            canonical_source_root=canonical_root,
        )


def test_final_bundle_portable_mode_uses_archived_builder_and_sealed_template(
    tmp_path: Path,
) -> None:
    export_dir = _make_export(tmp_path / "export", "A", "rf_kd")
    expected_export = _export_identity(export_dir)
    output = tmp_path / "bundle"
    prepare_final_bundle(
        export_dir=export_dir,
        output_dir=output,
        board="esp32c3",
        transport="usb_serial",
        build_contract=_build_contract("esp32c3"),
        verifier=_fake_verifier,
    )
    manifest = json.loads(
        (output / "final_bundle_manifest.json").read_text(encoding="utf-8")
    )
    builder_sha256 = manifest["bundle_identity_payload"]["bundle_builder_sha256"]

    canonical_root = tmp_path / "archive_execution"
    canonical_relatives = [
        final_bundles.USB_TEMPLATE_RELATIVES["esp32c3"],
        *[
            final_bundles.MODEL_COMMON_RELATIVE / name
            for name in final_bundles.MODEL_COMMON_FILES
        ],
    ]
    for relative in canonical_relatives:
        destination = canonical_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative, destination)
    bad_root = tmp_path / "bad_archive_execution"
    shutil.copytree(canonical_root, bad_root)
    (bad_root / final_bundles.USB_TEMPLATE_RELATIVES["esp32c3"]).write_text(
        "different source\n", encoding="ascii"
    )
    with pytest.raises(RuntimeError, match="canonical source"):
        verify_final_bundle(
            output,
            expected_export=expected_export,
            canonical_source_root=bad_root,
        )
    verified = verify_final_bundle(
        output,
        expected_export=expected_export,
        expected_builder_sha256=builder_sha256,
        canonical_source_root=canonical_root,
    )
    assert verified["bundle_id"] == manifest["bundle_id"]
    with pytest.raises(RuntimeError, match="builder source differs"):
        verify_final_bundle(
            output,
            expected_export=expected_export,
            expected_builder_sha256="0" * 64,
            canonical_source_root=canonical_root,
        )


def test_bundle_rejects_coherently_resealed_nonreproducible_sketch(
    tmp_path: Path,
) -> None:
    export_dir = _make_export(tmp_path / "export", "A", "rf_kd")
    expected_export = _export_identity(export_dir)
    output = tmp_path / "bundle"
    prepare_final_bundle(
        export_dir=export_dir,
        output_dir=output,
        board="esp32c3",
        transport="usb_serial",
        build_contract=_build_contract("esp32c3"),
        verifier=_fake_verifier,
    )
    manifest_path = output / "final_bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_bundle_id = manifest["bundle_id"]
    sketch_path = output / manifest["sketch_file"]
    sketch_path.write_text(
        sketch_path.read_text(encoding="utf-8") + "// coherently resealed\n",
        encoding="utf-8",
        newline="\n",
    )
    sketch_hash = sha256_file(sketch_path)
    identity = manifest["bundle_identity_payload"]
    identity["transformed_sketch_sha256"] = sketch_hash
    new_bundle_id = canonical_json_sha256(identity)
    manifest["bundle_id"] = new_bundle_id
    manifest["transformed_sketch_sha256"] = sketch_hash
    manifest["runtime_identity_response"] = manifest["runtime_identity_response"].replace(
        old_bundle_id, new_bundle_id
    )
    header_path = output / "cukd_final_bundle_identity.h"
    header_path.write_text(
        header_path.read_text(encoding="ascii").replace(old_bundle_id, new_bundle_id),
        encoding="ascii",
        newline="\n",
    )
    for item in manifest["files"]:
        member = output / item["path"]
        item["size_bytes"] = member.stat().st_size
        item["sha256"] = sha256_file(member)
    manifest.pop("manifest_payload_sha256")
    manifest["manifest_payload_sha256"] = canonical_json_sha256(manifest)
    _write_json(manifest_path, manifest)
    with pytest.raises(RuntimeError, match="does not reproduce"):
        verify_final_bundle(output, expected_export=expected_export)


def test_runtime_identity_rejects_field_substitution(tmp_path: Path) -> None:
    export_dir = _make_export(tmp_path / "export", "B", "scratch")
    export = validate_final_export(export_dir, verifier=_fake_verifier)
    line = runtime_identity(
        export,
        bundle_id="a" * 64,
        board="arduino_r4",
        transport="usb_serial",
        build_contract_id="b" * 64,
    )
    assert parse_runtime_identity(line)["route"] == "scratch"
    with pytest.raises(ValueError, match="invalid model route"):
        parse_runtime_identity(line.replace(",scratch,", ",unknown,"))


def test_response_verifier_rejects_zero_duration_timing() -> None:
    response = {
        "row_id": 0,
        "status": "OK",
        "predicted_class": 3,
        "logits": [0, 0, 0, 1, 0],
        "preprocess_us": 0,
        "inference_us": 0,
        "total_us": 0,
        "host_observed_rtt_us": 0,
        "transaction_elapsed_us": 0,
        "attempts": 1,
        "response_timeout_count": 0,
        "ignored_datagram_count": 0,
        **{f"ignored_{name}_count": 0 for name in IGNORED_DATAGRAM_CATEGORIES},
    }
    reference = {"row_id": 0, "fixed_pred": 3, "logits": [0, 0, 0, 1, 0]}
    with pytest.raises(RuntimeError, match="Invalid device/host timing"):
        verify_response_records(
            responses=[response], reference_rows=[reference], transport="usb_serial"
        )


def test_balanced_cohort_cannot_be_first_1000_prefix() -> None:
    rows = [
        {
            "timing_row_id": index,
            "original_full_test_row_id": 1000 + index,
            "source_row_index": 10_000 + index,
            "true_label": index % 5,
            "class_name": ["Blackhole", "Flooding", "Grayhole", "Normal", "TDMA"][
                index % 5
            ],
            "class_rank": index // 5,
            "selection_rank_sha256": f"{index + 1000:064x}",
            "feature_group_sha256": f"{index:064x}",
        }
        for index in range(1000)
    ]
    validate_cohort_selection(rows)
    for index, row in enumerate(rows):
        row["original_full_test_row_id"] = index
    with pytest.raises(RuntimeError, match="first-1,000"):
        validate_cohort_selection(rows)


def test_balanced_cohort_rejects_coherently_resealed_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deployment.hardware_hil.host import (
        generate_fgds_balanced_timing_cohort as cohort_source,
    )

    split_sizes = {"train": 5, "validation": 5, "test": 10}
    monkeypatch.setattr(contracts, "EXPECTED_FULL_ROWS", 10)
    monkeypatch.setattr(contracts, "EXPECTED_SPLIT_SIZES", split_sizes)
    monkeypatch.setattr(contracts, "GROUPS_PER_CLASS", 2)

    dataset = tmp_path / "WSN-DS.csv"
    feature_rows: list[list[float]] = []
    class_text: list[str] = []
    for label, class_name in enumerate(cohort_source.CLASS_NAMES):
        for group_index in range(4):
            feature_rows.append(
                [
                    float((label + 1) * 100_000 + group_index * 100 + feature)
                    for feature in range(17)
                ]
            )
            class_text.append(class_name)
    frame = pd.DataFrame(feature_rows, columns=cohort_source.FEATURE_NAMES)
    frame["Attack type"] = class_text
    frame.to_csv(dataset, index=False)

    train = np.asarray([label * 4 for label in range(5)], dtype=np.int64)
    validation = np.asarray([label * 4 + 1 for label in range(5)], dtype=np.int64)
    test = np.asarray(
        [label * 4 + offset for label in range(5) for offset in [2, 3]],
        dtype=np.int64,
    )
    split_root = tmp_path / "split"
    split_root.mkdir()
    np.savez_compressed(
        split_root / "split_indices.npz",
        train_indices=train,
        validation_indices=validation,
        test_indices=test,
    )
    mean = np.zeros(17, dtype=np.float64)
    scale = np.ones(17, dtype=np.float64)
    variance = np.ones(17, dtype=np.float64)
    np.savez_compressed(
        split_root / "scaler_parameters.npz",
        mean=mean,
        scale=scale,
        var=variance,
        n_samples_seen=np.asarray([len(train)], dtype=np.int64),
    )
    dataset_hash = sha256_file(dataset)
    split_hash = cohort_source.sha256_arrays(train, validation, test)
    scaler_hash = cohort_source.sha256_arrays(mean, scale, variance)
    execution = {
        "protocol_id": cohort_source.PROTOCOL_ID,
        "dataset_sha256": dataset_hash,
        "split_indices_sha256": split_hash,
        "scaler_sha256": scaler_hash,
        "seeds": cohort_source.EXPECTED_SEEDS,
    }
    execution["execution_fingerprint_sha256"] = canonical_json_sha256(execution)
    _write_json(split_root / "execution_contract.json", execution)
    _write_json(
        split_root / "preprocessing_contract.json",
        {
            "protocol_id": cohort_source.PROTOCOL_ID,
            "dataset_sha256": dataset_hash,
            "dataset_shape": [len(feature_rows), 17],
            "feature_names": cohort_source.FEATURE_NAMES,
            "class_names": cohort_source.CLASS_NAMES,
            "split_sizes": split_sizes,
            "split_indices_file": "split_indices.npz",
            "split_indices_file_sha256": sha256_file(
                split_root / "split_indices.npz"
            ),
            "split_indices_sha256": split_hash,
            "scaler_parameters_file": "scaler_parameters.npz",
            "scaler_parameters_file_sha256": sha256_file(
                split_root / "scaler_parameters.npz"
            ),
            "scaler_sha256": scaler_hash,
        },
    )
    split_files = [
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(split_root.iterdir())
        if path.is_file()
    ]
    _write_json(
        split_root / "artifact_manifest.json",
        {
            "status": "complete",
            "protocol_id": cohort_source.PROTOCOL_ID,
            "file_count_excluding_manifest": len(split_files),
            "files": split_files,
        },
    )

    export = _make_export(
        tmp_path / "export",
        "A",
        "scratch",
        hashes={
            "checkpoint": HASHES["checkpoint"],
            "trained": HASHES["trained"],
            "dataset": dataset_hash,
            "split": split_hash,
            "scaler": scaler_hash,
        },
        test_rows=len(test),
    )
    raw_features = np.asarray(feature_rows, dtype=np.float64)[test]
    full_replay = [
        {
            "row_id": row_id,
            "source_row_index": int(source_row),
            **{f"f{index}": int(raw_features[row_id, index]) for index in range(17)},
        }
        for row_id, source_row in enumerate(test)
    ]
    full_reference = []
    for row_id, source_row in enumerate(test):
        label = int(source_row // 4)
        full_reference.append(
            {
                "row_id": row_id,
                "source_row_index": int(source_row),
                "true_label": label,
                "fixed_pred": label,
                "fp32_pred": label,
                **{f"fixed_logit_{index}": int(index == label) for index in range(5)},
            }
        )
    with (export / "hil_replay_vectors.csv").open(
        "w", newline="", encoding="ascii"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(full_replay[0]))
        writer.writeheader()
        writer.writerows(full_replay)
    with (export / "hil_reference_predictions.csv").open(
        "w", newline="", encoding="ascii"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(full_reference[0]))
        writer.writeheader()
        writer.writerows(full_reference)
    _write_json(
        export / "preprocess_int_metadata.json", {"input_dim": 17, "raw_q_frac": 0}
    )
    export_manifest = json.loads(
        (export / "final_export_manifest.json").read_text(encoding="utf-8")
    )
    export_files = [
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(export.iterdir())
        if path.is_file() and path.name != "final_export_manifest.json"
    ]
    export_manifest["files"] = export_files
    export_manifest["file_count_excluding_manifest"] = len(export_files)
    export_manifest.pop("manifest_payload_sha256", None)
    export_manifest["manifest_payload_sha256"] = canonical_json_sha256(export_manifest)
    _write_json(export / "final_export_manifest.json", export_manifest)

    output = tmp_path / "cohort"
    manifest_path = generate_balanced_cohort(
        {"student_A_scratch": export},
        dataset_csv=dataset,
        split_root=split_root,
        output_dir=output,
        verifier=_fake_verifier,
    )
    assert manifest_path == output / "final_timing_cohort_manifest.json"
    identity = contracts.validate_final_export(export, verifier=_fake_verifier)
    validate_balanced_cohort(output, identities={"student_A_scratch": identity})
    original_manifest_text = manifest_path.read_text(encoding="utf-8")

    nested = output / "unmanifested" / "extra.txt"
    nested.parent.mkdir()
    nested.write_text("extra\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="extra or unlisted"):
        validate_balanced_cohort(output, identities={"student_A_scratch": identity})
    nested.unlink()
    nested.parent.rmdir()

    duplicate_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    duplicate_manifest["files"].append(dict(duplicate_manifest["files"][0]))
    duplicate_manifest["file_count_excluding_manifest"] += 1
    duplicate_manifest.pop("manifest_payload_sha256")
    duplicate_manifest["manifest_payload_sha256"] = canonical_json_sha256(
        duplicate_manifest
    )
    _write_json(manifest_path, duplicate_manifest)
    with pytest.raises(RuntimeError, match="inventory path is unsafe"):
        validate_balanced_cohort(output, identities={"student_A_scratch": identity})
    manifest_path.write_text(original_manifest_text, encoding="utf-8")

    portable_manifest = json.loads(original_manifest_text)
    portable_manifest["dataset"]["path_recorded"] = "Z:/unavailable/WSN-DS.csv"
    portable_manifest["split"]["path_recorded"] = "Z:/unavailable/split"
    portable_manifest["models"]["student_A_scratch"]["path_recorded"] = (
        "Z:/unavailable/export"
    )
    portable_manifest.pop("manifest_payload_sha256")
    portable_manifest["manifest_payload_sha256"] = canonical_json_sha256(
        portable_manifest
    )
    _write_json(manifest_path, portable_manifest)
    monkeypatch.setattr(
        final_runtime,
        "stage_contract",
        lambda name: {
            "name": name,
            "ordinal": 1,
            "rows": len(test),
            "input_role": "balanced_timing",
            "include_in_reported_metrics": True,
            "include_in_timing_metrics": True,
            "include_in_fidelity_metrics": False,
        },
    )
    stage_dataset = final_runtime.load_stage_dataset(
        export=identity,
        cohort_dir=output,
        stage_name="portable_timing_test",
    )
    assert len(stage_dataset.replay_rows) == len(test)
    assert len(stage_dataset.reference_rows) == len(test)
    manifest_path.write_text(original_manifest_text, encoding="utf-8")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing = {row["feature_group_sha256"] for row in manifest["rows"]}
    replacement = next(
        value * 64 for value in "0123456789abcdef" if value * 64 not in existing
    )
    manifest["rows"][0]["feature_group_sha256"] = replacement
    manifest.pop("manifest_payload_sha256")
    manifest["manifest_payload_sha256"] = canonical_json_sha256(manifest)
    _write_json(manifest_path, manifest)
    with pytest.raises(RuntimeError, match="selection does not reconstruct"):
        validate_balanced_cohort(output, identities={"student_A_scratch": identity})


def test_balanced_cohort_requires_its_canonical_payload_seal(
    tmp_path: Path,
) -> None:
    cohort = tmp_path / "cohort"
    cohort.mkdir()
    manifest = {
        "schema": "cukd_final_hil_balanced_timing_cohort_v1",
        "status": "passed",
    }
    _write_json(cohort / "final_timing_cohort_manifest.json", manifest)
    with pytest.raises(RuntimeError, match="canonical payload hash is missing"):
        validate_balanced_cohort(cohort, identities={})


def test_wifi_stage_rechecks_the_provisioning_board_before_udp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    connection_path = tmp_path / "connection.json"
    connection_path.write_text("{}\n", encoding="ascii")
    monkeypatch.setattr(final_runtime, "validate_final_export", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        final_runtime,
        "verify_final_bundle",
        lambda *_args, **_kwargs: {"transport": "wifi_udp"},
    )
    monkeypatch.setattr(
        final_runtime,
        "validate_wifi_connection",
        lambda *_args, **_kwargs: {
            "provisioning_port": "COM_TEST",
            "physical_port_serial": "SERIAL_TEST",
        },
    )
    monkeypatch.setattr(
        final_runtime,
        "load_stage_dataset",
        lambda **_kwargs: SimpleNamespace(replay_rows=[], reference_rows=[]),
    )
    calls: list[tuple[str, str]] = []

    def reject_wrong_specimen(port: str, serial: str) -> str:
        calls.append((port, serial))
        raise RuntimeError("physical specimen unavailable")

    monkeypatch.setattr(
        final_runtime, "require_physical_port_serial", reject_wrong_specimen
    )
    with pytest.raises(RuntimeError, match="physical specimen unavailable"):
        final_runtime.execute_wifi_stage(
            export_dir=tmp_path / "export",
            cohort_dir=tmp_path / "cohort",
            bundle_dir=tmp_path / "bundle",
            connection_json=connection_path,
            stage_name="smoke_10",
            output_root=tmp_path / "attempts",
            campaign_session_id="A" * 32,
        )
    assert calls == [("COM_TEST", "SERIAL_TEST")]


def test_physical_port_identity_accepts_stable_alias_for_enumerated_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from serial.tools import list_ports

    alias = "/dev/serial/by-id/cukd-board"
    endpoint = "/dev/ttyUSB0"
    monkeypatch.setattr(
        list_ports,
        "comports",
        lambda: [SimpleNamespace(device=endpoint, serial_number="CUKD123")],
    )
    monkeypatch.setattr(
        final_runtime,
        "_canonical_serial_endpoint",
        lambda value: endpoint if value in {alias, endpoint} else value,
    )

    assert final_runtime.require_physical_port_serial(alias, "CUKD123") == "CUKD123"


def test_malformed_wifi_stage_dataset_fails_before_hardware_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    connection_path = tmp_path / "connection.json"
    connection_path.write_text("{}\n", encoding="ascii")
    monkeypatch.setattr(
        final_runtime, "validate_final_export", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        final_runtime,
        "verify_final_bundle",
        lambda *_args, **_kwargs: {"transport": "wifi_udp"},
    )
    monkeypatch.setattr(
        final_runtime,
        "validate_wifi_connection",
        lambda *_args, **_kwargs: {
            "provisioning_port": "COM_TEST",
            "physical_port_serial": "SERIAL_TEST",
        },
    )
    monkeypatch.setattr(
        final_runtime,
        "load_stage_dataset",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("cohort is malformed")),
    )
    hardware_calls: list[str] = []

    def forbidden_hardware(*_args: object, **_kwargs: object) -> str:
        hardware_calls.append("called")
        raise AssertionError("malformed cohort reached hardware access")

    monkeypatch.setattr(
        final_runtime, "require_physical_port_serial", forbidden_hardware
    )
    with pytest.raises(RuntimeError, match="cohort is malformed"):
        final_runtime.execute_wifi_stage(
            export_dir=tmp_path / "export",
            cohort_dir=tmp_path / "cohort",
            bundle_dir=tmp_path / "bundle",
            connection_json=connection_path,
            stage_name="smoke_10",
            output_root=tmp_path / "attempts",
            campaign_session_id="A" * 32,
        )
    assert hardware_calls == []


def test_invalid_wifi_stage_fails_before_artifact_or_hardware_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> object:
        calls.append("called")
        raise AssertionError("invalid stage reached an artifact or hardware dependency")

    monkeypatch.setattr(final_runtime, "validate_final_export", forbidden)
    monkeypatch.setattr(final_runtime, "verify_final_bundle", forbidden)
    monkeypatch.setattr(final_runtime, "validate_wifi_connection", forbidden)
    monkeypatch.setattr(final_runtime, "require_physical_port_serial", forbidden)
    with pytest.raises(ValueError, match="Unknown final HIL stage"):
        final_runtime.execute_wifi_stage(
            export_dir=tmp_path / "export",
            cohort_dir=tmp_path / "cohort",
            bundle_dir=tmp_path / "bundle",
            connection_json=tmp_path / "connection.json",
            stage_name="invalid_stage",
            output_root=tmp_path / "attempts",
            campaign_session_id="A" * 32,
        )
    assert calls == []


def test_wifi_retry_and_device_duplicate_counters_are_exactly_reconciled() -> None:
    def exchange(attempts: int, *, prior_attempt_responses: int = 0) -> dict:
        ignored = {
            category: prior_attempt_responses if category == "wrong_attempt" else 0
            for category in IGNORED_DATAGRAM_CATEGORIES
        }
        return {
            "attempts": attempts,
            "response_timeout_count": attempts - 1,
            "ignored_datagram_count": sum(ignored.values()),
            **{f"ignored_{category}_count": value for category, value in ignored.items()},
        }

    rows = [exchange(2, prior_attempt_responses=1), exchange(1)]
    controls = {
        "udp_max_attempts": 3,
        "identity": exchange(1),
        "begin": exchange(1),
        "end": {
            **exchange(1),
            "device_stage_counters": {
                "completed_rows": 2,
                "inferences": 2,
                "received_datagrams": 5,
                "duplicate_replays": 1,
                "oversized_datagrams": 0,
                "short_reads": 0,
                "bad_envelopes": 0,
                "wrong_sessions": 0,
                "wrong_endpoints": 0,
                "wrong_stages": 0,
                "control_errors": 0,
                "data_errors": 0,
                "stale_transactions": 0,
            }
        },
    }
    host = {
        "stale_datagrams_drained_before_identity": 0,
        "device_protocol_errors": 0,
        "retransmissions": 1,
        "response_timeouts": 1,
        "datagrams_sent": 6,
        "datagrams_received": 6,
        **{
            f"ignored_{category}": 1 if category == "wrong_attempt" else 0
            for category in IGNORED_DATAGRAM_CATEGORIES
        },
    }
    result = validate_wifi_counter_reconciliation(
        rows=rows, controls=controls, session_counters=host, expected_rows=2
    )
    assert result["device_duplicate_replays"] == 1
    assert result["host_data_retransmissions_without_device_duplicate"] == 0
    assert result["host_ignored_prior_attempt_responses"] == 1

    controls["end"]["device_stage_counters"]["data_errors"] = 1
    with pytest.raises(RuntimeError, match="data_errors"):
        validate_wifi_counter_reconciliation(
            rows=rows, controls=controls, session_counters=host, expected_rows=2
        )

    controls["end"]["device_stage_counters"]["data_errors"] = 0
    rows[0]["ignored_wrong_attempt_count"] = 2
    rows[0]["ignored_datagram_count"] = 2
    host["ignored_wrong_attempt"] = 2
    host["datagrams_received"] = 7
    with pytest.raises(RuntimeError, match="prior-attempt responses"):
        validate_wifi_counter_reconciliation(
            rows=rows, controls=controls, session_counters=host, expected_rows=2
        )


def _stage_ledger() -> list[dict]:
    start = datetime(2026, 8, 12, tzinfo=timezone.utc)
    rows = []
    for index, stage in enumerate(FINAL_STAGES):
        started = start + timedelta(minutes=index * 2)
        rows.append(
            {
                **dict(stage),
                "status": "passed",
                "attempt_id": f"{index + 1:032X}",
                "started_utc": started.isoformat(),
                "finished_utc": (started + timedelta(minutes=1)).isoformat(),
            }
        )
    return rows


def test_session_completion_recomputes_raw_stage_responses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export_dir = tmp_path / "export"
    cohort_dir = tmp_path / "cohort"
    bundle_dir = tmp_path / "bundle"
    provenance_dir = tmp_path / "provenance"
    for directory in [export_dir, cohort_dir, bundle_dir, provenance_dir]:
        directory.mkdir()
    for path in [
        export_dir / "final_export_manifest.json",
        cohort_dir / "final_timing_cohort_manifest.json",
        bundle_dir / "final_bundle_manifest.json",
    ]:
        path.write_text("bound\n", encoding="ascii")
    provenance_path = provenance_dir / "build_upload_provenance.json"
    provenance_path.write_text("provenance\n", encoding="ascii")

    export = SimpleNamespace(
        export_id="1" * 64,
        trained_state_sha256="2" * 64,
        checkpoint_file_sha256="3" * 64,
        full_replay_sha256="4" * 64,
        full_reference_sha256="5" * 64,
        manifest_sha256=sha256_file(export_dir / "final_export_manifest.json"),
        model_key="student_A_rf_kd",
    )
    bundle = {
        "bundle_id": "6" * 64,
        "build_contract_id": "7" * 64,
        "_manifest_sha256": sha256_file(bundle_dir / "final_bundle_manifest.json"),
    }
    stages = _stage_ledger()
    attempts: dict[str, dict] = {}
    for stage in stages:
        attempt_dir = tmp_path / f"attempt_{stage['ordinal']}"
        attempt_dir.mkdir()
        (attempt_dir / "responses.csv").write_text("valid\n", encoding="ascii")
        (attempt_dir / "final_attempt.json").write_text("{}\n", encoding="ascii")
        stage["attempt_path_recorded"] = str(attempt_dir)
        stage["attempt_manifest_sha256"] = sha256_file(
            attempt_dir / "final_attempt.json"
        )
        attempts[stage["attempt_id"]] = {
            "status": "passed",
            "attempt_id": stage["attempt_id"],
            "campaign_session_id": "A" * 32,
            "bundle_id": bundle["bundle_id"],
            "build_contract_id": bundle["build_contract_id"],
            "combination": {
                "student": "A",
                "route": "rf_kd",
                "board": "esp32c3",
                "transport": "usb_serial",
            },
            "completed_rows": stage["rows"],
            "stage": {"name": stage["name"]},
            "verification": {
                "status": "passed",
                "sequence_exact": True,
                "predictions_exact": True,
                "logits_exact": True,
                "wifi_retry_reconciliation": None,
            },
            "physical_identity": {"physical_port_serial": "SERIAL"},
                "controls": {
                    "serial_open_policy": {"dtr": False, "rts": False},
                    "baud": 115200,
                },
        }

    monkeypatch.setattr(
        "deployment.final_hil.evidence.validate_final_export",
        lambda *args, **kwargs: export,
    )
    monkeypatch.setattr(
        "deployment.final_hil.evidence.validate_balanced_cohort",
        lambda *args, **kwargs: {
            "models": {
                "student_A_rf_kd": {
                    "export_id": export.export_id,
                    "trained_state_sha256": export.trained_state_sha256,
                    "full_replay_sha256": export.full_replay_sha256,
                    "full_reference_sha256": export.full_reference_sha256,
                }
            }
        },
    )
    monkeypatch.setattr(
        "deployment.final_hil.evidence.verify_final_bundle",
        lambda *args, **kwargs: bundle,
    )
    monkeypatch.setattr(
        "deployment.final_hil.evidence.validate_build_upload_provenance",
        lambda *args, **kwargs: {
            "provenance_id": "8" * 64,
            "bundle_id": bundle["bundle_id"],
            "build_contract_id": bundle["build_contract_id"],
            "board": "esp32c3",
            "transport": "usb_serial",
            "physical_port_serial": "SERIAL",
            "student": "A",
            "route": "rf_kd",
            "export_id": export.export_id,
            "model_sha256": export.trained_state_sha256,
            "checkpoint_file_sha256": export.checkpoint_file_sha256,
            "finished_utc": (
                datetime(2026, 8, 12, tzinfo=timezone.utc) - timedelta(minutes=1)
            ).isoformat(),
        },
    )

    by_path = {
        Path(stage["attempt_path_recorded"]).resolve(): attempts[stage["attempt_id"]]
        for stage in stages
    }

    def deep_verify(attempt_dir: Path, **_: object) -> dict:
        resolved = attempt_dir.resolve()
        if (resolved / "responses.csv").read_text(encoding="ascii") != "valid\n":
            raise RuntimeError("raw responses changed")
        return by_path[resolved]

    monkeypatch.setattr(
        "deployment.final_hil.evidence.verify_stage_attempt", deep_verify
    )
    payload = {
        "schema": "cukd_final_hil_six_stage_completion_v2",
        "status": "passed",
        "campaign_session_id": "A" * 32,
        "started_utc": stages[0]["started_utc"],
        "finished_utc": stages[-1]["finished_utc"],
        "combination_id": "student_A_rf_kd__esp32c3__usb_serial",
        "model_key": "student_A_rf_kd",
        "student": "A",
        "route": "rf_kd",
        "board": "esp32c3",
        "transport": "usb_serial",
        "export_id": export.export_id,
        "model_sha256": export.trained_state_sha256,
        "checkpoint_file_sha256": export.checkpoint_file_sha256,
        "bundle_id": bundle["bundle_id"],
        "build_contract_id": bundle["build_contract_id"],
        "export_path_recorded": str(export_dir),
        "export_manifest_sha256": export.manifest_sha256,
        "cohort_path_recorded": str(cohort_dir),
        "cohort_manifest_sha256": sha256_file(
            cohort_dir / "final_timing_cohort_manifest.json"
        ),
        "bundle_path_recorded": str(bundle_dir),
        "bundle_manifest_sha256": bundle["_manifest_sha256"],
        "physical_port_serial": "SERIAL",
        "wifi_mac_reported": None,
        "provenance_id": "8" * 64,
        "provenance_path_recorded": str(provenance_path),
        "provenance_sha256": sha256_file(provenance_path),
        "stages": stages,
        "row_totals": validate_session_stage_ledger(stages),
        "warmup_excluded_from_reported_metrics": True,
        "smoke_excluded_from_reported_metrics": True,
        "timing_statistical_unit": TIMING_STATISTICAL_UNIT,
        "wifi_network_session_id": None,
        "connection_path_recorded": None,
        "connection_record_sha256": None,
        "connection_payload_sha256": None,
        "udp_timeout_seconds": None,
        "udp_max_attempts": None,
    }
    payload["session_evidence_id"] = canonical_json_sha256(payload)
    session_path = tmp_path / "session.json"
    _write_json(session_path, payload)
    validate_session_completion(session_path, verifier=_fake_verifier)

    first_attempt_id = stages[0]["attempt_id"]
    attempts[first_attempt_id]["build_contract_id"] = "9" * 64
    with pytest.raises(RuntimeError, match="not exact passed evidence"):
        validate_session_completion(session_path, verifier=_fake_verifier)
    attempts[first_attempt_id]["build_contract_id"] = bundle["build_contract_id"]

    first_attempt = Path(stages[0]["attempt_path_recorded"])
    (first_attempt / "responses.csv").write_text("tampered\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="raw responses changed"):
        validate_session_completion(session_path, verifier=_fake_verifier)


def test_wifi_configuration_reserves_evidence_output_before_hardware(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "connection.json"
    output.with_suffix(".json.tmp").write_text("stale\n", encoding="ascii")
    monkeypatch.setattr(
        final_runtime, "validate_final_export", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        final_runtime,
        "verify_final_bundle",
        lambda *_args, **_kwargs: {"transport": "wifi_udp"},
    )
    hardware_calls: list[str] = []

    def forbidden_hardware(*_args: object, **_kwargs: object) -> str:
        hardware_calls.append("called")
        raise AssertionError("unwritable evidence destination reached hardware")

    monkeypatch.setattr(
        final_runtime, "require_physical_port_serial", forbidden_hardware
    )
    with pytest.raises(FileExistsError, match="Stale temporary evidence"):
        final_runtime.configure_final_wifi(
            export_dir=tmp_path / "export",
            bundle_dir=tmp_path / "bundle",
            port="COM_TEST",
            physical_port_serial="SERIAL_TEST",
            ssid="test-network",
            password="test-password",
            output_json=output,
        )
    assert hardware_calls == []


def test_wifi_attempts_require_one_sealed_connection_record() -> None:
    attempts = [
        {
            "physical_identity": {"wifi_network_session_id": "A" * 32},
            "controls": {
                "wifi_network_session_id": "A" * 32,
                "connection_path_recorded": "C:/sealed/connection.json",
                "connection_record_sha256": "b" * 64,
                "connection_payload_sha256": "c" * 64,
                "udp_timeout_seconds": 1.0,
                "udp_max_attempts": 3,
            },
        }
        for _ in range(6)
    ]
    final_runtime._validate_attempt_connection_set(attempts, "wifi_udp")
    attempts[-1]["controls"]["connection_record_sha256"] = "d" * 64
    with pytest.raises(RuntimeError, match="one sealed connection record"):
        final_runtime._validate_attempt_connection_set(attempts, "wifi_udp")
    attempts[-1]["controls"]["connection_record_sha256"] = "b" * 64
    attempts[-1]["controls"]["udp_max_attempts"] = 2
    with pytest.raises(RuntimeError, match="one sealed connection record"):
        final_runtime._validate_attempt_connection_set(attempts, "wifi_udp")


def test_campaign_preserves_supplied_session_path_for_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "nested").mkdir()
    session = tmp_path / "session.json"
    session.write_text('{"combination_id":"only"}\n', encoding="ascii")
    supplied = tmp_path / "nested" / ".." / "session.json"
    contract = {
        "status": "ready",
        "eligible_combinations": [{"combination_id": "only"}],
    }
    monkeypatch.setattr(final_evidence, "validate_campaign_contract", lambda _: contract)

    class ValidationReached(RuntimeError):
        pass

    def validate_without_preresolution(path: Path, **_: object) -> dict:
        assert path == supplied
        raise ValidationReached

    monkeypatch.setattr(
        final_evidence, "validate_session_completion", validate_without_preresolution
    )
    with pytest.raises(ValidationReached):
        final_evidence.verify_complete_campaign(
            campaign_contract=contract,
            session_jsons=[supplied],
        )


def test_six_stage_completion_rejects_partial_or_reordered_evidence() -> None:
    ledger = _stage_ledger()
    assert validate_session_stage_ledger(ledger)["all_rows"] == 59_321
    with pytest.raises(RuntimeError, match="incomplete or reordered"):
        validate_session_stage_ledger(ledger[:-1])
    swapped = list(ledger)
    swapped[2], swapped[3] = swapped[3], swapped[2]
    with pytest.raises(RuntimeError, match="incomplete or reordered"):
        validate_session_stage_ledger(swapped)
    wrong_role = [dict(item) for item in ledger]
    wrong_role[0]["input_role"] = "full_replay"
    with pytest.raises(RuntimeError, match="input_role"):
        validate_session_stage_ledger(wrong_role)


def test_blocked_route_keeps_full_matrix_and_blocks_preflight_contract(
    tmp_path: Path,
) -> None:
    exports = _four_exports(tmp_path)
    blocked_key = "student_B_rf_kd"
    blocked = {
        "status": "blocked",
        "quality_gates_passed": False,
        "identity": {
            "protocol": "wsnds_feature_group_split_train_only_scaler_10seed_v2",
            "seed": 42,
            "student": "B",
            "route": "rf_kd",
        },
        "reason": "fixed-point quality gate failed",
    }
    blocked["identity"]["blocked_audit_id"] = canonical_json_sha256(
        blocked["identity"]
    )
    blocked["audit_payload_sha256"] = canonical_json_sha256(blocked)
    blocked_path = tmp_path / "blocked.json"
    _write_json(blocked_path, blocked)
    sources = {key: path for key, path in exports.items() if key != blocked_key}
    sources[blocked_key] = {"blocked_audit": blocked_path}
    contract = build_campaign_contract(
        sources,
        verifier=_fake_verifier,
        blocked_verifier=_fake_blocked_verifier,
    )
    validate_campaign_contract(contract)
    assert contract["status"] == "ready_with_blocked_routes"
    assert contract["expected_combination_count"] == 16
    assert len(contract["combinations"]) == 16
    assert any(item["model_key"] == blocked_key for item in contract["combinations"])
    assert contract["expected_eligible_combination_count"] == 12
    assert len(contract["eligible_combinations"]) == 12
    assert len(contract["excluded_combinations"]) == 4
    assert contract["blocked_routes"] == [
        {"model_key": blocked_key, "reason": "fixed-point quality gate failed"}
    ]
    preflight = preflight_campaign(
        campaign_contract=contract,
        cohort_dir=None,
        export_dirs={},
        bundle_dirs={},
    )
    assert preflight["status"] == "blocked"
    assert preflight["intended_combination_count"] == 16
    assert preflight["eligible_combination_count"] == 12
    assert preflight["intended_matrix_was_retained"] is True
    assert preflight["execution_matrix_is_gate_derived"] is True
    assert preflight["blocked_routes"] == contract["blocked_routes"]

    sessions = [
        {
            "combination_id": item["combination_id"],
            "stages": [{} for _ in FINAL_STAGES],
            "row_totals": {
                "all_rows": 59_321,
                "timing_rows": 3_000,
                "full_rows": 56_301,
                "warmup_rows": 10,
                "smoke_rows": 10,
            },
        }
        for item in contract["eligible_combinations"]
    ]
    totals = validate_campaign_session_ledger(contract, sessions)
    assert totals["session_count"] == 12
    assert totals["balanced_timing_rows"] == 36_000
    assert totals["full_exact_replay_rows"] == 675_612


def test_preflight_uses_explicit_local_exports_not_recorded_cohort_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exports = _four_exports(tmp_path)
    blocked_key = "student_B_scratch"
    blocked = {
        "status": "blocked",
        "quality_gates_passed": False,
        "identity": {
            "protocol": "wsnds_feature_group_split_train_only_scaler_10seed_v2",
            "seed": 42,
            "student": "B",
            "route": "scratch",
        },
        "reason": "fixed-point quality gate failed",
    }
    blocked["identity"]["blocked_audit_id"] = canonical_json_sha256(
        blocked["identity"]
    )
    blocked["audit_payload_sha256"] = canonical_json_sha256(blocked)
    blocked_path = tmp_path / "blocked.json"
    _write_json(blocked_path, blocked)
    eligible_exports = {
        key: path for key, path in exports.items() if key != blocked_key
    }
    sources: dict[str, Path | dict[str, Path]] = dict(eligible_exports)
    sources[blocked_key] = {"blocked_audit": blocked_path}
    contract = build_campaign_contract(
        sources,
        verifier=_fake_verifier,
        blocked_verifier=_fake_blocked_verifier,
    )

    cohort_dir = tmp_path / "portable_cohort"
    cohort_dir.mkdir()
    (cohort_dir / "final_timing_cohort_manifest.json").write_text(
        "{}\n", encoding="ascii"
    )
    cohort_calls: list[dict[str, object]] = []

    def fake_validate_cohort(
        root: Path,
        *,
        identities: dict,
        reconstruct_sources: bool,
    ) -> dict:
        cohort_calls.append(
            {
                "root": root,
                "identities": identities,
                "reconstruct_sources": reconstruct_sources,
            }
        )
        first = contract["models"][next(iter(eligible_exports))]
        return {
            "source_protocol_id": first["protocol"],
            "dataset": {"sha256": first["dataset_sha256"]},
            "split": {
                "split_indices_sha256": first["split_indices_sha256"],
                "scaler_sha256": first["scaler_sha256"],
            },
            "models": {
                key: {
                    "export_id": identity.export_id,
                    "trained_state_sha256": identity.trained_state_sha256,
                    "full_replay_sha256": identity.full_replay_sha256,
                    "full_reference_sha256": identity.full_reference_sha256,
                    "path_recorded": "Z:/unavailable/original/export",
                }
                for key, identity in identities.items()
            },
        }

    combinations = {
        item["combination_id"]: item for item in contract["eligible_combinations"]
    }

    def fake_verify_bundle(path: Path, *, expected_export: object) -> dict:
        combination = combinations[path.name]
        model = contract["models"][combination["model_key"]]
        assert getattr(expected_export, "root") == str(
            eligible_exports[combination["model_key"]].resolve()
        )
        return {
            "student": model["student"],
            "route": model["route"],
            "export_id": model["export_id"],
            "model_sha256": model["trained_state_sha256"],
            "checkpoint_file_sha256": model["checkpoint_file_sha256"],
            "board": combination["board"],
            "transport": combination["transport"],
            "bundle_id": canonical_json_sha256(
                {"combination_id": combination["combination_id"]}
            ),
            "build_contract_id": (
                "a" * 64 if combination["board"] == "esp32c3" else "b" * 64
            ),
            "_manifest_sha256": "c" * 64,
        }

    monkeypatch.setattr(
        "deployment.final_hil.evidence.validate_balanced_cohort",
        fake_validate_cohort,
    )
    monkeypatch.setattr(
        "deployment.final_hil.evidence.verify_final_bundle", fake_verify_bundle
    )
    bundle_dirs = {key: tmp_path / key for key in combinations}
    preflight = preflight_campaign(
        campaign_contract=contract,
        cohort_dir=cohort_dir,
        export_dirs=eligible_exports,
        bundle_dirs=bundle_dirs,
        verifier=_fake_verifier,
    )

    assert preflight["status"] == "ready_with_blocked_routes"
    assert preflight["blockers"] == []
    assert len(cohort_calls) == 1
    assert cohort_calls[0]["root"] == cohort_dir
    assert cohort_calls[0]["reconstruct_sources"] is False
    assert set(cohort_calls[0]["identities"]) == set(eligible_exports)


def test_campaign_ledger_rejects_one_missing_combination(tmp_path: Path) -> None:
    exports = _four_exports(tmp_path)
    contract = build_campaign_contract(exports, verifier=_fake_verifier)
    assert contract["status"] == "ready"
    assert [item["execution_ordinal"] for item in contract["combinations"]] == list(
        range(1, 17)
    )
    assert [item["model_key"] for item in contract["combinations"][:8:2]] == list(
        MODEL_KEYS
    )
    assert [item["model_key"] for item in contract["combinations"][8::2]] == list(
        reversed(MODEL_KEYS)
    )
    for model in MODEL_KEYS:
        orders = []
        for board in ["esp32c3", "arduino_r4"]:
            orders.append(
                [
                    item["transport"]
                    for item in contract["combinations"]
                    if item["model_key"] == model and item["board"] == board
                ]
            )
        assert orders[1] == list(reversed(orders[0]))
    tampered_contract = json.loads(json.dumps(contract))
    tampered_contract["combinations"][0], tampered_contract["combinations"][1] = (
        tampered_contract["combinations"][1],
        tampered_contract["combinations"][0],
    )
    tampered_contract.pop("contract_id")
    tampered_contract["contract_id"] = canonical_json_sha256(tampered_contract)
    with pytest.raises(RuntimeError, match="Campaign-derived field"):
        validate_campaign_contract(tampered_contract)
    sessions = [
        {
            "combination_id": item["combination_id"],
            "stages": [{} for _ in FINAL_STAGES],
            "row_totals": {
                "all_rows": 59_321,
                "timing_rows": 3_000,
                "full_rows": 56_301,
                "warmup_rows": 10,
                "smoke_rows": 10,
            },
        }
        for item in contract["combinations"]
    ]
    totals = validate_campaign_session_ledger(contract, sessions)
    assert totals["balanced_timing_rows"] == 48_000
    assert totals["full_exact_replay_rows"] == 900_816
    with pytest.raises(RuntimeError, match="matrix is incomplete"):
        validate_campaign_session_ledger(contract, sessions[:-1])


def test_usb_only_contract_has_exact_six_session_gate_eligible_matrix(
    tmp_path: Path,
) -> None:
    exports = _four_exports(tmp_path)
    blocked_key = "student_B_scratch"
    blocked = {
        "status": "blocked",
        "quality_gates_passed": False,
        "identity": {
            "protocol": "wsnds_feature_group_split_train_only_scaler_10seed_v2",
            "seed": 42,
            "student": "B",
            "route": "scratch",
        },
        "reason": "fixed-point quality gate failed",
    }
    blocked["identity"]["blocked_audit_id"] = canonical_json_sha256(
        blocked["identity"]
    )
    blocked["audit_payload_sha256"] = canonical_json_sha256(blocked)
    blocked_path = tmp_path / "blocked.json"
    _write_json(blocked_path, blocked)
    sources: dict[str, Path | dict[str, Path]] = {
        key: path for key, path in exports.items() if key != blocked_key
    }
    sources[blocked_key] = {"blocked_audit": blocked_path}

    contract = build_campaign_contract(
        sources,
        transports=("usb_serial",),
        verifier=_fake_verifier,
        blocked_verifier=_fake_blocked_verifier,
    )
    validate_campaign_contract(contract)

    assert contract["status"] == "ready_with_blocked_routes"
    assert contract["transports"] == ["usb_serial"]
    assert contract["expected_combination_count"] == 8
    assert contract["expected_eligible_combination_count"] == 6
    assert all(
        item["transport"] == "usb_serial"
        for item in contract["eligible_combinations"]
    )
    assert [
        item["combination_id"] for item in contract["eligible_combinations"]
    ] == [
        "student_A_scratch__esp32c3__usb_serial",
        "student_A_rf_kd__esp32c3__usb_serial",
        "student_B_rf_kd__esp32c3__usb_serial",
        "student_B_rf_kd__arduino_r4__usb_serial",
        "student_A_rf_kd__arduino_r4__usb_serial",
        "student_A_scratch__arduino_r4__usb_serial",
    ]
    assert contract["expected_eligible_rows"] == {
        "warmup_excluded": 60,
        "smoke": 60,
        "balanced_timing": 18_000,
        "full_exact_replay": 337_806,
        "all_device_inferences": 355_926,
    }
    sessions = [
        {
            "combination_id": item["combination_id"],
            "stages": [{} for _ in FINAL_STAGES],
            "row_totals": {
                "all_rows": 59_321,
                "timing_rows": 3_000,
                "full_rows": 56_301,
                "warmup_rows": 10,
                "smoke_rows": 10,
            },
        }
        for item in contract["eligible_combinations"]
    ]
    assert validate_campaign_session_ledger(contract, sessions)["session_count"] == 6


def test_contract_rejects_invalid_transport_scope(tmp_path: Path) -> None:
    exports = _four_exports(tmp_path)
    with pytest.raises(ValueError, match="nonempty and unique"):
        build_campaign_contract(exports, transports=(), verifier=_fake_verifier)
    with pytest.raises(ValueError, match="Unsupported"):
        build_campaign_contract(
            exports, transports=("bluetooth",), verifier=_fake_verifier
        )
    with pytest.raises(ValueError, match="canonical order"):
        build_campaign_contract(
            exports,
            transports=("wifi_udp", "usb_serial"),
            verifier=_fake_verifier,
        )
