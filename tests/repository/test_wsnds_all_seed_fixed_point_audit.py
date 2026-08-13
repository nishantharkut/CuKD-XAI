from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from deployment.firmware_export.wsnds_final_hil import audit_all_seeds as audit
from deployment.firmware_export.wsnds_final_hil import export_final_seed42 as final_export


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _synthetic_shared() -> dict:
    return {
        "dataset": {"dataset_sha256": "1" * 64},
        "preprocessing": {
            "split_indices_sha256": "2" * 64,
            "scaler_sha256": "3" * 64,
        },
        "test_source_indices_sha256": "4" * 64,
        "root_artifact_manifest_sha256": "5" * 64,
        "execution_contract_sha256": "6" * 64,
        "preprocessing_contract_sha256": "7" * 64,
        "split": {"group_audit": {
            "train_validation_feature_overlap": 0,
            "train_test_feature_overlap": 0,
            "validation_test_feature_overlap": 0,
        }},
    }


def _synthetic_record(seed: int, student: str, route: str) -> dict:
    return audit._with_payload_hash({
        "status": "audit_error",
        "failure_class": "infrastructure_or_logic_exception",
        "requested_identity": {
            "model_id": audit.model_id(seed, student, route),
            "seed": seed,
            "student": student,
            "route": route,
        },
        "failure": {
            "exception_type": "SyntheticAuditError",
            "message": "focused test record",
        },
    }, "model_record_payload_sha256")


def _successful_c_evidence() -> dict:
    return {
        "status": "passed",
        "rows": audit.EXPECTED_SPLIT_SIZES["test"],
        "failed_phase": None,
        "failed_outcome": None,
        "phases": {
            phase: {
                "attempted": True,
                "status": "completed",
                "outcome": outcome,
            }
            for phase, outcome in audit.C_SUCCESS_OUTCOMES.items()
        },
        "preprocessed_inputs_exact": True,
        "fixed_logits_exact": True,
        "fixed_predictions_exact": True,
        "payload_sha256": "8" * 64,
        "raw_inputs_content_sha256": "a" * 64,
        "preprocessed_content_sha256": "b" * 64,
        "fixed_logits_content_sha256": "c" * 64,
        "fixed_predictions_content_sha256": "d" * 64,
        "sealed_source_sha256": {
            "cukd_preprocess.c": "e" * 64,
            "cukd_model.c": "f" * 64,
            "all_seed_stream_self_test.c": "0" * 64,
        },
        "generation_host_evidence": {
            "compiler_requested": "synthetic-cc",
            "compiler_version": {"returncode": 0, "stdout": "cc", "stderr": ""},
            "compile": {"returncode": 0, "stdout": "", "stderr": ""},
            "self_test": {"returncode": 0, "stdout": "", "stderr": ""},
            "temporary_executable_sha256": "9" * 64,
            "temporary_executable_retained": False,
        },
    }


def _failed_c_evidence(
    failed_phase: str = "compilation",
    failed_outcome: str = "compiler_nonzero_exit",
) -> dict:
    phases = audit._new_c_phase_ledger()
    failed_index = audit.C_PHASES.index(failed_phase)
    for phase in audit.C_PHASES[:failed_index]:
        phases[phase] = {
            "attempted": True,
            "status": "completed",
            "outcome": audit.C_SUCCESS_OUTCOMES[phase],
        }
    phases[failed_phase] = {
        "attempted": True,
        "status": "failed",
        "outcome": failed_outcome,
    }
    host: dict = {
        "compiler_requested": "synthetic-cc",
        "temporary_executable_retained": False,
    }
    if failed_index > 0:
        host["compiler_version"] = {
            "returncode": 0,
            "stdout": "cc",
            "stderr": "",
        }
    if failed_index > 1 or (
        failed_phase == "compilation" and failed_outcome == "compiler_nonzero_exit"
    ):
        host["compile"] = {
            "returncode": 0 if failed_index > 1 else 1,
            "stdout": "",
            "stderr": "compile failed" if failed_index == 1 else "",
        }
    if failed_phase == "verification":
        host["self_test"] = {
            "returncode": 1,
            "stdout": "",
            "stderr": "mismatch",
        }
    if failed_index > 1:
        host["temporary_executable_sha256"] = "9" * 64
    return {
        "status": "failed",
        "rows": audit.EXPECTED_SPLIT_SIZES["test"],
        "failed_phase": failed_phase,
        "failed_outcome": failed_outcome,
        "phases": phases,
        "failure": {
            "exception_type": "SyntheticCFailure",
            "message": "focused C failure",
        },
        "generation_host_evidence": host,
    }


def _bound_evidence(passed: bool) -> tuple[list[dict], list[dict]]:
    accumulator = [
        {
            "layer": index,
            "pre_rescale_absolute_bound": (
                1 if passed or index else int(np.iinfo(np.int32).max) + 1
            ),
            "output_shift": 0,
            "post_left_shift_absolute_bound": (
                1 if passed or index else int(np.iinfo(np.int32).max) + 1
            ),
            "int32_max": int(np.iinfo(np.int32).max),
            "passed": passed or index != 0,
        }
        for index in range(3)
    ]
    preprocess = [
        {
            "feature": index,
            "maximum_centered_absolute": 1,
            "inverse_scale_absolute": 1,
            "maximum_product_absolute": 1,
            "int64_max": int(np.iinfo(np.int64).max),
            "passed": True,
        }
        for index in range(17)
    ]
    return accumulator, preprocess


def _saturation_evidence(partition: str, rows: int) -> dict:
    return {
        "partition": partition,
        "rows_audited": rows,
        "chunk_size": 8192,
        "raw_input_saturation_count": 0,
        "weight_saturation_count": 0,
        "bias_saturation_count": 0,
        "integer_preprocess_saturation_count": 0,
        "parameter_layers": [
            {
                "layer": index,
                "source_prefix": f"net.{index * 2}",
                "weight_saturation_count": 0,
                "bias_saturation_count": 0,
            }
            for index in range(3)
        ],
        "activation_layers": [
            {
                "layer": index,
                "activation_saturation_count": 0,
                "minimum_before_clip": 0,
                "maximum_before_clip": 1,
            }
            for index in range(3)
        ],
        "activation_saturation_count": 0,
        "passed": True,
    }


def _completed_record(
    seed: int,
    student: str,
    route: str,
    *,
    numeric_passed: bool = True,
    c_evidence: dict | None = None,
) -> dict:
    accumulator, preprocess = _bound_evidence(numeric_passed)
    training = _saturation_evidence(
        "training calibration partition", audit.EXPECTED_SPLIT_SIZES["train"]
    )
    validation = _saturation_evidence(
        "validation partition", audit.EXPECTED_SPLIT_SIZES["validation"]
    )
    test = _saturation_evidence(
        "test partition", audit.EXPECTED_SPLIT_SIZES["test"]
    )
    gates = {
        "test_rows": audit.EXPECTED_SPLIT_SIZES["test"],
        "fixed_vs_fp32_agreement": 1.0,
        "minimum_fixed_vs_fp32_agreement": audit.MINIMUM_FIXED_FP32_AGREEMENT,
        "absolute_macro_f1_drop": 0.0,
        "maximum_absolute_macro_f1_drop": audit.MAXIMUM_ABSOLUTE_MACRO_F1_DROP,
        "accumulator_bounds_passed": numeric_passed,
        "preprocess_bounds_passed": True,
        "training_calibration_saturation_passed": True,
        "validation_calibration_saturation_passed": True,
        "test_saturation_passed": True,
        "standardized_input_bounds_passed": True,
        "fixed_vs_fp32_agreement_passed": True,
        "macro_f1_drop_passed": True,
        "zero_saturation_passed": True,
        "quality_gates_passed": numeric_passed,
    }
    gates["numeric_gate_ledger"] = {
        "accumulator_bounds": {
            "passed": numeric_passed,
            "evidence": accumulator,
        },
        "preprocess_bounds": {"passed": True, "evidence": preprocess},
        "training_calibration_saturation": {
            "passed": True,
            "evidence": training,
        },
        "validation_calibration_saturation": {
            "passed": True,
            "evidence": validation,
        },
        "test_saturation": {"passed": True, "evidence": test},
        "standardized_input_bounds": {
            "passed": True,
            "evidence": {"train": 0, "validation": 0, "test": 0},
        },
        "fixed_vs_fp32_agreement": {
            "passed": True,
            "evidence": {"comparison": "greater_than_or_equal"},
            "threshold": audit.MINIMUM_FIXED_FP32_AGREEMENT,
            "observed": 1.0,
        },
        "absolute_macro_f1_drop": {
            "passed": True,
            "evidence": {"comparison": "less_than_or_equal"},
            "threshold": audit.MAXIMUM_ABSOLUTE_MACRO_F1_DROP,
            "observed": 0.0,
        },
    }
    c_evidence = c_evidence or (
        _successful_c_evidence()
        if numeric_passed
        else audit._c_blocked_result(audit.EXPECTED_SPLIT_SIZES["test"])
    )
    gates["c_python_exact_equivalence_passed"] = c_evidence["status"] == "passed"
    payload = {
        "status": audit._status_from_gates(gates, c_evidence),
        "identity": {
            "model_id": audit.model_id(seed, student, route),
            "seed": seed,
            "student": student,
            "route": route,
        },
        "test_evaluation_scope": audit._c_scope(c_evidence),
        "gates": gates,
        "c_equivalence": c_evidence,
    }
    return audit._with_payload_hash(payload, "model_record_payload_sha256")


def _build_synthetic_complete_output(root: Path, shared: dict) -> dict:
    root.mkdir()
    snapshots = audit._source_snapshots(root)
    contract = audit._new_contract(shared, snapshots)
    _write_json(root / audit.CONTRACT_NAME, contract)
    model_root = root / audit.MODEL_DIR_NAME
    model_root.mkdir()
    for spec in audit.expected_model_matrix():
        record = _synthetic_record(
            spec["seed"], spec["student"], spec["route"]
        )
        _write_json(model_root / f"{spec['model_id']}.json", record)
    audit._finalize(root, contract)
    return contract


def _reseal_output(
    root: Path, contract: dict, *, claimed_status: str | None = None
) -> None:
    progress = audit._progress(root, contract["audit_contract_id"])
    _write_json(root / audit.PROGRESS_NAME, progress)
    status_counts: dict[str, int] = {}
    for item in progress["completed_models"]:
        status = item["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    report = audit._with_payload_hash({
        "audit_protocol_id": audit.AUDIT_PROTOCOL_ID,
        "audit_contract_id": contract["audit_contract_id"],
        "status": claimed_status or audit._report_status(status_counts),
        "model_count": audit.MODEL_COUNT,
        "status_counts": status_counts,
        "records": progress["completed_models"],
        "statistical_unit_disclosure": contract["statistical_unit_disclosure"],
        "post_hoc_method_disclosure": contract["quantization_contract"][
            "development_status"
        ],
    }, "report_payload_sha256")
    _write_json(root / audit.REPORT_NAME, report)
    inventory = audit.file_inventory(root, {audit.MANIFEST_NAME})
    manifest = audit._with_payload_hash({
        "audit_protocol_id": audit.AUDIT_PROTOCOL_ID,
        "audit_contract_id": contract["audit_contract_id"],
        "status": report["status"],
        "file_count_excluding_manifest": len(inventory),
        "files": inventory,
        "report_canonical_sha256": audit._canonical_hash(report),
    }, "manifest_payload_sha256")
    _write_json(root / audit.MANIFEST_NAME, manifest)


def test_exact_40_model_matrix_and_frozen_policy_aliases() -> None:
    matrix = audit.expected_model_matrix()
    assert len(matrix) == 40
    assert len({item["model_id"] for item in matrix}) == 40
    assert {item["seed"] for item in matrix} == set(final_export.EXPECTED_SEEDS)
    assert {item["student"] for item in matrix} == {"A", "B"}
    assert {item["route"] for item in matrix} == {"scratch", "rf_kd"}
    assert audit.select_quantization_policy is final_export.select_quantization_policy
    assert audit.MINIMUM_FIXED_FP32_AGREEMENT == 0.99
    assert audit.MAXIMUM_ABSOLUTE_MACRO_F1_DROP == 0.015


def test_policy_selection_api_cannot_accept_test_evidence() -> None:
    parameters = set(inspect.signature(audit.select_quantization_policy).parameters)
    assert parameters == {
        "layers",
        "preprocessing_metadata",
        "x_train",
        "x_train_raw",
        "x_validation",
        "x_validation_raw",
        "y_validation",
    }
    source = inspect.getsource(audit.evaluate_model)
    assert source.index("select_quantization_policy(") < source.index(
        "# Frozen test boundary"
    )
    assert "generate_e2e_artifacts" not in source


def test_cli_has_no_partial_model_or_weaker_gate_option() -> None:
    args = audit.parse_args(["--output-dir", "audit-out"])
    assert args.cc == "gcc"
    with pytest.raises(SystemExit):
        audit.parse_args(["--output-dir", "audit-out", "--seed", "42"])
    with pytest.raises(SystemExit):
        audit.parse_args(["--output-dir", "audit-out", "--agreement", "0.9"])


def test_normal_run_evaluates_each_model_once_and_resume_does_not_recompute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = _synthetic_shared()
    input_root = tmp_path / "input"
    input_root.mkdir()
    dataset_csv = input_root / "WSN-DS.csv"
    dataset_csv.write_text("test fixture\n", encoding="ascii")
    shared["confirmation_root"] = input_root
    shared["dataset_csv"] = dataset_csv
    monkeypatch.setattr(audit, "load_shared_lineage", lambda *_: shared)

    calls: list[tuple[int, str, str]] = []

    def evaluate(_shared, seed, student, route, _source_root, _cc):
        calls.append((seed, student, route))
        return _synthetic_record(seed, student, route)

    monkeypatch.setattr(audit, "evaluate_model", evaluate)
    output_root = tmp_path / "audit"
    audit.run_all_seed_audit(input_root, dataset_csv, output_root)
    assert calls == [
        (item["seed"], item["student"], item["route"])
        for item in audit.expected_model_matrix()
    ]

    audit.run_all_seed_audit(input_root, dataset_csv, output_root)
    assert len(calls) == audit.MODEL_COUNT


def test_stream_payload_is_compact_exact_and_little_endian(tmp_path: Path) -> None:
    rows = 3
    raw = np.arange(rows * 17, dtype=np.int32).reshape(rows, 17)
    preprocessed = np.arange(rows * 17, dtype=np.int16).reshape(rows, 17)
    logits = np.arange(rows * 5, dtype=np.int16).reshape(rows, 5)
    predictions = np.array([0, 1, 4], dtype=np.uint8)
    path = tmp_path / "equivalence.bin"
    digest = audit._write_equivalence_payload(
        path, raw, preprocessed, logits, predictions
    )
    assert digest == audit.sha256_file(path)
    assert path.stat().st_size == rows * (17 * 4 + 17 * 2 + 5 * 2 + 1)


def test_source_sealing_includes_native_harness_and_no_binary(tmp_path: Path) -> None:
    snapshots = audit._source_snapshots(tmp_path)
    paths = {item["snapshot_path"] for item in snapshots}
    assert "source_snapshot/c/all_seed_stream_self_test.c" in paths
    assert "source_snapshot/python/audit_all_seeds.py" in paths
    assert "source_snapshot/python/export_final_seed42.py" in paths
    assert not any(path.suffix.lower() in {".exe", ".out"} for path in tmp_path.rglob("*"))
    source = inspect.getsource(audit.run_native_c_equivalence)
    assert "TemporaryDirectory" in source
    assert "temporary_executable_retained" in source


def test_complete_synthetic_output_deep_verifies_and_rejects_resealed_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = _synthetic_shared()
    root = tmp_path / "audit"
    contract = _build_synthetic_complete_output(root, shared)
    monkeypatch.setattr(audit, "load_shared_lineage", lambda *_: shared)
    monkeypatch.setattr(
        audit,
        "evaluate_model",
        lambda _shared, seed, student, route, _source_root, _cc: _synthetic_record(
            seed, student, route
        ),
    )
    verified = audit.verify_all_seed_audit(root)
    assert verified == {
        "status": "verified",
        "model_count": 40,
        "status_counts": {"audit_error": 40},
    }

    target = root / audit.MODEL_DIR_NAME / "seed_42_student_A_scratch.json"
    record = json.loads(target.read_text(encoding="utf-8"))
    record["failure"]["message"] = "semantically tampered"
    record.pop("model_record_payload_sha256")
    record["model_record_payload_sha256"] = audit._canonical_hash(record)
    _write_json(target, record)
    _reseal_output(root, contract)
    with pytest.raises(RuntimeError, match="verified.seed_42_student_A_scratch"):
        audit.verify_all_seed_audit(root)


def test_progress_is_resumable_and_preserves_failed_records(tmp_path: Path) -> None:
    shared = _synthetic_shared()
    root = tmp_path / "resume"
    root.mkdir()
    snapshots = audit._source_snapshots(root)
    contract = audit._new_contract(shared, snapshots)
    _write_json(root / audit.CONTRACT_NAME, contract)
    model_root = root / audit.MODEL_DIR_NAME
    model_root.mkdir()
    record = _synthetic_record(42, "A", "scratch")
    _write_json(model_root / "seed_42_student_A_scratch.json", record)
    progress = audit._progress(root, contract["audit_contract_id"])
    assert progress["completed_count"] == 1
    assert progress["remaining_count"] == 39
    assert progress["completed_models"][0]["status"] == "audit_error"


def test_deep_verify_rejects_resealed_false_all_pass_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = _synthetic_shared()
    root = tmp_path / "false-all-pass"
    contract = _build_synthetic_complete_output(root, shared)
    _reseal_output(root, contract, claimed_status="complete_all_gates_passed")
    monkeypatch.setattr(audit, "load_shared_lineage", lambda *_: shared)
    with pytest.raises(
        RuntimeError,
        match="report status differs from recomputed per-model statuses",
    ):
        audit.verify_all_seed_audit(root)


def test_deep_verify_requires_manifest_report_status_agreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = _synthetic_shared()
    root = tmp_path / "status-disagreement"
    _build_synthetic_complete_output(root, shared)
    manifest_path = root / audit.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("manifest_payload_sha256")
    manifest["status"] = "complete_all_gates_passed"
    manifest = audit._with_payload_hash(manifest, "manifest_payload_sha256")
    _write_json(manifest_path, manifest)
    monkeypatch.setattr(audit, "load_shared_lineage", lambda *_: shared)
    with pytest.raises(RuntimeError, match="final documents disagree"):
        audit.verify_all_seed_audit(root)


def test_numeric_gate_failure_is_structured_gate_failed_and_tamper_rejected() -> None:
    record = _completed_record(42, "A", "scratch", numeric_passed=False)
    assert record["status"] == "gate_failed"
    assert record["c_equivalence"]["status"] == "blocked"
    assert set(record["gates"]["numeric_gate_ledger"]) == {
        "accumulator_bounds",
        "preprocess_bounds",
        "training_calibration_saturation",
        "validation_calibration_saturation",
        "test_saturation",
        "standardized_input_bounds",
        "fixed_vs_fp32_agreement",
        "absolute_macro_f1_drop",
    }
    assert all(
        type(entry["passed"]) is bool
        for entry in record["gates"]["numeric_gate_ledger"].values()
    )
    audit._validate_host_evidence(record)

    tampered = copy.deepcopy(record)
    tampered["status"] = "passed"
    tampered["gates"]["accumulator_bounds_passed"] = True
    tampered["gates"]["quality_gates_passed"] = True
    tampered["gates"]["numeric_gate_ledger"]["accumulator_bounds"][
        "passed"
    ] = True
    tampered.pop("model_record_payload_sha256")
    tampered = audit._with_payload_hash(tampered, "model_record_payload_sha256")
    with pytest.raises(RuntimeError, match="accumulator gate is inconsistent"):
        audit._validate_host_evidence(tampered)


def test_numeric_bound_helpers_return_false_ledgers_instead_of_raising() -> None:
    oversized_layer = {
        "weight": np.full((1, 70_000), 127, dtype=np.int8),
        "bias": np.zeros(1, dtype=np.int32),
        "output_shift": 0,
    }
    accumulator_passed, accumulator = audit._accumulator_gate([oversized_layer])
    assert accumulator_passed is False
    assert accumulator[0]["passed"] is False

    integer_metadata = {
        "scaler_mean_q": [0] * 17,
        "scaler_inv_scale_q": [int(np.iinfo(np.int64).max)] + [1] * 16,
    }
    preprocess_passed, preprocess = audit._preprocess_bounds_gate(integer_metadata)
    assert preprocess_passed is False
    assert preprocess[0]["passed"] is False


def test_stale_resealed_c_failure_is_recomputed_and_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = _synthetic_shared()
    root = tmp_path / "stale-c"
    contract = _build_synthetic_complete_output(root, shared)
    target = root / audit.MODEL_DIR_NAME / "seed_42_student_A_scratch.json"
    failed = _completed_record(
        42,
        "A",
        "scratch",
        c_evidence=_failed_c_evidence(),
    )
    assert failed["status"] == "c_equivalence_failed"
    assert failed["gates"]["c_python_exact_equivalence_passed"] is False
    audit._validate_host_evidence(failed)
    _write_json(target, failed)
    _reseal_output(root, contract)

    monkeypatch.setattr(audit, "load_shared_lineage", lambda *_: shared)

    def recompute(_shared, seed, student, route, _source_root, _cc):
        if (seed, student, route) == (42, "A", "scratch"):
            return _completed_record(seed, student, route)
        return _synthetic_record(seed, student, route)

    monkeypatch.setattr(audit, "evaluate_model", recompute)
    with pytest.raises(RuntimeError, match="verified.seed_42_student_A_scratch"):
        audit.verify_all_seed_audit(root)


def test_resume_refuses_resealed_snapshot_when_current_origin_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_repo = tmp_path / "repo"
    origin = fake_repo / "source.py"
    origin.parent.mkdir(parents=True)
    origin.write_text("version = 1\n", encoding="utf-8")
    monkeypatch.setattr(audit, "REPO_ROOT", fake_repo)
    selector_snapshot = "source_snapshot/python/export_final_seed42.py"
    monkeypatch.setattr(
        audit,
        "SOURCE_SNAPSHOT_SPECS",
        (
            (
                str(audit.final_export.SCRIPT_PATH),
                selector_snapshot,
            ),
            ("source.py", "source_snapshot/source.py"),
        ),
    )
    shared = _synthetic_shared()
    root = tmp_path / "mixed-source"
    audit._initialize_or_resume(root, shared)

    origin.write_text("version = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs from its sealed origin"):
        audit._initialize_or_resume(root, shared)


def test_resealed_compile_failure_cannot_claim_native_execution() -> None:
    record = _completed_record(
        42,
        "A",
        "rf_kd",
        c_evidence=_failed_c_evidence(),
    )
    assert record["test_evaluation_scope"] == {
        "selection_uses_train_and_validation_only": True,
        "python_frozen_test_evaluations_this_model_invocation": 1,
        "compiler_discovery_attempts_this_model_invocation": 1,
        "compiler_discovery_successes_this_model_invocation": 1,
        "compilation_attempts_this_model_invocation": 1,
        "successful_compilations_this_model_invocation": 0,
        "native_process_start_attempts_this_model_invocation": 0,
        "native_process_executions_this_model_invocation": 0,
        "exact_equivalence_verification_attempts_this_model_invocation": 0,
        "exact_equivalence_verifications_passed_this_model_invocation": 0,
        "claims_first_ever_test_observation": False,
        "historical_test_observations_preceded_this_audit": True,
    }
    tampered = copy.deepcopy(record)
    tampered["test_evaluation_scope"][
        "native_process_executions_this_model_invocation"
    ] = 1
    tampered.pop("model_record_payload_sha256")
    tampered = audit._with_payload_hash(tampered, "model_record_payload_sha256")
    with pytest.raises(RuntimeError, match="scope disagrees with attempted phases"):
        audit._validate_host_evidence(tampered)
