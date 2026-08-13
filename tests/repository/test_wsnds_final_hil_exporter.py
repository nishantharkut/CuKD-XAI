from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from deployment.firmware_export.wsnds_final_hil import export_final_seed42 as exporter


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = (
    REPO_ROOT
    / "deployment"
    / "firmware_export"
    / "wsnds_final_hil"
    / "export_final_seed42.py"
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _refresh_manifest_inventory(root: Path) -> None:
    path = root / "final_export_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for item in manifest["files"]:
        member = root / item["path"]
        item["size_bytes"] = member.stat().st_size
        item["sha256"] = exporter.sha256_file(member)
    manifest.pop("manifest_payload_sha256", None)
    manifest["manifest_payload_sha256"] = exporter.canonical_json_sha256(manifest)
    _write_json(path, manifest)


def _fully_reseal_after_csv_tamper(root: Path) -> None:
    identity_path = root / "final_export_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    for item in identity["core_files"]:
        member = root / item["path"]
        item["size_bytes"] = member.stat().st_size
        item["sha256"] = exporter.sha256_file(member)
    identity.pop("export_id", None)
    identity["export_id"] = exporter.canonical_json_sha256(identity)
    _write_json(identity_path, identity)
    exporter._write_identity_header(root / "cukd_export_identity.h", identity)

    report_path = root / "final_export_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["identity"] = identity
    report.pop("report_payload_sha256", None)
    report["report_payload_sha256"] = exporter.canonical_json_sha256(report)
    _write_json(report_path, report)

    manifest_path = root / "final_export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["export_id"] = identity["export_id"]
    manifest["identity_canonical_sha256"] = exporter.canonical_json_sha256(identity)
    manifest["report_canonical_sha256"] = exporter.canonical_json_sha256(report)
    for item in manifest["files"]:
        member = root / item["path"]
        item["size_bytes"] = member.stat().st_size
        item["sha256"] = exporter.sha256_file(member)
    manifest.pop("manifest_payload_sha256", None)
    manifest["manifest_payload_sha256"] = exporter.canonical_json_sha256(manifest)
    _write_json(manifest_path, manifest)


@pytest.fixture(scope="module")
def valid_final_export(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("final_hil_verified") / "student_A_rf_kd"
    context = exporter.load_verified_context(
        REPO_ROOT / exporter.EXPECTED_RELATIVE_ROOT,
        REPO_ROOT / "data" / "wsnds" / "WSN-DS.csv",
        "A",
        "rf_kd",
    )
    exporter.export(context, root, "gcc")
    return root


def _copy_export(source: Path, destination_root: Path) -> Path:
    destination = destination_root / "export"
    shutil.copytree(source, destination)
    return destination


def test_cli_requires_explicit_student_route_and_output() -> None:
    args = exporter.parse_args(
        ["--student", "B", "--route", "scratch", "--output-dir", "out"]
    )
    assert args.student == "B"
    assert args.route == "scratch"
    with pytest.raises(SystemExit):
        exporter.parse_args(["--student", "A", "--output-dir", "out"])


def test_frozen_quality_gates_use_absolute_drop_only() -> None:
    assert exporter.MINIMUM_FIXED_FP32_AGREEMENT == 0.99
    assert exporter.MAXIMUM_ABSOLUTE_MACRO_F1_DROP == 0.015
    exporter.enforce_quality_gates(0.99, 0.015)
    with pytest.raises(RuntimeError, match="agreement"):
        exporter.enforce_quality_gates(0.989999, 0.0)
    with pytest.raises(RuntimeError, match="macro-F1"):
        exporter.enforce_quality_gates(1.0, 0.015001)
    with pytest.raises(RuntimeError, match="finite"):
        exporter.enforce_quality_gates(float("nan"), 0.0)


def test_source_uses_explicit_legacy_imports_and_has_no_secondary_gate() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "importlib" not in source
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    numeric_import = next(
        node
        for node in imports
        if node.module
        == "deployment.firmware_export.wsnds_rfkd_hil.export_wsnds_student_a_rfkd_int8"
    )
    imported_names = {alias.name for alias in numeric_import.names}
    assert {
        "calibrate_quantized_layers",
        "generate_e2e_artifacts",
        "simulate_fixed_point_inference",
        "simulate_integer_preprocess_q",
        "write_header",
    } <= imported_names
    assert all(alias.name != "*" for alias in numeric_import.names)
    assert "ORIGINAL_CODE_LEVEL" not in source
    assert "informational_gate" not in source
    assert "verify_final_export(staging, cc=cc)" in source
    assert "verify_final_export(final_dir, cc=cc)" in source


def test_wrong_confirmation_root_is_rejected_before_consumption(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="must be exactly"):
        exporter.load_verified_context(
            tmp_path,
            REPO_ROOT / "data" / "wsnds" / "WSN-DS.csv",
            "A",
            "scratch",
        )


@pytest.mark.parametrize(
    ("student", "route", "expected_state", "expected_kd"),
    [
        (
            "A",
            "scratch",
            "5942a687a5d55d3146188596d184cbfe3c14e01f117b39acecc2cfb4755a6e3f",
            None,
        ),
        (
            "A",
            "rf_kd",
            "32dcb823f1602595fee5a60906a31eae5cec1b2df5c65bb091580c1f002ac36a",
            {"T": 4.0, "alpha": 0.7},
        ),
        (
            "B",
            "scratch",
            "3c2e132d4c9ba2e487dead7db95b739108196292689d68be149cb682aaa2f1ed",
            None,
        ),
        (
            "B",
            "rf_kd",
            "bddd7b25bb8b8faaef8a3ea67dca8d95de95384f86c9a44ab954d85a5fc3b1bf",
            {"T": 4.0, "alpha": 0.7},
        ),
    ],
)
def test_final_seed42_lineage_and_prediction_reproduction(
    student: str,
    route: str,
    expected_state: str,
    expected_kd: dict[str, float] | None,
) -> None:
    context = exporter.load_verified_context(
        REPO_ROOT / exporter.EXPECTED_RELATIVE_ROOT,
        REPO_ROOT / "data" / "wsnds" / "WSN-DS.csv",
        student,
        route,
    )
    assert context["trained_state_sha256"] == expected_state
    assert context["kd_hyperparameters"] == expected_kd
    assert len(context["fp32_predictions"]) == 56301
    if route == "scratch":
        assert context["teacher_soft_target_provenance"] is None
        assert context["teacher_probability_path"] is None
    else:
        assert context["teacher_soft_target_provenance"]["rf_seed"] == 42
        assert context["teacher_probability_path"].is_file()


def test_complete_export_verifies_and_uses_canonical_names(
    valid_final_export: Path,
) -> None:
    result = exporter.verify_final_export(valid_final_export)
    assert result["status"] == "passed"
    assert result["test_rows"] == 56301
    assert (valid_final_export / "final_export_manifest.json").is_file()
    assert (valid_final_export / "final_export_report.json").is_file()
    assert (valid_final_export / "final_export_identity.json").is_file()
    assert not (valid_final_export / "strict_export_manifest.json").exists()
    assert not (valid_final_export / "strict_export_report.json").exists()
    manifest = json.loads(
        (valid_final_export / "final_export_manifest.json").read_text(encoding="utf-8")
    )
    assert not any(item["path"].lower().endswith(".exe") for item in manifest["files"])
    host = json.loads(
        (valid_final_export / "final_export_report.json").read_text(encoding="utf-8")
    )["host_equivalence"]
    assert host["temporary_executable_retained"] is False
    assert "verifier-native temporary executable" in host["verification_contract"]


def test_verifier_rejects_extra_unlisted_file(
    valid_final_export: Path, tmp_path: Path
) -> None:
    tampered = _copy_export(valid_final_export, tmp_path)
    (tampered / "unlisted.txt").write_text("extra", encoding="ascii")
    with pytest.raises(RuntimeError, match="extra or unlisted"):
        exporter.verify_final_export(tampered)


def test_verifier_rejects_extra_manifest_listed_file_after_rehash(
    valid_final_export: Path, tmp_path: Path
) -> None:
    tampered = _copy_export(valid_final_export, tmp_path)
    extra = tampered / "listed_but_not_allowed.txt"
    extra.write_text("extra", encoding="ascii")
    manifest_path = tampered / "final_export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"].append({
        "path": extra.name,
        "size_bytes": extra.stat().st_size,
        "sha256": exporter.sha256_file(extra),
    })
    manifest["file_count_excluding_manifest"] = len(manifest["files"])
    manifest.pop("manifest_payload_sha256", None)
    manifest["manifest_payload_sha256"] = exporter.canonical_json_sha256(manifest)
    _write_json(manifest_path, manifest)
    with pytest.raises(RuntimeError, match="exact allowed inventory"):
        exporter.verify_final_export(tampered)


def test_verifier_rejects_report_tamper_after_outer_manifest_rehash(
    valid_final_export: Path, tmp_path: Path
) -> None:
    tampered = _copy_export(valid_final_export, tmp_path)
    report_path = tampered / "final_export_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["gates"]["quality_gates_passed"] = False
    _write_json(report_path, report)
    _refresh_manifest_inventory(tampered)
    with pytest.raises(RuntimeError, match="report canonical"):
        exporter.verify_final_export(tampered)


def test_verifier_rejects_sealed_source_tamper_after_manifest_rehash(
    valid_final_export: Path, tmp_path: Path
) -> None:
    tampered = _copy_export(valid_final_export, tmp_path)
    source = tampered / "source_snapshot" / "c" / "cukd_model.c"
    source.write_text(source.read_text(encoding="ascii") + "\n/* tampered */\n", encoding="ascii")
    _refresh_manifest_inventory(tampered)
    with pytest.raises(RuntimeError, match="Sealed source hash"):
        exporter.verify_final_export(tampered)


def test_verifier_rejects_coherently_resealed_model_header(
    valid_final_export: Path, tmp_path: Path
) -> None:
    tampered = _copy_export(valid_final_export, tmp_path)
    model_header = tampered / "model_weights.h"
    model_header.write_text(
        model_header.read_text(encoding="ascii").replace(
            "Generated from final lineage:", "Resealed but behavior-preserving:"
        ),
        encoding="ascii",
    )
    _fully_reseal_after_csv_tamper(tampered)
    with pytest.raises(RuntimeError, match="trusted reconstruction: model_weights.h"):
        exporter.verify_final_export(tampered)


def test_verifier_rejects_non_dense_rows_after_all_outer_hashes_are_resealed(
    valid_final_export: Path, tmp_path: Path
) -> None:
    tampered = _copy_export(valid_final_export, tmp_path)
    replay_path = tampered / "hil_replay_vectors.csv"
    replay = pd.read_csv(replay_path)
    replay.loc[0, "row_id"] = 1
    replay.to_csv(replay_path, index=False)
    _fully_reseal_after_csv_tamper(tampered)
    with pytest.raises(RuntimeError, match="row IDs are not dense"):
        exporter.verify_final_export(tampered)


def test_verifier_rejects_reference_argmax_tamper_after_resealing(
    valid_final_export: Path, tmp_path: Path
) -> None:
    tampered = _copy_export(valid_final_export, tmp_path)
    reference_path = tampered / "hil_reference_predictions.csv"
    reference = pd.read_csv(reference_path)
    reference.loc[0, "fixed_pred"] = (int(reference.loc[0, "fixed_pred"]) + 1) % 5
    reference.to_csv(reference_path, index=False)
    _fully_reseal_after_csv_tamper(tampered)
    with pytest.raises(RuntimeError, match="fixed-logit argmax"):
        exporter.verify_final_export(tampered)


def test_verifier_rejects_test_source_index_hash_tamper_after_resealing(
    valid_final_export: Path, tmp_path: Path
) -> None:
    tampered = _copy_export(valid_final_export, tmp_path)
    replay_path = tampered / "hil_replay_vectors.csv"
    reference_path = tampered / "hil_reference_predictions.csv"
    replay = pd.read_csv(replay_path)
    reference = pd.read_csv(reference_path)
    used = set(replay["source_row_index"].astype(int).tolist())
    replacement = next(index for index in range(374661) if index not in used)
    replay.loc[0, "source_row_index"] = replacement
    reference.loc[0, "source_row_index"] = replacement
    replay.to_csv(replay_path, index=False)
    reference.to_csv(reference_path, index=False)
    _fully_reseal_after_csv_tamper(tampered)
    with pytest.raises(RuntimeError, match="source-index content hash"):
        exporter.verify_final_export(tampered)


def test_student_b_scratch_has_machine_readable_honest_blocked_audit(
    tmp_path: Path,
) -> None:
    output = tmp_path / "blocked_export"
    audit_path = tmp_path / "student_B_scratch_blocked.json"
    with pytest.raises(exporter.FinalQualityGateError):
        exporter.main([
            "--student", "B",
            "--route", "scratch",
            "--output-dir", str(output),
            "--blocked-audit-json", str(audit_path),
        ])
    assert not output.exists()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    stored_hash = audit.pop("audit_payload_sha256")
    assert stored_hash == exporter.canonical_json_sha256(audit)
    assert audit["status"] == "blocked"
    assert audit["frozen_test_quality_gate_assessments_this_export_invocation"] == 1
    assert audit["selected_policy_python_fixed_test_forward_computations_before_gate"] == 3
    assert audit["claims_first_ever_test_evaluation"] is False
    assert audit["historical_test_observations_preceded_this_invocation"] is True
    policy = audit["quantization_policy"]
    assert policy["selection_uses_test_data"] is False
    assert policy["applied_uniformly_to_all_student_route_exports"] is True
    assert "post-hoc" in policy["development_status"]
    assert policy["validation_gate_passing_candidate_count"] == 0
    assert policy["selection_status"] == "no_candidate_met_validation_gates_baseline_frozen"
    assert audit["fixed_vs_fp32_agreement"] < 0.99
    assert audit["absolute_macro_f1_drop"] > 0.015
    verified = exporter.verify_blocked_audit(audit_path)
    assert verified["status"] == "blocked_verified"

    tampered = json.loads(audit_path.read_text(encoding="utf-8"))
    tampered["identity"]["test_source_indices_sha256"] = "0" * 64
    identity_payload = dict(tampered["identity"])
    identity_payload.pop("blocked_audit_id")
    tampered["identity"]["blocked_audit_id"] = exporter.canonical_json_sha256(
        identity_payload
    )
    tampered.pop("audit_payload_sha256")
    tampered["audit_payload_sha256"] = exporter.canonical_json_sha256(tampered)
    _write_json(audit_path, tampered)
    with pytest.raises(RuntimeError, match="lineage hash mismatch"):
        exporter.verify_blocked_audit(audit_path)
