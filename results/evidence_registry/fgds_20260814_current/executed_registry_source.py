"""Build a fail-closed evidence registry for the current FG-DS result lineages.

The registry does not recompute models or modify historical artifacts. It verifies
the preserved multi-seed, deployment, runtime, XAI, fixed-point, USB HIL,
Wi-Fi HIL, and Edge artifacts before recording which claims each lineage can
support.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]

TEN_SEED_PROTOCOL = "wsnds_feature_group_split_train_only_scaler_10seed_v2"
DEPLOYMENT_PROTOCOL = "wsnds_feature_group_split_train_only_scaler_5seed_v1"
RUNTIME_PROTOCOL = "wsnds_fgds_seed42_exact_runtime_v1"
WIRELESS_PROTOCOL = "cukd_fgds_wifi_udp_session_v2"
BEHAVIORAL_TRANSFER_PROTOCOL = "wsnds_fgds_behavioral_transfer_logits_10seed_v5"
MULTISPLIT_PROTOCOL = "wsnds_fgds_multisplit_core_10x2_v2"
ALL_SEED_FIXED_POINT_PROTOCOL = "wsnds_all_seed_software_fixed_point_audit_v1"
FINAL_HIL_REPORT_SCHEMA = "cukd_final_hil_archive_report_v1"
FINAL_HIL_ARCHIVE_SCHEMA = "cukd_final_hil_portable_archive_v2"
FINAL_HIL_REPORT_MANIFEST_SCHEMA = "cukd_final_hil_report_artifact_manifest_v1"
FINAL_HIL_LOCATOR_SCHEMA = "cukd_final_hil_external_archive_locator_v1"
EXPECTED_SEEDS = [42, 123, 456, 789, 1001, 2024, 3141, 5678, 8192, 9999]
EXPECTED_TEST_ROWS = 56_301
EXPECTED_DATASET_SHA256 = "c65d05b983a85753bd62b6f76c5739fc52fe0c14cbb7644255cee4742f5ff7c9"
EXPECTED_LABELS = [
    "esp32c3_student_A",
    "esp32c3_student_B",
    "arduino_r4_student_A",
    "arduino_r4_student_B",
]

TEN_SEED_ROOT = (
    REPO_ROOT
    / "results/wsnds/confirmation_runs_v2/local_feature_group_10seed_20260811"
    / "feature_group_10seed"
)
TEN_SEED_ANALYSIS_ROOT = TEN_SEED_ROOT.parent / "feature_group_10seed_analysis"
DEPLOYMENT_ROOT = (
    REPO_ROOT
    / "results/wsnds/confirmation_runs_v2/remote_winterfell_feature_group_5seed_20260805"
    / "feature_group_5seed"
)
RUNTIME_ROOT = REPO_ROOT / "results/runtime/onnx_openvino/wsnds/fgds_seed42_exact"
USB_ROOT = REPO_ROOT / "results/hardware_hil/fgds_seed42"
USB_REPORT_ROOT = USB_ROOT / "final_report_20260810T204825Z"
WIRELESS_ROOT = REPO_ROOT / "results/wireless_hil/fgds_seed42"
WIRELESS_REPORT_ROOT = WIRELESS_ROOT / "final_report_lineage_sealed_v2"
EDGE_ROOT = REPO_ROOT / "results/leftover_e2e_closure/04_edge_group_aware"
EVIDENCE_COMPLETION_ROOT = (
    REPO_ROOT / "results/wsnds/evidence_completion_20260811"
)
FULL_ROUTE_ROOT = EVIDENCE_COMPLETION_ROOT / "fgds_controlled_full_routes_10seed_v2"
SENSITIVITY_ROOT = (
    EVIDENCE_COMPLETION_ROOT / "fgds_rfkd_hyperparameter_sensitivity_10seed_v1"
)
GROUP_BALANCED_ROOT = (
    EVIDENCE_COMPLETION_ROOT / "fgds_group_balanced_routes_10seed_v1"
)
SHAP_ROOT = EVIDENCE_COMPLETION_ROOT / "fgds_seed42_reconstructed_teacher_shap_v3"
SHAP_V2_PARTIAL_ROOT = (
    EVIDENCE_COMPLETION_ROOT / "fgds_seed42_reconstructed_teacher_shap_v2"
)
FIXED_POINT_REFINEMENT_ROOT = (
    EVIDENCE_COMPLETION_ROOT / "fgds_fixed_point_refinement_seed42"
)
MSP430_STATIC_ROOT = REPO_ROOT / "deployment/msp430/current_fgds_static/artifacts"
POST_REGISTRY_EVIDENCE_ROOT = REPO_ROOT / "results/wsnds/evidence_completion_20260812"
BEHAVIORAL_TRANSFER_ROOT = (
    POST_REGISTRY_EVIDENCE_ROOT / "fgds_behavioral_transfer_logits_10seed_v5"
)
MULTISPLIT_ROOT = POST_REGISTRY_EVIDENCE_ROOT / "fgds_multisplit_core_10x2_v2"
ALL_SEED_FIXED_POINT_ROOT = (
    REPO_ROOT
    / "results/wsnds/evidence_completion_20260813"
    / "fgds_all_seed_fixed_point_audit_v1"
)
FINAL_HIL_REPORT_ROOT = (
    REPO_ROOT
    / "results/hardware_hil/final_fgds_seed42_v1"
    / "final_campaign_usb_v1"
)
FINAL_HIL_CAMPAIGN_ROOT = (
    REPO_ROOT / "results/hardware_hil/final_fgds_seed42_v1/campaign"
)
CONTROLLED_XAI_ROOT = (
    POST_REGISTRY_EVIDENCE_ROOT / "fgds_controlled_xai_transfer_10seed_v1"
)
EXTERNAL_EVIDENCE_ARCHIVE_ROOT = REPO_ROOT.parent / "Research_Evidence_Archives"
REGISTRY_ID = "cukd_fgds_evidence_registry_20260814_v3"
PREDECESSOR_REGISTRY_ID = "cukd_fgds_evidence_registry_20260812_v2"
PREDECESSOR_REGISTRY_ROOT = (
    REPO_ROOT / "results/evidence_registry/fgds_20260812_complete"
)
DEFAULT_OUTPUT = REPO_ROOT / "results/evidence_registry/fgds_20260814_current"

FULL_ROUTE_PROTOCOL = "wsnds_feature_group_disjoint_controlled_full_routes_10seed_v2"
SENSITIVITY_PROTOCOL = "wsnds_fgds_rfkd_hyperparameter_sensitivity_10seed_v1"
GROUP_BALANCED_PROTOCOL = "wsnds_fgds_group_balanced_route_sensitivity_10seed_v1"
SHAP_PROTOCOL = "wsnds_fgds_seed42_reconstructed_teacher_permutation_shap_v3"
FIXED_POINT_REFINEMENT_PROTOCOL = (
    "wsnds_fgds_seed42_frozen_fixed_point_refinement_v1"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Required JSON is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return sha256_bytes(encoded)


def verify_recorded_file(
    path: Path,
    expected_sha256: str,
    expected_size: int | None = None,
) -> str:
    """Verify exact bytes, with one explicit fallback for Git CRLF checkout conversion."""
    raw = path.read_bytes()
    if sha256_bytes(raw) == expected_sha256 and (
        expected_size is None or len(raw) == expected_size
    ):
        return "exact_bytes"
    require(
        path.suffix.lower() in {".csv", ".json", ".log", ".md", ".txt"},
        f"Recorded binary artifact differs: {path}",
    )
    normalized = raw.replace(b"\r\n", b"\n")
    require(normalized != raw, f"Recorded artifact differs without CRLF conversion: {path}")
    require(
        sha256_bytes(normalized) == expected_sha256
        and (expected_size is None or len(normalized) == expected_size),
        f"Recorded artifact differs after CRLF normalization: {path}",
    )
    return "working_tree_crlf_normalized_to_recorded_lf"


def repo_path(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def artifact(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Required artifact is missing: {path}")
    return {
        "path": repo_path(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def recorded_artifact(
    path: Path,
    expected_sha256: str,
    expected_size: int | None = None,
) -> dict[str, Any]:
    mode = verify_recorded_file(path, expected_sha256, expected_size)
    raw = path.read_bytes()
    recorded_size = (
        expected_size
        if expected_size is not None
        else len(raw if mode == "exact_bytes" else raw.replace(b"\r\n", b"\n"))
    )
    return {
        "path": repo_path(path),
        "size_bytes": recorded_size,
        "sha256": expected_sha256,
        "verification_mode": mode,
    }


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def verify_inventory(
    root: Path,
    manifest_name: str,
    accepted_statuses: set[str],
) -> dict[str, Any]:
    manifest_path = root / manifest_name
    manifest = read_json(manifest_path)
    require(
        str(manifest.get("status")) in accepted_statuses,
        f"Inventory status is not accepted: {manifest_path}",
    )
    entries = manifest.get("files")
    require(isinstance(entries, list) and entries, f"Empty inventory: {manifest_path}")
    require(
        manifest.get("file_count_excluding_manifest") == len(entries),
        f"Inventory count mismatch: {manifest_path}",
    )

    expected_paths: set[str] = set()
    verification_modes: dict[str, str] = {}
    resolved_root = root.resolve()
    for entry in entries:
        require(isinstance(entry, dict), f"Invalid inventory entry: {manifest_path}")
        relative = entry.get("path")
        require(isinstance(relative, str) and relative, f"Invalid inventory path")
        relative_path = Path(relative)
        require(
            not relative_path.is_absolute() and ".." not in relative_path.parts,
            f"Inventory path escapes root: {relative}",
        )
        normalized = relative_path.as_posix()
        require(normalized not in expected_paths, f"Duplicate inventory path: {relative}")
        expected_paths.add(normalized)
        path = (root / relative_path).resolve()
        try:
            path.relative_to(resolved_root)
        except ValueError as exc:
            raise RuntimeError(f"Inventory path escapes root: {relative}") from exc
        require(path.is_file(), f"Inventoried file is missing: {path}")
        verification_modes[normalized] = verify_recorded_file(
            path,
            str(entry.get("sha256")),
            int(entry.get("size_bytes")),
        )

    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    require(
        actual_paths == expected_paths,
        "Inventory differs from disk for "
        f"{manifest_path}; missing={sorted(expected_paths - actual_paths)}, "
        f"unexpected={sorted(actual_paths - expected_paths)}",
    )
    return {
        "manifest": artifact(manifest_path),
        "verified_files": len(entries),
        "status": manifest["status"],
        "protocol_id": manifest.get("protocol_id"),
        "verification_modes": {
            "exact_bytes": sum(mode == "exact_bytes" for mode in verification_modes.values()),
            "working_tree_crlf_normalized_to_recorded_lf": sum(
                mode == "working_tree_crlf_normalized_to_recorded_lf"
                for mode in verification_modes.values()
            ),
        },
    }


def verify_canonical_id(payload: dict[str, Any], id_field: str, label: str) -> str:
    recorded = payload.get(id_field)
    require(
        isinstance(recorded, str) and len(recorded) == 64,
        f"{label} lacks a valid {id_field}",
    )
    comparable = dict(payload)
    comparable.pop(id_field, None)
    require(
        recorded == canonical_json_sha256(comparable),
        f"{label} canonical {id_field} is invalid",
    )
    return recorded


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def summarize(values: list[float]) -> dict[str, Any]:
    require(bool(values), "Cannot summarize an empty list")
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def exact_signed_rank(differences: list[float]) -> dict[str, Any]:
    nonzero = [value for value in differences if value != 0.0]
    zero_count = len(differences) - len(nonzero)
    if not nonzero:
        return {
            "statistic": 0.0,
            "p_value_two_sided_exact": 1.0,
            "nonzero_pairs": 0,
            "positive_pairs": 0,
            "negative_pairs": 0,
            "zero_pairs": zero_count,
            "enumerated_sign_assignments": 1,
        }
    ranks = average_ranks([abs(value) for value in nonzero])
    total = sum(ranks)
    observed_positive = sum(
        rank for rank, value in zip(ranks, nonzero, strict=True) if value > 0
    )
    observed = min(observed_positive, total - observed_positive)
    assignments = 1 << len(ranks)
    extreme = 0
    for mask in range(assignments):
        positive = sum(rank for index, rank in enumerate(ranks) if mask & (1 << index))
        if min(positive, total - positive) <= observed + 1e-15:
            extreme += 1
    return {
        "statistic": observed,
        "p_value_two_sided_exact": extreme / assignments,
        "nonzero_pairs": len(nonzero),
        "positive_pairs": sum(value > 0 for value in nonzero),
        "negative_pairs": sum(value < 0 for value in nonzero),
        "zero_pairs": zero_count,
        "enumerated_sign_assignments": assignments,
    }


def compare_preprocessing_contracts(
    ten_seed_contract: dict[str, Any],
    deployment_contract: dict[str, Any],
) -> dict[str, Any]:
    comparable_keys = [
        "dataset_sha256",
        "dataset_shape",
        "feature_names",
        "class_names",
        "split_policy",
        "split_sizes",
        "split_class_counts",
        "split_hashes",
        "split_indices_sha256",
        "scaler_fit_partition",
        "scaler_fit_row_count",
        "scaler_fit_indices_sha256",
        "scaler_sha256",
        "transformed_split_hashes",
        "feature_overlap_audit",
    ]
    comparisons = {
        key: ten_seed_contract.get(key) == deployment_contract.get(key)
        for key in comparable_keys
    }
    require(all(comparisons.values()), "Five-seed and ten-seed preprocessing content differs")
    return {
        "all_semantic_fields_equal": True,
        "field_checks": comparisons,
        "binary_npz_files_expected_to_differ_across_hosts": True,
        "basis": "Content hashes and contract fields match; container-file hashes are not used as semantic identity.",
    }


def validate_ten_seed() -> tuple[dict[str, Any], dict[str, Any]]:
    run_inventory = verify_inventory(
        TEN_SEED_ROOT, "artifact_manifest.json", {"complete"}
    )
    analysis_inventory = verify_inventory(
        TEN_SEED_ANALYSIS_ROOT, "analysis_manifest.json", {"passed"}
    )
    analysis_path = TEN_SEED_ANALYSIS_ROOT / "feature_group_10seed_analysis.json"
    analysis = read_json(analysis_path)
    require(analysis.get("status") == "passed", "Ten-seed analysis did not pass")
    require(analysis.get("protocol_id") == TEN_SEED_PROTOCOL, "Ten-seed protocol mismatch")
    require(analysis.get("seeds") == EXPECTED_SEEDS, "Ten-seed seed set mismatch")
    require(analysis.get("seed_count") == 10, "Ten-seed count mismatch")
    require(
        analysis.get("test_rows_per_model_seed") == EXPECTED_TEST_ROWS,
        "Ten-seed test row count mismatch",
    )
    require(
        analysis.get("prediction_csv_files_recomputed") == 50,
        "Ten-seed prediction file count mismatch",
    )
    require(
        analysis.get("prediction_rows_recomputed") == 50 * EXPECTED_TEST_ROWS,
        "Ten-seed recomputed prediction row count mismatch",
    )
    preprocessing = read_json(TEN_SEED_ROOT / "preprocessing_contract.json")
    require(
        preprocessing.get("dataset_sha256") == EXPECTED_DATASET_SHA256,
        "Ten-seed dataset hash mismatch",
    )
    require(
        preprocessing.get("split_sizes")
        == {"train": 262_197, "validation": 56_163, "test": 56_301},
        "Ten-seed split sizes mismatch",
    )
    contracts = analysis.get("contracts", {})
    overlap = contracts.get("feature_overlap_audit", {})
    require(contracts.get("scaler_fit_partition") == "train only", "Scaler is not train only")
    for key in (
        "train_validation_feature_overlap",
        "train_test_feature_overlap",
        "validation_test_feature_overlap",
    ):
        require(overlap.get(key) == 0, f"Nonzero ten-seed feature overlap: {key}")

    selected_routes = {}
    for route in (
        "student_A_scratch",
        "student_A_rf_kd",
        "student_B_scratch",
        "student_B_rf_kd",
    ):
        source = analysis["routes"][route]
        selected_routes[route] = {
            "macro_f1_mean": source["mean"],
            "macro_f1_sample_std": source["sample_std"],
            "macro_f1_values": source["values"],
            "per_class_f1": source["per_class_f1"],
        }
    return analysis, {
        "status": "primary_multi_seed_evidence",
        "protocol_id": TEN_SEED_PROTOCOL,
        "evaluation_unit": "ten optimizer seeds on one fixed feature-group split",
        "seeds": EXPECTED_SEEDS,
        "test_rows_per_model_seed": EXPECTED_TEST_ROWS,
        "split_sizes": {"train": 262_197, "validation": 56_163, "test": 56_301},
        "scaler_fit_partition": "train only",
        "cross_partition_exact_raw_feature_group_overlap": 0,
        "conflicting_label_feature_groups": 3,
        "routes": selected_routes,
        "paired_tests": analysis["paired_tests"],
        "interpretation": "RF-KD is statistically indistinguishable from scratch for both students under this fixed clean protocol.",
        "boundary": "This estimates optimizer-seed variation on one fixed split. It does not estimate split-to-split uncertainty and it does not cover the full archived route matrix.",
        "analysis": artifact(analysis_path),
        "run_inventory": run_inventory,
        "analysis_inventory": analysis_inventory,
    }


def require_clean_contract_identity(
    execution: dict[str, Any],
    ten_seed_contract: dict[str, Any],
    label: str,
) -> None:
    for key in ("dataset_sha256", "split_indices_sha256", "scaler_sha256"):
        require(
            execution.get(key) == ten_seed_contract.get(key),
            f"{label} clean-data contract mismatch: {key}",
        )


def compact_route_metrics(route: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "ece_15_bin"):
        if metric in route:
            result[metric] = route[metric]
    for key in ("parameters", "fp32_parameter_payload_bytes"):
        if key in route:
            result[key] = route[key]
    return result


def validate_full_routes(ten_seed_contract: dict[str, Any]) -> dict[str, Any]:
    inventory = verify_inventory(FULL_ROUTE_ROOT, "artifact_manifest.json", {"complete"})
    aggregate_path = FULL_ROUTE_ROOT / "aggregate_results.json"
    execution_path = FULL_ROUTE_ROOT / "execution_contract.json"
    finalization_path = FULL_ROUTE_ROOT / "finalization_contract.json"
    aggregate = read_json(aggregate_path)
    execution = read_json(execution_path)
    finalization = read_json(finalization_path)

    require(aggregate.get("status") == "complete", "Full-route aggregate is incomplete")
    require(aggregate.get("protocol_id") == FULL_ROUTE_PROTOCOL, "Full-route protocol mismatch")
    require(execution.get("protocol_id") == FULL_ROUTE_PROTOCOL, "Full-route execution protocol mismatch")
    require(aggregate.get("seeds") == EXPECTED_SEEDS, "Full-route seed set mismatch")
    require(aggregate.get("seed_count") == 10, "Full-route seed count mismatch")
    require_clean_contract_identity(execution, ten_seed_contract, "Full-route")
    require(
        execution.get("evaluation_design", {}).get("cross_partition_exact_feature_group_overlap") == 0,
        "Full-route feature-group overlap is nonzero",
    )
    require(
        execution.get("student_routes")
        == [
            "D_Small_MLP",
            "E_KD_from_RF",
            "E2_KD_from_MLP",
            "F_KD_from_CL_MLP_fair",
            "F_KD_from_CL_MLP_ext",
            "G_KD_random_pacing",
            "I_KD_from_SMOTE_MLP",
            "J_CoDistill_RF_CL",
        ],
        "Full-route student route set mismatch",
    )
    require(
        finalization.get("status") == "complete"
        and finalization.get("schema_version") == 4
        and finalization.get("training_performed") is False,
        "Full-route finalization contract mismatch",
    )
    correction = finalization.get("correction", {})
    roundtrip = correction.get("roundtrip_audit", {})
    require(correction.get("model_training_or_prediction_changed") is False, "Full-route predictions changed during finalization")
    require(correction.get("non_ece_metrics_changed") is False, "Full-route non-ECE metrics changed during finalization")
    require(roundtrip.get("neural_artifact_count") == 180, "Full-route neural replay count mismatch")
    require(roundtrip.get("rf_artifact_count") == 10, "Full-route RF replay count mismatch")
    require(roundtrip.get("neural_float32_max_abs_delta") == 0.0, "Full-route neural replay differs")
    require(roundtrip.get("rf_persisted_roundtrip_max_abs_delta") == 0.0, "Full-route RF replay differs")
    inference = finalization.get("inference_correction", {})
    require(
        inference.get("audit", {}).get("test_count") == 26
        and inference.get("finalized_method") == "exact_signed_rank_enumeration",
        "Full-route exact inference finalization mismatch",
    )
    continuation = finalization.get("continuation_provenance", {})
    require(
        isinstance(continuation, dict) and continuation.get("status") == "complete",
        "Full-route continuation provenance is incomplete",
    )
    seed_evidence = finalization.get("seed_evidence", {})
    require(isinstance(seed_evidence, dict) and len(seed_evidence) == 10, "Full-route seed evidence mismatch")

    teacher_routes = {
        name: compact_route_metrics(value)
        for name, value in aggregate["teacher_aggregate"].items()
    }
    student_routes = {
        student: {
            name: compact_route_metrics(value)
            for name, value in routes.items()
        }
        for student, routes in aggregate["student_aggregate"].items()
    }
    require(sum(len(routes) for routes in student_routes.values()) == 16, "Full-route student aggregate count mismatch")
    require(len(teacher_routes) == 8, "Full-route teacher aggregate count mismatch")
    require(len(aggregate.get("paired_route_tests", {})) == 26, "Full-route paired-test count mismatch")

    return {
        "status": "current_controlled_route_evidence",
        "protocol_id": FULL_ROUTE_PROTOCOL,
        "seeds": EXPECTED_SEEDS,
        "statistical_unit": aggregate["statistical_unit"],
        "student_routes": student_routes,
        "teacher_routes": teacher_routes,
        "paired_test_count": 26,
        "aliases_excluded_from_inference": aggregate["aliases_excluded_from_inference"],
        "kd_contract": execution["kd"],
        "route_semantics": execution["route_semantics"],
        "finalization": {
            "training_or_predictions_changed": False,
            "non_ece_metrics_changed": False,
            "neural_artifacts_replayed_exactly": 180,
            "rf_artifacts_replayed_exactly": 10,
            "paired_tests_finalized_with_exact_enumeration": 26,
            "continuation_status": continuation["status"],
        },
        "boundary": "This is a controlled FG-DS reimplementation of the complete route matrix. Route-level RNG resets and shared initial states differ from the archived sequential execution, so archived-to-current changes cannot be attributed to split correction alone.",
        "aggregate": artifact(aggregate_path),
        "execution_contract": artifact(execution_path),
        "finalization_contract": artifact(finalization_path),
        "inventory": inventory,
    }


def sensitivity_comparison(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "temperature": entry["temperature"],
        "alpha": entry["alpha"],
        "difference": entry["difference"],
        "wilcoxon_exact_p": entry["wilcoxon_signed_rank"]["p_value_two_sided_exact_enumeration"],
        "holm_within_student_p": entry[
            "holm_adjusted_wilcoxon_within_student_9_test_family_p"
        ],
        "holm_global_p": entry["holm_adjusted_wilcoxon_global_18_test_family_p"],
    }


def validate_hyperparameter_sensitivity(
    ten_seed_contract: dict[str, Any],
) -> dict[str, Any]:
    inventory = verify_inventory(SENSITIVITY_ROOT, "artifact_manifest.json", {"complete"})
    aggregate_path = SENSITIVITY_ROOT / "response_surface_aggregate.json"
    execution_path = SENSITIVITY_ROOT / "execution_contract.json"
    aggregate = read_json(aggregate_path)
    execution = read_json(execution_path)
    require(aggregate.get("status") == "complete", "Sensitivity aggregate is incomplete")
    require(aggregate.get("protocol_id") == SENSITIVITY_PROTOCOL, "Sensitivity protocol mismatch")
    require(execution.get("protocol_id") == SENSITIVITY_PROTOCOL, "Sensitivity execution protocol mismatch")
    require(aggregate.get("seeds") == EXPECTED_SEEDS, "Sensitivity seed set mismatch")
    require(aggregate.get("seed_count") == 10, "Sensitivity seed count mismatch")
    require(aggregate.get("temperatures") == [1.0, 2.0, 4.0], "Sensitivity temperatures mismatch")
    require(aggregate.get("alphas") == [0.3, 0.5, 0.7], "Sensitivity alphas mismatch")
    require_clean_contract_identity(execution, ten_seed_contract, "Sensitivity")
    grid = execution.get("factorial_grid", {})
    require(grid.get("cell_count_per_student") == 9, "Sensitivity grid size mismatch")
    require(grid.get("total_training_jobs") == 180, "Sensitivity job count mismatch")
    require(aggregate.get("selection_performed") is False, "Sensitivity performed model selection")
    require(aggregate.get("primary_result_replaced") is False, "Sensitivity replaced the primary result")
    require(not aggregate.get("selected_hyperparameters"), "Sensitivity selected a winning cell")
    require(
        execution.get("selection_policy", {}).get("winning_cell_selected") is False,
        "Sensitivity execution contract selected a winning cell",
    )
    completions = sorted(SENSITIVITY_ROOT.glob("seed_*/seed_completion.json"))
    require(len(completions) == 10, "Sensitivity seed completion count mismatch")
    for completion_path in completions:
        completion = read_json(completion_path)
        require(completion.get("status") == "complete", f"Incomplete sensitivity seed: {completion_path}")
        require(completion.get("protocol_id") == SENSITIVITY_PROTOCOL, "Sensitivity seed protocol mismatch")
        require(completion.get("selection_performed") is False, "Sensitivity seed performed selection")
        grid_results = completion.get("grid_results", {})
        require(
            set(grid_results) == {"student_A", "student_B"}
            and all(len(grid_results[student]) == 9 for student in grid_results),
            "Sensitivity per-seed grid count mismatch",
        )

    paired = aggregate["paired_tests_against_persisted_scratch"]
    require(len(paired) == 18, "Sensitivity paired-test count mismatch")
    selected = {
        key: sensitivity_comparison(paired[key])
        for key in (
            "student_A:T1_alpha03_minus_persisted_scratch_test_macro_f1",
            "student_A:T4_alpha07_minus_persisted_scratch_test_macro_f1",
            "student_B:T4_alpha07_minus_persisted_scratch_test_macro_f1",
        )
    }
    return {
        "status": "descriptive_sensitivity_evidence",
        "protocol_id": SENSITIVITY_PROTOCOL,
        "seeds": EXPECTED_SEEDS,
        "training_jobs": 180,
        "temperatures": aggregate["temperatures"],
        "alphas": aggregate["alphas"],
        "selection_performed": False,
        "primary_result_replaced": False,
        "selected_comparisons": selected,
        "boundary": aggregate["interpretation_boundary"],
        "aggregate": artifact(aggregate_path),
        "execution_contract": artifact(execution_path),
        "inventory": inventory,
    }


def group_balanced_comparison(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "differences": entry["differences"],
        "exact_wilcoxon_p": entry["exact_paired_wilcoxon"]["p_value_two_sided"],
        "holm_within_family_p": entry["holm_exact_paired_wilcoxon_p"],
        "holm_view_global_p": entry["holm_view_global_exact_paired_wilcoxon_p"],
    }


def validate_group_balanced_sensitivity(
    ten_seed_contract: dict[str, Any],
) -> dict[str, Any]:
    inventory = verify_inventory(GROUP_BALANCED_ROOT, "artifact_manifest.json", {"complete"})
    aggregate_path = GROUP_BALANCED_ROOT / "aggregate_results.json"
    execution_path = GROUP_BALANCED_ROOT / "execution_contract.json"
    aggregate = read_json(aggregate_path)
    execution = read_json(execution_path)
    require(aggregate.get("status") == "complete", "Group-balanced aggregate is incomplete")
    require(aggregate.get("protocol_id") == GROUP_BALANCED_PROTOCOL, "Group-balanced protocol mismatch")
    require(execution.get("protocol_id") == GROUP_BALANCED_PROTOCOL, "Group-balanced execution protocol mismatch")
    require(aggregate.get("seeds") == EXPECTED_SEEDS, "Group-balanced seed set mismatch")
    require(aggregate.get("seed_count") == 10, "Group-balanced seed count mismatch")
    clean = execution.get("clean_data_contract", {})
    for key in ("split_indices_sha256", "scaler_sha256"):
        require(clean.get(key) == ten_seed_contract.get(key), f"Group-balanced clean contract mismatch: {key}")
    full_contract = execution.get("full_route_contract", {})
    require(full_contract.get("protocol_id") == FULL_ROUTE_PROTOCOL, "Group-balanced full-route source mismatch")
    require(
        full_contract.get("aggregate_results_sha256")
        == sha256_file(FULL_ROUTE_ROOT / "aggregate_results.json"),
        "Group-balanced source aggregate hash mismatch",
    )
    summary = aggregate.get("test_group_summary", {})
    require(summary.get("test_rows") == EXPECTED_TEST_ROWS, "Group-balanced test row mismatch")
    require(summary.get("test_exact_feature_groups") == 54_174, "Group-balanced group count mismatch")
    require(summary.get("mixed_label_groups") == 0, "Group-balanced mixed-label groups found")
    require(summary.get("mixed_label_rows") == 0, "Group-balanced mixed-label rows found")
    audit = aggregate.get("within_group_prediction_probability_audit", {})
    require(audit.get("route_seed_count") == 240, "Group-balanced route-seed audit count mismatch")
    require(audit.get("exact_prediction_equality_required") is True, "Group-balanced prediction equality gate missing")
    tolerance = float(audit.get("probability_max_abs_delta_tolerance"))
    maximum = float(audit.get("global_max_probability_abs_delta"))
    require(maximum <= tolerance, "Group-balanced probability consistency gate failed")
    require(len(aggregate.get("paired_route_tests", {})) == 78, "Group-balanced paired-test count mismatch")

    comparisons = {}
    for view in ("row_level", "inverse_test_group_size", "pure_group_representative"):
        comparisons[view] = {}
        for student in ("student_A", "student_B"):
            key = f"{view}:{student}:E_KD_from_RF_minus_D_Small_MLP"
            comparisons[view][student] = group_balanced_comparison(
                aggregate["paired_route_tests"][key]
            )
    return {
        "status": "repeated_pattern_sensitivity_evidence",
        "protocol_id": GROUP_BALANCED_PROTOCOL,
        "seeds": EXPECTED_SEEDS,
        "test_group_summary": summary,
        "within_group_prediction_probability_audit": audit,
        "rf_kd_minus_scratch": comparisons,
        "primary_inference_policy": aggregate["primary_inference_policy"],
        "boundary": "Row-level metrics remain primary because they preserve the benchmark record distribution. Inverse-group-size and one-representative-per-pure-group views are sensitivity analyses for repeated exact test patterns, not replacement test sets.",
        "aggregate": artifact(aggregate_path),
        "execution_contract": artifact(execution_path),
        "inventory": inventory,
    }


def validate_exact_teacher_shap() -> tuple[dict[str, Any], dict[str, Any]]:
    inventory = verify_inventory(SHAP_ROOT, "artifact_manifest.json", {"complete"})
    report_path = SHAP_ROOT / "shap_report.json"
    execution_path = SHAP_ROOT / "execution_contract.json"
    report = read_json(report_path)
    execution = read_json(execution_path)
    require(report.get("status") == "complete", "SHAP report is incomplete")
    require(report.get("protocol_id") == SHAP_PROTOCOL, "SHAP report protocol mismatch")
    require(execution.get("protocol_id") == SHAP_PROTOCOL, "SHAP execution protocol mismatch")
    require(execution.get("parameters", {}).get("masker_invariance_cache") == "bit-exact equality; near-but-not-equal values always trigger model evaluation", "SHAP exact masker contract mismatch")
    teacher = execution.get("teacher_reconstruction", {})
    require(teacher.get("train_and_test_output_validation_passed") is True, "SHAP teacher output validation failed")
    require(teacher.get("function_equivalence_on_permutation_masked_inputs_established") is False, "SHAP teacher reconstruction boundary missing")
    require(execution.get("dataset_sha256") == EXPECTED_DATASET_SHA256, "SHAP dataset mismatch")
    shap_files = sorted(SHAP_ROOT.glob("*_shap_values.npz"))
    require(len(shap_files) == 18, "SHAP artifact count mismatch")
    local_accuracy_atol = float(execution["parameters"]["local_accuracy_atol"])
    local_accuracy: dict[str, float] = {}
    for path in shap_files:
        with np.load(path, allow_pickle=False) as arrays:
            require(
                {"values", "base_values", "model_outputs"}.issubset(arrays.files),
                f"SHAP arrays missing from {path}",
            )
            residual = np.asarray(arrays["base_values"], dtype=np.float64) + np.asarray(
                arrays["values"], dtype=np.float64
            ).sum(axis=1) - np.asarray(arrays["model_outputs"], dtype=np.float64)
            maximum = float(np.max(np.abs(residual)))
        require(maximum <= local_accuracy_atol, f"SHAP local-accuracy gate failed: {path}")
        local_accuracy[path.name] = maximum

    condition_summaries: dict[str, Any] = {}
    for condition, value in report["conditions"].items():
        require(len(value.get("replicates", [])) == 3, f"SHAP replicate count mismatch: {condition}")
        student_summary = value["student_teacher_estimator_variation"]
        condition_summaries[condition] = {
            "temperature": value["temperature"],
            "student_A": student_summary["A"],
            "student_B": student_summary["B"],
        }
    require(set(condition_summaries) == {"fp32_deployment_source_probabilities_T1", "kd_softened_probabilities_T4"}, "SHAP condition set mismatch")

    require(SHAP_V2_PARTIAL_ROOT.is_dir(), "Expected preserved SHAP-v2 partial directory is missing")
    partial_files = sorted(path for path in SHAP_V2_PARTIAL_ROOT.rglob("*") if path.is_file())
    require(partial_files, "SHAP-v2 partial directory is empty")
    require(not (SHAP_V2_PARTIAL_ROOT / "artifact_manifest.json").exists(), "SHAP-v2 unexpectedly has a seal")
    require(not (SHAP_V2_PARTIAL_ROOT / "shap_report.json").exists(), "SHAP-v2 unexpectedly has a final report")
    digest = hashlib.sha256()
    for path in partial_files:
        relative = path.relative_to(SHAP_V2_PARTIAL_ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    partial = {
        "status": "incomplete_excluded",
        "path": repo_path(SHAP_V2_PARTIAL_ROOT),
        "file_count": len(partial_files),
        "tree_sha256": digest.hexdigest(),
        "boundary": "The v2 attempt has neither a final report nor an artifact manifest. It is preserved for forensic traceability and supports no result claim.",
    }
    return {
        "status": "current_single_specimen_xai_evidence",
        "protocol_id": SHAP_PROTOCOL,
        "deployment_seed": 42,
        "explained_test_rows": execution["parameters"]["explain_size"],
        "test_rows": EXPECTED_TEST_ROWS,
        "estimator_replicates": execution["parameters"]["estimator_replicates"],
        "local_accuracy_atol": local_accuracy_atol,
        "global_max_local_accuracy_residual": max(local_accuracy.values()),
        "condition_summaries": condition_summaries,
        "scope_boundary": report["scope_boundary"],
        "report": artifact(report_path),
        "execution_contract": artifact(execution_path),
        "inventory": inventory,
    }, partial


def validate_runtime_and_deployment(
    ten_seed_contract: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    deployment_inventory = verify_inventory(
        DEPLOYMENT_ROOT, "artifact_manifest.json", {"complete"}
    )
    runtime_inventory = verify_inventory(RUNTIME_ROOT, "artifact_manifest.json", {"complete"})
    deployment_contract_path = DEPLOYMENT_ROOT / "preprocessing_contract.json"
    deployment_contract = read_json(deployment_contract_path)
    require(
        deployment_contract.get("protocol_id") == DEPLOYMENT_PROTOCOL,
        "Deployment protocol mismatch",
    )
    runtime_path = RUNTIME_ROOT / "runtime_report.json"
    runtime = read_json(runtime_path)
    require(runtime.get("status") == "passed", "Runtime report did not pass")
    require(runtime.get("protocol_id") == RUNTIME_PROTOCOL, "Runtime protocol mismatch")
    require(
        runtime.get("deployment_source_protocol_id") == DEPLOYMENT_PROTOCOL,
        "Runtime deployment source mismatch",
    )
    require(runtime.get("deployment_seed") == 42, "Runtime seed mismatch")
    require(runtime.get("test_rows") == EXPECTED_TEST_ROWS, "Runtime row count mismatch")

    recorded_contracts = runtime["source_contracts"]
    local_contract_paths = {
        "execution_contract": DEPLOYMENT_ROOT / "execution_contract.json",
        "preprocessing_contract": deployment_contract_path,
        "split_indices": DEPLOYMENT_ROOT / "split_indices.npz",
        "scaler_parameters": DEPLOYMENT_ROOT / "scaler_parameters.npz",
    }
    for name, path in local_contract_paths.items():
        require(
            sha256_file(path) == recorded_contracts[name]["sha256"],
            f"Runtime source contract hash mismatch: {name}",
        )

    exports: dict[str, Any] = {}
    for student in ("student_A", "student_B"):
        entry = runtime["students"][student]
        suffix = student.removeprefix("student_")
        source_state = DEPLOYMENT_ROOT / f"seed_42/student_{suffix}_KD_from_RF_fp32.pt"
        source_predictions = (
            DEPLOYMENT_ROOT
            / f"seed_42/student_{suffix}_KD_from_RF_test_predictions.csv"
        )
        require(
            sha256_file(source_state) == entry["lineage"]["state_sha256"],
            f"Runtime source state hash mismatch: {student}",
        )
        require(
            sha256_file(source_predictions) == entry["lineage"]["prediction_sha256"],
            f"Runtime source prediction hash mismatch: {student}",
        )
        require(
            entry["onnxruntime_fp32"]["agreement_vs_pytorch_fp32"] == 1.0,
            f"ONNX Runtime FP32 mismatch: {student}",
        )
        require(
            entry["openvino_fp32_from_onnx"]["agreement_vs_pytorch_fp32"] == 1.0,
            f"OpenVINO FP32 mismatch: {student}",
        )
        require(
            entry["openvino_fp32_from_onnx"]["agreement_vs_onnxruntime_fp32"] == 1.0,
            f"OpenVINO/ONNX mismatch: {student}",
        )
        exports[student] = {
            "export_id": entry["lineage"]["export_id"],
            "source_state_sha256": entry["lineage"]["state_sha256"],
            "source_prediction_sha256": entry["lineage"]["prediction_sha256"],
            "pytorch_fp32": entry["pytorch_fp32"],
            "onnxruntime_fp32": entry["onnxruntime_fp32"],
            "openvino_fp32_from_onnx": entry["openvino_fp32_from_onnx"],
            "onnxruntime_dynamic_int8_weights": entry[
                "onnxruntime_dynamic_int8_weights"
            ],
        }

    local_seed42_hashes = {
        "student_A": sha256_file(TEN_SEED_ROOT / "seed_42/student_A_KD_from_RF_fp32.pt"),
        "student_B": sha256_file(TEN_SEED_ROOT / "seed_42/student_B_KD_from_RF_fp32.pt"),
    }
    deployment_seed42_hashes = {
        student: exports[student]["source_state_sha256"]
        for student in ("student_A", "student_B")
    }
    checkpoint_equality = {
        student: local_seed42_hashes[student] == deployment_seed42_hashes[student]
        for student in ("student_A", "student_B")
    }

    return runtime, {
        "status": "fixed_single_seed_deployment_evidence",
        "protocol_id": DEPLOYMENT_PROTOCOL,
        "seed": 42,
        "selection_boundary": "The preserved seed-42 checkpoint is the deployment specimen. Hardware execution is not replicated over all optimizer seeds.",
        "preprocessing_equivalence_to_ten_seed": compare_preprocessing_contracts(
            ten_seed_contract, deployment_contract
        ),
        "local_ten_seed_seed42_checkpoint_sha256": local_seed42_hashes,
        "deployment_checkpoint_sha256": deployment_seed42_hashes,
        "checkpoints_byte_identical": checkpoint_equality,
        "checkpoint_identity_boundary": "The Windows ten-seed and Winterfell five-seed checkpoints share protocol semantics but are separate trained artifacts. Their metrics and hashes must not be interchanged.",
        "runtime_protocol_id": RUNTIME_PROTOCOL,
        "runtime_test_rows": EXPECTED_TEST_ROWS,
        "exports": exports,
        "dynamic_onnx_boundary": "Dynamic ONNX quantization applies int8 weight quantization in ONNX Runtime. It is separate from the MCU fixed-point implementation.",
        "runtime_report": artifact(runtime_path),
        "deployment_inventory": deployment_inventory,
        "runtime_inventory": runtime_inventory,
    }


def validate_fixed_point_refinement(deployment: dict[str, Any]) -> dict[str, Any]:
    inventory = verify_inventory(
        FIXED_POINT_REFINEMENT_ROOT, "artifact_manifest.json", {"complete"}
    )
    report_path = FIXED_POINT_REFINEMENT_ROOT / "refinement_report.json"
    execution_path = FIXED_POINT_REFINEMENT_ROOT / "execution_contract.json"
    report = read_json(report_path)
    execution = read_json(execution_path)
    require(report.get("status") == "complete", "Fixed-point refinement is incomplete")
    require(
        report.get("protocol_id") == FIXED_POINT_REFINEMENT_PROTOCOL,
        "Fixed-point refinement report protocol mismatch",
    )
    require(
        execution.get("protocol_id") == FIXED_POINT_REFINEMENT_PROTOCOL,
        "Fixed-point refinement execution protocol mismatch",
    )
    boundary = report.get("claim_boundary", {})
    require(boundary.get("software_only") is True, "Fixed-point refinement is not marked software-only")
    require(boundary.get("hardware_replayed") is False, "Fixed-point refinement incorrectly claims hardware replay")
    require(
        boundary.get("existing_usb_and_wifi_hil_remain_bound_to_the_original_ptq_models") is True,
        "Fixed-point refinement does not preserve the original HIL boundary",
    )

    students: dict[str, Any] = {}
    for student in ("student_A", "student_B"):
        value = report["students"][student]
        source = value["source_checkpoint"]
        require(
            source["checkpoint_sha256"]
            == deployment["exports"][student]["source_state_sha256"],
            f"Fixed-point refinement source checkpoint mismatch: {student}",
        )
        require(
            source["strict_export_id"] == deployment["exports"][student]["export_id"],
            f"Fixed-point refinement source export mismatch: {student}",
        )
        require(
            value["saturation_and_range_audit"]["strict_zero_saturation_and_no_overflow_gate"] is True,
            f"Fixed-point refinement numeric gate failed: {student}",
        )
        claim = value["claim_boundary"]
        require(
            claim.get("software_only") is True
            and claim.get("hardware_replayed") is False
            and claim.get("firmware_exported") is False,
            f"Fixed-point refinement claim boundary mismatch: {student}",
        )
        output_root = FIXED_POINT_REFINEMENT_ROOT / student
        outputs = value["outputs"]
        hashes = value["output_hashes"]
        output_checks = {
            "plain_state_dict": (outputs["plain_state_dict"], hashes["plain_state_dict_sha256"]),
            "rich_artifact": (outputs["rich_artifact"], hashes["rich_artifact_sha256"]),
            "test_predictions": (outputs["test_predictions"], hashes["test_predictions_sha256"]),
        }
        output_artifacts = {}
        for name, (filename, expected_hash) in output_checks.items():
            path = output_root / filename
            require(sha256_file(path) == expected_hash, f"Fixed-point refinement output hash mismatch: {student}/{name}")
            output_artifacts[name] = artifact(path)
        test = value["test_evaluation_after_selection"]
        students[student] = {
            "source_export_id": source["strict_export_id"],
            "source_float_macro_f1": source["float_metrics"]["macro_f1"],
            "source_fixed_macro_f1": source["fixed_metrics"]["macro_f1"],
            "selected_epoch": value["refinement"]["selected_epoch"],
            "validation_fixed_macro_f1_delta": value["refinement"]["validation_macro_f1_delta"],
            "refined_float_macro_f1": test["float_metrics"]["macro_f1"],
            "refined_fixed_macro_f1": test["fixed_metrics"]["macro_f1"],
            "fixed_vs_float_prediction_agreement": test["fixed_vs_float_prediction_agreement"],
            "fixed_macro_f1_minus_current_ptq": test["fixed_macro_f1_minus_current_ptq"],
            "zero_saturation_and_no_overflow_gate": True,
            "outputs": output_artifacts,
        }
    return {
        "status": "software_only_candidate_evidence",
        "protocol_id": FIXED_POINT_REFINEMENT_PROTOCOL,
        "seed": 42,
        "students": students,
        "boundary": "The refinement changes the two model states after validation-based epoch selection. It has not been strictly exported or replayed on either board, so it cannot replace the preserved PTQ USB or Wi-Fi results.",
        "report": artifact(report_path),
        "execution_contract": artifact(execution_path),
        "inventory": inventory,
    }


def validate_current_msp430_static(deployment: dict[str, Any]) -> dict[str, Any]:
    inventory = verify_inventory(
        MSP430_STATIC_ROOT, "msp430_static_root_manifest.json", {"passed"}
    )
    root_manifest_path = MSP430_STATIC_ROOT / "msp430_static_root_manifest.json"
    summary_path = MSP430_STATIC_ROOT / "msp430_static_summary.json"
    root_manifest = read_json(root_manifest_path)
    summary = read_json(summary_path)
    require(summary.get("status") == "success", "Current MSP430 summary failed")
    require(
        root_manifest.get("evidence_scope")
        == "Static cross-compilation of the current FGDS seed-42 integer preprocessing and inference cores for MSP430F1611.",
        "Current MSP430 evidence scope mismatch",
    )
    target = root_manifest.get("target", {})
    require(
        target.get("mcu") == "msp430f1611"
        and target.get("flash_budget_bytes") == 49_152
        and target.get("ram_budget_bytes") == 10_240,
        "Current MSP430 target contract mismatch",
    )
    students: dict[str, Any] = {}
    for student in ("student_A", "student_B"):
        report_path = MSP430_STATIC_ROOT / student / "msp430_static_evidence.json"
        report = read_json(report_path)
        require(report.get("status") == "success", f"MSP430 static report failed: {student}")
        identity = report["contract_identity"]
        require(identity.get("seed") == 42, f"MSP430 seed mismatch: {student}")
        require(identity.get("dataset_sha256") == EXPECTED_DATASET_SHA256, f"MSP430 dataset mismatch: {student}")
        require(identity.get("protocol_id") == DEPLOYMENT_PROTOCOL, f"MSP430 protocol mismatch: {student}")
        require(
            identity.get("export_id") == deployment["exports"][student]["export_id"],
            f"MSP430 export mismatch: {student}",
        )
        expected_report_hash = root_manifest["students"][student]["report_sha256"]
        require(sha256_file(report_path) == expected_report_hash, f"MSP430 report hash mismatch: {student}")
        footprint = report["linked_footprint"]["memory_budget_context"]
        stack = report["stack_evidence"]
        require(not stack.get("missing_expected_functions"), f"MSP430 stack evidence is incomplete: {student}")
        students[student] = {
            "export_id": identity["export_id"],
            "dimensions": identity["dimensions"],
            "parameter_bytes": identity["parameter_bytes"],
            "activation_bytes_estimate": identity["activation_bytes_estimate"],
            "macs_per_inference": identity["macs_per_inference"],
            "static_flash_load_bytes": footprint["static_flash_load_bytes"],
            "static_ram_lower_bound_bytes": footprint["static_ram_lower_bound_bytes"],
            "maximum_single_function_stack_bytes": stack["maximum_single_function_bytes"],
            "report": artifact(report_path),
        }
    return {
        "status": "current_static_cross_compile_evidence",
        "target": target,
        "students": students,
        "claim_boundary": root_manifest["claim_boundary"],
        "toolchain_scope_boundary": root_manifest["toolchain_scope_boundary"],
        "summary": artifact(summary_path),
        "inventory": inventory,
    }


def local_path_from_recorded(base: Path, recorded: str) -> Path:
    return base / Path(recorded.replace("\\", "/")).name


def validate_hil(
    runtime: dict[str, Any],
) -> dict[str, Any]:
    usb_inventory = verify_inventory(
        USB_REPORT_ROOT, "final_report_manifest.json", {"complete"}
    )
    wireless_inventory = verify_inventory(
        WIRELESS_REPORT_ROOT, "wireless_report_manifest.json", {"passed"}
    )
    usb_report_path = USB_REPORT_ROOT / "fgds_seed42_hardware_summary.json"
    wireless_report_path = WIRELESS_REPORT_ROOT / "wireless_hil_final_report.json"
    usb_report_manifest = read_json(USB_REPORT_ROOT / "final_report_manifest.json")
    wireless_report_manifest = read_json(
        WIRELESS_REPORT_ROOT / "wireless_report_manifest.json"
    )
    usb_report_entry = next(
        entry
        for entry in usb_report_manifest["files"]
        if entry["path"] == usb_report_path.name
    )
    wireless_report_entry = next(
        entry
        for entry in wireless_report_manifest["files"]
        if entry["path"] == wireless_report_path.name
    )
    usb = read_json(usb_report_path)
    wireless = read_json(wireless_report_path)
    require(usb.get("status") == "passed", "USB HIL report did not pass")
    require(usb.get("run_count") == 4, "USB HIL run count mismatch")
    require(
        usb.get("full_test_board_predictions") == 4 * EXPECTED_TEST_ROWS,
        "USB HIL prediction count mismatch",
    )
    require(wireless.get("status") == "passed", "Wireless HIL report did not pass")
    require(wireless.get("protocol_id") == WIRELESS_PROTOCOL, "Wireless protocol mismatch")
    require(
        wireless.get("full_test_rows_per_board_model_pair") == EXPECTED_TEST_ROWS,
        "Wireless row count mismatch",
    )
    require(
        wireless.get("full_test_board_model_predictions") == 4 * EXPECTED_TEST_ROWS,
        "Wireless prediction count mismatch",
    )

    expected_exports = {
        student: runtime["students"][student]["lineage"]["export_id"]
        for student in ("student_A", "student_B")
    }
    expected_references = {
        student: sha256_file(
            REPO_ROOT
            / "deployment/firmware_export/wsnds_rfkd_hil"
            / f"generated_fgds_{student}_seed42/hil_reference_predictions.csv"
        )
        for student in ("student_A", "student_B")
    }
    usb_rows = {row["run"]: row for row in usb["runs"]}
    wireless_rows = {row["label"]: row for row in wireless["rows"]}
    require(set(usb_rows) == set(EXPECTED_LABELS), "USB label set mismatch")
    require(set(wireless_rows) == set(EXPECTED_LABELS), "Wireless label set mismatch")

    verified_rows: dict[str, Any] = {}
    reference_hashes: dict[str, set[str]] = {"student_A": set(), "student_B": set()}
    for label in EXPECTED_LABELS:
        board, _, suffix = label.partition("_student_")
        student = f"student_{suffix}"
        usb_metrics_path = USB_ROOT / label.replace("esp32c3", "pi5_esp32c3").replace(
            "arduino_r4", "pi5_arduino_r4"
        ) / "full_56301_metrics.json"
        wireless_metrics_path = WIRELESS_ROOT / label.replace(
            "esp32c3", "pi5_esp32c3"
        ).replace("arduino_r4", "pi5_arduino_r4") / "full_56301_metrics.json"
        usb_metrics = read_json(usb_metrics_path)
        wireless_metrics = read_json(wireless_metrics_path)
        for transport, metrics in (("USB", usb_metrics), ("Wi-Fi UDP", wireless_metrics)):
            require(metrics.get("status") == "passed", f"{transport} metrics failed: {label}")
            require(
                metrics.get("completed_vectors") == EXPECTED_TEST_ROWS,
                f"{transport} vector count mismatch: {label}",
            )
            require(
                metrics.get("mcu_vs_fixed_reference_agreement") == 1.0,
                f"{transport} prediction disagreement: {label}",
            )
            require(
                metrics.get("exact_logit_agreement") == 1.0,
                f"{transport} logit disagreement: {label}",
            )
            require(
                metrics.get("non_ok_status_count") == 0,
                f"{transport} non-OK statuses: {label}",
            )
            require(
                metrics["provenance"]["export_id"] == expected_exports[student],
                f"{transport} export ID mismatch: {label}",
            )
            require(
                metrics["provenance"]["board"] == board,
                f"{transport} board mismatch: {label}",
            )
            require(
                metrics["provenance"]["student"] == student,
                f"{transport} student mismatch: {label}",
            )
            reference_hashes[student].add(metrics["provenance"]["reference_csv_sha256"])

        usb_row = usb_rows[label]
        wireless_row = wireless_rows[label]
        require(usb_row["export_id"] == expected_exports[student], f"USB summary ID mismatch")
        for summary_key, metric_key in (
            ("vectors", "completed_vectors"),
            ("mcu_vs_fixed", "mcu_vs_fixed_reference_agreement"),
            ("exact_logit_agreement", "exact_logit_agreement"),
            ("mcu_vs_fp32", "mcu_vs_fp32_agreement"),
            ("accuracy", "accuracy"),
            ("macro_f1", "macro_f1"),
        ):
            require(
                close(usb_row[summary_key], usb_metrics[metric_key]),
                f"USB summary differs from metrics: {label}/{summary_key}",
            )
        for summary_key, metric_key in (
            ("full_vectors", "completed_vectors"),
            ("mcu_vs_fixed_reference", "mcu_vs_fixed_reference_agreement"),
            ("exact_logit_agreement", "exact_logit_agreement"),
            ("mcu_vs_fp32", "mcu_vs_fp32_agreement"),
            ("accuracy", "accuracy"),
            ("macro_f1", "macro_f1"),
        ):
            require(
                close(wireless_row[summary_key], wireless_metrics[metric_key]),
                f"Wireless summary differs from metrics: {label}/{summary_key}",
            )
        verify_recorded_file(usb_metrics_path, usb_row["metrics_sha256"])
        wireless_input = wireless["input_evidence"][label]
        verify_recorded_file(wireless_metrics_path, wireless_input["metrics_sha256"])

        usb_compile_path = local_path_from_recorded(
            USB_ROOT / "compile_evidence", usb_row["compile_evidence_path_recorded"]
        )
        wireless_compile_path = local_path_from_recorded(
            WIRELESS_ROOT / "compile_evidence", wireless_input["compile_path_recorded"]
        )
        verify_recorded_file(usb_compile_path, usb_row["compile_evidence_sha256"])
        verify_recorded_file(wireless_compile_path, wireless_input["compile_sha256"])

        for key in ("accuracy", "macro_f1"):
            require(
                close(usb_metrics[key], wireless_metrics[key]),
                f"USB/Wireless metric mismatch for {label}/{key}",
            )
        require(
            close(
                usb_metrics["mcu_vs_fp32_agreement"],
                wireless_metrics["mcu_vs_fp32_agreement"],
            ),
            f"USB/Wireless FP32 agreement mismatch: {label}",
        )
        require(
            usb_metrics["provenance"]["reference_csv_sha256"]
            == expected_references[student],
            f"Exported reference hash mismatch: {label}",
        )

        verified_rows[label] = {
            "board": board,
            "student": student,
            "vectors_per_transport": EXPECTED_TEST_ROWS,
            "export_id": expected_exports[student],
            "reference_csv_sha256": usb_metrics["provenance"]["reference_csv_sha256"],
            "accuracy": usb_metrics["accuracy"],
            "macro_f1": usb_metrics["macro_f1"],
            "mcu_vs_fixed_reference_agreement": 1.0,
            "exact_logit_agreement": 1.0,
            "mcu_vs_fp32_agreement": usb_metrics["mcu_vs_fp32_agreement"],
            "usb_metrics": recorded_artifact(
                usb_metrics_path, usb_row["metrics_sha256"]
            ),
            "wireless_metrics": recorded_artifact(
                wireless_metrics_path, wireless_input["metrics_sha256"]
            ),
            "usb_device_compute_mean_us": usb_row["compute_total_mean_us"],
            "wireless_device_compute_mean_us": wireless_row["device_compute_mean_us"],
            "wireless_host_rtt_mean_us": wireless_row["host_rtt_mean_us"],
            "wireless_data_retransmissions": wireless_row["data_retransmissions"],
        }

    require(
        all(len(hashes) == 1 for hashes in reference_hashes.values()),
        "Reference prediction hashes differ across boards or transports",
    )
    return {
        "status": "passed",
        "deployment_seed": 42,
        "board_model_pairs": 4,
        "full_test_rows_per_pair": EXPECTED_TEST_ROWS,
        "usb_full_predictions": 4 * EXPECTED_TEST_ROWS,
        "wireless_full_predictions": 4 * EXPECTED_TEST_ROWS,
        "combined_transport_executions": 8 * EXPECTED_TEST_ROWS,
        "independence_boundary": "USB and Wi-Fi replay the same two model artifacts on the same test records. They are transport and execution replications, not independent predictive samples.",
        "wireless_boundary": wireless["claim_boundary"],
        "usb_boundary": usb["claim_boundary"],
        "timing_boundary": "Firmware compute time and host-observed UDP round-trip time are separate measurements. Their difference is not pure wireless latency.",
        "rows": verified_rows,
        "usb_report": recorded_artifact(
            usb_report_path,
            usb_report_entry["sha256"],
            usb_report_entry["size_bytes"],
        ),
        "wireless_report": recorded_artifact(
            wireless_report_path,
            wireless_report_entry["sha256"],
            wireless_report_entry["size_bytes"],
        ),
        "usb_report_inventory": usb_inventory,
        "wireless_report_inventory": wireless_inventory,
    }


def validate_behavioral_transfer() -> dict[str, Any]:
    inventory = verify_inventory(
        BEHAVIORAL_TRANSFER_ROOT, "artifact_manifest.json", {"complete"}
    )
    report_path = BEHAVIORAL_TRANSFER_ROOT / "behavioral_transfer_summary.json"
    report = read_json(report_path)
    require(report.get("status") == "complete", "Behavioral-transfer report is incomplete")
    require(
        report.get("protocol_id") == BEHAVIORAL_TRANSFER_PROTOCOL,
        "Behavioral-transfer protocol mismatch",
    )
    contract = report.get("execution_contract")
    require(isinstance(contract, dict), "Behavioral-transfer execution contract is missing")
    require(
        contract.get("source_protocol_id") == TEN_SEED_PROTOCOL,
        "Behavioral-transfer source protocol mismatch",
    )
    require(contract.get("seeds") == EXPECTED_SEEDS, "Behavioral-transfer seed set mismatch")
    require(
        contract.get("test_rows_per_seed") == EXPECTED_TEST_ROWS,
        "Behavioral-transfer test-row count mismatch",
    )
    require(
        contract.get("analysis_status") == "post_hoc_secondary_evidence",
        "Behavioral-transfer post-hoc status is not explicit",
    )
    require(
        contract.get("primary_metric", {}).get("temperature") == 4.0,
        "Behavioral-transfer output temperature mismatch",
    )
    audit = report.get("test_group_audit")
    require(
        isinstance(audit, dict)
        and audit.get("rows") == EXPECTED_TEST_ROWS
        and audit.get("mixed_label_groups") == 0,
        "Behavioral-transfer test-group audit mismatch",
    )
    primary = report.get("primary_tests")
    require(
        isinstance(primary, dict) and set(primary) == {"student_A", "student_B"},
        "Behavioral-transfer primary-test matrix mismatch",
    )
    compact: dict[str, Any] = {}
    for student in ("student_A", "student_B"):
        item = primary[student]
        require(item.get("positive_seed_count") == 10, f"Transfer direction mismatch: {student}")
        require(item.get("negative_seed_count") == 0, f"Negative transfer seed: {student}")
        require(item.get("zero_seed_count") == 0, f"Zero transfer seed: {student}")
        require(item.get("reject_holm_alpha_0_05") is True, f"Transfer test did not pass: {student}")
        require(
            len(item.get("transfer_gain_values", [])) == len(EXPECTED_SEEDS),
            f"Transfer-gain vector length mismatch: {student}",
        )
        compact[student] = {
            "metric": item["metric"],
            "orientation": item["orientation"],
            "transfer_gain_mean": item["transfer_gain_mean"],
            "transfer_gain_sample_std": item["transfer_gain_sample_std"],
            "positive_seed_count": item["positive_seed_count"],
            "exact_paired_wilcoxon_p": item["exact_paired_wilcoxon"][
                "p_value_two_sided"
            ],
            "holm_adjusted_p": item["holm_adjusted_p_across_two_students"],
            "reject_holm_alpha_0_05": item["reject_holm_alpha_0_05"],
        }
    return {
        "status": "post_hoc_secondary_evidence",
        "protocol_id": BEHAVIORAL_TRANSFER_PROTOCOL,
        "seeds": EXPECTED_SEEDS,
        "test_rows_per_seed": EXPECTED_TEST_ROWS,
        "statistical_unit": contract["statistical_unit"],
        "primary_tests": compact,
        "interpretation": report["interpretation"],
        "claim_boundary": report["claim_boundary"],
        "report": artifact(report_path),
        "inventory": inventory,
    }


def validate_multisplit_confirmation() -> dict[str, Any]:
    inventory = verify_inventory(MULTISPLIT_ROOT, "artifact_manifest.json", {"complete"})
    report_path = MULTISPLIT_ROOT / "multisplit_core_summary.json"
    semantic_path = MULTISPLIT_ROOT / "semantic_verification.json"
    report = read_json(report_path)
    semantic = read_json(semantic_path)
    require(report.get("status") == "complete", "Multisplit report is incomplete")
    require(report.get("protocol_id") == MULTISPLIT_PROTOCOL, "Multisplit protocol mismatch")
    require(semantic.get("status") == "passed", "Multisplit semantic verification failed")
    require(
        semantic.get("protocol_id") == MULTISPLIT_PROTOCOL,
        "Multisplit semantic protocol mismatch",
    )
    require(report.get("split_count") == 10, "Multisplit split count mismatch")
    require(
        report.get("optimizer_seeds_per_split") == [42, 123],
        "Multisplit optimizer-seed contract mismatch",
    )
    require(report.get("training_jobs") == 80, "Multisplit training-job count mismatch")
    require(report.get("rf_fits") == 10, "Multisplit RF-fit count mismatch")
    require(
        report.get("formal_hypothesis_test_performed") is False,
        "Multisplit evidence must remain descriptive",
    )
    summaries = report.get("descriptive_summaries")
    require(
        isinstance(summaries, dict) and set(summaries) == {"student_A", "student_B"},
        "Multisplit student matrix mismatch",
    )
    for student in ("student_A", "student_B"):
        require(
            len(summaries[student].get("values", [])) == 10,
            f"Multisplit value count mismatch: {student}",
        )
    require(
        summaries["student_A"].get("positive_split_count") == 10,
        "Student A split-sensitivity direction mismatch",
    )
    require(
        summaries["student_B"].get("positive_split_count") == 5
        and summaries["student_B"].get("negative_split_count") == 5,
        "Student B split-sensitivity direction mismatch",
    )
    return {
        "status": "descriptive_split_sensitivity",
        "protocol_id": MULTISPLIT_PROTOCOL,
        "split_count": report["split_count"],
        "optimizer_seeds_per_split": report["optimizer_seeds_per_split"],
        "training_jobs": report["training_jobs"],
        "rf_fits": report["rf_fits"],
        "formal_hypothesis_test_performed": False,
        "descriptive_summaries": summaries,
        "claim_boundary": report["claim_boundary"],
        "report": artifact(report_path),
        "semantic_verification": artifact(semantic_path),
        "inventory": inventory,
    }


def validate_all_seed_fixed_point_audit() -> dict[str, Any]:
    inventory = verify_inventory(
        ALL_SEED_FIXED_POINT_ROOT,
        "all_seed_fixed_point_manifest.json",
        {"complete_with_retained_failures"},
    )
    contract_path = ALL_SEED_FIXED_POINT_ROOT / "all_seed_audit_contract.json"
    progress_path = ALL_SEED_FIXED_POINT_ROOT / "all_seed_audit_progress.json"
    report_path = ALL_SEED_FIXED_POINT_ROOT / "all_seed_fixed_point_report.json"
    manifest_path = ALL_SEED_FIXED_POINT_ROOT / "all_seed_fixed_point_manifest.json"
    contract = read_json(contract_path)
    progress = read_json(progress_path)
    report = read_json(report_path)
    manifest = read_json(manifest_path)
    for value, label in ((contract, "contract"), (progress, "progress"), (report, "report"), (manifest, "manifest")):
        require(
            value.get("audit_protocol_id") == ALL_SEED_FIXED_POINT_PROTOCOL,
            f"All-seed fixed-point {label} protocol mismatch",
        )
        require(
            value.get("audit_contract_id") == contract.get("audit_contract_id"),
            f"All-seed fixed-point {label} contract mismatch",
        )
    verify_canonical_id(contract, "audit_contract_id", "All-seed fixed-point contract")
    verify_canonical_id(progress, "progress_payload_sha256", "All-seed fixed-point progress")
    verify_canonical_id(report, "report_payload_sha256", "All-seed fixed-point report")
    verify_canonical_id(manifest, "manifest_payload_sha256", "All-seed fixed-point manifest")
    require(progress.get("status") == "complete", "All-seed fixed-point progress is incomplete")
    require(progress.get("completed_count") == 40, "All-seed fixed-point completion count mismatch")
    require(progress.get("remaining_count") == 0, "All-seed fixed-point models remain")
    require(
        report.get("status") == "complete_with_retained_failures"
        and manifest.get("status") == report.get("status"),
        "All-seed fixed-point final status mismatch",
    )
    require(report.get("model_count") == 40, "All-seed fixed-point model count mismatch")
    require(len(report.get("records", [])) == 40, "All-seed fixed-point record count mismatch")
    require(
        report.get("status_counts") == {"passed": 26, "gate_failed": 14},
        "All-seed fixed-point status counts mismatch",
    )
    model_ids = [item.get("model_id") for item in report["records"]]
    require(len(set(model_ids)) == 40, "All-seed fixed-point model IDs are not unique")
    return {
        "status": report["status"],
        "protocol_id": ALL_SEED_FIXED_POINT_PROTOCOL,
        "model_count": report["model_count"],
        "status_counts": report["status_counts"],
        "statistical_unit_disclosure": report["statistical_unit_disclosure"],
        "post_hoc_method_disclosure": report["post_hoc_method_disclosure"],
        "retention_boundary": contract["retention_boundary"],
        "verification_boundary": contract["verification_boundary"],
        "report": artifact(report_path),
        "contract": artifact(contract_path),
        "progress": artifact(progress_path),
        "inventory": inventory,
    }


def validate_final_usb_hil(ten_seed: dict[str, Any]) -> dict[str, Any]:
    report_path = FINAL_HIL_REPORT_ROOT / "final_hil_summary.json"
    report_manifest_path = FINAL_HIL_REPORT_ROOT / "report_manifest.json"
    archive_manifest_path = FINAL_HIL_REPORT_ROOT / "external_archive_manifest.json"
    locator_path = FINAL_HIL_REPORT_ROOT / "archive_locator.json"
    report = read_json(report_path)
    report_manifest = read_json(report_manifest_path)
    archive_manifest = read_json(archive_manifest_path)
    locator = read_json(locator_path)
    require(report.get("schema") == FINAL_HIL_REPORT_SCHEMA, "Final HIL report schema mismatch")
    require(
        report_manifest.get("schema") == FINAL_HIL_REPORT_MANIFEST_SCHEMA,
        "Final HIL report-manifest schema mismatch",
    )
    require(
        archive_manifest.get("schema") == FINAL_HIL_ARCHIVE_SCHEMA,
        "Final HIL archive-manifest schema mismatch",
    )
    require(locator.get("schema") == FINAL_HIL_LOCATOR_SCHEMA, "Final HIL locator schema mismatch")
    report_id = verify_canonical_id(report, "report_id", "Final HIL report")
    report_manifest_id = verify_canonical_id(
        report_manifest, "manifest_id", "Final HIL report manifest"
    )
    archive_id = verify_canonical_id(
        archive_manifest, "archive_id", "Final HIL archive manifest"
    )
    require(
        report.get("status") == "passed_with_blocked_routes",
        "Final HIL report status mismatch",
    )
    require(
        archive_manifest.get("status") == report.get("status"),
        "Final HIL archive/report status mismatch",
    )
    require(
        report_manifest.get("report_id") == report_id
        and report_manifest.get("archive_id") == archive_id,
        "Final HIL report-manifest identity mismatch",
    )
    report_files = report_manifest.get("files")
    expected_report_files = {
        "final_hil_summary.json",
        "final_hil_summary.md",
        "final_hil_results.csv",
        "final_hil_timing_repeats.csv",
    }
    require(
        isinstance(report_files, list)
        and {item.get("path") for item in report_files} == expected_report_files,
        "Final HIL compact report set is incomplete",
    )
    for item in report_files:
        verify_recorded_file(
            FINAL_HIL_REPORT_ROOT / item["path"], item["sha256"], item["size_bytes"]
        )
    require(
        archive_manifest.get("file_count_excluding_manifest") == 415
        and len(archive_manifest.get("inventory", [])) == 415,
        "Final HIL external archive inventory count mismatch",
    )
    require(
        report.get("archive_id") == archive_id
        and report.get("contract_id") == archive_manifest.get("contract_id")
        and report.get("campaign_evidence_id")
        == archive_manifest.get("campaign_evidence_id"),
        "Final HIL report/archive lineage mismatch",
    )
    for key in ("archive_id", "contract_id", "campaign_evidence_id", "report_id"):
        require(locator.get(key) == report.get(key), f"Final HIL locator {key} mismatch")
    require(
        locator.get("report_manifest_id") == report_manifest_id,
        "Final HIL locator report-manifest ID mismatch",
    )
    external_path = EXTERNAL_EVIDENCE_ARCHIVE_ROOT / locator["archive_filename"]
    require(external_path.is_file(), f"Final HIL external archive is missing: {external_path}")
    require(
        external_path.stat().st_size == locator.get("archive_compressed_size_bytes")
        and sha256_file(external_path) == locator.get("archive_compressed_sha256"),
        "Final HIL external archive bytes differ from the locator",
    )
    totals = report.get("totals")
    require(
        totals
        == {
            "session_count": 6,
            "stage_attempts": 36,
            "balanced_timing_rows": 18_000,
            "warmup_rows_excluded": 60,
            "smoke_rows_excluded": 60,
            "full_exact_replay_rows": 337_806,
            "all_device_inferences": 355_926,
        },
        "Final HIL report totals mismatch",
    )
    expected_models = {"student_A_scratch", "student_A_rf_kd", "student_B_rf_kd"}
    model_results = report.get("model_results")
    require(
        isinstance(model_results, dict) and set(model_results) == expected_models,
        "Final HIL accepted-model set mismatch",
    )
    route_by_model = {
        "student_A_scratch": "student_A_scratch",
        "student_A_rf_kd": "student_A_rf_kd",
        "student_B_rf_kd": "student_B_rf_kd",
    }
    seed_index = ten_seed["seeds"].index(42)
    for model_key, route_key in route_by_model.items():
        expected_macro_f1 = ten_seed["routes"][route_key]["macro_f1_values"][
            seed_index
        ]
        require(
            close(
                model_results[model_key]["fp32_metrics"]["macro_f1"],
                expected_macro_f1,
            ),
            f"Final HIL model does not match ten-seed seed-42 route: {model_key}",
        )
    sessions = report.get("sessions")
    require(isinstance(sessions, list) and len(sessions) == 6, "Final HIL session count mismatch")
    expected_combinations = {
        f"{model}__{board}__usb_serial"
        for model in expected_models
        for board in ("esp32c3", "arduino_r4")
    }
    require(
        {item.get("combination_id") for item in sessions} == expected_combinations,
        "Final HIL accepted combination set mismatch",
    )
    for session in sessions:
        fidelity = session.get("fidelity", {})
        require(fidelity.get("rows") == EXPECTED_TEST_ROWS, "Final HIL row count mismatch")
        require(
            fidelity.get("mcu_vs_fixed_reference_agreement") == 1.0
            and fidelity.get("mcu_fixed_logits_exact_fraction") == 1.0,
            f"Final HIL fixed-reference disagreement: {session.get('combination_id')}",
        )
        require(
            session.get("export_id") == model_results[session["model_key"]]["export_id"],
            f"Final HIL export mismatch: {session.get('combination_id')}",
        )
    blocked = report.get("blocked_routes")
    require(
        isinstance(blocked, list)
        and len(blocked) == 1
        and blocked[0].get("model_key") == "student_B_scratch"
        and blocked[0].get("status") == "blocked_before_firmware_generation",
        "Final HIL blocked-route ledger mismatch",
    )
    require(
        blocked[0].get("fixed_vs_fp32_agreement") < blocked[0].get(
            "minimum_fixed_vs_fp32_agreement"
        )
        and blocked[0].get("absolute_macro_f1_drop")
        > blocked[0].get("maximum_absolute_macro_f1_drop"),
        "Final HIL blocked-route gates do not support exclusion",
    )
    return {
        "status": report["status"],
        "seed": report["campaign_scope"]["seed"],
        "accepted_models": sorted(expected_models),
        "blocked_models": ["student_B_scratch"],
        "boards": report["campaign_scope"]["boards"],
        "transport": "usb_serial",
        "source_predictive_protocol": ten_seed["protocol_id"],
        "session_count": totals["session_count"],
        "full_replay_rows_per_session": EXPECTED_TEST_ROWS,
        "full_exact_replay_rows": totals["full_exact_replay_rows"],
        "all_device_inferences": totals["all_device_inferences"],
        "timing_statistical_unit": report["campaign_scope"]["timing_statistical_unit"],
        "models": model_results,
        "sessions": sessions,
        "blocked_routes": blocked,
        "claim_boundary": report["claim_boundary"],
        "report": recorded_artifact(
            report_path,
            next(item for item in report_files if item["path"] == report_path.name)[
                "sha256"
            ],
            next(item for item in report_files if item["path"] == report_path.name)[
                "size_bytes"
            ],
        ),
        "report_manifest": artifact(report_manifest_path),
        "external_archive_manifest": artifact(archive_manifest_path),
        "external_archive_locator": artifact(locator_path),
        "external_archive": {
            "filename": locator["archive_filename"],
            "compressed_size_bytes": locator["archive_compressed_size_bytes"],
            "compressed_sha256": locator["archive_compressed_sha256"],
            "archive_id": archive_id,
            "file_count_including_manifest": locator[
                "archive_file_count_including_manifest"
            ],
            "storage_policy": locator["storage_policy"],
        },
    }


def validate_predecessor_registry() -> dict[str, Any]:
    inventory = verify_inventory(
        PREDECESSOR_REGISTRY_ROOT, "artifact_manifest.json", {"complete"}
    )
    registry_path = PREDECESSOR_REGISTRY_ROOT / "evidence_registry.json"
    predecessor = read_json(registry_path)
    require(
        predecessor.get("registry_id") == PREDECESSOR_REGISTRY_ID,
        "Predecessor registry ID mismatch",
    )
    return {
        "registry_id": PREDECESSOR_REGISTRY_ID,
        "registry": artifact(registry_path),
        "manifest": artifact(
            PREDECESSOR_REGISTRY_ROOT / "artifact_manifest.json"
        ),
        "inventory": inventory,
    }


def validate_open_planned_work() -> dict[str, Any]:
    campaign_path = FINAL_HIL_CAMPAIGN_ROOT / "campaign_contract.json"
    campaign = read_json(campaign_path)
    require(
        campaign.get("status") == "ready_with_blocked_routes"
        and campaign.get("transports") == ["usb_serial", "wifi_udp"]
        and campaign.get("expected_eligible_combination_count") == 12,
        "Full final-HIL campaign contract mismatch",
    )
    require(
        not CONTROLLED_XAI_ROOT.exists(),
        "Controlled ten-seed XAI output now exists; register it in a new immutable registry",
    )
    return {
        "status": "open_planned_work",
        "controlled_ten_seed_xai": {
            "status": "not_executed",
            "expected_protocol_id": "wsnds_fgds_controlled_xai_transfer_10seed_v1",
            "expected_output": repo_path(CONTROLLED_XAI_ROOT),
            "boundary": "The available SHAP result is a completed seed-42 specimen audit. It is not a substitute for the planned ten-seed scratch-controlled XAI experiment.",
        },
        "final_lineage_wifi": {
            "status": "not_executed",
            "planned_eligible_sessions": 6,
            "planned_full_replay_rows": 6 * EXPECTED_TEST_ROWS,
            "boundary": "The completed Wi-Fi campaign belongs to the distinct five-seed deployment lineage. The final ten-seed seed-42 lineage currently has USB evidence only.",
            "campaign_contract": artifact(campaign_path),
        },
    }


def validate_edge() -> dict[str, Any]:
    summary_path = EDGE_ROOT / "edge_group_aware_summary.json"
    audit_path = EDGE_ROOT / "group_aware_split_audit.json"
    long_path = EDGE_ROOT / "edge_group_aware_long.csv"
    summary = read_json(summary_path)
    audit = read_json(audit_path)
    require(summary.get("status") == "complete", "Edge summary is incomplete")
    require(summary.get("audit") == audit, "Edge summary and split audit differ")
    require(audit.get("input_dim") == 40, "Edge group-aware input dimension mismatch")
    require(
        audit.get("split_rows")
        == {"train": 1_556_588, "validation": 332_240, "test": 330_373},
        "Edge split sizes mismatch",
    )
    require(audit.get("train_test_group_overlap") == 0, "Edge group overlap is nonzero")

    route_values: dict[tuple[str, str], list[tuple[int, float]]] = {}
    with long_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, "Edge long table has no header")
        for row in reader:
            key = (row["student"], row["route"])
            route_values.setdefault(key, []).append((int(row["seed"]), float(row["macro_f1"])))
    require(sum(len(values) for values in route_values.values()) == 25, "Edge row count mismatch")

    result_routes: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    for student in ("A", "B"):
        scratch = sorted(route_values[(student, "D_Small_MLP")])
        kd = sorted(route_values[(student, "E_KD_from_RF")])
        require([seed for seed, _ in scratch] == [42, 123, 456, 789, 1001], "Edge seeds mismatch")
        require([seed for seed, _ in scratch] == [seed for seed, _ in kd], "Edge pairing mismatch")
        scratch_values = [value for _, value in scratch]
        kd_values = [value for _, value in kd]
        differences = [right - left for left, right in zip(scratch_values, kd_values, strict=True)]
        result_routes[f"student_{student}_scratch"] = summarize(scratch_values)
        result_routes[f"student_{student}_rf_kd"] = summarize(kd_values)
        paired[f"student_{student}"] = {
            "comparison": "rf_kd_minus_scratch_macro_f1",
            "difference_summary": summarize(differences),
            "wilcoxon": exact_signed_rank(differences),
        }
    teacher_values = [value for _, value in sorted(route_values[("RF", "A_RF_200")])]
    result_routes["rf_teacher_200"] = summarize(teacher_values)

    return {
        "status": "secondary_protocol_evidence",
        "evaluation_unit": "five optimizer seeds on one fixed group-aware Edge-IIoTset split",
        "input_dim": 40,
        "class_count": 15,
        "split_rows": audit["split_rows"],
        "pre_encode_group_overlap": audit["train_test_group_overlap"],
        "encoded_exact_row_overlap": {
            "train_test": audit["encoded_train_test_row_overlap"],
            "train_validation": audit["encoded_train_val_row_overlap"],
            "validation_test": audit["encoded_val_test_row_overlap"],
        },
        "random_row_test_records_in_cross_partition_pre_encode_groups_percent": audit[
            "pct_test_in_cross_partition_groups_pre_split_protocol"
        ],
        "routes_recomputed_with_sample_std": result_routes,
        "paired_tests": paired,
        "boundary": "The group-aware Edge protocol changes the split, feature representation, teacher size, and training configuration relative to the literature-style run. Score differences cannot be attributed to grouping alone.",
        "summary": artifact(summary_path),
        "split_audit": artifact(audit_path),
        "long_table": artifact(long_path),
    }


def historical_boundaries() -> dict[str, Any]:
    paths = {
        "qat": REPO_ROOT
        / "results/wsnds/confirmation_runs_v2/deployment_seed_42_qat/qat_refinement_report.json",
        "shap": REPO_ROOT
        / "results/paper_strength_e2e/shap_train_only_deployment/shap_results.json",
        "msp430": REPO_ROOT / "deployment/msp430/MSP430_CROSS_COMPILE_REPORT.md",
        "codistillation": REPO_ROOT
        / "results/leftover_e2e_closure/01_j_codistill/j_summary.json",
    }
    return {
        "archived_random_row_full_route_matrix": {
            "status": "exploratory_historical_lineage",
            "boundary": "The archived route matrix uses a random-row train-only-scaler protocol with exact raw-feature groups crossing partitions. The current controlled FG-DS route matrix is registered separately and supersedes it for current route evidence.",
        },
        "qat": {
            "status": "historical_non_fgds_lineage",
            "artifact": artifact(paths["qat"]),
            "boundary": "This QAT refinement belongs to the older 56,200-row train-only deployment lineage. It must not be combined with the 56,301-row FG-DS deployment results.",
        },
        "shap": {
            "status": "historical_non_fgds_lineage",
            "artifact": artifact(paths["shap"]),
            "boundary": "This SHAP audit explains 500 of 56,200 test rows from the older train-only deployment lineage and compares students with a freshly refitted uncalibrated RF reference. Its repeated samples are not bootstrap samples.",
        },
        "legacy_msp430": {
            "status": "legacy_static_feasibility_only",
            "artifact": artifact(paths["msp430"]),
            "boundary": "This report covers a legacy Student A fixed-point core. Current FG-DS Student A and Student B MSP430F1611 static evidence is registered separately; neither lineage is physical TelosB execution.",
        },
        "archived_codistillation_and_curriculum_extensions": {
            "status": "exploratory_historical_lineage",
            "codistillation_artifact": artifact(paths["codistillation"]),
            "boundary": "These particular archived extension artifacts are not current FG-DS evidence. Controlled co-distillation and curriculum routes are present in the current full-route matrix. The separate archived curriculum recovery repetitions share an internal RNG reset and are deterministic repeats, not independent robustness replications.",
        },
    }


def build_registry() -> dict[str, Any]:
    ten_seed_analysis, ten_seed = validate_ten_seed()
    ten_seed_contract = read_json(TEN_SEED_ROOT / "preprocessing_contract.json")
    full_routes = validate_full_routes(ten_seed_contract)
    sensitivity = validate_hyperparameter_sensitivity(ten_seed_contract)
    group_balanced = validate_group_balanced_sensitivity(ten_seed_contract)
    runtime, deployment = validate_runtime_and_deployment(ten_seed_contract)
    fixed_point_refinement = validate_fixed_point_refinement(deployment)
    msp430_static = validate_current_msp430_static(deployment)
    hil = validate_hil(runtime)
    behavioral_transfer = validate_behavioral_transfer()
    multisplit = validate_multisplit_confirmation()
    all_seed_fixed_point = validate_all_seed_fixed_point_audit()
    final_usb_hil = validate_final_usb_hil(ten_seed)
    shap, incomplete_shap = validate_exact_teacher_shap()
    edge = validate_edge()
    predecessor = validate_predecessor_registry()
    open_work = validate_open_planned_work()
    return {
        "status": "passed_with_open_planned_work",
        "registry_id": REGISTRY_ID,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "predecessor_registry": predecessor,
        "primary_wsnds_multi_seed": ten_seed,
        "controlled_full_route_matrix": full_routes,
        "rfkd_hyperparameter_sensitivity": sensitivity,
        "repeated_pattern_sensitivity": group_balanced,
        "behavioral_transfer": behavioral_transfer,
        "multisplit_core_confirmation": multisplit,
        "current_exact_lineage_xai": shap,
        "fixed_deployment_specimen": deployment,
        "all_seed_software_fixed_point_audit": all_seed_fixed_point,
        "software_only_fixed_point_refinement": fixed_point_refinement,
        "hardware_execution": hil,
        "final_usb_hardware_campaign": final_usb_hil,
        "current_msp430_static": msp430_static,
        "secondary_edge_iiotset": edge,
        "open_planned_work": open_work,
        "excluded_incomplete": {
            "shap_v2_partial_attempt": incomplete_shap,
        },
        "historical_or_non_primary": historical_boundaries(),
        "claim_rules": {
            "allowed": [
                "Report ten-seed FG-DS scratch and RF-KD statistics as primary WSN-DS predictive evidence.",
                "Report the controlled ten-seed FG-DS route matrix as current route-comparison evidence under its RNG and execution-identity boundary.",
                "Report the RF-KD factorial surface as descriptive sensitivity evidence without selecting a winning cell or replacing the primary result.",
                "Report inverse-group-size and pure-group-representative metrics as repeated-pattern sensitivity analyses while retaining row-level metrics as primary.",
                "Report the ten-seed held-out T=4 response-distribution analysis as post-hoc secondary evidence that RF-KD students are closer to their calibrated RF teachers than matched scratch students.",
                "Report the ten-split core confirmation descriptively as split-sensitivity evidence; the overlapping holdouts are not independent inferential replications.",
                "Report the seed-42 reconstructed-teacher SHAP audit as partial feature-rank alignment for one fixed deployment specimen and two output contracts.",
                "Report the 40-model software fixed-point audit with all 26 passes and 14 retained gate failures.",
                "Report the older five-seed seed-42 runtime, USB, and Wi-Fi results as execution evidence for that fixed deployment specimen.",
                "Report the final seed-42 USB campaign as six exact fixed-reference sessions for three gate-eligible models on two physical board specimens, with Student B scratch blocked before firmware generation.",
                "Report exact MCU agreement as fixed-reference execution fidelity, not as independent model accuracy replication.",
                "Report current Student A and Student B MSP430F1611 results as static cross-compile and memory-footprint evidence only.",
                "Report fixed-point refinement only as a software-only candidate experiment that has not replaced the deployed PTQ artifacts.",
                "Report Edge group-aware results as secondary protocol evidence with its exact 40-input contract and residual encoded-row overlaps.",
            ],
            "forbidden": [
                "Do not substitute the local ten-seed seed-42 checkpoint for the distinct Winterfell deployment checkpoint.",
                "Do not mix 56,200-row historical artifacts with the 56,301-row FG-DS lineage.",
                "Do not count USB and Wi-Fi executions as independent predictive samples.",
                "Do not describe Wi-Fi application retransmissions as raw UDP losslessness.",
                "Do not describe dynamic ONNX int8 weights as the MCU fixed-point method.",
                "Do not use the sensitivity response surface for post-hoc hyperparameter selection.",
                "Do not interpret SHAP rank correlation as teacher-function equivalence, causal knowledge transfer, or MCU behavior.",
                "Do not interpret response-distribution proximity as causal mechanism transfer, off-manifold decision-boundary equivalence, explanation transfer, or deployment fidelity.",
                "Do not treat the ten overlapping multisplit holdouts as ten independent datasets or use them for a formal hypothesis test.",
                "Do not present the incomplete SHAP-v2 attempt as evidence.",
                "Do not hide or discard fixed-point gate failures when describing deployment eligibility.",
                "Do not present the software-only fixed-point refinement as firmware or hardware evidence.",
                "Do not claim Student B scratch was flashed or executed in the final USB campaign.",
                "Do not substitute the older five-seed Wi-Fi campaign for the unexecuted final-lineage Wi-Fi campaign.",
                "Do not claim that the planned ten-seed scratch-controlled XAI experiment was executed.",
                "Do not present static MSP430F1611 cross-compilation as physical TelosB execution, latency, energy, radio integration, or live feature extraction.",
                "Do not present historical QAT, old SHAP, legacy MSP430, or archived random-row extension results as current FG-DS evidence.",
            ],
        },
        "registry_script": artifact(SCRIPT_PATH),
        "source_analysis_sha256": sha256_file(
            TEN_SEED_ANALYSIS_ROOT / "feature_group_10seed_analysis.json"
        ),
        "source_report_sha256": {
            "runtime": sha256_file(RUNTIME_ROOT / "runtime_report.json"),
            "usb": hil["usb_report"]["sha256"],
            "wireless": hil["wireless_report"]["sha256"],
            "edge": sha256_file(EDGE_ROOT / "edge_group_aware_summary.json"),
            "full_routes": full_routes["aggregate"]["sha256"],
            "sensitivity": sensitivity["aggregate"]["sha256"],
            "group_balanced": group_balanced["aggregate"]["sha256"],
            "behavioral_transfer": behavioral_transfer["report"]["sha256"],
            "multisplit": multisplit["report"]["sha256"],
            "shap": shap["report"]["sha256"],
            "all_seed_fixed_point": all_seed_fixed_point["report"]["sha256"],
            "fixed_point_refinement": fixed_point_refinement["report"]["sha256"],
            "final_usb_hil": final_usb_hil["report"]["sha256"],
            "msp430_static": msp430_static["summary"]["sha256"],
        },
        "verified_prediction_rows_primary": ten_seed_analysis[
            "prediction_rows_recomputed"
        ],
    }


def registry_markdown(registry: dict[str, Any]) -> str:
    ten = registry["primary_wsnds_multi_seed"]
    full = registry["controlled_full_route_matrix"]
    sensitivity = registry["rfkd_hyperparameter_sensitivity"]
    repeated = registry["repeated_pattern_sensitivity"]
    transfer = registry["behavioral_transfer"]
    multisplit = registry["multisplit_core_confirmation"]
    xai = registry["current_exact_lineage_xai"]
    all_seed_fixed = registry["all_seed_software_fixed_point_audit"]
    refinement = registry["software_only_fixed_point_refinement"]
    hil = registry["hardware_execution"]
    final_hil = registry["final_usb_hardware_campaign"]
    msp430 = registry["current_msp430_static"]
    edge = registry["secondary_edge_iiotset"]
    open_work = registry["open_planned_work"]
    lines = [
        "# CuKD-XAI FG-DS Evidence Registry",
        "",
        f"Status: `{registry['status']}`",
        f"Registry: `{registry['registry_id']}`",
        "",
        "## Primary WSN-DS Evidence",
        "",
        "The primary predictive result is the ten-seed feature-group-disjoint run with a train-only scaler. It uses one fixed split and ten optimizer seeds.",
        "",
        "| Route | Macro-F1 mean | Sample SD |",
        "|---|---:|---:|",
    ]
    labels = {
        "student_A_scratch": "Student A scratch",
        "student_A_rf_kd": "Student A RF-KD",
        "student_B_scratch": "Student B scratch",
        "student_B_rf_kd": "Student B RF-KD",
    }
    for route, label in labels.items():
        data = ten["routes"][route]
        lines.append(
            f"| {label} | {data['macro_f1_mean']:.6f} | {data['macro_f1_sample_std']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"Student A RF-KD minus scratch: {ten['paired_tests']['student_A']['difference_summary']['mean']:.6f}, exact Wilcoxon p={ten['paired_tests']['student_A']['wilcoxon']['p_value_two_sided_exact']:.6f}, Holm-adjusted p={ten['paired_tests']['student_A']['holm_adjusted_p']:.6f}.",
            "",
            f"Student B RF-KD minus scratch: {ten['paired_tests']['student_B']['difference_summary']['mean']:.6f}, exact Wilcoxon p={ten['paired_tests']['student_B']['wilcoxon']['p_value_two_sided_exact']:.6f}, Holm-adjusted p={ten['paired_tests']['student_B']['holm_adjusted_p']:.6f}.",
            "",
            "## Controlled Full-Route Evidence",
            "",
            "The complete teacher and student route matrix was rerun on the same clean split for ten seeds. The table below reports the student routes. Exact signed-rank inference and sample SD are preserved in the sealed aggregate.",
            "",
            "| Student | Route | Macro-F1 mean | Sample SD |",
            "|---|---|---:|---:|",
        ]
    )
    for student in ("student_A", "student_B"):
        for route, value in full["student_routes"][student].items():
            metric = value["macro_f1"]
            lines.append(
                f"| {student} | {route} | {metric['mean']:.6f} | {metric['sample_std']:.6f} |"
            )
    lines.extend(
        [
            "",
            full["boundary"],
            "",
            "## RF-KD Hyperparameter Sensitivity",
            "",
            "The 3 x 3 temperature-alpha surface contains 180 training jobs across two students and ten seeds. It is descriptive only; no winning cell was selected and the primary result was not replaced.",
            "",
            "| Comparison | Mean macro-F1 difference | Exact Wilcoxon p | Within-student Holm p | Global Holm p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, value in sensitivity["selected_comparisons"].items():
        lines.append(
            f"| {name} | {value['difference']['mean']:.6f} | {value['wilcoxon_exact_p']:.6f} | {value['holm_within_student_p']:.6f} | {value['holm_global_p']:.6f} |"
        )
    lines.extend(
        [
            "",
            sensitivity["boundary"],
            "",
            "## Repeated-Pattern Sensitivity",
            "",
            f"The {repeated['test_group_summary']['test_rows']:,} test records form {repeated['test_group_summary']['test_exact_feature_groups']:,} exact feature groups. There are {repeated['test_group_summary']['mixed_label_groups']} mixed-label groups in the test partition.",
            "",
            "| View | Student | RF-KD minus scratch | Exact Wilcoxon p | Holm p |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for view, students in repeated["rf_kd_minus_scratch"].items():
        for student, value in students.items():
            lines.append(
                f"| {view} | {student} | {value['differences']['mean']:.6f} | {value['exact_wilcoxon_p']:.6f} | {value['holm_within_family_p']:.6f} |"
            )
    lines.extend(
        [
            "",
            repeated["boundary"],
            "",
            "## Behavioral-Transfer Evidence",
            "",
            "The post-hoc ten-seed analysis compares each RF-KD student with its matched scratch model under the same held-out, exact-group-balanced T=4 response-distribution contract.",
            "",
            "| Student | Mean scratch-minus-RF-KD KL | Sample SD | Positive seeds | Holm-adjusted p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for student, value in transfer["primary_tests"].items():
        lines.append(
            f"| {student} | {value['transfer_gain_mean']:.6f} | {value['transfer_gain_sample_std']:.6f} | {value['positive_seed_count']} | {value['holm_adjusted_p']:.6f} |"
        )
    lines.extend(
        [
            "",
            transfer["claim_boundary"],
            "",
            "## Split-Sensitivity Confirmation",
            "",
            f"The core scratch-versus-RF-KD comparison was repeated across {multisplit['split_count']} exact-feature-group splits with two paired optimizer seeds per split ({multisplit['training_jobs']} student training jobs). Student A had a positive split-level mean RF-KD effect on all 10 splits; Student B was positive on 5 and negative on 5.",
            "",
            multisplit["claim_boundary"],
            "",
            "## Current XAI Evidence",
            "",
            f"Permutation SHAP explains a fixed stratified subset of {xai['explained_test_rows']} of {xai['test_rows']:,} test records for one seed-42 deployment specimen. The reconstructed calibrated RF teacher passed train and test output validation. Exact equivalence on synthetic masked inputs is not claimed.",
            "",
            "| Output contract | Student | Global rank rho mean | Sample SD | Mean top-5 overlap |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for condition, condition_value in xai["condition_summaries"].items():
        for student in ("student_A", "student_B"):
            value = condition_value[student]
            lines.append(
                f"| {condition} | {student} | {value['global_spearman_rho']['mean']:.6f} | {value['global_spearman_rho']['sample_std']:.6f} | {value['top_5_overlap_count']['mean']:.3f} |"
            )
    lines.extend(
        [
            "",
            f"Maximum local-accuracy residual across all 18 SHAP artifacts: {xai['global_max_local_accuracy_residual']:.3e} (gate {xai['local_accuracy_atol']:.1e}).",
            "",
            "## Runtime and Hardware Evidence",
            "",
            f"One fixed seed-42 deployment specimen was replayed over USB and Wi-Fi UDP on four board-model pairs. Each transport executed {hil['usb_full_predictions']:,} full-test predictions. All pairs matched the fixed-point reference predictions and logits exactly.",
            "",
            "| Pair | MCU macro-F1 | MCU vs FP32 | USB compute mean (us) | Wi-Fi compute mean (us) | Wi-Fi retransmissions |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label in EXPECTED_LABELS:
        row = hil["rows"][label]
        lines.append(
            f"| {label} | {row['macro_f1']:.6f} | {row['mcu_vs_fp32_agreement']:.6f} | {row['usb_device_compute_mean_us']:.3f} | {row['wireless_device_compute_mean_us']:.3f} | {row['wireless_data_retransmissions']} |"
        )
    lines.extend(
        [
            "",
            "USB and Wi-Fi use the same model artifacts and test records. They are execution replications, not independent predictive samples. Wi-Fi retransmissions are application-level retries after response timeouts.",
            "",
            "## All-Seed Software Fixed-Point Audit",
            "",
            f"The fixed-point exporter audited {all_seed_fixed['model_count']} model-seed instances. {all_seed_fixed['status_counts']['passed']} passed every software quality and exact C/Python equivalence gate; {all_seed_fixed['status_counts']['gate_failed']} were retained as gate failures rather than omitted.",
            "",
            all_seed_fixed["statistical_unit_disclosure"],
            "",
            "## Final USB Hardware Campaign",
            "",
            f"The final campaign executed {final_hil['session_count']} gate-eligible model-board sessions, each with {final_hil['full_replay_rows_per_session']:,} full-test rows. Across {final_hil['full_exact_replay_rows']:,} reported replay rows, every MCU prediction and fixed-point logit matched the fixed reference exactly.",
            "",
            "| Model | ESP32-C3 | Arduino R4 | Deployment status |",
            "|---|---|---|---|",
            "| Student A scratch | exact | exact | passed |",
            "| Student A RF-KD | exact | exact | passed |",
            "| Student B RF-KD | exact | exact | passed |",
            "| Student B scratch | not executed | not executed | blocked by fixed-point gates |",
            "",
            f"The portable archive contains {final_hil['external_archive']['file_count_including_manifest']} files and is retained outside Git with SHA-256 `{final_hil['external_archive']['compressed_sha256']}`.",
            "",
            final_hil["claim_boundary"],
            "",
            "## Software-Only Fixed-Point Refinement",
            "",
            "| Student | Source PTQ fixed macro-F1 | Refined fixed macro-F1 | Delta | Fixed vs float agreement |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for student, value in refinement["students"].items():
        lines.append(
            f"| {student} | {value['source_fixed_macro_f1']:.6f} | {value['refined_fixed_macro_f1']:.6f} | {value['fixed_macro_f1_minus_current_ptq']:.6f} | {value['fixed_vs_float_prediction_agreement']:.6f} |"
        )
    lines.extend(
        [
            "",
            refinement["boundary"],
            "",
            "## Current MSP430F1611 Static Evidence",
            "",
            "| Student | Static flash (bytes) | Static RAM lower bound (bytes) | Maximum single-function stack (bytes) |",
            "|---|---:|---:|---:|",
        ]
    )
    for student, value in msp430["students"].items():
        lines.append(
            f"| {student} | {value['static_flash_load_bytes']} | {value['static_ram_lower_bound_bytes']} | {value['maximum_single_function_stack_bytes']} |"
        )
    lines.extend(
        [
            "",
            msp430["claim_boundary"],
            "",
            "## Secondary Edge-IIoTset Evidence",
            "",
            f"The group-aware Edge run uses 40 inputs and {edge['split_rows']['train']:,}/{edge['split_rows']['validation']:,}/{edge['split_rows']['test']:,} train/validation/test rows. Pre-encode group overlap is zero; encoded exact-row overlaps remain {edge['encoded_exact_row_overlap']['train_test']}, {edge['encoded_exact_row_overlap']['train_validation']}, and {edge['encoded_exact_row_overlap']['validation_test']} for train-test, train-validation, and validation-test.",
            "",
            "## Open Planned Work",
            "",
            "Two planned experiments are not part of this registry: the ten-seed scratch-controlled XAI audit and the six final-lineage Wi-Fi sessions. The completed seed-42 SHAP specimen and older five-seed Wi-Fi campaign remain valid only within their recorded lineages.",
            "",
            open_work["controlled_ten_seed_xai"]["boundary"],
            "",
            open_work["final_lineage_wifi"]["boundary"],
            "",
            "## Boundaries",
            "",
            "The incomplete SHAP-v2 attempt is explicitly excluded. Historical 56,200-row QAT and SHAP artifacts, the legacy MSP430 report, and archived random-row extensions remain preserved but are not current FG-DS evidence. See `evidence_registry.json` for machine-readable lineage rules, exact hashes, and claim boundaries.",
            "",
        ]
    )
    return "\n".join(lines)


def claims_csv(registry: dict[str, Any]) -> str:
    rows = [
        ["evidence", "status", "allowed_use", "boundary"],
        [
            "WSN-DS ten-seed FG-DS",
            "primary",
            "Predictive means, sample SDs, per-class metrics, paired tests",
            registry["primary_wsnds_multi_seed"]["boundary"],
        ],
        [
            "Controlled ten-seed full-route matrix",
            "current route evidence",
            "Teacher and student route metrics and predeclared paired comparisons",
            registry["controlled_full_route_matrix"]["boundary"],
        ],
        [
            "RF-KD temperature-alpha surface",
            "descriptive sensitivity",
            "Sensitivity to nine fixed hyperparameter cells per student",
            registry["rfkd_hyperparameter_sensitivity"]["boundary"],
        ],
        [
            "Repeated-pattern analysis",
            "sensitivity",
            "Inverse-group-size and pure-group-representative route metrics",
            registry["repeated_pattern_sensitivity"]["boundary"],
        ],
        [
            "Ten-seed behavioral transfer",
            "post-hoc secondary evidence",
            "Held-out T=4 response-distribution proximity for RF-KD versus matched scratch",
            registry["behavioral_transfer"]["claim_boundary"],
        ],
        [
            "Ten-split core confirmation",
            "descriptive split sensitivity",
            "Direction and dispersion of split-level RF-KD-minus-scratch effects",
            registry["multisplit_core_confirmation"]["claim_boundary"],
        ],
        [
            "Exact-lineage seed-42 SHAP",
            "single-specimen XAI",
            "Partial student-teacher feature-rank alignment under T=1 and T=4 output contracts",
            registry["current_exact_lineage_xai"]["scope_boundary"],
        ],
        [
            "All-seed software fixed-point audit",
            "complete with retained failures",
            "Deployment eligibility and exact native C/Python equivalence for 40 model-seed instances",
            registry["all_seed_software_fixed_point_audit"][
                "verification_boundary"
            ],
        ],
        [
            "Seed-42 runtime and HIL",
            "primary deployment evidence",
            "Runtime conversion and execution fidelity for one fixed model specimen",
            registry["hardware_execution"]["independence_boundary"],
        ],
        [
            "Final seed-42 USB hardware campaign",
            "passed with blocked route",
            "Exact fixed-reference execution for six eligible model-board sessions",
            registry["final_usb_hardware_campaign"]["claim_boundary"],
        ],
        [
            "Seed-42 fixed-point refinement",
            "software-only candidate",
            "Validation-selected fixed-point refinement metrics and numeric audits",
            registry["software_only_fixed_point_refinement"]["boundary"],
        ],
        [
            "Current FG-DS MSP430F1611",
            "static cross-compile evidence",
            "Static flash, RAM lower-bound, and per-function stack evidence",
            registry["current_msp430_static"]["claim_boundary"],
        ],
        [
            "Edge-IIoTset group-aware",
            "secondary",
            "Protocol-sensitive robustness evidence",
            registry["secondary_edge_iiotset"]["boundary"],
        ],
        [
            "SHAP-v2 partial attempt",
            "excluded incomplete",
            "Forensic traceability only",
            registry["excluded_incomplete"]["shap_v2_partial_attempt"]["boundary"],
        ],
        [
            "Ten-seed scratch-controlled XAI",
            "not executed",
            "No evidentiary use",
            registry["open_planned_work"]["controlled_ten_seed_xai"]["boundary"],
        ],
        [
            "Final-lineage Wi-Fi HIL",
            "not executed",
            "No evidentiary use",
            registry["open_planned_work"]["final_lineage_wifi"]["boundary"],
        ],
        [
            "Historical QAT/SHAP/legacy MSP430/random-row extensions",
            "non-primary",
            "Clearly labeled historical or exploratory discussion only",
            "These artifacts do not share the current FG-DS deployment lineage.",
        ],
    ]
    output: list[str] = []
    for row in rows:
        output.append(",".join('"' + str(value).replace('"', '""') + '"' for value in row))
    return "\n".join(output) + "\n"


def write_registry(output: Path, registry: dict[str, Any]) -> None:
    require(not output.exists(), f"Refusing to overwrite existing registry: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".building")
    require(not staging.exists(), f"Registry staging directory already exists: {staging}")
    staging.mkdir()
    registry_path = staging / "evidence_registry.json"
    markdown_path = staging / "EVIDENCE_REGISTRY.md"
    claims_path = staging / "claim_boundaries.csv"
    source_path = staging / "executed_registry_source.py"
    manifest_path = staging / "artifact_manifest.json"
    atomic_write(registry_path, json.dumps(registry, indent=2) + "\n")
    atomic_write(markdown_path, registry_markdown(registry) + "\n")
    atomic_write(claims_path, claims_csv(registry))
    temporary_source = source_path.with_suffix(source_path.suffix + ".tmp")
    shutil.copyfile(SCRIPT_PATH, temporary_source)
    os.replace(temporary_source, source_path)
    files = [
        artifact(path)
        for path in (registry_path, markdown_path, claims_path, source_path)
    ]
    for entry in files:
        entry["path"] = Path(entry["path"]).name
    manifest = {
        "status": "complete",
        "protocol_id": registry["registry_id"],
        "registry_id": registry["registry_id"],
        "file_count_excluding_manifest": len(files),
        "files": files,
    }
    atomic_write(manifest_path, json.dumps(manifest, indent=2) + "\n")
    os.replace(staging, output)


def verify_existing_registry(output: Path) -> dict[str, Any]:
    inventory = verify_inventory(output, "artifact_manifest.json", {"complete"})
    persisted = read_json(output / "evidence_registry.json")
    require(
        persisted.get("registry_id") == REGISTRY_ID,
        "Persisted registry ID mismatch",
    )
    source_snapshot = output / "executed_registry_source.py"
    require(
        source_snapshot.read_bytes() == SCRIPT_PATH.read_bytes(),
        "Persisted registry source differs from the executing source",
    )
    current = build_registry()
    persisted_comparable = dict(persisted)
    current_comparable = dict(current)
    persisted_comparable.pop("generated_utc", None)
    current_comparable.pop("generated_utc", None)
    require(
        persisted_comparable == current_comparable,
        "Persisted registry content differs from current source evidence",
    )
    return inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Verify an existing sealed registry without modifying it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    try:
        output.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise RuntimeError("Output directory must remain inside the repository") from exc
    if args.verify_existing:
        verify_existing_registry(output)
    else:
        registry = build_registry()
        write_registry(output, registry)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
