"""Build a fail-closed evidence registry for the current FG-DS result lineages.

The registry does not recompute models or modify historical artifacts. It verifies
the preserved multi-seed, deployment, runtime, USB HIL, Wi-Fi HIL, and Edge
artifacts before recording which claims each lineage can support.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]

TEN_SEED_PROTOCOL = "wsnds_feature_group_split_train_only_scaler_10seed_v2"
DEPLOYMENT_PROTOCOL = "wsnds_feature_group_split_train_only_scaler_5seed_v1"
RUNTIME_PROTOCOL = "wsnds_fgds_seed42_exact_runtime_v1"
WIRELESS_PROTOCOL = "cukd_fgds_wifi_udp_session_v2"
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
WIRELESS_REPORT_ROOT = WIRELESS_ROOT / "final_report_20260811T090515Z"
EDGE_ROOT = REPO_ROOT / "results/leftover_e2e_closure/04_edge_group_aware"
DEFAULT_OUTPUT = REPO_ROOT / "results/evidence_registry/fgds_20260811"


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
        "full_route_matrix": {
            "status": "exploratory_historical_lineage",
            "boundary": "The full route matrix uses a random-row train-only-scaler protocol with exact raw-feature groups crossing partitions. It is not primary FG-DS confirmation evidence.",
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
        "msp430": {
            "status": "legacy_static_feasibility_only",
            "artifact": artifact(paths["msp430"]),
            "boundary": "The MSP430 report covers a legacy Student A fixed-point core and static cross-compile footprint. It is not physical execution and is not the exact current FG-DS deployment model.",
        },
        "codistillation_and_curriculum": {
            "status": "exploratory_historical_lineage",
            "codistillation_artifact": artifact(paths["codistillation"]),
            "boundary": "Co-distillation and curriculum extensions were not rerun under FG-DS. Curriculum recovery repetitions share an internal RNG reset and are deterministic repeats, not independent robustness replications.",
        },
    }


def build_registry() -> dict[str, Any]:
    ten_seed_analysis, ten_seed = validate_ten_seed()
    ten_seed_contract = read_json(TEN_SEED_ROOT / "preprocessing_contract.json")
    runtime, deployment = validate_runtime_and_deployment(ten_seed_contract)
    hil = validate_hil(runtime)
    edge = validate_edge()
    return {
        "status": "passed",
        "registry_id": "cukd_fgds_evidence_registry_20260811_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "primary_wsnds_multi_seed": ten_seed,
        "fixed_deployment_specimen": deployment,
        "hardware_execution": hil,
        "secondary_edge_iiotset": edge,
        "historical_or_non_primary": historical_boundaries(),
        "claim_rules": {
            "allowed": [
                "Report ten-seed FG-DS scratch and RF-KD statistics as primary WSN-DS predictive evidence.",
                "Report seed-42 runtime, USB, and Wi-Fi results as execution evidence for one fixed deployment specimen.",
                "Report exact MCU agreement as fixed-reference execution fidelity, not as independent model accuracy replication.",
                "Report Edge group-aware results as secondary protocol evidence with its exact 40-input contract and residual encoded-row overlaps.",
            ],
            "forbidden": [
                "Do not substitute the local ten-seed seed-42 checkpoint for the distinct Winterfell deployment checkpoint.",
                "Do not mix 56,200-row historical artifacts with the 56,301-row FG-DS lineage.",
                "Do not count USB and Wi-Fi executions as independent predictive samples.",
                "Do not describe Wi-Fi application retransmissions as raw UDP losslessness.",
                "Do not describe dynamic ONNX int8 weights as the MCU fixed-point method.",
                "Do not present historical QAT, SHAP, MSP430, curriculum, or full-route results as current FG-DS evidence.",
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
        },
        "verified_prediction_rows_primary": ten_seed_analysis[
            "prediction_rows_recomputed"
        ],
    }


def registry_markdown(registry: dict[str, Any]) -> str:
    ten = registry["primary_wsnds_multi_seed"]
    hil = registry["hardware_execution"]
    edge = registry["secondary_edge_iiotset"]
    lines = [
        "# CuKD-XAI FG-DS Evidence Registry",
        "",
        f"Status: `{registry['status']}`",
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
            "## Secondary Edge-IIoTset Evidence",
            "",
            f"The group-aware Edge run uses 40 inputs and {edge['split_rows']['train']:,}/{edge['split_rows']['validation']:,}/{edge['split_rows']['test']:,} train/validation/test rows. Pre-encode group overlap is zero; encoded exact-row overlaps remain {edge['encoded_exact_row_overlap']['train_test']}, {edge['encoded_exact_row_overlap']['train_validation']}, and {edge['encoded_exact_row_overlap']['validation_test']} for train-test, train-validation, and validation-test.",
            "",
            "## Boundaries",
            "",
            "Historical 56,200-row QAT and SHAP artifacts, the legacy MSP430 static report, and the random-row full-route/curriculum/co-distillation evidence are retained but are not primary FG-DS evidence. See `evidence_registry.json` for the machine-readable lineage rules and source hashes.",
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
            "Seed-42 runtime and HIL",
            "primary deployment evidence",
            "Runtime conversion and execution fidelity for one fixed model specimen",
            registry["hardware_execution"]["independence_boundary"],
        ],
        [
            "Edge-IIoTset group-aware",
            "secondary",
            "Protocol-sensitive robustness evidence",
            registry["secondary_edge_iiotset"]["boundary"],
        ],
        [
            "Historical QAT/SHAP/MSP430/full-route extensions",
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
    output.mkdir(parents=True, exist_ok=True)
    registry_path = output / "evidence_registry.json"
    markdown_path = output / "EVIDENCE_REGISTRY.md"
    claims_path = output / "claim_boundaries.csv"
    manifest_path = output / "artifact_manifest.json"
    atomic_write(registry_path, json.dumps(registry, indent=2) + "\n")
    atomic_write(markdown_path, registry_markdown(registry))
    atomic_write(claims_path, claims_csv(registry))
    files = [artifact(path) for path in (registry_path, markdown_path, claims_path)]
    for entry in files:
        entry["path"] = Path(entry["path"]).name
    manifest = {
        "status": "complete",
        "registry_id": registry["registry_id"],
        "file_count_excluding_manifest": len(files),
        "files": files,
    }
    atomic_write(manifest_path, json.dumps(manifest, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    try:
        output.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise RuntimeError("Output directory must remain inside the repository") from exc
    registry = build_registry()
    write_registry(output, registry)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
