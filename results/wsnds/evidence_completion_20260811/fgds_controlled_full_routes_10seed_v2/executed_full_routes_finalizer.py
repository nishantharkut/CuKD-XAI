#!/usr/bin/env python3
"""Finalize the controlled full-route run with corrected verification and inference.

The executing training runner stores all probabilities as float32. Its post-run
verifier promotes those arrays to float64 before recomputing ECE, which changes
neural ECE relative to the original float32 calculation. The uncalibrated RF is
the inverse case: its executed route explicitly calculated metrics after a
float64-to-float32-to-float64 persistence round trip. This additive finalizer
preserves the executed runner unchanged, reproduces those route-specific
representations, and replaces its small-sample Wilcoxon approximation with
exact signed-rank enumeration before aggregation. It never trains or modifies
a seed directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import run_fgds_full_routes as runner


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
RUNNER_PATH = Path(runner.__file__).resolve()
COMMON_PATH = (
    REPO_ROOT / "experiments/wsnds/leakage_free_rerun/tier15_common.py"
).resolve()
FINALIZER_PROTOCOL_ID = "wsnds_fgds_controlled_full_routes_float32_ece_finalizer_v1"
FINALIZATION_CONTRACT_NAME = "finalization_contract.json"
RUNNER_SNAPSHOT_NAME = "executed_full_routes_source.py"
COMMON_SNAPSHOT_NAME = "bound_tier15_common.py"
FINALIZER_SNAPSHOT_NAME = "executed_full_routes_finalizer.py"
NEURAL_ECE_ATOL = 2e-9
RF_PERSISTED_ECE_ATOL = 2e-9
RF_PREDICTION_NAME = "teacher_A_RF_500_uncalibrated_test_predictions.npz"


class FinalizationError(RuntimeError):
    """Raised when the preserved run cannot be finalized exactly."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalizationError(f"Cannot read JSON evidence {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FinalizationError(f"Expected a JSON object: {path}")
    return value


def fsync_existing_file(path: Path) -> None:
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_existing_file(path)
    fsync_directory(path.parent)


def source_identity() -> dict[str, str]:
    return {
        "runner": sha256_file(RUNNER_PATH),
        "common": sha256_file(COMMON_PATH),
        "finalizer": sha256_file(SCRIPT_PATH),
    }


def assert_source_identity(expected: dict[str, str]) -> None:
    if source_identity() != expected:
        raise FinalizationError("A bound source changed during finalization")


def assert_finalization_identity(
    output_root: Path,
    expected_execution_contract_sha256: str,
    expected_sources: dict[str, str],
) -> None:
    _, observed_contract_sha256 = validate_execution_identity(output_root)
    if observed_contract_sha256 != expected_execution_contract_sha256:
        raise FinalizationError("Execution contract changed during finalization")
    assert_source_identity(expected_sources)


def validated_finalization_environment(
    execution_contract: dict[str, Any],
) -> dict[str, Any]:
    expected = execution_contract.get("environment")
    if not isinstance(expected, dict):
        raise FinalizationError("Execution contract has no environment record")
    if expected.get("device") != "cuda":
        raise FinalizationError("The bound execution environment is not CUDA")

    runner.set_seed(runner.PUBLICATION_SEEDS[0])
    device = runner.torch.device("cuda")
    observed = runner.environment_record(device)
    if observed != expected:
        raise FinalizationError("Finalization environment differs from execution environment")

    warn_only = bool(runner.torch.is_deterministic_algorithms_warn_only_enabled())
    controls = {
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_version": runner.torch.backends.cudnn.version(),
        "cudnn_available": bool(runner.torch.backends.cudnn.is_available()),
        "cudnn_deterministic": bool(runner.torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(runner.torch.backends.cudnn.benchmark),
        "deterministic_algorithms_warn_only": warn_only,
        "cuda_device_index": int(runner.torch.cuda.current_device()),
        "cuda_device_count": int(runner.torch.cuda.device_count()),
        "cuda_device_capability": list(runner.torch.cuda.get_device_capability(device)),
    }
    if controls["cublas_workspace_config"] != ":4096:8":
        raise FinalizationError("CUBLAS_WORKSPACE_CONFIG is not the bound deterministic value")
    if not observed["deterministic_algorithms_enabled"] or warn_only:
        raise FinalizationError("PyTorch deterministic algorithms are not fail-closed")
    if controls["cudnn_available"] and (
        not controls["cudnn_deterministic"] or controls["cudnn_benchmark"]
    ):
        raise FinalizationError("cuDNN determinism controls differ")
    return {
        "execution_environment_exact_match": True,
        "environment": observed,
        "determinism_controls": controls,
    }


def assert_finalization_environment(
    execution_contract: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    if validated_finalization_environment(execution_contract) != expected:
        raise FinalizationError("Finalization environment changed during finalization")


def corrected_metrics_from_npz_predictions(
    path: Path,
    expected_indices: np.ndarray,
    expected_labels: np.ndarray,
    expected_metrics: dict[str, Any],
) -> np.ndarray:
    """Reconcile metrics in the persisted probability representation."""
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "source_row_index",
            "true_label",
            "probability",
            "predicted_label",
        }
        if set(payload.files) != required:
            raise FinalizationError(f"Prediction NPZ schema differs: {path}")
        source_indices = payload["source_row_index"]
        labels = payload["true_label"]
        probabilities = payload["probability"]
        predictions = payload["predicted_label"]
    if probabilities.dtype != np.float32:
        raise FinalizationError(f"Prediction probabilities are not float32: {path}")
    if not np.array_equal(source_indices, expected_indices):
        raise FinalizationError(f"Prediction NPZ indices differ: {path}")
    if not np.array_equal(labels, expected_labels):
        raise FinalizationError(f"Prediction NPZ labels differ: {path}")
    if probabilities.shape != (len(expected_labels), len(runner.CLASS_NAMES)):
        raise FinalizationError(f"Prediction NPZ probability shape differs: {path}")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise FinalizationError(f"Prediction NPZ probabilities are invalid: {path}")
    if not np.allclose(
        probabilities.sum(axis=1), 1.0, rtol=0.0, atol=2e-6
    ):
        raise FinalizationError(f"Prediction NPZ probabilities do not sum to one: {path}")
    if not np.array_equal(predictions, probabilities.argmax(axis=1)):
        raise FinalizationError(f"Prediction NPZ labels differ from argmax: {path}")

    metric_probabilities = (
        probabilities.astype(np.float64)
        if path.name == RF_PREDICTION_NAME
        else probabilities
    )
    recomputed = runner.classification_metrics(labels, metric_probabilities)
    recomputed["ece_15_bin"] = runner.expected_calibration_error(
        metric_probabilities, labels
    )
    for key in [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "per_class_precision",
        "per_class_recall",
        "per_class_f1",
        "per_class_support",
        "confusion_matrix",
    ]:
        if key not in expected_metrics or not np.allclose(
            np.asarray(recomputed[key], dtype=np.float64),
            np.asarray(expected_metrics[key], dtype=np.float64),
            rtol=0.0,
            atol=2e-9,
        ):
            raise FinalizationError(
                f"Float32-faithful NPZ metric differs for {key}: {path}"
            )
    ece_atol = (
        RF_PERSISTED_ECE_ATOL
        if path.name == RF_PREDICTION_NAME
        else NEURAL_ECE_ATOL
    )
    if "ece_15_bin" not in expected_metrics or not np.isclose(
        float(recomputed["ece_15_bin"]),
        float(expected_metrics["ece_15_bin"]),
        rtol=0.0,
        atol=ece_atol,
    ):
        raise FinalizationError(f"Serialized NPZ ECE differs beyond its gate: {path}")
    return metric_probabilities


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
    nonzero = [float(value) for value in differences if value != 0.0]
    zero_count = len(differences) - len(nonzero)
    if not nonzero:
        return {
            "statistic": 0.0,
            "p_value_two_sided_exact": 1.0,
            "nonzero_pairs": 0,
            "positive_pairs": 0,
            "negative_pairs": 0,
            "zero_pairs": zero_count,
            "rank_ties_present": False,
            "enumerated_sign_assignments": 1,
        }
    absolute = [abs(value) for value in nonzero]
    ranks = average_ranks(absolute)
    total_rank = sum(ranks)
    observed_positive = sum(
        rank
        for rank, difference in zip(ranks, nonzero, strict=True)
        if difference > 0.0
    )
    observed_statistic = min(observed_positive, total_rank - observed_positive)
    assignments = 1 << len(ranks)
    extreme = 0
    for mask in range(assignments):
        positive = sum(
            rank for index, rank in enumerate(ranks) if mask & (1 << index)
        )
        statistic = min(positive, total_rank - positive)
        if statistic <= observed_statistic + 1e-15:
            extreme += 1
    return {
        "statistic": float(observed_statistic),
        "p_value_two_sided_exact": float(extreme / assignments),
        "nonzero_pairs": len(nonzero),
        "positive_pairs": sum(value > 0.0 for value in nonzero),
        "negative_pairs": sum(value < 0.0 for value in nonzero),
        "zero_pairs": zero_count,
        "rank_ties_present": len(set(absolute)) != len(absolute),
        "enumerated_sign_assignments": assignments,
    }


def corrected_paired_test(left: list[float], right: list[float]) -> dict[str, Any]:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape or left_array.ndim != 1:
        raise FinalizationError("Paired route vectors have incompatible shapes")
    difference = left_array - right_array
    signed_rank = exact_signed_rank(difference.tolist())
    exact_p = float(signed_rank["p_value_two_sided_exact"])
    return {
        "left_values": left_array.tolist(),
        "right_values": right_array.tolist(),
        "difference": runner.scalar_summary(difference.tolist()),
        "wilcoxon": {
            "statistic": float(signed_rank["statistic"]),
            "p_value_two_sided": exact_p,
            "p_value_two_sided_exact": exact_p,
            "method": "exact_signed_rank_enumeration",
            "zero_method": "wilcox",
            "zero_difference_count": int(signed_rank["zero_pairs"]),
            "nonzero_difference_count": int(signed_rank["nonzero_pairs"]),
            "positive_pair_count": int(signed_rank["positive_pairs"]),
            "negative_pair_count": int(signed_rank["negative_pairs"]),
            "rank_ties_present": bool(signed_rank["rank_ties_present"]),
            "enumerated_sign_assignments": int(
                signed_rank["enumerated_sign_assignments"]
            ),
        },
        "exact_sign_flip_mean_difference_p_two_sided": runner.exact_sign_flip_p(
            difference
        ),
    }


def atomic_copy(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination = destination.resolve()
    if destination.is_file():
        if sha256_file(destination) != sha256_file(source):
            raise FinalizationError(f"Existing source snapshot differs: {destination}")
        return
    if destination.exists():
        raise FinalizationError(f"Snapshot destination is not a file: {destination}")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with source.open("rb") as source_handle, temporary.open("wb") as output_handle:
        shutil.copyfileobj(source_handle, output_handle)
        output_handle.flush()
        os.fsync(output_handle.fileno())
    os.replace(temporary, destination)
    fsync_existing_file(destination)
    fsync_directory(destination.parent)


def validate_execution_identity(
    output_root: Path,
) -> tuple[dict[str, Any], str]:
    contract_path = output_root / "execution_contract.json"
    contract = read_json(contract_path)
    contract_sha256 = sha256_file(contract_path)
    if (
        contract.get("protocol_id") != runner.PROTOCOL_ID
        or contract.get("seeds") != runner.PUBLICATION_SEEDS
        or contract.get("script_sha256") != sha256_file(RUNNER_PATH)
        or contract.get("common_module_sha256") != sha256_file(COMMON_PATH)
    ):
        raise FinalizationError("Execution contract does not match preserved sources")
    encoded = json.dumps(
        {key: value for key, value in contract.items() if key != "execution_fingerprint_sha256"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if contract.get("execution_fingerprint_sha256") != hashlib.sha256(encoded).hexdigest():
        raise FinalizationError("Execution-contract fingerprint differs")
    return contract, contract_sha256


def complete_seed_ids(output_root: Path) -> list[int]:
    completed: list[int] = []
    for seed in runner.PUBLICATION_SEEDS:
        seed_root = output_root / f"seed_{seed}"
        if (seed_root / "seed_completion.json").is_file() and (
            seed_root / "artifact_manifest.json"
        ).is_file():
            completed.append(seed)
    return completed


def verify_seed_set(
    output_root: Path,
    seeds: list[int],
    contract_sha256: str,
    context: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    original = runner.metrics_from_npz_predictions
    runner.metrics_from_npz_predictions = corrected_metrics_from_npz_predictions
    try:
        completions = {
            seed: runner.verify_completed_seed_output(
                output_root / f"seed_{seed}",
                seed,
                contract_sha256,
                context,
            )
            for seed in seeds
        }
    finally:
        runner.metrics_from_npz_predictions = original
    return completions


def ece_roundtrip_audit(output_root: Path) -> dict[str, Any]:
    neural_deltas: list[float] = []
    rf_deltas: list[float] = []
    for seed in runner.PUBLICATION_SEEDS:
        seed_root = output_root / f"seed_{seed}"
        completion = read_json(seed_root / "seed_completion.json")
        records = [
            value
            for value in completion["teacher_results"].values()
            if "prediction_file" in value
        ]
        records.extend(
            value
            for routes in completion["student_results"].values()
            for value in routes.values()
            if "prediction_file" in value
        )
        for record in records:
            prediction_path = seed_root / record["prediction_file"]
            with np.load(prediction_path, allow_pickle=False) as payload:
                probabilities = payload["probability"]
                labels = payload["true_label"]
            metric_probabilities = (
                probabilities.astype(np.float64)
                if prediction_path.name == RF_PREDICTION_NAME
                else probabilities
            )
            observed = runner.expected_calibration_error(
                metric_probabilities, labels
            )
            delta = abs(observed - float(record["metrics"]["ece_15_bin"]))
            target = (
                rf_deltas
                if prediction_path.name == RF_PREDICTION_NAME
                else neural_deltas
            )
            target.append(float(delta))
    if not neural_deltas or not rf_deltas:
        raise FinalizationError("ECE round-trip audit found an empty route family")
    if max(neural_deltas) > NEURAL_ECE_ATOL:
        raise FinalizationError("Neural float32 ECE exceeds its reconciliation gate")
    if max(rf_deltas) > RF_PERSISTED_ECE_ATOL:
        raise FinalizationError("RF serialized ECE exceeds its reconciliation gate")
    return {
        "neural_artifact_count": len(neural_deltas),
        "neural_float32_max_abs_delta": max(neural_deltas),
        "neural_float32_gate": NEURAL_ECE_ATOL,
        "rf_artifact_count": len(rf_deltas),
        "rf_persisted_roundtrip_max_abs_delta": max(rf_deltas),
        "rf_persisted_roundtrip_gate": RF_PERSISTED_ECE_ATOL,
    }


def exact_inference_audit(output_root: Path) -> dict[str, Any]:
    aggregate = read_json(output_root / "aggregate_results.json")
    required_header = {
        "protocol_id": runner.PROTOCOL_ID,
        "status": "complete",
        "seeds": list(runner.PUBLICATION_SEEDS),
        "seed_count": len(runner.PUBLICATION_SEEDS),
        "aliases_excluded_from_inference": runner.ALIASES,
        "standard_deviation_definition": (
            "sample SD across algorithmic run seeds (ddof=1)"
        ),
        "holm_families": ["teacher", *runner.STUDENT_SPECS.keys()],
    }
    for key, expected in required_header.items():
        if aggregate.get(key) != expected:
            raise FinalizationError(f"Aggregate contract differs for {key}")
    observed_tests = aggregate.get("paired_route_tests")
    if not isinstance(observed_tests, dict) or not observed_tests:
        raise FinalizationError("Aggregate paired-route tests are absent")
    completions = {
        seed: read_json(output_root / f"seed_{seed}" / "seed_completion.json")
        for seed in runner.PUBLICATION_SEEDS
    }
    expected_teacher_aggregate = {
        route: runner.metric_aggregate(
            [
                completions[seed]["teacher_results"][route]["metrics"]
                for seed in runner.PUBLICATION_SEEDS
            ]
        )
        for route in runner.TEACHER_ROUTES
    }
    expected_student_aggregate = {
        student: {
            route: runner.metric_aggregate(
                [
                    completions[seed]["student_results"][student][route]["metrics"]
                    for seed in runner.PUBLICATION_SEEDS
                ]
            )
            for route in runner.STUDENT_ROUTES
        }
        for student in runner.STUDENT_SPECS
    }
    if aggregate.get("teacher_aggregate") != expected_teacher_aggregate:
        raise FinalizationError("Teacher aggregate differs from seed evidence")
    if aggregate.get("student_aggregate") != expected_student_aggregate:
        raise FinalizationError("Student aggregate differs from seed evidence")

    expected_tests: dict[str, dict[str, Any]] = {}
    for left, right in runner.TEACHER_COMPARISONS:
        name = f"teacher:{left}_minus_{right}"
        expected_tests[name] = corrected_paired_test(
            expected_teacher_aggregate[left]["macro_f1"]["values"],
            expected_teacher_aggregate[right]["macro_f1"]["values"],
        )
        expected_tests[name]["family"] = "teacher"
    for student in runner.STUDENT_SPECS:
        for left, right in runner.STUDENT_COMPARISONS:
            name = f"{student}:{left}_minus_{right}"
            expected_tests[name] = corrected_paired_test(
                expected_student_aggregate[student][left]["macro_f1"]["values"],
                expected_student_aggregate[student][right]["macro_f1"]["values"],
            )
            expected_tests[name]["family"] = student
    expected_test_count = len(runner.TEACHER_COMPARISONS) + len(
        runner.STUDENT_SPECS
    ) * len(runner.STUDENT_COMPARISONS)
    if len(expected_tests) != expected_test_count:
        raise FinalizationError("Expected paired-route contrast count differs")
    if set(observed_tests) != set(expected_tests):
        raise FinalizationError("Aggregate paired-route contrast set differs")
    for family in ["teacher", *runner.STUDENT_SPECS.keys()]:
        family_tests = {
            name: value
            for name, value in expected_tests.items()
            if value["family"] == family
        }
        runner.apply_holm(
            family_tests,
            lambda value: value["wilcoxon"]["p_value_two_sided"],
            "holm_adjusted_wilcoxon_within_family_p",
        )
        runner.apply_holm(
            family_tests,
            lambda value: value["exact_sign_flip_mean_difference_p_two_sided"],
            "holm_adjusted_sign_flip_within_family_p",
        )
    runner.apply_holm(
        expected_tests,
        lambda value: value["wilcoxon"]["p_value_two_sided"],
        "holm_adjusted_wilcoxon_global_p",
    )
    runner.apply_holm(
        expected_tests,
        lambda value: value["exact_sign_flip_mean_difference_p_two_sided"],
        "holm_adjusted_sign_flip_global_p",
    )
    if observed_tests != expected_tests:
        raise FinalizationError("Aggregate exact paired inference differs")
    assignments = [
        int(test["wilcoxon"]["enumerated_sign_assignments"])
        for test in expected_tests.values()
    ]
    return {
        "test_count": len(expected_tests),
        "method": "exact_signed_rank_enumeration",
        "maximum_enumerated_sign_assignments": max(assignments),
        "holm_scopes": ["within_route_family", "global"],
        "separate_exact_sign_flip_mean_difference_test_retained": True,
    }


def aggregate_with_corrections(
    output_root: Path,
    seeds: list[int],
    execution_contract_sha256: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    original_verifier = runner.metrics_from_npz_predictions
    original_paired_test = runner.paired_test
    runner.metrics_from_npz_predictions = corrected_metrics_from_npz_predictions
    runner.paired_test = corrected_paired_test
    try:
        return runner.aggregate(
            output_root,
            seeds,
            execution_contract_sha256,
            context,
        )
    finally:
        runner.metrics_from_npz_predictions = original_verifier
        runner.paired_test = original_paired_test


def seed_manifest_records(output_root: Path) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    for seed in runner.PUBLICATION_SEEDS:
        root = output_root / f"seed_{seed}"
        completion = root / "seed_completion.json"
        manifest = root / "artifact_manifest.json"
        if not completion.is_file() or not manifest.is_file():
            raise FinalizationError(f"Seed evidence is incomplete: {seed}")
        records[str(seed)] = {
            "completion_sha256": sha256_file(completion),
            "manifest_sha256": sha256_file(manifest),
        }
    return records


def assert_seed_identity(
    output_root: Path,
    expected: dict[str, dict[str, str]],
) -> None:
    if seed_manifest_records(output_root) != expected:
        raise FinalizationError("Bound seed evidence changed during finalization")


def build_semantic_contract(
    output_root: Path,
    execution_contract_sha256: str,
    bound_sources: dict[str, str] | None = None,
    bound_seed_evidence: dict[str, dict[str, str]] | None = None,
    finalization_environment: dict[str, Any] | None = None,
    continuation_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    aggregate_path = output_root / "aggregate_results.json"
    sources = source_identity() if bound_sources is None else bound_sources
    seeds = (
        seed_manifest_records(output_root)
        if bound_seed_evidence is None
        else bound_seed_evidence
    )
    if finalization_environment is None:
        execution_contract, _ = validate_execution_identity(output_root)
        finalization_environment = validated_finalization_environment(
            execution_contract
        )
    if continuation_provenance is None:
        continuation_provenance = validated_continuation_provenance(
            output_root,
            execution_contract_sha256,
            finalization_environment,
        )
    return {
        "schema_version": 4,
        "protocol_id": FINALIZER_PROTOCOL_ID,
        "status": "complete",
        "training_performed": False,
        "primary_protocol_id": runner.PROTOCOL_ID,
        "execution_contract_sha256": execution_contract_sha256,
        "executed_runner_sha256": sources["runner"],
        "bound_common_module_sha256": sources["common"],
        "finalizer_source_sha256": sources["finalizer"],
        "aggregate_results_sha256": sha256_file(aggregate_path),
        "seed_evidence": seeds,
        "finalization_environment": finalization_environment,
        "continuation_provenance": continuation_provenance,
        "correction": {
            "field": "ece_15_bin",
            "stored_probability_dtype": "float32",
            "neural_verification_dtype": "float32",
            "rf_verification_dtype": "persisted float32 promoted to float64",
            "reason": (
                "The executed runner promoted persisted float32 probabilities to "
                "float64 only during post-run verification. Neural ECE is recomputed "
                "in the original float32 representation. The uncalibrated RF route "
                "is recomputed from persisted float32 promoted to float64, matching "
                "the explicit persistence round trip in the executed route."
            ),
            "model_training_or_prediction_changed": False,
            "non_ece_metrics_changed": False,
            "roundtrip_audit": ece_roundtrip_audit(output_root),
        },
        "inference_correction": {
            "field": "paired_route_tests.wilcoxon",
            "executed_runner_method": "scipy_approximation",
            "finalized_method": "exact_signed_rank_enumeration",
            "reason": (
                "The route matrix contains ten paired algorithmic seeds. Exact signed-rank "
                "enumeration is feasible and matches the clean-base analysis contract; the "
                "executed runner's forced asymptotic approximation is not used for final "
                "paired inference."
            ),
            "training_predictions_or_per_seed_metrics_changed": False,
            "audit": exact_inference_audit(output_root),
        },
        "inferential_boundary": (
            "The primary Wilcoxon field in aggregate_results.json uses exact signed-rank "
            "enumeration with average ranks for ties and Wilcox zero handling. Exact "
            "sign-flip tests of the paired mean difference are reported separately."
        ),
        "route_boundary": (
            "J_CoDistill_RF_CL is multi-teacher RF-plus-curriculum distillation with "
            "a distinct training schedule; route differences cannot isolate the effect "
            "of adding a second teacher."
        ),
        "lifecycle_boundary": (
            "Continuation and finalization use one cooperative lifecycle lock. The "
            "unchanged original runner does not inspect that lock and must not be "
            "launched concurrently; detected artifact interference is rejected."
        ),
    }


def verify_finalized_root(
    output_root: Path,
    execution_contract_sha256: str,
    context: dict[str, Any],
    bound_sources: dict[str, str] | None = None,
    bound_seed_evidence: dict[str, dict[str, str]] | None = None,
    finalization_environment: dict[str, Any] | None = None,
    continuation_provenance: dict[str, Any] | None = None,
) -> None:
    sources = source_identity() if bound_sources is None else bound_sources
    seeds = (
        seed_manifest_records(output_root)
        if bound_seed_evidence is None
        else bound_seed_evidence
    )
    assert_seed_identity(output_root, seeds)
    verify_seed_set(
        output_root,
        list(runner.PUBLICATION_SEEDS),
        execution_contract_sha256,
        context,
    )
    for expected_hash, name in [
        (sources["runner"], RUNNER_SNAPSHOT_NAME),
        (sources["common"], COMMON_SNAPSHOT_NAME),
        (sources["finalizer"], FINALIZER_SNAPSHOT_NAME),
    ]:
        snapshot = output_root / name
        if not snapshot.is_file() or sha256_file(snapshot) != expected_hash:
            raise FinalizationError(f"Finalized source snapshot differs: {snapshot}")
    contract_path = output_root / FINALIZATION_CONTRACT_NAME
    observed = read_json(contract_path)
    expected = build_semantic_contract(
        output_root,
        execution_contract_sha256,
        sources,
        seeds,
        finalization_environment,
        continuation_provenance,
    )
    generated = observed.get("generated_at_utc")
    if not isinstance(generated, str) or not generated:
        raise FinalizationError("Finalization timestamp is absent")
    if {key: value for key, value in observed.items() if key != "generated_at_utc"} != expected:
        raise FinalizationError("Finalization contract semantics differ")
    assert_seed_identity(output_root, seeds)
    runner.verify_root_manifest(output_root, runner.PROTOCOL_ID)


def lifecycle_module() -> Any:
    from . import continue_fgds_full_routes as continuation

    return continuation


@contextmanager
def lifecycle_lock(output_root: Path) -> Any:
    continuation = lifecycle_module()
    try:
        with continuation.ExclusiveRunLock(
            output_root.parent / continuation.LOCK_NAME,
        ) as active_lock:
            active_lock.assert_owned()
            yield active_lock
            active_lock.assert_owned()
    except continuation.ContinuationError as exc:
        raise FinalizationError(f"Lifecycle lock failed: {exc}") from exc


def validated_continuation_provenance(
    output_root: Path,
    execution_contract_sha256: str,
    finalization_environment: dict[str, Any],
) -> dict[str, Any]:
    continuation = lifecycle_module()
    environment = {
        "execution_contract_environment": finalization_environment["environment"],
        "supplemental_determinism": continuation.supplemental_environment_record(),
    }
    try:
        return continuation.verify_completed_continuation(
            output_root,
            execution_contract_sha256,
            environment,
        )
    except continuation.ContinuationError as exc:
        raise FinalizationError(f"Continuation provenance differs: {exc}") from exc


def validated_locked_inputs(
    output_root: Path,
    execution_contract: dict[str, Any],
    execution_contract_sha256: str,
    bound_sources: dict[str, str],
    context: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, str]], dict[str, Any]]:
    assert_finalization_identity(
        output_root,
        execution_contract_sha256,
        bound_sources,
    )
    completed = complete_seed_ids(output_root)
    if completed != list(runner.PUBLICATION_SEEDS):
        raise FinalizationError("The publication seed set changed under lifecycle lock")
    verify_seed_set(output_root, completed, execution_contract_sha256, context)
    finalization_environment = validated_finalization_environment(execution_contract)
    continuation_provenance = validated_continuation_provenance(
        output_root,
        execution_contract_sha256,
        finalization_environment,
    )
    bound_seed_evidence = seed_manifest_records(output_root)
    assert_seed_identity(output_root, bound_seed_evidence)
    return (
        finalization_environment,
        bound_seed_evidence,
        continuation_provenance,
    )


def assert_continuation_provenance(
    output_root: Path,
    execution_contract_sha256: str,
    finalization_environment: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    observed = validated_continuation_provenance(
        output_root,
        execution_contract_sha256,
        finalization_environment,
    )
    if observed != expected:
        raise FinalizationError("Continuation provenance changed during finalization")


def verify_existing_locked(
    output_root: Path,
    execution_contract: dict[str, Any],
    execution_contract_sha256: str,
    bound_sources: dict[str, str],
    context: dict[str, Any],
    active_lock: Any,
) -> None:
    active_lock.assert_owned()
    (
        finalization_environment,
        bound_seed_evidence,
        continuation_provenance,
    ) = validated_locked_inputs(
        output_root,
        execution_contract,
        execution_contract_sha256,
        bound_sources,
        context,
    )
    verify_finalized_root(
        output_root,
        execution_contract_sha256,
        context,
        bound_sources,
        bound_seed_evidence,
        finalization_environment,
        continuation_provenance,
    )
    active_lock.assert_owned()


def finalize_locked(
    output_root: Path,
    execution_contract: dict[str, Any],
    execution_contract_sha256: str,
    bound_sources: dict[str, str],
    context: dict[str, Any],
    resume: bool,
    active_lock: Any,
) -> None:
    active_lock.assert_owned()
    (
        finalization_environment,
        bound_seed_evidence,
        continuation_provenance,
    ) = validated_locked_inputs(
        output_root,
        execution_contract,
        execution_contract_sha256,
        bound_sources,
        context,
    )
    manifest_path = output_root / "artifact_manifest.json"
    if manifest_path.exists():
        raise FinalizationError(
            "A root manifest already exists; use --verify-existing instead"
        )
    aggregate_path = output_root / "aggregate_results.json"
    contract_path = output_root / FINALIZATION_CONTRACT_NAME
    if not resume and (aggregate_path.exists() or contract_path.exists()):
        raise FinalizationError(
            "Partial finalization output exists; inspect it and use --resume"
        )

    atomic_copy(RUNNER_PATH, output_root / RUNNER_SNAPSHOT_NAME)
    atomic_copy(COMMON_PATH, output_root / COMMON_SNAPSHOT_NAME)
    atomic_copy(SCRIPT_PATH, output_root / FINALIZER_SNAPSHOT_NAME)
    assert_finalization_identity(
        output_root,
        execution_contract_sha256,
        bound_sources,
    )
    for name, expected_hash in [
        (RUNNER_SNAPSHOT_NAME, bound_sources["runner"]),
        (COMMON_SNAPSHOT_NAME, bound_sources["common"]),
        (FINALIZER_SNAPSHOT_NAME, bound_sources["finalizer"]),
    ]:
        if sha256_file(output_root / name) != expected_hash:
            raise FinalizationError(f"Finalization source snapshot differs: {name}")
    assert_seed_identity(output_root, bound_seed_evidence)
    assert_finalization_environment(execution_contract, finalization_environment)
    assert_continuation_provenance(
        output_root,
        execution_contract_sha256,
        finalization_environment,
        continuation_provenance,
    )
    active_lock.assert_owned()

    aggregate_with_corrections(
        output_root,
        list(runner.PUBLICATION_SEEDS),
        execution_contract_sha256,
        context,
    )
    fsync_existing_file(aggregate_path)
    assert_seed_identity(output_root, bound_seed_evidence)
    assert_finalization_identity(
        output_root,
        execution_contract_sha256,
        bound_sources,
    )
    assert_finalization_environment(execution_contract, finalization_environment)
    assert_continuation_provenance(
        output_root,
        execution_contract_sha256,
        finalization_environment,
        continuation_provenance,
    )
    active_lock.assert_owned()

    finalization = build_semantic_contract(
        output_root,
        execution_contract_sha256,
        bound_sources,
        bound_seed_evidence,
        finalization_environment,
        continuation_provenance,
    )
    finalization["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    durable_atomic_write_json(contract_path, finalization)
    assert_seed_identity(output_root, bound_seed_evidence)
    assert_finalization_identity(
        output_root,
        execution_contract_sha256,
        bound_sources,
    )
    assert_finalization_environment(execution_contract, finalization_environment)
    assert_continuation_provenance(
        output_root,
        execution_contract_sha256,
        finalization_environment,
        continuation_provenance,
    )
    active_lock.assert_owned()
    durable_atomic_write_json(
        manifest_path,
        runner.artifact_manifest(output_root, runner.PROTOCOL_ID, "complete"),
    )
    assert_seed_identity(output_root, bound_seed_evidence)
    assert_finalization_identity(
        output_root,
        execution_contract_sha256,
        bound_sources,
    )
    assert_finalization_environment(execution_contract, finalization_environment)
    assert_continuation_provenance(
        output_root,
        execution_contract_sha256,
        finalization_environment,
        continuation_provenance,
    )
    active_lock.assert_owned()
    verify_finalized_root(
        output_root,
        execution_contract_sha256,
        context,
        bound_sources,
        bound_seed_evidence,
        finalization_environment,
        continuation_provenance,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=runner.DEFAULT_OUTPUT)
    parser.add_argument("--base-root", type=Path, default=runner.DEFAULT_BASE)
    parser.add_argument("--dataset-csv", type=Path, default=runner.DEFAULT_DATASET)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--confirm-finalization", action="store_true")
    mode.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.resume and not args.confirm_finalization:
        raise FinalizationError("--resume requires --confirm-finalization")
    output_root = args.output_root.resolve()
    if not output_root.is_dir():
        raise FinalizationError(f"Output root is absent: {output_root}")
    execution_contract, execution_contract_sha256 = validate_execution_identity(
        output_root
    )
    bound_sources = source_identity()
    assert_finalization_identity(
        output_root,
        execution_contract_sha256,
        bound_sources,
    )
    context = runner.load_context(args.dataset_csv.resolve(), args.base_root.resolve())
    context["verified_base_seeds"] = {
        seed: runner.verify_base_seed(args.base_root.resolve(), seed, context)
        for seed in runner.PUBLICATION_SEEDS
    }
    completed = complete_seed_ids(output_root)
    verify_seed_set(output_root, completed, execution_contract_sha256, context)
    print(
        json.dumps(
            {
                "protocol_id": FINALIZER_PROTOCOL_ID,
                "completed_seed_count": len(completed),
                "completed_seeds": completed,
                "training_performed": False,
            },
            indent=2,
        ),
        flush=True,
    )
    if args.verify_existing:
        if completed != runner.PUBLICATION_SEEDS:
            raise FinalizationError("The publication seed set is incomplete")
        with lifecycle_lock(output_root) as active_lock:
            verify_existing_locked(
                output_root,
                execution_contract,
                execution_contract_sha256,
                bound_sources,
                context,
                active_lock,
            )
        print(output_root)
        return 0
    if not args.confirm_finalization:
        print("Read-only preflight complete; no output was written.")
        return 0
    if completed != runner.PUBLICATION_SEEDS:
        raise FinalizationError(
            f"Refusing finalization before all publication seeds complete: {completed}"
        )
    with lifecycle_lock(output_root) as active_lock:
        finalize_locked(
            output_root,
            execution_contract,
            execution_contract_sha256,
            bound_sources,
            context,
            args.resume,
            active_lock,
        )
    print(output_root)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FinalizationError as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
