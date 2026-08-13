"""Run a compact, source-sealed fixed-point audit of all 40 final WSN-DS models.

The audit is additive and resumable. It never trains, changes preserved run
artifacts, retains row-level predictions, or creates deployable model bundles.
Quantization selection is imported directly from the final seed-42 exporter and
uses training and validation evidence only. Audit generation gives each selected
model one frozen Python test evaluation. Models passing every numeric deployment
gate then receive a temporary verifier-native C replay. ``--verify-only`` is an
explicit, separate deterministic recomputation of a completed audit.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deployment.firmware_export.wsnds_final_hil import (  # noqa: E402
    export_final_seed42 as final_export,
)
from deployment.firmware_export.wsnds_rfkd_hil.export_fgds_seed42_deployment import (  # noqa: E402
    file_inventory,
    rescale_truncating_toward_zero,
    resolve_manifest_member,
    verify_manifest,
)
from deployment.firmware_export.wsnds_rfkd_hil.export_wsnds_student_a_rfkd_int8 import (  # noqa: E402
    build_integer_preprocessing_metadata,
    build_preprocessing_metadata,
    extract_linear_layers,
    forward_numpy,
    quantize_raw_features_q,
    quantize_standardized_q15,
    write_header,
    write_integer_preprocessing_header,
)
from experiments.wsnds.leakage_free_rerun.tier15_common import (  # noqa: E402
    CLASS_NAMES,
    RF_CONFIG,
    STUDENT_SPECS,
    TRAIN_CONFIG,
    classification_metrics,
    sha256_arrays,
    sha256_file,
    split_hashes,
)


AUDIT_PROTOCOL_ID = "wsnds_all_seed_software_fixed_point_audit_v1"
SOURCE_PROTOCOL_ID = final_export.PROTOCOL_ID
EXPECTED_SEEDS = tuple(final_export.EXPECTED_SEEDS)
EXPECTED_SPLIT_SIZES = dict(final_export.EXPECTED_SPLIT_SIZES)
EXPECTED_RELATIVE_ROOT = final_export.EXPECTED_RELATIVE_ROOT
MINIMUM_FIXED_FP32_AGREEMENT = final_export.MINIMUM_FIXED_FP32_AGREEMENT
MAXIMUM_ABSOLUTE_MACRO_F1_DROP = final_export.MAXIMUM_ABSOLUTE_MACRO_F1_DROP
select_quantization_policy = final_export.select_quantization_policy

STUDENTS = ("A", "B")
ROUTES = ("scratch", "rf_kd")
MODEL_COUNT = len(EXPECTED_SEEDS) * len(STUDENTS) * len(ROUTES)
CONTRACT_NAME = "all_seed_audit_contract.json"
PROGRESS_NAME = "all_seed_audit_progress.json"
REPORT_NAME = "all_seed_fixed_point_report.json"
MANIFEST_NAME = "all_seed_fixed_point_manifest.json"
MODEL_DIR_NAME = "models"

C_PHASES = (
    "compiler_discovery",
    "compilation",
    "execution",
    "verification",
)
C_SUCCESS_OUTCOMES = {
    "compiler_discovery": "compiler_version_obtained",
    "compilation": "compiler_process_completed",
    "execution": "native_process_completed",
    "verification": "exact_equivalence_passed",
}
NUMERIC_PRIMITIVE_GATE_KEYS = (
    "accumulator_bounds_passed",
    "preprocess_bounds_passed",
    "training_calibration_saturation_passed",
    "validation_calibration_saturation_passed",
    "test_saturation_passed",
    "standardized_input_bounds_passed",
    "fixed_vs_fp32_agreement_passed",
    "macro_f1_drop_passed",
)

_ADDITIONAL_SOURCE_SPECS = [
    (
        "deployment/firmware_export/wsnds_final_hil/audit_all_seeds.py",
        "source_snapshot/python/audit_all_seeds.py",
    ),
    (
        "deployment/firmware_export/wsnds_final_hil/all_seed_stream_self_test.c",
        "source_snapshot/c/all_seed_stream_self_test.c",
    ),
]
SOURCE_SNAPSHOT_SPECS = tuple(
    _ADDITIONAL_SOURCE_SPECS
    + [item for item in final_export.SOURCE_SNAPSHOT_SPECS]
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirmation-root",
        type=Path,
        default=REPO_ROOT / EXPECTED_RELATIVE_ROOT,
    )
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=REPO_ROOT / "data" / "wsnds" / "WSN-DS.csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cc", default="gcc")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Deeply recompute and verify an already completed audit.",
    )
    return parser.parse_args(argv)


def model_id(seed: int, student: str, route: str) -> str:
    if seed not in EXPECTED_SEEDS or student not in STUDENTS or route not in ROUTES:
        raise ValueError("Model identity is outside the frozen 40-model matrix")
    return f"seed_{seed}_student_{student}_{route}"


def expected_model_matrix() -> list[dict[str, Any]]:
    return [
        {
            "model_id": model_id(seed, student, route),
            "seed": seed,
            "student": student,
            "route": route,
        }
        for seed in EXPECTED_SEEDS
        for student in STUDENTS
        for route in ROUTES
    ]


def _read_json(path: Path) -> dict[str, Any]:
    return final_export.read_json(path)


def _canonical_hash(payload: dict[str, Any]) -> str:
    return final_export.canonical_json_sha256(payload)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{time.time_ns()}")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _with_payload_hash(payload: dict[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    result[field] = _canonical_hash(result)
    return result


def _verify_payload_hash(payload: dict[str, Any], field: str, label: str) -> None:
    unhashed = dict(payload)
    observed = unhashed.pop(field, None)
    if observed != _canonical_hash(unhashed):
        raise RuntimeError(f"{label} canonical payload hash is invalid")


def _nested_delta(fixed: Any, fp32: Any) -> Any:
    if isinstance(fp32, dict):
        if not isinstance(fixed, dict) or set(fixed) != set(fp32):
            raise RuntimeError("Metric payloads have different fields")
        return {key: _nested_delta(fixed[key], fp32[key]) for key in fp32}
    if isinstance(fp32, list):
        if not isinstance(fixed, list) or len(fixed) != len(fp32):
            raise RuntimeError("Metric payloads have different shapes")
        return [_nested_delta(a, b) for a, b in zip(fixed, fp32)]
    if isinstance(fp32, (int, float)) and not isinstance(fp32, bool):
        return float(fixed) - float(fp32)
    raise RuntimeError("Metric payload contains a nonnumeric leaf")


def _source_snapshots(output_root: Path) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    seen_origins: set[str] = set()
    seen_targets: set[str] = set()
    for origin_relative, snapshot_relative in SOURCE_SNAPSHOT_SPECS:
        if origin_relative in seen_origins or snapshot_relative in seen_targets:
            raise RuntimeError("Source snapshot specification contains a duplicate")
        seen_origins.add(origin_relative)
        seen_targets.add(snapshot_relative)
        origin = REPO_ROOT / origin_relative
        snapshot = output_root / snapshot_relative
        if not origin.is_file():
            raise FileNotFoundError(origin)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, snapshot)
        digest = sha256_file(origin)
        if sha256_file(snapshot) != digest:
            raise RuntimeError(f"Source snapshot copy differs: {origin_relative}")
        snapshots.append({
            "origin_relative_path": origin_relative,
            "snapshot_path": snapshot_relative,
            "size_bytes": snapshot.stat().st_size,
            "sha256": digest,
        })
    return snapshots


def _verify_snapshots(output_root: Path, snapshots: Any) -> None:
    if not isinstance(snapshots, list) or len(snapshots) != len(SOURCE_SNAPSHOT_SPECS):
        raise RuntimeError("Audit source snapshot ledger is incomplete")
    observed = {
        (item.get("origin_relative_path"), item.get("snapshot_path"))
        for item in snapshots
    }
    if observed != set(SOURCE_SNAPSHOT_SPECS):
        raise RuntimeError("Audit source snapshot mapping is invalid")
    for item in snapshots:
        final_export._assert_hash(item.get("sha256"), "audit source snapshot")
        origin = REPO_ROOT / item["origin_relative_path"]
        snapshot = output_root / item["snapshot_path"]
        if (
            not origin.is_file()
            or origin.stat().st_size != item.get("size_bytes")
            or sha256_file(origin) != item.get("sha256")
        ):
            raise RuntimeError(
                f"Current audit source differs from its sealed origin: {origin}"
            )
        if (
            not snapshot.is_file()
            or snapshot.stat().st_size != item.get("size_bytes")
            or sha256_file(snapshot) != item.get("sha256")
        ):
            raise RuntimeError(f"Audit source snapshot differs: {snapshot}")


def load_shared_lineage(confirmation_root: Path, dataset_csv: Path) -> dict[str, Any]:
    root = confirmation_root.resolve()
    expected_root = (REPO_ROOT / EXPECTED_RELATIVE_ROOT).resolve()
    if root != expected_root:
        raise RuntimeError(f"Confirmation root must be exactly {expected_root}")
    dataset_path = dataset_csv.resolve()
    root_manifest = verify_manifest(root)
    execution = _read_json(root / "execution_contract.json")
    preprocessing = _read_json(root / "preprocessing_contract.json")
    if (
        root_manifest.get("protocol_id") != SOURCE_PROTOCOL_ID
        or root_manifest.get("status") != "complete"
        or execution.get("protocol_id") != SOURCE_PROTOCOL_ID
        or preprocessing.get("protocol_id") != SOURCE_PROTOCOL_ID
    ):
        raise RuntimeError("Final ten-seed root does not satisfy the source protocol")
    if execution.get("seeds") != list(EXPECTED_SEEDS):
        raise RuntimeError("Execution contract does not contain the exact ten seeds")
    if execution.get("routes") != list(ROUTES):
        raise RuntimeError("Execution contract route matrix differs")
    if execution.get("students") != {
        name: list(dimensions) for name, dimensions in STUDENT_SPECS.items()
    }:
        raise RuntimeError("Execution contract student matrix differs")
    reconstructed = final_export._verify_preprocessing(
        root, dataset_path, preprocessing
    )
    test_indices_hash = sha256_arrays(
        np.asarray(reconstructed["split"]["test_indices"], dtype=np.int64)
    )
    return {
        **reconstructed,
        "confirmation_root": root,
        "dataset_csv": dataset_path,
        "root_manifest": root_manifest,
        "execution": execution,
        "preprocessing": preprocessing,
        "test_source_indices_sha256": test_indices_hash,
        "root_artifact_manifest_sha256": sha256_file(root / "artifact_manifest.json"),
        "execution_contract_sha256": sha256_file(root / "execution_contract.json"),
        "preprocessing_contract_sha256": sha256_file(
            root / "preprocessing_contract.json"
        ),
    }


def _verify_rf_provenance(
    seed: int,
    seed_root: Path,
    seed_manifest: dict[str, Any],
    completion: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    provenance = completion.get("teacher_soft_target_provenance")
    calibration = provenance.get("calibration_audit") if isinstance(provenance, dict) else None
    if (
        not isinstance(provenance, dict)
        or provenance.get("source_type") != "fresh_calibrated_rf_soft_targets"
        or provenance.get("rf_seed") != seed
        or provenance.get("rf_config") != RF_CONFIG
        or not isinstance(calibration, dict)
        or calibration.get("strategy") != "stratified_group_kfold"
        or calibration.get("folds") != 3
        or calibration.get("group_overlap_per_fold") != [0, 0, 0]
    ):
        raise RuntimeError("RF-KD teacher provenance violates the ten-seed contract")
    item = next(
        (
            candidate
            for candidate in seed_manifest.get("files", [])
            if candidate.get("path") == "rf_train_probabilities.npy"
        ),
        None,
    )
    if item is None:
        raise RuntimeError("RF-KD teacher probability artifact is missing")
    path = seed_root / "rf_train_probabilities.npy"
    probabilities = np.load(path, allow_pickle=False)
    if (
        probabilities.shape != (EXPECTED_SPLIT_SIZES["train"], len(CLASS_NAMES))
        or not np.isfinite(probabilities).all()
        or sha256_file(path) != item.get("sha256")
        or sha256_arrays(probabilities)
        != provenance.get("train_probability_content_sha256")
    ):
        raise RuntimeError("RF-KD teacher probability artifact is invalid")
    return provenance, path


def load_model_without_test(
    shared: dict[str, Any], seed: int, student: str, route: str
) -> dict[str, Any]:
    identifier = model_id(seed, student, route)
    root = shared["confirmation_root"]
    seed_root = root / f"seed_{seed}"
    seed_manifest = verify_manifest(seed_root)
    completion = _read_json(seed_root / "seed_completion.json")
    if (
        seed_manifest.get("protocol_id") != SOURCE_PROTOCOL_ID
        or completion.get("protocol_id") != SOURCE_PROTOCOL_ID
        or completion.get("status") != "complete"
        or completion.get("seed") != seed
    ):
        raise RuntimeError(f"Incomplete or mismatched seed contract: {seed}")
    for key in ["dataset_sha256", "split_indices_sha256", "scaler_sha256"]:
        if (
            completion.get(key) != shared["execution"].get(key)
            or completion.get(key) != shared["preprocessing"].get(key)
        ):
            raise RuntimeError(f"Seed {seed} lineage mismatch for {key}")
    final_export.verify_execution_contract(root, shared["execution"], completion)

    student_name = f"student_{student}"
    result_key = f"{student_name}_{route}"
    result = completion.get("student_results", {}).get(result_key)
    if not isinstance(result, dict) or result.get("route") != route:
        raise RuntimeError(f"Missing model result: {identifier}")
    model_path = resolve_manifest_member(
        seed_root,
        seed_manifest,
        result.get("plain_state_dict"),
        result.get("plain_state_dict_sha256"),
    )
    rich_path = resolve_manifest_member(
        seed_root,
        seed_manifest,
        result.get("rich_artifact"),
        result.get("rich_artifact_sha256"),
    )
    predictions_path = resolve_manifest_member(
        seed_root,
        seed_manifest,
        result.get("test_predictions"),
        result.get("test_predictions_sha256"),
    )
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    rich = torch.load(rich_path, map_location="cpu", weights_only=True)
    rich_state = rich.get("state_dict")
    if not isinstance(state, dict) or not isinstance(rich_state, dict):
        raise RuntimeError(f"Checkpoint state is invalid: {identifier}")
    if set(state) != set(rich_state):
        raise RuntimeError(f"Plain/rich checkpoint keys differ: {identifier}")
    for key in sorted(state):
        if (
            not torch.is_tensor(state[key])
            or not torch.isfinite(state[key]).all()
            or not torch.equal(state[key], rich_state[key])
        ):
            raise RuntimeError(f"Plain/rich checkpoint tensor differs: {identifier}:{key}")
    trained_state_sha256 = sha256_arrays(
        *[state[key].detach().cpu().numpy() for key in sorted(state)]
    )
    if (
        trained_state_sha256 != result.get("trained_state_sha256")
        or trained_state_sha256 != rich.get("trained_state_sha256")
    ):
        raise RuntimeError(f"Trained-state hash differs: {identifier}")

    teacher_provenance: dict[str, Any] | None = None
    teacher_path: Path | None = None
    kd_hyperparameters: dict[str, float] | None = None
    if route == "rf_kd":
        teacher_provenance, teacher_path = _verify_rf_provenance(
            seed, seed_root, seed_manifest, completion
        )
        kd_hyperparameters = {"T": 4.0, "alpha": 0.7}

    expected_rich = {
        "protocol_id": SOURCE_PROTOCOL_ID,
        "seed": seed,
        "student": student_name,
        "route": route,
        "hidden_dims": list(STUDENT_SPECS[student_name]),
        "input_dim": 17,
        "num_classes": len(CLASS_NAMES),
        "feature_names": shared["dataset"]["feature_names"],
        "class_names": CLASS_NAMES,
        "dataset_sha256": shared["dataset"]["dataset_sha256"],
        "split_hashes": split_hashes(shared["split"]),
        "scaler_sha256": shared["preprocessing"]["scaler_sha256"],
        "kd_hyperparameters": kd_hyperparameters,
        "training_config": TRAIN_CONFIG,
        "feature_overlap_audit": shared["split"]["group_audit"],
        "teacher_soft_target_provenance": teacher_provenance,
        "trained_state_sha256": trained_state_sha256,
        "initial_state_sha256": result.get("initial_state_sha256"),
    }
    for key, expected in expected_rich.items():
        if rich.get(key) != expected:
            raise RuntimeError(f"Rich checkpoint mismatch for {identifier}:{key}")
    layers = extract_linear_layers(state)
    hidden = [int(layers[0][1].shape[0]), int(layers[1][1].shape[0])]
    if len(layers) != 3 or hidden != list(STUDENT_SPECS[student_name]):
        raise RuntimeError(f"Checkpoint architecture differs: {identifier}")
    return {
        "model_id": identifier,
        "seed": seed,
        "student": student,
        "student_name": student_name,
        "route": route,
        "seed_root": seed_root,
        "seed_manifest": seed_manifest,
        "completion": completion,
        "result": result,
        "model_path": model_path,
        "rich_path": rich_path,
        "predictions_path": predictions_path,
        "layers": layers,
        "trained_state_sha256": trained_state_sha256,
        "kd_hyperparameters": kd_hyperparameters,
        "teacher_provenance": teacher_provenance,
        "teacher_path": teacher_path,
    }


def _model_identity(shared: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    identity = {
        "audit_protocol_id": AUDIT_PROTOCOL_ID,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "model_id": model["model_id"],
        "seed": model["seed"],
        "student": model["student"],
        "route": model["route"],
        "dataset_sha256": shared["dataset"]["dataset_sha256"],
        "split_indices_sha256": shared["preprocessing"]["split_indices_sha256"],
        "test_source_indices_sha256": shared["test_source_indices_sha256"],
        "scaler_sha256": shared["preprocessing"]["scaler_sha256"],
        "seed_manifest_sha256": sha256_file(
            model["seed_root"] / "artifact_manifest.json"
        ),
        "seed_completion_sha256": sha256_file(
            model["seed_root"] / "seed_completion.json"
        ),
        "checkpoint_file_sha256": sha256_file(model["model_path"]),
        "rich_artifact_sha256": sha256_file(model["rich_path"]),
        "prediction_artifact_sha256": sha256_file(model["predictions_path"]),
        "trained_state_sha256": model["trained_state_sha256"],
        "route_provenance": {
            "kd_hyperparameters": model["kd_hyperparameters"],
            "teacher_soft_target_provenance": model["teacher_provenance"],
            "teacher_probability_file_sha256": (
                sha256_file(model["teacher_path"])
                if model["teacher_path"] is not None
                else None
            ),
        },
    }
    identity["model_identity_sha256"] = _canonical_hash(identity)
    return identity


def _metadata(shared: dict[str, Any]) -> dict[str, Any]:
    metadata = build_preprocessing_metadata(
        target_col=shared["dataset"]["target_column"],
        feature_names=shared["dataset"]["feature_names"],
        class_names=shared["dataset"]["class_names"],
        scaler_mean=shared["scaler"].mean_,
        scaler_scale=shared["scaler"].scale_,
        split_sizes={"train": 262197, "val": 56163, "test": 56301},
    )
    metadata.update({
        "protocol_id": SOURCE_PROTOCOL_ID,
        "dataset_sha256": shared["dataset"]["dataset_sha256"],
        "split_indices_sha256": shared["preprocessing"]["split_indices_sha256"],
        "scaler_sha256": shared["preprocessing"]["scaler_sha256"],
        "scaler_fit_partition": "train only",
    })
    return metadata


def _run(command: list[str], cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(
        command, cwd=cwd, text=True, capture_output=True, check=False
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _new_c_phase_ledger() -> dict[str, dict[str, Any]]:
    return {
        phase: {"attempted": False, "status": "not_attempted", "outcome": None}
        for phase in C_PHASES
    }


def _c_failure_result(
    *,
    rows: int,
    phases: dict[str, dict[str, Any]],
    failed_phase: str,
    failed_outcome: str,
    exception_type: str,
    message: str,
    host_evidence: dict[str, Any],
) -> dict[str, Any]:
    if failed_phase not in C_PHASES:
        raise RuntimeError(f"Unknown C-equivalence failure phase: {failed_phase}")
    return {
        "status": "failed",
        "rows": int(rows),
        "failed_phase": failed_phase,
        "failed_outcome": failed_outcome,
        "phases": phases,
        "failure": {
            "exception_type": exception_type,
            "message": message,
        },
        "generation_host_evidence": host_evidence,
    }


def _c_blocked_result(rows: int) -> dict[str, Any]:
    return {
        "status": "blocked",
        "rows": int(rows),
        "failed_phase": None,
        "failed_outcome": None,
        "outcome": "blocked_by_numeric_gate_failure",
        "phases": _new_c_phase_ledger(),
    }


def _run_c_phase(
    command: list[str],
    cwd: Path,
    phases: dict[str, dict[str, Any]],
    phase: str,
    success_outcome: str,
) -> tuple[dict[str, Any] | None, Exception | None]:
    phases[phase] = {"attempted": True, "status": "running", "outcome": None}
    try:
        outcome = _run(command, cwd)
    except (OSError, subprocess.SubprocessError) as exc:
        phases[phase] = {
            "attempted": True,
            "status": "failed",
            "outcome": "process_start_failed",
        }
        return None, exc
    phases[phase] = {
        "attempted": True,
        "status": "completed",
        "outcome": success_outcome,
    }
    return outcome, None


def _c_scope(c_equivalence: dict[str, Any]) -> dict[str, Any]:
    phases = c_equivalence.get("phases", {})
    return {
        "selection_uses_train_and_validation_only": True,
        "python_frozen_test_evaluations_this_model_invocation": 1,
        "compiler_discovery_attempts_this_model_invocation": int(
            phases.get("compiler_discovery", {}).get("attempted") is True
        ),
        "compiler_discovery_successes_this_model_invocation": int(
            phases.get("compiler_discovery", {}).get("status") == "completed"
        ),
        "compilation_attempts_this_model_invocation": int(
            phases.get("compilation", {}).get("attempted") is True
        ),
        "successful_compilations_this_model_invocation": int(
            phases.get("compilation", {}).get("status") == "completed"
        ),
        "native_process_start_attempts_this_model_invocation": int(
            phases.get("execution", {}).get("attempted") is True
        ),
        "native_process_executions_this_model_invocation": int(
            phases.get("execution", {}).get("status") == "completed"
        ),
        "exact_equivalence_verification_attempts_this_model_invocation": int(
            phases.get("verification", {}).get("attempted") is True
        ),
        "exact_equivalence_verifications_passed_this_model_invocation": int(
            phases.get("verification", {}).get("status") == "completed"
        ),
        "claims_first_ever_test_observation": False,
        "historical_test_observations_preceded_this_audit": True,
    }


def _accumulator_gate(
    quantized_layers: list[dict[str, Any]],
) -> tuple[bool, list[dict[str, Any]]]:
    ledger: list[dict[str, Any]] = []
    int32_max = int(np.iinfo(np.int32).max)
    for index, layer in enumerate(quantized_layers):
        weights = np.asarray(layer["weight"], dtype=np.int64)
        biases = np.asarray(layer["bias"], dtype=np.int64)
        if weights.ndim != 2 or biases.shape != (weights.shape[0],):
            raise RuntimeError(f"Layer {index} has invalid quantized parameter shapes")
        maximum = max(
            (
                abs(int(bias))
                + 32768 * sum(abs(int(value)) for value in weights[row])
                for row, bias in enumerate(biases)
            ),
            default=0,
        )
        shift = int(layer["output_shift"])
        passed = shift >= 0 and maximum <= int32_max
        ledger.append({
            "layer": index,
            "pre_rescale_absolute_bound": maximum,
            "output_shift": shift,
            "post_left_shift_absolute_bound": maximum,
            "int32_max": int32_max,
            "passed": passed,
        })
    return all(item["passed"] for item in ledger), ledger


def _preprocess_bounds_gate(
    integer_metadata: dict[str, Any],
) -> tuple[bool, list[dict[str, Any]]]:
    means = np.asarray(integer_metadata["scaler_mean_q"], dtype=np.int64)
    inverse_scales = np.asarray(
        integer_metadata["scaler_inv_scale_q"], dtype=np.int64
    )
    if means.shape != (17,) or inverse_scales.shape != (17,):
        raise RuntimeError("Integer preprocessing constants must each contain 17 values")
    int32 = np.iinfo(np.int32)
    int64_max = int(np.iinfo(np.int64).max)
    ledger: list[dict[str, Any]] = []
    for index, (mean, inverse_scale) in enumerate(zip(means, inverse_scales)):
        maximum_centered = max(
            abs(int(int32.min) - int(mean)),
            abs(int(int32.max) - int(mean)),
        )
        maximum_product = maximum_centered * abs(int(inverse_scale))
        ledger.append({
            "feature": index,
            "maximum_centered_absolute": maximum_centered,
            "inverse_scale_absolute": abs(int(inverse_scale)),
            "maximum_product_absolute": maximum_product,
            "int64_max": int64_max,
            "passed": maximum_product <= int64_max,
        })
    return all(item["passed"] for item in ledger), ledger


def _saturation_chunk(
    layers: list[tuple[str, Any, Any]],
    quantized_layers: list[dict[str, Any]],
    raw_inputs_q: np.ndarray,
    integer_metadata: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    parameter_layers: list[dict[str, Any]] = []
    weight_saturation = 0
    bias_saturation = 0
    for index, ((prefix, weight, bias), quantized) in enumerate(
        zip(layers, quantized_layers)
    ):
        weight_array = np.asarray(weight, dtype=np.float64)
        bias_array = np.asarray(bias, dtype=np.float64)
        if not np.isfinite(weight_array).all() or not np.isfinite(bias_array).all():
            raise RuntimeError(f"Layer {index} contains non-finite FP32 parameters")
        weight_unbounded = np.rint(
            weight_array * float(1 << int(quantized["weight_frac"]))
        )
        bias_unbounded = np.rint(
            bias_array * float(1 << int(quantized["accum_frac"]))
        )
        weight_count = int(
            np.count_nonzero((weight_unbounded < -128) | (weight_unbounded > 127))
        )
        int32 = np.iinfo(np.int32)
        bias_count = int(
            np.count_nonzero(
                (bias_unbounded < int32.min) | (bias_unbounded > int32.max)
            )
        )
        weight_saturation += weight_count
        bias_saturation += bias_count
        parameter_layers.append({
            "layer": index,
            "source_prefix": prefix,
            "weight_saturation_count": weight_count,
            "bias_saturation_count": bias_count,
        })

    raw = np.asarray(raw_inputs_q, dtype=np.int64)
    centered = raw - np.asarray(
        integer_metadata["scaler_mean_q"], dtype=np.int64
    )
    scaled = centered * np.asarray(
        integer_metadata["scaler_inv_scale_q"], dtype=np.int64
    )
    preprocess_unclipped = rescale_truncating_toward_zero(
        scaled, int(integer_metadata["right_shift"])
    )
    preprocess_saturation = int(
        np.count_nonzero(
            (preprocess_unclipped < -32768) | (preprocess_unclipped > 32767)
        )
    )
    preprocessed = np.clip(preprocess_unclipped, -32768, 32767).astype(np.int16)
    activations = preprocessed.astype(np.int64)
    activation_layers: list[dict[str, Any]] = []
    final_logits: np.ndarray | None = None
    for index, quantized in enumerate(quantized_layers):
        accumulator = (
            activations @ np.asarray(quantized["weight"], dtype=np.int64).T
            + np.asarray(quantized["bias"], dtype=np.int64)
        )
        output = rescale_truncating_toward_zero(
            accumulator, int(quantized["output_shift"])
        )
        if index < len(quantized_layers) - 1:
            output = np.maximum(output, 0)
        count = int(np.count_nonzero((output < -32768) | (output > 32767)))
        activation_layers.append({
            "layer": index,
            "activation_saturation_count": count,
            "minimum_before_clip": int(output.min()),
            "maximum_before_clip": int(output.max()),
        })
        activations = np.clip(output, -32768, 32767).astype(np.int16).astype(
            np.int64
        )
        final_logits = activations.astype(np.int16)
    if final_logits is None:
        raise RuntimeError("Saturation audit produced no model output")
    audit = {
        "parameter_layers": parameter_layers,
        "weight_saturation_count": weight_saturation,
        "bias_saturation_count": bias_saturation,
        "integer_preprocess_saturation_count": preprocess_saturation,
        "activation_layers": activation_layers,
        "activation_saturation_count": int(
            sum(item["activation_saturation_count"] for item in activation_layers)
        ),
    }
    predictions = final_logits.astype(np.int32).argmax(axis=1).astype(np.int64)
    return audit, preprocessed, final_logits, predictions


def _partition_saturation_gate(
    *,
    partition: str,
    layers: list[tuple[str, Any, Any]],
    quantized_layers: list[dict[str, Any]],
    raw_features: np.ndarray,
    integer_metadata: dict[str, Any],
    chunk_size: int = 8192,
) -> tuple[bool, dict[str, Any]]:
    raw_features = np.asarray(raw_features)
    raw_saturation_count = 0
    preprocess_saturation_count = 0
    weight_saturation_count = 0
    bias_saturation_count = 0
    activation_counts = np.zeros(len(quantized_layers), dtype=np.int64)
    minimums = np.full(
        len(quantized_layers), np.iinfo(np.int64).max, dtype=np.int64
    )
    maximums = np.full(
        len(quantized_layers), np.iinfo(np.int64).min, dtype=np.int64
    )
    parameter_layers: list[dict[str, Any]] | None = None
    raw_scale = float(1 << int(integer_metadata["raw_q_frac"]))
    int32 = np.iinfo(np.int32)
    for start in range(0, len(raw_features), chunk_size):
        chunk = raw_features[start : start + chunk_size]
        unbounded = np.rint(np.asarray(chunk, dtype=np.float64) * raw_scale)
        raw_saturation_count += int(
            np.count_nonzero((unbounded < int32.min) | (unbounded > int32.max))
        )
        raw_q = np.clip(unbounded, int32.min, int32.max).astype(np.int32)
        chunk_audit, _, _, _ = _saturation_chunk(
            layers, quantized_layers, raw_q, integer_metadata
        )
        if parameter_layers is None:
            parameter_layers = chunk_audit["parameter_layers"]
            weight_saturation_count = chunk_audit["weight_saturation_count"]
            bias_saturation_count = chunk_audit["bias_saturation_count"]
        preprocess_saturation_count += chunk_audit[
            "integer_preprocess_saturation_count"
        ]
        for index, item in enumerate(chunk_audit["activation_layers"]):
            activation_counts[index] += item["activation_saturation_count"]
            minimums[index] = min(minimums[index], item["minimum_before_clip"])
            maximums[index] = max(maximums[index], item["maximum_before_clip"])
    ledger = {
        "partition": partition,
        "rows_audited": int(len(raw_features)),
        "chunk_size": chunk_size,
        "raw_input_saturation_count": raw_saturation_count,
        "weight_saturation_count": weight_saturation_count,
        "bias_saturation_count": bias_saturation_count,
        "integer_preprocess_saturation_count": preprocess_saturation_count,
        "parameter_layers": parameter_layers or [],
        "activation_layers": [
            {
                "layer": index,
                "activation_saturation_count": int(activation_counts[index]),
                "minimum_before_clip": int(minimums[index]),
                "maximum_before_clip": int(maximums[index]),
            }
            for index in range(len(quantized_layers))
        ],
        "activation_saturation_count": int(activation_counts.sum()),
    }
    passed = all(
        ledger[key] == 0
        for key in [
            "raw_input_saturation_count",
            "weight_saturation_count",
            "bias_saturation_count",
            "integer_preprocess_saturation_count",
            "activation_saturation_count",
        ]
    )
    ledger["passed"] = passed
    return passed, ledger


def _write_equivalence_payload(
    path: Path,
    raw_inputs: np.ndarray,
    preprocessed: np.ndarray,
    logits: np.ndarray,
    predictions: np.ndarray,
) -> str:
    if sys.byteorder != "little":
        raise RuntimeError("The streamed C audit payload currently requires little-endian host order")
    rows = len(predictions)
    dtype = np.dtype([
        ("raw", "<i4", (17,)),
        ("preprocessed", "<i2", (17,)),
        ("logits", "<i2", (5,)),
        ("prediction", "u1"),
    ], align=False)
    payload = np.empty(rows, dtype=dtype)
    payload["raw"] = np.asarray(raw_inputs, dtype=np.int32)
    payload["preprocessed"] = np.asarray(preprocessed, dtype=np.int16)
    payload["logits"] = np.asarray(logits, dtype=np.int16)
    payload["prediction"] = np.asarray(predictions, dtype=np.uint8)
    payload.tofile(path)
    expected_bytes = rows * (17 * 4 + 17 * 2 + 5 * 2 + 1)
    if path.stat().st_size != expected_bytes:
        raise RuntimeError("Temporary C equivalence payload has an invalid size")
    return sha256_file(path)


def run_native_c_equivalence(
    *,
    source_root: Path,
    cc: str,
    model: dict[str, Any],
    quantized_layers: list[dict[str, Any]],
    integer_metadata: dict[str, Any],
    raw_inputs: np.ndarray,
    preprocessed: np.ndarray,
    logits: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    rows = int(len(predictions))
    phases = _new_c_phase_ledger()
    host_evidence: dict[str, Any] = {
        "compiler_requested": cc,
        "temporary_executable_retained": False,
    }
    sealed_c = source_root / "source_snapshot" / "c"
    harness = sealed_c / "all_seed_stream_self_test.c"
    sources = [
        sealed_c / "cukd_preprocess.c",
        sealed_c / "cukd_model.c",
        harness,
    ]
    for source in sources:
        if not source.is_file():
            raise RuntimeError(f"Sealed C audit source is missing: {source.name}")
    compiler_version, compiler_exception = _run_c_phase(
        [cc, "--version"],
        source_root,
        phases,
        "compiler_discovery",
        "compiler_version_obtained",
    )
    if compiler_exception is not None:
        return _c_failure_result(
            rows=rows,
            phases=phases,
            failed_phase="compiler_discovery",
            failed_outcome="process_start_failed",
            exception_type=type(compiler_exception).__name__,
            message=str(compiler_exception),
            host_evidence=host_evidence,
        )
    assert compiler_version is not None
    host_evidence["compiler_version"] = compiler_version
    if compiler_version["returncode"] != 0:
        phases["compiler_discovery"].update({
            "status": "failed",
            "outcome": "compiler_version_nonzero_exit",
        })
        return _c_failure_result(
            rows=rows,
            phases=phases,
            failed_phase="compiler_discovery",
            failed_outcome="compiler_version_nonzero_exit",
            exception_type="CompilerDiscoveryError",
            message=compiler_version["stderr"],
            host_evidence=host_evidence,
        )
    with tempfile.TemporaryDirectory(prefix="cukd_all_seed_c_") as temporary:
        work = Path(temporary)
        write_header(
            work / "model_weights.h",
            quantized_layers,
            f"{model['model_id']} compact audit",
        )
        write_integer_preprocessing_header(
            work / "preprocess_int_metadata.h", integer_metadata
        )
        payload_path = work / "equivalence.bin"
        payload_sha256 = _write_equivalence_payload(
            payload_path, raw_inputs, preprocessed, logits, predictions
        )
        executable = work / "all_seed_self_test"
        if os.name == "nt":
            executable = executable.with_suffix(".exe")
        command = [
            cc,
            "-std=c99",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-O2",
            f"-DCUKD_AUDIT_ROWS={len(predictions)}",
            "-I",
            str(work),
            "-I",
            str(sealed_c),
            *(str(source) for source in sources),
            "-o",
            str(executable),
        ]
        compile_result, compile_exception = _run_c_phase(
            command,
            source_root,
            phases,
            "compilation",
            "compiler_process_completed",
        )
        if compile_exception is not None:
            return _c_failure_result(
                rows=rows,
                phases=phases,
                failed_phase="compilation",
                failed_outcome="process_start_failed",
                exception_type=type(compile_exception).__name__,
                message=str(compile_exception),
                host_evidence=host_evidence,
            )
        assert compile_result is not None
        host_evidence["compile"] = compile_result
        if compile_result["returncode"] != 0:
            phases["compilation"].update({
                "status": "failed",
                "outcome": "compiler_nonzero_exit",
            })
            return _c_failure_result(
                rows=rows,
                phases=phases,
                failed_phase="compilation",
                failed_outcome="compiler_nonzero_exit",
                exception_type="CompilationError",
                message=compile_result["stderr"],
                host_evidence=host_evidence,
            )
        executable_sha256 = sha256_file(executable)
        host_evidence["temporary_executable_sha256"] = executable_sha256
        self_test, execution_exception = _run_c_phase(
            [str(executable), str(payload_path)],
            source_root,
            phases,
            "execution",
            "native_process_completed",
        )
        if execution_exception is not None:
            return _c_failure_result(
                rows=rows,
                phases=phases,
                failed_phase="execution",
                failed_outcome="process_start_failed",
                exception_type=type(execution_exception).__name__,
                message=str(execution_exception),
                host_evidence=host_evidence,
            )
        assert self_test is not None
        host_evidence["self_test"] = self_test
        phases["verification"] = {
            "attempted": True,
            "status": "running",
            "outcome": None,
        }
        if self_test["returncode"] != 0:
            phases["verification"].update({
                "status": "failed",
                "outcome": "self_test_nonzero_exit",
            })
            return _c_failure_result(
                rows=rows,
                phases=phases,
                failed_phase="verification",
                failed_outcome="self_test_nonzero_exit",
                exception_type="EquivalenceVerificationError",
                message=(
                    "C/Python exact equivalence failed with code "
                    f"{self_test['returncode']}"
                ),
                host_evidence=host_evidence,
            )
        phases["verification"].update({
            "status": "completed",
            "outcome": "exact_equivalence_passed",
        })
    return {
        "status": "passed",
        "rows": rows,
        "failed_phase": None,
        "failed_outcome": None,
        "phases": phases,
        "preprocessed_inputs_exact": True,
        "fixed_logits_exact": True,
        "fixed_predictions_exact": True,
        "payload_contract": "little-endian streamed int32[17], int16[17], int16[5], uint8",
        "payload_sha256": payload_sha256,
        "raw_inputs_content_sha256": sha256_arrays(
            np.asarray(raw_inputs, dtype=np.int32)
        ),
        "preprocessed_content_sha256": sha256_arrays(
            np.asarray(preprocessed, dtype=np.int16)
        ),
        "fixed_logits_content_sha256": sha256_arrays(
            np.asarray(logits, dtype=np.int16)
        ),
        "fixed_predictions_content_sha256": sha256_arrays(
            np.asarray(predictions, dtype=np.uint8)
        ),
        "sealed_source_sha256": {
            source.name: sha256_file(source) for source in sources
        },
        "generation_host_evidence": host_evidence,
    }


def _policy_summary(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy_id": policy["policy_id"],
        "policy_sha256": _canonical_hash(policy),
        "development_status": policy["development_status"],
        "applied_uniformly_to_all_student_route_exports": policy[
            "applied_uniformly_to_all_student_route_exports"
        ],
        "selection_uses_test_data": policy["selection_uses_test_data"],
        "historical_test_observations_preceded_policy_development": policy[
            "historical_test_observations_preceded_policy_development"
        ],
        "selection_status": policy["selection_status"],
        "baseline_fracs": policy["baseline_fracs"],
        "selected_fracs": policy["selected_fracs"],
        "candidate_count": policy["candidate_count"],
        "validation_gate_passing_candidate_count": policy[
            "validation_gate_passing_candidate_count"
        ],
        "frozen_thresholds": policy["frozen_thresholds"],
    }


def _status_from_gates(gates: dict[str, Any], c_equivalence: dict[str, Any]) -> str:
    numeric_passed = all(gates.get(key) is True for key in NUMERIC_PRIMITIVE_GATE_KEYS)
    if not numeric_passed:
        return "gate_failed"
    if c_equivalence.get("status") != "passed":
        return "c_equivalence_failed"
    return "passed"


def _report_status(status_counts: dict[str, int]) -> str:
    allowed = {"passed", "gate_failed", "c_equivalence_failed", "audit_error"}
    if (
        set(status_counts) - allowed
        or any(type(count) is not int or count <= 0 for count in status_counts.values())
        or sum(status_counts.values()) != MODEL_COUNT
    ):
        raise RuntimeError("Per-model status counts do not describe the exact 40-model matrix")
    return (
        "complete_all_gates_passed"
        if status_counts == {"passed": MODEL_COUNT}
        else "complete_with_retained_failures"
    )


def _numeric_gate_entry(
    passed: bool, *, evidence: Any, threshold: Any = None, observed: Any = None
) -> dict[str, Any]:
    entry = {"passed": bool(passed), "evidence": evidence}
    if threshold is not None:
        entry["threshold"] = threshold
    if observed is not None:
        entry["observed"] = observed
    return entry


def evaluate_model(
    shared: dict[str, Any],
    seed: int,
    student: str,
    route: str,
    source_root: Path,
    cc: str,
) -> dict[str, Any]:
    """Evaluate one model; test inference occurs only after policy selection."""
    identifier = model_id(seed, student, route)
    try:
        model = load_model_without_test(shared, seed, student, route)
        identity = _model_identity(shared, model)
        metadata = _metadata(shared)
        quantized, _, policy = select_quantization_policy(
            layers=model["layers"],
            preprocessing_metadata=metadata,
            x_train=shared["scaled"]["X_train"],
            x_train_raw=shared["split"]["X_train_raw"],
            x_validation=shared["scaled"]["X_validation"],
            x_validation_raw=shared["split"]["X_validation_raw"],
            y_validation=shared["split"]["y_validation"],
        )
        policy_summary = _policy_summary(policy)
        integer_metadata = build_integer_preprocessing_metadata(
            metadata, output_q_frac=int(quantized[0]["input_frac"])
        )
        accumulator_passed, accumulator_ledger = _accumulator_gate(quantized)
        preprocess_bounds_passed, preprocess_ledger = _preprocess_bounds_gate(
            integer_metadata
        )
        training_saturation_passed, training_saturation = _partition_saturation_gate(
            partition="training calibration partition",
            layers=model["layers"],
            quantized_layers=quantized,
            raw_features=shared["split"]["X_train_raw"],
            integer_metadata=integer_metadata,
        )
        validation_saturation_passed, validation_saturation = (
            _partition_saturation_gate(
                partition="validation partition",
                layers=model["layers"],
                quantized_layers=quantized,
                raw_features=shared["split"]["X_validation_raw"],
                integer_metadata=integer_metadata,
            )
        )

        # Frozen test boundary: one FP32 forward and one fixed-point audit.
        fp32_logits = forward_numpy(model["layers"], shared["scaled"]["X_test"])
        if not np.isfinite(fp32_logits).all():
            raise RuntimeError("Frozen FP32 test logits contain non-finite values")
        fp32_predictions = fp32_logits.argmax(axis=1).astype(np.int64)
        raw_test_q = quantize_raw_features_q(
            shared["split"]["X_test_raw"], integer_metadata["raw_q_frac"]
        )
        test_saturation, preprocessed, fixed_logits, fixed_predictions = _saturation_chunk(
            model["layers"], quantized, raw_test_q, integer_metadata
        )
        test_saturation = dict(test_saturation)
        test_saturation["partition"] = "test partition"
        test_saturation["rows_audited"] = int(len(raw_test_q))
        test_saturation["raw_input_saturation_count"] = final_export._raw_int32_saturation_count(
            shared["split"]["X_test_raw"], integer_metadata["raw_q_frac"]
        )
        test_saturation_passed = all(
            test_saturation[key] == 0
            for key in [
                "raw_input_saturation_count",
                "weight_saturation_count",
                "bias_saturation_count",
                "integer_preprocess_saturation_count",
                "activation_saturation_count",
            ]
        )
        test_saturation["passed"] = test_saturation_passed
        labels = np.asarray(shared["split"]["y_test"], dtype=np.int64)
        if (
            np.any((labels < 0) | (labels >= len(CLASS_NAMES)))
            or np.any((fp32_predictions < 0) | (fp32_predictions >= len(CLASS_NAMES)))
            or np.any((fixed_predictions < 0) | (fixed_predictions >= len(CLASS_NAMES)))
            or np.any(fixed_logits.astype(np.int64) < np.iinfo(np.int16).min)
            or np.any(fixed_logits.astype(np.int64) > np.iinfo(np.int16).max)
            or not np.array_equal(
                fixed_logits.astype(np.int32).argmax(axis=1), fixed_predictions
            )
        ):
            raise RuntimeError("Frozen test labels, predictions, or fixed logits are invalid")
        standardized_saturation: dict[str, int] = {}
        for partition in ["train", "validation", "test"]:
            _, stats = quantize_standardized_q15(
                shared["scaled"][f"X_{partition}"],
                input_frac=int(quantized[0]["input_frac"]),
            )
            standardized_saturation[partition] = int(stats["saturation_count"])
        standardized_input_bounds_passed = not any(
            standardized_saturation.values()
        )

        saved = pd.read_csv(model["predictions_path"])
        probability_columns = [
            f"probability_{index}_{name}" for index, name in enumerate(CLASS_NAMES)
        ]
        expected_columns = [
            "source_row_index",
            "true_label",
            "predicted_label",
            *probability_columns,
        ]
        stable = fp32_logits.astype(np.float64) - fp32_logits.max(axis=1, keepdims=True)
        exponentials = np.exp(stable)
        probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
        if (
            saved.columns.tolist() != expected_columns
            or len(saved) != EXPECTED_SPLIT_SIZES["test"]
            or not np.array_equal(
                saved["source_row_index"].to_numpy(np.int64),
                shared["split"]["test_indices"],
            )
            or not np.array_equal(
                saved["true_label"].to_numpy(np.int64), shared["split"]["y_test"]
            )
            or not np.array_equal(
                saved["predicted_label"].to_numpy(np.int64), fp32_predictions
            )
            or not np.allclose(
                saved[probability_columns].to_numpy(np.float64),
                probabilities,
                rtol=final_export.PREDICTION_PROBABILITY_TOLERANCE,
                atol=final_export.PREDICTION_PROBABILITY_TOLERANCE,
            )
        ):
            raise RuntimeError("Preserved prediction artifact reproduction failed")
        fp32_metrics = final_export._metrics_from_predictions(
            labels, fp32_predictions
        )
        final_export._assert_nested_numeric_equal(
            fp32_metrics, model["result"]["metrics"], "all_seed.fp32_metrics"
        )
        fixed_metrics = final_export._metrics_from_predictions(
            labels, fixed_predictions
        )
        agreement = float(np.mean(fp32_predictions == fixed_predictions))
        absolute_drop = abs(
            float(fp32_metrics["macro_f1"]) - float(fixed_metrics["macro_f1"])
        )
        agreement_passed = agreement >= MINIMUM_FIXED_FP32_AGREEMENT
        macro_f1_drop_passed = absolute_drop <= MAXIMUM_ABSOLUTE_MACRO_F1_DROP
        gates = {
            "test_rows": EXPECTED_SPLIT_SIZES["test"],
            "fixed_vs_fp32_agreement": agreement,
            "minimum_fixed_vs_fp32_agreement": MINIMUM_FIXED_FP32_AGREEMENT,
            "absolute_macro_f1_drop": absolute_drop,
            "maximum_absolute_macro_f1_drop": MAXIMUM_ABSOLUTE_MACRO_F1_DROP,
            "accumulator_bounds_passed": accumulator_passed,
            "preprocess_bounds_passed": preprocess_bounds_passed,
            "training_calibration_saturation_passed": training_saturation_passed,
            "validation_calibration_saturation_passed": validation_saturation_passed,
            "test_saturation_passed": test_saturation_passed,
            "standardized_input_bounds_passed": standardized_input_bounds_passed,
            "fixed_vs_fp32_agreement_passed": agreement_passed,
            "macro_f1_drop_passed": macro_f1_drop_passed,
        }
        gates["zero_saturation_passed"] = all(
            gates[key]
            for key in [
                "training_calibration_saturation_passed",
                "validation_calibration_saturation_passed",
                "test_saturation_passed",
                "standardized_input_bounds_passed",
            ]
        )
        gates["quality_gates_passed"] = all(
            gates[key] for key in NUMERIC_PRIMITIVE_GATE_KEYS
        )
        gates["numeric_gate_ledger"] = {
            "accumulator_bounds": _numeric_gate_entry(
                accumulator_passed, evidence=accumulator_ledger
            ),
            "preprocess_bounds": _numeric_gate_entry(
                preprocess_bounds_passed, evidence=preprocess_ledger
            ),
            "training_calibration_saturation": _numeric_gate_entry(
                training_saturation_passed, evidence=training_saturation
            ),
            "validation_calibration_saturation": _numeric_gate_entry(
                validation_saturation_passed, evidence=validation_saturation
            ),
            "test_saturation": _numeric_gate_entry(
                test_saturation_passed, evidence=test_saturation
            ),
            "standardized_input_bounds": _numeric_gate_entry(
                standardized_input_bounds_passed,
                evidence=standardized_saturation,
            ),
            "fixed_vs_fp32_agreement": _numeric_gate_entry(
                agreement_passed,
                evidence={"comparison": "greater_than_or_equal"},
                observed=agreement,
                threshold=MINIMUM_FIXED_FP32_AGREEMENT,
            ),
            "absolute_macro_f1_drop": _numeric_gate_entry(
                macro_f1_drop_passed,
                evidence={"comparison": "less_than_or_equal"},
                observed=absolute_drop,
                threshold=MAXIMUM_ABSOLUTE_MACRO_F1_DROP,
            ),
        }
        if gates["quality_gates_passed"]:
            c_equivalence = run_native_c_equivalence(
                source_root=source_root,
                cc=cc,
                model=model,
                quantized_layers=quantized,
                integer_metadata=integer_metadata,
                raw_inputs=raw_test_q,
                preprocessed=preprocessed,
                logits=fixed_logits,
                predictions=fixed_predictions,
            )
        else:
            c_equivalence = _c_blocked_result(EXPECTED_SPLIT_SIZES["test"])
        gates["c_python_exact_equivalence_passed"] = (
            c_equivalence.get("status") == "passed"
        )
        result = {
            "status": _status_from_gates(gates, c_equivalence),
            "identity": identity,
            "quantization_policy": policy_summary,
            "test_evaluation_scope": _c_scope(c_equivalence),
            "fp32_metrics": fp32_metrics,
            "fixed_metrics": fixed_metrics,
            "fixed_minus_fp32_metric_deltas": _nested_delta(
                fixed_metrics, fp32_metrics
            ),
            "saturation_ledgers": {
                "training": training_saturation,
                "validation": validation_saturation,
                "test": test_saturation,
                "standardized_inputs": standardized_saturation,
            },
            "accumulator_bounds": accumulator_ledger,
            "preprocess_multiply_bounds": preprocess_ledger,
            "gates": gates,
            "c_equivalence": c_equivalence,
        }
    except Exception as exc:
        result = {
            "status": "audit_error",
            "failure_class": "infrastructure_or_logic_exception",
            "requested_identity": {
                "model_id": identifier,
                "seed": seed,
                "student": student,
                "route": route,
            },
            "failure": {
                "exception_type": type(exc).__name__,
                "message": str(exc),
            },
        }
    return _with_payload_hash(result, "model_record_payload_sha256")


def _shared_lineage_payload(shared: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "root_artifact_manifest_sha256": shared["root_artifact_manifest_sha256"],
        "execution_contract_sha256": shared["execution_contract_sha256"],
        "preprocessing_contract_sha256": shared["preprocessing_contract_sha256"],
        "dataset_sha256": shared["dataset"]["dataset_sha256"],
        "split_indices_sha256": shared["preprocessing"]["split_indices_sha256"],
        "test_source_indices_sha256": shared["test_source_indices_sha256"],
        "scaler_sha256": shared["preprocessing"]["scaler_sha256"],
        "split_sizes": EXPECTED_SPLIT_SIZES,
        "feature_overlap_audit": shared["split"]["group_audit"],
    }


def _new_contract(shared: dict[str, Any], snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "audit_protocol_id": AUDIT_PROTOCOL_ID,
        "status": "running",
        "model_count": MODEL_COUNT,
        "model_matrix": expected_model_matrix(),
        "shared_lineage": _shared_lineage_payload(shared),
        "quantization_contract": {
            "selector_module": (
                "deployment.firmware_export.wsnds_final_hil.export_final_seed42"
            ),
            "selector_name": "select_quantization_policy",
            "selector_source_sha256": sha256_file(final_export.SCRIPT_PATH),
            "minimum_fixed_vs_fp32_agreement": MINIMUM_FIXED_FP32_AGREEMENT,
            "maximum_absolute_macro_f1_drop": MAXIMUM_ABSOLUTE_MACRO_F1_DROP,
            "route_specific_exceptions": False,
            "selection_uses_test_data": False,
            "development_status": (
                "post-hoc deployment-method development after historical baseline "
                "fixed-point test-gate observations; not preregistered"
            ),
        },
        "statistical_unit_disclosure": (
            "The 40 models are training-run/model-seed instances sharing one fixed "
            "feature-group-disjoint split and one train-only-fitted scaler; they are "
            "not 40 independent data splits."
        ),
        "retention_boundary": (
            "Retain compact metrics and source evidence only; temporary model headers, "
            "binary row payloads, and native executables are deleted per model."
        ),
        "verification_boundary": (
            "Normal audit generation performs one frozen Python test evaluation per "
            "model. The explicit --verify-only mode performs a separate deterministic "
            "recomputation and is not run implicitly during generation or resume."
        ),
        "source_snapshots": snapshots,
    }
    payload["audit_contract_id"] = _canonical_hash(payload)
    return payload


def _verify_contract(output_root: Path, contract: dict[str, Any], shared: dict[str, Any]) -> None:
    payload = dict(contract)
    contract_id = payload.pop("audit_contract_id", None)
    if contract_id != _canonical_hash(payload):
        raise RuntimeError("All-seed audit contract hash is invalid")
    if (
        contract.get("audit_protocol_id") != AUDIT_PROTOCOL_ID
        or contract.get("model_count") != MODEL_COUNT
        or contract.get("model_matrix") != expected_model_matrix()
        or contract.get("shared_lineage") != _shared_lineage_payload(shared)
    ):
        raise RuntimeError("All-seed audit contract differs from final lineage")
    quantization = contract.get("quantization_contract", {})
    snapshot_hashes = {
        item["snapshot_path"]: item["sha256"]
        for item in contract.get("source_snapshots", [])
        if isinstance(item, dict)
        and isinstance(item.get("snapshot_path"), str)
        and isinstance(item.get("sha256"), str)
    }
    if (
        quantization.get("selector_source_sha256") != sha256_file(final_export.SCRIPT_PATH)
        or quantization.get("selector_source_sha256")
        != snapshot_hashes.get("source_snapshot/python/export_final_seed42.py")
        or quantization.get("minimum_fixed_vs_fp32_agreement")
        != MINIMUM_FIXED_FP32_AGREEMENT
        or quantization.get("maximum_absolute_macro_f1_drop")
        != MAXIMUM_ABSOLUTE_MACRO_F1_DROP
        or quantization.get("route_specific_exceptions") is not False
        or quantization.get("selection_uses_test_data") is not False
        or "post-hoc" not in str(quantization.get("development_status"))
    ):
        raise RuntimeError("All-seed quantization contract is invalid")
    _verify_snapshots(output_root, contract.get("source_snapshots"))


def _progress(output_root: Path, contract_id: str) -> dict[str, Any]:
    completed: list[dict[str, Any]] = []
    for spec in expected_model_matrix():
        path = output_root / MODEL_DIR_NAME / f"{spec['model_id']}.json"
        if not path.is_file():
            continue
        record = _read_json(path)
        _verify_payload_hash(record, "model_record_payload_sha256", spec["model_id"])
        _validate_record_semantics(record)
        completed.append({
            **spec,
            "status": record.get("status"),
            "record_sha256": sha256_file(path),
        })
    payload = {
        "audit_protocol_id": AUDIT_PROTOCOL_ID,
        "audit_contract_id": contract_id,
        "status": "complete" if len(completed) == MODEL_COUNT else "running",
        "completed_count": len(completed),
        "remaining_count": MODEL_COUNT - len(completed),
        "completed_models": completed,
    }
    return _with_payload_hash(payload, "progress_payload_sha256")


def _initialize_or_resume(
    output_root: Path, shared: dict[str, Any]
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    contract_path = output_root / CONTRACT_NAME
    if contract_path.exists():
        contract = _read_json(contract_path)
    else:
        if any(output_root.iterdir()):
            raise RuntimeError("Refusing to initialize a nonempty unsealed audit directory")
        snapshots = _source_snapshots(output_root)
        contract = _new_contract(shared, snapshots)
        _atomic_write_json(contract_path, contract)
    _verify_contract(output_root, contract, shared)
    (output_root / MODEL_DIR_NAME).mkdir(exist_ok=True)
    progress = _progress(output_root, contract["audit_contract_id"])
    _atomic_write_json(output_root / PROGRESS_NAME, progress)
    return contract


def _deterministic_record(record: dict[str, Any]) -> dict[str, Any]:
    payload = dict(record)
    payload.pop("model_record_payload_sha256", None)
    if payload.get("status") in {
        "passed",
        "gate_failed",
        "c_equivalence_failed",
    }:
        c_equivalence = dict(payload["c_equivalence"])
        c_equivalence.pop("generation_host_evidence", None)
        if c_equivalence.get("status") == "failed":
            c_equivalence.pop("failure", None)
        payload["c_equivalence"] = c_equivalence
    return payload


def _verification_projection(record: dict[str, Any]) -> dict[str, Any]:
    return _deterministic_record(record)


def _c_result_semantics(c_evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": c_evidence.get("status"),
        "failed_phase": c_evidence.get("failed_phase"),
        "failed_outcome": c_evidence.get("failed_outcome"),
        "phases": c_evidence.get("phases"),
    }


def _validate_c_phases(c_evidence: dict[str, Any]) -> None:
    phases = c_evidence.get("phases")
    if not isinstance(phases, dict) or set(phases) != set(C_PHASES):
        raise RuntimeError("Per-model C phase ledger is incomplete")
    failed_phase = c_evidence.get("failed_phase")
    failed_outcome = c_evidence.get("failed_outcome")
    if c_evidence.get("status") == "blocked":
        if (
            failed_phase is not None
            or failed_outcome is not None
            or c_evidence.get("outcome") != "blocked_by_numeric_gate_failure"
            or any(
                phases[phase]
                != {
                    "attempted": False,
                    "status": "not_attempted",
                    "outcome": None,
                }
                for phase in C_PHASES
            )
        ):
            raise RuntimeError("Numerically blocked C evidence claims attempted work")
        return
    if c_evidence.get("status") == "passed":
        if failed_phase is not None or failed_outcome is not None:
            raise RuntimeError("Passed C evidence retains a failure outcome")
        for phase, outcome in C_SUCCESS_OUTCOMES.items():
            if phases.get(phase) != {
                "attempted": True,
                "status": "completed",
                "outcome": outcome,
            }:
                raise RuntimeError(f"Passed C phase evidence is invalid: {phase}")
        return
    if c_evidence.get("status") != "failed" or failed_phase not in C_PHASES:
        raise RuntimeError("Failed C evidence lacks an exact failed phase")
    failed_index = C_PHASES.index(failed_phase)
    if not isinstance(failed_outcome, str) or not failed_outcome:
        raise RuntimeError("Failed C evidence lacks an exact failed outcome")
    permitted_failures = {
        "compiler_discovery": {
            "process_start_failed",
            "compiler_version_nonzero_exit",
        },
        "compilation": {"process_start_failed", "compiler_nonzero_exit"},
        "execution": {"process_start_failed"},
        "verification": {"self_test_nonzero_exit"},
    }
    if failed_outcome not in permitted_failures[failed_phase]:
        raise RuntimeError("Failed C evidence contains an unsupported outcome")
    for index, phase in enumerate(C_PHASES):
        item = phases.get(phase)
        if not isinstance(item, dict):
            raise RuntimeError(f"C phase evidence is invalid: {phase}")
        if index < failed_index and item != {
            "attempted": True,
            "status": "completed",
            "outcome": C_SUCCESS_OUTCOMES[phase],
        }:
            raise RuntimeError(f"C phase before failure is incomplete: {phase}")
        if index == failed_index and item != {
            "attempted": True,
            "status": "failed",
            "outcome": failed_outcome,
        }:
            raise RuntimeError(f"Failed C phase outcome is inconsistent: {phase}")
        if index > failed_index and item != {
            "attempted": False,
            "status": "not_attempted",
            "outcome": None,
        }:
            raise RuntimeError(f"C phase after failure was falsely claimed: {phase}")


def _validate_record_semantics(record: dict[str, Any]) -> None:
    status = record.get("status")
    if status == "audit_error":
        if "gates" in record or "c_equivalence" in record:
            raise RuntimeError("Audit-error record contains completed gate evidence")
        failure = record.get("failure")
        if (
            record.get("failure_class") != "infrastructure_or_logic_exception"
            or not isinstance(failure, dict)
            or not isinstance(failure.get("exception_type"), str)
            or not isinstance(failure.get("message"), str)
        ):
            raise RuntimeError("Audit-error record lacks infrastructure failure evidence")
        return
    if status not in {"passed", "gate_failed", "c_equivalence_failed"}:
        raise RuntimeError(f"Unknown per-model audit status: {status!r}")
    gates = record.get("gates")
    c_evidence = record.get("c_equivalence")
    if not isinstance(gates, dict) or not isinstance(c_evidence, dict):
        raise RuntimeError("Completed model record lacks gate or C evidence")
    ledger = gates.get("numeric_gate_ledger")
    expected_ledger_keys = {
        "accumulator_bounds",
        "preprocess_bounds",
        "training_calibration_saturation",
        "validation_calibration_saturation",
        "test_saturation",
        "standardized_input_bounds",
        "fixed_vs_fp32_agreement",
        "absolute_macro_f1_drop",
    }
    if not isinstance(ledger, dict) or set(ledger) != expected_ledger_keys:
        raise RuntimeError("Numeric deployment gate ledger is incomplete")
    primitive_to_ledger = {
        "accumulator_bounds_passed": "accumulator_bounds",
        "preprocess_bounds_passed": "preprocess_bounds",
        "training_calibration_saturation_passed": (
            "training_calibration_saturation"
        ),
        "validation_calibration_saturation_passed": (
            "validation_calibration_saturation"
        ),
        "test_saturation_passed": "test_saturation",
        "standardized_input_bounds_passed": "standardized_input_bounds",
        "fixed_vs_fp32_agreement_passed": "fixed_vs_fp32_agreement",
        "macro_f1_drop_passed": "absolute_macro_f1_drop",
    }
    for primitive, ledger_key in primitive_to_ledger.items():
        if not isinstance(gates.get(primitive), bool):
            raise RuntimeError(f"Numeric gate is not Boolean: {primitive}")
        ledger_entry = ledger.get(ledger_key)
        if (
            not isinstance(ledger_entry, dict)
            or not isinstance(ledger_entry.get("passed"), bool)
            or "evidence" not in ledger_entry
            or ledger_entry["passed"] is not gates[primitive]
        ):
            raise RuntimeError(f"Numeric gate ledger disagrees: {primitive}")
    if (
        gates.get("test_rows") != EXPECTED_SPLIT_SIZES["test"]
        or gates.get("minimum_fixed_vs_fp32_agreement")
        != MINIMUM_FIXED_FP32_AGREEMENT
        or gates.get("maximum_absolute_macro_f1_drop")
        != MAXIMUM_ABSOLUTE_MACRO_F1_DROP
    ):
        raise RuntimeError("Numeric gate thresholds or test scope were altered")
    agreement = gates.get("fixed_vs_fp32_agreement")
    macro_drop = gates.get("absolute_macro_f1_drop")
    if (
        type(agreement) not in {int, float}
        or not np.isfinite(agreement)
        or agreement < 0.0
        or agreement > 1.0
        or type(macro_drop) not in {int, float}
        or not np.isfinite(macro_drop)
        or macro_drop < 0.0
    ):
        raise RuntimeError("Numeric fidelity gate observation is invalid")
    agreement_entry = ledger["fixed_vs_fp32_agreement"]
    drop_entry = ledger["absolute_macro_f1_drop"]
    if (
        agreement_entry.get("threshold") != MINIMUM_FIXED_FP32_AGREEMENT
        or agreement_entry.get("observed") != agreement
        or agreement_entry.get("evidence")
        != {"comparison": "greater_than_or_equal"}
        or gates["fixed_vs_fp32_agreement_passed"]
        is not (agreement >= MINIMUM_FIXED_FP32_AGREEMENT)
        or drop_entry.get("threshold") != MAXIMUM_ABSOLUTE_MACRO_F1_DROP
        or drop_entry.get("observed") != macro_drop
        or drop_entry.get("evidence") != {"comparison": "less_than_or_equal"}
        or gates["macro_f1_drop_passed"]
        is not (macro_drop <= MAXIMUM_ABSOLUTE_MACRO_F1_DROP)
    ):
        raise RuntimeError("Numeric fidelity gate is not derived from its observation")
    accumulator_evidence = ledger["accumulator_bounds"]["evidence"]
    preprocess_evidence = ledger["preprocess_bounds"]["evidence"]
    if (
        not isinstance(accumulator_evidence, list)
        or len(accumulator_evidence) != 3
        or not all(isinstance(item, dict) for item in accumulator_evidence)
        or not isinstance(preprocess_evidence, list)
        or len(preprocess_evidence) != 17
        or not all(isinstance(item, dict) for item in preprocess_evidence)
    ):
        raise RuntimeError("Numeric bound gate evidence has an invalid shape")
    for index, item in enumerate(accumulator_evidence):
        maximum = item.get("pre_rescale_absolute_bound")
        shift = item.get("output_shift")
        derived = (
            type(maximum) is int
            and maximum >= 0
            and type(shift) is int
            and shift >= 0
            and maximum <= int(np.iinfo(np.int32).max)
        )
        if (
            item.get("layer") != index
            or item.get("post_left_shift_absolute_bound") != maximum
            or item.get("int32_max") != int(np.iinfo(np.int32).max)
            or item.get("passed") is not derived
        ):
            raise RuntimeError("Accumulator gate is not derived from its evidence")
    if gates["accumulator_bounds_passed"] is not all(
        item["passed"] for item in accumulator_evidence
    ):
        raise RuntimeError("Aggregate accumulator gate is inconsistent")
    for index, item in enumerate(preprocess_evidence):
        centered = item.get("maximum_centered_absolute")
        inverse = item.get("inverse_scale_absolute")
        product = item.get("maximum_product_absolute")
        derived = (
            type(centered) is int
            and centered >= 0
            and type(inverse) is int
            and inverse >= 0
            and product == centered * inverse
            and product <= int(np.iinfo(np.int64).max)
        )
        if (
            item.get("feature") != index
            or item.get("int64_max") != int(np.iinfo(np.int64).max)
            or item.get("passed") is not derived
        ):
            raise RuntimeError("Preprocessing gate is not derived from its evidence")
    if gates["preprocess_bounds_passed"] is not all(
        item["passed"] for item in preprocess_evidence
    ):
        raise RuntimeError("Aggregate preprocessing gate is inconsistent")
    saturation_keys = [
        "raw_input_saturation_count",
        "weight_saturation_count",
        "bias_saturation_count",
        "integer_preprocess_saturation_count",
        "activation_saturation_count",
    ]
    for gate_key, ledger_key, partition, expected_rows in [
        (
            "training_calibration_saturation_passed",
            "training_calibration_saturation",
            "training calibration partition",
            EXPECTED_SPLIT_SIZES["train"],
        ),
        (
            "validation_calibration_saturation_passed",
            "validation_calibration_saturation",
            "validation partition",
            EXPECTED_SPLIT_SIZES["validation"],
        ),
        (
            "test_saturation_passed",
            "test_saturation",
            "test partition",
            EXPECTED_SPLIT_SIZES["test"],
        ),
    ]:
        evidence = ledger[ledger_key]["evidence"]
        parameter_layers = evidence.get("parameter_layers", []) if isinstance(
            evidence, dict
        ) else []
        activation_layers = evidence.get("activation_layers", []) if isinstance(
            evidence, dict
        ) else []
        if (
            not isinstance(evidence, dict)
            or evidence.get("partition") != partition
            or evidence.get("rows_audited") != expected_rows
            or any(
                type(evidence.get(key)) is not int or evidence[key] < 0
                for key in saturation_keys
            )
            or not isinstance(parameter_layers, list)
            or len(parameter_layers) != 3
            or not all(isinstance(item, dict) for item in parameter_layers)
            or evidence["weight_saturation_count"]
            != sum(item.get("weight_saturation_count", -1) for item in parameter_layers)
            or evidence["bias_saturation_count"]
            != sum(item.get("bias_saturation_count", -1) for item in parameter_layers)
            or not isinstance(activation_layers, list)
            or len(activation_layers) != 3
            or not all(isinstance(item, dict) for item in activation_layers)
            or evidence["activation_saturation_count"]
            != sum(
                item.get("activation_saturation_count", -1)
                for item in activation_layers
            )
            or gates[gate_key]
            is not all(evidence[key] == 0 for key in saturation_keys)
            or evidence.get("passed") is not gates[gate_key]
        ):
            raise RuntimeError(f"Saturation gate is not derived from evidence: {gate_key}")
    standardized_evidence = ledger["standardized_input_bounds"]["evidence"]
    if (
        not isinstance(standardized_evidence, dict)
        or set(standardized_evidence) != {"train", "validation", "test"}
        or any(
            type(value) is not int or value < 0
            for value in standardized_evidence.values()
        )
        or gates["standardized_input_bounds_passed"]
        is not all(value == 0 for value in standardized_evidence.values())
    ):
        raise RuntimeError("Standardized-input gate is not derived from evidence")
    expected_zero_saturation = all(
        gates[key]
        for key in [
            "training_calibration_saturation_passed",
            "validation_calibration_saturation_passed",
            "test_saturation_passed",
            "standardized_input_bounds_passed",
        ]
    )
    expected_numeric = all(gates[key] for key in NUMERIC_PRIMITIVE_GATE_KEYS)
    if gates.get("zero_saturation_passed") is not expected_zero_saturation:
        raise RuntimeError("Aggregate zero-saturation gate is inconsistent")
    if gates.get("quality_gates_passed") is not expected_numeric:
        raise RuntimeError("Aggregate numeric quality gate is inconsistent")
    c_passed = c_evidence.get("status") == "passed"
    if gates.get("c_python_exact_equivalence_passed") is not c_passed:
        raise RuntimeError("C-equivalence Boolean gate disagrees with C evidence")
    if expected_numeric and c_evidence.get("status") == "blocked":
        raise RuntimeError("Numerically valid record has blocked C evidence")
    if not expected_numeric and c_evidence.get("status") != "blocked":
        raise RuntimeError("Numerically failed record must not claim C work")
    _validate_c_phases(c_evidence)
    if record.get("test_evaluation_scope") != _c_scope(c_evidence):
        raise RuntimeError("Test/C evaluation scope disagrees with attempted phases")
    expected_status = _status_from_gates(gates, c_evidence)
    if status != expected_status:
        raise RuntimeError(
            f"Per-model status {status!r} differs from gate-derived {expected_status!r}"
        )


def _validate_host_evidence(record: dict[str, Any]) -> None:
    _validate_record_semantics(record)
    c_evidence = record.get("c_equivalence", {})
    if c_evidence.get("status") == "blocked":
        if (
            record.get("status") != "gate_failed"
            or c_evidence.get("rows") != EXPECTED_SPLIT_SIZES["test"]
            or "generation_host_evidence" in c_evidence
        ):
            raise RuntimeError("Numerically blocked C evidence is inconsistent")
        return
    if record.get("status") == "c_equivalence_failed":
        failure = c_evidence.get("failure")
        host = c_evidence.get("generation_host_evidence", {})
        failed_phase = c_evidence.get("failed_phase")
        failed_outcome = c_evidence.get("failed_outcome")
        if (
            c_evidence.get("status") != "failed"
            or c_evidence.get("rows") != EXPECTED_SPLIT_SIZES["test"]
            or not isinstance(failure, dict)
            or not isinstance(failure.get("exception_type"), str)
            or not isinstance(failure.get("message"), str)
            or not isinstance(host, dict)
            or not isinstance(host.get("compiler_requested"), str)
            or not host["compiler_requested"]
            or host.get("temporary_executable_retained") is not False
        ):
            raise RuntimeError("Retained per-model C failure evidence is incomplete")
        compiler = host.get("compiler_version")
        compilation = host.get("compile")
        execution = host.get("self_test")
        if failed_phase == "compiler_discovery":
            if failed_outcome == "process_start_failed":
                if compiler is not None:
                    raise RuntimeError("Compiler-start failure contains a compiler result")
            elif (
                failed_outcome != "compiler_version_nonzero_exit"
                or not isinstance(compiler, dict)
                or compiler.get("returncode") == 0
            ):
                raise RuntimeError("Compiler-discovery failure outcome is unsupported")
            if compilation is not None or execution is not None:
                raise RuntimeError("C evidence claims work after compiler discovery failed")
            if host.get("temporary_executable_sha256") is not None:
                raise RuntimeError("Compiler-discovery failure claims an executable")
        elif failed_phase == "compilation":
            if not isinstance(compiler, dict) or compiler.get("returncode") != 0:
                raise RuntimeError("Compilation failure lacks successful compiler discovery")
            if failed_outcome == "process_start_failed":
                if compilation is not None:
                    raise RuntimeError("Compiler-start failure contains a compile result")
            elif (
                failed_outcome != "compiler_nonzero_exit"
                or not isinstance(compilation, dict)
                or compilation.get("returncode") == 0
            ):
                raise RuntimeError("Compilation failure outcome is unsupported")
            if execution is not None:
                raise RuntimeError("C evidence claims execution after compilation failed")
            if host.get("temporary_executable_sha256") is not None:
                raise RuntimeError("Compilation failure claims an executable")
        elif failed_phase == "execution":
            if (
                not isinstance(compiler, dict)
                or compiler.get("returncode") != 0
                or not isinstance(compilation, dict)
                or compilation.get("returncode") != 0
                or failed_outcome != "process_start_failed"
                or execution is not None
            ):
                raise RuntimeError("Native execution failure evidence is inconsistent")
            final_export._assert_hash(
                host.get("temporary_executable_sha256"),
                "temporary audit executable",
            )
        elif failed_phase == "verification":
            if (
                not isinstance(compiler, dict)
                or compiler.get("returncode") != 0
                or not isinstance(compilation, dict)
                or compilation.get("returncode") != 0
                or not isinstance(execution, dict)
                or execution.get("returncode") == 0
                or failed_outcome != "self_test_nonzero_exit"
            ):
                raise RuntimeError("C verification failure evidence is inconsistent")
            final_export._assert_hash(
                host.get("temporary_executable_sha256"),
                "temporary audit executable",
            )
        return
    if record.get("status") not in {"passed", "gate_failed"}:
        return
    host = c_evidence.get("generation_host_evidence", {})
    if (
        c_evidence.get("status") != "passed"
        or c_evidence.get("rows") != EXPECTED_SPLIT_SIZES["test"]
        or c_evidence.get("preprocessed_inputs_exact") is not True
        or c_evidence.get("fixed_logits_exact") is not True
        or c_evidence.get("fixed_predictions_exact") is not True
        or not isinstance(host.get("compiler_requested"), str)
        or not host["compiler_requested"]
        or host.get("compiler_version", {}).get("returncode") != 0
        or host.get("compile", {}).get("returncode") != 0
        or host.get("self_test", {}).get("returncode") != 0
        or host.get("temporary_executable_retained") is not False
    ):
        raise RuntimeError("Per-model C equivalence evidence is incomplete")
    final_export._assert_hash(
        host.get("temporary_executable_sha256"), "temporary audit executable"
    )
    for key in [
        "payload_sha256",
        "raw_inputs_content_sha256",
        "preprocessed_content_sha256",
        "fixed_logits_content_sha256",
        "fixed_predictions_content_sha256",
    ]:
        final_export._assert_hash(c_evidence.get(key), f"C evidence {key}")
    sealed_sources = c_evidence.get("sealed_source_sha256")
    if not isinstance(sealed_sources, dict) or set(sealed_sources) != {
        "cukd_preprocess.c",
        "cukd_model.c",
        "all_seed_stream_self_test.c",
    }:
        raise RuntimeError("C evidence sealed-source ledger is incomplete")
    for name, digest in sealed_sources.items():
        final_export._assert_hash(digest, f"C evidence sealed source {name}")


def _finalize(output_root: Path, contract: dict[str, Any]) -> None:
    progress = _progress(output_root, contract["audit_contract_id"])
    if progress["completed_count"] != MODEL_COUNT:
        raise RuntimeError("Cannot finalize an incomplete 40-model audit")
    for spec in expected_model_matrix():
        record = _read_json(
            output_root / MODEL_DIR_NAME / f"{spec['model_id']}.json"
        )
        _verify_payload_hash(
            record, "model_record_payload_sha256", spec["model_id"]
        )
        _validate_host_evidence(record)
    _atomic_write_json(output_root / PROGRESS_NAME, progress)
    records = []
    status_counts: dict[str, int] = {}
    for item in progress["completed_models"]:
        status = str(item["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        records.append(item)
    report = _with_payload_hash({
        "audit_protocol_id": AUDIT_PROTOCOL_ID,
        "audit_contract_id": contract["audit_contract_id"],
        "status": _report_status(status_counts),
        "model_count": MODEL_COUNT,
        "status_counts": status_counts,
        "records": records,
        "statistical_unit_disclosure": contract["statistical_unit_disclosure"],
        "post_hoc_method_disclosure": contract["quantization_contract"][
            "development_status"
        ],
    }, "report_payload_sha256")
    _atomic_write_json(output_root / REPORT_NAME, report)
    inventory = file_inventory(output_root, {MANIFEST_NAME})
    manifest = _with_payload_hash({
        "audit_protocol_id": AUDIT_PROTOCOL_ID,
        "audit_contract_id": contract["audit_contract_id"],
        "status": report["status"],
        "file_count_excluding_manifest": len(inventory),
        "files": inventory,
        "report_canonical_sha256": _canonical_hash(report),
    }, "manifest_payload_sha256")
    _atomic_write_json(output_root / MANIFEST_NAME, manifest)


def run_all_seed_audit(
    confirmation_root: Path,
    dataset_csv: Path,
    output_dir: Path,
    cc: str = "gcc",
) -> Path:
    shared = load_shared_lineage(confirmation_root, dataset_csv)
    output_root = output_dir.resolve()
    final_export.require_output_outside_inputs(
        output_root, [shared["confirmation_root"], shared["dataset_csv"].parent]
    )
    contract = _initialize_or_resume(output_root, shared)
    if (output_root / MANIFEST_NAME).exists():
        verify_all_seed_audit(output_root, cc=cc, recompute_models=False)
        return output_root / MANIFEST_NAME
    for spec in expected_model_matrix():
        record_path = output_root / MODEL_DIR_NAME / f"{spec['model_id']}.json"
        if record_path.exists():
            record = _read_json(record_path)
            _verify_payload_hash(
                record, "model_record_payload_sha256", spec["model_id"]
            )
            _validate_host_evidence(record)
            continue
        record = evaluate_model(
            shared,
            spec["seed"],
            spec["student"],
            spec["route"],
            output_root,
            cc,
        )
        _atomic_write_json(record_path, record)
        _atomic_write_json(
            output_root / PROGRESS_NAME,
            _progress(output_root, contract["audit_contract_id"]),
        )
    _finalize(output_root, contract)
    verify_all_seed_audit(output_root, cc=cc, recompute_models=False)
    return output_root / MANIFEST_NAME


def _verify_manifest_inventory(output_root: Path, manifest: dict[str, Any]) -> None:
    files = manifest.get("files")
    if not isinstance(files, list) or manifest.get("file_count_excluding_manifest") != len(files):
        raise RuntimeError("All-seed manifest file count is invalid")
    listed: dict[str, dict[str, Any]] = {}
    for item in files:
        relative = item.get("path")
        if not isinstance(relative, str) or not relative:
            raise RuntimeError("All-seed manifest path is invalid")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts or relative in listed:
            raise RuntimeError("All-seed manifest contains an unsafe or duplicate path")
        member = (output_root / relative_path).resolve()
        try:
            member.relative_to(output_root)
        except ValueError as exc:
            raise RuntimeError("All-seed manifest path escapes output") from exc
        if (
            not member.is_file()
            or member.stat().st_size != item.get("size_bytes")
            or sha256_file(member) != item.get("sha256")
        ):
            raise RuntimeError(f"All-seed manifest member differs: {relative}")
        listed[relative_path.as_posix()] = item
    actual = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file() and path.name != MANIFEST_NAME
    }
    expected = {
        CONTRACT_NAME,
        PROGRESS_NAME,
        REPORT_NAME,
        *[snapshot for _, snapshot in SOURCE_SNAPSHOT_SPECS],
        *[
            f"{MODEL_DIR_NAME}/{item['model_id']}.json"
            for item in expected_model_matrix()
        ],
    }
    if actual != set(listed) or set(listed) != expected:
        raise RuntimeError("All-seed audit inventory is not exact")


def verify_all_seed_audit(
    path: Path,
    *,
    cc: str = "gcc",
    recompute_models: bool = True,
) -> dict[str, Any]:
    """Verify a completed audit, optionally replaying all 40 model evaluations."""
    output_root = path.resolve()
    manifest = _read_json(output_root / MANIFEST_NAME)
    contract = _read_json(output_root / CONTRACT_NAME)
    progress = _read_json(output_root / PROGRESS_NAME)
    report = _read_json(output_root / REPORT_NAME)
    _verify_payload_hash(manifest, "manifest_payload_sha256", "all-seed manifest")
    _verify_payload_hash(progress, "progress_payload_sha256", "all-seed progress")
    _verify_payload_hash(report, "report_payload_sha256", "all-seed report")
    _verify_manifest_inventory(output_root, manifest)
    shared = load_shared_lineage(
        REPO_ROOT / EXPECTED_RELATIVE_ROOT,
        REPO_ROOT / "data" / "wsnds" / "WSN-DS.csv",
    )
    _verify_contract(output_root, contract, shared)
    if (
        manifest.get("audit_protocol_id") != AUDIT_PROTOCOL_ID
        or manifest.get("audit_contract_id") != contract.get("audit_contract_id")
        or manifest.get("report_canonical_sha256") != _canonical_hash(report)
        or manifest.get("status") != report.get("status")
        or report.get("model_count") != MODEL_COUNT
        or progress.get("completed_count") != MODEL_COUNT
        or progress.get("remaining_count") != 0
        or progress.get("status") != "complete"
    ):
        raise RuntimeError("All-seed final documents disagree")
    expected_progress = _progress(output_root, contract["audit_contract_id"])
    if progress != expected_progress or report.get("records") != progress.get(
        "completed_models"
    ):
        raise RuntimeError("All-seed progress/report record ledger is invalid")
    record_status_counts: dict[str, int] = {}
    for item in expected_progress["completed_models"]:
        status = str(item.get("status"))
        record_status_counts[status] = record_status_counts.get(status, 0) + 1
    if report.get("status_counts") != record_status_counts:
        raise RuntimeError("All-seed report status counts differ from model records")
    if report.get("status") != _report_status(record_status_counts):
        raise RuntimeError(
            "All-seed report status differs from recomputed per-model statuses"
        )
    for spec in expected_model_matrix():
        record_path = output_root / MODEL_DIR_NAME / f"{spec['model_id']}.json"
        record = _read_json(record_path)
        _verify_payload_hash(
            record, "model_record_payload_sha256", spec["model_id"]
        )
        _validate_host_evidence(record)
    if not recompute_models:
        return {
            "status": "structurally_verified",
            "model_count": MODEL_COUNT,
            "status_counts": record_status_counts,
            "models_recomputed": False,
        }
    recomputed_counts: dict[str, int] = {}
    for spec in expected_model_matrix():
        record_path = output_root / MODEL_DIR_NAME / f"{spec['model_id']}.json"
        record = _read_json(record_path)
        _verify_payload_hash(
            record, "model_record_payload_sha256", spec["model_id"]
        )
        _validate_host_evidence(record)
        recomputed = evaluate_model(
            shared,
            spec["seed"],
            spec["student"],
            spec["route"],
            output_root,
            cc,
        )
        _validate_host_evidence(recomputed)
        if _c_result_semantics(record.get("c_equivalence", {})) != (
            _c_result_semantics(recomputed.get("c_equivalence", {}))
        ):
            raise RuntimeError(
                "Current C result semantics differ for "
                f"verified.{spec['model_id']}"
            )
        final_export._assert_nested_numeric_equal(
            _verification_projection(record),
            _verification_projection(recomputed),
            f"verified.{spec['model_id']}",
        )
        status = str(record.get("status"))
        recomputed_counts[status] = recomputed_counts.get(status, 0) + 1
    if recomputed_counts != record_status_counts:
        raise RuntimeError("Deeply recomputed per-model statuses differ from records")
    expected_report_status = _report_status(recomputed_counts)
    if report.get("status") != expected_report_status:
        raise RuntimeError(
            "All-seed report status differs from recomputed per-model statuses"
        )
    return {
        "status": "verified",
        "model_count": MODEL_COUNT,
        "status_counts": recomputed_counts,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verify_only:
        result = verify_all_seed_audit(args.output_dir, cc=args.cc)
        print(json.dumps(result, indent=2))
        return 0
    manifest = run_all_seed_audit(
        args.confirmation_root,
        args.dataset_csv,
        args.output_dir,
        cc=args.cc,
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
