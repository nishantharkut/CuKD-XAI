"""Export a manifest-bound final WSN-DS seed-42 student for MCU HIL replay.

The exporter is additive and fail-closed. It consumes the completed ten-seed
feature-group-disjoint confirmation run, reproduces the selected checkpoint's
test predictions, and exports one Student A/B scratch or RF-KD route without
training or changing any preserved artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
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

from deployment.firmware_export.wsnds_rfkd_hil.export_fgds_seed42_deployment import (  # noqa: E402
    accumulator_bounds,
    bind_replay_source_rows,
    calibration_partition_saturation_audit,
    file_inventory,
    preprocess_multiply_bounds,
    require_output_outside_inputs,
    resolve_manifest_member,
    saturation_audit,
    verify_manifest,
    write_reference_with_logits,
)
from deployment.firmware_export.wsnds_rfkd_hil.export_wsnds_student_a_rfkd_int8 import (  # noqa: E402
    build_preprocessing_metadata,
    build_integer_preprocessing_metadata,
    calibrate_quantized_layers,
    extract_linear_layers,
    forward_numpy,
    generate_e2e_artifacts,
    quantize_raw_features_q,
    quantize_layer,
    quantize_standardized_q15,
    simulate_fixed_point_inference,
    simulate_integer_preprocess_q,
    write_header,
    write_integer_preprocessing_header,
    write_preprocessing_header,
)
from experiments.wsnds.leakage_free_rerun.tier15_common import (  # noqa: E402
    CLASS_NAMES,
    RF_CONFIG,
    STUDENT_SPECS,
    TRAIN_CONFIG,
    apply_train_scaler,
    classification_metrics,
    feature_group_split,
    load_wsnds,
    sha256_arrays,
    sha256_file,
    split_hashes,
)


PROTOCOL_ID = "wsnds_feature_group_split_train_only_scaler_10seed_v2"
SEED = 42
EXPECTED_SEEDS = [42, 123, 456, 789, 1001, 2024, 3141, 5678, 8192, 9999]
EXPECTED_RELATIVE_ROOT = Path(
    "results/wsnds/confirmation_runs_v2/"
    "local_feature_group_10seed_20260811/feature_group_10seed"
)
EXPECTED_SPLIT_SIZES = {"train": 262197, "validation": 56163, "test": 56301}
MINIMUM_FIXED_FP32_AGREEMENT = 0.99
MAXIMUM_ABSOLUTE_MACRO_F1_DROP = 0.015
PREDICTION_PROBABILITY_TOLERANCE = 2.0e-6
FIRMWARE_COMMON_DIR = REPO_ROOT / "deployment" / "hardware_hil" / "firmware" / "common"
HOST_SELF_TEST_SOURCE = (
    REPO_ROOT
    / "deployment"
    / "firmware_export"
    / "wsnds_rfkd_hil"
    / "wsnds_train_only_self_test.c"
)
LEGACY_NUMERIC_EXPORTER = (
    REPO_ROOT
    / "deployment"
    / "firmware_export"
    / "wsnds_rfkd_hil"
    / "export_wsnds_student_a_rfkd_int8.py"
)
CORE_EXPORT_FILES = [
    "model_weights.h",
    "preprocess_metadata.h",
    "preprocess_metadata.json",
    "preprocess_int_metadata.h",
    "preprocess_int_metadata.json",
    "test_vectors.h",
    "hil_replay_vectors.csv",
    "hil_reference_predictions.csv",
    "equivalence_report.json",
]
SOURCE_SNAPSHOT_SPECS = [
    (
        "deployment/firmware_export/wsnds_final_hil/export_final_seed42.py",
        "source_snapshot/python/export_final_seed42.py",
    ),
    (
        "deployment/firmware_export/wsnds_rfkd_hil/export_fgds_seed42_deployment.py",
        "source_snapshot/python/export_fgds_seed42_deployment.py",
    ),
    (
        "deployment/firmware_export/wsnds_rfkd_hil/export_wsnds_student_a_rfkd_int8.py",
        "source_snapshot/python/export_wsnds_student_a_rfkd_int8.py",
    ),
    (
        "experiments/wsnds/leakage_free_rerun/tier15_common.py",
        "source_snapshot/python/tier15_common.py",
    ),
    (
        "deployment/firmware_export/wsnds_rfkd_hil/wsnds_train_only_self_test.c",
        "source_snapshot/c/wsnds_train_only_self_test.c",
    ),
    (
        "deployment/hardware_hil/firmware/common/cukd_model.c",
        "source_snapshot/c/cukd_model.c",
    ),
    (
        "deployment/hardware_hil/firmware/common/cukd_model.h",
        "source_snapshot/c/cukd_model.h",
    ),
    (
        "deployment/hardware_hil/firmware/common/cukd_preprocess.c",
        "source_snapshot/c/cukd_preprocess.c",
    ),
    (
        "deployment/hardware_hil/firmware/common/cukd_preprocess.h",
        "source_snapshot/c/cukd_preprocess.h",
    ),
]
EXPECTED_FINAL_LINEAGE_FILES = {
    "artifact_manifest.json": "6bb4a7d9456ea3bd93dbb479c4ea9c34b9061881179c943521b18db96b491e92",
    "execution_contract.json": "21722565d0ec4b9e6e25a8e0a48617db0e7c4debff4891295dd4e5ef32ffb3ad",
    "preprocessing_contract.json": "2c2499caa28fec5e3e4253596631d3e8fd0dc8ac65623b3e9080c02f4bd30c22",
    "split_indices.npz": "f2ddf428ea89e8a32809e9bce69f6df3bc046e4c66dfec220f2ffe72c49bc036",
    "scaler_parameters.npz": "641def12f200ff2922b0e3b8cc9525b5d123d4361354ffbede9c083852ccee6a",
    "seed_42/artifact_manifest.json": "0ed0e6708e3f1a9f9e3fc78315a1dc49d3c26a4bea53a17494d9c62f6e14f57b",
    "seed_42/seed_completion.json": "25e36afaf33b68ec5433887f03fcd888966e40b966c465fec2ca912fcb82d87b",
}


class FinalQualityGateError(RuntimeError):
    """A frozen test gate failed after quantization policy selection."""

    def __init__(self, message: str, audit: dict[str, Any]):
        super().__init__(message)
        self.audit = audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", choices=["A", "B"], required=True)
    parser.add_argument("--route", choices=["scratch", "rf_kd"], required=True)
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
    parser.add_argument(
        "--blocked-audit-json",
        type=Path,
        help="Write a machine-readable audit only when frozen test gates block export.",
    )
    parser.add_argument("--cc", default="gcc")
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_hash(value: Any, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError(f"{label} is not a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise RuntimeError(f"{label} is not a SHA-256 digest") from exc


def _assert_nested_numeric_equal(observed: Any, expected: Any, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(observed) != set(expected):
            raise RuntimeError(f"{label} fields differ from the preserved result")
        for key, value in expected.items():
            _assert_nested_numeric_equal(observed[key], value, f"{label}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise RuntimeError(f"{label} shape differs from the preserved result")
        for index, value in enumerate(expected):
            _assert_nested_numeric_equal(observed[index], value, f"{label}[{index}]")
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not np.isclose(float(observed), float(expected), rtol=1e-12, atol=1e-12):
            raise RuntimeError(f"{label} differs from the preserved result")
        return
    if observed != expected:
        raise RuntimeError(f"{label} differs from the preserved result")


def verify_execution_contract(
    root: Path,
    execution: dict[str, Any],
    completion: dict[str, Any],
) -> None:
    expected = {
        "protocol_id": PROTOCOL_ID,
        "mode": "duplicate-sensitivity",
        "seeds": EXPECTED_SEEDS,
        "students": {name: list(dims) for name, dims in STUDENT_SPECS.items()},
        "routes": ["scratch", "rf_kd"],
        "kd_hyperparameters": {"T": 4.0, "alpha": 0.7},
        "training_config": TRAIN_CONFIG,
        "teacher_config": RF_CONFIG,
        "teacher_calibration_strategy": (
            "stratified_group_kfold_with_zero_exact_feature_group_overlap"
        ),
        "bound_teacher_soft_target_provenance": None,
    }
    for key, value in expected.items():
        if execution.get(key) != value:
            raise RuntimeError(f"Execution contract mismatch for {key}")
    fingerprint_payload = dict(execution)
    observed_fingerprint = fingerprint_payload.pop("execution_fingerprint_sha256", None)
    if observed_fingerprint != canonical_json_sha256(fingerprint_payload):
        raise RuntimeError("Execution contract fingerprint is invalid")
    _assert_hash(execution.get("script_sha256"), "executed runner hash")
    _assert_hash(execution.get("common_module_sha256"), "executed common-module hash")
    current_common = (
        REPO_ROOT / "experiments" / "wsnds" / "leakage_free_rerun" / "tier15_common.py"
    )
    if execution["common_module_sha256"] != sha256_file(current_common):
        raise RuntimeError(
            "The reconstruction module differs from the manifest-bound executed module"
        )
    execution_file_hash = sha256_file(root / "execution_contract.json")
    if completion.get("execution_contract_sha256") != execution_file_hash:
        raise RuntimeError("Seed completion is not bound to the execution contract")


def _verify_preprocessing(
    root: Path,
    dataset_csv: Path,
    preprocessing: dict[str, Any],
) -> dict[str, Any]:
    dataset = load_wsnds(dataset_csv)
    if dataset["dataset_sha256"] != preprocessing.get("dataset_sha256"):
        raise RuntimeError("Dataset SHA-256 differs from the preprocessing contract")
    split = feature_group_split(dataset["features"], dataset["labels"])
    scaled, scaler = apply_train_scaler(split)
    observed_split_hashes = split_hashes(split)
    observed_indices_hash = sha256_arrays(
        split["train_indices"], split["validation_indices"], split["test_indices"]
    )
    observed_scaler_hash = sha256_arrays(
        np.asarray(scaler.mean_, dtype=np.float64),
        np.asarray(scaler.scale_, dtype=np.float64),
        np.asarray(scaler.var_, dtype=np.float64),
    )
    observed_transformed_hashes = {
        name: sha256_arrays(scaled[f"X_{name}"])
        for name in ["train", "validation", "test"]
    }
    observed_sizes = {
        name: int(len(split[f"{name}_indices"])) for name in EXPECTED_SPLIT_SIZES
    }
    expected_fields = {
        "split_policy": "seed42_stratified_feature_group_split",
        "split_sizes": EXPECTED_SPLIT_SIZES,
        "split_hashes": observed_split_hashes,
        "split_indices_sha256": observed_indices_hash,
        "scaler_fit_partition": "train only",
        "scaler_fit_row_count": EXPECTED_SPLIT_SIZES["train"],
        "scaler_sha256": observed_scaler_hash,
        "transformed_split_hashes": observed_transformed_hashes,
        "feature_overlap_audit": split["group_audit"],
    }
    if observed_sizes != EXPECTED_SPLIT_SIZES:
        raise RuntimeError(f"Recomputed split sizes differ: {observed_sizes}")
    for key, value in expected_fields.items():
        if preprocessing.get(key) != value:
            raise RuntimeError(f"Preprocessing contract mismatch for {key}")
    for overlap_key in [
        "train_validation_feature_overlap",
        "train_test_feature_overlap",
        "validation_test_feature_overlap",
    ]:
        if split["group_audit"].get(overlap_key) != 0:
            raise RuntimeError("Feature groups cross final split partitions")

    indices_path = root / str(preprocessing.get("split_indices_file"))
    scaler_path = root / str(preprocessing.get("scaler_parameters_file"))
    if sha256_file(indices_path) != preprocessing.get("split_indices_file_sha256"):
        raise RuntimeError("Split-index NPZ hash mismatch")
    if sha256_file(scaler_path) != preprocessing.get("scaler_parameters_file_sha256"):
        raise RuntimeError("Scaler NPZ hash mismatch")
    with np.load(indices_path, allow_pickle=False) as saved:
        for name in ["train", "validation", "test"]:
            if not np.array_equal(saved[f"{name}_indices"], split[f"{name}_indices"]):
                raise RuntimeError(f"Saved {name} indices differ from reconstruction")
    with np.load(scaler_path, allow_pickle=False) as saved:
        for name, expected in [
            ("mean", scaler.mean_),
            ("scale", scaler.scale_),
            ("var", scaler.var_),
        ]:
            if not np.array_equal(saved[name], np.asarray(expected, dtype=np.float64)):
                raise RuntimeError(f"Saved scaler {name} differs from reconstruction")
    return {"dataset": dataset, "split": split, "scaled": scaled, "scaler": scaler}


def _verify_teacher_provenance(
    seed_root: Path,
    seed_manifest: dict[str, Any],
    completion: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    provenance = completion.get("teacher_soft_target_provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError("RF-KD route lacks teacher soft-target provenance")
    calibration = provenance.get("calibration_audit")
    if (
        provenance.get("source_type") != "fresh_calibrated_rf_soft_targets"
        or provenance.get("rf_seed") != SEED
        or provenance.get("rf_config") != RF_CONFIG
        or not isinstance(calibration, dict)
        or calibration.get("strategy") != "stratified_group_kfold"
        or calibration.get("folds") != 3
        or calibration.get("group_overlap_per_fold") != [0, 0, 0]
    ):
        raise RuntimeError("RF-KD teacher provenance violates the final contract")
    item = next(
        (entry for entry in seed_manifest["files"] if entry.get("path") == "rf_train_probabilities.npy"),
        None,
    )
    if item is None:
        raise RuntimeError("RF probability file is absent from the seed manifest")
    path = seed_root / "rf_train_probabilities.npy"
    probabilities = np.load(path, allow_pickle=False)
    if probabilities.shape != (EXPECTED_SPLIT_SIZES["train"], len(CLASS_NAMES)):
        raise RuntimeError("RF probability array has the wrong shape")
    if not np.isfinite(probabilities).all():
        raise RuntimeError("RF probability array contains non-finite values")
    if sha256_file(path) != item.get("sha256"):
        raise RuntimeError("RF probability file hash differs from the seed manifest")
    if sha256_arrays(probabilities) != provenance.get("train_probability_content_sha256"):
        raise RuntimeError("RF probability content hash differs from teacher provenance")
    return provenance, path


def load_verified_context(
    confirmation_root: Path,
    dataset_csv: Path,
    student: str,
    route: str,
) -> dict[str, Any]:
    root = confirmation_root.resolve()
    expected_root = (REPO_ROOT / EXPECTED_RELATIVE_ROOT).resolve()
    if root != expected_root:
        raise RuntimeError(f"Confirmation root must be exactly {expected_root}")
    for relative, expected_sha256 in EXPECTED_FINAL_LINEAGE_FILES.items():
        source = root / relative
        if not source.is_file() or sha256_file(source) != expected_sha256:
            raise RuntimeError(
                f"Preserved final lineage trust anchor changed: {relative}"
            )
    seed_root = root / f"seed_{SEED}"
    root_manifest = verify_manifest(root)
    seed_manifest = verify_manifest(seed_root)
    preprocessing = read_json(root / "preprocessing_contract.json")
    execution = read_json(root / "execution_contract.json")
    completion = read_json(seed_root / "seed_completion.json")
    for label, document in [
        ("root manifest", root_manifest),
        ("seed manifest", seed_manifest),
        ("preprocessing contract", preprocessing),
        ("execution contract", execution),
        ("seed completion", completion),
    ]:
        if document.get("protocol_id") != PROTOCOL_ID:
            raise RuntimeError(f"{label} protocol is not {PROTOCOL_ID}")
    if completion.get("status") != "complete" or completion.get("seed") != SEED:
        raise RuntimeError("Seed-42 completion is not complete")
    for key in ["dataset_sha256", "split_indices_sha256", "scaler_sha256"]:
        if completion.get(key) != execution.get(key) or completion.get(key) != preprocessing.get(key):
            raise RuntimeError(f"Root and seed contracts disagree for {key}")
    verify_execution_contract(root, execution, completion)
    reconstructed = _verify_preprocessing(root, dataset_csv.resolve(), preprocessing)

    student_name = f"student_{student}"
    result_key = f"{student_name}_{route}"
    result = completion.get("student_results", {}).get(result_key)
    if not isinstance(result, dict) or result.get("route") != route:
        raise RuntimeError(f"Missing final result {result_key}")
    model_path = resolve_manifest_member(
        seed_root, seed_manifest, result.get("plain_state_dict"), result.get("plain_state_dict_sha256")
    )
    rich_path = resolve_manifest_member(
        seed_root, seed_manifest, result.get("rich_artifact"), result.get("rich_artifact_sha256")
    )
    predictions_path = resolve_manifest_member(
        seed_root, seed_manifest, result.get("test_predictions"), result.get("test_predictions_sha256")
    )
    plain_state = torch.load(model_path, map_location="cpu", weights_only=True)
    rich = torch.load(rich_path, map_location="cpu", weights_only=False)
    rich_state = rich.get("state_dict")
    if not isinstance(plain_state, dict) or not isinstance(rich_state, dict):
        raise RuntimeError("Checkpoint artifacts do not contain state dictionaries")
    if set(plain_state) != set(rich_state):
        raise RuntimeError("Plain and rich checkpoint keys differ")
    for key in sorted(plain_state):
        if not torch.is_tensor(plain_state[key]) or not torch.isfinite(plain_state[key]).all():
            raise RuntimeError(f"Invalid checkpoint tensor: {key}")
        if not torch.equal(plain_state[key], rich_state[key]):
            raise RuntimeError(f"Plain and rich checkpoint tensors differ: {key}")
    trained_state_sha256 = sha256_arrays(
        *[plain_state[key].detach().cpu().numpy() for key in sorted(plain_state)]
    )
    if trained_state_sha256 != result.get("trained_state_sha256"):
        raise RuntimeError("Checkpoint tensors differ from the completion state hash")
    if trained_state_sha256 != rich.get("trained_state_sha256"):
        raise RuntimeError("Checkpoint tensors differ from the rich-artifact state hash")

    teacher_provenance: dict[str, Any] | None = None
    teacher_probability_path: Path | None = None
    kd_hyperparameters: dict[str, float] | None = None
    if route == "rf_kd":
        teacher_provenance, teacher_probability_path = _verify_teacher_provenance(
            seed_root, seed_manifest, completion
        )
        kd_hyperparameters = {"T": 4.0, "alpha": 0.7}
    expected_rich = {
        "protocol_id": PROTOCOL_ID,
        "seed": SEED,
        "student": student_name,
        "route": route,
        "hidden_dims": list(STUDENT_SPECS[student_name]),
        "input_dim": 17,
        "num_classes": len(CLASS_NAMES),
        "feature_names": reconstructed["dataset"]["feature_names"],
        "class_names": CLASS_NAMES,
        "dataset_sha256": reconstructed["dataset"]["dataset_sha256"],
        "split_hashes": split_hashes(reconstructed["split"]),
        "scaler_sha256": preprocessing["scaler_sha256"],
        "kd_hyperparameters": kd_hyperparameters,
        "training_config": TRAIN_CONFIG,
        "feature_overlap_audit": reconstructed["split"]["group_audit"],
        "teacher_soft_target_provenance": teacher_provenance,
        "trained_state_sha256": trained_state_sha256,
        "initial_state_sha256": result.get("initial_state_sha256"),
    }
    for key, value in expected_rich.items():
        if rich.get(key) != value:
            raise RuntimeError(f"Rich-artifact contract mismatch for {key}")

    layers = extract_linear_layers(plain_state)
    observed_hidden = [int(layers[0][1].shape[0]), int(layers[1][1].shape[0])]
    if len(layers) != 3 or observed_hidden != list(STUDENT_SPECS[student_name]):
        raise RuntimeError("Checkpoint architecture differs from student identity")
    logits = forward_numpy(layers, reconstructed["scaled"]["X_test"])
    if not np.isfinite(logits).all():
        raise RuntimeError("Reproduced FP32 logits contain non-finite values")
    predictions = logits.argmax(axis=1).astype(np.int64)
    stable_logits = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(stable_logits)
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    saved = pd.read_csv(predictions_path)
    probability_columns = [
        f"probability_{index}_{name}" for index, name in enumerate(CLASS_NAMES)
    ]
    expected_columns = ["source_row_index", "true_label", "predicted_label", *probability_columns]
    if saved.columns.tolist() != expected_columns or len(saved) != EXPECTED_SPLIT_SIZES["test"]:
        raise RuntimeError("Saved prediction table violates the final schema or row count")
    if not np.array_equal(saved["source_row_index"].to_numpy(np.int64), reconstructed["split"]["test_indices"]):
        raise RuntimeError("Saved predictions reference different test rows")
    if not np.array_equal(saved["true_label"].to_numpy(np.int64), reconstructed["split"]["y_test"]):
        raise RuntimeError("Saved predictions contain different test labels")
    if not np.array_equal(saved["predicted_label"].to_numpy(np.int64), predictions):
        raise RuntimeError("Checkpoint prediction reproduction is not exact")
    saved_probabilities = saved[probability_columns].to_numpy(np.float64)
    if not np.allclose(
        saved_probabilities,
        probabilities,
        rtol=PREDICTION_PROBABILITY_TOLERANCE,
        atol=PREDICTION_PROBABILITY_TOLERANCE,
    ):
        raise RuntimeError("Checkpoint probability reproduction exceeds tolerance")
    reproduced_metrics = classification_metrics(reconstructed["split"]["y_test"], logits)
    _assert_nested_numeric_equal(reproduced_metrics, result.get("metrics"), "metrics")

    return {
        **reconstructed,
        "dataset_csv": dataset_csv.resolve(),
        "confirmation_root": root,
        "seed_root": seed_root,
        "root_manifest": root_manifest,
        "seed_manifest": seed_manifest,
        "preprocessing": preprocessing,
        "execution": execution,
        "completion": completion,
        "student": student,
        "student_name": student_name,
        "route": route,
        "result": result,
        "model_path": model_path,
        "rich_path": rich_path,
        "predictions_path": predictions_path,
        "plain_state": plain_state,
        "layers": layers,
        "fp32_logits": logits,
        "fp32_predictions": predictions,
        "fp32_metrics": reproduced_metrics,
        "trained_state_sha256": trained_state_sha256,
        "kd_hyperparameters": kd_hyperparameters,
        "teacher_soft_target_provenance": teacher_provenance,
        "teacher_probability_path": teacher_probability_path,
    }


def _prepare_output(path: Path) -> tuple[Path, Path]:
    final = path.resolve()
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite export path: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.parent / f".{final.name}.tmp.{os.getpid()}.{time.time_ns()}"
    staging.mkdir()
    return final, staging


def seal_output_sources(output_dir: Path) -> list[dict[str, Any]]:
    sealed: list[dict[str, Any]] = []
    for origin_relative, snapshot_relative in SOURCE_SNAPSHOT_SPECS:
        origin = REPO_ROOT / origin_relative
        snapshot = output_dir / snapshot_relative
        if not origin.is_file():
            raise FileNotFoundError(origin)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, snapshot)
        origin_hash = sha256_file(origin)
        snapshot_hash = sha256_file(snapshot)
        if origin_hash != snapshot_hash:
            raise RuntimeError(f"Source snapshot differs after copy: {origin_relative}")
        sealed.append({
            "origin_relative_path": origin_relative,
            "snapshot_path": snapshot_relative,
            "size_bytes": snapshot.stat().st_size,
            "sha256": snapshot_hash,
        })
    return sealed


def _raw_int32_saturation_count(raw_features: np.ndarray, raw_q_frac: int) -> int:
    unbounded = np.rint(
        np.asarray(raw_features, dtype=np.float64) * float(1 << int(raw_q_frac))
    )
    limits = np.iinfo(np.int32)
    return int(np.count_nonzero((unbounded < limits.min) | (unbounded > limits.max)))


def _quantized_layers_for_fracs(
    layers: list[tuple[str, Any, Any]],
    input_frac: int,
    output_fracs: tuple[int, int, int],
) -> list[dict[str, Any]]:
    quantized: list[dict[str, Any]] = []
    current_input_frac = input_frac
    for (_, weight, bias), output_frac in zip(layers, output_fracs):
        quantized.append(
            quantize_layer(
                weight,
                bias,
                input_frac=current_input_frac,
                output_frac=output_frac,
            )
        )
        current_input_frac = output_frac
    return quantized


def select_quantization_policy(
    *,
    layers: list[tuple[str, Any, Any]],
    preprocessing_metadata: dict[str, Any],
    x_train: np.ndarray,
    x_train_raw: np.ndarray,
    x_validation: np.ndarray,
    x_validation_raw: np.ndarray,
    y_validation: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Select scales without accepting test features, labels, or metrics."""
    baseline_layers, calibration = calibrate_quantized_layers(layers, x_train)
    baseline_fracs = (
        int(baseline_layers[0]["input_frac"]),
        *(int(layer["output_frac"]) for layer in baseline_layers),
    )
    input_options = sorted({max(0, baseline_fracs[0] - 1), baseline_fracs[0]})
    output_options = [
        sorted({max(0, output_frac - 1), output_frac})
        for output_frac in baseline_fracs[1:]
    ]
    fp32_train_predictions = forward_numpy(layers, x_train).argmax(axis=1)
    fp32_validation_logits = forward_numpy(layers, x_validation)
    fp32_validation_predictions = fp32_validation_logits.argmax(axis=1)
    fp32_validation_macro_f1 = classification_metrics(
        y_validation, fp32_validation_logits
    )["macro_f1"]

    candidates: list[dict[str, Any]] = []
    candidate_layers: dict[tuple[int, int, int, int], list[dict[str, Any]]] = {}
    for values in itertools.product(input_options, *output_options):
        fracs = tuple(int(value) for value in values)
        quantized = _quantized_layers_for_fracs(layers, fracs[0], fracs[1:])
        candidate_layers[fracs] = quantized
        candidate: dict[str, Any] = {
            "input_frac": fracs[0],
            "output_fracs": list(fracs[1:]),
            "is_legacy_range_baseline": fracs == baseline_fracs,
            "validation_gate_passed": False,
            "training_safety_passed": False,
            "eligible_for_selection": False,
        }
        try:
            accumulator_audit = accumulator_bounds(quantized)
            integer_metadata = build_integer_preprocessing_metadata(
                preprocessing_metadata, output_q_frac=fracs[0]
            )
            preprocessing_bounds = preprocess_multiply_bounds(integer_metadata)
            raw_validation_saturation = _raw_int32_saturation_count(
                x_validation_raw, integer_metadata["raw_q_frac"]
            )
            if raw_validation_saturation != 0:
                raise RuntimeError("validation raw-input saturation")
            validation_raw_q = quantize_raw_features_q(
                x_validation_raw, integer_metadata["raw_q_frac"]
            )
            validation_audit, _, validation_logits, validation_predictions = saturation_audit(
                layers, quantized, validation_raw_q, integer_metadata
            )
            _, standardized_validation = quantize_standardized_q15(
                x_validation, input_frac=fracs[0]
            )
            if standardized_validation["saturation_count"] != 0:
                raise RuntimeError("validation standardized-input saturation")
            validation_macro_f1 = classification_metrics(
                y_validation, validation_logits
            )["macro_f1"]
            validation_agreement = float(
                np.mean(validation_predictions == fp32_validation_predictions)
            )
            validation_drop = abs(
                float(fp32_validation_macro_f1) - float(validation_macro_f1)
            )
            validation_gate_passed = (
                validation_agreement >= MINIMUM_FIXED_FP32_AGREEMENT
                and validation_drop <= MAXIMUM_ABSOLUTE_MACRO_F1_DROP
            )
            candidate.update({
                "validation_fixed_vs_fp32_agreement": validation_agreement,
                "validation_fp32_macro_f1": fp32_validation_macro_f1,
                "validation_fixed_macro_f1": validation_macro_f1,
                "validation_absolute_macro_f1_drop": validation_drop,
                "validation_gate_passed": validation_gate_passed,
                "validation_saturation": validation_audit,
                "accumulator_bounds": accumulator_audit,
                "preprocess_multiply_bounds": preprocessing_bounds,
            })
            if validation_gate_passed or fracs == baseline_fracs:
                raw_train_saturation = _raw_int32_saturation_count(
                    x_train_raw, integer_metadata["raw_q_frac"]
                )
                if raw_train_saturation != 0:
                    raise RuntimeError("training raw-input saturation")
                train_raw_q = quantize_raw_features_q(
                    x_train_raw, integer_metadata["raw_q_frac"]
                )
                train_audit, _, _, train_predictions = saturation_audit(
                    layers, quantized, train_raw_q, integer_metadata
                )
                _, standardized_train = quantize_standardized_q15(
                    x_train, input_frac=fracs[0]
                )
                if standardized_train["saturation_count"] != 0:
                    raise RuntimeError("training standardized-input saturation")
                candidate.update({
                    "training_fixed_vs_fp32_agreement": float(
                        np.mean(train_predictions == fp32_train_predictions)
                    ),
                    "training_saturation": train_audit,
                    "training_safety_passed": True,
                    "eligible_for_selection": validation_gate_passed,
                })
        except RuntimeError as exc:
            candidate["rejection_reason"] = str(exc)
        candidates.append(candidate)

    eligible = [item for item in candidates if item["eligible_for_selection"]]
    if eligible:
        selected = max(
            eligible,
            key=lambda item: (
                item["validation_fixed_vs_fp32_agreement"],
                -item["validation_absolute_macro_f1_drop"],
                item["training_fixed_vs_fp32_agreement"],
                sum(item["output_fracs"]) + item["input_frac"],
                tuple(item["output_fracs"]),
            ),
        )
        selection_status = "selected_from_validation_gate_passing_candidates"
    else:
        selected = next(
            item for item in candidates if item["is_legacy_range_baseline"]
        )
        if not selected["training_safety_passed"]:
            raise RuntimeError("Legacy baseline is unsafe on the training partition")
        selection_status = "no_candidate_met_validation_gates_baseline_frozen"
    selected_fracs = (
        int(selected["input_frac"]),
        *(int(value) for value in selected["output_fracs"]),
    )
    policy_audit = {
        "policy_id": "train_range_adjacent_frac_validation_v1",
        "development_status": (
            "post-hoc deployment-method development after historical baseline "
            "fixed-point test-gate observations; not preregistered"
        ),
        "applied_uniformly_to_all_student_route_exports": True,
        "selection_status": selection_status,
        "selection_uses_test_data": False,
        "historical_test_observations_preceded_policy_development": True,
        "selection_inputs": [
            "training features for range calibration and safety",
            "validation features and labels for frozen fidelity gates",
        ],
        "frozen_thresholds": {
            "minimum_fixed_vs_fp32_agreement": MINIMUM_FIXED_FP32_AGREEMENT,
            "maximum_absolute_macro_f1_drop": MAXIMUM_ABSOLUTE_MACRO_F1_DROP,
        },
        "baseline_fracs": {
            "input_frac": baseline_fracs[0],
            "output_fracs": list(baseline_fracs[1:]),
        },
        "selected_fracs": {
            "input_frac": selected_fracs[0],
            "output_fracs": list(selected_fracs[1:]),
        },
        "candidate_count": len(candidates),
        "validation_gate_passing_candidate_count": len(eligible),
        "candidates": candidates,
    }
    calibration["quantization_policy"] = policy_audit
    calibration["input_frac"] = selected_fracs[0]
    for index, layer in enumerate(candidate_layers[selected_fracs]):
        calibration["layers"][index].update({
            "input_frac": int(layer["input_frac"]),
            "weight_frac": int(layer["weight_frac"]),
            "accum_frac": int(layer["accum_frac"]),
            "output_frac": int(layer["output_frac"]),
            "output_shift": int(layer["output_shift"]),
        })
    return candidate_layers[selected_fracs], calibration, policy_audit


def _normalize_headers(output_dir: Path, context: dict[str, Any]) -> None:
    test_vectors = output_dir / "test_vectors.h"
    text = test_vectors.read_text(encoding="ascii").replace(
        "CUKD_WSNDS_STUDENT_A_RFKD_TEST_VECTORS_H",
        "CUKD_WSNDS_FINAL_TEST_VECTORS_H",
    )
    test_vectors.write_text(text, encoding="ascii")
    model_weights = output_dir / "model_weights.h"
    text = model_weights.read_text(encoding="ascii").replace(
        "Generated by deployment/firmware_export/wsnds_rfkd_hil/"
        "export_wsnds_student_a_rfkd_int8.py.",
        f"Generated from final lineage: {context['student_name']} {context['route']} seed 42.",
    )
    model_weights.write_text(text, encoding="ascii")


def _run(command: list[str], *, cwd: Path = REPO_ROOT) -> dict[str, Any]:
    completed = subprocess.run(
        command, cwd=cwd, text=True, capture_output=True, check=False
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _run_native_host_self_test(export_root: Path, cc: str) -> dict[str, Any]:
    """Compile sealed sources for this host and remove the native executable."""
    root = export_root.resolve()
    sealed_c_dir = root / "source_snapshot" / "c"
    source_names = ["cukd_preprocess.c", "cukd_model.c", "wsnds_train_only_self_test.c"]
    for name in source_names:
        if not (sealed_c_dir / name).is_file():
            raise RuntimeError(f"Sealed host self-test source is missing: {name}")
    compiler_version = _run([cc, "--version"], cwd=root)
    if compiler_version["returncode"] != 0:
        raise RuntimeError(
            f"Host C compiler is unavailable: {compiler_version['stderr']}"
        )
    with tempfile.TemporaryDirectory(prefix="cukd_final_verify_") as temporary:
        executable = Path(temporary) / "cukd_final_self_test"
        if os.name == "nt":
            executable = executable.with_suffix(".exe")
        compile_command = [
            cc, "-std=c99", "-Wall", "-Wextra", "-Werror", "-O2",
            "-I", str(root), "-I", str(sealed_c_dir),
            *(str(sealed_c_dir / name) for name in source_names),
            "-o", str(executable),
        ]
        compile_result = _run(compile_command, cwd=root)
        if compile_result["returncode"] != 0:
            raise RuntimeError(
                f"Host C compilation failed: {compile_result['stderr']}"
            )
        executable_sha256 = sha256_file(executable)
        self_test_result = _run([str(executable)], cwd=root)
        if self_test_result["returncode"] != 0:
            raise RuntimeError(
                "Host C/Python fixed-reference equivalence failed with code "
                f"{self_test_result['returncode']}"
            )
    return {
        "status": "passed",
        "rows": EXPECTED_SPLIT_SIZES["test"],
        "preprocessed_inputs_exact": True,
        "fixed_logits_exact": True,
        "fixed_predictions_exact": True,
        "compiler_requested": cc,
        "compiler_version": compiler_version,
        "compile": compile_result,
        "self_test": self_test_result,
        "temporary_executable_sha256": executable_sha256,
        "temporary_executable_retained": False,
        "verification_contract": (
            "Compile and run a verifier-native temporary executable from the sealed "
            "C sources and immutable generated headers; never consume a bundled binary."
        ),
    }


def _identity_payload(
    context: dict[str, Any],
    core_files: list[dict[str, Any]],
    source_snapshots: list[dict[str, Any]],
    quantization_policy: dict[str, Any],
) -> dict[str, Any]:
    route_provenance = {
        "kd_hyperparameters": context["kd_hyperparameters"],
        "teacher_soft_target_provenance": context["teacher_soft_target_provenance"],
        "teacher_probability_file_sha256": (
            sha256_file(context["teacher_probability_path"])
            if context["teacher_probability_path"] is not None
            else None
        ),
    }
    identity = {
        "protocol": PROTOCOL_ID,
        "seed": SEED,
        "student": context["student"],
        "route": context["route"],
        "checkpoint_file_sha256": sha256_file(context["model_path"]),
        "trained_state_sha256": context["trained_state_sha256"],
        "dataset_sha256": context["dataset"]["dataset_sha256"],
        "split_indices_sha256": context["preprocessing"]["split_indices_sha256"],
        "test_source_indices_sha256": sha256_arrays(
            np.asarray(context["split"]["test_indices"], dtype=np.int64)
        ),
        "scaler_sha256": context["preprocessing"]["scaler_sha256"],
        "route_provenance": route_provenance,
        "quantization_policy_sha256": canonical_json_sha256(quantization_policy),
        "core_files": core_files,
        "source_snapshots": source_snapshots,
    }
    identity["export_id"] = canonical_json_sha256(identity)
    return identity


def _final_preprocessing_metadata(context: dict[str, Any]) -> dict[str, Any]:
    metadata = build_preprocessing_metadata(
        target_col=context["dataset"]["target_column"],
        feature_names=context["dataset"]["feature_names"],
        class_names=context["dataset"]["class_names"],
        scaler_mean=context["scaler"].mean_,
        scaler_scale=context["scaler"].scale_,
        split_sizes={"train": 262197, "val": 56163, "test": 56301},
    )
    metadata.update({
        "protocol_id": PROTOCOL_ID,
        "dataset_sha256": context["dataset"]["dataset_sha256"],
        "split_hashes": context["preprocessing"]["split_hashes"],
        "split_indices_sha256": context["preprocessing"]["split_indices_sha256"],
        "scaler_sha256": context["preprocessing"]["scaler_sha256"],
        "scaler_fit_partition": "train only",
        "preprocessing_contract": (
            "Feature-group-disjoint split over all 17 raw features, followed by "
            "StandardScaler fit on training rows only."
        ),
    })
    return metadata


def _identity_header_text(identity: dict[str, Any]) -> str:
    return (
        "#ifndef CUKD_FINAL_EXPORT_IDENTITY_H\n"
        "#define CUKD_FINAL_EXPORT_IDENTITY_H\n"
        f"#define CUKD_EXPORT_ID \"{identity['export_id']}\"\n"
        f"#define CUKD_PROTOCOL_ID \"{identity['protocol']}\"\n"
        f"#define CUKD_EXPORT_SEED {identity['seed']}\n"
        f"#define CUKD_STUDENT_ID \"{identity['student']}\"\n"
        f"#define CUKD_ROUTE_ID \"{identity['route']}\"\n"
        f"#define CUKD_CHECKPOINT_FILE_SHA256 \"{identity['checkpoint_file_sha256']}\"\n"
        f"#define CUKD_TRAINED_STATE_SHA256 \"{identity['trained_state_sha256']}\"\n"
        "#endif\n"
    )


def _write_identity_header(path: Path, identity: dict[str, Any]) -> None:
    path.write_text(_identity_header_text(identity), encoding="ascii")


def enforce_quality_gates(agreement: float, absolute_macro_f1_drop: float) -> None:
    if not np.isfinite(agreement) or not np.isfinite(absolute_macro_f1_drop):
        raise RuntimeError("Fixed-point quality gate values must be finite")
    failures: list[str] = []
    if agreement < MINIMUM_FIXED_FP32_AGREEMENT:
        failures.append(f"fixed/FP32 agreement {agreement:.6f} is below 0.99")
    if absolute_macro_f1_drop > MAXIMUM_ABSOLUTE_MACRO_F1_DROP:
        failures.append(
            f"absolute macro-F1 drop {absolute_macro_f1_drop:.6f} exceeds 0.015"
        )
    if failures:
        raise RuntimeError("Fixed-point quality gates failed: " + "; ".join(failures))


def export(context: dict[str, Any], output_dir: Path, cc: str) -> Path:
    require_output_outside_inputs(
        output_dir,
        [context["confirmation_root"], context["dataset_csv"].parent],
    )
    final_dir, staging = _prepare_output(output_dir)
    published = False
    try:
        layers = context["layers"]
        split = context["split"]
        scaled = context["scaled"]
        source_snapshots = seal_output_sources(staging)
        metadata = _final_preprocessing_metadata(context)
        quantized_layers, calibration, quantization_policy = select_quantization_policy(
            layers=layers,
            preprocessing_metadata=metadata,
            x_train=scaled["X_train"],
            x_train_raw=split["X_train_raw"],
            x_validation=scaled["X_validation"],
            x_validation_raw=split["X_validation_raw"],
            y_validation=split["y_validation"],
        )
        calibration["calibration_source"] = "final FGDS seed-42 training partition only"
        calibration["protocol_id"] = PROTOCOL_ID
        write_header(staging / "model_weights.h", quantized_layers, context["model_path"].name)
        dataset_for_export = {
            "metadata": metadata,
            "x_calibration": scaled["X_train"],
            "y_calibration": split["y_train"],
            "x_test": scaled["X_test"],
            "x_test_raw": split["X_test_raw"],
            "y_test": split["y_test"],
            "x_train_shape": list(scaled["X_train"].shape),
            "x_val_shape": list(scaled["X_validation"].shape),
            "x_test_shape": list(scaled["X_test"].shape),
        }
        generated = generate_e2e_artifacts(
            output_dir=staging,
            layers=layers,
            quantized_layers=quantized_layers,
            dataset=dataset_for_export,
            dataset_csv=context["dataset_csv"],
            calibration_summary=calibration,
            num_test_vectors=EXPECTED_SPLIT_SIZES["test"],
            test_vector_seed=SEED,
        )
        _normalize_headers(staging, context)
        integer_metadata = generated["integer_preprocess_metadata"]
        accumulator_audit = accumulator_bounds(quantized_layers)
        preprocessing_bounds = preprocess_multiply_bounds(integer_metadata)
        raw_scale = float(1 << int(integer_metadata["raw_q_frac"]))
        raw_unbounded = np.rint(np.asarray(split["X_test_raw"], dtype=np.float64) * raw_scale)
        int32 = np.iinfo(np.int32)
        raw_input_saturation = int(np.count_nonzero(
            (raw_unbounded < int32.min) | (raw_unbounded > int32.max)
        ))
        if raw_input_saturation:
            raise RuntimeError(f"Raw input saturation count is {raw_input_saturation}")
        raw_inputs_q = quantize_raw_features_q(split["X_test_raw"], integer_metadata["raw_q_frac"])
        preprocessed_q = simulate_integer_preprocess_q(raw_inputs_q, integer_metadata)
        fixed_logits, fixed_predictions = simulate_fixed_point_inference(
            quantized_layers, preprocessed_q
        )
        saturation, audited_q, audited_logits, audited_predictions = saturation_audit(
            layers, quantized_layers, raw_inputs_q, integer_metadata
        )
        test_saturation = dict(saturation)
        test_saturation["partition"] = "test partition"
        test_saturation["raw_input_saturation_count"] = raw_input_saturation
        if not np.array_equal(audited_q, preprocessed_q):
            raise RuntimeError("Python preprocessing references differ")
        if not np.array_equal(audited_logits, fixed_logits):
            raise RuntimeError("Python fixed-logit references differ")
        if not np.array_equal(audited_predictions, fixed_predictions):
            raise RuntimeError("Python fixed-prediction references differ")
        training_saturation = calibration_partition_saturation_audit(
            layers, quantized_layers, split["X_train_raw"], integer_metadata
        )
        validation_saturation = calibration_partition_saturation_audit(
            layers, quantized_layers, split["X_validation_raw"], integer_metadata
        )
        validation_saturation["partition"] = "validation partition"
        standardized_input_saturation: dict[str, int] = {}
        for partition in ["train", "validation", "test"]:
            _, direct_stats = quantize_standardized_q15(
                scaled[f"X_{partition}"],
                input_frac=int(quantized_layers[0]["input_frac"]),
            )
            standardized_input_saturation[partition] = int(
                direct_stats["saturation_count"]
            )
        if any(standardized_input_saturation.values()):
            raise RuntimeError(
                "Standardized input quantization saturates: "
                f"{standardized_input_saturation}"
            )
        bind_replay_source_rows(
            staging / "hil_replay_vectors.csv", split["test_indices"], raw_inputs_q
        )
        write_reference_with_logits(
            staging / "hil_reference_predictions.csv",
            split["test_indices"],
            split["y_test"],
            context["fp32_predictions"],
            fixed_predictions,
            fixed_logits,
        )
        fixed_metrics = classification_metrics(split["y_test"], fixed_logits)
        agreement = float(np.mean(context["fp32_predictions"] == fixed_predictions))
        absolute_macro_f1_drop = abs(
            float(context["fp32_metrics"]["macro_f1"]) - float(fixed_metrics["macro_f1"])
        )
        blocked_identity = {
            "protocol": PROTOCOL_ID,
            "seed": SEED,
            "student": context["student"],
            "route": context["route"],
            "dataset_sha256": context["dataset"]["dataset_sha256"],
            "split_indices_sha256": context["preprocessing"]["split_indices_sha256"],
            "test_source_indices_sha256": sha256_arrays(
                np.asarray(split["test_indices"], dtype=np.int64)
            ),
            "scaler_sha256": context["preprocessing"]["scaler_sha256"],
            "checkpoint_file_sha256": sha256_file(context["model_path"]),
            "trained_state_sha256": context["trained_state_sha256"],
            "rich_artifact_sha256": sha256_file(context["rich_path"]),
            "predictions_sha256": sha256_file(context["predictions_path"]),
            "route_provenance": {
                "kd_hyperparameters": context["kd_hyperparameters"],
                "teacher_soft_target_provenance": context[
                    "teacher_soft_target_provenance"
                ],
                "teacher_probability_file_sha256": (
                    sha256_file(context["teacher_probability_path"])
                    if context["teacher_probability_path"] is not None
                    else None
                ),
            },
            "quantization_policy_sha256": canonical_json_sha256(quantization_policy),
            "source_snapshots": source_snapshots,
        }
        blocked_identity["blocked_audit_id"] = canonical_json_sha256(blocked_identity)
        test_gate_audit = {
            "status": "passed",
            "identity": blocked_identity,
            "quantization_policy": quantization_policy,
            "frozen_test_quality_gate_assessments_this_export_invocation": 1,
            "selected_policy_python_fixed_test_forward_computations_before_gate": 3,
            "claims_first_ever_test_evaluation": False,
            "historical_test_observations_preceded_this_invocation": True,
            "test_rows": EXPECTED_SPLIT_SIZES["test"],
            "fixed_vs_fp32_agreement": agreement,
            "minimum_fixed_vs_fp32_agreement": MINIMUM_FIXED_FP32_AGREEMENT,
            "fp32_macro_f1": context["fp32_metrics"]["macro_f1"],
            "fixed_macro_f1": fixed_metrics["macro_f1"],
            "absolute_macro_f1_drop": absolute_macro_f1_drop,
            "maximum_absolute_macro_f1_drop": MAXIMUM_ABSOLUTE_MACRO_F1_DROP,
            "quality_gates_passed": False,
            "zero_saturation_passed": True,
            "accumulator_bounds_passed": True,
            "preprocess_bounds_passed": True,
            "saturation_ledgers": {
                "training": training_saturation,
                "validation": validation_saturation,
                "test": test_saturation,
                "standardized_inputs": standardized_input_saturation,
            },
            "accumulator_bounds": accumulator_audit,
            "preprocess_multiply_bounds": preprocessing_bounds,
        }
        try:
            enforce_quality_gates(agreement, absolute_macro_f1_drop)
        except RuntimeError as exc:
            test_gate_audit["status"] = "blocked"
            test_gate_audit["reason"] = str(exc)
            raise FinalQualityGateError(str(exc), test_gate_audit) from exc

        core_inventory = [
            {
                "path": name,
                "size_bytes": (staging / name).stat().st_size,
                "sha256": sha256_file(staging / name),
            }
            for name in CORE_EXPORT_FILES
        ]
        identity = _identity_payload(
            context, core_inventory, source_snapshots, quantization_policy
        )
        _write_identity_header(staging / "cukd_export_identity.h", identity)
        (staging / "final_export_identity.json").write_text(
            json.dumps(identity, indent=2), encoding="utf-8"
        )

        host_equivalence = _run_native_host_self_test(staging, cc)
        report = {
            "status": "passed",
            "identity": identity,
            "route_provenance": identity["route_provenance"],
            "quantization_policy": quantization_policy,
            "test_evaluation_scope": {
                "frozen_test_quality_gate_assessments_this_export_invocation": 1,
                "selected_policy_python_fixed_test_forward_computations_before_gate": 3,
                "host_c_full_test_executions_recorded_before_report_seal": 1,
                "claims_first_ever_test_evaluation": False,
                "historical_test_observations_preceded_this_invocation": True,
            },
            "source_artifacts": {
                "root_manifest_sha256": sha256_file(context["confirmation_root"] / "artifact_manifest.json"),
                "seed_manifest_sha256": sha256_file(context["seed_root"] / "artifact_manifest.json"),
                "execution_contract_sha256": sha256_file(context["confirmation_root"] / "execution_contract.json"),
                "preprocessing_contract_sha256": sha256_file(context["confirmation_root"] / "preprocessing_contract.json"),
                "seed_completion_sha256": sha256_file(context["seed_root"] / "seed_completion.json"),
                "checkpoint_file_sha256": identity["checkpoint_file_sha256"],
                "rich_artifact_sha256": sha256_file(context["rich_path"]),
                "predictions_sha256": sha256_file(context["predictions_path"]),
                "executed_runner_sha256_recorded": context["execution"]["script_sha256"],
                "executed_common_module_sha256_recorded": context["execution"]["common_module_sha256"],
                "exporter_sha256": sha256_file(SCRIPT_PATH),
                "legacy_numeric_exporter_sha256": sha256_file(LEGACY_NUMERIC_EXPORTER),
                "host_self_test_source_sha256": sha256_file(HOST_SELF_TEST_SOURCE),
                "source_snapshots": source_snapshots,
            },
            "gates": {
                "test_rows": EXPECTED_SPLIT_SIZES["test"],
                "checkpoint_predictions_exact": True,
                "checkpoint_probabilities_reproduced_within_absolute_tolerance": PREDICTION_PROBABILITY_TOLERANCE,
                "raw_input_saturation_count": raw_input_saturation,
                "standardized_input_saturation_count": standardized_input_saturation,
                "weight_saturation_count": saturation["weight_saturation_count"],
                "bias_saturation_count": saturation["bias_saturation_count"],
                "activation_saturation_count": saturation["activation_saturation_count"],
                "integer_preprocess_saturation_count": saturation["integer_preprocess_saturation_count"],
                "training_partition_saturation": training_saturation,
                "validation_partition_saturation": validation_saturation,
                "test_partition_saturation": test_saturation,
                "accumulator_bounds": accumulator_audit,
                "preprocess_multiply_bounds": preprocessing_bounds,
                "fixed_vs_fp32_agreement": agreement,
                "minimum_fixed_vs_fp32_agreement": MINIMUM_FIXED_FP32_AGREEMENT,
                "fp32_macro_f1": context["fp32_metrics"]["macro_f1"],
                "fixed_macro_f1": fixed_metrics["macro_f1"],
                "absolute_macro_f1_drop": absolute_macro_f1_drop,
                "maximum_absolute_macro_f1_drop": MAXIMUM_ABSOLUTE_MACRO_F1_DROP,
                "quality_gates_passed": True,
                "zero_saturation_passed": True,
                "accumulator_bounds_passed": True,
                "preprocess_bounds_passed": True,
                "host_equivalence_passed": True,
                "dense_row_ids_passed": True,
                "source_row_ids_complete_unique_passed": True,
            },
            "fp32_metrics": context["fp32_metrics"],
            "fixed_metrics": fixed_metrics,
            "fixed_point_calibration": calibration,
            "host_equivalence": host_equivalence,
            "claim_boundary": (
                "Firmware-level replay of 56,301 already extracted WSN-DS records "
                "from one preserved final seed-42 checkpoint. Timing is board compute "
                "only when reported by the firmware; transport time, feature extraction, "
                "energy, live capture, and multi-seed hardware replication are excluded."
            ),
        }
        report["report_payload_sha256"] = canonical_json_sha256(report)
        (staging / "final_export_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        inventory = file_inventory(staging, {"final_export_manifest.json"})
        manifest = {
            "status": "passed",
            "protocol_id": PROTOCOL_ID,
            "seed": SEED,
            "student": context["student"],
            "route": context["route"],
            "export_id": identity["export_id"],
            "identity_canonical_sha256": canonical_json_sha256(identity),
            "report_canonical_sha256": canonical_json_sha256(report),
            "file_count_excluding_manifest": len(inventory),
            "files": inventory,
        }
        manifest["manifest_payload_sha256"] = canonical_json_sha256(manifest)
        (staging / "final_export_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        verify_final_export(staging, cc=cc)
        if final_dir.exists():
            raise FileExistsError(f"Export path appeared during generation: {final_dir}")
        os.replace(staging, final_dir)
        published = True
        verify_final_export(final_dir, cc=cc)
        return final_dir / "final_export_manifest.json"
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if published:
            shutil.rmtree(final_dir, ignore_errors=True)
        raise


def _macro_f1_from_predictions(
    labels: np.ndarray, predictions: np.ndarray, class_count: int = 5
) -> float:
    return float(_metrics_from_predictions(labels, predictions, class_count)["macro_f1"])


def _metrics_from_predictions(
    labels: np.ndarray, predictions: np.ndarray, class_count: int = 5
) -> dict[str, Any]:
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    precision: list[float] = []
    recall: list[float] = []
    f1: list[float] = []
    for class_index in range(class_count):
        true_positive = int(matrix[class_index, class_index])
        predicted_count = int(matrix[:, class_index].sum())
        support = int(matrix[class_index, :].sum())
        class_precision = 0.0 if predicted_count == 0 else true_positive / predicted_count
        class_recall = 0.0 if support == 0 else true_positive / support
        denominator = class_precision + class_recall
        precision.append(float(class_precision))
        recall.append(float(class_recall))
        f1.append(0.0 if denominator == 0.0 else 2.0 * class_precision * class_recall / denominator)
    return {
        "accuracy": float(np.trace(matrix) / len(labels)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
        "per_class_support": matrix.sum(axis=1).astype(np.int64).tolist(),
        "confusion_matrix": matrix.tolist(),
    }


def _verify_zero_saturation_partition(audit: Any, label: str) -> None:
    if not isinstance(audit, dict):
        raise RuntimeError(f"{label} saturation audit is missing")
    for key in [
        "raw_input_saturation_count",
        "integer_preprocess_saturation_count",
        "activation_saturation_count",
    ]:
        if audit.get(key) != 0:
            raise RuntimeError(f"{label} {key} is not zero")
    layers = audit.get("activation_layers")
    if not isinstance(layers, list) or len(layers) != 3:
        raise RuntimeError(f"{label} activation-layer audit is incomplete")
    if any(item.get("activation_saturation_count") != 0 for item in layers):
        raise RuntimeError(f"{label} contains a saturating activation layer")
    parameters = audit.get("parameter_layers")
    if not isinstance(parameters, list) or len(parameters) != 3:
        raise RuntimeError(f"{label} parameter saturation audit is incomplete")
    if any(
        item.get("weight_saturation_count") != 0
        or item.get("bias_saturation_count") != 0
        for item in parameters
    ):
        raise RuntimeError(f"{label} contains saturated weights or biases")


def _verify_final_lineage_binding(
    *,
    root: Path,
    identity: dict[str, Any],
    report: dict[str, Any],
    replay_sources: np.ndarray,
    replay_features: np.ndarray,
    labels: np.ndarray,
    fixed_predictions: np.ndarray,
    fp32_predictions: np.ndarray,
    fixed_logits: np.ndarray,
) -> dict[str, Any]:
    """Reconstruct exported behavior from the immutable final training lineage."""

    context = load_verified_context(
        REPO_ROOT / EXPECTED_RELATIVE_ROOT,
        REPO_ROOT / "data" / "wsnds" / "WSN-DS.csv",
        str(identity["student"]),
        str(identity["route"]),
    )
    expected_route_provenance = {
        "kd_hyperparameters": context["kd_hyperparameters"],
        "teacher_soft_target_provenance": context["teacher_soft_target_provenance"],
        "teacher_probability_file_sha256": (
            sha256_file(context["teacher_probability_path"])
            if context["teacher_probability_path"] is not None
            else None
        ),
    }
    expected_identity = {
        "checkpoint_file_sha256": sha256_file(context["model_path"]),
        "trained_state_sha256": context["trained_state_sha256"],
        "dataset_sha256": context["dataset"]["dataset_sha256"],
        "split_indices_sha256": context["preprocessing"]["split_indices_sha256"],
        "test_source_indices_sha256": sha256_arrays(
            np.asarray(context["split"]["test_indices"], dtype=np.int64)
        ),
        "scaler_sha256": context["preprocessing"]["scaler_sha256"],
        "route_provenance": expected_route_provenance,
    }
    for field, expected in expected_identity.items():
        if identity.get(field) != expected:
            raise RuntimeError(f"Final export differs from trusted lineage: {field}")

    expected_source_artifacts = {
        "root_manifest_sha256": sha256_file(
            context["confirmation_root"] / "artifact_manifest.json"
        ),
        "seed_manifest_sha256": sha256_file(
            context["seed_root"] / "artifact_manifest.json"
        ),
        "execution_contract_sha256": sha256_file(
            context["confirmation_root"] / "execution_contract.json"
        ),
        "preprocessing_contract_sha256": sha256_file(
            context["confirmation_root"] / "preprocessing_contract.json"
        ),
        "seed_completion_sha256": sha256_file(
            context["seed_root"] / "seed_completion.json"
        ),
        "checkpoint_file_sha256": sha256_file(context["model_path"]),
        "rich_artifact_sha256": sha256_file(context["rich_path"]),
        "predictions_sha256": sha256_file(context["predictions_path"]),
        "executed_runner_sha256_recorded": context["execution"]["script_sha256"],
        "executed_common_module_sha256_recorded": context["execution"][
            "common_module_sha256"
        ],
    }
    source_artifacts = report.get("source_artifacts", {})
    for field, expected in expected_source_artifacts.items():
        if source_artifacts.get(field) != expected:
            raise RuntimeError(
                f"Final report differs from trusted source artifact: {field}"
            )

    for item in identity["source_snapshots"]:
        origin = REPO_ROOT / item["origin_relative_path"]
        snapshot = root / item["snapshot_path"]
        if (
            not origin.is_file()
            or sha256_file(origin) != item["sha256"]
            or snapshot.read_bytes() != origin.read_bytes()
        ):
            raise RuntimeError(
                f"Sealed source differs from its trusted origin: "
                f"{item['origin_relative_path']}"
            )

    metadata = _final_preprocessing_metadata(context)
    quantized_layers, _, quantization_policy = select_quantization_policy(
        layers=context["layers"],
        preprocessing_metadata=metadata,
        x_train=context["scaled"]["X_train"],
        x_train_raw=context["split"]["X_train_raw"],
        x_validation=context["scaled"]["X_validation"],
        x_validation_raw=context["split"]["X_validation_raw"],
        y_validation=context["split"]["y_validation"],
    )
    if report.get("quantization_policy") != quantization_policy:
        raise RuntimeError("Final quantization policy differs from trusted reconstruction")
    if identity.get("quantization_policy_sha256") != canonical_json_sha256(
        quantization_policy
    ):
        raise RuntimeError("Final quantization-policy identity differs from reconstruction")

    integer_metadata = build_integer_preprocessing_metadata(
        metadata, output_q_frac=int(quantized_layers[0]["input_frac"])
    )
    if read_json(root / "preprocess_metadata.json") != metadata:
        raise RuntimeError("Float preprocessing metadata differs from trusted reconstruction")
    if read_json(root / "preprocess_int_metadata.json") != integer_metadata:
        raise RuntimeError("Integer preprocessing metadata differs from trusted reconstruction")
    with tempfile.TemporaryDirectory(prefix="cukd_lineage_headers_") as temporary:
        expected_root = Path(temporary)
        expected_model = expected_root / "model_weights.h"
        write_header(expected_model, quantized_layers, context["model_path"].name)
        expected_model.write_text(
            expected_model.read_text(encoding="ascii").replace(
                "Generated by deployment/firmware_export/wsnds_rfkd_hil/"
                "export_wsnds_student_a_rfkd_int8.py.",
                f"Generated from final lineage: {context['student_name']} "
                f"{context['route']} seed 42.",
            ),
            encoding="ascii",
        )
        write_preprocessing_header(
            expected_root / "preprocess_metadata.h", metadata
        )
        write_integer_preprocessing_header(
            expected_root / "preprocess_int_metadata.h", integer_metadata
        )
        for name in [
            "model_weights.h",
            "preprocess_metadata.h",
            "preprocess_int_metadata.h",
        ]:
            if (root / name).read_bytes() != (expected_root / name).read_bytes():
                raise RuntimeError(
                    f"Generated firmware input differs from trusted reconstruction: {name}"
                )

    expected_sources = np.asarray(context["split"]["test_indices"], dtype=np.int64)
    expected_labels = np.asarray(context["split"]["y_test"], dtype=np.int64)
    expected_raw = quantize_raw_features_q(
        context["split"]["X_test_raw"], integer_metadata["raw_q_frac"]
    ).astype(np.int64)
    expected_preprocessed = simulate_integer_preprocess_q(
        expected_raw, integer_metadata
    )
    expected_fixed_logits, expected_fixed_predictions = simulate_fixed_point_inference(
        quantized_layers, expected_preprocessed
    )
    expected_fp32_predictions = np.asarray(
        context["fp32_predictions"], dtype=np.int64
    )
    comparisons = {
        "source rows": (replay_sources, expected_sources),
        "raw replay features": (replay_features, expected_raw),
        "labels": (labels, expected_labels),
        "FP32 predictions": (fp32_predictions, expected_fp32_predictions),
        "fixed predictions": (fixed_predictions, expected_fixed_predictions),
        "fixed logits": (fixed_logits, expected_fixed_logits.astype(np.int64)),
    }
    for label, (observed, expected) in comparisons.items():
        if not np.array_equal(observed, expected):
            raise RuntimeError(f"Final export {label} differ from trusted reconstruction")
    return {
        "trusted_lineage_files": len(EXPECTED_FINAL_LINEAGE_FILES),
        "test_rows_reconstructed": int(len(expected_sources)),
        "checkpoint_file_sha256": expected_identity["checkpoint_file_sha256"],
        "trained_state_sha256": expected_identity["trained_state_sha256"],
        "quantization_policy_sha256": canonical_json_sha256(quantization_policy),
    }


def verify_final_export(path: Path, *, cc: str = "gcc") -> dict[str, Any]:
    """Independently verify one published or staged final export directory."""
    root = path.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    manifest_path = root / "final_export_manifest.json"
    report_path = root / "final_export_report.json"
    identity_path = root / "final_export_identity.json"
    for required in [manifest_path, report_path, identity_path]:
        if not required.is_file():
            raise RuntimeError(f"Required final export file is missing: {required.name}")
    if (root / "strict_export_manifest.json").exists() or (
        root / "strict_export_report.json"
    ).exists():
        raise RuntimeError("Final export must not expose historical strict-export filenames")

    manifest = read_json(manifest_path)
    report = read_json(report_path)
    identity = read_json(identity_path)
    manifest_payload = dict(manifest)
    manifest_payload_hash = manifest_payload.pop("manifest_payload_sha256", None)
    if manifest_payload_hash != canonical_json_sha256(manifest_payload):
        raise RuntimeError("Final manifest canonical payload hash is invalid")
    report_payload = dict(report)
    report_payload_hash = report_payload.pop("report_payload_sha256", None)
    if report_payload_hash != canonical_json_sha256(report_payload):
        raise RuntimeError("Final report canonical payload hash is invalid")
    identity_payload = dict(identity)
    export_id = identity_payload.pop("export_id", None)
    if export_id != canonical_json_sha256(identity_payload):
        raise RuntimeError("Final identity export ID is invalid")
    if report.get("identity") != identity:
        raise RuntimeError("Report identity differs from final_export_identity.json")
    if manifest.get("export_id") != export_id:
        raise RuntimeError("Manifest export ID differs from final identity")
    if manifest.get("identity_canonical_sha256") != canonical_json_sha256(identity):
        raise RuntimeError("Manifest identity canonical hash is invalid")
    if manifest.get("report_canonical_sha256") != canonical_json_sha256(report):
        raise RuntimeError("Manifest report canonical hash is invalid")
    if manifest.get("status") != "passed" or report.get("status") != "passed":
        raise RuntimeError("Final manifest/report status is not passed")
    for key, expected in [
        ("protocol_id", PROTOCOL_ID),
        ("seed", SEED),
        ("student", identity.get("student")),
        ("route", identity.get("route")),
    ]:
        if manifest.get(key) != expected:
            raise RuntimeError(f"Final manifest identity mismatch for {key}")
    if identity.get("protocol") != PROTOCOL_ID or identity.get("seed") != SEED:
        raise RuntimeError("Final identity protocol or seed is invalid")
    for hash_key in [
        "checkpoint_file_sha256",
        "trained_state_sha256",
        "dataset_sha256",
        "split_indices_sha256",
        "test_source_indices_sha256",
        "scaler_sha256",
        "quantization_policy_sha256",
    ]:
        _assert_hash(identity.get(hash_key), f"final identity {hash_key}")
    identity_header = root / "cukd_export_identity.h"
    if (
        not identity_header.is_file()
        or identity_header.read_text(encoding="ascii") != _identity_header_text(identity)
    ):
        raise RuntimeError("C identity header differs from canonical JSON identity")
    if identity.get("student") not in ["A", "B"] or identity.get("route") not in [
        "scratch", "rf_kd"
    ]:
        raise RuntimeError("Final identity student or route is invalid")
    route_provenance = identity.get("route_provenance")
    if not isinstance(route_provenance, dict):
        raise RuntimeError("Route provenance is missing")
    if report.get("route_provenance") != route_provenance:
        raise RuntimeError("Report route provenance differs from final identity")
    if identity["route"] == "scratch":
        if (
            route_provenance.get("kd_hyperparameters") is not None
            or route_provenance.get("teacher_soft_target_provenance") is not None
            or route_provenance.get("teacher_probability_file_sha256") is not None
        ):
            raise RuntimeError("Scratch export contains RF-KD provenance")
    elif (
        route_provenance.get("kd_hyperparameters") != {"T": 4.0, "alpha": 0.7}
        or not isinstance(route_provenance.get("teacher_soft_target_provenance"), dict)
    ):
        raise RuntimeError("RF-KD export lacks its frozen KD provenance")

    files = manifest.get("files")
    if not isinstance(files, list) or manifest.get("file_count_excluding_manifest") != len(files):
        raise RuntimeError("Final manifest inventory count is invalid")
    listed: dict[str, dict[str, Any]] = {}
    for item in files:
        relative = item.get("path")
        if not isinstance(relative, str) or not relative:
            raise RuntimeError("Final manifest contains an invalid path")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts or relative in listed:
            raise RuntimeError(f"Unsafe or duplicate manifest path: {relative!r}")
        member = (root / relative_path).resolve()
        try:
            member.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"Manifest path escapes export: {relative!r}") from exc
        if (
            not member.is_file()
            or member.stat().st_size != item.get("size_bytes")
            or sha256_file(member) != item.get("sha256")
        ):
            raise RuntimeError(f"Manifest member mismatch: {relative}")
        listed[relative_path.as_posix()] = item
    actual = {
        member.relative_to(root).as_posix()
        for member in root.rglob("*")
        if member.is_file() and member.resolve() != manifest_path.resolve()
    }
    if actual != set(listed):
        raise RuntimeError("Final export contains extra or unlisted files")
    required_names = {
        *CORE_EXPORT_FILES,
        "cukd_export_identity.h",
        "final_export_identity.json",
        "final_export_report.json",
    }
    if not required_names <= set(listed):
        raise RuntimeError("Final export inventory lacks required canonical files")

    expected_snapshot_paths = {snapshot for _, snapshot in SOURCE_SNAPSHOT_SPECS}
    exact_inventory = required_names | expected_snapshot_paths
    if set(listed) != exact_inventory:
        raise RuntimeError("Final export manifest does not match the exact allowed inventory")

    snapshots = identity.get("source_snapshots")
    if not isinstance(snapshots, list) or len(snapshots) != len(SOURCE_SNAPSHOT_SPECS):
        raise RuntimeError("Identity source snapshot ledger is missing")
    expected_snapshot_pairs = set(SOURCE_SNAPSHOT_SPECS)
    observed_snapshot_pairs = {
        (item.get("origin_relative_path"), item.get("snapshot_path"))
        for item in snapshots
    }
    if observed_snapshot_pairs != expected_snapshot_pairs:
        raise RuntimeError("Identity source origin/snapshot mapping is incomplete")
    if {item.get("snapshot_path") for item in snapshots} != expected_snapshot_paths:
        raise RuntimeError("Identity source snapshot set is incomplete")
    if report.get("source_artifacts", {}).get("source_snapshots") != snapshots:
        raise RuntimeError("Report source snapshot ledger differs from identity")
    for item in snapshots:
        snapshot_path = item["snapshot_path"]
        _assert_hash(item.get("sha256"), f"sealed source {snapshot_path}")
        snapshot = root / snapshot_path
        if (
            snapshot_path not in listed
            or snapshot.stat().st_size != item.get("size_bytes")
            or sha256_file(snapshot) != item.get("sha256")
            or listed[snapshot_path].get("sha256") != item.get("sha256")
        ):
            raise RuntimeError(f"Sealed source hash mismatch: {snapshot_path}")
    snapshot_hashes = {item["snapshot_path"]: item["sha256"] for item in snapshots}
    source_artifacts = report["source_artifacts"]
    expected_named_source_hashes = {
        "exporter_sha256": snapshot_hashes[
            "source_snapshot/python/export_final_seed42.py"
        ],
        "legacy_numeric_exporter_sha256": snapshot_hashes[
            "source_snapshot/python/export_wsnds_student_a_rfkd_int8.py"
        ],
        "host_self_test_source_sha256": snapshot_hashes[
            "source_snapshot/c/wsnds_train_only_self_test.c"
        ],
        "executed_common_module_sha256_recorded": snapshot_hashes[
            "source_snapshot/python/tier15_common.py"
        ],
    }
    for key, expected in expected_named_source_hashes.items():
        if source_artifacts.get(key) != expected:
            raise RuntimeError(f"Named source hash differs from sealed source: {key}")
    identity_core_files = identity.get("core_files")
    if (
        not isinstance(identity_core_files, list)
        or {item.get("path") for item in identity_core_files}
        != set(CORE_EXPORT_FILES)
        or len(identity_core_files) != len(CORE_EXPORT_FILES)
    ):
        raise RuntimeError("Identity core-file ledger is not exact")
    for item in identity_core_files:
        relative = item.get("path")
        if (
            relative not in listed
            or listed[relative].get("size_bytes") != item.get("size_bytes")
            or listed[relative].get("sha256") != item.get("sha256")
        ):
            raise RuntimeError(f"Identity core-file binding is invalid: {relative}")
    policy = report.get("quantization_policy")
    if identity.get("quantization_policy_sha256") != canonical_json_sha256(policy):
        raise RuntimeError("Quantization policy hash differs from identity")
    if policy.get("selection_uses_test_data") is not False:
        raise RuntimeError("Quantization policy does not exclude test evidence")
    if (
        policy.get("applied_uniformly_to_all_student_route_exports") is not True
        or "post-hoc" not in str(policy.get("development_status"))
    ):
        raise RuntimeError("Quantization policy development scope is incomplete")
    if report.get("test_evaluation_scope") != {
        "frozen_test_quality_gate_assessments_this_export_invocation": 1,
        "selected_policy_python_fixed_test_forward_computations_before_gate": 3,
        "host_c_full_test_executions_recorded_before_report_seal": 1,
        "claims_first_ever_test_evaluation": False,
        "historical_test_observations_preceded_this_invocation": True,
    }:
        raise RuntimeError("Final report test-observation scope is misleading")

    replay = pd.read_csv(root / "hil_replay_vectors.csv")
    reference = pd.read_csv(root / "hil_reference_predictions.csv")
    expected_rows = EXPECTED_SPLIT_SIZES["test"]
    expected_row_ids = np.arange(expected_rows, dtype=np.int64)
    replay_columns = ["row_id", "source_row_index", *[f"f{i}" for i in range(17)]]
    reference_columns = [
        "row_id", "source_row_index", "true_label", "fixed_pred", "fp32_pred",
        *[f"fixed_logit_{i}" for i in range(5)],
    ]
    if replay.columns.tolist() != replay_columns or len(replay) != expected_rows:
        raise RuntimeError("Replay CSV schema or row count is invalid")
    if reference.columns.tolist() != reference_columns or len(reference) != expected_rows:
        raise RuntimeError("Reference CSV schema or row count is invalid")
    replay_rows = replay["row_id"].to_numpy(dtype=np.int64)
    reference_rows = reference["row_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(replay_rows, expected_row_ids) or not np.array_equal(
        reference_rows, expected_row_ids
    ):
        raise RuntimeError("Replay/reference row IDs are not dense 0..56,300")
    replay_sources = replay["source_row_index"].to_numpy(dtype=np.int64)
    reference_sources = reference["source_row_index"].to_numpy(dtype=np.int64)
    if (
        not np.array_equal(replay_sources, reference_sources)
        or len(np.unique(replay_sources)) != expected_rows
        or replay_sources.min(initial=0) < 0
        or replay_sources.max(initial=0) >= 374661
    ):
        raise RuntimeError("Source-row identities are incomplete, duplicated, or inconsistent")
    observed_source_hash = sha256_arrays(reference_sources.astype(np.int64, copy=False))
    if identity.get("test_source_indices_sha256") != observed_source_hash:
        raise RuntimeError("Test source-index content hash differs from final identity")
    replay_feature_frame = replay[[f"f{i}" for i in range(17)]]
    if any(not np.issubdtype(dtype, np.integer) for dtype in replay_feature_frame.dtypes):
        raise RuntimeError("Replay features are not encoded as integers")
    replay_features = replay_feature_frame.to_numpy(dtype=np.int64)
    if replay_features.shape != (expected_rows, 17):
        raise RuntimeError("Replay feature matrix is incomplete")
    int32 = np.iinfo(np.int32)
    if np.any((replay_features < int32.min) | (replay_features > int32.max)):
        raise RuntimeError("Replay feature value is outside int32 range")
    for column in reference_columns:
        if not np.issubdtype(reference[column].dtype, np.integer):
            raise RuntimeError(f"Reference column is not integer encoded: {column}")
    labels = reference["true_label"].to_numpy(dtype=np.int64)
    fixed_predictions = reference["fixed_pred"].to_numpy(dtype=np.int64)
    fp32_predictions = reference["fp32_pred"].to_numpy(dtype=np.int64)
    fixed_logits = reference[[f"fixed_logit_{i}" for i in range(5)]].to_numpy(
        dtype=np.int64
    )
    if np.any((labels < 0) | (labels >= 5)):
        raise RuntimeError("Reference CSV contains an invalid class label")
    if np.any((fixed_predictions < 0) | (fixed_predictions >= 5)):
        raise RuntimeError("Reference CSV contains an invalid fixed prediction")
    if np.any((fp32_predictions < 0) | (fp32_predictions >= 5)):
        raise RuntimeError("Reference CSV contains an invalid FP32 prediction")
    int16 = np.iinfo(np.int16)
    if np.any((fixed_logits < int16.min) | (fixed_logits > int16.max)):
        raise RuntimeError("Reference fixed logit is outside int16 range")
    if not np.array_equal(fixed_logits.argmax(axis=1), fixed_predictions):
        raise RuntimeError("Reference fixed predictions differ from fixed-logit argmax")

    gates = report.get("gates")
    if not isinstance(gates, dict) or gates.get("test_rows") != expected_rows:
        raise RuntimeError("Final gate ledger has an invalid test-row count")
    agreement = float(np.mean(fixed_predictions == fp32_predictions))
    fp32_metrics = _metrics_from_predictions(labels, fp32_predictions)
    fixed_metrics = _metrics_from_predictions(labels, fixed_predictions)
    fp32_macro_f1 = fp32_metrics["macro_f1"]
    fixed_macro_f1 = fixed_metrics["macro_f1"]
    absolute_drop = abs(fp32_macro_f1 - fixed_macro_f1)
    for key, observed in [
        ("fixed_vs_fp32_agreement", agreement),
        ("fp32_macro_f1", fp32_macro_f1),
        ("fixed_macro_f1", fixed_macro_f1),
        ("absolute_macro_f1_drop", absolute_drop),
    ]:
        if not np.isclose(float(gates.get(key)), observed, rtol=1e-12, atol=1e-12):
            raise RuntimeError(f"Reported gate value is inconsistent: {key}")
    _assert_nested_numeric_equal(
        report.get("fp32_metrics"), fp32_metrics, "verified.fp32_metrics"
    )
    _assert_nested_numeric_equal(
        report.get("fixed_metrics"), fixed_metrics, "verified.fixed_metrics"
    )
    if gates.get("minimum_fixed_vs_fp32_agreement") != MINIMUM_FIXED_FP32_AGREEMENT:
        raise RuntimeError("Agreement threshold is not the frozen threshold")
    if gates.get("maximum_absolute_macro_f1_drop") != MAXIMUM_ABSOLUTE_MACRO_F1_DROP:
        raise RuntimeError("Macro-F1 threshold is not the frozen threshold")
    enforce_quality_gates(agreement, absolute_drop)
    for boolean_key in [
        "checkpoint_predictions_exact",
        "quality_gates_passed",
        "zero_saturation_passed",
        "accumulator_bounds_passed",
        "preprocess_bounds_passed",
        "host_equivalence_passed",
        "dense_row_ids_passed",
        "source_row_ids_complete_unique_passed",
    ]:
        if gates.get(boolean_key) is not True:
            raise RuntimeError(f"Required gate boolean is not true: {boolean_key}")
    if gates.get("raw_input_saturation_count") != 0:
        raise RuntimeError("Test raw-input saturation is not zero")
    standardized = gates.get("standardized_input_saturation_count")
    if standardized != {"train": 0, "validation": 0, "test": 0}:
        raise RuntimeError("Standardized-input saturation ledger is not all zero")
    for key in [
        "weight_saturation_count",
        "bias_saturation_count",
        "activation_saturation_count",
        "integer_preprocess_saturation_count",
    ]:
        if gates.get(key) != 0:
            raise RuntimeError(f"Test saturation count is not zero: {key}")
    _verify_zero_saturation_partition(
        gates.get("training_partition_saturation"), "training"
    )
    _verify_zero_saturation_partition(
        gates.get("validation_partition_saturation"), "validation"
    )
    _verify_zero_saturation_partition(
        gates.get("test_partition_saturation"), "test"
    )
    accumulator_audit = gates.get("accumulator_bounds")
    if not isinstance(accumulator_audit, list) or len(accumulator_audit) != 3:
        raise RuntimeError("Accumulator-bound ledger is incomplete")
    if any(
        item.get("passed") is not True
        or item.get("output_shift", -1) < 0
        or item.get("pre_rescale_absolute_bound", 1) > item.get("int32_max", 0)
        for item in accumulator_audit
    ):
        raise RuntimeError("An accumulator bound is unsafe")
    preprocessing_bounds = gates.get("preprocess_multiply_bounds")
    if not isinstance(preprocessing_bounds, list) or len(preprocessing_bounds) != 17:
        raise RuntimeError("Preprocessing-bound ledger is incomplete")
    if any(
        item.get("passed") is not True
        or item.get("maximum_product_absolute", 1) > item.get("int64_max", 0)
        for item in preprocessing_bounds
    ):
        raise RuntimeError("An integer preprocessing bound is unsafe")

    host = report.get("host_equivalence")
    if (
        not isinstance(host, dict)
        or host.get("status") != "passed"
        or host.get("rows") != expected_rows
        or host.get("preprocessed_inputs_exact") is not True
        or host.get("fixed_logits_exact") is not True
        or host.get("fixed_predictions_exact") is not True
        or host.get("compile", {}).get("returncode") != 0
        or host.get("self_test", {}).get("returncode") != 0
        or host.get("temporary_executable_retained") is not False
    ):
        raise RuntimeError("Host self-test evidence is incomplete or failed")
    _assert_hash(
        host.get("temporary_executable_sha256"),
        "generation-time temporary executable",
    )
    lineage_verification = _verify_final_lineage_binding(
        root=root,
        identity=identity,
        report=report,
        replay_sources=replay_sources,
        replay_features=replay_features,
        labels=labels,
        fixed_predictions=fixed_predictions,
        fp32_predictions=fp32_predictions,
        fixed_logits=fixed_logits,
    )
    native_verification = _run_native_host_self_test(root, cc)
    return {
        "status": "passed",
        "export_id": export_id,
        "student": identity.get("student"),
        "route": identity.get("route"),
        "test_rows": expected_rows,
        "fixed_vs_fp32_agreement": agreement,
        "absolute_macro_f1_drop": absolute_drop,
        "native_verification": native_verification,
        "lineage_verification": lineage_verification,
    }


def verify_blocked_audit(path: Path) -> dict[str, Any]:
    """Reconstruct and verify a machine-readable blocked-export audit."""
    audit = read_json(path.resolve())
    payload = dict(audit)
    stored_payload_hash = payload.pop("audit_payload_sha256", None)
    if stored_payload_hash != canonical_json_sha256(payload):
        raise RuntimeError("Blocked audit canonical payload hash is invalid")
    if audit.get("status") != "blocked":
        raise RuntimeError("Blocked audit status is not blocked")
    if (
        audit.get("quality_gates_passed") is not False
        or audit.get("zero_saturation_passed") is not True
        or audit.get("accumulator_bounds_passed") is not True
        or audit.get("preprocess_bounds_passed") is not True
    ):
        raise RuntimeError("Blocked audit gate booleans are invalid")
    if (
        audit.get("frozen_test_quality_gate_assessments_this_export_invocation") != 1
        or audit.get("selected_policy_python_fixed_test_forward_computations_before_gate") != 3
        or audit.get("claims_first_ever_test_evaluation") is not False
        or audit.get("historical_test_observations_preceded_this_invocation") is not True
    ):
        raise RuntimeError("Blocked audit test-observation scope is misleading")
    identity = audit.get("identity")
    if not isinstance(identity, dict):
        raise RuntimeError("Blocked audit identity is missing")
    identity_payload = dict(identity)
    blocked_audit_id = identity_payload.pop("blocked_audit_id", None)
    if blocked_audit_id != canonical_json_sha256(identity_payload):
        raise RuntimeError("Blocked audit identity hash is invalid")
    if (
        identity.get("protocol") != PROTOCOL_ID
        or identity.get("seed") != SEED
        or identity.get("student") not in ["A", "B"]
        or identity.get("route") not in ["scratch", "rf_kd"]
    ):
        raise RuntimeError("Blocked audit lineage identity is invalid")
    context = load_verified_context(
        REPO_ROOT / EXPECTED_RELATIVE_ROOT,
        REPO_ROOT / "data" / "wsnds" / "WSN-DS.csv",
        identity["student"],
        identity["route"],
    )
    expected_lineage = {
        "dataset_sha256": context["dataset"]["dataset_sha256"],
        "split_indices_sha256": context["preprocessing"]["split_indices_sha256"],
        "test_source_indices_sha256": sha256_arrays(
            np.asarray(context["split"]["test_indices"], dtype=np.int64)
        ),
        "scaler_sha256": context["preprocessing"]["scaler_sha256"],
        "checkpoint_file_sha256": sha256_file(context["model_path"]),
        "trained_state_sha256": context["trained_state_sha256"],
        "rich_artifact_sha256": sha256_file(context["rich_path"]),
        "predictions_sha256": sha256_file(context["predictions_path"]),
    }
    for key, expected in expected_lineage.items():
        if identity.get(key) != expected:
            raise RuntimeError(f"Blocked audit lineage hash mismatch: {key}")
    expected_route_provenance = {
        "kd_hyperparameters": context["kd_hyperparameters"],
        "teacher_soft_target_provenance": context["teacher_soft_target_provenance"],
        "teacher_probability_file_sha256": (
            sha256_file(context["teacher_probability_path"])
            if context["teacher_probability_path"] is not None
            else None
        ),
    }
    if identity.get("route_provenance") != expected_route_provenance:
        raise RuntimeError("Blocked audit route provenance is invalid")
    snapshots = identity.get("source_snapshots")
    if (
        not isinstance(snapshots, list)
        or len(snapshots) != len(SOURCE_SNAPSHOT_SPECS)
        or {
            (item.get("origin_relative_path"), item.get("snapshot_path"))
            for item in snapshots
        }
        != set(SOURCE_SNAPSHOT_SPECS)
    ):
        raise RuntimeError("Blocked audit source ledger is incomplete")
    for item in snapshots:
        origin = REPO_ROOT / item["origin_relative_path"]
        _assert_hash(item.get("sha256"), f"blocked source {origin}")
        if (
            not origin.is_file()
            or origin.stat().st_size != item.get("size_bytes")
            or sha256_file(origin) != item.get("sha256")
        ):
            raise RuntimeError(f"Blocked audit source hash mismatch: {origin}")

    metadata = build_preprocessing_metadata(
        target_col=context["dataset"]["target_column"],
        feature_names=context["dataset"]["feature_names"],
        class_names=context["dataset"]["class_names"],
        scaler_mean=context["scaler"].mean_,
        scaler_scale=context["scaler"].scale_,
        split_sizes={"train": 262197, "val": 56163, "test": 56301},
    )
    quantized, _, recomputed_policy = select_quantization_policy(
        layers=context["layers"],
        preprocessing_metadata=metadata,
        x_train=context["scaled"]["X_train"],
        x_train_raw=context["split"]["X_train_raw"],
        x_validation=context["scaled"]["X_validation"],
        x_validation_raw=context["split"]["X_validation_raw"],
        y_validation=context["split"]["y_validation"],
    )
    _assert_nested_numeric_equal(
        audit.get("quantization_policy"), recomputed_policy, "blocked.quantization_policy"
    )
    if identity.get("quantization_policy_sha256") != canonical_json_sha256(
        recomputed_policy
    ):
        raise RuntimeError("Blocked audit quantization policy hash is invalid")
    split = context["split"]
    scaled = context["scaled"]
    integer_metadata = build_integer_preprocessing_metadata(
        metadata, output_q_frac=int(quantized[0]["input_frac"])
    )
    raw_test_saturation = _raw_int32_saturation_count(
        split["X_test_raw"], integer_metadata["raw_q_frac"]
    )
    raw_test_q = quantize_raw_features_q(
        split["X_test_raw"], integer_metadata["raw_q_frac"]
    )
    test_saturation, _, fixed_logits, fixed_predictions = saturation_audit(
        context["layers"], quantized, raw_test_q, integer_metadata
    )
    test_saturation = dict(test_saturation)
    test_saturation["partition"] = "test partition"
    test_saturation["raw_input_saturation_count"] = raw_test_saturation
    training_saturation = calibration_partition_saturation_audit(
        context["layers"], quantized, split["X_train_raw"], integer_metadata
    )
    validation_saturation = calibration_partition_saturation_audit(
        context["layers"], quantized, split["X_validation_raw"], integer_metadata
    )
    validation_saturation["partition"] = "validation partition"
    standardized_input_saturation: dict[str, int] = {}
    for partition in ["train", "validation", "test"]:
        _, stats = quantize_standardized_q15(
            scaled[f"X_{partition}"], input_frac=int(quantized[0]["input_frac"])
        )
        standardized_input_saturation[partition] = int(stats["saturation_count"])
    expected_saturation_ledgers = {
        "training": training_saturation,
        "validation": validation_saturation,
        "test": test_saturation,
        "standardized_inputs": standardized_input_saturation,
    }
    _assert_nested_numeric_equal(
        audit.get("saturation_ledgers"),
        expected_saturation_ledgers,
        "blocked.saturation_ledgers",
    )
    expected_accumulator = accumulator_bounds(quantized)
    expected_preprocess_bounds = preprocess_multiply_bounds(integer_metadata)
    _assert_nested_numeric_equal(
        audit.get("accumulator_bounds"), expected_accumulator, "blocked.accumulator_bounds"
    )
    _assert_nested_numeric_equal(
        audit.get("preprocess_multiply_bounds"),
        expected_preprocess_bounds,
        "blocked.preprocess_multiply_bounds",
    )
    fixed_metrics = classification_metrics(split["y_test"], fixed_logits)
    agreement = float(np.mean(context["fp32_predictions"] == fixed_predictions))
    absolute_drop = abs(
        float(context["fp32_metrics"]["macro_f1"]) - float(fixed_metrics["macro_f1"])
    )
    expected_gate_values = {
        "test_rows": EXPECTED_SPLIT_SIZES["test"],
        "fixed_vs_fp32_agreement": agreement,
        "minimum_fixed_vs_fp32_agreement": MINIMUM_FIXED_FP32_AGREEMENT,
        "fp32_macro_f1": context["fp32_metrics"]["macro_f1"],
        "fixed_macro_f1": fixed_metrics["macro_f1"],
        "absolute_macro_f1_drop": absolute_drop,
        "maximum_absolute_macro_f1_drop": MAXIMUM_ABSOLUTE_MACRO_F1_DROP,
    }
    for key, expected in expected_gate_values.items():
        if not np.isclose(float(audit.get(key)), float(expected), rtol=1e-12, atol=1e-12):
            raise RuntimeError(f"Blocked audit gate mismatch: {key}")
    try:
        enforce_quality_gates(agreement, absolute_drop)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("Blocked audit now passes frozen quality gates")
    return {
        "status": "blocked_verified",
        "blocked_audit_id": blocked_audit_id,
        "student": identity["student"],
        "route": identity["route"],
        "fixed_vs_fp32_agreement": agreement,
        "absolute_macro_f1_drop": absolute_drop,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    context = load_verified_context(
        args.confirmation_root, args.dataset_csv, args.student, args.route
    )
    try:
        manifest = export(context, args.output_dir, args.cc)
    except FinalQualityGateError as exc:
        if args.blocked_audit_json is not None:
            audit_path = args.blocked_audit_json.resolve()
            if audit_path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite blocked audit: {audit_path}"
                ) from exc
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            payload = dict(exc.audit)
            payload["audit_payload_sha256"] = canonical_json_sha256(payload)
            temporary = audit_path.with_suffix(audit_path.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(temporary, audit_path)
            verify_blocked_audit(audit_path)
            print(audit_path, file=sys.stderr)
        raise
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
