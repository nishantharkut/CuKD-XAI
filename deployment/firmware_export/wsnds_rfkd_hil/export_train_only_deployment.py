"""Export a provenance-locked train-only-scaler RF-KD deployment model.

This script never fits a scaler and never chooses a model. It consumes the
artifacts produced by run_tier15_confirmation.py --mode deployment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import time
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.wsnds.leakage_free_rerun.tier15_common import (  # noqa: E402
    CLASS_NAMES,
    KD_ALPHA,
    KD_T,
    RF_CONFIG,
    STUDENT_SPECS,
    TRAIN_CONFIG,
    apply_train_scaler,
    archived_random_split,
    classification_metrics,
    load_wsnds,
    sha256_arrays,
    sha256_file,
    split_hashes,
)


LEGACY_EXPORTER_PATH = SCRIPT_DIR / "export_wsnds_student_a_rfkd_int8.py"
DEPLOYMENT_PROTOCOL = "wsnds_archive_split_train_only_scaler_deployment_seed42_v1"
MINIMUM_FIXED_FP32_AGREEMENT = 0.99
MAXIMUM_MACRO_F1_DROP = 0.01
FIRMWARE_COMMON_DIR = REPO_ROOT / "deployment" / "hardware_hil" / "firmware" / "common"
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


def load_legacy_exporter():
    spec = importlib.util.spec_from_file_location("cukd_legacy_exporter", LEGACY_EXPORTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load exporter module: {LEGACY_EXPORTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deployment-root",
        type=Path,
        default=(
            REPO_ROOT
            / "results"
            / "wsnds"
            / "confirmation_runs_v2"
            / "deployment_seed_42"
        ),
    )
    parser.add_argument("--student", choices=["A", "B"], required=True)
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=REPO_ROOT / "data" / "wsnds" / "WSN-DS.csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cc", default="gcc")
    parser.add_argument("--skip-host-compile", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def verify_manifest(root: Path, manifest_name: str = "artifact_manifest.json") -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / manifest_name
    manifest = read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Artifact manifest is not complete: {manifest_path}")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError(f"Artifact manifest has no file inventory: {manifest_path}")
    if manifest.get("file_count_excluding_manifest") != len(files):
        raise RuntimeError(f"Artifact manifest file count is inconsistent: {manifest_path}")
    seen: set[str] = set()
    for item in files:
        relative = item.get("path")
        if not isinstance(relative, str) or not relative:
            raise RuntimeError(f"Artifact manifest contains an invalid path: {manifest_path}")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(f"Artifact manifest path escapes its root: {relative!r}")
        normalized = relative_path.as_posix()
        if normalized in seen:
            raise RuntimeError(f"Artifact manifest contains a duplicate path: {relative!r}")
        seen.add(normalized)
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"Artifact manifest path escapes its root: {relative!r}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        size = item.get("size_bytes")
        digest = item.get("sha256")
        if not isinstance(size, int) or size < 0 or not isinstance(digest, str):
            raise RuntimeError(f"Artifact manifest metadata is invalid for: {relative!r}")
        if path.stat().st_size != size or sha256_file(path) != digest:
            raise RuntimeError(f"Artifact manifest mismatch: {path}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.resolve() != manifest_path.resolve()
    }
    if actual != seen:
        raise RuntimeError(f"Artifact manifest inventory differs from files on disk: {manifest_path}")
    return manifest


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_manifest_member(
    root: Path,
    manifest: dict[str, Any],
    relative: Any,
    expected_sha256: Any,
) -> Path:
    if not isinstance(relative, str) or not relative:
        raise RuntimeError("Completion record contains an invalid artifact path")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise RuntimeError(f"Completion artifact path escapes seed root: {relative!r}")
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Completion artifact path escapes seed root: {relative!r}") from exc
    item = next(
        (entry for entry in manifest.get("files", []) if entry.get("path") == relative_path.as_posix()),
        None,
    )
    if (
        item is None
        or item.get("sha256") != expected_sha256
        or not path.is_file()
        or path.stat().st_size != item.get("size_bytes")
        or sha256_file(path) != expected_sha256
    ):
        raise RuntimeError(f"Completion artifact is not bound by the seed manifest: {relative}")
    return path


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="ascii")
    os.replace(temporary, path)


def verify_execution_contract(
    deployment_root: Path,
    execution: dict[str, Any],
    completion: dict[str, Any],
) -> None:
    expected = {
        "protocol_id": DEPLOYMENT_PROTOCOL,
        "mode": "deployment",
        "seeds": [42],
        "students": {name: list(dims) for name, dims in STUDENT_SPECS.items()},
        "routes": ["rf_kd"],
        "kd_hyperparameters": {"T": KD_T, "alpha": KD_ALPHA},
        "training_config": TRAIN_CONFIG,
        "teacher_config": RF_CONFIG,
    }
    for key, value in expected.items():
        if execution.get(key) != value:
            raise RuntimeError(f"Execution contract mismatch for {key}: {execution.get(key)!r}")
    fingerprint_payload = dict(execution)
    observed_fingerprint = fingerprint_payload.pop("execution_fingerprint_sha256", None)
    expected_fingerprint = canonical_json_sha256(fingerprint_payload)
    if observed_fingerprint != expected_fingerprint:
        raise RuntimeError("Execution contract fingerprint is invalid")
    runner = REPO_ROOT / "experiments" / "wsnds" / "leakage_free_rerun" / "run_tier15_confirmation.py"
    common = runner.parent / "tier15_common.py"
    if execution.get("script_sha256") != sha256_file(runner):
        raise RuntimeError("Execution contract is not bound to the current confirmation runner")
    if execution.get("common_module_sha256") != sha256_file(common):
        raise RuntimeError("Execution contract is not bound to the current common module")
    execution_sha256 = sha256_file(deployment_root / "execution_contract.json")
    if completion.get("execution_contract_sha256") != execution_sha256:
        raise RuntimeError("Seed completion is not bound to the execution contract")
    teacher_provenance = execution.get("bound_teacher_soft_target_provenance")
    if not isinstance(teacher_provenance, dict):
        raise RuntimeError("Deployment execution lacks bound RF soft-target provenance")
    if teacher_provenance.get("rf_seed") != 42:
        raise RuntimeError("Deployment RF soft-target cache is not identified as seed 42")
    if teacher_provenance.get("rf_config") != RF_CONFIG:
        raise RuntimeError("Deployment RF soft-target configuration differs from the protocol")
    if teacher_provenance.get("cache_file_sha256") != teacher_provenance.get(
        "expected_cache_file_sha256"
    ):
        raise RuntimeError("Deployment RF soft-target cache is not the expected preserved cache")
    if completion.get("teacher_soft_target_provenance") != teacher_provenance:
        raise RuntimeError("Seed completion is not bound to the execution RF soft targets")


def prepare_output(path: Path) -> tuple[Path, Path]:
    final = path.resolve()
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite export path: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.parent / f".{final.name}.tmp.{os.getpid()}.{time.time_ns()}"
    staging.mkdir()
    return final, staging


def require_output_outside_inputs(output: Path, protected: list[Path]) -> None:
    output = output.resolve()
    for root in protected:
        root = root.resolve()
        try:
            output.relative_to(root)
        except ValueError:
            continue
        raise RuntimeError(f"Export output cannot be inside protected input: {output}")


def load_verified_context(
    deployment_root: Path,
    dataset_csv: Path,
    student_letter: str,
) -> dict[str, Any]:
    deployment_root = deployment_root.resolve()
    seed_root = deployment_root / "seed_42"
    root_manifest = verify_manifest(deployment_root)
    seed_manifest = verify_manifest(seed_root)
    preprocessing = read_json(deployment_root / "preprocessing_contract.json")
    execution = read_json(deployment_root / "execution_contract.json")
    completion = read_json(seed_root / "seed_completion.json")
    for document in [preprocessing, execution, completion]:
        if document.get("protocol_id") != DEPLOYMENT_PROTOCOL:
            raise RuntimeError(f"Protocol mismatch in deployment input: {document.get('protocol_id')}")
    if root_manifest.get("protocol_id") != DEPLOYMENT_PROTOCOL:
        raise RuntimeError("Deployment root manifest protocol mismatch")
    if seed_manifest.get("protocol_id") != DEPLOYMENT_PROTOCOL:
        raise RuntimeError("Deployment seed manifest protocol mismatch")
    if completion.get("status") != "complete" or completion.get("seed") != 42:
        raise RuntimeError("Deployment seed completion is not the complete seed-42 run")
    verify_execution_contract(deployment_root, execution, completion)

    dataset_csv = dataset_csv.resolve()
    dataset = load_wsnds(dataset_csv)
    if dataset["dataset_sha256"] != preprocessing["dataset_sha256"]:
        raise RuntimeError("Dataset SHA-256 differs from deployment preprocessing contract")
    split = archived_random_split(dataset["features"], dataset["labels"])
    scaled, scaler = apply_train_scaler(split)
    computed_split_hashes = split_hashes(split)
    computed_scaler_hash = sha256_arrays(
        np.asarray(scaler.mean_, dtype=np.float64),
        np.asarray(scaler.scale_, dtype=np.float64),
        np.asarray(scaler.var_, dtype=np.float64),
    )
    if computed_split_hashes != preprocessing["split_hashes"]:
        raise RuntimeError("Recomputed raw split hashes differ from deployment contract")
    if computed_scaler_hash != preprocessing["scaler_sha256"]:
        raise RuntimeError("Recomputed train-only scaler differs from deployment contract")
    if split["group_audit"] != preprocessing.get("feature_overlap_audit"):
        raise RuntimeError("Recomputed exact-feature overlap audit differs from deployment contract")

    indices_file = deployment_root / preprocessing["split_indices_file"]
    scaler_file = deployment_root / preprocessing["scaler_parameters_file"]
    if sha256_file(indices_file) != preprocessing["split_indices_file_sha256"]:
        raise RuntimeError("Split-index file SHA-256 mismatch")
    if sha256_file(scaler_file) != preprocessing["scaler_parameters_file_sha256"]:
        raise RuntimeError("Scaler-parameter file SHA-256 mismatch")
    with np.load(indices_file, allow_pickle=False) as saved_indices:
        for name in ["train", "validation", "test"]:
            if not np.array_equal(saved_indices[f"{name}_indices"], split[f"{name}_indices"]):
                raise RuntimeError(f"Saved {name} indices differ from recomputed split")
    with np.load(scaler_file, allow_pickle=False) as saved_scaler:
        for name, expected in [
            ("mean", scaler.mean_),
            ("scale", scaler.scale_),
            ("var", scaler.var_),
        ]:
            if not np.array_equal(saved_scaler[name], np.asarray(expected, dtype=np.float64)):
                raise RuntimeError(f"Saved scaler {name} differs from recomputed value")

    student_name = f"student_{student_letter}"
    result_key = f"{student_name}_rf_kd"
    result = completion["student_results"][result_key]
    model_path = resolve_manifest_member(
        seed_root, seed_manifest, result.get("plain_state_dict"),
        result.get("plain_state_dict_sha256"),
    )
    rich_path = resolve_manifest_member(
        seed_root, seed_manifest, result.get("rich_artifact"),
        result.get("rich_artifact_sha256"),
    )
    predictions_path = resolve_manifest_member(
        seed_root, seed_manifest, result.get("test_predictions"),
        result.get("test_predictions_sha256"),
    )

    rich = torch.load(rich_path, map_location="cpu", weights_only=False)
    plain_state = torch.load(model_path, map_location="cpu", weights_only=False)
    rich_state = rich.get("state_dict")
    if not isinstance(plain_state, dict) or not isinstance(rich_state, dict):
        raise RuntimeError("Deployment model files do not contain state dictionaries")
    if set(plain_state) != set(rich_state):
        raise RuntimeError("Plain and rich deployment state dictionaries have different keys")
    for key in plain_state:
        if not torch.equal(plain_state[key], rich_state[key]):
            raise RuntimeError(f"Plain and rich deployment tensors differ: {key}")
    trained_state_sha256 = sha256_arrays(*[
        plain_state[key].detach().cpu().numpy() for key in sorted(plain_state)
    ])
    if trained_state_sha256 != rich.get("trained_state_sha256"):
        raise RuntimeError("Rich artifact trained-state hash does not match its tensors")
    if trained_state_sha256 != result.get("trained_state_sha256"):
        raise RuntimeError("Completion record trained-state hash does not match the model")
    expected_hidden = list(STUDENT_SPECS[student_name])
    expected_rich = {
        "protocol_id": DEPLOYMENT_PROTOCOL,
        "seed": 42,
        "student": student_name,
        "route": "rf_kd",
        "hidden_dims": expected_hidden,
        "input_dim": 17,
        "num_classes": len(CLASS_NAMES),
        "feature_names": dataset["feature_names"],
        "class_names": CLASS_NAMES,
        "dataset_sha256": dataset["dataset_sha256"],
        "split_hashes": computed_split_hashes,
        "scaler_sha256": computed_scaler_hash,
        "kd_hyperparameters": {"T": KD_T, "alpha": KD_ALPHA},
        "training_config": TRAIN_CONFIG,
        "feature_overlap_audit": split["group_audit"],
        "teacher_soft_target_provenance": execution["bound_teacher_soft_target_provenance"],
    }
    for key, value in expected_rich.items():
        if rich.get(key) != value:
            raise RuntimeError(f"Rich model contract mismatch for {key}: {rich.get(key)!r}")
    for key, tensor in plain_state.items():
        if not torch.is_tensor(tensor) or not torch.isfinite(tensor).all():
            raise RuntimeError(f"Deployment model tensor is non-finite or invalid: {key}")

    return {
        "deployment_root": deployment_root,
        "seed_root": seed_root,
        "dataset_csv": dataset_csv,
        "dataset": dataset,
        "split": split,
        "scaled": scaled,
        "scaler": scaler,
        "preprocessing": preprocessing,
        "execution": execution,
        "completion": completion,
        "student_name": student_name,
        "hidden_dims": expected_hidden,
        "result": result,
        "model_path": model_path,
        "rich_path": rich_path,
        "predictions_path": predictions_path,
    }


def accumulator_bounds(quantized_layers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bounds = []
    max_activation = 32768
    int32_max = np.iinfo(np.int32).max
    for index, layer in enumerate(quantized_layers):
        weights = np.asarray(layer["weight"], dtype=np.int64)
        biases = np.asarray(layer["bias"], dtype=np.int64)
        per_output = np.abs(biases) + max_activation * np.abs(weights).sum(axis=1)
        maximum = int(per_output.max(initial=0))
        shift = int(layer["output_shift"])
        if shift < 0:
            raise RuntimeError(
                f"Layer {index} requires signed left-shift rescaling, which is "
                "not accepted by the strict C numeric contract"
            )
        shifted_bound = maximum
        passed = shifted_bound <= int32_max
        bounds.append({
            "layer": index,
            "pre_rescale_absolute_bound": maximum,
            "output_shift": shift,
            "post_left_shift_absolute_bound": shifted_bound,
            "int32_max": int(int32_max),
            "passed": passed,
        })
        if not passed:
            raise RuntimeError(f"Layer {index} can overflow the firmware int32 accumulator")
    return bounds


def preprocess_multiply_bounds(integer_metadata: dict[str, Any]) -> list[dict[str, Any]]:
    means = np.asarray(integer_metadata["scaler_mean_q"], dtype=np.int64)
    inverse_scales = np.asarray(integer_metadata["scaler_inv_scale_q"], dtype=np.int64)
    if means.shape != (17,) or inverse_scales.shape != (17,):
        raise RuntimeError("Integer preprocessing constants must each contain 17 values")
    int32 = np.iinfo(np.int32)
    int64_max = int(np.iinfo(np.int64).max)
    bounds: list[dict[str, Any]] = []
    for index, (mean, inverse_scale) in enumerate(zip(means, inverse_scales)):
        maximum_centered = max(
            abs(int(int32.min) - int(mean)),
            abs(int(int32.max) - int(mean)),
        )
        maximum_product = maximum_centered * abs(int(inverse_scale))
        passed = maximum_product <= int64_max
        bounds.append({
            "feature": index,
            "maximum_centered_absolute": maximum_centered,
            "inverse_scale_absolute": abs(int(inverse_scale)),
            "maximum_product_absolute": maximum_product,
            "int64_max": int64_max,
            "passed": passed,
        })
        if not passed:
            raise RuntimeError(
                f"Feature {index} can overflow the firmware int64 preprocessing multiply"
            )
    return bounds


def rescale_truncating_toward_zero(values: np.ndarray, shift: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if shift > 0:
        return np.where(values >= 0, values >> shift, -((-values) >> shift))
    if shift < 0:
        return values << (-shift)
    return values


def saturation_audit(
    layers: list[tuple[str, Any, Any]],
    quantized_layers: list[dict[str, Any]],
    raw_inputs_q: np.ndarray,
    integer_metadata: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    weight_saturation = 0
    bias_saturation = 0
    layer_parameter_counts: list[dict[str, Any]] = []
    for index, ((prefix, weight, bias), quantized) in enumerate(zip(layers, quantized_layers)):
        if not np.isfinite(np.asarray(weight, dtype=np.float64)).all() or not np.isfinite(
            np.asarray(bias, dtype=np.float64)
        ).all():
            raise RuntimeError(f"Layer {index} contains non-finite FP32 parameters")
        weight_unbounded = np.rint(
            np.asarray(weight, dtype=np.float64) * float(1 << int(quantized["weight_frac"]))
        )
        bias_unbounded = np.rint(
            np.asarray(bias, dtype=np.float64) * float(1 << int(quantized["accum_frac"]))
        )
        weight_count = int(np.count_nonzero((weight_unbounded < -128) | (weight_unbounded > 127)))
        bias_count = int(np.count_nonzero(
            (bias_unbounded < np.iinfo(np.int32).min)
            | (bias_unbounded > np.iinfo(np.int32).max)
        ))
        weight_saturation += weight_count
        bias_saturation += bias_count
        layer_parameter_counts.append({
            "layer": index,
            "source_prefix": prefix,
            "weight_saturation_count": weight_count,
            "bias_saturation_count": bias_count,
        })

    raw = np.asarray(raw_inputs_q, dtype=np.int64)
    centered = raw - np.asarray(integer_metadata["scaler_mean_q"], dtype=np.int64)
    scaled = centered * np.asarray(integer_metadata["scaler_inv_scale_q"], dtype=np.int64)
    preprocess_unclipped = rescale_truncating_toward_zero(
        scaled, int(integer_metadata["right_shift"])
    )
    preprocess_saturation = int(np.count_nonzero(
        (preprocess_unclipped < -32768) | (preprocess_unclipped > 32767)
    ))
    preprocessed_q = np.clip(preprocess_unclipped, -32768, 32767).astype(np.int16)
    activations = preprocessed_q.astype(np.int64)

    activation_counts: list[dict[str, Any]] = []
    final_logits = None
    for index, quantized in enumerate(quantized_layers):
        accumulator = (
            activations @ np.asarray(quantized["weight"], dtype=np.int64).T
            + np.asarray(quantized["bias"], dtype=np.int64)
        )
        output = rescale_truncating_toward_zero(accumulator, int(quantized["output_shift"]))
        if index < len(quantized_layers) - 1:
            output = np.maximum(output, 0)
        count = int(np.count_nonzero((output < -32768) | (output > 32767)))
        activation_counts.append({
            "layer": index,
            "activation_saturation_count": count,
            "minimum_before_clip": int(output.min()),
            "maximum_before_clip": int(output.max()),
        })
        activations = np.clip(output, -32768, 32767).astype(np.int16).astype(np.int64)
        final_logits = activations.astype(np.int16)

    audit = {
        "parameter_layers": layer_parameter_counts,
        "weight_saturation_count": weight_saturation,
        "bias_saturation_count": bias_saturation,
        "integer_preprocess_saturation_count": preprocess_saturation,
        "activation_layers": activation_counts,
        "activation_saturation_count": int(sum(item["activation_saturation_count"] for item in activation_counts)),
    }
    if any(
        audit[key] != 0
        for key in [
            "weight_saturation_count",
            "bias_saturation_count",
            "integer_preprocess_saturation_count",
            "activation_saturation_count",
        ]
    ):
        raise RuntimeError(f"Strict fixed-point saturation audit failed: {audit}")
    if final_logits is None:
        raise RuntimeError("Saturation audit produced no model output")
    predictions = np.argmax(final_logits.astype(np.int32), axis=1).astype(np.int64)
    return audit, preprocessed_q, final_logits, predictions


def calibration_partition_saturation_audit(
    layers: list[tuple[str, Any, Any]],
    quantized_layers: list[dict[str, Any]],
    raw_features: np.ndarray,
    integer_metadata: dict[str, Any],
    chunk_size: int = 8192,
) -> dict[str, Any]:
    raw_features = np.asarray(raw_features)
    raw_saturation_count = 0
    preprocess_saturation_count = 0
    activation_saturation_counts = np.zeros(len(quantized_layers), dtype=np.int64)
    layer_minimums = np.full(len(quantized_layers), np.iinfo(np.int64).max, dtype=np.int64)
    layer_maximums = np.full(len(quantized_layers), np.iinfo(np.int64).min, dtype=np.int64)
    parameter_audit = None
    raw_scale = float(1 << int(integer_metadata["raw_q_frac"]))
    int32 = np.iinfo(np.int32)
    for start in range(0, len(raw_features), chunk_size):
        chunk = raw_features[start : start + chunk_size]
        unbounded = np.rint(np.asarray(chunk, dtype=np.float64) * raw_scale)
        raw_saturation_count += int(np.count_nonzero(
            (unbounded < int32.min) | (unbounded > int32.max)
        ))
        raw_q = np.clip(unbounded, int32.min, int32.max).astype(np.int32)
        audit, _, _, _ = saturation_audit(
            layers, quantized_layers, raw_q, integer_metadata
        )
        if parameter_audit is None:
            parameter_audit = audit["parameter_layers"]
        preprocess_saturation_count += audit["integer_preprocess_saturation_count"]
        for index, item in enumerate(audit["activation_layers"]):
            activation_saturation_counts[index] += item["activation_saturation_count"]
            layer_minimums[index] = min(layer_minimums[index], item["minimum_before_clip"])
            layer_maximums[index] = max(layer_maximums[index], item["maximum_before_clip"])
    result = {
        "partition": "training calibration partition",
        "rows_audited": int(len(raw_features)),
        "chunk_size": chunk_size,
        "raw_input_saturation_count": raw_saturation_count,
        "integer_preprocess_saturation_count": preprocess_saturation_count,
        "parameter_layers": parameter_audit or [],
        "activation_layers": [
            {
                "layer": index,
                "activation_saturation_count": int(activation_saturation_counts[index]),
                "minimum_before_clip": int(layer_minimums[index]),
                "maximum_before_clip": int(layer_maximums[index]),
            }
            for index in range(len(quantized_layers))
        ],
        "activation_saturation_count": int(activation_saturation_counts.sum()),
    }
    if raw_saturation_count or preprocess_saturation_count or np.any(
        activation_saturation_counts
    ):
        raise RuntimeError(f"Calibration-partition saturation audit failed: {result}")
    return result


def write_reference_with_logits(
    path: Path,
    source_row_indices: np.ndarray,
    labels: np.ndarray,
    fp32_predictions: np.ndarray,
    fixed_predictions: np.ndarray,
    fixed_logits: np.ndarray,
) -> None:
    data: dict[str, Any] = {
        "row_id": np.arange(len(labels), dtype=np.int64),
        "source_row_index": np.asarray(source_row_indices, dtype=np.int64),
        "true_label": labels.astype(np.int64),
        "fixed_pred": fixed_predictions.astype(np.int64),
        "fp32_pred": fp32_predictions.astype(np.int64),
    }
    for index in range(fixed_logits.shape[1]):
        data[f"fixed_logit_{index}"] = fixed_logits[:, index].astype(np.int64)
    temporary = path.with_suffix(path.suffix + ".tmp")
    expected_frame = pd.DataFrame(data)
    expected_frame.to_csv(temporary, index=False)
    os.replace(temporary, path)
    observed = pd.read_csv(path)
    if observed.columns.tolist() != expected_frame.columns.tolist() or not np.array_equal(
        observed.to_numpy(dtype=np.int64), expected_frame.to_numpy(dtype=np.int64)
    ):
        raise RuntimeError("Written HIL reference CSV differs from its in-memory reference")


def bind_replay_source_rows(
    path: Path,
    source_row_indices: np.ndarray,
    expected_features: np.ndarray,
) -> None:
    frame = pd.read_csv(path)
    expected_row_ids = np.arange(len(source_row_indices), dtype=np.int64)
    if "row_id" not in frame or not np.array_equal(
        frame["row_id"].to_numpy(dtype=np.int64), expected_row_ids
    ):
        raise RuntimeError("Generated replay CSV row IDs differ from the full test sequence")
    if "source_row_index" in frame:
        raise RuntimeError("Generated replay CSV unexpectedly already has source row indices")
    feature_columns = [f"f{index}" for index in range(17)]
    if frame.columns.tolist() != ["row_id", *feature_columns]:
        raise RuntimeError("Generated replay CSV has an unexpected feature contract")
    observed_features = frame[feature_columns].to_numpy()
    if not np.issubdtype(observed_features.dtype, np.integer) or not np.array_equal(
        observed_features.astype(np.int64, copy=False),
        np.asarray(expected_features, dtype=np.int64),
    ):
        raise RuntimeError("Generated replay CSV features differ from raw fixed-point inputs")
    frame.insert(1, "source_row_index", np.asarray(source_row_indices, dtype=np.int64))
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)
    rebound = pd.read_csv(path)
    if not np.array_equal(
        rebound["source_row_index"].to_numpy(dtype=np.int64),
        np.asarray(source_row_indices, dtype=np.int64),
    ):
        raise RuntimeError("Written replay CSV source-row indices differ")


def normalize_generated_headers(output_dir: Path) -> None:
    test_vectors = output_dir / "test_vectors.h"
    text = test_vectors.read_text(encoding="ascii")
    text = text.replace(
        "CUKD_WSNDS_STUDENT_A_RFKD_TEST_VECTORS_H",
        "CUKD_WSNDS_RFKD_TEST_VECTORS_H",
    )
    test_vectors.write_text(text, encoding="ascii")

    model_weights = output_dir / "model_weights.h"
    text = model_weights.read_text(encoding="ascii")
    text = text.replace(
        "Generated by deployment/firmware_export/wsnds_rfkd_hil/"
        "export_wsnds_student_a_rfkd_int8.py.",
        "Generated by the strict train-only deployment exporter using the shared "
        "fixed-point numeric core.",
    )
    model_weights.write_text(text, encoding="ascii")


def run_command(command: list[str], cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def file_inventory(root: Path, excluded: set[str]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in excluded
    ]


def export_identity(core_files: list[dict[str, Any]], provenance: dict[str, Any]) -> str:
    payload = json.dumps(
        {"provenance": provenance, "core_files": core_files},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    args = parse_args()
    legacy = load_legacy_exporter()
    context = load_verified_context(args.deployment_root, args.dataset_csv, args.student)
    require_output_outside_inputs(
        args.output_dir,
        [context["deployment_root"], context["dataset_csv"].parent],
    )
    final_output_dir, output_dir = prepare_output(args.output_dir)

    state_dict = legacy.load_state_dict(str(context["model_path"]))
    layers = legacy.extract_linear_layers(state_dict)
    if len(layers) != 3:
        raise RuntimeError(f"Expected three linear layers, found {len(layers)}")
    observed_hidden = [int(layers[0][1].shape[0]), int(layers[1][1].shape[0])]
    if observed_hidden != context["hidden_dims"]:
        raise RuntimeError(f"Model architecture mismatch: {observed_hidden}")

    split = context["split"]
    scaled = context["scaled"]
    x_test = scaled["X_test"]
    y_test = split["y_test"]
    fp32_logits = legacy.forward_numpy(layers, x_test)
    if not np.isfinite(fp32_logits).all():
        raise RuntimeError("Deployment FP32 logits contain NaN or infinity")
    fp32_predictions = fp32_logits.argmax(axis=1).astype(np.int64)
    saved_predictions = pd.read_csv(context["predictions_path"])
    probability_columns = [
        f"probability_{class_index}_{class_name}"
        for class_index, class_name in enumerate(CLASS_NAMES)
    ]
    expected_prediction_columns = [
        "source_row_index",
        "true_label",
        "predicted_label",
        *probability_columns,
    ]
    if saved_predictions.columns.tolist() != expected_prediction_columns:
        raise RuntimeError(
            "Saved deployment prediction columns differ from the protocol contract"
        )
    if len(saved_predictions) != len(y_test):
        raise RuntimeError("Saved deployment prediction row count differs from the test split")
    if not np.array_equal(
        saved_predictions["source_row_index"].to_numpy(dtype=np.int64),
        split["test_indices"],
    ):
        raise RuntimeError("Saved deployment predictions use different test rows")
    if not np.array_equal(
        saved_predictions["true_label"].to_numpy(dtype=np.int64),
        y_test,
    ):
        raise RuntimeError("Saved deployment true labels differ from the test split")
    saved_probabilities = saved_predictions[probability_columns].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(saved_probabilities).all()
        or np.any(saved_probabilities < 0.0)
        or not np.allclose(
            saved_probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-7
        )
    ):
        raise RuntimeError("Saved deployment probabilities are invalid")
    if not np.array_equal(
        saved_probabilities.argmax(axis=1).astype(np.int64),
        saved_predictions["predicted_label"].to_numpy(dtype=np.int64),
    ):
        raise RuntimeError("Saved deployment probability argmax differs from its prediction")
    if not np.array_equal(
        saved_predictions["predicted_label"].to_numpy(dtype=np.int64),
        fp32_predictions,
    ):
        raise RuntimeError("Exporter FP32 predictions differ from saved deployment predictions")

    metadata = legacy.build_preprocessing_metadata(
        target_col=context["dataset"]["target_column"],
        feature_names=context["dataset"]["feature_names"],
        class_names=context["dataset"]["class_names"],
        scaler_mean=context["scaler"].mean_,
        scaler_scale=context["scaler"].scale_,
        split_sizes={
            "train": len(split["train_indices"]),
            "val": len(split["validation_indices"]),
            "test": len(split["test_indices"]),
        },
    )
    metadata["preprocessing_contract"] = (
        "Seed-42 stratified raw-row 70/15/15 split followed by StandardScaler "
        "fit on the training partition only; validation and test are transform-only."
    )
    metadata["protocol_id"] = DEPLOYMENT_PROTOCOL
    metadata["dataset_sha256"] = context["dataset"]["dataset_sha256"]
    metadata["split_hashes"] = context["preprocessing"]["split_hashes"]
    metadata["scaler_sha256"] = context["preprocessing"]["scaler_sha256"]
    metadata["scaler_fit_partition"] = "train only"

    dataset_for_export = {
        "metadata": metadata,
        "x_calibration": scaled["X_train"],
        "y_calibration": split["y_train"],
        "x_test": x_test,
        "x_test_raw": split["X_test_raw"],
        "y_test": y_test,
        "x_train_shape": list(scaled["X_train"].shape),
        "x_val_shape": list(scaled["X_validation"].shape),
        "x_test_shape": list(x_test.shape),
    }
    quantized_layers, calibration = legacy.calibrate_quantized_layers(
        layers, dataset_for_export["x_calibration"]
    )
    calibration["calibration_source"] = (
        "WSN-DS training partition only after the bound train-only StandardScaler"
    )
    calibration["protocol_id"] = DEPLOYMENT_PROTOCOL
    bounds = accumulator_bounds(quantized_layers)
    legacy.write_header(
        output_dir / "model_weights.h", quantized_layers, context["model_path"].name
    )
    e2e = legacy.generate_e2e_artifacts(
        output_dir=output_dir,
        layers=layers,
        quantized_layers=quantized_layers,
        dataset=dataset_for_export,
        dataset_csv=context["dataset_csv"],
        calibration_summary=calibration,
        num_test_vectors=len(y_test),
        test_vector_seed=42,
    )
    normalize_generated_headers(output_dir)

    integer_metadata = e2e["integer_preprocess_metadata"]
    integer_metadata["formula"] = (
        "standardized_q = trunc_toward_zero(((raw_q - scaler_mean_q) * "
        "scaler_inv_scale_q) / 2^right_shift)"
    )
    atomic_write_json(output_dir / "preprocess_int_metadata.json", integer_metadata)
    int32_info = np.iinfo(np.int32)
    for key in ["scaler_mean_q", "scaler_inv_scale_q"]:
        values = integer_metadata.get(key)
        if not isinstance(values, list) or len(values) != 17 or any(
            not isinstance(value, int)
            or value < int32_info.min
            or value > int32_info.max
            for value in values
        ):
            raise RuntimeError(f"Integer preprocessing field {key} does not fit int32[17]")
    right_shift = integer_metadata.get("right_shift")
    if not isinstance(right_shift, int) or not 0 <= right_shift <= 62:
        raise RuntimeError("Integer preprocessing right shift is outside the strict C range")
    preprocessing_bounds = preprocess_multiply_bounds(integer_metadata)
    raw_inputs_q = legacy.quantize_raw_features_q(
        split["X_test_raw"], integer_metadata["raw_q_frac"]
    )
    unbounded_raw = np.rint(
        np.asarray(split["X_test_raw"], dtype=np.float64)
        * float(1 << int(integer_metadata["raw_q_frac"]))
    )
    raw_saturation_count = int(np.sum(
        (unbounded_raw < np.iinfo(np.int32).min)
        | (unbounded_raw > np.iinfo(np.int32).max)
    ))
    if raw_saturation_count != 0:
        raise RuntimeError(f"Raw fixed-point input saturation count is {raw_saturation_count}")
    preprocessed_q = legacy.simulate_integer_preprocess_q(raw_inputs_q, integer_metadata)
    fixed_logits, fixed_predictions = legacy.simulate_fixed_point_inference(
        quantized_layers, preprocessed_q
    )
    saturation, audited_preprocessed_q, audited_logits, audited_predictions = saturation_audit(
        layers,
        quantized_layers,
        raw_inputs_q,
        integer_metadata,
    )
    if not np.array_equal(audited_preprocessed_q, preprocessed_q):
        raise RuntimeError("Strict preprocessing audit differs from the generated Python reference")
    if not np.array_equal(audited_logits, fixed_logits):
        raise RuntimeError("Strict activation audit differs from the generated fixed-point logits")
    if not np.array_equal(audited_predictions, fixed_predictions):
        raise RuntimeError("Strict activation audit differs from the generated fixed predictions")
    calibration_saturation = calibration_partition_saturation_audit(
        layers,
        quantized_layers,
        split["X_train_raw"],
        integer_metadata,
    )
    bind_replay_source_rows(
        output_dir / "hil_replay_vectors.csv", split["test_indices"], raw_inputs_q
    )
    write_reference_with_logits(
        output_dir / "hil_reference_predictions.csv",
        split["test_indices"],
        y_test,
        fp32_predictions,
        fixed_predictions,
        fixed_logits,
    )

    direct_q, direct_q_stats = legacy.quantize_standardized_q15(
        x_test, input_frac=int(quantized_layers[0]["input_frac"])
    )
    preprocess_delta = np.asarray(preprocessed_q, dtype=np.int32) - np.asarray(
        direct_q, dtype=np.int32
    )
    preprocess_reference_delta = {
        "max_abs_delta_vs_direct_standardized_q": (
            int(np.max(np.abs(preprocess_delta))) if preprocess_delta.size else 0
        ),
        "exact_match_vs_direct_standardized_q": (
            bool(np.all(preprocess_delta == 0)) if preprocess_delta.size else True
        ),
        "interpretation": (
            "The integer scaler is an approximation of direct floating-point "
            "StandardScaler quantization. Deployment preservation is therefore "
            "gated by full-test prediction agreement and macro-F1, not by numeric "
            "identity of the two preprocessing paths."
        ),
    }
    if direct_q_stats["saturation_count"] != 0:
        raise RuntimeError(
            f"Standardized fixed-point input saturation count is "
            f"{direct_q_stats['saturation_count']}"
        )
    fp32_metrics = classification_metrics(y_test, fp32_logits)
    fixed_metrics = classification_metrics(y_test, fixed_logits)
    agreement = float(np.mean(fp32_predictions == fixed_predictions))
    macro_f1_drop = fp32_metrics["macro_f1"] - fixed_metrics["macro_f1"]
    if not np.isfinite(agreement) or not np.isfinite(macro_f1_drop):
        raise RuntimeError("Fixed-point quality gates produced a non-finite value")
    if agreement < MINIMUM_FIXED_FP32_AGREEMENT:
        raise RuntimeError(f"Fixed/FP32 agreement {agreement:.6f} is below the gate")
    if macro_f1_drop > MAXIMUM_MACRO_F1_DROP:
        raise RuntimeError(f"Fixed-point macro-F1 drop {macro_f1_drop:.6f} exceeds the gate")

    provenance = {
        "protocol_id": DEPLOYMENT_PROTOCOL,
        "student": context["student_name"],
        "seed": 42,
        "dataset_sha256": context["dataset"]["dataset_sha256"],
        "split_hashes": context["preprocessing"]["split_hashes"],
        "scaler_sha256": context["preprocessing"]["scaler_sha256"],
        "model_file_sha256": sha256_file(context["model_path"]),
        "model_artifact_sha256": sha256_file(context["rich_path"]),
        "execution_contract_sha256": sha256_file(
            context["deployment_root"] / "execution_contract.json"
        ),
        "seed_completion_sha256": sha256_file(context["seed_root"] / "seed_completion.json"),
        "exporter_sha256": sha256_file(SCRIPT_PATH),
        "legacy_numeric_exporter_sha256": sha256_file(LEGACY_EXPORTER_PATH),
        "firmware_common_files": {
            name: sha256_file(FIRMWARE_COMMON_DIR / name)
            for name in [
                "cukd_model.c", "cukd_model.h",
                "cukd_preprocess.c", "cukd_preprocess.h",
            ]
        },
        "host_self_test_source_sha256": sha256_file(
            SCRIPT_DIR / "wsnds_train_only_self_test.c"
        ),
        "calibration_partition": "train only",
        "feature_overlap_audit": split["group_audit"],
        "teacher_soft_target_provenance": context["execution"][
            "bound_teacher_soft_target_provenance"
        ],
    }
    core_inventory = [
        {
            "path": name,
            "size_bytes": (output_dir / name).stat().st_size,
            "sha256": sha256_file(output_dir / name),
        }
        for name in CORE_EXPORT_FILES
    ]
    export_id = export_identity(core_inventory, provenance)
    (output_dir / "cukd_export_identity.h").write_text(
        "#ifndef CUKD_EXPORT_IDENTITY_H\n"
        "#define CUKD_EXPORT_IDENTITY_H\n"
        f"#define CUKD_EXPORT_ID \"{export_id}\"\n"
        f"#define CUKD_STUDENT_ID \"{context['student_name']}\"\n"
        "#endif\n",
        encoding="ascii",
    )

    host_report = None
    if not args.skip_host_compile:
        executable = output_dir / "cukd_train_only_self_test"
        if os.name == "nt":
            executable = executable.with_suffix(".exe")
        compile_command = [
            args.cc,
            "-std=c99",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-O2",
            "-I",
            str(output_dir),
            "-I",
            str(FIRMWARE_COMMON_DIR),
            str(FIRMWARE_COMMON_DIR / "cukd_preprocess.c"),
            str(FIRMWARE_COMMON_DIR / "cukd_model.c"),
            str(SCRIPT_DIR / "wsnds_train_only_self_test.c"),
            "-o",
            str(executable),
        ]
        compile_result = run_command(compile_command, REPO_ROOT)
        if compile_result["returncode"] != 0:
            raise RuntimeError(f"Host compile failed: {compile_result['stderr']}")
        self_test_result = run_command([str(executable)], REPO_ROOT)
        if self_test_result["returncode"] != 0:
            raise RuntimeError(
                f"Host preprocessing/inference equivalence failed with "
                f"code {self_test_result['returncode']}"
            )
        host_report = {
            "compile": compile_result,
            "self_test": self_test_result,
            "executable": executable.name,
            "executable_sha256": sha256_file(executable),
        }

    export_status = "passed" if host_report is not None else "host_compile_skipped"
    report = {
        "status": export_status,
        "export_id": export_id,
        "provenance": provenance,
        "export_identity_payload": {
            "provenance": provenance,
            "core_files": core_inventory,
        },
        "gates": {
            "full_test_rows": len(y_test),
            "saved_test_rows_and_labels_exact": True,
            "saved_fp32_predictions_exact": True,
            "raw_input_saturation_count": raw_saturation_count,
            "standardized_input_saturation_count": direct_q_stats["saturation_count"],
            "strict_saturation_audit": saturation,
            "calibration_partition_saturation_audit": calibration_saturation,
            "accumulator_bounds": bounds,
            "preprocess_multiply_bounds": preprocessing_bounds,
            "integer_preprocess_reference_delta": preprocess_reference_delta,
            "fixed_vs_fp32_agreement": agreement,
            "minimum_fixed_vs_fp32_agreement": MINIMUM_FIXED_FP32_AGREEMENT,
            "fp32_macro_f1": fp32_metrics["macro_f1"],
            "fixed_macro_f1": fixed_metrics["macro_f1"],
            "macro_f1_drop": macro_f1_drop,
            "maximum_macro_f1_drop": MAXIMUM_MACRO_F1_DROP,
        },
        "fp32_metrics": fp32_metrics,
        "fixed_metrics": fixed_metrics,
        "fixed_point_calibration": calibration,
        "host_equivalence": host_report,
        "claim_boundary": (
            "Replay of already extracted 17-feature WSN-DS records; no live packet "
            "capture, feature extraction, board energy, or TelosB execution. The "
            "archive-compatible random-row split contains exact feature groups that "
            "cross partitions; deployment evidence establishes execution fidelity, "
            "not duplicate-free generalization."
        ),
    }
    (output_dir / "strict_export_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    inventory = file_inventory(output_dir, {"strict_export_manifest.json"})
    manifest = {
        "status": export_status,
        "export_id": export_id,
        "protocol_id": DEPLOYMENT_PROTOCOL,
        "student": context["student_name"],
        "export_identity_payload_sha256": export_id,
        "file_count_excluding_manifest": len(inventory),
        "files": inventory,
    }
    (output_dir / "strict_export_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if final_output_dir.exists():
        raise FileExistsError(f"Export path appeared during generation: {final_output_dir}")
    os.replace(output_dir, final_output_dir)
    print(final_output_dir / "strict_export_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
