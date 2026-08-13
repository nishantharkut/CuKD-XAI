"""Run the additive WSN-DS RF-KD hyperparameter sensitivity analysis.

This driver is intentionally bound to the completed ten-seed, feature-group-
disjoint WSN-DS lineage. It evaluates a declared 3 x 3 factorial grid for both
students while preserving each seed's calibrated RF probabilities, student
initialization, split, scaler, and minibatch order.

The analysis is descriptive sensitivity evidence. It does not optimize or
replace the primary model, and it must not be used to select a winning
temperature or alpha from the repeatedly reused validation/test partitions.

Default execution is read-only preflight. Model training requires
``--confirm-run``. Existing source and result artifacts are never overwritten.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

try:
    from ..leakage_free_rerun.tier15_common import (
        CLASS_NAMES,
        PUBLICATION_SEEDS,
        STUDENT_SPECS,
        TRAIN_CONFIG,
        StudentMLP,
        batched_probs,
        class_weights,
        classification_metrics,
        load_wsnds,
        set_seed,
        sha256_arrays,
        sha256_file,
        split_hashes,
        verified_feature_hashes,
    )
except ImportError:
    from experiments.wsnds.leakage_free_rerun.tier15_common import (  # type: ignore[no-redef]
        CLASS_NAMES,
        PUBLICATION_SEEDS,
        STUDENT_SPECS,
        TRAIN_CONFIG,
        StudentMLP,
        batched_probs,
        class_weights,
        classification_metrics,
        load_wsnds,
        set_seed,
        sha256_arrays,
        sha256_file,
        split_hashes,
        verified_feature_hashes,
    )


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
COMMON_SOURCE = (
    REPO_ROOT / "experiments" / "wsnds" / "leakage_free_rerun" / "tier15_common.py"
)
BASE_SOURCE = (
    REPO_ROOT
    / "experiments"
    / "wsnds"
    / "leakage_free_rerun"
    / "run_feature_group_10seed_confirmation.py"
)
DEFAULT_DATASET = REPO_ROOT / "data" / "wsnds" / "WSN-DS.csv"
DEFAULT_BASE_ROOT = (
    REPO_ROOT
    / "results"
    / "wsnds"
    / "confirmation_runs_v2"
    / "local_feature_group_10seed_20260811"
    / "feature_group_10seed"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "wsnds"
    / "evidence_completion_20260811"
    / "fgds_rfkd_hyperparameter_sensitivity_10seed_v1"
)

PROTOCOL_ID = "wsnds_fgds_rfkd_hyperparameter_sensitivity_10seed_v1"
BASE_PROTOCOL_ID = "wsnds_feature_group_split_train_only_scaler_10seed_v2"
TEMPERATURES = (1.0, 2.0, 4.0)
ALPHAS = (0.3, 0.5, 0.7)
ANCHOR_T = 4.0
ANCHOR_ALPHA = 0.7
GRID = tuple((temperature, alpha) for temperature in TEMPERATURES for alpha in ALPHAS)

EXPECTED_DATASET_SHA256 = (
    "c65d05b983a85753bd62b6f76c5739fc52fe0c14cbb7644255cee4742f5ff7c9"
)
EXPECTED_SPLIT_SHA256 = (
    "3d4061aa020122d4c5c5b2f7722de71e0c223c533869d3fdfa1f10784a0a0473"
)
EXPECTED_SCALER_SHA256 = (
    "5303fb570aeb82ffaf88e2d4cceda94a7611762f67c86761990e6a4f09af5dd6"
)
EXPECTED_BASE_ROOT_MANIFEST_SHA256 = (
    "6bb4a7d9456ea3bd93dbb479c4ea9c34b9061881179c943521b18db96b491e92"
)
EXPECTED_BASE_EXECUTION_SHA256 = (
    "21722565d0ec4b9e6e25a8e0a48617db0e7c4debff4891295dd4e5ef32ffb3ad"
)
EXPECTED_BASE_PREPROCESSING_SHA256 = (
    "2c2499caa28fec5e3e4253596631d3e8fd0dc8ac65623b3e9080c02f4bd30c22"
)
EXPECTED_BASE_AGGREGATE_SHA256 = (
    "91da6486726fcc19887e7c9fcf362982e1c70b02b9289d126572c819398748db"
)
EXPECTED_BASE_SPLIT_FILE_SHA256 = (
    "f2ddf428ea89e8a32809e9bce69f6df3bc046e4c66dfec220f2ffe72c49bc036"
)
EXPECTED_BASE_SCALER_FILE_SHA256 = (
    "641def12f200ff2922b0e3b8cc9525b5d123d4361354ffbede9c083852ccee6a"
)
EXPECTED_BASE_SOURCE_SHA256 = (
    "6b76bb0eba54691d7d1ac43324e95160ff21328374c991f0b34843d167a3546e"
)
EXPECTED_COMMON_SOURCE_SHA256 = (
    "78c79a377a57a2bc9ef6b532ed280b352cb5ba246d862409f874253cc768ff47"
)

PREDICTION_REPLAY_ATOL = 1e-6
BASE_CSV_REPLAY_ATOL = 5e-6
METRIC_ATOL = 2e-9
PROBABILITY_SUM_ATOL = 2e-6

ROOT_IMMUTABLE_FILES = {
    "base_lineage_contract.json",
    "minibatch_order_contract.json",
    "execution_contract.json",
    "executed_sensitivity_source.py",
    "bound_common_source.py",
    "bound_base_runner_source.py",
}
ROOT_FINAL_FILES = ROOT_IMMUTABLE_FILES | {
    "response_surface_aggregate.json",
    "artifact_manifest.json",
}
JOB_FILES_EXCLUDING_MANIFEST = {
    "checkpoint.pt",
    "validation_predictions.npz",
    "test_predictions.npz",
    "training_history.json",
    "completion.json",
}
SCRATCH_FILES_EXCLUDING_MANIFEST = {
    "validation_predictions.npz",
    "test_predictions.npz",
    "completion.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--confirm-run",
        action="store_true",
        help="Required to write artifacts and train the full factorial sensitivity run.",
    )
    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate the complete base lineage and synthetic checks without writing.",
    )
    mode.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Run only in-memory and temporary-directory checks without base data access.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Strictly verify completed artifacts and continue missing grid cells.",
    )
    return parser.parse_args()


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_bytes(path, canonical_json_bytes(payload))


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_torch_mapping(path: Path, *, rich: bool = False) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = torch.load(
            path, map_location="cpu", weights_only=False if rich else True
        )
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a mapping in {path}")
    return value


def state_dict_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    if not state or not all(torch.is_tensor(value) for value in state.values()):
        raise RuntimeError("State dictionary is empty or contains non-tensor values")
    return sha256_arrays(
        *[state[name].detach().cpu().numpy() for name in sorted(state)]
    )


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def resource_record(hidden_dims: tuple[int, int]) -> dict[str, Any]:
    model = StudentMLP(17, hidden_dims, len(CLASS_NAMES))
    parameters = count_parameters(model)
    linear_macs = (
        17 * hidden_dims[0]
        + hidden_dims[0] * hidden_dims[1]
        + hidden_dims[1] * len(CLASS_NAMES)
    )
    return {
        "trainable_parameters": parameters,
        "fp32_parameter_payload_bytes": parameters * 4,
        "linear_macs_per_record": int(linear_macs),
        "linear_flops_per_record_multiply_add_as_two": int(2 * linear_macs),
        "hidden_activation_values_per_record": int(sum(hidden_dims)),
        "scope": (
            "Analytical neural-core counts. They exclude framework, optimizer, "
            "serialization, firmware, preprocessing, communication, peak RAM, "
            "latency, energy, and RF-teacher storage."
        ),
    }


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 15
) -> float:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        selected = (confidence > lower) & (confidence <= upper)
        if index == 0:
            selected |= confidence == 0.0
        if np.any(selected):
            value += float(selected.mean()) * abs(
                float(correct[selected].mean())
                - float(confidence[selected].mean())
            )
    return float(value)


def metrics_with_resources(
    labels: np.ndarray,
    probabilities: np.ndarray,
    hidden_dims: tuple[int, int],
) -> dict[str, Any]:
    metrics = classification_metrics(labels, probabilities)
    metrics["ece_15_bin"] = expected_calibration_error(probabilities, labels)
    metrics.update(resource_record(hidden_dims))
    return metrics


def assert_nested_close(
    observed: Any,
    expected: Any,
    *,
    path: str,
    atol: float = METRIC_ATOL,
) -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(observed) != set(expected):
            raise RuntimeError(f"Mapping schema differs at {path}")
        for key in expected:
            assert_nested_close(
                observed[key], expected[key], path=f"{path}.{key}", atol=atol
            )
        return
    if isinstance(expected, list):
        if not isinstance(observed, list):
            raise RuntimeError(f"List schema differs at {path}")
        if not np.allclose(
            np.asarray(observed, dtype=np.float64),
            np.asarray(expected, dtype=np.float64),
            rtol=0.0,
            atol=atol,
        ):
            raise RuntimeError(f"Numeric list differs at {path}")
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=atol):
            raise RuntimeError(f"Numeric value differs at {path}: {observed} != {expected}")
        return
    if observed != expected:
        raise RuntimeError(f"Value differs at {path}: {observed!r} != {expected!r}")


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def environment_record(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor()
        ),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def require_hash(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    if observed.lower() != expected.lower():
        raise RuntimeError(f"{label} SHA-256 differs: {observed} != {expected}")


def require_disjoint_output(output: Path, protected: Iterable[Path]) -> Path:
    resolved = output.resolve()
    for item in protected:
        protected_path = item.resolve()
        overlaps = False
        try:
            resolved.relative_to(protected_path)
            overlaps = True
        except ValueError:
            try:
                protected_path.relative_to(resolved)
                overlaps = True
            except ValueError:
                pass
        if overlaps:
            raise RuntimeError(
                f"Output path overlaps protected source/evidence: {resolved}"
            )
    return resolved


def manifest_payload(root: Path, protocol_id: str) -> dict[str, Any]:
    manifest_path = root / "artifact_manifest.json"
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == manifest_path or path.name.endswith(".tmp"):
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "protocol_id": protocol_id,
        "status": "complete",
        "file_count_excluding_manifest": len(files),
        "files": files,
    }


def write_manifest(root: Path, protocol_id: str) -> dict[str, Any]:
    payload = manifest_payload(root, protocol_id)
    atomic_write_json(root / "artifact_manifest.json", payload)
    return payload


def verify_manifest(
    root: Path,
    protocol_id: str,
    *,
    expected_files_excluding_manifest: set[str] | None = None,
) -> dict[str, Any]:
    path = root / "artifact_manifest.json"
    manifest = load_json(path)
    if manifest.get("protocol_id") != protocol_id or manifest.get("status") != "complete":
        raise RuntimeError(f"Manifest contract differs: {path}")
    items = manifest.get("files")
    if not isinstance(items, list) or not items:
        raise RuntimeError(f"Manifest inventory is empty: {path}")
    declared: dict[str, Any] = {}
    for item in items:
        relative = item.get("path")
        if not isinstance(relative, str) or not relative:
            raise RuntimeError(f"Manifest contains an invalid path: {path}")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(f"Manifest path escapes root: {relative}")
        normalized = relative_path.as_posix()
        if normalized in declared:
            raise RuntimeError(f"Manifest path is duplicated: {relative}")
        declared[normalized] = item
    actual = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file() and item != path and not item.name.endswith(".tmp")
    }
    if set(declared) != actual:
        raise RuntimeError(f"Manifest inventory differs from disk: {path}")
    if expected_files_excluding_manifest is not None and actual != expected_files_excluding_manifest:
        raise RuntimeError(f"Expected inventory differs: {root}")
    if manifest.get("file_count_excluding_manifest") != len(actual):
        raise RuntimeError(f"Manifest count differs: {path}")
    for relative, item in declared.items():
        artifact = root / relative
        if (
            artifact.stat().st_size != item.get("size_bytes")
            or sha256_file(artifact) != item.get("sha256")
        ):
            raise RuntimeError(f"Manifest hash/size verification failed: {artifact}")
    return manifest


def prediction_payload(
    source_indices: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    return {
        "source_row_index": np.asarray(source_indices, dtype=np.int64),
        "true_label": np.asarray(labels, dtype=np.int64),
        "probability": probabilities,
        "predicted_label": probabilities.argmax(axis=1).astype(np.int64),
    }


def save_predictions(
    path: Path,
    source_indices: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    atomic_save_npz(path, **prediction_payload(source_indices, labels, probabilities))


def load_predictions(
    path: Path,
    expected_indices: np.ndarray,
    expected_labels: np.ndarray,
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "source_row_index",
            "true_label",
            "probability",
            "predicted_label",
        }
        if set(payload.files) != required:
            raise RuntimeError(f"Prediction schema differs: {path}")
        source_indices = payload["source_row_index"]
        labels = payload["true_label"]
        probabilities = payload["probability"]
        predictions = payload["predicted_label"]
    if source_indices.dtype != np.int64 or not np.array_equal(source_indices, expected_indices):
        raise RuntimeError(f"Prediction source indices differ: {path}")
    if labels.dtype != np.int64 or not np.array_equal(labels, expected_labels):
        raise RuntimeError(f"Prediction labels differ: {path}")
    expected_shape = (len(expected_labels), len(CLASS_NAMES))
    if probabilities.dtype != np.float32 or probabilities.shape != expected_shape:
        raise RuntimeError(f"Prediction probability shape/dtype differs: {path}")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
        raise RuntimeError(f"Prediction probabilities are invalid: {path}")
    if not np.allclose(
        probabilities.sum(axis=1), 1.0, rtol=0.0, atol=PROBABILITY_SUM_ATOL
    ):
        raise RuntimeError(f"Prediction probabilities do not sum to one: {path}")
    if predictions.dtype != np.int64 or not np.array_equal(
        predictions, probabilities.argmax(axis=1)
    ):
        raise RuntimeError(f"Prediction argmax differs: {path}")
    return probabilities


def load_base_prediction_csv(
    path: Path,
    expected_indices: np.ndarray,
    expected_labels: np.ndarray,
) -> np.ndarray:
    frame = pd.read_csv(path)
    expected_columns = [
        "source_row_index",
        "true_label",
        "predicted_label",
        *[
            f"probability_{index}_{name}"
            for index, name in enumerate(CLASS_NAMES)
        ],
    ]
    if frame.columns.tolist() != expected_columns:
        raise RuntimeError(f"Base prediction schema differs: {path}")
    if not np.array_equal(
        frame["source_row_index"].to_numpy(dtype=np.int64), expected_indices
    ):
        raise RuntimeError(f"Base prediction source indices differ: {path}")
    if not np.array_equal(
        frame["true_label"].to_numpy(dtype=np.int64), expected_labels
    ):
        raise RuntimeError(f"Base prediction labels differ: {path}")
    probabilities = frame[expected_columns[3:]].to_numpy(dtype=np.float64)
    predictions = frame["predicted_label"].to_numpy(dtype=np.int64)
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
        raise RuntimeError(f"Base prediction probabilities are invalid: {path}")
    if not np.allclose(
        probabilities.sum(axis=1), 1.0, rtol=0.0, atol=PROBABILITY_SUM_ATOL
    ):
        raise RuntimeError(f"Base prediction probabilities do not sum to one: {path}")
    if not np.array_equal(predictions, probabilities.argmax(axis=1)):
        raise RuntimeError(f"Base prediction argmax differs: {path}")
    return probabilities


def build_data_context(dataset_csv: Path, base_root: Path) -> dict[str, Any]:
    dataset_csv = dataset_csv.resolve()
    require_hash(dataset_csv, EXPECTED_DATASET_SHA256, "WSN-DS dataset")
    dataset = load_wsnds(dataset_csv)
    if dataset["dataset_sha256"] != EXPECTED_DATASET_SHA256:
        raise RuntimeError("Loaded WSN-DS dataset hash differs")

    preprocessing = load_json(base_root / "preprocessing_contract.json")
    execution = load_json(base_root / "execution_contract.json")
    split_path = base_root / "split_indices.npz"
    scaler_path = base_root / "scaler_parameters.npz"
    with np.load(split_path, allow_pickle=False) as payload:
        if set(payload.files) != {
            "train_indices",
            "validation_indices",
            "test_indices",
        }:
            raise RuntimeError("Base split-index schema differs")
        indices = {
            name: payload[f"{name}_indices"].astype(np.int64, copy=True)
            for name in ["train", "validation", "test"]
        }
    all_indices = np.concatenate([indices[name] for name in ["train", "validation", "test"]])
    if (
        len(all_indices) != len(dataset["labels"])
        or len(np.unique(all_indices)) != len(all_indices)
        or not np.array_equal(np.sort(all_indices), np.arange(len(all_indices)))
    ):
        raise RuntimeError("Base split indices are not an exact partition of WSN-DS")
    observed_split_sha = sha256_arrays(
        indices["train"], indices["validation"], indices["test"]
    )
    if observed_split_sha != EXPECTED_SPLIT_SHA256:
        raise RuntimeError("Base split semantic SHA-256 differs")

    features = dataset["features"]
    labels = dataset["labels"]
    split: dict[str, Any] = {}
    for name in ["train", "validation", "test"]:
        index = indices[name]
        split[f"{name}_indices"] = index
        split[f"X_{name}_raw"] = features[index]
        split[f"y_{name}"] = labels[index]
    observed_split_hashes = split_hashes(split)
    if observed_split_hashes != preprocessing.get("split_hashes"):
        raise RuntimeError("Reconstructed raw split hashes differ from base")

    feature_hashes = verified_feature_hashes(features)
    group_sets = {
        name: set(map(int, feature_hashes[indices[name]]))
        for name in ["train", "validation", "test"]
    }
    overlap = {
        "train_validation_feature_overlap": len(
            group_sets["train"] & group_sets["validation"]
        ),
        "train_test_feature_overlap": len(group_sets["train"] & group_sets["test"]),
        "validation_test_feature_overlap": len(
            group_sets["validation"] & group_sets["test"]
        ),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Feature groups cross the clean base split: {overlap}")
    expected_audit = preprocessing.get("feature_overlap_audit", {})
    for key, value in overlap.items():
        if expected_audit.get(key) != value:
            raise RuntimeError(f"Base feature-overlap audit differs for {key}")
    if expected_audit.get("num_feature_groups") != len(np.unique(feature_hashes)):
        raise RuntimeError("Base feature-group count differs")
    group_labels = pd.DataFrame(
        {"group": feature_hashes, "label": labels}
    ).drop_duplicates()
    conflicting_groups = int(
        (group_labels.groupby("group", sort=False)["label"].nunique() > 1).sum()
    )
    if expected_audit.get("conflicting_label_feature_groups") != conflicting_groups:
        raise RuntimeError("Base conflicting-label feature-group count differs")
    if expected_audit.get("group_stratification_label") != (
        "majority label; smallest class index breaks ties"
    ):
        raise RuntimeError("Base feature-group stratification rule differs")

    with np.load(scaler_path, allow_pickle=False) as payload:
        if set(payload.files) != {"mean", "scale", "var", "n_samples_seen"}:
            raise RuntimeError("Base scaler schema differs")
        saved_scaler = {name: payload[name].copy() for name in payload.files}
    scaler_sha = sha256_arrays(
        saved_scaler["mean"], saved_scaler["scale"], saved_scaler["var"]
    )
    if scaler_sha != EXPECTED_SCALER_SHA256:
        raise RuntimeError("Base scaler semantic SHA-256 differs")
    independently_fit = StandardScaler().fit(split["X_train_raw"])
    for name in ["mean", "scale", "var"]:
        if not np.array_equal(
            np.asarray(getattr(independently_fit, f"{name}_"), dtype=np.float64),
            np.asarray(saved_scaler[name], dtype=np.float64),
        ):
            raise RuntimeError(f"Saved scaler {name} is not the train-only fit")
    if int(np.asarray(saved_scaler["n_samples_seen"]).reshape(-1)[0]) != len(indices["train"]):
        raise RuntimeError("Saved scaler fit-row count differs")

    scaler = StandardScaler()
    scaler.mean_ = np.asarray(saved_scaler["mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(saved_scaler["scale"], dtype=np.float64)
    scaler.var_ = np.asarray(saved_scaler["var"], dtype=np.float64)
    scaler.n_features_in_ = features.shape[1]
    scaler.n_samples_seen_ = int(saved_scaler["n_samples_seen"].reshape(-1)[0])
    scaled = {
        name: scaler.transform(split[f"X_{name}_raw"]).astype(np.float32, copy=False)
        for name in ["train", "validation", "test"]
    }
    transformed_hashes = {
        name: sha256_arrays(scaled[name])
        for name in ["train", "validation", "test"]
    }
    if transformed_hashes != preprocessing.get("transformed_split_hashes"):
        raise RuntimeError("Saved-scaler transformed split hashes differ from base")

    required_contract = {
        "protocol_id": BASE_PROTOCOL_ID,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "split_indices_sha256": EXPECTED_SPLIT_SHA256,
        "scaler_sha256": EXPECTED_SCALER_SHA256,
        "seeds": list(PUBLICATION_SEEDS),
        "students": {name: list(value) for name, value in STUDENT_SPECS.items()},
        "routes": ["scratch", "rf_kd"],
    }
    for key, expected in required_contract.items():
        if execution.get(key) != expected:
            raise RuntimeError(f"Base execution contract differs for {key}")
    if execution.get("kd_hyperparameters") != {
        "T": ANCHOR_T,
        "alpha": ANCHOR_ALPHA,
    }:
        raise RuntimeError("Base RF-KD hyperparameter declaration differs")
    hyperparameter_source = execution.get("kd_hyperparameter_source", "")
    if (
        "preliminary MLP-teacher validation grid" not in hyperparameter_source
        or "not an RF-KD optimum" not in hyperparameter_source
    ):
        raise RuntimeError("Base hyperparameter-source limitation is absent or changed")
    if execution.get("training_config") != TRAIN_CONFIG:
        raise RuntimeError("Base student training configuration differs")
    if execution.get("teacher_calibration_strategy") != (
        "stratified_group_kfold_with_zero_exact_feature_group_overlap"
    ):
        raise RuntimeError("Base RF calibration strategy differs")
    if preprocessing.get("scaler_fit_partition") != "train only":
        raise RuntimeError("Base scaler was not declared train-only")
    if preprocessing.get("feature_names") != dataset["feature_names"]:
        raise RuntimeError("Base feature order differs")
    if preprocessing.get("class_names") != CLASS_NAMES:
        raise RuntimeError("Base class order differs")
    expected_sizes = {
        name: len(indices[name]) for name in ["train", "validation", "test"]
    }
    if preprocessing.get("split_sizes") != expected_sizes:
        raise RuntimeError("Base split-size contract differs")
    expected_class_counts = {
        name: np.bincount(
            split[f"y_{name}"], minlength=len(CLASS_NAMES)
        ).astype(int).tolist()
        for name in ["train", "validation", "test"]
    }
    if preprocessing.get("split_class_counts") != expected_class_counts:
        raise RuntimeError("Base split class counts differ")

    return {
        "dataset": dataset,
        "preprocessing": preprocessing,
        "execution": execution,
        "indices": indices,
        "split": split,
        "scaled": scaled,
        "feature_hashes": feature_hashes,
        "transformed_hashes": transformed_hashes,
    }


def expected_base_seed_files() -> set[str]:
    files = {
        "artifact_manifest.json",
        "seed_completion.json",
        "rf_train_probabilities.npy",
        "RF_teacher_test_predictions.csv",
    }
    for student in STUDENT_SPECS:
        files.update(
            {
                f"{student}_Small_MLP_scratch_fp32.pt",
                f"{student}_Small_MLP_scratch_artifact.pt",
                f"{student}_Small_MLP_scratch_test_predictions.csv",
                f"{student}_KD_from_RF_fp32.pt",
                f"{student}_KD_from_RF_artifact.pt",
                f"{student}_KD_from_RF_test_predictions.csv",
            }
        )
    return files


def verify_base_neural_route(
    seed_root: Path,
    seed: int,
    student: str,
    route: str,
    result: dict[str, Any],
    context: dict[str, Any],
    verification_device: torch.device,
    expected_teacher_provenance: dict[str, Any] | None,
) -> dict[str, Any]:
    hidden_dims = STUDENT_SPECS[student]
    if route == "scratch":
        prefix = f"{student}_Small_MLP_scratch"
    else:
        prefix = f"{student}_KD_from_RF"
    expected_names = {
        "plain_state_dict": f"{prefix}_fp32.pt",
        "rich_artifact": f"{prefix}_artifact.pt",
        "test_predictions": f"{prefix}_test_predictions.csv",
    }
    if result.get("route") != route:
        raise RuntimeError(f"Base route label differs: seed {seed}, {student}, {route}")
    for key, expected in expected_names.items():
        if result.get(key) != expected:
            raise RuntimeError(f"Base route filename differs: {seed}, {student}, {key}")
    plain_path = seed_root / expected_names["plain_state_dict"]
    rich_path = seed_root / expected_names["rich_artifact"]
    prediction_path = seed_root / expected_names["test_predictions"]
    for path_key, hash_key in [
        (plain_path, "plain_state_dict_sha256"),
        (rich_path, "rich_artifact_sha256"),
        (prediction_path, "test_predictions_sha256"),
    ]:
        if sha256_file(path_key) != result.get(hash_key):
            raise RuntimeError(f"Base route artifact hash differs: {path_key}")

    plain = load_torch_mapping(plain_path)
    rich = load_torch_mapping(rich_path, rich=True)
    rich_state = rich.get("state_dict")
    if not isinstance(rich_state, dict) or set(rich_state) != set(plain):
        raise RuntimeError(f"Base rich/plain state schema differs: {rich_path}")
    for name in plain:
        if not torch.equal(plain[name], rich_state[name]):
            raise RuntimeError(f"Base rich/plain state differs: {rich_path}:{name}")
    trained_hash = state_dict_sha256(plain)
    if trained_hash != result.get("trained_state_sha256"):
        raise RuntimeError(f"Base trained-state content hash differs: {plain_path}")
    required_rich = {
        "protocol_id": BASE_PROTOCOL_ID,
        "seed": seed,
        "student": student,
        "route": route,
        "input_dim": 17,
        "hidden_dims": list(hidden_dims),
        "num_classes": len(CLASS_NAMES),
        "feature_names": context["dataset"]["feature_names"],
        "class_names": CLASS_NAMES,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "split_hashes": context["preprocessing"]["split_hashes"],
        "scaler_sha256": EXPECTED_SCALER_SHA256,
        "feature_overlap_audit": context["preprocessing"]["feature_overlap_audit"],
        "initial_state_sha256": result["initial_state_sha256"],
        "trained_state_sha256": result["trained_state_sha256"],
        "training_config": TRAIN_CONFIG,
        "metrics": result["metrics"],
    }
    for key, expected in required_rich.items():
        if rich.get(key) != expected:
            raise RuntimeError(f"Base rich artifact differs at {rich_path}:{key}")
    if route == "scratch":
        if (
            rich.get("kd_hyperparameters") is not None
            or rich.get("teacher_soft_target_provenance") is not None
        ):
            raise RuntimeError(f"Base scratch artifact contains KD provenance: {rich_path}")
    else:
        if rich.get("kd_hyperparameters") != {"T": ANCHOR_T, "alpha": ANCHOR_ALPHA}:
            raise RuntimeError(f"Base RF-KD hyperparameters differ: {rich_path}")
        if rich.get("teacher_soft_target_provenance") != expected_teacher_provenance:
            raise RuntimeError(f"Base RF-KD teacher provenance differs: {rich_path}")

    model = StudentMLP(17, hidden_dims, len(CLASS_NAMES)).to(verification_device)
    model.load_state_dict(plain, strict=True)
    stored_probabilities = load_base_prediction_csv(
        prediction_path,
        context["indices"]["test"],
        context["split"]["y_test"],
    )
    recomputed_metrics = classification_metrics(
        context["split"]["y_test"], stored_probabilities
    )
    assert_nested_close(
        recomputed_metrics,
        result["metrics"],
        path=f"base.{seed}.{student}.{route}.metrics",
        atol=1e-12,
    )
    replayed = batched_probs(
        model, torch.from_numpy(context["scaled"]["test"]), verification_device
    )
    max_delta = float(np.max(np.abs(replayed - stored_probabilities)))
    if max_delta > BASE_CSV_REPLAY_ATOL:
        raise RuntimeError(
            f"Base checkpoint does not replay stored probabilities: {plain_path}, {max_delta}"
        )
    if not np.array_equal(replayed.argmax(axis=1), stored_probabilities.argmax(axis=1)):
        raise RuntimeError(f"Base checkpoint replay changes predictions: {plain_path}")
    return {
        "plain_state_dict": str(plain_path.resolve()),
        "plain_state_dict_sha256": sha256_file(plain_path),
        "rich_artifact": str(rich_path.resolve()),
        "rich_artifact_sha256": sha256_file(rich_path),
        "test_predictions": str(prediction_path.resolve()),
        "test_predictions_sha256": sha256_file(prediction_path),
        "initial_state_sha256": result["initial_state_sha256"],
        "trained_state_sha256": result["trained_state_sha256"],
        "metrics": result["metrics"],
    }


def verify_base_seed(
    base_root: Path,
    seed: int,
    context: dict[str, Any],
    verification_device: torch.device,
) -> dict[str, Any]:
    seed_root = base_root / f"seed_{seed}"
    actual_names = {path.name for path in seed_root.iterdir() if path.is_file()}
    if actual_names != expected_base_seed_files():
        raise RuntimeError(f"Base seed file schema differs: {seed_root}")
    verify_manifest(
        seed_root,
        BASE_PROTOCOL_ID,
        expected_files_excluding_manifest=expected_base_seed_files()
        - {"artifact_manifest.json"},
    )
    completion_path = seed_root / "seed_completion.json"
    manifest_path = seed_root / "artifact_manifest.json"
    completion = load_json(completion_path)
    required = {
        "protocol_id": BASE_PROTOCOL_ID,
        "seed": seed,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "split_indices_sha256": EXPECTED_SPLIT_SHA256,
        "scaler_sha256": EXPECTED_SCALER_SHA256,
        "execution_contract_sha256": EXPECTED_BASE_EXECUTION_SHA256,
        "status": "complete",
    }
    for key, expected in required.items():
        if completion.get(key) != expected:
            raise RuntimeError(f"Base seed completion differs: {seed}:{key}")

    probability_path = seed_root / "rf_train_probabilities.npy"
    rf_probabilities = np.load(probability_path, allow_pickle=False)
    expected_shape = (len(context["indices"]["train"]), len(CLASS_NAMES))
    if rf_probabilities.shape != expected_shape or rf_probabilities.dtype != np.float32:
        raise RuntimeError(f"Base RF target shape/dtype differs: {probability_path}")
    if not np.isfinite(rf_probabilities).all() or np.any(rf_probabilities < 0.0):
        raise RuntimeError(f"Base RF probabilities are invalid: {probability_path}")
    if not np.allclose(
        rf_probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6
    ):
        raise RuntimeError(f"Base RF probabilities do not sum to one: {probability_path}")
    content_sha = sha256_arrays(rf_probabilities)
    provenance = completion.get("teacher_soft_target_provenance", {})
    if provenance.get("rf_seed") != seed:
        raise RuntimeError(f"Base RF target seed differs: {seed}")
    if provenance.get("train_probability_content_sha256") != content_sha:
        raise RuntimeError(f"Base RF target content hash differs: {seed}")
    if provenance.get("source_type") != "fresh_calibrated_rf_soft_targets":
        raise RuntimeError(f"Base RF target source differs: {seed}")
    if completion.get("teacher_config") != context["execution"].get("teacher_config"):
        raise RuntimeError(f"Base RF teacher configuration differs: {seed}")
    if provenance.get("rf_config") != completion.get("teacher_config"):
        raise RuntimeError(f"Base RF soft-target configuration differs: {seed}")
    calibration = provenance.get("calibration_audit", {})
    if (
        calibration.get("strategy") != "stratified_group_kfold"
        or calibration.get("folds") != 3
        or calibration.get("group_overlap_per_fold") != [0, 0, 0]
        or calibration.get("unique_groups")
        != len(np.unique(context["feature_hashes"][context["indices"]["train"]]))
    ):
        raise RuntimeError(f"Base RF group-calibration audit differs: {seed}")

    students: dict[str, Any] = {}
    results = completion.get("student_results", {})
    expected_result_keys = {
        f"{student}_{route}"
        for student in STUDENT_SPECS
        for route in ["scratch", "rf_kd"]
    }
    if set(results) != expected_result_keys:
        raise RuntimeError(f"Base student-result schema differs: seed {seed}")
    for student, hidden_dims in STUDENT_SPECS.items():
        scratch_result = results[f"{student}_scratch"]
        kd_result = results[f"{student}_rf_kd"]
        if scratch_result["initial_state_sha256"] != kd_result["initial_state_sha256"]:
            raise RuntimeError(f"Base scratch/KD initial states differ: {seed}, {student}")
        set_seed(seed)
        initial_model = StudentMLP(17, hidden_dims, len(CLASS_NAMES))
        recreated_initial_hash = state_dict_sha256(state_dict_cpu(initial_model))
        if recreated_initial_hash != scratch_result["initial_state_sha256"]:
            raise RuntimeError(f"Base student initial state cannot be recreated: {seed}, {student}")
        del initial_model
        students[student] = {
            "initial_state_sha256": recreated_initial_hash,
            "scratch": verify_base_neural_route(
                seed_root,
                seed,
                student,
                "scratch",
                scratch_result,
                context,
                verification_device,
                None,
            ),
            "rf_kd_anchor": verify_base_neural_route(
                seed_root,
                seed,
                student,
                "rf_kd",
                kd_result,
                context,
                verification_device,
                provenance,
            ),
        }
    return {
        "seed": seed,
        "seed_root_recorded": str(seed_root.resolve()),
        "seed_completion_sha256": sha256_file(completion_path),
        "seed_manifest_sha256": sha256_file(manifest_path),
        "rf_train_probabilities": str(probability_path.resolve()),
        "rf_train_probability_file_sha256": sha256_file(probability_path),
        "rf_train_probability_content_sha256": content_sha,
        "rf_target_shape": list(rf_probabilities.shape),
        "rf_target_dtype": str(rf_probabilities.dtype),
        "teacher_soft_target_provenance": provenance,
        "students": students,
    }


def validate_base_lineage(
    base_root: Path,
    dataset_csv: Path,
    verification_device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_root = base_root.resolve()
    if not base_root.is_dir():
        raise FileNotFoundError(base_root)
    pinned = [
        (base_root / "artifact_manifest.json", EXPECTED_BASE_ROOT_MANIFEST_SHA256, "base root manifest"),
        (base_root / "execution_contract.json", EXPECTED_BASE_EXECUTION_SHA256, "base execution contract"),
        (base_root / "preprocessing_contract.json", EXPECTED_BASE_PREPROCESSING_SHA256, "base preprocessing contract"),
        (base_root / "aggregate_results.json", EXPECTED_BASE_AGGREGATE_SHA256, "base aggregate"),
        (base_root / "split_indices.npz", EXPECTED_BASE_SPLIT_FILE_SHA256, "base split file"),
        (base_root / "scaler_parameters.npz", EXPECTED_BASE_SCALER_FILE_SHA256, "base scaler file"),
        (BASE_SOURCE, EXPECTED_BASE_SOURCE_SHA256, "base runner source"),
        (COMMON_SOURCE, EXPECTED_COMMON_SOURCE_SHA256, "shared training source"),
    ]
    for path, expected, label in pinned:
        require_hash(path, expected, label)
    expected_top_level = {
        "artifact_manifest.json",
        "execution_contract.json",
        "preprocessing_contract.json",
        "aggregate_results.json",
        "split_indices.npz",
        "scaler_parameters.npz",
        *[f"seed_{seed}" for seed in PUBLICATION_SEEDS],
    }
    if {path.name for path in base_root.iterdir()} != expected_top_level:
        raise RuntimeError("Base root top-level inventory differs")
    verify_manifest(base_root, BASE_PROTOCOL_ID)
    aggregate = load_json(base_root / "aggregate_results.json")
    if (
        aggregate.get("protocol_id") != BASE_PROTOCOL_ID
        or aggregate.get("status") != "complete"
        or aggregate.get("seeds") != list(PUBLICATION_SEEDS)
        or aggregate.get("seed_count") != len(PUBLICATION_SEEDS)
    ):
        raise RuntimeError("Base aggregate contract differs")

    context = build_data_context(dataset_csv, base_root)
    seed_bindings = {
        str(seed): verify_base_seed(base_root, seed, context, verification_device)
        for seed in PUBLICATION_SEEDS
    }
    lineage = {
        "base_protocol_id": BASE_PROTOCOL_ID,
        "base_root_recorded": str(base_root),
        "base_root_manifest_sha256": EXPECTED_BASE_ROOT_MANIFEST_SHA256,
        "base_execution_contract_sha256": EXPECTED_BASE_EXECUTION_SHA256,
        "base_preprocessing_contract_sha256": EXPECTED_BASE_PREPROCESSING_SHA256,
        "base_aggregate_sha256": EXPECTED_BASE_AGGREGATE_SHA256,
        "base_split_file_sha256": EXPECTED_BASE_SPLIT_FILE_SHA256,
        "base_scaler_file_sha256": EXPECTED_BASE_SCALER_FILE_SHA256,
        "base_runner_source_sha256": EXPECTED_BASE_SOURCE_SHA256,
        "common_source_sha256": EXPECTED_COMMON_SOURCE_SHA256,
        "dataset_path_recorded": str(dataset_csv.resolve()),
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "split_indices_sha256": EXPECTED_SPLIT_SHA256,
        "scaler_sha256": EXPECTED_SCALER_SHA256,
        "split_hashes": context["preprocessing"]["split_hashes"],
        "transformed_split_hashes": context["transformed_hashes"],
        "feature_overlap_audit": context["preprocessing"]["feature_overlap_audit"],
        "seeds": list(PUBLICATION_SEEDS),
        "seed_bindings": seed_bindings,
    }
    lineage["lineage_semantic_sha256"] = canonical_sha256(lineage)
    return context, lineage


def draw_epoch_order(size: int) -> tuple[int, int, np.ndarray]:
    discarded = int(torch.empty((), dtype=torch.int64).random_().item())
    sampler_seed = int(torch.empty((), dtype=torch.int64).random_().item())
    generator = torch.Generator()
    generator.manual_seed(sampler_seed)
    order = torch.randperm(size, generator=generator)
    return discarded, sampler_seed, order


def build_minibatch_order_contract(
    lineage: dict[str, Any], train_rows: int, config: dict[str, Any]
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    epochs = int(config["epochs"])
    batch_size = int(config["batch_size"])
    for seed in PUBLICATION_SEEDS:
        records[str(seed)] = {}
        for student, hidden_dims in STUDENT_SPECS.items():
            set_seed(seed)
            model = StudentMLP(17, hidden_dims, len(CLASS_NAMES))
            initial_hash = state_dict_sha256(state_dict_cpu(model))
            expected_initial = lineage["seed_bindings"][str(seed)]["students"][student][
                "initial_state_sha256"
            ]
            if initial_hash != expected_initial:
                raise RuntimeError(f"Schedule initial-state binding differs: {seed}, {student}")
            del model
            epoch_records = []
            for epoch in range(epochs):
                discarded, sampler_seed, order = draw_epoch_order(train_rows)
                epoch_records.append(
                    {
                        "epoch": epoch + 1,
                        "discarded_global_rng_draw": discarded,
                        "sampler_seed": sampler_seed,
                        "index_order_sha256": sha256_arrays(order.numpy()),
                        "batch_count": int(math.ceil(train_rows / batch_size)),
                    }
                )
            records[str(seed)][student] = {
                "initial_state_sha256": initial_hash,
                "train_rows": train_rows,
                "batch_size": batch_size,
                "maximum_epochs": epochs,
                "epochs": epoch_records,
            }
    payload = {
        "protocol_id": PROTOCOL_ID,
        "source_semantics": (
            "The order recreates tier15_common.shuffled_batches after set_seed(seed) "
            "and StudentMLP construction. The first int64 draw is discarded, the "
            "second seeds a CPU torch.Generator, and torch.randperm defines the row order."
        ),
        "common_source_sha256": EXPECTED_COMMON_SOURCE_SHA256,
        "torch_version": torch.__version__,
        "train_rows": train_rows,
        "training_config": config,
        "records": records,
    }
    payload["order_contract_semantic_sha256"] = canonical_sha256(payload)
    return payload


def checked_epoch_batches(
    tensors: tuple[torch.Tensor, ...],
    batch_size: int,
    expected_epoch: dict[str, Any],
) -> Iterable[tuple[torch.Tensor, ...]]:
    if not tensors or any(len(tensor) != len(tensors[0]) for tensor in tensors):
        raise RuntimeError("Training tensors have unequal row counts")
    discarded, sampler_seed, order_cpu = draw_epoch_order(len(tensors[0]))
    actual = {
        "epoch": expected_epoch["epoch"],
        "discarded_global_rng_draw": discarded,
        "sampler_seed": sampler_seed,
        "index_order_sha256": sha256_arrays(order_cpu.numpy()),
        "batch_count": int(math.ceil(len(tensors[0]) / batch_size)),
    }
    if actual != expected_epoch:
        raise RuntimeError(
            f"Actual minibatch order differs from declared contract at epoch "
            f"{expected_epoch['epoch']}"
        )
    order = order_cpu.to(tensors[0].device)
    for start in range(0, len(order), batch_size):
        index = order[start : start + batch_size]
        yield tuple(tensor[index] for tensor in tensors)


def train_rf_kd_cell(
    model: StudentMLP,
    rf_probabilities: np.ndarray,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_validation: torch.Tensor,
    y_validation: torch.Tensor,
    weights: torch.Tensor,
    device: torch.device,
    *,
    temperature: float,
    alpha: float,
    schedule: dict[str, Any],
    config: dict[str, Any],
) -> tuple[StudentMLP, dict[str, Any]]:
    expected_shape = (len(X_train), len(CLASS_NAMES))
    if rf_probabilities.shape != expected_shape or rf_probabilities.dtype != np.float32:
        raise RuntimeError("RF soft-target shape/dtype differs")
    if temperature not in TEMPERATURES or alpha not in ALPHAS:
        raise RuntimeError("Requested cell is outside the declared factorial grid")
    if schedule["train_rows"] != len(X_train) or schedule["batch_size"] != config["batch_size"]:
        raise RuntimeError("Minibatch schedule does not match training data/config")

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["epochs"])
    )
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    X_train_d = X_train.to(device)
    y_train_d = y_train.to(device)
    X_validation_d = X_validation.to(device)
    y_validation_np = y_validation.numpy()
    raw = torch.tensor(rf_probabilities, dtype=torch.float32, device=device)
    teacher_targets = F.softmax(
        torch.log(raw.clamp(min=1e-8)) / float(temperature), dim=1
    ).detach()
    teacher_target_hash = sha256_arrays(teacher_targets.detach().cpu().numpy())

    best_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch: int | None = None
    stale = 0
    history = []
    started = time.perf_counter()
    for epoch in range(int(config["epochs"])):
        model.train()
        loss_sum = 0.0
        example_count = 0
        expected_epoch = schedule["epochs"][epoch]
        for X_batch, y_batch, teacher_batch in checked_epoch_batches(
            (X_train_d, y_train_d, teacher_targets),
            int(config["batch_size"]),
            expected_epoch,
        ):
            optimizer.zero_grad()
            logits = model(X_batch)
            kd_loss = F.kl_div(
                F.log_softmax(logits / float(temperature), dim=1),
                teacher_batch,
                reduction="batchmean",
            ) * (float(temperature) ** 2)
            ce_loss = criterion(logits, y_batch)
            loss = float(alpha) * kd_loss + (1.0 - float(alpha)) * ce_loss
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(X_batch)
            example_count += len(X_batch)
        scheduler.step()
        validation_probabilities = batched_probs(model, X_validation_d, device)
        validation_predictions = validation_probabilities.argmax(axis=1)
        validation_macro_f1 = float(
            f1_score(
                y_validation_np,
                validation_predictions,
                average="macro",
                zero_division=0,
            )
        )
        improved = validation_macro_f1 > best_f1
        if improved:
            best_f1 = validation_macro_f1
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
        history.append(
            {
                "epoch": epoch + 1,
                "mean_training_loss": loss_sum / max(example_count, 1),
                "validation_macro_f1": validation_macro_f1,
                "learning_rate_after_scheduler_step": float(
                    optimizer.param_groups[0]["lr"]
                ),
                "improved_checkpoint": improved,
                "stale_epochs": stale,
                "minibatch_order": expected_epoch,
            }
        )
        if stale >= int(config["patience"]):
            break
    if best_state is None or best_epoch is None:
        raise RuntimeError("RF-KD sensitivity training produced no validation checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "temperature": float(temperature),
        "alpha": float(alpha),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_macro_f1_during_training": best_f1,
        "early_stopped": len(history) < int(config["epochs"]),
        "teacher_target_content_sha256": teacher_target_hash,
        "minibatch_order_contract_prefix": schedule["epochs"][: len(history)],
        "history": history,
        "wall_seconds": time.perf_counter() - started,
    }


def cell_id(temperature: float, alpha: float) -> str:
    return f"T{int(temperature)}_alpha{int(round(alpha * 10)):02d}"


def job_root(seed_root: Path, student: str, temperature: float, alpha: float) -> Path:
    return seed_root / student / cell_id(temperature, alpha)


def scratch_root(seed_root: Path, student: str) -> Path:
    return seed_root / student / "scratch_reference"


def checkpoint_payload(
    *,
    execution_contract_sha256: str,
    seed: int,
    student: str,
    hidden_dims: tuple[int, int],
    temperature: float,
    alpha: float,
    initial_state_sha256: str,
    state: dict[str, torch.Tensor],
    base_binding: dict[str, Any],
    schedule: dict[str, Any],
    training: dict[str, Any],
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "analysis_role": "factorial_sensitivity_only",
        "execution_contract_sha256": execution_contract_sha256,
        "seed": seed,
        "student": student,
        "input_dim": 17,
        "hidden_dims": list(hidden_dims),
        "num_classes": len(CLASS_NAMES),
        "temperature": float(temperature),
        "alpha": float(alpha),
        "initial_state_sha256": initial_state_sha256,
        "trained_state_sha256": state_dict_sha256(state),
        "base_rf_target_file_sha256": base_binding[
            "rf_train_probability_file_sha256"
        ],
        "base_rf_target_content_sha256": base_binding[
            "rf_train_probability_content_sha256"
        ],
        "minibatch_order_contract_semantic_sha256": schedule[
            "order_contract_semantic_sha256"
        ],
        "training_config": TRAIN_CONFIG,
        "training": training,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "resource_counts": resource_record(hidden_dims),
        "state_dict": state,
    }


def create_scratch_reference(
    root: Path,
    context: dict[str, Any],
    lineage: dict[str, Any],
    execution_contract_sha256: str,
    seed: int,
    student: str,
    device: torch.device,
) -> dict[str, Any]:
    hidden_dims = STUDENT_SPECS[student]
    base = lineage["seed_bindings"][str(seed)]["students"][student]["scratch"]
    state = load_torch_mapping(Path(base["plain_state_dict"]))
    model = StudentMLP(17, hidden_dims, len(CLASS_NAMES)).to(device)
    model.load_state_dict(state, strict=True)
    validation_probabilities = batched_probs(
        model, torch.from_numpy(context["scaled"]["validation"]), device
    )
    test_probabilities = batched_probs(
        model, torch.from_numpy(context["scaled"]["test"]), device
    )
    base_test_probabilities = load_base_prediction_csv(
        Path(base["test_predictions"]),
        context["indices"]["test"],
        context["split"]["y_test"],
    )
    max_delta = float(np.max(np.abs(test_probabilities - base_test_probabilities)))
    if max_delta > BASE_CSV_REPLAY_ATOL or not np.array_equal(
        test_probabilities.argmax(axis=1), base_test_probabilities.argmax(axis=1)
    ):
        raise RuntimeError(f"Scratch reference replay differs: {seed}, {student}")
    validation_metrics = metrics_with_resources(
        context["split"]["y_validation"], validation_probabilities, hidden_dims
    )
    test_metrics = metrics_with_resources(
        context["split"]["y_test"], test_probabilities, hidden_dims
    )
    assert_nested_close(
        {key: test_metrics[key] for key in base["metrics"]},
        base["metrics"],
        path=f"scratch_reference.{seed}.{student}.base_metrics",
        atol=1e-12,
    )
    save_predictions(
        root / "validation_predictions.npz",
        context["indices"]["validation"],
        context["split"]["y_validation"],
        validation_probabilities,
    )
    save_predictions(
        root / "test_predictions.npz",
        context["indices"]["test"],
        context["split"]["y_test"],
        test_probabilities,
    )
    completion = {
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "role": "persisted_scratch_checkpoint_reference_without_retraining",
        "execution_contract_sha256": execution_contract_sha256,
        "seed": seed,
        "student": student,
        "source_checkpoint": base["plain_state_dict"],
        "source_checkpoint_sha256": base["plain_state_dict_sha256"],
        "source_rich_artifact_sha256": base["rich_artifact_sha256"],
        "source_test_predictions_sha256": base["test_predictions_sha256"],
        "initial_state_sha256": base["initial_state_sha256"],
        "trained_state_sha256": base["trained_state_sha256"],
        "source_csv_replay_max_abs_delta": max_delta,
        "validation_predictions_sha256": sha256_file(
            root / "validation_predictions.npz"
        ),
        "test_predictions_sha256": sha256_file(root / "test_predictions.npz"),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "resource_counts": resource_record(hidden_dims),
        "training_performed": False,
    }
    atomic_write_json(root / "completion.json", completion)
    write_manifest(root, PROTOCOL_ID)
    return completion


def verify_scratch_reference(
    root: Path,
    context: dict[str, Any],
    lineage: dict[str, Any],
    execution_contract_sha256: str,
    seed: int,
    student: str,
    device: torch.device,
) -> dict[str, Any]:
    verify_manifest(
        root,
        PROTOCOL_ID,
        expected_files_excluding_manifest=SCRATCH_FILES_EXCLUDING_MANIFEST,
    )
    completion = load_json(root / "completion.json")
    base = lineage["seed_bindings"][str(seed)]["students"][student]["scratch"]
    required = {
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "role": "persisted_scratch_checkpoint_reference_without_retraining",
        "execution_contract_sha256": execution_contract_sha256,
        "seed": seed,
        "student": student,
        "source_checkpoint": base["plain_state_dict"],
        "source_checkpoint_sha256": base["plain_state_dict_sha256"],
        "source_rich_artifact_sha256": base["rich_artifact_sha256"],
        "source_test_predictions_sha256": base["test_predictions_sha256"],
        "initial_state_sha256": base["initial_state_sha256"],
        "trained_state_sha256": base["trained_state_sha256"],
        "training_performed": False,
    }
    for key, expected in required.items():
        if completion.get(key) != expected:
            raise RuntimeError(f"Scratch resume contract differs: {root}:{key}")
    validation_path = root / "validation_predictions.npz"
    test_path = root / "test_predictions.npz"
    if completion.get("validation_predictions_sha256") != sha256_file(validation_path):
        raise RuntimeError(f"Scratch validation prediction hash differs: {root}")
    if completion.get("test_predictions_sha256") != sha256_file(test_path):
        raise RuntimeError(f"Scratch test prediction hash differs: {root}")
    stored_validation = load_predictions(
        validation_path,
        context["indices"]["validation"],
        context["split"]["y_validation"],
    )
    stored_test = load_predictions(
        test_path, context["indices"]["test"], context["split"]["y_test"]
    )
    state = load_torch_mapping(Path(base["plain_state_dict"]))
    if state_dict_sha256(state) != base["trained_state_sha256"]:
        raise RuntimeError(f"Scratch source checkpoint content differs: {root}")
    model = StudentMLP(17, STUDENT_SPECS[student], len(CLASS_NAMES)).to(device)
    model.load_state_dict(state, strict=True)
    replayed_validation = batched_probs(
        model, torch.from_numpy(context["scaled"]["validation"]), device
    )
    replayed_test = batched_probs(
        model, torch.from_numpy(context["scaled"]["test"]), device
    )
    for name, replayed, stored in [
        ("validation", replayed_validation, stored_validation),
        ("test", replayed_test, stored_test),
    ]:
        delta = float(np.max(np.abs(replayed - stored)))
        if delta > PREDICTION_REPLAY_ATOL or not np.array_equal(
            replayed.argmax(axis=1), stored.argmax(axis=1)
        ):
            raise RuntimeError(f"Scratch checkpoint replay differs for {name}: {root}")
    expected_validation_metrics = metrics_with_resources(
        context["split"]["y_validation"], stored_validation, STUDENT_SPECS[student]
    )
    expected_test_metrics = metrics_with_resources(
        context["split"]["y_test"], stored_test, STUDENT_SPECS[student]
    )
    assert_nested_close(
        completion["validation_metrics"],
        expected_validation_metrics,
        path=f"scratch.{seed}.{student}.validation",
    )
    assert_nested_close(
        completion["test_metrics"],
        expected_test_metrics,
        path=f"scratch.{seed}.{student}.test",
    )
    expected_resources = resource_record(STUDENT_SPECS[student])
    if completion.get("resource_counts") != expected_resources:
        raise RuntimeError(f"Scratch resource counts differ: {root}")
    base_test = load_base_prediction_csv(
        Path(base["test_predictions"]),
        context["indices"]["test"],
        context["split"]["y_test"],
    )
    base_delta = float(np.max(np.abs(stored_test - base_test)))
    if (
        base_delta > BASE_CSV_REPLAY_ATOL
        or not np.array_equal(stored_test.argmax(axis=1), base_test.argmax(axis=1))
        or not math.isclose(
            float(completion.get("source_csv_replay_max_abs_delta", -1.0)),
            base_delta,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError(f"Scratch source-CSV replay record differs: {root}")
    return completion


def create_grid_job(
    root: Path,
    context: dict[str, Any],
    lineage: dict[str, Any],
    order_contract: dict[str, Any],
    execution_contract_sha256: str,
    seed: int,
    student: str,
    temperature: float,
    alpha: float,
    device: torch.device,
) -> dict[str, Any]:
    hidden_dims = STUDENT_SPECS[student]
    base_seed = lineage["seed_bindings"][str(seed)]
    rf_probabilities = np.load(
        base_seed["rf_train_probabilities"], allow_pickle=False
    )
    if sha256_file(Path(base_seed["rf_train_probabilities"])) != base_seed[
        "rf_train_probability_file_sha256"
    ]:
        raise RuntimeError(f"RF target file changed after preflight: seed {seed}")
    if sha256_arrays(rf_probabilities) != base_seed[
        "rf_train_probability_content_sha256"
    ]:
        raise RuntimeError(f"RF target content changed after preflight: seed {seed}")

    set_seed(seed)
    model = StudentMLP(17, hidden_dims, len(CLASS_NAMES))
    initial_hash = state_dict_sha256(state_dict_cpu(model))
    expected_initial = base_seed["students"][student]["initial_state_sha256"]
    if initial_hash != expected_initial:
        raise RuntimeError(f"Grid-cell initial state differs: {seed}, {student}")
    schedule = order_contract["records"][str(seed)][student]
    schedule = {
        **schedule,
        "order_contract_semantic_sha256": order_contract[
            "order_contract_semantic_sha256"
        ],
    }
    model, training = train_rf_kd_cell(
        model,
        rf_probabilities,
        torch.from_numpy(context["scaled"]["train"]),
        torch.from_numpy(context["split"]["y_train"]),
        torch.from_numpy(context["scaled"]["validation"]),
        torch.from_numpy(context["split"]["y_validation"]),
        class_weights(context["split"]["y_train"]),
        device,
        temperature=temperature,
        alpha=alpha,
        schedule=schedule,
        config=TRAIN_CONFIG,
    )
    validation_probabilities = batched_probs(
        model, torch.from_numpy(context["scaled"]["validation"]), device
    )
    test_probabilities = batched_probs(
        model, torch.from_numpy(context["scaled"]["test"]), device
    )
    validation_metrics = metrics_with_resources(
        context["split"]["y_validation"], validation_probabilities, hidden_dims
    )
    test_metrics = metrics_with_resources(
        context["split"]["y_test"], test_probabilities, hidden_dims
    )
    state = state_dict_cpu(model)
    trained_hash = state_dict_sha256(state)

    anchor_check: dict[str, Any] | None = None
    if temperature == ANCHOR_T and alpha == ANCHOR_ALPHA:
        anchor = base_seed["students"][student]["rf_kd_anchor"]
        if trained_hash != anchor["trained_state_sha256"]:
            raise RuntimeError(
                f"T=4, alpha=0.7 lineage anchor state failed: {seed}, {student}"
            )
        anchor_probabilities = load_base_prediction_csv(
            Path(anchor["test_predictions"]),
            context["indices"]["test"],
            context["split"]["y_test"],
        )
        max_delta = float(np.max(np.abs(test_probabilities - anchor_probabilities)))
        if max_delta > BASE_CSV_REPLAY_ATOL or not np.array_equal(
            test_probabilities.argmax(axis=1), anchor_probabilities.argmax(axis=1)
        ):
            raise RuntimeError(
                f"T=4, alpha=0.7 lineage anchor predictions failed: {seed}, {student}"
            )
        anchor_check = {
            "base_trained_state_sha256": anchor["trained_state_sha256"],
            "exact_state_hash_match": True,
            "base_test_predictions_sha256": anchor["test_predictions_sha256"],
            "max_abs_probability_delta_vs_base_csv": max_delta,
            "exact_argmax_match": True,
        }

    save_predictions(
        root / "validation_predictions.npz",
        context["indices"]["validation"],
        context["split"]["y_validation"],
        validation_probabilities,
    )
    save_predictions(
        root / "test_predictions.npz",
        context["indices"]["test"],
        context["split"]["y_test"],
        test_probabilities,
    )
    history_payload = {
        key: value for key, value in training.items() if key != "history"
    }
    history_payload["history"] = training["history"]
    atomic_write_json(root / "training_history.json", history_payload)
    checkpoint = checkpoint_payload(
        execution_contract_sha256=execution_contract_sha256,
        seed=seed,
        student=student,
        hidden_dims=hidden_dims,
        temperature=temperature,
        alpha=alpha,
        initial_state_sha256=initial_hash,
        state=state,
        base_binding=base_seed,
        schedule=schedule,
        training={key: value for key, value in training.items() if key != "history"},
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
    )
    atomic_torch_save(root / "checkpoint.pt", checkpoint)
    completion = {
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "analysis_role": "factorial_sensitivity_only",
        "execution_contract_sha256": execution_contract_sha256,
        "seed": seed,
        "student": student,
        "temperature": float(temperature),
        "alpha": float(alpha),
        "initial_state_sha256": initial_hash,
        "trained_state_sha256": trained_hash,
        "base_rf_target_file_sha256": base_seed[
            "rf_train_probability_file_sha256"
        ],
        "base_rf_target_content_sha256": base_seed[
            "rf_train_probability_content_sha256"
        ],
        "transformed_teacher_target_content_sha256": training[
            "teacher_target_content_sha256"
        ],
        "minibatch_order_contract_semantic_sha256": order_contract[
            "order_contract_semantic_sha256"
        ],
        "checkpoint_sha256": sha256_file(root / "checkpoint.pt"),
        "checkpoint_size_bytes": (root / "checkpoint.pt").stat().st_size,
        "validation_predictions_sha256": sha256_file(
            root / "validation_predictions.npz"
        ),
        "test_predictions_sha256": sha256_file(root / "test_predictions.npz"),
        "training_history_sha256": sha256_file(root / "training_history.json"),
        "training": {key: value for key, value in training.items() if key != "history"},
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "resource_counts": resource_record(hidden_dims),
        "anchor_reproduction_check": anchor_check,
        "selection_eligible": False,
    }
    atomic_write_json(root / "completion.json", completion)
    write_manifest(root, PROTOCOL_ID)
    return completion


def verify_grid_job(
    root: Path,
    context: dict[str, Any],
    lineage: dict[str, Any],
    order_contract: dict[str, Any],
    execution_contract_sha256: str,
    seed: int,
    student: str,
    temperature: float,
    alpha: float,
    device: torch.device,
) -> dict[str, Any]:
    verify_manifest(
        root,
        PROTOCOL_ID,
        expected_files_excluding_manifest=JOB_FILES_EXCLUDING_MANIFEST,
    )
    completion = load_json(root / "completion.json")
    base_seed = lineage["seed_bindings"][str(seed)]
    initial_hash = base_seed["students"][student]["initial_state_sha256"]
    required = {
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "analysis_role": "factorial_sensitivity_only",
        "execution_contract_sha256": execution_contract_sha256,
        "seed": seed,
        "student": student,
        "temperature": float(temperature),
        "alpha": float(alpha),
        "initial_state_sha256": initial_hash,
        "base_rf_target_file_sha256": base_seed[
            "rf_train_probability_file_sha256"
        ],
        "base_rf_target_content_sha256": base_seed[
            "rf_train_probability_content_sha256"
        ],
        "minibatch_order_contract_semantic_sha256": order_contract[
            "order_contract_semantic_sha256"
        ],
        "selection_eligible": False,
    }
    for key, expected in required.items():
        if completion.get(key) != expected:
            raise RuntimeError(f"Grid-job resume contract differs: {root}:{key}")
    artifact_hashes = {
        "checkpoint_sha256": root / "checkpoint.pt",
        "validation_predictions_sha256": root / "validation_predictions.npz",
        "test_predictions_sha256": root / "test_predictions.npz",
        "training_history_sha256": root / "training_history.json",
    }
    for key, path in artifact_hashes.items():
        if completion.get(key) != sha256_file(path):
            raise RuntimeError(f"Grid-job artifact hash differs: {root}:{key}")
    if completion.get("checkpoint_size_bytes") != (root / "checkpoint.pt").stat().st_size:
        raise RuntimeError(f"Grid-job checkpoint size differs: {root}")
    if completion.get("resource_counts") != resource_record(STUDENT_SPECS[student]):
        raise RuntimeError(f"Grid-job resource counts differ: {root}")
    if completion.get("transformed_teacher_target_content_sha256") != completion.get(
        "training", {}
    ).get("teacher_target_content_sha256"):
        raise RuntimeError(f"Grid-job transformed teacher-target hash differs: {root}")

    history = load_json(root / "training_history.json")
    if set(history) != set(completion["training"]) | {"history"}:
        raise RuntimeError(f"Grid-job training-history schema differs: {root}")
    for key, value in completion["training"].items():
        if history.get(key) != value:
            raise RuntimeError(f"Grid-job training history differs: {root}:{key}")
    if (
        not isinstance(history["history"], list)
        or len(history["history"]) != int(completion["training"]["epochs_completed"])
    ):
        raise RuntimeError(f"Grid-job epoch-history length differs: {root}")
    completed_epochs = int(completion["training"]["epochs_completed"])
    expected_prefix = order_contract["records"][str(seed)][student]["epochs"][:completed_epochs]
    if completion["training"].get("minibatch_order_contract_prefix") != expected_prefix:
        raise RuntimeError(f"Grid-job completion minibatch prefix differs: {root}")
    if [row.get("minibatch_order") for row in history["history"]] != expected_prefix:
        raise RuntimeError(f"Grid-job history minibatch prefix differs: {root}")

    checkpoint = load_torch_mapping(root / "checkpoint.pt", rich=True)
    checkpoint_state = checkpoint.get("state_dict")
    if not isinstance(checkpoint_state, dict):
        raise RuntimeError(f"Grid-job checkpoint lacks state_dict: {root}")
    trained_hash = state_dict_sha256(checkpoint_state)
    if trained_hash != completion.get("trained_state_sha256"):
        raise RuntimeError(f"Grid-job checkpoint state hash differs: {root}")
    checkpoint_required = {
        "protocol_id": PROTOCOL_ID,
        "analysis_role": "factorial_sensitivity_only",
        "execution_contract_sha256": execution_contract_sha256,
        "seed": seed,
        "student": student,
        "input_dim": 17,
        "hidden_dims": list(STUDENT_SPECS[student]),
        "num_classes": len(CLASS_NAMES),
        "temperature": float(temperature),
        "alpha": float(alpha),
        "initial_state_sha256": initial_hash,
        "trained_state_sha256": trained_hash,
        "base_rf_target_file_sha256": base_seed[
            "rf_train_probability_file_sha256"
        ],
        "base_rf_target_content_sha256": base_seed[
            "rf_train_probability_content_sha256"
        ],
        "minibatch_order_contract_semantic_sha256": order_contract[
            "order_contract_semantic_sha256"
        ],
        "training_config": TRAIN_CONFIG,
        "training": completion["training"],
        "validation_metrics": completion["validation_metrics"],
        "test_metrics": completion["test_metrics"],
        "resource_counts": resource_record(STUDENT_SPECS[student]),
    }
    for key, expected in checkpoint_required.items():
        if checkpoint.get(key) != expected:
            raise RuntimeError(f"Grid-job checkpoint metadata differs: {root}:{key}")

    stored_validation = load_predictions(
        root / "validation_predictions.npz",
        context["indices"]["validation"],
        context["split"]["y_validation"],
    )
    stored_test = load_predictions(
        root / "test_predictions.npz",
        context["indices"]["test"],
        context["split"]["y_test"],
    )
    model = StudentMLP(17, STUDENT_SPECS[student], len(CLASS_NAMES)).to(device)
    model.load_state_dict(checkpoint_state, strict=True)
    replayed_validation = batched_probs(
        model, torch.from_numpy(context["scaled"]["validation"]), device
    )
    replayed_test = batched_probs(
        model, torch.from_numpy(context["scaled"]["test"]), device
    )
    replay_deltas = {}
    for name, replayed, stored in [
        ("validation", replayed_validation, stored_validation),
        ("test", replayed_test, stored_test),
    ]:
        delta = float(np.max(np.abs(replayed - stored)))
        replay_deltas[name] = delta
        if delta > PREDICTION_REPLAY_ATOL:
            raise RuntimeError(f"Grid-job checkpoint probability replay differs: {root}:{name}")
        if not np.array_equal(replayed.argmax(axis=1), stored.argmax(axis=1)):
            raise RuntimeError(f"Grid-job checkpoint prediction replay differs: {root}:{name}")
    expected_validation_metrics = metrics_with_resources(
        context["split"]["y_validation"], stored_validation, STUDENT_SPECS[student]
    )
    expected_test_metrics = metrics_with_resources(
        context["split"]["y_test"], stored_test, STUDENT_SPECS[student]
    )
    assert_nested_close(
        completion["validation_metrics"],
        expected_validation_metrics,
        path=f"grid.{seed}.{student}.{cell_id(temperature, alpha)}.validation",
    )
    assert_nested_close(
        completion["test_metrics"],
        expected_test_metrics,
        path=f"grid.{seed}.{student}.{cell_id(temperature, alpha)}.test",
    )

    if temperature == ANCHOR_T and alpha == ANCHOR_ALPHA:
        anchor = base_seed["students"][student]["rf_kd_anchor"]
        if trained_hash != anchor["trained_state_sha256"]:
            raise RuntimeError(f"Grid-job lineage anchor state differs: {root}")
        base_probabilities = load_base_prediction_csv(
            Path(anchor["test_predictions"]),
            context["indices"]["test"],
            context["split"]["y_test"],
        )
        max_delta = float(np.max(np.abs(stored_test - base_probabilities)))
        anchor_check = completion.get("anchor_reproduction_check") or {}
        if (
            max_delta > BASE_CSV_REPLAY_ATOL
            or anchor_check.get("exact_state_hash_match") is not True
            or anchor_check.get("exact_argmax_match") is not True
            or not math.isclose(
                float(anchor_check.get("max_abs_probability_delta_vs_base_csv", -1.0)),
                max_delta,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not np.array_equal(stored_test.argmax(axis=1), base_probabilities.argmax(axis=1))
        ):
            raise RuntimeError(f"Grid-job lineage anchor predictions differ: {root}")
    elif completion.get("anchor_reproduction_check") is not None:
        raise RuntimeError(f"Non-anchor grid job contains an anchor claim: {root}")
    return completion


def scalar_summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise RuntimeError("Cannot summarize empty or non-finite values")
    return {
        "values": array.tolist(),
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def metric_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    scalar_keys = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "ece_15_bin",
    ]
    aggregate = {
        key: scalar_summary(float(record[key]) for record in records)
        for key in scalar_keys
    }
    per_class = np.asarray([record["per_class_f1"] for record in records], dtype=np.float64)
    aggregate["per_class_f1"] = {
        "class_names": CLASS_NAMES,
        "values": per_class.tolist(),
        "mean": per_class.mean(axis=0).tolist(),
        "sample_std": per_class.std(axis=0, ddof=1).tolist(),
    }
    resources = [record["trainable_parameters"] for record in records]
    if len(set(resources)) != 1:
        raise RuntimeError("Resource counts differ across seeds")
    aggregate["resource_counts"] = {
        key: records[0][key]
        for key in [
            "trainable_parameters",
            "fp32_parameter_payload_bytes",
            "linear_macs_per_record",
            "linear_flops_per_record_multiply_add_as_two",
            "hidden_activation_values_per_record",
            "scope",
        ]
    }
    return aggregate


def exact_signed_rank_test(differences: np.ndarray) -> dict[str, Any]:
    differences = np.asarray(differences, dtype=np.float64)
    nonzero = differences[differences != 0.0]
    if len(nonzero) == 0:
        return {
            "statistic_min_signed_rank": 0.0,
            "p_value_two_sided_exact_enumeration": 1.0,
            "zero_difference_count": int(len(differences)),
            "nonzero_difference_count": 0,
            "rank_tie_method": "average",
            "enumerated_sign_assignments": 1,
        }
    ranks = stats.rankdata(np.abs(nonzero), method="average")
    total = float(ranks.sum())
    observed_positive = float(ranks[nonzero > 0].sum())
    observed_statistic = min(observed_positive, total - observed_positive)
    statistics = []
    for mask in range(1 << len(nonzero)):
        positive = sum(
            float(ranks[index])
            for index in range(len(nonzero))
            if (mask >> index) & 1
        )
        statistics.append(min(positive, total - positive))
    p_value = float(
        np.mean(np.asarray(statistics, dtype=np.float64) <= observed_statistic + 1e-15)
    )
    scipy_result = stats.wilcoxon(
        differences,
        alternative="two-sided",
        zero_method="wilcox",
        method="approx",
    )
    return {
        "statistic_min_signed_rank": observed_statistic,
        "p_value_two_sided_exact_enumeration": p_value,
        "zero_difference_count": int(np.count_nonzero(differences == 0.0)),
        "nonzero_difference_count": int(len(nonzero)),
        "rank_tie_method": "average",
        "enumerated_sign_assignments": int(1 << len(nonzero)),
        "scipy_approximation_crosscheck": {
            "statistic": float(scipy_result.statistic),
            "p_value_two_sided": float(scipy_result.pvalue),
            "zero_method": "wilcox",
            "method": "approx",
        },
    }


def exact_sign_flip_mean_test(differences: np.ndarray) -> dict[str, Any]:
    differences = np.asarray(differences, dtype=np.float64)
    observed = abs(float(differences.mean()))
    values = []
    for mask in range(1 << len(differences)):
        signed = np.asarray(
            [
                differences[index] if (mask >> index) & 1 else -differences[index]
                for index in range(len(differences))
            ],
            dtype=np.float64,
        )
        values.append(abs(float(signed.mean())))
    return {
        "observed_abs_mean_difference": observed,
        "p_value_two_sided_exact_enumeration": float(
            np.mean(np.asarray(values) >= observed - 1e-15)
        ),
        "enumerated_sign_assignments": int(1 << len(differences)),
    }


def paired_test(left: list[float], right: list[float]) -> dict[str, Any]:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape or len(left_array) != len(PUBLICATION_SEEDS):
        raise RuntimeError("Paired sensitivity test does not contain ten aligned seeds")
    differences = left_array - right_array
    return {
        "left_values": left_array.tolist(),
        "right_values": right_array.tolist(),
        "difference": scalar_summary(differences),
        "wilcoxon_signed_rank": exact_signed_rank_test(differences),
        "sign_flip_mean_difference": exact_sign_flip_mean_test(differences),
    }


def apply_holm(
    tests: dict[str, dict[str, Any]],
    getter: Callable[[dict[str, Any]], float],
    field: str,
) -> None:
    ordered = sorted(tests, key=lambda name: (getter(tests[name]), name))
    running = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        adjusted = min(1.0, getter(tests[name]) * (total - rank))
        running = max(running, adjusted)
        tests[name][field] = running


def build_response_surface(
    seed_completions: dict[int, dict[str, Any]],
    execution_contract_sha256: str,
) -> dict[str, Any]:
    students: dict[str, Any] = {}
    tests: dict[str, Any] = {}
    for student in STUDENT_SPECS:
        scratch_validation_records = [
            seed_completions[seed]["scratch_references"][student][
                "validation_metrics"
            ]
            for seed in PUBLICATION_SEEDS
        ]
        scratch_test_records = [
            seed_completions[seed]["scratch_references"][student]["test_metrics"]
            for seed in PUBLICATION_SEEDS
        ]
        surface: dict[str, Any] = {}
        for temperature, alpha in GRID:
            key = cell_id(temperature, alpha)
            validation_records = [
                seed_completions[seed]["grid_results"][student][key][
                    "validation_metrics"
                ]
                for seed in PUBLICATION_SEEDS
            ]
            test_records = [
                seed_completions[seed]["grid_results"][student][key]["test_metrics"]
                for seed in PUBLICATION_SEEDS
            ]
            validation_aggregate = metric_aggregate(validation_records)
            test_aggregate = metric_aggregate(test_records)
            validation_difference = np.asarray(
                validation_aggregate["macro_f1"]["values"]
            ) - np.asarray(
                [record["macro_f1"] for record in scratch_validation_records]
            )
            test_difference = np.asarray(test_aggregate["macro_f1"]["values"]) - np.asarray(
                [record["macro_f1"] for record in scratch_test_records]
            )
            surface[key] = {
                "temperature": temperature,
                "alpha": alpha,
                "validation": validation_aggregate,
                "test": test_aggregate,
                "validation_macro_f1_minus_scratch": scalar_summary(validation_difference),
                "test_macro_f1_minus_scratch": scalar_summary(test_difference),
                "selection_eligible": False,
            }
            test_name = f"{student}:{key}_minus_persisted_scratch_test_macro_f1"
            tests[test_name] = paired_test(
                test_aggregate["macro_f1"]["values"],
                [record["macro_f1"] for record in scratch_test_records],
            )
            tests[test_name]["student_family"] = student
            tests[test_name]["temperature"] = temperature
            tests[test_name]["alpha"] = alpha

        marginal_temperature = {}
        for temperature in TEMPERATURES:
            values = [
                float(
                    np.mean(
                        [
                            surface[cell_id(temperature, alpha)]["test"]["macro_f1"]
                            ["values"][seed_index]
                            for alpha in ALPHAS
                        ]
                    )
                )
                for seed_index, _seed in enumerate(PUBLICATION_SEEDS)
            ]
            marginal_temperature[str(int(temperature))] = {
                "test_macro_f1_seed_marginal_mean_over_3_alphas": scalar_summary(values),
                "statistical_unit": (
                    "For each of ten aligned algorithmic seeds, macro-F1 is averaged "
                    "over the three alpha cells before the seed-level values are summarized."
                ),
            }
        marginal_alpha = {}
        for alpha in ALPHAS:
            values = [
                float(
                    np.mean(
                        [
                            surface[cell_id(temperature, alpha)]["test"]["macro_f1"]
                            ["values"][seed_index]
                            for temperature in TEMPERATURES
                        ]
                    )
                )
                for seed_index, _seed in enumerate(PUBLICATION_SEEDS)
            ]
            marginal_alpha[f"{alpha:.1f}"] = {
                "test_macro_f1_seed_marginal_mean_over_3_temperatures": scalar_summary(values),
                "statistical_unit": (
                    "For each of ten aligned algorithmic seeds, macro-F1 is averaged "
                    "over the three temperature cells before the seed-level values are summarized."
                ),
            }
        students[student] = {
            "scratch_reference": {
                "validation": metric_aggregate(scratch_validation_records),
                "test": metric_aggregate(scratch_test_records),
                "source": "completed clean ten-seed base checkpoints; no scratch retraining",
            },
            "factorial_response_surface": surface,
            "descriptive_marginal_temperature": marginal_temperature,
            "descriptive_marginal_alpha": marginal_alpha,
        }

    for student in STUDENT_SPECS:
        family = {
            name: value
            for name, value in tests.items()
            if value["student_family"] == student
        }
        apply_holm(
            family,
            lambda item: item["wilcoxon_signed_rank"][
                "p_value_two_sided_exact_enumeration"
            ],
            "holm_adjusted_wilcoxon_within_student_9_test_family_p",
        )
        apply_holm(
            family,
            lambda item: item["sign_flip_mean_difference"][
                "p_value_two_sided_exact_enumeration"
            ],
            "holm_adjusted_sign_flip_within_student_9_test_family_p",
        )
    apply_holm(
        tests,
        lambda item: item["wilcoxon_signed_rank"][
            "p_value_two_sided_exact_enumeration"
        ],
        "holm_adjusted_wilcoxon_global_18_test_family_p",
    )
    apply_holm(
        tests,
        lambda item: item["sign_flip_mean_difference"][
            "p_value_two_sided_exact_enumeration"
        ],
        "holm_adjusted_sign_flip_global_18_test_family_p",
    )
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "execution_contract_sha256": execution_contract_sha256,
        "analysis_role": "prespecified_factorial_sensitivity_only",
        "seeds": list(PUBLICATION_SEEDS),
        "seed_count": len(PUBLICATION_SEEDS),
        "temperatures": list(TEMPERATURES),
        "alphas": list(ALPHAS),
        "students": students,
        "paired_tests_against_persisted_scratch": tests,
        "standard_deviation_definition": "sample SD across ten aligned algorithmic seeds (ddof=1)",
        "statistical_unit": (
            "One algorithmic seed on one fixed feature-group-disjoint WSN-DS split. "
            "Seeds do not represent independently resampled train/test partitions."
        ),
        "multiple_testing_families": {
            "global_prespecified_family": "18 test-macro-F1 comparisons: 9 cells x 2 students",
            "reported_secondary_families": "9 comparisons within each student",
            "holm_applied_to": [
                "exact enumerated Wilcoxon signed-rank p-values",
                "exact sign-flip mean-difference p-values",
            ],
        },
        "inference_policy": {
            "primary_test": "exact enumerated Wilcoxon signed-rank",
            "sensitivity_test": "exact sign-flip mean difference",
            "reporting_rule": (
                "Wilcoxon results govern inferential statements. Sign-flip results are "
                "reported as sensitivity evidence and cannot replace the primary test."
            ),
        },
        "selection_performed": False,
        "selected_hyperparameters": None,
        "primary_result_replaced": False,
        "interpretation_boundary": (
            "This response surface evaluates sensitivity after the clean primary run. "
            "The same fixed validation partition is used for early stopping in every "
            "cell and the same fixed test partition is evaluated repeatedly. The "
            "surface must not be used for post-hoc model selection, a new primary "
            "claim, or an unbiased estimate of a selected cell's generalization."
        ),
    }


def build_execution_contract(
    lineage: dict[str, Any],
    order_contract: dict[str, Any],
    device: torch.device,
    lineage_file_sha256: str,
    order_file_sha256: str,
) -> dict[str, Any]:
    set_seed(0)
    contract = {
        "protocol_id": PROTOCOL_ID,
        "analysis_role": "prespecified_factorial_sensitivity_only",
        "script_sha256": sha256_file(SCRIPT_PATH),
        "common_source_sha256": EXPECTED_COMMON_SOURCE_SHA256,
        "base_runner_source_sha256": EXPECTED_BASE_SOURCE_SHA256,
        "source_snapshots": {
            "executed_sensitivity_source.py": sha256_file(SCRIPT_PATH),
            "bound_common_source.py": sha256_file(COMMON_SOURCE),
            "bound_base_runner_source.py": sha256_file(BASE_SOURCE),
        },
        "base_lineage_contract_file_sha256": lineage_file_sha256,
        "base_lineage_semantic_sha256": lineage["lineage_semantic_sha256"],
        "minibatch_order_contract_file_sha256": order_file_sha256,
        "minibatch_order_contract_semantic_sha256": order_contract[
            "order_contract_semantic_sha256"
        ],
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "split_indices_sha256": EXPECTED_SPLIT_SHA256,
        "scaler_sha256": EXPECTED_SCALER_SHA256,
        "seeds": list(PUBLICATION_SEEDS),
        "students": {name: list(value) for name, value in STUDENT_SPECS.items()},
        "factorial_grid": {
            "temperature": list(TEMPERATURES),
            "alpha": list(ALPHAS),
            "cell_count_per_student": len(GRID),
            "total_training_jobs": len(PUBLICATION_SEEDS) * len(STUDENT_SPECS) * len(GRID),
        },
        "training_config": TRAIN_CONFIG,
        "paired_reference": (
            "Persisted scratch checkpoint from the same seed, student architecture, "
            "clean split, scaler, initialization lineage, and minibatch-order contract."
        ),
        "lineage_anchor": {
            "temperature": ANCHOR_T,
            "alpha": ANCHOR_ALPHA,
            "purpose": (
                "Reproduce the completed base RF-KD state and predictions. This is a "
                "logic/provenance gate and is not a selected optimum."
            ),
            "exact_trained_state_hash_required": True,
            "exact_argmax_required": True,
            "maximum_probability_delta_vs_base_csv": BASE_CSV_REPLAY_ATOL,
        },
        "checkpoint_resume_policy": {
            "exact_file_inventory_and_sha256_required": True,
            "state_content_hash_required": True,
            "checkpoint_inference_replay_required": True,
            "maximum_probability_delta": PREDICTION_REPLAY_ATOL,
            "exact_argmax_required": True,
            "metrics_recomputed": True,
            "incomplete_or_unknown_artifacts": "refuse; never overwrite or auto-delete",
        },
        "metrics": [
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "per_class precision/recall/F1/support",
            "confusion matrix",
            "15-bin ECE",
            "analytical neural resource counts",
        ],
        "inference": {
            "paired_test_metric": "test macro-F1",
            "primary_test": "exact enumerated Wilcoxon signed-rank",
            "sensitivity_test": "exact sign-flip mean difference",
            "paired_tests": [
                "exact enumerated Wilcoxon signed-rank",
                "exact sign-flip mean difference",
            ],
            "reporting_rule": (
                "Wilcoxon results govern inferential statements. Sign-flip results are "
                "reported only as sensitivity evidence."
            ),
            "holm_global_family_size": len(GRID) * len(STUDENT_SPECS),
            "holm_within_student_family_size": len(GRID),
            "sample_standard_deviation_ddof": 1,
        },
        "selection_policy": {
            "optimization_performed": False,
            "winning_cell_selected": False,
            "primary_model_replaced": False,
            "allowed_use": "descriptive RF-KD hyperparameter sensitivity only",
        },
        "partition_reuse_boundary": (
            "All 180 jobs reuse one fixed validation partition for early stopping and "
            "one fixed test partition for reporting. Report every declared cell. Do "
            "not choose a cell using these results and call it independently tested."
        ),
        "environment": environment_record(device),
    }
    contract["execution_fingerprint_sha256"] = canonical_sha256(contract)
    return contract


def require_base_environment_for_real_run(
    base_environment: dict[str, Any], current: dict[str, Any]
) -> None:
    for key in [
        "python",
        "numpy",
        "pandas",
        "scikit_learn",
        "torch",
        "cuda_version",
        "device",
        "device_name",
        "cublas_workspace_config",
    ]:
        if current.get(key) != base_environment.get(key):
            raise RuntimeError(
                f"Real sensitivity run must match the clean base environment for {key}: "
                f"{current.get(key)!r} != {base_environment.get(key)!r}"
            )
    required_flags = {
        "deterministic_algorithms_enabled": True,
        "deterministic_algorithms_warn_only": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }
    for key, expected in required_flags.items():
        if current.get(key) != expected:
            raise RuntimeError(f"Deterministic execution flag differs: {key}")


def expected_seed_recursive_files() -> set[str]:
    files = {"seed_completion.json"}
    for student in STUDENT_SPECS:
        scratch_prefix = f"{student}/scratch_reference"
        files.update(
            {f"{scratch_prefix}/{name}" for name in SCRATCH_FILES_EXCLUDING_MANIFEST}
        )
        files.add(f"{scratch_prefix}/artifact_manifest.json")
        for temperature, alpha in GRID:
            prefix = f"{student}/{cell_id(temperature, alpha)}"
            files.update({f"{prefix}/{name}" for name in JOB_FILES_EXCLUDING_MANIFEST})
            files.add(f"{prefix}/artifact_manifest.json")
    return files


def validate_partial_seed_schema(seed_root: Path) -> None:
    allowed_students = set(STUDENT_SPECS)
    for path in seed_root.iterdir():
        if path.is_file():
            if path.name not in {"seed_completion.json", "artifact_manifest.json"}:
                raise RuntimeError(f"Unknown file in partial seed root: {path}")
        elif path.is_dir():
            if path.name not in allowed_students:
                raise RuntimeError(f"Unknown directory in partial seed root: {path}")
            allowed_jobs = {"scratch_reference"} | {
                cell_id(temperature, alpha) for temperature, alpha in GRID
            }
            for job in path.iterdir():
                if not job.is_dir() or job.name not in allowed_jobs:
                    raise RuntimeError(f"Unknown job artifact in partial seed: {job}")


def complete_seed(
    output_root: Path,
    context: dict[str, Any],
    lineage: dict[str, Any],
    order_contract: dict[str, Any],
    execution_contract_sha256: str,
    seed: int,
    device: torch.device,
    resume: bool,
) -> dict[str, Any]:
    seed_root = output_root / f"seed_{seed}"
    if seed_root.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite existing seed output: {seed_root}")
    seed_root.mkdir(parents=True, exist_ok=True)
    validate_partial_seed_schema(seed_root)

    scratch_results: dict[str, Any] = {}
    grid_results: dict[str, Any] = {}
    for student in STUDENT_SPECS:
        student_root = seed_root / student
        student_root.mkdir(exist_ok=True)
        reference_root = scratch_root(seed_root, student)
        if reference_root.exists():
            if not resume:
                raise FileExistsError(reference_root)
            scratch_results[student] = verify_scratch_reference(
                reference_root,
                context,
                lineage,
                execution_contract_sha256,
                seed,
                student,
                device,
            )
        else:
            reference_root.mkdir()
            scratch_results[student] = create_scratch_reference(
                reference_root,
                context,
                lineage,
                execution_contract_sha256,
                seed,
                student,
                device,
            )

        grid_results[student] = {}
        for temperature, alpha in GRID:
            root = job_root(seed_root, student, temperature, alpha)
            key = cell_id(temperature, alpha)
            if root.exists():
                if not resume:
                    raise FileExistsError(root)
                grid_results[student][key] = verify_grid_job(
                    root,
                    context,
                    lineage,
                    order_contract,
                    execution_contract_sha256,
                    seed,
                    student,
                    temperature,
                    alpha,
                    device,
                )
            else:
                root.mkdir()
                grid_results[student][key] = create_grid_job(
                    root,
                    context,
                    lineage,
                    order_contract,
                    execution_contract_sha256,
                    seed,
                    student,
                    temperature,
                    alpha,
                    device,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    completion = {
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "execution_contract_sha256": execution_contract_sha256,
        "seed": seed,
        "base_seed_completion_sha256": lineage["seed_bindings"][str(seed)][
            "seed_completion_sha256"
        ],
        "base_seed_manifest_sha256": lineage["seed_bindings"][str(seed)][
            "seed_manifest_sha256"
        ],
        "scratch_references": scratch_results,
        "grid_results": grid_results,
        "selection_performed": False,
    }
    completion_path = seed_root / "seed_completion.json"
    if completion_path.exists():
        if not resume or load_json(completion_path) != completion:
            raise RuntimeError(f"Existing seed completion differs: {seed_root}")
    else:
        atomic_write_json(completion_path, completion)
    expected_files = expected_seed_recursive_files()
    manifest_path = seed_root / "artifact_manifest.json"
    if manifest_path.exists():
        verify_manifest(
            seed_root,
            PROTOCOL_ID,
            expected_files_excluding_manifest=expected_files,
        )
    else:
        actual = {
            path.relative_to(seed_root).as_posix()
            for path in seed_root.rglob("*")
            if path.is_file()
        }
        if actual != expected_files:
            raise RuntimeError(f"Completed seed inventory differs before manifest: {seed_root}")
        write_manifest(seed_root, PROTOCOL_ID)
    return completion


def verify_completed_seed(
    output_root: Path,
    context: dict[str, Any],
    lineage: dict[str, Any],
    order_contract: dict[str, Any],
    execution_contract_sha256: str,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    seed_root = output_root / f"seed_{seed}"
    verify_manifest(
        seed_root,
        PROTOCOL_ID,
        expected_files_excluding_manifest=expected_seed_recursive_files(),
    )
    stored = load_json(seed_root / "seed_completion.json")
    required = {
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "execution_contract_sha256": execution_contract_sha256,
        "seed": seed,
        "base_seed_completion_sha256": lineage["seed_bindings"][str(seed)][
            "seed_completion_sha256"
        ],
        "base_seed_manifest_sha256": lineage["seed_bindings"][str(seed)][
            "seed_manifest_sha256"
        ],
        "selection_performed": False,
    }
    for key, expected in required.items():
        if stored.get(key) != expected:
            raise RuntimeError(f"Stored seed completion differs: {seed}:{key}")
    scratch = {}
    grid = {}
    for student in STUDENT_SPECS:
        scratch[student] = verify_scratch_reference(
            scratch_root(seed_root, student),
            context,
            lineage,
            execution_contract_sha256,
            seed,
            student,
            device,
        )
        grid[student] = {}
        for temperature, alpha in GRID:
            key = cell_id(temperature, alpha)
            grid[student][key] = verify_grid_job(
                job_root(seed_root, student, temperature, alpha),
                context,
                lineage,
                order_contract,
                execution_contract_sha256,
                seed,
                student,
                temperature,
                alpha,
                device,
            )
    semantic = {
        **required,
        "scratch_references": scratch,
        "grid_results": grid,
    }
    if stored != semantic:
        raise RuntimeError(f"Stored seed completion does not replay semantically: {seed}")
    return stored


def write_or_verify_immutable_contracts(
    output_root: Path,
    lineage: dict[str, Any],
    order_contract: dict[str, Any],
    device: torch.device,
    resume: bool,
) -> tuple[dict[str, Any], str]:
    lineage_path = output_root / "base_lineage_contract.json"
    order_path = output_root / "minibatch_order_contract.json"
    execution_path = output_root / "execution_contract.json"
    source_snapshots = {
        "executed_sensitivity_source.py": SCRIPT_PATH,
        "bound_common_source.py": COMMON_SOURCE,
        "bound_base_runner_source.py": BASE_SOURCE,
    }
    if resume:
        required_paths = [lineage_path, order_path, execution_path] + [
            output_root / name for name in source_snapshots
        ]
        if not all(path.is_file() for path in required_paths):
            raise RuntimeError("Resume root lacks immutable contracts")
        for name, source in source_snapshots.items():
            snapshot = output_root / name
            if snapshot.read_bytes() != source.read_bytes():
                raise RuntimeError(f"Resume source snapshot differs: {snapshot}")
        if load_json(lineage_path) != lineage:
            raise RuntimeError("Resume base-lineage contract differs")
        if load_json(order_path) != order_contract:
            raise RuntimeError("Resume minibatch-order contract differs")
    else:
        if output_root.exists() and any(output_root.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite non-empty sensitivity output: {output_root}"
            )
        output_root.mkdir(parents=True, exist_ok=True)
        for name, source in source_snapshots.items():
            atomic_write_bytes(output_root / name, source.read_bytes())
        atomic_write_json(lineage_path, lineage)
        atomic_write_json(order_path, order_contract)
    execution = build_execution_contract(
        lineage,
        order_contract,
        device,
        sha256_file(lineage_path),
        sha256_file(order_path),
    )
    if resume:
        if load_json(execution_path) != execution:
            raise RuntimeError("Resume execution contract differs from code/config/environment")
    else:
        atomic_write_json(execution_path, execution)
    verify_bound_source_snapshots(output_root, execution)
    return execution, sha256_file(execution_path)


def verify_bound_source_snapshots(
    output_root: Path, execution: dict[str, Any]
) -> None:
    expected = execution.get("source_snapshots")
    if not isinstance(expected, dict) or set(expected) != {
        "executed_sensitivity_source.py",
        "bound_common_source.py",
        "bound_base_runner_source.py",
    }:
        raise RuntimeError("Execution contract source-snapshot schema differs")
    for name, expected_sha256 in expected.items():
        snapshot = output_root / name
        if not snapshot.is_file() or sha256_file(snapshot) != expected_sha256:
            raise RuntimeError(f"Bound source snapshot differs: {snapshot}")
    if (
        expected["executed_sensitivity_source.py"] != execution["script_sha256"]
        or expected["bound_common_source.py"] != execution["common_source_sha256"]
        or expected["bound_base_runner_source.py"]
        != execution["base_runner_source_sha256"]
    ):
        raise RuntimeError("Bound source snapshots disagree with execution source hashes")


def validate_root_partial_schema(output_root: Path) -> None:
    allowed = ROOT_FINAL_FILES | {f"seed_{seed}" for seed in PUBLICATION_SEEDS}
    for path in output_root.iterdir():
        if path.name not in allowed:
            raise RuntimeError(f"Unknown artifact in sensitivity output root: {path}")
        if path.name.startswith("seed_") and not path.is_dir():
            raise RuntimeError(f"Seed output is not a directory: {path}")
        if not path.name.startswith("seed_") and not path.is_file():
            raise RuntimeError(f"Root contract artifact is not a file: {path}")


def run_real(
    args: argparse.Namespace,
    context: dict[str, Any],
    lineage: dict[str, Any],
    order_contract: dict[str, Any],
    device: torch.device,
) -> Path:
    output_root = require_disjoint_output(
        args.output_dir,
        [args.base_root, SCRIPT_PATH, COMMON_SOURCE, BASE_SOURCE],
    )
    require_base_environment_for_real_run(
        context["execution"]["environment"], environment_record(device)
    )
    execution, execution_contract_sha256 = write_or_verify_immutable_contracts(
        output_root, lineage, order_contract, device, args.resume
    )
    if execution["execution_fingerprint_sha256"] != canonical_sha256(
        {key: value for key, value in execution.items() if key != "execution_fingerprint_sha256"}
    ):
        raise RuntimeError("Execution fingerprint is internally inconsistent")
    validate_root_partial_schema(output_root)

    seed_completions: dict[int, dict[str, Any]] = {}
    for seed in PUBLICATION_SEEDS:
        seed_root = output_root / f"seed_{seed}"
        if (seed_root / "artifact_manifest.json").is_file():
            if not args.resume:
                raise FileExistsError(seed_root)
            seed_completions[seed] = verify_completed_seed(
                output_root,
                context,
                lineage,
                order_contract,
                execution_contract_sha256,
                seed,
                device,
            )
        else:
            seed_completions[seed] = complete_seed(
                output_root,
                context,
                lineage,
                order_contract,
                execution_contract_sha256,
                seed,
                device,
                args.resume,
            )

    aggregate = build_response_surface(seed_completions, execution_contract_sha256)
    aggregate_path = output_root / "response_surface_aggregate.json"
    if aggregate_path.exists():
        if not args.resume or load_json(aggregate_path) != aggregate:
            raise RuntimeError("Existing response-surface aggregate differs")
    else:
        atomic_write_json(aggregate_path, aggregate)
    verify_bound_source_snapshots(output_root, execution)
    root_manifest_path = output_root / "artifact_manifest.json"
    if root_manifest_path.exists():
        verify_manifest(output_root, PROTOCOL_ID)
    else:
        write_manifest(output_root, PROTOCOL_ID)
    if {path.name for path in output_root.iterdir()} != (
        ROOT_FINAL_FILES | {f"seed_{seed}" for seed in PUBLICATION_SEEDS}
    ):
        raise RuntimeError("Final sensitivity root inventory differs")
    return aggregate_path


def run_synthetic_checks() -> dict[str, Any]:
    config = {
        "epochs": 2,
        "batch_size": 10,
        "lr": 1e-3,
        "weight_decay": 1e-3,
        "patience": 2,
    }
    rng = np.random.default_rng(20260811)
    X_train = rng.normal(size=(50, 17)).astype(np.float32)
    y_train = np.tile(np.arange(len(CLASS_NAMES), dtype=np.int64), 10)
    X_validation = rng.normal(size=(25, 17)).astype(np.float32)
    y_validation = np.tile(np.arange(len(CLASS_NAMES), dtype=np.int64), 5)
    raw_teacher = rng.uniform(0.01, 1.0, size=(50, len(CLASS_NAMES))).astype(np.float32)
    rf_probabilities = (raw_teacher / raw_teacher.sum(axis=1, keepdims=True)).astype(
        np.float32
    )
    seed = 123
    student = "student_A"
    hidden_dims = STUDENT_SPECS[student]
    set_seed(seed)
    planned_model = StudentMLP(17, hidden_dims, len(CLASS_NAMES))
    initial_hash = state_dict_sha256(state_dict_cpu(planned_model))
    del planned_model
    epoch_records = []
    for epoch in range(config["epochs"]):
        discarded, sampler_seed, order = draw_epoch_order(len(X_train))
        epoch_records.append(
            {
                "epoch": epoch + 1,
                "discarded_global_rng_draw": discarded,
                "sampler_seed": sampler_seed,
                "index_order_sha256": sha256_arrays(order.numpy()),
                "batch_count": int(math.ceil(len(X_train) / config["batch_size"])),
            }
        )
    schedule = {
        "initial_state_sha256": initial_hash,
        "train_rows": len(X_train),
        "batch_size": config["batch_size"],
        "maximum_epochs": config["epochs"],
        "epochs": epoch_records,
    }
    observed_prefixes = []
    prediction_sets = []
    for temperature, alpha in [(1.0, 0.3), (4.0, 0.7)]:
        set_seed(seed)
        model = StudentMLP(17, hidden_dims, len(CLASS_NAMES))
        if state_dict_sha256(state_dict_cpu(model)) != initial_hash:
            raise RuntimeError("Synthetic initial-state recreation failed")
        model, training = train_rf_kd_cell(
            model,
            rf_probabilities,
            torch.from_numpy(X_train),
            torch.from_numpy(y_train),
            torch.from_numpy(X_validation),
            torch.from_numpy(y_validation),
            class_weights(y_train),
            torch.device("cpu"),
            temperature=temperature,
            alpha=alpha,
            schedule=schedule,
            config=config,
        )
        observed_prefixes.append(training["minibatch_order_contract_prefix"])
        prediction_sets.append(
            batched_probs(model, torch.from_numpy(X_validation), torch.device("cpu"))
        )
    if observed_prefixes[0] != observed_prefixes[1]:
        raise RuntimeError("Synthetic grid cells did not reuse the minibatch order")

    paired = paired_test(
        [0.80 + index * 0.001 for index in range(10)],
        [0.79 + index * 0.001 for index in range(10)],
    )
    test_family = {
        "a": copy.deepcopy(paired),
        "b": copy.deepcopy(paired),
    }
    apply_holm(
        test_family,
        lambda item: item["wilcoxon_signed_rank"][
            "p_value_two_sided_exact_enumeration"
        ],
        "adjusted",
    )
    if not all(0.0 <= item["adjusted"] <= 1.0 for item in test_family.values()):
        raise RuntimeError("Synthetic Holm correction failed")

    with tempfile.TemporaryDirectory(prefix="cukd_rfkd_sensitivity_") as temporary:
        root = Path(temporary)
        probabilities = prediction_sets[0]
        indices = np.arange(len(y_validation), dtype=np.int64)
        save_predictions(root / "predictions.npz", indices, y_validation, probabilities)
        replayed = load_predictions(root / "predictions.npz", indices, y_validation)
        if not np.array_equal(replayed, probabilities.astype(np.float32)):
            raise RuntimeError("Synthetic prediction persistence failed")
        atomic_write_json(root / "payload.json", {"status": "complete"})
        write_manifest(root, PROTOCOL_ID)
        verify_manifest(root, PROTOCOL_ID)

        strict_root = root / "strict_resume"
        strict_root.mkdir()
        rf_path = strict_root / "rf_train_probabilities.npy"
        with rf_path.open("wb") as handle:
            np.save(handle, rf_probabilities, allow_pickle=False)
        set_seed(seed)
        scratch_model = StudentMLP(17, hidden_dims, len(CLASS_NAMES))
        scratch_state = state_dict_cpu(scratch_model)
        scratch_checkpoint = strict_root / "scratch_fp32.pt"
        torch.save(scratch_state, scratch_checkpoint)
        X_test = rng.normal(size=(25, 17)).astype(np.float32)
        y_test = np.tile(np.arange(len(CLASS_NAMES), dtype=np.int64), 5)
        scratch_test_probabilities = batched_probs(
            scratch_model, torch.from_numpy(X_test), torch.device("cpu")
        )
        scratch_csv = strict_root / "scratch_test_predictions.csv"
        scratch_frame: dict[str, Any] = {
            "source_row_index": np.arange(200, 225, dtype=np.int64),
            "true_label": y_test,
            "predicted_label": scratch_test_probabilities.argmax(axis=1),
        }
        for class_index, class_name in enumerate(CLASS_NAMES):
            scratch_frame[
                f"probability_{class_index}_{class_name}"
            ] = scratch_test_probabilities[:, class_index]
        pd.DataFrame(scratch_frame).to_csv(scratch_csv, index=False)
        scratch_rich = strict_root / "scratch_artifact.pt"
        torch.save({"state_dict": scratch_state}, scratch_rich)

        synthetic_context = {
            "scaled": {
                "train": X_train,
                "validation": X_validation,
                "test": X_test,
            },
            "split": {
                "y_train": y_train,
                "y_validation": y_validation,
                "y_test": y_test,
            },
            "indices": {
                "train": np.arange(50, dtype=np.int64),
                "validation": np.arange(100, 125, dtype=np.int64),
                "test": np.arange(200, 225, dtype=np.int64),
            },
        }
        scratch_metrics = classification_metrics(y_test, scratch_test_probabilities)
        synthetic_lineage = {
            "seed_bindings": {
                str(seed): {
                    "rf_train_probabilities": str(rf_path),
                    "rf_train_probability_file_sha256": sha256_file(rf_path),
                    "rf_train_probability_content_sha256": sha256_arrays(
                        rf_probabilities
                    ),
                    "students": {
                        student: {
                            "initial_state_sha256": initial_hash,
                            "scratch": {
                                "plain_state_dict": str(scratch_checkpoint),
                                "plain_state_dict_sha256": sha256_file(
                                    scratch_checkpoint
                                ),
                                "rich_artifact_sha256": sha256_file(scratch_rich),
                                "test_predictions": str(scratch_csv),
                                "test_predictions_sha256": sha256_file(scratch_csv),
                                "initial_state_sha256": initial_hash,
                                "trained_state_sha256": state_dict_sha256(
                                    scratch_state
                                ),
                                "metrics": scratch_metrics,
                            },
                        }
                    },
                }
            }
        }
        set_seed(seed)
        schedule_model = StudentMLP(17, hidden_dims, len(CLASS_NAMES))
        if state_dict_sha256(state_dict_cpu(schedule_model)) != initial_hash:
            raise RuntimeError("Synthetic strict-resume initial state differs")
        del schedule_model
        full_epochs = []
        for epoch in range(int(TRAIN_CONFIG["epochs"])):
            discarded, sampler_seed, order = draw_epoch_order(len(X_train))
            full_epochs.append(
                {
                    "epoch": epoch + 1,
                    "discarded_global_rng_draw": discarded,
                    "sampler_seed": sampler_seed,
                    "index_order_sha256": sha256_arrays(order.numpy()),
                    "batch_count": int(
                        math.ceil(len(X_train) / int(TRAIN_CONFIG["batch_size"]))
                    ),
                }
            )
        synthetic_order_contract = {
            "order_contract_semantic_sha256": "synthetic-order-contract",
            "records": {
                str(seed): {
                    student: {
                        "initial_state_sha256": initial_hash,
                        "train_rows": len(X_train),
                        "batch_size": int(TRAIN_CONFIG["batch_size"]),
                        "maximum_epochs": int(TRAIN_CONFIG["epochs"]),
                        "epochs": full_epochs,
                    }
                }
            },
        }
        synthetic_execution_sha = "synthetic-execution-contract"
        scratch_output = strict_root / "scratch_reference"
        scratch_output.mkdir()
        create_scratch_reference(
            scratch_output,
            synthetic_context,
            synthetic_lineage,
            synthetic_execution_sha,
            seed,
            student,
            torch.device("cpu"),
        )
        verify_scratch_reference(
            scratch_output,
            synthetic_context,
            synthetic_lineage,
            synthetic_execution_sha,
            seed,
            student,
            torch.device("cpu"),
        )
        grid_output = strict_root / "grid_job"
        grid_output.mkdir()
        create_grid_job(
            grid_output,
            synthetic_context,
            synthetic_lineage,
            synthetic_order_contract,
            synthetic_execution_sha,
            seed,
            student,
            1.0,
            0.3,
            torch.device("cpu"),
        )
        verified_grid = verify_grid_job(
            grid_output,
            synthetic_context,
            synthetic_lineage,
            synthetic_order_contract,
            synthetic_execution_sha,
            seed,
            student,
            1.0,
            0.3,
            torch.device("cpu"),
        )
        if "resume_checkpoint_replay_max_abs_delta" in verified_grid:
            raise RuntimeError("Grid verification introduced non-persistent resume metadata")
        synthetic_seed_completions: dict[int, dict[str, Any]] = {}
        for seed_index, synthetic_seed in enumerate(PUBLICATION_SEEDS):
            scratch_records = {}
            grid_records = {}
            for student_index, (
                synthetic_student,
                synthetic_hidden_dims,
            ) in enumerate(STUDENT_SPECS.items()):
                metrics = metrics_with_resources(
                    y_test, scratch_test_probabilities, synthetic_hidden_dims
                )
                metrics["macro_f1"] = 0.55 + 0.001 * seed_index + 0.01 * student_index
                scratch_records[synthetic_student] = {
                    "validation_metrics": copy.deepcopy(metrics),
                    "test_metrics": copy.deepcopy(metrics),
                }
                grid_records[synthetic_student] = {}
                for temperature, alpha in GRID:
                    cell_metrics = copy.deepcopy(metrics)
                    cell_metrics["macro_f1"] += 0.002 * temperature + 0.003 * alpha
                    grid_records[synthetic_student][cell_id(temperature, alpha)] = {
                        "validation_metrics": copy.deepcopy(cell_metrics),
                        "test_metrics": copy.deepcopy(cell_metrics),
                    }
            synthetic_seed_completions[synthetic_seed] = {
                "scratch_references": scratch_records,
                "grid_results": grid_records,
            }
        synthetic_surface = build_response_surface(
            synthetic_seed_completions, synthetic_execution_sha
        )
        if (
            len(synthetic_surface["paired_tests_against_persisted_scratch"]) != 18
            or synthetic_surface["selection_performed"] is not False
            or synthetic_surface["selected_hyperparameters"] is not None
        ):
            raise RuntimeError("Synthetic response-surface aggregation failed")
        for student_index, synthetic_student in enumerate(STUDENT_SPECS):
            student_surface = synthetic_surface["students"][synthetic_student]
            for temperature in TEMPERATURES:
                marginal = student_surface["descriptive_marginal_temperature"][
                    str(int(temperature))
                ]
                summary = marginal["test_macro_f1_seed_marginal_mean_over_3_alphas"]
                expected = np.asarray(
                    [
                        0.55
                        + 0.001 * seed_index
                        + 0.01 * student_index
                        + 0.002 * temperature
                        + 0.003 * float(np.mean(ALPHAS))
                        for seed_index, _seed in enumerate(PUBLICATION_SEEDS)
                    ],
                    dtype=np.float64,
                )
                if len(summary["values"]) != len(PUBLICATION_SEEDS) or not np.allclose(
                    np.asarray(summary["values"], dtype=np.float64),
                    expected,
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise RuntimeError("Temperature marginal values are incorrect")
            for alpha in ALPHAS:
                marginal = student_surface["descriptive_marginal_alpha"][f"{alpha:.1f}"]
                summary = marginal[
                    "test_macro_f1_seed_marginal_mean_over_3_temperatures"
                ]
                expected = np.asarray(
                    [
                        0.55
                        + 0.001 * seed_index
                        + 0.01 * student_index
                        + 0.002 * float(np.mean(TEMPERATURES))
                        + 0.003 * alpha
                        for seed_index, _seed in enumerate(PUBLICATION_SEEDS)
                    ],
                    dtype=np.float64,
                )
                if len(summary["values"]) != len(PUBLICATION_SEEDS) or not np.allclose(
                    np.asarray(summary["values"], dtype=np.float64),
                    expected,
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise RuntimeError("Alpha marginal values are incorrect")
        if synthetic_surface["inference_policy"] != {
            "primary_test": "exact enumerated Wilcoxon signed-rank",
            "sensitivity_test": "exact sign-flip mean difference",
            "reporting_rule": (
                "Wilcoxon results govern inferential statements. Sign-flip results are "
                "reported as sensitivity evidence and cannot replace the primary test."
            ),
        }:
            raise RuntimeError("Synthetic inferential hierarchy differs")
    return {
        "status": "passed",
        "training_jobs_executed": 3,
        "short_primitive_jobs": 2,
        "strict_resume_round_trip_jobs": 1,
        "short_primitive_epochs_per_job": config["epochs"],
        "identical_initial_state_reused": True,
        "identical_minibatch_order_reused": True,
        "prediction_round_trip_checked": True,
        "manifest_round_trip_checked": True,
        "strict_scratch_reference_resume_checked": True,
        "strict_grid_checkpoint_resume_checked": True,
        "resume_metadata_stability_checked": True,
        "response_surface_and_18_test_family_checked": True,
        "seed_level_factorial_marginals_checked": True,
        "exact_tests_and_holm_checked": True,
        "repository_outputs_written": False,
    }


def main() -> int:
    args = parse_args()
    if args.resume and not args.confirm_run:
        raise RuntimeError("--resume is valid only with --confirm-run")
    if args.synthetic_only:
        print(json.dumps({"synthetic_checks": run_synthetic_checks()}, indent=2))
        return 0

    device = resolve_device(args.device)
    verification_device = torch.device("cpu") if not args.confirm_run else device
    synthetic = run_synthetic_checks()
    context, lineage = validate_base_lineage(
        args.base_root, args.dataset_csv, verification_device
    )
    order_contract = build_minibatch_order_contract(
        lineage, len(context["indices"]["train"]), TRAIN_CONFIG
    )
    if not args.confirm_run:
        report = {
            "protocol_id": PROTOCOL_ID,
            "status": "preflight_passed",
            "training_started": False,
            "repository_outputs_written": False,
            "script_sha256": sha256_file(SCRIPT_PATH),
            "base_lineage_semantic_sha256": lineage["lineage_semantic_sha256"],
            "minibatch_order_contract_semantic_sha256": order_contract[
                "order_contract_semantic_sha256"
            ],
            "dataset_sha256": EXPECTED_DATASET_SHA256,
            "split_indices_sha256": EXPECTED_SPLIT_SHA256,
            "scaler_sha256": EXPECTED_SCALER_SHA256,
            "seeds": list(PUBLICATION_SEEDS),
            "students": {name: list(value) for name, value in STUDENT_SPECS.items()},
            "factorial_grid": {
                "temperature": list(TEMPERATURES),
                "alpha": list(ALPHAS),
                "total_training_jobs": len(PUBLICATION_SEEDS)
                * len(STUDENT_SPECS)
                * len(GRID),
            },
            "synthetic_checks": synthetic,
            "selection_performed": False,
            "output_if_confirmed": str(args.output_dir.resolve()),
        }
        print(json.dumps(report, indent=2))
        return 0

    aggregate_path = run_real(args, context, lineage, order_contract, device)
    print(aggregate_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
