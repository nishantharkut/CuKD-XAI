"""Run the frozen ten-seed, scratch-controlled FG-DS XAI transfer audit.

The default invocation performs preflight only. Permutation SHAP starts only when
``--confirm-explanations`` is supplied. Historical SHAP artifacts are never read,
modified, or relabeled by this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
from itertools import product
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import scipy
import shap
import sklearn
import torch
import torch.nn.functional as F
from scipy.stats import rankdata, spearmanr, t
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.wsnds.leakage_free_rerun.tier15_common import (  # noqa: E402
    CLASS_NAMES,
    RF_CONFIG,
    STUDENT_SPECS,
    StudentMLP,
    apply_train_scaler,
    feature_group_split,
    load_wsnds,
    sha256_arrays,
    sha256_file,
    verified_feature_hashes,
)


PROTOCOL_ID = "wsnds_fgds_controlled_xai_transfer_10seed_v1"
SOURCE_PROTOCOL_ID = "wsnds_feature_group_split_train_only_scaler_10seed_v2"
EXPECTED_DATASET_SHA256 = (
    "c65d05b983a85753bd62b6f76c5739fc52fe0c14cbb7644255cee4742f5ff7c9"
)
EXPECTED_SPLIT_SHA256 = (
    "3d4061aa020122d4c5c5b2f7722de71e0c223c533869d3fdfa1f10784a0a0473"
)
EXPECTED_SCALER_SHA256 = (
    "5303fb570aeb82ffaf88e2d4cceda94a7611762f67c86761990e6a4f09af5dd6"
)
EXPECTED_SEEDS = [42, 123, 456, 789, 1001, 2024, 3141, 5678, 8192, 9999]
EXPECTED_SPLIT_SIZES = {"train": 262_197, "validation": 56_163, "test": 56_301}
KD_TEMPERATURE = 4.0
PROBABILITY_FLOOR = 1e-8
COHORT_SEED = 2042
BACKGROUND_SEED = 1042
PERMUTATION_SEED = 3042
COHORT_GROUPS_PER_CLASS = 50
BACKGROUND_GROUPS_PER_CLASS = 4
COHORT_SIZE = COHORT_GROUPS_PER_CLASS * len(CLASS_NAMES)
BACKGROUND_SIZE = BACKGROUND_GROUPS_PER_CLASS * len(CLASS_NAMES)
PERMUTATION_REPEATS = 5
LOCAL_ACCURACY_ATOL = 1e-6
MODEL_REPLAY_ATOL = 1e-6
ATTRIBUTION_NORM_EPSILON = 1e-6
COMPONENT_MATERIALITY_EPSILON = 1e-6
MIN_ELIGIBLE_PER_CLASS = 40
RANDOMIZATION_SEEDS = {"student_A": 5042, "student_B": 6042}
STUDENT_KEYS = ["student_A", "student_B"]
ROUTES = ["scratch", "rf_kd"]
SUBJECT_KEYS = [
    "teacher",
    "student_A_scratch",
    "student_A_rf_kd",
    "student_B_scratch",
    "student_B_rf_kd",
]
RANDOM_CONTROL_KEYS = [
    "control_student_A_fully_reinitialized",
    "control_student_B_fully_reinitialized",
]

DEFAULT_DATASET = REPO_ROOT / "data/wsnds/WSN-DS.csv"
DEFAULT_SOURCE_ROOT = (
    REPO_ROOT
    / "results/wsnds/confirmation_runs_v2/local_feature_group_10seed_20260811"
    / "feature_group_10seed"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results/wsnds/evidence_completion_20260812"
    / "fgds_controlled_xai_transfer_10seed_v1"
)
COMMON_SOURCE = REPO_ROOT / "experiments/wsnds/leakage_free_rerun/tier15_common.py"
SOURCE_SNAPSHOTS = {
    "executed_controlled_xai_source.py": SCRIPT_PATH,
    "bound_common_source.py": COMMON_SOURCE,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def repo_path(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Required JSON is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=False) + "\n")


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def state_content_sha256(state: dict[str, torch.Tensor]) -> str:
    arrays = [state[name].detach().cpu().numpy() for name in sorted(state)]
    return sha256_arrays(*arrays)


def build_inventory(root: Path, status: str) -> dict[str, Any]:
    manifest_path = root / "artifact_manifest.json"
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == manifest_path or path.name.endswith((".tmp", ".tmp.npz")):
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "file_count_excluding_manifest": len(files),
        "files": files,
    }


def verify_inventory(
    root: Path,
    accepted_status: set[str],
    *,
    protocol_id: str = PROTOCOL_ID,
) -> dict[str, Any]:
    manifest_path = root / "artifact_manifest.json"
    manifest = read_json(manifest_path)
    require(manifest.get("protocol_id") == protocol_id, f"Manifest protocol mismatch: {root}")
    require(manifest.get("status") in accepted_status, f"Manifest status mismatch: {root}")
    files = manifest.get("files")
    require(isinstance(files, list) and files, f"Manifest inventory is empty: {root}")
    require(manifest.get("file_count_excluding_manifest") == len(files), "Manifest count mismatch")
    seen: set[str] = set()
    for item in files:
        relative = item.get("path")
        require(isinstance(relative, str) and relative, "Invalid manifest path")
        relative_path = Path(relative)
        require(not relative_path.is_absolute() and ".." not in relative_path.parts, "Unsafe path")
        require(relative not in seen, f"Duplicate manifest path: {relative}")
        seen.add(relative)
        path = root / relative_path
        require(path.is_file(), f"Manifest artifact is missing: {path}")
        require(path.stat().st_size == item.get("size_bytes"), f"Size mismatch: {path}")
        require(sha256_file(path) == item.get("sha256"), f"Hash mismatch: {path}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path != manifest_path
        and not path.name.endswith((".tmp", ".tmp.npz"))
    }
    require(actual == seen, f"Manifest does not exactly cover files under {root}")
    return manifest


def environment_record() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "shap": shap.__version__,
    }


def soften_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    require(values.ndim == 2 and values.shape[1] == len(CLASS_NAMES), "Probability shape differs")
    require(np.all(np.isfinite(values)) and np.all(values >= 0.0), "Invalid probabilities")
    require(np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-6), "Rows do not sum to one")
    softened = np.maximum(values, PROBABILITY_FLOOR) ** (1.0 / KD_TEMPERATURE)
    softened /= softened.sum(axis=1, keepdims=True)
    return softened


def select_balanced_group_representatives(
    raw_features: np.ndarray,
    labels: np.ndarray,
    source_indices: np.ndarray,
    *,
    groups_per_class: int,
    seed: int,
    global_purity: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    source_indices = np.asarray(source_indices, dtype=np.int64)
    require(len(raw_features) == len(labels) == len(source_indices), "Sampling arrays differ")
    group_hashes = verified_feature_hashes(raw_features)
    unique_hashes, inverse = np.unique(group_hashes, return_inverse=True)
    group_count = len(unique_hashes)
    minimum_label = np.full(group_count, len(CLASS_NAMES), dtype=np.int64)
    maximum_label = np.full(group_count, -1, dtype=np.int64)
    np.minimum.at(minimum_label, inverse, labels)
    np.maximum.at(maximum_label, inverse, labels)
    pure = minimum_label == maximum_label
    global_hashes = global_purity["unique_hashes"]
    positions = np.searchsorted(global_hashes, unique_hashes)
    require(
        np.all(positions < len(global_hashes))
        and np.array_equal(global_hashes[positions], unique_hashes),
        "A partition feature group is absent from the global purity table",
    )
    globally_pure = global_purity["pure"][positions]
    global_labels = global_purity["labels"][positions]
    pure &= globally_pure & (minimum_label == global_labels)

    first_partition_index = np.full(group_count, -1, dtype=np.int64)
    for partition_index in np.argsort(source_indices, kind="stable"):
        group_index = int(inverse[partition_index])
        if first_partition_index[group_index] < 0:
            first_partition_index[group_index] = int(partition_index)
    require(np.all(first_partition_index >= 0), "A feature group has no representative")

    rng = np.random.RandomState(seed)
    selected_by_class: list[np.ndarray] = []
    for class_index in range(len(CLASS_NAMES)):
        candidates = np.flatnonzero(pure & (minimum_label == class_index))
        require(
            len(candidates) >= groups_per_class,
            f"Class {class_index} has only {len(candidates)} eligible groups",
        )
        selected_by_class.append(
            np.asarray(rng.choice(candidates, groups_per_class, replace=False), dtype=np.int64)
        )
    interleaved_groups = np.asarray(
        [selected_by_class[class_index][offset] for offset in range(groups_per_class) for class_index in range(len(CLASS_NAMES))],
        dtype=np.int64,
    )
    partition_indices = first_partition_index[interleaved_groups]
    observed_labels = labels[partition_indices]
    expected_labels = np.tile(np.arange(len(CLASS_NAMES), dtype=np.int64), groups_per_class)
    require(np.array_equal(observed_labels, expected_labels), "Balanced class interleave differs")
    require(len(np.unique(interleaved_groups)) == len(interleaved_groups), "Selected groups repeat")
    return {
        "partition_indices": partition_indices,
        "source_row_indices": source_indices[partition_indices],
        "labels": observed_labels,
        "exact_feature_group_hashes": unique_hashes[interleaved_groups].astype(np.uint64),
    }


def build_global_group_purity(
    raw_features: np.ndarray,
    labels: np.ndarray,
) -> dict[str, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    group_hashes = verified_feature_hashes(raw_features)
    unique_hashes, inverse = np.unique(group_hashes, return_inverse=True)
    minimum_label = np.full(len(unique_hashes), len(CLASS_NAMES), dtype=np.int64)
    maximum_label = np.full(len(unique_hashes), -1, dtype=np.int64)
    np.minimum.at(minimum_label, inverse, labels)
    np.maximum.at(maximum_label, inverse, labels)
    pure = minimum_label == maximum_label
    return {
        "unique_hashes": unique_hashes.astype(np.uint64),
        "pure": pure,
        "labels": minimum_label,
    }


def exact_paired_signed_rank(differences: np.ndarray) -> dict[str, Any]:
    values = np.asarray(differences, dtype=np.float64)
    require(values.ndim == 1 and np.all(np.isfinite(values)), "Signed-rank values are invalid")
    nonzero = values[values != 0.0]
    zero_count = int(len(values) - len(nonzero))
    if len(nonzero) == 0:
        return {
            "statistic_abs_signed_rank_sum": 0.0,
            "p_value_two_sided": 1.0,
            "nonzero_difference_count": 0,
            "zero_difference_count": zero_count,
            "enumerated_sign_assignments": 1,
            "rank_tie_method": "average",
            "zero_method": "wilcox",
            "enumeration": "exhaustive",
        }
    ranks = rankdata(np.abs(nonzero), method="average")
    observed = float(abs(np.sum(np.sign(nonzero) * ranks)))
    possible = np.asarray(
        [abs(float(np.dot(signs, ranks))) for signs in product((-1.0, 1.0), repeat=len(nonzero))],
        dtype=np.float64,
    )
    p_value = float(np.mean(possible >= observed - 1e-15))
    return {
        "statistic_abs_signed_rank_sum": observed,
        "p_value_two_sided": p_value,
        "nonzero_difference_count": int(len(nonzero)),
        "zero_difference_count": zero_count,
        "enumerated_sign_assignments": int(2 ** len(nonzero)),
        "rank_tie_method": "average",
        "zero_method": "wilcox",
        "enumeration": "exhaustive",
    }


def holm_adjust(raw: dict[str, float]) -> dict[str, float]:
    require(raw, "Holm family is empty")
    ordered = sorted(raw, key=lambda key: (raw[key], key))
    adjusted: dict[str, float] = {}
    running = 0.0
    family_size = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (family_size - rank) * float(raw[key])))
        adjusted[key] = running
    return adjusted


class ExactInvariantIndependentMasker(shap.maskers.Independent):
    """Independent masker with exact cache invalidation."""

    def invariants(self, x: np.ndarray) -> np.ndarray:
        value = np.asarray(x)
        if value.shape != self.data.shape[1:]:
            raise ValueError("Explained row does not match background shape")
        return np.equal(value, self.data)


def fit_reconstructed_calibrated_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    groups: np.ndarray,
) -> tuple[CalibratedClassifierCV, dict[str, Any]]:
    groups = np.asarray(groups)
    require(groups.shape == (len(y_train),), "RF calibration group vector has the wrong shape")
    splitter = StratifiedGroupKFold(
        n_splits=RF_CONFIG["calibration_cv"],
        shuffle=True,
        random_state=seed,
    )
    folds = list(splitter.split(X_train, y_train, groups))
    overlap_counts = []
    for train_indices, validation_indices in folds:
        overlap_counts.append(
            len(
                set(map(int, groups[train_indices]))
                & set(map(int, groups[validation_indices]))
            )
        )
    require(not any(overlap_counts), f"Exact feature groups cross RF folds: {overlap_counts}")
    teacher = CalibratedClassifierCV(
        RandomForestClassifier(
            n_estimators=RF_CONFIG["n_estimators"],
            max_depth=RF_CONFIG["max_depth"],
            random_state=seed,
            n_jobs=-1,
        ),
        method=RF_CONFIG["calibration_method"],
        cv=folds,
    )
    teacher.fit(X_train, y_train)
    return teacher, {
        "strategy": "stratified_group_kfold",
        "folds": len(folds),
        "group_overlap_per_fold": overlap_counts,
        "unique_groups": int(len(np.unique(groups))),
    }


def student_predictor(
    model: StudentMLP,
    *,
    temperature: float = KD_TEMPERATURE,
) -> Callable[[np.ndarray], np.ndarray]:
    require(temperature > 0.0, "Student probability temperature must be positive")

    def predict(values: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(np.asarray(values, dtype=np.float32))
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(tensor), 4096):
                chunks.append(F.softmax(model(tensor[start : start + 4096]) / temperature, dim=1).numpy())
        result = np.concatenate(chunks).astype(np.float64, copy=False)
        require(np.all(np.isfinite(result)), "Student predictor returned non-finite values")
        return result

    return predict


def teacher_predictor(model: Any) -> Callable[[np.ndarray], np.ndarray]:
    def predict(values: np.ndarray) -> np.ndarray:
        return soften_probabilities(model.predict_proba(values))

    return predict


def load_torch_mapping(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    require(isinstance(value, dict), f"Expected a mapping in {path}")
    return value


def load_student(path: Path, hidden_dims: tuple[int, int]) -> tuple[StudentMLP, str]:
    state = load_torch_mapping(path)
    if "state_dict" in state:
        state = state["state_dict"]
    require(all(isinstance(value, torch.Tensor) for value in state.values()), "Invalid state dictionary")
    model = StudentMLP(17, hidden_dims, len(CLASS_NAMES))
    model.load_state_dict(state)
    model.eval()
    return model, state_content_sha256(state)


def probability_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame.columns if column.startswith("probability_")]
    require(len(columns) == len(CLASS_NAMES), f"Unexpected probability columns: {columns}")
    return columns


def verify_external_manifest(root: Path, expected_protocol: str) -> dict[str, Any]:
    return verify_inventory(root, {"complete"}, protocol_id=expected_protocol)


def reconstruct_context(dataset_path: Path, source_root: Path) -> dict[str, Any]:
    require(sha256_file(dataset_path) == EXPECTED_DATASET_SHA256, "Dataset hash mismatch")
    dataset = load_wsnds(dataset_path)
    require(dataset["dataset_sha256"] == EXPECTED_DATASET_SHA256, "Loaded dataset hash mismatch")
    split = feature_group_split(dataset["features"], dataset["labels"])
    scaled, scaler = apply_train_scaler(split)
    split_sha = sha256_arrays(split["train_indices"], split["validation_indices"], split["test_indices"])
    scaler_sha = sha256_arrays(
        np.asarray(scaler.mean_, dtype=np.float64),
        np.asarray(scaler.scale_, dtype=np.float64),
        np.asarray(scaler.var_, dtype=np.float64),
    )
    require(split_sha == EXPECTED_SPLIT_SHA256, "Reconstructed split hash mismatch")
    require(scaler_sha == EXPECTED_SCALER_SHA256, "Reconstructed scaler hash mismatch")
    for name, expected_size in EXPECTED_SPLIT_SIZES.items():
        require(len(split[f"y_{name}"]) == expected_size, f"{name} split size differs")
    require(split["group_audit"]["train_validation_feature_overlap"] == 0, "Train/validation groups overlap")
    require(split["group_audit"]["train_test_feature_overlap"] == 0, "Train/test groups overlap")
    require(split["group_audit"]["validation_test_feature_overlap"] == 0, "Validation/test groups overlap")

    with np.load(source_root / "split_indices.npz", allow_pickle=False) as payload:
        expected_names = {"train_indices", "validation_indices", "test_indices"}
        require(set(payload.files) == expected_names, "Source split NPZ schema differs")
        for name in expected_names:
            require(np.array_equal(payload[name], split[name]), f"Source split array differs: {name}")
    with np.load(source_root / "scaler_parameters.npz", allow_pickle=False) as payload:
        require(
            set(payload.files) == {"mean", "scale", "var", "n_samples_seen"},
            "Scaler NPZ schema differs",
        )
        require(np.array_equal(payload["mean"], scaler.mean_), "Scaler mean differs")
        require(np.array_equal(payload["scale"], scaler.scale_), "Scaler scale differs")
        require(np.array_equal(payload["var"], scaler.var_), "Scaler variance differs")
        require(
            np.array_equal(
                payload["n_samples_seen"],
                np.asarray([EXPECTED_SPLIT_SIZES["train"]], dtype=np.int64),
            ),
            "Scaler fitted-row count differs",
        )
    execution = read_json(source_root / "execution_contract.json")
    require(execution.get("protocol_id") == SOURCE_PROTOCOL_ID, "Source protocol differs")
    require(execution.get("seeds") == EXPECTED_SEEDS, "Source seed list differs")
    require(execution.get("dataset_sha256") == EXPECTED_DATASET_SHA256, "Source dataset differs")
    require(execution.get("split_indices_sha256") == EXPECTED_SPLIT_SHA256, "Source split differs")
    require(execution.get("scaler_sha256") == EXPECTED_SCALER_SHA256, "Source scaler differs")
    require(execution.get("kd_hyperparameters") == {"T": 4.0, "alpha": 0.7}, "Source KD settings differ")
    require(execution.get("teacher_config") == RF_CONFIG, "Source RF configuration differs")
    require(execution.get("common_module_sha256") == sha256_file(COMMON_SOURCE), "Common source differs")
    return {
        "dataset": dataset,
        "split": split,
        "scaled": scaled,
        "scaler": scaler,
        "split_sha256": split_sha,
        "scaler_sha256": scaler_sha,
        "source_execution": execution,
    }


def build_sampling(context: dict[str, Any]) -> dict[str, np.ndarray]:
    split = context["split"]
    dataset = context["dataset"]
    global_purity = build_global_group_purity(dataset["features"], dataset["labels"])
    cohort = select_balanced_group_representatives(
        split["X_test_raw"],
        split["y_test"],
        split["test_indices"],
        groups_per_class=COHORT_GROUPS_PER_CLASS,
        seed=COHORT_SEED,
        global_purity=global_purity,
    )
    background = select_balanced_group_representatives(
        split["X_train_raw"],
        split["y_train"],
        split["train_indices"],
        groups_per_class=BACKGROUND_GROUPS_PER_CLASS,
        seed=BACKGROUND_SEED,
        global_purity=global_purity,
    )
    arrays: dict[str, np.ndarray] = {}
    for prefix, values in (("cohort", cohort), ("background", background)):
        for name, value in values.items():
            arrays[f"{prefix}_{name}"] = np.asarray(value)
    arrays["global_exact_feature_group_count"] = np.asarray(
        [len(global_purity["unique_hashes"])], dtype=np.int64
    )
    arrays["global_mixed_label_group_count"] = np.asarray(
        [int(np.sum(~global_purity["pure"]))], dtype=np.int64
    )
    return arrays


def sampling_array_contract(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        name: {
            "shape": list(np.asarray(value).shape),
            "dtype": str(np.asarray(value).dtype),
            "content_sha256": sha256_arrays(np.asarray(value)),
        }
        for name, value in sorted(arrays.items())
    }


def load_sampling(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name].copy() for name in payload.files}


def verify_sampling(path: Path, expected: dict[str, np.ndarray]) -> None:
    observed = load_sampling(path)
    require(set(observed) == set(expected), "Sampling artifact schema differs")
    for name, value in expected.items():
        require(observed[name].dtype == value.dtype, f"Sampling dtype differs: {name}")
        require(np.array_equal(observed[name], value), f"Sampling values differ: {name}")


def run_permutation_shap(
    predictor: Callable[[np.ndarray], np.ndarray],
    background: np.ndarray,
    explained: np.ndarray,
) -> dict[str, np.ndarray]:
    require(np.all(np.isfinite(background)) and np.all(np.isfinite(explained)), "SHAP inputs are non-finite")
    masker = ExactInvariantIndependentMasker(background, max_samples=len(background))
    explainer = shap.Explainer(
        predictor,
        masker,
        algorithm="permutation",
        output_names=CLASS_NAMES,
        seed=PERMUTATION_SEED,
    )
    explanation = explainer(
        explained,
        max_evals=(2 * explained.shape[1] + 1) * PERMUTATION_REPEATS,
        batch_size=256,
        silent=False,
    )
    values = np.asarray(explanation.values, dtype=np.float64)
    base_values = np.asarray(explanation.base_values, dtype=np.float64)
    outputs = predictor(explained)
    require(values.shape == (COHORT_SIZE, 17, len(CLASS_NAMES)), "SHAP tensor shape differs")
    require(base_values.shape == outputs.shape == (COHORT_SIZE, len(CLASS_NAMES)), "SHAP output shape differs")
    require(np.all(np.isfinite(values)) and np.all(np.isfinite(base_values)), "SHAP values are non-finite")
    residual = np.abs(base_values + values.sum(axis=1) - outputs)
    require(float(residual.max()) <= LOCAL_ACCURACY_ATOL, "SHAP local accuracy gate failed")
    return {
        "values": values,
        "base_values": base_values,
        "model_outputs": outputs,
        "local_accuracy_residual": residual,
    }


def selected_attributions(values: np.ndarray, teacher_classes: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    teacher_classes = np.asarray(teacher_classes, dtype=np.int64)
    require(values.shape == (COHORT_SIZE, 17, len(CLASS_NAMES)), "Attribution tensor differs")
    require(teacher_classes.shape == (COHORT_SIZE,), "Teacher class vector differs")
    require(np.all((teacher_classes >= 0) & (teacher_classes < len(CLASS_NAMES))), "Teacher class out of range")
    return values[np.arange(COHORT_SIZE), :, teacher_classes]


def cosine_rows(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    require(left.shape == right.shape and left.ndim == 2, "Cosine arrays differ")
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    eligible = (left_norm >= ATTRIBUTION_NORM_EPSILON) & (right_norm >= ATTRIBUTION_NORM_EPSILON)
    values = np.full(len(left), np.nan, dtype=np.float64)
    values[eligible] = np.sum(left[eligible] * right[eligible], axis=1) / (
        left_norm[eligible] * right_norm[eligible]
    )
    return values, eligible


def rank_agreement(left: np.ndarray, right: np.ndarray) -> float | None:
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else None


def pair_metrics(
    teacher_values: np.ndarray,
    scratch_values: np.ndarray,
    kd_values: np.ndarray,
    teacher_classes: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    teacher = selected_attributions(teacher_values, teacher_classes)
    scratch = selected_attributions(scratch_values, teacher_classes)
    kd = selected_attributions(kd_values, teacher_classes)
    teacher_norm = np.linalg.norm(teacher, axis=1)
    scratch_norm = np.linalg.norm(scratch, axis=1)
    kd_norm = np.linalg.norm(kd, axis=1)
    eligible = (
        (teacher_norm >= ATTRIBUTION_NORM_EPSILON)
        & (scratch_norm >= ATTRIBUTION_NORM_EPSILON)
        & (kd_norm >= ATTRIBUTION_NORM_EPSILON)
    )
    scratch_cosine, _ = cosine_rows(teacher, scratch)
    kd_cosine, _ = cosine_rows(teacher, kd)
    delta = kd_cosine - scratch_cosine
    class_rows = []
    inconclusive_reasons = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        selected = (labels == class_index) & eligible
        count = int(selected.sum())
        if count < MIN_ELIGIBLE_PER_CLASS:
            inconclusive_reasons.append(
                f"{class_name} has {count} eligible rows; minimum is {MIN_ELIGIBLE_PER_CLASS}"
            )
        class_rows.append(
            {
                "class_index": class_index,
                "class_name": class_name,
                "eligible_rows": count,
                "total_rows": int(np.sum(labels == class_index)),
                "scratch_alignment_mean": float(np.mean(scratch_cosine[selected])) if count else None,
                "rf_kd_alignment_mean": float(np.mean(kd_cosine[selected])) if count else None,
                "alignment_gain_mean": float(np.mean(delta[selected])) if count else None,
            }
        )
    defined_rows = [row for row in class_rows if row["alignment_gain_mean"] is not None]
    if np.any(eligible):
        scratch_global = np.mean(np.abs(scratch[eligible]), axis=0)
        kd_global = np.mean(np.abs(kd[eligible]), axis=0)
        teacher_global = np.mean(np.abs(teacher[eligible]), axis=0)
        teacher_order = np.argsort(-teacher_global, kind="stable")
        scratch_order = np.argsort(-scratch_global, kind="stable")
        kd_order = np.argsort(-kd_global, kind="stable")
    else:
        scratch_global = kd_global = teacher_global = None
        teacher_order = scratch_order = kd_order = None
    material_scratch = np.maximum(np.abs(teacher), np.abs(scratch)) >= COMPONENT_MATERIALITY_EPSILON
    material_kd = np.maximum(np.abs(teacher), np.abs(kd)) >= COMPONENT_MATERIALITY_EPSILON
    scratch_sign = (
        float(np.mean(np.sign(teacher[material_scratch]) == np.sign(scratch[material_scratch])))
        if np.any(material_scratch)
        else None
    )
    kd_sign = (
        float(np.mean(np.sign(teacher[material_kd]) == np.sign(kd[material_kd])))
        if np.any(material_kd)
        else None
    )
    conclusive = not inconclusive_reasons
    return {
        "eligibility": {
            "rule": "RF, scratch, and RF-KD selected-class L2 norms must all be at least 1e-6",
            "eligible_rows": int(eligible.sum()),
            "undefined_rows": int((~eligible).sum()),
            "coverage_fraction": float(np.mean(eligible)),
            "minimum_per_class_gate": MIN_ELIGIBLE_PER_CLASS,
            "status": "conclusive" if conclusive else "inconclusive",
            "inconclusive_reasons": inconclusive_reasons,
        },
        "primary": {
            "status": "conclusive" if conclusive else "inconclusive",
            "inconclusive_reasons": inconclusive_reasons,
            "descriptive_defined_class_count": len(defined_rows),
            "scratch_alignment_macro_class_mean": (
                float(np.mean([row["scratch_alignment_mean"] for row in defined_rows]))
                if defined_rows
                else None
            ),
            "rf_kd_alignment_macro_class_mean": (
                float(np.mean([row["rf_kd_alignment_mean"] for row in defined_rows]))
                if defined_rows
                else None
            ),
            "rf_kd_minus_scratch_alignment_gain": (
                float(np.mean([row["alignment_gain_mean"] for row in defined_rows]))
                if defined_rows
                else None
            ),
        },
        "per_class": class_rows,
        "secondary": {
            "scratch_global_rank_rho": (
                rank_agreement(teacher_global, scratch_global) if teacher_global is not None else None
            ),
            "rf_kd_global_rank_rho": (
                rank_agreement(teacher_global, kd_global) if teacher_global is not None else None
            ),
            "scratch_top5_overlap": (
                int(len(set(teacher_order[:5]) & set(scratch_order[:5])))
                if teacher_order is not None
                else None
            ),
            "rf_kd_top5_overlap": (
                int(len(set(teacher_order[:5]) & set(kd_order[:5])))
                if teacher_order is not None
                else None
            ),
            "scratch_thresholded_sign_agreement": scratch_sign,
            "rf_kd_thresholded_sign_agreement": kd_sign,
            "teacher_global_mean_absolute_shap": teacher_global.tolist() if teacher_global is not None else None,
            "scratch_global_mean_absolute_shap": scratch_global.tolist() if scratch_global is not None else None,
            "rf_kd_global_mean_absolute_shap": kd_global.tolist() if kd_global is not None else None,
        },
    }


def verify_prediction_artifact(
    path: Path,
    source_indices: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    atol: float,
) -> dict[str, Any]:
    frame = pd.read_csv(path)
    columns = probability_columns(frame)
    saved = frame[columns].to_numpy(dtype=np.float64)
    require(len(frame) == len(labels), f"Prediction row count differs: {path}")
    require(np.array_equal(frame["source_row_index"].to_numpy(np.int64), source_indices), f"Source indices differ: {path}")
    require(np.array_equal(frame["true_label"].to_numpy(np.int64), labels), f"Labels differ: {path}")
    require(np.array_equal(frame["predicted_label"].to_numpy(np.int64), probabilities.argmax(axis=1)), f"Predictions differ: {path}")
    maximum_delta = float(np.max(np.abs(saved - probabilities)))
    require(maximum_delta <= atol, f"Probability replay differs by {maximum_delta}: {path}")
    return {"file_sha256": sha256_file(path), "maximum_absolute_probability_delta": maximum_delta}


def verify_student_artifact(
    seed_root: Path,
    completion: dict[str, Any],
    student_key: str,
    route: str,
    expected_teacher_provenance: dict[str, Any],
) -> tuple[StudentMLP, dict[str, Any]]:
    result_key = f"{student_key}_{route}"
    result = completion["student_results"][result_key]
    require(result.get("route") == route, f"Route metadata differs: {result_key}")
    letter = "A" if student_key == "student_A" else "B"
    route_label = "Small_MLP_scratch" if route == "scratch" else "KD_from_RF"
    plain_path = seed_root / f"student_{letter}_{route_label}_fp32.pt"
    rich_path = seed_root / f"student_{letter}_{route_label}_artifact.pt"
    prediction_path = seed_root / f"student_{letter}_{route_label}_test_predictions.csv"
    require(plain_path.name == result["plain_state_dict"], f"Plain checkpoint name differs: {result_key}")
    require(rich_path.name == result["rich_artifact"], f"Rich artifact name differs: {result_key}")
    require(prediction_path.name == result["test_predictions"], f"Prediction name differs: {result_key}")
    require(sha256_file(plain_path) == result["plain_state_dict_sha256"], f"Plain hash differs: {result_key}")
    require(sha256_file(rich_path) == result["rich_artifact_sha256"], f"Rich hash differs: {result_key}")
    require(sha256_file(prediction_path) == result["test_predictions_sha256"], f"Prediction hash differs: {result_key}")
    model, trained_state_hash = load_student(plain_path, STUDENT_SPECS[student_key])
    require(trained_state_hash == result["trained_state_sha256"], f"Trained state differs: {result_key}")
    rich = load_torch_mapping(rich_path)
    required = {
        "protocol_id": SOURCE_PROTOCOL_ID,
        "seed": completion["seed"],
        "student": student_key,
        "route": route,
        "input_dim": 17,
        "hidden_dims": list(STUDENT_SPECS[student_key]),
        "num_classes": len(CLASS_NAMES),
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "scaler_sha256": EXPECTED_SCALER_SHA256,
        "initial_state_sha256": result["initial_state_sha256"],
        "trained_state_sha256": trained_state_hash,
        "kd_hyperparameters": ({"T": 4.0, "alpha": 0.7} if route == "rf_kd" else None),
    }
    for key, expected in required.items():
        require(rich.get(key) == expected, f"Rich artifact differs for {result_key}: {key}")
    rich_state = rich.get("state_dict")
    require(isinstance(rich_state, dict), f"Rich state missing: {result_key}")
    require(state_content_sha256(rich_state) == trained_state_hash, f"Rich/plain state differs: {result_key}")
    if route == "scratch":
        require(rich.get("teacher_soft_target_provenance") is None, "Scratch route has teacher provenance")
    else:
        provenance = rich.get("teacher_soft_target_provenance")
        require(isinstance(provenance, dict), "RF-KD route lacks teacher provenance")
        require(
            provenance == expected_teacher_provenance,
            f"RF-KD teacher provenance differs: {result_key}",
        )
    return model, {
        "student": student_key,
        "route": route,
        "plain_checkpoint": repo_path(plain_path),
        "plain_checkpoint_sha256": sha256_file(plain_path),
        "rich_artifact": repo_path(rich_path),
        "rich_artifact_sha256": sha256_file(rich_path),
        "prediction_artifact": repo_path(prediction_path),
        "prediction_artifact_sha256": sha256_file(prediction_path),
        "initial_state_sha256": result["initial_state_sha256"],
        "trained_state_sha256": trained_state_hash,
    }


def reconstruct_seed_subjects(
    source_root: Path,
    context: dict[str, Any],
    seed: int,
    *,
    include_random_controls: bool,
) -> dict[str, Any]:
    seed_root = source_root / f"seed_{seed}"
    verify_external_manifest(seed_root, SOURCE_PROTOCOL_ID)
    completion = read_json(seed_root / "seed_completion.json")
    required_completion = {
        "protocol_id": SOURCE_PROTOCOL_ID,
        "seed": seed,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "split_indices_sha256": EXPECTED_SPLIT_SHA256,
        "scaler_sha256": EXPECTED_SCALER_SHA256,
        "status": "complete",
        "teacher_config": RF_CONFIG,
    }
    for key, expected in required_completion.items():
        require(completion.get(key) == expected, f"Seed completion differs for {seed}: {key}")
    split = context["split"]
    scaled = context["scaled"]
    groups = verified_feature_hashes(split["X_train_raw"])
    teacher, calibration_audit = fit_reconstructed_calibrated_rf(
        scaled["X_train"], split["y_train"], seed, groups=groups
    )
    train_probabilities = teacher.predict_proba(scaled["X_train"]).astype(np.float32)
    preserved_train_path = seed_root / "rf_train_probabilities.npy"
    preserved_train = np.load(preserved_train_path, allow_pickle=False)
    require(preserved_train.dtype == np.float32, f"RF train dtype differs for seed {seed}")
    require(np.array_equal(preserved_train, train_probabilities), f"RF train probabilities differ for seed {seed}")
    expected_teacher_provenance = {
        "source_type": "fresh_calibrated_rf_soft_targets",
        "rf_seed": seed,
        "rf_config": RF_CONFIG,
        "calibration_audit": calibration_audit,
        "train_probability_content_sha256": sha256_arrays(train_probabilities),
    }
    require(
        completion.get("teacher_soft_target_provenance") == expected_teacher_provenance,
        f"Seed-level teacher provenance differs for seed {seed}",
    )
    test_probabilities = teacher.predict_proba(scaled["X_test"]).astype(np.float32).astype(np.float64)
    teacher_prediction = verify_prediction_artifact(
        seed_root / "RF_teacher_test_predictions.csv",
        split["test_indices"],
        split["y_test"],
        test_probabilities,
        atol=5e-8,
    )
    models: dict[str, StudentMLP] = {}
    identities: dict[str, Any] = {}
    for student_key in STUDENT_KEYS:
        for route in ROUTES:
            key = f"{student_key}_{route}"
            model, identity = verify_student_artifact(
                seed_root,
                completion,
                student_key,
                route,
                expected_teacher_provenance,
            )
            replay = student_predictor(model, temperature=1.0)(scaled["X_test"])
            letter = "A" if student_key == "student_A" else "B"
            route_label = "Small_MLP_scratch" if route == "scratch" else "KD_from_RF"
            identity["test_replay"] = verify_prediction_artifact(
                seed_root / f"student_{letter}_{route_label}_test_predictions.csv",
                split["test_indices"],
                split["y_test"],
                replay,
                atol=MODEL_REPLAY_ATOL,
            )
            models[key] = model
            identities[key] = identity
        require(
            identities[f"{student_key}_scratch"]["initial_state_sha256"]
            == identities[f"{student_key}_rf_kd"]["initial_state_sha256"],
            f"Scratch/KD initial states differ for {student_key}, seed {seed}",
        )
    random_models: dict[str, StudentMLP] = {}
    random_identities: dict[str, Any] = {}
    if include_random_controls:
        require(seed == 42, "Randomization controls are restricted to seed 42")
        for student_key in STUDENT_KEYS:
            random_seed = RANDOMIZATION_SEEDS[student_key]
            torch.manual_seed(random_seed)
            model = StudentMLP(17, STUDENT_SPECS[student_key], len(CLASS_NAMES)).eval()
            state_hash = state_content_sha256(model.state_dict())
            trained_hashes = {
                identities[f"{student_key}_{route}"]["trained_state_sha256"] for route in ROUTES
            }
            require(state_hash not in trained_hashes, f"Random control duplicates a trained {student_key} state")
            key = f"control_{student_key}_fully_reinitialized"
            random_models[key] = model
            random_identities[key] = {
                "subject_role": "architecture_specific_fully_reinitialized_sanity_control",
                "student_architecture": student_key,
                "hidden_dims": list(STUDENT_SPECS[student_key]),
                "randomization_seed": random_seed,
                "state_sha256": state_hash,
                "trained_route": False,
                "must_not_be_interpreted_as": ["scratch", "rf_kd", "trained_model"],
            }
    return {
        "teacher": teacher,
        "models": models,
        "random_models": random_models,
        "identity": {
            "seed": seed,
            "seed_completion": repo_path(seed_root / "seed_completion.json"),
            "seed_completion_sha256": sha256_file(seed_root / "seed_completion.json"),
            "teacher": {
                "identity": "deterministic output-validated reconstruction; RF object is not serialized",
                "calibration_audit": calibration_audit,
                "train_probability_file_sha256": sha256_file(preserved_train_path),
                "train_probability_content_sha256": sha256_arrays(train_probabilities),
                "test_prediction_verification": teacher_prediction,
            },
            "trained_subjects": identities,
            "randomization_controls": random_identities,
        },
    }


def seed_job_contract(
    contract: dict[str, Any],
    seed_identity: dict[str, Any],
    sampling_arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    value = {
        "protocol_id": PROTOCOL_ID,
        "execution_contract_id": contract["execution_contract_id"],
        "seed": seed_identity["seed"],
        "source_seed_completion_sha256": seed_identity["seed_completion_sha256"],
        "teacher": seed_identity["teacher"],
        "trained_subjects": seed_identity["trained_subjects"],
        "randomization_controls": seed_identity["randomization_controls"],
        "sampling": sampling_array_contract(sampling_arrays),
        "output_contract": "five-class probabilities softened at T=4",
        "selected_output": "RF teacher argmax class at T=4 for each cohort row",
        "permutation_seed": PERMUTATION_SEED,
        "permutation_repeats": PERMUTATION_REPEATS,
        "attribution_norm_epsilon": ATTRIBUTION_NORM_EPSILON,
    }
    value["job_contract_id"] = canonical_sha256(value)
    return value


def expected_subjects(seed: int) -> list[str]:
    return SUBJECT_KEYS + (RANDOM_CONTROL_KEYS if seed == 42 else [])


def subject_arrays(payload: dict[str, np.ndarray], subject: str) -> dict[str, np.ndarray]:
    return {
        "values": payload[f"{subject}__values"],
        "base_values": payload[f"{subject}__base_values"],
        "model_outputs": payload[f"{subject}__model_outputs"],
        "local_accuracy_residual": payload[f"{subject}__local_accuracy_residual"],
    }


def validate_subject_arrays(values: dict[str, np.ndarray], subject: str) -> None:
    require(values["values"].shape == (COHORT_SIZE, 17, len(CLASS_NAMES)), f"SHAP shape differs: {subject}")
    require(values["base_values"].shape == (COHORT_SIZE, len(CLASS_NAMES)), f"Base-value shape differs: {subject}")
    require(values["model_outputs"].shape == (COHORT_SIZE, len(CLASS_NAMES)), f"Output shape differs: {subject}")
    require(values["local_accuracy_residual"].shape == (COHORT_SIZE, len(CLASS_NAMES)), f"Residual shape differs: {subject}")
    for name, array in values.items():
        require(np.asarray(array).dtype == np.float64, f"{subject} {name} dtype differs")
        require(np.all(np.isfinite(array)), f"{subject} {name} contains non-finite values")
    require(
        np.allclose(values["model_outputs"].sum(axis=1), 1.0, rtol=0.0, atol=1e-10),
        f"Output simplex differs: {subject}",
    )
    recomputed_residual = np.abs(
        values["base_values"] + values["values"].sum(axis=1) - values["model_outputs"]
    )
    require(
        np.array_equal(recomputed_residual, values["local_accuracy_residual"]),
        f"Local residual differs: {subject}",
    )
    require(float(recomputed_residual.max()) <= LOCAL_ACCURACY_ATOL, f"Local accuracy failed: {subject}")


def macro_class_pairwise_cosine(
    left: np.ndarray,
    right: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    cosine, eligible = cosine_rows(left, right)
    per_class = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        selected = eligible & (labels == class_index)
        per_class.append(
            {
                "class_index": class_index,
                "class_name": class_name,
                "eligible_rows": int(selected.sum()),
                "mean_cosine": float(np.mean(cosine[selected])) if np.any(selected) else None,
            }
        )
    finite_class_means = [row["mean_cosine"] for row in per_class if row["mean_cosine"] is not None]
    return {
        "eligible_rows": int(eligible.sum()),
        "undefined_rows": int((~eligible).sum()),
        "macro_class_mean_cosine": float(np.mean(finite_class_means)) if finite_class_means else None,
        "per_class": per_class,
    }


def compute_seed_metrics(
    payload: dict[str, np.ndarray],
    sampling: dict[str, np.ndarray],
    seed: int,
) -> dict[str, Any]:
    subjects = expected_subjects(seed)
    for subject in subjects:
        validate_subject_arrays(subject_arrays(payload, subject), subject)
    labels = sampling["cohort_labels"].astype(np.int64)
    teacher_payload = subject_arrays(payload, "teacher")
    teacher_classes = teacher_payload["model_outputs"].argmax(axis=1).astype(np.int64)
    students: dict[str, Any] = {}
    for student_key in STUDENT_KEYS:
        scratch_payload = subject_arrays(payload, f"{student_key}_scratch")
        kd_payload = subject_arrays(payload, f"{student_key}_rf_kd")
        metrics = pair_metrics(
            teacher_payload["values"],
            scratch_payload["values"],
            kd_payload["values"],
            teacher_classes,
            labels,
        )
        metrics["secondary"]["scratch_hard_agreement_with_rf"] = float(
            np.mean(scratch_payload["model_outputs"].argmax(axis=1) == teacher_classes)
        )
        metrics["secondary"]["rf_kd_hard_agreement_with_rf"] = float(
            np.mean(kd_payload["model_outputs"].argmax(axis=1) == teacher_classes)
        )
        students[student_key] = metrics
    randomization: dict[str, Any] = {}
    if seed == 42:
        teacher_selected = selected_attributions(teacher_payload["values"], teacher_classes)
        for student_key in STUDENT_KEYS:
            scratch_selected = selected_attributions(
                subject_arrays(payload, f"{student_key}_scratch")["values"], teacher_classes
            )
            kd_selected = selected_attributions(
                subject_arrays(payload, f"{student_key}_rf_kd")["values"], teacher_classes
            )
            control_key = f"control_{student_key}_fully_reinitialized"
            random_selected = selected_attributions(
                subject_arrays(payload, control_key)["values"], teacher_classes
            )
            randomization[student_key] = {
                "control_subject_key": control_key,
                "control_role": "architecture_specific_fully_reinitialized_sanity_control",
                "trained_scratch_to_randomized": macro_class_pairwise_cosine(
                    scratch_selected, random_selected, labels
                ),
                "trained_rf_kd_to_randomized": macro_class_pairwise_cosine(
                    kd_selected, random_selected, labels
                ),
                "rf_teacher_to_randomized": macro_class_pairwise_cosine(
                    teacher_selected, random_selected, labels
                ),
                "inference": "descriptive sanity control; no p-value or pass threshold",
            }
    return {
        "seed": seed,
        "teacher_predicted_class_counts": np.bincount(
            teacher_classes, minlength=len(CLASS_NAMES)
        ).astype(int).tolist(),
        "local_accuracy": {
            subject: {
                "maximum_absolute_residual": float(
                    subject_arrays(payload, subject)["local_accuracy_residual"].max()
                ),
                "mean_absolute_residual": float(
                    subject_arrays(payload, subject)["local_accuracy_residual"].mean()
                ),
            }
            for subject in subjects
        },
        "students": students,
        "randomization_sanity": randomization,
    }


def load_seed_npz(path: Path, seed: int, job_contract: dict[str, Any]) -> dict[str, np.ndarray]:
    require(path.is_file(), f"Seed SHAP artifact is missing: {path}")
    with np.load(path, allow_pickle=False) as stored:
        payload = {name: stored[name].copy() for name in stored.files}
    expected_names = {"job_contract_json_utf8"}
    for subject in expected_subjects(seed):
        expected_names |= {
            f"{subject}__values",
            f"{subject}__base_values",
            f"{subject}__model_outputs",
            f"{subject}__local_accuracy_residual",
        }
    require(set(payload) == expected_names, f"Seed SHAP schema differs: {path}")
    observed_contract = payload.pop("job_contract_json_utf8").tobytes()
    require(observed_contract == canonical_json_bytes(job_contract), f"Seed job contract differs: {path}")
    return payload


def predictors_for_subjects(subjects: dict[str, Any]) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    predictors: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "teacher": teacher_predictor(subjects["teacher"]),
    }
    for key, model in subjects["models"].items():
        predictors[key] = student_predictor(model)
    for key, model in subjects["random_models"].items():
        predictors[key] = student_predictor(model)
    return predictors


def verify_seed_payload(
    seed_dir: Path,
    contract: dict[str, Any],
    context: dict[str, Any],
    sampling: dict[str, np.ndarray],
    subjects: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = verify_inventory(seed_dir, {"complete"})
    completion = read_json(seed_dir / "seed_completion.json")
    seed = int(completion.get("seed", -1))
    require(seed in EXPECTED_SEEDS, f"Unexpected seed directory: {seed_dir}")
    if subjects is None:
        subjects = reconstruct_seed_subjects(
            Path(REPO_ROOT / contract["source_run"]["path"]),
            context,
            seed,
            include_random_controls=seed == 42,
        )
    job_contract = seed_job_contract(contract, subjects["identity"], sampling)
    require(completion.get("protocol_id") == PROTOCOL_ID, "Seed protocol differs")
    require(completion.get("status") == "complete", "Seed completion status differs")
    require(completion.get("execution_contract_id") == contract["execution_contract_id"], "Seed execution contract differs")
    require(completion.get("job_contract") == job_contract, "Persisted seed job contract differs")
    artifact_path = seed_dir / "controlled_xai_shap_arrays.npz"
    require(completion.get("shap_artifact_sha256") == sha256_file(artifact_path), "Seed SHAP hash differs")
    payload = load_seed_npz(artifact_path, seed, job_contract)
    cohort = context["scaled"]["X_test"][sampling["cohort_partition_indices"]]
    predictors = predictors_for_subjects(subjects)
    require(set(predictors) == set(expected_subjects(seed)), "Subject predictor inventory differs")
    for subject, predictor in predictors.items():
        predicted = predictor(cohort)
        stored = subject_arrays(payload, subject)["model_outputs"]
        require(np.allclose(stored, predicted, rtol=0.0, atol=1e-10), f"Stored model outputs differ: {subject}")
    recomputed_metrics = compute_seed_metrics(payload, sampling, seed)
    require(completion.get("metrics") == recomputed_metrics, "Persisted seed metrics differ")
    require(completion.get("subject_identity") == subjects["identity"], "Subject identity differs")
    return {
        "seed": seed,
        "completion": completion,
        "metrics": recomputed_metrics,
        "verified_files": manifest["file_count_excluding_manifest"],
    }


def run_seed_job(
    output_dir: Path,
    contract: dict[str, Any],
    context: dict[str, Any],
    sampling: dict[str, np.ndarray],
    seed: int,
    resume: bool,
) -> dict[str, Any]:
    seed_dir = output_dir / f"seed_{seed}"
    if seed_dir.exists():
        require(resume, f"Refusing to overwrite existing seed output: {seed_dir}")
        return verify_seed_payload(seed_dir, contract, context, sampling)["completion"]
    subjects = reconstruct_seed_subjects(
        Path(REPO_ROOT / contract["source_run"]["path"]),
        context,
        seed,
        include_random_controls=seed == 42,
    )
    job_contract = seed_job_contract(contract, subjects["identity"], sampling)
    background = context["scaled"]["X_train"][sampling["background_partition_indices"]]
    cohort = context["scaled"]["X_test"][sampling["cohort_partition_indices"]]
    predictors = predictors_for_subjects(subjects)
    require(set(predictors) == set(expected_subjects(seed)), "Subject inventory differs")
    started = time.time()
    arrays: dict[str, np.ndarray] = {
        "job_contract_json_utf8": np.frombuffer(
            canonical_json_bytes(job_contract), dtype=np.uint8
        ).copy()
    }
    subject_seconds: dict[str, float] = {}
    for subject in expected_subjects(seed):
        subject_started = time.time()
        result = run_permutation_shap(predictors[subject], background, cohort)
        subject_seconds[subject] = time.time() - subject_started
        for name, value in result.items():
            arrays[f"{subject}__{name}"] = np.asarray(value, dtype=np.float64)
    metrics = compute_seed_metrics(
        {name: value for name, value in arrays.items() if name != "job_contract_json_utf8"},
        sampling,
        seed,
    )
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}_seed_{seed}_", dir=output_dir.parent))
    try:
        artifact_path = staging / "controlled_xai_shap_arrays.npz"
        atomic_save_npz(artifact_path, **arrays)
        completion = {
            "protocol_id": PROTOCOL_ID,
            "status": "complete",
            "seed": seed,
            "execution_contract_id": contract["execution_contract_id"],
            "job_contract": job_contract,
            "subject_identity": subjects["identity"],
            "shap_artifact": artifact_path.name,
            "shap_artifact_sha256": sha256_file(artifact_path),
            "subject_computation_wall_seconds": subject_seconds,
            "wall_seconds": time.time() - started,
            "metrics": metrics,
        }
        atomic_write_json(staging / "seed_completion.json", completion)
        atomic_write_json(staging / "artifact_manifest.json", build_inventory(staging, "complete"))
        verify_seed_payload(staging, contract, context, sampling, subjects)
        os.replace(staging, seed_dir)
        return completion
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def t_interval_95(values: np.ndarray) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    require(len(array) >= 2 and np.all(np.isfinite(array)), "Interval values are invalid")
    mean = float(array.mean())
    margin = float(t.ppf(0.975, len(array) - 1) * array.std(ddof=1) / math.sqrt(len(array)))
    return [mean - margin, mean + margin]


def aggregate_results(completions: list[dict[str, Any]]) -> tuple[dict[str, Any], pd.DataFrame]:
    require([item["seed"] for item in completions] == EXPECTED_SEEDS, "Completion seed order differs")
    tests: dict[str, Any] = {}
    rows = []
    for completion in completions:
        seed = completion["seed"]
        for student_key in STUDENT_KEYS:
            metrics = completion["metrics"]["students"][student_key]
            primary = metrics["primary"]
            rows.append(
                {
                    "seed": seed,
                    "student": student_key,
                    "primary_status": primary["status"],
                    "primary_inconclusive_reasons": (
                        "; ".join(primary["inconclusive_reasons"]) or "none"
                    ),
                    "descriptive_defined_class_count": primary["descriptive_defined_class_count"],
                    "scratch_alignment_macro_class_mean": primary["scratch_alignment_macro_class_mean"],
                    "rf_kd_alignment_macro_class_mean": primary["rf_kd_alignment_macro_class_mean"],
                    "rf_kd_minus_scratch_alignment_gain": primary["rf_kd_minus_scratch_alignment_gain"],
                    "eligible_rows": metrics["eligibility"]["eligible_rows"],
                    "undefined_rows": metrics["eligibility"]["undefined_rows"],
                }
            )
    frame = pd.DataFrame(rows)
    family_is_conclusive = bool(
        len(frame) == len(EXPECTED_SEEDS) * len(STUDENT_KEYS)
        and np.all(frame["primary_status"].to_numpy() == "conclusive")
        and np.all(
            np.isfinite(
                pd.to_numeric(
                    frame["rf_kd_minus_scratch_alignment_gain"], errors="coerce"
                ).to_numpy(np.float64)
            )
        )
    )
    raw_p: dict[str, float] = {}
    for student_key in STUDENT_KEYS:
        selected = frame[frame["student"] == student_key].sort_values("seed")
        ordered = selected.set_index("seed").loc[EXPECTED_SEEDS]
        raw_values = ordered["rf_kd_minus_scratch_alignment_gain"].tolist()
        differences = pd.to_numeric(
            ordered["rf_kd_minus_scratch_alignment_gain"], errors="coerce"
        ).to_numpy(np.float64)
        finite = differences[np.isfinite(differences)]
        seed_reasons = [
            f"seed {seed}: {reason}"
            for seed, status, reason in zip(
                EXPECTED_SEEDS,
                ordered["primary_status"].tolist(),
                ordered["primary_inconclusive_reasons"].tolist(),
            )
            if status != "conclusive"
        ]
        if not family_is_conclusive and not seed_reasons:
            seed_reasons = [
                "The predeclared two-test Holm family is incomplete because the other "
                "student has at least one inconclusive seed."
            ]
        signed_rank = exact_paired_signed_rank(differences) if family_is_conclusive else None
        if signed_rank is not None:
            raw_p[student_key] = signed_rank["p_value_two_sided"]
        tests[student_key] = {
            "status": "conclusive" if family_is_conclusive else "inconclusive",
            "inconclusive_reasons": seed_reasons,
            "student": student_key,
            "metric": "macro-class mean local signed-attribution cosine alignment gain",
            "orientation": "RF-KD minus scratch; positive means RF-KD is closer to RF",
            "seed_values": [float(value) if value is not None and np.isfinite(value) else None for value in raw_values],
            "defined_seed_count": int(len(finite)),
            "mean": float(finite.mean()) if len(finite) else None,
            "sample_std": float(finite.std(ddof=1)) if len(finite) >= 2 else None,
            "median": float(np.median(finite)) if len(finite) else None,
            "minimum": float(finite.min()) if len(finite) else None,
            "maximum": float(finite.max()) if len(finite) else None,
            "t_interval_95_percent": t_interval_95(differences) if family_is_conclusive else None,
            "positive_seed_count": int(np.sum(finite > 0.0)),
            "negative_seed_count": int(np.sum(finite < 0.0)),
            "zero_seed_count": int(np.sum(finite == 0.0)),
            "exact_paired_wilcoxon": signed_rank,
            "holm_family": "Student A and Student B primary XAI tests",
            "holm_family_size": 2,
            "holm_adjusted_p": None,
            "reject_holm_alpha_0_05": None,
        }
    if family_is_conclusive:
        adjusted = holm_adjust(raw_p)
        for student_key in STUDENT_KEYS:
            tests[student_key]["holm_adjusted_p"] = adjusted[student_key]
            tests[student_key]["reject_holm_alpha_0_05"] = adjusted[student_key] <= 0.05
    result = {
        "protocol_id": PROTOCOL_ID,
        "status": "complete" if family_is_conclusive else "complete_primary_inconclusive",
        "seeds": EXPECTED_SEEDS,
        "seed_count": len(EXPECTED_SEEDS),
        "primary_family_status": "conclusive" if family_is_conclusive else "inconclusive",
        "primary_tests": tests,
        "seed_42_randomization_sanity": completions[0]["metrics"]["randomization_sanity"],
        "secondary_results": {
            student_key: [
                completion["metrics"]["students"][student_key]["secondary"]
                for completion in completions
            ]
            for student_key in STUDENT_KEYS
        },
        "claim_boundary": (
            "This fixed-split analysis tests whether RF-KD has stronger local T=4 "
            "RF-predicted-class permutation-SHAP alignment with the corresponding RF than "
            "matched scratch across paired training-run/model seeds. It does not prove causal mechanism "
            "transfer, feature causality, attribution identity, explanation transfer "
            "independent of response-distribution closeness, off-manifold boundary "
            "equivalence, hardware explanation preservation, or cross-split generalization."
            " The RF subject is a deterministic reconstruction with exact float32 train "
            "replay and bounded test-CSV replay (maximum absolute tolerance 5e-8); no "
            "original serialized RF was retained."
        ),
    }
    return result, frame


def markdown_summary(result: dict[str, Any]) -> str:
    lines = [
        "# Controlled FG-DS XAI Transfer Audit",
        "",
        "This audit compares each final ten-seed RF-KD student with its matched scratch control.",
        "",
        f"Primary family status: **{result['primary_family_status']}**.",
        "",
        "| Student | Status | Defined seeds | Mean gain | Sample SD | 95% t interval | Exact p | Holm p | Positive seeds |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for student_key in STUDENT_KEYS:
        row = result["primary_tests"][student_key]
        interval = row["t_interval_95_percent"]
        mean = "NA" if row["mean"] is None else f"{row['mean']:.6f}"
        sample_std = "NA" if row["sample_std"] is None else f"{row['sample_std']:.6f}"
        interval_text = "NA" if interval is None else f"[{interval[0]:.6f}, {interval[1]:.6f}]"
        exact_p = (
            "NA"
            if row["exact_paired_wilcoxon"] is None
            else f"{row['exact_paired_wilcoxon']['p_value_two_sided']:.6f}"
        )
        holm_p = "NA" if row["holm_adjusted_p"] is None else f"{row['holm_adjusted_p']:.6f}"
        lines.append(
            f"| {student_key.replace('_', ' ').title()} | {row['status']} | "
            f"{row['defined_seed_count']}/10 | {mean} | {sample_std} | {interval_text} | "
            f"{exact_p} | {holm_p} | {row['positive_seed_count']}/{row['defined_seed_count']} |"
        )
        if row["inconclusive_reasons"]:
            lines.extend(["", f"{student_key}: " + " ".join(row["inconclusive_reasons"])])
    lines += ["", "## Claim Boundary", "", result["claim_boundary"], ""]
    return "\n".join(lines)


def build_contract(dataset_path: Path, source_root: Path, sampling: dict[str, np.ndarray]) -> dict[str, Any]:
    source_manifest = verify_external_manifest(source_root, SOURCE_PROTOCOL_ID)
    source_execution = read_json(source_root / "execution_contract.json")
    recorded_runner_hash = source_execution.get("script_sha256")
    require(isinstance(recorded_runner_hash, str) and len(recorded_runner_hash) == 64, "Recorded runner hash is invalid")
    matching_archived_runner_snapshots = sorted(
        item["path"]
        for item in source_manifest["files"]
        if item.get("sha256") == recorded_runner_hash
        and item.get("path") != "execution_contract.json"
    )
    value: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "analysis_status": "post_hoc_secondary_evidence",
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "dataset": {"path": repo_path(dataset_path), "sha256": EXPECTED_DATASET_SHA256},
        "source_run": {
            "path": repo_path(source_root),
            "root_manifest_sha256": sha256_file(source_root / "artifact_manifest.json"),
            "root_manifest_file_count": source_manifest["file_count_excluding_manifest"],
            "execution_contract_sha256": sha256_file(source_root / "execution_contract.json"),
            "recorded_training_runner_sha256": recorded_runner_hash,
            "executed_runner_snapshot_available": bool(matching_archived_runner_snapshots),
            "matching_snapshot_paths": matching_archived_runner_snapshots,
            "provenance_boundary": (
                "The historical training runner is not executed by this XAI analysis. Its "
                "recorded hash and archived snapshot availability are immutable source-run "
                "provenance. The current live trainer is deliberately excluded from this "
                "execution contract and resume identity."
            ),
        },
        "split_indices_sha256": EXPECTED_SPLIT_SHA256,
        "scaler_sha256": EXPECTED_SCALER_SHA256,
        "seeds": EXPECTED_SEEDS,
        "subjects_per_seed": SUBJECT_KEYS,
        "seed_42_randomization_controls": {
            "student_A": {
                "subject_key": "control_student_A_fully_reinitialized",
                "architecture": list(STUDENT_SPECS["student_A"]),
                "randomization_seed": RANDOMIZATION_SEEDS["student_A"],
            },
            "student_B": {
                "subject_key": "control_student_B_fully_reinitialized",
                "architecture": list(STUDENT_SPECS["student_B"]),
                "randomization_seed": RANDOMIZATION_SEEDS["student_B"],
            },
        },
        "sampling": {
            "cohort": "50 globally label-pure unique exact-feature groups per test class, lowest source-row representative, class interleaved",
            "background": "4 globally label-pure unique exact-feature groups per training class, lowest source-row representative, class interleaved",
            "global_purity_verification": (
                "Purity is computed over all dataset rows before partition sampling. The "
                "feature-group-disjoint split is also verified, so every selected partition "
                "group maps to exactly one globally pure group and label."
            ),
            "cohort_seed": COHORT_SEED,
            "background_seed": BACKGROUND_SEED,
            "arrays": sampling_array_contract(sampling),
        },
        "explainer": {
            "name": "model-agnostic permutation SHAP with independent masker",
            "output": "five-class probabilities softened at T=4",
            "selected_local_output": "RF teacher argmax class at T=4 for every subject",
            "permutation_seed": PERMUTATION_SEED,
            "permutation_repeats": PERMUTATION_REPEATS,
            "max_evals_per_record": (2 * 17 + 1) * PERMUTATION_REPEATS,
            "local_accuracy_atol": LOCAL_ACCURACY_ATOL,
            "masker_boundary": "Independent feature replacement may create combinations outside the observed joint distribution",
            "teacher_identity_boundary": (
                "Each RF is reconstructed from the recorded RF configuration through the "
                "fitting recipe implemented in this sealed XAI script. The hash-matched common "
                "module supplies the recorded RF configuration, split, scaler, and model "
                "definitions used by the reconstruction. The RF must exactly reproduce "
                "preserved train probabilities and bounded test CSV probabilities before "
                "SHAP. The historical training runner is provenance-only because it is not "
                "executed here. No original serialized RF was retained."
            ),
        },
        "primary_metric": {
            "name": "macro-class mean local signed-attribution cosine alignment gain",
            "orientation": "RF-KD minus scratch",
            "attribution_norm_epsilon": ATTRIBUTION_NORM_EPSILON,
            "undefined_handling": "exclude only rows where RF, scratch, or RF-KD norm is below threshold; never replace undefined cosine with zero",
            "minimum_eligible_rows_per_true_class": MIN_ELIGIBLE_PER_CLASS,
            "statistical_unit": (
                "paired training-run/model seed on one fixed feature-group-disjoint split; "
                "the seed jointly controls RF construction and student training"
            ),
            "test": "exact two-sided signed-rank enumeration; average ranks; Wilcox zero removal",
            "multiplicity": "Holm across Student A and Student B",
        },
        "secondary_metrics": [
            "separate scratch/RF-KD local cosine alignment",
            "global absolute-importance Spearman rank",
            "top-five overlap",
            "thresholded sign agreement",
            "local-accuracy residual",
            "class-conditional summaries",
            "hard-label agreement sensitivity",
        ],
        "software": {
            "executed_source_sha256": sha256_file(SCRIPT_PATH),
            "common_source_sha256": sha256_file(COMMON_SOURCE),
            "source_snapshots": {
                name: sha256_file(source) for name, source in SOURCE_SNAPSHOTS.items()
            },
        },
        "environment": environment_record(),
        "retention": "compressed per-seed SHAP arrays and compact summaries; reconstructed RF objects are not serialized",
    }
    value["execution_contract_id"] = canonical_sha256(value)
    return value


def verify_source_snapshots(output_dir: Path, contract: dict[str, Any]) -> None:
    snapshots = contract["software"]["source_snapshots"]
    require(set(snapshots) == set(SOURCE_SNAPSHOTS), "Source snapshot inventory differs")
    for name, expected_hash in snapshots.items():
        require(sha256_file(output_dir / name) == expected_hash, f"Source snapshot differs: {name}")
        require(sha256_file(SOURCE_SNAPSHOTS[name]) == expected_hash, f"Live source differs: {name}")


def prepare_output_root(
    output_dir: Path,
    contract: dict[str, Any],
    sampling: dict[str, np.ndarray],
    resume: bool,
) -> str:
    if output_dir.exists() and any(output_dir.iterdir()):
        require(resume, f"Refusing to overwrite existing output: {output_dir}")
        existing = read_json(output_dir / "execution_contract.json")
        require(existing == contract, "Resume execution contract differs")
        verify_source_snapshots(output_dir, contract)
        verify_sampling(output_dir / "sampling_contract.npz", sampling)
        manifest = read_json(output_dir / "artifact_manifest.json")
        require(manifest.get("protocol_id") == PROTOCOL_ID, "Resume manifest protocol differs")
        require(manifest.get("status") in {"running", "complete"}, "Resume manifest status differs")
        allowed_root_files = {
            "execution_contract.json",
            "sampling_contract.npz",
            "artifact_manifest.json",
            *SOURCE_SNAPSHOTS.keys(),
        }
        if manifest.get("status") == "complete":
            verify_existing(output_dir)
            return "complete"
        actual_root_files = {path.name for path in output_dir.iterdir() if path.is_file()}
        require(actual_root_files == allowed_root_files, "Running root file inventory differs")
        actual_dirs = {path.name for path in output_dir.iterdir() if path.is_dir()}
        require(actual_dirs <= {f"seed_{seed}" for seed in EXPECTED_SEEDS}, "Unexpected output directory")
        atomic_write_json(output_dir / "artifact_manifest.json", build_inventory(output_dir, "running"))
        return "resume"
    require(not resume, "Resume requires a non-empty output directory")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.initializing_", dir=output_dir.parent))
    try:
        atomic_write_json(staging / "execution_contract.json", contract)
        atomic_save_npz(staging / "sampling_contract.npz", **sampling)
        for name, source in SOURCE_SNAPSHOTS.items():
            shutil.copy2(source, staging / name)
        atomic_write_json(staging / "artifact_manifest.json", build_inventory(staging, "running"))
        verify_inventory(staging, {"running"})
        verify_source_snapshots(staging, contract)
        verify_sampling(staging / "sampling_contract.npz", sampling)
        if output_dir.exists():
            output_dir.rmdir()
        os.replace(staging, output_dir)
        return "fresh"
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def semantic_verification_record(
    contract: dict[str, Any],
    status: str,
    error: BaseException | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "execution_contract_id": contract["execution_contract_id"],
        "verified_seed_count": len(EXPECTED_SEEDS) if status == "passed" else 0,
        "checks": [
            "exact source-run manifests and lineage hashes",
            "deterministic pure-group cohort and background reconstruction",
            "exact train-only split and scaler reconstruction",
            "exact RF train-probability and bounded test-probability replay",
            "scratch and RF-KD checkpoint identity and prediction replay",
            "architecture-specific randomization-control identity",
            "per-seed SHAP schema, finiteness, local accuracy, and model-output replay",
            "RF-selected output attribution and three-way near-zero eligibility",
            "per-class and seed-level metric recomputation",
            "exact signed-rank/Holm recomputation or predeclared inconclusive withholding",
            "CSV, Markdown, summary, and exact inventory recomputation",
        ],
    }
    if error is not None:
        record["failure_type"] = type(error).__name__
        record["failure_message"] = str(error)
    return record


def verify_result_payload(
    output_dir: Path,
    contract: dict[str, Any],
    context: dict[str, Any],
    sampling: dict[str, np.ndarray],
) -> dict[str, Any]:
    verify_source_snapshots(output_dir, contract)
    source_root = (REPO_ROOT / contract["source_run"]["path"]).resolve()
    source_manifest = verify_external_manifest(source_root, SOURCE_PROTOCOL_ID)
    require(
        sha256_file(source_root / "artifact_manifest.json")
        == contract["source_run"]["root_manifest_sha256"],
        "Source root manifest hash differs during final verification",
    )
    require(
        source_manifest["file_count_excluding_manifest"]
        == contract["source_run"]["root_manifest_file_count"],
        "Source root manifest count differs during final verification",
    )
    completions = []
    for seed in EXPECTED_SEEDS:
        verification = verify_seed_payload(
            output_dir / f"seed_{seed}", contract, context, sampling
        )
        require(verification["seed"] == seed, "Verified seed order differs")
        completions.append(verification["completion"])
    recomputed_result, recomputed_frame = aggregate_results(completions)
    persisted_result = read_json(output_dir / "controlled_xai_transfer_summary.json")
    require(persisted_result == recomputed_result, "Persisted XAI summary differs")
    persisted_frame = pd.read_csv(output_dir / "controlled_xai_transfer_seed_table.csv")
    require(list(persisted_frame.columns) == list(recomputed_frame.columns), "Seed table columns differ")
    require(persisted_frame.shape == recomputed_frame.shape, "Seed table shape differs")
    for column in recomputed_frame.columns:
        if np.issubdtype(recomputed_frame[column].dtype, np.number):
            require(
                np.allclose(
                    persisted_frame[column].to_numpy(),
                    recomputed_frame[column].to_numpy(),
                    rtol=0.0,
                    atol=1e-15,
                    equal_nan=True,
                ),
                f"Seed table numeric column differs: {column}",
            )
        else:
            require(
                persisted_frame[column].astype(str).tolist()
                == recomputed_frame[column].astype(str).tolist(),
                f"Seed table text column differs: {column}",
            )
    require(
        (output_dir / "CONTROLLED_XAI_TRANSFER_SUMMARY.md").read_text(encoding="utf-8")
        == markdown_summary(recomputed_result),
        "Persisted Markdown summary differs",
    )
    return {
        "result": recomputed_result,
        "seed_count": len(completions),
    }


def verify_existing(output_dir: Path) -> dict[str, Any]:
    manifest = verify_inventory(output_dir, {"complete"})
    contract = read_json(output_dir / "execution_contract.json")
    without_id = {key: value for key, value in contract.items() if key != "execution_contract_id"}
    require(contract.get("execution_contract_id") == canonical_sha256(without_id), "Execution contract ID differs")
    verify_source_snapshots(output_dir, contract)
    dataset_path = (REPO_ROOT / contract["dataset"]["path"]).resolve()
    source_root = (REPO_ROOT / contract["source_run"]["path"]).resolve()
    require(
        sha256_file(source_root / "artifact_manifest.json")
        == contract["source_run"]["root_manifest_sha256"],
        "Source root manifest hash differs",
    )
    context = reconstruct_context(dataset_path, source_root)
    expected_sampling = build_sampling(context)
    verify_sampling(output_dir / "sampling_contract.npz", expected_sampling)
    require(
        contract["sampling"]["arrays"] == sampling_array_contract(expected_sampling),
        "Sampling contract differs",
    )
    semantic = read_json(output_dir / "semantic_verification.json")
    require(
        semantic == semantic_verification_record(contract, "passed"),
        "Semantic verification record differs",
    )
    verify_result_payload(output_dir, contract, context, expected_sampling)
    return {
        "status": "verified",
        "protocol_id": PROTOCOL_ID,
        "verified_files": manifest["file_count_excluding_manifest"],
        "verified_seeds": len(EXPECTED_SEEDS),
        "execution_contract_id": contract["execution_contract_id"],
    }


def run(
    dataset_path: Path,
    source_root: Path,
    output_dir: Path,
    resume: bool,
) -> None:
    context = reconstruct_context(dataset_path, source_root)
    sampling = build_sampling(context)
    contract = build_contract(dataset_path, source_root, sampling)
    state = prepare_output_root(output_dir, contract, sampling, resume)
    if state == "complete":
        return
    completions = []
    for seed in EXPECTED_SEEDS:
        completion = run_seed_job(
            output_dir, contract, context, sampling, seed, resume
        )
        completions.append(completion)
        atomic_write_json(output_dir / "artifact_manifest.json", build_inventory(output_dir, "running"))
        print(
            json.dumps(
                {
                    "status": "seed_complete",
                    "seed": seed,
                    "completed_seeds": len(completions),
                    "total_seeds": len(EXPECTED_SEEDS),
                }
            ),
            flush=True,
        )
    result, frame = aggregate_results(completions)
    atomic_write_json(output_dir / "controlled_xai_transfer_summary.json", result)
    atomic_write_csv(output_dir / "controlled_xai_transfer_seed_table.csv", frame)
    atomic_write_text(
        output_dir / "CONTROLLED_XAI_TRANSFER_SUMMARY.md", markdown_summary(result)
    )
    try:
        verify_result_payload(output_dir, contract, context, sampling)
    except BaseException as error:
        atomic_write_json(
            output_dir / "semantic_verification.json",
            semantic_verification_record(contract, "failed", error),
        )
        atomic_write_json(output_dir / "artifact_manifest.json", build_inventory(output_dir, "failed"))
        raise
    atomic_write_json(
        output_dir / "semantic_verification.json",
        semantic_verification_record(contract, "passed"),
    )
    atomic_write_json(output_dir / "artifact_manifest.json", build_inventory(output_dir, "complete"))
    verify_inventory(output_dir, {"complete"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--confirm-explanations", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.verify_existing:
        print(json.dumps(verify_existing(output_dir), indent=2))
        return 0
    dataset_path = args.dataset.resolve()
    source_root = args.source_root.resolve()
    context = reconstruct_context(dataset_path, source_root)
    sampling = build_sampling(context)
    contract = build_contract(dataset_path, source_root, sampling)
    preflight = {
        "status": "passed",
        "protocol_id": PROTOCOL_ID,
        "execution_contract_id": contract["execution_contract_id"],
        "cohort_rows": COHORT_SIZE,
        "background_rows": BACKGROUND_SIZE,
        "global_exact_feature_groups": int(sampling["global_exact_feature_group_count"][0]),
        "global_mixed_label_groups": int(sampling["global_mixed_label_group_count"][0]),
        "seeds": EXPECTED_SEEDS,
        "recorded_training_runner_sha256": contract["source_run"]["recorded_training_runner_sha256"],
        "executed_runner_snapshot_available": contract["source_run"]["executed_runner_snapshot_available"],
        "executing_source_snapshots": contract["software"]["source_snapshots"],
        "shap_started": False,
    }
    print(json.dumps(preflight, indent=2), flush=True)
    if not args.confirm_explanations:
        print("SHAP was not started. Pass --confirm-explanations to execute the audit.")
        return 0
    run(dataset_path, source_root, output_dir, args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
