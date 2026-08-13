"""Run a compact multi-split robustness check for the core WSN-DS comparison.

The experiment uses ten seeded exact-feature-group-disjoint holdout assignments.
Within every assignment, two paired optimizer seeds train matched scratch and
RF-KD versions of Student A and Student B. The calibrated RF is fitted once per
assignment and supplies the same soft targets to both optimizer seeds.

Only compact metrics, predicted labels, split assignments, and hashes are
persisted. RF objects, training probabilities, student checkpoints, and copied
datasets are intentionally omitted because this experiment measures split-level
robustness and is not a deployment lineage.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.wsnds.leakage_free_rerun.run_feature_group_10seed_confirmation import (  # noqa: E402
    fit_calibrated_rf,
    state_dict_sha256,
)
from experiments.wsnds.leakage_free_rerun.tier15_common import (  # noqa: E402
    CLASS_NAMES,
    KD_ALPHA,
    KD_T,
    RF_CONFIG,
    STUDENT_SPECS,
    TRAIN_CONFIG,
    StudentMLP,
    _build_split,
    apply_train_scaler,
    batched_probs,
    class_weights,
    classification_metrics,
    feature_overlap_audit,
    load_wsnds,
    set_seed,
    sha256_arrays,
    sha256_file,
    shuffled_batches,
    verified_feature_hashes,
)


PROTOCOL_ID = "wsnds_fgds_multisplit_core_10x2_v2"
EXPECTED_DATASET_SHA256 = (
    "c65d05b983a85753bd62b6f76c5739fc52fe0c14cbb7644255cee4742f5ff7c9"
)
SPLIT_SEEDS = [42, 123, 456, 789, 1001, 2024, 3141, 5678, 8192, 9999]
OPTIMIZER_SEEDS = [42, 123]
RF_SEED = 42
EXPECTED_GROUP_COUNT = 361_156
EXPECTED_MIXED_LABEL_GROUPS = 3
EXPECTED_MIXED_LABEL_ROWS = 6
EXPECTED_TOTAL_ROWS = 374_661
DEFAULT_DATASET = REPO_ROOT / "data/wsnds/WSN-DS.csv"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results/wsnds/evidence_completion_20260812"
    / "fgds_multisplit_core_10x2_v2"
)
COMMON_SOURCE = REPO_ROOT / "experiments/wsnds/leakage_free_rerun/tier15_common.py"
RF_SOURCE = (
    REPO_ROOT
    / "experiments/wsnds/leakage_free_rerun/run_feature_group_10seed_confirmation.py"
)
SOURCE_SNAPSHOTS = {
    "executed_multisplit_source.py": SCRIPT_PATH,
    "bound_common_source.py": COMMON_SOURCE,
    "bound_rf_source.py": RF_SOURCE,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def repo_path(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "ascii"
        )
    )


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


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Required JSON is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


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


def verify_inventory(root: Path, accepted_status: set[str]) -> dict[str, Any]:
    manifest_path = root / "artifact_manifest.json"
    manifest = read_json(manifest_path)
    require(manifest.get("protocol_id") == PROTOCOL_ID, "Manifest protocol mismatch")
    require(manifest.get("status") in accepted_status, "Manifest status mismatch")
    files = manifest.get("files")
    require(isinstance(files, list) and files, "Manifest inventory is empty")
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
    require(actual == seen, f"Manifest does not exactly cover the files under {root}")
    return manifest


def environment_record(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "deterministic_algorithms_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def configure_determinism() -> None:
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def prepare_group_table(features: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    hashes = verified_feature_hashes(features)
    frame = pd.DataFrame({"group": hashes, "label": labels})
    counts = pd.crosstab(frame["group"], frame["label"])
    counts = counts.reindex(columns=range(len(CLASS_NAMES)), fill_value=0)
    group_ids = counts.index.to_numpy(dtype=np.uint64)
    group_labels = counts.to_numpy().argmax(axis=1).astype(np.int64)
    mixed_mask = counts.gt(0).sum(axis=1).to_numpy() > 1
    mixed_groups = group_ids[mixed_mask]
    mixed_rows = int(np.isin(hashes, mixed_groups).sum())
    require(len(group_ids) == EXPECTED_GROUP_COUNT, "Feature-group count mismatch")
    require(int(mixed_mask.sum()) == EXPECTED_MIXED_LABEL_GROUPS, "Mixed-label group count mismatch")
    require(mixed_rows == EXPECTED_MIXED_LABEL_ROWS, "Mixed-label row count mismatch")
    return {
        "row_group_ids": hashes,
        "group_ids": group_ids,
        "group_labels": group_labels,
        "mixed_group_ids": mixed_groups,
    }


def make_split(
    features: np.ndarray,
    labels: np.ndarray,
    group_table: dict[str, Any],
    split_seed: int,
) -> dict[str, Any]:
    group_ids = group_table["group_ids"]
    group_labels = group_table["group_labels"]
    row_groups = group_table["row_group_ids"]
    trainval_groups, test_groups = train_test_split(
        group_ids,
        test_size=0.15,
        random_state=split_seed,
        stratify=group_labels,
    )
    trainval_labels = pd.Series(group_labels, index=group_ids).loc[trainval_groups].to_numpy()
    train_groups, validation_groups = train_test_split(
        trainval_groups,
        test_size=0.1765,
        random_state=split_seed,
        stratify=trainval_labels,
    )
    train_mask = np.isin(row_groups, train_groups)
    validation_mask = np.isin(row_groups, validation_groups)
    test_mask = np.isin(row_groups, test_groups)
    assignments = train_mask.astype(np.int8) + validation_mask + test_mask
    require(np.all(assignments == 1), "Split did not assign every row exactly once")
    split = _build_split(
        features,
        labels,
        np.flatnonzero(train_mask),
        np.flatnonzero(validation_mask),
        np.flatnonzero(test_mask),
    )
    for partition in ("train", "validation", "test"):
        counts = np.bincount(split[f"y_{partition}"], minlength=len(CLASS_NAMES))
        require(
            counts.shape == (len(CLASS_NAMES),) and np.all(counts > 0),
            f"The {partition} partition does not contain all classes",
        )
    audit = feature_overlap_audit(features, split)
    require(all(value == 0 for value in audit.values()), "Exact feature groups cross partitions")
    mixed_groups = group_table["mixed_group_ids"]
    mixed_assignment = {}
    for partition in ("train", "validation", "test"):
        partition_groups = row_groups[split[f"{partition}_indices"]]
        mixed_assignment[partition] = {
            "groups": int(len(set(map(int, partition_groups)) & set(map(int, mixed_groups)))),
            "rows": int(np.isin(partition_groups, mixed_groups).sum()),
        }
    require(
        sum(row["groups"] for row in mixed_assignment.values()) == EXPECTED_MIXED_LABEL_GROUPS,
        "A mixed-label group appears in more than one partition",
    )
    require(
        sum(row["rows"] for row in mixed_assignment.values()) == EXPECTED_MIXED_LABEL_ROWS,
        "Mixed-label rows were lost",
    )
    split["policy"] = "stratified_exact_feature_group_split"
    split["split_seed"] = split_seed
    split["group_audit"] = {
        **audit,
        "num_feature_groups": len(group_ids),
        "conflicting_label_feature_groups": len(mixed_groups),
        "mixed_label_group_assignment": mixed_assignment,
        "group_stratification_label": "majority label; smallest class index breaks ties",
    }
    return split


def split_index_hash(split: dict[str, Any]) -> str:
    return sha256_arrays(
        split["train_indices"], split["validation_indices"], split["test_indices"]
    )


def scaler_hash(scaler: Any) -> str:
    return sha256_arrays(
        np.asarray(scaler.mean_, dtype=np.float64),
        np.asarray(scaler.scale_, dtype=np.float64),
        np.asarray(scaler.var_, dtype=np.float64),
    )


def validate_probability_matrix(
    probabilities: np.ndarray,
    expected_rows: int,
    context: str,
) -> np.ndarray:
    values = np.asarray(probabilities)
    require(
        values.shape == (expected_rows, len(CLASS_NAMES)),
        f"{context} probability shape mismatch: {values.shape}",
    )
    require(np.issubdtype(values.dtype, np.floating), f"{context} is not floating point")
    require(np.isfinite(values).all(), f"{context} contains NaN or infinity")
    require(np.all(values >= 0.0), f"{context} contains negative probabilities")
    require(np.all(values <= 1.0), f"{context} contains probabilities above one")
    require(
        np.allclose(values.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6),
        f"{context} probability rows do not sum to one",
    )
    return values


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "per_class_f1": metrics["per_class_f1"],
        "per_class_support": metrics["per_class_support"],
        "confusion_matrix": metrics["confusion_matrix"],
    }


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_standard_checked(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_validation: torch.Tensor,
    y_validation: torch.Tensor,
    weights: torch.Tensor,
    device: torch.device,
    context: str,
) -> nn.Module:
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG["lr"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TRAIN_CONFIG["epochs"]
    )
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    X_train_device = X_train.to(device)
    y_train_device = y_train.to(device)
    X_validation_device = X_validation.to(device)
    y_validation_numpy = y_validation.numpy()
    best_f1 = -1.0
    best_state = None
    stale = 0
    for epoch in range(TRAIN_CONFIG["epochs"]):
        model.train()
        for X_batch, y_batch in shuffled_batches(
            X_train_device, y_train_device, batch_size=TRAIN_CONFIG["batch_size"]
        ):
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()
        validation_probabilities = validate_probability_matrix(
            batched_probs(model, X_validation_device, device),
            len(y_validation_numpy),
            f"{context} validation epoch {epoch}",
        )
        score = f1_score(
            y_validation_numpy,
            validation_probabilities.argmax(axis=1),
            labels=np.arange(len(CLASS_NAMES)),
            average="macro",
            zero_division=0,
        )
        require(np.isfinite(score), f"{context} validation macro-F1 is non-finite")
        if score > best_f1:
            best_f1 = float(score)
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= TRAIN_CONFIG["patience"]:
                break
    require(best_state is not None, f"{context} produced no validation checkpoint")
    model.load_state_dict(best_state)
    return model


def train_rf_kd_checked(
    model: nn.Module,
    teacher_probabilities: np.ndarray,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_validation: torch.Tensor,
    y_validation: torch.Tensor,
    weights: torch.Tensor,
    device: torch.device,
    context: str,
) -> nn.Module:
    teacher_values = validate_probability_matrix(
        teacher_probabilities, len(X_train), f"{context} teacher targets"
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG["lr"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TRAIN_CONFIG["epochs"]
    )
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    X_train_device = X_train.to(device)
    y_train_device = y_train.to(device)
    X_validation_device = X_validation.to(device)
    y_validation_numpy = y_validation.numpy()
    raw_teacher = torch.tensor(teacher_values, dtype=torch.float32, device=device)
    teacher_targets = F.softmax(
        torch.log(raw_teacher.clamp(min=1e-8)) / KD_T, dim=1
    ).detach()
    best_f1 = -1.0
    best_state = None
    stale = 0
    for epoch in range(TRAIN_CONFIG["epochs"]):
        model.train()
        for X_batch, y_batch, teacher_batch in shuffled_batches(
            X_train_device,
            y_train_device,
            teacher_targets,
            batch_size=TRAIN_CONFIG["batch_size"],
        ):
            optimizer.zero_grad()
            logits = model(X_batch)
            kd_loss = F.kl_div(
                F.log_softmax(logits / KD_T, dim=1),
                teacher_batch,
                reduction="batchmean",
            ) * (KD_T * KD_T)
            ce_loss = criterion(logits, y_batch)
            loss = KD_ALPHA * kd_loss + (1.0 - KD_ALPHA) * ce_loss
            loss.backward()
            optimizer.step()
        scheduler.step()
        validation_probabilities = validate_probability_matrix(
            batched_probs(model, X_validation_device, device),
            len(y_validation_numpy),
            f"{context} validation epoch {epoch}",
        )
        score = f1_score(
            y_validation_numpy,
            validation_probabilities.argmax(axis=1),
            labels=np.arange(len(CLASS_NAMES)),
            average="macro",
            zero_division=0,
        )
        require(np.isfinite(score), f"{context} validation macro-F1 is non-finite")
        if score > best_f1:
            best_f1 = float(score)
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= TRAIN_CONFIG["patience"]:
                break
    require(best_state is not None, f"{context} produced no validation checkpoint")
    model.load_state_dict(best_state)
    return model


def train_pair(
    student_name: str,
    hidden_dims: tuple[int, int],
    optimizer_seed: int,
    X_train_t: torch.Tensor,
    y_train_t: torch.Tensor,
    X_validation_t: torch.Tensor,
    y_validation_t: torch.Tensor,
    X_test_t: torch.Tensor,
    y_test: np.ndarray,
    weights: torch.Tensor,
    teacher_train_probabilities: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    set_seed(optimizer_seed)
    scratch = StudentMLP(17, hidden_dims, len(CLASS_NAMES))
    scratch_initial = state_dict_sha256(cpu_state_dict(scratch))
    scratch = train_standard_checked(
        scratch,
        X_train_t,
        y_train_t,
        X_validation_t,
        y_validation_t,
        weights,
        device,
        f"{student_name} scratch seed {optimizer_seed}",
    )
    scratch_probabilities = validate_probability_matrix(
        batched_probs(scratch, X_test_t, device),
        len(y_test),
        f"{student_name} scratch seed {optimizer_seed}",
    )
    scratch_metrics = classification_metrics(y_test, scratch_probabilities)
    scratch_state = state_dict_sha256(cpu_state_dict(scratch))
    del scratch

    set_seed(optimizer_seed)
    rf_kd = StudentMLP(17, hidden_dims, len(CLASS_NAMES))
    rf_kd_initial = state_dict_sha256(cpu_state_dict(rf_kd))
    require(scratch_initial == rf_kd_initial, "Scratch and RF-KD initial states differ")
    rf_kd = train_rf_kd_checked(
        rf_kd,
        teacher_train_probabilities,
        X_train_t,
        y_train_t,
        X_validation_t,
        y_validation_t,
        weights,
        device,
        f"{student_name} RF-KD seed {optimizer_seed}",
    )
    rf_kd_probabilities = validate_probability_matrix(
        batched_probs(rf_kd, X_test_t, device),
        len(y_test),
        f"{student_name} RF-KD seed {optimizer_seed}",
    )
    rf_kd_metrics = classification_metrics(y_test, rf_kd_probabilities)
    rf_kd_state = state_dict_sha256(cpu_state_dict(rf_kd))
    del rf_kd

    result = {
        "student": student_name,
        "hidden_dims": list(hidden_dims),
        "optimizer_seed": optimizer_seed,
        "initial_state_sha256": scratch_initial,
        "scratch_trained_state_sha256": scratch_state,
        "rf_kd_trained_state_sha256": rf_kd_state,
        "scratch": compact_metrics(scratch_metrics),
        "rf_kd": compact_metrics(rf_kd_metrics),
        "rf_kd_minus_scratch_macro_f1": float(
            rf_kd_metrics["macro_f1"] - scratch_metrics["macro_f1"]
        ),
    }
    return (
        result,
        scratch_probabilities.argmax(axis=1).astype(np.uint8),
        rf_kd_probabilities.argmax(axis=1).astype(np.uint8),
    )


def completion_expected(contract: dict[str, Any], split_seed: int) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "execution_contract_id": contract["execution_contract_id"],
        "split_seed": split_seed,
        "status": "complete",
    }


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def metrics_from_labels(true_labels: np.ndarray, predicted_labels: np.ndarray) -> dict[str, Any]:
    true_values = np.asarray(true_labels, dtype=np.int64)
    predicted_values = np.asarray(predicted_labels, dtype=np.int64)
    require(true_values.shape == predicted_values.shape, "Prediction-label shape mismatch")
    require(true_values.ndim == 1 and len(true_values) > 0, "Prediction labels are empty")
    require(
        np.all((true_values >= 0) & (true_values < len(CLASS_NAMES))),
        "True labels are outside the class contract",
    )
    require(
        np.all((predicted_values >= 0) & (predicted_values < len(CLASS_NAMES))),
        "Predicted labels are outside the class contract",
    )
    hard_probabilities = np.eye(len(CLASS_NAMES), dtype=np.float32)[predicted_values]
    return compact_metrics(classification_metrics(true_values, hard_probabilities))


def require_metrics_equal(expected: dict[str, Any], observed: dict[str, Any], context: str) -> None:
    require(set(expected) == set(observed), f"{context} metric keys differ")
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        require(
            abs(float(expected[key]) - float(observed[key])) <= 1e-15,
            f"{context} differs for {key}",
        )
    require(
        np.allclose(
            np.asarray(expected["per_class_f1"], dtype=np.float64),
            np.asarray(observed["per_class_f1"], dtype=np.float64),
            rtol=0.0,
            atol=1e-15,
        ),
        f"{context} per-class F1 differs",
    )
    require(
        expected["per_class_support"] == observed["per_class_support"],
        f"{context} class support differs",
    )
    require(
        expected["confusion_matrix"] == observed["confusion_matrix"],
        f"{context} confusion matrix differs",
    )


def verify_split_semantics(
    split_root: Path,
    completion: dict[str, Any],
    contract: dict[str, Any],
    dataset: dict[str, Any],
    group_table: dict[str, Any],
) -> None:
    split_contract_path = split_root / "split_contract.json"
    prediction_path = split_root / "prediction_labels.npz"
    require(
        completion.get("split_contract_sha256") == sha256_file(split_contract_path),
        f"Split-contract hash differs: {split_root}",
    )
    require(
        completion.get("prediction_labels_sha256") == sha256_file(prediction_path),
        f"Prediction-label hash differs: {split_root}",
    )
    split_contract = read_json(split_contract_path)
    split_seed = int(completion["split_seed"])
    required_split_contract = {
        "protocol_id": PROTOCOL_ID,
        "execution_contract_id": contract["execution_contract_id"],
        "split_seed": split_seed,
        "rf_seed": RF_SEED,
        "split_policy": "stratified_exact_feature_group_split",
        "scaler_fit_partition": "train only",
    }
    for key, value in required_split_contract.items():
        require(
            split_contract.get(key) == value,
            f"Split contract differs for {key}: {split_root}",
        )

    with np.load(prediction_path, allow_pickle=False) as persisted:
        arrays = {name: persisted[name] for name in persisted.files}
    expected_array_names = {
        "train_indices",
        "validation_indices",
        "test_source_indices",
        "true_labels",
        "teacher_predictions",
    }
    for optimizer_seed in OPTIMIZER_SEEDS:
        for student_name in STUDENT_SPECS:
            suffix = f"{student_name}_seed_{optimizer_seed}"
            expected_array_names.update({f"{suffix}_scratch", f"{suffix}_rf_kd"})
    require(set(arrays) == expected_array_names, f"Prediction-array inventory differs: {split_root}")

    reconstructed = make_split(
        dataset["features"], dataset["labels"], group_table, split_seed
    )
    index_names = {
        "train": "train_indices",
        "validation": "validation_indices",
        "test": "test_source_indices",
    }
    for partition, saved_name in index_names.items():
        saved = arrays[saved_name]
        require(saved.dtype == np.int64, f"{saved_name} dtype differs: {split_root}")
        require(
            np.array_equal(saved, reconstructed[f"{partition}_indices"]),
            f"{partition} indices differ from deterministic reconstruction: {split_root}",
        )
    require(
        split_contract.get("split_indices_content_sha256") == split_index_hash(reconstructed),
        f"Split-index content hash differs: {split_root}",
    )
    expected_sizes = {
        partition: int(len(reconstructed[f"{partition}_indices"]))
        for partition in ("train", "validation", "test")
    }
    require(split_contract.get("split_sizes") == expected_sizes, f"Split sizes differ: {split_root}")
    expected_class_counts = {
        partition: np.bincount(
            reconstructed[f"y_{partition}"], minlength=len(CLASS_NAMES)
        ).tolist()
        for partition in ("train", "validation", "test")
    }
    require(
        split_contract.get("class_counts") == expected_class_counts,
        f"Split class counts differ: {split_root}",
    )
    require(
        split_contract.get("feature_group_audit") == reconstructed["group_audit"],
        f"Feature-group audit differs: {split_root}",
    )
    _, reconstructed_scaler = apply_train_scaler(reconstructed)
    require(
        split_contract.get("scaler_content_sha256") == scaler_hash(reconstructed_scaler),
        f"Train-only scaler hash differs: {split_root}",
    )
    calibration = split_contract.get("rf_calibration_audit")
    calibration_groups = verified_feature_hashes(reconstructed["X_train_raw"])
    calibration_folds = list(
        StratifiedGroupKFold(
            n_splits=RF_CONFIG["calibration_cv"],
            shuffle=True,
            random_state=RF_SEED,
        ).split(
            reconstructed["X_train_raw"],
            reconstructed["y_train"],
            calibration_groups,
        )
    )
    overlap_counts = []
    for train_indices, validation_indices in calibration_folds:
        overlap_counts.append(
            int(
                len(
                    set(map(int, calibration_groups[train_indices]))
                    & set(map(int, calibration_groups[validation_indices]))
                )
            )
        )
    expected_calibration = {
        "strategy": "stratified_group_kfold",
        "folds": RF_CONFIG["calibration_cv"],
        "group_overlap_per_fold": overlap_counts,
        "unique_groups": int(len(np.unique(calibration_groups))),
    }
    require(
        calibration == expected_calibration and overlap_counts == [0] * RF_CONFIG["calibration_cv"],
        f"RF calibration audit differs: {split_root}",
    )
    require(
        is_sha256(split_contract.get("teacher_train_probability_content_sha256")),
        f"Teacher probability content hash is invalid: {split_root}",
    )

    true_labels = arrays["true_labels"]
    require(true_labels.dtype == np.uint8, f"True-label dtype differs: {split_root}")
    require(
        np.array_equal(true_labels, reconstructed["y_test"].astype(np.uint8)),
        f"Persisted true labels differ: {split_root}",
    )
    test_rows = len(true_labels)
    for name in expected_array_names - {
        "train_indices",
        "validation_indices",
        "test_source_indices",
        "true_labels",
    }:
        require(arrays[name].dtype == np.uint8, f"{name} dtype differs: {split_root}")
        require(arrays[name].shape == (test_rows,), f"{name} shape differs: {split_root}")
        require(
            np.all(arrays[name] < len(CLASS_NAMES)),
            f"{name} contains an invalid class: {split_root}",
        )

    require_metrics_equal(
        completion["teacher_metrics"],
        metrics_from_labels(true_labels, arrays["teacher_predictions"]),
        f"Teacher metrics for split {split_seed}",
    )
    rows = completion.get("student_results")
    require(isinstance(rows, list), f"Student results are missing: {split_root}")
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (row.get("student"), row.get("optimizer_seed"))
        require(key not in indexed, f"Duplicate student result {key}: {split_root}")
        indexed[key] = row
    expected_keys = {
        (student_name, optimizer_seed)
        for optimizer_seed in OPTIMIZER_SEEDS
        for student_name in STUDENT_SPECS
    }
    require(set(indexed) == expected_keys, f"Student-result coverage differs: {split_root}")
    for (student_name, optimizer_seed), row in indexed.items():
        require(
            row.get("hidden_dims") == list(STUDENT_SPECS[student_name]),
            f"Hidden dimensions differ for {(student_name, optimizer_seed)}",
        )
        set_seed(optimizer_seed)
        initial_model = StudentMLP(17, STUDENT_SPECS[student_name], len(CLASS_NAMES))
        initial_hash = state_dict_sha256(cpu_state_dict(initial_model))
        del initial_model
        require(
            row.get("initial_state_sha256") == initial_hash,
            f"Initial-state hash differs for {(student_name, optimizer_seed)}",
        )
        require(
            is_sha256(row.get("scratch_trained_state_sha256"))
            and is_sha256(row.get("rf_kd_trained_state_sha256")),
            f"Trained-state hash is invalid for {(student_name, optimizer_seed)}",
        )
        suffix = f"{student_name}_seed_{optimizer_seed}"
        scratch_metrics = metrics_from_labels(true_labels, arrays[f"{suffix}_scratch"])
        rf_kd_metrics = metrics_from_labels(true_labels, arrays[f"{suffix}_rf_kd"])
        require_metrics_equal(
            row["scratch"], scratch_metrics, f"Scratch {(student_name, optimizer_seed)}"
        )
        require_metrics_equal(
            row["rf_kd"], rf_kd_metrics, f"RF-KD {(student_name, optimizer_seed)}"
        )
        expected_delta = rf_kd_metrics["macro_f1"] - scratch_metrics["macro_f1"]
        require(
            abs(float(row["rf_kd_minus_scratch_macro_f1"]) - expected_delta) <= 1e-15,
            f"Paired macro-F1 delta differs for {(student_name, optimizer_seed)}",
        )


def verify_split_completion(
    split_root: Path,
    expected: dict[str, Any],
    contract: dict[str, Any],
    dataset: dict[str, Any],
    group_table: dict[str, Any],
) -> dict[str, Any] | None:
    completion_path = split_root / "split_completion.json"
    manifest_path = split_root / "artifact_manifest.json"
    if not completion_path.is_file() or not manifest_path.is_file():
        return None
    completion = read_json(completion_path)
    for key, value in expected.items():
        require(completion.get(key) == value, f"Resume contract mismatch for {split_root}: {key}")
    verify_inventory(split_root, {"complete"})
    verify_split_semantics(split_root, completion, contract, dataset, group_table)
    return completion


def quarantine_split_attempt(
    root: Path,
    split_root: Path,
    split_seed: int,
    reason: BaseException | None,
) -> Path:
    failed_root = root / "failed_split_attempts"
    failed_root.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    destination = failed_root / f"split_{split_seed}_{timestamp}"
    os.replace(split_root, destination)
    failure = {
        "protocol_id": PROTOCOL_ID,
        "status": "quarantined",
        "split_seed": split_seed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reason_type": type(reason).__name__ if reason is not None else "IncompleteSplitAttempt",
        "reason": str(reason) if reason is not None else "Completion artifacts were absent",
    }
    atomic_write_json(destination / "quarantine_reason.json", failure)
    atomic_write_json(destination / "artifact_manifest.json", build_inventory(destination, "failed"))
    return destination


def run_split(
    root: Path,
    contract: dict[str, Any],
    dataset: dict[str, Any],
    group_table: dict[str, Any],
    split_seed: int,
    device: torch.device,
    resume: bool,
) -> dict[str, Any]:
    split_root = root / f"split_{split_seed}"
    expected = completion_expected(contract, split_seed)
    if split_root.exists() and any(split_root.iterdir()):
        if resume:
            try:
                completed = verify_split_completion(
                    split_root, expected, contract, dataset, group_table
                )
            except Exception as error:
                quarantine_split_attempt(root, split_root, split_seed, error)
                completed = None
            if completed is not None:
                return completed
            if split_root.exists():
                quarantine_split_attempt(root, split_root, split_seed, None)
        else:
            raise FileExistsError(f"Refusing to overwrite split output: {split_root}")
    elif split_root.exists():
        split_root.rmdir()
    split_root.mkdir(parents=True, exist_ok=False)
    started = time.time()
    split = make_split(dataset["features"], dataset["labels"], group_table, split_seed)
    scaled, scaler = apply_train_scaler(split)
    training_groups = verified_feature_hashes(split["X_train_raw"])
    teacher_started = time.time()
    teacher, calibration_audit = fit_calibrated_rf(
        scaled["X_train"], split["y_train"], RF_SEED, groups=training_groups
    )
    teacher_fit_seconds = time.time() - teacher_started
    teacher_train_probabilities = validate_probability_matrix(
        teacher.predict_proba(scaled["X_train"]).astype(np.float32, copy=False),
        len(split["y_train"]),
        f"RF training split {split_seed}",
    )
    teacher_test_probabilities = validate_probability_matrix(
        teacher.predict_proba(scaled["X_test"]).astype(np.float32, copy=False),
        len(split["y_test"]),
        f"RF test split {split_seed}",
    )
    teacher_train_probability_content_sha256 = sha256_arrays(
        teacher_train_probabilities
    )
    teacher_metrics = classification_metrics(split["y_test"], teacher_test_probabilities)
    teacher_prediction_labels = teacher_test_probabilities.argmax(axis=1).astype(np.uint8)
    del teacher

    X_train_t = torch.from_numpy(scaled["X_train"])
    y_train_t = torch.from_numpy(split["y_train"])
    X_validation_t = torch.from_numpy(scaled["X_validation"])
    y_validation_t = torch.from_numpy(split["y_validation"])
    X_test_t = torch.from_numpy(scaled["X_test"])
    weights = class_weights(split["y_train"])

    student_results: list[dict[str, Any]] = []
    prediction_arrays: dict[str, np.ndarray] = {
        "train_indices": split["train_indices"].astype(np.int64),
        "validation_indices": split["validation_indices"].astype(np.int64),
        "test_source_indices": split["test_indices"].astype(np.int64),
        "true_labels": split["y_test"].astype(np.uint8),
        "teacher_predictions": teacher_prediction_labels,
    }
    for optimizer_seed in OPTIMIZER_SEEDS:
        for student_name, hidden_dims in STUDENT_SPECS.items():
            result, scratch_predictions, rf_kd_predictions = train_pair(
                student_name,
                hidden_dims,
                optimizer_seed,
                X_train_t,
                y_train_t,
                X_validation_t,
                y_validation_t,
                X_test_t,
                split["y_test"],
                weights,
                teacher_train_probabilities,
                device,
            )
            student_results.append(result)
            suffix = f"{student_name}_seed_{optimizer_seed}"
            prediction_arrays[f"{suffix}_scratch"] = scratch_predictions
            prediction_arrays[f"{suffix}_rf_kd"] = rf_kd_predictions
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del teacher_train_probabilities

    class_counts = {
        partition: np.bincount(split[f"y_{partition}"], minlength=len(CLASS_NAMES)).tolist()
        for partition in ("train", "validation", "test")
    }
    split_contract = {
        "protocol_id": PROTOCOL_ID,
        "execution_contract_id": contract["execution_contract_id"],
        "split_seed": split_seed,
        "rf_seed": RF_SEED,
        "split_policy": split["policy"],
        "split_sizes": {
            partition: int(len(split[f"{partition}_indices"]))
            for partition in ("train", "validation", "test")
        },
        "class_counts": class_counts,
        "split_indices_content_sha256": split_index_hash(split),
        "scaler_content_sha256": scaler_hash(scaler),
        "scaler_fit_partition": "train only",
        "feature_group_audit": split["group_audit"],
        "rf_calibration_audit": calibration_audit,
        "teacher_train_probability_content_sha256": teacher_train_probability_content_sha256,
    }
    atomic_write_json(split_root / "split_contract.json", split_contract)
    atomic_save_npz(split_root / "prediction_labels.npz", **prediction_arrays)
    completion = {
        **expected,
        "split_contract_sha256": sha256_file(split_root / "split_contract.json"),
        "prediction_labels_sha256": sha256_file(split_root / "prediction_labels.npz"),
        "teacher_fit_seconds": teacher_fit_seconds,
        "teacher_metrics": compact_metrics(teacher_metrics),
        "student_results": student_results,
        "wall_seconds": time.time() - started,
    }
    atomic_write_json(split_root / "split_completion.json", completion)
    atomic_write_json(split_root / "artifact_manifest.json", build_inventory(split_root, "complete"))
    verified = verify_split_completion(
        split_root, expected, contract, dataset, group_table
    )
    require(verified is not None, f"Fresh split verification failed: {split_seed}")
    return verified


def split_level_rows(completions: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for completion in completions:
        split_seed = completion["split_seed"]
        for student in ("student_A", "student_B"):
            matches = [
                row for row in completion["student_results"] if row["student"] == student
            ]
            require(len(matches) == len(OPTIMIZER_SEEDS), "Optimizer-seed coverage mismatch")
            deltas = [row["rf_kd_minus_scratch_macro_f1"] for row in matches]
            rows.append(
                {
                    "split_seed": split_seed,
                    "student": student,
                    "optimizer_seed_count": len(matches),
                    "scratch_macro_f1_mean": float(
                        np.mean([row["scratch"]["macro_f1"] for row in matches])
                    ),
                    "rf_kd_macro_f1_mean": float(
                        np.mean([row["rf_kd"]["macro_f1"] for row in matches])
                    ),
                    "rf_kd_minus_scratch_macro_f1": float(np.mean(deltas)),
                    "rf_kd_minus_scratch_optimizer_seed_values": json.dumps(deltas),
                    "test_rows": int(sum(completion["teacher_metrics"]["per_class_support"])),
                }
            )
    frame = pd.DataFrame(rows).sort_values(["split_seed", "student"])
    require(len(frame) == len(SPLIT_SEEDS) * 2, "Split-level row coverage mismatch")
    return frame


def aggregate_results(completions: list[dict[str, Any]]) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = split_level_rows(completions)
    summaries: dict[str, Any] = {}
    for student in ("student_A", "student_B"):
        values = frame[frame["student"] == student][
            "rf_kd_minus_scratch_macro_f1"
        ].to_numpy(dtype=np.float64)
        summaries[student] = {
            "unit": "seeded split assignment; each value averages two paired optimizer seeds",
            "values": values.tolist(),
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "median": float(np.median(values)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "interquartile_range": np.quantile(values, [0.25, 0.75]).tolist(),
            "positive_split_count": int(np.count_nonzero(values > 0.0)),
            "negative_split_count": int(np.count_nonzero(values < 0.0)),
            "zero_split_count": int(np.count_nonzero(values == 0.0)),
            "inference": (
                "descriptive sensitivity analysis; repeated holdouts from one dataset overlap "
                "and are not treated as independent inferential units"
            ),
        }
    return (
        {
            "protocol_id": PROTOCOL_ID,
            "status": "complete",
            "split_count": len(SPLIT_SEEDS),
            "optimizer_seeds_per_split": OPTIMIZER_SEEDS,
            "training_jobs": len(SPLIT_SEEDS) * len(OPTIMIZER_SEEDS) * 4,
            "rf_fits": len(SPLIT_SEEDS),
            "descriptive_summaries": summaries,
            "formal_hypothesis_test_performed": False,
            "claim_boundary": (
                "This confirmation estimates sensitivity to ten exact-feature-group split "
                "seeds for the core scratch versus RF-KD comparison. Each split-level value "
                "averages two paired optimizer seeds. The repeated holdouts overlap and are "
                "therefore reported descriptively, not as independent replications. This does "
                "not replace the finalized ten-optimizer-seed result on the fixed primary split "
                "and does not cover the full route matrix, deployment, XAI, or Edge-IIoTset."
            ),
        },
        frame,
    )


def markdown_summary(result: dict[str, Any]) -> str:
    lines = [
        "# FG-DS Multi-Split Core Confirmation",
        "",
        "Ten seeded exact-feature-group holdout assignments were evaluated. Each split-level comparison averages two paired optimizer seeds. Because the repeated holdouts overlap, this is descriptive sensitivity evidence.",
        "",
        "| Student | Mean RF-KD minus scratch macro-F1 | Sample SD | Range | Positive splits |",
        "|---|---:|---:|---:|---:|",
    ]
    for student in ("student_A", "student_B"):
        row = result["descriptive_summaries"][student]
        lines.append(
            f"| {student.replace('_', ' ').title()} | {row['mean']:.6f} | "
            f"{row['sample_std']:.6f} | [{row['minimum']:.6f}, {row['maximum']:.6f}] | "
            f"{row['positive_split_count']}/{len(SPLIT_SEEDS)} |"
        )
    lines.extend(["", "## Scope", "", result["claim_boundary"], ""])
    return "\n".join(lines)


def build_contract(dataset_path: Path, device: torch.device) -> dict[str, Any]:
    contract = {
        "protocol_id": PROTOCOL_ID,
        "dataset": {
            "path": repo_path(dataset_path),
            "sha256": sha256_file(dataset_path),
            "row_count": EXPECTED_TOTAL_ROWS,
            "class_count": len(CLASS_NAMES),
        },
        "split_seeds": SPLIT_SEEDS,
        "optimizer_seeds_per_split": OPTIMIZER_SEEDS,
        "rf_seed_fixed_across_splits": RF_SEED,
        "split_policy": (
            "70/15/15 approximate row ratios from stratified majority-label exact-feature "
            "groups; both split stages use the split seed; mixed-label groups remain intact"
        ),
        "scaler_fit_partition": "training partition of each split only",
        "student_routes": ["scratch", "rf_kd"],
        "students": {key: list(value) for key, value in STUDENT_SPECS.items()},
        "kd_hyperparameters": {"T": KD_T, "alpha": KD_ALPHA},
        "kd_hyperparameter_boundary": (
            "Historical fixed values; no multi-split validation or test retuning"
        ),
        "training_config": TRAIN_CONFIG,
        "teacher_config": RF_CONFIG,
        "teacher_calibration": "three-fold StratifiedGroupKFold within each training partition",
        "summary_unit": "split-level mean over two paired optimizer seeds",
        "primary_outcome": "RF-KD minus scratch test macro-F1",
        "inference_policy": (
            "descriptive split sensitivity only; overlapping repeated holdouts are not "
            "treated as independent inferential units"
        ),
        "multiplicity": "not applicable because no multi-split hypothesis test is performed",
        "retained_artifacts": (
            "compact metrics, hashes, split contracts, split test indices, true labels, and predicted labels"
        ),
        "intentionally_not_retained": (
            "RF objects, RF train probabilities, student checkpoints, test probabilities, copied dataset"
        ),
        "software": {
            "executed_source_sha256": sha256_file(SCRIPT_PATH),
            "common_source_sha256": sha256_file(COMMON_SOURCE),
            "rf_source_sha256": sha256_file(RF_SOURCE),
            "source_snapshots": {
                name: sha256_file(source) for name, source in SOURCE_SNAPSHOTS.items()
            },
        },
        "environment": environment_record(device),
    }
    require(contract["dataset"]["sha256"] == EXPECTED_DATASET_SHA256, "Dataset hash mismatch")
    contract["execution_contract_id"] = canonical_sha256(contract)
    return contract


def verify_source_snapshots(output_dir: Path, contract: dict[str, Any]) -> None:
    snapshots = contract["software"]["source_snapshots"]
    require(set(snapshots) == set(SOURCE_SNAPSHOTS), "Source-snapshot inventory differs")
    for name, expected_hash in snapshots.items():
        require(
            sha256_file(output_dir / name) == expected_hash,
            f"Source snapshot mismatch: {name}",
        )
    require(
        snapshots["executed_multisplit_source.py"]
        == contract["software"]["executed_source_sha256"],
        "Executed-source contract fields disagree",
    )
    require(
        snapshots["bound_common_source.py"] == contract["software"]["common_source_sha256"],
        "Common-source contract fields disagree",
    )
    require(
        snapshots["bound_rf_source.py"] == contract["software"]["rf_source_sha256"],
        "RF-source contract fields disagree",
    )


def verify_live_sources_match_contract(contract: dict[str, Any]) -> None:
    snapshots = contract["software"]["source_snapshots"]
    require(set(snapshots) == set(SOURCE_SNAPSHOTS), "Live-source inventory differs")
    for name, source in SOURCE_SNAPSHOTS.items():
        require(
            sha256_file(source) == snapshots[name],
            f"Live executable source differs from the sealed run: {name}",
        )


def load_contract_dataset(contract: dict[str, Any]) -> dict[str, Any]:
    dataset_relative = Path(contract["dataset"]["path"])
    require(
        not dataset_relative.is_absolute() and ".." not in dataset_relative.parts,
        "Dataset path in the contract is unsafe",
    )
    dataset_path = (REPO_ROOT / dataset_relative).resolve()
    require(sha256_file(dataset_path) == contract["dataset"]["sha256"], "Dataset hash differs")
    dataset = load_wsnds(dataset_path)
    require(
        len(dataset["labels"]) == contract["dataset"]["row_count"],
        "Dataset row count differs",
    )
    return dataset


def semantic_verification_record(
    contract: dict[str, Any],
    status: str,
    error: BaseException | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "execution_contract_id": contract["execution_contract_id"],
        "verified_split_count": len(SPLIT_SEEDS) if status == "passed" else 0,
        "checks": [
            "deterministic split reconstruction",
            "zero exact-feature-group partition overlap",
            "train-only scaler reconstruction",
            "five-class partition coverage",
            "group-disjoint RF calibration reconstruction",
            "prediction-label inventory and bounds",
            "teacher and student metric recomputation",
            "paired macro-F1 delta recomputation",
            "aggregate table and descriptive summary recomputation",
        ],
    }
    if error is not None:
        record["failure_type"] = type(error).__name__
        record["failure_message"] = str(error)
    return record


def verify_result_payload(
    output_dir: Path,
    contract: dict[str, Any],
    dataset: dict[str, Any],
    group_table: dict[str, Any],
) -> dict[str, Any]:
    result = read_json(output_dir / "multisplit_core_summary.json")
    require(result.get("status") == "complete", "Multi-split result is incomplete")
    require(result.get("split_count") == len(SPLIT_SEEDS), "Split count mismatch")
    completions = []
    for split_seed in SPLIT_SEEDS:
        completion = verify_split_completion(
            output_dir / f"split_{split_seed}",
            completion_expected(contract, split_seed),
            contract,
            dataset,
            group_table,
        )
        require(completion is not None, f"Split completion is missing: {split_seed}")
        completions.append(completion)
    recomputed_result, recomputed_rows = aggregate_results(completions)
    require(
        canonical_sha256(result) == canonical_sha256(recomputed_result),
        "Persisted aggregate result differs from split completions",
    )
    rows = pd.read_csv(output_dir / "multisplit_core_split_table.csv")
    try:
        pd.testing.assert_frame_equal(
            rows.reset_index(drop=True),
            recomputed_rows.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=1e-15,
        )
    except AssertionError as error:
        raise RuntimeError("Persisted split table differs from recomputation") from error
    require(
        (output_dir / "MULTISPLIT_CORE_SUMMARY.md").read_text(encoding="utf-8")
        == markdown_summary(recomputed_result),
        "Persisted Markdown summary differs from recomputation",
    )
    return {"split_count": len(completions), "result": recomputed_result}


def prepare_output_root(
    output_dir: Path,
    contract: dict[str, Any],
    resume: bool,
) -> str:
    if output_dir.exists():
        require(output_dir.is_dir(), "Output path is not a directory")
        nonempty = any(output_dir.iterdir())
    else:
        nonempty = False
    if nonempty:
        require(resume, f"Refusing to overwrite existing output: {output_dir}")
        existing = read_json(output_dir / "execution_contract.json")
        require(existing == contract, "Resume execution contract differs")
        verify_source_snapshots(output_dir, contract)
        verify_live_sources_match_contract(contract)
        return "resume"
    require(not resume, "Resume requires a non-empty existing output directory")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.initializing_",
            dir=output_dir.parent,
        )
    )
    atomic_write_json(staging / "execution_contract.json", contract)
    for name, source in SOURCE_SNAPSHOTS.items():
        shutil.copy2(source, staging / name)
    atomic_write_json(staging / "artifact_manifest.json", build_inventory(staging, "running"))
    verify_inventory(staging, {"running"})
    verify_source_snapshots(staging, contract)
    verify_live_sources_match_contract(contract)
    if output_dir.exists():
        output_dir.rmdir()
    os.replace(staging, output_dir)
    return "fresh"


def run(dataset_path: Path, output_dir: Path, device: torch.device, resume: bool) -> None:
    dataset = load_wsnds(dataset_path)
    require(dataset["dataset_sha256"] == EXPECTED_DATASET_SHA256, "Loaded dataset hash mismatch")
    require(len(dataset["labels"]) == EXPECTED_TOTAL_ROWS, "Dataset row count mismatch")
    group_table = prepare_group_table(dataset["features"], dataset["labels"])
    contract = build_contract(dataset_path, device)
    prepare_output_root(output_dir, contract, resume)

    completions = []
    for split_seed in SPLIT_SEEDS:
        completion = run_split(
            output_dir, contract, dataset, group_table, split_seed, device, resume
        )
        completions.append(completion)
        atomic_write_json(output_dir / "artifact_manifest.json", build_inventory(output_dir, "running"))
        print(
            json.dumps(
                {
                    "status": "split_complete",
                    "split_seed": split_seed,
                    "completed_splits": len(completions),
                    "total_splits": len(SPLIT_SEEDS),
                    "wall_seconds": completion["wall_seconds"],
                }
            ),
            flush=True,
        )

    result, rows = aggregate_results(completions)
    atomic_write_json(output_dir / "multisplit_core_summary.json", result)
    atomic_write_csv(output_dir / "multisplit_core_split_table.csv", rows)
    atomic_write_text(output_dir / "MULTISPLIT_CORE_SUMMARY.md", markdown_summary(result))
    try:
        verify_result_payload(output_dir, contract, dataset, group_table)
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
    try:
        verify_existing(output_dir)
    except BaseException as error:
        atomic_write_json(
            output_dir / "semantic_verification.json",
            semantic_verification_record(contract, "failed", error),
        )
        atomic_write_json(output_dir / "artifact_manifest.json", build_inventory(output_dir, "failed"))
        raise


def verify_existing(output_dir: Path) -> dict[str, Any]:
    manifest = verify_inventory(output_dir, {"complete"})
    contract = read_json(output_dir / "execution_contract.json")
    require(contract.get("execution_contract_id") == canonical_sha256({k: v for k, v in contract.items() if k != "execution_contract_id"}), "Execution contract ID mismatch")
    verify_source_snapshots(output_dir, contract)
    verify_live_sources_match_contract(contract)
    dataset = load_contract_dataset(contract)
    group_table = prepare_group_table(dataset["features"], dataset["labels"])
    semantic_report = read_json(output_dir / "semantic_verification.json")
    require(
        semantic_report == semantic_verification_record(contract, "passed"),
        "Semantic-verification report differs",
    )
    verify_result_payload(output_dir, contract, dataset, group_table)
    return {
        "status": "verified",
        "protocol_id": PROTOCOL_ID,
        "verified_files": manifest["file_count_excluding_manifest"],
        "execution_contract_id": contract["execution_contract_id"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--confirm-training", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if args.verify_existing:
        print(json.dumps(verify_existing(output), indent=2))
        return 0
    require(args.confirm_training, "Training requires --confirm-training")
    device = resolve_device(args.device)
    configure_determinism()
    run(args.dataset.resolve(), output, device, args.resume)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
