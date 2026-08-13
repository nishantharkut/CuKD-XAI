"""Shared, side-effect-free components for the compact confirmation runs."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


CLASS_NAMES = ["Blackhole", "Flooding", "Grayhole", "Normal", "TDMA"]
STUDENT_SPECS = {
    "student_A": (32, 16),
    "student_B": (64, 32),
}
PUBLICATION_SEEDS = [42, 123, 456, 789, 1001, 2024, 3141, 5678, 8192, 9999]
SENSITIVITY_SEEDS = [42, 123, 456, 789, 1001]
KD_T = 4.0
KD_ALPHA = 0.7
TRAIN_CONFIG = {
    "epochs": 30,
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay": 1e-3,
    "patience": 8,
}
RF_CONFIG = {
    "n_estimators": 500,
    "max_depth": 15,
    "calibration_method": "isotonic",
    "calibration_cv": 3,
}


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-encode {type(value)!r}")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=json_default)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(temporary, path)


def atomic_save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def load_wsnds(csv_path: Path) -> dict[str, Any]:
    frame = pd.read_csv(csv_path)
    frame.columns = frame.columns.str.strip()
    target = next(
        (
            name
            for name in ["Attack type", "Attack_Type", "attack_type", "Attack Type", "class"]
            if name in frame.columns
        ),
        frame.columns[-1],
    )
    for candidate in ["id", "Id", "ID"]:
        if candidate in frame.columns:
            frame = frame.drop(columns=[candidate])
            break
    frame[target] = frame[target].astype(str).str.strip()
    encoder = LabelEncoder()
    labels = encoder.fit_transform(frame[target]).astype(np.int64)
    features = frame.drop(columns=[target])
    feature_names = features.columns.tolist()
    class_names = encoder.classes_.tolist()
    if feature_names != [
        "Time", "Is_CH", "who CH", "Dist_To_CH", "ADV_S", "ADV_R", "JOIN_S",
        "JOIN_R", "SCH_S", "SCH_R", "Rank", "DATA_S", "DATA_R",
        "Data_Sent_To_BS", "dist_CH_To_BS", "send_code", "Expaned Energy",
    ]:
        raise RuntimeError(f"Unexpected WSN-DS feature contract: {feature_names}")
    if class_names != CLASS_NAMES:
        raise RuntimeError(f"Unexpected WSN-DS class contract: {class_names}")
    values = features.to_numpy(dtype=np.float32)
    if values.shape != (374661, 17):
        raise RuntimeError(f"Unexpected WSN-DS matrix shape: {values.shape}")
    if not np.isfinite(values).all():
        raise RuntimeError("WSN-DS feature matrix contains NaN or infinity")
    class_counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    if len(class_counts) != len(CLASS_NAMES) or np.any(class_counts == 0):
        raise RuntimeError(f"WSN-DS class distribution is incomplete: {class_counts.tolist()}")
    return {
        "features": values,
        "labels": labels,
        "feature_names": feature_names,
        "class_names": class_names,
        "target_column": target,
        "dataset_sha256": sha256_file(csv_path),
    }


def archived_random_split(features: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    indices = np.arange(len(labels), dtype=np.int64)
    trainval_idx, test_idx = train_test_split(
        indices, test_size=0.15, random_state=42, stratify=labels
    )
    train_idx, validation_idx = train_test_split(
        trainval_idx,
        test_size=0.1765,
        random_state=42,
        stratify=labels[trainval_idx],
    )
    split = _build_split(features, labels, train_idx, validation_idx, test_idx)
    split["policy"] = "archived_seed42_stratified_random_row"
    split["group_audit"] = feature_overlap_audit(features, split)
    return split


def feature_group_split(features: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    feature_hashes = verified_feature_hashes(features)
    group_table = pd.DataFrame({"group": feature_hashes, "label": labels})
    counts = pd.crosstab(group_table["group"], group_table["label"])
    counts = counts.reindex(columns=range(len(CLASS_NAMES)), fill_value=0)
    group_ids = counts.index.to_numpy(dtype=np.uint64)
    group_labels = counts.to_numpy().argmax(axis=1).astype(np.int64)
    trainval_groups, test_groups = train_test_split(
        group_ids, test_size=0.15, random_state=42, stratify=group_labels
    )
    trainval_lookup = pd.Series(group_labels, index=group_ids).loc[trainval_groups].to_numpy()
    train_groups, validation_groups = train_test_split(
        trainval_groups,
        test_size=0.1765,
        random_state=42,
        stratify=trainval_lookup,
    )
    train_mask = np.isin(feature_hashes, train_groups)
    validation_mask = np.isin(feature_hashes, validation_groups)
    test_mask = np.isin(feature_hashes, test_groups)
    assignment_count = train_mask.astype(np.int8) + validation_mask + test_mask
    if not np.all(assignment_count == 1):
        raise RuntimeError("Feature-group split did not assign every row exactly once")
    split = _build_split(
        features,
        labels,
        np.flatnonzero(train_mask),
        np.flatnonzero(validation_mask),
        np.flatnonzero(test_mask),
    )
    audit = feature_overlap_audit(features, split)
    if any(audit[key] != 0 for key in [
        "train_validation_feature_overlap",
        "train_test_feature_overlap",
        "validation_test_feature_overlap",
    ]):
        raise RuntimeError(f"Feature groups cross partitions: {audit}")
    conflicting_groups = int((counts.gt(0).sum(axis=1) > 1).sum())
    audit.update({
        "num_feature_groups": int(len(group_ids)),
        "conflicting_label_feature_groups": conflicting_groups,
        "group_stratification_label": "majority label; smallest class index breaks ties",
    })
    split["policy"] = "seed42_stratified_feature_group_split"
    split["group_audit"] = audit
    return split


def _build_split(
    features: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, Any]:
    result = {
        "train_indices": np.asarray(train_idx, dtype=np.int64),
        "validation_indices": np.asarray(validation_idx, dtype=np.int64),
        "test_indices": np.asarray(test_idx, dtype=np.int64),
    }
    for name in ["train", "validation", "test"]:
        index = result[f"{name}_indices"]
        result[f"X_{name}_raw"] = features[index]
        result[f"y_{name}"] = labels[index]
    return result


def feature_overlap_audit(features: np.ndarray, split: dict[str, Any]) -> dict[str, int]:
    hashes = verified_feature_hashes(features)
    train = set(map(int, hashes[split["train_indices"]]))
    validation = set(map(int, hashes[split["validation_indices"]]))
    test = set(map(int, hashes[split["test_indices"]]))
    return {
        "train_validation_feature_overlap": len(train & validation),
        "train_test_feature_overlap": len(train & test),
        "validation_test_feature_overlap": len(validation & test),
    }


def verified_feature_hashes(features: np.ndarray) -> np.ndarray:
    """Hash feature rows and fail if one hash represents unequal rows."""
    hashes = pd.util.hash_pandas_object(
        pd.DataFrame(features), index=False
    ).to_numpy(dtype=np.uint64)
    duplicated = pd.Series(hashes).duplicated(keep=False).to_numpy()
    duplicate_indices = np.flatnonzero(duplicated)
    if len(duplicate_indices) == 0:
        return hashes

    order = np.argsort(hashes[duplicate_indices], kind="stable")
    sorted_indices = duplicate_indices[order]
    for previous, current in zip(sorted_indices[:-1], sorted_indices[1:]):
        if hashes[previous] != hashes[current]:
            continue
        if not np.array_equal(features[previous], features[current], equal_nan=True):
            raise RuntimeError(
                "A 64-bit feature hash maps to unequal WSN-DS rows; "
                "the feature-group split is unsafe"
            )
    return hashes


def apply_train_scaler(split: dict[str, Any]) -> tuple[dict[str, np.ndarray], StandardScaler]:
    scaler = StandardScaler()
    transformed = {
        "X_train": scaler.fit_transform(split["X_train_raw"]).astype(np.float32, copy=False),
        "X_validation": scaler.transform(split["X_validation_raw"]).astype(np.float32, copy=False),
        "X_test": scaler.transform(split["X_test_raw"]).astype(np.float32, copy=False),
    }
    return transformed, scaler


def split_hashes(split: dict[str, Any]) -> dict[str, str]:
    return {
        name: sha256_arrays(split[f"X_{name}_raw"], split[f"y_{name}"])
        for name in ["train", "validation", "test"]
    }


class StudentMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, int], num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], num_classes),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def shuffled_batches(*tensors: torch.Tensor, batch_size: int):
    if not tensors:
        return
    size = len(tensors[0])
    if any(len(tensor) != size for tensor in tensors):
        raise ValueError("Batch tensors have unequal first dimensions")
    torch.empty((), dtype=torch.int64).random_()
    sampler_seed = int(torch.empty((), dtype=torch.int64).random_().item())
    generator = torch.Generator()
    generator.manual_seed(sampler_seed)
    order = torch.randperm(size, generator=generator).to(tensors[0].device)
    for start in range(0, size, batch_size):
        index = order[start:start + batch_size]
        yield tuple(tensor[index] for tensor in tensors)


def batched_probs(model: nn.Module, values: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(values), 4096):
            logits = model(values[start:start + 4096].to(device))
            outputs.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(outputs)


def train_standard(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_validation: torch.Tensor,
    y_validation: torch.Tensor,
    class_weights: torch.Tensor,
    device: torch.device,
) -> nn.Module:
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=TRAIN_CONFIG["lr"], weight_decay=TRAIN_CONFIG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TRAIN_CONFIG["epochs"]
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    X_train_d = X_train.to(device)
    y_train_d = y_train.to(device)
    X_validation_d = X_validation.to(device)
    y_validation_np = y_validation.numpy()
    best_f1 = -1.0
    best_state = None
    stale = 0
    for _ in range(TRAIN_CONFIG["epochs"]):
        model.train()
        for X_batch, y_batch in shuffled_batches(
            X_train_d, y_train_d, batch_size=TRAIN_CONFIG["batch_size"]
        ):
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()
        predictions = batched_probs(model, X_validation_d, device).argmax(axis=1)
        score = f1_score(y_validation_np, predictions, average="macro", zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= TRAIN_CONFIG["patience"]:
                break
    if best_state is None:
        raise RuntimeError("Supervised training produced no validation checkpoint")
    model.load_state_dict(best_state)
    return model


def train_rf_kd(
    model: nn.Module,
    rf_probabilities: np.ndarray,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_validation: torch.Tensor,
    y_validation: torch.Tensor,
    class_weights: torch.Tensor,
    device: torch.device,
) -> nn.Module:
    if rf_probabilities.shape != (len(X_train), len(CLASS_NAMES)):
        raise RuntimeError(f"Unexpected RF probability shape: {rf_probabilities.shape}")
    if not np.isfinite(rf_probabilities).all() or np.any(rf_probabilities < 0):
        raise RuntimeError("RF probabilities are invalid")
    if not np.allclose(rf_probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6):
        raise RuntimeError("RF probability rows do not sum to one")
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=TRAIN_CONFIG["lr"], weight_decay=TRAIN_CONFIG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TRAIN_CONFIG["epochs"]
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    X_train_d = X_train.to(device)
    y_train_d = y_train.to(device)
    X_validation_d = X_validation.to(device)
    y_validation_np = y_validation.numpy()
    raw = torch.tensor(rf_probabilities, dtype=torch.float32, device=device)
    teacher_targets = F.softmax(torch.log(raw.clamp(min=1e-8)) / KD_T, dim=1).detach()
    best_f1 = -1.0
    best_state = None
    stale = 0
    for _ in range(TRAIN_CONFIG["epochs"]):
        model.train()
        for X_batch, y_batch, teacher_batch in shuffled_batches(
            X_train_d,
            y_train_d,
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
        predictions = batched_probs(model, X_validation_d, device).argmax(axis=1)
        score = f1_score(y_validation_np, predictions, average="macro", zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= TRAIN_CONFIG["patience"]:
                break
    if best_state is None:
        raise RuntimeError("RF-KD training produced no validation checkpoint")
    model.load_state_dict(best_state)
    return model


def class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    return torch.tensor(
        len(labels) / (len(CLASS_NAMES) * np.maximum(counts, 1)), dtype=torch.float32
    )


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=np.arange(len(CLASS_NAMES)), zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_support": support.tolist(),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=np.arange(len(CLASS_NAMES))
        ).tolist(),
    }


def artifact_manifest(root: Path, protocol_id: str, status: str) -> dict[str, Any]:
    manifest_path = root / "artifact_manifest.json"
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == manifest_path or path.name.endswith(".tmp"):
            continue
        files.append({
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return {
        "protocol_id": protocol_id,
        "status": status,
        "file_count_excluding_manifest": len(files),
        "files": files,
    }
