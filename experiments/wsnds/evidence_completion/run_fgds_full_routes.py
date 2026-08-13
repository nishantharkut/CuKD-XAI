"""Run a controlled WSN-DS route matrix on a feature-group-disjoint split.

This runner extends, but never overwrites, the verified ten-seed scratch/RF-KD
confirmation. It binds to that run's dataset, split, scaler, and RF soft-target
artifacts. Repeated feature rows remain within partitions; only cross-partition
feature-group overlap is prohibited. Every trainable route is reset to the same
architecture-specific initial state within a seed. This controlled RNG policy is
deliberately different from the archived sequential route execution, so changes
relative to the archive cannot be attributed to split correction alone. Neural
teacher routes are trained once per seed and shared by Student A and Student B.

The default action is a non-training preflight. Model training requires the
explicit ``--confirm-training`` flag.
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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import scipy
import sklearn
import imblearn
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

try:
    from ..leakage_free_rerun.tier15_common import (
        CLASS_NAMES,
        KD_ALPHA,
        KD_T,
        PUBLICATION_SEEDS,
        RF_CONFIG,
        STUDENT_SPECS,
        TRAIN_CONFIG,
        StudentMLP,
        apply_train_scaler,
        artifact_manifest,
        atomic_torch_save,
        atomic_write_json,
        batched_probs,
        class_weights,
        classification_metrics,
        feature_group_split,
        load_wsnds,
        set_seed,
        sha256_arrays,
        sha256_file,
        shuffled_batches,
        split_hashes,
    )
except ImportError:
    from experiments.wsnds.leakage_free_rerun.tier15_common import (
        CLASS_NAMES,
        KD_ALPHA,
        KD_T,
        PUBLICATION_SEEDS,
        RF_CONFIG,
        STUDENT_SPECS,
        TRAIN_CONFIG,
        StudentMLP,
        apply_train_scaler,
        artifact_manifest,
        atomic_torch_save,
        atomic_write_json,
        batched_probs,
        class_weights,
        classification_metrics,
        feature_group_split,
        load_wsnds,
        set_seed,
        sha256_arrays,
        sha256_file,
        shuffled_batches,
        split_hashes,
    )


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_DATASET = REPO_ROOT / "data" / "wsnds" / "WSN-DS.csv"
DEFAULT_BASE = (
    REPO_ROOT
    / "results"
    / "wsnds"
    / "confirmation_runs_v2"
    / "local_feature_group_10seed_20260811"
    / "feature_group_10seed"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results"
    / "wsnds"
    / "evidence_completion_20260811"
    / "fgds_controlled_full_routes_10seed_v2"
)
PROTOCOL_ID = "wsnds_feature_group_disjoint_controlled_full_routes_10seed_v2"
BASE_PROTOCOL_ID = "wsnds_feature_group_split_train_only_scaler_10seed_v2"

CL_STAGES_FAIR = ((0.33, 3), (0.66, 3), (1.0, 24))
CL_STAGES_EXTENDED = ((0.33, 5), (0.66, 5), (1.0, 30))
CODISTILL_CONFIG = {
    "ce_weight": 0.30,
    "rf_weight": 0.40,
    "curriculum_weight": 0.30,
    "epochs": 40,
    "lr": 7e-4,
    "weight_decay": 1e-3,
    "patience": 10,
    "batch_size": 256,
}

TRAINED_TEACHER_ROUTES = [
    "B_Full_MLP",
    "C_CL_MLP_loss_fair",
    "C_CL_MLP_loss_ext",
    "C2_CL_MLP_domain",
    "G_random_pacing_teacher",
    "I_SMOTE_MLP_teacher",
]
TEACHER_ROUTES = [
    "A_RF_500_uncalibrated",
    "A_calibrated_RF_KD_teacher",
    *TRAINED_TEACHER_ROUTES,
]
STUDENT_ROUTES = [
    "D_Small_MLP",
    "E_KD_from_RF",
    "E2_KD_from_MLP",
    "F_KD_from_CL_MLP_fair",
    "F_KD_from_CL_MLP_ext",
    "G_KD_random_pacing",
    "I_KD_from_SMOTE_MLP",
    "J_CoDistill_RF_CL",
]
ALIASES = {
    "C_CL_MLP_loss": "C_CL_MLP_loss_fair",
    "F_KD_from_CL_MLP": "F_KD_from_CL_MLP_fair",
}
TEACHER_COMPARISONS = [
    ("C_CL_MLP_loss_fair", "B_Full_MLP"),
    ("C_CL_MLP_loss_ext", "B_Full_MLP"),
]
STUDENT_COMPARISONS = [
    ("E_KD_from_RF", "D_Small_MLP"),
    ("E2_KD_from_MLP", "D_Small_MLP"),
    ("F_KD_from_CL_MLP_fair", "E2_KD_from_MLP"),
    ("F_KD_from_CL_MLP_ext", "E2_KD_from_MLP"),
    ("F_KD_from_CL_MLP_fair", "D_Small_MLP"),
    ("F_KD_from_CL_MLP_fair", "G_KD_random_pacing"),
    ("F_KD_from_CL_MLP_fair", "I_KD_from_SMOTE_MLP"),
    ("E_KD_from_RF", "E2_KD_from_MLP"),
    ("I_KD_from_SMOTE_MLP", "E2_KD_from_MLP"),
    ("J_CoDistill_RF_CL", "E_KD_from_RF"),
    ("J_CoDistill_RF_CL", "E2_KD_from_MLP"),
    ("J_CoDistill_RF_CL", "F_KD_from_CL_MLP_fair"),
]
ROUTE_SEMANTICS = {
    "C_CL_MLP_loss_fair": (
        "Legacy route identifier. The schedule matches 30 nominal epochs but "
        "processes fewer rows than 30 full-data epochs; it is not compute matched."
    ),
    "F_KD_from_CL_MLP_fair": (
        "Student distilled from C_CL_MLP_loss_fair. The inherited 'fair' token is "
        "a legacy identifier, not an equal-compute claim."
    ),
    "C_CL_MLP_loss_ext": (
        "Extended curriculum schedule with 40 nominal epochs and nonuniform row exposure."
    ),
    "I_SMOTE_MLP_teacher": (
        "Algorithmic SMOTE control on standardized tabular features; synthetic-record "
        "physical validity is not asserted."
    ),
}


class TeacherMLP(nn.Module):
    """The archived 69,893-parameter neural teacher architecture."""

    def __init__(self, input_dim: int = 17, num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.67),
            nn.Linear(128, num_classes),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=PUBLICATION_SEEDS,
        help="Defaults to the fixed ten-seed publication set.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--confirm-training", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def state_dict_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    return sha256_arrays(*[state[name].numpy() for name in sorted(state)])


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def model_resource_record(model: nn.Module) -> dict[str, Any]:
    parameters = count_parameters(model)
    return {
        "parameters": parameters,
        "fp32_parameter_payload_bytes": parameters * 4,
        "resource_scope": (
            "Trainable neural parameters only. This is not serialized artifact size, "
            "mixed-width fixed-point model size, firmware flash, peak RAM, or energy."
        ),
    }


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    total = len(labels)
    value = 0.0
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        selected = (confidence > lower) & (confidence <= upper)
        if index == 0:
            selected |= confidence == 0.0
        if not np.any(selected):
            continue
        value += float(selected.sum() / total) * abs(
            float(correct[selected].mean()) - float(confidence[selected].mean())
        )
    return float(value)


def evaluate(model: nn.Module, X: torch.Tensor, labels: np.ndarray, device: torch.device) -> tuple[dict[str, Any], np.ndarray]:
    probabilities = batched_probs(model, X, device)
    metrics = classification_metrics(labels, probabilities)
    metrics["ece_15_bin"] = expected_calibration_error(probabilities, labels)
    metrics.update(model_resource_record(model))
    return metrics, probabilities


def train_supervised(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_validation: torch.Tensor,
    y_validation: torch.Tensor,
    weights: torch.Tensor,
    device: torch.device,
    *,
    epochs: int = int(TRAIN_CONFIG["epochs"]),
    batch_size: int = int(TRAIN_CONFIG["batch_size"]),
    lr: float = float(TRAIN_CONFIG["lr"]),
    weight_decay: float = float(TRAIN_CONFIG["weight_decay"]),
    patience: int = int(TRAIN_CONFIG["patience"]),
) -> tuple[nn.Module, dict[str, Any]]:
    model = model.to(device)
    X_train_d = X_train.to(device)
    y_train_d = y_train.to(device)
    X_validation_d = X_validation.to(device)
    validation_labels = y_validation.numpy()
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_score = -1.0
    best_state = None
    stale = 0
    history: list[dict[str, float]] = []
    total_rows_processed = 0
    total_optimizer_steps = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batches = 0
        rows_processed = 0
        for X_batch, y_batch in shuffled_batches(X_train_d, y_train_d, batch_size=batch_size):
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
            rows_processed += int(len(y_batch))
        total_rows_processed += rows_processed
        total_optimizer_steps += batches
        scheduler.step()
        predictions = batched_probs(model, X_validation_d, device).argmax(axis=1)
        score = float(f1_score(validation_labels, predictions, average="macro", zero_division=0))
        history.append({
            "epoch": epoch + 1,
            "mean_train_loss": total_loss / max(batches, 1),
            "validation_macro_f1": score,
            "rows_processed": rows_processed,
            "optimizer_steps": batches,
        })
        if score > best_score:
            best_score = score
            best_state = state_dict_cpu(model)
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("Supervised route produced no validation checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "best_validation_macro_f1": best_score,
        "epochs_completed": len(history),
        "total_rows_processed": total_rows_processed,
        "total_optimizer_steps": total_optimizer_steps,
        "history": history,
    }


def train_curriculum(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    order: np.ndarray,
    X_validation: torch.Tensor,
    y_validation: torch.Tensor,
    weights: torch.Tensor,
    device: torch.device,
    stages: tuple[tuple[float, int], ...],
) -> tuple[nn.Module, dict[str, Any]]:
    if not np.array_equal(np.sort(order), np.arange(len(X_train), dtype=order.dtype)):
        raise RuntimeError("Curriculum order is not a permutation of the training rows")
    model = model.to(device)
    X_train_d = X_train.to(device)
    y_train_d = y_train.to(device)
    X_validation_d = X_validation.to(device)
    validation_labels = y_validation.numpy()
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    best_score = -1.0
    best_state = None
    stale = 0
    stopped = False
    history: list[dict[str, Any]] = []
    total_rows_processed = 0
    total_optimizer_steps = 0
    for stage_index, (fraction, stage_epochs) in enumerate(stages):
        if stopped:
            break
        count = int(len(X_train) * fraction)
        indices = torch.from_numpy(order[:count]).long().to(device)
        X_stage = X_train_d[indices]
        y_stage = y_train_d[indices]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(TRAIN_CONFIG["lr"]),
            weight_decay=float(TRAIN_CONFIG["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=stage_epochs
        )
        for stage_epoch in range(stage_epochs):
            model.train()
            total_loss = 0.0
            batches = 0
            rows_processed = 0
            for X_batch, y_batch in shuffled_batches(
                X_stage, y_stage, batch_size=int(TRAIN_CONFIG["batch_size"])
            ):
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu())
                batches += 1
                rows_processed += int(len(y_batch))
            total_rows_processed += rows_processed
            total_optimizer_steps += batches
            scheduler.step()
            predictions = batched_probs(model, X_validation_d, device).argmax(axis=1)
            score = float(
                f1_score(validation_labels, predictions, average="macro", zero_division=0)
            )
            history.append({
                "stage": stage_index + 1,
                "stage_epoch": stage_epoch + 1,
                "training_fraction": fraction,
                "training_rows": count,
                "mean_train_loss": total_loss / max(batches, 1),
                "validation_macro_f1": score,
                "rows_processed": rows_processed,
                "optimizer_steps": batches,
            })
            if score > best_score:
                best_score = score
                best_state = state_dict_cpu(model)
                stale = 0
            else:
                stale += 1
                if stale >= int(TRAIN_CONFIG["patience"]):
                    stopped = True
                    break
    if best_state is None:
        raise RuntimeError("Curriculum route produced no validation checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "best_validation_macro_f1": best_score,
        "epochs_completed": len(history),
        "total_rows_processed": total_rows_processed,
        "total_optimizer_steps": total_optimizer_steps,
        "stages": [list(stage) for stage in stages],
        "history": history,
    }


def compute_loss_order(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    set_seed(seed)
    probe = StudentMLP(17, STUDENT_SPECS["student_B"], len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    X_d = X_train.to(device)
    y_d = y_train.to(device)
    for _ in range(3):
        probe.train()
        for X_batch, y_batch in shuffled_batches(X_d, y_d, batch_size=512):
            optimizer.zero_grad()
            criterion(probe(X_batch), y_batch).backward()
            optimizer.step()
    probe.eval()
    per_row: list[np.ndarray] = []
    criterion_none = nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        for start in range(0, len(X_d), 4096):
            per_row.append(
                criterion_none(
                    probe(X_d[start : start + 4096]),
                    y_d[start : start + 4096],
                ).cpu().numpy()
            )
    losses = np.concatenate(per_row)
    order = np.argsort(losses, kind="stable").astype(np.int64)
    return order, {
        "probe_seed": seed,
        "probe_epochs": 3,
        "probe_architecture": [17, 64, 32, 5],
        "loss_min": float(losses.min()),
        "loss_mean": float(losses.mean()),
        "loss_max": float(losses.max()),
        "order_sha256": sha256_arrays(order),
    }


def domain_order(labels: np.ndarray) -> np.ndarray:
    tier_by_class = {
        "Normal": 1,
        "Blackhole": 1,
        "Grayhole": 2,
        "Flooding": 2,
        "TDMA": 3,
    }
    tiers = np.asarray([tier_by_class[CLASS_NAMES[int(value)]] for value in labels])
    within_tier = np.random.RandomState(42).random_sample(len(labels))
    return np.lexsort((within_tier, tiers)).astype(np.int64)


def teacher_targets(
    teacher: nn.Module,
    X_train: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    teacher = teacher.to(device).eval()
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(X_train), 4096):
            logits = teacher(X_train[start : start + 4096].to(device))
            chunks.append(F.softmax(logits / KD_T, dim=1).cpu())
    return torch.cat(chunks, dim=0)


def rf_targets(probabilities: np.ndarray) -> torch.Tensor:
    raw = torch.from_numpy(np.asarray(probabilities, dtype=np.float32))
    return F.softmax(torch.log(raw.clamp(min=1e-8)) / KD_T, dim=1).detach()


def train_kd_targets(
    model: nn.Module,
    soft_targets: torch.Tensor,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_validation: torch.Tensor,
    y_validation: torch.Tensor,
    weights: torch.Tensor,
    device: torch.device,
    *,
    epochs: int = int(TRAIN_CONFIG["epochs"]),
    lr: float = float(TRAIN_CONFIG["lr"]),
    weight_decay: float = float(TRAIN_CONFIG["weight_decay"]),
    patience: int = int(TRAIN_CONFIG["patience"]),
    batch_size: int = int(TRAIN_CONFIG["batch_size"]),
    hard_label_weight: float,
    component_weights: tuple[float, ...],
    extra_targets: tuple[torch.Tensor, ...] = (),
) -> tuple[nn.Module, dict[str, Any]]:
    all_targets = (soft_targets,) + extra_targets
    if any(target.shape != (len(X_train), len(CLASS_NAMES)) for target in all_targets):
        raise RuntimeError("A KD target tensor has the wrong shape")
    if len(component_weights) != len(all_targets):
        raise RuntimeError("KD component weights do not match target tensors")
    if any(weight < 0 for weight in component_weights):
        raise RuntimeError("KD component weights must be non-negative")
    if hard_label_weight < 0:
        raise RuntimeError("KD hard-label weight must be non-negative")
    if not np.isclose(
        hard_label_weight + sum(component_weights), 1.0, rtol=0.0, atol=1e-12
    ):
        raise RuntimeError("KD hard-label and teacher-component weights must sum to one")
    model = model.to(device)
    X_train_d = X_train.to(device)
    y_train_d = y_train.to(device)
    target_d = tuple(target.to(device) for target in all_targets)
    X_validation_d = X_validation.to(device)
    validation_labels = y_validation.numpy()
    weights_d = weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_d)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_score = -1.0
    best_state = None
    stale = 0
    history: list[dict[str, float]] = []
    total_rows_processed = 0
    total_optimizer_steps = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batches = 0
        rows_processed = 0
        for batch in shuffled_batches(
            X_train_d,
            y_train_d,
            *target_d,
            batch_size=batch_size,
        ):
            X_batch, y_batch, *target_batches = batch
            optimizer.zero_grad()
            logits = model(X_batch)
            log_soft = F.log_softmax(logits / KD_T, dim=1)
            loss = hard_label_weight * criterion(logits, y_batch)
            for component_weight, target_batch in zip(component_weights, target_batches):
                loss = loss + component_weight * F.kl_div(
                    log_soft, target_batch, reduction="batchmean"
                ) * (KD_T * KD_T)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
            rows_processed += int(len(y_batch))
        total_rows_processed += rows_processed
        total_optimizer_steps += batches
        scheduler.step()
        predictions = batched_probs(model, X_validation_d, device).argmax(axis=1)
        score = float(
            f1_score(validation_labels, predictions, average="macro", zero_division=0)
        )
        history.append({
            "epoch": epoch + 1,
            "mean_train_loss": total_loss / max(batches, 1),
            "validation_macro_f1": score,
            "rows_processed": rows_processed,
            "optimizer_steps": batches,
        })
        if score > best_score:
            best_score = score
            best_state = state_dict_cpu(model)
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("KD route produced no validation checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "best_validation_macro_f1": best_score,
        "epochs_completed": len(history),
        "total_rows_processed": total_rows_processed,
        "total_optimizer_steps": total_optimizer_steps,
        "T": KD_T,
        "hard_label_weight": hard_label_weight,
        "teacher_component_weights": list(component_weights),
        "history": history,
    }


def atomic_save_predictions(
    path: Path,
    source_indices: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            source_row_index=source_indices.astype(np.int64),
            true_label=labels.astype(np.int64),
            probability=probabilities.astype(np.float32),
            predicted_label=probabilities.argmax(axis=1).astype(np.int64),
        )
    os.replace(temporary, path)


def save_model_result(
    seed_root: Path,
    category: str,
    name: str,
    model: nn.Module,
    metrics: dict[str, Any],
    training: dict[str, Any],
    initial_state_sha256: str,
    probabilities: np.ndarray,
    test_indices: np.ndarray,
    test_labels: np.ndarray,
) -> dict[str, Any]:
    prefix = f"{category}_{name}"
    state = state_dict_cpu(model)
    model_path = seed_root / f"{prefix}.pt"
    prediction_path = seed_root / f"{prefix}_test_predictions.npz"
    atomic_torch_save(model_path, state)
    atomic_save_predictions(
        prediction_path, test_indices, test_labels, probabilities
    )
    return {
        "model_file": model_path.name,
        "model_file_sha256": sha256_file(model_path),
        "prediction_file": prediction_path.name,
        "prediction_file_sha256": sha256_file(prediction_path),
        "initial_state_sha256": initial_state_sha256,
        "trained_state_sha256": state_dict_sha256(state),
        "metrics": metrics,
        "training": training,
    }


def verify_root_manifest(root: Path, expected_protocol_id: str) -> dict[str, Any]:
    manifest_path = root / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("Base root manifest is not complete")
    if manifest.get("protocol_id") != expected_protocol_id:
        raise RuntimeError("Base root manifest protocol differs")
    declared = {item["path"]: item for item in manifest.get("files", [])}
    if len(declared) != len(manifest.get("files", [])):
        raise RuntimeError("Base root manifest contains duplicate paths")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if set(declared) != actual:
        raise RuntimeError("Base root inventory differs from its manifest")
    if manifest.get("file_count_excluding_manifest") != len(actual):
        raise RuntimeError("Base root manifest file count differs")
    for relative, item in declared.items():
        path = root / relative
        if path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Base root artifact failed verification: {path}")
    return {
        "payload": manifest,
        "declared": declared,
        "sha256": sha256_file(manifest_path),
    }


def metrics_from_preserved_predictions(
    path: Path,
    expected_indices: np.ndarray,
    expected_labels: np.ndarray,
    expected_metrics: dict[str, Any],
) -> dict[str, Any]:
    frame = pd.read_csv(path)
    probability_columns = [
        column for column in frame.columns if column.startswith("probability_")
    ]
    if len(probability_columns) != len(CLASS_NAMES):
        raise RuntimeError(f"Preserved probability schema differs: {path}")
    source_indices = frame["source_row_index"].to_numpy(dtype=np.int64)
    labels = frame["true_label"].to_numpy(dtype=np.int64)
    predictions = frame["predicted_label"].to_numpy(dtype=np.int64)
    probabilities = frame[probability_columns].to_numpy(dtype=np.float64)
    if not np.array_equal(source_indices, expected_indices):
        raise RuntimeError(f"Preserved source indices differ: {path}")
    if not np.array_equal(labels, expected_labels):
        raise RuntimeError(f"Preserved labels differ: {path}")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise RuntimeError(f"Preserved probabilities are invalid: {path}")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=2e-6):
        raise RuntimeError(f"Preserved probabilities do not sum to one: {path}")
    if not np.array_equal(predictions, probabilities.argmax(axis=1)):
        raise RuntimeError(f"Preserved predicted labels differ from argmax: {path}")
    metrics = classification_metrics(labels, probabilities)
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
        if key not in expected_metrics:
            raise RuntimeError(f"Expected base metrics are missing {key}: {path}")
        if not np.allclose(
            np.asarray(metrics[key], dtype=np.float64),
            np.asarray(expected_metrics[key], dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(f"Recomputed base metric differs for {key}: {path}")
    metrics["ece_15_bin"] = expected_calibration_error(probabilities, labels)
    return metrics


def metrics_from_npz_predictions(
    path: Path,
    expected_indices: np.ndarray,
    expected_labels: np.ndarray,
    expected_metrics: dict[str, Any],
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        required = {"source_row_index", "true_label", "probability", "predicted_label"}
        if set(payload.files) != required:
            raise RuntimeError(f"Prediction NPZ schema differs: {path}")
        source_indices = payload["source_row_index"]
        labels = payload["true_label"]
        probabilities = payload["probability"].astype(np.float64)
        predictions = payload["predicted_label"]
    if not np.array_equal(source_indices, expected_indices):
        raise RuntimeError(f"Prediction NPZ indices differ: {path}")
    if not np.array_equal(labels, expected_labels):
        raise RuntimeError(f"Prediction NPZ labels differ: {path}")
    if probabilities.shape != (len(expected_labels), len(CLASS_NAMES)):
        raise RuntimeError(f"Prediction NPZ probability shape differs: {path}")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise RuntimeError(f"Prediction NPZ probabilities are invalid: {path}")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=2e-6):
        raise RuntimeError(f"Prediction NPZ probabilities do not sum to one: {path}")
    if not np.array_equal(predictions, probabilities.argmax(axis=1)):
        raise RuntimeError(f"Prediction NPZ labels differ from argmax: {path}")
    recomputed = classification_metrics(labels, probabilities)
    recomputed["ece_15_bin"] = expected_calibration_error(probabilities, labels)
    for key in [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "ece_15_bin",
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
            raise RuntimeError(f"Prediction NPZ metric differs for {key}: {path}")
    return probabilities


def load_torch_mapping(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a mapping in {path}")
    return value


def verify_base_student_artifact(
    seed_root: Path,
    seed: int,
    result_name: str,
    result: dict[str, Any],
    context: dict[str, Any],
    expected_teacher_provenance: dict[str, Any] | None,
) -> None:
    student_name, expected_route = (
        (result_name.removesuffix("_scratch"), "scratch")
        if result_name.endswith("_scratch")
        else (result_name.removesuffix("_rf_kd"), "rf_kd")
    )
    rich_path = seed_root / result["rich_artifact"]
    plain_path = seed_root / result["plain_state_dict"]
    if sha256_file(rich_path) != result["rich_artifact_sha256"]:
        raise RuntimeError(f"Base rich artifact hash differs: {result_name}")
    if sha256_file(plain_path) != result["plain_state_dict_sha256"]:
        raise RuntimeError(f"Base plain checkpoint hash differs: {result_name}")
    rich = load_torch_mapping(rich_path)
    plain = load_torch_mapping(plain_path)
    required = {
        "protocol_id": BASE_PROTOCOL_ID,
        "seed": seed,
        "student": student_name,
        "route": expected_route,
        "dataset_sha256": context["dataset"]["dataset_sha256"],
        "split_hashes": context["preprocessing"]["split_hashes"],
        "scaler_sha256": context["execution"]["scaler_sha256"],
        "feature_overlap_audit": context["split"]["group_audit"],
        "initial_state_sha256": result["initial_state_sha256"],
        "trained_state_sha256": result["trained_state_sha256"],
        "metrics": result["metrics"],
    }
    for key, expected in required.items():
        if rich.get(key) != expected:
            raise RuntimeError(f"Base rich artifact differs for {result_name}:{key}")
    rich_state = rich.get("state_dict")
    if not isinstance(rich_state, dict) or set(rich_state) != set(plain):
        raise RuntimeError(f"Base rich/plain state schema differs: {result_name}")
    for key in plain:
        if not torch.equal(rich_state[key], plain[key]):
            raise RuntimeError(f"Base rich/plain state differs: {result_name}:{key}")
    observed_state_hash = state_dict_sha256(plain)
    if observed_state_hash != result["trained_state_sha256"]:
        raise RuntimeError(f"Base trained state hash differs: {result_name}")
    if expected_route == "rf_kd":
        provenance = rich.get("teacher_soft_target_provenance") or {}
        if provenance != expected_teacher_provenance:
            raise RuntimeError(f"Base RF target provenance differs: {result_name}")
    elif rich.get("teacher_soft_target_provenance") is not None:
        raise RuntimeError(f"Base scratch artifact has teacher provenance: {result_name}")


def load_context(dataset_path: Path, base_root: Path) -> dict[str, Any]:
    execution_path = base_root / "execution_contract.json"
    preprocessing_path = base_root / "preprocessing_contract.json"
    split_path = base_root / "split_indices.npz"
    scaler_path = base_root / "scaler_parameters.npz"
    aggregate_path = base_root / "aggregate_results.json"
    required = [execution_path, preprocessing_path, split_path, scaler_path, aggregate_path]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    root_manifest = verify_root_manifest(base_root, BASE_PROTOCOL_ID)
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    if execution.get("protocol_id") != BASE_PROTOCOL_ID:
        raise RuntimeError("Base run is not the verified ten-seed feature-group protocol")
    if preprocessing.get("protocol_id") != BASE_PROTOCOL_ID:
        raise RuntimeError("Base preprocessing protocol differs")
    if execution.get("seeds") != PUBLICATION_SEEDS:
        raise RuntimeError("Base run does not contain the fixed publication seed set")
    dataset = load_wsnds(dataset_path.resolve())
    if dataset["dataset_sha256"] != execution.get("dataset_sha256"):
        raise RuntimeError("Dataset hash differs from the base execution contract")
    split = feature_group_split(dataset["features"], dataset["labels"])
    scaled, scaler = apply_train_scaler(split)
    observed_split_hashes = split_hashes(split)
    observed_index_hash = sha256_arrays(
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
    checks = {
        "split_hashes": (observed_split_hashes, preprocessing.get("split_hashes")),
        "split_indices_sha256": (observed_index_hash, execution.get("split_indices_sha256")),
        "scaler_sha256": (observed_scaler_hash, execution.get("scaler_sha256")),
        "feature_overlap_audit": (
            split["group_audit"], preprocessing.get("feature_overlap_audit")
        ),
        "transformed_split_hashes": (
            observed_transformed_hashes,
            preprocessing.get("transformed_split_hashes"),
        ),
    }
    for name, (observed, expected) in checks.items():
        if observed != expected:
            raise RuntimeError(f"Reconstructed clean data contract differs for {name}")
    with np.load(split_path, allow_pickle=False) as saved:
        for partition in ["train", "validation", "test"]:
            if not np.array_equal(
                saved[f"{partition}_indices"], split[f"{partition}_indices"]
            ):
                raise RuntimeError(f"Saved {partition} indices differ from reconstruction")
    with np.load(scaler_path, allow_pickle=False) as saved:
        for name, observed in [
            ("mean", scaler.mean_),
            ("scale", scaler.scale_),
            ("var", scaler.var_),
        ]:
            if not np.array_equal(saved[name], np.asarray(observed, dtype=np.float64)):
                raise RuntimeError(f"Saved scaler {name} differs from reconstruction")
    return {
        "dataset": dataset,
        "split": split,
        "scaled": scaled,
        "scaler": scaler,
        "execution": execution,
        "preprocessing": preprocessing,
        "base_contract_sha256": sha256_file(execution_path),
        "base_preprocessing_sha256": sha256_file(preprocessing_path),
        "base_root_manifest": root_manifest,
        "transformed_split_hashes": observed_transformed_hashes,
    }


def verify_base_seed(base_root: Path, seed: int, context: dict[str, Any]) -> dict[str, Any]:
    seed_root = base_root / f"seed_{seed}"
    completion_path = seed_root / "seed_completion.json"
    manifest_path = seed_root / "artifact_manifest.json"
    if not completion_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"Base seed {seed} is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Base seed {seed} manifest is not complete")
    if manifest.get("protocol_id") != BASE_PROTOCOL_ID:
        raise RuntimeError(f"Base seed {seed} manifest protocol differs")
    declared = {item["path"]: item for item in manifest.get("files", [])}
    actual = {
        path.relative_to(seed_root).as_posix()
        for path in seed_root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if set(declared) != actual:
        raise RuntimeError(f"Base seed {seed} inventory differs from its manifest")
    for relative, item in declared.items():
        path = seed_root / relative
        if path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Base seed {seed} artifact failed verification: {path}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    required = {
        "protocol_id": BASE_PROTOCOL_ID,
        "seed": seed,
        "dataset_sha256": context["dataset"]["dataset_sha256"],
        "split_indices_sha256": context["execution"]["split_indices_sha256"],
        "scaler_sha256": context["execution"]["scaler_sha256"],
        "status": "complete",
        "execution_contract_sha256": context["base_contract_sha256"],
    }
    for key, expected in required.items():
        if completion.get(key) != expected:
            raise RuntimeError(f"Base seed {seed} differs for {key}")
    probability_path = seed_root / "rf_train_probabilities.npy"
    rf_probability = np.load(probability_path, allow_pickle=False)
    provenance = completion.get("teacher_soft_target_provenance", {})
    if sha256_arrays(rf_probability) != provenance.get("train_probability_content_sha256"):
        raise RuntimeError(f"Base seed {seed} RF probability content hash differs")
    if rf_probability.shape != (len(context["split"]["train_indices"]), len(CLASS_NAMES)):
        raise RuntimeError(f"Base seed {seed} RF probability shape differs")
    root_declared = context["base_root_manifest"]["declared"]
    for relative, observed_sha in [
        (f"seed_{seed}/artifact_manifest.json", sha256_file(manifest_path)),
        (f"seed_{seed}/seed_completion.json", sha256_file(completion_path)),
    ]:
        if relative not in root_declared or root_declared[relative]["sha256"] != observed_sha:
            raise RuntimeError(f"Base root manifest does not bind {relative}")
    test_indices = context["split"]["test_indices"]
    test_labels = context["split"]["y_test"]
    rf_prediction_path = seed_root / "RF_teacher_test_predictions.csv"
    rf_metrics = metrics_from_preserved_predictions(
        rf_prediction_path, test_indices, test_labels, completion["teacher_metrics"]
    )
    student_metrics: dict[str, dict[str, Any]] = {}
    for result_name, result in completion.get("student_results", {}).items():
        verify_base_student_artifact(
            seed_root,
            seed,
            result_name,
            result,
            context,
            (
                completion.get("teacher_soft_target_provenance")
                if result_name.endswith("_rf_kd")
                else None
            ),
        )
        prediction_path = seed_root / result["test_predictions"]
        if sha256_file(prediction_path) != result["test_predictions_sha256"]:
            raise RuntimeError(f"Base seed {seed} prediction hash differs: {result_name}")
        student_metrics[result_name] = metrics_from_preserved_predictions(
            prediction_path, test_indices, test_labels, result["metrics"]
        )
    return {
        "root": seed_root,
        "completion": completion,
        "rf_probability": rf_probability,
        "rf_metrics": rf_metrics,
        "rf_prediction_file": rf_prediction_path,
        "student_metrics": student_metrics,
        "manifest_sha256": sha256_file(manifest_path),
        "completion_sha256": sha256_file(completion_path),
        "probability_file_sha256": sha256_file(probability_path),
    }


def instantiate_with_state(
    factory: Callable[[], nn.Module],
    initial_state: dict[str, torch.Tensor],
    seed: int,
) -> nn.Module:
    set_seed(seed)
    model = factory()
    model.load_state_dict(initial_state)
    if state_dict_sha256(state_dict_cpu(model)) != state_dict_sha256(initial_state):
        raise RuntimeError("Model initialization state changed while cloning a route")
    return model


def verify_completed_seed_output(
    seed_root: Path,
    seed: int,
    execution_contract_sha256: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    completion_path = seed_root / "seed_completion.json"
    manifest_path = seed_root / "artifact_manifest.json"
    if not completion_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"Seed {seed} completion files are absent")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    required = {
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "seed": seed,
        "execution_contract_sha256": execution_contract_sha256,
    }
    for key, expected in required.items():
        if completion.get(key) != expected:
            raise RuntimeError(f"Seed {seed} completion differs for {key}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_id") != PROTOCOL_ID or manifest.get("status") != "complete":
        raise RuntimeError(f"Seed {seed} manifest contract differs")
    declared = {item["path"]: item for item in manifest.get("files", [])}
    if len(declared) != len(manifest.get("files", [])):
        raise RuntimeError(f"Seed {seed} manifest contains duplicate paths")
    actual = {
        path.relative_to(seed_root).as_posix()
        for path in seed_root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if set(declared) != actual:
        raise RuntimeError(f"Seed {seed} output inventory differs from its manifest")
    trained_student_routes = [
        route for route in STUDENT_ROUTES if route not in {"D_Small_MLP", "E_KD_from_RF"}
    ]
    expected_files = {
        "seed_completion.json",
        "teacher_A_RF_500_uncalibrated.joblib",
        "teacher_A_RF_500_uncalibrated_test_predictions.npz",
    }
    for route in TRAINED_TEACHER_ROUTES:
        expected_files.update(
            {f"teacher_{route}.pt", f"teacher_{route}_test_predictions.npz"}
        )
    for student_name in STUDENT_SPECS:
        for route in trained_student_routes:
            expected_files.update(
                {
                    f"{student_name}_{route}.pt",
                    f"{student_name}_{route}_test_predictions.npz",
                }
            )
    if actual != expected_files:
        raise RuntimeError(f"Seed {seed} output does not match the expected route schema")
    if manifest.get("file_count_excluding_manifest") != len(actual):
        raise RuntimeError(f"Seed {seed} manifest file count differs")
    for relative, item in declared.items():
        path = seed_root / relative
        if path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Seed {seed} output artifact failed verification: {path}")
    expected_teacher_keys = set(TEACHER_ROUTES) | {"C_CL_MLP_loss"}
    if set(completion.get("teacher_results", {})) != expected_teacher_keys:
        raise RuntimeError(f"Seed {seed} teacher result schema differs")
    expected_student_keys = set(STUDENT_ROUTES) | {"F_KD_from_CL_MLP"}
    if set(completion.get("student_results", {})) != set(STUDENT_SPECS):
        raise RuntimeError(f"Seed {seed} student schema differs")
    for routes in completion["student_results"].values():
        if set(routes) != expected_student_keys:
            raise RuntimeError(f"Seed {seed} student route schema differs")

    base = context["verified_base_seeds"][seed]
    expected_base_record = {
        "root_manifest_sha256": context["base_root_manifest"]["sha256"],
        "completion_sha256": base["completion_sha256"],
        "manifest_sha256": base["manifest_sha256"],
        "rf_probability_file_sha256": base["probability_file_sha256"],
        "rf_probability_content_sha256": sha256_arrays(base["rf_probability"]),
    }
    if completion.get("base_seed") != expected_base_record:
        raise RuntimeError(f"Seed {seed} base-source binding differs")
    calibrated = completion["teacher_results"]["A_calibrated_RF_KD_teacher"]
    if calibrated.get("metrics") != base["rf_metrics"]:
        raise RuntimeError(f"Seed {seed} calibrated RF metrics differ from base")
    set_seed(seed)
    independently_initialized_teacher = TeacherMLP(17, len(CLASS_NAMES))
    expected_teacher_initial_hash = state_dict_sha256(
        state_dict_cpu(independently_initialized_teacher)
    )
    del independently_initialized_teacher
    if completion.get("teacher_initial_state_sha256") != expected_teacher_initial_hash:
        raise RuntimeError(f"Seed {seed} teacher initialization binding differs")
    for student_name in STUDENT_SPECS:
        base_results = base["completion"]["student_results"]
        base_scratch = base_results[f"{student_name}_scratch"]
        base_rf_kd = base_results[f"{student_name}_rf_kd"]
        if base_scratch["initial_state_sha256"] != base_rf_kd["initial_state_sha256"]:
            raise RuntimeError(f"Seed {seed} verified base student initial states differ: {student_name}")
        expected_student_initial_hash = base_scratch["initial_state_sha256"]
        expected_resource = model_resource_record(
            StudentMLP(17, STUDENT_SPECS[student_name], len(CLASS_NAMES))
        )
        for route, suffix in [
            ("D_Small_MLP", "scratch"),
            ("E_KD_from_RF", "rf_kd"),
        ]:
            current = completion["student_results"][student_name][route]
            source = base_results[f"{student_name}_{suffix}"]
            if current.get("source_artifact_sha256") != source["rich_artifact_sha256"]:
                raise RuntimeError(f"Seed {seed} reused artifact differs: {student_name}:{route}")
            if current.get("initial_state_sha256") != expected_student_initial_hash:
                raise RuntimeError(
                    f"Seed {seed} reused initialization differs: {student_name}:{route}"
                )
            current_metrics = current.get("metrics", {})
            source_metrics = base["student_metrics"][f"{student_name}_{suffix}"]
            expected_metric_keys = set(source_metrics) | set(expected_resource)
            if set(current_metrics) != expected_metric_keys:
                raise RuntimeError(
                    f"Seed {seed} reused metric/resource schema differs: {student_name}:{route}"
                )
            for key, expected in source_metrics.items():
                if current_metrics.get(key) != expected:
                    raise RuntimeError(
                        f"Seed {seed} reused metric differs for {key}: {student_name}:{route}"
                    )
            for key, expected in expected_resource.items():
                if current_metrics.get(key) != expected:
                    raise RuntimeError(
                        f"Seed {seed} reused resource field differs for {key}: "
                        f"{student_name}:{route}"
                    )

    test_indices = context["split"]["test_indices"]
    test_labels = context["split"]["y_test"]
    rf_result = completion["teacher_results"]["A_RF_500_uncalibrated"]
    if rf_result.get("model_file") != "teacher_A_RF_500_uncalibrated.joblib":
        raise RuntimeError(f"Seed {seed} uncalibrated RF model filename differs")
    if (
        rf_result.get("prediction_file")
        != "teacher_A_RF_500_uncalibrated_test_predictions.npz"
    ):
        raise RuntimeError(f"Seed {seed} uncalibrated RF prediction filename differs")
    rf_model_path = seed_root / rf_result["model_file"]
    rf_prediction_path = seed_root / rf_result["prediction_file"]
    if sha256_file(rf_model_path) != rf_result["model_file_sha256"]:
        raise RuntimeError(f"Seed {seed} uncalibrated RF model hash differs")
    if sha256_file(rf_prediction_path) != rf_result["prediction_file_sha256"]:
        raise RuntimeError(f"Seed {seed} uncalibrated RF prediction hash differs")
    if rf_result.get("serialized_joblib_bytes") != rf_model_path.stat().st_size:
        raise RuntimeError(f"Seed {seed} uncalibrated RF serialized size differs")
    rf_probabilities = metrics_from_npz_predictions(
        rf_prediction_path, test_indices, test_labels, rf_result["metrics"]
    )
    rf_model = joblib.load(rf_model_path)
    if not isinstance(rf_model, RandomForestClassifier):
        raise RuntimeError(f"Seed {seed} RF artifact type differs")
    expected_rf_configuration = {
        "n_estimators": 500,
        "max_depth": 15,
        "random_state": seed,
        "n_jobs": -1,
        "calibrated": False,
    }
    if rf_result.get("configuration") != expected_rf_configuration:
        raise RuntimeError(f"Seed {seed} RF declared configuration differs")
    for key, expected in expected_rf_configuration.items():
        if key == "calibrated":
            continue
        if rf_model.get_params().get(key) != expected:
            raise RuntimeError(f"Seed {seed} RF parameter differs: {key}")
    if not np.array_equal(rf_model.classes_, np.arange(len(CLASS_NAMES))):
        raise RuntimeError(f"Seed {seed} RF class order differs")
    replayed_rf_probabilities = rf_model.predict_proba(context["scaled"]["X_test"])
    if not np.allclose(
        replayed_rf_probabilities, rf_probabilities, rtol=0.0, atol=6e-8
    ):
        raise RuntimeError(f"Seed {seed} RF model does not reproduce saved probabilities")
    del rf_model, replayed_rf_probabilities, rf_probabilities

    for route in TRAINED_TEACHER_ROUTES:
        result = completion["teacher_results"][route]
        model = TeacherMLP(17, len(CLASS_NAMES))
        _verify_neural_output_result(
            seed_root,
            result,
            model,
            test_indices,
            test_labels,
            context["scaled"]["X_test"],
            expected_model_file=f"teacher_{route}.pt",
            expected_prediction_file=f"teacher_{route}_test_predictions.npz",
            expected_initial_state_sha256=expected_teacher_initial_hash,
        )
    for student_name, hidden_dims in STUDENT_SPECS.items():
        base_results = base["completion"]["student_results"]
        expected_student_initial = base_results[f"{student_name}_scratch"][
            "initial_state_sha256"
        ]
        if (
            base_results[f"{student_name}_rf_kd"]["initial_state_sha256"]
            != expected_student_initial
        ):
            raise RuntimeError(f"Seed {seed} verified base initial states differ: {student_name}")
        if (
            completion["student_results"][student_name]["E_KD_from_RF"].get(
                "initial_state_sha256"
            )
            != expected_student_initial
        ):
            raise RuntimeError(f"Seed {seed} reused student initial states differ: {student_name}")
        for route in trained_student_routes:
            result = completion["student_results"][student_name][route]
            model = StudentMLP(17, hidden_dims, len(CLASS_NAMES))
            _verify_neural_output_result(
                seed_root,
                result,
                model,
                test_indices,
                test_labels,
                context["scaled"]["X_test"],
                expected_model_file=f"{student_name}_{route}.pt",
                expected_prediction_file=f"{student_name}_{route}_test_predictions.npz",
                expected_initial_state_sha256=expected_student_initial,
            )
    return completion


def _verify_neural_output_result(
    seed_root: Path,
    result: dict[str, Any],
    model: nn.Module,
    test_indices: np.ndarray,
    test_labels: np.ndarray,
    X_test: np.ndarray,
    *,
    expected_model_file: str,
    expected_prediction_file: str,
    expected_initial_state_sha256: str,
) -> None:
    if result.get("model_file") != expected_model_file:
        raise RuntimeError(f"Neural model filename differs: {expected_model_file}")
    if result.get("prediction_file") != expected_prediction_file:
        raise RuntimeError(f"Neural prediction filename differs: {expected_prediction_file}")
    if result.get("initial_state_sha256") != expected_initial_state_sha256:
        raise RuntimeError(f"Neural initial-state binding differs: {expected_model_file}")
    model_path = seed_root / result["model_file"]
    prediction_path = seed_root / result["prediction_file"]
    if sha256_file(model_path) != result["model_file_sha256"]:
        raise RuntimeError(f"Neural model hash differs: {model_path}")
    if sha256_file(prediction_path) != result["prediction_file_sha256"]:
        raise RuntimeError(f"Neural prediction hash differs: {prediction_path}")
    state = load_torch_mapping(model_path)
    model.load_state_dict(state, strict=True)
    if state_dict_sha256(state) != result["trained_state_sha256"]:
        raise RuntimeError(f"Neural state content hash differs: {model_path}")
    expected_resource = model_resource_record(model)
    for key, expected in expected_resource.items():
        if result.get("metrics", {}).get(key) != expected:
            raise RuntimeError(f"Neural resource field differs for {key}: {model_path}")
    probabilities = metrics_from_npz_predictions(
        prediction_path, test_indices, test_labels, result["metrics"]
    )
    replayed = batched_probs(
        model, torch.from_numpy(np.asarray(X_test, dtype=np.float32)), torch.device("cpu")
    )
    replay_delta = float(np.max(np.abs(replayed - probabilities)))
    if replay_delta > 5e-6:
        raise RuntimeError(
            f"Neural model CPU replay differs from saved GPU probabilities by "
            f"{replay_delta:.9g}: {model_path}"
        )
    if not np.array_equal(replayed.argmax(axis=1), probabilities.argmax(axis=1)):
        raise RuntimeError(
            f"Neural model CPU replay changes at least one predicted label: {model_path}"
        )


def train_teacher_route(
    route: str,
    factory: Callable[[], TeacherMLP],
    initial_state: dict[str, torch.Tensor],
    seed: int,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_validation: torch.Tensor,
    y_validation: torch.Tensor,
    weights: torch.Tensor,
    device: torch.device,
    loss_order: np.ndarray,
    domain_difficulty_order: np.ndarray,
    random_order: np.ndarray,
    smote_arrays: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[TeacherMLP, dict[str, Any]]:
    model = instantiate_with_state(factory, initial_state, seed)
    if route == "B_Full_MLP":
        return train_supervised(
            model, X_train, y_train, X_validation, y_validation, weights, device
        )
    if route == "C_CL_MLP_loss_fair":
        return train_curriculum(
            model,
            X_train,
            y_train,
            loss_order,
            X_validation,
            y_validation,
            weights,
            device,
            CL_STAGES_FAIR,
        )
    if route == "C_CL_MLP_loss_ext":
        return train_curriculum(
            model,
            X_train,
            y_train,
            loss_order,
            X_validation,
            y_validation,
            weights,
            device,
            CL_STAGES_EXTENDED,
        )
    if route == "C2_CL_MLP_domain":
        return train_curriculum(
            model,
            X_train,
            y_train,
            domain_difficulty_order,
            X_validation,
            y_validation,
            weights,
            device,
            CL_STAGES_FAIR,
        )
    if route == "G_random_pacing_teacher":
        return train_curriculum(
            model,
            X_train,
            y_train,
            random_order,
            X_validation,
            y_validation,
            weights,
            device,
            CL_STAGES_FAIR,
        )
    if route == "I_SMOTE_MLP_teacher":
        if smote_arrays is None:
            raise RuntimeError("SMOTE arrays were not prepared")
        X_smote, y_smote = smote_arrays
        return train_supervised(
            model,
            torch.from_numpy(X_smote.astype(np.float32, copy=False)),
            torch.from_numpy(y_smote.astype(np.int64, copy=False)),
            X_validation,
            y_validation,
            class_weights(y_smote),
            device,
        )
    raise KeyError(route)


def run_seed(
    output_root: Path,
    base_root: Path,
    context: dict[str, Any],
    seed: int,
    device: torch.device,
    execution_contract_sha256: str,
    resume: bool,
) -> dict[str, Any]:
    seed_root = output_root / f"seed_{seed}"
    completion_path = seed_root / "seed_completion.json"
    manifest_path = seed_root / "artifact_manifest.json"
    if seed_root.exists() and any(seed_root.iterdir()):
        if resume and completion_path.is_file() and manifest_path.is_file():
            return verify_completed_seed_output(
                seed_root, seed, execution_contract_sha256, context
            )
        failed = output_root / "failed_seed_attempts"
        failed.mkdir(parents=True, exist_ok=True)
        suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        os.replace(seed_root, failed / f"seed_{seed}_{suffix}")
    seed_root.mkdir(parents=True, exist_ok=False)

    base = context["verified_base_seeds"][seed]
    split = context["split"]
    scaled = context["scaled"]
    X_train = torch.from_numpy(scaled["X_train"])
    y_train = torch.from_numpy(split["y_train"])
    X_validation = torch.from_numpy(scaled["X_validation"])
    y_validation = torch.from_numpy(split["y_validation"])
    X_test = torch.from_numpy(scaled["X_test"])
    test_labels = split["y_test"]
    weights = class_weights(split["y_train"])
    started = time.time()

    set_seed(seed)
    teacher_factory: Callable[[], TeacherMLP] = lambda: TeacherMLP(17, len(CLASS_NAMES))
    teacher_initial = state_dict_cpu(teacher_factory())
    teacher_initial_hash = state_dict_sha256(teacher_initial)

    loss_order, difficulty_record = compute_loss_order(X_train, y_train, seed, device)
    domain_difficulty_order = domain_order(split["y_train"])
    random_order = np.random.RandomState(seed).permutation(len(X_train)).astype(np.int64)
    if np.array_equal(loss_order, random_order):
        raise RuntimeError("Loss-based and random curriculum orders unexpectedly match")

    print(f"seed={seed} teacher=A_RF_500_uncalibrated", flush=True)
    rf_started = time.time()
    uncalibrated_rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        random_state=seed,
        n_jobs=-1,
    )
    uncalibrated_rf.fit(scaled["X_train"], split["y_train"])
    if not np.array_equal(uncalibrated_rf.classes_, np.arange(len(CLASS_NAMES))):
        raise RuntimeError("Uncalibrated RF class order differs")
    uncalibrated_rf_probabilities = uncalibrated_rf.predict_proba(scaled["X_test"])
    persisted_uncalibrated_rf_probabilities = uncalibrated_rf_probabilities.astype(
        np.float32
    ).astype(np.float64)
    uncalibrated_rf_metrics = classification_metrics(
        test_labels, persisted_uncalibrated_rf_probabilities
    )
    uncalibrated_rf_metrics["ece_15_bin"] = expected_calibration_error(
        persisted_uncalibrated_rf_probabilities, test_labels
    )
    rf_model_path = seed_root / "teacher_A_RF_500_uncalibrated.joblib"
    rf_model_temporary = rf_model_path.with_suffix(rf_model_path.suffix + ".tmp")
    joblib.dump(uncalibrated_rf, rf_model_temporary, compress=3)
    os.replace(rf_model_temporary, rf_model_path)
    rf_prediction_path = seed_root / "teacher_A_RF_500_uncalibrated_test_predictions.npz"
    atomic_save_predictions(
        rf_prediction_path,
        split["test_indices"],
        test_labels,
        persisted_uncalibrated_rf_probabilities,
    )
    uncalibrated_rf_result = {
        "source": "fresh_controlled_route",
        "configuration": {
            "n_estimators": 500,
            "max_depth": 15,
            "random_state": seed,
            "n_jobs": -1,
            "calibrated": False,
        },
        "model_file": rf_model_path.name,
        "model_file_sha256": sha256_file(rf_model_path),
        "serialized_joblib_bytes": rf_model_path.stat().st_size,
        "prediction_file": rf_prediction_path.name,
        "prediction_file_sha256": sha256_file(rf_prediction_path),
        "metrics": uncalibrated_rf_metrics,
        "training": {"wall_seconds": time.time() - rf_started},
    }
    del (
        uncalibrated_rf,
        uncalibrated_rf_probabilities,
        persisted_uncalibrated_rf_probabilities,
    )

    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:
        raise RuntimeError("imbalanced-learn is required for the full route matrix") from exc
    smote = SMOTE(random_state=seed, k_neighbors=3)
    smote_started = time.time()
    X_smote, y_smote = smote.fit_resample(scaled["X_train"], split["y_train"])
    smote_record = {
        "random_state": seed,
        "k_neighbors": 3,
        "input_rows": int(len(split["y_train"])),
        "output_rows": int(len(y_smote)),
        "input_class_counts": np.bincount(
            split["y_train"], minlength=len(CLASS_NAMES)
        ).astype(int).tolist(),
        "output_class_counts": np.bincount(
            y_smote, minlength=len(CLASS_NAMES)
        ).astype(int).tolist(),
        "preparation_seconds": time.time() - smote_started,
        "interpretation": ROUTE_SEMANTICS["I_SMOTE_MLP_teacher"],
    }

    teacher_results: dict[str, Any] = {
        "A_RF_500_uncalibrated": uncalibrated_rf_result,
        "A_calibrated_RF_KD_teacher": {
            "source": "verified_ten_seed_base",
            "source_prediction_file": str(
                base["rf_prediction_file"].relative_to(REPO_ROOT)
            ).replace("\\", "/"),
            "source_prediction_file_sha256": sha256_file(base["rf_prediction_file"]),
            "metrics": base["rf_metrics"],
            "resource_scope": (
                "The calibrated RF was not serialized in the base run. Tree-node memory "
                "and artifact bytes are therefore unavailable and are not inferred."
            ),
        }
    }
    teachers: dict[str, TeacherMLP] = {}
    for route in TRAINED_TEACHER_ROUTES:
        print(f"seed={seed} teacher={route}", flush=True)
        route_started = time.time()
        model, training = train_teacher_route(
            route,
            teacher_factory,
            teacher_initial,
            seed,
            X_train,
            y_train,
            X_validation,
            y_validation,
            weights,
            device,
            loss_order,
            domain_difficulty_order,
            random_order,
            (X_smote, y_smote),
        )
        metrics, probabilities = evaluate(model, X_test, test_labels, device)
        training["wall_seconds"] = time.time() - route_started
        teacher_results[route] = save_model_result(
            seed_root,
            "teacher",
            route,
            model,
            metrics,
            training,
            teacher_initial_hash,
            probabilities,
            split["test_indices"],
            test_labels,
        )
        teachers[route] = model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del X_smote, y_smote

    target_by_route = {
        "E_KD_from_RF": rf_targets(base["rf_probability"]),
        "E2_KD_from_MLP": teacher_targets(teachers["B_Full_MLP"], X_train, device),
        "F_KD_from_CL_MLP_fair": teacher_targets(
            teachers["C_CL_MLP_loss_fair"], X_train, device
        ),
        "F_KD_from_CL_MLP_ext": teacher_targets(
            teachers["C_CL_MLP_loss_ext"], X_train, device
        ),
        "G_KD_random_pacing": teacher_targets(
            teachers["G_random_pacing_teacher"], X_train, device
        ),
        "I_KD_from_SMOTE_MLP": teacher_targets(
            teachers["I_SMOTE_MLP_teacher"], X_train, device
        ),
    }

    student_results: dict[str, dict[str, Any]] = {}
    for student_name, hidden_dims in STUDENT_SPECS.items():
        print(f"seed={seed} student={student_name}", flush=True)
        student_factory: Callable[[], StudentMLP] = lambda h=hidden_dims: StudentMLP(
            17, h, len(CLASS_NAMES)
        )
        set_seed(seed)
        student_initial = state_dict_cpu(student_factory())
        student_initial_hash = state_dict_sha256(student_initial)
        base_results = base["completion"]["student_results"]
        scratch_base = base_results[f"{student_name}_scratch"]
        rf_base = base_results[f"{student_name}_rf_kd"]
        if scratch_base["initial_state_sha256"] != student_initial_hash:
            raise RuntimeError(f"{student_name} scratch initialization differs from base")
        if rf_base["initial_state_sha256"] != student_initial_hash:
            raise RuntimeError(f"{student_name} RF-KD initialization differs from base")
        scratch_metrics = copy.deepcopy(base["student_metrics"][f"{student_name}_scratch"])
        scratch_metrics.update(model_resource_record(student_factory()))
        rf_metrics = copy.deepcopy(base["student_metrics"][f"{student_name}_rf_kd"])
        rf_metrics.update(model_resource_record(student_factory()))
        routes: dict[str, Any] = {
            "D_Small_MLP": {
                "source": "verified_ten_seed_base",
                "source_artifact": str(
                    (base["root"] / scratch_base["rich_artifact"]).relative_to(REPO_ROOT)
                ).replace("\\", "/"),
                "source_artifact_sha256": scratch_base["rich_artifact_sha256"],
                "initial_state_sha256": student_initial_hash,
                "metrics": scratch_metrics,
            },
            "E_KD_from_RF": {
                "source": "verified_ten_seed_base",
                "source_artifact": str(
                    (base["root"] / rf_base["rich_artifact"]).relative_to(REPO_ROOT)
                ).replace("\\", "/"),
                "source_artifact_sha256": rf_base["rich_artifact_sha256"],
                "initial_state_sha256": student_initial_hash,
                "metrics": rf_metrics,
                "teacher_probability_content_sha256": sha256_arrays(
                    base["rf_probability"]
                ),
            },
        }
        for route in [
            "E2_KD_from_MLP",
            "F_KD_from_CL_MLP_fair",
            "F_KD_from_CL_MLP_ext",
            "G_KD_random_pacing",
            "I_KD_from_SMOTE_MLP",
            "J_CoDistill_RF_CL",
        ]:
            print(f"  route={route}", flush=True)
            route_started = time.time()
            model = instantiate_with_state(student_factory, student_initial, seed)
            if route == "J_CoDistill_RF_CL":
                model, training = train_kd_targets(
                    model,
                    target_by_route["E_KD_from_RF"],
                    X_train,
                    y_train,
                    X_validation,
                    y_validation,
                    weights,
                    device,
                    epochs=int(CODISTILL_CONFIG["epochs"]),
                    lr=float(CODISTILL_CONFIG["lr"]),
                    weight_decay=float(CODISTILL_CONFIG["weight_decay"]),
                    patience=int(CODISTILL_CONFIG["patience"]),
                    batch_size=int(CODISTILL_CONFIG["batch_size"]),
                    hard_label_weight=float(CODISTILL_CONFIG["ce_weight"]),
                    component_weights=(
                        float(CODISTILL_CONFIG["rf_weight"]),
                        float(CODISTILL_CONFIG["curriculum_weight"]),
                    ),
                    extra_targets=(target_by_route["F_KD_from_CL_MLP_fair"],),
                )
            else:
                model, training = train_kd_targets(
                    model,
                    target_by_route[route],
                    X_train,
                    y_train,
                    X_validation,
                    y_validation,
                    weights,
                    device,
                    hard_label_weight=1.0 - KD_ALPHA,
                    component_weights=(KD_ALPHA,),
                )
            metrics, probabilities = evaluate(model, X_test, test_labels, device)
            training["wall_seconds"] = time.time() - route_started
            routes[route] = save_model_result(
                seed_root,
                student_name,
                route,
                model,
                metrics,
                training,
                student_initial_hash,
                probabilities,
                split["test_indices"],
                test_labels,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        routes["F_KD_from_CL_MLP"] = {
            "alias_of": "F_KD_from_CL_MLP_fair",
            "metrics": copy.deepcopy(routes["F_KD_from_CL_MLP_fair"]["metrics"]),
        }
        student_results[student_name] = routes

    for alias, source in ALIASES.items():
        if alias.startswith("C_"):
            teacher_results[alias] = {
                "alias_of": source,
                "metrics": copy.deepcopy(teacher_results[source]["metrics"]),
            }

    completion = {
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "seed": seed,
        "execution_contract_sha256": execution_contract_sha256,
        "base_seed": {
            "root_manifest_sha256": context["base_root_manifest"]["sha256"],
            "completion_sha256": base["completion_sha256"],
            "manifest_sha256": base["manifest_sha256"],
            "rf_probability_file_sha256": base["probability_file_sha256"],
            "rf_probability_content_sha256": sha256_arrays(base["rf_probability"]),
        },
        "difficulty": difficulty_record,
        "orders": {
            "domain_sha256": sha256_arrays(domain_difficulty_order),
            "random_sha256": sha256_arrays(random_order),
        },
        "smote": smote_record,
        "route_semantics": ROUTE_SEMANTICS,
        "teacher_initial_state_sha256": teacher_initial_hash,
        "teacher_results": teacher_results,
        "student_results": student_results,
        "wall_seconds": time.time() - started,
    }
    atomic_write_json(completion_path, completion)
    atomic_write_json(
        manifest_path, artifact_manifest(seed_root, PROTOCOL_ID, "complete")
    )
    return completion


def scalar_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "values": array.tolist(),
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def metric_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    required_metrics = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "ece_15_bin",
    ]
    for index, record in enumerate(records):
        missing = [metric for metric in required_metrics if metric not in record]
        if missing:
            raise RuntimeError(f"Aggregate record {index} lacks metrics: {missing}")
    result = {
        metric: scalar_summary([float(record[metric]) for record in records])
        for metric in required_metrics
    }
    per_class = np.asarray([record["per_class_f1"] for record in records], dtype=np.float64)
    result["per_class_f1"] = {
        "class_names": CLASS_NAMES,
        "values": per_class.tolist(),
        "mean": per_class.mean(axis=0).tolist(),
        "sample_std": per_class.std(axis=0, ddof=1).tolist(),
    }
    if all("parameters" in record for record in records):
        parameters = {int(record["parameters"]) for record in records}
        payloads = {int(record["fp32_parameter_payload_bytes"]) for record in records}
        if len(parameters) != 1 or len(payloads) != 1:
            raise RuntimeError("Neural resource records differ across seeds")
        result["parameters"] = parameters.pop()
        result["fp32_parameter_payload_bytes"] = payloads.pop()
    return result


def exact_sign_flip_p(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=np.float64)
    observed = abs(float(differences.mean()))
    signs = np.asarray(
        [
            [1.0 if (mask >> index) & 1 else -1.0 for index in range(len(differences))]
            for mask in range(1 << len(differences))
        ],
        dtype=np.float64,
    )
    permuted = np.abs((signs * differences).mean(axis=1))
    return float(np.mean(permuted >= observed - 1e-15))


def paired_test(left: list[float], right: list[float]) -> dict[str, Any]:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    difference = left_array - right_array
    if np.allclose(difference, 0.0, rtol=0.0, atol=0.0):
        wilcoxon_statistic, wilcoxon_p = 0.0, 1.0
        method = "all_zero_differences"
    else:
        result = stats.wilcoxon(
            left_array,
            right_array,
            alternative="two-sided",
            zero_method="wilcox",
            method="approx",
        )
        wilcoxon_statistic = float(result.statistic)
        wilcoxon_p = float(result.pvalue)
        method = "scipy_approximation"
    return {
        "left_values": left_array.tolist(),
        "right_values": right_array.tolist(),
        "difference": scalar_summary(difference.tolist()),
        "wilcoxon": {
            "statistic": wilcoxon_statistic,
            "p_value_two_sided": wilcoxon_p,
            "method": method,
            "zero_method": "wilcox",
            "zero_difference_count": int(np.count_nonzero(difference == 0.0)),
            "nonzero_difference_count": int(np.count_nonzero(difference != 0.0)),
        },
        "exact_sign_flip_mean_difference_p_two_sided": exact_sign_flip_p(difference),
    }


def apply_holm(
    tests: dict[str, dict[str, Any]], p_value_getter: Callable[[dict[str, Any]], float],
    output_field: str,
) -> None:
    ordered = sorted(
        tests,
        key=lambda name: p_value_getter(tests[name]),
    )
    running = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        raw = p_value_getter(tests[name])
        adjusted = min(1.0, raw * (total - rank))
        running = max(running, adjusted)
        tests[name][output_field] = running


def aggregate(
    output_root: Path,
    seeds: list[int],
    execution_contract_sha256: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    completions = {
        seed: verify_completed_seed_output(
            output_root / f"seed_{seed}", seed, execution_contract_sha256, context
        )
        for seed in seeds
    }
    teacher_aggregate: dict[str, Any] = {}
    for route in TEACHER_ROUTES:
        teacher_aggregate[route] = metric_aggregate(
            [completions[seed]["teacher_results"][route]["metrics"] for seed in seeds]
        )
    student_aggregate: dict[str, Any] = {}
    tests: dict[str, Any] = {}
    for left, right in TEACHER_COMPARISONS:
        name = f"teacher:{left}_minus_{right}"
        tests[name] = paired_test(
            teacher_aggregate[left]["macro_f1"]["values"],
            teacher_aggregate[right]["macro_f1"]["values"],
        )
        tests[name]["family"] = "teacher"
    for student in STUDENT_SPECS:
        student_aggregate[student] = {}
        for route in STUDENT_ROUTES:
            student_aggregate[student][route] = metric_aggregate(
                [
                    completions[seed]["student_results"][student][route]["metrics"]
                    for seed in seeds
                ]
            )
        for left, right in STUDENT_COMPARISONS:
            name = f"{student}:{left}_minus_{right}"
            tests[name] = paired_test(
                student_aggregate[student][left]["macro_f1"]["values"],
                student_aggregate[student][right]["macro_f1"]["values"],
            )
            tests[name]["family"] = student
    for family in ["teacher", *STUDENT_SPECS.keys()]:
        family_tests = {
            name: value for name, value in tests.items() if value["family"] == family
        }
        apply_holm(
            family_tests,
            lambda value: value["wilcoxon"]["p_value_two_sided"],
            "holm_adjusted_wilcoxon_within_family_p",
        )
        apply_holm(
            family_tests,
            lambda value: value["exact_sign_flip_mean_difference_p_two_sided"],
            "holm_adjusted_sign_flip_within_family_p",
        )
    apply_holm(
        tests,
        lambda value: value["wilcoxon"]["p_value_two_sided"],
        "holm_adjusted_wilcoxon_global_p",
    )
    apply_holm(
        tests,
        lambda value: value["exact_sign_flip_mean_difference_p_two_sided"],
        "holm_adjusted_sign_flip_global_p",
    )
    payload = {
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "seeds": seeds,
        "seed_count": len(seeds),
        "teacher_aggregate": teacher_aggregate,
        "student_aggregate": student_aggregate,
        "paired_route_tests": tests,
        "aliases_excluded_from_inference": ALIASES,
        "statistical_unit": (
            "algorithmic run seed on one fixed feature-group-disjoint split; the seed "
            "controls initialization, dropout, minibatch order, difficulty probing, "
            "random pacing, SMOTE, and random-forest construction; repeated rows "
            "within a partition are retained"
        ),
        "standard_deviation_definition": "sample SD across algorithmic run seeds (ddof=1)",
        "holm_families": ["teacher", *STUDENT_SPECS.keys()],
    }
    atomic_write_json(output_root / "aggregate_results.json", payload)
    return payload


def environment_record(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "imbalanced_learn": imblearn.__version__,
        "joblib": joblib.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor()
        ),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
    }


def execution_contract(
    context: dict[str, Any], base_root: Path, seeds: list[int], device: torch.device
) -> dict[str, Any]:
    codistill_weight_sum = sum(
        float(CODISTILL_CONFIG[key])
        for key in ["ce_weight", "rf_weight", "curriculum_weight"]
    )
    if not np.isclose(codistill_weight_sum, 1.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("Co-distillation component weights must sum to one")
    payload = {
        "protocol_id": PROTOCOL_ID,
        "script_sha256": sha256_file(SCRIPT_PATH),
        "common_module_sha256": sha256_file(
            REPO_ROOT / "experiments/wsnds/leakage_free_rerun/tier15_common.py"
        ),
        "base_root_recorded": str(base_root.resolve()),
        "base_execution_contract_sha256": context["base_contract_sha256"],
        "base_preprocessing_contract_sha256": context["base_preprocessing_sha256"],
        "base_root_manifest_sha256": context["base_root_manifest"]["sha256"],
        "base_seed_sources": {
            str(seed): {
                "completion_sha256": context["verified_base_seeds"][seed][
                    "completion_sha256"
                ],
                "manifest_sha256": context["verified_base_seeds"][seed][
                    "manifest_sha256"
                ],
                "rf_probability_file_sha256": context["verified_base_seeds"][seed][
                    "probability_file_sha256"
                ],
            }
            for seed in seeds
        },
        "dataset_sha256": context["dataset"]["dataset_sha256"],
        "split_indices_sha256": context["execution"]["split_indices_sha256"],
        "scaler_sha256": context["execution"]["scaler_sha256"],
        "transformed_split_hashes": context["transformed_split_hashes"],
        "feature_overlap_audit": context["split"]["group_audit"],
        "evaluation_design": {
            "name": "controlled feature-group-disjoint reimplementation",
            "cross_partition_exact_feature_group_overlap": 0,
            "within_partition_repeated_rows_retained": True,
            "mixed_label_feature_groups_retained": int(
                context["split"]["group_audit"]["conflicting_label_feature_groups"]
            ),
            "archive_comparison_boundary": (
                "Route-level RNG resets and shared initial states differ from the archived "
                "sequential execution. Archived-to-current changes cannot be attributed "
                "to split correction alone."
            ),
        },
        "seeds": seeds,
        "teacher_routes": TEACHER_ROUTES,
        "student_routes": STUDENT_ROUTES,
        "aliases": ALIASES,
        "route_semantics": ROUTE_SEMANTICS,
        "student_specs": {name: list(value) for name, value in STUDENT_SPECS.items()},
        "kd": {
            "T": KD_T,
            "alpha": KD_ALPHA,
            "source": context["execution"].get("kd_hyperparameter_source"),
            "current_protocol_policy": (
                "Historical values are retained without confirmation-test retuning and "
                "are used for every KD route and seed. They are not asserted to be an "
                "RF-KD optimum."
            ),
        },
        "training": TRAIN_CONFIG,
        "curriculum": {
            "legacy_fair_route_stages": [list(stage) for stage in CL_STAGES_FAIR],
            "extended_stages": [list(stage) for stage in CL_STAGES_EXTENDED],
            "loss_probe_epochs": 3,
            "budget_boundary": (
                "The legacy fair route is nominal-epoch-matched only. Actual processed "
                "rows and optimizer steps are persisted for newly trained extension "
                "routes and can differ because stage fractions and early stopping differ. "
                "The reused base D/E artifacts do not preserve equivalent exposure logs."
            ),
            "route_rng_policy": (
                "Difficulty construction is followed by an explicit route seed reset. "
                "All same-architecture teacher routes share initial weights; all "
                "same-architecture student routes share initial weights."
            ),
        },
        "codistillation": CODISTILL_CONFIG,
        "rf_teachers": {
            "A_RF_500_uncalibrated": {
                "configuration": {
                    "n_estimators": 500,
                    "max_depth": 15,
                    "random_state": "algorithmic run seed",
                    "n_jobs": -1,
                    "calibrated": False,
                },
                "source": "fresh fit in this controlled extension",
            },
            "A_calibrated_RF_KD_teacher": {
                "configuration": RF_CONFIG,
                "source": (
                    "Verified per-seed calibrated RF probability cache from the clean "
                    "ten-seed base; no calibrated RF refit in this extension."
                ),
            },
        },
        "statistical_unit": (
            "algorithmic run seed on one fixed feature-group-disjoint split; repeated rows "
            "within a partition are retained"
        ),
        "environment": environment_record(device),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["execution_fingerprint_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def ensure_output(root: Path, resume: bool) -> None:
    protected = [DEFAULT_BASE.resolve(), DEFAULT_DATASET.resolve()]
    resolved = root.resolve()
    for item in protected:
        try:
            resolved.relative_to(item)
            raise RuntimeError(f"Output is inside protected evidence/source: {resolved}")
        except ValueError:
            pass
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError(f"Refusing to overwrite non-empty output: {root}")
    root.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    if args.seeds != PUBLICATION_SEEDS:
        raise RuntimeError(
            f"Publication run requires exactly {PUBLICATION_SEEDS}; received {args.seeds}"
        )
    device = resolve_device(args.device)
    set_seed(args.seeds[0])
    context = load_context(args.dataset_csv, args.base_root)
    context["verified_base_seeds"] = {
        seed: verify_base_seed(args.base_root, seed, context) for seed in args.seeds
    }
    contract = execution_contract(context, args.base_root, args.seeds, device)
    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "dataset_sha256": context["dataset"]["dataset_sha256"],
        "split_indices_sha256": context["execution"]["split_indices_sha256"],
        "scaler_sha256": context["execution"]["scaler_sha256"],
        "feature_overlap_audit": context["split"]["group_audit"],
        "device": str(device),
        "output": str(args.output_dir.resolve()),
    }, indent=2), flush=True)
    if args.preflight_only or not args.confirm_training:
        if not args.preflight_only:
            print("Training was not started. Pass --confirm-training to run.")
        return 0
    ensure_output(args.output_dir, args.resume)
    contract_path = args.output_dir / "execution_contract.json"
    if contract_path.is_file():
        observed = json.loads(contract_path.read_text(encoding="utf-8"))
        if observed != contract:
            raise RuntimeError("Existing execution contract differs from current code/config")
    else:
        atomic_write_json(contract_path, contract)
    contract_sha256 = sha256_file(contract_path)
    for seed in args.seeds:
        completion = run_seed(
            args.output_dir,
            args.base_root,
            context,
            seed,
            device,
            contract_sha256,
            args.resume,
        )
        print(
            f"completed seed={seed} wall_seconds={completion['wall_seconds']:.1f}",
            flush=True,
        )
    aggregate(args.output_dir, args.seeds, contract_sha256, context)
    atomic_write_json(
        args.output_dir / "artifact_manifest.json",
        artifact_manifest(args.output_dir, PROTOCOL_ID, "complete"),
    )
    print(f"complete: {args.output_dir / 'aggregate_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
