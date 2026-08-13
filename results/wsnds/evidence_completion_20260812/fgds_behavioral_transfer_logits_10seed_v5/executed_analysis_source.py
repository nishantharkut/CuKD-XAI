"""Quantify RF-to-student response transfer from checkpoint logits.

This additive v2 analysis preserves the completed probability-artifact audit as
historical evidence. It reconstructs each saved student checkpoint and computes
student T=4 outputs directly as softmax(logits / 4), matching the KD and XAI
contracts. T=4 evaluation uses float64 arithmetic on the original float32
checkpoint logits so representable nonzero tails are not lost to float32
exponential underflow. The calibrated RF T=4 target uses the same 1e-8
probability floor as training. No model is fitted and no existing artifact is
modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.wsnds.evidence_completion.analyze_fgds_group_balanced_routes import (  # noqa: E402
    exact_paired_wilcoxon,
    holm_adjust,
)
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


PROTOCOL_ID = "wsnds_fgds_behavioral_transfer_logits_10seed_v5"
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
EXPECTED_TEST_ROWS = 56_301
EXPECTED_TEST_GROUPS = 54_174
TEMPERATURE = 4.0
PROBABILITY_FLOOR = 1e-8
PROBABILITY_SUM_ATOL = 2e-6
CHECKPOINT_REPLAY_ATOL = 3e-6
WITHIN_GROUP_PROBABILITY_ATOL = 2e-6

DEFAULT_RUN_ROOT = (
    REPO_ROOT
    / "results/wsnds/confirmation_runs_v2/local_feature_group_10seed_20260811"
    / "feature_group_10seed"
)
DEFAULT_DATASET = REPO_ROOT / "data/wsnds/WSN-DS.csv"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results/wsnds/evidence_completion_20260812"
    / "fgds_behavioral_transfer_logits_10seed_v5"
)

STUDENTS = {
    "student_A": {
        "scratch": {
            "predictions": "student_A_Small_MLP_scratch_test_predictions.csv",
            "checkpoint": "student_A_Small_MLP_scratch_fp32.pt",
            "artifact": "student_A_Small_MLP_scratch_artifact.pt",
        },
        "rf_kd": {
            "predictions": "student_A_KD_from_RF_test_predictions.csv",
            "checkpoint": "student_A_KD_from_RF_fp32.pt",
            "artifact": "student_A_KD_from_RF_artifact.pt",
        },
    },
    "student_B": {
        "scratch": {
            "predictions": "student_B_Small_MLP_scratch_test_predictions.csv",
            "checkpoint": "student_B_Small_MLP_scratch_fp32.pt",
            "artifact": "student_B_Small_MLP_scratch_artifact.pt",
        },
        "rf_kd": {
            "predictions": "student_B_KD_from_RF_test_predictions.csv",
            "checkpoint": "student_B_KD_from_RF_fp32.pt",
            "artifact": "student_B_KD_from_RF_artifact.pt",
        },
    },
}
TEACHER_FILE = "RF_teacher_test_predictions.csv"
PRIMARY_METRIC = "kl_teacher_to_student_T4"

SIMILARITY_METRICS = {
    "hard_agreement_T1",
    "top2_overlap_T1",
    "macro_class_hard_agreement_T1",
}
DISTANCE_METRICS = {
    "kl_teacher_to_student_T4",
    "js_T4",
    "l1_T4",
    "margin_abs_error_T1",
    "macro_class_kl_T4",
}
SECONDARY_TEST_METRICS = [
    "js_T4",
    "l1_T4",
    "top2_overlap_T1",
    "hard_agreement_T1",
    "macro_class_kl_T4",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def configure_deterministic_inference() -> None:
    """Make checkpoint replay independent of threaded CPU reduction order."""

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def repo_path(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Required JSON is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    normalized = json.loads(
        json.dumps(value, ensure_ascii=True, allow_nan=False)
    )
    payload = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return sha256_bytes(payload)


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


def build_inventory(root: Path, protocol_id: str, status: str) -> dict[str, Any]:
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
        "status": status,
        "file_count_excluding_manifest": len(files),
        "files": files,
    }


def verify_inventory(root: Path, manifest_name: str = "artifact_manifest.json") -> dict[str, Any]:
    manifest_path = root / manifest_name
    manifest = read_json(manifest_path)
    require(manifest.get("status") == "complete", f"Incomplete manifest: {manifest_path}")
    files = manifest.get("files")
    require(isinstance(files, list) and files, f"Empty manifest: {manifest_path}")
    require(
        manifest.get("file_count_excluding_manifest") == len(files),
        f"Manifest count mismatch: {manifest_path}",
    )
    seen: set[str] = set()
    for item in files:
        relative = item.get("path")
        require(isinstance(relative, str) and relative, "Invalid manifest path")
        relative_path = Path(relative)
        require(not relative_path.is_absolute() and ".." not in relative_path.parts, "Unsafe path")
        normalized = relative_path.as_posix()
        require(normalized not in seen, f"Duplicate manifest path: {normalized}")
        seen.add(normalized)
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


def manifest_index(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["path"]: item for item in manifest["files"]}


def verify_manifest_file(root: Path, index: dict[str, dict[str, Any]], relative: str) -> Path:
    require(relative in index, f"Artifact is not recorded in the source manifest: {relative}")
    path = root / relative
    item = index[relative]
    require(path.is_file(), f"Recorded artifact is missing: {path}")
    require(path.stat().st_size == item["size_bytes"], f"Recorded size differs: {path}")
    require(sha256_file(path) == item["sha256"], f"Recorded hash differs: {path}")
    return path


def soften(probabilities: np.ndarray, temperature: float = TEMPERATURE) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    require(probabilities.ndim == 2, "Probability array must be two-dimensional")
    require(np.isfinite(probabilities).all(), "Probability array contains non-finite values")
    require(np.all(probabilities >= 0.0), "Probability array contains negative values")
    row_sums = probabilities.sum(axis=1)
    require(
        np.allclose(row_sums, 1.0, rtol=0.0, atol=PROBABILITY_SUM_ATOL),
        "Probability rows do not sum to one",
    )
    clipped = np.maximum(probabilities, PROBABILITY_FLOOR)
    logits = np.log(clipped) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    exponentials = np.exp(logits)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def probability_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame.columns if column.startswith("probability_")]
    require(len(columns) == len(CLASS_NAMES), "Prediction file does not contain five probabilities")
    return columns


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    frame = pd.read_csv(path)
    required = {"source_row_index", "true_label", "predicted_label"}
    require(required.issubset(frame.columns), f"Prediction schema mismatch: {path}")
    columns = probability_columns(frame)
    probabilities = frame[columns].to_numpy(dtype=np.float64)
    require(len(frame) == EXPECTED_TEST_ROWS, f"Prediction row count mismatch: {path}")
    require(np.isfinite(probabilities).all(), f"Non-finite probabilities: {path}")
    require(np.all(probabilities >= 0.0), f"Negative probabilities: {path}")
    require(
        np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=PROBABILITY_SUM_ATOL),
        f"Probability sum mismatch: {path}",
    )
    predicted = frame["predicted_label"].to_numpy(dtype=np.int64)
    require(
        np.array_equal(predicted, probabilities.argmax(axis=1)),
        f"Predicted labels do not match argmax: {path}",
    )
    return {
        "source_indices": frame["source_row_index"].to_numpy(dtype=np.int64),
        "true_labels": frame["true_label"].to_numpy(dtype=np.int64),
        "predicted_labels": predicted,
        "probabilities_t1": probabilities,
    }


def load_torch_mapping(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    require(isinstance(value, dict), f"Expected a mapping in checkpoint: {path}")
    return value


def state_content_sha256(state: dict[str, torch.Tensor]) -> str:
    require(state, "State dictionary is empty")
    require(
        all(isinstance(value, torch.Tensor) for value in state.values()),
        "State dictionary contains a non-tensor value",
    )
    return sha256_arrays(
        *[state[name].detach().cpu().numpy() for name in sorted(state)]
    )


def checkpoint_probabilities(
    checkpoint_path: Path,
    hidden_dims: tuple[int, int],
    X_test: np.ndarray,
) -> tuple[dict[str, np.ndarray], str]:
    state = load_torch_mapping(checkpoint_path)
    if "state_dict" in state:
        nested = state["state_dict"]
        require(isinstance(nested, dict), f"Invalid nested state: {checkpoint_path}")
        state = nested
    model = StudentMLP(17, hidden_dims, len(CLASS_NAMES))
    model.load_state_dict(state, strict=True)
    model.eval()
    tensor = torch.from_numpy(np.asarray(X_test, dtype=np.float32))
    t1_chunks: list[np.ndarray] = []
    t4_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(tensor), 4096):
            logits = model(tensor[start : start + 4096])
            t1_chunks.append(F.softmax(logits, dim=1).cpu().numpy())
            t4_chunks.append(
                F.softmax(logits.to(torch.float64) / TEMPERATURE, dim=1)
                .cpu()
                .numpy()
            )
    probabilities_t1 = np.concatenate(t1_chunks).astype(np.float64, copy=False)
    probabilities_t4 = np.concatenate(t4_chunks).astype(np.float64, copy=False)
    for name, values in (
        ("T1", probabilities_t1),
        ("T4", probabilities_t4),
    ):
        require(values.shape == (EXPECTED_TEST_ROWS, len(CLASS_NAMES)), f"{name} shape differs")
        require(np.isfinite(values).all(), f"{name} probabilities are non-finite")
        require(
            np.allclose(
                values.sum(axis=1),
                1.0,
                rtol=0.0,
                atol=PROBABILITY_SUM_ATOL,
            ),
            f"{name} probability rows do not sum to one",
        )
    return {
        "probabilities_t1": probabilities_t1,
        "probabilities_t4": probabilities_t4,
        "predicted_labels": probabilities_t1.argmax(axis=1).astype(np.int64),
    }, state_content_sha256(state)


def validate_student_lineage(
    run_root: Path,
    source_index: dict[str, dict[str, Any]],
    completion: dict[str, Any],
    seed: int,
    student: str,
    route: str,
    files: dict[str, str],
    X_test: np.ndarray,
    saved_predictions: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], list[dict[str, str]], dict[str, Any]]:
    seed_prefix = f"seed_{seed}"
    result_key = f"{student}_{route}"
    result = completion["student_results"][result_key]
    require(result.get("route") == route, f"Route metadata differs: {result_key}")
    expected_names = {
        "plain_state_dict": files["checkpoint"],
        "rich_artifact": files["artifact"],
        "test_predictions": files["predictions"],
    }
    for key, expected in expected_names.items():
        require(result.get(key) == expected, f"Artifact name differs: {result_key}:{key}")

    checkpoint_relative = f"{seed_prefix}/{files['checkpoint']}"
    artifact_relative = f"{seed_prefix}/{files['artifact']}"
    checkpoint_path = verify_manifest_file(run_root, source_index, checkpoint_relative)
    artifact_path = verify_manifest_file(run_root, source_index, artifact_relative)
    require(
        sha256_file(checkpoint_path) == result["plain_state_dict_sha256"],
        f"Checkpoint hash differs: {result_key}",
    )
    require(
        sha256_file(artifact_path) == result["rich_artifact_sha256"],
        f"Rich-artifact hash differs: {result_key}",
    )
    replay, trained_state_hash = checkpoint_probabilities(
        checkpoint_path, STUDENT_SPECS[student], X_test
    )
    require(
        trained_state_hash == result["trained_state_sha256"],
        f"Trained-state hash differs: {result_key}",
    )
    maximum_t1_delta = float(
        np.max(np.abs(replay["probabilities_t1"] - saved_predictions["probabilities_t1"]))
    )
    require(
        maximum_t1_delta <= CHECKPOINT_REPLAY_ATOL,
        f"Checkpoint T1 replay differs: {seed}:{result_key}",
    )
    require(
        np.array_equal(replay["predicted_labels"], saved_predictions["predicted_labels"]),
        f"Checkpoint predictions differ: {result_key}",
    )

    rich = load_torch_mapping(artifact_path)
    required_rich = {
        "protocol_id": SOURCE_PROTOCOL_ID,
        "seed": seed,
        "student": student,
        "route": route,
        "input_dim": 17,
        "hidden_dims": list(STUDENT_SPECS[student]),
        "num_classes": len(CLASS_NAMES),
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "scaler_sha256": EXPECTED_SCALER_SHA256,
        "initial_state_sha256": result["initial_state_sha256"],
        "trained_state_sha256": trained_state_hash,
        "kd_hyperparameters": ({"T": 4.0, "alpha": 0.7} if route == "rf_kd" else None),
    }
    for key, expected in required_rich.items():
        require(rich.get(key) == expected, f"Rich artifact differs: {result_key}:{key}")
    rich_state = rich.get("state_dict")
    require(isinstance(rich_state, dict), f"Rich state is missing: {result_key}")
    require(
        state_content_sha256(rich_state) == trained_state_hash,
        f"Rich and plain states differ: {result_key}",
    )
    provenance = rich.get("teacher_soft_target_provenance")
    if route == "scratch":
        require(provenance is None, f"Scratch has teacher provenance: {result_key}")
    else:
        require(
            provenance == completion["teacher_soft_target_provenance"],
            f"RF-KD teacher provenance differs: {result_key}",
        )
        require(provenance.get("rf_seed") == seed, f"RF seed differs: {result_key}")
        require(provenance.get("rf_config") == RF_CONFIG, f"RF config differs: {result_key}")
        calibration = provenance.get("calibration_audit", {})
        require(
            calibration.get("strategy") == "stratified_group_kfold"
            and calibration.get("folds") == RF_CONFIG["calibration_cv"]
            and calibration.get("group_overlap_per_fold") == [0, 0, 0],
            f"RF calibration provenance differs: {result_key}",
        )
        train_probability_path = verify_manifest_file(
            run_root, source_index, f"{seed_prefix}/rf_train_probabilities.npy"
        )
        train_probabilities = np.load(train_probability_path, allow_pickle=False)
        require(
            sha256_arrays(train_probabilities)
            == provenance["train_probability_content_sha256"],
            f"RF train-probability provenance differs: {result_key}",
        )

    artifacts = [
        {"path": repo_path(checkpoint_path), "sha256": sha256_file(checkpoint_path)},
        {"path": repo_path(artifact_path), "sha256": sha256_file(artifact_path)},
    ]
    if route == "rf_kd":
        artifacts.append(
            {
                "path": repo_path(train_probability_path),
                "sha256": sha256_file(train_probability_path),
            }
        )
    diagnostics = {
        "checkpoint_t1_maximum_absolute_probability_delta": maximum_t1_delta,
        "checkpoint_t4_source": "direct softmax(checkpoint logits / 4)",
        "checkpoint_t4_computation_dtype": "float64 from original float32 logits",
        "trained_state_sha256": trained_state_hash,
    }
    return replay, artifacts, diagnostics


def exact_group_context(group_ids: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    unique_groups, inverse, counts = np.unique(
        group_ids, return_inverse=True, return_counts=True
    )
    require(len(group_ids) == EXPECTED_TEST_ROWS, "Test group vector length mismatch")
    require(len(unique_groups) == EXPECTED_TEST_GROUPS, "Exact test group count mismatch")
    label_min = np.full(len(unique_groups), len(CLASS_NAMES), dtype=np.int64)
    label_max = np.full(len(unique_groups), -1, dtype=np.int64)
    np.minimum.at(label_min, inverse, labels)
    np.maximum.at(label_max, inverse, labels)
    mixed = label_min != label_max
    require(not mixed.any(), "The finalized test partition contains mixed-label groups")
    inverse_size_weights = 1.0 / counts[inverse].astype(np.float64)
    return {
        "unique_groups": unique_groups,
        "inverse": inverse,
        "counts": counts,
        "row_weights": np.ones(len(group_ids), dtype=np.float64),
        "group_balanced_weights": inverse_size_weights,
        "mixed_label_groups": int(mixed.sum()),
    }


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    require(values.shape == weights.shape, "Weighted mean shape mismatch")
    require(np.isfinite(values).all() and np.isfinite(weights).all(), "Non-finite weighted mean")
    require(np.all(weights >= 0.0) and weights.sum() > 0.0, "Invalid weights")
    numerator = math.fsum(
        float(value) * float(weight)
        for value, weight in zip(values, weights, strict=True)
    )
    denominator = math.fsum(float(weight) for weight in weights)
    return numerator / denominator


def top2_overlap(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_top = np.argsort(-left, axis=1, kind="stable")[:, :2]
    right_top = np.argsort(-right, axis=1, kind="stable")[:, :2]
    overlap = (
        (left_top[:, :, None] == right_top[:, None, :]).any(axis=2).sum(axis=1)
    )
    return overlap.astype(np.float64) / 2.0


def top_margin(probabilities: np.ndarray) -> np.ndarray:
    top = np.partition(probabilities, -2, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def per_row_metrics(
    teacher_t1: np.ndarray,
    student_t1: np.ndarray,
    teacher_t4: np.ndarray,
    student_t4: np.ndarray,
) -> dict[str, np.ndarray]:
    for name, values in (
        ("teacher_t1", teacher_t1),
        ("student_t1", student_t1),
        ("teacher_t4", teacher_t4),
        ("student_t4", student_t4),
    ):
        require(
            values.shape == (EXPECTED_TEST_ROWS, len(CLASS_NAMES)),
            f"{name} shape differs",
        )
        require(np.isfinite(values).all() and np.all(values >= 0.0), f"{name} is invalid")
    require(np.all(teacher_t4 > 0.0) and np.all(student_t4 > 0.0), "T4 values must be positive")
    midpoint = 0.5 * (teacher_t4 + student_t4)
    kl = np.sum(
        teacher_t4 * (np.log(teacher_t4) - np.log(student_t4)), axis=1
    )
    js = 0.5 * np.sum(
        teacher_t4 * (np.log(teacher_t4) - np.log(midpoint)), axis=1
    ) + 0.5 * np.sum(
        student_t4 * (np.log(student_t4) - np.log(midpoint)), axis=1
    )
    teacher_pred = teacher_t1.argmax(axis=1)
    student_pred = student_t1.argmax(axis=1)
    return {
        "kl_teacher_to_student_T4": kl,
        "js_T4": js,
        "l1_T4": np.abs(teacher_t4 - student_t4).sum(axis=1),
        "hard_agreement_T1": (teacher_pred == student_pred).astype(np.float64),
        "top2_overlap_T1": top2_overlap(teacher_t1, student_t1),
        "margin_abs_error_T1": np.abs(top_margin(teacher_t1) - top_margin(student_t1)),
    }


def within_group_probability_delta(
    probabilities: np.ndarray, inverse: np.ndarray, group_count: int
) -> float:
    maximum = 0.0
    for class_index in range(probabilities.shape[1]):
        minima = np.full(group_count, np.inf, dtype=np.float64)
        maxima = np.full(group_count, -np.inf, dtype=np.float64)
        np.minimum.at(minima, inverse, probabilities[:, class_index])
        np.maximum.at(maxima, inverse, probabilities[:, class_index])
        maximum = max(maximum, float(np.max(maxima - minima)))
    return maximum


def route_metrics(
    teacher: dict[str, np.ndarray],
    student: dict[str, np.ndarray],
    group_context: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    teacher_probs = teacher["probabilities_t1"]
    student_probs = student["probabilities_t1"]
    teacher_t4 = teacher["probabilities_t4"]
    student_t4 = student["probabilities_t4"]
    labels = teacher["true_labels"]
    rows = per_row_metrics(teacher_probs, student_probs, teacher_t4, student_t4)
    teacher_pred = teacher["predicted_labels"]
    student_pred = student["predicted_labels"]
    teacher_correct = teacher_pred == labels
    student_correct = student_pred == labels
    rows["teacher_correct_student_wrong_rate"] = (
        teacher_correct & ~student_correct
    ).astype(np.float64)
    rows["teacher_wrong_student_correct_rate"] = (
        ~teacher_correct & student_correct
    ).astype(np.float64)

    aggregate_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    for view, weights in (
        ("row_weighted", group_context["row_weights"]),
        ("exact_group_balanced", group_context["group_balanced_weights"]),
    ):
        aggregate: dict[str, Any] = {"weighting": view}
        for metric, values in rows.items():
            aggregate[metric] = weighted_mean(values, weights)
        class_hard = []
        class_kl = []
        for class_index, class_name in enumerate(CLASS_NAMES):
            mask = labels == class_index
            require(mask.any(), f"Test class is empty: {class_name}")
            class_record: dict[str, Any] = {
                "weighting": view,
                "class_index": class_index,
                "class_name": class_name,
                "support_rows": int(mask.sum()),
                "exact_group_count": int(
                    np.unique(group_context["inverse"][mask]).size
                ),
            }
            for metric, values in rows.items():
                class_record[metric] = weighted_mean(values[mask], weights[mask])
            class_rows.append(class_record)
            class_hard.append(class_record["hard_agreement_T1"])
            class_kl.append(class_record["kl_teacher_to_student_T4"])
        aggregate["macro_class_hard_agreement_T1"] = float(np.mean(class_hard))
        aggregate["macro_class_kl_T4"] = float(np.mean(class_kl))
        aggregate_rows.append(aggregate)

    teacher_margin = top_margin(teacher_probs)
    student_margin = top_margin(student_probs)
    spearman = stats.spearmanr(teacher_margin, student_margin)
    diagnostics = {
        "margin_spearman_rho_T1_row_weighted": float(spearman.statistic),
        "teacher_accuracy": float(teacher_correct.mean()),
        "student_accuracy": float(student_correct.mean()),
        "student_teacher_hard_disagreements": int(np.count_nonzero(teacher_pred != student_pred)),
        "max_within_exact_group_probability_delta": within_group_probability_delta(
            student_probs,
            group_context["inverse"],
            len(group_context["unique_groups"]),
        ),
    }
    require(
        diagnostics["max_within_exact_group_probability_delta"]
        <= WITHIN_GROUP_PROBABILITY_ATOL,
        "Student probabilities differ within an exact feature group",
    )
    return aggregate_rows, class_rows, diagnostics


def t_interval_95(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    require(values.shape == (len(EXPECTED_SEEDS),), "Ten paired values are required")
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / math.sqrt(len(values)))
    half_width = float(stats.t.ppf(0.975, len(values) - 1) * standard_error)
    return [mean - half_width, mean + half_width]


def paired_summary(
    metrics: pd.DataFrame,
    student: str,
    weighting: str,
    metric: str,
) -> dict[str, Any]:
    subset = metrics[
        (metrics["student"] == student) & (metrics["weighting"] == weighting)
    ]
    scratch = (
        subset[subset["route"] == "scratch"].set_index("seed")[metric].loc[EXPECTED_SEEDS]
    )
    kd = subset[subset["route"] == "rf_kd"].set_index("seed")[metric].loc[EXPECTED_SEEDS]
    if metric in DISTANCE_METRICS:
        differences = scratch.to_numpy(dtype=np.float64) - kd.to_numpy(dtype=np.float64)
        orientation = "scratch_minus_rf_kd; positive means RF-KD is closer to teacher"
    elif metric in SIMILARITY_METRICS:
        differences = kd.to_numpy(dtype=np.float64) - scratch.to_numpy(dtype=np.float64)
        orientation = "rf_kd_minus_scratch; positive means RF-KD is closer to teacher"
    else:
        raise RuntimeError(f"No paired-test orientation is defined for {metric}")
    test = exact_paired_wilcoxon(differences)
    return {
        "student": student,
        "weighting": weighting,
        "metric": metric,
        "orientation": orientation,
        "scratch_values": scratch.to_list(),
        "rf_kd_values": kd.to_list(),
        "transfer_gain_values": differences.tolist(),
        "transfer_gain_mean": float(differences.mean()),
        "transfer_gain_sample_std": float(differences.std(ddof=1)),
        "transfer_gain_median": float(np.median(differences)),
        "transfer_gain_95_percent_t_interval": t_interval_95(differences),
        "positive_seed_count": int(np.count_nonzero(differences > 0.0)),
        "negative_seed_count": int(np.count_nonzero(differences < 0.0)),
        "zero_seed_count": int(np.count_nonzero(differences == 0.0)),
        "exact_paired_wilcoxon": test,
    }


def validate_source_contract(run_root: Path, dataset_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = verify_inventory(run_root)
    require(manifest.get("protocol_id") == SOURCE_PROTOCOL_ID, "Source protocol mismatch")
    index = manifest_index(manifest)
    execution = read_json(
        verify_manifest_file(run_root, index, "execution_contract.json")
    )
    preprocessing = read_json(
        verify_manifest_file(run_root, index, "preprocessing_contract.json")
    )
    require(execution.get("protocol_id") == SOURCE_PROTOCOL_ID, "Execution protocol mismatch")
    require(execution.get("seeds") == EXPECTED_SEEDS, "Source seed list mismatch")
    require(execution.get("dataset_sha256") == EXPECTED_DATASET_SHA256, "Dataset hash mismatch")
    require(execution.get("split_indices_sha256") == EXPECTED_SPLIT_SHA256, "Split hash mismatch")
    require(execution.get("scaler_sha256") == EXPECTED_SCALER_SHA256, "Scaler hash mismatch")
    require(preprocessing.get("scaler_fit_partition") == "train only", "Scaler is not train only")
    require(
        preprocessing.get("split_sizes")
        == {"train": 262_197, "validation": 56_163, "test": EXPECTED_TEST_ROWS},
        "Split sizes mismatch",
    )
    overlap = preprocessing.get("feature_overlap_audit", {})
    for key in (
        "train_validation_feature_overlap",
        "train_test_feature_overlap",
        "validation_test_feature_overlap",
    ):
        require(overlap.get(key) == 0, f"Nonzero feature-group overlap: {key}")
    require(dataset_path.is_file(), f"Dataset is missing: {dataset_path}")
    require(sha256_file(dataset_path) == EXPECTED_DATASET_SHA256, "Local dataset bytes differ")
    return {"execution": execution, "preprocessing": preprocessing, "manifest": manifest}, index


def analyze(run_root: Path, dataset_path: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    configure_deterministic_inference()
    source, index = validate_source_contract(run_root, dataset_path)
    dataset = load_wsnds(dataset_path)
    require(dataset["dataset_sha256"] == EXPECTED_DATASET_SHA256, "Loaded dataset hash mismatch")
    split = feature_group_split(dataset["features"], dataset["labels"])
    scaled, scaler = apply_train_scaler(split)
    reconstructed_split_hash = sha256_arrays(
        split["train_indices"], split["validation_indices"], split["test_indices"]
    )
    reconstructed_scaler_hash = sha256_arrays(
        np.asarray(scaler.mean_, dtype=np.float64),
        np.asarray(scaler.scale_, dtype=np.float64),
        np.asarray(scaler.var_, dtype=np.float64),
    )
    require(reconstructed_split_hash == EXPECTED_SPLIT_SHA256, "Reconstructed split differs")
    require(reconstructed_scaler_hash == EXPECTED_SCALER_SHA256, "Reconstructed scaler differs")
    indices_path = verify_manifest_file(run_root, index, "split_indices.npz")
    with np.load(indices_path, allow_pickle=False) as saved:
        require(
            set(saved.files) == {"train_indices", "validation_indices", "test_indices"},
            "Split-index schema differs",
        )
        require(np.array_equal(saved["train_indices"], split["train_indices"]), "Train indices differ")
        require(
            np.array_equal(saved["validation_indices"], split["validation_indices"]),
            "Validation indices differ",
        )
        test_indices = np.asarray(saved["test_indices"], dtype=np.int64)
    require(np.array_equal(test_indices, split["test_indices"]), "Test indices differ")
    require(test_indices.shape == (EXPECTED_TEST_ROWS,), "Test index shape mismatch")
    test_features = dataset["features"][test_indices]
    test_labels = dataset["labels"][test_indices]
    X_test = scaled["X_test"]
    group_ids = verified_feature_hashes(test_features)
    groups = exact_group_context(group_ids, test_labels)

    metric_records: list[dict[str, Any]] = []
    class_records: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    teacher_hashes: dict[str, str] = {}
    source_artifacts: list[dict[str, Any]] = []

    for seed in EXPECTED_SEEDS:
        seed_prefix = f"seed_{seed}"
        completion_relative = f"{seed_prefix}/seed_completion.json"
        completion_path = verify_manifest_file(run_root, index, completion_relative)
        completion = read_json(completion_path)
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
            require(completion.get(key) == expected, f"Seed completion differs: {seed}:{key}")
        for student in STUDENTS:
            scratch_initial = completion["student_results"][
                f"{student}_scratch"
            ].get("initial_state_sha256")
            rf_kd_initial = completion["student_results"][
                f"{student}_rf_kd"
            ].get("initial_state_sha256")
            require(
                isinstance(scratch_initial, str)
                and len(scratch_initial) == 64
                and scratch_initial == rf_kd_initial,
                f"Scratch/RF-KD initial states are not paired: {seed}:{student}",
            )
        source_artifacts.append(
            {"path": repo_path(completion_path), "sha256": sha256_file(completion_path)}
        )
        teacher_relative = f"{seed_prefix}/{TEACHER_FILE}"
        teacher_path = verify_manifest_file(run_root, index, teacher_relative)
        teacher = load_predictions(teacher_path)
        teacher["probabilities_t4"] = soften(teacher["probabilities_t1"], TEMPERATURE)
        require(np.array_equal(teacher["source_indices"], test_indices), "Teacher row order mismatch")
        require(np.array_equal(teacher["true_labels"], test_labels), "Teacher labels mismatch")
        teacher_hashes[str(seed)] = sha256_file(teacher_path)
        teacher_group_delta = within_group_probability_delta(
            teacher["probabilities_t1"], groups["inverse"], len(groups["unique_groups"])
        )
        require(
            teacher_group_delta <= WITHIN_GROUP_PROBABILITY_ATOL,
            "Teacher probabilities differ within an exact feature group",
        )
        source_artifacts.append(
            {"path": repo_path(teacher_path), "sha256": teacher_hashes[str(seed)]}
        )
        for student, route_files in STUDENTS.items():
            for route, files in route_files.items():
                relative = f"{seed_prefix}/{files['predictions']}"
                path = verify_manifest_file(run_root, index, relative)
                prediction = load_predictions(path)
                require(
                    np.array_equal(prediction["source_indices"], teacher["source_indices"]),
                    f"Source-row mismatch for {relative}",
                )
                require(
                    np.array_equal(prediction["true_labels"], teacher["true_labels"]),
                    f"Label mismatch for {relative}",
                )
                checkpoint_replay, checkpoint_artifacts, checkpoint_diagnostics = (
                    validate_student_lineage(
                        run_root,
                        index,
                        completion,
                        seed,
                        student,
                        route,
                        files,
                        X_test,
                        prediction,
                    )
                )
                student_payload = {
                    **prediction,
                    "probabilities_t1": checkpoint_replay["probabilities_t1"],
                    "probabilities_t4": checkpoint_replay["probabilities_t4"],
                    "predicted_labels": checkpoint_replay["predicted_labels"],
                }
                aggregates, classes, route_diagnostics = route_metrics(
                    teacher, student_payload, groups
                )
                for aggregate in aggregates:
                    metric_records.append(
                        {"seed": seed, "student": student, "route": route, **aggregate}
                    )
                for class_record in classes:
                    class_records.append(
                        {"seed": seed, "student": student, "route": route, **class_record}
                    )
                diagnostics[f"{seed}:{student}:{route}"] = {
                    **route_diagnostics,
                    **checkpoint_diagnostics,
                }
                source_artifacts.append(
                    {"path": repo_path(path), "sha256": sha256_file(path)}
                )
                source_artifacts.extend(checkpoint_artifacts)

    metrics = pd.DataFrame(metric_records).sort_values(
        ["seed", "student", "route", "weighting"]
    )
    per_class = pd.DataFrame(class_records).sort_values(
        ["seed", "student", "route", "weighting", "class_index"]
    )
    require(len(metrics) == len(EXPECTED_SEEDS) * 2 * 2 * 2, "Metric coverage mismatch")
    require(len(per_class) == len(metrics) * len(CLASS_NAMES), "Class coverage mismatch")

    primary_tests = {
        student: paired_summary(
            metrics, student, "exact_group_balanced", PRIMARY_METRIC
        )
        for student in STUDENTS
    }
    primary_raw = {
        student: value["exact_paired_wilcoxon"]["p_value_two_sided"]
        for student, value in primary_tests.items()
    }
    primary_adjusted = holm_adjust(primary_raw)
    for student, value in primary_tests.items():
        value["holm_adjusted_p_across_two_students"] = primary_adjusted[student]
        value["reject_holm_alpha_0_05"] = primary_adjusted[student] < 0.05

    secondary_tests: dict[str, dict[str, Any]] = {}
    for student in STUDENTS:
        for metric in SECONDARY_TEST_METRICS:
            key = f"{student}:{metric}"
            secondary_tests[key] = paired_summary(
                metrics, student, "exact_group_balanced", metric
            )
    secondary_raw = {
        key: value["exact_paired_wilcoxon"]["p_value_two_sided"]
        for key, value in secondary_tests.items()
    }
    secondary_adjusted = holm_adjust(secondary_raw)
    for key, value in secondary_tests.items():
        value["holm_adjusted_p_across_secondary_family"] = secondary_adjusted[key]
        value["reject_holm_alpha_0_05"] = secondary_adjusted[key] < 0.05

    execution_contract = {
        "protocol_id": PROTOCOL_ID,
        "analysis_status": "post_hoc_secondary_evidence",
        "analysis_status_reason": (
            "The behavioral-transfer question was specified after the finalized "
            "prediction artifacts had been inspected; inferential outputs are therefore "
            "reported as secondary evidence rather than preregistered confirmation."
        ),
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "source_run_root": repo_path(run_root),
        "dataset": {"path": repo_path(dataset_path), "sha256": EXPECTED_DATASET_SHA256},
        "split_indices_sha256": EXPECTED_SPLIT_SHA256,
        "scaler_sha256": EXPECTED_SCALER_SHA256,
        "seeds": EXPECTED_SEEDS,
        "test_rows_per_seed": EXPECTED_TEST_ROWS,
        "test_exact_feature_groups": EXPECTED_TEST_GROUPS,
        "statistical_unit": (
            "paired training-run/model seed on one fixed feature-group-disjoint split; "
            "the same seed also controls RF construction and student training"
        ),
        "primary_metric": {
            "name": PRIMARY_METRIC,
            "weighting": "one total unit per exact raw-feature group",
            "direction": "teacher to student",
            "temperature": TEMPERATURE,
            "teacher_T4_definition": (
                "softmax(log(max(calibrated_RF_probability, 1e-8)) / 4), matching KD"
            ),
            "student_T4_definition": "softmax(saved_checkpoint_logits / 4), matching KD and XAI",
            "student_T4_computation_dtype": (
                "float64 evaluation of the original float32 checkpoint logits to avoid "
                "float32 exponential underflow; no student probability floor"
            ),
            "teacher_probability_floor": PROBABILITY_FLOOR,
            "student_probability_floor": None,
            "checkpoint_T1_replay_absolute_tolerance": CHECKPOINT_REPLAY_ATOL,
            "checkpoint_T1_replay_requirement": (
                "bounded probability drift plus exact hard-label equality across all rows"
            ),
            "comparison": "scratch minus RF-KD; positive means RF-KD is closer",
            "multiplicity": "Holm adjustment across Student A and Student B",
        },
        "secondary_test_metrics": SECONDARY_TEST_METRICS,
        "secondary_multiplicity": "Holm adjustment across ten student-metric tests",
        "row_weighted_sensitivity": True,
        "per_class_results": "descriptive; no class-wise null-hypothesis tests",
        "software_dependencies": {
            "analysis_source_sha256": sha256_file(SCRIPT_PATH),
            "checkpoint_inference_runtime": {
                "device": "cpu",
                "torch_deterministic_algorithms": True,
                "torch_intraop_threads": 1,
            },
            "weighted_reduction": (
                "fixed row order with Python math.fsum for numerator and denominator"
            ),
            "statistics_source": repo_path(
                REPO_ROOT
                / "experiments/wsnds/evidence_completion/analyze_fgds_group_balanced_routes.py"
            ),
            "statistics_source_sha256": sha256_file(
                REPO_ROOT
                / "experiments/wsnds/evidence_completion/analyze_fgds_group_balanced_routes.py"
            ),
            "common_source": repo_path(
                REPO_ROOT / "experiments/wsnds/leakage_free_rerun/tier15_common.py"
            ),
            "common_source_sha256": sha256_file(
                REPO_ROOT / "experiments/wsnds/leakage_free_rerun/tier15_common.py"
            ),
        },
        "claim_boundary": (
            "This analysis tests held-out in-distribution response-distribution transfer "
            "from the calibrated RF to each checkpoint-reconstructed student and compares "
            "RF-KD with matched scratch under the same T=4 output contract used by KD and XAI. "
            "It does not establish causal mechanism transfer, off-manifold decision-boundary "
            "equivalence, explanation transfer, or deployment fidelity."
        ),
    }
    execution_contract["execution_contract_id"] = canonical_json_sha256(execution_contract)

    result = {
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "execution_contract": execution_contract,
        "source_contract": {
            "source_execution_contract_sha256": sha256_file(
                run_root / "execution_contract.json"
            ),
            "source_preprocessing_contract_sha256": sha256_file(
                run_root / "preprocessing_contract.json"
            ),
            "source_artifact_manifest_sha256": sha256_file(
                run_root / "artifact_manifest.json"
            ),
            "teacher_prediction_sha256_by_seed": teacher_hashes,
            "validated_source_artifacts": sorted(
                {item["path"]: item for item in source_artifacts}.values(),
                key=lambda item: item["path"],
            ),
        },
        "test_group_audit": {
            "rows": EXPECTED_TEST_ROWS,
            "exact_feature_groups": len(groups["unique_groups"]),
            "repeated_rows_beyond_one_per_group": EXPECTED_TEST_ROWS
            - len(groups["unique_groups"]),
            "mixed_label_groups": groups["mixed_label_groups"],
            "maximum_group_size": int(groups["counts"].max()),
        },
        "primary_tests": primary_tests,
        "secondary_tests": secondary_tests,
        "route_diagnostics": diagnostics,
        "interpretation": (
            "Positive transfer gain means RF-KD reproduces the RF response distribution "
            "more closely than the matched scratch model under checkpoint-logit T=4 replay. "
            "Classification macro-F1 is a separate outcome and is not reinterpreted by this "
            "transfer analysis."
        ),
        "claim_boundary": execution_contract["claim_boundary"],
    }
    return result, metrics, per_class


def markdown_summary(result: dict[str, Any]) -> str:
    lines = [
        "# FG-DS Checkpoint-Logit Behavioral-Transfer Audit",
        "",
        "This additive analysis reconstructs the finalized ten-seed student checkpoints on the fixed feature-group-disjoint WSN-DS test partition. It does not retrain a model.",
        "",
        "## Primary Metric",
        "",
        "The primary metric is exact-feature-group-balanced KL divergence from the calibrated RF distribution to direct checkpoint-logit student outputs at T=4. Positive transfer gain means scratch KL minus RF-KD KL is positive.",
        "",
        "| Student | Scratch KL | RF-KD KL | Transfer gain | 95% t interval | Exact p | Holm p | Positive seeds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for student in STUDENTS:
        row = result["primary_tests"][student]
        scratch_mean = float(np.mean(row["scratch_values"]))
        kd_mean = float(np.mean(row["rf_kd_values"]))
        interval = row["transfer_gain_95_percent_t_interval"]
        lines.append(
            f"| {student.replace('_', ' ').title()} | {scratch_mean:.6f} | "
            f"{kd_mean:.6f} | {row['transfer_gain_mean']:.6f} | "
            f"[{interval[0]:.6f}, {interval[1]:.6f}] | "
            f"{row['exact_paired_wilcoxon']['p_value_two_sided']:.6f} | "
            f"{row['holm_adjusted_p_across_two_students']:.6f} | "
            f"{row['positive_seed_count']}/{len(EXPECTED_SEEDS)} |"
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            result["claim_boundary"],
            "",
            "The inference unit is the paired training-run/model seed on one fixed clean split. Per-class outputs and row-weighted values are descriptive sensitivities.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    output_dir: Path,
    result: dict[str, Any],
    metrics: pd.DataFrame,
    per_class: pd.DataFrame,
) -> None:
    require(not output_dir.exists(), f"Refusing to overwrite existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp.", dir=output_dir.parent)
    )
    try:
        atomic_write_json(staging / "execution_contract.json", result["execution_contract"])
        atomic_write_json(staging / "behavioral_transfer_summary.json", result)
        atomic_write_csv(staging / "behavioral_transfer_seed_metrics.csv", metrics)
        atomic_write_csv(staging / "behavioral_transfer_per_class.csv", per_class)
        atomic_write_text(staging / "BEHAVIORAL_TRANSFER_SUMMARY.md", markdown_summary(result))
        shutil.copy2(SCRIPT_PATH, staging / "executed_analysis_source.py")
        manifest = build_inventory(staging, PROTOCOL_ID, "complete")
        atomic_write_json(staging / "artifact_manifest.json", manifest)
        verify_inventory(staging)
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_existing(output_dir: Path) -> dict[str, Any]:
    manifest = verify_inventory(output_dir)
    require(manifest.get("protocol_id") == PROTOCOL_ID, "Output protocol mismatch")
    contract = read_json(output_dir / "execution_contract.json")
    result = read_json(output_dir / "behavioral_transfer_summary.json")
    require(contract.get("protocol_id") == PROTOCOL_ID, "Contract protocol mismatch")
    require(result.get("status") == "complete", "Behavioral-transfer result is incomplete")
    require(
        result.get("execution_contract", {}).get("execution_contract_id")
        == contract.get("execution_contract_id"),
        "Persisted execution contracts differ",
    )
    require(
        sha256_file(output_dir / "executed_analysis_source.py")
        == contract["software_dependencies"]["analysis_source_sha256"],
        "Executed source snapshot differs from its contract",
    )
    require(
        sha256_file(SCRIPT_PATH)
        == contract["software_dependencies"]["analysis_source_sha256"],
        "Live verifier source differs from the sealed analysis source",
    )
    metrics = pd.read_csv(output_dir / "behavioral_transfer_seed_metrics.csv")
    per_class = pd.read_csv(output_dir / "behavioral_transfer_per_class.csv")
    require(len(metrics) == len(EXPECTED_SEEDS) * 2 * 2 * 2, "Persisted metric coverage mismatch")
    require(len(per_class) == len(metrics) * len(CLASS_NAMES), "Persisted class coverage mismatch")
    source_root = (REPO_ROOT / contract["source_run_root"]).resolve()
    dataset_path = (REPO_ROOT / contract["dataset"]["path"]).resolve()
    recomputed_result, recomputed_metrics, recomputed_per_class = analyze(
        source_root, dataset_path
    )
    require(
        canonical_json_sha256(result) == canonical_json_sha256(recomputed_result),
        "Persisted summary differs from source-artifact recomputation",
    )
    try:
        pd.testing.assert_frame_equal(
            metrics.reset_index(drop=True),
            recomputed_metrics.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=1e-15,
        )
        pd.testing.assert_frame_equal(
            per_class.reset_index(drop=True),
            recomputed_per_class.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=1e-15,
        )
    except AssertionError as error:
        raise RuntimeError("Persisted behavioral tables differ from recomputation") from error
    require(
        (output_dir / "BEHAVIORAL_TRANSFER_SUMMARY.md").read_text(encoding="utf-8")
        == markdown_summary(recomputed_result),
        "Persisted Markdown summary differs from recomputation",
    )
    return {
        "status": "verified",
        "protocol_id": PROTOCOL_ID,
        "verified_files": manifest["file_count_excluding_manifest"],
        "execution_contract_id": contract["execution_contract_id"],
        "verified_seeds": len(EXPECTED_SEEDS),
        "verification_mode": "full source-artifact recomputation",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.verify_existing:
        print(json.dumps(verify_existing(args.output_dir.resolve()), indent=2))
        return 0
    result, metrics, per_class = analyze(
        args.run_root.resolve(), args.dataset.resolve()
    )
    write_outputs(args.output_dir.resolve(), result, metrics, per_class)
    print(args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
