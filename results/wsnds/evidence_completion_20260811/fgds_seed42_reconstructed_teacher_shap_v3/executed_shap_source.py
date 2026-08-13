"""Explain an output-validated seed-42 RF reconstruction and RF-KD students.

The calibrated RF is deterministically rebuilt from the immutable FG-DS split.
Its train probabilities and full-test probabilities must match the preserved
seed-42 artifacts before SHAP begins. This does not establish equivalence on
the synthetic masked inputs evaluated by permutation SHAP, so the teacher
explanations are attributed only to the validated reconstruction. The audit
separately explains FP32 deployment-source probabilities and the T=4 softened
probabilities used by KD. Within each replicate, all three subjects use the
same model-agnostic permutation SHAP estimator, background rows, explained
rows, and permutation seed.

SHAP 0.48's tabular masker uses ``np.isclose`` to decide whether a changed
background-row feature can reuse a previous model output. That optimization is
unsafe for threshold models because near-but-not-equal values can fall on
opposite sides of a tree split. This audit therefore uses exact equality for
the cache decision while retaining the standard independent-background masking
distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import scipy
import shap
import sklearn
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

try:
    from ..leakage_free_rerun.run_feature_group_10seed_confirmation import (
        fit_calibrated_rf,
    )
    from ..leakage_free_rerun.tier15_common import (
        CLASS_NAMES,
        STUDENT_SPECS,
        StudentMLP,
        apply_train_scaler,
        artifact_manifest,
        atomic_write_json,
        classification_metrics,
        feature_group_split,
        load_wsnds,
        set_seed,
        sha256_arrays,
        sha256_file,
        verified_feature_hashes,
    )
except ImportError:
    from experiments.wsnds.leakage_free_rerun.run_feature_group_10seed_confirmation import (
        fit_calibrated_rf,
    )
    from experiments.wsnds.leakage_free_rerun.tier15_common import (
        CLASS_NAMES,
        STUDENT_SPECS,
        StudentMLP,
        apply_train_scaler,
        artifact_manifest,
        atomic_write_json,
        classification_metrics,
        feature_group_split,
        load_wsnds,
        set_seed,
        sha256_arrays,
        sha256_file,
        verified_feature_hashes,
    )


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
DATASET = REPO_ROOT / "data" / "wsnds" / "WSN-DS.csv"
DEPLOYMENT_ROOT = (
    REPO_ROOT
    / "results/wsnds/confirmation_runs_v2"
    / "remote_winterfell_feature_group_5seed_20260805"
    / "feature_group_5seed"
)
SEED_ROOT = DEPLOYMENT_ROOT / "seed_42"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results/wsnds/evidence_completion_20260811"
    / "fgds_seed42_reconstructed_teacher_shap_v3"
)
PROTOCOL_ID = "wsnds_fgds_seed42_reconstructed_teacher_permutation_shap_v3"
KD_TEMPERATURE = 4.0
FEATURE_NAMES = [
    "Time",
    "Is_CH",
    "who CH",
    "Dist_To_CH",
    "ADV_S",
    "ADV_R",
    "JOIN_S",
    "JOIN_R",
    "SCH_S",
    "SCH_R",
    "Rank",
    "DATA_S",
    "DATA_R",
    "Data_Sent_To_BS",
    "dist_CH_To_BS",
    "send_code",
    "Expaned Energy",
]
EXPECTED_TRAIN_PROBABILITY_CONTENT_SHA256 = (
    "809755ca6ec3e8e317648e08947e41cc5f0fbcb1377a3e272d283a050888e452"
)
EXPECTED_TRAIN_PROBABILITY_FILE_SHA256 = (
    "df4c961c8510296336425639e86d0c862a25629bcb99267a42729637f5a127bf"
)
EXPECTED_DEPLOYMENT_PROTOCOL_ID = "wsnds_feature_group_split_train_only_scaler_5seed_v1"
LOCAL_ACCURACY_ATOL = 1e-6
STUDENT_CROSS_PLATFORM_REPLAY_ATOL = 1e-6
SOURCE_SNAPSHOT_NAMES = {
    "executed_shap_source.py",
    "bound_common_source.py",
    "bound_rf_fitter_source.py",
}
BASE_OUTPUT_NAMES = {
    "execution_contract.json",
    "sampling_contract.npz",
    "preflight_verification.json",
    "reconstructed_calibrated_rf_teacher_seed42.joblib",
    "shap_report.json",
    "artifact_manifest.json",
} | SOURCE_SNAPSHOT_NAMES


class ExactInvariantIndependentMasker(shap.maskers.Independent):
    """Independent masker with threshold-safe, exact cache invalidation."""

    def invariants(self, x: np.ndarray) -> np.ndarray:
        value = np.asarray(x)
        if value.shape != self.data.shape[1:]:
            raise ValueError(
                "The explained row does not match the independent background shape"
            )
        return np.equal(value, self.data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-csv", type=Path, default=DATASET)
    parser.add_argument("--deployment-root", type=Path, default=DEPLOYMENT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--background-size", type=int, default=32)
    parser.add_argument("--explain-size", type=int, default=500)
    parser.add_argument("--estimator-replicates", type=int, default=3)
    parser.add_argument("--permutation-repeats", type=int, default=5)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confirm-explanations", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def atomic_joblib_dump(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary, compress=3)
    os.replace(temporary, path)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
    os.replace(temporary, path)


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def array_contract(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        name: {
            "shape": list(np.asarray(value).shape),
            "dtype": str(np.asarray(value).dtype),
            "content_sha256": sha256_arrays(np.asarray(value)),
        }
        for name, value in sorted(arrays.items())
    }


def verify_npz_exact(path: Path, expected: dict[str, np.ndarray]) -> str:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != set(expected):
            raise RuntimeError(f"NPZ schema differs: {path}")
        for name, expected_value in expected.items():
            observed = payload[name]
            if observed.dtype != expected_value.dtype or not np.array_equal(
                observed, expected_value
            ):
                raise RuntimeError(f"NPZ array differs for {name}: {path}")
    return sha256_file(path)


def verify_artifact_manifest(
    root: Path,
    manifest_path: Path,
    *,
    expected_status: str = "complete",
    expected_protocol_id: str = EXPECTED_DEPLOYMENT_PROTOCOL_ID,
    expected_files_excluding_manifest: set[str] | None = None,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_id") != expected_protocol_id:
        raise RuntimeError(f"Manifest protocol differs: {manifest_path}")
    if manifest.get("status") != expected_status:
        raise RuntimeError(f"Manifest is incomplete: {manifest_path}")
    entries = manifest.get("files", [])
    declared = {entry["path"]: entry for entry in entries}
    if len(declared) != len(entries):
        raise RuntimeError(f"Manifest contains duplicate paths: {manifest_path}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if set(declared) != actual:
        raise RuntimeError(f"Manifest inventory differs: {manifest_path}")
    if expected_files_excluding_manifest is not None and actual != expected_files_excluding_manifest:
        raise RuntimeError(f"Manifest expected inventory differs: {manifest_path}")
    if manifest.get("file_count_excluding_manifest") != len(actual):
        raise RuntimeError(f"Manifest file count differs: {manifest_path}")
    for relative, entry in declared.items():
        path = root / relative
        if path.stat().st_size != entry["size_bytes"] or sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"Manifest artifact differs: {path}")
    return {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "file_count_excluding_manifest": len(actual),
    }


def state_content_sha256(state: dict[str, torch.Tensor]) -> str:
    return sha256_arrays(
        *[
            state[name].detach().cpu().numpy()
            for name in sorted(state)
        ]
    )


def expected_output_names(args: argparse.Namespace) -> set[str]:
    names = set(BASE_OUTPUT_NAMES)
    conditions = ["fp32_deployment_source_probabilities_T1", "kd_softened_probabilities_T4"]
    subjects = ["teacher", "student_A", "student_B"]
    for condition in conditions:
        for replicate in range(1, args.estimator_replicates + 1):
            for subject in subjects:
                names.add(f"{condition}_replicate_{replicate}_{subject}_shap_values.npz")
    return names


def validate_resume_inventory(output_dir: Path, args: argparse.Namespace) -> None:
    allowed = expected_output_names(args)
    actual = {path.name for path in output_dir.iterdir() if path.is_file()}
    directories = [path for path in output_dir.iterdir() if path.is_dir()]
    if directories:
        raise RuntimeError(f"Unexpected directories in SHAP output: {directories}")
    unknown = sorted(actual - allowed)
    if unknown:
        raise RuntimeError(f"Unexpected files in SHAP output: {unknown}")
    report_exists = "shap_report.json" in actual
    manifest_exists = "artifact_manifest.json" in actual
    if report_exists != manifest_exists:
        raise RuntimeError("Final SHAP report and manifest must either both exist or both be absent")


def stratified_sample(labels: np.ndarray, size: int, seed: int) -> np.ndarray:
    if size < len(CLASS_NAMES):
        raise ValueError("Stratified sample must be large enough to include every class")
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    if np.any(counts == 0):
        raise RuntimeError("A class is absent from the candidate sample population")
    proportions = counts / counts.sum()
    allocation = np.maximum(1, np.floor(proportions * size).astype(int))
    while allocation.sum() < size:
        residual = proportions * size - allocation
        allocation[int(np.argmax(residual))] += 1
    while allocation.sum() > size:
        candidates = np.flatnonzero(allocation > 1)
        index = candidates[int(np.argmax(allocation[candidates] - proportions[candidates] * size))]
        allocation[index] -= 1
    rng = np.random.RandomState(seed)
    selected = []
    for class_index, count in enumerate(allocation):
        candidates = np.flatnonzero(labels == class_index)
        selected.append(rng.choice(candidates, int(count), replace=False))
    result = np.concatenate(selected).astype(np.int64)
    rng.shuffle(result)
    if len(result) != size or len(np.unique(result)) != size:
        raise RuntimeError("Stratified sampling did not produce unique requested rows")
    return result


def load_student(path: Path, hidden: tuple[int, int]) -> StudentMLP:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model = StudentMLP(17, hidden, len(CLASS_NAMES))
    model.load_state_dict(state)
    return model.eval()


def load_torch_mapping(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a mapping in {path}")
    return value


def verify_student_deployment_lineage(
    letter: str,
    student_key: str,
    seed_root: Path,
    completion: dict[str, Any],
    execution_contract_sha256: str,
    dataset_sha256: str,
    scaler_sha256: str,
    teacher_probability_file_sha256: str,
) -> dict[str, Any]:
    result = completion["student_results"][f"{student_key}_rf_kd"]
    plain_path = seed_root / result["plain_state_dict"]
    rich_path = seed_root / result["rich_artifact"]
    if sha256_file(plain_path) != result["plain_state_dict_sha256"]:
        raise RuntimeError(f"Student {letter} plain checkpoint hash differs")
    if sha256_file(rich_path) != result["rich_artifact_sha256"]:
        raise RuntimeError(f"Student {letter} rich artifact hash differs")
    plain_state = load_torch_mapping(plain_path)
    rich = load_torch_mapping(rich_path)
    rich_state = rich.get("state_dict")
    if not isinstance(rich_state, dict):
        raise RuntimeError(f"Student {letter} rich artifact has no state dictionary")
    plain_content_hash = state_content_sha256(plain_state)
    if plain_content_hash != result["trained_state_sha256"]:
        raise RuntimeError(f"Student {letter} trained-state hash differs")
    if state_content_sha256(rich_state) != plain_content_hash:
        raise RuntimeError(f"Student {letter} rich/plain states differ")
    required_rich = {
        "protocol_id": EXPECTED_DEPLOYMENT_PROTOCOL_ID,
        "seed": 42,
        "student": student_key,
        "route": "rf_kd",
        "input_dim": 17,
        "hidden_dims": list(STUDENT_SPECS[student_key]),
        "num_classes": len(CLASS_NAMES),
        "feature_names": FEATURE_NAMES,
        "class_names": CLASS_NAMES,
        "dataset_sha256": dataset_sha256,
        "scaler_sha256": scaler_sha256,
        "trained_state_sha256": plain_content_hash,
        "kd_hyperparameters": {"T": KD_TEMPERATURE, "alpha": 0.7},
    }
    for key, expected in required_rich.items():
        if rich.get(key) != expected:
            raise RuntimeError(f"Student {letter} rich artifact differs for {key}")
    teacher_provenance = rich.get("teacher_soft_target_provenance", {})
    if (
        teacher_provenance.get("train_probability_content_sha256")
        != EXPECTED_TRAIN_PROBABILITY_CONTENT_SHA256
    ):
        raise RuntimeError(f"Student {letter} teacher target provenance differs")

    generated_dir = (
        REPO_ROOT
        / "deployment/firmware_export/wsnds_rfkd_hil"
        / f"generated_fgds_student_{letter}_seed42"
    )
    strict_manifest_path = generated_dir / "strict_export_manifest.json"
    strict_manifest = verify_artifact_manifest(
        generated_dir,
        strict_manifest_path,
        expected_status="passed",
    )
    strict_report_path = generated_dir / "strict_export_report.json"
    strict_report = json.loads(strict_report_path.read_text(encoding="utf-8"))
    provenance = strict_report.get("provenance", {})
    gates = strict_report.get("gates", {})
    required_provenance = {
        "protocol_id": EXPECTED_DEPLOYMENT_PROTOCOL_ID,
        "student": student_key,
        "seed": 42,
        "dataset_sha256": dataset_sha256,
        "scaler_sha256": scaler_sha256,
        "model_file_sha256": sha256_file(plain_path),
        "model_artifact_sha256": sha256_file(rich_path),
        "execution_contract_sha256": execution_contract_sha256,
        "seed_completion_sha256": sha256_file(seed_root / "seed_completion.json"),
        "teacher_probability_file_sha256": teacher_probability_file_sha256,
    }
    if strict_report.get("status") != "passed":
        raise RuntimeError(f"Student {letter} strict export did not pass")
    for key, expected in required_provenance.items():
        if provenance.get(key) != expected:
            raise RuntimeError(f"Student {letter} strict provenance differs for {key}")
    if not (
        gates.get("full_test_rows") == 56301
        and gates.get("saved_test_rows_and_labels_exact") is True
        and gates.get("saved_fp32_predictions_exact") is True
    ):
        raise RuntimeError(f"Student {letter} strict export gates differ")
    return {
        "plain_checkpoint": str(plain_path.relative_to(REPO_ROOT)),
        "plain_checkpoint_sha256": sha256_file(plain_path),
        "rich_artifact": str(rich_path.relative_to(REPO_ROOT)),
        "rich_artifact_sha256": sha256_file(rich_path),
        "trained_state_content_sha256": plain_content_hash,
        "strict_export_report": str(strict_report_path.relative_to(REPO_ROOT)),
        "strict_export_report_sha256": sha256_file(strict_report_path),
        "strict_export_manifest": strict_manifest,
        "strict_export_id": strict_report["export_id"],
        "hardware_replay_scope": (
            "The verified FP32 checkpoint is the source of the strict fixed-point export. "
            "SHAP still explains FP32 outputs, not integer MCU outputs."
        ),
    }


def temperature_scale_probabilities(
    probabilities: np.ndarray,
    temperature: float,
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float32)
    if temperature == 1.0:
        return values.astype(np.float64, copy=False)
    tensor = torch.from_numpy(values)
    softened = F.softmax(
        torch.log(tensor.clamp_min(1e-8)) / temperature,
        dim=1,
    )
    return softened.numpy().astype(np.float64, copy=False)


def student_predictor(
    model: StudentMLP,
    temperature: float = 1.0,
) -> Callable[[np.ndarray], np.ndarray]:
    def predict(values: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(np.asarray(values, dtype=np.float32))
        chunks = []
        with torch.no_grad():
            for start in range(0, len(tensor), 4096):
                chunks.append(
                    F.softmax(
                        model(tensor[start : start + 4096]) / temperature,
                        dim=1,
                    ).numpy()
                )
        return np.concatenate(chunks).astype(np.float64, copy=False)

    return predict


def teacher_predictor(
    model: Any,
    temperature: float = 1.0,
) -> Callable[[np.ndarray], np.ndarray]:
    def predict(values: np.ndarray) -> np.ndarray:
        return temperature_scale_probabilities(model.predict_proba(values), temperature)

    return predict


def probability_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame.columns if column.startswith("probability_")]
    if len(columns) != len(CLASS_NAMES):
        raise RuntimeError(f"Unexpected probability columns: {columns}")
    return columns


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
    saved_probabilities = frame[columns].to_numpy(dtype=np.float64)
    maximum_probability_delta = float(
        np.max(np.abs(saved_probabilities - probabilities))
    )
    checks: dict[str, bool] = {
        "row_count": len(frame) == len(labels),
        "source_indices_exact": np.array_equal(
            frame["source_row_index"].to_numpy(dtype=np.int64), source_indices
        ),
        "labels_exact": np.array_equal(
            frame["true_label"].to_numpy(dtype=np.int64), labels
        ),
        "predictions_exact": np.array_equal(
            frame["predicted_label"].to_numpy(dtype=np.int64), probabilities.argmax(axis=1)
        ),
        "probabilities_within_tolerance": bool(
            maximum_probability_delta <= atol
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"Prediction artifact verification failed for {path}: {checks}; "
            f"maximum_absolute_probability_delta={maximum_probability_delta:.9g}"
        )
    return {
        **checks,
        "maximum_absolute_probability_delta": maximum_probability_delta,
        "file_sha256": sha256_file(path),
    }


def run_permutation_shap(
    name: str,
    predictor: Callable[[np.ndarray], np.ndarray],
    background: np.ndarray,
    explained: np.ndarray,
    output_dir: Path,
    seed: int,
    permutation_repeats: int,
    resume: bool,
    execution_fingerprint: str,
) -> dict[str, Any]:
    if permutation_repeats < 1:
        raise ValueError("permutation_repeats must be positive")
    path = output_dir / f"{name}_shap_values.npz"
    started = time.time()
    resumed = False
    job_contract = {
        "execution_fingerprint_sha256": execution_fingerprint,
        "job_name": name,
        "background_content_sha256": sha256_arrays(
            np.asarray(background, dtype=np.float64)
        ),
        "explained_content_sha256": sha256_arrays(
            np.asarray(explained, dtype=np.float64)
        ),
        "permutation_seed": int(seed),
        "permutation_repeats": int(permutation_repeats),
        "explainer": (
            "shap_permutation_with_independent_masker_and_exact_invariance_cache"
        ),
    }
    job_contract["job_fingerprint_sha256"] = canonical_json_sha256(job_contract)
    expected_contract_bytes = canonical_json_bytes(job_contract)
    if path.is_file():
        if not resume:
            raise RuntimeError(f"Unexpected existing SHAP artifact: {path}")
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != {
                "values",
                "base_values",
                "model_outputs",
                "job_contract_json_utf8",
                "computation_wall_seconds",
            }:
                raise RuntimeError(f"SHAP artifact schema differs: {path}")
            values = payload["values"].astype(np.float64)
            base_values = payload["base_values"].astype(np.float64)
            stored_outputs = payload["model_outputs"].astype(np.float64)
            observed_contract_bytes = payload["job_contract_json_utf8"].tobytes()
            computation_wall_seconds = float(
                np.asarray(payload["computation_wall_seconds"], dtype=np.float64).reshape(-1)[0]
            )
        if observed_contract_bytes != expected_contract_bytes:
            raise RuntimeError(f"SHAP job contract differs: {path}")
        if not np.isfinite(computation_wall_seconds) or computation_wall_seconds < 0:
            raise RuntimeError(f"SHAP computation timing is invalid: {path}")
        resumed = True
    else:
        print(f"SHAP subject={name} rows={len(explained)}", flush=True)
        if not (
            np.all(np.isfinite(background)) and np.all(np.isfinite(explained))
        ):
            raise RuntimeError("SHAP background and explained inputs must be finite")
        masker = ExactInvariantIndependentMasker(
            background, max_samples=len(background)
        )
        explainer = shap.Explainer(
            predictor,
            masker,
            algorithm="permutation",
            output_names=CLASS_NAMES,
            seed=seed,
        )
        explanation = explainer(
            explained,
            max_evals=(2 * explained.shape[1] + 1) * permutation_repeats,
            batch_size=256,
            silent=False,
        )
        values = np.asarray(explanation.values, dtype=np.float64)
        base_values = np.asarray(explanation.base_values, dtype=np.float64)
        computation_wall_seconds = time.time() - started
    if values.shape != (len(explained), len(FEATURE_NAMES), len(CLASS_NAMES)):
        raise RuntimeError(f"Unexpected SHAP shape for {name}: {values.shape}")
    predicted = predictor(explained)
    if not resumed:
        stored_outputs = predicted
    if stored_outputs.shape != predicted.shape or not np.allclose(
        stored_outputs, predicted, rtol=0.0, atol=1e-10
    ):
        raise RuntimeError(f"SHAP artifact model outputs differ: {path}")
    if not (
        np.all(np.isfinite(values))
        and np.all(np.isfinite(base_values))
        and np.all(np.isfinite(predicted))
    ):
        raise RuntimeError(f"SHAP output contains NaN or infinity: {name}")
    if base_values.shape != predicted.shape:
        raise RuntimeError(f"Unexpected SHAP base-value shape for {name}: {base_values.shape}")
    reconstructed = base_values + values.sum(axis=1)
    if reconstructed.shape != predicted.shape or not np.all(np.isfinite(reconstructed)):
        raise RuntimeError(f"SHAP reconstruction is invalid: {name}")
    local_accuracy_delta = np.abs(predicted - reconstructed)
    maximum_local_delta = float(local_accuracy_delta.max())
    if maximum_local_delta > LOCAL_ACCURACY_ATOL:
        raise RuntimeError(
            f"SHAP local-accuracy delta {maximum_local_delta:.9g} exceeds "
            f"{LOCAL_ACCURACY_ATOL}: {name}"
        )
    if not resumed:
        atomic_save_npz(
            path,
            values=values,
            base_values=base_values,
            model_outputs=predicted,
            job_contract_json_utf8=np.frombuffer(
                expected_contract_bytes, dtype=np.uint8
            ).copy(),
            computation_wall_seconds=np.asarray(
                [computation_wall_seconds], dtype=np.float64
            ),
        )
    return {
        "values": values,
        "artifact": path.name,
        "artifact_sha256": sha256_file(path),
        "computation_wall_seconds": computation_wall_seconds,
        "resume_verification_wall_seconds": time.time() - started if resumed else 0.0,
        "resumed_from_verified_artifact": resumed,
        "job_contract": job_contract,
        "permutation_repeats": permutation_repeats,
        "local_accuracy": {
            "maximum_absolute_delta": maximum_local_delta,
            "mean_absolute_delta": float(local_accuracy_delta.mean()),
        },
    }


def rank_agreement(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    rho = float(spearmanr(left, right).statistic)
    if not np.isfinite(rho):
        raise RuntimeError("Spearman rank agreement is not finite")
    return {
        "spearman_rho": rho,
        "inference": "descriptive; no p-value is reported for these related comparisons",
    }


def bootstrap_rank_agreement(
    student_values: np.ndarray,
    teacher_values: np.ndarray,
    feature_groups: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    if student_values.shape != teacher_values.shape:
        raise RuntimeError("Student and teacher SHAP tensors have different shapes")
    if len(feature_groups) != len(student_values):
        raise RuntimeError("Feature-group vector does not match explained SHAP rows")
    unique_groups, inverse = np.unique(feature_groups, return_inverse=True)
    members = [np.flatnonzero(inverse == index) for index in range(len(unique_groups))]
    rng = np.random.RandomState(seed)
    values = []
    for _ in range(repeats):
        sampled_groups = rng.choice(len(unique_groups), len(unique_groups), replace=True)
        sampled = np.concatenate([members[int(index)] for index in sampled_groups])
        student_importance = np.abs(student_values[sampled]).mean(axis=(0, 2))
        teacher_importance = np.abs(teacher_values[sampled]).mean(axis=(0, 2))
        values.append(float(spearmanr(student_importance, teacher_importance).statistic))
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise RuntimeError("Feature-group bootstrap produced non-finite rank agreement")
    return {
        "sampling": (
            "exact-feature-group cluster bootstrap with replacement from the fixed "
            "explained cohort"
        ),
        "unique_feature_groups": int(len(unique_groups)),
        "explained_rows": int(len(feature_groups)),
        "repeats": repeats,
        "seed": seed,
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)),
        "percentile_95_interval": [
            float(np.quantile(array, 0.025)),
            float(np.quantile(array, 0.975)),
        ],
        "values": array.tolist(),
    }


def summarize_pair(
    student_values: np.ndarray,
    teacher_values: np.ndarray,
    feature_groups: np.ndarray,
    repeats: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    student_global = np.abs(student_values).mean(axis=(0, 2))
    teacher_global = np.abs(teacher_values).mean(axis=(0, 2))
    per_output_class = {}
    for class_index, class_name in enumerate(CLASS_NAMES):
        per_output_class[class_name] = rank_agreement(
            np.abs(student_values[:, :, class_index]).mean(axis=0),
            np.abs(teacher_values[:, :, class_index]).mean(axis=0),
        )
    student_order = np.argsort(-student_global, kind="stable")
    teacher_order = np.argsort(-teacher_global, kind="stable")
    return {
        "global_rank_agreement": rank_agreement(student_global, teacher_global),
        "per_output_class_rank_agreement": per_output_class,
        "bootstrap_global_rank_agreement": bootstrap_rank_agreement(
            student_values, teacher_values, feature_groups, repeats, bootstrap_seed
        ),
        "student_global_mean_absolute_shap": student_global.tolist(),
        "teacher_global_mean_absolute_shap": teacher_global.tolist(),
        "student_ranked_features": [FEATURE_NAMES[index] for index in student_order],
        "teacher_ranked_features": [FEATURE_NAMES[index] for index in teacher_order],
        "top_5_overlap_count": int(
            len(set(student_order[:5].tolist()) & set(teacher_order[:5].tolist()))
        ),
        "interpretation_boundary": (
            f"This measures feature-ranking agreement on one {len(student_values)}-row stratified "
            "test cohort. It is not evidence that the student reproduces the RF's "
            "decision logic on every test record."
        ),
    }


def summarize_estimator_replicates(replicates: list[dict[str, Any]]) -> dict[str, Any]:
    if len(replicates) < 2:
        raise RuntimeError("Estimator-variation summary requires at least two replicates")
    rho = np.asarray(
        [record["global_rank_agreement"]["spearman_rho"] for record in replicates],
        dtype=np.float64,
    )
    top_five = np.asarray(
        [record["top_5_overlap_count"] for record in replicates], dtype=np.int64
    )
    student_importance = np.asarray(
        [record["student_global_mean_absolute_shap"] for record in replicates],
        dtype=np.float64,
    )
    teacher_importance = np.asarray(
        [record["teacher_global_mean_absolute_shap"] for record in replicates],
        dtype=np.float64,
    )
    mean_student = student_importance.mean(axis=0)
    mean_teacher = teacher_importance.mean(axis=0)
    return {
        "replicate_count": len(replicates),
        "variation_source": (
            "independent stratified training backgrounds and permutation-estimator seeds; "
            "the explained cohort and trained model states are fixed"
        ),
        "global_spearman_rho": {
            "values": rho.tolist(),
            "mean": float(rho.mean()),
            "sample_std": float(rho.std(ddof=1)),
            "min": float(rho.min()),
            "max": float(rho.max()),
        },
        "top_5_overlap_count": {
            "values": top_five.tolist(),
            "mean": float(top_five.mean()),
            "min": int(top_five.min()),
            "max": int(top_five.max()),
        },
        "mean_importance_rank_agreement": rank_agreement(mean_student, mean_teacher),
        "student_mean_absolute_shap_across_replicates": mean_student.tolist(),
        "student_sample_std_across_replicates": student_importance.std(
            axis=0, ddof=1
        ).tolist(),
        "teacher_mean_absolute_shap_across_replicates": mean_teacher.tolist(),
        "teacher_sample_std_across_replicates": teacher_importance.std(
            axis=0, ddof=1
        ).tolist(),
    }


def main() -> int:
    args = parse_args()
    if args.seed != 42:
        raise RuntimeError("The preserved deployment specimen is seed 42")
    output_is_nonempty = args.output_dir.exists() and any(args.output_dir.iterdir())
    if args.confirm_explanations and output_is_nonempty and not args.resume:
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    if args.confirm_explanations and args.resume and not output_is_nonempty:
        raise FileNotFoundError(
            f"Resume requires an existing non-empty output directory: {args.output_dir}"
        )
    if args.confirm_explanations and args.resume:
        validate_resume_inventory(args.output_dir, args)
    if args.estimator_replicates < 2:
        raise ValueError("At least two estimator replicates are required")
    if args.permutation_repeats < 1 or args.bootstrap_repeats < 2:
        raise ValueError("Permutation and bootstrap repeat counts are invalid")
    set_seed(args.seed)
    dataset = load_wsnds(args.dataset_csv.resolve())
    if dataset["feature_names"] != FEATURE_NAMES:
        raise RuntimeError("Feature names differ from the deployment contract")
    split = feature_group_split(dataset["features"], dataset["labels"])
    scaled, scaler = apply_train_scaler(split)
    execution_path = args.deployment_root / "execution_contract.json"
    preprocessing_path = args.deployment_root / "preprocessing_contract.json"
    root_manifest_path = args.deployment_root / "artifact_manifest.json"
    seed_root = args.deployment_root / "seed_42"
    completion_path = seed_root / "seed_completion.json"
    seed_manifest_path = seed_root / "artifact_manifest.json"
    for path in [
        execution_path,
        preprocessing_path,
        root_manifest_path,
        completion_path,
        seed_manifest_path,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)
    root_manifest_verification = verify_artifact_manifest(
        args.deployment_root, root_manifest_path
    )
    seed_manifest_verification = verify_artifact_manifest(seed_root, seed_manifest_path)
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    observed_indices = sha256_arrays(
        split["train_indices"], split["validation_indices"], split["test_indices"]
    )
    observed_scaler = sha256_arrays(
        np.asarray(scaler.mean_, dtype=np.float64),
        np.asarray(scaler.scale_, dtype=np.float64),
        np.asarray(scaler.var_, dtype=np.float64),
    )
    required = {
        "protocol_id": EXPECTED_DEPLOYMENT_PROTOCOL_ID,
        "dataset_sha256": dataset["dataset_sha256"],
        "split_indices_sha256": observed_indices,
        "scaler_sha256": observed_scaler,
    }
    for key, value in required.items():
        if execution.get(key) != value:
            raise RuntimeError(f"Deployment execution contract differs for {key}")
    execution_contract_sha256 = sha256_file(execution_path)
    if execution.get("kd_hyperparameters") != {"T": KD_TEMPERATURE, "alpha": 0.7}:
        raise RuntimeError("Deployment KD hyperparameters differ")
    expected_teacher_config = {
        "n_estimators": 500,
        "max_depth": 15,
        "calibration_method": "isotonic",
        "calibration_cv": 3,
    }
    if execution.get("teacher_config") != expected_teacher_config:
        raise RuntimeError("Deployment teacher configuration differs")
    required_completion = {
        "protocol_id": EXPECTED_DEPLOYMENT_PROTOCOL_ID,
        "seed": 42,
        "status": "complete",
        "dataset_sha256": dataset["dataset_sha256"],
        "split_indices_sha256": observed_indices,
        "scaler_sha256": observed_scaler,
        "execution_contract_sha256": execution_contract_sha256,
        "teacher_config": expected_teacher_config,
    }
    for key, expected in required_completion.items():
        if completion.get(key) != expected:
            raise RuntimeError(f"Deployment seed completion differs for {key}")
    if preprocessing.get("feature_overlap_audit") != split["group_audit"]:
        raise RuntimeError("Deployment feature-overlap audit differs")

    print("Rebuilding deterministic group-calibrated RF teacher", flush=True)
    groups = verified_feature_hashes(split["X_train_raw"])
    teacher_started = time.time()
    teacher, calibration_audit = fit_calibrated_rf(
        scaled["X_train"], split["y_train"], args.seed, groups=groups
    )
    train_probability = teacher.predict_proba(scaled["X_train"]).astype(np.float32)
    preserved_train_path = seed_root / "rf_train_probabilities.npy"
    preserved_train_probability = np.load(preserved_train_path, allow_pickle=False)
    if (
        preserved_train_probability.dtype != np.float32
        or preserved_train_probability.shape != train_probability.shape
        or not np.array_equal(preserved_train_probability, train_probability)
    ):
        raise RuntimeError("Rebuilt RF does not exactly reproduce the preserved training array")
    if sha256_file(preserved_train_path) != EXPECTED_TRAIN_PROBABILITY_FILE_SHA256:
        raise RuntimeError("Preserved RF training-probability file hash differs")
    train_content_hash = sha256_arrays(train_probability)
    if train_content_hash != EXPECTED_TRAIN_PROBABILITY_CONTENT_SHA256:
        raise RuntimeError(
            "Rebuilt calibrated RF does not reproduce the preserved soft targets: "
            f"{train_content_hash}"
        )
    if sha256_arrays(preserved_train_probability) != EXPECTED_TRAIN_PROBABILITY_CONTENT_SHA256:
        raise RuntimeError("Preserved RF training-probability content hash differs")
    test_probability = teacher.predict_proba(scaled["X_test"]).astype(
        np.float32
    ).astype(np.float64)
    teacher_verification = verify_prediction_artifact(
        seed_root / "RF_teacher_test_predictions.csv",
        split["test_indices"],
        split["y_test"],
        test_probability,
        atol=5e-8,
    )
    teacher_metrics = classification_metrics(split["y_test"], test_probability)
    if abs(teacher_metrics["macro_f1"] - completion["teacher_metrics"]["macro_f1"]) > 1e-12:
        raise RuntimeError("Rebuilt RF teacher macro-F1 differs from preserved completion")
    fitter_source = inspect.getsourcefile(fit_calibrated_rf)
    if fitter_source is None:
        raise RuntimeError("Cannot resolve calibrated RF fitter source")
    fitter_source_path = Path(fitter_source).resolve()
    common_source = inspect.getsourcefile(StudentMLP)
    if common_source is None:
        raise RuntimeError("Cannot resolve the shared WSN-DS model/preprocessing source")
    common_source_path = Path(common_source).resolve()

    student_models = {}
    student_verification = {}
    student_deployment_lineage = {}
    for letter, key in [("A", "student_A"), ("B", "student_B")]:
        student_deployment_lineage[letter] = verify_student_deployment_lineage(
            letter,
            key,
            seed_root,
            completion,
            execution_contract_sha256,
            dataset["dataset_sha256"],
            observed_scaler,
            sha256_file(preserved_train_path),
        )
        model_path = seed_root / f"student_{letter}_KD_from_RF_fp32.pt"
        expected_hash = completion["student_results"][f"{key}_rf_kd"][
            "plain_state_dict_sha256"
        ]
        if sha256_file(model_path) != expected_hash:
            raise RuntimeError(f"Student {letter} state file hash differs")
        model = load_student(model_path, STUDENT_SPECS[key])
        probability = student_predictor(model)(scaled["X_test"])
        artifact_verification = verify_prediction_artifact(
            seed_root / f"student_{letter}_KD_from_RF_test_predictions.csv",
            split["test_indices"],
            split["y_test"],
            probability,
            atol=STUDENT_CROSS_PLATFORM_REPLAY_ATOL,
        )
        replay_metrics = classification_metrics(split["y_test"], probability)
        expected_metrics = completion["student_results"][f"{key}_rf_kd"]["metrics"]
        if replay_metrics != expected_metrics:
            raise RuntimeError(f"Student {letter} checkpoint replay metrics differ")
        student_verification[letter] = {
            **artifact_verification,
            "cross_platform_fp32_probability_replay_atol": (
                STUDENT_CROSS_PLATFORM_REPLAY_ATOL
            ),
            "metrics_exact": True,
            "metrics": replay_metrics,
        }
        student_models[letter] = model

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "shap": shap.__version__,
    }
    preflight = {
        "protocol_id": PROTOCOL_ID,
        "status": "passed",
        "script_sha256": sha256_file(SCRIPT_PATH),
        "dataset_sha256": dataset["dataset_sha256"],
        "deployment_execution_contract_sha256": sha256_file(execution_path),
        "deployment_preprocessing_contract_sha256": sha256_file(preprocessing_path),
        "deployment_seed_completion_sha256": sha256_file(completion_path),
        "split_indices_sha256": observed_indices,
        "scaler_sha256": observed_scaler,
        "feature_overlap_audit": split["group_audit"],
        "teacher": {
            "identity": (
                "output-validated deterministic reconstruction of the calibrated RF; "
                "no original serialized RF was retained"
            ),
            "original_executed_training_source_sha256": execution["script_sha256"],
            "fitter_source_path": str(fitter_source_path.relative_to(REPO_ROOT)),
            "fitter_source_sha256": sha256_file(fitter_source_path),
            "function_equivalence_on_permutation_masked_inputs_established": False,
            "fit_seconds": time.time() - teacher_started,
            "calibration_audit": calibration_audit,
            "train_probability_content_sha256": train_content_hash,
            "test_artifact_verification": teacher_verification,
            "metrics": teacher_metrics,
        },
        "students": student_verification,
        "student_deployment_lineage": student_deployment_lineage,
        "artifact_manifests": {
            "deployment_root": root_manifest_verification,
            "seed_42": seed_manifest_verification,
        },
        "shared_model_preprocessing_source_path": str(
            common_source_path.relative_to(REPO_ROOT)
        ),
        "shared_model_preprocessing_source_sha256": sha256_file(common_source_path),
        "environment": environment,
        "deployment_specimen_scope": (
            "Separate Linux seed-42 deployment replication from the five-seed "
            "Winterfell lineage; it is not the local Windows ten-seed seed-42 checkpoint."
        ),
        "explanations_started": False,
    }
    print(json.dumps(preflight, indent=2), flush=True)
    if not args.confirm_explanations:
        print("SHAP was not started. Pass --confirm-explanations to run.")
        return 0

    explain_indices = stratified_sample(
        split["y_test"], args.explain_size, args.seed + 2000
    )
    explained = scaled["X_test"][explain_indices]
    explained_feature_groups = verified_feature_hashes(split["X_test_raw"])[
        explain_indices
    ]
    background_seeds = np.asarray(
        [args.seed + 1000 + replicate for replicate in range(args.estimator_replicates)],
        dtype=np.int64,
    )
    permutation_seeds = np.asarray(
        [args.seed + 3000 + replicate for replicate in range(args.estimator_replicates)],
        dtype=np.int64,
    )
    background_indices = np.stack(
        [
            stratified_sample(split["y_train"], args.background_size, int(seed))
            for seed in background_seeds
        ]
    )
    sampling_arrays = {
        "background_partition_indices": background_indices,
        "background_source_row_indices": np.stack(
            [split["train_indices"][indices] for indices in background_indices]
        ),
        "background_labels": np.stack(
            [split["y_train"][indices] for indices in background_indices]
        ),
        "background_seeds": background_seeds,
        "permutation_seeds": permutation_seeds,
        "explained_partition_indices": explain_indices,
        "explained_source_row_indices": split["test_indices"][explain_indices],
        "explained_labels": split["y_test"][explain_indices],
        "explained_exact_feature_group_hashes": explained_feature_groups,
    }

    output_contracts = {
        "fp32_deployment_source_probabilities_T1": 1.0,
        "kd_softened_probabilities_T4": KD_TEMPERATURE,
    }
    source_snapshots = {
        "executed_shap_source.py": SCRIPT_PATH,
        "bound_common_source.py": common_source_path,
        "bound_rf_fitter_source.py": fitter_source_path,
    }
    execution_contract = {
        "protocol_id": PROTOCOL_ID,
        "script_sha256": sha256_file(SCRIPT_PATH),
        "source_snapshots": {
            name: sha256_file(source) for name, source in source_snapshots.items()
        },
        "deployment_specimen_scope": (
            "Separate Linux seed-42 deployment replication from the five-seed "
            "Winterfell lineage; it is not the local Windows ten-seed seed-42 checkpoint."
        ),
        "dataset_sha256": dataset["dataset_sha256"],
        "split_indices_sha256": observed_indices,
        "scaler_sha256": observed_scaler,
        "deployment_execution_contract_sha256": execution_contract_sha256,
        "deployment_preprocessing_contract_sha256": sha256_file(preprocessing_path),
        "deployment_seed_completion_sha256": sha256_file(completion_path),
        "deployment_root_manifest_sha256": root_manifest_verification["sha256"],
        "deployment_seed_manifest_sha256": seed_manifest_verification["sha256"],
        "preserved_teacher_train_probability_file_sha256": sha256_file(
            preserved_train_path
        ),
        "preserved_teacher_train_probability_content_sha256": train_content_hash,
        "teacher_reconstruction": {
            "original_executed_training_source_sha256": execution["script_sha256"],
            "current_fitter_source_sha256": sha256_file(fitter_source_path),
            "train_and_test_output_validation_passed": True,
            "function_equivalence_on_permutation_masked_inputs_established": False,
        },
        "shared_model_preprocessing_source_sha256": sha256_file(common_source_path),
        "environment": environment,
        "student_deployment_lineage": student_deployment_lineage,
        "parameters": {
            "seed": args.seed,
            "background_size": args.background_size,
            "explain_size": args.explain_size,
            "estimator_replicates": args.estimator_replicates,
            "permutation_repeats": args.permutation_repeats,
            "bootstrap_repeats": args.bootstrap_repeats,
            "local_accuracy_atol": LOCAL_ACCURACY_ATOL,
            "student_cross_platform_replay_atol": (
                STUDENT_CROSS_PLATFORM_REPLAY_ATOL
            ),
            "output_contracts": output_contracts,
            "masker_invariance_cache": (
                "bit-exact equality; near-but-not-equal values always trigger model evaluation"
            ),
        },
        "sampling_arrays": array_contract(sampling_arrays),
    }
    execution_fingerprint = canonical_json_sha256(execution_contract)
    execution_contract["execution_fingerprint_sha256"] = execution_fingerprint
    contract_path = args.output_dir / "execution_contract.json"
    sampling_path = args.output_dir / "sampling_contract.npz"
    preflight_path = args.output_dir / "preflight_verification.json"
    teacher_path = args.output_dir / "reconstructed_calibrated_rf_teacher_seed42.joblib"
    if args.resume:
        for name, source in source_snapshots.items():
            snapshot = args.output_dir / name
            if not snapshot.is_file() or snapshot.read_bytes() != source.read_bytes():
                raise RuntimeError(f"Resume source snapshot differs: {snapshot}")
        observed_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if observed_contract != execution_contract:
            raise RuntimeError("Resume execution contract differs from the requested run")
        verify_npz_exact(sampling_path, sampling_arrays)
        stored_preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        if not (
            stored_preflight.get("status") == "passed"
            and stored_preflight.get("explanations_started") is True
            and stored_preflight.get("execution_fingerprint_sha256")
            == execution_fingerprint
        ):
            raise RuntimeError("Resume preflight contract is incomplete or differs")
        if sha256_file(teacher_path) != stored_preflight["teacher"][
            "serialized_reconstruction_sha256"
        ]:
            raise RuntimeError("Serialized reconstructed teacher hash differs on resume")
        stored_teacher = joblib.load(teacher_path)
        stored_train_probability = stored_teacher.predict_proba(
            scaled["X_train"]
        ).astype(np.float32)
        stored_test_probability = stored_teacher.predict_proba(scaled["X_test"]).astype(
            np.float32
        )
        if not (
            np.array_equal(stored_train_probability, preserved_train_probability)
            and np.array_equal(stored_test_probability, test_probability.astype(np.float32))
        ):
            raise RuntimeError("Serialized reconstructed teacher outputs differ on resume")
        teacher = stored_teacher
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for name, source in source_snapshots.items():
            atomic_write_bytes(args.output_dir / name, source.read_bytes())
        atomic_write_json(contract_path, execution_contract)
        atomic_save_npz(sampling_path, **sampling_arrays)
        atomic_joblib_dump(teacher_path, teacher)
        preflight["explanations_started"] = True
        preflight["execution_fingerprint_sha256"] = execution_fingerprint
        preflight["teacher"]["serialized_reconstruction"] = teacher_path.name
        preflight["teacher"]["serialized_reconstruction_sha256"] = sha256_file(
            teacher_path
        )
        atomic_write_json(preflight_path, preflight)

    final_report_path = args.output_dir / "shap_report.json"
    final_manifest_path = args.output_dir / "artifact_manifest.json"
    if args.resume and final_manifest_path.is_file():
        verify_artifact_manifest(
            args.output_dir,
            final_manifest_path,
            expected_protocol_id=PROTOCOL_ID,
            expected_files_excluding_manifest=expected_output_names(args)
            - {"artifact_manifest.json"},
        )
        stored_report = json.loads(final_report_path.read_text(encoding="utf-8"))
        if (
            stored_report.get("protocol_id") != PROTOCOL_ID
            or stored_report.get("status") != "complete"
            or stored_report.get("execution_fingerprint_sha256") != execution_fingerprint
        ):
            raise RuntimeError("Completed SHAP report differs from the execution contract")
        print(f"complete and verified: {final_report_path}")
        return 0

    condition_results: dict[str, Any] = {}
    for condition_index, (condition, temperature) in enumerate(output_contracts.items()):
        replicate_reports = []
        pair_records: dict[str, list[dict[str, Any]]] = {"A": [], "B": []}
        for replicate in range(args.estimator_replicates):
            background = scaled["X_train"][background_indices[replicate]]
            permutation_seed = int(permutation_seeds[replicate])
            prefix = f"{condition}_replicate_{replicate + 1}"
            subjects = {
                "teacher": run_permutation_shap(
                    f"{prefix}_teacher",
                    teacher_predictor(teacher, temperature),
                    background,
                    explained,
                    args.output_dir,
                    permutation_seed,
                    args.permutation_repeats,
                    args.resume,
                    execution_fingerprint,
                )
            }
            for letter in ["A", "B"]:
                subjects[f"student_{letter}"] = run_permutation_shap(
                    f"{prefix}_student_{letter}",
                    student_predictor(student_models[letter], temperature),
                    background,
                    explained,
                    args.output_dir,
                    permutation_seed,
                    args.permutation_repeats,
                    args.resume,
                    execution_fingerprint,
                )
            pairs = {}
            for letter_index, letter in enumerate(["A", "B"]):
                pair = summarize_pair(
                    subjects[f"student_{letter}"]["values"],
                    subjects["teacher"]["values"],
                    explained_feature_groups,
                    args.bootstrap_repeats,
                    args.seed
                    + 10_000 * (condition_index + 1)
                    + 1_000 * (replicate + 1)
                    + 100 * (letter_index + 1),
                )
                pairs[letter] = pair
                pair_records[letter].append(pair)
            for subject in subjects.values():
                subject.pop("values")
            replicate_reports.append(
                {
                    "replicate": replicate + 1,
                    "background_seed": int(background_seeds[replicate]),
                    "permutation_seed": permutation_seed,
                    "subjects": subjects,
                    "student_teacher_agreement": pairs,
                }
            )
        condition_results[condition] = {
            "temperature": temperature,
            "replicates": replicate_reports,
            "student_teacher_estimator_variation": {
                letter: summarize_estimator_replicates(pair_records[letter])
                for letter in ["A", "B"]
            },
        }
    report = {
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "subject_identity": (
            "An output-validated deterministic reconstruction of the seed-42 "
            "group-calibrated RF, plus the exact Student A/B FP32 checkpoints that "
            "were the source of the strict fixed-point deployments. These checkpoints "
            "belong to the separate Linux five-seed deployment replication, not the "
            "local Windows ten-seed seed-42 member."
        ),
        "method": {
            "explainer": (
                "model-agnostic permutation SHAP with an independent training-background "
                "masker and exact-equality cache invalidation for all three subjects"
            ),
            "masker_boundary": (
                "Independent feature replacement uses the sampled training background and "
                "can create masked combinations outside the observed joint feature distribution. "
                "Exact equality is used only to decide whether a changed background-row value "
                "requires reevaluation; it does not alter replacement values."
            ),
            "output_contracts": {
                "fp32_deployment_source_probabilities_T1": (
                    "ordinary FP32 checkpoint probabilities; these are not MCU outputs"
                ),
                "kd_softened_probabilities_T4": (
                    "probabilities softened at T=4, matching the KD temperature space; this "
                    "does not by itself identify causal knowledge transfer"
                ),
            },
            "feature_count": len(FEATURE_NAMES),
            "estimator_replicates": args.estimator_replicates,
            "permutation_repeats": args.permutation_repeats,
            "max_evals_per_record": (
                2 * len(FEATURE_NAMES) + 1
            ) * args.permutation_repeats,
            "background_size": args.background_size,
            "explain_size": args.explain_size,
            "explained_class_counts": np.bincount(
                split["y_test"][explain_indices], minlength=len(CLASS_NAMES)
            ).astype(int).tolist(),
            "bootstrap_repeats": args.bootstrap_repeats,
            "bootstrap_sampling_unit": "exact-feature-group cluster",
        },
        "feature_names": FEATURE_NAMES,
        "class_names": CLASS_NAMES,
        "teacher_model": {
            "artifact": teacher_path.name,
            "artifact_sha256": sha256_file(teacher_path),
            "identity": "deterministic reconstructed calibrated RF; original RF serialization was not preserved",
            "original_executed_training_source_sha256": execution["script_sha256"],
            "fitter_source_path": str(fitter_source_path.relative_to(REPO_ROOT)),
            "fitter_source_sha256": sha256_file(fitter_source_path),
            "train_probability_content_sha256": train_content_hash,
            "train_and_test_output_validation_passed": True,
            "function_equivalence_on_permutation_masked_inputs_established": False,
        },
        "student_deployment_lineage": student_deployment_lineage,
        "execution_fingerprint_sha256": execution_fingerprint,
        "conditions": condition_results,
        "scope_boundary": (
            f"The audit explains a fixed stratified subset of {args.explain_size} of the "
            f"{len(split['y_test']):,} "
            "FG-DS test records for one preserved Linux deployment-replication seed-42 "
            "trained state per student. These are distinct from the local Windows "
            "ten-seed seed-42 checkpoints. "
            "Cluster-bootstrap intervals resample exact feature groups and condition on each "
            "background and explainer seed. Across-replicate variation reflects "
            f"{args.estimator_replicates} training-background and permutation-seed pairs. "
            "The RF explanations belong to an output-validated reconstruction; equivalence "
            "to the unavailable original RF on synthetic masked inputs is not established. "
            "The independent masker can evaluate feature combinations outside the observed "
            "joint distribution. "
            "These quantities do not represent retrained-model uncertainty, MCU behavior, "
            "or causal knowledge transfer."
        ),
    }
    validate_resume_inventory(args.output_dir, args)
    atomic_write_json(args.output_dir / "shap_report.json", report)
    atomic_write_json(
        args.output_dir / "artifact_manifest.json",
        artifact_manifest(args.output_dir, PROTOCOL_ID, "complete"),
    )
    verify_artifact_manifest(
        args.output_dir,
        args.output_dir / "artifact_manifest.json",
        expected_protocol_id=PROTOCOL_ID,
        expected_files_excluding_manifest=expected_output_names(args)
        - {"artifact_manifest.json"},
    )
    print(f"complete: {args.output_dir / 'shap_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
