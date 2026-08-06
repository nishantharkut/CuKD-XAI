"""Run the compact, provenance-locked WSN-DS confirmation experiments.

The default mode is preflight and performs no model training.  Training modes
require --confirm-training and write only below a new output root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

try:
    from .tier15_common import (
        CLASS_NAMES,
        KD_ALPHA,
        KD_T,
        RF_CONFIG,
        SENSITIVITY_SEEDS,
        STUDENT_SPECS,
        TRAIN_CONFIG,
        StudentMLP,
        apply_train_scaler,
        archived_random_split,
        artifact_manifest,
        atomic_save_npy,
        atomic_save_npz,
        atomic_torch_save,
        atomic_write_csv,
        atomic_write_json,
        batched_probs,
        class_weights,
        classification_metrics,
        feature_group_split,
        load_wsnds,
        set_seed,
        sha256_arrays,
        sha256_file,
        split_hashes,
        train_rf_kd,
        train_standard,
        verified_feature_hashes,
    )
except ImportError:
    from tier15_common import (  # type: ignore[no-redef]
        CLASS_NAMES,
        KD_ALPHA,
        KD_T,
        RF_CONFIG,
        SENSITIVITY_SEEDS,
        STUDENT_SPECS,
        TRAIN_CONFIG,
        StudentMLP,
        apply_train_scaler,
        archived_random_split,
        artifact_manifest,
        atomic_save_npy,
        atomic_save_npz,
        atomic_torch_save,
        atomic_write_csv,
        atomic_write_json,
        batched_probs,
        class_weights,
        classification_metrics,
        feature_group_split,
        load_wsnds,
        set_seed,
        sha256_arrays,
        sha256_file,
        split_hashes,
        train_rf_kd,
        train_standard,
        verified_feature_hashes,
    )


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_DATASET = REPO_ROOT / "data" / "wsnds" / "WSN-DS.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "wsnds" / "confirmation_runs_v2"
DEPLOYMENT_PROTOCOL = "wsnds_archive_split_train_only_scaler_deployment_seed42_v1"
GROUP_PROTOCOL = "wsnds_feature_group_split_train_only_scaler_5seed_v1"
PREFLIGHT_PROTOCOL = "wsnds_confirmation_preflight_v2"
ACTIVE_V1_ROOT = REPO_ROOT / "results" / "wsnds" / "leakage_free_rerun" / "main_10seed"
EXPECTED_DEPLOYMENT_RF_CACHE_SHA256 = (
    "9ac2937644343782322fc397d0b8a3d7c8dc1d5047812e031e0829b79b8c069c"
)
EXPECTED_ACTIVE_V1_SOURCE_SHA256 = (
    "728eb19b1330e94652db9cf6f57f3d0d698fae0b9849e48b4e8ba25ba20eee27"
)
EXPECTED_ACTIVE_V1_MANIFEST_SHA256 = (
    "bf4dfe1fff61e170d6967d99b842433dbcf0d3a84f8a7a3692e12df4dfd6815d"
)
EXPECTED_ACTIVE_V1_SKLEARN_VERSION = "1.8.0"
EXPECTED_ACTIVE_V1_SPLIT_HASHES = {
    "train": "e40f6710abaa97013f2241275cdeff3fa262c7cae61a946292c0b4722703bb93",
    "validation": "4688ecc06fe61f590bef069b17830cd1a6b9e3e6afafa211dab3f5d075ff7d37",
    "test": "64256808f3cf9f06ffe4be7e423e5780b4c5e4344eb52d20e8fdd45db79130ff",
}
EXPECTED_ACTIVE_V1_TRANSFORMED_HASHES = {
    "train": "95091edecd00aefa3268d7f39655a3c042c31e314a3dd8ae2397592e11641432",
    "validation": "dc1cfdbcf63a360b0ca105be227430e88d36653cb7698f165f3a5d4b965dfa5e",
    "test": "d1651b05c8c50b1f22530fbbfe2bfc54364b9dc781dfc98a40db4d134ebda776",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["preflight", "deployment", "duplicate-sensitivity"],
        default="preflight",
    )
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--deployment-rf-cache",
        type=Path,
        default=ACTIVE_V1_ROOT / "rf_soft_seed_42.npy",
    )
    parser.add_argument(
        "--deployment-preprocessing-report",
        type=Path,
        default=ACTIVE_V1_ROOT / "leakage_free_preprocessing.json",
    )
    parser.add_argument(
        "--deployment-execution-manifest",
        type=Path,
        default=ACTIVE_V1_ROOT / "executed_source_snapshot" / "execution_manifest.json",
    )
    parser.add_argument(
        "--deployment-source-snapshot",
        type=Path,
        default=(
            ACTIVE_V1_ROOT
            / "executed_source_snapshot"
            / "run_leakage_free_wsnds.executed.py"
        ),
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--confirm-training",
        action="store_true",
        help="Required for either training mode; omitted in safe preflight mode.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Verify and skip completed seed directories; never trusts names alone.",
    )
    return parser.parse_args()


def require_disjoint_output(path: Path, protected_roots: list[Path]) -> Path:
    resolved = path.resolve()
    for protected in protected_roots:
        protected = protected.resolve()
        try:
            resolved.relative_to(protected)
            overlaps = True
        except ValueError:
            try:
                protected.relative_to(resolved)
                overlaps = True
            except ValueError:
                overlaps = False
        if overlaps:
            raise RuntimeError(f"Output path overlaps protected evidence/source: {resolved}")
    return resolved


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_new_or_resume(root: Path, resume: bool) -> None:
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {root}. "
            "Use --resume only after checking that this is the intended protocol."
        )
    root.mkdir(parents=True, exist_ok=True)


def active_v1_worker_is_running() -> bool:
    """Check the preserved Windows worker PID without adding a psutil dependency."""
    if os.name != "nt":
        return False
    manifest_path = ACTIVE_V1_ROOT / "executed_source_snapshot" / "execution_manifest.json"
    if not manifest_path.is_file():
        return False
    worker_pid = int(json.loads(manifest_path.read_text(encoding="utf-8"))["worker_pid"])
    import ctypes

    process_query_limited_information = 0x1000
    handle = ctypes.windll.kernel32.OpenProcess(
        process_query_limited_information, False, worker_pid
    )
    if not handle:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)
    return True


def environment_record(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()
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


def count_distribution(labels: np.ndarray) -> list[int]:
    return np.bincount(labels, minlength=len(CLASS_NAMES)).astype(int).tolist()


def scaler_hash(scaler: Any) -> str:
    return sha256_arrays(
        np.asarray(scaler.mean_, dtype=np.float64),
        np.asarray(scaler.scale_, dtype=np.float64),
        np.asarray(scaler.var_, dtype=np.float64),
    )


def build_data_context(
    dataset: dict[str, Any],
    split: dict[str, Any],
) -> dict[str, Any]:
    scaled, scaler = apply_train_scaler(split)
    return {
        "dataset": dataset,
        "split": split,
        "scaled": scaled,
        "scaler": scaler,
        "split_hashes": split_hashes(split),
        "split_indices_sha256": sha256_arrays(
            split["train_indices"],
            split["validation_indices"],
            split["test_indices"],
        ),
        "scaler_sha256": scaler_hash(scaler),
        "transformed_hashes": {
            name: sha256_arrays(scaled[f"X_{name}"])
            for name in ["train", "validation", "test"]
        },
    }


def write_data_contract(root: Path, context: dict[str, Any], protocol_id: str) -> None:
    dataset = context["dataset"]
    split = context["split"]
    scaler = context["scaler"]
    indices_path = root / "split_indices.npz"
    scaler_path = root / "scaler_parameters.npz"
    atomic_save_npz(
        indices_path,
        train_indices=split["train_indices"],
        validation_indices=split["validation_indices"],
        test_indices=split["test_indices"],
    )
    atomic_save_npz(
        scaler_path,
        mean=np.asarray(scaler.mean_, dtype=np.float64),
        scale=np.asarray(scaler.scale_, dtype=np.float64),
        var=np.asarray(scaler.var_, dtype=np.float64),
        n_samples_seen=np.atleast_1d(scaler.n_samples_seen_).astype(np.int64),
    )
    atomic_write_json(root / "preprocessing_contract.json", {
        "protocol_id": protocol_id,
        "dataset_path_recorded": str(Path(dataset["dataset_path_recorded"])),
        "dataset_sha256": dataset["dataset_sha256"],
        "dataset_shape": list(dataset["features"].shape),
        "target_column": dataset["target_column"],
        "feature_names": dataset["feature_names"],
        "class_names": dataset["class_names"],
        "split_policy": split["policy"],
        "split_sizes": {
            name: int(len(split[f"{name}_indices"]))
            for name in ["train", "validation", "test"]
        },
        "split_class_counts": {
            name: count_distribution(split[f"y_{name}"])
            for name in ["train", "validation", "test"]
        },
        "split_hashes": context["split_hashes"],
        "split_indices_sha256": context["split_indices_sha256"],
        "split_indices_file": indices_path.name,
        "split_indices_file_sha256": sha256_file(indices_path),
        "scaler_fit_partition": "train only",
        "scaler_fit_row_count": int(len(split["train_indices"])),
        "scaler_fit_indices_sha256": sha256_arrays(split["train_indices"]),
        "scaler_sha256": context["scaler_sha256"],
        "scaler_parameters_file": scaler_path.name,
        "scaler_parameters_file_sha256": sha256_file(scaler_path),
        "transformed_split_hashes": context["transformed_hashes"],
        "feature_overlap_audit": split["group_audit"],
    })


def dataset_with_recorded_path(dataset_csv: Path) -> dict[str, Any]:
    resolved = dataset_csv.resolve()
    dataset = load_wsnds(resolved)
    dataset["dataset_path_recorded"] = str(resolved)
    return dataset


def verify_existing_preflight(root: Path, dataset: dict[str, Any]) -> Path:
    report_path = root / "preflight_report.json"
    manifest_path = root / "artifact_manifest.json"
    if not report_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("Existing preflight directory is incomplete")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_report = {
        "protocol_id": PREFLIGHT_PROTOCOL,
        "status": "passed",
        "script_sha256": sha256_file(SCRIPT_PATH),
        "common_module_sha256": sha256_file(SCRIPT_DIR / "tier15_common.py"),
        "dataset_sha256": dataset["dataset_sha256"],
        "training_started": False,
    }
    for key, value in expected_report.items():
        if report.get(key) != value:
            raise RuntimeError(f"Existing preflight differs for {key}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_id") != PREFLIGHT_PROTOCOL or manifest.get("status") != "complete":
        raise RuntimeError("Existing preflight manifest has the wrong protocol or status")
    files = manifest.get("files")
    if not isinstance(files, list) or manifest.get("file_count_excluding_manifest") != len(files):
        raise RuntimeError("Existing preflight manifest inventory is invalid")
    declared: set[str] = set()
    for item in files:
        relative = item.get("path")
        if not isinstance(relative, str) or not relative:
            raise RuntimeError("Existing preflight manifest contains an invalid path")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(f"Existing preflight manifest path escapes its root: {relative!r}")
        normalized = relative_path.as_posix()
        if normalized in declared:
            raise RuntimeError(f"Existing preflight manifest duplicates {relative!r}")
        declared.add(normalized)
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise RuntimeError(f"Existing preflight artifact escapes its root: {relative!r}") from exc
        if (
            not path.is_file()
            or path.stat().st_size != item.get("size_bytes")
            or sha256_file(path) != item.get("sha256")
        ):
            raise RuntimeError(f"Existing preflight artifact failed verification: {path}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != declared:
        raise RuntimeError("Existing preflight inventory differs from its manifest")
    return report_path


def preflight_report(dataset: dict[str, Any], output_root: Path, resume: bool) -> Path:
    root = output_root / "preflight"
    if root.exists() and any(root.iterdir()):
        if not resume:
            raise FileExistsError(
                f"Refusing to overwrite existing preflight evidence: {root}. "
                "Use --resume to verify and reuse the exact existing preflight."
            )
        return verify_existing_preflight(root, dataset)
    root.mkdir(parents=True, exist_ok=True)
    random_context = build_data_context(
        dataset,
        archived_random_split(dataset["features"], dataset["labels"]),
    )
    grouped_context = build_data_context(
        dataset,
        feature_group_split(dataset["features"], dataset["labels"]),
    )
    write_data_contract(root / "archive_split", random_context, DEPLOYMENT_PROTOCOL)
    write_data_contract(root / "feature_group_split", grouped_context, GROUP_PROTOCOL)

    hashes = verified_feature_hashes(dataset["features"])
    _, multiplicity = np.unique(hashes, return_counts=True)
    report_path = root / "preflight_report.json"
    atomic_write_json(report_path, {
        "protocol_id": PREFLIGHT_PROTOCOL,
        "status": "passed",
        "script_sha256": sha256_file(SCRIPT_PATH),
        "common_module_sha256": sha256_file(SCRIPT_DIR / "tier15_common.py"),
        "dataset_sha256": dataset["dataset_sha256"],
        "dataset_shape": list(dataset["features"].shape),
        "class_counts": count_distribution(dataset["labels"]),
        "exact_duplicate_audit": {
            "unique_feature_rows": int(len(multiplicity)),
            "duplicate_rows_excluding_first": int(np.maximum(multiplicity - 1, 0).sum()),
            "rows_in_non_unique_feature_groups": int(multiplicity[multiplicity > 1].sum()),
            "non_unique_feature_groups": int((multiplicity > 1).sum()),
            "hash_collisions_with_unequal_rows": 0,
        },
        "archive_split": {
            "split_hashes": random_context["split_hashes"],
            "scaler_sha256": random_context["scaler_sha256"],
            "feature_overlap_audit": random_context["split"]["group_audit"],
        },
        "feature_group_split": {
            "split_hashes": grouped_context["split_hashes"],
            "scaler_sha256": grouped_context["scaler_sha256"],
            "feature_overlap_audit": grouped_context["split"]["group_audit"],
        },
        "training_started": False,
    })
    atomic_write_json(root / "artifact_manifest.json", artifact_manifest(
        root, PREFLIGHT_PROTOCOL, "complete"
    ))
    return report_path


def atomic_joblib_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary, compress=0)
    os.replace(temporary, path)


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    arrays = [state_dict[name].numpy() for name in sorted(state_dict)]
    return sha256_arrays(*arrays)


def prediction_frame(
    row_indices: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    result: dict[str, Any] = {
        "source_row_index": row_indices.astype(np.int64),
        "true_label": labels.astype(np.int64),
        "predicted_label": probabilities.argmax(axis=1).astype(np.int64),
    }
    for class_index, class_name in enumerate(CLASS_NAMES):
        result[f"probability_{class_index}_{class_name}"] = probabilities[:, class_index]
    return pd.DataFrame(result)


def fit_calibrated_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    groups: np.ndarray | None = None,
) -> tuple[CalibratedClassifierCV, dict[str, Any]]:
    calibration_cv: int | list[tuple[np.ndarray, np.ndarray]] = RF_CONFIG["calibration_cv"]
    calibration_audit: dict[str, Any] = {
        "strategy": "stratified_kfold_without_group_constraint",
        "folds": RF_CONFIG["calibration_cv"],
        "group_overlap_per_fold": None,
    }
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape != (len(y_train),):
            raise RuntimeError("RF calibration group vector has the wrong shape")
        splitter = StratifiedGroupKFold(
            n_splits=RF_CONFIG["calibration_cv"],
            shuffle=True,
            random_state=seed,
        )
        calibration_cv = list(splitter.split(X_train, y_train, groups))
        overlap_counts = []
        for train_indices, validation_indices in calibration_cv:
            overlap = set(map(int, groups[train_indices])) & set(
                map(int, groups[validation_indices])
            )
            overlap_counts.append(len(overlap))
        if any(overlap_counts):
            raise RuntimeError(
                f"Exact feature groups cross RF calibration folds: {overlap_counts}"
            )
        calibration_audit = {
            "strategy": "stratified_group_kfold",
            "folds": len(calibration_cv),
            "group_overlap_per_fold": overlap_counts,
            "unique_groups": int(len(np.unique(groups))),
        }
    teacher = CalibratedClassifierCV(
        RandomForestClassifier(
            n_estimators=RF_CONFIG["n_estimators"],
            max_depth=RF_CONFIG["max_depth"],
            random_state=seed,
            n_jobs=-1,
        ),
        method=RF_CONFIG["calibration_method"],
        cv=calibration_cv,
    )
    teacher.fit(X_train, y_train)
    return teacher, calibration_audit


def load_bound_deployment_cache(
    context: dict[str, Any],
    cache_path: Path,
    preprocessing_report_path: Path,
    execution_manifest_path: Path,
    source_snapshot_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    paths = [
        cache_path,
        preprocessing_report_path,
        execution_manifest_path,
        source_snapshot_path,
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    cache_file_sha256 = sha256_file(cache_path)
    if cache_file_sha256 != EXPECTED_DEPLOYMENT_RF_CACHE_SHA256:
        raise RuntimeError(
            "Deployment RF cache is not the preserved seed-42 cache: "
            f"{cache_file_sha256}"
        )

    report = json.loads(preprocessing_report_path.read_text(encoding="utf-8"))
    manifest = json.loads(execution_manifest_path.read_text(encoding="utf-8"))
    source_sha256 = sha256_file(source_snapshot_path)
    if source_sha256.lower() != EXPECTED_ACTIVE_V1_SOURCE_SHA256:
        raise RuntimeError("Executed-source snapshot is not the pinned active-v1 source")
    if sha256_file(execution_manifest_path).lower() != EXPECTED_ACTIVE_V1_MANIFEST_SHA256:
        raise RuntimeError("Execution manifest is not the pinned active-v1 manifest")
    if manifest.get("source_sha256", "").lower() != source_sha256.lower():
        raise RuntimeError("Executed-source snapshot does not match its execution manifest")
    if manifest.get("protocol_id") != "archive_random_split_train_scaler_retuned_v1":
        raise RuntimeError(f"Unexpected active-run protocol: {manifest.get('protocol_id')!r}")
    if report.get("dataset_sha256") != context["dataset"]["dataset_sha256"]:
        raise RuntimeError("RF cache preprocessing report has a different dataset SHA-256")
    if report.get("scaler_fit_partition") != "train":
        raise RuntimeError("RF cache was not produced with a train-only scaler")
    expected_shapes = {
        name: list(context["scaled"][f"X_{name}"].shape)
        for name in ["train", "validation", "test"]
    }
    if report.get("split_shapes") != expected_shapes:
        raise RuntimeError(
            f"RF cache split shapes differ: {report.get('split_shapes')} != {expected_shapes}"
        )
    if report.get("feature_names") != context["dataset"]["feature_names"]:
        raise RuntimeError("RF cache feature order differs from the confirmation contract")
    if report.get("class_names") != CLASS_NAMES:
        raise RuntimeError("RF cache class order differs from the confirmation contract")
    if report.get("environment", {}).get("sklearn") != EXPECTED_ACTIVE_V1_SKLEARN_VERSION:
        raise RuntimeError("RF cache preprocessing report has an unexpected scikit-learn version")
    if sklearn.__version__ != EXPECTED_ACTIVE_V1_SKLEARN_VERSION:
        raise RuntimeError(
            "Current scikit-learn version cannot reproduce the preserved active-v1 row order"
        )
    if context["split_hashes"] != EXPECTED_ACTIVE_V1_SPLIT_HASHES:
        raise RuntimeError("Reconstructed active-v1 raw split membership/order differs")
    if context["transformed_hashes"] != EXPECTED_ACTIVE_V1_TRANSFORMED_HASHES:
        raise RuntimeError("Reconstructed active-v1 transformed split membership/order differs")
    if not np.array_equal(
        np.asarray(report.get("scaler_mean"), dtype=np.float64),
        np.asarray(context["scaler"].mean_, dtype=np.float64),
    ):
        raise RuntimeError("RF cache scaler mean is not the recomputed train-only mean")
    if not np.array_equal(
        np.asarray(report.get("scaler_scale"), dtype=np.float64),
        np.asarray(context["scaler"].scale_, dtype=np.float64),
    ):
        raise RuntimeError("RF cache scaler scale is not the recomputed train-only scale")

    probabilities = np.load(cache_path, allow_pickle=False)
    expected_shape = (len(context["split"]["train_indices"]), len(CLASS_NAMES))
    if probabilities.shape != expected_shape or probabilities.dtype != np.float32:
        raise RuntimeError(
            f"RF cache shape/dtype is {probabilities.shape}/{probabilities.dtype}, "
            f"expected {expected_shape}/float32"
        )
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise RuntimeError("RF cache contains non-finite or negative probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6):
        raise RuntimeError("RF cache probability rows do not sum to one")
    return probabilities, {
        "source_type": "bound_active_v1_calibrated_rf_soft_targets",
        "rf_seed": 42,
        "rf_config": RF_CONFIG,
        "cache_path_recorded": str(cache_path.resolve()),
        "cache_file_sha256": cache_file_sha256,
        "expected_cache_file_sha256": EXPECTED_DEPLOYMENT_RF_CACHE_SHA256,
        "preprocessing_report_path_recorded": str(preprocessing_report_path.resolve()),
        "preprocessing_report_sha256": sha256_file(preprocessing_report_path),
        "execution_manifest_path_recorded": str(execution_manifest_path.resolve()),
        "execution_manifest_sha256": sha256_file(execution_manifest_path),
        "executed_source_path_recorded": str(source_snapshot_path.resolve()),
        "executed_source_sha256": source_sha256,
        "active_protocol_id": manifest["protocol_id"],
        "verified_split_hashes": context["split_hashes"],
        "verified_transformed_split_hashes": context["transformed_hashes"],
        "verified_sklearn_version": sklearn.__version__,
        "cache_content_sha256": sha256_arrays(probabilities),
    }


def execution_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_resume_root(
    root: Path,
    context: dict[str, Any],
    expected_execution: dict[str, Any],
) -> None:
    execution_path = root / "execution_contract.json"
    preprocessing_path = root / "preprocessing_contract.json"
    indices_path = root / "split_indices.npz"
    scaler_path = root / "scaler_parameters.npz"
    for path in [execution_path, preprocessing_path, indices_path, scaler_path]:
        if not path.is_file():
            raise RuntimeError(f"Resume root is missing its immutable contract: {path}")
    observed_execution = json.loads(execution_path.read_text(encoding="utf-8"))
    if observed_execution != expected_execution:
        raise RuntimeError(
            "Resume execution contract differs from the current code/configuration"
        )
    preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    required = {
        "protocol_id": expected_execution["protocol_id"],
        "dataset_sha256": context["dataset"]["dataset_sha256"],
        "feature_names": context["dataset"]["feature_names"],
        "class_names": CLASS_NAMES,
        "split_policy": context["split"]["policy"],
        "split_hashes": context["split_hashes"],
        "split_indices_sha256": context["split_indices_sha256"],
        "scaler_sha256": context["scaler_sha256"],
        "transformed_split_hashes": context["transformed_hashes"],
        "feature_overlap_audit": context["split"]["group_audit"],
    }
    for key, value in required.items():
        if preprocessing.get(key) != value:
            raise RuntimeError(f"Resume preprocessing contract differs for {key}")
    if preprocessing.get("split_indices_file_sha256") != sha256_file(indices_path):
        raise RuntimeError("Resume split-index file hash differs")
    if preprocessing.get("scaler_parameters_file_sha256") != sha256_file(scaler_path):
        raise RuntimeError("Resume scaler file hash differs")
    split = context["split"]
    with np.load(indices_path, allow_pickle=False) as saved:
        for name in ["train", "validation", "test"]:
            if not np.array_equal(saved[f"{name}_indices"], split[f"{name}_indices"]):
                raise RuntimeError(f"Resume {name} indices differ")
    scaler = context["scaler"]
    with np.load(scaler_path, allow_pickle=False) as saved:
        for name, expected in [
            ("mean", scaler.mean_),
            ("scale", scaler.scale_),
            ("var", scaler.var_),
        ]:
            if not np.array_equal(saved[name], np.asarray(expected, dtype=np.float64)):
                raise RuntimeError(f"Resume scaler {name} differs")


def verify_completed_seed(seed_root: Path, expected: dict[str, Any]) -> bool:
    completion_path = seed_root / "seed_completion.json"
    manifest_path = seed_root / "artifact_manifest.json"
    if not completion_path.is_file() or not manifest_path.is_file():
        return False
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if completion.get(key) != value:
            raise RuntimeError(
                f"Resume contract mismatch for {seed_root}: {key} is "
                f"{completion.get(key)!r}, expected {value!r}"
            )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        return False
    if manifest.get("protocol_id") != expected["protocol_id"]:
        raise RuntimeError(f"Resume artifact protocol mismatch: {seed_root}")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError(f"Resume artifact manifest has no inventory: {seed_root}")
    if manifest.get("file_count_excluding_manifest") != len(files):
        raise RuntimeError(f"Resume artifact manifest count mismatch: {seed_root}")
    seen: set[str] = set()
    for item in files:
        relative = item.get("path")
        if not isinstance(relative, str) or not relative:
            raise RuntimeError(f"Resume artifact manifest has an invalid path: {seed_root}")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(f"Resume artifact path escapes its seed root: {relative!r}")
        normalized = relative_path.as_posix()
        if normalized in seen:
            raise RuntimeError(f"Resume artifact path is duplicated: {relative!r}")
        seen.add(normalized)
        path = (seed_root / relative_path).resolve()
        try:
            path.relative_to(seed_root.resolve())
        except ValueError as exc:
            raise RuntimeError(f"Resume artifact path escapes its seed root: {relative!r}") from exc
        if (
            not path.is_file()
            or path.stat().st_size != item.get("size_bytes")
            or sha256_file(path) != item.get("sha256")
        ):
            raise RuntimeError(f"Resume artifact failed SHA-256 verification: {path}")
    actual = {
        path.relative_to(seed_root).as_posix()
        for path in seed_root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != seen:
        raise RuntimeError(f"Resume artifact inventory differs from files on disk: {seed_root}")
    return True


def save_student_artifacts(
    seed_root: Path,
    protocol_id: str,
    student_name: str,
    route: str,
    seed: int,
    hidden_dims: tuple[int, int],
    model: StudentMLP,
    initial_state_sha256: str,
    metrics: dict[str, Any],
    probabilities: np.ndarray,
    context: dict[str, Any],
    teacher_soft_target_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    route_label = "Small_MLP_scratch" if route == "scratch" else "KD_from_RF"
    prefix = f"{student_name}_{route_label}"
    state = cpu_state_dict(model)
    plain_path = seed_root / f"{prefix}_fp32.pt"
    rich_path = seed_root / f"{prefix}_artifact.pt"
    atomic_torch_save(plain_path, state)
    atomic_torch_save(rich_path, {
        "protocol_id": protocol_id,
        "seed": seed,
        "student": student_name,
        "route": route,
        "input_dim": 17,
        "hidden_dims": list(hidden_dims),
        "num_classes": len(CLASS_NAMES),
        "feature_names": context["dataset"]["feature_names"],
        "class_names": CLASS_NAMES,
        "dataset_sha256": context["dataset"]["dataset_sha256"],
        "split_hashes": context["split_hashes"],
        "scaler_sha256": context["scaler_sha256"],
        "feature_overlap_audit": context["split"]["group_audit"],
        "initial_state_sha256": initial_state_sha256,
        "trained_state_sha256": state_dict_sha256(state),
        "kd_hyperparameters": ({"T": KD_T, "alpha": KD_ALPHA} if route == "rf_kd" else None),
        "teacher_soft_target_provenance": (
            teacher_soft_target_provenance if route == "rf_kd" else None
        ),
        "training_config": TRAIN_CONFIG,
        "metrics": metrics,
        "state_dict": state,
    })
    prediction_path = seed_root / f"{prefix}_test_predictions.csv"
    atomic_write_csv(prediction_path, prediction_frame(
        context["split"]["test_indices"],
        context["split"]["y_test"],
        probabilities,
    ))
    return {
        "route": route,
        "plain_state_dict": plain_path.name,
        "plain_state_dict_sha256": sha256_file(plain_path),
        "rich_artifact": rich_path.name,
        "rich_artifact_sha256": sha256_file(rich_path),
        "test_predictions": prediction_path.name,
        "test_predictions_sha256": sha256_file(prediction_path),
        "initial_state_sha256": initial_state_sha256,
        "trained_state_sha256": state_dict_sha256(state),
        "metrics": metrics,
    }


def run_seed(
    root: Path,
    protocol_id: str,
    context: dict[str, Any],
    seed: int,
    device: torch.device,
    include_scratch: bool,
    save_teacher: bool,
    resume: bool,
    execution_contract_sha256: str,
    bound_rf_train_probs: np.ndarray | None = None,
    bound_rf_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    seed_root = root / f"seed_{seed}"
    expected = {
        "protocol_id": protocol_id,
        "seed": seed,
        "dataset_sha256": context["dataset"]["dataset_sha256"],
        "split_indices_sha256": context["split_indices_sha256"],
        "scaler_sha256": context["scaler_sha256"],
        "execution_contract_sha256": execution_contract_sha256,
    }
    if seed_root.exists() and any(seed_root.iterdir()):
        if resume and verify_completed_seed(seed_root, expected):
            return json.loads((seed_root / "seed_completion.json").read_text(encoding="utf-8"))
        if not resume:
            raise FileExistsError(f"Refusing to overwrite incomplete seed directory: {seed_root}")
        failed_root = root / "failed_seed_attempts"
        failed_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        preserved = failed_root / f"seed_{seed}_{timestamp}"
        os.replace(seed_root, preserved)
    seed_root.mkdir(parents=True, exist_ok=False)

    scaled = context["scaled"]
    split = context["split"]
    X_train = scaled["X_train"]
    X_validation = scaled["X_validation"]
    X_test = scaled["X_test"]
    y_train = split["y_train"]
    y_validation = split["y_validation"]
    y_test = split["y_test"]
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_validation_t = torch.from_numpy(X_validation)
    y_validation_t = torch.from_numpy(y_validation)
    X_test_t = torch.from_numpy(X_test)
    weights = class_weights(y_train)

    started = time.time()
    teacher = None
    teacher_metrics = None
    teacher_seconds = 0.0
    teacher_soft_target_provenance = bound_rf_provenance
    if bound_rf_train_probs is None:
        teacher_started = time.time()
        calibration_groups = (
            verified_feature_hashes(split["X_train_raw"])
            if protocol_id == GROUP_PROTOCOL
            else None
        )
        teacher, calibration_audit = fit_calibrated_rf(
            X_train,
            y_train,
            seed,
            groups=calibration_groups,
        )
        teacher_seconds = time.time() - teacher_started
        rf_train_probs = teacher.predict_proba(X_train).astype(np.float32, copy=False)
        rf_test_probs = teacher.predict_proba(X_test).astype(np.float32, copy=False)
        teacher_metrics = classification_metrics(y_test, rf_test_probs)
        teacher_soft_target_provenance = {
            "source_type": "fresh_calibrated_rf_soft_targets",
            "rf_seed": seed,
            "rf_config": RF_CONFIG,
            "calibration_audit": calibration_audit,
            "train_probability_content_sha256": sha256_arrays(rf_train_probs),
        }
        atomic_write_csv(seed_root / "RF_teacher_test_predictions.csv", prediction_frame(
            split["test_indices"], y_test, rf_test_probs
        ))
    else:
        if bound_rf_provenance is None:
            raise RuntimeError("Bound RF probabilities require provenance")
        rf_train_probs = np.asarray(bound_rf_train_probs, dtype=np.float32)
    atomic_save_npy(seed_root / "rf_train_probabilities.npy", rf_train_probs)
    if save_teacher and teacher is not None:
        atomic_joblib_dump(seed_root / "calibrated_rf_teacher.joblib", teacher)

    student_results: dict[str, Any] = {}
    for student_name, hidden_dims in STUDENT_SPECS.items():
        initial_hashes: dict[str, str] = {}
        if include_scratch:
            set_seed(seed)
            scratch = StudentMLP(17, hidden_dims, len(CLASS_NAMES))
            initial_hashes["scratch"] = state_dict_sha256(cpu_state_dict(scratch))
            scratch = train_standard(
                scratch,
                X_train_t,
                y_train_t,
                X_validation_t,
                y_validation_t,
                weights,
                device,
            )
            scratch_probs = batched_probs(scratch, X_test_t, device)
            student_results[f"{student_name}_scratch"] = save_student_artifacts(
                seed_root,
                protocol_id,
                student_name,
                "scratch",
                seed,
                hidden_dims,
                scratch,
                initial_hashes["scratch"],
                classification_metrics(y_test, scratch_probs),
                scratch_probs,
                context,
                None,
            )
            del scratch

        set_seed(seed)
        kd_student = StudentMLP(17, hidden_dims, len(CLASS_NAMES))
        initial_hashes["rf_kd"] = state_dict_sha256(cpu_state_dict(kd_student))
        if include_scratch and initial_hashes["scratch"] != initial_hashes["rf_kd"]:
            raise RuntimeError(f"Scratch/KD initialization mismatch for {student_name}, seed {seed}")
        kd_student = train_rf_kd(
            kd_student,
            rf_train_probs,
            X_train_t,
            y_train_t,
            X_validation_t,
            y_validation_t,
            weights,
            device,
        )
        kd_probs = batched_probs(kd_student, X_test_t, device)
        student_results[f"{student_name}_rf_kd"] = save_student_artifacts(
            seed_root,
            protocol_id,
            student_name,
            "rf_kd",
            seed,
            hidden_dims,
            kd_student,
            initial_hashes["rf_kd"],
            classification_metrics(y_test, kd_probs),
            kd_probs,
            context,
            teacher_soft_target_provenance,
        )
        del kd_student
        if device.type == "cuda":
            torch.cuda.empty_cache()

    completion = {
        **expected,
        "status": "complete",
        "teacher_config": RF_CONFIG,
        "teacher_metrics": teacher_metrics,
        "teacher_fit_seconds": teacher_seconds,
        "teacher_soft_target_provenance": teacher_soft_target_provenance,
        "student_results": student_results,
        "wall_seconds": time.time() - started,
    }
    atomic_write_json(seed_root / "seed_completion.json", completion)
    atomic_write_json(seed_root / "artifact_manifest.json", artifact_manifest(
        seed_root, protocol_id, "complete"
    ))
    return completion


def aggregate_results(
    root: Path,
    protocol_id: str,
    completions: list[dict[str, Any]],
) -> Path:
    route_names = sorted(completions[0]["student_results"])
    aggregate: dict[str, Any] = {}
    for route_name in route_names:
        rows = [item["student_results"][route_name]["metrics"] for item in completions]
        aggregate[route_name] = {}
        for metric in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]:
            values = np.asarray([row[metric] for row in rows], dtype=np.float64)
            aggregate[route_name][metric] = {
                "values": values.tolist(),
                "mean": float(values.mean()),
                "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
            }
        per_class = np.asarray([row["per_class_f1"] for row in rows], dtype=np.float64)
        aggregate[route_name]["per_class_f1"] = {
            "class_names": CLASS_NAMES,
            "values": per_class.tolist(),
            "mean": per_class.mean(axis=0).tolist(),
            "sample_std": (
                per_class.std(axis=0, ddof=1).tolist()
                if len(per_class) > 1
                else np.zeros(per_class.shape[1]).tolist()
            ),
        }
    paired_differences: dict[str, Any] = {}
    for student_name in STUDENT_SPECS:
        scratch_key = f"{student_name}_scratch"
        kd_key = f"{student_name}_rf_kd"
        if scratch_key not in route_names or kd_key not in route_names:
            continue
        paired_differences[student_name] = {}
        for metric in ["accuracy", "macro_f1"]:
            differences = np.asarray([
                item["student_results"][kd_key]["metrics"][metric]
                - item["student_results"][scratch_key]["metrics"][metric]
                for item in completions
            ], dtype=np.float64)
            paired_differences[student_name][f"rf_kd_minus_scratch_{metric}"] = {
                "values": differences.tolist(),
                "mean": float(differences.mean()),
                "sample_std": (
                    float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
                ),
            }
    path = root / "aggregate_results.json"
    atomic_write_json(path, {
        "protocol_id": protocol_id,
        "status": "complete",
        "seeds": [item["seed"] for item in completions],
        "seed_count": len(completions),
        "aggregate": aggregate,
        "paired_differences": paired_differences,
        "inference_boundary": (
            "Five-seed exact-feature-group sensitivity summaries are descriptive; "
            "they are not a matched causal ablation against the archived random-row "
            "route and make no statistical-significance claim."
            if protocol_id == GROUP_PROTOCOL
            else
            "One-seed deployment training is an artifact-generation route, not a "
            "multi-seed statistical estimate or an exact recovery of active-v1 weights."
        ),
    })
    return path


def run_training_mode(
    mode: str,
    dataset: dict[str, Any],
    output_root: Path,
    device: torch.device,
    resume: bool,
    deployment_rf_cache: Path,
    deployment_preprocessing_report: Path,
    deployment_execution_manifest: Path,
    deployment_source_snapshot: Path,
) -> Path:
    if mode == "deployment":
        root = output_root / "deployment_seed_42"
        protocol_id = DEPLOYMENT_PROTOCOL
        split = archived_random_split(dataset["features"], dataset["labels"])
        seeds = [42]
        include_scratch = False
        save_teacher = False
    else:
        root = output_root / "feature_group_5seed"
        protocol_id = GROUP_PROTOCOL
        split = feature_group_split(dataset["features"], dataset["labels"])
        seeds = SENSITIVITY_SEEDS
        include_scratch = True
        save_teacher = False

    root_had_content = root.exists() and any(root.iterdir())
    ensure_new_or_resume(root, resume)
    context = build_data_context(dataset, split)
    bound_rf_train_probs = None
    bound_rf_provenance = None
    if mode == "deployment":
        bound_rf_train_probs, bound_rf_provenance = load_bound_deployment_cache(
            context,
            deployment_rf_cache.resolve(),
            deployment_preprocessing_report.resolve(),
            deployment_execution_manifest.resolve(),
            deployment_source_snapshot.resolve(),
        )
    contract_without_fingerprint = {
        "protocol_id": protocol_id,
        "mode": mode,
        "script_sha256": sha256_file(SCRIPT_PATH),
        "common_module_sha256": sha256_file(SCRIPT_DIR / "tier15_common.py"),
        "dataset_sha256": dataset["dataset_sha256"],
        "split_indices_sha256": context["split_indices_sha256"],
        "scaler_sha256": context["scaler_sha256"],
        "seeds": seeds,
        "students": {name: list(dims) for name, dims in STUDENT_SPECS.items()},
        "routes": (["scratch", "rf_kd"] if include_scratch else ["rf_kd"]),
        "kd_hyperparameters": {"T": KD_T, "alpha": KD_ALPHA},
        "kd_hyperparameter_source": (
            "fixed values selected once by the active-v1 preliminary MLP-teacher "
            "validation grid; candidate initialization/shuffle streams were not held "
            "constant, so this is not an RF-KD optimum; no confirmation-test retuning"
        ),
        "training_config": TRAIN_CONFIG,
        "teacher_config": RF_CONFIG,
        "teacher_calibration_strategy": (
            "stratified_group_kfold_with_zero_exact_feature_group_overlap"
            if protocol_id == GROUP_PROTOCOL
            else "bound_preserved_active_v1_calibrated_rf_targets"
        ),
        "bound_teacher_soft_target_provenance": bound_rf_provenance,
        "environment": environment_record(device),
    }
    execution_contract = {
        **contract_without_fingerprint,
        "execution_fingerprint_sha256": execution_fingerprint(
            contract_without_fingerprint
        ),
    }
    if root_had_content:
        verify_resume_root(root, context, execution_contract)
    else:
        write_data_contract(root, context, protocol_id)
        atomic_write_json(root / "execution_contract.json", execution_contract)
    execution_contract_sha256 = sha256_file(root / "execution_contract.json")

    completions = [
        run_seed(
            root,
            protocol_id,
            context,
            seed,
            device,
            include_scratch,
            save_teacher,
            resume,
            execution_contract_sha256,
            bound_rf_train_probs,
            bound_rf_provenance,
        )
        for seed in seeds
    ]
    result_path = aggregate_results(root, protocol_id, completions)
    atomic_write_json(root / "artifact_manifest.json", artifact_manifest(
        root, protocol_id, "complete"
    ))
    return result_path


def main() -> int:
    args = parse_args()
    if args.mode != "preflight" and not args.confirm_training:
        raise RuntimeError(
            f"Mode {args.mode!r} trains models. Re-run with --confirm-training "
            "after the active WSN-DS process has finished."
        )
    if args.mode != "preflight" and active_v1_worker_is_running():
        raise RuntimeError(
            "The preserved active WSN-DS worker is still running. Confirmation "
            "training is blocked until it exits."
        )
    dataset = dataset_with_recorded_path(args.dataset_csv)
    output_root = require_disjoint_output(args.output_root, [
        ACTIVE_V1_ROOT,
        REPO_ROOT / "experiments",
        REPO_ROOT / "deployment",
        REPO_ROOT / "data",
        REPO_ROOT / "manuscript",
        REPO_ROOT / "tests",
        REPO_ROOT / ".git",
    ])
    if args.mode == "preflight":
        print(preflight_report(dataset, output_root, args.resume))
        return 0
    device = resolve_device(args.device)
    set_seed(0)
    print(run_training_mode(
        args.mode,
        dataset,
        output_root,
        device,
        args.resume,
        args.deployment_rf_cache,
        args.deployment_preprocessing_report,
        args.deployment_execution_manifest,
        args.deployment_source_snapshot,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
