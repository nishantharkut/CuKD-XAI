"""Post-process the completed controlled WSN-DS full-route experiment.

This program is read-only with respect to its dataset and source experiment
roots. It validates the preserved row-level evidence before calculating two
within-test repeated-pattern sensitivity views:

1. Every row is retained and receives weight 1 / test-group-size.
2. One deterministic row is retained per label-pure exact feature group.
   Mixed-label groups are excluded and enumerated without majority labelling.

The default action is a non-writing preflight. Analysis output is created only
with ``--confirm-analysis``. An existing output is refused unless ``--resume``
is supplied and the output is already complete and exactly reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import sklearn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder, StandardScaler


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_DATASET = REPO_ROOT / "data" / "wsnds" / "WSN-DS.csv"
DEFAULT_BASE_ROOT = (
    REPO_ROOT
    / "results"
    / "wsnds"
    / "confirmation_runs_v2"
    / "local_feature_group_10seed_20260811"
    / "feature_group_10seed"
)
DEFAULT_FULL_ROOT = (
    REPO_ROOT
    / "results"
    / "wsnds"
    / "evidence_completion_20260811"
    / "fgds_controlled_full_routes_10seed_v2"
)
DEFAULT_FULL_SOURCE = DEFAULT_FULL_ROOT / "executed_full_routes_source.py"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results"
    / "wsnds"
    / "evidence_completion_20260811"
    / "fgds_group_balanced_routes_10seed_v1"
)

PROTOCOL_ID = "wsnds_fgds_group_balanced_route_sensitivity_10seed_v1"
BASE_PROTOCOL_ID = "wsnds_feature_group_split_train_only_scaler_10seed_v2"
FULL_PROTOCOL_ID = "wsnds_feature_group_disjoint_controlled_full_routes_10seed_v2"
FINALIZER_PROTOCOL_ID = "wsnds_fgds_controlled_full_routes_float32_ece_finalizer_v1"
CONTINUATION_PROTOCOL_ID = "wsnds_fgds_full_routes_dtype_faithful_continuation_v1"
FINALIZER_SCHEMA_VERSION = 4
SEEDS = [42, 123, 456, 789, 1001, 2024, 3141, 5678, 8192, 9999]
CLASS_NAMES = ["Blackhole", "Flooding", "Grayhole", "Normal", "TDMA"]
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
STUDENTS = ["student_A", "student_B"]
TEACHER_ROUTES = [
    "A_RF_500_uncalibrated",
    "A_calibrated_RF_KD_teacher",
    "B_Full_MLP",
    "C_CL_MLP_loss_fair",
    "C_CL_MLP_loss_ext",
    "C2_CL_MLP_domain",
    "G_random_pacing_teacher",
    "I_SMOTE_MLP_teacher",
]
TRAINED_TEACHER_ROUTES = [
    "B_Full_MLP",
    "C_CL_MLP_loss_fair",
    "C_CL_MLP_loss_ext",
    "C2_CL_MLP_domain",
    "G_random_pacing_teacher",
    "I_SMOTE_MLP_teacher",
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
TRAINED_STUDENT_ROUTES = STUDENT_ROUTES[2:]
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
VIEWS = ["row_level", "inverse_test_group_size", "pure_group_representative"]
SENSITIVITY_VIEWS = VIEWS[1:]
PRIMARY_INFERENCE_POLICY = {
    "view": "row_level",
    "test": "exact paired Wilcoxon signed-rank test",
    "multiplicity_control": (
        "Holm adjustment within each predeclared route family: teacher, "
        "Student A, and Student B"
    ),
    "rationale": (
        "The row-level view preserves the benchmark's empirical test-record "
        "distribution. Equal-weight exact-group and pure representative views are "
        "reported as repeated-pattern sensitivity analyses."
    ),
}
EXPECTED_TEST_ROWS = 56_301
ROW_METRIC_ATOL = 2e-9
PROBABILITY_SUM_ATOL = 2e-6
WITHIN_GROUP_PROBABILITY_ATOL = 2e-6

OUTPUT_FILES = {
    "execution_contract.json",
    "bound_analyzer_source.py",
    "bound_full_route_executed_source.py",
    "bound_shared_data_contract_source.py",
    "test_group_assignments.npz",
    "test_group_summary.json",
    "mixed_label_groups.csv",
    "mixed_label_rows.csv",
    "route_seed_metrics.csv",
    "route_seed_per_class_f1.csv",
    "route_aggregate.csv",
    "route_per_class_f1_aggregate.csv",
    "sensitivity_shifts.csv",
    "route_pair_deltas.csv",
    "paired_tests.csv",
    "aggregate_results.json",
    "analysis_summary.md",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE_ROOT)
    parser.add_argument("--full-route-root", type=Path, default=DEFAULT_FULL_ROOT)
    parser.add_argument(
        "--full-route-executed-source", type=Path, default=DEFAULT_FULL_SOURCE
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate completed inputs and compute no output files.",
    )
    parser.add_argument(
        "--confirm-analysis",
        action="store_true",
        help="Create the additive post-processing result directory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Validate and reuse only an already complete identical output.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run isolated synthetic tests without reading real experiment data.",
    )
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON encode {type(value)!r}")


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=json_default
    ).encode("utf-8")


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


def atomic_write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, default=json_default)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    os.replace(temporary, path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def manifest_payload(root: Path, protocol_id: str, status: str) -> dict[str, Any]:
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


def verify_manifest(
    root: Path,
    expected_protocol: str,
    *,
    expected_inventory: set[str] | None = None,
) -> dict[str, Any]:
    manifest_path = root / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            f"Completed artifact manifest is absent; refusing incomplete input: {root}"
        )
    manifest = read_json(manifest_path)
    if manifest.get("protocol_id") != expected_protocol:
        raise RuntimeError(f"Manifest protocol differs: {manifest_path}")
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Manifest is not complete: {manifest_path}")
    items = manifest.get("files")
    if not isinstance(items, list):
        raise RuntimeError(f"Manifest file list is invalid: {manifest_path}")
    declared = {item.get("path"): item for item in items}
    if None in declared or len(declared) != len(items):
        raise RuntimeError(f"Manifest paths are missing or duplicated: {manifest_path}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path and not path.name.endswith(".tmp")
    }
    if set(declared) != actual:
        raise RuntimeError(f"Manifest inventory differs from disk: {manifest_path}")
    if expected_inventory is not None and actual != expected_inventory:
        raise RuntimeError(f"Output inventory differs from the fixed schema: {root}")
    if manifest.get("file_count_excluding_manifest") != len(actual):
        raise RuntimeError(f"Manifest file count differs: {manifest_path}")
    for relative, item in declared.items():
        path = root / relative
        if path.stat().st_size != item.get("size_bytes"):
            raise RuntimeError(f"Manifest size differs: {path}")
        if sha256_file(path) != item.get("sha256"):
            raise RuntimeError(f"Manifest hash differs: {path}")
    return {
        "payload": manifest,
        "sha256": sha256_file(manifest_path),
        "declared": declared,
    }


def expected_base_seed_inventory() -> set[str]:
    files = {
        "seed_completion.json",
        "RF_teacher_test_predictions.csv",
        "rf_train_probabilities.npy",
    }
    for student in STUDENTS:
        for route_name in ["Small_MLP_scratch", "KD_from_RF"]:
            files.update(
                {
                    f"{student}_{route_name}_fp32.pt",
                    f"{student}_{route_name}_artifact.pt",
                    f"{student}_{route_name}_test_predictions.csv",
                }
            )
    return files


def expected_base_root_inventory() -> set[str]:
    files = {
        "aggregate_results.json",
        "execution_contract.json",
        "preprocessing_contract.json",
        "scaler_parameters.npz",
        "split_indices.npz",
    }
    seed_files = expected_base_seed_inventory() | {"artifact_manifest.json"}
    for seed in SEEDS:
        files.update(f"seed_{seed}/{name}" for name in seed_files)
    return files


def expected_full_seed_inventory() -> set[str]:
    files = {
        "seed_completion.json",
        "teacher_A_RF_500_uncalibrated.joblib",
        "teacher_A_RF_500_uncalibrated_test_predictions.npz",
    }
    for route in TRAINED_TEACHER_ROUTES:
        files.update(
            {
                f"teacher_{route}.pt",
                f"teacher_{route}_test_predictions.npz",
            }
        )
    for student in STUDENTS:
        for route in TRAINED_STUDENT_ROUTES:
            files.update(
                {
                    f"{student}_{route}.pt",
                    f"{student}_{route}_test_predictions.npz",
                }
            )
    return files


def expected_full_root_inventory() -> set[str]:
    files = {
        "aggregate_results.json",
        "bound_tier15_common.py",
        "continuation_attempt_contract.json",
        "executed_full_routes_finalizer.py",
        "executed_full_routes_source.py",
        "execution_contract.json",
        "finalization_contract.json",
    }
    seed_files = expected_full_seed_inventory() | {"artifact_manifest.json"}
    for seed in SEEDS:
        files.update(f"seed_{seed}/{name}" for name in seed_files)
    return files


def resolve_recorded_repo_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"Recorded {label} path is invalid")
    candidate = Path(value)
    resolved = candidate.resolve() if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Recorded {label} path escapes the repository: {value}") from exc
    return resolved


def validate_finalized_full_root(
    full_root: Path,
    full_execution_path: Path,
    full_aggregate_path: Path,
    full_source_path: Path,
    common_source_path: Path,
    full_execution: dict[str, Any],
) -> dict[str, Any]:
    contract_path = full_root / "finalization_contract.json"
    runner_snapshot = full_root / "executed_full_routes_source.py"
    common_snapshot = full_root / "bound_tier15_common.py"
    finalizer_snapshot = full_root / "executed_full_routes_finalizer.py"
    attempt_path = full_root / "continuation_attempt_contract.json"
    for path in [
        contract_path,
        runner_snapshot,
        common_snapshot,
        finalizer_snapshot,
        attempt_path,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    finalization = read_json(contract_path)
    required = {
        "schema_version": FINALIZER_SCHEMA_VERSION,
        "protocol_id": FINALIZER_PROTOCOL_ID,
        "status": "complete",
        "training_performed": False,
        "primary_protocol_id": FULL_PROTOCOL_ID,
        "execution_contract_sha256": sha256_file(full_execution_path),
        "executed_runner_sha256": sha256_file(runner_snapshot),
        "bound_common_module_sha256": sha256_file(common_snapshot),
        "finalizer_source_sha256": sha256_file(finalizer_snapshot),
        "aggregate_results_sha256": sha256_file(full_aggregate_path),
    }
    for key, expected in required.items():
        if finalization.get(key) != expected:
            raise RuntimeError(f"Finalization contract differs for {key}")

    correction = finalization.get("correction")
    if not isinstance(correction, dict):
        raise RuntimeError("Finalization correction audit is absent")
    if correction.get("field") != "ece_15_bin":
        raise RuntimeError("Finalization correction field differs")
    if correction.get("model_training_or_prediction_changed") is not False:
        raise RuntimeError("Finalization unexpectedly changed training or predictions")
    if correction.get("non_ece_metrics_changed") is not False:
        raise RuntimeError("Finalization unexpectedly changed non-ECE metrics")
    roundtrip = correction.get("roundtrip_audit")
    if not isinstance(roundtrip, dict):
        raise RuntimeError("Finalization round-trip audit is absent")
    expected_roundtrip = {
        "neural_artifact_count": 180,
        "neural_float32_max_abs_delta": 0.0,
        "rf_artifact_count": 10,
        "rf_persisted_roundtrip_max_abs_delta": 0.0,
    }
    for key, expected in expected_roundtrip.items():
        if roundtrip.get(key) != expected:
            raise RuntimeError(f"Finalization round-trip audit differs for {key}")

    inference_correction = finalization.get("inference_correction")
    if not isinstance(inference_correction, dict):
        raise RuntimeError("Finalization inference correction is absent")
    if inference_correction.get("finalized_method") != "exact_signed_rank_enumeration":
        raise RuntimeError("Finalization inference method differs")
    if inference_correction.get("training_predictions_or_per_seed_metrics_changed") is not False:
        raise RuntimeError("Finalization unexpectedly changed per-seed evidence")
    inference_audit = inference_correction.get("audit")
    if not isinstance(inference_audit, dict):
        raise RuntimeError("Finalization inference audit is absent")
    if inference_audit.get("test_count") != 26:
        raise RuntimeError("Finalization inferential test count differs")
    if inference_audit.get("maximum_enumerated_sign_assignments") != 1024:
        raise RuntimeError("Finalization exact-enumeration scope differs")

    if sha256_file(full_source_path) != finalization["executed_runner_sha256"]:
        raise RuntimeError("Provided full-route source differs from the finalized snapshot")
    if full_execution.get("script_sha256") != finalization["executed_runner_sha256"]:
        raise RuntimeError("Full-route execution contract differs from the finalized runner")
    if sha256_file(common_source_path) != finalization["bound_common_module_sha256"]:
        raise RuntimeError("Current common source differs from the finalized snapshot")
    if full_execution.get("common_module_sha256") != finalization[
        "bound_common_module_sha256"
    ]:
        raise RuntimeError("Full-route execution contract differs from finalized common source")

    provenance = finalization.get("continuation_provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError("Finalization continuation provenance is absent")
    if provenance.get("protocol_id") != CONTINUATION_PROTOCOL_ID:
        raise RuntimeError("Continuation provenance protocol differs")
    if provenance.get("status") != "complete":
        raise RuntimeError("Continuation provenance is not complete")
    continuation_root = resolve_recorded_repo_path(
        provenance.get("evidence_root"), "continuation evidence root"
    )
    if not continuation_root.is_dir():
        raise FileNotFoundError(continuation_root)
    continuation_manifest = verify_manifest(
        continuation_root, CONTINUATION_PROTOCOL_ID
    )
    continuation_contract_path = continuation_root / "continuation_contract.json"
    if not continuation_contract_path.is_file():
        raise FileNotFoundError(continuation_contract_path)
    continuation = read_json(continuation_contract_path)
    if continuation.get("protocol_id") != CONTINUATION_PROTOCOL_ID:
        raise RuntimeError("Continuation contract protocol differs")
    if continuation.get("status") != "complete":
        raise RuntimeError("Continuation contract is not complete")
    if sha256_file(continuation_contract_path) != provenance.get(
        "continuation_contract_sha256"
    ):
        raise RuntimeError("Continuation contract hash differs from finalization")
    if continuation_manifest["sha256"] != provenance.get(
        "continuation_manifest_sha256"
    ):
        raise RuntimeError("Continuation manifest hash differs from finalization")
    if resolve_recorded_repo_path(continuation.get("output_root"), "continuation output root") != full_root:
        raise RuntimeError("Continuation output root differs from the finalized root")
    if continuation.get("completed_seeds_after_continuation") != SEEDS:
        raise RuntimeError("Continuation did not bind the complete publication seed set")
    if continuation.get("target_seeds") != [8192, 9999]:
        raise RuntimeError("Continuation target-seed contract differs")
    if continuation.get("aggregation_performed") is not False:
        raise RuntimeError("Continuation unexpectedly claims aggregate generation")

    attempt_sha256 = sha256_file(attempt_path)
    if attempt_sha256 != provenance.get("attempt_contract_sha256"):
        raise RuntimeError("Continuation attempt contract differs from finalization")
    if attempt_sha256 != continuation.get("attempt_contract_sha256"):
        raise RuntimeError("Continuation attempt contract differs from continuation evidence")
    attempt_snapshot = continuation_root / "continuation_attempt_contract.json"
    if sha256_file(attempt_snapshot) != attempt_sha256:
        raise RuntimeError("Continuation attempt snapshot differs")

    snapshots = continuation.get("source_snapshots")
    if not isinstance(snapshots, dict):
        raise RuntimeError("Continuation source-snapshot contract is absent")
    expected_snapshots = {
        "executed_continuation_source.py": continuation.get(
            "continuation_source_sha256"
        ),
        "bound_original_runner_source.py": finalization["executed_runner_sha256"],
        "bound_finalizer_source.py": finalization["finalizer_source_sha256"],
        "bound_common_source.py": finalization["bound_common_module_sha256"],
    }
    if not isinstance(expected_snapshots["executed_continuation_source.py"], str):
        raise RuntimeError("Continuation source hash is absent")
    if set(snapshots) != set(expected_snapshots):
        raise RuntimeError("Continuation source-snapshot inventory differs")
    for name, expected_hash in expected_snapshots.items():
        if snapshots.get(name) != expected_hash:
            raise RuntimeError(f"Continuation source record differs: {name}")
        if sha256_file(continuation_root / name) != expected_hash:
            raise RuntimeError(f"Continuation source snapshot differs: {name}")

    interrupted_manifest = resolve_recorded_repo_path(
        continuation.get("interrupted_attempt_archive_manifest"),
        "interrupted-attempt manifest",
    )
    if not interrupted_manifest.is_file():
        raise FileNotFoundError(interrupted_manifest)
    interrupted_hash = sha256_file(interrupted_manifest)
    if interrupted_hash != continuation.get("interrupted_attempt_archive_manifest_sha256"):
        raise RuntimeError("Interrupted-attempt manifest differs from continuation evidence")
    if interrupted_hash != provenance.get("interrupted_archive_manifest_sha256"):
        raise RuntimeError("Interrupted-attempt manifest differs from finalization")

    return {
        "payload": finalization,
        "sha256": sha256_file(contract_path),
        "continuation_root": continuation_root,
        "continuation_contract_sha256": sha256_file(continuation_contract_path),
        "continuation_manifest_sha256": continuation_manifest["sha256"],
        "interrupted_archive_manifest_sha256": interrupted_hash,
    }


def load_dataset(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    target = next(
        (
            name
            for name in [
                "Attack type",
                "Attack_Type",
                "attack_type",
                "Attack Type",
                "class",
            ]
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
    feature_frame = frame.drop(columns=[target])
    if feature_frame.columns.tolist() != FEATURE_NAMES:
        raise RuntimeError("WSN-DS feature contract differs")
    if encoder.classes_.tolist() != CLASS_NAMES:
        raise RuntimeError("WSN-DS class contract differs")
    features = feature_frame.to_numpy(dtype=np.float32)
    if features.shape != (374_661, 17):
        raise RuntimeError(f"WSN-DS shape differs: {features.shape}")
    if not np.isfinite(features).all():
        raise RuntimeError("WSN-DS features contain non-finite values")
    return {
        "features": features,
        "labels": labels,
        "dataset_sha256": sha256_file(path),
        "target_column": target,
    }


def canonical_feature_rows(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError(f"Feature matrix shape is invalid: {values.shape}")
    if not np.isfinite(values).all():
        raise RuntimeError("Feature matrix contains non-finite values")
    canonical = np.array(values, dtype="<f4", order="C", copy=True)
    canonical[canonical == 0.0] = 0.0
    return canonical


def exact_row_keys(features: np.ndarray) -> np.ndarray:
    canonical = canonical_feature_rows(features)
    row_bytes = canonical.shape[1] * canonical.dtype.itemsize
    return canonical.view(np.dtype((np.void, row_bytes))).reshape(-1)


def reconstruct_split_and_scaler(
    dataset: dict[str, Any], base_root: Path, base_execution: dict[str, Any],
    preprocessing: dict[str, Any]
) -> dict[str, Any]:
    split_path = base_root / "split_indices.npz"
    scaler_path = base_root / "scaler_parameters.npz"
    if not split_path.is_file() or not scaler_path.is_file():
        raise RuntimeError("Base split or scaler artifact is absent")
    if sha256_file(split_path) != preprocessing.get("split_indices_file_sha256"):
        raise RuntimeError("Saved split-index file hash differs")
    if sha256_file(scaler_path) != preprocessing.get("scaler_parameters_file_sha256"):
        raise RuntimeError("Saved scaler file hash differs")
    with np.load(split_path, allow_pickle=False) as payload:
        if set(payload.files) != {
            "train_indices",
            "validation_indices",
            "test_indices",
        }:
            raise RuntimeError("Saved split index schema differs")
        split = {
            name: np.asarray(payload[f"{name}_indices"], dtype=np.int64)
            for name in ["train", "validation", "test"]
        }
    all_indices = np.concatenate([split[name] for name in ["train", "validation", "test"]])
    if len(all_indices) != len(dataset["labels"]):
        raise RuntimeError("Saved partitions do not cover the dataset row count")
    if np.any(all_indices < 0) or np.any(all_indices >= len(dataset["labels"])):
        raise RuntimeError("Saved split contains an out-of-range source index")
    if len(np.unique(all_indices)) != len(all_indices):
        raise RuntimeError("Saved partitions overlap or contain duplicate indices")
    if not np.array_equal(np.sort(all_indices), np.arange(len(all_indices), dtype=np.int64)):
        raise RuntimeError("Saved partitions do not cover every dataset row exactly once")
    if len(split["test"]) != EXPECTED_TEST_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_TEST_ROWS} test rows")
    index_hash = sha256_arrays(split["train"], split["validation"], split["test"])
    if index_hash != base_execution.get("split_indices_sha256"):
        raise RuntimeError("Split-index content hash differs from the base contract")
    if sha256_arrays(split["train"]) != preprocessing.get(
        "scaler_fit_indices_sha256"
    ):
        raise RuntimeError("Scaler fit-index hash differs from the preprocessing contract")
    if preprocessing.get("scaler_fit_partition") != "train only":
        raise RuntimeError("Scaler fit partition is not train-only")
    if preprocessing.get("scaler_fit_row_count") != len(split["train"]):
        raise RuntimeError("Scaler fit-row count differs from the preprocessing contract")

    features = dataset["features"]
    labels = dataset["labels"]
    observed_split_hashes = {
        name: sha256_arrays(features[split[name]], labels[split[name]])
        for name in ["train", "validation", "test"]
    }
    if observed_split_hashes != preprocessing.get("split_hashes"):
        raise RuntimeError("Raw split hashes differ from the preprocessing contract")

    all_keys = exact_row_keys(features)
    unique_keys = {name: np.unique(all_keys[split[name]]) for name in split}
    overlap = {
        "train_validation": int(
            len(np.intersect1d(unique_keys["train"], unique_keys["validation"]))
        ),
        "train_test": int(len(np.intersect1d(unique_keys["train"], unique_keys["test"]))),
        "validation_test": int(
            len(np.intersect1d(unique_keys["validation"], unique_keys["test"]))
        ),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Exact raw feature groups cross partitions: {overlap}")
    recorded_audit = preprocessing.get("feature_overlap_audit", {})
    if any(
        int(recorded_audit.get(key, -1)) != 0
        for key in [
            "train_validation_feature_overlap",
            "train_test_feature_overlap",
            "validation_test_feature_overlap",
        ]
    ):
        raise RuntimeError("Base preprocessing contract does not record zero group overlap")

    fitted_scaler = StandardScaler().fit(features[split["train"]])
    with np.load(scaler_path, allow_pickle=False) as saved:
        if set(saved.files) != {"mean", "scale", "var", "n_samples_seen"}:
            raise RuntimeError("Saved scaler schema differs")
        saved_scaler = {
            name: np.asarray(saved[name], dtype=np.float64)
            for name in ["mean", "scale", "var"]
        }
        n_samples_seen = np.asarray(saved["n_samples_seen"])
    if (
        n_samples_seen.dtype != np.int64
        or n_samples_seen.shape != (1,)
        or int(n_samples_seen[0]) != len(split["train"])
    ):
        raise RuntimeError("Saved scaler fit-row count differs")
    for name, observed in [
        ("mean", fitted_scaler.mean_),
        ("scale", fitted_scaler.scale_),
        ("var", fitted_scaler.var_),
    ]:
        if not np.array_equal(saved_scaler[name], np.asarray(observed, dtype=np.float64)):
            raise RuntimeError(f"Saved scaler {name} differs from a train-only refit")
    scaler_hash = sha256_arrays(
        saved_scaler["mean"], saved_scaler["scale"], saved_scaler["var"]
    )
    if scaler_hash != base_execution.get("scaler_sha256"):
        raise RuntimeError("Scaler content hash differs from the base contract")
    transformed_hashes = {}
    for name in ["train", "validation", "test"]:
        transformed = fitted_scaler.transform(features[split[name]]).astype(
            np.float32, copy=False
        )
        transformed_hashes[name] = sha256_arrays(transformed)
    if transformed_hashes != preprocessing.get("transformed_split_hashes"):
        raise RuntimeError("Transformed split hashes differ from the base contract")
    return {
        "indices": split,
        "index_hash": index_hash,
        "scaler_hash": scaler_hash,
        "split_hashes": observed_split_hashes,
        "transformed_hashes": transformed_hashes,
        "exact_overlap": overlap,
        "test_features": features[split["test"]],
        "test_labels": labels[split["test"]],
    }


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, sample_weight: np.ndarray | None = None
) -> float:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correctness = predictions == labels
    if sample_weight is None:
        value = 0.0
        edges = np.linspace(0.0, 1.0, 16)
        for index in range(15):
            mask = (confidence > edges[index]) & (confidence <= edges[index + 1])
            if index == 0:
                mask |= confidence == 0.0
            if not np.any(mask):
                continue
            value += float(mask.sum() / len(labels)) * abs(
                float(correctness[mask].mean()) - float(confidence[mask].mean())
            )
        return float(value)

    weights = np.asarray(sample_weight, dtype=np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        raise RuntimeError("Calibration weights have zero total")
    result = 0.0
    edges = np.linspace(0.0, 1.0, 16)
    for index in range(15):
        mask = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if index == 0:
            mask |= confidence == 0.0
        weight = float(weights[mask].sum())
        if weight == 0.0:
            continue
        accuracy = float(
            np.average(correctness[mask].astype(np.float64), weights=weights[mask])
        )
        mean_confidence = float(np.average(confidence[mask], weights=weights[mask]))
        result += (weight / total) * abs(accuracy - mean_confidence)
    return float(result)


def classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if labels.shape != predictions.shape or labels.ndim != 1:
        raise RuntimeError("Metric labels and predictions have incompatible shapes")
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    if weights is not None:
        if weights.shape != labels.shape or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise RuntimeError("Metric sample weights are invalid")
        denominator = float(weights.sum())
        accuracy = float(weights[predictions == labels].sum() / denominator)
    else:
        denominator = float(len(labels))
        accuracy = float(np.mean(predictions == labels))
    class_ids = np.arange(len(CLASS_NAMES), dtype=np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=class_ids,
        sample_weight=weights,
        zero_division=0,
    )
    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))
    matrix = confusion_matrix(
        labels, predictions, labels=class_ids, sample_weight=weights
    ).astype(np.float64)
    return {
        "row_count": int(len(labels)),
        "effective_weight": denominator,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class_precision": precision.astype(float).tolist(),
        "per_class_recall": recall.astype(float).tolist(),
        "per_class_f1": f1.astype(float).tolist(),
        "per_class_support": support.astype(float).tolist(),
        "confusion_matrix": matrix.tolist(),
    }


def validate_metrics(observed: dict[str, Any], expected: dict[str, Any], path: Path) -> None:
    required = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "per_class_precision",
        "per_class_recall",
        "per_class_f1",
        "per_class_support",
        "confusion_matrix",
    ]
    for key in required:
        if key not in expected:
            raise RuntimeError(f"Persisted metric {key} is absent: {path}")
        if not np.allclose(
            np.asarray(observed[key], dtype=np.float64),
            np.asarray(expected[key], dtype=np.float64),
            rtol=0.0,
            atol=ROW_METRIC_ATOL,
        ):
            raise RuntimeError(f"Persisted row metric differs for {key}: {path}")
    if "ece_15_bin" in expected:
        if "ece_15_bin" not in observed or not np.isclose(
            observed["ece_15_bin"], expected["ece_15_bin"], rtol=0.0, atol=ROW_METRIC_ATOL
        ):
            raise RuntimeError(f"Persisted row ECE differs: {path}")


def validate_prediction_arrays(
    path: Path,
    source_indices: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    expected_metrics: dict[str, Any],
    ece_probability_representation: str = "stored_dtype",
) -> dict[str, Any]:
    raw_source_indices = np.asarray(source_indices)
    raw_labels = np.asarray(labels)
    raw_predictions = np.asarray(predictions)
    source_indices = raw_source_indices.astype(np.int64)
    labels = raw_labels.astype(np.int64)
    probabilities = np.asarray(probabilities)
    if probabilities.dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise RuntimeError(f"Probability dtype differs: {path}")
    predictions = raw_predictions.astype(np.int64)
    if not np.array_equal(raw_source_indices, source_indices):
        raise RuntimeError(f"Source index dtype conversion changed values: {path}")
    if not np.array_equal(raw_labels, labels):
        raise RuntimeError(f"True-label dtype conversion changed values: {path}")
    if not np.array_equal(raw_predictions, predictions):
        raise RuntimeError(f"Predicted-label dtype conversion changed values: {path}")
    if probabilities.shape != (len(labels), len(CLASS_NAMES)):
        raise RuntimeError(f"Probability shape differs: {path}")
    if predictions.shape != labels.shape or source_indices.shape != labels.shape:
        raise RuntimeError(f"Prediction row shapes differ: {path}")
    if np.any(labels < 0) or np.any(labels >= len(CLASS_NAMES)):
        raise RuntimeError(f"True labels are outside the class contract: {path}")
    if np.any(predictions < 0) or np.any(predictions >= len(CLASS_NAMES)):
        raise RuntimeError(f"Predicted labels are outside the class contract: {path}")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
        raise RuntimeError(f"Probabilities are invalid: {path}")
    if not np.allclose(
        probabilities.sum(axis=1), 1.0, rtol=0.0, atol=PROBABILITY_SUM_ATOL
    ):
        raise RuntimeError(f"Probabilities do not sum to one: {path}")
    if not np.array_equal(predictions, probabilities.argmax(axis=1)):
        raise RuntimeError(f"Persisted predictions differ from argmax: {path}")
    if ece_probability_representation == "stored_dtype":
        ece_probabilities = probabilities
    elif ece_probability_representation == "stored_float32_promoted_to_float64":
        if probabilities.dtype != np.float32:
            raise RuntimeError(f"RF probability dtype differs: {path}")
        ece_probabilities = probabilities.astype(np.float64)
    else:
        raise RuntimeError(
            f"Unknown ECE probability representation: {ece_probability_representation}"
        )
    metrics = classification_metrics(labels, predictions)
    metrics["ece_15_bin"] = expected_calibration_error(ece_probabilities, labels)
    validate_metrics(metrics, expected_metrics, path)
    return {
        "source_indices": source_indices,
        "labels": labels,
        "probabilities": probabilities,
        "predictions": predictions,
        "row_metrics": metrics,
    }


def load_prediction_csv(
    path: Path,
    expected_indices: np.ndarray,
    expected_labels: np.ndarray,
    expected_metrics: dict[str, Any],
) -> dict[str, Any]:
    frame = pd.read_csv(path)
    probability_columns = [
        f"probability_{index}_{name}" for index, name in enumerate(CLASS_NAMES)
    ]
    expected_columns = [
        "source_row_index",
        "true_label",
        "predicted_label",
        *probability_columns,
    ]
    if frame.columns.tolist() != expected_columns:
        raise RuntimeError(f"Prediction CSV schema differs: {path}")
    result = validate_prediction_arrays(
        path,
        frame["source_row_index"].to_numpy(),
        frame["true_label"].to_numpy(),
        frame[probability_columns].to_numpy(dtype=np.float64),
        frame["predicted_label"].to_numpy(),
        expected_metrics,
    )
    if not np.array_equal(result["source_indices"], expected_indices):
        raise RuntimeError(f"Prediction CSV source indices differ: {path}")
    if not np.array_equal(result["labels"], expected_labels):
        raise RuntimeError(f"Prediction CSV labels differ: {path}")
    return result


def load_prediction_npz(
    path: Path,
    expected_indices: np.ndarray,
    expected_labels: np.ndarray,
    expected_metrics: dict[str, Any],
    ece_probability_representation: str = "stored_dtype",
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {
            "source_row_index",
            "true_label",
            "probability",
            "predicted_label",
        }:
            raise RuntimeError(f"Prediction NPZ schema differs: {path}")
        result = validate_prediction_arrays(
            path,
            payload["source_row_index"],
            payload["true_label"],
            payload["probability"],
            payload["predicted_label"],
            expected_metrics,
            ece_probability_representation,
        )
    if not np.array_equal(result["source_indices"], expected_indices):
        raise RuntimeError(f"Prediction NPZ source indices differ: {path}")
    if not np.array_equal(result["labels"], expected_labels):
        raise RuntimeError(f"Prediction NPZ labels differ: {path}")
    return result


def verify_base_seed(
    base_root: Path,
    seed: int,
    base_contract_sha256: str,
    base_root_manifest: dict[str, Any],
    dataset_sha256: str,
    split_hash: str,
    scaler_hash: str,
) -> dict[str, Any]:
    root = base_root / f"seed_{seed}"
    manifest = verify_manifest(
        root,
        BASE_PROTOCOL_ID,
        expected_inventory=expected_base_seed_inventory(),
    )
    completion_path = root / "seed_completion.json"
    completion = read_json(completion_path)
    required = {
        "protocol_id": BASE_PROTOCOL_ID,
        "status": "complete",
        "seed": seed,
        "execution_contract_sha256": base_contract_sha256,
        "dataset_sha256": dataset_sha256,
        "split_indices_sha256": split_hash,
        "scaler_sha256": scaler_hash,
    }
    for key, expected in required.items():
        if completion.get(key) != expected:
            raise RuntimeError(f"Base seed {seed} differs for {key}")
    for relative, observed in [
        (f"seed_{seed}/artifact_manifest.json", manifest["sha256"]),
        (f"seed_{seed}/seed_completion.json", sha256_file(completion_path)),
    ]:
        item = base_root_manifest["declared"].get(relative)
        if item is None or item.get("sha256") != observed:
            raise RuntimeError(f"Base root manifest does not bind {relative}")
    expected_results = {
        f"{student}_{route}"
        for student in STUDENTS
        for route in ["scratch", "rf_kd"]
    }
    if set(completion.get("student_results", {})) != expected_results:
        raise RuntimeError(f"Base seed {seed} student route schema differs")
    route_bindings: dict[str, Any] = {}
    for result_name, result in completion["student_results"].items():
        if result_name.endswith("_scratch"):
            student = result_name[: -len("_scratch")]
            route_stem = "Small_MLP_scratch"
        elif result_name.endswith("_rf_kd"):
            student = result_name[: -len("_rf_kd")]
            route_stem = "KD_from_RF"
        else:
            raise RuntimeError(f"Base route name is not canonical: {result_name}")
        expected_names = {
            "plain_state_dict": f"{student}_{route_stem}_fp32.pt",
            "rich_artifact": f"{student}_{route_stem}_artifact.pt",
            "test_predictions": f"{student}_{route_stem}_test_predictions.csv",
        }
        for file_key, hash_key in [
            ("plain_state_dict", "plain_state_dict_sha256"),
            ("rich_artifact", "rich_artifact_sha256"),
            ("test_predictions", "test_predictions_sha256"),
        ]:
            if result.get(file_key) != expected_names[file_key]:
                raise RuntimeError(f"Base route filename differs: {result_name}:{file_key}")
            path = root / result[file_key]
            if sha256_file(path) != result[hash_key]:
                raise RuntimeError(f"Base route artifact differs: {path}")
        route_bindings[result_name] = {
            "plain_state_dict": result["plain_state_dict"],
            "plain_state_dict_sha256": result["plain_state_dict_sha256"],
            "rich_artifact": result["rich_artifact"],
            "rich_artifact_sha256": result["rich_artifact_sha256"],
            "test_predictions": result["test_predictions"],
            "test_predictions_sha256": result["test_predictions_sha256"],
        }
    rf_path = root / "RF_teacher_test_predictions.csv"
    if not rf_path.is_file():
        raise RuntimeError(f"Base RF prediction evidence is absent: {rf_path}")
    rf_train_path = root / "rf_train_probabilities.npy"
    rf_train_probabilities = np.load(rf_train_path, allow_pickle=False)
    expected_train_probability_hash = completion.get(
        "teacher_soft_target_provenance", {}
    ).get("train_probability_content_sha256")
    observed_train_probability_hash = sha256_arrays(rf_train_probabilities)
    if observed_train_probability_hash != expected_train_probability_hash:
        raise RuntimeError(f"Base RF probability content differs for seed {seed}")
    return {
        "root": root,
        "completion": completion,
        "completion_sha256": sha256_file(completion_path),
        "manifest_sha256": manifest["sha256"],
        "rf_prediction_path": rf_path,
        "rf_prediction_sha256": sha256_file(rf_path),
        "route_bindings": route_bindings,
        "rf_train_probability_sha256": sha256_file(rf_train_path),
        "rf_train_probability_content_sha256": observed_train_probability_hash,
    }


def verify_full_seed(
    full_root: Path,
    seed: int,
    full_contract_sha256: str,
    full_root_manifest: dict[str, Any],
    base: dict[str, Any],
) -> dict[str, Any]:
    root = full_root / f"seed_{seed}"
    manifest = verify_manifest(
        root,
        FULL_PROTOCOL_ID,
        expected_inventory=expected_full_seed_inventory(),
    )
    completion_path = root / "seed_completion.json"
    completion = read_json(completion_path)
    for key, expected in {
        "protocol_id": FULL_PROTOCOL_ID,
        "status": "complete",
        "seed": seed,
        "execution_contract_sha256": full_contract_sha256,
    }.items():
        if completion.get(key) != expected:
            raise RuntimeError(f"Full-route seed {seed} differs for {key}")
    for relative, observed in [
        (f"seed_{seed}/artifact_manifest.json", manifest["sha256"]),
        (f"seed_{seed}/seed_completion.json", sha256_file(completion_path)),
    ]:
        item = full_root_manifest["declared"].get(relative)
        if item is None or item.get("sha256") != observed:
            raise RuntimeError(f"Full root manifest does not bind {relative}")
    expected_base = {
        "root_manifest_sha256": base["root_manifest_sha256"],
        "completion_sha256": base["completion_sha256"],
        "manifest_sha256": base["manifest_sha256"],
        "rf_probability_file_sha256": base["rf_train_probability_sha256"],
        "rf_probability_content_sha256": base[
            "rf_train_probability_content_sha256"
        ],
    }
    if completion.get("base_seed") != expected_base:
        raise RuntimeError(f"Full-route seed {seed} base binding differs")
    teacher_keys = set(TEACHER_ROUTES) | {"C_CL_MLP_loss"}
    if set(completion.get("teacher_results", {})) != teacher_keys:
        raise RuntimeError(f"Full-route seed {seed} teacher schema differs")
    if set(completion.get("student_results", {})) != set(STUDENTS):
        raise RuntimeError(f"Full-route seed {seed} student schema differs")
    expected_student_keys = set(STUDENT_ROUTES) | {"F_KD_from_CL_MLP"}
    for student, routes in completion["student_results"].items():
        if set(routes) != expected_student_keys:
            raise RuntimeError(f"Full-route seed {seed} {student} schema differs")
    for alias, source in ALIASES.items():
        collection = (
            completion["teacher_results"]
            if alias.startswith("C_")
            else completion["student_results"]["student_A"]
        )
        if alias.startswith("C_"):
            if collection[alias].get("alias_of") != source:
                raise RuntimeError(f"Teacher alias differs for seed {seed}")
            if collection[alias].get("metrics") != collection[source].get("metrics"):
                raise RuntimeError(f"Teacher alias metrics differ for seed {seed}")
        else:
            for student in STUDENTS:
                if completion["student_results"][student][alias].get("alias_of") != source:
                    raise RuntimeError(f"Student alias differs for seed {seed}: {student}")
                if completion["student_results"][student][alias].get("metrics") != completion[
                    "student_results"
                ][student][source].get("metrics"):
                    raise RuntimeError(
                        f"Student alias metrics differ for seed {seed}: {student}"
                    )

    uncalibrated = completion["teacher_results"]["A_RF_500_uncalibrated"]
    if uncalibrated.get("model_file") != "teacher_A_RF_500_uncalibrated.joblib":
        raise RuntimeError(f"Full-route RF model filename differs for seed {seed}")
    if (
        uncalibrated.get("prediction_file")
        != "teacher_A_RF_500_uncalibrated_test_predictions.npz"
    ):
        raise RuntimeError(f"Full-route RF prediction filename differs for seed {seed}")
    for file_key, hash_key in [
        ("model_file", "model_file_sha256"),
        ("prediction_file", "prediction_file_sha256"),
    ]:
        path = root / uncalibrated[file_key]
        if sha256_file(path) != uncalibrated[hash_key]:
            raise RuntimeError(f"Full-route RF artifact differs: {path}")
    calibrated = completion["teacher_results"]["A_calibrated_RF_KD_teacher"]
    if calibrated.get("source_prediction_file_sha256") != base["rf_prediction_sha256"]:
        raise RuntimeError(f"Calibrated RF prediction binding differs for seed {seed}")
    expected_rf_relative = base["rf_prediction_path"].relative_to(REPO_ROOT).as_posix()
    if calibrated.get("source_prediction_file") != expected_rf_relative:
        raise RuntimeError(f"Calibrated RF prediction path differs for seed {seed}")
    for route in TRAINED_TEACHER_ROUTES:
        result = completion["teacher_results"][route]
        if result.get("model_file") != f"teacher_{route}.pt":
            raise RuntimeError(f"Full-route teacher model filename differs: {seed}:{route}")
        if result.get("prediction_file") != f"teacher_{route}_test_predictions.npz":
            raise RuntimeError(
                f"Full-route teacher prediction filename differs: {seed}:{route}"
            )
        for file_key, hash_key in [
            ("model_file", "model_file_sha256"),
            ("prediction_file", "prediction_file_sha256"),
        ]:
            path = root / result[file_key]
            if sha256_file(path) != result[hash_key]:
                raise RuntimeError(f"Full-route teacher artifact differs: {path}")
    for student in STUDENTS:
        base_results = base["completion"]["student_results"]
        for route, suffix in [
            ("D_Small_MLP", "scratch"),
            ("E_KD_from_RF", "rf_kd"),
        ]:
            current = completion["student_results"][student][route]
            source = base_results[f"{student}_{suffix}"]
            expected_artifact = (base["root"] / source["rich_artifact"]).relative_to(
                REPO_ROOT
            ).as_posix()
            if current.get("source_artifact") != expected_artifact:
                raise RuntimeError(
                    f"Reused base artifact path differs for seed {seed}: {student}:{route}"
                )
            if current.get("source_artifact_sha256") != source["rich_artifact_sha256"]:
                raise RuntimeError(
                    f"Reused base artifact hash differs for seed {seed}: {student}:{route}"
                )
        for route in TRAINED_STUDENT_ROUTES:
            result = completion["student_results"][student][route]
            if result.get("model_file") != f"{student}_{route}.pt":
                raise RuntimeError(
                    f"Full-route student model filename differs: {seed}:{student}:{route}"
                )
            if result.get("prediction_file") != f"{student}_{route}_test_predictions.npz":
                raise RuntimeError(
                    f"Full-route student prediction filename differs: {seed}:{student}:{route}"
                )
            for file_key, hash_key in [
                ("model_file", "model_file_sha256"),
                ("prediction_file", "prediction_file_sha256"),
            ]:
                path = root / result[file_key]
                if sha256_file(path) != result[hash_key]:
                    raise RuntimeError(f"Full-route student artifact differs: {path}")
    return {
        "root": root,
        "completion": completion,
        "completion_sha256": sha256_file(completion_path),
        "manifest_sha256": manifest["sha256"],
    }


def load_context(
    dataset_path: Path,
    base_root: Path,
    full_root: Path,
    full_source_path: Path,
) -> dict[str, Any]:
    base_root = base_root.resolve()
    full_root = full_root.resolve()
    dataset_path = dataset_path.resolve()
    full_source_path = full_source_path.resolve()
    common_source_path = (
        REPO_ROOT / "experiments" / "wsnds" / "leakage_free_rerun" / "tier15_common.py"
    ).resolve()
    if base_root == full_root:
        raise RuntimeError("Base and full-route roots must differ")
    base_manifest = verify_manifest(
        base_root,
        BASE_PROTOCOL_ID,
        expected_inventory=expected_base_root_inventory(),
    )
    full_manifest = verify_manifest(
        full_root,
        FULL_PROTOCOL_ID,
        expected_inventory=expected_full_root_inventory(),
    )
    base_execution_path = base_root / "execution_contract.json"
    preprocessing_path = base_root / "preprocessing_contract.json"
    base_aggregate_path = base_root / "aggregate_results.json"
    full_execution_path = full_root / "execution_contract.json"
    full_aggregate_path = full_root / "aggregate_results.json"
    for path in [
        base_execution_path,
        preprocessing_path,
        base_aggregate_path,
        full_execution_path,
        full_aggregate_path,
        full_source_path,
        common_source_path,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)
    base_execution = read_json(base_execution_path)
    preprocessing = read_json(preprocessing_path)
    base_aggregate = read_json(base_aggregate_path)
    full_execution = read_json(full_execution_path)
    full_aggregate = read_json(full_aggregate_path)
    if base_execution.get("protocol_id") != BASE_PROTOCOL_ID:
        raise RuntimeError("Base execution protocol differs")
    if preprocessing.get("protocol_id") != BASE_PROTOCOL_ID:
        raise RuntimeError("Base preprocessing protocol differs")
    if base_aggregate.get("protocol_id") != BASE_PROTOCOL_ID or base_aggregate.get(
        "status"
    ) != "complete":
        raise RuntimeError("Base aggregate is absent or incomplete")
    if base_aggregate.get("seeds") != SEEDS or base_aggregate.get("seed_count") != len(SEEDS):
        raise RuntimeError("Base aggregate seed contract differs")
    if full_execution.get("protocol_id") != FULL_PROTOCOL_ID:
        raise RuntimeError("Full-route execution protocol differs")
    if full_aggregate.get("protocol_id") != FULL_PROTOCOL_ID or full_aggregate.get("status") != "complete":
        raise RuntimeError("Full-route aggregate is absent or incomplete")
    if base_execution.get("seeds") != SEEDS or full_execution.get("seeds") != SEEDS:
        raise RuntimeError("Source runs do not use the fixed ten-seed set")
    if full_aggregate.get("seeds") != SEEDS or full_aggregate.get("seed_count") != len(SEEDS):
        raise RuntimeError("Full-route aggregate seed contract differs")
    if full_execution.get("teacher_routes") != TEACHER_ROUTES:
        raise RuntimeError("Full-route teacher route contract differs")
    if full_execution.get("student_routes") != STUDENT_ROUTES:
        raise RuntimeError("Full-route student route contract differs")
    if full_execution.get("aliases") != ALIASES:
        raise RuntimeError("Full-route alias contract differs")
    finalization = validate_finalized_full_root(
        full_root,
        full_execution_path,
        full_aggregate_path,
        full_source_path,
        common_source_path,
        full_execution,
    )
    full_source_sha256 = sha256_file(full_source_path)
    if full_source_sha256 != full_execution.get("script_sha256"):
        raise RuntimeError(
            "Provided full-route executed source does not match its execution contract"
        )
    common_source_sha256 = sha256_file(common_source_path)
    if common_source_sha256 != base_execution.get("common_module_sha256"):
        raise RuntimeError("Current shared data-contract source differs from the base run")
    if common_source_sha256 != full_execution.get("common_module_sha256"):
        raise RuntimeError("Current shared data-contract source differs from the full-route run")
    dataset = load_dataset(dataset_path)
    if dataset["dataset_sha256"] != base_execution.get("dataset_sha256"):
        raise RuntimeError("Dataset hash differs from the base execution contract")
    for key in ["dataset_sha256", "split_indices_sha256", "scaler_sha256"]:
        if full_execution.get(key) != base_execution.get(key):
            raise RuntimeError(f"Full-route and base contracts differ for {key}")
    if full_execution.get("base_root_manifest_sha256") != base_manifest["sha256"]:
        raise RuntimeError("Full-route contract does not bind the verified base manifest")
    if full_execution.get("base_execution_contract_sha256") != sha256_file(base_execution_path):
        raise RuntimeError("Full-route contract does not bind the base execution contract")
    if full_execution.get("base_preprocessing_contract_sha256") != sha256_file(preprocessing_path):
        raise RuntimeError("Full-route contract does not bind preprocessing")
    split = reconstruct_split_and_scaler(dataset, base_root, base_execution, preprocessing)
    base_seeds: dict[int, Any] = {}
    full_seeds: dict[int, Any] = {}
    for seed in SEEDS:
        base = verify_base_seed(
            base_root,
            seed,
            sha256_file(base_execution_path),
            base_manifest,
            dataset["dataset_sha256"],
            split["index_hash"],
            split["scaler_hash"],
        )
        base["root_manifest_sha256"] = base_manifest["sha256"]
        base_seeds[seed] = base
        full_seeds[seed] = verify_full_seed(
            full_root,
            seed,
            sha256_file(full_execution_path),
            full_manifest,
            base,
        )
    finalized_seed_evidence = {
        str(seed): {
            "completion_sha256": full_seeds[seed]["completion_sha256"],
            "manifest_sha256": full_seeds[seed]["manifest_sha256"],
        }
        for seed in SEEDS
    }
    if finalization["payload"].get("seed_evidence") != finalized_seed_evidence:
        raise RuntimeError("Finalization contract seed evidence differs")
    return {
        "dataset_path": dataset_path,
        "base_root": base_root,
        "full_root": full_root,
        "full_source_path": full_source_path,
        "common_source_path": common_source_path,
        "dataset": dataset,
        "split": split,
        "base_execution": base_execution,
        "preprocessing": preprocessing,
        "base_aggregate": base_aggregate,
        "full_execution": full_execution,
        "full_aggregate": full_aggregate,
        "base_execution_sha256": sha256_file(base_execution_path),
        "preprocessing_sha256": sha256_file(preprocessing_path),
        "base_aggregate_sha256": sha256_file(base_aggregate_path),
        "full_execution_sha256": sha256_file(full_execution_path),
        "base_manifest": base_manifest,
        "full_manifest": full_manifest,
        "finalization": finalization,
        "full_source_sha256": full_source_sha256,
        "common_source_sha256": common_source_sha256,
        "base_seeds": base_seeds,
        "full_seeds": full_seeds,
    }


def build_test_groups(
    test_features: np.ndarray, test_labels: np.ndarray, source_indices: np.ndarray
) -> dict[str, Any]:
    keys = exact_row_keys(test_features)
    unique_keys, group_ids, group_sizes = np.unique(
        keys, return_inverse=True, return_counts=True
    )
    group_ids = group_ids.astype(np.int64, copy=False)
    group_sizes = group_sizes.astype(np.int64, copy=False)
    group_count = len(unique_keys)
    label_counts = np.zeros((group_count, len(CLASS_NAMES)), dtype=np.int64)
    np.add.at(label_counts, (group_ids, test_labels), 1)
    mixed = (label_counts > 0).sum(axis=1) > 1
    pure = ~mixed
    representative_source = np.full(group_count, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(representative_source, group_ids, source_indices)
    representative = source_indices == representative_source[group_ids]
    if np.any(np.bincount(group_ids, weights=representative.astype(np.int64)) != 1):
        raise RuntimeError("Deterministic group representative selection failed")
    representative_pure = representative & pure[group_ids]
    inverse_weights = 1.0 / group_sizes[group_ids].astype(np.float64)
    if not np.isclose(inverse_weights.sum(), float(group_count), rtol=0.0, atol=1e-10):
        raise RuntimeError("Inverse group weights do not assign one total unit per group")
    group_key_sha256 = np.asarray(
        [hashlib.sha256(bytes(value)).hexdigest() for value in unique_keys], dtype="U64"
    )
    if len(set(group_key_sha256.tolist())) != group_count:
        raise RuntimeError("SHA-256 collision occurred among exact test feature groups")
    mixed_group_ids = np.flatnonzero(mixed)
    mixed_rows = mixed[group_ids]
    if int(mixed_rows.sum()) != int(group_sizes[mixed].sum()):
        raise RuntimeError("Mixed-label group row accounting differs")
    return {
        "group_ids": group_ids,
        "group_sizes": group_sizes,
        "group_count": group_count,
        "group_key_sha256": group_key_sha256,
        "label_counts": label_counts,
        "pure": pure,
        "mixed": mixed,
        "mixed_group_ids": mixed_group_ids,
        "mixed_rows": mixed_rows,
        "representative": representative,
        "representative_pure": representative_pure,
        "representative_source": representative_source,
        "inverse_weights": inverse_weights,
        "repeated_group_count": int(np.count_nonzero(group_sizes > 1)),
        "rows_in_repeated_groups": int(group_sizes[group_sizes > 1].sum()),
        "mixed_group_count": int(mixed.sum()),
        "mixed_row_count": int(group_sizes[mixed].sum()),
        "pure_group_count": int(pure.sum()),
    }


def route_sources(context: dict[str, Any], seed: int) -> Iterable[dict[str, Any]]:
    base = context["base_seeds"][seed]
    full = context["full_seeds"][seed]
    completion = full["completion"]
    yield {
        "category": "teacher",
        "student": "",
        "route": "A_RF_500_uncalibrated",
        "format": "npz",
        "path": full["root"] / completion["teacher_results"]["A_RF_500_uncalibrated"]["prediction_file"],
        "expected_metrics": completion["teacher_results"]["A_RF_500_uncalibrated"]["metrics"],
        "ece_probability_representation": "stored_float32_promoted_to_float64",
    }
    yield {
        "category": "teacher",
        "student": "",
        "route": "A_calibrated_RF_KD_teacher",
        "format": "csv",
        "path": base["rf_prediction_path"],
        "expected_metrics": completion["teacher_results"]["A_calibrated_RF_KD_teacher"]["metrics"],
    }
    for route in TRAINED_TEACHER_ROUTES:
        result = completion["teacher_results"][route]
        yield {
            "category": "teacher",
            "student": "",
            "route": route,
            "format": "npz",
            "path": full["root"] / result["prediction_file"],
            "expected_metrics": result["metrics"],
        }
    for student in STUDENTS:
        base_results = base["completion"]["student_results"]
        for route, base_suffix in [
            ("D_Small_MLP", "scratch"),
            ("E_KD_from_RF", "rf_kd"),
        ]:
            base_result = base_results[f"{student}_{base_suffix}"]
            current = completion["student_results"][student][route]
            if current.get("source_artifact_sha256") != base_result["rich_artifact_sha256"]:
                raise RuntimeError(f"Full route does not bind base artifact: {seed}:{student}:{route}")
            yield {
                "category": "student",
                "student": student,
                "route": route,
                "format": "csv",
                "path": base["root"] / base_result["test_predictions"],
                "expected_metrics": current["metrics"],
            }
        for route in TRAINED_STUDENT_ROUTES:
            result = completion["student_results"][student][route]
            yield {
                "category": "student",
                "student": student,
                "route": route,
                "format": "npz",
                "path": full["root"] / result["prediction_file"],
                "expected_metrics": result["metrics"],
            }


def analyze_routes(context: dict[str, Any], groups: dict[str, Any]) -> dict[str, Any]:
    expected_indices = context["split"]["indices"]["test"]
    expected_labels = context["split"]["test_labels"]
    metric_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        seen: set[tuple[str, str, str]] = set()
        for route in route_sources(context, seed):
            key = (route["category"], route["student"], route["route"])
            if key in seen:
                raise RuntimeError(f"Duplicate route source for seed {seed}: {key}")
            seen.add(key)
            if route["format"] == "csv":
                evidence = load_prediction_csv(
                    route["path"], expected_indices, expected_labels, route["expected_metrics"]
                )
            else:
                evidence = load_prediction_npz(
                    route["path"],
                    expected_indices,
                    expected_labels,
                    route["expected_metrics"],
                    route.get("ece_probability_representation", "stored_dtype"),
                )
            predictions = evidence["predictions"]
            probabilities = evidence["probabilities"]
            group_prediction_min = np.full(groups["group_count"], len(CLASS_NAMES), dtype=np.int64)
            group_prediction_max = np.full(groups["group_count"], -1, dtype=np.int64)
            np.minimum.at(group_prediction_min, groups["group_ids"], predictions)
            np.maximum.at(group_prediction_max, groups["group_ids"], predictions)
            if not np.array_equal(group_prediction_min, group_prediction_max):
                raise RuntimeError(
                    f"Identical test features have inconsistent predictions: "
                    f"seed={seed} route={route['route']}"
                )
            group_probability_min = np.full(
                (groups["group_count"], len(CLASS_NAMES)), np.inf, dtype=np.float64
            )
            group_probability_max = np.full(
                (groups["group_count"], len(CLASS_NAMES)), -np.inf, dtype=np.float64
            )
            np.minimum.at(group_probability_min, groups["group_ids"], probabilities)
            np.maximum.at(group_probability_max, groups["group_ids"], probabilities)
            maximum_probability_delta = float(
                np.max(np.abs(group_probability_max - group_probability_min))
            )
            if maximum_probability_delta > WITHIN_GROUP_PROBABILITY_ATOL:
                raise RuntimeError(
                    "Identical test features have inconsistent probability vectors: "
                    f"seed={seed} route={route['route']} "
                    f"max_delta={maximum_probability_delta:.9g} "
                    f"tolerance={WITHIN_GROUP_PROBABILITY_ATOL:.9g}"
                )
            source_sha256 = sha256_file(route["path"])
            view_inputs = {
                "row_level": (expected_labels, predictions, None, len(groups["group_sizes"])),
                "inverse_test_group_size": (
                    expected_labels,
                    predictions,
                    groups["inverse_weights"],
                    groups["group_count"],
                ),
                "pure_group_representative": (
                    expected_labels[groups["representative_pure"]],
                    predictions[groups["representative_pure"]],
                    None,
                    groups["pure_group_count"],
                ),
            }
            for view, (labels, view_predictions, weights, represented_groups) in view_inputs.items():
                metrics = classification_metrics(labels, view_predictions, weights)
                row = {
                    "view": view,
                    "seed": seed,
                    "category": route["category"],
                    "student": route["student"],
                    "route": route["route"],
                    "prediction_source": route["path"].relative_to(REPO_ROOT).as_posix(),
                    "prediction_source_sha256": source_sha256,
                    "max_within_exact_group_probability_delta": maximum_probability_delta,
                    "within_exact_group_probability_tolerance": (
                        WITHIN_GROUP_PROBABILITY_ATOL
                    ),
                    "represented_group_count": int(represented_groups),
                    "excluded_mixed_group_count": (
                        groups["mixed_group_count"] if view == "pure_group_representative" else 0
                    ),
                    "excluded_mixed_row_count": (
                        groups["mixed_row_count"] if view == "pure_group_representative" else 0
                    ),
                    **{key: metrics[key] for key in [
                        "row_count",
                        "effective_weight",
                        "accuracy",
                        "macro_precision",
                        "macro_recall",
                        "macro_f1",
                    ]},
                    "confusion_matrix_json": json.dumps(metrics["confusion_matrix"], separators=(",", ":")),
                }
                metric_rows.append(row)
                for class_id, class_name in enumerate(CLASS_NAMES):
                    class_rows.append(
                        {
                            "view": view,
                            "seed": seed,
                            "category": route["category"],
                            "student": route["student"],
                            "route": route["route"],
                            "class_id": class_id,
                            "class_name": class_name,
                            "precision": metrics["per_class_precision"][class_id],
                            "recall": metrics["per_class_recall"][class_id],
                            "f1": metrics["per_class_f1"][class_id],
                            "support": metrics["per_class_support"][class_id],
                        }
                    )
        expected_route_count = len(TEACHER_ROUTES) + len(STUDENTS) * len(STUDENT_ROUTES)
        if len(seen) != expected_route_count:
            raise RuntimeError(f"Seed {seed} route coverage differs: {len(seen)}")
    metrics_frame = pd.DataFrame(metric_rows)
    classes_frame = pd.DataFrame(class_rows)
    expected_metric_rows = len(SEEDS) * (len(TEACHER_ROUTES) + 2 * len(STUDENT_ROUTES)) * len(VIEWS)
    if len(metrics_frame) != expected_metric_rows:
        raise RuntimeError("Route metric row count differs")
    return {"metrics": metrics_frame, "classes": classes_frame}


def scalar_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(SEEDS),) or not np.isfinite(values).all():
        raise RuntimeError("Aggregate values do not contain ten finite paired seeds")
    return {
        "values": values.tolist(),
        "mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def exact_sign_flip_mean_p(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=np.float64)
    if differences.shape != (len(SEEDS),):
        raise RuntimeError("Exact paired test requires ten differences")
    observed = abs(float(differences.mean()))
    exceed = 0
    total = 1 << len(differences)
    for mask in range(total):
        signed_sum = 0.0
        for index, value in enumerate(differences):
            signed_sum += value if (mask >> index) & 1 else -value
        if abs(signed_sum / len(differences)) >= observed - 1e-15:
            exceed += 1
    return float(exceed / total)


def exact_paired_wilcoxon(differences: np.ndarray) -> dict[str, Any]:
    differences = np.asarray(differences, dtype=np.float64)
    nonzero = differences[differences != 0.0]
    if len(nonzero) == 0:
        return {
            "statistic_abs_signed_rank_sum": 0.0,
            "p_value_two_sided": 1.0,
            "nonzero_difference_count": 0,
            "zero_difference_count": int(len(differences)),
            "enumerated_sign_assignments": 1,
            "rank_tie_method": "average",
            "zero_method": "wilcox",
            "enumeration": "exhaustive",
        }
    ranks = stats.rankdata(np.abs(nonzero), method="average")
    observed = abs(float(np.dot(np.sign(nonzero), ranks)))
    exceed = 0
    total = 1 << len(nonzero)
    for mask in range(total):
        signed = sum(
            rank if (mask >> index) & 1 else -rank
            for index, rank in enumerate(ranks)
        )
        if abs(float(signed)) >= observed - 1e-15:
            exceed += 1
    return {
        "statistic_abs_signed_rank_sum": observed,
        "p_value_two_sided": float(exceed / total),
        "nonzero_difference_count": int(len(nonzero)),
        "zero_difference_count": int(len(differences) - len(nonzero)),
        "enumerated_sign_assignments": int(total),
        "rank_tie_method": "average",
        "zero_method": "wilcox",
        "enumeration": "exhaustive",
    }


def holm_adjust(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=lambda name: (raw[name], name))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        candidate = min(1.0, raw[name] * (total - rank))
        running = max(running, candidate)
        adjusted[name] = float(running)
    return adjusted


def aggregate_analysis(route_data: dict[str, Any]) -> dict[str, Any]:
    metrics = route_data["metrics"]
    classes = route_data["classes"]
    probability_rows = metrics[metrics["view"] == "row_level"].copy()
    expected_probability_rows = len(SEEDS) * (
        len(TEACHER_ROUTES) + len(STUDENTS) * len(STUDENT_ROUTES)
    )
    if len(probability_rows) != expected_probability_rows:
        raise RuntimeError("Probability-consistency route coverage differs")
    if not np.all(
        probability_rows["within_exact_group_probability_tolerance"].to_numpy(
            dtype=np.float64
        )
        == WITHIN_GROUP_PROBABILITY_ATOL
    ):
        raise RuntimeError("Probability-consistency tolerance differs across routes")
    maximum_probability_row = probability_rows.loc[
        probability_rows["max_within_exact_group_probability_delta"].idxmax()
    ]
    probability_consistency = {
        "route_seed_count": int(len(probability_rows)),
        "exact_prediction_equality_required": True,
        "probability_max_abs_delta_tolerance": WITHIN_GROUP_PROBABILITY_ATOL,
        "nonzero_probability_delta_route_seed_count": int(
            np.count_nonzero(
                probability_rows[
                    "max_within_exact_group_probability_delta"
                ].to_numpy(dtype=np.float64)
            )
        ),
        "global_max_probability_abs_delta": float(
            maximum_probability_row["max_within_exact_group_probability_delta"]
        ),
        "global_max_seed": int(maximum_probability_row["seed"]),
        "global_max_category": str(maximum_probability_row["category"]),
        "global_max_student": str(maximum_probability_row["student"]),
        "global_max_route": str(maximum_probability_row["route"]),
        "global_max_prediction_source": str(
            maximum_probability_row["prediction_source"]
        ),
        "global_max_prediction_source_sha256": str(
            maximum_probability_row["prediction_source_sha256"]
        ),
    }
    identity = ["view", "category", "student", "route"]
    aggregate_rows: list[dict[str, Any]] = []
    aggregate_json: dict[str, Any] = {}
    for keys, frame in metrics.groupby(identity, sort=True, dropna=False):
        view, category, student, route = keys
        frame = frame.sort_values("seed")
        if frame["seed"].tolist() != sorted(SEEDS):
            raise RuntimeError(f"Aggregate seed order differs: {keys}")
        record = {
            "view": view,
            "category": category,
            "student": student,
            "route": route,
        }
        route_payload = {}
        for metric in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]:
            summary = scalar_summary(frame[metric].to_numpy(dtype=np.float64))
            route_payload[metric] = summary
            record[f"{metric}_mean"] = summary["mean"]
            record[f"{metric}_sample_sd"] = summary["sample_std"]
        aggregate_rows.append(record)
        aggregate_json.setdefault(view, {}).setdefault(category, {}).setdefault(
            student or "all", {}
        )[route] = route_payload

    class_aggregate_rows: list[dict[str, Any]] = []
    for keys, frame in classes.groupby(
        identity + ["class_id", "class_name"], sort=True, dropna=False
    ):
        view, category, student, route, class_id, class_name = keys
        frame = frame.sort_values("seed")
        if frame["seed"].tolist() != sorted(SEEDS):
            raise RuntimeError(f"Per-class aggregate seed order differs: {keys}")
        summary = scalar_summary(frame["f1"].to_numpy(dtype=np.float64))
        class_aggregate_rows.append(
            {
                "view": view,
                "category": category,
                "student": student,
                "route": route,
                "class_id": int(class_id),
                "class_name": class_name,
                "f1_mean": summary["mean"],
                "f1_sample_sd": summary["sample_std"],
                "f1_values_json": json.dumps(summary["values"], separators=(",", ":")),
            }
        )

    shift_rows: list[dict[str, Any]] = []
    for _, row in metrics[metrics["view"].isin(SENSITIVITY_VIEWS)].iterrows():
        base = metrics[
            (metrics["view"] == "row_level")
            & (metrics["seed"] == row["seed"])
            & (metrics["category"] == row["category"])
            & (metrics["student"] == row["student"])
            & (metrics["route"] == row["route"])
        ]
        if len(base) != 1:
            raise RuntimeError("Sensitivity shift row-level match is not unique")
        base_row = base.iloc[0]
        shift_rows.append(
            {
                "view": row["view"],
                "seed": int(row["seed"]),
                "category": row["category"],
                "student": row["student"],
                "route": row["route"],
                "accuracy_minus_row_level": float(row["accuracy"] - base_row["accuracy"]),
                "macro_f1_minus_row_level": float(row["macro_f1"] - base_row["macro_f1"]),
            }
        )

    delta_rows: list[dict[str, Any]] = []
    test_records: dict[str, dict[str, Any]] = {}
    for view in VIEWS:
        comparisons = [("teacher", "", TEACHER_COMPARISONS)] + [
            ("student", student, STUDENT_COMPARISONS) for student in STUDENTS
        ]
        for category, student, pairs in comparisons:
            family = "teacher" if category == "teacher" else student
            for left, right in pairs:
                left_frame = metrics[
                    (metrics["view"] == view)
                    & (metrics["category"] == category)
                    & (metrics["student"] == student)
                    & (metrics["route"] == left)
                ].sort_values("seed")
                right_frame = metrics[
                    (metrics["view"] == view)
                    & (metrics["category"] == category)
                    & (metrics["student"] == student)
                    & (metrics["route"] == right)
                ].sort_values("seed")
                if left_frame["seed"].tolist() != sorted(SEEDS) or not np.array_equal(
                    left_frame["seed"].to_numpy(), right_frame["seed"].to_numpy()
                ):
                    raise RuntimeError(f"Paired route seeds differ: {view}:{family}:{left}:{right}")
                differences = (
                    left_frame["macro_f1"].to_numpy(dtype=np.float64)
                    - right_frame["macro_f1"].to_numpy(dtype=np.float64)
                )
                per_class_differences: dict[int, np.ndarray] = {}
                for class_id in range(len(CLASS_NAMES)):
                    left_class = classes[
                        (classes["view"] == view)
                        & (classes["category"] == category)
                        & (classes["student"] == student)
                        & (classes["route"] == left)
                        & (classes["class_id"] == class_id)
                    ].sort_values("seed")
                    right_class = classes[
                        (classes["view"] == view)
                        & (classes["category"] == category)
                        & (classes["student"] == student)
                        & (classes["route"] == right)
                        & (classes["class_id"] == class_id)
                    ].sort_values("seed")
                    if left_class["seed"].tolist() != sorted(SEEDS) or not np.array_equal(
                        left_class["seed"].to_numpy(), right_class["seed"].to_numpy()
                    ):
                        raise RuntimeError(
                            f"Paired per-class seeds differ: {view}:{family}:{left}:{right}:{class_id}"
                        )
                    per_class_differences[class_id] = (
                        left_class["f1"].to_numpy(dtype=np.float64)
                        - right_class["f1"].to_numpy(dtype=np.float64)
                    )
                name = f"{view}:{family}:{left}_minus_{right}"
                for index, seed in enumerate(sorted(SEEDS)):
                    delta_row = {
                        "view": view,
                        "family": family,
                        "seed": seed,
                        "left_route": left,
                        "right_route": right,
                        "left_macro_f1": float(left_frame.iloc[index]["macro_f1"]),
                        "right_macro_f1": float(right_frame.iloc[index]["macro_f1"]),
                        "macro_f1_delta": float(differences[index]),
                    }
                    for class_id, class_name in enumerate(CLASS_NAMES):
                        delta_row[f"class_{class_id}_{class_name}_f1_delta"] = float(
                            per_class_differences[class_id][index]
                        )
                    delta_rows.append(delta_row)
                paired_wilcoxon = exact_paired_wilcoxon(differences)
                test_records[name] = {
                    "name": name,
                    "view": view,
                    "family": family,
                    "left_route": left,
                    "right_route": right,
                    "differences": scalar_summary(differences),
                    "exact_paired_wilcoxon": paired_wilcoxon,
                    "exact_sign_flip_mean_p_two_sided": exact_sign_flip_mean_p(differences),
                }

    for view in VIEWS:
        view_names = [name for name, item in test_records.items() if item["view"] == view]
        for family in ["teacher", *STUDENTS]:
            names = [name for name in view_names if test_records[name]["family"] == family]
            signed = holm_adjust(
                {name: test_records[name]["exact_paired_wilcoxon"]["p_value_two_sided"] for name in names}
            )
            mean = holm_adjust(
                {name: test_records[name]["exact_sign_flip_mean_p_two_sided"] for name in names}
            )
            for name in names:
                test_records[name]["holm_family"] = f"{view}:{family}"
                test_records[name]["holm_family_size"] = len(names)
                test_records[name]["holm_exact_paired_wilcoxon_p"] = signed[name]
                test_records[name]["holm_exact_sign_flip_mean_p"] = mean[name]
        signed_global = holm_adjust(
            {name: test_records[name]["exact_paired_wilcoxon"]["p_value_two_sided"] for name in view_names}
        )
        mean_global = holm_adjust(
            {name: test_records[name]["exact_sign_flip_mean_p_two_sided"] for name in view_names}
        )
        for name in view_names:
            test_records[name]["holm_view_global_family"] = view
            test_records[name]["holm_view_global_family_size"] = len(view_names)
            test_records[name]["holm_view_global_exact_paired_wilcoxon_p"] = signed_global[name]
            test_records[name]["holm_view_global_exact_sign_flip_mean_p"] = mean_global[name]

    test_rows = []
    for name in sorted(test_records):
        item = test_records[name]
        test_rows.append(
            {
                "name": name,
                "view": item["view"],
                "family": item["family"],
                "left_route": item["left_route"],
                "right_route": item["right_route"],
                "mean_macro_f1_delta": item["differences"]["mean"],
                "sample_sd_macro_f1_delta": item["differences"]["sample_std"],
                "exact_paired_wilcoxon_p_two_sided": item["exact_paired_wilcoxon"]["p_value_two_sided"],
                "exact_sign_flip_mean_p_two_sided": item["exact_sign_flip_mean_p_two_sided"],
                "holm_family": item["holm_family"],
                "holm_family_size": item["holm_family_size"],
                "holm_exact_paired_wilcoxon_p": item["holm_exact_paired_wilcoxon_p"],
                "holm_exact_sign_flip_mean_p": item["holm_exact_sign_flip_mean_p"],
                "holm_view_global_family_size": item["holm_view_global_family_size"],
                "holm_view_global_exact_paired_wilcoxon_p": item["holm_view_global_exact_paired_wilcoxon_p"],
                "holm_view_global_exact_sign_flip_mean_p": item["holm_view_global_exact_sign_flip_mean_p"],
            }
        )
    return {
        "route_aggregate": pd.DataFrame(aggregate_rows),
        "class_aggregate": pd.DataFrame(class_aggregate_rows),
        "sensitivity_shifts": pd.DataFrame(shift_rows),
        "route_pair_deltas": pd.DataFrame(delta_rows),
        "paired_tests": pd.DataFrame(test_rows),
        "aggregate_json": aggregate_json,
        "test_records": test_records,
        "probability_consistency": probability_consistency,
    }


def group_output_frames(
    groups: dict[str, Any], source_indices: np.ndarray, labels: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_rows = []
    row_rows = []
    for group_id in groups["mixed_group_ids"]:
        mask = groups["group_ids"] == group_id
        sources = source_indices[mask].astype(int).tolist()
        group_labels = labels[mask].astype(int).tolist()
        counts = groups["label_counts"][group_id]
        row = {
            "exact_group_id": int(group_id),
            "group_key_sha256": groups["group_key_sha256"][group_id],
            "row_count": int(mask.sum()),
            "source_row_indices_json": json.dumps(sources, separators=(",", ":")),
            "row_labels_json": json.dumps(group_labels, separators=(",", ":")),
        }
        for class_id, class_name in enumerate(CLASS_NAMES):
            row[f"label_count_{class_id}_{class_name}"] = int(counts[class_id])
        group_rows.append(row)
        for position in np.flatnonzero(mask):
            row_rows.append(
                {
                    "exact_group_id": int(group_id),
                    "group_key_sha256": groups["group_key_sha256"][group_id],
                    "source_row_index": int(source_indices[position]),
                    "true_label": int(labels[position]),
                    "class_name": CLASS_NAMES[int(labels[position])],
                }
            )
    return pd.DataFrame(group_rows), pd.DataFrame(row_rows)


def claim_boundaries() -> list[str]:
    return [
        "This is post-processing of one fixed 56,301-row WSN-DS test partition; no new split or model fit is performed.",
        "The inferential unit is the paired algorithmic run seed, with ten seeds for every unique route.",
        "Primary inference uses the row-level view, the exact paired Wilcoxon signed-rank test, and Holm adjustment within each predeclared teacher or student route family.",
        "View-global Holm results are reported as a stricter multiplicity sensitivity analysis; they do not replace the predeclared family-wise primary policy.",
        "The inverse-size view retains every row and assigns each exact test feature group total weight one; mixed-label rows keep their recorded labels.",
        "The representative view keeps the smallest source-row index from each label-pure exact feature group and excludes every mixed-label group and row.",
        "No mixed-label feature group is assigned a majority label in either sensitivity view.",
        "The sensitivity views quantify within-test repeated-pattern weighting and do not estimate performance on a new dataset, new partition, live traffic, or independent network events.",
        "Holm families are declared separately by view for teachers, Student A routes, and Student B routes; view-global adjustments are also reported.",
        "Alias route names are excluded from inference because they duplicate an already included route exactly.",
        "Per-class route deltas are descriptive because no per-class hypothesis-test family is performed.",
        "Predicted classes must agree exactly within every exact-feature group. Persisted probability vectors are audited with a maximum absolute tolerance of 2e-6 for float32 and decimal serialization effects; group weighting does not use probabilities.",
        "The exact paired p-values in this analysis supersede the full-route runner's approximate Wilcoxon fields; saved models, predictions, and performance metrics are unchanged.",
    ]


def build_execution_contract(context: dict[str, Any], groups: dict[str, Any]) -> dict[str, Any]:
    base_bindings = {}
    full_bindings = {}
    for seed in SEEDS:
        base = context["base_seeds"][seed]
        full = context["full_seeds"][seed]
        base_bindings[str(seed)] = {
            "seed_manifest_sha256": base["manifest_sha256"],
            "seed_completion_sha256": base["completion_sha256"],
            "rf_test_predictions_sha256": base["rf_prediction_sha256"],
            "rf_train_probabilities_sha256": base["rf_train_probability_sha256"],
            "scratch_and_rfkd_artifacts": base["route_bindings"],
        }
        full_bindings[str(seed)] = {
            "seed_manifest_sha256": full["manifest_sha256"],
            "seed_completion_sha256": full["completion_sha256"],
        }
    payload = {
        "protocol_id": PROTOCOL_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analyzer_source_sha256": sha256_file(SCRIPT_PATH),
        "dataset": {
            "recorded_path": str(context["dataset_path"]),
            "sha256": context["dataset"]["dataset_sha256"],
            "rows": len(context["dataset"]["labels"]),
            "features": len(FEATURE_NAMES),
        },
        "clean_data_contract": {
            "base_protocol_id": BASE_PROTOCOL_ID,
            "base_execution_contract_sha256": context["base_execution_sha256"],
            "base_preprocessing_contract_sha256": context["preprocessing_sha256"],
            "base_aggregate_results_sha256": context["base_aggregate_sha256"],
            "base_root_manifest_sha256": context["base_manifest"]["sha256"],
            "shared_data_contract_source_sha256": context["common_source_sha256"],
            "split_indices_sha256": context["split"]["index_hash"],
            "scaler_sha256": context["split"]["scaler_hash"],
            "split_hashes": context["split"]["split_hashes"],
            "transformed_split_hashes": context["split"]["transformed_hashes"],
            "exact_raw_feature_group_overlap": context["split"]["exact_overlap"],
            "test_rows": len(context["split"]["indices"]["test"]),
        },
        "full_route_contract": {
            "protocol_id": FULL_PROTOCOL_ID,
            "recorded_root": str(context["full_root"]),
            "execution_contract_sha256": context["full_execution_sha256"],
            "root_manifest_sha256": context["full_manifest"]["sha256"],
            "executed_source_recorded_path": str(context["full_source_path"]),
            "executed_source_sha256": context["full_source_sha256"],
            "aggregate_results_sha256": sha256_file(context["full_root"] / "aggregate_results.json"),
            "finalization_contract_sha256": context["finalization"]["sha256"],
            "finalizer_protocol_id": FINALIZER_PROTOCOL_ID,
            "continuation_contract_sha256": context["finalization"][
                "continuation_contract_sha256"
            ],
            "continuation_manifest_sha256": context["finalization"][
                "continuation_manifest_sha256"
            ],
            "interrupted_archive_manifest_sha256": context["finalization"][
                "interrupted_archive_manifest_sha256"
            ],
        },
        "base_seed_bindings": base_bindings,
        "full_route_seed_bindings": full_bindings,
        "seeds": SEEDS,
        "teacher_routes": TEACHER_ROUTES,
        "student_routes": STUDENT_ROUTES,
        "aliases_excluded_from_inference": ALIASES,
        "inference_route_policy": (
            "Only the canonical teacher_routes and student_routes are analyzed. "
            "Alias records must identify their canonical source and contain exactly "
            "equal metrics, but aliases are not duplicated as inferential routes."
        ),
        "views": {
            "row_level": "All persisted test rows, used to validate the source metrics.",
            "inverse_test_group_size": "All rows weighted by one divided by exact within-test feature-group size.",
            "pure_group_representative": "Smallest source-row index per label-pure exact test feature group; all mixed groups excluded.",
        },
        "grouping": {
            "features_only": True,
            "feature_count": len(FEATURE_NAMES),
            "input_representation": "finite float32 values parsed from the raw 17 WSN-DS feature columns",
            "equality": "bit-exact little-endian float32 rows after canonicalizing signed zero",
            "labels_used_to_form_group_ids": False,
            "labels_used_only_for_purity_audit": True,
            "deterministic_representative": "minimum source_row_index",
            "prediction_consistency": {
                "predicted_class_equality": "exact",
                "probability_max_abs_delta_tolerance": (
                    WITHIN_GROUP_PROBABILITY_ATOL
                ),
                "tolerance_scope": (
                    "persisted float32 and decimal probability representations only"
                ),
                "probabilities_used_for_group_weighting": False,
            },
            "test_group_count": groups["group_count"],
            "repeated_group_count": groups["repeated_group_count"],
            "rows_in_repeated_groups": groups["rows_in_repeated_groups"],
            "mixed_label_group_count": groups["mixed_group_count"],
            "mixed_label_row_count": groups["mixed_row_count"],
            "pure_group_count": groups["pure_group_count"],
        },
        "statistics": {
            "unit": "paired algorithmic run seed on one fixed test split",
            "seed_count": len(SEEDS),
            "standard_deviation": "sample SD across seeds, ddof=1",
            "primary_inference_policy": PRIMARY_INFERENCE_POLICY,
            "paired_tests": [
                "exact paired Wilcoxon signed-rank test by exhaustive sign enumeration of nonzero paired differences; absolute signed-rank statistic, average ranks for ties, and zero_method=wilcox",
                "exact enumeration of all 2^10 sign assignments for the absolute mean difference",
            ],
            "teacher_comparisons": [list(pair) for pair in TEACHER_COMPARISONS],
            "student_comparisons": [list(pair) for pair in STUDENT_COMPARISONS],
            "holm_families": "separate view:teacher, view:student_A, and view:student_B families; a view-global family is also reported; no correction is pooled across the three distinct views",
            "supersession": (
                "For route inference, exhaustive Wilcoxon p-values and their Holm "
                "adjustments from this analyzer supersede approximate Wilcoxon fields "
                "in the source full-route aggregate. Training outputs and metrics are "
                "not changed."
            ),
        },
        "claim_boundaries": claim_boundaries(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    fingerprint_payload = dict(payload)
    fingerprint_payload.pop("created_utc", None)
    payload["execution_fingerprint_sha256"] = hashlib.sha256(
        canonical_json_bytes(fingerprint_payload)
    ).hexdigest()
    return payload


def build_group_summary(groups: dict[str, Any], source_indices: np.ndarray) -> dict[str, Any]:
    size_distribution = pd.Series(groups["group_sizes"]).value_counts().sort_index()
    return {
        "protocol_id": PROTOCOL_ID,
        "test_rows": int(len(source_indices)),
        "test_exact_feature_groups": groups["group_count"],
        "singleton_groups": int(np.count_nonzero(groups["group_sizes"] == 1)),
        "repeated_groups": groups["repeated_group_count"],
        "rows_in_repeated_groups": groups["rows_in_repeated_groups"],
        "mixed_label_groups": groups["mixed_group_count"],
        "mixed_label_rows": groups["mixed_row_count"],
        "pure_groups": groups["pure_group_count"],
        "representative_view_rows": int(groups["representative_pure"].sum()),
        "inverse_weight_total": float(groups["inverse_weights"].sum()),
        "group_size_distribution": {
            str(int(size)): int(count) for size, count in size_distribution.items()
        },
        "mixed_label_policy": {
            "inverse_test_group_size": "retain every row with its recorded label",
            "pure_group_representative": "exclude every mixed-label group and row",
            "majority_label_assignment": False,
        },
    }


def make_markdown(
    group_summary: dict[str, Any], aggregate: dict[str, Any]
) -> str:
    lines = [
        "# WSN-DS Group-Balanced Route Sensitivity",
        "",
        "## Scope",
        "",
        f"The analysis validates and post-processes {group_summary['test_rows']:,} rows from one fixed feature-group-disjoint test split across ten paired algorithmic seeds. No model is fitted.",
        "",
        "Primary inference uses row-level metrics, exhaustive paired Wilcoxon tests, and Holm adjustment within the predeclared teacher, Student A, or Student B route family. View-global Holm values and the two group-balanced views are stricter multiplicity and repeated-pattern sensitivity analyses.",
        "",
        "The exact p-values reported here supersede the source runner's approximate Wilcoxon fields. Model artifacts, predictions, and performance metrics are unchanged.",
        "",
        "## Test-Pattern Accounting",
        "",
        "| Quantity | Count |",
        "|---|---:|",
        f"| Exact feature groups | {group_summary['test_exact_feature_groups']:,} |",
        f"| Singleton groups | {group_summary['singleton_groups']:,} |",
        f"| Repeated groups | {group_summary['repeated_groups']:,} |",
        f"| Rows in repeated groups | {group_summary['rows_in_repeated_groups']:,} |",
        f"| Mixed-label groups | {group_summary['mixed_label_groups']:,} |",
        f"| Mixed-label rows | {group_summary['mixed_label_rows']:,} |",
        f"| Label-pure representative rows | {group_summary['representative_view_rows']:,} |",
        "",
        "The inverse-size view retains mixed-label rows with their recorded labels. The representative view excludes every mixed-label group and row. Neither view assigns a majority label.",
        "",
        "## Route Macro-F1",
        "",
        "| View | Category | Student | Route | Mean | Sample SD |",
        "|---|---|---|---|---:|---:|",
    ]
    route_frame = aggregate["route_aggregate"].sort_values(
        ["view", "category", "student", "route"]
    )
    for _, row in route_frame.iterrows():
        lines.append(
            f"| {row['view']} | {row['category']} | {row['student'] or '-'} | {row['route']} | {row['macro_f1_mean']:.6f} | {row['macro_f1_sample_sd']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Paired Route Tests",
            "",
            "| View | Family | Contrast | Mean delta | Exact signed-rank p | Family Holm p | View-global Holm p |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in aggregate["paired_tests"].sort_values(["view", "family", "name"]).iterrows():
        lines.append(
            f"| {row['view']} | {row['family']} | {row['left_route']} - {row['right_route']} | {row['mean_macro_f1_delta']:.6f} | {row['exact_paired_wilcoxon_p_two_sided']:.6f} | {row['holm_exact_paired_wilcoxon_p']:.6f} | {row['holm_view_global_exact_paired_wilcoxon_p']:.6f} |"
        )
    lines.extend(["", "## Claim Boundaries", ""])
    lines.extend(f"- {item}" for item in claim_boundaries())
    lines.append("")
    return "\n".join(lines)


def prepare_output(
    output: Path,
    resume: bool,
    contract: dict[str, Any],
    protected: Iterable[Path],
) -> bool:
    protected = {Path(path).resolve() for path in protected}
    resolved = output.resolve()
    for root in protected:
        if resolved == root:
            raise RuntimeError(f"Output equals a protected source path: {resolved}")
        try:
            resolved.relative_to(root)
            raise RuntimeError(f"Output is inside a protected source path: {resolved}")
        except ValueError:
            pass
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise FileExistsError(f"Refusing non-empty output without --resume: {output}")
        expected = set(OUTPUT_FILES)
        manifest = verify_manifest(output, PROTOCOL_ID, expected_inventory=expected)
        observed_contract = read_json(output / "execution_contract.json")
        comparable_observed = dict(observed_contract)
        comparable_expected = dict(contract)
        comparable_observed.pop("created_utc", None)
        comparable_expected.pop("created_utc", None)
        if comparable_observed != comparable_expected:
            raise RuntimeError("Existing complete output contract differs from current inputs")
        aggregate = read_json(output / "aggregate_results.json")
        if aggregate.get("status") != "complete" or aggregate.get("protocol_id") != PROTOCOL_ID:
            raise RuntimeError("Existing aggregate is not complete")
        if aggregate.get("execution_contract_sha256") != sha256_file(
            output / "execution_contract.json"
        ):
            raise RuntimeError("Existing aggregate is not bound to its execution contract")
        if manifest["payload"].get("file_count_excluding_manifest") != len(OUTPUT_FILES):
            raise RuntimeError("Existing output manifest count differs")
        return True
    if resume:
        raise RuntimeError("--resume requires an already complete non-empty output")
    output.mkdir(parents=True, exist_ok=False)
    return False


def write_analysis(
    output: Path,
    context: dict[str, Any],
    groups: dict[str, Any],
    route_data: dict[str, Any],
    aggregate: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    if prepare_output(
        output,
        False,
        contract,
        [context["dataset_path"], context["base_root"], context["full_root"]],
    ):
        raise AssertionError("Fresh output unexpectedly resumed")
    atomic_write_json(output / "execution_contract.json", contract)
    shutil.copyfile(SCRIPT_PATH, output / "bound_analyzer_source.py")
    shutil.copyfile(
        context["full_source_path"], output / "bound_full_route_executed_source.py"
    )
    shutil.copyfile(
        context["common_source_path"], output / "bound_shared_data_contract_source.py"
    )
    source_indices = context["split"]["indices"]["test"]
    labels = context["split"]["test_labels"]
    atomic_save_npz(
        output / "test_group_assignments.npz",
        source_row_index=source_indices.astype(np.int64),
        true_label=labels.astype(np.int64),
        exact_group_id=groups["group_ids"].astype(np.int64),
        exact_group_size=groups["group_sizes"][groups["group_ids"]].astype(np.int64),
        inverse_group_size_weight=groups["inverse_weights"].astype(np.float64),
        label_pure=groups["pure"][groups["group_ids"]].astype(np.bool_),
        deterministic_representative=groups["representative"].astype(np.bool_),
        retained_pure_representative=groups["representative_pure"].astype(np.bool_),
        group_key_sha256=groups["group_key_sha256"][groups["group_ids"]],
    )
    group_summary = build_group_summary(groups, source_indices)
    atomic_write_json(output / "test_group_summary.json", group_summary)
    mixed_groups, mixed_rows = group_output_frames(groups, source_indices, labels)
    atomic_write_csv(output / "mixed_label_groups.csv", mixed_groups)
    atomic_write_csv(output / "mixed_label_rows.csv", mixed_rows)
    atomic_write_csv(output / "route_seed_metrics.csv", route_data["metrics"])
    atomic_write_csv(output / "route_seed_per_class_f1.csv", route_data["classes"])
    atomic_write_csv(output / "route_aggregate.csv", aggregate["route_aggregate"])
    atomic_write_csv(
        output / "route_per_class_f1_aggregate.csv", aggregate["class_aggregate"]
    )
    atomic_write_csv(output / "sensitivity_shifts.csv", aggregate["sensitivity_shifts"])
    atomic_write_csv(output / "route_pair_deltas.csv", aggregate["route_pair_deltas"])
    atomic_write_csv(output / "paired_tests.csv", aggregate["paired_tests"])
    contract_sha256 = sha256_file(output / "execution_contract.json")
    class_aggregate_json = []
    for row in aggregate["class_aggregate"].to_dict(orient="records"):
        item = dict(row)
        item["f1_values"] = json.loads(item.pop("f1_values_json"))
        class_aggregate_json.append(item)
    aggregate_payload = {
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "execution_contract_sha256": contract_sha256,
        "seeds": SEEDS,
        "seed_count": len(SEEDS),
        "views": VIEWS,
        "primary_inference_policy": PRIMARY_INFERENCE_POLICY,
        "statistical_supersession": (
            "Exact paired Wilcoxon and Holm fields in this analysis supersede the "
            "source full-route aggregate's approximate Wilcoxon fields."
        ),
        "test_group_summary": group_summary,
        "route_aggregate": aggregate["aggregate_json"],
        "route_per_class_f1_aggregate": class_aggregate_json,
        "paired_route_tests": aggregate["test_records"],
        "within_group_prediction_probability_audit": aggregate[
            "probability_consistency"
        ],
        "aliases_excluded_from_inference": ALIASES,
        "standard_deviation_definition": "sample SD across ten paired algorithmic seeds, ddof=1",
        "claim_boundaries": claim_boundaries(),
    }
    atomic_write_json(output / "aggregate_results.json", aggregate_payload)
    atomic_write_text(output / "analysis_summary.md", make_markdown(group_summary, aggregate))
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    if actual != OUTPUT_FILES:
        raise RuntimeError(f"Output inventory differs before manifest: {sorted(actual ^ OUTPUT_FILES)}")
    atomic_write_json(
        output / "artifact_manifest.json", manifest_payload(output, PROTOCOL_ID, "complete")
    )
    verify_manifest(output, PROTOCOL_ID, expected_inventory=OUTPUT_FILES)


def run_synthetic_tests() -> None:
    features = np.asarray(
        [
            [1.0] * 17,
            [1.0] * 17,
            [1.0] * 17,
            [2.0] * 17,
            [2.0] * 17,
            [3.0] * 17,
            [4.0] * 17,
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 0, 1, 2, 2, 1], dtype=np.int64)
    source = np.asarray([50, 10, 30, 70, 60, 90, 80], dtype=np.int64)
    groups = build_test_groups(features, labels, source)
    assert groups["group_count"] == 4
    assert groups["mixed_group_count"] == 1
    assert groups["mixed_row_count"] == 2
    assert groups["pure_group_count"] == 3
    assert groups["representative_pure"].sum() == 3
    assert set(source[groups["representative_pure"]].tolist()) == {10, 80, 90}
    assert np.isclose(groups["inverse_weights"].sum(), 4.0)
    predictions = np.asarray([0, 0, 1, 1, 1, 2, 1], dtype=np.int64)
    weighted = classification_metrics(labels, predictions, groups["inverse_weights"])
    assert np.isclose(weighted["effective_weight"], 4.0)
    assert np.isclose(weighted["confusion_matrix"][0][0], 2.0 / 3.0)
    representative = classification_metrics(
        labels[groups["representative_pure"]],
        predictions[groups["representative_pure"]],
    )
    assert representative["row_count"] == 3
    differences = np.asarray([0.1] * len(SEEDS), dtype=np.float64)
    assert np.isclose(exact_sign_flip_mean_p(differences), 2.0 / (1 << len(SEEDS)))
    signed = exact_paired_wilcoxon(differences)
    assert np.isclose(signed["p_value_two_sided"], 2.0 / (1 << len(SEEDS)))
    holm = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.02})
    assert all(0.0 <= value <= 1.0 for value in holm.values())

    synthetic_metric_rows = []
    synthetic_class_rows = []
    canonical_routes = [
        ("teacher", "", route) for route in TEACHER_ROUTES
    ] + [
        ("student", student, route)
        for student in STUDENTS
        for route in STUDENT_ROUTES
    ]
    for view_index, view in enumerate(VIEWS):
        for route_index, (category, student, route) in enumerate(canonical_routes):
            for seed_index, seed in enumerate(sorted(SEEDS)):
                value = 0.70 + route_index * 0.001 + seed_index * 0.0001 + view_index * 0.0002
                synthetic_metric_rows.append(
                    {
                        "view": view,
                        "seed": seed,
                        "category": category,
                        "student": student,
                        "route": route,
                        "prediction_source": (
                            f"synthetic/seed_{seed}/{student or 'all'}/{route}.npz"
                        ),
                        "prediction_source_sha256": "0" * 64,
                        "max_within_exact_group_probability_delta": 0.0,
                        "within_exact_group_probability_tolerance": (
                            WITHIN_GROUP_PROBABILITY_ATOL
                        ),
                        "accuracy": value + 0.02,
                        "macro_precision": value - 0.01,
                        "macro_recall": value + 0.01,
                        "macro_f1": value,
                    }
                )
                for class_id, class_name in enumerate(CLASS_NAMES):
                    synthetic_class_rows.append(
                        {
                            "view": view,
                            "seed": seed,
                            "category": category,
                            "student": student,
                            "route": route,
                            "class_id": class_id,
                            "class_name": class_name,
                            "f1": value + class_id * 0.00001,
                        }
                    )
    synthetic_aggregate = aggregate_analysis(
        {
            "metrics": pd.DataFrame(synthetic_metric_rows),
            "classes": pd.DataFrame(synthetic_class_rows),
        }
    )
    assert len(synthetic_aggregate["route_aggregate"]) == len(VIEWS) * len(canonical_routes)
    assert len(synthetic_aggregate["class_aggregate"]) == len(VIEWS) * len(canonical_routes) * len(CLASS_NAMES)
    expected_tests_per_view = len(TEACHER_COMPARISONS) + len(STUDENTS) * len(STUDENT_COMPARISONS)
    assert len(synthetic_aggregate["paired_tests"]) == len(VIEWS) * expected_tests_per_view
    assert len(synthetic_aggregate["route_pair_deltas"]) == len(VIEWS) * expected_tests_per_view * len(SEEDS)
    assert {
        "exact_paired_wilcoxon_p_two_sided",
        "holm_exact_paired_wilcoxon_p",
        "exact_sign_flip_mean_p_two_sided",
        "holm_exact_sign_flip_mean_p",
    }.issubset(synthetic_aggregate["paired_tests"].columns)
    assert all(
        synthetic_aggregate["paired_tests"]["holm_family_size"]
        == synthetic_aggregate["paired_tests"]["family"].map(
            {"teacher": len(TEACHER_COMPARISONS), **{
                student: len(STUDENT_COMPARISONS) for student in STUDENTS
            }}
        )
    )
    probabilities = np.eye(len(CLASS_NAMES), dtype=np.float64)[labels] * 0.9 + 0.02
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    expected = classification_metrics(labels, probabilities.argmax(axis=1))
    expected["ece_15_bin"] = expected_calibration_error(probabilities, labels)
    with tempfile.TemporaryDirectory(prefix="fgds_group_balance_synthetic_") as temporary:
        root = Path(temporary)
        npz_path = root / "predictions.npz"
        with npz_path.open("wb") as handle:
            np.savez(
                handle,
                source_row_index=source,
                true_label=labels,
                probability=probabilities.astype(np.float32),
                predicted_label=probabilities.argmax(axis=1).astype(np.int64),
            )
        expected_npz = classification_metrics(labels, probabilities.argmax(axis=1))
        stored_probabilities = probabilities.astype(np.float32).astype(np.float64)
        expected_npz["ece_15_bin"] = expected_calibration_error(stored_probabilities, labels)
        load_prediction_npz(npz_path, source, labels, expected_npz)
        manifest_root = root / "manifest_test"
        manifest_root.mkdir()
        atomic_write_text(manifest_root / "a.txt", "a\n")
        atomic_write_json(
            manifest_root / "artifact_manifest.json",
            manifest_payload(manifest_root, "synthetic", "complete"),
        )
        verify_manifest(manifest_root, "synthetic", expected_inventory={"a.txt"})
        atomic_write_text(manifest_root / "a.txt", "tampered\n")
        try:
            verify_manifest(manifest_root, "synthetic", expected_inventory={"a.txt"})
        except RuntimeError:
            pass
        else:
            raise AssertionError("Synthetic manifest tampering was not detected")
    print("synthetic tests passed")


def main() -> int:
    args = parse_args()
    selected_modes = sum(
        int(value) for value in [args.preflight_only, args.confirm_analysis, args.self_test]
    )
    if selected_modes > 1:
        raise RuntimeError(
            "Choose only one of --preflight-only, --confirm-analysis, or --self-test"
        )
    if args.self_test:
        if args.resume:
            raise RuntimeError("--resume cannot be combined with --self-test")
        run_synthetic_tests()
        return 0
    context = load_context(
        args.dataset_csv,
        args.base_root,
        args.full_route_root,
        args.full_route_executed_source,
    )
    groups = build_test_groups(
        context["split"]["test_features"],
        context["split"]["test_labels"],
        context["split"]["indices"]["test"],
    )
    contract = build_execution_contract(context, groups)
    print(
        json.dumps(
            {
                "protocol_id": PROTOCOL_ID,
                "full_route_manifest_sha256": context["full_manifest"]["sha256"],
                "full_route_executed_source_sha256": context["full_source_sha256"],
                "dataset_sha256": context["dataset"]["dataset_sha256"],
                "split_indices_sha256": context["split"]["index_hash"],
                "scaler_sha256": context["split"]["scaler_hash"],
                "test_rows": EXPECTED_TEST_ROWS,
                "test_exact_feature_groups": groups["group_count"],
                "mixed_label_groups": groups["mixed_group_count"],
                "mixed_label_rows": groups["mixed_row_count"],
                "output": str(args.output_dir.resolve()),
            },
            indent=2,
        ),
        flush=True,
    )
    if not args.confirm_analysis:
        if args.resume:
            raise RuntimeError("--resume requires --confirm-analysis")
        print("Preflight complete. No analysis output was created.", flush=True)
        return 0
    if args.resume and not args.output_dir.exists():
        raise RuntimeError("--resume requires an already complete existing output")
    if args.output_dir.exists():
        if args.resume and any(args.output_dir.iterdir()):
            if prepare_output(
                args.output_dir,
                True,
                contract,
                [context["dataset_path"], context["base_root"], context["full_root"]],
            ):
                print(f"Verified complete existing analysis: {args.output_dir}")
                return 0
        raise FileExistsError(
            f"Refusing existing output unless it is complete and --resume is used: "
            f"{args.output_dir}"
        )
    route_data = analyze_routes(context, groups)
    aggregate = aggregate_analysis(route_data)
    write_analysis(args.output_dir, context, groups, route_data, aggregate, contract)
    print(f"Analysis complete: {args.output_dir / 'aggregate_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
