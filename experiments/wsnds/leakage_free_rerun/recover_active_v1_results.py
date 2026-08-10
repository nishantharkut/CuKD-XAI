"""Recover complete ten-seed WSN-DS metrics from active-v1 seed checkpoints.

This is a report-only fallback for failures after model training. It does not
train models and refuses to operate until all 20 A/B checkpoint files exist.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy.stats import wilcoxon


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.wsnds.leakage_free_rerun.tier15_common import (  # noqa: E402
    apply_train_scaler,
    archived_random_split,
    load_wsnds,
    sha256_arrays,
    split_hashes,
)

ACTIVE_ROOT = REPO_ROOT / "results" / "wsnds" / "leakage_free_rerun" / "main_10seed"
DEFAULT_OUTPUT = (
    REPO_ROOT / "results" / "wsnds" / "leakage_free_rerun" / "recovered_main_10seed_v1"
)
SEEDS = [42, 123, 456, 789, 1001, 2024, 3141, 5678, 8192, 9999]
CLASS_NAMES = ["Blackhole", "Flooding", "Grayhole", "Normal", "TDMA"]
REQUIRED_CONFIGS = {
    "A_RF_500",
    "B_Full_MLP",
    "C_CL_MLP_loss_fair",
    "C_CL_MLP_loss_ext",
    "C_CL_MLP_loss",
    "C2_CL_MLP_domain",
    "D_Small_MLP",
    "E_KD_from_RF",
    "E2_KD_from_MLP",
    "F_KD_from_CL_MLP_fair",
    "F_KD_from_CL_MLP_ext",
    "F_KD_from_CL_MLP",
    "G_KD_random_pacing",
}
OPTIONAL_CONFIGS = {"I_KD_from_SMOTE_MLP"}
EXPECTED_CONFIGS = REQUIRED_CONFIGS | OPTIONAL_CONFIGS
SOURCE_PROTOCOL = "archive_random_split_train_scaler_retuned_v1"
EXPECTED_ACTIVE_V1_SOURCE_SHA256 = (
    "728eb19b1330e94652db9cf6f57f3d0d698fae0b9849e48b4e8ba25ba20eee27"
)
EXPECTED_ACTIVE_V1_MANIFEST_SHA256 = (
    "bf4dfe1fff61e170d6967d99b842433dbcf0d3a84f8a7a3692e12df4dfd6815d"
)
BEST_KD_PATTERN = re.compile(
    r"^Best KD hyperparameters: T=4, alpha=0\.7 \(val F1 [0-9]+\.[0-9]+\)$",
    re.MULTILINE,
)
COMPARISONS = [
    ("C_CL_MLP_loss_fair", "B_Full_MLP"),
    ("C_CL_MLP_loss_ext", "B_Full_MLP"),
    ("F_KD_from_CL_MLP_fair", "E2_KD_from_MLP"),
    ("F_KD_from_CL_MLP_ext", "E2_KD_from_MLP"),
    ("F_KD_from_CL_MLP", "D_Small_MLP"),
    ("E2_KD_from_MLP", "D_Small_MLP"),
    ("F_KD_from_CL_MLP", "G_KD_random_pacing"),
    ("F_KD_from_CL_MLP", "I_KD_from_SMOTE_MLP"),
    ("E_KD_from_RF", "E2_KD_from_MLP"),
]
EXPECTED_TEST_SUPPORT = np.asarray([1507, 497, 2189, 51011, 996], dtype=np.int64)
STUDENT_CONFIGS = {
    "D_Small_MLP",
    "E_KD_from_RF",
    "E2_KD_from_MLP",
    "F_KD_from_CL_MLP_fair",
    "F_KD_from_CL_MLP_ext",
    "F_KD_from_CL_MLP",
    "G_KD_random_pacing",
    "I_KD_from_SMOTE_MLP",
}
ALIAS_PAIRS = [
    ("C_CL_MLP_loss", "C_CL_MLP_loss_fair"),
    ("F_KD_from_CL_MLP", "F_KD_from_CL_MLP_fair"),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_output(path: Path) -> tuple[Path, Path]:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite recovery output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    if temporary.exists():
        raise FileExistsError(f"Stale recovery temporary directory exists: {temporary}")
    temporary.mkdir()
    return path, temporary


def require_disjoint_recovery_output(path: Path, active_root: Path) -> Path:
    requested = path.resolve()
    protected_roots = [
        active_root.resolve(),
        SCRIPT_PATH.parent.resolve(),
        (REPO_ROOT / "data").resolve(),
        (REPO_ROOT / ".git").resolve(),
    ]
    for protected in protected_roots:
        try:
            requested.relative_to(protected)
            overlaps = True
        except ValueError:
            try:
                protected.relative_to(requested)
                overlaps = True
            except ValueError:
                overlaps = False
        if overlaps:
            raise RuntimeError(
                f"Recovery output overlaps protected evidence/source: {requested}"
            )
    return requested


def validate_metric_result(
    config: str,
    result: Any,
    path: Path,
    *,
    student: str | None = None,
) -> None:
    if not isinstance(result, dict):
        raise RuntimeError(f"Checkpoint result is not an object: {path}/{config}")
    for key in ["accuracy", "macro_f1"]:
        value = result.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise RuntimeError(f"Checkpoint metric is not finite: {path}/{config}/{key}")
        if not 0.0 <= float(value) <= 1.0:
            raise RuntimeError(f"Checkpoint metric is outside [0,1]: {path}/{config}/{key}")
    per_class = result.get("per_class_f1")
    if not isinstance(per_class, list) or len(per_class) != len(CLASS_NAMES):
        raise RuntimeError(f"Checkpoint per-class F1 has the wrong shape: {path}/{config}")
    if any(
        not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
        for value in per_class
    ):
        raise RuntimeError(f"Checkpoint per-class F1 is invalid: {path}/{config}")
    for key in ["params", "model_size_kb"]:
        value = result.get(key)
        if value is not None and (
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise RuntimeError(f"Checkpoint resource field is invalid: {path}/{config}/{key}")
    if config in STUDENT_CONFIGS and student is not None:
        expected_params = 1189 if student == "A" else 3397
        if result.get("params") != expected_params:
            raise RuntimeError(
                f"Checkpoint parameter count differs: {path}/{config}: "
                f"{result.get('params')!r} != {expected_params}"
            )

    confusion = result.get("confusion_matrix")
    if confusion is None:
        if config != "A_RF_500":
            raise RuntimeError(f"Checkpoint neural result lacks a confusion matrix: {path}/{config}")
        return
    matrix = np.asarray(confusion)
    if matrix.shape != (len(CLASS_NAMES), len(CLASS_NAMES)):
        raise RuntimeError(f"Checkpoint confusion matrix has the wrong shape: {path}/{config}")
    if not np.issubdtype(matrix.dtype, np.integer) or np.any(matrix < 0):
        raise RuntimeError(f"Checkpoint confusion matrix is not non-negative integer data: {path}/{config}")
    matrix = matrix.astype(np.int64, copy=False)
    if not np.array_equal(matrix.sum(axis=1), EXPECTED_TEST_SUPPORT):
        raise RuntimeError(f"Checkpoint confusion-matrix support differs: {path}/{config}")
    total = int(matrix.sum())
    true_positive = np.diag(matrix).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    actual = matrix.sum(axis=1).astype(np.float64)
    precision = np.divide(
        true_positive, predicted, out=np.zeros_like(true_positive), where=predicted != 0
    )
    recall = np.divide(
        true_positive, actual, out=np.zeros_like(true_positive), where=actual != 0
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    recomputed = {
        "accuracy": float(true_positive.sum() / total),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
    }
    for key, expected in recomputed.items():
        if key not in result or not math.isclose(
            float(result[key]), expected, rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError(f"Checkpoint metric does not match its confusion matrix: {path}/{config}/{key}")
    for key, expected in [
        ("per_class_precision", precision),
        ("per_class_recall", recall),
        ("per_class_f1", f1),
    ]:
        observed = np.asarray(result.get(key), dtype=np.float64)
        if observed.shape != (len(CLASS_NAMES),) or not np.allclose(
            observed, expected, rtol=0.0, atol=1e-12
        ):
            raise RuntimeError(f"Checkpoint per-class metric differs: {path}/{config}/{key}")


def validate_aliases(results: dict[str, Any], path: Path) -> None:
    metric_keys = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "per_class_precision",
        "per_class_recall",
        "per_class_f1",
        "confusion_matrix",
        "params",
        "model_size_kb",
    ]
    for alias, source in ALIAS_PAIRS:
        for key in metric_keys:
            if results[alias].get(key) != results[source].get(key):
                raise RuntimeError(f"Checkpoint alias differs from source: {path}/{alias}/{key}")


def load_checkpoints(active_root: Path, student: str) -> tuple[dict[int, Any], list[dict[str, Any]]]:
    results = {}
    evidence = []
    manifest = json.loads(
        (active_root / "executed_source_snapshot" / "execution_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    process_started = datetime.fromisoformat(manifest["process_start_local"]).timestamp()
    for seed in SEEDS:
        path = active_root / f"checkpoint_student_{student}_seed_{seed}.json"
        if not path.is_file():
            raise RuntimeError(f"Active run is incomplete; missing {path.name}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("student") != student or payload.get("seed") != seed:
            raise RuntimeError(f"Checkpoint identity mismatch: {path}")
        if payload.get("preprocessing") != "train-only scaler":
            raise RuntimeError(f"Checkpoint preprocessing mismatch: {path}")
        checkpoint_results = payload.get("results")
        if not isinstance(checkpoint_results, dict):
            raise RuntimeError(f"Checkpoint has no result mapping: {path}")
        configs = set(checkpoint_results)
        if not REQUIRED_CONFIGS.issubset(configs) or not configs.issubset(
            REQUIRED_CONFIGS | OPTIONAL_CONFIGS
        ):
            raise RuntimeError(
                f"Configuration set differs in {path}: "
                f"missing={sorted(REQUIRED_CONFIGS - configs)}, "
                f"unexpected={sorted(configs - REQUIRED_CONFIGS - OPTIONAL_CONFIGS)}"
            )
        if path.stat().st_mtime < process_started:
            raise RuntimeError(f"Checkpoint predates the recorded active process: {path}")
        for config, result in checkpoint_results.items():
            validate_metric_result(config, result, path, student=student)
        validate_aliases(checkpoint_results, path)
        results[seed] = checkpoint_results
        evidence.append({
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    for optional in OPTIONAL_CONFIGS:
        present = [seed for seed in SEEDS if optional in results[seed]]
        if present and len(present) != len(SEEDS):
            raise RuntimeError(
                f"Optional configuration {optional} is only present for seeds {present}; "
                "recovery requires optional routes to be all-or-none"
            )
    return results, evidence


def aggregate(seed_results: dict[int, Any]) -> pd.DataFrame:
    rows = []
    configs = sorted(set().union(*(set(results) for results in seed_results.values())))
    for config in configs:
        config_rows = [seed_results[seed][config] for seed in SEEDS if config in seed_results[seed]]
        accuracies = np.asarray([row["accuracy"] for row in config_rows], dtype=np.float64)
        macro_f1 = np.asarray([row["macro_f1"] for row in config_rows], dtype=np.float64)
        per_class = np.asarray([row["per_class_f1"] for row in config_rows], dtype=np.float64)
        row: dict[str, Any] = {
            "Config": config,
            "Accuracy_mean": float(accuracies.mean()),
            "Accuracy_std": float(accuracies.std(ddof=1)) if len(accuracies) > 1 else 0.0,
            "MacroF1_mean": float(macro_f1.mean()),
            "MacroF1_std": float(macro_f1.std(ddof=1)) if len(macro_f1) > 1 else 0.0,
            "n_seeds": len(config_rows),
            "params": config_rows[0].get("params"),
            "size_kb": config_rows[0].get("model_size_kb"),
        }
        for index, class_name in enumerate(CLASS_NAMES):
            row[f"{class_name}_F1_mean"] = float(per_class[:, index].mean())
            row[f"{class_name}_F1_std"] = (
                float(per_class[:, index].std(ddof=1)) if len(per_class) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def paired_tests(seed_results: dict[int, Any]) -> dict[str, Any]:
    tests = []
    for left, right in COMPARISONS:
        common_seeds = [
            seed for seed in SEEDS
            if left in seed_results[seed] and right in seed_results[seed]
        ]
        if len(common_seeds) != len(SEEDS):
            continue
        left_values = np.asarray(
            [seed_results[seed][left]["macro_f1"] for seed in common_seeds], dtype=np.float64
        )
        right_values = np.asarray(
            [seed_results[seed][right]["macro_f1"] for seed in common_seeds], dtype=np.float64
        )
        differences = left_values - right_values
        if np.all(differences == 0):
            statistic, p_value = 0.0, 1.0
        else:
            statistic, p_value = wilcoxon(
                left_values,
                right_values,
                zero_method="wilcox",
                alternative="two-sided",
                method="auto",
            )
        tests.append({
            "left": left,
            "right": right,
            "paired_seeds": common_seeds,
            "left_values": left_values.tolist(),
            "right_values": right_values.tolist(),
            "difference_mean": float(differences.mean()),
            "statistic": float(statistic),
            "p_raw": float(p_value),
        })

    order = sorted(range(len(tests)), key=lambda index: tests[index]["p_raw"])
    running_max = 0.0
    count = len(tests)
    for rank, index in enumerate(order):
        adjusted = min(1.0, tests[index]["p_raw"] * (count - rank))
        running_max = max(running_max, adjusted)
        tests[index]["p_holm"] = running_max
        tests[index]["reject_holm_0_05"] = running_max <= 0.05
    return {f"{item['left']}_vs_{item['right']}": item for item in tests}


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-root", type=Path, default=ACTIVE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    active_root = args.active_root.resolve()
    source_manifest_path = active_root / "executed_source_snapshot" / "execution_manifest.json"
    preprocessing_path = active_root / "leakage_free_preprocessing.json"
    run_log_path = active_root / "run.log"
    for path in [source_manifest_path, preprocessing_path, run_log_path]:
        if not path.is_file():
            raise FileNotFoundError(path)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    if sha256_file(source_manifest_path).lower() != EXPECTED_ACTIVE_V1_MANIFEST_SHA256:
        raise RuntimeError("Active execution manifest is not the pinned v1 manifest")
    if source_manifest.get("protocol_id") != SOURCE_PROTOCOL:
        raise RuntimeError("Active source manifest protocol is not the expected v1 protocol")
    if source_manifest.get("source_file") != "run_leakage_free_wsnds.executed.py":
        raise RuntimeError("Active source manifest names an unexpected source snapshot")
    if not isinstance(source_manifest.get("launcher_pid"), int) or not isinstance(
        source_manifest.get("worker_pid"), int
    ):
        raise RuntimeError("Active source manifest lacks recorded process identities")
    expected_output = (REPO_ROOT / source_manifest.get("output_directory", "")).resolve()
    if expected_output != active_root:
        raise RuntimeError("Active source manifest records a different output directory")
    snapshot_root = (active_root / "executed_source_snapshot").resolve()
    source_path = (snapshot_root / source_manifest["source_file"]).resolve()
    try:
        source_path.relative_to(snapshot_root)
    except ValueError as exc:
        raise RuntimeError("Active source snapshot escapes its evidence directory") from exc
    source_sha256 = sha256_file(source_path).lower()
    if source_sha256 != EXPECTED_ACTIVE_V1_SOURCE_SHA256:
        raise RuntimeError("Active executed-source snapshot is not the pinned v1 source")
    if source_sha256 != source_manifest["source_sha256"].lower():
        raise RuntimeError("Active executed-source snapshot SHA-256 mismatch")
    dataset_path = REPO_ROOT / "data" / "wsnds" / "WSN-DS.csv"
    expected_preprocessing = {
        "dataset_sha256": sha256_file(dataset_path),
        "split_random_state": 42,
        "scaler_fit_partition": "train",
        "seeds": SEEDS,
        "feature_names": [
            "Time", "Is_CH", "who CH", "Dist_To_CH", "ADV_S", "ADV_R",
            "JOIN_S", "JOIN_R", "SCH_S", "SCH_R", "Rank", "DATA_S",
            "DATA_R", "Data_Sent_To_BS", "dist_CH_To_BS", "send_code",
            "Expaned Energy",
        ],
        "class_names": CLASS_NAMES,
        "split_shapes": {
            "train": [262252, 17],
            "validation": [56209, 17],
            "test": [56200, 17],
        },
    }
    for key, value in expected_preprocessing.items():
        if preprocessing.get(key) != value:
            raise RuntimeError(f"Active preprocessing evidence differs for {key}")
    dataset = load_wsnds(dataset_path)
    split = archived_random_split(dataset["features"], dataset["labels"])
    _, scaler = apply_train_scaler(split)
    recomputed_shapes = {
        "train": list(split["X_train_raw"].shape),
        "validation": list(split["X_validation_raw"].shape),
        "test": list(split["X_test_raw"].shape),
    }
    if recomputed_shapes != preprocessing.get("split_shapes"):
        raise RuntimeError("Recomputed split shapes differ from active preprocessing evidence")
    for key, expected in [
        ("scaler_mean", np.asarray(scaler.mean_, dtype=np.float64)),
        ("scaler_scale", np.asarray(scaler.scale_, dtype=np.float64)),
    ]:
        values = np.asarray(preprocessing.get(key), dtype=np.float64)
        if values.shape != (17,) or not np.array_equal(values, expected):
            raise RuntimeError(f"Active preprocessing {key} differs from recomputation")
    recomputed_preprocessing = {
        "dataset_sha256": dataset["dataset_sha256"],
        "split_hashes": split_hashes(split),
        "scaler_sha256": sha256_arrays(
            np.asarray(scaler.mean_, dtype=np.float64),
            np.asarray(scaler.scale_, dtype=np.float64),
            np.asarray(scaler.var_, dtype=np.float64),
        ),
    }
    run_log = run_log_path.read_text(encoding="utf-8", errors="replace")
    kd_matches = BEST_KD_PATTERN.findall(run_log)
    if len(kd_matches) != 1:
        raise RuntimeError(
            f"Expected exactly one preserved T=4/alpha=0.7 selection line, found {len(kd_matches)}"
        )

    student_a, evidence_a = load_checkpoints(active_root, "A")
    student_b, evidence_b = load_checkpoints(active_root, "B")
    if set(student_a) != set(SEEDS) or set(student_b) != set(SEEDS):
        raise RuntimeError("Recovered checkpoint seeds differ from the ten-seed protocol")
    requested_output = require_disjoint_recovery_output(args.output_dir, active_root)
    output_dir, temporary_dir = prepare_output(requested_output)
    aggregate_a = aggregate(student_a)
    aggregate_b = aggregate(student_b)
    atomic_csv(temporary_dir / "recovered_wsnds_results_student_A.csv", aggregate_a)
    atomic_csv(temporary_dir / "recovered_wsnds_results_student_B.csv", aggregate_b)

    paired_a = paired_tests(student_a)
    paired_b = paired_tests(student_b)
    report = {
        "status": "metric_recovery_from_complete_checkpoint_set",
        "scope": (
            "Metric recovery only. SHAP, quantization, runtime, deployment models, "
            "and plots are not reconstructed by this tool."
        ),
        "seeds": SEEDS,
        "optimization_seed_count": 10,
        "fixed_split": True,
        "split_interpretation": (
            "Ten optimizer seeds evaluated on one fixed archived random-row split. "
            "Exact duplicate feature rows cross partitions, so these statistics are "
            "conditional on that split and are not population-level uncertainty estimates."
        ),
        "source_execution_manifest": source_manifest,
        "source_execution_manifest_sha256": sha256_file(source_manifest_path),
        "source_preprocessing_report": preprocessing,
        "source_preprocessing_report_sha256": sha256_file(preprocessing_path),
        "independent_preprocessing_recomputation": recomputed_preprocessing,
        "source_run_log_path_recorded": str(run_log_path),
        "source_run_log_sha256": sha256_file(run_log_path),
        "selected_kd_hyperparameters": {"T": 4, "alpha": 0.7},
        "recovery_script_sha256": sha256_file(SCRIPT_PATH),
        "statistical_procedure": {
            "scipy_version": scipy.__version__,
            "test": "scipy.stats.wilcoxon",
            "zero_method": "wilcox",
            "alternative": "two-sided",
            "method": "auto",
            "multiplicity": "Holm step-down adjustment",
            "family_size_by_student": {
                "student_A": len(paired_a),
                "student_B": len(paired_b),
            },
            "alpha": 0.05,
            "standard_deviation_ddof": 1,
            "initialization_matching_boundary": (
                "The active-v1 checkpoint routes were not recorded as matched model "
                "initializations across every comparison; paired tests pair optimizer "
                "seed labels on the fixed split, not guaranteed identical initial states."
            ),
        },
        "checkpoint_evidence": {"student_A": evidence_a, "student_B": evidence_b},
        "student_A_seed_results": student_a,
        "student_B_seed_results": student_b,
        "student_A_paired_tests_holm": paired_a,
        "student_B_paired_tests_holm": paired_b,
        "checkpoint_provenance_boundary": (
            "The v1 checkpoint JSON files do not independently embed source, split, "
            "scaler, or KD hashes. This fallback binds them by exact directory, filename, "
            "identity fields, post-process-start modification time, the preserved executed "
            "source manifest, independently recomputed deterministic split/scaler, "
            "preprocessing report, and run log. It recovers reported metrics only, is not "
            "model-artifact recovery, and is retained as fallback rather than primary "
            "checkpoint-level provenance."
        ),
    }
    report_path = temporary_dir / "recovered_results.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    files = [
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(temporary_dir.iterdir())
        if path.is_file() and path.name != "artifact_manifest.json"
    ]
    (temporary_dir / "artifact_manifest.json").write_text(
        json.dumps({
            "status": "complete",
            "source_protocol_id": source_manifest["protocol_id"],
            "file_count_excluding_manifest": len(files),
            "files": files,
        }, indent=2),
        encoding="utf-8",
    )
    try:
        os.replace(temporary_dir, output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    print(output_dir / "recovered_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
