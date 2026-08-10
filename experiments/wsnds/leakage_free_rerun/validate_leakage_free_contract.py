"""Preflight checks for the additive WSN-DS train-only-scaler v2 rerun."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DATASET = REPO_ROOT / "data" / "wsnds" / "WSN-DS.csv"
MAIN_RUNNER = SCRIPT_DIR / "run_leakage_free_wsnds.py"
J_RUNNER = SCRIPT_DIR / "run_leakage_free_codistillation.py"
REPORT = REPO_ROOT / "results" / "wsnds" / "leakage_free_rerun" / "preflight_report_v2.json"
SEEDS = [42, 123, 456, 789, 1001, 2024, 3141, 5678, 8192, 9999]
BASE_PROTOCOL_ID = "archive_random_split_train_scaler_controlled_tuning_v2"
J_PROTOCOL_ID = "archive_random_split_train_scaler_codistillation_v2"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_raw() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    frame = pd.read_csv(DATASET)
    frame.columns = frame.columns.str.strip()
    target = next(
        (name for name in ["Attack type", "Attack_Type", "attack_type", "Attack Type", "class"] if name in frame.columns),
        frame.columns[-1],
    )
    for candidate in ["id", "Id", "ID"]:
        if candidate in frame.columns:
            frame = frame.drop(columns=[candidate])
            break
    encoder = LabelEncoder()
    labels = encoder.fit_transform(frame[target].astype(str).str.strip()).astype(np.int64)
    features = frame.drop(columns=[target])
    return (
        features.to_numpy(dtype=np.float32),
        labels,
        features.columns.tolist(),
        encoder.classes_.tolist(),
    )


def main() -> int:
    main_source = MAIN_RUNNER.read_text(encoding="utf-8")
    j_source = J_RUNNER.read_text(encoding="utf-8")
    for name, source in [("main", main_source), ("codistillation", j_source)]:
        require(
            "fit_transform(X_all" not in source,
            f"{name} fits a scaler and transforms all rows before splitting",
        )
        require(
            "X_all_std" not in source,
            f"{name} still uses the archived globally scaled matrix",
        )
        require(
            "scaler.fit_transform(X_train_raw)" in source,
            f"{name} does not fit its model-input scaler on raw training rows",
        )
    require(str(SEEDS) in main_source, "Publication seed set is absent from the main runner")
    require(
        f"PROTOCOL_ID = '{BASE_PROTOCOL_ID}'" in main_source,
        "Main protocol ID is missing or incorrect",
    )
    require("main_10seed_v2" in main_source, "Main v2 output isolation is absent")
    require(
        "set_seed(GRID_TUNING_SEED)" in main_source,
        "Main KD grid does not reset the candidate seed",
    )
    require(
        "archived_global_scaler = StandardScaler().fit(X_all)" in main_source,
        "Expected diagnostic global scaler is absent",
    )
    require(
        "global_scaler_role': 'diagnostic_only_never_transforms_model_inputs'" in main_source,
        "Diagnostic-only global scaler scope is not recorded",
    )
    require(
        "X_ctr = scaler_c.fit_transform(X_ctr_raw)" in main_source
        and "scaler_c.fit_transform(X_c)" not in main_source,
        "Optional CICIoT preprocessing is not train-only",
    )
    require(
        f"BASE_PROTOCOL_ID = '{BASE_PROTOCOL_ID}'" in j_source
        and f"J_PROTOCOL_ID = '{J_PROTOCOL_ID}'" in j_source,
        "Config J protocol IDs are missing or incorrect",
    )
    require("main_10seed_v2" in j_source, "Config J defaults do not target the v2 base")
    require(
        "_load_validated_rf_cache" in j_source
        and "probabilities_sha256" in j_source
        and "rf_calibration_config" in j_source,
        "Config J does not validate RF cache provenance and content",
    )
    require(
        "subprocess.check_call" not in j_source,
        "Config J still installs packages during an experiment",
    )

    X, y, feature_names, class_names = load_raw()
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1765, random_state=42, stratify=y_trainval
    )
    require(X.shape == (374661, 17), f"Unexpected dataset shape: {X.shape}")
    require(X_train.shape == (262252, 17), f"Unexpected train shape: {X_train.shape}")
    require(X_val.shape == (56209, 17), f"Unexpected validation shape: {X_val.shape}")
    require(X_test.shape == (56200, 17), f"Unexpected test shape: {X_test.shape}")

    scaler = StandardScaler().fit(X_train)
    global_scaler = StandardScaler().fit(X)
    mean_shift_sd = np.abs(scaler.mean_ - global_scaler.mean_) / global_scaler.scale_
    scale_change = np.abs(scaler.scale_ / global_scaler.scale_ - 1.0)

    def row_hashes(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        hash_frame = pd.DataFrame(features, columns=feature_names)
        hash_frame["__label"] = labels
        return pd.util.hash_pandas_object(hash_frame, index=False).to_numpy(dtype=np.uint64)

    all_hashes = row_hashes(X, y)
    train_hashes = row_hashes(X_train, y_train)
    validation_hashes = row_hashes(X_val, y_val)
    test_hashes = row_hashes(X_test, y_test)
    all_counts = pd.Series(all_hashes).value_counts()
    train_hash_set = set(map(int, train_hashes))
    validation_hash_set = set(map(int, validation_hashes))
    duplicate_audit = {
        "duplicate_rows_excluding_first": int(len(all_hashes) - len(set(map(int, all_hashes)))),
        "rows_in_duplicate_groups": int(all_counts[all_counts > 1].sum()),
        "validation_rows_exactly_present_in_train": int(sum(
            int(value) in train_hash_set for value in validation_hashes
        )),
        "test_rows_exactly_present_in_train": int(sum(
            int(value) in train_hash_set for value in test_hashes
        )),
        "test_rows_exactly_present_in_validation": int(sum(
            int(value) in validation_hash_set for value in test_hashes
        )),
    }
    require(
        duplicate_audit["duplicate_rows_excluding_first"] == 13502,
        f"Unexpected whole-dataset duplicate count: {duplicate_audit}",
    )
    require(
        duplicate_audit["rows_in_duplicate_groups"] == 23277,
        f"Unexpected duplicate-group row count: {duplicate_audit}",
    )
    require(
        duplicate_audit["test_rows_exactly_present_in_train"] == 2695,
        f"Unexpected test-to-train duplicate crossover: {duplicate_audit}",
    )

    report = {
        "status": "passed",
        "dataset": str(DATASET),
        "raw_shape": list(X.shape),
        "split_shapes": {
            "train": list(X_train.shape),
            "validation": list(X_val.shape),
            "test": list(X_test.shape),
        },
        "split_random_state": 42,
        "scaler_fit_partition": "train",
        "feature_names": feature_names,
        "class_names": class_names,
        "seeds": SEEDS,
        "class_counts": {
            "train": np.bincount(y_train).tolist(),
            "validation": np.bincount(y_val).tolist(),
            "test": np.bincount(y_test).tolist(),
        },
        "scaler_diagnostic": {
            "mean_shift_in_global_sd_max": float(mean_shift_sd.max()),
            "mean_shift_in_global_sd_mean": float(mean_shift_sd.mean()),
            "relative_scale_change_max": float(scale_change.max()),
            "relative_scale_change_mean": float(scale_change.mean()),
        },
        "duplicate_audit": duplicate_audit,
        "source_guards": {
            "global_scaler_is_diagnostic_only": True,
            "global_scaled_matrix_absent": True,
            "train_fit_present": True,
            "ten_seed_set_present": True,
            "controlled_kd_grid_present": True,
            "optional_ciciot_train_only_scaling_present": True,
            "config_j_cache_hash_validation_present": True,
            "runtime_package_install_absent": True,
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(REPORT)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
