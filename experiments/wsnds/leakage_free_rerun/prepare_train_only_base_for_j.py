"""Reconstruct a J-compatible train-only base package from main_10seed checkpoints.

The official run_leakage_free_codistillation.py expects:
  - complete 10-seed cukd_xai_results.json with protocol metadata
  - artifact_manifest.json (sha256 + sizes)
  - rf_soft_seed_{seed}.npy + matching .manifest.json

main_10seed already has checkpoints + RF soft npy files, but the summary JSON
only contains seeds 8192/9999 and lacks protocol/completion fields.

This script writes a *copy* package under main_10seed_v2_reconstructed/ and
does not modify the incomplete original summary.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "results/wsnds/leakage_free_rerun/main_10seed"
OUT = ROOT / "results/wsnds/leakage_free_rerun/main_10seed_v2_reconstructed"
WSNDS_PATH = ROOT / "data/wsnds/WSN-DS.csv"
SEEDS = [42, 123, 456, 789, 1001, 2024, 3141, 5678, 8192, 9999]
BASE_PROTOCOL_ID = "archive_random_split_train_scaler_controlled_tuning_v2"
REQUIRED_CONFIGS = [
    "A_RF_500",
    "B_Full_MLP",
    "C_CL_MLP_loss_fair",
    "D_Small_MLP",
    "E_KD_from_RF",
    "E2_KD_from_MLP",
    "F_KD_from_CL_MLP_fair",
    "F_KD_from_CL_MLP",
]
RF_CALIBRATION_CONFIG = {
    "n_estimators": 500,
    "max_depth": 15,
    "calibration_method": "isotonic",
    "calibration_cv": 3,
}


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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def rebuild_split_and_hashes() -> dict:
    df = pd.read_csv(WSNDS_PATH)
    df.columns = df.columns.str.strip()
    target_col = None
    for cand in ["Attack type", "Attack_Type", "attack_type", "Attack Type", "class"]:
        if cand in df.columns:
            target_col = cand
            break
    if target_col is None:
        target_col = df.columns[-1]
    for id_col in ["id", "Id", "ID"]:
        if id_col in df.columns:
            df = df.drop(columns=[id_col])
            break
    df[target_col] = df[target_col].astype(str).str.strip()
    le = LabelEncoder()
    y_all = le.fit_transform(df[target_col]).astype(np.int64)
    X_all = df.drop(columns=[target_col]).values.astype(np.float32)
    feature_names = df.drop(columns=[target_col]).columns.tolist()
    class_names = le.classes_.tolist()

    X_trainval_raw, X_test_raw, y_trainval, y_test = train_test_split(
        X_all, y_all, test_size=0.15, random_state=42, stratify=y_all
    )
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_trainval_raw, y_trainval, test_size=0.1765, random_state=42, stratify=y_trainval
    )
    scaler = StandardScaler()
    scaler.fit(X_train_raw)
    return {
        "feature_names": feature_names,
        "class_names": class_names,
        "dataset_sha256": sha256_file(WSNDS_PATH),
        "split_hashes": {
            "train_raw_features_and_labels_sha256": sha256_arrays(X_train_raw, y_train),
            "validation_raw_features_and_labels_sha256": sha256_arrays(X_val_raw, y_val),
            "test_raw_features_and_labels_sha256": sha256_arrays(X_test_raw, y_test),
        },
        "scaler_mean": scaler.mean_.astype(np.float64).tolist(),
        "scaler_scale": scaler.scale_.astype(np.float64).tolist(),
        "scaler_sha256": sha256_arrays(scaler.mean_, scaler.scale_),
        "n_train": int(len(X_train_raw)),
        "n_val": int(len(X_val_raw)),
        "n_test": int(len(X_test_raw)),
        "input_dim": int(X_all.shape[1]),
        "num_classes": int(len(class_names)),
    }


def load_checkpoints() -> tuple[dict, dict]:
    student_a: dict = {}
    student_b: dict = {}
    for seed in SEEDS:
        for student, store in (("A", student_a), ("B", student_b)):
            path = SRC / f"checkpoint_student_{student}_seed_{seed}.json"
            payload = load_json(path)
            results = payload["results"]
            for cfg in REQUIRED_CONFIGS:
                if cfg not in results:
                    raise RuntimeError(f"Missing {cfg} in {path.name}")
            store[str(seed)] = results
    return student_a, student_b


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    split_meta = rebuild_split_and_hashes()
    prep_src = load_json(SRC / "leakage_free_preprocessing.json")
    incomplete = load_json(SRC / "cukd_xai_results.json")

    # Exact-match scaler against recorded train-only prep.
    if not np.allclose(split_meta["scaler_mean"], prep_src["scaler_mean"]):
        raise RuntimeError("Recomputed scaler mean does not match leakage_free_preprocessing.json")
    if not np.allclose(split_meta["scaler_scale"], prep_src["scaler_scale"]):
        raise RuntimeError("Recomputed scaler scale does not match leakage_free_preprocessing.json")
    if split_meta["dataset_sha256"] != prep_src["dataset_sha256"]:
        raise RuntimeError("Dataset hash mismatch vs leakage_free_preprocessing.json")
    if split_meta["class_names"] != prep_src["class_names"]:
        raise RuntimeError("Class names mismatch")
    if split_meta["feature_names"] != prep_src["feature_names"]:
        raise RuntimeError("Feature names mismatch")

    student_a, student_b = load_checkpoints()
    preprocessing_protocol = {
        "protocol": prep_src["protocol"],
        "protocol_id": BASE_PROTOCOL_ID,
        "dataset_path": str(WSNDS_PATH.resolve()),
        "dataset_sha256": split_meta["dataset_sha256"],
        "split_random_state": 42,
        "split_shapes": {
            "train": [split_meta["n_train"], split_meta["input_dim"]],
            "validation": [split_meta["n_val"], split_meta["input_dim"]],
            "test": [split_meta["n_test"], split_meta["input_dim"]],
        },
        "feature_names": split_meta["feature_names"],
        "class_names": split_meta["class_names"],
        "scaler_fit_partition": "train",
        "scaler_mean": split_meta["scaler_mean"],
        "scaler_scale": split_meta["scaler_scale"],
        "scaler_sha256": split_meta["scaler_sha256"],
        "split_hashes": split_meta["split_hashes"],
        "archived_global_scaler_diagnostic": prep_src.get("archived_global_scaler_diagnostic"),
        "seeds": SEEDS,
        "environment": prep_src.get("environment", {}),
    }

    results = {
        "run_status": "complete",
        "protocol_id": BASE_PROTOCOL_ID,
        "completion_gate": {
            "student_A_complete": True,
            "student_B_complete": True,
            "n_seeds": len(SEEDS),
            "source": "reconstructed_from_main_10seed_checkpoints",
        },
        "kd_grid_search": {
            "mode": "controlled_single_seed_grid",
            "tuning_seed": 42,
            "note": "Inherited from original train-only 10-seed run; KD T/alpha taken from cukd_xai_results.json",
        },
        "kd_hyperparameters": incomplete.get("kd_hyperparameters", {"T": 4, "alpha": 0.7}),
        "seeds": SEEDS,
        "class_names": split_meta["class_names"],
        "feature_names": split_meta["feature_names"],
        "preprocessing_protocol": preprocessing_protocol,
        "wsn_ds_multi_seed_student_A": student_a,
        "wsn_ds_multi_seed_student_B": student_b,
        "result_scope": (
            "Reconstructed full 10-seed train-only-scaler multi-config package from "
            "main_10seed checkpoints for co-distillation J merge. Summary CSV in the "
            "source folder remains incomplete (n=2); checkpoints are authoritative."
        ),
        "reconstruction_metadata": {
            "source_checkpoint_dir": str(SRC.relative_to(ROOT)).replace("\\", "/"),
            "source_incomplete_json": "cukd_xai_results.json",
            "n_checkpoints": 20,
            "required_configs": REQUIRED_CONFIGS,
        },
    }

    # Copy RF soft caches and write manifests.
    for seed in SEEDS:
        src_npy = SRC / f"rf_soft_seed_{seed}.npy"
        if not src_npy.is_file():
            raise FileNotFoundError(src_npy)
        dst_npy = OUT / src_npy.name
        if not dst_npy.exists() or sha256_file(dst_npy) != sha256_file(src_npy):
            shutil.copy2(src_npy, dst_npy)
        array = np.load(dst_npy, allow_pickle=False)
        if array.shape != (split_meta["n_train"], split_meta["num_classes"]):
            raise RuntimeError(
                f"RF soft shape mismatch seed {seed}: {array.shape} "
                f"vs ({split_meta['n_train']}, {split_meta['num_classes']})"
            )
        if array.dtype != np.float32:
            raise RuntimeError(f"RF soft dtype mismatch seed {seed}: {array.dtype}")
        if not np.isfinite(array).all() or np.any(array < 0.0):
            raise RuntimeError(f"RF soft invalid values seed {seed}")
        if not np.allclose(array.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6):
            raise RuntimeError(f"RF soft rows do not sum to 1 seed {seed}")
        manifest = {
            "protocol_id": BASE_PROTOCOL_ID,
            "seed": int(seed),
            "dataset_sha256": split_meta["dataset_sha256"],
            "train_split_sha256": split_meta["split_hashes"]["train_raw_features_and_labels_sha256"],
            "scaler_sha256": split_meta["scaler_sha256"],
            "class_names": split_meta["class_names"],
            "rf_calibration_config": RF_CALIBRATION_CONFIG,
            "shape": [split_meta["n_train"], split_meta["num_classes"]],
            "dtype": "float32",
            "cache_file": dst_npy.name,
            "probabilities_sha256": sha256_arrays(array),
            "sklearn_version": prep_src.get("environment", {}).get("sklearn"),
            "source_npy": str(src_npy.relative_to(ROOT)).replace("\\", "/"),
        }
        write_json(OUT / f"rf_soft_seed_{seed}.manifest.json", manifest)

    # Also copy leakage_free_preprocessing for provenance.
    shutil.copy2(SRC / "leakage_free_preprocessing.json", OUT / "leakage_free_preprocessing.json")
    write_json(OUT / "cukd_xai_results.json", results)

    # Artifact manifest over all tracked files in OUT.
    entries = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "artifact_manifest.json":
            continue
        rel = path.relative_to(OUT).as_posix()
        entries.append(
            {
                "path": rel,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "protocol_id": BASE_PROTOCOL_ID,
        "status": "complete",
        "source": "prepare_train_only_base_for_j.py",
        "file_count_excluding_manifest": len(entries),
        "files": entries,
    }
    write_json(OUT / "artifact_manifest.json", manifest)

    # Quick validation print.
    print(f"Wrote base package to {OUT}")
    print(f"  seeds={len(SEEDS)} configs_checked={len(REQUIRED_CONFIGS)}")
    print(f"  train_rows={split_meta['n_train']} rf_soft_manifests={len(SEEDS)}")
    print(f"  artifact_files={len(entries)}")
    print(f"  scaler_sha256={split_meta['scaler_sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
