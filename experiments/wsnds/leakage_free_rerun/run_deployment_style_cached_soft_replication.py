"""Replicate deployment-style RF-KD using *cached* soft targets (not re-calibrated).

This answers whether seed-42 deployment F1≈0.9485 is:
  (a) a one-off unreplicated high point, or
  (b) a systematic property of: set_seed(seed) → RF-KD only + main_10seed rf_soft_seed_{seed}.npy

Protocol (matches run_tier15_confirmation deployment route):
  - train-only StandardScaler after seed-42 raw split (same as main_10seed)
  - load bound soft targets from main_10seed/rf_soft_seed_{seed}.npy
  - set_seed(seed)
  - construct StudentMLP and train_rf_kd only (no prior configs)
  - evaluate on the fixed test partition

Does NOT re-HIL. Writes copy-only results under leftover_e2e_closure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from tier15_common import (  # noqa: E402
    CLASS_NAMES,
    STUDENT_SPECS,
    StudentMLP,
    apply_train_scaler,
    archived_random_split,
    batched_probs,
    class_weights,
    classification_metrics,
    load_wsnds,
    set_seed,
    sha256_arrays,
    sha256_file,
    train_rf_kd,
)

MAIN10 = REPO_ROOT / "results/wsnds/leakage_free_rerun/main_10seed"
DEPLOY42 = REPO_ROOT / "results/wsnds/confirmation_runs_v2/deployment_seed_42"
OUT = REPO_ROOT / "results/leftover_e2e_closure/06_deployment_style_cached_soft_replication"
SEEDS = [42, 123, 456, 789, 1001, 2024, 3141, 5678, 8192, 9999]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    loaded = load_wsnds(REPO_ROOT / "data/wsnds/WSN-DS.csv")
    X_all = loaded["features"]
    y_all = loaded["labels"]
    assert list(loaded["class_names"]) == CLASS_NAMES
    split = archived_random_split(X_all, y_all)
    scaled, _scaler = apply_train_scaler(split)
    X_train = scaled["X_train"].astype(np.float32)
    y_train = split["y_train"].astype(np.int64)
    X_val = scaled["X_validation"].astype(np.float32)
    y_val = split["y_validation"].astype(np.int64)
    X_test = scaled["X_test"].astype(np.float32)
    y_test = split["y_test"].astype(np.int64)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    weights = class_weights(y_train)

    # Optional: confirm seed-42 deployment soft cache identity
    dep_soft = DEPLOY42 / "seed_42" / "rf_train_probabilities.npy"
    main_soft_42 = MAIN10 / "rf_soft_seed_42.npy"
    soft_identity = None
    if dep_soft.is_file() and main_soft_42.is_file():
        a = np.load(dep_soft)
        b = np.load(main_soft_42)
        soft_identity = {
            "deployment_soft_sha256": sha256_arrays(a),
            "main10_soft_sha256": sha256_arrays(b),
            "byte_identical": bool(a.shape == b.shape and np.array_equal(a, b)),
        }
        print("soft_identity_42", soft_identity)

    rows = []
    by_seed: dict[str, dict] = {}
    for seed in SEEDS:
        cache = MAIN10 / f"rf_soft_seed_{seed}.npy"
        if not cache.is_file():
            raise FileNotFoundError(cache)
        rf_soft = np.load(cache).astype(np.float32)
        if rf_soft.shape != (len(X_train), 5):
            raise RuntimeError(f"soft shape {rf_soft.shape} for seed {seed}")
        soft_sha = sha256_arrays(rf_soft)
        print(f"\n=== seed {seed} soft_sha={soft_sha[:16]}... ===")
        seed_payload = {
            "seed": seed,
            "soft_cache": str(cache.relative_to(REPO_ROOT)).replace("\\", "/"),
            "soft_sha256": soft_sha,
            "students": {},
        }
        for student_name, hidden in STUDENT_SPECS.items():
            set_seed(seed)
            student = StudentMLP(17, hidden, 5)
            student = train_rf_kd(
                student,
                rf_soft,
                X_train_t,
                y_train_t,
                X_val_t,
                y_val_t,
                weights,
                device,
            )
            probs = batched_probs(student, X_test_t, device)
            metrics = classification_metrics(y_test, probs)
            short = "A" if student_name.endswith("A") else "B"
            seed_payload["students"][short] = {
                "hidden": list(hidden),
                "macro_f1": float(metrics["macro_f1"]),
                "accuracy": float(metrics["accuracy"]),
                "macro_precision": float(metrics["macro_precision"]),
                "macro_recall": float(metrics["macro_recall"]),
                "per_class_f1": metrics["per_class_f1"],
            }
            rows.append(
                {
                    "seed": seed,
                    "student": short,
                    "macro_f1": float(metrics["macro_f1"]),
                    "accuracy": float(metrics["accuracy"]),
                    "soft_sha256": soft_sha,
                    "protocol": "deployment_style_cached_soft_v1",
                }
            )
            print(
                f"  student {short}: macro_f1={metrics['macro_f1']:.6f} "
                f"acc={metrics['accuracy']:.6f}"
            )
            del student
            if device.type == "cuda":
                torch.cuda.empty_cache()
        by_seed[str(seed)] = seed_payload
        write_json(OUT / f"seed_{seed}_metrics.json", seed_payload)

    # Comparisons vs known units for seed 42
    # Pipeline L1 from checkpoint
    pipe_a = json.loads(
        (MAIN10 / "checkpoint_student_A_seed_42.json").read_text(encoding="utf-8")
    )["results"]["E_KD_from_RF"]["macro_f1"]
    pipe_b = json.loads(
        (MAIN10 / "checkpoint_student_B_seed_42.json").read_text(encoding="utf-8")
    )["results"]["E_KD_from_RF"]["macro_f1"]
    dep_agg = json.loads(
        (DEPLOY42 / "aggregate_results.json").read_text(encoding="utf-8")
    )
    dep_a = float(dep_agg["aggregate"]["student_A_rf_kd"]["macro_f1"]["mean"])
    dep_b = float(dep_agg["aggregate"]["student_B_rf_kd"]["macro_f1"]["mean"])

    # Per-route recalibrated (from leftover reseed)
    reseed_path = (
        REPO_ROOT
        / "results/leftover_e2e_closure/03_per_route_set_seed/seed_42_checkpoint.json"
    )
    reseed_a = reseed_b = None
    if reseed_path.is_file():
        rs = json.loads(reseed_path.read_text(encoding="utf-8"))
        reseed_a = float(rs["A"]["E"]["macro_f1"])
        reseed_b = float(rs["B"]["E"]["macro_f1"])

    rep_a = by_seed["42"]["students"]["A"]["macro_f1"]
    rep_b = by_seed["42"]["students"]["B"]["macro_f1"]

    f1s_a = [by_seed[str(s)]["students"]["A"]["macro_f1"] for s in SEEDS]
    f1s_b = [by_seed[str(s)]["students"]["B"]["macro_f1"] for s in SEEDS]

    report = {
        "status": "complete",
        "protocol": "deployment_style_cached_soft_v1",
        "definition": (
            "set_seed(seed); StudentMLP; train_rf_kd with main_10seed/rf_soft_seed_{seed}.npy; "
            "no multi-config prior training; no RF re-calibration."
        ),
        "soft_identity_seed42": soft_identity,
        "seeds": SEEDS,
        "seed42_four_way_comparison": {
            "student_A": {
                "pipeline_multiconfig": float(pipe_a),
                "per_route_recalibrated_soft": reseed_a,
                "deployment_recorded": dep_a,
                "this_replication_cached_soft": rep_a,
                "replication_minus_deployment": rep_a - dep_a,
                "replication_minus_pipeline": rep_a - float(pipe_a),
            },
            "student_B": {
                "pipeline_multiconfig": float(pipe_b),
                "per_route_recalibrated_soft": reseed_b,
                "deployment_recorded": dep_b,
                "this_replication_cached_soft": rep_b,
                "replication_minus_deployment": rep_b - dep_b,
                "replication_minus_pipeline": rep_b - float(pipe_b),
            },
        },
        "deployment_style_cached_soft_aggregate": {
            "A": {
                "macro_f1_mean": float(np.mean(f1s_a)),
                "macro_f1_std_ddof1": float(np.std(f1s_a, ddof=1)),
                "macro_f1_min": float(np.min(f1s_a)),
                "macro_f1_max": float(np.max(f1s_a)),
                "values_by_seed": {str(s): by_seed[str(s)]["students"]["A"]["macro_f1"] for s in SEEDS},
                "n": len(SEEDS),
            },
            "B": {
                "macro_f1_mean": float(np.mean(f1s_b)),
                "macro_f1_std_ddof1": float(np.std(f1s_b, ddof=1)),
                "macro_f1_min": float(np.min(f1s_b)),
                "macro_f1_max": float(np.max(f1s_b)),
                "values_by_seed": {str(s): by_seed[str(s)]["students"]["B"]["macro_f1"] for s in SEEDS},
                "n": len(SEEDS),
            },
        },
        "interpretation": {
            "seed42_replication_ok": abs(rep_a - dep_a) < 0.002,
            "note_if_not_exact": (
                "Small residual differences can come from GPU nondeterminism / library "
                "versions; large gaps mean the recorded deployment unit is not reproduced."
            ),
            "key_question": (
                "Does deployment-style + cached soft produce systematically high F1 "
                "across seeds, or is 0.9485 a seed-42 outlier under that protocol?"
            ),
        },
        "by_seed": by_seed,
    }
    write_json(OUT / "deployment_style_cached_soft_report.json", report)

    import pandas as pd

    pd.DataFrame(rows).to_csv(OUT / "deployment_style_cached_soft_long.csv", index=False)
    print("\n=== REPORT ===")
    print(json.dumps(report["seed42_four_way_comparison"], indent=2))
    print("A aggregate", report["deployment_style_cached_soft_aggregate"]["A"])
    print("B aggregate", report["deployment_style_cached_soft_aggregate"]["B"])
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
