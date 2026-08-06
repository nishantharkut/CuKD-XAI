"""Rebuild full 10-seed train-only tables from complete checkpoint set.

Why this exists:
  main_10seed/wsnds_results_student_{A,B}.csv currently report n_seeds=2
  (only seeds 8192/9999 were merged into the summary files). All 20 checkpoints
  (10 seeds x 2 students) already contain full per-config metrics.

This script does not retrain. It aggregates existing checkpoints with sample
std (ddof=1) and writes copy-only outputs for research use. Manuscript rebuild
is intentionally deferred until research completion is signed off.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
CKPT_DIR = ROOT / "results/wsnds/leakage_free_rerun/main_10seed"
OUT = ROOT / "results/wsnds/leakage_free_rerun/main_10seed_full_aggregate_copy"
CLASS_NAMES = ["Blackhole", "Flooding", "Grayhole", "Normal", "TDMA"]
SEEDS = [42, 123, 456, 789, 1001, 2024, 3141, 5678, 8192, 9999]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    by_student: dict[str, dict[str, dict[str, list]]] = {
        "A": defaultdict(lambda: defaultdict(list)),
        "B": defaultdict(lambda: defaultdict(list)),
    }
    meta: dict[str, dict[str, dict]] = {"A": {}, "B": {}}
    anomalies = []

    for student in ("A", "B"):
        for seed in SEEDS:
            path = CKPT_DIR / f"checkpoint_student_{student}_seed_{seed}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("student") not in (student, f"student_{student}"):
                # tolerate student_A naming
                pass
            results = payload["results"]
            for cfg, metrics in results.items():
                if not isinstance(metrics, dict) or "macro_f1" not in metrics:
                    continue
                mf1 = float(metrics["macro_f1"])
                if mf1 < 0.5:
                    anomalies.append(
                        {
                            "student": student,
                            "seed": seed,
                            "config": cfg,
                            "macro_f1": mf1,
                            "note": "collapsed_or_failed_run_suspected",
                        }
                    )
                by_student[student][cfg]["macro_f1"].append(mf1)
                by_student[student][cfg]["accuracy"].append(float(metrics["accuracy"]))
                by_student[student][cfg]["seeds"].append(seed)
                if isinstance(metrics.get("per_class_f1"), list) and len(metrics["per_class_f1"]) == 5:
                    for name, value in zip(CLASS_NAMES, metrics["per_class_f1"]):
                        by_student[student][cfg][f"{name}_f1"].append(float(value))
                meta[student][cfg] = {
                    "params": metrics.get("params"),
                    "size_kb": metrics.get("model_size_kb"),
                }

    summary_rows = []
    for student in ("A", "B"):
        rows = []
        for cfg, mets in sorted(by_student[student].items()):
            n = len(mets["macro_f1"])
            if n != 10:
                raise RuntimeError(f"{student}/{cfg} has n={n}, expected 10")
            row = {
                "Config": cfg,
                "n_seeds": n,
                "Accuracy_mean": float(np.mean(mets["accuracy"])),
                "Accuracy_std": float(np.std(mets["accuracy"], ddof=1)),
                "MacroF1_mean": float(np.mean(mets["macro_f1"])),
                "MacroF1_std": float(np.std(mets["macro_f1"], ddof=1)),
            }
            for name in CLASS_NAMES:
                key = f"{name}_f1"
                if key in mets:
                    row[f"{name}_F1_mean"] = float(np.mean(mets[key]))
                    row[f"{name}_F1_std"] = float(np.std(mets[key], ddof=1))
            row["params"] = meta[student].get(cfg, {}).get("params")
            row["size_kb"] = meta[student].get(cfg, {}).get("size_kb")
            rows.append(row)
            summary_rows.append({"student": student, **row})
        df = pd.DataFrame(rows)
        out_csv = OUT / f"wsnds_results_student_{student}_10seed.csv"
        df.to_csv(out_csv, index=False)
        print("Wrote", out_csv)

    # Load recovered holm tests if present
    recovered = ROOT / "results/wsnds/leakage_free_rerun/recovered_main_10seed_v1/recovered_results.json"
    holm = None
    if recovered.is_file():
        rec = json.loads(recovered.read_text(encoding="utf-8"))
        holm = {
            "student_A_paired_tests_holm": rec.get("student_A_paired_tests_holm"),
            "student_B_paired_tests_holm": rec.get("student_B_paired_tests_holm"),
            "statistical_procedure": rec.get("statistical_procedure"),
            "source": str(recovered.relative_to(ROOT)).replace("\\", "/"),
        }

    report = {
        "protocol": "wsnds_train_only_scaler_random_row_10seed_full_aggregate_v1",
        "status": "complete_from_checkpoints",
        "seeds": SEEDS,
        "checkpoint_dir": str(CKPT_DIR.relative_to(ROOT)).replace("\\", "/"),
        "n_checkpoints_expected": 20,
        "n_checkpoints_found": len(list(CKPT_DIR.glob("checkpoint_student_*_seed_*.json"))),
        "configs_per_student": 14,
        "co_distillation_J_present": False,
        "co_distillation_J_note": (
            "Train-only 10-seed rerun does not include J_CoDistill_RF_CL. "
            "Archived co-distillation claims remain under pre-split-scaler lineage unless J is retrained."
        ),
        "std_definition": "sample standard deviation (ddof=1)",
        "anomalies": anomalies,
        "anomaly_interpretation": (
            "Any macro_f1 < 0.5 is flagged. Curriculum-ext routes include a known collapse "
            "on seed 5678; do not hide it. For primary RF-KD claims, verify RF-KD seeds are healthy."
        ),
        "headline_rf_kd": {
            row["student"]: {
                "MacroF1_mean": row["MacroF1_mean"],
                "MacroF1_std": row["MacroF1_std"],
                "Accuracy_mean": row["Accuracy_mean"],
                "n_seeds": row["n_seeds"],
            }
            for row in summary_rows
            if row["Config"] == "E_KD_from_RF"
        },
        "headline_scratch": {
            row["student"]: {
                "MacroF1_mean": row["MacroF1_mean"],
                "MacroF1_std": row["MacroF1_std"],
                "n_seeds": row["n_seeds"],
            }
            for row in summary_rows
            if row["Config"] == "D_Small_MLP"
        },
        "holm_tests": holm,
        "comparison_to_incomplete_summary_csvs": {
            "main_10seed_csv_n_seeds": 2,
            "main_10seed_csv_note": (
                "results/wsnds/leakage_free_rerun/main_10seed/wsnds_results_student_*.csv "
                "are incomplete merges (n=2). Do not use them as primary train-only evidence."
            ),
            "full_aggregate_n_seeds": 10,
        },
        "manuscript_policy": (
            "Manuscript finalization is DEFERRED until research completion sign-off. "
            "Do not treat partial manuscript edits as submission-ready."
        ),
    }
    (OUT / "full_10seed_aggregate_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(summary_rows).to_csv(OUT / "wsnds_results_both_students_10seed_long.csv", index=False)
    print(json.dumps({k: report[k] for k in ["status", "headline_rf_kd", "co_distillation_J_present", "anomalies"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
