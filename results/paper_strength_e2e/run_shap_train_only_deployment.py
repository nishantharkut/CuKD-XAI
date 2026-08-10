"""SHAP audit for train-only deployment RF-KD students A/B (e2e, software).

Mirrors the manuscript audit protocol:
  - DeepExplainer on student MLP
  - TreeExplainer on RF teacher (same train partition, seed 42)
  - Global + per-class Spearman rank agreement
  - 5-subsample bootstrap of global Spearman

Subjects are the **deployment-clean seed-42** weights (not multi-seed pipeline
seed-42). Soft targets used in KD are bound from main_10seed; the RF refit here
is for TreeExplainer structure on the same train-only scaled data and RF config.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.wsnds.leakage_free_rerun.tier15_common import (  # noqa: E402
    CLASS_NAMES,
    STUDENT_SPECS,
    StudentMLP,
    apply_train_scaler,
    archived_random_split,
    load_wsnds,
    set_seed,
)

OUT = ROOT / "results/paper_strength_e2e/shap_train_only_deployment"
DEPLOY = ROOT / "results/wsnds/confirmation_runs_v2/deployment_seed_42/seed_42"
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


def load_student(path: Path, hidden: tuple[int, int]) -> StudentMLP:
    model = StudentMLP(17, hidden, 5)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(str(k).startswith("net.") for k in state):
        model.load_state_dict(state)
    else:
        model.net.load_state_dict(state)
    model.eval()
    return model


def to_list_per_class(shap_values, n_classes: int = 5):
    if isinstance(shap_values, list):
        return [np.asarray(v) for v in shap_values]
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        # (n, features, classes) or (n, classes, features)
        if arr.shape[-1] == n_classes:
            return [arr[:, :, i] for i in range(n_classes)]
        if arr.shape[1] == n_classes:
            return [arr[:, i, :] for i in range(n_classes)]
    raise RuntimeError(f"Unexpected SHAP value shape: {arr.shape}")


def global_abs_mean(shap_list: list[np.ndarray]) -> np.ndarray:
    return np.abs(np.stack(shap_list, axis=0)).mean(axis=(0, 1))


def rank_spearman(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    ra = pd.Series(a).rank(ascending=False)
    rb = pd.Series(b).rank(ascending=False)
    rho, p = spearmanr(ra, rb)
    return float(rho), float(p)


def explain_student(
    model: StudentMLP,
    X_train: np.ndarray,
    X_test: np.ndarray,
    bg_idx: np.ndarray,
    explain_idx: np.ndarray,
    device: torch.device,
) -> list[np.ndarray]:
    model = model.to(device)
    model.eval()
    bg = torch.tensor(X_train[bg_idx], dtype=torch.float32, device=device)
    tx = torch.tensor(X_test[explain_idx], dtype=torch.float32, device=device)
    explainer = shap.DeepExplainer(model, bg)
    values = explainer.shap_values(tx)
    return to_list_per_class(values, n_classes=5)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    set_seed(42)
    device = torch.device("cpu")  # SHAP DeepExplainer more reliable on CPU here

    dataset = load_wsnds(ROOT / "data/wsnds/WSN-DS.csv")
    split = archived_random_split(dataset["features"], dataset["labels"])
    scaled, _ = apply_train_scaler(split)
    X_train = np.asarray(scaled["X_train"], dtype=np.float32)
    X_test = np.asarray(scaled["X_test"], dtype=np.float32)
    y_train = np.asarray(split["y_train"], dtype=np.int64)

    print("Fitting RF teacher for TreeExplainer (seed=42, train-only scaled data)...")
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=15, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    students = {
        "A": {
            "hidden": STUDENT_SPECS["student_A"],
            "path": DEPLOY / "student_A_KD_from_RF_fp32.pt",
        },
        "B": {
            "hidden": STUDENT_SPECS["student_B"],
            "path": DEPLOY / "student_B_KD_from_RF_fp32.pt",
        },
    }

    master = {
        "protocol": "train_only_deployment_seed42_shap_v1",
        "subject": "deployment_clean_seed42_rf_kd",
        "not_subject": "multi_seed_pipeline_seed42_checkpoint",
        "feature_names": FEATURE_NAMES,
        "class_names": CLASS_NAMES,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "background_size": 100,
        "explain_size": 500,
        "bootstrap_repeats": 5,
        "rf_config": {"n_estimators": 500, "max_depth": 15, "random_state": 42},
        "students": {},
    }

    for stu, meta in students.items():
        print(f"\n===== Student {stu} =====")
        model = load_student(meta["path"], meta["hidden"])
        rng = np.random.RandomState(42)
        bg_idx = rng.choice(len(X_train), 100, replace=False)
        explain_idx = rng.choice(len(X_test), 500, replace=False)

        print("DeepExplainer student...")
        student_list = explain_student(model, X_train, X_test, bg_idx, explain_idx, device)
        student_global = global_abs_mean(student_list)

        print("TreeExplainer RF teacher...")
        rf_explainer = shap.TreeExplainer(rf)
        rf_values = rf_explainer.shap_values(X_test[explain_idx])
        rf_list = to_list_per_class(rf_values, n_classes=5)
        teacher_global = global_abs_mean(rf_list)

        rho, rho_p = rank_spearman(student_global, teacher_global)
        print(f"Global Spearman rho={rho:.4f} p={rho_p:.4e}")

        per_class = {}
        for i, cname in enumerate(CLASS_NAMES):
            s_imp = np.abs(student_list[i]).mean(axis=0)
            t_imp = np.abs(rf_list[i]).mean(axis=0)
            cr, cp = rank_spearman(s_imp, t_imp)
            per_class[cname] = {"rho": cr, "p": cp}
            print(f"  {cname:10s} rho={cr:+.4f} p={cp:.4e}")

        # bootstrap stability (5)
        print("Bootstrap (5)...")
        boot_rhos, boot_ps = [], []
        for bi in range(5):
            br = np.random.RandomState(42 + bi * 37)
            b_bg = br.choice(len(X_train), 100, replace=False)
            b_ex = br.choice(len(X_test), 500, replace=False)
            try:
                s_list = explain_student(model, X_train, X_test, b_bg, b_ex, device)
                r_vals = shap.TreeExplainer(rf).shap_values(X_test[b_ex])
                r_list = to_list_per_class(r_vals, n_classes=5)
                brho, bp = rank_spearman(global_abs_mean(s_list), global_abs_mean(r_list))
                boot_rhos.append(brho)
                boot_ps.append(bp)
                print(f"  boot {bi+1}: rho={brho:.4f}")
            except Exception as exc:  # noqa: BLE001
                print(f"  boot {bi+1} failed: {exc}")

        rank_df = pd.DataFrame(
            {
                "feature": FEATURE_NAMES,
                "student_global_abs_shap": student_global,
                "teacher_global_abs_shap": teacher_global,
                "student_rank": pd.Series(student_global).rank(ascending=False).to_numpy(),
                "teacher_rank": pd.Series(teacher_global).rank(ascending=False).to_numpy(),
            }
        ).sort_values("student_global_abs_shap", ascending=False)
        rank_path = OUT / f"student_{stu}_feature_ranks.csv"
        rank_df.to_csv(rank_path, index=False)

        # summary plot
        try:
            shap.summary_plot(
                student_list,
                X_test[explain_idx],
                feature_names=FEATURE_NAMES,
                class_names=CLASS_NAMES,
                show=False,
                plot_size=(10, 6),
            )
            plt.tight_layout()
            plt.savefig(OUT / f"student_{stu}_shap_summary.png", dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as exc:  # noqa: BLE001
            print("plot failed", exc)

        master["students"][stu] = {
            "weights_path": str(meta["path"].relative_to(ROOT)).replace("\\", "/"),
            "weights_sha256": __import__("hashlib")
            .sha256(meta["path"].read_bytes())
            .hexdigest(),
            "ranking_agreement_spearman": rho,
            "ranking_agreement_p": rho_p,
            "per_class_spearman": per_class,
            "bootstrap_spearman_values": boot_rhos,
            "bootstrap_spearman_ps": boot_ps,
            "bootstrap_spearman_mean": float(np.mean(boot_rhos)) if boot_rhos else None,
            "bootstrap_spearman_std": float(np.std(boot_rhos, ddof=1))
            if len(boot_rhos) > 1
            else None,
            "student_global_importance": {
                f: float(v) for f, v in zip(FEATURE_NAMES, student_global)
            },
            "teacher_global_importance": {
                f: float(v) for f, v in zip(FEATURE_NAMES, teacher_global)
            },
            "student_top5": rank_df.head(5)["feature"].tolist(),
            "teacher_top5": rank_df.sort_values(
                "teacher_global_abs_shap", ascending=False
            )
            .head(5)["feature"]
            .tolist(),
            "rank_table": str(rank_path.relative_to(ROOT)).replace("\\", "/"),
            "interpretation": (
                "low_alignment_student_diverges_from_rf_global_ranks"
                if rho < 0.7
                else "high_alignment"
            ),
        }

    (OUT / "shap_results.json").write_text(
        json.dumps(master, indent=2) + "\n", encoding="utf-8"
    )
    print("\nWrote", OUT / "shap_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
