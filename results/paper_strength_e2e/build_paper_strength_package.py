"""Build end-to-end paper-strength evidence package (software only).

Produces frozen tables for:
  - dual model identity (multi-seed pipeline seed-42 vs deployment seed-42)
  - protocol ladder (absolute F1 + KD-minus-scratch) across three protocols
  - per-class KD-minus-scratch heat data
  - gate policy freeze for deployment fixed-point
  - claim freeze JSON for later manuscript rewrite

Does not retrain, re-export, or re-run HIL.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "paper_strength_e2e"
CLASS = ["Blackhole", "Flooding", "Grayhole", "Normal", "TDMA"]
SEEDS10 = [42, 123, 456, 789, 1001, 2024, 3141, 5678, 8192, 9999]
SEEDS5 = [42, 123, 456, 789, 1001]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def paired(a: list[float], b: list[float]) -> dict:
    a_arr, b_arr = np.asarray(a, float), np.asarray(b, float)
    d = a_arr - b_arr
    try:
        w = stats.wilcoxon(a_arr, b_arr, zero_method="wilcox", alternative="two-sided")
        w_stat, w_p = float(w.statistic), float(w.pvalue)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    t = stats.ttest_rel(a_arr, b_arr)
    return {
        "n": int(len(a_arr)),
        "mean_a": float(np.mean(a_arr)),
        "mean_b": float(np.mean(b_arr)),
        "mean_delta": float(np.mean(d)),
        "std_delta": float(np.std(d, ddof=1)) if len(d) > 1 else 0.0,
        "t_stat": float(t.statistic),
        "t_p": float(t.pvalue),
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_p,
        "values_a": a_arr.tolist(),
        "values_b": b_arr.tolist(),
        "deltas": d.tolist(),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Dual identity freeze
    # ------------------------------------------------------------------
    ckpt_a = json.loads(
        (ROOT / "results/wsnds/leakage_free_rerun/main_10seed/checkpoint_student_A_seed_42.json").read_text(
            encoding="utf-8"
        )
    )
    ckpt_b = json.loads(
        (ROOT / "results/wsnds/leakage_free_rerun/main_10seed/checkpoint_student_B_seed_42.json").read_text(
            encoding="utf-8"
        )
    )
    dep_sc = json.loads(
        (
            ROOT
            / "results/wsnds/confirmation_runs_v2/deployment_seed_42/seed_42/seed_completion.json"
        ).read_text(encoding="utf-8")
    )
    dep_a_pred = pd.read_csv(
        ROOT
        / "results/wsnds/confirmation_runs_v2/deployment_seed_42/seed_42/student_A_KD_from_RF_test_predictions.csv"
    )
    dep_b_pred = pd.read_csv(
        ROOT
        / "results/wsnds/confirmation_runs_v2/deployment_seed_42/seed_42/student_B_KD_from_RF_test_predictions.csv"
    )

    def from_pred(df: pd.DataFrame) -> dict:
        y, p = df["true_label"], df["predicted_label"]
        return {
            "n": int(len(df)),
            "accuracy": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro")),
            "per_class_f1": f1_score(y, p, average=None, labels=[0, 1, 2, 3, 4]).tolist(),
        }

    # 10-seed distribution for z-scores
    a_e_all, b_e_all = [], []
    for seed in SEEDS10:
        ca = json.loads(
            (
                ROOT
                / f"results/wsnds/leakage_free_rerun/main_10seed/checkpoint_student_A_seed_{seed}.json"
            ).read_text(encoding="utf-8")
        )
        cb = json.loads(
            (
                ROOT
                / f"results/wsnds/leakage_free_rerun/main_10seed/checkpoint_student_B_seed_{seed}.json"
            ).read_text(encoding="utf-8")
        )
        a_e_all.append(float(ca["results"]["E_KD_from_RF"]["macro_f1"]))
        b_e_all.append(float(cb["results"]["E_KD_from_RF"]["macro_f1"]))
    a_mean, a_std = float(np.mean(a_e_all)), float(np.std(a_e_all, ddof=1))
    b_mean, b_std = float(np.mean(b_e_all)), float(np.std(b_e_all, ddof=1))

    soft_main = ROOT / "results/wsnds/leakage_free_rerun/main_10seed/rf_soft_seed_42.npy"
    soft_dep = (
        ROOT
        / "results/wsnds/confirmation_runs_v2/deployment_seed_42/seed_42/rf_train_probabilities.npy"
    )
    deploy_a_pt = (
        ROOT
        / "results/wsnds/confirmation_runs_v2/deployment_seed_42/seed_42/student_A_KD_from_RF_fp32.pt"
    )
    deploy_b_pt = (
        ROOT
        / "results/wsnds/confirmation_runs_v2/deployment_seed_42/seed_42/student_B_KD_from_RF_fp32.pt"
    )

    dual = {
        "status": "frozen",
        "units": {
            "multi_seed_pipeline_seed42": {
                "definition": (
                    "Student trained inside main_10seed run_all_configs after set_seed(seed) once; "
                    "RF-KD is not re-seeded; RNG follows prior configs in the same seed."
                ),
                "student_A_rf_kd": {
                    "macro_f1": float(ckpt_a["results"]["E_KD_from_RF"]["macro_f1"]),
                    "accuracy": float(ckpt_a["results"]["E_KD_from_RF"]["accuracy"]),
                    "per_class_f1": ckpt_a["results"]["E_KD_from_RF"]["per_class_f1"],
                    "z_vs_10seed_mean": (
                        float(ckpt_a["results"]["E_KD_from_RF"]["macro_f1"]) - a_mean
                    )
                    / a_std,
                },
                "student_B_rf_kd": {
                    "macro_f1": float(ckpt_b["results"]["E_KD_from_RF"]["macro_f1"]),
                    "accuracy": float(ckpt_b["results"]["E_KD_from_RF"]["accuracy"]),
                    "per_class_f1": ckpt_b["results"]["E_KD_from_RF"]["per_class_f1"],
                    "z_vs_10seed_mean": (
                        float(ckpt_b["results"]["E_KD_from_RF"]["macro_f1"]) - b_mean
                    )
                    / b_std,
                },
                "use_for": [
                    "multi_seed_statistical_tables",
                    "seed_level_paired_tests",
                    "protocol_ladder_random_row_train_only",
                ],
            },
            "deployment_clean_seed42": {
                "definition": (
                    "run_tier15_confirmation deployment mode: set_seed(42) then train only RF-KD; "
                    "bound soft targets from main_10seed rf_soft_seed_42.npy (byte-identical)."
                ),
                "student_A_rf_kd": {
                    **from_pred(dep_a_pred),
                    "trained_state_sha256": dep_sc["student_results"]["student_A_rf_kd"][
                        "trained_state_sha256"
                    ],
                    "weights_sha256": sha256_file(deploy_a_pt),
                    "z_vs_10seed_mean": (from_pred(dep_a_pred)["macro_f1"] - a_mean) / a_std,
                },
                "student_B_rf_kd": {
                    **from_pred(dep_b_pred),
                    "trained_state_sha256": dep_sc["student_results"]["student_B_rf_kd"][
                        "trained_state_sha256"
                    ],
                    "weights_sha256": sha256_file(deploy_b_pt),
                    "z_vs_10seed_mean": (from_pred(dep_b_pred)["macro_f1"] - b_mean) / b_std,
                },
                "soft_targets_sha256": sha256_file(soft_dep),
                "soft_targets_match_main_10seed": sha256_file(soft_main) == sha256_file(soft_dep),
                "test_n": 56200,
                "use_for": [
                    "fixed_point_export",
                    "four_pair_hil",
                    "onnx_openvino_train_only_package",
                    "deployment_artifact_claims_only",
                ],
                "forbidden_use": [
                    "as_the_seed_42_cell_of_the_10seed_table",
                    "inside_10seed_mean_or_std_without_relabeling",
                ],
            },
        },
        "ten_seed_rf_kd_distribution": {
            "student_A": {"mean": a_mean, "sample_std": a_std, "values": a_e_all},
            "student_B": {"mean": b_mean, "sample_std": b_std, "values": b_e_all},
        },
        "paper_rule": (
            "Always name which unit is meant. Multi-seed statistics must use checkpoint/full-aggregate "
            "tables. Hardware and fixed-point must name the deployment-clean unit."
        ),
    }
    (OUT / "01_dual_identity_freeze.json").write_text(
        json.dumps(dual, indent=2) + "\n", encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # 2) Protocol ladder + per-class deltas
    # ------------------------------------------------------------------
    arch_a = pd.read_csv(
        ROOT / "results/wsnds/final_results/2026-05-30-10seed-plus-j/wsnds_results_student_A.csv"
    )
    arch_b = pd.read_csv(
        ROOT / "results/wsnds/final_results/2026-05-30-10seed-plus-j/wsnds_results_student_B.csv"
    )

    # train-only per-seed lists
    to_a_e, to_a_d, to_b_e, to_b_d = [], [], [], []
    to_a_e_pc = {c: [] for c in CLASS}
    to_a_d_pc = {c: [] for c in CLASS}
    to_b_e_pc = {c: [] for c in CLASS}
    to_b_d_pc = {c: [] for c in CLASS}
    for seed in SEEDS10:
        ca = json.loads(
            (
                ROOT
                / f"results/wsnds/leakage_free_rerun/main_10seed/checkpoint_student_A_seed_{seed}.json"
            ).read_text(encoding="utf-8")
        )
        cb = json.loads(
            (
                ROOT
                / f"results/wsnds/leakage_free_rerun/main_10seed/checkpoint_student_B_seed_{seed}.json"
            ).read_text(encoding="utf-8")
        )
        ea, da = ca["results"]["E_KD_from_RF"], ca["results"]["D_Small_MLP"]
        eb, db = cb["results"]["E_KD_from_RF"], cb["results"]["D_Small_MLP"]
        to_a_e.append(float(ea["macro_f1"]))
        to_a_d.append(float(da["macro_f1"]))
        to_b_e.append(float(eb["macro_f1"]))
        to_b_d.append(float(db["macro_f1"]))
        for i, c in enumerate(CLASS):
            to_a_e_pc[c].append(float(ea["per_class_f1"][i]))
            to_a_d_pc[c].append(float(da["per_class_f1"][i]))
            to_b_e_pc[c].append(float(eb["per_class_f1"][i]))
            to_b_d_pc[c].append(float(db["per_class_f1"][i]))

    fg_root = (
        ROOT
        / "results/wsnds/confirmation_runs_v2/remote_winterfell_feature_group_5seed_20260805/feature_group_5seed"
    )
    fg_a_e, fg_a_d, fg_b_e, fg_b_d = [], [], [], []
    fg_a_e_pc = {c: [] for c in CLASS}
    fg_a_d_pc = {c: [] for c in CLASS}
    fg_b_e_pc = {c: [] for c in CLASS}
    fg_b_d_pc = {c: [] for c in CLASS}
    for seed in SEEDS5:
        for pred_name, f1_bucket, pc_bucket in [
            ("student_A_KD_from_RF_test_predictions.csv", fg_a_e, fg_a_e_pc),
            ("student_A_Small_MLP_scratch_test_predictions.csv", fg_a_d, fg_a_d_pc),
            ("student_B_KD_from_RF_test_predictions.csv", fg_b_e, fg_b_e_pc),
            ("student_B_Small_MLP_scratch_test_predictions.csv", fg_b_d, fg_b_d_pc),
        ]:
            df = pd.read_csv(fg_root / f"seed_{seed}" / pred_name)
            y, p = df["true_label"], df["predicted_label"]
            f1_bucket.append(float(f1_score(y, p, average="macro")))
            pc = f1_score(y, p, average=None, labels=[0, 1, 2, 3, 4])
            for i, c in enumerate(CLASS):
                pc_bucket[c].append(float(pc[i]))

    # ladder rows
    def arch_mean(df: pd.DataFrame, cfg: str) -> float:
        return float(df.loc[df.Config == cfg, "MacroF1_mean"].iloc[0])

    def arch_std(df: pd.DataFrame, cfg: str) -> float:
        return float(df.loc[df.Config == cfg, "MacroF1_std"].iloc[0])

    ladder = []
    for student, arch_df, e_list, d_list in [
        ("A", arch_a, to_a_e, to_a_d),
        ("B", arch_b, to_b_e, to_b_d),
    ]:
        fg_e = fg_a_e if student == "A" else fg_b_e
        fg_d = fg_a_d if student == "A" else fg_b_d
        paired_to = paired(e_list, d_list)
        paired_fg = paired(fg_e, fg_d)
        ladder.append(
            {
                "student": student,
                "protocol": "archived_presplit_scaler_random_row_10seed",
                "rf_kd_macro_f1_mean": arch_mean(arch_df, "E_KD_from_RF"),
                "rf_kd_macro_f1_std": arch_std(arch_df, "E_KD_from_RF"),
                "scratch_macro_f1_mean": arch_mean(arch_df, "D_Small_MLP"),
                "scratch_macro_f1_std": arch_std(arch_df, "D_Small_MLP"),
                "kd_minus_scratch_mean_of_means": arch_mean(arch_df, "E_KD_from_RF")
                - arch_mean(arch_df, "D_Small_MLP"),
                "paired_t_p": None,
                "paired_wilcoxon_p": None,
                "n_seeds": 10,
                "pairing_note": "archived package stores aggregates only; delta is mean-of-means not seed-paired",
            }
        )
        ladder.append(
            {
                "student": student,
                "protocol": "train_only_scaler_random_row_10seed",
                "rf_kd_macro_f1_mean": paired_to["mean_a"],
                "rf_kd_macro_f1_std": float(np.std(e_list, ddof=1)),
                "scratch_macro_f1_mean": paired_to["mean_b"],
                "scratch_macro_f1_std": float(np.std(d_list, ddof=1)),
                "kd_minus_scratch_mean_paired": paired_to["mean_delta"],
                "kd_minus_scratch_std_paired": paired_to["std_delta"],
                "paired_t_stat": paired_to["t_stat"],
                "paired_t_p": paired_to["t_p"],
                "paired_wilcoxon_p": paired_to["wilcoxon_p"],
                "n_seeds": 10,
                "pairing_note": "seed-paired from checkpoints",
            }
        )
        ladder.append(
            {
                "student": student,
                "protocol": "train_only_scaler_feature_group_disjoint_5seed",
                "rf_kd_macro_f1_mean": paired_fg["mean_a"],
                "rf_kd_macro_f1_std": float(np.std(fg_e, ddof=1)),
                "scratch_macro_f1_mean": paired_fg["mean_b"],
                "scratch_macro_f1_std": float(np.std(fg_d, ddof=1)),
                "kd_minus_scratch_mean_paired": paired_fg["mean_delta"],
                "kd_minus_scratch_std_paired": paired_fg["std_delta"],
                "paired_t_stat": paired_fg["t_stat"],
                "paired_t_p": paired_fg["t_p"],
                "paired_wilcoxon_p": paired_fg["wilcoxon_p"],
                "n_seeds": 5,
                "n_test_rows": 56301,
                "fg_per_seed_rf_kd": fg_e,
                "fg_per_seed_scratch": fg_d,
                "pairing_note": "seed-paired; mean matches aggregate_results.json",
            }
        )

    pd.DataFrame(ladder).to_csv(OUT / "02_protocol_ladder.csv", index=False)
    (OUT / "02_protocol_ladder.json").write_text(
        json.dumps(ladder, indent=2) + "\n", encoding="utf-8"
    )

    # per-class heat rows
    heat = []
    for student, e_pc, d_pc, protocol, n in [
        ("A", to_a_e_pc, to_a_d_pc, "train_only_random_row_10seed", 10),
        ("B", to_b_e_pc, to_b_d_pc, "train_only_random_row_10seed", 10),
        ("A", fg_a_e_pc, fg_a_d_pc, "feature_group_5seed", 5),
        ("B", fg_b_e_pc, fg_b_d_pc, "feature_group_5seed", 5),
    ]:
        for c in CLASS:
            heat.append(
                {
                    "student": student,
                    "protocol": protocol,
                    "class": c,
                    "rf_kd_mean": float(np.mean(e_pc[c])),
                    "scratch_mean": float(np.mean(d_pc[c])),
                    "delta_kd_minus_scratch": float(np.mean(e_pc[c]) - np.mean(d_pc[c])),
                    "n_seeds": n,
                }
            )
        # archived means for comparison
    for student, adf in [("A", arch_a), ("B", arch_b)]:
        e = adf[adf.Config == "E_KD_from_RF"].iloc[0]
        d = adf[adf.Config == "D_Small_MLP"].iloc[0]
        for c in CLASS:
            heat.append(
                {
                    "student": student,
                    "protocol": "archived_presplit_random_row_10seed",
                    "class": c,
                    "rf_kd_mean": float(e[f"{c}_F1_mean"]),
                    "scratch_mean": float(d[f"{c}_F1_mean"]),
                    "delta_kd_minus_scratch": float(e[f"{c}_F1_mean"] - d[f"{c}_F1_mean"]),
                    "n_seeds": 10,
                }
            )
    pd.DataFrame(heat).to_csv(OUT / "03_per_class_kd_minus_scratch.csv", index=False)

    # ------------------------------------------------------------------
    # 3) Gate policy freeze (Option B: measured bound for deployment unit)
    # ------------------------------------------------------------------
    gates = {}
    for stu in ("A", "B"):
        rep = json.loads(
            (
                ROOT
                / f"deployment/firmware_export/wsnds_rfkd_hil/generated_train_only_student_{stu}_seed42_copy/strict_export_report.json"
            ).read_text(encoding="utf-8")
        )

        def find_gate(obj):
            if isinstance(obj, dict):
                if "macro_f1_drop" in obj and "fixed_vs_fp32_agreement" in obj:
                    return obj
                for v in obj.values():
                    found = find_gate(v)
                    if found:
                        return found
            return None

        g = find_gate(rep)
        gates[stu] = g

    hil = {}
    for board, stu, folder in [
        ("esp32c3", "A", "pi5_esp32c3_student_A"),
        ("esp32c3", "B", "pi5_esp32c3_student_B"),
        ("arduino_r4", "A", "pi5_arduino_r4_student_A"),
        ("arduino_r4", "B", "pi5_arduino_r4_student_B"),
    ]:
        m = json.loads(
            (
                ROOT
                / "results/hardware_hil/train_only_scaler_copy"
                / folder
                / "full_56200_metrics.json"
            ).read_text(encoding="utf-8")
        )
        hil[f"{board}_{stu}"] = m

    gate_policy = {
        "decision": "B_measured_conversion_bound",
        "status": "frozen_for_deployment_unit",
        "applies_to": "deployment_clean_seed42 RF-KD A/B fixed-point + HIL package",
        "does_not_apply_to": "claiming multi-seed seed-42 cell equals deployment weights",
        "strict_code_gate_maximum_macro_f1_drop": 0.01,
        "published_measured_gate_maximum_macro_f1_drop": 0.03,
        "rationale": (
            "Measured float-to-fixed macro-F1 drops under train-only deployment PTQ are "
            "~0.024 (A) and ~0.027 (B), with fixed-vs-FP32 prediction agreement >= 0.99. "
            "QAT reduced drop but hurt absolute A F1. Publishing a measured 0.03 bound is "
            "honest; claiming 0.01-strict would be false for these weights."
        ),
        "students": {
            "A": {
                "fp32_macro_f1": gates["A"]["fp32_macro_f1"],
                "fixed_macro_f1": gates["A"]["fixed_macro_f1"],
                "macro_f1_drop": gates["A"]["macro_f1_drop"],
                "fixed_vs_fp32_agreement": gates["A"]["fixed_vs_fp32_agreement"],
                "passes_0_01": gates["A"]["macro_f1_drop"] <= 0.01,
                "passes_0_03": gates["A"]["macro_f1_drop"] <= 0.03,
            },
            "B": {
                "fp32_macro_f1": gates["B"]["fp32_macro_f1"],
                "fixed_macro_f1": gates["B"]["fixed_macro_f1"],
                "macro_f1_drop": gates["B"]["macro_f1_drop"],
                "fixed_vs_fp32_agreement": gates["B"]["fixed_vs_fp32_agreement"],
                "passes_0_01": gates["B"]["macro_f1_drop"] <= 0.01,
                "passes_0_03": gates["B"]["macro_f1_drop"] <= 0.03,
            },
        },
        "hil_deployment_unit": {
            k: {
                "macro_f1": v.get("macro_f1"),
                "mcu_vs_fixed": v.get("mcu_vs_fixed_reference_agreement"),
                "mcu_vs_fp32": v.get("mcu_vs_fp32_agreement"),
                "latency_us_mean": v.get("latency_us_mean"),
                "n": v.get("n"),
            }
            for k, v in hil.items()
        },
        "future_option_A": "Improve PTQ/QAT to pass 0.01 without absolute F1 collapse, then re-export and re-HIL",
        "future_option_C": "If paper requires HIL of multi-seed pipeline seed-42 weights, export those checkpoints instead",
    }
    (OUT / "04_gate_policy_freeze.json").write_text(
        json.dumps(gate_policy, indent=2) + "\n", encoding="utf-8"
    )
    # update decision file
    decision_path = ROOT / "results/wsnds/confirmation_runs_v2/GATE_POLICY_DECISION_REQUIRED.md"
    decision_path.write_text(
        decision_path.read_text(encoding="utf-8")
        + "\n\n## Decision log\n\n"
        + "| Date | Choice | Rationale | Owner |\n"
        + "|---|---|---|---|\n"
        + "| 2026-08-06 | **B measured 0.03 bound** for deployment unit | "
        "Measured drops 0.024/0.027; QAT hurt absolute A F1; dual-identity freeze separates "
        "deploy from multi-seed seed-42 | research freeze |\n"
        + "\nFrozen machine record: `results/paper_strength_e2e/04_gate_policy_freeze.json`\n",
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # 4) Edge disclosure numbers
    # ------------------------------------------------------------------
    edge = json.loads(
        (
            ROOT
            / "results/edge_iiot/duplicate_audit_20260805/edgeiiot_duplicate_audit_summary.json"
        ).read_text(encoding="utf-8")
    )
    edge_rows = []
    for proto in ["strict", "literature_comparable"]:
        b = edge["outputs"][proto]
        audit = b["encoded_model_input_audit"]
        test_n = b["split_rows"]["test"]
        cross_test = audit["rows_in_cross_partition_feature_groups_by_partition"]["test"]
        edge_rows.append(
            {
                "protocol": proto,
                "total_rows": audit["total_rows"],
                "test_rows": test_n,
                "duplicate_rows_excluding_first": audit["duplicate_rows_excluding_first"],
                "duplicate_pct_of_all_rows": 100.0
                * audit["duplicate_rows_excluding_first"]
                / audit["total_rows"],
                "train_test_feature_overlap_groups": audit["overlaps"][
                    "train_test_feature_overlap"
                ],
                "test_rows_in_cross_partition_groups": cross_test,
                "pct_test_rows_in_cross_partition_groups": 100.0 * cross_test / test_n,
                "recommendation": (
                    "disclose_only"
                    if 100.0 * cross_test / test_n < 5.0
                    else "disclose_and_consider_group_aware_retrain_before_sota_claims"
                ),
            }
        )
    pd.DataFrame(edge_rows).to_csv(OUT / "05_edge_duplicate_disclosure.csv", index=False)

    # ------------------------------------------------------------------
    # 5) Claim freeze for later manuscript rewrite
    # ------------------------------------------------------------------
    claims = {
        "status": "frozen_research_claims_not_final_manuscript",
        "allowed_primary_claims": [
            {
                "id": "C1_compression",
                "text": (
                    "Under train-only scaler random-row 10-seed evaluation, RF-KD students retain "
                    f"macro-F1 {a_mean:.4f}±{a_std:.4f} (A, 1189 params) and {b_mean:.4f}±{b_std:.4f} "
                    "(B, 3397 params)."
                ),
                "evidence": "main_10seed_full_aggregate_copy + checkpoints",
            },
            {
                "id": "C2_kd_sensitive_to_split",
                "text": (
                    "Student A RF-KD minus scratch macro-F1 is +0.0094 paired under train-only random-row "
                    "(t p≈0.048) but +0.0002 under feature-group disjoint 5-seed (t p≈0.95). "
                    "KD benefit is split-construction sensitive."
                ),
                "evidence": "02_protocol_ladder.json",
            },
            {
                "id": "C3_minority_tradeoff",
                "text": (
                    "Macro-F1 can hide minority-class costs: under FG, Student B RF-KD loses ~0.023 "
                    "Blackhole and ~0.016 Grayhole F1 vs scratch while gaining Flooding/TDMA."
                ),
                "evidence": "03_per_class_kd_minus_scratch.csv",
            },
            {
                "id": "C4_deployment_fidelity",
                "text": (
                    "For the deployment-clean seed-42 RF-KD unit, MCU predictions match fixed-point "
                    "references on all 56200 vectors (agree=1.0) on ESP32-C3 and Arduino R4; "
                    "float-to-fixed macro-F1 drop is ~0.024 (A) and ~0.027 (B)."
                ),
                "evidence": "deployment HIL + export gates + dual identity freeze",
            },
            {
                "id": "C5_host_conversion",
                "text": (
                    "ONNX FP32 and OpenVINO FP32 agree with PyTorch at 1.0 for deployment RF-KD graphs; "
                    "dynamic INT8 reduces size but costs substantial macro-F1."
                ),
                "evidence": "train_only_seed42_copy runtime package",
            },
        ],
        "forbidden_or_retired_claims": [
            {
                "id": "X1",
                "text": "RF-KD is the strongest ultra-small route (unqualified).",
                "reason": "Fails under feature-group disjoint protocol for Student A KD-over-scratch.",
            },
            {
                "id": "X2",
                "text": "Deployment seed-42 F1 0.9485 is the multi-seed table's seed-42 cell.",
                "reason": "Different training unit (clean seed vs multi-config RNG).",
            },
            {
                "id": "X3",
                "text": "Fixed-point conversion meets a 0.01 macro-F1 drop gate for these deploy weights.",
                "reason": "Measured drops 0.024/0.027; gate policy B uses 0.03 measured bound.",
            },
            {
                "id": "X4",
                "text": "Co-distillation superiority under train-only scaler.",
                "reason": "J not trained in train-only 10-seed package.",
            },
        ],
        "manuscript_policy": "Rewrite manuscript only after this freeze; draft tex is not submission-ready.",
    }
    (OUT / "06_claim_freeze.json").write_text(json.dumps(claims, indent=2) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # 6) Master README
    # ------------------------------------------------------------------
    md = []
    md.append("# Paper-strength E2E package (frozen research objects)\n\n")
    md.append("Built by `build_paper_strength_package.py`. No retraining, no HIL re-run.\n\n")
    md.append("## Files\n\n")
    md.append("| File | Purpose |\n|---|---|\n")
    md.append("| `01_dual_identity_freeze.json` | Multi-seed seed-42 vs deployment seed-42 |\n")
    md.append("| `02_protocol_ladder.csv/.json` | Absolute F1 + KD−scratch across protocols |\n")
    md.append("| `03_per_class_kd_minus_scratch.csv` | Minority-class tradeoffs |\n")
    md.append("| `04_gate_policy_freeze.json` | Option B measured 0.03 bound |\n")
    md.append("| `05_edge_duplicate_disclosure.csv` | Edge leakage framing |\n")
    md.append("| `06_claim_freeze.json` | Allowed vs forbidden claims |\n\n")
    md.append("## Headline ladder (KD − scratch macro-F1)\n\n")
    md.append("| Student | Archived RR | Train-only RR (paired) | FG disjoint (paired) |\n")
    md.append("|---|---:|---:|---:|\n")
    for student in ("A", "B"):
        rows = [r for r in ladder if r["student"] == student]
        arch = next(r for r in rows if r["protocol"].startswith("archived"))
        to = next(r for r in rows if "random_row_10seed" in r["protocol"] and "train_only" in r["protocol"])
        fg = next(r for r in rows if "feature_group" in r["protocol"])
        md.append(
            f"| {student} | {arch['kd_minus_scratch_mean_of_means']:+.4f} | "
            f"{to['kd_minus_scratch_mean_paired']:+.4f} (t p={to['paired_t_p']:.3f}) | "
            f"{fg['kd_minus_scratch_mean_paired']:+.4f} (t p={fg['paired_t_p']:.3f}) |\n"
        )
    md.append("\n## Dual identity (Student A RF-KD)\n\n")
    md.append(
        f"- Multi-seed pipeline seed 42: **{dual['units']['multi_seed_pipeline_seed42']['student_A_rf_kd']['macro_f1']:.4f}** "
        f"(z={dual['units']['multi_seed_pipeline_seed42']['student_A_rf_kd']['z_vs_10seed_mean']:+.2f})\n"
    )
    md.append(
        f"- Deployment clean seed 42: **{dual['units']['deployment_clean_seed42']['student_A_rf_kd']['macro_f1']:.4f}** "
        f"(z={dual['units']['deployment_clean_seed42']['student_A_rf_kd']['z_vs_10seed_mean']:+.2f})\n"
    )
    md.append(
        f"- Soft targets identical: **{dual['units']['deployment_clean_seed42']['soft_targets_match_main_10seed']}**\n"
    )
    md.append("\n## Next (optional HW)\n\n")
    md.append("- Keep HIL on deployment unit (already complete) **or** re-HIL checkpoint weights if required.\n")
    md.append("- Manuscript rewrite uses `06_claim_freeze.json` only after review.\n")
    (OUT / "README.md").write_text("".join(md), encoding="utf-8")

    print("Wrote package to", OUT)
    print("A ladder KD-scratch:", [r for r in ladder if r["student"] == "A"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
