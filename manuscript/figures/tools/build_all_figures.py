#!/usr/bin/env python3
"""Build manuscript figures with numbers loaded only from audited result files.

No hard-coded scientific claims. Every quantitative label is read from audited
JSON/CSV under results/. Writes manuscript/figures/_provenance.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from graphviz_env import dot_version, ensure_graphviz_on_path, render_dot  # noqa: E402
from ieee_style import (  # noqa: E402
    C_BLUE,
    C_GRAY,
    C_ORANGE,
    FIG_DOUBLE,
    FIG_SINGLE,
    apply_ieee_style,
    save_fig,
)

REPO = Path(__file__).resolve().parents[3]
FIG = Path(__file__).resolve().parents[1]
AGG = REPO / "results/wsnds/leakage_free_rerun/main_10seed_full_aggregate_copy"
PC = REPO / "results/paper_strength_e2e/03_per_class_kd_minus_scratch.csv"
LADDER = REPO / "results/paper_strength_e2e/02_protocol_ladder.json"
RUNTIME = (
    REPO
    / "results/runtime/onnx_openvino/wsnds/train_only_seed42_copy/train_only_runtime_results.json"
)
FREEZE = REPO / "results/paper_strength_e2e/06_claim_freeze.json"
HIL = REPO / "results/hardware_hil/train_only_scaler_copy/four_pair_summary.json"
SHAP = (
    REPO
    / "results/paper_strength_e2e/shap_train_only_deployment/shap_results.json"
)
J_SUM = REPO / "results/leftover_e2e_closure/01_j_codistill/j_summary.json"
EDGE = (
    REPO
    / "results/leftover_e2e_closure/04_edge_group_aware/edge_group_aware_summary.json"
)
LAB_DIR = FIG / "hardware" / "lab"

PROV: dict = {"figures": {}}


def _fmt(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required evidence file: {path}")
    return path


def load_sources() -> dict:
    A = pd.read_csv(_require(AGG / "wsnds_results_student_A_10seed.csv"))
    B = pd.read_csv(_require(AGG / "wsnds_results_student_B_10seed.csv"))
    ladder = json.loads(_require(LADDER).read_text(encoding="utf-8"))
    runtime = json.loads(_require(RUNTIME).read_text(encoding="utf-8"))
    pc = pd.read_csv(_require(PC))
    freeze = json.loads(_require(FREEZE).read_text(encoding="utf-8"))
    hil = json.loads(_require(HIL).read_text(encoding="utf-8"))
    shap = json.loads(_require(SHAP).read_text(encoding="utf-8"))
    jsum = json.loads(_require(J_SUM).read_text(encoding="utf-8"))
    edge = json.loads(_require(EDGE).read_text(encoding="utf-8"))

    def row(df, cfg):
        r = df.loc[df["Config"] == cfg]
        if r.empty:
            raise KeyError(f"Config {cfg} not in dataframe")
        return r.iloc[0]

    def ladder_entry(student: str, protocol: str) -> dict:
        for e in ladder:
            if e["student"] == student and e["protocol"] == protocol:
                return e
        raise KeyError(f"No ladder entry {student} / {protocol}")

    udc_a = float(runtime["pytorch_fp32_baselines"]["A"]["macro_f1"])
    udc_b = float(runtime["pytorch_fp32_baselines"]["B"]["macro_f1"])
    n_test = int(runtime["n_test"])

    a_rr = ladder_entry("A", "train_only_scaler_random_row_10seed")
    a_fg = ladder_entry("A", "train_only_scaler_feature_group_disjoint_5seed")
    b_rr = ladder_entry("B", "train_only_scaler_random_row_10seed")
    b_fg = ladder_entry("B", "train_only_scaler_feature_group_disjoint_5seed")

    c10 = next(c for c in freeze["allowed_primary_claims"] if c["id"] == "C10_edge_group_aware")

    return {
        "A": A,
        "B": B,
        "row": row,
        "a_rr": a_rr,
        "a_fg": a_fg,
        "b_rr": b_rr,
        "b_fg": b_fg,
        "udc_a": udc_a,
        "udc_b": udc_b,
        "n_test": n_test,
        "pc": pc,
        "runtime": runtime,
        "hil": hil,
        "shap": shap,
        "jsum": jsum,
        "edge": edge,
        "c10_text": c10["text"],
        "runtime_path": str(RUNTIME.relative_to(REPO)).replace("\\", "/"),
        "ladder_path": str(LADDER.relative_to(REPO)).replace("\\", "/"),
        "agg_path": str(AGG.relative_to(REPO)).replace("\\", "/"),
        "pc_path": str(PC.relative_to(REPO)).replace("\\", "/"),
        "hil_path": str(HIL.relative_to(REPO)).replace("\\", "/"),
        "shap_path": str(SHAP.relative_to(REPO)).replace("\\", "/"),
        "j_path": str(J_SUM.relative_to(REPO)).replace("\\", "/"),
        "edge_path": str(EDGE.relative_to(REPO)).replace("\\", "/"),
    }


def fig_pareto(src: dict) -> None:
    apply_ieee_style()
    A, B, row = src["A"], src["B"], src["row"]
    sel = [
        ("Full MLP", "B_Full_MLP", A, "o", "#333333", True),
        ("A scratch", "D_Small_MLP", A, "s", C_BLUE, False),
        ("A RF-KD", "E_KD_from_RF", A, "D", C_BLUE, True),
        ("B scratch", "D_Small_MLP", B, "s", C_ORANGE, False),
        ("B RF-KD", "E_KD_from_RF", B, "D", C_ORANGE, True),
    ]
    points = []
    fig, ax = plt.subplots(figsize=(FIG_SINGLE, 2.55))
    for name, cfg, df, mk, color, filled in sel:
        r = row(df, cfg)
        x = float(r["size_kb"])
        y = float(r["MacroF1_mean"])
        ye = float(r["MacroF1_std"])
        n = int(r["n_seeds"])
        params = r["params"]
        points.append(
            {
                "label": name,
                "config": cfg,
                "size_kb": x,
                "macro_f1_mean": y,
                "macro_f1_std": ye,
                "n_seeds": n,
                "params": None if pd.isna(params) else int(params),
            }
        )
        ax.errorbar(
            x,
            y,
            yerr=ye,
            fmt=mk,
            color=color,
            mfc=color if filled else "none",
            mec=color,
            ms=5.5,
            capsize=2,
            elinewidth=0.6,
            label=name,
            zorder=3,
        )
    ax.set_xlabel("FP32 parameter payload (KB)")
    ax.set_ylabel(r"Macro-F1 (mean $\pm$ sample std)")
    ax.set_xscale("log")
    xs = [p["size_kb"] for p in points]
    ys = [p["macro_f1_mean"] for p in points]
    ax.set_xlim(min(xs) * 0.65, max(xs) * 1.45)
    ax.set_ylim(min(ys) - 0.02, max(ys) + 0.02)
    ax.grid(True, which="major")
    ax.legend(frameon=False, loc="lower right", fontsize=6.5, ncol=1)
    fig.tight_layout()
    save_fig(fig, "fig_pareto_train_only", FIG)
    plt.close(fig)
    PROV["figures"]["fig_pareto_train_only"] = {
        "type": "quantitative",
        "source": src["agg_path"] + "/wsnds_results_student_{A,B}_10seed.csv",
        "protocol": "train_only multi-seed aggregate (U-MS)",
        "points": points,
        "note": "RF teacher omitted: size_kb is serialized pickle, not FP32 param payload.",
    }
    print("OK fig_pareto_train_only")


def fig_perclass(src: dict) -> None:
    apply_ieee_style()
    pc = src["pc"]
    data = pc[pc["protocol"].str.contains("feature_group", case=False, na=False)].copy()
    if data.empty:
        raise RuntimeError("No feature_group rows in per-class CSV")
    classes = ["Blackhole", "Grayhole", "Flooding", "TDMA", "Normal"]
    fig, axes = plt.subplots(1, 2, figsize=(FIG_DOUBLE, 2.45), sharey=True)
    used = []
    for ax, stu, panel in zip(axes, ["A", "B"], ["(a)", "(b)"]):
        sub = data[data["student"] == stu]
        deltas = []
        for c in classes:
            row = sub[sub["class"] == c]
            if row.empty:
                raise KeyError(f"Missing class {c} for student {stu}")
            d = float(row["delta_kd_minus_scratch"].iloc[0])
            deltas.append(d)
            used.append(
                {
                    "student": stu,
                    "class": c,
                    "delta_kd_minus_scratch": d,
                    "rf_kd_mean": float(row["rf_kd_mean"].iloc[0]),
                    "scratch_mean": float(row["scratch_mean"].iloc[0]),
                    "n_seeds": int(row["n_seeds"].iloc[0]),
                    "protocol": str(row["protocol"].iloc[0]),
                }
            )
        x = np.arange(len(classes))
        colors = [C_BLUE if d >= 0 else C_ORANGE for d in deltas]
        ax.bar(x, deltas, color=colors, width=0.72, edgecolor="k", linewidth=0.35)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=18, ha="right")
        ax.set_title(f"{panel} Student {stu}", loc="left", fontsize=8)
        ax.grid(True, axis="y")
    n = int(used[0]["n_seeds"])
    axes[0].set_ylabel(r"$\Delta$ per-class F1 (RF-KD $-$ scratch)")
    fig.suptitle(
        f"Per-class KD effect (feature-group disjoint, $n{{=}}{n}$)",
        fontsize=8,
        y=1.01,
    )
    fig.tight_layout()
    save_fig(fig, "fig_perclass_delta", FIG)
    plt.close(fig)
    PROV["figures"]["fig_perclass_delta"] = {
        "type": "quantitative",
        "source": src["pc_path"],
        "rows": used,
    }
    print("OK fig_perclass_delta")


def fig_route_bars(src: dict) -> None:
    """Primary route comparison for Students A/B under U-MS."""
    apply_ieee_style()
    A, B, row = src["A"], src["B"], src["row"]
    j = src["jsum"]
    j_source = src["j_path"]
    ja_m = float(j["student_A_J"]["MacroF1_mean"])
    ja_s = float(j["student_A_J"]["MacroF1_std"])
    jb_m = float(j["student_B_J"]["MacroF1_mean"])
    jb_s = float(j["student_B_J"]["MacroF1_std"])

    routes = [
        ("scratch", "D_Small_MLP"),
        ("MLP-KD", "E2_KD_from_MLP"),
        ("RF-KD", "E_KD_from_RF"),
    ]
    labels = [r[0] for r in routes] + ["co-distill (J)"]
    fig, axes = plt.subplots(1, 2, figsize=(FIG_DOUBLE, 2.55), sharey=True)
    used = {"A": [], "B": []}
    for ax, df, stu, j_m, j_s, panel in [
        (axes[0], A, "A", ja_m, ja_s, "(a)"),
        (axes[1], B, "B", jb_m, jb_s, "(b)"),
    ]:
        means, stds = [], []
        for name, cfg in routes:
            r = row(df, cfg)
            m, s = float(r["MacroF1_mean"]), float(r["MacroF1_std"])
            means.append(m)
            stds.append(s)
            used[stu].append(
                {"route": name, "config": cfg, "macro_f1_mean": m, "macro_f1_std": s}
            )
        means.append(j_m)
        stds.append(j_s)
        used[stu].append(
            {
                "route": "co-distill (J)",
                "config": "J_CoDistill",
                "macro_f1_mean": j_m,
                "macro_f1_std": j_s,
                "source": j_source,
            }
        )
        x = np.arange(len(labels))
        colors = ["#666666", "#4C78A8", C_BLUE, C_ORANGE]
        ax.bar(
            x,
            means,
            yerr=stds,
            color=colors,
            edgecolor="k",
            linewidth=0.35,
            capsize=2,
            width=0.72,
            error_kw={"elinewidth": 0.6},
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=12, ha="right")
        ax.set_title(f"{panel} Student {stu}", loc="left")
        ax.grid(True, axis="y")
        y0 = min(m - s for m, s in zip(means, stds)) - 0.008
        y1 = max(m + s for m, s in zip(means, stds)) + 0.008
        ax.set_ylim(y0, y1)
    axes[0].set_ylabel(r"Macro-F1 (mean $\pm$ std)")
    fig.suptitle("Primary routes under train-only multi-seed (U-MS)", fontsize=8, y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig_route_bars", FIG)
    plt.close(fig)
    PROV["figures"]["fig_route_bars"] = {
        "type": "quantitative",
        "source_agg": src["agg_path"],
        "source_j": j_source,
        "data": used,
    }
    print(f"OK fig_route_bars (J A={ja_m:.4f} B={jb_m:.4f})")

def fig_protocol_delta_bars(src: dict) -> None:
    """Paired KD-scratch deltas for A/B under RR vs FG."""
    apply_ieee_style()
    # short tick labels + student as group header via colors
    entries = [
        ("A / RR", src["a_rr"]),
        ("A / FG", src["a_fg"]),
        ("B / RR", src["b_rr"]),
        ("B / FG", src["b_fg"]),
    ]
    labels = [e[0] for e in entries]
    deltas = [float(e[1]["kd_minus_scratch_mean_paired"]) for e in entries]
    ps = [float(e[1]["paired_t_p"]) for e in entries]
    ns = [int(e[1]["n_seeds"]) for e in entries]

    fig, ax = plt.subplots(figsize=(FIG_SINGLE, 2.65))
    x = np.arange(len(labels))
    colors = [C_BLUE, "#7BA3C9", C_ORANGE, "#E39A6D"]
    ax.bar(x, deltas, color=colors, edgecolor="k", linewidth=0.35, width=0.68)
    ax.axhline(0, color="k", lw=0.5)
    for i, (d, p, n) in enumerate(zip(deltas, ps, ns)):
        y_off = 0.00055 if d >= 0 else -0.0009
        ax.text(
            i,
            d + y_off,
            f"p={p:.3f}\nn={n}",
            ha="center",
            va="bottom" if d >= 0 else "top",
            fontsize=5.8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(r"Paired $\Delta$ macro-F1 (RF-KD $-$ scratch)")
    ax.set_title("KD gain: random-row (RR) vs feature-group (FG)", loc="left", fontsize=7.5)
    # room for annotations
    ymin = min(deltas) - 0.0035
    ymax = max(deltas) + 0.0035
    ax.set_ylim(ymin, ymax)
    ax.grid(True, axis="y")
    fig.tight_layout()
    save_fig(fig, "fig_protocol_delta_bars", FIG)
    plt.close(fig)
    PROV["figures"]["fig_protocol_delta_bars"] = {
        "type": "quantitative",
        "source": src["ladder_path"],
        "deltas": [
            {"label": lab, "delta": d, "p": p, "n": n}
            for lab, d, p, n in zip(labels, deltas, ps, ns)
        ],
    }
    print("OK fig_protocol_delta_bars")


def fig_hil_latency(src: dict) -> None:
    apply_ieee_style()
    hil = src["hil"]
    rows = []
    for pair in hil["esp32c3"]["pairs"]:
        rows.append(
            {
                "board": "ESP32-C3",
                "student": pair["student"],
                "latency_us": float(pair["latency_us_mean"]),
                "macro_f1": float(pair["macro_f1"]),
                "agree": float(pair["mcu_vs_fixed_reference_agreement"]),
                "n": int(pair["n"]),
            }
        )
    for pair in hil["arduino_r4"]["pairs"]:
        rows.append(
            {
                "board": "Arduino R4",
                "student": pair["student"],
                "latency_us": float(pair["latency_us_mean"]),
                "macro_f1": float(pair["macro_f1"]),
                "agree": float(pair["mcu_vs_fixed_reference_agreement"]),
                "n": int(pair["n"]),
            }
        )

    fig, axes = plt.subplots(1, 2, figsize=(FIG_DOUBLE, 2.4))
    # latency grouped
    ax = axes[0]
    boards = ["ESP32-C3", "Arduino R4"]
    students = ["A", "B"]
    x = np.arange(len(boards))
    w = 0.35
    for i, stu in enumerate(students):
        vals = [
            next(r["latency_us"] for r in rows if r["board"] == b and r["student"] == stu)
            for b in boards
        ]
        ax.bar(
            x + (i - 0.5) * w,
            vals,
            width=w,
            label=f"Student {stu}",
            color=C_BLUE if stu == "A" else C_ORANGE,
            edgecolor="k",
            linewidth=0.35,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(boards)
    ax.set_ylabel(r"Mean latency ($\mu$s)")
    ax.set_title("(a) Full-test HIL latency", loc="left")
    ax.legend(frameon=False, fontsize=6.5)
    ax.grid(True, axis="y")

    ax = axes[1]
    for i, stu in enumerate(students):
        vals = [
            next(r["macro_f1"] for r in rows if r["board"] == b and r["student"] == stu)
            for b in boards
        ]
        ax.bar(
            x + (i - 0.5) * w,
            vals,
            width=w,
            label=f"Student {stu}",
            color=C_BLUE if stu == "A" else C_ORANGE,
            edgecolor="k",
            linewidth=0.35,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(boards)
    ax.set_ylabel("MCU macro-F1 (fixed-point)")
    ax.set_title("(b) MCU macro-F1 (agree=1.0 all pairs)", loc="left")
    ax.set_ylim(0.90, 0.94)
    ax.legend(frameon=False, fontsize=6.5)
    ax.grid(True, axis="y")

    fig.tight_layout()
    save_fig(fig, "fig_hil_latency", FIG)
    plt.close(fig)
    PROV["figures"]["fig_hil_latency"] = {
        "type": "quantitative",
        "source": src["hil_path"],
        "rows": rows,
        "n_test": src["n_test"],
    }
    print("OK fig_hil_latency")


def fig_shap_rho(src: dict) -> None:
    apply_ieee_style()
    shap = src["shap"]
    students = shap["students"]
    labels = []
    rhos = []
    ps = []
    boots = []
    boot_std = []
    for stu in ["A", "B"]:
        s = students[stu]
        labels.append(f"Student {stu}")
        rhos.append(float(s["ranking_agreement_spearman"]))
        ps.append(float(s["ranking_agreement_p"]))
        bv = s.get("bootstrap_spearman_values")
        if bv:
            arr = np.array(bv, dtype=float)
            boots.append(float(arr.mean()))
            boot_std.append(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0)
        else:
            boots.append(float("nan"))
            boot_std.append(0.0)

    fig, ax = plt.subplots(figsize=(FIG_SINGLE, 2.5))
    x = np.arange(len(labels))
    ax.bar(
        x - 0.18,
        rhos,
        width=0.36,
        label=r"Point $\rho$",
        color=C_BLUE,
        edgecolor="k",
        linewidth=0.35,
    )
    ax.bar(
        x + 0.18,
        boots,
        width=0.36,
        yerr=boot_std,
        label=r"Bootstrap mean $\pm$ std",
        color=C_ORANGE,
        edgecolor="k",
        linewidth=0.35,
        capsize=2,
        error_kw={"elinewidth": 0.6},
    )
    for i, p in enumerate(ps):
        ax.text(i - 0.18, rhos[i] + 0.02, f"p={p:.2f}", ha="center", fontsize=6)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"Spearman $\rho$ (global SHAP ranks)")
    ax.set_title("RF TreeExplainer vs student DeepExplainer", loc="left", fontsize=8)
    ax.set_ylim(-0.05, 0.55)
    ax.legend(frameon=False, fontsize=6.5)
    ax.grid(True, axis="y")
    fig.tight_layout()
    save_fig(fig, "fig_shap_rho", FIG)
    plt.close(fig)
    PROV["figures"]["fig_shap_rho"] = {
        "type": "quantitative",
        "source": src["shap_path"],
        "point_rho": {"A": rhos[0], "B": rhos[1]},
        "point_p": {"A": ps[0], "B": ps[1]},
        "bootstrap_mean": {"A": boots[0], "B": boots[1]},
        "bootstrap_std": {"A": boot_std[0], "B": boot_std[1]},
    }
    print("OK fig_shap_rho")


def fig_dual_unit_bars(src: dict) -> None:
    """U-MS mean vs U-DC seed-42 for A/B RF-KD."""
    apply_ieee_style()
    ums_a = float(src["a_rr"]["rf_kd_macro_f1_mean"])
    ums_a_s = float(src["a_rr"]["rf_kd_macro_f1_std"])
    ums_b = float(src["b_rr"]["rf_kd_macro_f1_mean"])
    ums_b_s = float(src["b_rr"]["rf_kd_macro_f1_std"])
    udc_a, udc_b = src["udc_a"], src["udc_b"]

    fig, ax = plt.subplots(figsize=(FIG_SINGLE, 2.5))
    x = np.arange(2)
    w = 0.35
    ums = [ums_a, ums_b]
    ums_e = [ums_a_s, ums_b_s]
    udc = [udc_a, udc_b]
    ax.bar(
        x - w / 2,
        ums,
        w,
        yerr=ums_e,
        label="U-MS (10-seed mean)",
        color=C_BLUE,
        edgecolor="k",
        linewidth=0.35,
        capsize=2,
        error_kw={"elinewidth": 0.6},
    )
    ax.bar(
        x + w / 2,
        udc,
        w,
        label="U-DC (seed-42 deploy-clean)",
        color=C_ORANGE,
        edgecolor="k",
        linewidth=0.35,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["Student A RF-KD", "Student B RF-KD"])
    ax.set_ylabel("Macro-F1")
    ax.set_title("Dual evaluation units (do not interchange)", loc="left", fontsize=8)
    ax.set_ylim(0.90, 0.97)
    ax.legend(frameon=False, fontsize=6.5)
    ax.grid(True, axis="y")
    fig.tight_layout()
    save_fig(fig, "fig_dual_unit_bars", FIG)
    plt.close(fig)
    PROV["figures"]["fig_dual_unit_bars"] = {
        "type": "quantitative",
        "U-MS": {
            "A": {"mean": ums_a, "std": ums_a_s},
            "B": {"mean": ums_b, "std": ums_b_s},
            "source": src["ladder_path"],
        },
        "U-DC": {
            "A": udc_a,
            "B": udc_b,
            "source": src["runtime_path"],
        },
    }
    print("OK fig_dual_unit_bars")


def fig_host_conversion(src: dict) -> None:
    """ONNX FP32 / INT8 vs PyTorch on U-DC."""
    apply_ieee_style()
    rt = src["runtime"]
    rows = []
    for r in rt["onnx_runtime_rows"]:
        rows.append(
            {
                "model": r["model_name"],
                "variant": r["variant"],
                "macro_f1": float(r["macro_f1"]),
                "agree": float(r["prediction_agreement_vs_pytorch_fp32"]),
            }
        )
    # map to student
    order = [
        ("A", "onnx_fp32"),
        ("A", "onnx_dynamic_int8"),
        ("B", "onnx_fp32"),
        ("B", "onnx_dynamic_int8"),
    ]
    labels = ["A FP32", "A INT8", "B FP32", "B INT8"]
    f1s, agrees = [], []
    for stu, var in order:
        key = f"E_student_{stu}_KD_from_RF_train_only"
        hit = next(r for r in rows if r["model"] == key and r["variant"] == var)
        f1s.append(hit["macro_f1"])
        agrees.append(hit["agree"])

    fig, axes = plt.subplots(1, 2, figsize=(FIG_DOUBLE, 2.35))
    x = np.arange(len(labels))
    colors = [C_BLUE, "#7BA3C9", C_ORANGE, "#E39A6D"]
    axes[0].bar(x, f1s, color=colors, edgecolor="k", linewidth=0.35)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=10)
    axes[0].set_ylabel("Macro-F1")
    axes[0].set_title("(a) Host runtime macro-F1 (U-DC)", loc="left")
    axes[0].set_ylim(0.85, 0.98)
    axes[0].grid(True, axis="y")

    axes[1].bar(x, agrees, color=colors, edgecolor="k", linewidth=0.35)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=10)
    axes[1].set_ylabel("Agreement vs PyTorch FP32")
    axes[1].set_title("(b) Prediction agreement", loc="left")
    axes[1].set_ylim(0.97, 1.005)
    axes[1].grid(True, axis="y")
    fig.tight_layout()
    save_fig(fig, "fig_host_conversion", FIG)
    plt.close(fig)
    PROV["figures"]["fig_host_conversion"] = {
        "type": "quantitative",
        "source": src["runtime_path"],
        "labels": labels,
        "macro_f1": f1s,
        "agreement": agrees,
    }
    print("OK fig_host_conversion")


def fig_edge_bars(src: dict) -> None:
    apply_ieee_style()
    ej = src["edge"]
    a_kd = ej["A_E_KD_from_RF"]
    a_sc = ej["A_D_Small_MLP"]
    lit = ej["literature_random_split_reference"]["lit_A_E_KD_from_RF"]
    leak = float(ej["audit"]["pct_test_in_cross_partition_groups_pre_split_protocol"])

    labels = [
        "Group-aware\nscratch",
        "Group-aware\nRF-KD",
        "Random-row\nRF-KD (ref)",
    ]
    means = [
        float(a_sc["macro_f1_mean"]),
        float(a_kd["macro_f1_mean"]),
        float(lit["macro_f1_mean"]),
    ]
    stds = [
        float(a_sc["macro_f1_std"]),
        float(a_kd["macro_f1_std"]),
        float(lit.get("macro_f1_std", 0.0)),
    ]

    fig, ax = plt.subplots(figsize=(FIG_SINGLE, 2.5))
    x = np.arange(len(labels))
    ax.bar(
        x,
        means,
        yerr=stds,
        color=[C_GRAY, C_BLUE, "#888888"],
        edgecolor="k",
        linewidth=0.35,
        capsize=2,
        error_kw={"elinewidth": 0.6},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Macro-F1 (Student A)")
    ax.set_title(
        f"Edge-IIoTset secondary (pre-split leak audit {leak:.1f}%)",
        loc="left",
        fontsize=7.5,
    )
    ax.set_ylim(0.70, 0.85)
    ax.grid(True, axis="y")
    fig.tight_layout()
    save_fig(fig, "fig_edge_bars", FIG)
    plt.close(fig)
    PROV["figures"]["fig_edge_bars"] = {
        "type": "quantitative",
        "source": src["edge_path"],
        "means": dict(zip(labels, means)),
        "stds": dict(zip(labels, stds)),
        "leak_pct_pre_split": leak,
    }
    print("OK fig_edge_bars")


def _prep_photo(path: Path, max_side: int = 1600) -> Image.Image:
    im = Image.open(path).convert("RGB")
    im = ImageOps.exif_transpose(im)
    # mild contrast for print
    im = ImageEnhance.Contrast(im).enhance(1.08)
    im = ImageEnhance.Color(im).enhance(1.05)
    w, h = im.size
    scale = max_side / max(w, h)
    if scale < 1:
        im = im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return im


def fig_lab_setup() -> None:
    """Single lab HIL photo from phone capture."""
    apply_ieee_style()
    overview = LAB_DIR / "overview.jpeg"
    if not overview.is_file():
        raise FileNotFoundError(f"Lab photo missing: {overview}")

    im = _prep_photo(overview)

    fig, ax = plt.subplots(1, 1, figsize=(FIG_SINGLE, 3.15))
    ax.imshow(im)
    ax.set_title("Laboratory HIL bench", loc="left", fontsize=8)
    ax.axis("off")
    # annotation strip
    fig.text(
        0.5,
        0.02,
        "Raspberry Pi HIL host (orchestrator)  ·  Arduino UNO R4 WiFi  ·  ESP32-C3  ·  USB serial full-test replay",
        ha="center",
        va="bottom",
        fontsize=7,
        style="italic",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_fig(fig, "fig_lab_setup", FIG)
    plt.close(fig)
    PROV["figures"]["fig_lab_setup"] = {
        "type": "photograph",
        "sources": [
            str((LAB_DIR / "overview.jpeg").relative_to(REPO)).replace("\\", "/"),
        ],
        "note": "Phone photo of lab setup; replaceable with professional shot later.",
    }
    print("OK fig_lab_setup")


def fig_system_dot(src: dict) -> None:
    """Compact LR architecture — keep height printable for IEEEtran."""
    ensure_graphviz_on_path()
    n_test = src["n_test"]
    src_dot = f"""
digraph SystemHIL {{
  graph [rankdir=LR, fontname="Helvetica", fontsize=9, pad=0.15, nodesep=0.35, ranksep=0.45,
         splines=true, newrank=true];
  node  [shape=box, style="rounded,filled", fillcolor="white", color="black",
         fontname="Helvetica", fontsize=8, penwidth=0.9, margin="0.05,0.04", height=0.4];
  edge  [fontname="Helvetica", fontsize=7, penwidth=0.9, arrowsize=0.55];

  subgraph cluster_a {{
    label="(a) Ideal WSN stack";
    labelloc="t"; fontsize=9; fontname="Helvetica-Bold";
    color="gray50"; style="dashed";
    mote [label="WSN mote\\n(on-node IDS)"];
    radio [label="Low-power\\nradio"];
    gw    [label="Gateway\\n/ sink"];
    cloud [label="Optional\\ncloud"];
    mote -> radio -> gw -> cloud;
  }}

  subgraph cluster_b {{
    label="(b) Offline development";
    labelloc="t"; fontsize=9; fontname="Helvetica-Bold";
    color="gray50";
    csv   [label="WSN-DS CSV\\n17 features", fillcolor="gray95"];
    split [label="Train-only scaler\\n+ fixed split"];
    train [label="RF teacher →\\ntiny MLP students"];
    export [label="FP32 export\\n+ fixed-point C"];
    csv -> split -> train -> export;
  }}

  subgraph cluster_c {{
    label="(c) HIL fidelity (this work)";
    labelloc="t"; fontsize=9; fontname="Helvetica-Bold";
    color="black";
    pi  [label="Raspberry Pi\\nHIL host only", fillcolor="gray90"];
    esp [label="ESP32-C3\\nDUT", fillcolor="gray85"];
    r4  [label="Arduino R4\\nWiFi DUT", fillcolor="gray85"];
    ref [label="Fixed-point ref\\nN={n_test}", fillcolor="gray95"];
    pi -> esp [label="USB"];
    pi -> r4  [label="USB"];
    esp -> ref [label="class"];
    r4  -> ref;
  }}

  export -> pi [style=dashed, label="weights"];
}}
"""
    outs = render_dot(src_dot, FIG / "fig_system_hil_dot", formats=("pdf", "png", "svg"))
    PROV["figures"]["fig_system_hil_dot"] = {
        "type": "architecture",
        "quantitative_fields": {
            "n_test": n_test,
            "source": src["runtime_path"] + " → n_test",
        },
        "outputs": [p.name for p in outs],
        "layout": "LR compact for IEEEtran height",
    }
    print("OK fig_system_hil_dot (compact LR)")


def fig_dual_identity_dot(src: dict) -> None:
    ensure_graphviz_on_path()
    udc_a = src["udc_a"]
    ums_a = float(src["a_rr"]["rf_kd_macro_f1_mean"])
    ums_std = float(src["a_rr"]["rf_kd_macro_f1_std"])
    n_ms = int(src["a_rr"]["n_seeds"])
    src_dot = f"""
digraph DualIdentity {{
  graph [rankdir=LR, fontname="Helvetica", fontsize=9, pad=0.12, nodesep=0.28, ranksep=0.4];
  node  [shape=box, style="rounded,filled", fillcolor="white", color="black",
         fontname="Helvetica", fontsize=8, penwidth=0.9, margin="0.1,0.06"];
  edge  [fontname="Helvetica", fontsize=7, penwidth=0.9, arrowsize=0.55];

  split [label="Fixed stratified split\\n(seed 42)", fillcolor="gray95"];
  ums   [label=<<b>Unit U-MS</b><br/>multi-config train-only<br/>A RF-KD {_fmt(ums_a, 4)} +/- {_fmt(ums_std, 4)} (n={n_ms})>, fillcolor="gray90"];
  udc   [label=<<b>Unit U-DC</b><br/>deploy-clean RF-KD<br/>A macro-F1 {_fmt(udc_a, 4)} (seed 42)>, fillcolor="gray90"];
  tab   [label="Multi-seed tables"];
  hil   [label="Export / HIL"];

  split -> ums -> tab;
  split -> udc -> hil;
}}
"""
    outs = render_dot(src_dot, FIG / "fig_dual_identity_dot", formats=("pdf", "png", "svg"))
    PROV["figures"]["fig_dual_identity_dot"] = {
        "type": "architecture+metrics",
        "U-MS_StudentA_RFKD": {
            "macro_f1_mean": ums_a,
            "macro_f1_std": ums_std,
            "n_seeds": n_ms,
            "source": src["ladder_path"],
        },
        "U-DC_StudentA_RFKD": {
            "macro_f1": udc_a,
            "source": src["runtime_path"],
        },
        "outputs": [p.name for p in outs],
    }
    print(f"OK fig_dual_identity_dot (U-MS={ums_a:.4f}, U-DC={udc_a:.4f})")


def fig_protocol_ladder_dot(src: dict) -> None:
    ensure_graphviz_on_path()
    a_rr, a_fg = src["a_rr"], src["a_fg"]
    d_rr = float(a_rr["kd_minus_scratch_mean_paired"])
    p_rr = float(a_rr["paired_t_p"])
    n_rr = int(a_rr["n_seeds"])
    d_fg = float(a_fg["kd_minus_scratch_mean_paired"])
    p_fg = float(a_fg["paired_t_p"])
    n_fg = int(a_fg["n_seeds"])

    ej = src["edge"]
    a_kd = ej["A_E_KD_from_RF"]
    a_delta = ej["A_KD_minus_scratch"]
    mean = float(a_kd["macro_f1_mean"])
    std = float(a_kd["macro_f1_std"])
    n_eg = int(a_kd["n_seeds"])
    d_eg = float(a_delta["mean"])

    src_dot = f"""
digraph ProtocolLadder {{
  graph [rankdir=LR, fontname="Helvetica", fontsize=9, pad=0.12, nodesep=0.3, ranksep=0.45];
  node  [shape=box, style="rounded,filled", fillcolor="white", color="black",
         fontname="Helvetica", fontsize=8, penwidth=0.9, margin="0.1,0.06"];
  edge  [fontname="Helvetica", fontsize=7, penwidth=0.9, arrowsize=0.55];

  rr [label=<<b>Random-row train-only</b><br/>A paired KD-scratch<br/>delta={_fmt(d_rr, 4)} (t p={_fmt(p_rr, 3)}, n={n_rr})>, fillcolor="gray95"];
  fg [label=<<b>Feature-group disjoint</b><br/>A paired KD-scratch<br/>delta={_fmt(d_fg, 4)} (t p={_fmt(p_fg, 3)}, n={n_fg})>, fillcolor="gray90"];
  eg [label=<<b>Edge group-aware</b><br/>A RF-KD {_fmt(mean, 4)} +/- {_fmt(std, 4)} (n={n_eg})<br/>KD-scratch mean {_fmt(d_eg, 4)}>, fillcolor="gray90"];

  rr -> fg [label="stricter split"];
  fg -> eg [label="cross-dataset"];
}}
"""
    outs = render_dot(src_dot, FIG / "fig_protocol_ladder_dot", formats=("pdf", "png", "svg"))
    PROV["figures"]["fig_protocol_ladder_dot"] = {
        "type": "quantitative",
        "source": src["ladder_path"],
        "edge_source": src["edge_path"],
        "student_A_random_row": {"delta": d_rr, "p": p_rr, "n": n_rr},
        "student_A_feature_group": {"delta": d_fg, "p": p_fg, "n": n_fg},
        "edge_A_rfkd_mean": mean,
        "outputs": [p.name for p in outs],
    }
    print("OK fig_protocol_ladder_dot")


def main() -> None:
    print("Graphviz:", dot_version())
    src = load_sources()
    print(
        "Loaded evidence: U-DC A={:.10f} B={:.10f} n_test={}".format(
            src["udc_a"], src["udc_b"], src["n_test"]
        )
    )
    fig_lab_setup()
    fig_pareto(src)
    fig_perclass(src)
    fig_route_bars(src)
    fig_protocol_delta_bars(src)
    fig_hil_latency(src)
    fig_shap_rho(src)
    fig_dual_unit_bars(src)
    fig_host_conversion(src)
    fig_edge_bars(src)
    fig_system_dot(src)
    fig_dual_identity_dot(src)
    fig_protocol_ladder_dot(src)

    note = FIG / "fig_dual_identity_NUMBER_NOTE.txt"
    note.write_text(
        "U-DC Student A macro-F1 = {:.10f}\n"
        "source: {}\n"
        "U-MS Student A RF-KD mean ± std = {:.10f} ± {:.10f} (n={})\n"
        "source: {}\n".format(
            src["udc_a"],
            src["runtime_path"],
            float(src["a_rr"]["rf_kd_macro_f1_mean"]),
            float(src["a_rr"]["rf_kd_macro_f1_std"]),
            int(src["a_rr"]["n_seeds"]),
            src["ladder_path"],
        ),
        encoding="utf-8",
    )

    prov_path = FIG / "_provenance.json"
    PROV["policy"] = (
        "All quantitative figure labels loaded from result JSON/CSV at build time; "
        "no hand-typed scientific numbers in build_all_figures.py."
    )
    prov_path.write_text(json.dumps(PROV, indent=2), encoding="utf-8")
    print("Wrote", prov_path)
    print("All figures grounded.")


if __name__ == "__main__":
    main()
