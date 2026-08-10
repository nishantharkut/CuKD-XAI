#!/usr/bin/env python3
"""Build manuscript tables and plot data directly from repository evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "manuscript" / "generated"

WSN_DIR = ROOT / "results" / "wsnds" / "final_results" / "2026-05-30-10seed-plus-j"
WSN_A = WSN_DIR / "wsnds_results_student_A.csv"
WSN_B = WSN_DIR / "wsnds_results_student_B.csv"
WSN_JSON = WSN_DIR / "cukd_xai_results_with_J.json"
RUNTIME = ROOT / "results" / "runtime" / "onnx_openvino" / "wsnds"
ONNX_RUNTIME = RUNTIME / "runtime_from_existing_outputs" / "wsnds_existing_artifact_runtime_summary.csv"
QAT = RUNTIME / "wsnds_qat_summary.csv"
DEPLOYMENT_RESULTS = RUNTIME / "wsnds_deployment_results.json"
HIL = ROOT / "results" / "hardware_hil" / "reports" / "final_postprocessing"
HIL_REPLAY = ROOT / "results" / "hardware_hil" / "board_replay"
COMPILE_LOG_DIR = ROOT / "results" / "hardware_hil" / "compile_logs"
EXPORT_ROOT = ROOT / "deployment" / "firmware_export" / "wsnds_rfkd_hil"
HIL_EXPORT_A = EXPORT_ROOT / "generated_student_a_rfkd_hil_full"
HIL_EXPORT_B = EXPORT_ROOT / "generated_student_b_rfkd_hil_full"
EDGE_STRICT = ROOT / "results" / "edge_iiot" / "strict_generalization" / "edgeiiot_v23_config_rankings.csv"
EDGE_COMPARABLE = ROOT / "results" / "edge_iiot" / "literature_comparable" / "edgeiiot_v23_config_rankings.csv"

SEQUENCE_FILES = [
    HIL_REPLAY / "pi5_esp32c3" / "full_56200_sequence.json",
    HIL_REPLAY / "pi5_arduino_r4" / "full_56200_sequence.json",
    HIL_REPLAY / "pi5_esp32c3_student_b" / "full_56200_sequence.json",
    HIL_REPLAY / "pi5_arduino_r4_student_b" / "full_56200_sequence.json",
]
COMPILE_LOG_FILES = [
    COMPILE_LOG_DIR / "esp32c3_serial_baseline_compile.txt",
    COMPILE_LOG_DIR / "arduino_r4_serial_baseline_compile.txt",
    COMPILE_LOG_DIR / "esp32c3_student_a_compile.txt",
    COMPILE_LOG_DIR / "arduino_r4_student_a_compile.txt",
    COMPILE_LOG_DIR / "esp32c3_student_b_compile.txt",
    COMPILE_LOG_DIR / "arduino_r4_student_b_compile.txt",
]

SOURCES = [
    WSN_A,
    WSN_B,
    WSN_JSON,
    ONNX_RUNTIME,
    QAT,
    DEPLOYMENT_RESULTS,
    HIL_EXPORT_A / "export_summary.json",
    HIL_EXPORT_A / "preprocess_metadata.json",
    HIL_EXPORT_B / "export_summary.json",
    HIL_EXPORT_B / "preprocess_metadata.json",
    HIL / "hil_fidelity.csv",
    HIL / "cycles_per_mac.csv",
    HIL / "compile_framework_baseline.csv",
    HIL / "model_only_footprint.csv",
    HIL / "quantization_drift_summary.csv",
    *SEQUENCE_FILES,
    *COMPILE_LOG_FILES,
    EDGE_STRICT,
    EDGE_COMPARABLE,
]


CONFIG_LABELS = {
    "A_RF_500": "RF-500 teacher",
    "A_LightGBM": "LightGBM teacher",
    "B_Full_MLP": "Full MLP",
    "C2_CL_MLP_domain": "Domain curriculum MLP",
    "C_CL_MLP_loss": "Loss curriculum MLP (alias)",
    "C_CL_MLP_loss_ext": "Loss curriculum MLP (extended)",
    "C_CL_MLP_loss_fair": "Loss curriculum MLP (nominal schedule)",
    "D_Small_MLP": "Scratch",
    "E2_KD_from_MLP": "MLP-KD",
    "E_KD_from_RF": "RF-KD",
    "E3_KD_from_LightGBM": "LightGBM-KD",
    "F_KD_from_CL_MLP": "Curriculum-KD (alias)",
    "F_KD_from_CL_MLP_ext": "Curriculum-KD (extended)",
    "F_KD_from_CL_MLP_fair": "Curriculum-KD (nominal schedule)",
    "G_KD_random_pacing": "Random pacing KD",
    "I_KD_from_SMOTE_MLP": "SMOTE-teacher KD",
    "J_CoDistill_RF_CL": "RF+curriculum co-distillation",
}

STUDENT_LABELS = {
    "student_A_32_16": "A (32--16)",
    "student_B_64_32": "B (64--32)",
    "student_C_128_64": "C (128--64)",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write(name: str, content: str) -> None:
    (OUT / name).write_text(content.rstrip() + "\n", encoding="ascii")


def tex(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def f(value: str | float, digits: int = 4) -> str:
    if value in ("", None):
        return "--"
    return f"{float(value):.{digits}f}"


def integer(value: str | float) -> str:
    if value in ("", None):
        return "--"
    return f"{int(float(value)):,}"


def rows_to_tex(rows: Iterable[Iterable[object]]) -> str:
    return "\n".join(" & ".join(str(cell) for cell in row) + r" \\" for row in rows)


def build_wsn_tables() -> dict[str, list[dict[str, str]]]:
    tables: dict[str, list[dict[str, str]]] = {}
    for student, path in (("A", WSN_A), ("B", WSN_B)):
        data = read_csv(path)
        tables[student] = data
        rows = []
        for row in data:
            config = row["Config"]
            label = CONFIG_LABELS.get(config, config)
            rows.append(
                (
                    tex(label),
                    f(row["Accuracy_mean"]),
                    f(row["Accuracy_std"]),
                    f(row["MacroF1_mean"]),
                    f(row["MacroF1_std"]),
                    integer(row["params"]),
                    f(row["size_kb"], 2),
                )
            )
        write(f"wsn_student_{student.lower()}_all_rows.tex", rows_to_tex(rows))

    selected = [
        ("RF", tables["A"], "A_RF_500"),
        ("Full MLP", tables["A"], "B_Full_MLP"),
        ("A RF-KD", tables["A"], "E_KD_from_RF"),
        ("A co-distill", tables["A"], "J_CoDistill_RF_CL"),
        ("B scratch", tables["B"], "D_Small_MLP"),
        ("B RF-KD", tables["B"], "E_KD_from_RF"),
        ("B co-distill", tables["B"], "J_CoDistill_RF_CL"),
    ]
    class_fields = ["Blackhole_F1_mean", "Flooding_F1_mean", "Grayhole_F1_mean", "Normal_F1_mean", "TDMA_F1_mean"]
    rows = []
    for label, table, config in selected:
        row = next(item for item in table if item["Config"] == config)
        rows.append((tex(label), *(f(row[field], 3) for field in class_fields), f(row["MacroF1_mean"], 3)))
    write("wsn_selected_per_class_rows.tex", rows_to_tex(rows))

    pareto_rows = []
    for student, table in tables.items():
        for config in ("D_Small_MLP", "E2_KD_from_MLP", "E_KD_from_RF", "F_KD_from_CL_MLP_fair", "J_CoDistill_RF_CL"):
            row = next(item for item in table if item["Config"] == config)
            pareto_rows.append(
                f"{student}-{config}\t{float(row['size_kb']):.8f}\t{float(row['MacroF1_mean']):.8f}\n"
            )
    write("wsn_pareto.tsv", "label\tsize_kb\tmacro_f1\n" + "".join(pareto_rows))
    return tables


def build_statistics_and_shap() -> None:
    payload = json.loads(WSN_JSON.read_text(encoding="utf-8"))
    comparisons = payload["wilcoxon_results_with_J"]
    selected: list[tuple[str, dict[str, object]]] = []
    for student_key, label in (("student_A_32_16", "A"), ("student_B_64_32", "B")):
        for _, result in comparisons[student_key].items():
            if result["b_config"] == "F_KD_from_CL_MLP":
                continue
            selected.append((label, result))

    # The archived result JSON stores five unique co-distillation comparisons
    # for each student plus duplicate alias rows. Control family-wise error
    # across the ten unique tests with the step-down Holm procedure.
    ordered = sorted(enumerate(selected), key=lambda item: float(item[1][1]["p"]))
    adjusted = [0.0] * len(selected)
    running_max = 0.0
    family_size = len(selected)
    for rank, (original_index, (_, result)) in enumerate(ordered):
        candidate = (family_size - rank) * float(result["p"])
        running_max = max(running_max, candidate)
        adjusted[original_index] = min(1.0, running_max)

    rows = []
    for index, (label, result) in enumerate(selected):
        rows.append(
            (
                label,
                tex(CONFIG_LABELS[result["b_config"]]),
                f(result["diff_mean"], 4),
                f(result["stat"], 1),
                f(result["p"], 4),
                f(adjusted[index], 4),
            )
        )
    write("wsn_codistill_wilcoxon_rows.tex", rows_to_tex(rows))

    shap = payload["shap_results"]
    student_rank = {item["feature"]: rank for rank, item in enumerate(shap["student_global_importance"], 1)}
    teacher_rank = {item["feature"]: rank for rank, item in enumerate(shap["teacher_global_importance"], 1)}
    rows = []
    for feature in student_rank:
        rows.append((tex(feature), teacher_rank[feature], student_rank[feature], student_rank[feature] - teacher_rank[feature]))
    write("shap_rank_rows.tex", rows_to_tex(rows))
    write(
        "shap_summary_macros.tex",
        "\n".join(
            [
                rf"\newcommand{{\ShapRho}}{{{f(shap['ranking_agreement_spearman'], 4)}}}",
                rf"\newcommand{{\ShapP}}{{{f(shap['ranking_agreement_p'], 4)}}}",
                rf"\newcommand{{\ShapRepeatMean}}{{{f(shap['bootstrap_spearman_mean'], 4)}}}",
                rf"\newcommand{{\ShapRepeatStd}}{{{f(shap['bootstrap_spearman_std'], 4)}}}",
            ]
        ),
    )


def build_runtime_tables() -> None:
    runtime_rows = []
    for row in read_csv(ONNX_RUNTIME):
        runtime_label = {
            "onnx_fp32": "ONNX FP32",
            "onnx_dynamic_int8": "ONNX dynamic INT8",
            "openvino_fp32_from_onnx": "OpenVINO FP32",
        }[row["variant"]]
        agreement = row["openvino_prediction_agreement_vs_onnx"] or "--"
        runtime_rows.append(
            (
                tex(row["model_name"].replace("_student_", " ").replace("_", " ")),
                runtime_label,
                f(row["accuracy"]),
                f(row["macro_f1"]),
                f(row["serialized_size_kb"], 2),
                f(row["latency_p50_ms_b1"], 4),
                f(row["latency_p95_ms_b1"], 4),
                f(agreement, 3) if agreement != "--" else "--",
            )
        )
    write("runtime_all_rows.tex", rows_to_tex(runtime_rows))

    qat_rows = []
    for row in read_csv(QAT):
        qat_rows.append(
            (
                tex(row["model_name"].replace("_student_", " ").replace("_", " ")),
                f(row["accuracy"]),
                f(row["macro_f1"]),
                f(row["macro_f1_delta_vs_fp32"]),
            )
        )
    write("qat_all_rows.tex", rows_to_tex(qat_rows))


def build_hil_tables() -> None:
    rows = []
    for row in read_csv(HIL / "hil_fidelity.csv"):
        rows.append(
            (
                tex(row["model"]),
                tex(row["board"]),
                integer(row["vectors"]),
                f(row["macro_f1"]),
                f(row["mcu_vs_fixed"], 3),
                f(row["mcu_vs_fp32"], 4),
                f(row["mean_total_us"], 2),
                f(row["p99_total_us"], 0),
            )
        )
    write("hil_fidelity_rows.tex", rows_to_tex(rows))

    rows = []
    for row in read_csv(HIL / "cycles_per_mac.csv"):
        rows.append(
            (
                tex(row["model"]),
                tex(row["board"].replace(" DevKitM-1", "").replace(" WiFi", "")),
                integer(row["macs"]),
                f(row["mean_inference_us"], 2),
                f(row["inference_cycles"], 0),
                f(row["cycles_per_mac"], 2),
                f(row["total_throughput_per_s"], 1),
            )
        )
    write("hil_cycles_rows.tex", rows_to_tex(rows))

    rows = []
    for row in read_csv(HIL / "compile_framework_baseline.csv"):
        rows.append(
            (
                tex(row["model"].replace("Student ", "")),
                "C3" if row["board"].startswith("ESP32") else "R4",
                integer(row["program_bytes"]),
                integer(row["global_bytes"]),
                integer(row["program_delta_vs_serial_baseline"]),
                integer(row["global_delta_vs_serial_baseline"]),
            )
        )
    write("hil_compile_rows.tex", rows_to_tex(rows))

    footprints = {row["model"]: row for row in read_csv(HIL / "model_only_footprint.csv")}
    drifts = {row["model"]: row for row in read_csv(HIL / "quantization_drift_summary.csv")}
    rows = []
    for model, fp in footprints.items():
        drift = drifts[model]
        rows.append(
            (
                tex(model),
                tex(fp["architecture"]),
                integer(fp["macs"]),
                integer(fp["param_bytes"]),
                integer(drift["drift_count"]),
                f(100 * float(drift["drift_fraction"]), 3) + r"\%",
            )
        )
    write("hil_model_drift_rows.tex", rows_to_tex(rows))


def build_figures(wsn_tables: dict[str, list[dict[str, str]]]) -> None:
    def get(table: list[dict[str, str]], config: str, field: str) -> float:
        return float(next(row for row in table if row["Config"] == config)[field])

    a = wsn_tables["A"]
    b = wsn_tables["B"]
    wsn_figure = rf"""
\begin{{figure}}[t]
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
  width=\columnwidth,height=48mm,
  xmode=log,log basis x=10,
  xmin=3.6,xmax=400,ymin=0.905,ymax=0.938,
  xlabel={{FP32 parameter payload (KB)}},ylabel={{Macro-F1}},
  grid=both,major grid style={{gray!25}},minor grid style={{gray!10}},
  legend style={{font=\scriptsize,at={{(0.5,-0.30)}},anchor=north,legend columns=3}},
  tick label style={{font=\scriptsize}},label style={{font=\scriptsize}}
]
\addplot[only marks,mark=*,mark size=2.2pt,blue] coordinates {{
 ({get(a,'D_Small_MLP','size_kb'):.5f},{get(a,'D_Small_MLP','MacroF1_mean'):.6f})
 ({get(a,'E_KD_from_RF','size_kb'):.5f},{get(a,'E_KD_from_RF','MacroF1_mean'):.6f})
 ({get(a,'J_CoDistill_RF_CL','size_kb'):.5f},{get(a,'J_CoDistill_RF_CL','MacroF1_mean'):.6f}) }};
\addlegendentry{{Student A}}
\addplot[only marks,mark=square*,mark size=2.2pt,red!75!black] coordinates {{
 ({get(b,'D_Small_MLP','size_kb'):.5f},{get(b,'D_Small_MLP','MacroF1_mean'):.6f})
 ({get(b,'E_KD_from_RF','size_kb'):.5f},{get(b,'E_KD_from_RF','MacroF1_mean'):.6f})
 ({get(b,'J_CoDistill_RF_CL','size_kb'):.5f},{get(b,'J_CoDistill_RF_CL','MacroF1_mean'):.6f}) }};
\addlegendentry{{Student B}}
\addplot[only marks,mark=triangle*,mark size=2.5pt,black] coordinates {{
 ({get(a,'B_Full_MLP','size_kb'):.5f},{get(a,'B_Full_MLP','MacroF1_mean'):.6f}) }};
\addlegendentry{{Full MLP}}
\node[font=\tiny,anchor=south west] at (axis cs:4.6445,{get(a,'E_KD_from_RF','MacroF1_mean'):.6f}) {{A RF-KD}};
\node[font=\tiny,anchor=south west] at (axis cs:13.2695,{get(b,'J_CoDistill_RF_CL','MacroF1_mean'):.6f}) {{B co-distill}};
\end{{axis}}
\end{{tikzpicture}}
\caption{{WSN-DS size--performance trade-off. Points are ten-seed means; all configurations at a student capacity share the same FP32 parameter size.}}
\label{{fig:wsn-tradeoff}}
\end{{figure}}
"""
    write("fig_wsn_tradeoff.tex", wsn_figure)

    hil = read_csv(HIL / "hil_fidelity.csv")
    values = [float(row["mean_total_us"]) for row in hil]
    hil_figure = rf"""
\begin{{figure}}[t]
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
  width=\columnwidth,height=46mm,ybar,bar width=8mm,
  symbolic x coords={{A-C3,A-R4,B-C3,B-R4}},xtick=data,
  ymin=0,ymax=900,ylabel={{Mean total latency ($\mu$s)}},
  nodes near coords,nodes near coords style={{font=\tiny}},
  tick label style={{font=\scriptsize}},label style={{font=\scriptsize}},
  ymajorgrids=true,major grid style={{gray!25}}
]
\addplot[fill=blue!45,draw=blue!70!black] coordinates {{
 (A-C3,{values[0]:.2f}) (A-R4,{values[1]:.2f})
 (B-C3,{values[2]:.2f}) (B-R4,{values[3]:.2f}) }};
\end{{axis}}
\end{{tikzpicture}}
\caption{{Measured fixed-point preprocessing-plus-inference latency over all 56,200 records. C3 denotes ESP32-C3 and R4 denotes Arduino UNO R4 WiFi.}}
\label{{fig:hil-latency}}
\end{{figure}}
"""
    write("fig_hil_latency.tex", hil_figure)

    strict = read_csv(EDGE_STRICT)
    comparable = read_csv(EDGE_COMPARABLE)
    strict_rf = get(strict, "A_RF_500", "macro_f1_mean")
    strict_best = max(float(row["macro_f1_mean"]) for row in strict if row["params"])
    comp_rf = get(comparable, "A_RF_500", "macro_f1_mean")
    comp_best = max(float(row["macro_f1_mean"]) for row in comparable if row["params"])
    edge_figure = rf"""
\begin{{figure}}[t]
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
  width=\columnwidth,height=45mm,ybar,bar width=7mm,
  symbolic x coords={{Strict,Literature-oriented}},xtick=data,
  ymin=0.60,ymax=0.92,ylabel={{Macro-F1}},
  legend style={{font=\scriptsize,at={{(0.5,-0.27)}},anchor=north,legend columns=2}},
  tick label style={{font=\scriptsize}},label style={{font=\scriptsize}},
  ymajorgrids=true,major grid style={{gray!25}}
]
\addplot[fill=black!35] coordinates {{(Strict,{strict_rf:.6f}) (Literature-oriented,{comp_rf:.6f})}};
\addplot[fill=red!50] coordinates {{(Strict,{strict_best:.6f}) (Literature-oriented,{comp_best:.6f})}};
\legend{{RF teacher,Best compact student}}
\end{{axis}}
\end{{tikzpicture}}
\caption{{Protocol sensitivity on Edge-IIoTset. The protocols use different source files and retained features, so the gap is not attributed to training alone.}}
\label{{fig:edge-protocol}}
\end{{figure}}
"""
    write("fig_edge_protocol.tex", edge_figure)


def dedupe_teacher_rows(data: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    output = []
    for row in data:
        config = row["Config"]
        if config in {"A_RF_500", "A_LightGBM", "B_Full_MLP", "C_CL_MLP_loss_fair"}:
            if config in seen:
                continue
            seen.add(config)
        output.append(row)
    return output


def parse_compile_log(path: Path) -> tuple[int, int]:
    text = path.read_text(encoding="utf-8")
    program = re.search(r"Sketch uses (\d+) bytes", text)
    globals_ = re.search(r"Global variables use (\d+) bytes", text)
    if program is None or globals_ is None:
        raise RuntimeError(f"Cannot parse Arduino footprint from {path}")
    return int(program.group(1)), int(globals_.group(1))


def validate_hil_evidence() -> dict[str, int]:
    completed = 0
    for path in SEQUENCE_FILES:
        sequence = read_json(path)
        if not isinstance(sequence, dict):
            raise RuntimeError(f"Invalid HIL sequence document: {path}")
        expected = sequence.get("expected")
        observed = sequence.get("completed")
        if expected != 56_200 or observed != expected:
            raise RuntimeError(f"Incomplete HIL replay in {path}: {observed}/{expected}")
        if sequence.get("missing") or sequence.get("duplicates") or sequence.get("unexpected"):
            raise RuntimeError(f"Sequence integrity failure in {path}")
        if sequence.get("status_counts") != {"OK": expected}:
            raise RuntimeError(f"Non-OK HIL status in {path}: {sequence.get('status_counts')}")
        completed += int(observed)

    compile_rows = read_csv(HIL / "compile_framework_baseline.csv")
    for row in compile_rows:
        program, globals_ = parse_compile_log(ROOT / row["compile_log"])
        baseline_program, baseline_globals = parse_compile_log(ROOT / row["baseline_log"])
        observed = {
            "program_bytes": program,
            "global_bytes": globals_,
            "serial_baseline_program_bytes": baseline_program,
            "serial_baseline_global_bytes": baseline_globals,
            "program_delta_vs_serial_baseline": program - baseline_program,
            "global_delta_vs_serial_baseline": globals_ - baseline_globals,
        }
        expected = {key: int(row[key]) for key in observed}
        if observed != expected:
            raise RuntimeError(
                f"Compile-footprint mismatch for {row['model']} on {row['board']}: "
                f"{observed} != {expected}"
            )

    return {
        "hil_full_replays": len(SEQUENCE_FILES),
        "hil_completed_vectors": completed,
        "compile_logs": len(COMPILE_LOG_FILES),
        "compile_table_rows": len(compile_rows),
    }


def build_edge_tables() -> None:
    for name, path in (("strict", EDGE_STRICT), ("comparable", EDGE_COMPARABLE)):
        rows = []
        for row in dedupe_teacher_rows(read_csv(path)):
            rows.append(
                (
                    tex(STUDENT_LABELS.get(row["student_name"], row["student_name"])),
                    tex(CONFIG_LABELS.get(row["Config"], row["Config"])),
                    f(row["accuracy_mean"]),
                    f(row["accuracy_std"]),
                    f(row["macro_f1_mean"]),
                    f(row["macro_f1_std"]),
                    integer(row["params"]),
                    f(row["size_kb"], 2),
                )
            )
        write(f"edge_{name}_all_rows.tex", rows_to_tex(rows))


def build_manifest() -> None:
    hil_invariants = validate_hil_evidence()
    manifest = {"generator": str(Path(__file__).relative_to(ROOT)), "sources": {}}
    for path in SOURCES:
        raw = path.read_bytes()
        entry: dict[str, object] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
        if path.suffix == ".csv":
            entry["data_rows"] = len(read_csv(path))
        elif path.suffix == ".json":
            value = read_json(path)
            if isinstance(value, dict) and {"expected", "completed"} <= value.keys():
                entry["expected_vectors"] = value["expected"]
                entry["completed_vectors"] = value["completed"]
        manifest["sources"][str(path.relative_to(ROOT)).replace("\\", "/")] = entry
    manifest["invariants"] = {
        "wsn_student_a_rows": len(read_csv(WSN_A)),
        "wsn_student_b_rows": len(read_csv(WSN_B)),
        "runtime_rows": len(read_csv(ONNX_RUNTIME)),
        "qat_rows": len(read_csv(QAT)),
        "hil_pairs": len(read_csv(HIL / "hil_fidelity.csv")),
        "edge_strict_rows": len(read_csv(EDGE_STRICT)),
        "edge_comparable_rows": len(read_csv(EDGE_COMPARABLE)),
        **hil_invariants,
    }
    expected = {
        "wsn_student_a_rows": 15,
        "wsn_student_b_rows": 15,
        "runtime_rows": 18,
        "qat_rows": 6,
        "hil_pairs": 4,
        "edge_strict_rows": 16,
        "edge_comparable_rows": 18,
        "hil_full_replays": 4,
        "hil_completed_vectors": 224800,
        "compile_logs": 6,
        "compile_table_rows": 4,
    }
    if manifest["invariants"] != expected:
        raise RuntimeError(f"Evidence shape changed: {manifest['invariants']} != {expected}")
    (OUT / "evidence_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    wsn_tables = build_wsn_tables()
    build_statistics_and_shap()
    build_runtime_tables()
    build_hil_tables()
    build_edge_tables()
    build_figures(wsn_tables)
    build_manifest()
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
