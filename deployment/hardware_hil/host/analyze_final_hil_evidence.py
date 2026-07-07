"""Generate final post-processing tables for CuKD-XAI hardware evidence.

This script does not run hardware. It consumes already generated HIL metrics,
reference prediction CSV files, and optional Arduino IDE compile logs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


CLASS_NAMES = {
    0: "Blackhole",
    1: "Flooding",
    2: "Grayhole",
    3: "Normal",
    4: "TDMA",
}

MODEL_SPECS = {
    "student_a": {
        "label": "Student A RF-KD",
        "architecture": "17-32-16-5",
        "macs": 1136,
        "weight_bytes": 1136,
        "bias_bytes": 212,
        "param_bytes": 1348,
    },
    "student_b": {
        "label": "Student B RF-KD",
        "architecture": "17-64-32-5",
        "macs": 3296,
        "weight_bytes": 3296,
        "bias_bytes": 404,
        "param_bytes": 3700,
    },
}

BOARD_SPECS = {
    "esp32c3": {
        "label": "ESP32-C3 DevKitM-1",
        "clock_mhz": 160.0,
        "baseline_compile_log": "results/hardware_hil/compile_logs/esp32c3_serial_baseline_compile.txt",
    },
    "arduino_r4": {
        "label": "Arduino R4 WiFi",
        "clock_mhz": 48.0,
        "baseline_compile_log": "results/hardware_hil/compile_logs/arduino_r4_serial_baseline_compile.txt",
    },
}

RUNS = [
    {
        "id": "student_a_esp32c3",
        "model": "student_a",
        "board": "esp32c3",
        "metrics": [
            "results/hardware_hil/board_replay/pi5_esp32c3/full_56200_metrics.json",
            "results/hardware_hil/board_replay/esp32c3/full_56200_metrics.json",
        ],
        "mcu_csv": [
            "results/hardware_hil/board_replay/pi5_esp32c3/full_56200_mcu.csv",
            "results/hardware_hil/board_replay/esp32c3/full_56200_mcu.csv",
        ],
        "reference_csv": [
            "deployment/firmware_export/wsnds_rfkd_hil/generated_student_a_rfkd_hil_full/hil_reference_predictions.csv",
            "deployment/firmware_export/wsnds_rfkd_hil/generated_student_a_rfkd_hil_full_2nd_command/hil_reference_predictions.csv",
        ],
        "compile_log": "results/hardware_hil/compile_logs/esp32c3_student_a_compile.txt",
    },
    {
        "id": "student_a_arduino_r4",
        "model": "student_a",
        "board": "arduino_r4",
        "metrics": [
            "results/hardware_hil/board_replay/pi5_arduino_r4/full_56200_metrics.json",
            "results/hardware_hil/board_replay/arduino_r4/full_56200_metrics.json",
        ],
        "mcu_csv": [
            "results/hardware_hil/board_replay/pi5_arduino_r4/full_56200_mcu.csv",
            "results/hardware_hil/board_replay/arduino_r4/full_56200_mcu.csv",
        ],
        "reference_csv": [
            "deployment/firmware_export/wsnds_rfkd_hil/generated_student_a_rfkd_hil_full/hil_reference_predictions.csv",
            "deployment/firmware_export/wsnds_rfkd_hil/generated_student_a_rfkd_hil_full_2nd_command/hil_reference_predictions.csv",
        ],
        "compile_log": "results/hardware_hil/compile_logs/arduino_r4_student_a_compile.txt",
    },
    {
        "id": "student_b_esp32c3",
        "model": "student_b",
        "board": "esp32c3",
        "metrics": [
            "results/hardware_hil/board_replay/pi5_esp32c3_student_b/full_56200_metrics.json",
        ],
        "mcu_csv": [
            "results/hardware_hil/board_replay/pi5_esp32c3_student_b/full_56200_mcu.csv",
        ],
        "reference_csv": [
            "deployment/firmware_export/wsnds_rfkd_hil/generated_student_b_rfkd_hil_full/hil_reference_predictions.csv",
        ],
        "compile_log": "results/hardware_hil/compile_logs/esp32c3_student_b_compile.txt",
    },
    {
        "id": "student_b_arduino_r4",
        "model": "student_b",
        "board": "arduino_r4",
        "metrics": [
            "results/hardware_hil/board_replay/pi5_arduino_r4_student_b/full_56200_metrics.json",
        ],
        "mcu_csv": [
            "results/hardware_hil/board_replay/pi5_arduino_r4_student_b/full_56200_mcu.csv",
        ],
        "reference_csv": [
            "deployment/firmware_export/wsnds_rfkd_hil/generated_student_b_rfkd_hil_full/hil_reference_predictions.csv",
        ],
        "compile_log": "results/hardware_hil/compile_logs/arduino_r4_student_b_compile.txt",
    },
]


def parse_compile_log(text: str) -> dict[str, int] | None:
    program = re.search(
        r"Sketch uses\s+(\d+)\s+bytes\s+\((\d+)%\).*?Maximum is\s+(\d+)\s+bytes",
        text,
        re.DOTALL,
    )
    global_mem = re.search(
        r"Global variables use\s+(\d+)\s+bytes\s+\((\d+)%\).*?Maximum is\s+(\d+)\s+bytes",
        text,
        re.DOTALL,
    )
    if not program or not global_mem:
        return None
    return {
        "program_bytes": int(program.group(1)),
        "program_percent": int(program.group(2)),
        "program_max_bytes": int(program.group(3)),
        "global_bytes": int(global_mem.group(1)),
        "global_percent": int(global_mem.group(2)),
        "global_max_bytes": int(global_mem.group(3)),
    }


def compute_efficiency(
    *,
    mean_inference_us: float,
    mean_total_us: float,
    clock_mhz: float,
    macs: int,
) -> dict[str, float]:
    inference_cycles = mean_inference_us * clock_mhz
    total_cycles = mean_total_us * clock_mhz
    return {
        "inference_cycles": inference_cycles,
        "total_cycles": total_cycles,
        "cycles_per_mac": inference_cycles / macs,
        "total_throughput_per_s": 1_000_000.0 / mean_total_us,
    }


def first_existing(root: Path, candidates: list[str]) -> Path | None:
    for rel in candidates:
        path = root / rel
        if path.exists():
            return path
    return None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 5) -> str:
    if value is None:
        return "NR"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def compile_info(root: Path, rel_path: str) -> dict[str, int] | None:
    path = root / rel_path
    if not path.exists():
        return None
    return parse_compile_log(path.read_text(encoding="utf-8", errors="replace"))


def compile_status(root: Path, rel_path: str) -> str:
    path = root / rel_path
    if not path.exists():
        return "missing_optional"
    parsed = parse_compile_log(path.read_text(encoding="utf-8", errors="replace"))
    return "present" if parsed else "parse_failed"


def candidate_sources(candidates: list[str]) -> str:
    return " OR ".join(candidates)


def quantization_drift(reference_csv: Path) -> dict[str, Any]:
    by_true: Counter[int] = Counter()
    pairs: Counter[tuple[int, int]] = Counter()
    total = 0
    drift = 0
    with reference_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total += 1
            true_label = int(row["true_label"])
            fixed_pred = int(row["fixed_pred"])
            fp32_pred = int(row["fp32_pred"])
            if fixed_pred != fp32_pred:
                drift += 1
                by_true[true_label] += 1
                pairs[(fp32_pred, fixed_pred)] += 1
    return {
        "total": total,
        "drift_count": drift,
        "drift_fraction": drift / total if total else 0.0,
        "by_true": by_true,
        "pairs": pairs,
    }


def build_analysis(root: Path) -> dict[str, Any]:
    rows = []
    efficiency_rows = []
    compile_rows = []
    evidence_rows = []
    drift_summary_rows = []
    drift_by_class_rows = []
    drift_pair_rows = []
    seen_drift_models: set[str] = set()

    for run in RUNS:
        model = MODEL_SPECS[run["model"]]
        board = BOARD_SPECS[run["board"]]
        compile_data = compile_info(root, run["compile_log"])
        baseline_data = compile_info(root, board["baseline_compile_log"])
        compile_rows.append({
            "model": model["label"],
            "board": board["label"],
            "program_bytes": compile_data["program_bytes"] if compile_data else None,
            "global_bytes": compile_data["global_bytes"] if compile_data else None,
            "serial_baseline_program_bytes": baseline_data["program_bytes"] if baseline_data else None,
            "serial_baseline_global_bytes": baseline_data["global_bytes"] if baseline_data else None,
            "program_delta_vs_serial_baseline": (
                compile_data["program_bytes"] - baseline_data["program_bytes"]
                if compile_data and baseline_data else None
            ),
            "global_delta_vs_serial_baseline": (
                compile_data["global_bytes"] - baseline_data["global_bytes"]
                if compile_data and baseline_data else None
            ),
            "compile_log": run["compile_log"],
            "baseline_log": board["baseline_compile_log"],
        })

        metrics_path = first_existing(root, run["metrics"])
        if metrics_path is None:
            evidence_rows.extend([
                {
                    "claim": f"{model['label']} {board['label']} HIL metrics",
                    "source": candidate_sources(run["metrics"]),
                    "status": "missing_required",
                },
                {
                    "claim": f"{model['label']} {board['label']} raw MCU replay CSV",
                    "source": candidate_sources(run["mcu_csv"]),
                    "status": "present" if first_existing(root, run["mcu_csv"]) else "missing_optional",
                },
                {
                    "claim": f"{model['label']} {board['label']} compile summary",
                    "source": run["compile_log"],
                    "status": compile_status(root, run["compile_log"]),
                },
            ])
        else:
            metrics = read_json(metrics_path)
            latency = metrics["latency"]
            total_mean = float(latency["total_us"]["mean"])
            inference_mean = float(latency["inference_us"]["mean"])
            eff = compute_efficiency(
                mean_inference_us=inference_mean,
                mean_total_us=total_mean,
                clock_mhz=board["clock_mhz"],
                macs=int(model["macs"]),
            )

            rows.append({
                "model": model["label"],
                "board": board["label"],
                "vectors": metrics["completed_vectors"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "mcu_vs_fixed": metrics["mcu_vs_fixed_reference_agreement"],
                "mcu_vs_fp32": metrics["mcu_vs_fp32_agreement"],
                "mean_total_us": total_mean,
                "p99_total_us": latency["total_us"]["p99"],
                "source": str(metrics_path.relative_to(root)),
            })
            efficiency_rows.append({
                "model": model["label"],
                "board": board["label"],
                "clock_mhz": board["clock_mhz"],
                "macs": model["macs"],
                "mean_inference_us": inference_mean,
                "mean_total_us": total_mean,
                "inference_cycles": eff["inference_cycles"],
                "cycles_per_mac": eff["cycles_per_mac"],
                "total_throughput_per_s": eff["total_throughput_per_s"],
            })

            evidence_rows.extend([
                {
                    "claim": f"{model['label']} {board['label']} HIL metrics",
                    "source": str(metrics_path.relative_to(root)),
                    "status": "present",
                },
                {
                    "claim": f"{model['label']} {board['label']} raw MCU replay CSV",
                    "source": candidate_sources(run["mcu_csv"]),
                    "status": "present" if first_existing(root, run["mcu_csv"]) else "missing_optional",
                },
                {
                    "claim": f"{model['label']} {board['label']} compile summary",
                    "source": run["compile_log"],
                    "status": compile_status(root, run["compile_log"]),
                },
            ])

        if run["model"] in seen_drift_models:
            continue
        seen_drift_models.add(run["model"])

        reference_path = first_existing(root, run["reference_csv"])
        if reference_path is None:
            evidence_rows.append({
                "claim": f"{model['label']} FP32-to-fixed drift profile",
                "source": candidate_sources(run["reference_csv"]),
                "status": "missing_required",
            })
            continue

        drift = quantization_drift(reference_path)
        drift_summary_rows.append({
            "model": model["label"],
            "reference_rows": drift["total"],
            "drift_count": drift["drift_count"],
            "drift_fraction": drift["drift_fraction"],
            "reference_csv": str(reference_path.relative_to(root)),
        })
        for label, count in sorted(drift["by_true"].items()):
            drift_by_class_rows.append({
                "model": model["label"],
                "true_label": label,
                "true_class": CLASS_NAMES.get(label, str(label)),
                "drift_count": count,
                "total_reference_rows": drift["total"],
                "drift_fraction_of_all_rows": count / drift["total"] if drift["total"] else 0.0,
                "reference_csv": str(reference_path.relative_to(root)),
            })
        for (fp32_pred, fixed_pred), count in drift["pairs"].most_common():
            drift_pair_rows.append({
                "model": model["label"],
                "fp32_pred": fp32_pred,
                "fp32_class": CLASS_NAMES.get(fp32_pred, str(fp32_pred)),
                "fixed_pred": fixed_pred,
                "fixed_class": CLASS_NAMES.get(fixed_pred, str(fixed_pred)),
                "count": count,
            })
        evidence_rows.append({
            "claim": f"{model['label']} FP32-to-fixed drift profile",
            "source": str(reference_path.relative_to(root)),
            "status": "present",
        })

    for board in BOARD_SPECS.values():
        evidence_rows.append({
            "claim": f"{board['label']} serial baseline compile summary",
            "source": board["baseline_compile_log"],
            "status": compile_status(root, board["baseline_compile_log"]),
        })

    footprint_rows = [
        {
            "model": spec["label"],
            "architecture": spec["architecture"],
            "macs": spec["macs"],
            "weight_bytes": spec["weight_bytes"],
            "bias_bytes": spec["bias_bytes"],
            "param_bytes": spec["param_bytes"],
            "numeric_format": "int8 weights + int32 biases + int16 activations",
        }
        for spec in MODEL_SPECS.values()
    ]

    return {
        "hil_rows": rows,
        "efficiency_rows": efficiency_rows,
        "footprint_rows": footprint_rows,
        "compile_rows": compile_rows,
        "drift_summary_rows": drift_summary_rows,
        "drift_by_class_rows": drift_by_class_rows,
        "drift_pair_rows": drift_pair_rows,
        "evidence_rows": evidence_rows,
    }


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def write_markdown(path: Path, analysis: dict[str, Any]) -> None:
    lines = [
        "# Final HIL Post-Processing Analysis",
        "",
        "Boundary: this report post-processes existing replay logs, reference CSVs, and optional compile logs. It does not run new hardware, measure energy, perform live WSN packet capture, or validate packet-to-feature extraction.",
        "",
        "## HIL Fidelity",
        markdown_table(
            ["Model", "Board", "Vectors", "Accuracy", "Macro-F1", "MCU vs Fixed", "MCU vs FP32", "Mean Total us", "P99 Total us"],
            [
                [
                    row["model"],
                    row["board"],
                    str(row["vectors"]),
                    fmt(row["accuracy"]),
                    fmt(row["macro_f1"]),
                    fmt(row["mcu_vs_fixed"]),
                    fmt(row["mcu_vs_fp32"]),
                    fmt(row["mean_total_us"], 2),
                    fmt(row["p99_total_us"], 0),
                ]
                for row in analysis["hil_rows"]
            ],
        ),
        "",
        "## Cycles Per MAC",
        markdown_table(
            ["Model", "Board", "Clock MHz", "MACs", "Mean Inference us", "Inference Cycles", "Cycles/MAC", "Total Throughput Ceiling/s"],
            [
                [
                    row["model"],
                    row["board"],
                    fmt(row["clock_mhz"], 0),
                    str(row["macs"]),
                    fmt(row["mean_inference_us"], 2),
                    fmt(row["inference_cycles"], 0),
                    fmt(row["cycles_per_mac"], 2),
                    fmt(row["total_throughput_per_s"], 1),
                ]
                for row in analysis["efficiency_rows"]
            ],
        ),
        "",
        "Throughput ceiling is computed from on-device measured total processing time only. It is not a claim about serial, radio, or live network packet throughput.",
        "",
        "## Model-Only Fixed-Point Footprint",
        markdown_table(
            ["Model", "Architecture", "MACs", "Weight Bytes", "Bias Bytes", "Param Bytes", "Format"],
            [
                [
                    row["model"],
                    row["architecture"],
                    str(row["macs"]),
                    str(row["weight_bytes"]),
                    str(row["bias_bytes"]),
                    str(row["param_bytes"]),
                    row["numeric_format"],
                ]
                for row in analysis["footprint_rows"]
            ],
        ),
        "",
        "## Compile And Framework Baseline",
        markdown_table(
            ["Model", "Board", "Program Bytes", "Global Bytes", "Serial Baseline Program", "Serial Baseline Globals", "Program Delta", "Global Delta"],
            [
                [
                    row["model"],
                    row["board"],
                    fmt(row["program_bytes"], 0),
                    fmt(row["global_bytes"], 0),
                    fmt(row["serial_baseline_program_bytes"], 0),
                    fmt(row["serial_baseline_global_bytes"], 0),
                    fmt(row["program_delta_vs_serial_baseline"], 0),
                    fmt(row["global_delta_vs_serial_baseline"], 0),
                ]
                for row in analysis["compile_rows"]
            ],
        ),
        "",
        "If baseline columns show `NR`, compile the serial-baseline sketches and paste the Arduino IDE output into `results/hardware_hil/compile_logs/*_serial_baseline_compile.txt`, then rerun this script.",
        "",
        "## Quantization Drift Summary",
        markdown_table(
            ["Model", "Reference Rows", "Drift Count", "Drift Fraction", "Reference"],
            [
                [
                    row["model"],
                    str(row["reference_rows"]),
                    str(row["drift_count"]),
                    fmt(row["drift_fraction"], 6),
                    row["reference_csv"],
                ]
                for row in analysis["drift_summary_rows"]
            ],
        ),
        "",
        "## Quantization Drift By True Class",
        markdown_table(
            ["Model", "True Class", "Drift Count", "Fraction Of All Rows", "Reference"],
            [
                [
                    row["model"],
                    row["true_class"],
                    str(row["drift_count"]),
                    fmt(row["drift_fraction_of_all_rows"], 6),
                    row["reference_csv"],
                ]
                for row in analysis["drift_by_class_rows"]
            ],
        ),
        "",
        "Drift means `fixed_pred != fp32_pred` in the generated reference CSV. Because MCU-vs-fixed agreement is expected to be 1.00000, this is the fixed-point quantization drift profile, not a serial transport error.",
        "",
        "## Evidence Traceability",
        markdown_table(
            ["Claim", "Source", "Status"],
            [[row["claim"], row["source"], row["status"]] for row in analysis["evidence_rows"]],
        ),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".", help="Repository root.")
    parser.add_argument("--output-dir", default="results/hardware_hil/reports/final_postprocessing")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    output_dir = root / args.output_dir
    analysis = build_analysis(root)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "final_postprocessing_analysis.json").write_text(
        json.dumps(analysis, indent=2),
        encoding="utf-8",
    )
    write_markdown(output_dir / "final_postprocessing_analysis.md", analysis)
    write_csv(output_dir / "hil_fidelity.csv", analysis["hil_rows"], [
        "model", "board", "vectors", "accuracy", "macro_f1", "weighted_f1",
        "mcu_vs_fixed", "mcu_vs_fp32", "mean_total_us", "p99_total_us", "source",
    ])
    write_csv(output_dir / "cycles_per_mac.csv", analysis["efficiency_rows"], [
        "model", "board", "clock_mhz", "macs", "mean_inference_us",
        "mean_total_us", "inference_cycles", "cycles_per_mac",
        "total_throughput_per_s",
    ])
    write_csv(output_dir / "model_only_footprint.csv", analysis["footprint_rows"], [
        "model", "architecture", "macs", "weight_bytes", "bias_bytes",
        "param_bytes", "numeric_format",
    ])
    write_csv(output_dir / "compile_framework_baseline.csv", analysis["compile_rows"], [
        "model", "board", "program_bytes", "global_bytes",
        "serial_baseline_program_bytes", "serial_baseline_global_bytes",
        "program_delta_vs_serial_baseline", "global_delta_vs_serial_baseline",
        "compile_log", "baseline_log",
    ])
    write_csv(output_dir / "quantization_drift_by_true_class.csv", analysis["drift_by_class_rows"], [
        "model", "true_label", "true_class", "drift_count",
        "total_reference_rows", "drift_fraction_of_all_rows", "reference_csv",
    ])
    write_csv(output_dir / "quantization_drift_summary.csv", analysis["drift_summary_rows"], [
        "model", "reference_rows", "drift_count", "drift_fraction", "reference_csv",
    ])
    write_csv(output_dir / "quantization_drift_pairs.csv", analysis["drift_pair_rows"], [
        "model", "fp32_pred", "fp32_class", "fixed_pred", "fixed_class", "count",
    ])
    write_csv(output_dir / "evidence_traceability.csv", analysis["evidence_rows"], [
        "claim", "source", "status",
    ])

    print(output_dir / "final_postprocessing_analysis.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
