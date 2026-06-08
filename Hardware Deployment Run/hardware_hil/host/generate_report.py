"""Generate Markdown and CSV summary tables for HIL results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def render_markdown(board_metrics: list[tuple[str, dict[str, object]]]) -> str:
    lines = [
        "# CuKD-XAI Hardware HIL Summary",
        "",
        "This report covers MCU-class replay of WSN-DS 17-feature records. It does not claim live WSN packet capture, energy measurement, or physical TelosB deployment.",
        "",
        "## Fidelity",
        "",
        "| Board | Vectors | MCU vs Fixed Ref | MCU vs FP32 | Accuracy | Macro-F1 | Weighted-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for board, metrics in board_metrics:
        lines.append(
            "| {board} | {vectors} | {fixed:.6f} | {fp32:.6f} | {acc:.6f} | {macro:.6f} | {weighted:.6f} |".format(
                board=board,
                vectors=int(metrics.get("completed_vectors", 0)),
                fixed=float(metrics.get("mcu_vs_fixed_reference_agreement", 0.0)),
                fp32=float(metrics.get("mcu_vs_fp32_agreement", 0.0)),
                acc=float(metrics.get("accuracy", 0.0)),
                macro=float(metrics.get("macro_f1", 0.0)),
                weighted=float(metrics.get("weighted_f1", 0.0)),
            )
        )
    lines.extend(["", "## Claim Boundary", ""])
    lines.append(
        "The hardware experiments validate firmware-level fixed-point execution on available MCU-class development boards using replayed WSN-DS records. They exclude live packet-to-feature extraction and energy profiling."
    )
    return "\n".join(lines) + "\n"


def write_csv(path: Path, board_metrics: list[tuple[str, dict[str, object]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "board",
                "completed_vectors",
                "mcu_vs_fixed_reference_agreement",
                "mcu_vs_fp32_agreement",
                "accuracy",
                "macro_f1",
                "weighted_f1",
            ],
        )
        writer.writeheader()
        for board, metrics in board_metrics:
            writer.writerow({
                "board": board,
                "completed_vectors": metrics.get("completed_vectors", 0),
                "mcu_vs_fixed_reference_agreement": metrics.get("mcu_vs_fixed_reference_agreement", 0),
                "mcu_vs_fp32_agreement": metrics.get("mcu_vs_fp32_agreement", 0),
                "accuracy": metrics.get("accuracy", 0),
                "macro_f1": metrics.get("macro_f1", 0),
                "weighted_f1": metrics.get("weighted_f1", 0),
            })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", action="append", default=[], help="BOARD=metrics.json")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    board_metrics = []
    for item in args.metric:
        if "=" not in item:
            raise ValueError("--metric must be BOARD=path")
        board, path = item.split("=", 1)
        board_metrics.append((board, load_json(Path(path))))

    md = render_markdown(board_metrics)
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(md, encoding="utf-8")
    write_csv(Path(args.output_csv), board_metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

