"""Verify MCU replay logs against fixed-point and FP32 references."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

DEPLOYMENT_ROOT = Path(__file__).resolve().parents[2]
if str(DEPLOYMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_ROOT))

from hardware_hil.host.hil_common import compute_classification_metrics, summarize_latency


def _read_reference(path: Path) -> dict[int, dict[str, int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"row_id", "true_label", "fixed_pred", "fp32_pred"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"reference missing columns: {sorted(missing)}")
        return {
            int(row["row_id"]): {
                "true_label": int(row["true_label"]),
                "fixed_pred": int(row["fixed_pred"]),
                "fp32_pred": int(row["fp32_pred"]),
            }
            for row in reader
        }


def _read_mcu(path: Path) -> list[dict[str, int | str]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({
                "row_id": int(row["row_id"]),
                "status": row["status"],
                "predicted_class": int(row["predicted_class"]),
                "preprocess_us": int(row["preprocess_us"]),
                "inference_us": int(row["inference_us"]),
                "total_us": int(row["total_us"]),
            })
    return rows


def verify(mcu_csv: Path, reference_csv: Path, labels: list[int]) -> dict[str, object]:
    references = _read_reference(reference_csv)
    mcu_rows = _read_mcu(mcu_csv)
    matched = [row for row in mcu_rows if int(row["row_id"]) in references]
    y_true = [references[int(row["row_id"])]["true_label"] for row in matched]
    y_mcu = [int(row["predicted_class"]) for row in matched]
    y_fixed = [references[int(row["row_id"])]["fixed_pred"] for row in matched]
    y_fp32 = [references[int(row["row_id"])]["fp32_pred"] for row in matched]

    fixed_matches = sum(1 for a, b in zip(y_mcu, y_fixed) if a == b)
    fp32_matches = sum(1 for a, b in zip(y_mcu, y_fp32) if a == b)

    result = compute_classification_metrics(y_true=y_true, y_pred=y_mcu, labels=labels)
    result["completed_vectors"] = len(matched)
    result["mcu_vs_fixed_reference_agreement"] = fixed_matches / len(matched) if matched else 0.0
    result["mcu_vs_fp32_agreement"] = fp32_matches / len(matched) if matched else 0.0
    result["latency"] = {
        "preprocess_us": summarize_latency(int(row["preprocess_us"]) for row in matched),
        "inference_us": summarize_latency(int(row["inference_us"]) for row in matched),
        "total_us": summarize_latency(int(row["total_us"]) for row in matched),
    }
    result["non_ok_status_count"] = sum(1 for row in mcu_rows if row["status"] != "OK")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcu-csv", required=True)
    parser.add_argument("--reference-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--labels", default="0,1,2,3,4")
    args = parser.parse_args()

    labels = [int(v) for v in args.labels.split(",") if v.strip()]
    metrics = verify(Path(args.mcu_csv), Path(args.reference_csv), labels)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

