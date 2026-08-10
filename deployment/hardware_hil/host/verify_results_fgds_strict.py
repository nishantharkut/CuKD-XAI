"""Fail-closed FG-DS verification of identity, predictions, logits, and latency."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

try:
    from .hil_common import compute_classification_metrics, summarize_latency
    from .stream_vectors_fgds_strict import FULL_TEST_ROWS, verify_bundle, verify_export
except ImportError:
    from hil_common import compute_classification_metrics, summarize_latency
    from stream_vectors_fgds_strict import FULL_TEST_ROWS, verify_bundle, verify_export


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_reference(path: Path) -> dict[int, dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "row_id", "source_row_index", "true_label", "fixed_pred", "fp32_pred",
            *[f"fixed_logit_{index}" for index in range(5)],
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"Reference CSV lacks strict columns: {sorted(missing)}")
        result: dict[int, dict[str, Any]] = {}
        for row in reader:
            row_id = int(row["row_id"])
            if row_id in result:
                raise RuntimeError(f"Duplicate reference row_id: {row_id}")
            result[row_id] = {
                "source_row_index": int(row["source_row_index"]),
                "true_label": int(row["true_label"]),
                "fixed_pred": int(row["fixed_pred"]),
                "fp32_pred": int(row["fp32_pred"]),
                "fixed_logits": [int(row[f"fixed_logit_{index}"]) for index in range(5)],
            }
    return result


def read_replay_source_rows(path: Path) -> dict[int, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"row_id", "source_row_index", *[f"f{index}" for index in range(17)]}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"Replay CSV lacks strict columns: {sorted(missing)}")
        result: dict[int, int] = {}
        for row in reader:
            row_id = int(row["row_id"])
            if row_id in result:
                raise RuntimeError(f"Duplicate replay row_id: {row_id}")
            result[row_id] = int(row["source_row_index"])
        return result


def read_mcu(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "row_id", "status", "predicted_class", "logits",
            "preprocess_us", "inference_us", "total_us",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"MCU CSV lacks columns: {sorted(missing)}")
        return [
            {
                "row_id": int(row["row_id"]),
                "status": row["status"],
                "predicted_class": int(row["predicted_class"]),
                "logits": [int(value) for value in row["logits"].split()],
                "preprocess_us": int(row["preprocess_us"]),
                "inference_us": int(row["inference_us"]),
                "total_us": int(row["total_us"]),
            }
            for row in reader
        ]


def require_manifest_file(
    manifest: dict[str, Any],
    generated_dir: Path,
    file_name: str,
) -> str:
    item = next(
        (entry for entry in manifest.get("files", []) if entry["path"] == file_name),
        None,
    )
    path = generated_dir / file_name
    if item is None or not path.is_file() or sha256_file(path) != item["sha256"]:
        raise RuntimeError(f"Strict export does not bind {path}")
    return item["sha256"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcu-csv", type=Path, required=True)
    parser.add_argument("--sequence-json", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    generated_dir = args.generated_dir.resolve()
    bundle_dir = args.bundle_dir.resolve()
    mcu_csv = args.mcu_csv.resolve()
    sequence_path = args.sequence_json.resolve()
    reference_csv = args.reference_csv.resolve()
    output_path = args.output_json.resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite verification output: {output_path}")
    output_temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    if output_temporary.exists():
        raise FileExistsError(f"Stale verification temporary exists: {output_temporary}")
    if not 1 <= args.expected_count <= FULL_TEST_ROWS:
        raise ValueError(f"--expected-count must be between 1 and {FULL_TEST_ROWS}")
    if len({mcu_csv, sequence_path, reference_csv, output_path}) != 4:
        raise RuntimeError("Verification inputs and output must be distinct files")
    for protected_root in [generated_dir, bundle_dir]:
        try:
            output_path.relative_to(protected_root)
        except ValueError:
            continue
        raise RuntimeError("Verification output cannot be written inside a protected input")

    export_manifest = verify_export(generated_dir)
    bundle_manifest = verify_bundle(bundle_dir, export_manifest)
    reference_hash = require_manifest_file(
        export_manifest, generated_dir, "hil_reference_predictions.csv"
    )
    if reference_csv != (generated_dir / "hil_reference_predictions.csv").resolve():
        raise RuntimeError("Reference CSV must be the file bound by the strict export")

    sequence = read_json(sequence_path)
    expected_identity = (
        f"CUKDBUILD,{export_manifest['student']},{export_manifest['export_id']},"
        f"{bundle_manifest['bundle_id']}"
    )
    expected_provenance = {
        "export_id": export_manifest["export_id"],
        "bundle_id": bundle_manifest["bundle_id"],
        "board": bundle_manifest["board"],
        "student": bundle_manifest["student"],
        "device_identity": expected_identity,
        "vector_sha256": require_manifest_file(
            export_manifest, generated_dir, "hil_replay_vectors.csv"
        ),
        "strict_export_manifest_sha256": sha256_file(
            generated_dir / "strict_export_manifest.json"
        ),
        "strict_bundle_manifest_sha256": sha256_file(
            bundle_dir / "strict_bundle_manifest.json"
        ),
    }
    for key, value in expected_provenance.items():
        if sequence.get("provenance", {}).get(key) != value:
            raise RuntimeError(f"Sequence provenance mismatch for {key}")
    expected_stream_implementation = {
        "stream_script_sha256": sha256_file(
            Path(__file__).with_name("stream_vectors_fgds_strict.py")
        ),
        "protocol_helper_sha256": sha256_file(Path(__file__).with_name("hil_common.py")),
        "vector_loader_sha256": sha256_file(Path(__file__).with_name("stream_vectors.py")),
    }
    for key, value in expected_stream_implementation.items():
        if sequence.get("provenance", {}).get(key) != value:
            raise RuntimeError(f"Sequence implementation provenance mismatch for {key}")
    if sequence.get("output_csv_sha256") != sha256_file(mcu_csv):
        raise RuntimeError("MCU CSV SHA-256 differs from sequence summary")
    if Path(sequence.get("output_csv", "")).resolve() != mcu_csv:
        raise RuntimeError("Sequence summary records a different MCU CSV path")
    if sequence.get("status") != "passed" or sequence.get("error") is not None:
        raise RuntimeError("Sequence summary is not passed")
    if sequence.get("expected") != args.expected_count:
        raise RuntimeError("Sequence expected count differs from declared expected count")
    if sequence.get("completed") != args.expected_count:
        raise RuntimeError("Sequence is incomplete")
    if any(sequence.get(key) for key in ["missing", "duplicates", "unexpected"]):
        raise RuntimeError("Sequence summary contains missing, duplicate, or unexpected rows")
    if sequence.get("status_counts") != {"OK": args.expected_count}:
        raise RuntimeError("Sequence contains non-OK status rows")

    references = read_reference(reference_csv)
    if len(references) != FULL_TEST_ROWS or set(references) != set(range(FULL_TEST_ROWS)):
        raise RuntimeError(
            f"FG-DS reference must contain row IDs 0..{FULL_TEST_ROWS - 1} exactly once"
        )
    replay_source_rows = read_replay_source_rows(
        generated_dir / "hil_replay_vectors.csv"
    )
    if len(replay_source_rows) != FULL_TEST_ROWS or set(replay_source_rows) != set(
        range(FULL_TEST_ROWS)
    ):
        raise RuntimeError(
            f"FG-DS replay must contain row IDs 0..{FULL_TEST_ROWS - 1} exactly once"
        )
    source_values = list(replay_source_rows.values())
    if min(source_values) < 0 or len(set(source_values)) != FULL_TEST_ROWS:
        raise RuntimeError("Strict replay source-row indices are not unique non-negative rows")
    for row_id, reference in references.items():
        if replay_source_rows[row_id] != reference["source_row_index"]:
            raise RuntimeError(f"Replay/reference source-row mismatch at row {row_id}")
    rows = read_mcu(mcu_csv)
    expected_ids = list(range(args.expected_count))
    observed_ids = [row["row_id"] for row in rows]
    if observed_ids != expected_ids:
        raise RuntimeError("MCU rows are not the exact ordered prefix requested")

    prediction_mismatches = []
    logit_mismatches = []
    argmax_mismatches = []
    latency_mismatches = []
    for row in rows:
        reference = references[row["row_id"]]
        if (
            not 0 <= reference["true_label"] < 5
            or not 0 <= reference["fp32_pred"] < 5
            or any(value < -32768 or value > 32767 for value in reference["fixed_logits"])
        ):
            raise RuntimeError(
                f"Reference class/logit is outside its strict range at row {row['row_id']}"
            )
        reference_argmax = max(
            range(5), key=lambda index: reference["fixed_logits"][index]
        )
        if reference["fixed_pred"] != reference_argmax:
            raise RuntimeError(f"Reference prediction/logit mismatch at row {row['row_id']}")
        if row["status"] != "OK" or row["predicted_class"] != reference["fixed_pred"]:
            prediction_mismatches.append(row["row_id"])
        if row["logits"] != reference["fixed_logits"]:
            logit_mismatches.append(row["row_id"])
        if len(row["logits"]) != 5 or row["predicted_class"] != max(
            range(5), key=lambda index: row["logits"][index]
        ):
            argmax_mismatches.append(row["row_id"])
        if (
            row["preprocess_us"] < 0
            or row["inference_us"] < 0
            or row["total_us"] < 0
            or row["total_us"] != row["preprocess_us"] + row["inference_us"]
        ):
            latency_mismatches.append(row["row_id"])
    if prediction_mismatches or logit_mismatches or argmax_mismatches or latency_mismatches:
        raise RuntimeError(
            "Strict HIL mismatch: "
            f"predictions={prediction_mismatches[:5]}, logits={logit_mismatches[:5]}, "
            f"argmax={argmax_mismatches[:5]}, latency={latency_mismatches[:5]}"
        )

    y_true = [references[row_id]["true_label"] for row_id in expected_ids]
    y_mcu = [row["predicted_class"] for row in rows]
    y_fp32 = [references[row_id]["fp32_pred"] for row_id in expected_ids]
    metrics = compute_classification_metrics(y_true, y_mcu, range(5))
    metrics.update({
        "status": "passed",
        "completed_vectors": len(rows),
        "mcu_vs_fixed_reference_agreement": 1.0,
        "mcu_vs_fp32_agreement": (
            sum(int(left == right) for left, right in zip(y_mcu, y_fp32)) / len(rows)
        ),
        "exact_logit_agreement": 1.0,
        "non_ok_status_count": 0,
        "latency_identity_mismatch_count": 0,
        "latency": {
            key: summarize_latency(row[key] for row in rows)
            for key in ["preprocess_us", "inference_us", "total_us"]
        },
        "latency_boundary": (
            "Firmware preprocessing plus inference compute time measured with micros(); "
            "request parsing, response formatting, USB serial, and host overhead are excluded."
        ),
        "provenance": {
            **expected_provenance,
            "mcu_csv_sha256": sha256_file(mcu_csv),
            "sequence_json_sha256": sha256_file(sequence_path),
            "reference_csv_sha256": reference_hash,
            "stream_script_sha256": sequence["provenance"].get("stream_script_sha256"),
            "protocol_helper_sha256": sequence["provenance"].get("protocol_helper_sha256"),
            "vector_loader_sha256": sequence["provenance"].get("vector_loader_sha256"),
            "stream_python": sequence["provenance"].get("python"),
            "pyserial_version": sequence["provenance"].get("pyserial_version"),
            "strict_export_manifest_sha256": sha256_file(
                generated_dir / "strict_export_manifest.json"
            ),
            "strict_bundle_manifest_sha256": sha256_file(
                bundle_dir / "strict_bundle_manifest.json"
            ),
            "verification_script_sha256": sha256_file(Path(__file__).resolve()),
            "metric_helper_sha256": sha256_file(Path(__file__).with_name("hil_common.py")),
            "python": sys.version,
        },
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_temporary.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    os.replace(output_temporary, output_path)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
