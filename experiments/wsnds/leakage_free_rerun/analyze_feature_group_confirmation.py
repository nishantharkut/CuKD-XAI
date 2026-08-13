"""Verify and summarize a completed feature-group WSN-DS confirmation run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import time
from pathlib import Path
from typing import Any


EXPECTED_SEEDS = [42, 123, 456, 789, 1001, 2024, 3141, 5678, 8192, 9999]
EXPECTED_PROTOCOL = "wsnds_feature_group_split_train_only_scaler_10seed_v2"
EXPECTED_ROUTES = [
    "student_A_scratch",
    "student_A_rf_kd",
    "student_B_scratch",
    "student_B_rf_kd",
]
CLASS_NAMES = ["Blackhole", "Flooding", "Grayhole", "Normal", "TDMA"]
FLOAT_TOLERANCE = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def verify_inventory(root: Path, manifest_name: str = "artifact_manifest.json") -> int:
    manifest_path = root / manifest_name
    manifest = read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Manifest is not complete: {manifest_path}")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError(f"Manifest has no file inventory: {manifest_path}")
    if manifest.get("file_count_excluding_manifest") != len(files):
        raise RuntimeError(f"Manifest file count differs from inventory: {manifest_path}")

    seen: set[str] = set()
    root_resolved = root.resolve()
    for item in files:
        if not isinstance(item, dict):
            raise RuntimeError(f"Invalid inventory entry in {manifest_path}")
        relative = item.get("path")
        if not isinstance(relative, str) or not relative:
            raise RuntimeError(f"Invalid inventory path in {manifest_path}")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(f"Inventory path escapes root: {relative!r}")
        normalized = relative_path.as_posix()
        if normalized in seen:
            raise RuntimeError(f"Duplicate inventory path: {relative!r}")
        seen.add(normalized)
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root_resolved)
        except ValueError as exc:
            raise RuntimeError(f"Inventory path escapes root: {relative!r}") from exc
        if not path.is_file():
            raise RuntimeError(f"Inventoried file is missing: {path}")
        if path.stat().st_size != item.get("size_bytes"):
            raise RuntimeError(f"Inventoried file size differs: {path}")
        if sha256_file(path) != item.get("sha256"):
            raise RuntimeError(f"Inventoried file hash differs: {path}")

    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != seen:
        missing = sorted(seen - actual)
        unexpected = sorted(actual - seen)
        raise RuntimeError(
            f"Manifest inventory differs from disk; missing={missing}, unexpected={unexpected}"
        )
    return len(files)


def close_float(observed: float, expected: float) -> bool:
    return math.isclose(observed, expected, rel_tol=0.0, abs_tol=FLOAT_TOLERANCE)


def metrics_from_confusion(confusion: list[list[int]]) -> dict[str, Any]:
    total = sum(sum(row) for row in confusion)
    if total == 0:
        raise RuntimeError("Cannot compute metrics from an empty confusion matrix")
    accuracy = sum(confusion[index][index] for index in range(len(confusion))) / total
    precision: list[float] = []
    recall: list[float] = []
    f1: list[float] = []
    support: list[int] = []
    for index in range(len(confusion)):
        true_positive = confusion[index][index]
        predicted = sum(row[index] for row in confusion)
        actual = sum(confusion[index])
        class_precision = true_positive / predicted if predicted else 0.0
        class_recall = true_positive / actual if actual else 0.0
        class_f1 = (
            2.0 * class_precision * class_recall / (class_precision + class_recall)
            if class_precision + class_recall
            else 0.0
        )
        precision.append(class_precision)
        recall.append(class_recall)
        f1.append(class_f1)
        support.append(actual)
    return {
        "accuracy": accuracy,
        "macro_precision": statistics.fmean(precision),
        "macro_recall": statistics.fmean(recall),
        "macro_f1": statistics.fmean(f1),
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
        "per_class_support": support,
        "confusion_matrix": confusion,
    }


def recompute_prediction_metrics(
    path: Path,
    expected_rows: int,
    canonical_rows: list[tuple[int, int]] | None,
) -> tuple[dict[str, Any], list[tuple[int, int]]]:
    confusion = [[0 for _ in CLASS_NAMES] for _ in CLASS_NAMES]
    observed_rows: list[tuple[int, int]] = []
    seen_indices: set[int] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"source_row_index", "true_label", "predicted_label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError(f"Prediction CSV has an invalid header: {path}")
        for row in reader:
            source_index = int(row["source_row_index"])
            true_label = int(row["true_label"])
            predicted_label = int(row["predicted_label"])
            if source_index in seen_indices:
                raise RuntimeError(f"Duplicate source row index in {path}: {source_index}")
            if not 0 <= true_label < len(CLASS_NAMES):
                raise RuntimeError(f"Invalid true label in {path}: {true_label}")
            if not 0 <= predicted_label < len(CLASS_NAMES):
                raise RuntimeError(f"Invalid predicted label in {path}: {predicted_label}")
            seen_indices.add(source_index)
            observed_rows.append((source_index, true_label))
            confusion[true_label][predicted_label] += 1
    if len(observed_rows) != expected_rows:
        raise RuntimeError(
            f"Prediction row count differs for {path}: {len(observed_rows)} != {expected_rows}"
        )
    if canonical_rows is not None and observed_rows != canonical_rows:
        raise RuntimeError(f"Prediction row order or labels differ: {path}")
    return metrics_from_confusion(confusion), observed_rows


def verify_metrics(observed: dict[str, Any], expected: dict[str, Any], source: str) -> None:
    scalar_keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    vector_keys = [
        "per_class_precision",
        "per_class_recall",
        "per_class_f1",
    ]
    for key in scalar_keys:
        if not close_float(float(observed[key]), float(expected[key])):
            raise RuntimeError(f"Metric differs for {source}/{key}")
    for key in vector_keys:
        if len(observed[key]) != len(expected[key]) or any(
            not close_float(float(left), float(right))
            for left, right in zip(observed[key], expected[key], strict=True)
        ):
            raise RuntimeError(f"Metric vector differs for {source}/{key}")
    if observed["per_class_support"] != expected["per_class_support"]:
        raise RuntimeError(f"Class support differs for {source}")
    if observed["confusion_matrix"] != expected["confusion_matrix"]:
        raise RuntimeError(f"Confusion matrix differs for {source}")


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        raise RuntimeError("Cannot summarize an empty value list")
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def exact_signed_rank(differences: list[float]) -> dict[str, Any]:
    nonzero = [value for value in differences if value != 0.0]
    zeros = len(differences) - len(nonzero)
    if not nonzero:
        return {
            "statistic": 0.0,
            "p_value_two_sided_exact": 1.0,
            "nonzero_pairs": 0,
            "positive_pairs": 0,
            "negative_pairs": 0,
            "zero_pairs": zeros,
            "rank_ties_present": False,
            "enumerated_sign_assignments": 1,
        }
    absolute = [abs(value) for value in nonzero]
    ranks = average_ranks(absolute)
    rank_ties = len(set(absolute)) != len(absolute)
    total_rank = sum(ranks)
    observed_positive = sum(
        rank for rank, difference in zip(ranks, nonzero, strict=True) if difference > 0
    )
    observed_statistic = min(observed_positive, total_rank - observed_positive)
    extreme = 0
    assignments = 1 << len(ranks)
    for mask in range(assignments):
        positive = sum(
            rank for index, rank in enumerate(ranks) if mask & (1 << index)
        )
        statistic = min(positive, total_rank - positive)
        if statistic <= observed_statistic + 1e-15:
            extreme += 1
    return {
        "statistic": observed_statistic,
        "p_value_two_sided_exact": extreme / assignments,
        "nonzero_pairs": len(nonzero),
        "positive_pairs": sum(value > 0 for value in nonzero),
        "negative_pairs": sum(value < 0 for value in nonzero),
        "zero_pairs": zeros,
        "rank_ties_present": rank_ties,
        "enumerated_sign_assignments": assignments,
    }


def apply_holm(tests: dict[str, dict[str, Any]], alpha: float = 0.05) -> None:
    ordered = sorted(
        tests,
        key=lambda name: float(tests[name]["wilcoxon"]["p_value_two_sided_exact"]),
    )
    running_adjusted = 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        raw = float(tests[name]["wilcoxon"]["p_value_two_sided_exact"])
        adjusted = min(1.0, (count - rank) * raw)
        running_adjusted = max(running_adjusted, adjusted)
        tests[name]["holm_adjusted_p"] = running_adjusted
        tests[name]["reject_holm_alpha_0_05"] = running_adjusted <= alpha


def compare_aggregate(
    aggregate: dict[str, Any],
    route_metrics: dict[str, list[dict[str, Any]]],
) -> None:
    if set(aggregate) != set(EXPECTED_ROUTES):
        raise RuntimeError("Aggregate route set differs from the expected four routes")
    for route in EXPECTED_ROUTES:
        for metric in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]:
            values = [float(row[metric]) for row in route_metrics[route]]
            summary = summarize(values)
            recorded = aggregate[route][metric]
            if len(recorded["values"]) != len(values) or any(
                not close_float(float(left), float(right))
                for left, right in zip(recorded["values"], values, strict=True)
            ):
                raise RuntimeError(f"Aggregate values differ for {route}/{metric}")
            for key in ["mean", "sample_std", "min", "max"]:
                if not close_float(float(recorded[key]), float(summary[key])):
                    raise RuntimeError(f"Aggregate summary differs for {route}/{metric}/{key}")


def build_analysis(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir = run_dir.resolve()
    contract = read_json(run_dir / "execution_contract.json")
    preprocessing = read_json(run_dir / "preprocessing_contract.json")
    aggregate = read_json(run_dir / "aggregate_results.json")
    if contract.get("protocol_id") != EXPECTED_PROTOCOL:
        raise RuntimeError("Execution protocol is not the ten-seed feature-group protocol")
    if contract.get("seeds") != EXPECTED_SEEDS:
        raise RuntimeError("Execution seed list differs from the publication seed set")
    if aggregate.get("protocol_id") != EXPECTED_PROTOCOL:
        raise RuntimeError("Aggregate protocol differs from the execution contract")
    if aggregate.get("status") != "complete":
        raise RuntimeError("Aggregate status is not complete")
    if aggregate.get("seeds") != EXPECTED_SEEDS or aggregate.get("seed_count") != 10:
        raise RuntimeError("Aggregate seed inventory is incomplete or out of order")
    if preprocessing.get("protocol_id") != EXPECTED_PROTOCOL:
        raise RuntimeError("Preprocessing protocol differs from the execution contract")
    if preprocessing.get("scaler_fit_partition") != "train only":
        raise RuntimeError("Scaler was not recorded as train-only")

    fingerprint_payload = dict(contract)
    recorded_fingerprint = fingerprint_payload.pop("execution_fingerprint_sha256", None)
    recomputed_fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if recorded_fingerprint != recomputed_fingerprint:
        raise RuntimeError("Execution-contract fingerprint differs from its contents")

    script_dir = Path(__file__).resolve().parent
    runner_path = script_dir / "run_feature_group_10seed_confirmation.py"
    common_path = script_dir / "tier15_common.py"
    dataset_path = Path(str(preprocessing["dataset_path_recorded"]))
    source_checks = {
        "runner": {
            "path_recorded": str(runner_path),
            "sha256": sha256_file(runner_path),
            "expected_sha256": contract["script_sha256"],
        },
        "common_module": {
            "path_recorded": str(common_path),
            "sha256": sha256_file(common_path),
            "expected_sha256": contract["common_module_sha256"],
        },
        "dataset": {
            "path_recorded": str(dataset_path),
            "sha256": sha256_file(dataset_path),
            "expected_sha256": contract["dataset_sha256"],
        },
    }
    for name, check in source_checks.items():
        if check["sha256"] != check["expected_sha256"]:
            raise RuntimeError(f"Current {name} hash differs from the execution contract")
    overlap = preprocessing.get("feature_overlap_audit", {})
    for key in [
        "train_validation_feature_overlap",
        "train_test_feature_overlap",
        "validation_test_feature_overlap",
    ]:
        if overlap.get(key) != 0:
            raise RuntimeError(f"Feature-group overlap is nonzero: {key}")
    expected_rows = int(preprocessing["split_sizes"]["test"])
    if expected_rows != 56301:
        raise RuntimeError(f"Unexpected test split size: {expected_rows}")

    manifest_files = verify_inventory(run_dir)
    if sha256_file(run_dir / "split_indices.npz") != preprocessing.get(
        "split_indices_file_sha256"
    ):
        raise RuntimeError("Split-index file hash differs from preprocessing contract")
    if sha256_file(run_dir / "scaler_parameters.npz") != preprocessing.get(
        "scaler_parameters_file_sha256"
    ):
        raise RuntimeError("Scaler file hash differs from preprocessing contract")

    execution_hash = sha256_file(run_dir / "execution_contract.json")
    route_metrics: dict[str, list[dict[str, Any]]] = {
        route: [] for route in EXPECTED_ROUTES
    }
    teacher_metrics: list[dict[str, Any]] = []
    canonical_rows: list[tuple[int, int]] | None = None
    seed_rows: list[dict[str, Any]] = []

    for seed in EXPECTED_SEEDS:
        seed_root = run_dir / f"seed_{seed}"
        completion = read_json(seed_root / "seed_completion.json")
        if completion.get("status") != "complete" or completion.get("seed") != seed:
            raise RuntimeError(f"Seed completion is invalid: {seed}")
        for key in ["protocol_id", "dataset_sha256", "split_indices_sha256", "scaler_sha256"]:
            expected = contract[key]
            if completion.get(key) != expected:
                raise RuntimeError(f"Seed {seed} contract mismatch: {key}")
        if completion.get("execution_contract_sha256") != execution_hash:
            raise RuntimeError(f"Seed {seed} execution-contract hash differs")
        if set(completion.get("student_results", {})) != set(EXPECTED_ROUTES):
            raise RuntimeError(f"Seed {seed} route inventory differs")
        if (
            completion["student_results"]["student_A_scratch"]["initial_state_sha256"]
            != completion["student_results"]["student_A_rf_kd"]["initial_state_sha256"]
        ):
            raise RuntimeError(f"Student A paired initialization differs for seed {seed}")
        if (
            completion["student_results"]["student_B_scratch"]["initial_state_sha256"]
            != completion["student_results"]["student_B_rf_kd"]["initial_state_sha256"]
        ):
            raise RuntimeError(f"Student B paired initialization differs for seed {seed}")

        teacher_csv = seed_root / "RF_teacher_test_predictions.csv"
        observed_teacher, observed_rows = recompute_prediction_metrics(
            teacher_csv, expected_rows, canonical_rows
        )
        if canonical_rows is None:
            canonical_rows = observed_rows
        verify_metrics(observed_teacher, completion["teacher_metrics"], f"seed {seed}/teacher")
        teacher_metrics.append(observed_teacher)

        row: dict[str, Any] = {
            "seed": seed,
            "teacher_macro_f1": observed_teacher["macro_f1"],
        }
        for route in EXPECTED_ROUTES:
            record = completion["student_results"][route]
            prediction_csv = seed_root / str(record["test_predictions"])
            observed, _ = recompute_prediction_metrics(
                prediction_csv, expected_rows, canonical_rows
            )
            verify_metrics(observed, record["metrics"], f"seed {seed}/{route}")
            route_metrics[route].append(observed)
            row[f"{route}_macro_f1"] = observed["macro_f1"]
        row["student_A_rf_kd_minus_scratch_macro_f1"] = (
            row["student_A_rf_kd_macro_f1"] - row["student_A_scratch_macro_f1"]
        )
        row["student_B_rf_kd_minus_scratch_macro_f1"] = (
            row["student_B_rf_kd_macro_f1"] - row["student_B_scratch_macro_f1"]
        )
        seed_rows.append(row)

    compare_aggregate(aggregate["aggregate"], route_metrics)
    teacher_macro_f1 = [float(row["macro_f1"]) for row in teacher_metrics]
    teacher_summary = summarize(teacher_macro_f1)
    route_summaries: dict[str, Any] = {}
    for route in EXPECTED_ROUTES:
        macro_f1 = [float(row["macro_f1"]) for row in route_metrics[route]]
        summary = summarize(macro_f1)
        paired_gaps = [
            teacher - student
            for teacher, student in zip(teacher_macro_f1, macro_f1, strict=True)
        ]
        summary["teacher_minus_route_macro_f1"] = summarize(paired_gaps)
        summary["mean_macro_f1_retention_fraction"] = (
            summary["mean"] / teacher_summary["mean"]
        )
        per_class = []
        for class_index, class_name in enumerate(CLASS_NAMES):
            per_class.append(
                {
                    "class_index": class_index,
                    "class_name": class_name,
                    **summarize(
                        [
                            float(metrics["per_class_f1"][class_index])
                            for metrics in route_metrics[route]
                        ]
                    ),
                }
            )
        summary["per_class_f1"] = per_class
        route_summaries[route] = summary

    paired_tests: dict[str, dict[str, Any]] = {}
    for student in ["student_A", "student_B"]:
        scratch = [
            float(row["macro_f1"]) for row in route_metrics[f"{student}_scratch"]
        ]
        rf_kd = [
            float(row["macro_f1"]) for row in route_metrics[f"{student}_rf_kd"]
        ]
        differences = [right - left for left, right in zip(scratch, rf_kd, strict=True)]
        paired_tests[student] = {
            "comparison": "rf_kd_minus_scratch_macro_f1",
            "seeds": EXPECTED_SEEDS,
            "differences": differences,
            "difference_summary": summarize(differences),
            "wilcoxon": exact_signed_rank(differences),
        }
    apply_holm(paired_tests)

    result = {
        "status": "passed",
        "protocol_id": EXPECTED_PROTOCOL,
        "run_dir_recorded": str(run_dir),
        "run_manifest_sha256": sha256_file(run_dir / "artifact_manifest.json"),
        "verified_manifest_files": manifest_files,
        "prediction_csv_files_recomputed": 50,
        "prediction_rows_recomputed": 50 * expected_rows,
        "test_rows_per_model_seed": expected_rows,
        "seeds": EXPECTED_SEEDS,
        "seed_count": len(EXPECTED_SEEDS),
        "contracts": {
            "execution_contract_sha256": execution_hash,
            "execution_fingerprint_sha256": recomputed_fingerprint,
            "dataset_sha256": contract["dataset_sha256"],
            "split_indices_sha256": contract["split_indices_sha256"],
            "scaler_sha256": contract["scaler_sha256"],
            "scaler_fit_partition": preprocessing["scaler_fit_partition"],
            "feature_overlap_audit": overlap,
            "kd_hyperparameters": contract["kd_hyperparameters"],
            "kd_hyperparameter_source": contract["kd_hyperparameter_source"],
            "teacher_calibration_strategy": contract["teacher_calibration_strategy"],
            "environment": contract["environment"],
        },
        "source_checks": source_checks,
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "teacher": {"macro_f1": teacher_summary},
        "routes": route_summaries,
        "paired_tests": paired_tests,
        "statistical_unit": (
            "Ten paired optimizer seeds on one fixed feature-group-disjoint split; "
            "the analysis does not estimate variation across independently sampled splits."
        ),
        "test_definition": (
            "Two-sided paired Wilcoxon signed-rank permutation test obtained by "
            "enumerating all sign assignments after removing zero differences; "
            "Holm correction is applied across the Student A and Student B tests."
        ),
    }
    return result, seed_rows


def write_outputs(output_dir: Path, result: dict[str, Any], seed_rows: list[dict[str, Any]]) -> None:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite analysis directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.tmp.{os.getpid()}.{time.time_ns()}"
    staging.mkdir()
    try:
        json_path = staging / "feature_group_10seed_analysis.json"
        csv_path = staging / "feature_group_10seed_seed_table.csv"
        markdown_path = staging / "feature_group_10seed_summary.md"
        atomic_write_json(json_path, result)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0]))
            writer.writeheader()
            writer.writerows(seed_rows)

        lines = [
            "# WSN-DS Feature-Group 10-Seed Confirmation",
            "",
            "All values below were recomputed from the manifest-bound prediction CSV files.",
            "",
            "| Route | Macro-F1 mean | Sample SD | Teacher gap | Retention |",
            "|---|---:|---:|---:|---:|",
        ]
        teacher_mean = result["teacher"]["macro_f1"]["mean"]
        teacher_sd = result["teacher"]["macro_f1"]["sample_std"]
        lines.append(f"| Calibrated RF teacher | {teacher_mean:.6f} | {teacher_sd:.6f} | 0.000000 | 1.000000 |")
        labels = {
            "student_A_scratch": "Student A scratch",
            "student_A_rf_kd": "Student A RF-KD",
            "student_B_scratch": "Student B scratch",
            "student_B_rf_kd": "Student B RF-KD",
        }
        for route in EXPECTED_ROUTES:
            summary = result["routes"][route]
            lines.append(
                f"| {labels[route]} | {summary['mean']:.6f} | "
                f"{summary['sample_std']:.6f} | "
                f"{summary['teacher_minus_route_macro_f1']['mean']:.6f} | "
                f"{summary['mean_macro_f1_retention_fraction']:.6f} |"
            )
        lines.extend(
            [
                "",
                "## Paired RF-KD versus scratch",
                "",
                "| Student | Mean difference | Exact p | Holm p | Reject at 0.05 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for student, label in [("student_A", "Student A"), ("student_B", "Student B")]:
            test = result["paired_tests"][student]
            lines.append(
                f"| {label} | {test['difference_summary']['mean']:.6f} | "
                f"{test['wilcoxon']['p_value_two_sided_exact']:.6f} | "
                f"{test['holm_adjusted_p']:.6f} | "
                f"{str(test['reject_holm_alpha_0_05']).lower()} |"
            )
        lines.extend(
            [
                "",
                "## Evaluation boundary",
                "",
                result["statistical_unit"],
                "",
                result["test_definition"],
                "",
            ]
        )
        markdown_path.write_text("\n".join(lines), encoding="utf-8")

        files = []
        for path in [json_path, csv_path, markdown_path]:
            files.append(
                {
                    "path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        atomic_write_json(
            staging / "analysis_manifest.json",
            {
                "status": "passed",
                "protocol_id": EXPECTED_PROTOCOL,
                "input_run_manifest_sha256": result["run_manifest_sha256"],
                "analysis_script_sha256": result["analysis_script_sha256"],
                "file_count_excluding_manifest": len(files),
                "files": files,
            },
        )
        os.replace(staging, output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result, seed_rows = build_analysis(args.run_dir)
    write_outputs(args.output_dir, result, seed_rows)
    print(args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
