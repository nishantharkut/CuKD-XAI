"""Derive compact final-HIL tables from a verified portable campaign archive."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping, Sequence

from deployment.hardware_hil.host.hil_common import (
    compute_classification_metrics,
    summarize_latency,
)
from deployment.hardware_hil.host.record_fgds_compile_evidence import (
    FLASH_PATTERN,
    RAM_PATTERN,
    parsed_match,
    validate_footprint,
)

from .archive import ARCHIVE_MANIFEST, verify_campaign_archive
from .contracts import (
    EXPECTED_FULL_ROWS,
    canonical_json_sha256,
    read_json,
    sha256_file,
)


REPORT_SCHEMA = "cukd_final_hil_archive_report_v1"
BOARD_LABELS = {
    "esp32c3": "ESP32-C3",
    "arduino_r4": "Arduino UNO R4 WiFi",
}
MODEL_LABELS = {
    "student_A_scratch": "Student A scratch",
    "student_A_rf_kd": "Student A RF-KD",
    "student_B_scratch": "Student B scratch",
    "student_B_rf_kd": "Student B RF-KD",
}
MODEL_MACROS = (
    "CUKD_INPUT_DIM",
    "CUKD_H1_DIM",
    "CUKD_H2_DIM",
    "CUKD_OUTPUT_DIM",
    "CUKD_WEIGHT_BYTES",
    "CUKD_BIAS_BYTES",
    "CUKD_PARAM_BYTES",
    "CUKD_ACTIVATION_BYTES_EST",
    "CUKD_MACS_PER_INFERENCE",
)
RESPONSE_FIELDS = {
    "row_id",
    "status",
    "predicted_class",
    "fixed_logit_0",
    "fixed_logit_1",
    "fixed_logit_2",
    "fixed_logit_3",
    "fixed_logit_4",
    "preprocess_us",
    "inference_us",
    "total_us",
    "attempts",
    "response_timeout_count",
    "host_observed_rtt_us",
    "transaction_elapsed_us",
}
REFERENCE_FIELDS = {
    "row_id",
    "true_label",
    "fixed_pred",
    "fp32_pred",
    "fixed_logit_0",
    "fixed_logit_1",
    "fixed_logit_2",
    "fixed_logit_3",
    "fixed_logit_4",
}
DEVICE_TIMING_METRICS = ("preprocess_us", "inference_us", "total_us")
HOST_TIMING_METRICS = ("host_observed_rtt_us", "transaction_elapsed_us")
TIMING_METRICS = DEVICE_TIMING_METRICS + HOST_TIMING_METRICS


def _safe_local(root: Path, relative: str) -> Path:
    path = Path(relative)
    if (
        not relative
        or "\\" in relative
        or path.as_posix() != relative
        or path.is_absolute()
        or path.drive
        or ".." in path.parts
    ):
        raise RuntimeError(f"Unsafe archive-relative path: {relative!r}")
    resolved_root = root.resolve()
    resolved = (resolved_root / path).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError(f"Archive path escapes its root: {relative!r}") from exc
    if not resolved.is_file() and not resolved.is_dir():
        raise FileNotFoundError(resolved)
    return resolved


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = required - fields
        if missing:
            raise RuntimeError(f"CSV lacks required fields {sorted(missing)}: {path}")
        return list(reader)


def _integer(row: Mapping[str, str], field: str) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid integer field {field!r}: {row.get(field)!r}") from exc


def _require_dense_rows(rows: Sequence[Mapping[str, str]], expected: int) -> None:
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} rows, found {len(rows)}")
    observed = [_integer(row, "row_id") for row in rows]
    if observed != list(range(expected)):
        raise RuntimeError("Rows are not the exact dense sequence 0..N-1")


def _validate_response_rows(
    rows: Sequence[Mapping[str, str]], expected: int
) -> None:
    _require_dense_rows(rows, expected)
    ignored_fields = sorted(
        field for field in rows[0] if field.startswith("ignored_")
    ) if rows else []
    for row in rows:
        if row.get("status") != "OK":
            raise RuntimeError("Report input contains a non-OK device response")
        if _integer(row, "attempts") != 1:
            raise RuntimeError("Report input contains a retried device response")
        if _integer(row, "response_timeout_count") != 0:
            raise RuntimeError("Report input contains a response timeout")
        if any(_integer(row, field) != 0 for field in ignored_fields):
            raise RuntimeError("Report input contains a nonzero ignored-response counter")
        preprocess = _integer(row, "preprocess_us")
        inference = _integer(row, "inference_us")
        total = _integer(row, "total_us")
        if min(preprocess, inference, total) < 0 or preprocess + inference != total:
            raise RuntimeError("Device timing arithmetic is invalid")
        for field in HOST_TIMING_METRICS:
            if field in row and _integer(row, field) < 0:
                raise RuntimeError(f"Host timing field is negative: {field}")


def _model_macros(header: Path) -> dict[str, Any]:
    text = header.read_text(encoding="ascii")
    values: dict[str, int] = {}
    for name in MODEL_MACROS:
        matches = re.findall(
            rf"^#define\s+{re.escape(name)}\s+(\d+)\s*$",
            text,
            re.MULTILINE,
        )
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one {name} macro in {header}")
        values[name] = int(matches[0])
    if values["CUKD_PARAM_BYTES"] != (
        values["CUKD_WEIGHT_BYTES"] + values["CUKD_BIAS_BYTES"]
    ):
        raise RuntimeError("Model parameter-byte macros are internally inconsistent")
    return {
        "architecture": (
            f"{values['CUKD_INPUT_DIM']}-{values['CUKD_H1_DIM']}-"
            f"{values['CUKD_H2_DIM']}-{values['CUKD_OUTPUT_DIM']}"
        ),
        "weight_bytes": values["CUKD_WEIGHT_BYTES"],
        "bias_bytes": values["CUKD_BIAS_BYTES"],
        "parameter_bytes": values["CUKD_PARAM_BYTES"],
        "activation_bytes_estimate": values["CUKD_ACTIVATION_BYTES_EST"],
        "macs_per_inference": values["CUKD_MACS_PER_INFERENCE"],
    }


def _compile_footprint(compile_log: Path) -> dict[str, int]:
    text = compile_log.read_text(encoding="utf-8", errors="replace")
    flash = parsed_match(FLASH_PATTERN, text, "flash")
    ram = parsed_match(RAM_PATTERN, text, "RAM")
    validate_footprint(flash, ram)
    return {
        "program_bytes": flash["used"],
        "program_max_bytes": flash["maximum"],
        "program_percent_reported": flash["percent"],
        "global_bytes": ram["used"],
        "global_max_bytes": ram["maximum"],
        "global_percent_reported": ram["percent"],
        "remaining_dynamic_bytes": ram["remaining"],
    }


def _repeat_summary(repeats: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(repeats) != 3:
        raise RuntimeError("Final timing evidence must contain exactly three repeats")
    result: dict[str, Any] = {"repeat_count": len(repeats)}
    for metric in TIMING_METRICS:
        repeat_means = [float(item[metric]["mean"]) for item in repeats]
        all_values = [
            int(value)
            for item in repeats
            for value in item[f"_{metric}_values"]
        ]
        result[metric] = {
            "mean_of_repeat_means": mean(repeat_means),
            "sample_sd_of_repeat_means": stdev(repeat_means),
            "repeat_means": repeat_means,
            "row_level_descriptive_3000": summarize_latency(all_values),
        }
    return result


def _assert_close(observed: float, expected: float, label: str) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"{label} differs: {observed!r} != {expected!r}")


def _full_replay_metrics(
    response_path: Path,
    reference_path: Path,
    export_report: Mapping[str, Any],
) -> dict[str, Any]:
    responses = _read_csv(response_path, RESPONSE_FIELDS)
    references = _read_csv(reference_path, REFERENCE_FIELDS)
    _validate_response_rows(responses, EXPECTED_FULL_ROWS)
    _require_dense_rows(references, EXPECTED_FULL_ROWS)

    true_labels: list[int] = []
    mcu_predictions: list[int] = []
    fp32_predictions: list[int] = []
    exact_predictions = 0
    exact_logits = 0
    for response, reference in zip(responses, references):
        prediction = _integer(response, "predicted_class")
        fixed_prediction = _integer(reference, "fixed_pred")
        true_label = _integer(reference, "true_label")
        fp32_prediction = _integer(reference, "fp32_pred")
        response_logits = [
            _integer(response, f"fixed_logit_{index}") for index in range(5)
        ]
        reference_logits = [
            _integer(reference, f"fixed_logit_{index}") for index in range(5)
        ]
        if any(
            value not in range(5)
            for value in (prediction, fixed_prediction, true_label, fp32_prediction)
        ):
            raise RuntimeError("Report input contains a class outside 0..4")
        if any(value < -32768 or value > 32767 for value in response_logits + reference_logits):
            raise RuntimeError("Report input contains a fixed logit outside int16")
        reference_argmax = max(range(5), key=reference_logits.__getitem__)
        if fixed_prediction != reference_argmax or prediction != max(
            range(5), key=response_logits.__getitem__
        ):
            raise RuntimeError("A reported prediction differs from fixed-logit argmax")
        true_labels.append(true_label)
        fp32_predictions.append(fp32_prediction)
        mcu_predictions.append(prediction)
        if prediction == fixed_prediction:
            exact_predictions += 1
        if response_logits == reference_logits:
            exact_logits += 1

    if exact_predictions != EXPECTED_FULL_ROWS or exact_logits != EXPECTED_FULL_ROWS:
        raise RuntimeError("MCU output is not exactly equal to the fixed reference")
    metrics = compute_classification_metrics(true_labels, mcu_predictions, range(5))
    fixed_report = export_report["fixed_metrics"]
    _assert_close(float(metrics["accuracy"]), float(fixed_report["accuracy"]), "accuracy")
    _assert_close(float(metrics["macro_f1"]), float(fixed_report["macro_f1"]), "macro-F1")
    if metrics["confusion_matrix"] != fixed_report["confusion_matrix"]:
        raise RuntimeError("Recomputed fixed confusion matrix differs from export")
    for label in range(5):
        derived = metrics["per_class"][str(label)]
        if derived["support"] != fixed_report["per_class_support"][label]:
            raise RuntimeError(f"Recomputed fixed support differs for class {label}")
        for metric, report_key in (
            ("precision", "per_class_precision"),
            ("recall", "per_class_recall"),
            ("f1", "per_class_f1"),
        ):
            _assert_close(
                float(derived[metric]),
                float(fixed_report[report_key][label]),
                f"class-{label} {metric}",
            )
    mcu_vs_fp32 = sum(
        observed == expected
        for observed, expected in zip(mcu_predictions, fp32_predictions)
    ) / EXPECTED_FULL_ROWS
    _assert_close(
        mcu_vs_fp32,
        float(export_report["gates"]["fixed_vs_fp32_agreement"]),
        "MCU/FP32 agreement",
    )
    return {
        "rows": EXPECTED_FULL_ROWS,
        **metrics,
        "mcu_vs_fixed_reference_agreement": exact_predictions / EXPECTED_FULL_ROWS,
        "mcu_fixed_logits_exact_fraction": exact_logits / EXPECTED_FULL_ROWS,
        "mcu_vs_fp32_agreement": mcu_vs_fp32,
    }


def _duration_seconds(started: str, finished: str) -> float:
    start = datetime.fromisoformat(started)
    end = datetime.fromisoformat(finished)
    value = (end - start).total_seconds()
    if value < 0:
        raise RuntimeError("Session duration is negative")
    return value


def _blocked_results(
    root: Path,
    semantic: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_model = {item["model_key"]: item for item in evidence["blocked_routes"]}
    results: list[dict[str, Any]] = []
    for model_key, item in semantic["models"].items():
        if item.get("status") != "blocked":
            continue
        audit = read_json(_safe_local(root, item["blocked_audit_local_path"]))
        results.append(
            {
                "model_key": model_key,
                "model": MODEL_LABELS[model_key],
                "status": "blocked_before_firmware_generation",
                "reason": by_model[model_key]["reason"],
                "fixed_vs_fp32_agreement": audit["fixed_vs_fp32_agreement"],
                "absolute_macro_f1_drop": audit["absolute_macro_f1_drop"],
                "minimum_fixed_vs_fp32_agreement": audit[
                    "minimum_fixed_vs_fp32_agreement"
                ],
                "maximum_absolute_macro_f1_drop": audit[
                    "maximum_absolute_macro_f1_drop"
                ],
                "excluded_combinations": [
                    excluded["combination_id"]
                    for excluded in evidence["excluded_combinations"]
                    if excluded["model_key"] == model_key
                ],
            }
        )
    return results


def build_report(archive_dir: Path) -> dict[str, Any]:
    root = Path(archive_dir).resolve()
    verification = verify_campaign_archive(root)
    manifest = read_json(root / ARCHIVE_MANIFEST)
    semantic = manifest["semantic_map"]
    campaign_map = semantic["campaign"]
    contract = read_json(_safe_local(root, campaign_map["contract_local_path"]))
    evidence = read_json(_safe_local(root, campaign_map["evidence_local_path"]))

    model_results: dict[str, dict[str, Any]] = {}
    sessions: list[dict[str, Any]] = []
    timing_repeat_rows: list[dict[str, Any]] = []
    combinations = sorted(
        semantic["combinations"].items(),
        key=lambda item: item[1]["execution_ordinal"],
    )
    stage_contract = {item["name"]: item for item in contract["stages"]}
    timing_stage_names = [
        item["name"] for item in contract["stages"] if item["include_in_timing_metrics"]
    ]
    fidelity_stage_names = [
        item["name"] for item in contract["stages"] if item["include_in_fidelity_metrics"]
    ]
    if len(timing_stage_names) != 3 or len(fidelity_stage_names) != 1:
        raise RuntimeError(
            "Final campaign must define three timing stages and one fidelity stage"
        )
    timing_row_counts = {stage_contract[name]["rows"] for name in timing_stage_names}
    if len(timing_row_counts) != 1:
        raise RuntimeError("Final timing stages do not have one common row count")
    timing_rows_per_repeat = next(iter(timing_row_counts))
    fidelity_stage_name = fidelity_stage_names[0]
    if stage_contract[fidelity_stage_name]["rows"] != EXPECTED_FULL_ROWS:
        raise RuntimeError("Final fidelity stage row count differs from the export contract")
    for combination_id, combination in combinations:
        model_key = combination["model_key"]
        export_dir = _safe_local(
            root, semantic["models"][model_key]["export_local_dir"]
        )
        report = read_json(export_dir / "final_export_report.json")
        session = read_json(_safe_local(root, combination["session_local_path"]))
        if session["combination_id"] != combination_id:
            raise RuntimeError("Session identity differs from archive semantic map")

        timing_repeats: list[dict[str, Any]] = []
        full_path: Path | None = None
        for attempt in sorted(combination["attempts"], key=lambda item: item["ordinal"]):
            attempt_dir = _safe_local(root, attempt["local_dir"])
            response_path = attempt_dir / "responses.csv"
            if attempt["stage_name"] in timing_stage_names:
                rows = _read_csv(response_path, RESPONSE_FIELDS)
                _validate_response_rows(rows, timing_rows_per_repeat)
                repeat: dict[str, Any] = {
                    "stage": attempt["stage_name"],
                    "repeat": len(timing_repeats) + 1,
                }
                for metric in TIMING_METRICS:
                    values = [_integer(row, metric) for row in rows]
                    repeat[metric] = summarize_latency(values)
                    repeat[f"_{metric}_values"] = values
                timing_repeats.append(repeat)
            elif attempt["stage_name"] == fidelity_stage_name:
                full_path = response_path
        if full_path is None:
            raise RuntimeError(f"Full replay is absent: {combination_id}")

        fidelity = _full_replay_metrics(
            full_path,
            export_dir / "hil_reference_predictions.csv",
            report,
        )
        timing = _repeat_summary(timing_repeats)
        provenance_dir = _safe_local(root, combination["provenance"]["local_dir"])
        provenance = read_json(provenance_dir / "build_upload_provenance.json")
        compile_log_relative = provenance["commands"]["compile"]["log"]["path"]
        footprint = _compile_footprint(
            _safe_local(provenance_dir, compile_log_relative)
        )
        build_contract = provenance["build_contract"]
        model_footprint = _model_macros(export_dir / "model_weights.h")

        if model_key not in model_results:
            model_results[model_key] = {
                "model": MODEL_LABELS[model_key],
                "student": session["student"],
                "route": session["route"],
                "seed": report["identity"]["seed"],
                "export_id": session["export_id"],
                "model_footprint": model_footprint,
                "fp32_metrics": report["fp32_metrics"],
                "fixed_metrics": report["fixed_metrics"],
                "fixed_point_quality_gates": {
                    key: report["gates"][key]
                    for key in (
                        "fixed_vs_fp32_agreement",
                        "absolute_macro_f1_drop",
                        "minimum_fixed_vs_fp32_agreement",
                        "maximum_absolute_macro_f1_drop",
                        "zero_saturation_passed",
                        "accumulator_bounds_passed",
                        "preprocess_bounds_passed",
                    )
                },
                "signed_fixed_minus_fp32_macro_f1": (
                    float(report["fixed_metrics"]["macro_f1"])
                    - float(report["fp32_metrics"]["macro_f1"])
                ),
            }
        elif model_results[model_key]["export_id"] != session["export_id"]:
            raise RuntimeError("Boards did not execute the same model export")

        session_row = {
            "execution_ordinal": combination["execution_ordinal"],
            "combination_id": combination_id,
            "model_key": model_key,
            "model": MODEL_LABELS[model_key],
            "student": session["student"],
            "route": session["route"],
            "board": session["board"],
            "board_label": BOARD_LABELS[session["board"]],
            "transport": session["transport"],
            "physical_port_serial": session["physical_port_serial"],
            "session_evidence_id": session["session_evidence_id"],
            "campaign_session_id": session["campaign_session_id"],
            "export_id": session["export_id"],
            "bundle_id": session["bundle_id"],
            "provenance_id": session["provenance_id"],
            "duration_seconds": _duration_seconds(
                session["started_utc"], session["finished_utc"]
            ),
            "fidelity": fidelity,
            "timing": timing,
            "compile_footprint": footprint,
            "build_environment": {
                "fqbn": build_contract["fqbn"],
                "board_core_version": build_contract["board_core_version"],
                "arduino_cli_version": build_contract["frontend_version"],
                "toolchain_version": build_contract["toolchain_version"],
                "build_contract_id": build_contract["build_contract_id"],
            },
        }
        sessions.append(session_row)
        for repeat in timing_repeats:
            timing_repeat_rows.append(
                {
                    "combination_id": combination_id,
                    "model": MODEL_LABELS[model_key],
                    "board": BOARD_LABELS[session["board"]],
                    "repeat": repeat["repeat"],
                    **{
                        f"{metric}_{stat}": repeat[metric][stat]
                        for metric in TIMING_METRICS
                        for stat in ("mean", "median", "p95", "p99", "min", "max")
                    },
                }
            )

    payload: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": verification["status"],
        "archive_verification": verification,
        "contract_id": contract["contract_id"],
        "campaign_evidence_id": evidence["campaign_evidence_id"],
        "archive_id": manifest["archive_id"],
        "campaign_scope": {
            "seed": 42,
            "boards": evidence["boards"],
            "transports": evidence["transports"],
            "eligible_session_count": len(sessions),
            "full_replay_rows_per_session": EXPECTED_FULL_ROWS,
            "timing_repeats_per_session": 3,
            "timing_rows_per_repeat": timing_rows_per_repeat,
            "timing_statistical_unit": (
                "three ordered repeat summaries on one physical board specimen; "
                "the 3,000 rows are not independent hardware replications"
            ),
        },
        "totals": evidence["totals"],
        "physical_specimens": evidence["physical_specimens"],
        "blocked_routes": _blocked_results(root, semantic, evidence),
        "model_results": model_results,
        "sessions": sessions,
        "timing_repeat_rows": timing_repeat_rows,
        "claim_boundary": evidence["claim_boundary"],
    }
    payload["report_id"] = canonical_json_sha256(payload)
    return payload


def _fmt(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Final FGDS Seed-42 Hardware HIL Results",
        "",
        "This report is derived from a deeply verified portable evidence archive. "
        "It reports USB serial replay only.",
        "",
        "## Fidelity and task metrics",
        "",
        "| Model | Board | Rows | Accuracy | Macro-F1 | Weighted-F1 | MCU vs fixed | MCU vs FP32 | Exact logits |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for session in report["sessions"]:
        fidelity = session["fidelity"]
        lines.append(
            "| {model} | {board} | {rows} | {accuracy} | {macro} | {weighted} | "
            "{fixed} | {fp32} | {logits} |".format(
                model=session["model"],
                board=session["board_label"],
                rows=fidelity["rows"],
                accuracy=_fmt(fidelity["accuracy"]),
                macro=_fmt(fidelity["macro_f1"]),
                weighted=_fmt(fidelity["weighted_f1"]),
                fixed=_fmt(fidelity["mcu_vs_fixed_reference_agreement"]),
                fp32=_fmt(fidelity["mcu_vs_fp32_agreement"]),
                logits=_fmt(fidelity["mcu_fixed_logits_exact_fraction"]),
            )
        )

    lines.extend(
        [
            "",
            "## Fixed-point quality by model",
            "",
            "The signed delta is fixed-point macro-F1 minus FP32 macro-F1. "
            "The gate uses the absolute delta.",
            "",
            "| Model | FP32 macro-F1 | Fixed macro-F1 | Signed delta | Absolute gate delta | Fixed vs FP32 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model in report["model_results"].values():
        gate = model["fixed_point_quality_gates"]
        lines.append(
            "| {model} | {fp32} | {fixed} | {signed} | {absolute} | {agreement} |".format(
                model=model["model"],
                fp32=_fmt(model["fp32_metrics"]["macro_f1"]),
                fixed=_fmt(model["fixed_metrics"]["macro_f1"]),
                signed=_fmt(model["signed_fixed_minus_fp32_macro_f1"]),
                absolute=_fmt(gate["absolute_macro_f1_drop"]),
                agreement=_fmt(gate["fixed_vs_fp32_agreement"]),
            )
        )

    lines.extend(
        [
            "",
            "## Device timing",
            "",
            "Each entry uses three ordered 1,000-row repetitions on one physical board. "
            "The reported SD is the sample SD of the three repeat means. P95 and p99 are "
            "descriptive percentiles across the combined 3,000 device timings.",
            "",
            "| Model | Board | Preprocess mean (us) | Inference mean (us) | Total mean (us) | Total repeat SD (us) | Total p95 (us) | Total p99 (us) |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for session in report["sessions"]:
        timing = session["timing"]
        total_rows = timing["total_us"]["row_level_descriptive_3000"]
        lines.append(
            "| {model} | {board} | {pre} | {inf} | {total} | {sd} | {p95} | {p99} |".format(
                model=session["model"],
                board=session["board_label"],
                pre=_fmt(timing["preprocess_us"]["mean_of_repeat_means"], 3),
                inf=_fmt(timing["inference_us"]["mean_of_repeat_means"], 3),
                total=_fmt(timing["total_us"]["mean_of_repeat_means"], 3),
                sd=_fmt(timing["total_us"]["sample_sd_of_repeat_means"], 3),
                p95=total_rows["p95"],
                p99=total_rows["p99"],
            )
        )

    lines.extend(
        [
            "",
            "## Host-observed transaction timing",
            "",
            "These values are descriptive. They include serial transfer, operating-system "
            "overhead, response handling, and device computation; they are not pure "
            "transport latency.",
            "",
            "| Model | Board | Host RTT mean (us) | Host RTT p95 (us) | Transaction mean (us) | Transaction p95 (us) |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for session in report["sessions"]:
        timing = session["timing"]
        host_rtt = timing["host_observed_rtt_us"]
        transaction = timing["transaction_elapsed_us"]
        lines.append(
            "| {model} | {board} | {rtt_mean} | {rtt_p95} | {transaction_mean} | {transaction_p95} |".format(
                model=session["model"],
                board=session["board_label"],
                rtt_mean=_fmt(host_rtt["mean_of_repeat_means"], 3),
                rtt_p95=host_rtt["row_level_descriptive_3000"]["p95"],
                transaction_mean=_fmt(transaction["mean_of_repeat_means"], 3),
                transaction_p95=transaction["row_level_descriptive_3000"]["p95"],
            )
        )

    lines.extend(
        [
            "",
            "## Compile and model footprint",
            "",
            "| Model | Board | Program bytes | Program capacity | Global bytes | Dynamic-memory capacity | Parameter bytes | MACs/inference |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for session in report["sessions"]:
        compile_data = session["compile_footprint"]
        model_data = report["model_results"][session["model_key"]]["model_footprint"]
        lines.append(
            "| {model} | {board} | {program} | {program_max} | {global_used} | "
            "{global_max} | {params} | {macs} |".format(
                model=session["model"],
                board=session["board_label"],
                program=compile_data["program_bytes"],
                program_max=compile_data["program_max_bytes"],
                global_used=compile_data["global_bytes"],
                global_max=compile_data["global_max_bytes"],
                params=model_data["parameter_bytes"],
                macs=model_data["macs_per_inference"],
            )
        )

    lines.extend(
        [
            "",
            "## Per-class fixed-point F1",
            "",
            "| Model | Blackhole | Flooding | Grayhole | Normal | TDMA |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model in report["model_results"].values():
        values = model["fixed_metrics"]["per_class_f1"]
        lines.append(
            "| {model} | {values} |".format(
                model=model["model"],
                values=" | ".join(_fmt(value) for value in values),
            )
        )

    lines.extend(["", "## Blocked route", ""])
    for blocked in report["blocked_routes"]:
        reason = str(blocked["reason"]).rstrip(". ") + "."
        lines.append(
            f"- {blocked['model']}: {reason} It was blocked before firmware "
            "generation on both boards."
        )
    lines.extend(
        [
            "",
            "## Evidence identity",
            "",
            f"- Report ID: `{report['report_id']}`",
            f"- Archive ID: `{report['archive_id']}`",
            f"- Campaign evidence ID: `{report['campaign_evidence_id']}`",
            f"- Contract ID: `{report['contract_id']}`",
            "",
            "## Claim boundary",
            "",
            str(report["claim_boundary"]),
            "",
            "The replay inputs are already extracted 17-feature WSN-DS records. The "
            "campaign does not measure live packet-to-feature extraction, energy, "
            "transport latency, multi-unit variation, or multi-seed hardware variation.",
            "",
            "Compiled program storage includes the firmware and platform runtime. Static "
            "global memory is not a measurement of peak RAM usage.",
            "",
        ]
    )
    return "\n".join(lines)


def _flatten_session(session: Mapping[str, Any]) -> dict[str, Any]:
    fidelity = session["fidelity"]
    timing = session["timing"]
    total_rows = timing["total_us"]["row_level_descriptive_3000"]
    compile_data = session["compile_footprint"]
    return {
        "execution_ordinal": session["execution_ordinal"],
        "combination_id": session["combination_id"],
        "model": session["model"],
        "student": session["student"],
        "route": session["route"],
        "board": session["board"],
        "transport": session["transport"],
        "physical_port_serial": session["physical_port_serial"],
        "full_rows": fidelity["rows"],
        "accuracy": fidelity["accuracy"],
        "macro_f1": fidelity["macro_f1"],
        "weighted_f1": fidelity["weighted_f1"],
        "mcu_vs_fixed_reference_agreement": fidelity[
            "mcu_vs_fixed_reference_agreement"
        ],
        "mcu_fixed_logits_exact_fraction": fidelity[
            "mcu_fixed_logits_exact_fraction"
        ],
        "mcu_vs_fp32_agreement": fidelity["mcu_vs_fp32_agreement"],
        "timing_repeat_count": timing["repeat_count"],
        "preprocess_mean_us": timing["preprocess_us"]["mean_of_repeat_means"],
        "inference_mean_us": timing["inference_us"]["mean_of_repeat_means"],
        "total_mean_us": timing["total_us"]["mean_of_repeat_means"],
        "total_repeat_mean_sample_sd_us": timing["total_us"][
            "sample_sd_of_repeat_means"
        ],
        "total_p95_us_descriptive_3000": total_rows["p95"],
        "total_p99_us_descriptive_3000": total_rows["p99"],
        "host_observed_rtt_mean_us": timing["host_observed_rtt_us"][
            "mean_of_repeat_means"
        ],
        "host_observed_rtt_p95_us_descriptive_3000": timing[
            "host_observed_rtt_us"
        ]["row_level_descriptive_3000"]["p95"],
        "transaction_elapsed_mean_us": timing["transaction_elapsed_us"][
            "mean_of_repeat_means"
        ],
        "transaction_elapsed_p95_us_descriptive_3000": timing[
            "transaction_elapsed_us"
        ]["row_level_descriptive_3000"]["p95"],
        "program_bytes": compile_data["program_bytes"],
        "program_max_bytes": compile_data["program_max_bytes"],
        "global_bytes": compile_data["global_bytes"],
        "global_max_bytes": compile_data["global_max_bytes"],
        "session_evidence_id": session["session_evidence_id"],
    }


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise RuntimeError(f"Refusing to write an empty report table: {path.name}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def write_report(report: Mapping[str, Any], output_dir: Path) -> Path:
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite report directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp.{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"Stale report directory exists: {temporary}")
    temporary.mkdir()
    try:
        (temporary / "final_hil_summary.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        (temporary / "final_hil_summary.md").write_text(
            _markdown(report), encoding="utf-8"
        )
        _write_csv(
            temporary / "final_hil_results.csv",
            (_flatten_session(session) for session in report["sessions"]),
        )
        _write_csv(
            temporary / "final_hil_timing_repeats.csv",
            report["timing_repeat_rows"],
        )
        manifest_payload: dict[str, Any] = {
            "schema": "cukd_final_hil_report_artifact_manifest_v1",
            "report_id": report["report_id"],
            "archive_id": report["archive_id"],
            "files": [
                {
                    "path": name,
                    "size_bytes": (temporary / name).stat().st_size,
                    "sha256": sha256_file(temporary / name),
                }
                for name in (
                    "final_hil_summary.json",
                    "final_hil_summary.md",
                    "final_hil_results.csv",
                    "final_hil_timing_repeats.csv",
                )
            ],
        }
        manifest_payload["manifest_id"] = canonical_json_sha256(manifest_payload)
        (temporary / "report_manifest.json").write_text(
            json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def verify_report(
    report_dir: Path,
    *,
    archive_dir: Path | None = None,
) -> dict[str, Any]:
    root = Path(report_dir).resolve()
    manifest = read_json(root / "report_manifest.json")
    if manifest.get("schema") != "cukd_final_hil_report_artifact_manifest_v1":
        raise RuntimeError("Unsupported final-HIL report manifest schema")
    recorded_id = manifest.get("manifest_id")
    identity_payload = dict(manifest)
    identity_payload.pop("manifest_id", None)
    if recorded_id != canonical_json_sha256(identity_payload):
        raise RuntimeError("Final-HIL report manifest ID is invalid")
    expected_names = {
        "final_hil_summary.json",
        "final_hil_summary.md",
        "final_hil_results.csv",
        "final_hil_timing_repeats.csv",
    }
    files = manifest.get("files")
    if not isinstance(files, list) or {item.get("path") for item in files} != expected_names:
        raise RuntimeError("Final-HIL report artifact set is incomplete")
    for item in files:
        path = _safe_local(root, item["path"])
        if path.stat().st_size != item.get("size_bytes") or sha256_file(path) != item.get(
            "sha256"
        ):
            raise RuntimeError(f"Final-HIL report artifact changed: {item['path']}")
    report = read_json(root / "final_hil_summary.json")
    recorded_report_id = report.get("report_id")
    report_payload = dict(report)
    report_payload.pop("report_id", None)
    if recorded_report_id != canonical_json_sha256(report_payload):
        raise RuntimeError("Final-HIL report ID is invalid")
    if (
        manifest.get("report_id") != recorded_report_id
        or manifest.get("archive_id") != report.get("archive_id")
    ):
        raise RuntimeError("Final-HIL report manifest identity differs from its summary")
    archive_rederived = False
    if archive_dir is not None:
        with tempfile.TemporaryDirectory(prefix="cukd_final_hil_report_verify_") as tmp:
            rebuilt = Path(tmp) / "rebuilt"
            write_report(build_report(archive_dir), rebuilt)
            for name in expected_names | {"report_manifest.json"}:
                if (root / name).read_bytes() != (rebuilt / name).read_bytes():
                    raise RuntimeError(
                        f"Final-HIL report differs from archive rederivation: {name}"
                    )
        archive_rederived = True
    return {
        "schema": manifest["schema"],
        "status": report["status"],
        "manifest_id": recorded_id,
        "report_id": recorded_report_id,
        "archive_id": report["archive_id"],
        "artifact_count": len(files),
        "archive_rederived": archive_rederived,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify-report-dir", type=Path)
    args = parser.parse_args(argv)
    if args.verify_report_dir is not None:
        if args.output_dir is not None:
            parser.error("--verify-report-dir cannot be combined with --output-dir")
        print(
            json.dumps(
                verify_report(
                    args.verify_report_dir,
                    archive_dir=args.archive_dir,
                ),
                sort_keys=True,
            )
        )
        return 0
    if args.archive_dir is None or args.output_dir is None:
        parser.error("generation requires --archive-dir and --output-dir")
    output = write_report(build_report(args.archive_dir), args.output_dir)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
