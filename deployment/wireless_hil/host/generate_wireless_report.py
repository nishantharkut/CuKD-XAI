"""Generate a four-pair FG-DS Wi-Fi UDP HIL report from sealed evidence."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from deployment.hardware_hil.host.record_fgds_compile_evidence import (
    BOARD_FQBN_PREFIXES,
    FLASH_PATTERN,
    RAM_PATTERN,
    parsed_match,
    validate_footprint,
)

try:
    from .wireless_common import (
        REQUIRED_WIRELESS_STAGES,
        WIRELESS_COMPLETION_PROTOCOL_ID,
        WIRELESS_PROTOCOL_ID,
        read_compile_log_text,
        read_json,
        sha256_file,
        validate_compile_log_metadata,
    )
except ImportError:
    from wireless_common import (  # type: ignore
        REQUIRED_WIRELESS_STAGES,
        WIRELESS_COMPLETION_PROTOCOL_ID,
        WIRELESS_PROTOCOL_ID,
        read_compile_log_text,
        read_json,
        sha256_file,
        validate_compile_log_metadata,
    )


EXPECTED_LABELS = {
    "esp32c3_student_A": ("esp32c3", "student_A"),
    "esp32c3_student_B": ("esp32c3", "student_B"),
    "arduino_r4_student_A": ("arduino_r4", "student_A"),
    "arduino_r4_student_B": ("arduino_r4", "student_B"),
}


def parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH")
    label, path = value.split("=", 1)
    if label not in EXPECTED_LABELS or not path:
        raise argparse.ArgumentTypeError(f"Unknown or empty labeled path: {value}")
    return label, Path(path)


def verify_inventory(root: Path, manifest_path: Path, manifest: dict[str, Any]) -> None:
    files = manifest.get("files")
    if not isinstance(files, list) or manifest.get("file_count_excluding_manifest") != len(files):
        raise RuntimeError(f"Invalid inventory in {manifest_path}")
    declared: set[str] = set()
    for item in files:
        relative = item.get("path")
        if not isinstance(relative, str) or not relative or relative in declared:
            raise RuntimeError(f"Invalid inventory path in {manifest_path}")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(f"Inventory path escapes root: {relative}")
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise RuntimeError(f"Inventory path escapes root: {relative}") from exc
        if (
            not path.is_file()
            or path.stat().st_size != item.get("size_bytes")
            or sha256_file(path) != item.get("sha256")
        ):
            raise RuntimeError(f"Inventory artifact is missing or changed: {path}")
        declared.add(relative_path.as_posix())
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.resolve() != manifest_path.resolve()
    }
    if actual != declared:
        raise RuntimeError(f"Inventory differs from files on disk: {manifest_path}")


def verify_run(label: str, metrics_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics_path = metrics_path.resolve()
    if metrics_path.name != "full_56301_metrics.json":
        raise RuntimeError(f"{label} is not a full_56301 metrics artifact")
    root = metrics_path.parent
    completion_path = root / "wireless_hil_completion_manifest.json"
    completion = read_json(completion_path)
    metrics = read_json(metrics_path)
    board, student = EXPECTED_LABELS[label]
    if (
        completion.get("status") != "complete"
        or completion.get("protocol_id") != WIRELESS_COMPLETION_PROTOCOL_ID
        or completion.get("wireless_application_protocol_id") != WIRELESS_PROTOCOL_ID
        or completion.get("required_stages") != REQUIRED_WIRELESS_STAGES
        or completion.get("board") != board
        or completion.get("student") != student
        or completion.get("completion_script_sha256")
        != sha256_file(Path(__file__).with_name("complete_wireless_run.py"))
    ):
        raise RuntimeError(f"Completion identity is invalid: {label}")
    verify_inventory(root, completion_path, completion)
    full_evidence = completion.get("stage_evidence", {}).get("full_56301", {})
    if full_evidence.get("metrics_json_sha256") != sha256_file(metrics_path):
        raise RuntimeError(f"Completion does not bind full metrics: {label}")
    if (
        metrics.get("status") != "passed"
        or metrics.get("protocol_id") != WIRELESS_PROTOCOL_ID
        or metrics.get("stage_contract", {}).get("name") != "full_56301"
        or metrics.get("completed_vectors") != 56301
        or metrics.get("mcu_vs_fixed_reference_agreement") != 1.0
        or metrics.get("exact_logit_agreement") != 1.0
        or metrics.get("non_ok_status_count") != 0
        or metrics.get("strict_export_metric_reconciliation", {}).get("status")
        != "passed"
        or metrics.get("provenance", {}).get("board") != board
        or metrics.get("provenance", {}).get("student") != student
        or metrics.get("provenance", {}).get("export_id") != completion.get("export_id")
        or metrics.get("provenance", {}).get("wireless_bundle_id")
        != completion.get("wireless_bundle_id")
    ):
        raise RuntimeError(f"Full wireless metrics are invalid: {label}")
    verification_script = Path(__file__).with_name("verify_results_udp.py")
    if metrics.get("provenance", {}).get("verification_script_sha256") != sha256_file(
        verification_script
    ):
        raise RuntimeError(f"Full metrics used another verifier: {label}")
    connection = read_json(root / "connection.json")
    if connection.get("session_id") != completion.get("session_id"):
        raise RuntimeError(f"Connection/completion session differs: {label}")
    return metrics, connection


def resolve_portable(evidence_path: Path, item: dict[str, Any]) -> Path:
    relative = item.get("path")
    if not isinstance(relative, str) or not relative:
        raise RuntimeError("Compile evidence has an invalid portable path")
    path = (evidence_path.parent / relative).resolve()
    try:
        path.relative_to(evidence_path.parent.resolve())
    except ValueError as exc:
        raise RuntimeError("Compile portable artifact escapes evidence root") from exc
    if (
        not path.is_file()
        or sha256_file(path) != item.get("sha256")
        or ("size_bytes" in item and path.stat().st_size != item["size_bytes"])
    ):
        raise RuntimeError(f"Compile portable artifact is missing or changed: {path}")
    return path


def verify_compile(
    label: str,
    evidence_path: Path,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    evidence_path = evidence_path.resolve()
    evidence = read_json(evidence_path)
    board, student = EXPECTED_LABELS[label]
    if (
        evidence.get("status") != "passed"
        or evidence.get("protocol_id") != WIRELESS_PROTOCOL_ID
        or evidence.get("board") != board
        or evidence.get("student") != student
        or evidence.get("export_id") != metrics["provenance"]["export_id"]
        or evidence.get("wireless_bundle_id")
        != metrics["provenance"]["wireless_bundle_id"]
        or evidence.get("compile_evidence_script_sha256")
        != sha256_file(Path(__file__).with_name("record_wireless_compile_evidence.py"))
    ):
        raise RuntimeError(f"Compile evidence identity is invalid: {label}")
    portable = evidence.get("portable_artifacts")
    required = {
        "compile_log",
        "firmware_binary",
        "strict_export_manifest",
        "strict_export_report",
        "wireless_bundle_manifest",
        "hil_reference_predictions",
    }
    if not isinstance(portable, dict) or set(portable) != required:
        raise RuntimeError(f"Compile evidence portable set differs: {label}")
    resolved = {name: resolve_portable(evidence_path, portable[name]) for name in required}
    if (
        portable["compile_log"]["sha256"] != evidence["compile_log_sha256"]
        or portable["firmware_binary"]["sha256"]
        != evidence["firmware_binary_sha256"]
        or portable["wireless_bundle_manifest"]["sha256"]
        != evidence["wireless_bundle_manifest_sha256"]
        or portable["strict_export_manifest"]["sha256"]
        != evidence["strict_export_manifest_sha256"]
        or portable["hil_reference_predictions"]["sha256"]
        != metrics["provenance"]["reference_csv_sha256"]
    ):
        raise RuntimeError(f"Compile evidence hashes do not reconcile: {label}")
    strict_manifest = read_json(resolved["strict_export_manifest"])
    strict_files = {
        item.get("path"): item
        for item in strict_manifest.get("files", [])
        if isinstance(item, dict)
    }
    if (
        strict_manifest.get("status") != "passed"
        or strict_manifest.get("student") != student
        or strict_manifest.get("export_id") != evidence["export_id"]
        or strict_files.get("strict_export_report.json", {}).get("sha256")
        != portable["strict_export_report"]["sha256"]
        or strict_files.get("hil_reference_predictions.csv", {}).get("sha256")
        != portable["hil_reference_predictions"]["sha256"]
    ):
        raise RuntimeError(f"Portable strict-export evidence is inconsistent: {label}")
    compile_text = read_compile_log_text(resolved["compile_log"])
    parsed_flash = parsed_match(FLASH_PATTERN, compile_text, "flash")
    parsed_ram = parsed_match(RAM_PATTERN, compile_text, "RAM")
    validate_footprint(parsed_flash, parsed_ram)
    if evidence.get("flash") != parsed_flash or evidence.get("ram") != parsed_ram:
        raise RuntimeError(f"Compile footprint differs from its preserved log: {label}")
    fqbn = evidence.get("fqbn")
    if not isinstance(fqbn, str) or not any(
        fqbn == prefix or fqbn.startswith(prefix + ":")
        for prefix in BOARD_FQBN_PREFIXES[board]
    ):
        raise RuntimeError(f"Compile FQBN does not match its board: {label}")
    metadata_names = [
        "fqbn",
        "board_core_version",
        "frontend_version",
        "toolchain_version",
    ]
    if any(
        not isinstance(evidence.get(name), str) or not evidence[name]
        for name in metadata_names
    ):
        raise RuntimeError(f"Compile evidence has invalid metadata: {label}")
    validate_compile_log_metadata(
        compile_text,
        fqbn=evidence["fqbn"],
        board_core_version=evidence["board_core_version"],
        frontend_version=evidence["frontend_version"],
        toolchain_version=evidence["toolchain_version"],
    )
    if evidence.get("sketch_file") not in compile_text:
        raise RuntimeError(f"Compile log does not identify its sketch: {label}")
    binary_payload = resolved["firmware_binary"].read_bytes()
    for value in [
        evidence["export_id"],
        evidence["wireless_bundle_id"],
        WIRELESS_PROTOCOL_ID,
    ]:
        if value.encode("ascii") not in binary_payload:
            raise RuntimeError(f"Portable firmware lacks its embedded identity: {label}")
    return evidence


def fmt(value: Any, digits: int = 6) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_labeled_path, required=True)
    parser.add_argument("--compile", action="append", type=parse_labeled_path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    runs = dict(args.run)
    compiles = dict(args.compile)
    if (
        set(runs) != set(EXPECTED_LABELS)
        or len(args.run) != 4
        or set(compiles) != set(EXPECTED_LABELS)
        or len(args.compile) != 4
    ):
        raise RuntimeError(
            "Exactly four distinct board/student run and compile inputs are required"
        )
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite wireless report directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / (
        f".{output_dir.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    if staging.exists():
        raise FileExistsError(f"Stale wireless report staging directory: {staging}")

    verified_runs: dict[str, dict[str, Any]] = {}
    connections: dict[str, dict[str, Any]] = {}
    verified_compiles: dict[str, dict[str, Any]] = {}
    for label in EXPECTED_LABELS:
        metrics, connection = verify_run(label, runs[label])
        verified_runs[label] = metrics
        connections[label] = connection
        verified_compiles[label] = verify_compile(label, compiles[label], metrics)

    for student in ["student_A", "student_B"]:
        pair = [
            verified_runs[label]
            for label in EXPECTED_LABELS
            if EXPECTED_LABELS[label][1] == student
        ]
        for key in [
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "mcu_vs_fp32_agreement",
            "mcu_vs_fixed_reference_agreement",
            "exact_logit_agreement",
            "confusion_matrix",
        ]:
            if pair[0].get(key) != pair[1].get(key):
                raise RuntimeError(f"Cross-board model result differs for {student}/{key}")

    rows: list[dict[str, Any]] = []
    for label, (board, student) in EXPECTED_LABELS.items():
        metrics = verified_runs[label]
        compile_evidence = verified_compiles[label]
        data_network = metrics["data_network_counters"]
        device_latency = metrics["device_compute_latency"]["total_us"]
        host_latency = metrics["host_observed_transport_timing"][
            "host_observed_datagram_rtt_us"
        ]
        rows.append(
            {
                "label": label,
                "board": board,
                "student": student,
                "full_vectors": 56301,
                "mcu_vs_fixed_reference": metrics[
                    "mcu_vs_fixed_reference_agreement"
                ],
                "exact_logit_agreement": metrics["exact_logit_agreement"],
                "mcu_vs_fp32": metrics["mcu_vs_fp32_agreement"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "device_compute_mean_us": device_latency["mean"],
                "device_compute_p95_us": device_latency["p95"],
                "host_rtt_mean_us": host_latency["mean"],
                "host_rtt_p95_us": host_latency["p95"],
                "data_retransmissions": data_network["retransmissions"],
                "data_response_timeouts": data_network["response_timeouts"],
                "rssi_dbm_at_connection": connections[label]["rssi_dbm_at_connection"],
                "flash_used_bytes": compile_evidence["flash"]["used"],
                "flash_maximum_bytes": compile_evidence["flash"]["maximum"],
                "ram_used_bytes": compile_evidence["ram"]["used"],
                "ram_maximum_bytes": compile_evidence["ram"]["maximum"],
                "fqbn": compile_evidence["fqbn"],
                "board_core_version": compile_evidence["board_core_version"],
            }
        )

    payload = {
        "status": "passed",
        "protocol_id": WIRELESS_PROTOCOL_ID,
        "full_test_rows_per_board_model_pair": 56301,
        "full_test_board_model_predictions": 4 * 56301,
        "all_three_stage_board_model_inferences": 4
        * sum(int(value["rows"]) for value in REQUIRED_WIRELESS_STAGES.values()),
        "rows": rows,
        "timing_boundary": (
            "The MCU timed code region and host-observed datagram RTT are separate "
            "measurements. MCU wall-clock timing may include interrupt preemption. "
            "Their difference is not reported as pure wireless latency."
        ),
        "claim_boundary": (
            "Controlled-LAN Wi-Fi UDP replay of extracted FG-DS features into exact "
            "fixed-point MCU inference. The evidence does not establish live WSN capture, "
            "on-device feature extraction, transport security, energy efficiency, BLE, "
            "or physical WSN-radio deployment."
        ),
        "input_evidence": {
            label: {
                "metrics_path_recorded": str(runs[label].resolve()),
                "metrics_sha256": sha256_file(runs[label].resolve()),
                "compile_path_recorded": str(compiles[label].resolve()),
                "compile_sha256": sha256_file(compiles[label].resolve()),
            }
            for label in EXPECTED_LABELS
        },
        "report_script_sha256": sha256_file(Path(__file__).resolve()),
    }

    staging.mkdir()
    try:
        json_path = staging / "wireless_hil_final_report.json"
        csv_path = staging / "wireless_hil_final_table.csv"
        markdown_path = staging / "wireless_hil_final_report.md"
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        markdown_lines = [
            "# FG-DS Wi-Fi UDP Hardware-in-the-Loop Report",
            "",
            "| Board | Student | Vectors | Fixed pred. | Exact logits | "
            "FP32 agree. | Macro-F1 | Compute mean (us) | Host RTT mean (us) | "
            "Retries | Flash (B) | RAM (B) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            markdown_lines.append(
                "| "
                + " | ".join(
                    [
                        row["board"],
                        row["student"],
                        str(row["full_vectors"]),
                        fmt(row["mcu_vs_fixed_reference"]),
                        fmt(row["exact_logit_agreement"]),
                        fmt(row["mcu_vs_fp32"]),
                        fmt(row["macro_f1"]),
                        fmt(row["device_compute_mean_us"], 3),
                        fmt(row["host_rtt_mean_us"], 3),
                        str(row["data_retransmissions"]),
                        str(row["flash_used_bytes"]),
                        str(row["ram_used_bytes"]),
                    ]
                )
                + " |"
            )
        markdown_lines.extend(
            [
                "",
                "## Measurement Boundary",
                "",
                payload["timing_boundary"],
                "",
                "## Claim Boundary",
                "",
                payload["claim_boundary"],
                "",
            ]
        )
        markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")
        manifest_path = staging / "wireless_report_manifest.json"
        files = [
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in [json_path, csv_path, markdown_path]
        ]
        manifest = {
            "status": "passed",
            "protocol_id": WIRELESS_PROTOCOL_ID,
            "report_script_sha256": sha256_file(Path(__file__).resolve()),
            "file_count_excluding_manifest": len(files),
            "files": files,
        }
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, manifest_path)
        os.replace(staging, output_dir)
    except Exception:
        if staging.exists():
            resolved = staging.resolve()
            if (
                resolved.parent != output_dir.parent.resolve()
                or not resolved.name.startswith(f".{output_dir.name}.tmp.")
            ):
                raise RuntimeError("Refusing to remove unsafe wireless report staging path")
            shutil.rmtree(resolved)
        raise
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
