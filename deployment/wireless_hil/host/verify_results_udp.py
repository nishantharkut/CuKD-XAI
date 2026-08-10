"""Verify one session-bound FG-DS Wi-Fi UDP replay stage from primary artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any

from deployment.hardware_hil.host.hil_common import (
    compute_classification_metrics,
    summarize_latency,
)
from deployment.hardware_hil.host.stream_vectors_fgds_strict import FULL_TEST_ROWS
from deployment.hardware_hil.host.verify_results_fgds_strict import (
    read_reference,
    read_replay_source_rows,
    require_manifest_file,
)

try:
    from .stream_vectors_udp import CSV_FIELDS
    from .udp_session import IGNORED_DATAGRAM_CATEGORIES
    from .wireless_common import (
        DEFAULT_DEVICE_UDP_PORT,
        DEFAULT_HOST_UDP_PORT,
        REQUIRED_WIRELESS_STAGES,
        WIRELESS_PROTOCOL_ID,
        expected_device_identity,
        read_json,
        sha256_file,
        validate_connection_record,
        verify_export_for_wireless,
        verify_wireless_bundle,
    )
except ImportError:
    from stream_vectors_udp import CSV_FIELDS  # type: ignore
    from udp_session import IGNORED_DATAGRAM_CATEGORIES  # type: ignore
    from wireless_common import (  # type: ignore
        DEFAULT_DEVICE_UDP_PORT,
        DEFAULT_HOST_UDP_PORT,
        REQUIRED_WIRELESS_STAGES,
        WIRELESS_PROTOCOL_ID,
        expected_device_identity,
        read_json,
        sha256_file,
        validate_connection_record,
        verify_export_for_wireless,
        verify_wireless_bundle,
    )


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_fgds_wireless_hil.sh"
INTEGER_FIELDS = {
    "row_id",
    "predicted_class",
    "preprocess_us",
    "inference_us",
    "total_us",
    "attempts",
    "response_timeout_count",
    "ignored_datagram_count",
    *{f"ignored_{category}_count" for category in IGNORED_DATAGRAM_CATEGORIES},
    "successful_request_bytes",
    "request_datagram_bytes_sent",
    "response_bytes",
    "host_observed_datagram_rtt_us",
    "transaction_elapsed_us",
}


def read_wireless_mcu(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CSV_FIELDS:
            raise RuntimeError(
                "Wireless MCU CSV schema differs from the executing streamer: "
                f"observed={reader.fieldnames}, expected={CSV_FIELDS}"
            )
        rows: list[dict[str, Any]] = []
        for csv_row in reader:
            if None in csv_row or any(value is None for value in csv_row.values()):
                raise RuntimeError("Wireless MCU CSV contains a malformed row")
            row: dict[str, Any] = {
                key: int(value) if key in INTEGER_FIELDS else value
                for key, value in csv_row.items()
                if key != "logits"
            }
            row["logits"] = [int(value) for value in csv_row["logits"].split()]
            rows.append(row)
        return rows


def validate_exchange_evidence(record: dict[str, Any], label: str) -> None:
    ignored_sum = sum(
        int(record.get(f"ignored_{category}_count", -1))
        for category in IGNORED_DATAGRAM_CATEGORIES
    )
    if (
        int(record.get("attempts", 0)) < 1
        or int(record.get("response_timeout_count", -1))
        != int(record.get("attempts", 0)) - 1
        or ignored_sum < 0
        or int(record.get("ignored_datagram_count", -1)) != ignored_sum
        or int(record.get("successful_request_bytes", 0)) <= 0
        or int(record.get("request_datagram_bytes_sent", 0))
        < int(record.get("successful_request_bytes", 0))
        or int(record.get("response_bytes", 0)) <= 0
        or int(record.get("host_observed_datagram_rtt_us", -1)) < 0
        or int(record.get("transaction_elapsed_us", -1))
        < int(record.get("host_observed_datagram_rtt_us", 0))
    ):
        raise RuntimeError(f"Invalid exchange evidence for {label}")


def derive_data_network_counters(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "successful_transactions": len(rows),
        "datagrams_sent": sum(row["attempts"] for row in rows),
        "retransmissions": sum(row["attempts"] - 1 for row in rows),
        "response_timeouts": sum(row["response_timeout_count"] for row in rows),
        "ignored_datagrams": sum(row["ignored_datagram_count"] for row in rows),
        "request_bytes_sent": sum(row["request_datagram_bytes_sent"] for row in rows),
        "response_bytes_received": sum(row["response_bytes"] for row in rows),
    }


def validate_session_counters(
    session_counters: dict[str, Any],
    rows: list[dict[str, Any]],
    controls: dict[str, Any],
) -> None:
    for name in ["identity", "begin", "end"]:
        record = controls.get(name)
        if not isinstance(record, dict):
            raise RuntimeError(f"Passed replay lacks {name} control evidence")
        validate_exchange_evidence(record, name)
    if "abort" in controls:
        raise RuntimeError("Passed replay unexpectedly contains an abort transaction")

    control_records = [controls[name] for name in ["identity", "begin", "end"]]
    expected_sent = sum(int(record["attempts"]) for record in control_records) + sum(
        row["attempts"] for row in rows
    )
    expected_retransmissions = expected_sent - (len(rows) + len(control_records))
    expected_timeouts = sum(
        int(record["response_timeout_count"]) for record in control_records
    ) + sum(row["response_timeout_count"] for row in rows)
    expected_ignored = sum(
        int(record["ignored_datagram_count"]) for record in control_records
    ) + sum(row["ignored_datagram_count"] for row in rows)
    expected_received = len(rows) + len(control_records) + expected_ignored
    expected_request_bytes = sum(
        int(record["request_datagram_bytes_sent"]) for record in control_records
    ) + sum(row["request_datagram_bytes_sent"] for row in rows)
    accepted_response_bytes = sum(
        int(record["response_bytes"]) for record in control_records
    ) + sum(row["response_bytes"] for row in rows)
    expected_exact = {
        "datagrams_sent": expected_sent,
        "retransmissions": expected_retransmissions,
        "response_timeouts": expected_timeouts,
        "datagrams_received": expected_received,
        "request_bytes_sent": expected_request_bytes,
        "device_protocol_errors": 0,
    }
    for key, value in expected_exact.items():
        if session_counters.get(key) != value:
            raise RuntimeError(f"Session counter differs for {key}")
    if int(session_counters.get("response_bytes_received", -1)) < accepted_response_bytes:
        raise RuntimeError("Session response-byte counter is below accepted responses")
    if session_counters.get("stale_datagrams_drained_before_identity") != controls.get(
        "stale_datagrams_drained_before_identity"
    ):
        raise RuntimeError("Pre-identity stale datagram count differs")
    for category in IGNORED_DATAGRAM_CATEGORIES:
        expected_category = sum(
            int(record[f"ignored_{category}_count"])
            for record in [*control_records, *rows]
        )
        if session_counters.get(f"ignored_{category}") != expected_category:
            raise RuntimeError(
                f"Session ignored-datagram category differs for {category}"
            )


def reconcile_full_metrics_with_strict_export(
    metrics: dict[str, Any],
    report: dict[str, Any],
    *,
    student: str,
    export_id: str,
) -> dict[str, Any]:
    provenance = report.get("provenance")
    gates = report.get("gates")
    expected = report.get("fixed_metrics")
    if (
        report.get("status") != "passed"
        or report.get("export_id") != export_id
        or not isinstance(provenance, dict)
        or provenance.get("student") != student
        or provenance.get("seed") != 42
        or not isinstance(gates, dict)
        or gates.get("full_test_rows") != FULL_TEST_ROWS
        or not isinstance(expected, dict)
    ):
        raise RuntimeError("Strict export report identity or full-test contract is invalid")

    if metrics.get("confusion_matrix") != expected.get("confusion_matrix"):
        raise RuntimeError("Full-stage confusion matrix differs from the strict export")
    for key in ["accuracy", "macro_f1"]:
        if not math.isclose(
            float(metrics.get(key, float("nan"))),
            float(expected.get(key, float("nan"))),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"Full-stage {key} differs from the strict export")

    expected_fields = {
        "precision": expected.get("per_class_precision"),
        "recall": expected.get("per_class_recall"),
        "f1": expected.get("per_class_f1"),
        "support": expected.get("per_class_support"),
    }
    if any(
        not isinstance(values, list) or len(values) != 5
        for values in expected_fields.values()
    ):
        raise RuntimeError("Strict export report has incomplete per-class metrics")
    observed_per_class = metrics.get("per_class")
    if not isinstance(observed_per_class, dict) or set(observed_per_class) != {
        str(index) for index in range(5)
    }:
        raise RuntimeError("Full-stage per-class metric set is invalid")
    for index in range(5):
        observed = observed_per_class[str(index)]
        if not isinstance(observed, dict):
            raise RuntimeError(f"Full-stage class {index} metrics are invalid")
        if observed.get("support") != expected_fields["support"][index]:
            raise RuntimeError(
                f"Full-stage class {index} support differs from the strict export"
            )
        for key in ["precision", "recall", "f1"]:
            if not math.isclose(
                float(observed.get(key, float("nan"))),
                float(expected_fields[key][index]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    f"Full-stage class {index} {key} differs from the strict export"
                )

    expected_fp32_agreement = gates.get("fixed_vs_fp32_agreement")
    if not math.isclose(
        float(metrics.get("mcu_vs_fp32_agreement", float("nan"))),
        float(expected_fp32_agreement),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Full-stage FP32 agreement differs from the strict export")
    return {
        "status": "passed",
        "accuracy": expected["accuracy"],
        "macro_f1": expected["macro_f1"],
        "fixed_vs_fp32_agreement": expected_fp32_agreement,
        "confusion_matrix_exact": True,
        "per_class_metrics_reconciled": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcu-csv", type=Path, required=True)
    parser.add_argument("--sequence-json", type=Path, required=True)
    parser.add_argument("--connection-json", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--stage-name", choices=sorted(REQUIRED_WIRELESS_STAGES), required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    generated_dir = args.generated_dir.resolve()
    bundle_dir = args.bundle_dir.resolve()
    mcu_csv = args.mcu_csv.resolve()
    sequence_path = args.sequence_json.resolve()
    connection_path = args.connection_json.resolve()
    reference_csv = args.reference_csv.resolve()
    output_path = args.output_json.resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite wireless metrics: {output_path}")
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Stale wireless metrics temporary exists: {temporary}")
    if len({mcu_csv, sequence_path, connection_path, reference_csv, output_path}) != 5:
        raise RuntimeError("Wireless verification inputs and output must be distinct")

    export_manifest = verify_export_for_wireless(generated_dir)
    bundle_manifest = verify_wireless_bundle(bundle_dir, export_manifest)
    connection = validate_connection_record(
        connection_path,
        bundle_manifest,
        generated_dir / "strict_export_manifest.json",
    )
    reference_hash = require_manifest_file(
        export_manifest,
        generated_dir,
        "hil_reference_predictions.csv",
    )
    strict_report_hash = require_manifest_file(
        export_manifest,
        generated_dir,
        "strict_export_report.json",
    )
    if reference_csv != (generated_dir / "hil_reference_predictions.csv").resolve():
        raise RuntimeError("Reference CSV is not the file bound by the strict export")

    frozen_stage = REQUIRED_WIRELESS_STAGES[args.stage_name]
    expected_count = int(frozen_stage["rows"])
    stage_ordinal = int(frozen_stage["ordinal"])
    sequence = read_json(sequence_path)
    stage_contract = sequence.get("stage_contract", {})
    if (
        stage_contract.get("name") != args.stage_name
        or stage_contract.get("ordinal") != stage_ordinal
        or stage_contract.get("expected_rows") != expected_count
        or re.fullmatch(r"[0-9A-F]{16}", str(stage_contract.get("stage_id", "")))
        is None
    ):
        raise RuntimeError("Wireless stage contract is not the frozen requested stage")
    stage_id = stage_contract["stage_id"]
    expected_provenance = {
        "transport": "IEEE 802.11 station mode with UDP datagrams",
        "application_protocol": WIRELESS_PROTOCOL_ID,
        "board": bundle_manifest["board"],
        "student": bundle_manifest["student"],
        "export_id": bundle_manifest["export_id"],
        "wireless_bundle_id": bundle_manifest["wireless_bundle_id"],
        "session_id": connection["session_id"],
        "device_identity": expected_device_identity(bundle_manifest),
        "device_endpoint": f"{connection['device_ip']}:{DEFAULT_DEVICE_UDP_PORT}",
        "host_udp_port": DEFAULT_HOST_UDP_PORT,
        "vector_sha256": require_manifest_file(
            export_manifest,
            generated_dir,
            "hil_replay_vectors.csv",
        ),
        "strict_export_manifest_sha256": sha256_file(
            generated_dir / "strict_export_manifest.json"
        ),
        "strict_export_report_sha256": strict_report_hash,
        "wireless_bundle_manifest_sha256": bundle_manifest["_manifest_sha256"],
        "connection_json_sha256": sha256_file(connection_path),
        "stream_script_sha256": sha256_file(
            Path(__file__).with_name("stream_vectors_udp.py")
        ),
        "udp_session_sha256": sha256_file(Path(__file__).with_name("udp_session.py")),
        "wireless_common_sha256": sha256_file(
            Path(__file__).with_name("wireless_common.py")
        ),
        "protocol_helper_sha256": sha256_file(
            REPO_ROOT / "deployment" / "hardware_hil" / "host" / "hil_common.py"
        ),
        "vector_loader_sha256": sha256_file(
            REPO_ROOT
            / "deployment"
            / "hardware_hil"
            / "host"
            / "stream_vectors.py"
        ),
        "run_script_sha256": sha256_file(RUN_SCRIPT_PATH),
    }
    for key, value in expected_provenance.items():
        if sequence.get("provenance", {}).get(key) != value:
            raise RuntimeError(f"Wireless sequence provenance differs for {key}")
    if sequence.get("protocol_id") != WIRELESS_PROTOCOL_ID:
        raise RuntimeError("Wireless sequence protocol differs")
    if sequence.get("output_csv_sha256") != sha256_file(mcu_csv):
        raise RuntimeError("Wireless MCU CSV differs from its sequence summary")
    if Path(sequence.get("output_csv", "")).resolve() != mcu_csv:
        raise RuntimeError("Wireless sequence summary records another MCU CSV")
    if sequence.get("status") != "passed" or sequence.get("error") is not None:
        raise RuntimeError("Wireless sequence summary is not passed")
    if sequence.get("expected") != expected_count or sequence.get("completed") != expected_count:
        raise RuntimeError("Wireless sequence count differs from its frozen stage")
    if any(sequence.get(key) for key in ["missing", "duplicates", "unexpected"]):
        raise RuntimeError("Wireless sequence contains missing, duplicate, or unexpected IDs")
    if sequence.get("status_counts") != {"OK": expected_count}:
        raise RuntimeError("Wireless sequence contains non-OK status rows")

    references = read_reference(reference_csv)
    replay_source_rows = read_replay_source_rows(
        generated_dir / "hil_replay_vectors.csv"
    )
    expected_full_ids = set(range(FULL_TEST_ROWS))
    if set(references) != expected_full_ids or set(replay_source_rows) != expected_full_ids:
        raise RuntimeError("FG-DS reference/replay does not contain the full strict row set")
    for row_id, reference in references.items():
        if replay_source_rows[row_id] != reference["source_row_index"]:
            raise RuntimeError(f"Replay/reference source-row mismatch at {row_id}")

    rows = read_wireless_mcu(mcu_csv)
    expected_ids = list(range(expected_count))
    if [row["row_id"] for row in rows] != expected_ids:
        raise RuntimeError("Wireless MCU rows are not the exact ordered requested prefix")

    mismatch_groups: dict[str, list[int]] = {
        "prediction": [],
        "logit": [],
        "argmax": [],
        "device_timing": [],
        "transport": [],
    }
    for row in rows:
        row_id = row["row_id"]
        reference = references[row_id]
        if row["status"] != "OK" or row["predicted_class"] != reference["fixed_pred"]:
            mismatch_groups["prediction"].append(row_id)
        if row["logits"] != reference["fixed_logits"]:
            mismatch_groups["logit"].append(row_id)
        if len(row["logits"]) != 5 or row["predicted_class"] != max(
            range(5), key=lambda index: row["logits"][index]
        ):
            mismatch_groups["argmax"].append(row_id)
        if (
            row["preprocess_us"] < 0
            or row["inference_us"] < 0
            or row["total_us"] < 0
            or row["total_us"] != row["preprocess_us"] + row["inference_us"]
        ):
            mismatch_groups["device_timing"].append(row_id)
        try:
            validate_exchange_evidence(row, f"row {row_id}")
        except RuntimeError:
            mismatch_groups["transport"].append(row_id)
    if any(mismatch_groups.values()):
        raise RuntimeError(
            "Wireless HIL mismatch: "
            + ", ".join(
                f"{key}={values[:5]}" for key, values in mismatch_groups.items()
            )
        )

    observed_data_counters = derive_data_network_counters(rows)
    if sequence.get("data_network_counters") != observed_data_counters:
        raise RuntimeError("Wireless data counters differ from the MCU CSV")
    controls = sequence.get("control_transactions")
    session_counters = sequence.get("session_network_counters")
    if not isinstance(controls, dict) or not isinstance(session_counters, dict):
        raise RuntimeError("Wireless sequence lacks network/control evidence")
    validate_session_counters(session_counters, rows, controls)
    device_stage_counters = controls["end"].get("device_stage_counters")
    if not isinstance(device_stage_counters, dict):
        raise RuntimeError("Wireless end control lacks device counters")
    if (
        device_stage_counters.get("completed_rows") != expected_count
        or device_stage_counters.get("expected_rows") != expected_count
        or device_stage_counters.get("inferences") != expected_count
        or device_stage_counters.get("stage_ordinal") != stage_ordinal
    ):
        raise RuntimeError("Device stage counters do not prove one inference per row")

    y_true = [references[row_id]["true_label"] for row_id in expected_ids]
    y_mcu = [row["predicted_class"] for row in rows]
    y_fp32 = [references[row_id]["fp32_pred"] for row_id in expected_ids]
    metrics = compute_classification_metrics(y_true, y_mcu, range(5))
    host_minus_compute = [
        row["host_observed_datagram_rtt_us"] - row["total_us"] for row in rows
    ]
    metrics.update(
        {
            "status": "passed",
            "protocol_id": WIRELESS_PROTOCOL_ID,
            "stage_contract": stage_contract,
            "completed_vectors": len(rows),
            "mcu_vs_fixed_reference_agreement": 1.0,
            "exact_logit_agreement": 1.0,
            "mcu_vs_fp32_agreement": sum(
                int(left == right) for left, right in zip(y_mcu, y_fp32)
            )
            / len(rows),
            "non_ok_status_count": 0,
            "data_network_counters": observed_data_counters,
            "session_network_counters": session_counters,
            "device_stage_counters": device_stage_counters,
            "device_compute_latency": {
                key: summarize_latency(row[key] for row in rows)
                for key in ["preprocess_us", "inference_us", "total_us"]
            },
            "host_observed_transport_timing": {
                "host_observed_datagram_rtt_us": summarize_latency(
                    row["host_observed_datagram_rtt_us"] for row in rows
                ),
                "transaction_elapsed_us": summarize_latency(
                    row["transaction_elapsed_us"] for row in rows
                ),
                "host_rtt_minus_device_compute_us_descriptive_only": summarize_latency(
                    host_minus_compute
                ),
            },
            "timing_boundaries": {
                "total_us": (
                    "MCU micros() interval around integer preprocessing and fixed-point "
                    "inference only; UDP receive, envelope parsing, formatting, send, and "
                    "radio time are outside the timed code region. The wall-clock interval "
                    "can include interrupt preemption."
                ),
                "host_observed_datagram_rtt_us": (
                    "Host monotonic interval from immediately before the successful "
                    "attempt sendto call to immediately after its matching recvfrom call."
                ),
                "transaction_elapsed_us": (
                    "Host monotonic interval from the first attempt through the matching "
                    "response, including earlier response timeouts and retransmissions."
                ),
                "host_rtt_minus_device_compute_us_descriptive_only": (
                    "Difference between measurements from independent host and MCU clocks; "
                    "it is not pure radio, network, or protocol latency."
                ),
            },
            "provenance": {
                **expected_provenance,
                "stage_id": stage_id,
                "mcu_csv_sha256": sha256_file(mcu_csv),
                "sequence_json_sha256": sha256_file(sequence_path),
                "reference_csv_sha256": reference_hash,
                "verification_script_sha256": sha256_file(Path(__file__).resolve()),
            },
            "claim_boundary": (
                "Controlled-LAN Wi-Fi UDP replay of already extracted 17-feature FG-DS "
                "records into the fixed-point MCU inference path, with exact fixed-logit "
                "verification. This does not establish live WSN capture, packet-to-feature "
                "extraction, production transport security, energy efficiency, BLE, or "
                "WSN-radio deployment."
            ),
        }
    )
    if args.stage_name == "full_56301":
        strict_report = read_json(generated_dir / "strict_export_report.json")
        metrics["strict_export_metric_reconciliation"] = (
            reconcile_full_metrics_with_strict_export(
                metrics,
                strict_report,
                student=bundle_manifest["student"],
                export_id=bundle_manifest["export_id"],
            )
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
