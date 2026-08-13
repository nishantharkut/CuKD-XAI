"""Replay one fixed FG-DS stage through a session-bound Wi-Fi UDP endpoint."""

from __future__ import annotations

import argparse
import csv
import json
import os
import secrets
import sys
from pathlib import Path
from typing import Any

from deployment.hardware_hil.host.hil_common import (
    decode_response_line,
    encode_request_line,
    verify_response_sequence,
)
from deployment.hardware_hil.host.stream_vectors import load_vectors
from deployment.hardware_hil.host.stream_vectors_fgds_strict import (
    require_export_file_hash,
    validate_output_paths,
    validate_replay_rows,
)

try:
    from .udp_session import (
        IGNORED_DATAGRAM_CATEGORIES,
        StrictUdpSession,
        parse_abort_response,
        parse_begin_response,
        parse_end_response,
    )
    from .wireless_common import (
        ABORT_TRANSACTION_ID,
        BEGIN_TRANSACTION_ID,
        DEFAULT_DEVICE_UDP_PORT,
        DEFAULT_HOST_UDP_PORT,
        END_TRANSACTION_ID,
        IDENTITY_TRANSACTION_ID,
        REQUEST_ENVELOPE_PREFIX,
        REQUIRED_WIRELESS_STAGES,
        WIRELESS_PROTOCOL_ID,
        encode_wireless_envelope,
        expected_device_identity,
        sha256_file,
        validate_connection_record,
        verify_export_for_wireless,
        verify_wireless_bundle,
    )
except ImportError:
    from udp_session import (  # type: ignore
        IGNORED_DATAGRAM_CATEGORIES,
        StrictUdpSession,
        parse_abort_response,
        parse_begin_response,
        parse_end_response,
    )
    from wireless_common import (  # type: ignore
        ABORT_TRANSACTION_ID,
        BEGIN_TRANSACTION_ID,
        DEFAULT_DEVICE_UDP_PORT,
        DEFAULT_HOST_UDP_PORT,
        END_TRANSACTION_ID,
        IDENTITY_TRANSACTION_ID,
        REQUEST_ENVELOPE_PREFIX,
        REQUIRED_WIRELESS_STAGES,
        WIRELESS_PROTOCOL_ID,
        encode_wireless_envelope,
        expected_device_identity,
        sha256_file,
        validate_connection_record,
        verify_export_for_wireless,
        verify_wireless_bundle,
    )


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_fgds_wireless_hil.sh"
CSV_BASE_FIELDS = [
    "row_id",
    "status",
    "predicted_class",
    "logits",
    "preprocess_us",
    "inference_us",
    "total_us",
    "attempts",
    "response_timeout_count",
    "ignored_datagram_count",
]
CSV_IGNORED_FIELDS = [
    f"ignored_{category}_count" for category in IGNORED_DATAGRAM_CATEGORIES
]
CSV_TRAILING_FIELDS = [
    "successful_request_bytes",
    "request_datagram_bytes_sent",
    "response_bytes",
    "host_observed_datagram_rtt_us",
    "transaction_elapsed_us",
]
CSV_FIELDS = CSV_BASE_FIELDS + CSV_IGNORED_FIELDS + CSV_TRAILING_FIELDS


def _control_evidence(result: object) -> dict[str, Any]:
    return result.evidence()  # type: ignore[attr-defined]


def _data_network_counters(responses: list[dict[str, object]]) -> dict[str, int]:
    return {
        "successful_transactions": len(responses),
        "datagrams_sent": sum(int(row["attempts"]) for row in responses),
        "retransmissions": sum(int(row["attempts"]) - 1 for row in responses),
        "response_timeouts": sum(
            int(row["response_timeout_count"]) for row in responses
        ),
        "ignored_datagrams": sum(
            int(row["ignored_datagram_count"]) for row in responses
        ),
        "request_bytes_sent": sum(
            int(row["request_datagram_bytes_sent"]) for row in responses
        ),
        "response_bytes_received": sum(
            int(row["response_bytes"]) for row in responses
        ),
    }


def write_results(
    *,
    output_csv: Path,
    summary_json: Path,
    responses: list[dict[str, object]],
    expected_ids: list[int],
    stage_contract: dict[str, Any],
    network_counters: dict[str, int],
    controls: dict[str, Any],
    provenance: dict[str, Any],
    error: dict[str, Any] | None,
) -> dict[str, Any]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_temporary = output_csv.with_suffix(output_csv.suffix + ".tmp")
    with csv_temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for response in responses:
            row = dict(response)
            row["logits"] = " ".join(str(value) for value in row["logits"])
            writer.writerow(row)
    os.replace(csv_temporary, output_csv)

    sequence = verify_response_sequence(expected_ids, responses)
    end_record = controls.get("end")
    passed = (
        sequence["expected"] == sequence["completed"]
        and not sequence["missing"]
        and not sequence["duplicates"]
        and not sequence["unexpected"]
        and sequence["status_counts"] == {"OK": sequence["expected"]}
        and isinstance(end_record, dict)
        and end_record.get("device_stage_counters", {}).get("inferences")
        == sequence["expected"]
        and error is None
    )
    summary = {
        **sequence,
        "status": "passed" if passed else "failed",
        "protocol_id": WIRELESS_PROTOCOL_ID,
        "stage_contract": stage_contract,
        "output_csv": str(output_csv),
        "output_csv_sha256": sha256_file(output_csv),
        "data_network_counters": _data_network_counters(responses),
        "session_network_counters": network_counters,
        "control_transactions": controls,
        "provenance": provenance,
        "error": error,
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    json_temporary = summary_json.with_suffix(summary_json.suffix + ".tmp")
    json_temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    os.replace(json_temporary, summary_json)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--vectors-csv", type=Path, required=True)
    parser.add_argument("--connection-json", type=Path, required=True)
    parser.add_argument("--stage-name", choices=sorted(REQUIRED_WIRELESS_STAGES), required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    if args.timeout <= 0 or not 1 <= args.max_attempts <= 255:
        raise ValueError("Timeout must be positive and max attempts must be in 1..255")

    generated_dir = args.generated_dir.resolve()
    bundle_dir = args.bundle_dir.resolve()
    output_csv = args.output_csv.resolve()
    summary_json = args.summary_json.resolve()
    connection_path = args.connection_json.resolve()
    validate_output_paths(output_csv, summary_json, [generated_dir, bundle_dir])

    export_manifest = verify_export_for_wireless(generated_dir)
    bundle_manifest = verify_wireless_bundle(bundle_dir, export_manifest)
    connection = validate_connection_record(
        connection_path,
        bundle_manifest,
        generated_dir / "strict_export_manifest.json",
    )
    vector_hash = require_export_file_hash(
        export_manifest,
        generated_dir,
        args.vectors_csv.resolve(),
    )
    frozen_stage = REQUIRED_WIRELESS_STAGES[args.stage_name]
    expected_rows = int(frozen_stage["rows"])
    stage_ordinal = int(frozen_stage["ordinal"])
    rows = load_vectors(args.vectors_csv.resolve(), limit=expected_rows)
    validate_replay_rows(rows)
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Stage {args.stage_name} requires {expected_rows} rows, got {len(rows)}"
        )

    stage_id = secrets.token_hex(8).upper()
    stage_contract = {
        "name": args.stage_name,
        "stage_id": stage_id,
        "ordinal": stage_ordinal,
        "expected_rows": expected_rows,
    }
    prepared_rows: list[tuple[int, str]] = []
    for row in rows:
        row_id = int(row["row_id"])
        inner_request = encode_request_line(
            row_id,
            list(row["features"]),
        ).rstrip("\n")
        if "\r" in inner_request or "\n" in inner_request:
            raise RuntimeError("Inner request contains an unexpected line terminator")
        transaction_id = f"{row_id:016X}"
        encode_wireless_envelope(
            prefix=REQUEST_ENVELOPE_PREFIX,
            session_id=str(connection["session_id"]),
            stage_id=stage_id,
            transaction_id=transaction_id,
            attempt=args.max_attempts,
            inner_text=inner_request,
        )
        prepared_rows.append((row_id, inner_request))
    session = StrictUdpSession(
        device_ip=str(connection["device_ip"]),
        device_port=DEFAULT_DEVICE_UDP_PORT,
        host_port=DEFAULT_HOST_UDP_PORT,
        session_id=str(connection["session_id"]),
        timeout_seconds=args.timeout,
        max_attempts=args.max_attempts,
    )
    responses: list[dict[str, object]] = []
    controls: dict[str, Any] = {}
    error: dict[str, Any] | None = None
    begin_attempted = False
    stage_closed = False
    observed_identity: str | None = None
    try:
        controls["stale_datagrams_drained_before_identity"] = (
            session.drain_stale_datagrams()
        )
        identity = session.exchange(
            stage_id=stage_id,
            transaction_id=IDENTITY_TRANSACTION_ID,
            inner_text="CUKDWID?",
        )
        observed_identity = identity.inner_text
        expected_identity = expected_device_identity(bundle_manifest)
        if observed_identity != expected_identity:
            raise RuntimeError(
                f"UDP firmware identity is {observed_identity!r}; "
                f"expected {expected_identity!r}"
            )
        controls["identity"] = _control_evidence(identity)

        begin_attempted = True
        begin = session.exchange(
            stage_id=stage_id,
            transaction_id=BEGIN_TRANSACTION_ID,
            inner_text=(
                f"CUKDWBEGIN,{stage_id},{stage_ordinal},{expected_rows}"
            ),
        )
        begin_parsed = parse_begin_response(
            begin.inner_text,
            stage_id=stage_id,
            ordinal=stage_ordinal,
            expected_rows=expected_rows,
        )
        controls["begin"] = {
            **_control_evidence(begin),
            "device_acknowledgement": begin_parsed,
        }
        for row_id, inner_request in prepared_rows:
            exchange = session.exchange(
                stage_id=stage_id,
                transaction_id=f"{row_id:016X}",
                inner_text=inner_request,
            )
            decoded = decode_response_line(exchange.inner_text)
            if int(decoded["row_id"]) != row_id:
                raise RuntimeError(
                    f"Device returned row {decoded['row_id']} for request {row_id}"
                )
            responses.append({**decoded, **exchange.evidence()})

        end = session.exchange(
            stage_id=stage_id,
            transaction_id=END_TRANSACTION_ID,
            inner_text=f"CUKDWEND,{stage_id},{len(responses)}",
        )
        end_parsed = parse_end_response(
            end.inner_text,
            stage_id=stage_id,
            ordinal=stage_ordinal,
            expected_rows=expected_rows,
        )
        controls["end"] = {
            **_control_evidence(end),
            "device_stage_counters": end_parsed,
        }
        stage_closed = True
    except Exception as exc:
        error = {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "completed_before_error": len(responses),
        }
        if begin_attempted and not stage_closed:
            abort_errors: list[dict[str, Any]] = []
            candidates = [len(responses)]
            if len(responses) < expected_rows:
                candidates.append(len(responses) + 1)
            for completed_candidate in candidates:
                try:
                    abort = session.exchange(
                        stage_id=stage_id,
                        transaction_id=ABORT_TRANSACTION_ID,
                        inner_text=(
                            f"CUKDWABORT,{stage_id},{completed_candidate}"
                        ),
                    )
                    controls["abort"] = {
                        **_control_evidence(abort),
                        "device_acknowledgement": parse_abort_response(
                            abort.inner_text,
                            stage_id=stage_id,
                            completed_rows=completed_candidate,
                            expected_rows=expected_rows,
                        ),
                    }
                    stage_closed = True
                    break
                except Exception as abort_exc:
                    abort_errors.append(
                        {
                            "completed_candidate": completed_candidate,
                            "type": abort_exc.__class__.__name__,
                            "message": str(abort_exc),
                        }
                    )
            if not stage_closed:
                error["abort_errors"] = abort_errors
    finally:
        network_counters = session.counter_evidence()
        session.close()

    summary = write_results(
        output_csv=output_csv,
        summary_json=summary_json,
        responses=responses,
        expected_ids=[int(row["row_id"]) for row in rows],
        stage_contract=stage_contract,
        network_counters=network_counters,
        controls=controls,
        provenance={
            "transport": "IEEE 802.11 station mode with UDP datagrams",
            "application_protocol": WIRELESS_PROTOCOL_ID,
            "board": bundle_manifest["board"],
            "student": bundle_manifest["student"],
            "export_id": bundle_manifest["export_id"],
            "wireless_bundle_id": bundle_manifest["wireless_bundle_id"],
            "session_id": connection["session_id"],
            "device_identity": observed_identity,
            "device_endpoint": (
                f"{connection['device_ip']}:{DEFAULT_DEVICE_UDP_PORT}"
            ),
            "host_udp_port": DEFAULT_HOST_UDP_PORT,
            "vector_sha256": vector_hash,
            "strict_export_manifest_sha256": sha256_file(
                generated_dir / "strict_export_manifest.json"
            ),
            "strict_export_report_sha256": require_export_file_hash(
                export_manifest,
                generated_dir,
                generated_dir / "strict_export_report.json",
            ),
            "wireless_bundle_manifest_sha256": bundle_manifest[
                "_manifest_sha256"
            ],
            "connection_json_sha256": sha256_file(connection_path),
            "stream_script_sha256": sha256_file(Path(__file__).resolve()),
            "udp_session_sha256": sha256_file(
                Path(__file__).with_name("udp_session.py")
            ),
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
            "timeout_seconds": args.timeout,
            "max_attempts": args.max_attempts,
            "python": sys.version,
            "timing_boundary": (
                "host_observed_datagram_rtt_us uses the host monotonic clock from "
                "immediately before the successful-attempt sendto call until immediately "
                "after the matching recvfrom call. transaction_elapsed_us also includes "
                "earlier timed-out attempts. Firmware total_us measures integer "
                "preprocessing plus fixed-point inference in a bounded MCU code region; "
                "its wall-clock interval may include interrupt preemption. Their "
                "difference is not claimed as pure wireless latency."
            ),
        },
        error=error,
    )
    print(summary_json)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
