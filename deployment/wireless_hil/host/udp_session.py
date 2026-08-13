"""Strict session, retry, and timing layer for the FG-DS Wi-Fi UDP HIL path."""

from __future__ import annotations

import ipaddress
import re
import socket
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

try:
    from .wireless_common import (
        MAX_ATTEMPTS,
        MAX_DATAGRAM_BYTES,
        REQUEST_ENVELOPE_PREFIX,
        RESPONSE_ENVELOPE_PREFIX,
        EnvelopeError,
        decode_wireless_envelope,
        encode_wireless_envelope,
    )
except ImportError:
    from wireless_common import (  # type: ignore
        MAX_ATTEMPTS,
        MAX_DATAGRAM_BYTES,
        REQUEST_ENVELOPE_PREFIX,
        RESPONSE_ENVELOPE_PREFIX,
        EnvelopeError,
        decode_wireless_envelope,
        encode_wireless_envelope,
    )


IGNORED_DATAGRAM_CATEGORIES = (
    "wrong_endpoint",
    "bad_length",
    "non_ascii",
    "malformed",
    "bad_prefix",
    "bad_identity",
    "bad_attempt",
    "bad_inner_hex",
    "bad_crc",
    "bad_inner_text",
    "wrong_session",
    "wrong_stage",
    "wrong_transaction",
    "wrong_attempt",
)


class DeviceProtocolError(RuntimeError):
    """A valid matching response that explicitly reports a device-side error."""


@dataclass(frozen=True)
class ExchangeResult:
    inner_text: str
    attempts: int
    response_timeout_count: int
    ignored_by_category: dict[str, int]
    successful_request_bytes: int
    request_datagram_bytes_sent: int
    response_bytes: int
    host_observed_datagram_rtt_us: int
    transaction_elapsed_us: int

    def evidence(self) -> dict[str, Any]:
        ignored = {
            category: int(self.ignored_by_category.get(category, 0))
            for category in IGNORED_DATAGRAM_CATEGORIES
        }
        return {
            "attempts": self.attempts,
            "response_timeout_count": self.response_timeout_count,
            "ignored_datagram_count": sum(ignored.values()),
            **{f"ignored_{key}_count": value for key, value in ignored.items()},
            "successful_request_bytes": self.successful_request_bytes,
            "request_datagram_bytes_sent": self.request_datagram_bytes_sent,
            "response_bytes": self.response_bytes,
            "host_observed_datagram_rtt_us": self.host_observed_datagram_rtt_us,
            "transaction_elapsed_us": self.transaction_elapsed_us,
        }


class StrictUdpSession:
    """One fixed local endpoint and one outstanding stop-and-wait transaction."""

    def __init__(
        self,
        *,
        device_ip: str,
        device_port: int,
        host_port: int,
        session_id: str,
        timeout_seconds: float,
        max_attempts: int,
    ) -> None:
        parsed_ip = ipaddress.ip_address(device_ip)
        if parsed_ip.version != 4 or parsed_ip.is_unspecified or parsed_ip.is_multicast:
            raise ValueError("device_ip must be a unicast IPv4 address")
        if not 1 <= int(device_port) <= 65535 or not 1 <= int(host_port) <= 65535:
            raise ValueError("UDP ports must be in 1..65535")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not 1 <= int(max_attempts) <= MAX_ATTEMPTS:
            raise ValueError(f"max_attempts must be in 1..{MAX_ATTEMPTS}")
        if re.fullmatch(r"[0-9A-F]{32}", session_id) is None:
            raise ValueError("session_id must be exactly 32 uppercase hexadecimal digits")
        self.device_endpoint = (str(parsed_ip), int(device_port))
        self.host_port = int(host_port)
        self.session_id = session_id
        self.timeout_ns = int(float(timeout_seconds) * 1_000_000_000)
        self.max_attempts = int(max_attempts)
        if self.timeout_ns <= 0:
            raise ValueError("timeout_seconds is below monotonic clock resolution")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.socket.bind(("0.0.0.0", self.host_port))
        except Exception:
            self.socket.close()
            raise
        self.counters: Counter[str] = Counter()

    def close(self) -> None:
        self.socket.close()

    def drain_stale_datagrams(self) -> int:
        drained = 0
        self.socket.setblocking(False)
        try:
            while True:
                try:
                    self.socket.recvfrom(MAX_DATAGRAM_BYTES + 1)
                except BlockingIOError:
                    break
                drained += 1
        finally:
            self.socket.setblocking(True)
        self.counters["stale_datagrams_drained_before_identity"] += drained
        return drained

    def exchange(
        self,
        *,
        stage_id: str,
        transaction_id: str,
        inner_text: str,
    ) -> ExchangeResult:
        transaction_start_ns: int | None = None
        response_timeouts = 0
        request_bytes_sent = 0
        ignored: Counter[str] = Counter()

        for attempt in range(1, self.max_attempts + 1):
            request = encode_wireless_envelope(
                prefix=REQUEST_ENVELOPE_PREFIX,
                session_id=self.session_id,
                stage_id=stage_id,
                transaction_id=transaction_id,
                attempt=attempt,
                inner_text=inner_text,
            )
            send_start_ns = time.monotonic_ns()
            if transaction_start_ns is None:
                transaction_start_ns = send_start_ns
            sent = self.socket.sendto(request, self.device_endpoint)
            if sent != len(request):
                raise RuntimeError("UDP send did not accept the complete request datagram")
            request_bytes_sent += sent
            self.counters["datagrams_sent"] += 1
            self.counters["request_bytes_sent"] += sent
            if attempt > 1:
                self.counters["retransmissions"] += 1
            deadline_ns = send_start_ns + self.timeout_ns

            while True:
                remaining_ns = deadline_ns - time.monotonic_ns()
                if remaining_ns <= 0:
                    break
                self.socket.settimeout(max(0.000001, remaining_ns / 1_000_000_000))
                try:
                    response_payload, source = self.socket.recvfrom(
                        MAX_DATAGRAM_BYTES + 1
                    )
                except socket.timeout:
                    break
                response_ns = time.monotonic_ns()
                self.counters["datagrams_received"] += 1
                self.counters["response_bytes_received"] += len(response_payload)
                if source != self.device_endpoint:
                    ignored["wrong_endpoint"] += 1
                    self.counters["ignored_wrong_endpoint"] += 1
                    continue
                try:
                    envelope = decode_wireless_envelope(
                        response_payload,
                        expected_prefix=RESPONSE_ENVELOPE_PREFIX,
                    )
                except EnvelopeError as exc:
                    category = exc.category
                    ignored[category] += 1
                    self.counters[f"ignored_{category}"] += 1
                    continue
                if envelope.session_id != self.session_id:
                    category = "wrong_session"
                elif envelope.stage_id != stage_id:
                    category = "wrong_stage"
                elif envelope.transaction_id != transaction_id:
                    category = "wrong_transaction"
                elif envelope.attempt < attempt:
                    category = "wrong_attempt"
                elif envelope.attempt > attempt:
                    raise RuntimeError(
                        "UDP response carries an attempt number that has not been sent"
                    )
                else:
                    category = ""
                if category:
                    ignored[category] += 1
                    self.counters[f"ignored_{category}"] += 1
                    continue
                if envelope.inner_text.startswith("CUKDWERR,"):
                    self.counters["device_protocol_errors"] += 1
                    raise DeviceProtocolError(envelope.inner_text)
                return ExchangeResult(
                    inner_text=envelope.inner_text,
                    attempts=attempt,
                    response_timeout_count=response_timeouts,
                    ignored_by_category=dict(ignored),
                    successful_request_bytes=len(request),
                    request_datagram_bytes_sent=request_bytes_sent,
                    response_bytes=len(response_payload),
                    host_observed_datagram_rtt_us=(
                        response_ns - send_start_ns
                    ) // 1000,
                    transaction_elapsed_us=(
                        response_ns - transaction_start_ns
                    ) // 1000,
                )
            response_timeouts += 1
            self.counters["response_timeouts"] += 1

        raise TimeoutError(
            f"No matching UDP response for transaction {transaction_id} after "
            f"{self.max_attempts} attempts"
        )

    def counter_evidence(self) -> dict[str, int]:
        required = {
            "stale_datagrams_drained_before_identity",
            "datagrams_sent",
            "request_bytes_sent",
            "retransmissions",
            "datagrams_received",
            "response_bytes_received",
            "response_timeouts",
            "device_protocol_errors",
            *{f"ignored_{category}" for category in IGNORED_DATAGRAM_CATEGORIES},
        }
        return {
            key: int(self.counters.get(key, 0))
            for key in sorted(required | set(self.counters))
        }


def _canonical_uint(text: str, *, minimum: int, maximum: int, label: str) -> int:
    try:
        value = int(text, 10)
    except ValueError as exc:
        raise ValueError(f"{label} is not decimal") from exc
    if str(value) != text or not minimum <= value <= maximum:
        raise ValueError(f"{label} is not canonical or outside range")
    return value


def parse_begin_response(
    text: str,
    *,
    stage_id: str,
    ordinal: int,
    expected_rows: int,
) -> dict[str, Any]:
    parts = text.split(",")
    expected = [
        "CUKDWBEGINR",
        stage_id,
        str(int(ordinal)),
        str(int(expected_rows)),
        "OK",
    ]
    if parts != expected:
        raise ValueError("Stage-begin response differs from the requested contract")
    return {"stage_id": stage_id, "ordinal": ordinal, "expected_rows": expected_rows}


def parse_abort_response(
    text: str,
    *,
    stage_id: str,
    completed_rows: int,
    expected_rows: int,
) -> dict[str, Any]:
    parts = text.split(",")
    if len(parts) != 6 or parts[0] != "CUKDWABORTR" or parts[-1] != "OK":
        raise ValueError("Stage-abort response has an invalid shape")
    if parts[1] != stage_id:
        raise ValueError("Stage-abort response has another stage ID")
    completed = _canonical_uint(
        parts[2], minimum=0, maximum=56301, label="abort completed_rows"
    )
    expected = _canonical_uint(
        parts[3], minimum=1, maximum=56301, label="abort expected_rows"
    )
    inferences = _canonical_uint(
        parts[4], minimum=0, maximum=56301, label="abort inferences"
    )
    if completed != completed_rows or expected != expected_rows or inferences != completed:
        raise ValueError("Stage-abort response counts are inconsistent")
    return {"completed_rows": completed, "expected_rows": expected, "inferences": inferences}


def parse_end_response(
    text: str,
    *,
    stage_id: str,
    ordinal: int,
    expected_rows: int,
) -> dict[str, Any]:
    parts = text.split(",")
    if len(parts) != 18 or parts[0] != "CUKDWENDR" or parts[-1] != "OK":
        raise ValueError("Stage-end response has an invalid shape")
    if parts[1] != stage_id:
        raise ValueError("Stage-end response has another stage ID")
    labels = [
        "completed_rows",
        "expected_rows",
        "received_datagrams",
        "oversized_datagrams",
        "short_reads",
        "bad_envelopes",
        "wrong_sessions",
        "wrong_endpoints",
        "wrong_stages",
        "control_errors",
        "data_errors",
        "duplicate_replays",
        "stale_transactions",
        "inferences",
        "stage_ordinal",
    ]
    values = {
        label: _canonical_uint(
            value,
            minimum=0,
            maximum=(2**32 - 1),
            label=label,
        )
        for label, value in zip(labels, parts[2:-1])
    }
    if (
        values["completed_rows"] != expected_rows
        or values["expected_rows"] != expected_rows
        or values["inferences"] != expected_rows
        or values["stage_ordinal"] != ordinal
    ):
        raise ValueError("Stage-end response does not prove the requested completed stage")
    if values["received_datagrams"] < expected_rows + 2:
        raise ValueError("Stage-end response undercounts begin/data/end datagrams")
    return values
