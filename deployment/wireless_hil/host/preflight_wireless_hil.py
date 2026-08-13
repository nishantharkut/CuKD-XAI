"""Verify one provisioned wireless firmware endpoint before staged replay."""

from __future__ import annotations

import argparse
import json
import os
import platform
import secrets
import sys
from pathlib import Path

try:
    from .udp_session import StrictUdpSession
    from .wireless_common import (
        DEFAULT_DEVICE_UDP_PORT,
        DEFAULT_HOST_UDP_PORT,
        IDENTITY_TRANSACTION_ID,
        WIRELESS_PROTOCOL_ID,
        expected_device_identity,
        sha256_file,
        validate_connection_record,
        verify_export_for_wireless,
        verify_wireless_bundle,
    )
except ImportError:
    from udp_session import StrictUdpSession  # type: ignore
    from wireless_common import (  # type: ignore
        DEFAULT_DEVICE_UDP_PORT,
        DEFAULT_HOST_UDP_PORT,
        IDENTITY_TRANSACTION_ID,
        WIRELESS_PROTOCOL_ID,
        expected_device_identity,
        sha256_file,
        validate_connection_record,
        verify_export_for_wireless,
        verify_wireless_bundle,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--connection-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    if args.timeout <= 0 or not 1 <= args.max_attempts <= 255:
        raise ValueError("Timeout must be positive and max attempts must be in 1..255")

    generated_dir = args.generated_dir.resolve()
    bundle_dir = args.bundle_dir.resolve()
    connection_path = args.connection_json.resolve()
    output = args.output_json.resolve()
    temporary = output.with_suffix(output.suffix + ".tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite preflight evidence: {output}")
    export_manifest = verify_export_for_wireless(generated_dir)
    bundle_manifest = verify_wireless_bundle(bundle_dir, export_manifest)
    connection = validate_connection_record(
        connection_path,
        bundle_manifest,
        generated_dir / "strict_export_manifest.json",
    )

    session = StrictUdpSession(
        device_ip=str(connection["device_ip"]),
        device_port=DEFAULT_DEVICE_UDP_PORT,
        host_port=DEFAULT_HOST_UDP_PORT,
        session_id=str(connection["session_id"]),
        timeout_seconds=args.timeout,
        max_attempts=args.max_attempts,
    )
    stage_id = secrets.token_hex(8).upper()
    try:
        drained = session.drain_stale_datagrams()
        identity = session.exchange(
            stage_id=stage_id,
            transaction_id=IDENTITY_TRANSACTION_ID,
            inner_text="CUKDWID?",
        )
        expected = expected_device_identity(bundle_manifest)
        if identity.inner_text != expected:
            raise RuntimeError(
                f"UDP firmware identity is {identity.inner_text!r}; expected {expected!r}"
            )
        counters = session.counter_evidence()
    finally:
        session.close()

    payload = {
        "status": "passed",
        "protocol_id": WIRELESS_PROTOCOL_ID,
        "board": bundle_manifest["board"],
        "student": bundle_manifest["student"],
        "export_id": bundle_manifest["export_id"],
        "wireless_bundle_id": bundle_manifest["wireless_bundle_id"],
        "session_id": connection["session_id"],
        "device_identity": identity.inner_text,
        "device_endpoint": f"{connection['device_ip']}:{DEFAULT_DEVICE_UDP_PORT}",
        "host_udp_port": DEFAULT_HOST_UDP_PORT,
        "identity_stage_nonce": stage_id,
        "identity_exchange": identity.evidence(),
        "stale_datagrams_drained": drained,
        "network_counters": counters,
        "strict_export_manifest_sha256": sha256_file(
            generated_dir / "strict_export_manifest.json"
        ),
        "wireless_bundle_manifest_sha256": bundle_manifest["_manifest_sha256"],
        "connection_json_sha256": sha256_file(connection_path),
        "preflight_script_sha256": sha256_file(Path(__file__).resolve()),
        "udp_session_sha256": sha256_file(Path(__file__).with_name("udp_session.py")),
        "wireless_common_sha256": sha256_file(
            Path(__file__).with_name("wireless_common.py")
        ),
        "host": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "hostname": platform.node(),
        },
        "boundary": (
            "Preflight proves that the exact hash-bound firmware answered through the "
            "configured Wi-Fi UDP endpoint from the fixed host port. It does not measure "
            "model accuracy, energy, live traffic capture, or credential persistence."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
