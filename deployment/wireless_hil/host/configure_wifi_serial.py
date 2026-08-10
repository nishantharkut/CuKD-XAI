"""Provision Wi-Fi over USB serial without placing credentials in CLI arguments."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import secrets
import sys
import time
from pathlib import Path

try:
    import serial
except ImportError:  # pragma: no cover
    serial = None

try:
    from .wireless_common import (
        CONFIG_RESPONSE_PREFIX,
        DEFAULT_DEVICE_UDP_PORT,
        DEFAULT_HOST_UDP_PORT,
        SERIAL_IDENTITY_PREFIX,
        SERIAL_IDENTITY_QUERY,
        WIRELESS_PROTOCOL_ID,
        decode_wifi_config_response,
        encode_wifi_config_line,
        expected_device_identity,
        sha256_file,
        verify_export_for_wireless,
        verify_wireless_bundle,
    )
except ImportError:
    from wireless_common import (  # type: ignore
        CONFIG_RESPONSE_PREFIX,
        DEFAULT_DEVICE_UDP_PORT,
        DEFAULT_HOST_UDP_PORT,
        SERIAL_IDENTITY_PREFIX,
        SERIAL_IDENTITY_QUERY,
        WIRELESS_PROTOCOL_ID,
        decode_wifi_config_response,
        encode_wifi_config_line,
        expected_device_identity,
        sha256_file,
        verify_export_for_wireless,
        verify_wireless_bundle,
    )


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite connection evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Stale connection evidence temporary exists: {temporary}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    try:
        os.chmod(temporary, 0o600)
    except OSError:
        pass
    os.replace(temporary, path)


def read_matching_line(
    device: object,
    prefix: str,
    deadline: float,
    *,
    expose_observed: bool,
) -> str:
    observed: list[str] = []
    while time.monotonic() < deadline:
        raw = device.readline()
        try:
            line = raw.decode("ascii", errors="strict").strip()
        except UnicodeDecodeError:
            if expose_observed:
                observed.append("<non-ASCII serial line>")
            continue
        if not line:
            continue
        if expose_observed:
            observed.append(line)
        if line.startswith(prefix):
            return line
    suffix = f"; observed={observed[-5:]}" if expose_observed else ""
    raise TimeoutError(f"No {prefix!r} response{suffix}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device-udp-port", type=int, default=DEFAULT_DEVICE_UDP_PORT)
    parser.add_argument("--host-udp-port", type=int, default=DEFAULT_HOST_UDP_PORT)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--connect-timeout", type=float, default=45.0)
    args = parser.parse_args()
    if serial is None:
        raise RuntimeError("pyserial is required")
    if args.timeout <= 0 or args.connect_timeout <= 0 or args.baud <= 0:
        raise ValueError("Baud and timeout values must be positive")
    if args.device_udp_port != DEFAULT_DEVICE_UDP_PORT:
        raise ValueError(
            f"The hash-bound firmware contract fixes device UDP port "
            f"{DEFAULT_DEVICE_UDP_PORT}"
        )
    if not 1 <= args.host_udp_port <= 65535:
        raise ValueError("Host UDP port must be in 1..65535")
    if args.host_udp_port != DEFAULT_HOST_UDP_PORT:
        raise ValueError(
            f"The session/endpoint contract fixes host UDP port "
            f"{DEFAULT_HOST_UDP_PORT}"
        )

    generated_dir = args.generated_dir.resolve()
    bundle_dir = args.bundle_dir.resolve()
    export_manifest = verify_export_for_wireless(generated_dir)
    bundle_manifest = verify_wireless_bundle(bundle_dir, export_manifest)
    expected_identity = expected_device_identity(bundle_manifest)

    ssid = input("Wi-Fi SSID (not recorded): ")
    password = getpass.getpass("Wi-Fi password (not recorded): ")
    session_id = secrets.token_hex(16).upper()
    config_line = encode_wifi_config_line(
        ssid,
        password,
        args.device_udp_port,
        session_id,
    )

    device = serial.Serial(
        port=args.port,
        baudrate=args.baud,
        timeout=args.timeout,
        write_timeout=args.timeout,
    )
    response_line: str
    observed_identity: str
    try:
        time.sleep(2.0)
        device.reset_input_buffer()
        device.reset_output_buffer()
        device.write((SERIAL_IDENTITY_QUERY + "\n").encode("ascii"))
        device.flush()
        observed_identity = read_matching_line(
            device,
            SERIAL_IDENTITY_PREFIX + ",",
            time.monotonic() + max(5.0, args.timeout * 5.0),
            expose_observed=True,
        )
        if observed_identity != expected_identity:
            raise RuntimeError(
                f"Wireless firmware identity is {observed_identity!r}; "
                f"expected {expected_identity!r}"
            )
        device.write(config_line.encode("ascii"))
        device.flush()
        response_line = read_matching_line(
            device,
            CONFIG_RESPONSE_PREFIX + ",",
            time.monotonic() + args.connect_timeout,
            expose_observed=False,
        )
    finally:
        device.close()
        config_line = ""
        password = ""
        ssid = ""

    response = decode_wifi_config_response(response_line)
    if response["session_id"] != session_id:
        raise RuntimeError("Board Wi-Fi response belongs to another run session")
    if response["status"] != "OK":
        raise RuntimeError(f"Board rejected Wi-Fi configuration: {response['status']}")
    if response["udp_port"] != args.device_udp_port:
        raise RuntimeError("Board bound a different UDP port than the firmware contract")

    payload: dict[str, object] = {
        "status": "connected",
        "protocol_id": WIRELESS_PROTOCOL_ID,
        "transport": "wifi_udp",
        "board": bundle_manifest["board"],
        "student": bundle_manifest["student"],
        "device_ip": response["device_ip"],
        "device_udp_port": response["udp_port"],
        "host_udp_port": args.host_udp_port,
        "session_id": session_id,
        "provisioning_serial_endpoint_recorded": args.port,
        "serial_closed_before_udp_replay": True,
        "device_identity": observed_identity,
        "export_id": bundle_manifest["export_id"],
        "wireless_bundle_id": bundle_manifest["wireless_bundle_id"],
        "rssi_dbm_at_connection": response["rssi_dbm"],
        "connectivity_firmware_reported": response["connectivity_firmware"],
        "wifi_mac_reported": response["wifi_mac"],
        "strict_export_manifest_sha256": sha256_file(
            generated_dir / "strict_export_manifest.json"
        ),
        "wireless_bundle_manifest_sha256": bundle_manifest["_manifest_sha256"],
        "provisioning_script_sha256": sha256_file(Path(__file__).resolve()),
        "credential_boundary": (
            "SSID and password were entered interactively and sent over local USB "
            "serial. They are not embedded in firmware source, command arguments, "
            "environment variables, this record, or result artifacts. ESP32 Arduino "
            "persistence is disabled before Wi-Fi initialization. Persistence by the "
            "UNO R4 WiFi connectivity coprocessor is not asserted."
        ),
        "security_boundary": (
            "The random session identifier provides run correlation and endpoint "
            "binding, not cryptographic authentication. The application payload is "
            "not encrypted or authenticated by this experiment."
        ),
        "python": sys.version,
        "pyserial_version": getattr(serial, "__version__", None),
    }
    atomic_write_json(args.output_json.resolve(), payload)
    print(args.output_json.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
