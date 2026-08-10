"""Shared provenance and protocol helpers for the FG-DS Wi-Fi UDP HIL path."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deployment.hardware_hil.host.hil_common import crc16_ccitt
from deployment.hardware_hil.host.stream_vectors_fgds_strict import verify_export


WIRELESS_PROTOCOL_ID = "cukd_fgds_wifi_udp_session_v2"
WIRELESS_COMPLETION_PROTOCOL_ID = "fgds_seed42_wifi_udp_three_stage_v2"
REQUIRED_WIRELESS_STAGES = {
    "smoke_10": {"ordinal": 1, "rows": 10},
    "validation_1000": {"ordinal": 2, "rows": 1000},
    "full_56301": {"ordinal": 3, "rows": 56301},
}
SUPPORTED_BOARDS = {"esp32c3", "arduino_r4"}
DEFAULT_DEVICE_UDP_PORT = 42101
DEFAULT_HOST_UDP_PORT = 42102
MAX_DATAGRAM_BYTES = 768
SESSION_HEX_LENGTH = 32
STAGE_HEX_LENGTH = 16
TRANSACTION_HEX_LENGTH = 16
MAX_ATTEMPTS = 255
IDENTITY_TRANSACTION_ID = "FFFFFFFFFFFFFFFC"
ABORT_TRANSACTION_ID = "FFFFFFFFFFFFFFFD"
BEGIN_TRANSACTION_ID = "FFFFFFFFFFFFFFFE"
END_TRANSACTION_ID = "FFFFFFFFFFFFFFFF"
REQUEST_ENVELOPE_PREFIX = "CUKDW2Q"
RESPONSE_ENVELOPE_PREFIX = "CUKDW2R"
SERIAL_IDENTITY_QUERY = "CUKDWID?"
SERIAL_IDENTITY_PREFIX = "CUKDWBUILD"
UDP_IDENTITY_QUERY = "CUKDWID?"
CONFIG_PREFIX = "CUKDWCFG2"
CONFIG_RESPONSE_PREFIX = "CUKDWCFG2R"

MODEL_COMMON_FILES = {
    "cukd_model.h",
    "cukd_model.c",
    "cukd_preprocess.h",
    "cukd_preprocess.c",
    "cukd_protocol.h",
    "cukd_protocol.c",
}
WIRELESS_COMMON_FILES = {
    "cukd_wifi_config.h",
    "cukd_wifi_config.c",
    "cukd_wifi_envelope.h",
    "cukd_wifi_envelope.c",
}
GENERATED_FILES = {
    "model_weights.h",
    "preprocess_int_metadata.h",
    "cukd_export_identity.h",
}
WIRELESS_BUNDLE_FILES = (
    MODEL_COMMON_FILES
    | WIRELESS_COMMON_FILES
    | GENERATED_FILES
    | {"cukd_wireless_bundle_identity.h"}
)

_HEX_PATTERNS = {
    SESSION_HEX_LENGTH: re.compile(rf"[0-9A-F]{{{SESSION_HEX_LENGTH}}}\Z"),
    STAGE_HEX_LENGTH: re.compile(rf"[0-9A-F]{{{STAGE_HEX_LENGTH}}}\Z"),
    TRANSACTION_HEX_LENGTH: re.compile(
        rf"[0-9A-F]{{{TRANSACTION_HEX_LENGTH}}}\Z"
    ),
}
_ANSI_ESCAPE_PATTERN = re.compile(
    r"(?:\x1B\][^\x07]*(?:\x07|\x1B\\))|(?:\x1B\[[0-?]*[ -/]*[@-~])"
)


class EnvelopeError(ValueError):
    """Protocol error carrying a stable counter category."""

    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category


@dataclass(frozen=True)
class WirelessEnvelope:
    prefix: str
    session_id: str
    stage_id: str
    transaction_id: str
    attempt: int
    inner_text: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def read_compile_log_text(path: Path) -> str:
    """Decode a preserved Arduino CLI log without silently replacing bytes."""
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = path.read_bytes()
    if payload.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        raise RuntimeError(f"Unsupported UTF-32 compile log: {path}")
    try:
        if payload.startswith((b"\xff\xfe", b"\xfe\xff")):
            text = payload.decode("utf-16")
        else:
            text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            f"Compile log is not valid BOM-marked UTF-16 or UTF-8: {path}"
        ) from exc
    if "\x00" in text:
        raise RuntimeError(f"Compile log contains an embedded NUL character: {path}")
    return _ANSI_ESCAPE_PATTERN.sub("", text)


def validate_compile_log_metadata(
    text: str,
    *,
    fqbn: str,
    board_core_version: str,
    frontend_version: str,
    toolchain_version: str,
) -> None:
    observed_lines = set(text.splitlines())
    required = {
        f"CUKD_FQBN={fqbn}",
        f"CUKD_BOARD_CORE_VERSION={board_core_version}",
        f"CUKD_FRONTEND_VERSION={frontend_version}",
        f"CUKD_TOOLCHAIN_VERSION={toolchain_version}",
    }
    missing = sorted(required - observed_lines)
    if missing:
        raise RuntimeError(f"Compile log lacks exact provenance markers: {missing}")


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _require_hex(value: str, length: int, label: str) -> str:
    if not isinstance(value, str) or _HEX_PATTERNS[length].fullmatch(value) is None:
        raise ValueError(f"{label} must be exactly {length} uppercase hexadecimal digits")
    return value


def _require_printable_ascii(text: str, label: str) -> bytes:
    try:
        encoded = text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} must be ASCII") from exc
    if not encoded or any(value < 0x20 or value > 0x7E for value in encoded):
        raise ValueError(f"{label} must contain printable ASCII without CR/LF/NUL")
    return encoded


def _crc_packet(body: str) -> str:
    return f"{body},{crc16_ccitt(body.encode('ascii')):04X}"


def encode_wireless_envelope(
    *,
    prefix: str,
    session_id: str,
    stage_id: str,
    transaction_id: str,
    attempt: int,
    inner_text: str,
) -> bytes:
    if prefix not in {REQUEST_ENVELOPE_PREFIX, RESPONSE_ENVELOPE_PREFIX}:
        raise ValueError("Unsupported wireless envelope prefix")
    _require_hex(session_id, SESSION_HEX_LENGTH, "session_id")
    _require_hex(stage_id, STAGE_HEX_LENGTH, "stage_id")
    _require_hex(transaction_id, TRANSACTION_HEX_LENGTH, "transaction_id")
    if not 1 <= int(attempt) <= MAX_ATTEMPTS:
        raise ValueError(f"attempt must be in 1..{MAX_ATTEMPTS}")
    inner = _require_printable_ascii(inner_text, "inner_text")
    body = ",".join(
        [
            prefix,
            session_id,
            stage_id,
            transaction_id,
            str(int(attempt)),
            inner.hex().upper(),
        ]
    )
    packet = _crc_packet(body).encode("ascii")
    if len(packet) > MAX_DATAGRAM_BYTES:
        raise ValueError(
            f"Wireless envelope is {len(packet)} bytes; limit is {MAX_DATAGRAM_BYTES}"
        )
    return packet


def decode_wireless_envelope(
    payload: bytes,
    *,
    expected_prefix: str | None = None,
) -> WirelessEnvelope:
    if not payload or len(payload) > MAX_DATAGRAM_BYTES:
        raise EnvelopeError("bad_length", "Datagram length is outside the protocol limit")
    if any(value < 0x20 or value > 0x7E for value in payload):
        raise EnvelopeError(
            "non_ascii", "Datagram contains NUL, control, or non-ASCII bytes"
        )
    text = payload.decode("ascii")
    parts = text.split(",")
    if len(parts) != 7:
        raise EnvelopeError("malformed", "Wireless envelope must contain seven fields")
    prefix, session_id, stage_id, transaction_id, attempt_text, inner_hex, crc_text = (
        parts
    )
    if prefix not in {REQUEST_ENVELOPE_PREFIX, RESPONSE_ENVELOPE_PREFIX}:
        raise EnvelopeError("bad_prefix", "Wireless envelope prefix is invalid")
    if expected_prefix is not None and prefix != expected_prefix:
        raise EnvelopeError("bad_prefix", "Wireless envelope direction is invalid")
    try:
        _require_hex(session_id, SESSION_HEX_LENGTH, "session_id")
        _require_hex(stage_id, STAGE_HEX_LENGTH, "stage_id")
        _require_hex(transaction_id, TRANSACTION_HEX_LENGTH, "transaction_id")
    except ValueError as exc:
        raise EnvelopeError("bad_identity", str(exc)) from exc
    try:
        attempt = int(attempt_text, 10)
    except ValueError as exc:
        raise EnvelopeError("bad_attempt", "Envelope attempt is not decimal") from exc
    if str(attempt) != attempt_text or not 1 <= attempt <= MAX_ATTEMPTS:
        raise EnvelopeError("bad_attempt", "Envelope attempt is not canonical or in range")
    if not inner_hex or len(inner_hex) % 2 or re.fullmatch(r"[0-9A-F]+", inner_hex) is None:
        raise EnvelopeError("bad_inner_hex", "Envelope payload is not canonical hex")
    if re.fullmatch(r"[0-9A-F]{4}", crc_text) is None:
        raise EnvelopeError("bad_crc", "Envelope CRC is not four uppercase hex digits")
    body = ",".join(parts[:-1])
    expected_crc = crc16_ccitt(body.encode("ascii"))
    if int(crc_text, 16) != expected_crc:
        raise EnvelopeError("bad_crc", "Envelope CRC does not match")
    try:
        inner_bytes = bytes.fromhex(inner_hex)
    except ValueError as exc:  # guarded above; retained as fail-closed defense
        raise EnvelopeError("bad_inner_hex", "Envelope payload hex cannot be decoded") from exc
    if not inner_bytes or any(value < 0x20 or value > 0x7E for value in inner_bytes):
        raise EnvelopeError(
            "bad_inner_text", "Inner payload contains NUL, control, or non-ASCII bytes"
        )
    return WirelessEnvelope(
        prefix=prefix,
        session_id=session_id,
        stage_id=stage_id,
        transaction_id=transaction_id,
        attempt=attempt,
        inner_text=inner_bytes.decode("ascii"),
    )


def verify_inventory(root: Path, manifest: dict[str, Any], manifest_path: Path) -> None:
    root = root.resolve()
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError(f"Wireless bundle manifest has no inventory: {manifest_path}")
    if manifest.get("file_count_excluding_manifest") != len(files):
        raise RuntimeError("Wireless bundle file count differs from its inventory")
    seen: set[str] = set()
    for item in files:
        relative = item.get("path")
        if not isinstance(relative, str) or not relative:
            raise RuntimeError("Wireless bundle inventory contains an invalid path")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(f"Wireless bundle path escapes its root: {relative}")
        normalized = relative_path.as_posix()
        if normalized in seen:
            raise RuntimeError(f"Wireless bundle inventory duplicates {relative}")
        seen.add(normalized)
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"Wireless bundle path escapes its root: {relative}") from exc
        if not path.is_file() or path.stat().st_size != item.get("size_bytes"):
            raise RuntimeError(f"Wireless bundle file is missing or changed: {path}")
        if sha256_file(path) != item.get("sha256"):
            raise RuntimeError(f"Wireless bundle SHA-256 mismatch: {path}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.resolve() != manifest_path.resolve()
    }
    if actual != seen:
        raise RuntimeError(
            "Wireless bundle inventory differs from files on disk: "
            f"missing={sorted(seen - actual)}, unexpected={sorted(actual - seen)}"
        )


def expected_device_identity(manifest: dict[str, Any]) -> str:
    return ",".join(
        [
            SERIAL_IDENTITY_PREFIX,
            manifest["student"],
            manifest["export_id"],
            manifest["wireless_bundle_id"],
            manifest["board"],
            WIRELESS_PROTOCOL_ID,
        ]
    )


def verify_wireless_bundle(
    bundle_dir: Path,
    export_manifest: dict[str, Any],
) -> dict[str, Any]:
    bundle_dir = bundle_dir.resolve()
    manifest_path = bundle_dir / "wireless_bundle_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "passed":
        raise RuntimeError("Wireless bundle status is not passed")
    expected_top_level = {
        "protocol_id": WIRELESS_PROTOCOL_ID,
        "student": export_manifest.get("student"),
        "export_id": export_manifest.get("export_id"),
        "device_udp_port": DEFAULT_DEVICE_UDP_PORT,
        "max_datagram_bytes": MAX_DATAGRAM_BYTES,
    }
    for key, expected in expected_top_level.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"Wireless bundle differs for {key}")
    if manifest.get("board") not in SUPPORTED_BOARDS:
        raise RuntimeError("Wireless bundle has an unsupported board")
    identity = manifest.get("wireless_bundle_identity_payload")
    bundle_id = manifest.get("wireless_bundle_id")
    if not isinstance(identity, dict) or not isinstance(bundle_id, str) or len(bundle_id) != 64:
        raise RuntimeError("Wireless bundle lacks a valid identity payload")
    if canonical_json_sha256(identity) != bundle_id:
        raise RuntimeError("Wireless bundle ID is not derived from its identity payload")
    for key in [
        "protocol_id",
        "board",
        "student",
        "export_id",
        "device_udp_port",
        "max_datagram_bytes",
        "strict_export_manifest_sha256",
    ]:
        if identity.get(key) != manifest.get(key):
            raise RuntimeError(f"Wireless bundle identity differs for {key}")
    export_manifest_path = Path(export_manifest["_manifest_path"])
    if manifest.get("strict_export_manifest_sha256") != sha256_file(export_manifest_path):
        raise RuntimeError("Wireless bundle was built from another strict export manifest")
    bundler = Path(__file__).with_name("prepare_fgds_wireless_bundle.py")
    if manifest.get("bundler_sha256") != sha256_file(bundler):
        raise RuntimeError("Wireless bundle was created by another bundler implementation")
    if identity.get("bundler_sha256") != manifest.get("bundler_sha256"):
        raise RuntimeError("Wireless identity and manifest bundler hashes differ")

    verify_inventory(bundle_dir, manifest, manifest_path)
    sketch = manifest.get("sketch_file")
    if not isinstance(sketch, str) or Path(sketch).name != sketch:
        raise RuntimeError("Wireless bundle sketch name is invalid")
    expected_files = {*WIRELESS_BUNDLE_FILES, sketch}
    observed_files = {item["path"] for item in manifest["files"]}
    if observed_files != expected_files:
        raise RuntimeError(
            "Wireless bundle has the wrong source set: "
            f"missing={sorted(expected_files - observed_files)}, "
            f"unexpected={sorted(observed_files - expected_files)}"
        )

    source_files = identity.get("source_files")
    if not isinstance(source_files, list):
        raise RuntimeError("Wireless bundle identity has no source inventory")
    source_records: dict[str, dict[str, Any]] = {}
    for item in source_files:
        name = item.get("name")
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not isinstance(item.get("sha256"), str)
            or not isinstance(item.get("size_bytes"), int)
            or item["size_bytes"] < 0
            or name in source_records
        ):
            raise RuntimeError("Wireless bundle identity has an invalid source entry")
        source_records[name] = item
    expected_source_names = (
        MODEL_COMMON_FILES | WIRELESS_COMMON_FILES | GENERATED_FILES | {"cukd_wireless_fgds.ino"}
    )
    if set(source_records) != expected_source_names:
        raise RuntimeError("Wireless bundle identity has the wrong source inventory")
    for name, record in source_records.items():
        bundled_name = sketch if name == "cukd_wireless_fgds.ino" else name
        bundled_path = bundle_dir / bundled_name
        if (
            bundled_path.stat().st_size != record["size_bytes"]
            or sha256_file(bundled_path) != record["sha256"]
        ):
            raise RuntimeError(f"Wireless bundle source identity differs for {name}")

    export_hashes = {
        item["path"]: item["sha256"] for item in export_manifest.get("files", [])
    }
    for name in GENERATED_FILES:
        if sha256_file(bundle_dir / name) != export_hashes.get(name):
            raise RuntimeError(f"Wireless bundle changed strict export input: {name}")
    tested_common = export_manifest.get("_verified_report", {}).get(
        "provenance", {}
    ).get("firmware_common_files")
    if not isinstance(tested_common, dict):
        raise RuntimeError("Strict export lacks host-tested model-core identities")
    for name, expected_hash in tested_common.items():
        if sha256_file(bundle_dir / name) != expected_hash:
            raise RuntimeError(f"Wireless model core differs from host-tested code: {name}")

    board_macro = (
        "CUKD_WIRELESS_BOARD_ESP32C3"
        if manifest["board"] == "esp32c3"
        else "CUKD_WIRELESS_BOARD_ARDUINO_R4"
    )
    expected_header = (
        "#ifndef CUKD_WIRELESS_BUNDLE_IDENTITY_H\n"
        "#define CUKD_WIRELESS_BUNDLE_IDENTITY_H\n"
        f'#define CUKD_WIRELESS_BUNDLE_ID "{bundle_id}"\n'
        f'#define CUKD_WIRELESS_PROTOCOL_ID "{WIRELESS_PROTOCOL_ID}"\n'
        f'#define CUKD_WIRELESS_BOARD_ID "{manifest["board"]}"\n'
        f"#define CUKD_WIRELESS_UDP_PORT {DEFAULT_DEVICE_UDP_PORT}u\n"
        f"#define CUKD_WIRELESS_MAX_DATAGRAM {MAX_DATAGRAM_BYTES}u\n"
        f"#define {board_macro} 1\n"
        "#endif\n"
    )
    if (bundle_dir / "cukd_wireless_bundle_identity.h").read_text(
        encoding="ascii"
    ) != expected_header:
        raise RuntimeError("Wireless bundle identity header differs from its manifest")
    manifest["_manifest_path"] = str(manifest_path)
    manifest["_manifest_sha256"] = sha256_file(manifest_path)
    return manifest


def verify_export_for_wireless(generated_dir: Path) -> dict[str, Any]:
    generated_dir = generated_dir.resolve()
    manifest = verify_export(generated_dir)
    manifest["_manifest_path"] = str(generated_dir / "strict_export_manifest.json")
    return manifest


def validate_connection_record(
    path: Path,
    bundle: dict[str, Any],
    export_manifest_path: Path,
) -> dict[str, Any]:
    connection = read_json(path.resolve())
    expected = {
        "status": "connected",
        "protocol_id": WIRELESS_PROTOCOL_ID,
        "transport": "wifi_udp",
        "board": bundle["board"],
        "student": bundle["student"],
        "export_id": bundle["export_id"],
        "wireless_bundle_id": bundle["wireless_bundle_id"],
        "device_identity": expected_device_identity(bundle),
        "device_udp_port": DEFAULT_DEVICE_UDP_PORT,
        "host_udp_port": DEFAULT_HOST_UDP_PORT,
        "strict_export_manifest_sha256": sha256_file(export_manifest_path),
        "wireless_bundle_manifest_sha256": bundle["_manifest_sha256"],
        "serial_closed_before_udp_replay": True,
        "provisioning_script_sha256": sha256_file(
            Path(__file__).with_name("configure_wifi_serial.py")
        ),
    }
    for key, value in expected.items():
        if connection.get(key) != value:
            raise RuntimeError(f"Wireless connection evidence differs for {key}")
    device_ip = connection.get("device_ip")
    try:
        parsed_ip = ipaddress.ip_address(device_ip)
    except ValueError as exc:
        raise RuntimeError("Wireless connection evidence has an invalid device IP") from exc
    if parsed_ip.version != 4 or parsed_ip.is_unspecified or parsed_ip.is_multicast:
        raise RuntimeError("Wireless device endpoint must be a unicast IPv4 address")
    try:
        _require_hex(connection.get("session_id"), SESSION_HEX_LENGTH, "session_id")
    except ValueError as exc:
        raise RuntimeError("Wireless connection evidence has an invalid session") from exc
    return connection


def encode_wifi_config_line(
    ssid: str,
    password: str,
    udp_port: int,
    session_id: str,
) -> str:
    _require_hex(session_id, SESSION_HEX_LENGTH, "session_id")
    ssid_bytes = _require_printable_ascii(ssid, "SSID")
    password_bytes = _require_printable_ascii(password, "password")
    if not 1 <= len(ssid_bytes) <= 32:
        raise ValueError("SSID must contain 1-32 printable ASCII bytes")
    if not 8 <= len(password_bytes) <= 63:
        raise ValueError("Password must contain 8-63 printable ASCII bytes")
    if not 1 <= int(udp_port) <= 65535:
        raise ValueError("UDP port must be in 1..65535")
    body = ",".join(
        [
            CONFIG_PREFIX,
            session_id,
            str(int(udp_port)),
            ssid_bytes.hex().upper(),
            password_bytes.hex().upper(),
        ]
    )
    return _crc_packet(body) + "\n"


def decode_wifi_config_response(line: str) -> dict[str, Any]:
    stripped = line.rstrip("\r\n")
    if not stripped or line not in {stripped, stripped + "\n", stripped + "\r\n"}:
        raise ValueError("Wi-Fi response has invalid leading or trailing framing")
    try:
        encoded = stripped.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("Wi-Fi response is not ASCII") from exc
    if any(value < 0x20 or value > 0x7E for value in encoded):
        raise ValueError("Wi-Fi response contains a control byte")
    parts = stripped.split(",")
    if len(parts) != 9 or parts[0] != CONFIG_RESPONSE_PREFIX:
        raise ValueError("Unexpected Wi-Fi configuration response")
    body = ",".join(parts[:-1])
    if re.fullmatch(r"[0-9A-F]{4}", parts[-1]) is None:
        raise ValueError("Wi-Fi response checksum is not canonical hex")
    expected = crc16_ccitt(body.encode("ascii"))
    if int(parts[-1], 16) != expected:
        raise ValueError("Wi-Fi response checksum does not match")
    _require_hex(parts[1], SESSION_HEX_LENGTH, "session_id")
    try:
        port = int(parts[4], 10)
        rssi_dbm = int(parts[5], 10)
        if (
            not parts[6]
            or len(parts[6]) % 2
            or re.fullmatch(r"[0-9A-F]+", parts[6]) is None
        ):
            raise ValueError("connectivity firmware is not canonical uppercase hex")
        firmware = bytes.fromhex(parts[6]).decode("ascii")
        mac = parts[7]
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("Wi-Fi response contains an invalid numeric/hex field") from exc
    if not 0 <= port <= 65535 or str(port) != parts[4]:
        raise ValueError("Wi-Fi response port is invalid")
    if re.fullmatch(r"[A-Z0-9_]+", parts[2]) is None:
        raise ValueError("Wi-Fi response status is invalid")
    _require_printable_ascii(firmware, "connectivity_firmware")
    if re.fullmatch(r"(?:[0-9A-F]{2}:){5}[0-9A-F]{2}", mac) is None:
        raise ValueError("Wi-Fi response MAC address is invalid")
    if parts[2] == "OK":
        parsed_ip = ipaddress.ip_address(parts[3])
        if parsed_ip.version != 4 or parsed_ip.is_unspecified or parsed_ip.is_multicast:
            raise ValueError("Connected Wi-Fi response has an invalid IPv4 endpoint")
        if port == 0:
            raise ValueError("Connected Wi-Fi response has port zero")
        if not -127 <= rssi_dbm <= 0:
            raise ValueError("Connected Wi-Fi response RSSI is outside -127..0 dBm")
    return {
        "session_id": parts[1],
        "status": parts[2],
        "device_ip": parts[3],
        "udp_port": port,
        "rssi_dbm": rssi_dbm,
        "connectivity_firmware": firmware,
        "wifi_mac": mac,
    }
