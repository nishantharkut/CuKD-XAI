"""Versioned USB and Wi-Fi bundle preparation for the final HIL campaign."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

from deployment.wireless_hil.host.wireless_common import (
    IDENTITY_TRANSACTION_ID,
    MAX_ATTEMPTS,
    RESPONSE_ENVELOPE_PREFIX,
    encode_wireless_envelope,
)

from .contracts import (
    FinalExportIdentity,
    Verifier,
    _is_sha256,
    _manifest_files,
    _verify_optional_payload_hash,
    canonical_json_sha256,
    read_json,
    sha256_file,
    validate_final_export,
)


BUNDLE_PROTOCOL_ID = "cukd_final_hil_bundle_v1"
RUNTIME_IDENTITY_PREFIX = "CUKDF1"
REPO_ROOT = Path(__file__).resolve().parents[2]
USB_ROOT_RELATIVE = Path("deployment/hardware_hil")
WIFI_ROOT_RELATIVE = Path("deployment/wireless_hil")
USB_ROOT = REPO_ROOT / USB_ROOT_RELATIVE
WIFI_ROOT = REPO_ROOT / WIFI_ROOT_RELATIVE
MODEL_COMMON = USB_ROOT / "firmware" / "common"
WIFI_COMMON = WIFI_ROOT / "firmware" / "common"
MODEL_COMMON_RELATIVE = USB_ROOT_RELATIVE / "firmware" / "common"
WIFI_COMMON_RELATIVE = WIFI_ROOT_RELATIVE / "firmware" / "common"
MODEL_COMMON_FILES = (
    "cukd_model.h",
    "cukd_model.c",
    "cukd_preprocess.h",
    "cukd_preprocess.c",
    "cukd_protocol.h",
    "cukd_protocol.c",
)
WIFI_COMMON_FILES = (
    "cukd_wifi_config.h",
    "cukd_wifi_config.c",
    "cukd_wifi_envelope.h",
    "cukd_wifi_envelope.c",
)
GENERATED_FILES = (
    "model_weights.h",
    "preprocess_int_metadata.h",
    "cukd_export_identity.h",
)
USB_TEMPLATE_RELATIVES = {
    "esp32c3": USB_ROOT_RELATIVE / "firmware" / "esp32c3" / "src" / "main.cpp",
    "arduino_r4": (
        USB_ROOT_RELATIVE
        / "firmware"
        / "arduino_r4"
        / "cukd_hil_r4"
        / "cukd_hil_r4.ino"
    ),
}
USB_TEMPLATES = {
    board: REPO_ROOT / relative for board, relative in USB_TEMPLATE_RELATIVES.items()
}
USB_PROCESS_ANCHORS = {
    "esp32c3": "static void process_line(const char *line) {\n",
    "arduino_r4": "static void cukd_process_line(const char *line) {\n",
}
WIFI_TEMPLATE_RELATIVE = (
    WIFI_ROOT_RELATIVE / "firmware" / "cukd_wireless_fgds" / "cukd_wireless_fgds.ino"
)
WIFI_TEMPLATE = REPO_ROOT / WIFI_TEMPLATE_RELATIVE
TEMPLATE_SNAPSHOT_FILE = "final_source_template.txt"
MAX_RUNTIME_IDENTITY_BYTES = 383
BOARD_TOKENS = {"esp32c3": "c3", "arduino_r4": "r4"}
TRANSPORT_TOKENS = {"usb_serial": "usb", "wifi_udp": "udp"}
BOARD_NAMES = {value: key for key, value in BOARD_TOKENS.items()}
TRANSPORT_NAMES = {value: key for key, value in TRANSPORT_TOKENS.items()}
EXPECTED_BOARD_TARGETS = {
    "esp32c3": {
        "fqbn": "esp32:esp32:esp32c3",
        "platform_id": "esp32:esp32",
    },
    "arduino_r4": {
        "fqbn": "arduino:renesas_uno:unor4wifi",
        "platform_id": "arduino:renesas_uno",
    },
}


def _repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _source_entry(path: Path) -> dict[str, Any]:
    return {
        "origin": _repo_path(path),
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _canonical_source_entry(root: Path, relative: Path) -> dict[str, Any]:
    path = (root.resolve() / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Canonical bundle source escapes its root: {relative}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "origin": relative.as_posix(),
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_build_contract(
    payload_or_path: Mapping[str, Any] | Path,
    *,
    board: str | None = None,
) -> dict[str, Any]:
    payload = (
        read_json(payload_or_path.resolve())
        if isinstance(payload_or_path, Path)
        else dict(payload_or_path)
    )
    recorded_id = payload.pop("build_contract_id", None)
    if payload.get("schema") != "cukd_final_hil_build_contract_v1":
        raise RuntimeError("Unsupported final build-contract schema")
    if payload.get("board") not in USB_TEMPLATES:
        raise RuntimeError("Build contract has an unsupported board")
    if board is not None and payload.get("board") != board:
        raise RuntimeError("Build contract is for another board")
    target = EXPECTED_BOARD_TARGETS[str(payload["board"])]
    for field in [
        "fqbn",
        "platform_id",
        "board_core_version",
        "frontend_version",
        "toolchain_version",
    ]:
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise RuntimeError(f"Build contract field is missing: {field}")
    for field in ["compile_command", "upload_command"]:
        command = payload.get(field)
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(token, str) or not token for token in command)
        ):
            raise RuntimeError(f"Build contract requires a tokenized {field}")
    compile_command = payload["compile_command"]
    upload_command = payload["upload_command"]
    if payload.get("fqbn") != target["fqbn"]:
        raise RuntimeError("Build contract FQBN does not match the physical board target")
    if payload.get("platform_id") != target["platform_id"]:
        raise RuntimeError(
            "Build contract platform ID does not match the physical board target"
        )
    for label, command, subcommand in [
        ("compile", compile_command, "compile"),
        ("upload", upload_command, "upload"),
    ]:
        if Path(command[0]).name.lower() not in {"arduino-cli", "arduino-cli.exe"}:
            raise RuntimeError(f"Build contract {label} must execute arduino-cli directly")
        if len(command) < 2 or command[1] != subcommand:
            raise RuntimeError(f"Build contract {label} subcommand is invalid")
        if any("\n" in token or "\r" in token for token in command):
            raise RuntimeError(f"Build contract {label} contains a multiline token")
    required_compile = {"{bundle}", "{fqbn}", "{build_dir}"}
    required_upload = {"{bundle}", "{fqbn}", "{port}", "{build_dir}"}
    if not required_compile <= set(compile_command):
        raise RuntimeError(
            "Compile command must bind {bundle}, {fqbn}, and {build_dir}"
        )
    if not required_upload <= set(upload_command):
        raise RuntimeError(
            "Upload command must bind {bundle}, {fqbn}, {port}, and {build_dir}"
        )
    if "--output-dir" not in compile_command:
        raise RuntimeError("Compile command must use the sealed {build_dir} output")
    if "--input-dir" not in upload_command:
        raise RuntimeError("Upload command must upload from the sealed {build_dir}")
    for placeholder in required_compile:
        if compile_command.count(placeholder) != 1:
            raise RuntimeError(f"Compile command must bind {placeholder} exactly once")
    for placeholder in required_upload:
        if upload_command.count(placeholder) != 1:
            raise RuntimeError(f"Upload command must bind {placeholder} exactly once")
    allowed_placeholders = required_compile | required_upload
    for label, command in [("compile", compile_command), ("upload", upload_command)]:
        if any(
            ("{" in token or "}" in token) and token not in allowed_placeholders
            for token in command
        ):
            raise RuntimeError(f"Build contract {label} has an unknown placeholder")
    for label, command in [
        ("compile", compile_command),
        ("upload", upload_command),
    ]:
        if not any(token in {"-v", "--verbose"} for token in command):
            raise RuntimeError(f"Build contract {label} command is not verbose")
    if any(
        "CUKD_HIL_VERIFY_PREDICT_WRAPPER" in token for token in compile_command
    ):
        raise RuntimeError(
            "Final timing build cannot enable the second-pass predict wrapper"
        )
    expected_compile = [
        "compile",
        "--fqbn",
        "{fqbn}",
        "--verbose",
        "--output-dir",
        "{build_dir}",
        "{bundle}",
    ]
    expected_upload = [
        "upload",
        "--fqbn",
        "{fqbn}",
        "--port",
        "{port}",
        "--verbose",
        "--input-dir",
        "{build_dir}",
        "{bundle}",
    ]
    if compile_command[1:] != expected_compile:
        raise RuntimeError("Final compile command contains an unapproved argument")
    if upload_command[1:] != expected_upload:
        raise RuntimeError("Final upload command contains an unapproved argument")
    build_id = canonical_json_sha256(payload)
    if recorded_id is not None and recorded_id != build_id:
        raise RuntimeError("Build-contract ID is invalid")
    return {**payload, "build_contract_id": build_id}


def runtime_identity(
    export: FinalExportIdentity,
    *,
    bundle_id: str,
    board: str,
    transport: str,
    build_contract_id: str,
) -> str:
    if board not in USB_TEMPLATES or transport not in {"usb_serial", "wifi_udp"}:
        raise ValueError("Unsupported final HIL board or transport")
    for value in [bundle_id, build_contract_id, export.trained_state_sha256, export.export_id]:
        if not _is_sha256(value):
            raise ValueError("Runtime identity contains a malformed SHA-256 value")
    line = ",".join(
        [
            RUNTIME_IDENTITY_PREFIX,
            export.protocol,
            str(export.seed),
            export.student,
            export.route,
            export.trained_state_sha256,
            export.export_id,
            bundle_id,
            BOARD_TOKENS[board],
            TRANSPORT_TOKENS[transport],
            build_contract_id,
        ]
    )
    encoded = line.encode("ascii")
    if len(encoded) > MAX_RUNTIME_IDENTITY_BYTES:
        raise RuntimeError(
            f"Final runtime identity is {len(encoded)} bytes; firmware limit is "
            f"{MAX_RUNTIME_IDENTITY_BYTES}"
        )
    if transport == "wifi_udp":
        encode_wireless_envelope(
            prefix=RESPONSE_ENVELOPE_PREFIX,
            session_id="F" * 32,
            stage_id="F" * 16,
            transaction_id=IDENTITY_TRANSACTION_ID,
            attempt=MAX_ATTEMPTS,
            inner_text=line,
        )
    return line


def parse_runtime_identity(line: str) -> dict[str, Any]:
    parts = line.strip().split(",")
    if len(parts) != 11 or parts[0] != RUNTIME_IDENTITY_PREFIX:
        raise ValueError("Final runtime identity has an invalid shape")
    try:
        seed = int(parts[2])
    except ValueError as exc:
        raise ValueError("Final runtime identity seed is invalid") from exc
    result = {
        "prefix": parts[0],
        "protocol": parts[1],
        "seed": seed,
        "student": parts[3],
        "route": parts[4],
        "model_sha256": parts[5],
        "export_id": parts[6],
        "bundle_id": parts[7],
        "board": BOARD_NAMES.get(parts[8]),
        "transport": TRANSPORT_NAMES.get(parts[9]),
        "build_contract_id": parts[10],
    }
    for field in ["model_sha256", "export_id", "bundle_id", "build_contract_id"]:
        if not _is_sha256(result[field]):
            raise ValueError(f"Final runtime identity has an invalid {field}")
    if result["student"] not in {"A", "B"} or result["route"] not in {
        "scratch",
        "rf_kd",
    }:
        raise ValueError("Final runtime identity has an invalid model route")
    if result["board"] is None or result["transport"] is None:
        raise ValueError("Final runtime identity has an invalid target")
    return result


def _identity_header(
    *,
    bundle_id: str,
    board: str,
    transport: str,
    build_contract_id: str,
) -> str:
    board_macro = (
        "CUKD_WIRELESS_BOARD_ESP32C3"
        if board == "esp32c3"
        else "CUKD_WIRELESS_BOARD_ARDUINO_R4"
    )
    wireless_compatibility = ""
    if transport == "wifi_udp":
        wireless_compatibility = (
            f'#define CUKD_WIRELESS_BUNDLE_ID "{bundle_id}"\n'
            '#define CUKD_WIRELESS_PROTOCOL_ID "cukd_fgds_wifi_udp_session_v2"\n'
            f'#define CUKD_WIRELESS_BOARD_ID "{board}"\n'
            "#define CUKD_WIRELESS_UDP_PORT 42101u\n"
            "#define CUKD_WIRELESS_MAX_DATAGRAM 768u\n"
            f"#define {board_macro} 1\n"
        )
    return (
        "#ifndef CUKD_FINAL_BUNDLE_IDENTITY_H\n"
        "#define CUKD_FINAL_BUNDLE_IDENTITY_H\n"
        f'#define CUKD_FINAL_BUNDLE_ID "{bundle_id}"\n'
        f'#define CUKD_FINAL_BOARD_ID "{board}"\n'
        f'#define CUKD_FINAL_TRANSPORT_ID "{transport}"\n'
        f'#define CUKD_FINAL_BOARD_TOKEN "{BOARD_TOKENS[board]}"\n'
        f'#define CUKD_FINAL_TRANSPORT_TOKEN "{TRANSPORT_TOKENS[transport]}"\n'
        f'#define CUKD_BUILD_CONTRACT_ID "{build_contract_id}"\n'
        "#define CUKD_FINAL_STRINGIFY_INNER(value) #value\n"
        "#define CUKD_FINAL_STRINGIFY(value) CUKD_FINAL_STRINGIFY_INNER(value)\n"
        "#define CUKD_FINAL_RUNTIME_IDENTITY \\\n"
        f"    \"{RUNTIME_IDENTITY_PREFIX},\" CUKD_PROTOCOL_ID \",\" \\\n"
        "    CUKD_FINAL_STRINGIFY(CUKD_EXPORT_SEED) \",\" CUKD_STUDENT_ID \",\" \\\n"
        "    CUKD_ROUTE_ID \",\" CUKD_TRAINED_STATE_SHA256 \",\" CUKD_EXPORT_ID \",\" \\\n"
        "    CUKD_FINAL_BUNDLE_ID \",\" CUKD_FINAL_BOARD_TOKEN \",\" \\\n"
        "    CUKD_FINAL_TRANSPORT_TOKEN \",\" CUKD_BUILD_CONTRACT_ID\n"
        + wireless_compatibility
        + "#endif\n"
    )


def _normalize_source(text: str) -> str:
    normalized = text.replace("\r\n", "\n")
    if "\r" in normalized:
        raise RuntimeError("Firmware template contains mixed line endings")
    return normalized


def _transform_usb(source: str, board: str) -> str:
    source = _normalize_source(source)
    include_anchor = "#include <string.h>\n"
    include_replacement = (
        "#include <string.h>\n\n"
        '#include "cukd_export_identity.h"\n'
        '#include "cukd_final_bundle_identity.h"\n'
    )
    if source.count(include_anchor) != 1:
        raise RuntimeError("USB template include anchor is missing or ambiguous")
    source = source.replace(include_anchor, include_replacement, 1)
    function_anchor = USB_PROCESS_ANCHORS[board]
    handler = function_anchor + (
        '    if (strcmp(line, "CUKDID?") == 0) {\n'
        "        Serial.println(CUKD_FINAL_RUNTIME_IDENTITY);\n"
        "        return;\n"
        "    }\n"
    )
    if source.count(function_anchor) != 1:
        raise RuntimeError("USB process-line anchor is missing or ambiguous")
    return source.replace(function_anchor, handler, 1)


def _transform_wifi(source: str) -> str:
    source = _normalize_source(source)
    corrected_timer = (
        "    cukd_forward_q15(input_q15, logits);\n"
        "    const uint8_t prediction = cukd_argmax_logits(logits);\n"
        "    const uint32_t inference_end = cukd_now_us();\n"
    )
    if source.count(corrected_timer) != 1:
        raise RuntimeError(
            "Canonical Wi-Fi timer is not the corrected forward-plus-argmax boundary"
        )
    serial_old = (
        "static void cukd_send_build_identity_serial() {\n"
        '    Serial.print("CUKDWBUILD,");\n'
        "    Serial.print(CUKD_STUDENT_ID);\n"
        '    Serial.print(",");\n'
        "    Serial.print(CUKD_EXPORT_ID);\n"
        '    Serial.print(",");\n'
        "    Serial.print(CUKD_WIRELESS_BUNDLE_ID);\n"
        '    Serial.print(",");\n'
        "    Serial.print(CUKD_WIRELESS_BOARD_ID);\n"
        '    Serial.print(",");\n'
        "    Serial.println(CUKD_WIRELESS_PROTOCOL_ID);\n"
        "}\n"
    )
    serial_new = (
        "static void cukd_send_build_identity_serial() {\n"
        "    Serial.println(CUKD_FINAL_RUNTIME_IDENTITY);\n"
        "}\n"
    )
    if source.count(serial_old) != 1:
        raise RuntimeError("Wi-Fi serial identity anchor is missing or ambiguous")
    source = source.replace(serial_old, serial_new, 1)
    udp_old = (
        "        \"CUKDWBUILD,%s,%s,%s,%s,%s\",\n"
        "        CUKD_STUDENT_ID,\n"
        "        CUKD_EXPORT_ID,\n"
        "        CUKD_WIRELESS_BUNDLE_ID,\n"
        "        CUKD_WIRELESS_BOARD_ID,\n"
        "        CUKD_WIRELESS_PROTOCOL_ID\n"
    )
    udp_new = "        \"%s\",\n        CUKD_FINAL_RUNTIME_IDENTITY\n"
    if source.count(udp_old) != 1:
        raise RuntimeError("Wi-Fi UDP identity anchor is missing or ambiguous")
    transformed = source.replace(udp_old, udp_new, 1)
    if transformed.count(corrected_timer) != 1:
        raise RuntimeError("Identity transformation changed the Wi-Fi timing boundary")
    return transformed


def _tested_model_source_hashes(export_root: Path) -> dict[str, str]:
    identity = read_json(export_root / "final_export_identity.json")
    snapshots = identity.get("source_snapshots")
    if not isinstance(snapshots, list):
        raise RuntimeError("Final export lacks sealed source snapshots")
    result: dict[str, str] = {}
    for item in snapshots:
        if not isinstance(item, dict):
            continue
        origin = item.get("origin_relative_path")
        digest = item.get("sha256")
        if isinstance(origin, str) and _is_sha256(digest):
            result[origin] = digest
    required = {
        f"deployment/hardware_hil/firmware/common/{name}"
        for name in [
            "cukd_model.c",
            "cukd_model.h",
            "cukd_preprocess.c",
            "cukd_preprocess.h",
        ]
    }
    if not required <= set(result):
        raise RuntimeError("Final export lacks the host-tested model/preprocess sources")
    return result


def prepare_final_bundle(
    *,
    export_dir: Path,
    output_dir: Path,
    board: str,
    transport: str,
    build_contract: Mapping[str, Any] | Path,
    verifier: Verifier | None = None,
) -> Path:
    """Create a new immutable final bundle without touching historical bundles."""

    if board not in USB_TEMPLATES or transport not in {"usb_serial", "wifi_udp"}:
        raise ValueError("Unsupported final HIL board or transport")
    export = validate_final_export(export_dir, verifier=verifier)
    build = validate_build_contract(build_contract, board=board)
    root = Path(export.root)
    destination = output_dir.resolve()
    try:
        destination.relative_to(root)
    except ValueError:
        pass
    else:
        raise RuntimeError("Final bundle cannot be created inside its export")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite final bundle: {destination}")

    template = USB_TEMPLATES[board] if transport == "usb_serial" else WIFI_TEMPLATE
    source_paths = [template, *[MODEL_COMMON / name for name in MODEL_COMMON_FILES]]
    if transport == "wifi_udp":
        source_paths.extend(WIFI_COMMON / name for name in WIFI_COMMON_FILES)
    source_paths.extend(root / name for name in GENERATED_FILES)
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    names = [path.name for path in source_paths]
    if len(names) != len(set(names)):
        raise RuntimeError("Final bundle source filenames are not unique")

    tested_hashes = _tested_model_source_hashes(root)
    for name in ["cukd_model.c", "cukd_model.h", "cukd_preprocess.c", "cukd_preprocess.h"]:
        origin = f"deployment/hardware_hil/firmware/common/{name}"
        if sha256_file(MODEL_COMMON / name) != tested_hashes[origin]:
            raise RuntimeError(f"Host-tested canonical source changed: {name}")
    export_manifest = read_json(root / "final_export_manifest.json")
    export_inventory = {item["path"]: item["sha256"] for item in _manifest_files(export_manifest)}
    for name in GENERATED_FILES:
        if sha256_file(root / name) != export_inventory.get(name):
            raise RuntimeError(f"Generated bundle input changed after verification: {name}")

    template_text = template.read_text(encoding="utf-8")
    transformed = (
        _transform_usb(template_text, board)
        if transport == "usb_serial"
        else _transform_wifi(template_text)
    )
    export_identity_payload = read_json(root / "final_export_identity.json")
    identity_payload: dict[str, Any] = {
        "protocol_id": BUNDLE_PROTOCOL_ID,
        "board": board,
        "transport": transport,
        "final_export_identity": export_identity_payload,
        "final_export_manifest_sha256": export.manifest_sha256,
        "build_contract_id": build["build_contract_id"],
        "template_source": _source_entry(template),
        "template_snapshot_file": TEMPLATE_SNAPSHOT_FILE,
        "source_files": [_source_entry(path) for path in source_paths],
        "transformed_sketch_sha256": hashlib.sha256(
            transformed.encode("utf-8")
        ).hexdigest(),
        "bundle_builder_sha256": sha256_file(Path(__file__).resolve()),
    }
    bundle_id = canonical_json_sha256(identity_payload)
    expected_runtime_identity = runtime_identity(
        export,
        bundle_id=bundle_id,
        board=board,
        transport=transport,
        build_contract_id=build["build_contract_id"],
    )
    header_name = (
        "cukd_final_bundle_identity.h"
        if transport == "usb_serial"
        else "cukd_wireless_bundle_identity.h"
    )
    header = _identity_header(
        bundle_id=bundle_id,
        board=board,
        transport=transport,
        build_contract_id=build["build_contract_id"],
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    manifest_path = temporary / "final_bundle_manifest.json"
    try:
        (temporary / TEMPLATE_SNAPSHOT_FILE).write_bytes(template.read_bytes())
        for path in source_paths:
            if path == template:
                continue
            (temporary / path.name).write_bytes(path.read_bytes())
        sketch_name = f"{destination.name}.ino"
        (temporary / sketch_name).write_text(transformed, encoding="utf-8", newline="\n")
        (temporary / header_name).write_text(header, encoding="ascii", newline="\n")
        build_path = temporary / "final_build_contract.json"
        build_path.write_text(json.dumps(build, indent=2) + "\n", encoding="utf-8")
        files = [
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(temporary.iterdir())
            if path.is_file() and path != manifest_path
        ]
        manifest: dict[str, Any] = {
            "schema": "cukd_final_hil_bundle_manifest_v1",
            "status": "passed",
            "protocol_id": BUNDLE_PROTOCOL_ID,
            "board": board,
            "transport": transport,
            "student": export.student,
            "route": export.route,
            "seed": export.seed,
            "model_sha256": export.trained_state_sha256,
            "checkpoint_file_sha256": export.checkpoint_file_sha256,
            "export_id": export.export_id,
            "bundle_id": bundle_id,
            "build_contract_id": build["build_contract_id"],
            "runtime_identity_query": "CUKDID?" if transport == "usb_serial" else "CUKDWID?",
            "runtime_identity_response": expected_runtime_identity,
            "bundle_identity_payload": identity_payload,
            "template_snapshot_file": TEMPLATE_SNAPSHOT_FILE,
            "sketch_file": sketch_name,
            "transformed_sketch_sha256": identity_payload[
                "transformed_sketch_sha256"
            ],
            "file_count_excluding_manifest": len(files),
            "files": files,
            "claim_boundary": (
                "Hash-bound build and runtime provenance for replay firmware. This is "
                "not cryptographic attestation by the MCU or compiler frontend."
            ),
        }
        manifest["manifest_payload_sha256"] = canonical_json_sha256(manifest)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        verify_final_bundle(temporary, expected_export=export)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination / "final_bundle_manifest.json"


def verify_final_bundle(
    bundle_dir: Path,
    *,
    expected_export: FinalExportIdentity,
    expected_builder_sha256: str | None = None,
    canonical_source_root: Path | None = None,
) -> dict[str, Any]:
    root = bundle_dir.resolve()
    manifest_path = root / "final_bundle_manifest.json"
    manifest = read_json(manifest_path)
    _verify_optional_payload_hash(manifest, "manifest_payload_sha256", "Bundle manifest")
    if (
        manifest.get("schema") != "cukd_final_hil_bundle_manifest_v1"
        or manifest.get("protocol_id") != BUNDLE_PROTOCOL_ID
        or manifest.get("status") != "passed"
    ):
        raise RuntimeError("Final bundle manifest contract is invalid")
    identity = manifest.get("bundle_identity_payload")
    if not isinstance(identity, dict) or canonical_json_sha256(identity) != manifest.get(
        "bundle_id"
    ):
        raise RuntimeError("Final bundle ID does not bind its identity payload")
    if identity.get("board") != manifest.get("board") or identity.get(
        "transport"
    ) != manifest.get("transport"):
        raise RuntimeError("Final bundle target differs from its identity payload")
    if identity.get("build_contract_id") != manifest.get("build_contract_id"):
        raise RuntimeError("Final bundle build contract differs from its identity payload")
    export_identity = identity.get("final_export_identity")
    if not isinstance(export_identity, dict):
        raise RuntimeError("Final bundle lacks its canonical export identity")
    export_payload = dict(export_identity)
    export_id = export_payload.pop("export_id", None)
    if export_id != canonical_json_sha256(export_payload):
        raise RuntimeError("Bundled final export identity hash is invalid")
    for field, expected in {
        "student": export_identity.get("student"),
        "route": export_identity.get("route"),
        "seed": export_identity.get("seed"),
        "model_sha256": export_identity.get("trained_state_sha256"),
        "checkpoint_file_sha256": export_identity.get("checkpoint_file_sha256"),
        "export_id": export_id,
    }.items():
        if manifest.get(field) != expected:
            raise RuntimeError(f"Bundle manifest/export identity mismatch for {field}")
    if not _is_sha256(identity.get("final_export_manifest_sha256")):
        raise RuntimeError("Bundle lacks the final export manifest hash")
    if not _is_sha256(identity.get("bundle_builder_sha256")):
        raise RuntimeError("Bundle lacks its builder source hash")
    builder_sha256 = expected_builder_sha256 or sha256_file(Path(__file__).resolve())
    if not _is_sha256(builder_sha256):
        raise RuntimeError("Expected bundle-builder source hash is invalid")
    if identity.get("bundle_builder_sha256") != builder_sha256:
        raise RuntimeError("Bundle builder source differs from the verified implementation")
    files = _manifest_files(manifest)
    if manifest.get("file_count_excluding_manifest") != len(files):
        raise RuntimeError("Final bundle inventory count is invalid")
    listed: set[str] = set()
    for item in files:
        relative = item.get("path")
        if not isinstance(relative, str) or Path(relative).name != relative or relative in listed:
            raise RuntimeError("Final bundle inventory path is unsafe or duplicated")
        member = root / relative
        if (
            not member.is_file()
            or member.stat().st_size != item.get("size_bytes")
            or sha256_file(member) != item.get("sha256")
        ):
            raise RuntimeError(f"Final bundle inventory mismatch: {relative}")
        listed.add(relative)
    directories = [path for path in root.rglob("*") if path.is_dir()]
    if directories:
        raise RuntimeError("Final bundle contains an unlisted directory")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != listed:
        raise RuntimeError("Final bundle contains extra or missing files")
    source_files = identity.get("source_files")
    template_source = identity.get("template_source")
    if not isinstance(source_files, list) or not isinstance(template_source, dict):
        raise RuntimeError("Final bundle source-input ledger is missing")
    source_by_name: dict[str, dict[str, Any]] = {}
    for entry in source_files:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("name"), str)
            or not isinstance(entry.get("origin"), str)
            or not _is_sha256(entry.get("sha256"))
            or not isinstance(entry.get("size_bytes"), int)
            or entry["name"] in source_by_name
        ):
            raise RuntimeError("Final bundle source-input ledger is malformed")
        source_by_name[entry["name"]] = entry
    expected_source_names = {
        *MODEL_COMMON_FILES,
        *GENERATED_FILES,
        template_source.get("name"),
        *(WIFI_COMMON_FILES if manifest.get("transport") == "wifi_udp" else ()),
    }
    if set(source_by_name) != expected_source_names:
        raise RuntimeError("Final bundle source-input set is incomplete or unexpected")
    if template_source not in source_files:
        raise RuntimeError("Final bundle template is not present in its source ledger")
    if manifest.get("board") not in USB_TEMPLATES or manifest.get("transport") not in {
        "usb_serial",
        "wifi_udp",
    }:
        raise RuntimeError("Final bundle target is unsupported")
    template_relative = (
        USB_TEMPLATE_RELATIVES[str(manifest.get("board"))]
        if manifest.get("transport") == "usb_serial"
        else WIFI_TEMPLATE_RELATIVE
    )
    canonical_root = (canonical_source_root or REPO_ROOT).resolve()
    canonical_relatives = [
        template_relative,
        *[MODEL_COMMON_RELATIVE / name for name in MODEL_COMMON_FILES],
        *(
            [WIFI_COMMON_RELATIVE / name for name in WIFI_COMMON_FILES]
            if manifest.get("transport") == "wifi_udp"
            else []
        ),
    ]
    for relative in canonical_relatives:
        expected_entry = _canonical_source_entry(canonical_root, relative)
        if source_by_name.get(relative.name) != expected_entry:
            raise RuntimeError(
                f"Final bundle source differs from the canonical source: {relative.name}"
            )
    snapshot_name = identity.get("template_snapshot_file")
    if (
        snapshot_name != TEMPLATE_SNAPSHOT_FILE
        or manifest.get("template_snapshot_file") != TEMPLATE_SNAPSHOT_FILE
        or snapshot_name not in listed
    ):
        raise RuntimeError("Final bundle template snapshot is missing or unbound")
    snapshot = root / TEMPLATE_SNAPSHOT_FILE
    if (
        snapshot.stat().st_size != template_source.get("size_bytes")
        or sha256_file(snapshot) != template_source.get("sha256")
    ):
        raise RuntimeError("Final bundle template snapshot differs from its source")
    for name, entry in source_by_name.items():
        if entry == template_source:
            continue
        member = root / name
        if (
            not member.is_file()
            or member.stat().st_size != entry["size_bytes"]
            or sha256_file(member) != entry["sha256"]
        ):
            raise RuntimeError(f"Copied bundle source differs from its input: {name}")
    source_snapshots = export_identity.get("source_snapshots")
    if not isinstance(source_snapshots, list):
        raise RuntimeError("Final export identity lacks its source-snapshot ledger")
    snapshot_by_name: dict[str, Mapping[str, Any]] = {}
    for item in source_snapshots:
        if not isinstance(item, Mapping) or not isinstance(
            item.get("snapshot_path"), str
        ):
            raise RuntimeError("Final export source-snapshot ledger is malformed")
        name = Path(str(item["snapshot_path"])).name
        if name in snapshot_by_name:
            raise RuntimeError("Final export source-snapshot name is duplicated")
        snapshot_by_name[name] = item
    for name in ["cukd_model.c", "cukd_model.h", "cukd_preprocess.c", "cukd_preprocess.h"]:
        snapshot_entry = snapshot_by_name.get(name)
        source = source_by_name[name]
        if (
            snapshot_entry is None
            or snapshot_entry.get("size_bytes") != source.get("size_bytes")
            or snapshot_entry.get("sha256") != source.get("sha256")
        ):
            raise RuntimeError(
                f"Bundled model source differs from host-tested export snapshot: {name}"
            )
    core_files = export_identity.get("core_files")
    if not isinstance(core_files, list):
        raise RuntimeError("Final export identity lacks its core-file ledger")
    core_by_name = {
        item.get("path"): item for item in core_files if isinstance(item, Mapping)
    }
    for name in ["model_weights.h", "preprocess_int_metadata.h"]:
        core = core_by_name.get(name)
        source = source_by_name[name]
        if (
            not isinstance(core, Mapping)
            or core.get("size_bytes") != source.get("size_bytes")
            or core.get("sha256") != source.get("sha256")
        ):
            raise RuntimeError(f"Bundled generated source differs from export identity: {name}")
    expected_export_header = (
        "#ifndef CUKD_FINAL_EXPORT_IDENTITY_H\n"
        "#define CUKD_FINAL_EXPORT_IDENTITY_H\n"
        f"#define CUKD_EXPORT_ID \"{export_id}\"\n"
        f"#define CUKD_PROTOCOL_ID \"{export_identity['protocol']}\"\n"
        f"#define CUKD_EXPORT_SEED {export_identity['seed']}\n"
        f"#define CUKD_STUDENT_ID \"{export_identity['student']}\"\n"
        f"#define CUKD_ROUTE_ID \"{export_identity['route']}\"\n"
        f"#define CUKD_CHECKPOINT_FILE_SHA256 \"{export_identity['checkpoint_file_sha256']}\"\n"
        f"#define CUKD_TRAINED_STATE_SHA256 \"{export_identity['trained_state_sha256']}\"\n"
        "#endif\n"
    )
    if (root / "cukd_export_identity.h").read_text(
        encoding="ascii"
    ) != expected_export_header:
        raise RuntimeError("Bundled export identity header is not reproducible")
    build = validate_build_contract(root / "final_build_contract.json", board=manifest["board"])
    if build["build_contract_id"] != manifest.get("build_contract_id"):
        raise RuntimeError("Bundle build-contract binding is invalid")
    sketch = root / str(manifest.get("sketch_file"))
    if sha256_file(sketch) != next(
        item["sha256"] for item in files if item["path"] == sketch.name
    ):
        raise RuntimeError("Bundle sketch is not hash-bound")
    if sha256_file(sketch) != manifest.get("transformed_sketch_sha256"):
        raise RuntimeError("Bundle sketch differs from transformed-source identity")
    if manifest.get("transformed_sketch_sha256") != identity.get(
        "transformed_sketch_sha256"
    ):
        raise RuntimeError("Bundle sketch identity differs from manifest")
    template_text = snapshot.read_text(encoding="utf-8")
    expected_sketch = (
        _transform_usb(template_text, str(manifest.get("board")))
        if manifest.get("transport") == "usb_serial"
        else _transform_wifi(template_text)
    )
    if sketch.read_bytes() != expected_sketch.encode("utf-8"):
        raise RuntimeError("Bundle sketch does not reproduce from its sealed template")
    header_name = (
        "cukd_final_bundle_identity.h"
        if manifest.get("transport") == "usb_serial"
        else "cukd_wireless_bundle_identity.h"
    )
    expected_header = _identity_header(
        bundle_id=str(manifest.get("bundle_id")),
        board=str(manifest.get("board")),
        transport=str(manifest.get("transport")),
        build_contract_id=str(manifest.get("build_contract_id")),
    )
    header_path = root / header_name
    if not header_path.is_file() or header_path.read_text(encoding="ascii") != expected_header:
        raise RuntimeError("Bundle runtime-identity header is invalid")
    if identity.get("final_export_manifest_sha256") != expected_export.manifest_sha256:
        raise RuntimeError("Final bundle differs from export manifest hash")
    expected = {
        "student": expected_export.student,
        "route": expected_export.route,
        "seed": expected_export.seed,
        "model_sha256": expected_export.trained_state_sha256,
        "checkpoint_file_sha256": expected_export.checkpoint_file_sha256,
        "export_id": expected_export.export_id,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise RuntimeError(f"Final bundle differs from export for {field}")
    for name in GENERATED_FILES:
        expected_member = Path(expected_export.root) / name
        source = source_by_name[name]
        if (
            not expected_member.is_file()
            or expected_member.stat().st_size != source["size_bytes"]
            or sha256_file(expected_member) != source["sha256"]
        ):
            raise RuntimeError(f"Final bundle differs from export source: {name}")
    parsed = parse_runtime_identity(str(manifest.get("runtime_identity_response")))
    expected_fields = {
        "seed": manifest.get("seed"),
        "student": manifest.get("student"),
        "route": manifest.get("route"),
        "model_sha256": manifest.get("model_sha256"),
        "export_id": manifest.get("export_id"),
        "bundle_id": manifest.get("bundle_id"),
        "board": manifest.get("board"),
        "transport": manifest.get("transport"),
        "build_contract_id": manifest.get("build_contract_id"),
    }
    for field, expected in expected_fields.items():
        if parsed[field] != expected:
            raise RuntimeError(f"Runtime identity differs from bundle for {field}")
    export_protocol = identity.get("final_export_identity", {}).get("protocol")
    if parsed["protocol"] != export_protocol:
        raise RuntimeError("Runtime protocol differs from final export identity")
    return {**manifest, "_manifest_sha256": sha256_file(manifest_path)}
