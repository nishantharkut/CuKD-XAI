"""Build provenance, six-stage completion, preflight, and campaign verification."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .bundles import parse_runtime_identity, validate_build_contract, verify_final_bundle
from .contracts import (
    BOARDS,
    FINAL_STAGES,
    MODEL_KEYS,
    TRANSPORTS,
    FinalExportIdentity,
    Verifier,
    _is_sha256,
    atomic_write_json,
    canonical_json_sha256,
    read_json,
    sha256_file,
    validate_balanced_cohort,
    validate_campaign_contract,
    validate_final_export,
)
from .runtime import (
    _absolute_path_without_symlinks,
    _bundle_source_options,
    _validate_attempt_connection_set,
    collect_host_environment,
    query_runtime_identity_serial,
    require_physical_port_serial,
    require_session_id,
    utc_now,
    validate_host_environment,
    validate_wifi_connection,
    verify_stage_attempt,
)


PROVENANCE_SCHEMA = "cukd_final_hil_build_upload_provenance_v2"
SESSION_SCHEMA = "cukd_final_hil_six_stage_completion_v2"
CAMPAIGN_EVIDENCE_SCHEMA = "cukd_final_hil_campaign_evidence_v1"
TIMING_STATISTICAL_UNIT = (
    "three ordered repeat summaries on one physical board specimen; the "
    "3,000 rows are not independent hardware replications"
)
SESSION_CONNECTION_FIELDS = (
    "wifi_network_session_id",
    "connection_path_recorded",
    "connection_record_sha256",
    "connection_payload_sha256",
    "udp_timeout_seconds",
    "udp_max_attempts",
)


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("Evidence timestamp must use a zero UTC offset")
    return parsed


def _sealed_payload(payload: Mapping[str, Any], id_field: str) -> dict[str, Any]:
    copy = dict(payload)
    copy.pop(id_field, None)
    result = dict(copy)
    result[id_field] = canonical_json_sha256(copy)
    return result


def _session_connection_fields(
    binding: tuple[str, str, str, str, float, int] | None,
) -> dict[str, Any]:
    values: tuple[Any, ...] = binding or (None,) * len(SESSION_CONNECTION_FIELDS)
    return dict(zip(SESSION_CONNECTION_FIELDS, values))


def validate_session_connection_record(
    *,
    session: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    bundle: Mapping[str, Any],
    connection_json: Path | None = None,
    host_source_root: Path | None = None,
) -> dict[str, Any] | None:
    binding = _validate_attempt_connection_set(attempts, str(session["transport"]))
    expected_fields = _session_connection_fields(binding)
    if any(
        field not in session or session.get(field) != value
        for field, value in expected_fields.items()
    ):
        raise RuntimeError("Session connection/policy binding differs from its attempts")
    if binding is None:
        if connection_json is not None:
            raise RuntimeError("USB session cannot carry a Wi-Fi connection record")
        return None

    network_session_id, recorded_path, record_sha, payload_sha, _, _ = binding
    selected = Path(connection_json) if connection_json is not None else Path(recorded_path)
    selected = _absolute_path_without_symlinks(selected, "Wi-Fi connection")
    if connection_json is None and recorded_path != str(selected):
        raise RuntimeError("Session Wi-Fi connection path is not canonical")
    before = sha256_file(selected)
    connection = validate_wifi_connection(
        selected,
        bundle=bundle,
        host_source_root=host_source_root,
    )
    if before != record_sha or sha256_file(selected) != record_sha:
        raise RuntimeError("Session Wi-Fi connection record changed")
    if (
        connection.get("session_id") != network_session_id
        or connection.get("connection_payload_sha256") != payload_sha
        or connection.get("physical_port_serial")
        != session.get("physical_port_serial")
        or connection.get("wifi_mac_reported") != session.get("wifi_mac_reported")
    ):
        raise RuntimeError("Session Wi-Fi connection identity differs")
    if _parse_timestamp(str(connection["finished_utc"])) > _parse_timestamp(
        str(session["stages"][0]["started_utc"])
    ):
        raise RuntimeError("Wi-Fi provisioning overlaps stage execution")
    return connection


def _memoized_export_verifier(verifier: Verifier | None) -> Verifier:
    cache: dict[str, Mapping[str, Any]] = {}

    def verify(root: Path) -> Mapping[str, Any]:
        key = str(root.resolve())
        if key not in cache:
            if verifier is None:
                from .contracts import _canonical_final_verifier

                cache[key] = dict(_canonical_final_verifier(root.resolve()))
            else:
                cache[key] = dict(verifier(root.resolve()))
        return dict(cache[key])

    return verify


def _expand_build_command(
    template: Sequence[str],
    *,
    bundle_dir: Path,
    fqbn: str,
    port: str,
    build_dir: Path,
) -> list[str]:
    return _render_build_command(
        template,
        bundle=str(bundle_dir.resolve()),
        fqbn=fqbn,
        port=port,
        build_dir=str(build_dir.resolve()),
    )


def _render_build_command(
    template: Sequence[str],
    *,
    bundle: str,
    fqbn: str,
    port: str,
    build_dir: str,
) -> list[str]:
    replacements = {
        "{bundle}": bundle,
        "{fqbn}": fqbn,
        "{port}": port,
        "{build_dir}": build_dir,
    }
    return [replacements.get(token, token) for token in template]


def _artifact_inventory(root: Path, *, exclude: set[str] | None = None) -> list[dict[str, Any]]:
    excluded = exclude or set()
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"Build provenance cannot contain symlinks: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return files


def _validate_upload_artifact_delta(
    before: Any,
    after: Any,
    *,
    board: str,
    bundle_side_effects: Any = (),
) -> list[dict[str, Any]]:
    """Require compile outputs to remain unchanged while recording uploader copies."""

    if not isinstance(before, list) or not before or not isinstance(after, list):
        raise RuntimeError("Build provenance compile-output inventory is missing")

    def indexed(inventory: list[Any], label: str) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for item in inventory:
            if not isinstance(item, Mapping):
                raise RuntimeError(f"Build provenance {label} inventory is malformed")
            relative = item.get("path")
            size = item.get("size_bytes")
            digest = item.get("sha256")
            if (
                not isinstance(relative, str)
                or not relative
                or "\\" in relative
                or Path(relative).as_posix() != relative
                or Path(relative).is_absolute()
                or Path(relative).drive
                or ".." in Path(relative).parts
                or relative in result
                or not isinstance(size, int)
                or isinstance(size, bool)
                or size < 0
                or not _is_sha256(digest)
            ):
                raise RuntimeError(f"Build provenance {label} inventory is malformed")
            result[relative] = {
                "path": relative,
                "size_bytes": size,
                "sha256": digest,
            }
        return result

    before_by_path = indexed(before, "pre-upload")
    after_by_path = indexed(after, "post-upload")
    for relative, item in before_by_path.items():
        if after_by_path.get(relative) != item:
            raise RuntimeError(
                f"Upload changed or removed a compile output: {relative}"
            )

    if board not in {"esp32c3", "arduino_r4"}:
        raise RuntimeError(f"Unsupported board in upload artifact inventory: {board}")
    external_by_name: dict[str, set[tuple[int, str]]] = {}
    if not isinstance(bundle_side_effects, list) and bundle_side_effects != ():
        raise RuntimeError("Build provenance bundle side-effect inventory is malformed")
    for item in bundle_side_effects:
        if not isinstance(item, Mapping):
            raise RuntimeError("Build provenance bundle side-effect inventory is malformed")
        name = Path(str(item.get("path", ""))).name
        size = item.get("size_bytes")
        digest = item.get("sha256")
        if not name or not isinstance(size, int) or not _is_sha256(digest):
            raise RuntimeError("Build provenance bundle side-effect inventory is malformed")
        external_by_name.setdefault(name, set()).add((size, digest))

    added = [
        item for relative, item in after_by_path.items() if relative not in before_by_path
    ]
    for item in added:
        relative = item["path"]
        path = Path(relative)
        if not path.name.endswith("_flashed.bin"):
            raise RuntimeError(f"Upload added an unexpected build artifact: {relative}")
        source_name = path.name.removesuffix("_flashed.bin") + ".bin"
        source_relative = (path.parent / source_name).as_posix()
        source = before_by_path.get(source_relative)
        if source is None:
            allowed_external = board == "esp32c3" and path.name == "boot_app0_flashed.bin"
            if not allowed_external or (
                item["size_bytes"], item["sha256"]
            ) not in external_by_name.get(source_name, set()):
                raise RuntimeError(
                    f"Uploader-added artifact has no matching source: {relative}"
                )
        elif (
            item["size_bytes"] != source["size_bytes"]
            or item["sha256"] != source["sha256"]
        ):
            raise RuntimeError(
                f"Uploader-added artifact differs from its compile output: {relative}"
            )
    return added


def _preserve_bundle_build_side_effect(
    bundle_root: Path,
    destination: Path,
    *,
    command_name: str,
) -> list[dict[str, Any]]:
    """Move Arduino CLI's sketch-local build cache into run provenance."""

    source = bundle_root / "build"
    if not source.exists():
        return []
    if source.is_symlink() or not source.is_dir():
        raise RuntimeError("Arduino CLI bundle build side effect is not a directory")
    target_name = f"arduino_cli_{command_name}_build_side_effect"
    target = destination / target_name
    if target.exists():
        raise RuntimeError("Arduino CLI bundle build side-effect destination exists")
    inventory = _artifact_inventory(source)
    source.rename(target)
    preserved = [
        {**item, "path": f"{target_name}/{item['path']}"}
        for item in inventory
    ]
    if _artifact_inventory(target) != inventory:
        raise RuntimeError("Arduino CLI bundle build side effect changed while preserving")
    return preserved


def _run_logged_command(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    timeout_seconds: float,
    runner: Callable[..., Any],
) -> dict[str, Any]:
    started = utc_now()
    timed_out = False
    try:
        result = runner(
            list(command),
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout_seconds,
        )
        returncode = int(result.returncode)
        output = result.stdout or ""
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = -1
        partial = exc.stdout or ""
        output = partial.decode("utf-8", errors="replace") if isinstance(partial, bytes) else str(partial)
    log_path.write_text(output, encoding="utf-8", errors="strict")
    return {
        "command": list(command),
        "started_utc": started,
        "finished_utc": utc_now(),
        "returncode": returncode,
        "timed_out": timed_out,
        "log": {
            "path": log_path.name,
            "size_bytes": log_path.stat().st_size,
            "sha256": sha256_file(log_path),
        },
    }


def _validate_arduino_inspection(
    *,
    contract: Mapping[str, Any],
    frontend_log: Path,
    cores_log: Path,
    board_log: Path,
) -> str:
    payloads: dict[str, Mapping[str, Any]] = {}
    for name, path in {
        "frontend": frontend_log,
        "cores": cores_log,
        "board": board_log,
    }.items():
        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="strict"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Arduino CLI {name} inspection is not valid JSON") from exc
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"Arduino CLI {name} inspection is not a JSON object")
        payloads[name] = payload

    frontend = payloads["frontend"]
    observed_frontend = f"{frontend.get('Application')} {frontend.get('VersionString')}"
    if observed_frontend != contract["frontend_version"]:
        raise RuntimeError(
            "Arduino CLI frontend identity differs from the build contract: "
            f"observed {observed_frontend!r}, expected {contract['frontend_version']!r}"
        )

    platforms = payloads["cores"].get("platforms")
    if not isinstance(platforms, list):
        raise RuntimeError("Arduino CLI core inspection has no platform list")
    matching_platforms = [
        item
        for item in platforms
        if isinstance(item, Mapping) and item.get("id") == contract["platform_id"]
    ]
    if len(matching_platforms) != 1:
        raise RuntimeError("Arduino CLI core inspection has no unique contracted platform")
    installed_version = matching_platforms[0].get("installed_version")
    if installed_version != contract["board_core_version"]:
        raise RuntimeError(
            "Installed board core differs from the build contract: "
            f"observed {installed_version!r}, expected {contract['board_core_version']!r}"
        )

    board = payloads["board"]
    if board.get("fqbn") != contract["fqbn"]:
        raise RuntimeError("Arduino CLI board FQBN differs from the build contract")
    if board.get("version") != contract["board_core_version"]:
        raise RuntimeError("Arduino CLI board-details core version differs from the contract")

    return "\n".join(
        path.read_text(encoding="utf-8", errors="strict")
        for path in [frontend_log, cores_log, board_log]
    )


def record_build_upload_provenance(
    *,
    export_dir: Path,
    bundle_dir: Path,
    physical_port: str,
    physical_port_serial: str,
    output_dir: Path,
    command_timeout_seconds: float = 1800.0,
    runner: Callable[..., Any] = subprocess.run,
    identity_query: Callable[..., str] = query_runtime_identity_serial,
    verifier: Verifier | None = None,
) -> Path:
    """Execute, preserve, and seal one exact compile/upload/identity sequence."""

    if command_timeout_seconds <= 0:
        raise ValueError("Build/upload command timeout must be positive")
    destination = output_dir.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite build provenance: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    build_dir = destination / "build_artifacts"
    bundle_root = bundle_dir.resolve()
    bundle: dict[str, Any] | None = None
    contract: dict[str, Any] | None = None
    environment: dict[str, Any] | None = None
    observed_port_serial: str | None = None
    commands: dict[str, Any] = {}
    bundle_tool_side_effects: list[dict[str, Any]] = []
    overall_started = utc_now()
    try:
        build_dir.mkdir()
        cached_verifier = _memoized_export_verifier(verifier)
        export = validate_final_export(export_dir, verifier=cached_verifier)
        bundle = verify_final_bundle(bundle_root, expected_export=export)
        contract = validate_build_contract(bundle_root / "final_build_contract.json")
        if contract["build_contract_id"] != bundle["build_contract_id"]:
            raise RuntimeError("Bundle and build contract IDs differ")
        observed_port_serial = require_physical_port_serial(
            physical_port, physical_port_serial
        )
        environment = collect_host_environment()
        revision = environment.get("git_revision")
        if (
            not isinstance(revision, str)
            or len(revision) != 40
            or any(
                character not in "0123456789abcdef"
                for character in revision.lower()
            )
        ):
            raise RuntimeError("Build provenance requires an exact host Git revision")

        cli = contract["compile_command"][0]
        inspection_templates = {
            "frontend": [cli, "version", "--format", "json"],
            "cores": [cli, "core", "list", "--format", "json"],
            "board": [
                cli,
                "board",
                "details",
                "--fqbn",
                contract["fqbn"],
                "--format",
                "json",
            ],
        }
        for name, command in inspection_templates.items():
            commands[name] = _run_logged_command(
                command,
                cwd=Path(__file__).resolve().parents[2],
                log_path=destination / f"{name}.log",
                timeout_seconds=min(command_timeout_seconds, 120.0),
                runner=runner,
            )
            if commands[name]["returncode"] != 0:
                raise RuntimeError(f"Arduino CLI {name} inspection failed")
        inspection_text = _validate_arduino_inspection(
            contract=contract,
            frontend_log=destination / commands["frontend"]["log"]["path"],
            cores_log=destination / commands["cores"]["log"]["path"],
            board_log=destination / commands["board"]["log"]["path"],
        )

        compile_command = _expand_build_command(
            contract["compile_command"],
            bundle_dir=bundle_root,
            fqbn=contract["fqbn"],
            port=physical_port,
            build_dir=build_dir,
        )
        commands["compile"] = _run_logged_command(
            compile_command,
            cwd=Path(__file__).resolve().parents[2],
            log_path=destination / "compile.log",
            timeout_seconds=command_timeout_seconds,
            runner=runner,
        )
        bundle_tool_side_effects.extend(
            _preserve_bundle_build_side_effect(
                bundle_root,
                destination,
                command_name="compile",
            )
        )
        if commands["compile"]["returncode"] != 0:
            raise RuntimeError("Final firmware compile command failed")
        if verify_final_bundle(
            bundle_root,
            expected_export=export,
        )["bundle_id"] != bundle["bundle_id"]:
            raise RuntimeError("Final bundle identity changed during compile")
        compile_text = (destination / "compile.log").read_text(
            encoding="utf-8", errors="strict"
        )
        before_upload = _artifact_inventory(build_dir)
        if not before_upload or not any(
            Path(item["path"]).suffix.lower() in {".bin", ".elf", ".hex"}
            and item["size_bytes"] > 0
            for item in before_upload
        ):
            raise RuntimeError("Compile produced no nonempty firmware binary artifact")

        upload_command = _expand_build_command(
            contract["upload_command"],
            bundle_dir=bundle_root,
            fqbn=contract["fqbn"],
            port=physical_port,
            build_dir=build_dir,
        )
        commands["upload"] = _run_logged_command(
            upload_command,
            cwd=Path(__file__).resolve().parents[2],
            log_path=destination / "upload.log",
            timeout_seconds=command_timeout_seconds,
            runner=runner,
        )
        bundle_tool_side_effects.extend(
            _preserve_bundle_build_side_effect(
                bundle_root,
                destination,
                command_name="upload",
            )
        )
        if commands["upload"]["returncode"] != 0:
            raise RuntimeError("Final firmware upload command failed")
        if verify_final_bundle(
            bundle_root,
            expected_export=export,
        )["bundle_id"] != bundle["bundle_id"]:
            raise RuntimeError("Final bundle identity changed during upload")
        after_upload = _artifact_inventory(build_dir)
        uploader_added_artifacts = _validate_upload_artifact_delta(
            before_upload,
            after_upload,
            board=str(contract["board"]),
            bundle_side_effects=bundle_tool_side_effects,
        )
        post_reset_identity = identity_query(
            export_dir=export_dir,
            bundle_dir=bundle_root,
            port=physical_port,
            physical_port_serial=observed_port_serial,
            verifier=cached_verifier,
        )
        if post_reset_identity != bundle["runtime_identity_response"]:
            raise RuntimeError("Post-reset board identity differs from the final bundle")
    except BaseException as exc:
        failure = _sealed_payload(
            {
                "schema": PROVENANCE_SCHEMA,
                "status": "failed",
                "bundle_id": None if bundle is None else bundle["bundle_id"],
                "build_contract_id": (
                    None if bundle is None else bundle["build_contract_id"]
                ),
                "physical_port": physical_port,
                "physical_port_serial": observed_port_serial,
                "physical_port_serial_expected": physical_port_serial,
                "started_utc": overall_started,
                "finished_utc": utc_now(),
                "commands": commands,
                "bundle_tool_side_effect_artifacts": bundle_tool_side_effects,
                "error": {"type": exc.__class__.__name__, "message": str(exc)},
            },
            "failure_id",
        )
        atomic_write_json(destination / "failed_build_upload.json", failure)
        raise

    assert bundle is not None
    assert contract is not None
    assert environment is not None
    assert observed_port_serial is not None
    artifact_inventory = _artifact_inventory(
        destination, exclude={"build_upload_provenance.json"}
    )
    payload: dict[str, Any] = {
        "schema": PROVENANCE_SCHEMA,
        "status": "passed",
        "bundle_path_recorded": str(bundle_root),
        "build_dir_path_recorded": str(build_dir.resolve()),
        "bundle_manifest_sha256": bundle["_manifest_sha256"],
        "bundle_id": bundle["bundle_id"],
        "build_contract_id": bundle["build_contract_id"],
        "build_contract": contract,
        "board": bundle["board"],
        "transport": bundle["transport"],
        "student": bundle["student"],
        "route": bundle["route"],
        "export_id": bundle["export_id"],
        "model_sha256": bundle["model_sha256"],
        "checkpoint_file_sha256": bundle["checkpoint_file_sha256"],
        "physical_port": physical_port,
        "physical_port_serial": observed_port_serial,
        "started_utc": overall_started,
        "finished_utc": utc_now(),
        "commands": commands,
        "build_artifacts_before_upload": before_upload,
        "build_artifacts_after_upload": after_upload,
        "uploader_added_artifacts": uploader_added_artifacts,
        "bundle_tool_side_effect_artifacts": bundle_tool_side_effects,
        "post_reset_runtime_identity": post_reset_identity,
        "artifact_count_excluding_manifest": len(artifact_inventory),
        "artifacts": artifact_inventory,
        "host_environment": environment,
        "secure_attestation": False,
        "claim_boundary": (
            "This host-observed record binds tokenized commands, successful return "
            "codes, verbose logs, byte-identical pre-upload compile outputs, recorded "
            "uploader-added flashed-image copies, "
            "preserved Arduino CLI sketch-local build side effects, "
            "physical port identity, and a direct post-upload board identity query. "
            "It is reproducibility provenance, not secure compiler or MCU attestation."
        ),
    }
    payload = _sealed_payload(payload, "provenance_id")
    output_json = destination / "build_upload_provenance.json"
    atomic_write_json(output_json, payload)
    validate_build_upload_provenance(
        output_json,
        bundle_dir=bundle_root,
        expected_export=export,
    )
    return output_json


def validate_build_upload_provenance(
    path: Path,
    *,
    bundle_dir: Path | None = None,
    expected_export: FinalExportIdentity | None = None,
    artifact_root: Path | None = None,
    host_source_root: Path | None = None,
) -> dict[str, Any]:
    source_manifest = Path(path)
    if source_manifest.is_symlink():
        raise RuntimeError("Build provenance manifest cannot be a symlink")
    manifest_path = source_manifest.resolve()
    payload = read_json(manifest_path)
    recorded_id = payload.get("provenance_id")
    copy = dict(payload)
    copy.pop("provenance_id", None)
    if recorded_id != canonical_json_sha256(copy):
        raise RuntimeError("Build/upload provenance ID is invalid")
    if payload.get("schema") != PROVENANCE_SCHEMA or payload.get("status") != "passed":
        raise RuntimeError("Build/upload provenance is not passed final evidence")
    if payload.get("secure_attestation") is not False:
        raise RuntimeError("Build provenance must not claim secure attestation")
    if _parse_timestamp(payload["finished_utc"]) < _parse_timestamp(
        payload["started_utc"]
    ):
        raise RuntimeError("Build/upload provenance UTC interval is reversed")
    source_root = Path(artifact_root) if artifact_root is not None else manifest_path.parent
    if source_root.is_symlink():
        raise RuntimeError("Build provenance artifact root cannot be a symlink")
    root = source_root.resolve()
    members = list(root.rglob("*"))
    if any(item.is_symlink() for item in members):
        raise RuntimeError("Build provenance cannot contain symlinks")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RuntimeError("Build provenance artifact inventory is missing")
    listed = set()
    for item in artifacts:
        relative = item.get("path") if isinstance(item, Mapping) else None
        if (
            not isinstance(relative, str)
            or not relative
            or "\\" in relative
            or Path(relative).as_posix() != relative
            or Path(relative).is_absolute()
            or Path(relative).drive
            or ".." in Path(relative).parts
            or relative in listed
            or not _is_sha256(item.get("sha256"))
        ):
            raise RuntimeError("Build provenance artifact inventory is malformed")
        listed.add(relative)
        artifact = root / relative
        if (
            not artifact.is_file()
            or artifact.stat().st_size != item.get("size_bytes")
            or sha256_file(artifact) != item["sha256"]
        ):
            raise RuntimeError(f"Build provenance artifact changed: {relative}")
    actual = {
        item.relative_to(root).as_posix()
        for item in members
        if item.is_file() and item.resolve() != manifest_path
    }
    if actual != listed or len(artifacts) != payload.get(
        "artifact_count_excluding_manifest"
    ):
        raise RuntimeError("Build provenance contains extra or missing artifacts")
    recorded_side_effects = payload.get("bundle_tool_side_effect_artifacts")
    if not isinstance(recorded_side_effects, list):
        raise RuntimeError("Build provenance bundle side-effect inventory is missing")
    actual_side_effects = [
        item
        for item in artifacts
        if str(item["path"]).startswith("arduino_cli_compile_build_side_effect/")
        or str(item["path"]).startswith("arduino_cli_upload_build_side_effect/")
    ]
    if recorded_side_effects != actual_side_effects:
        raise RuntimeError("Build provenance bundle side-effect inventory differs")
    before = payload.get("build_artifacts_before_upload")
    after = payload.get("build_artifacts_after_upload")
    uploader_added = _validate_upload_artifact_delta(
        before,
        after,
        board=str(payload.get("board")),
        bundle_side_effects=recorded_side_effects,
    )
    if payload.get("uploader_added_artifacts") != uploader_added:
        raise RuntimeError("Build provenance uploader-added inventory differs")
    current_build = [
        {
            "path": item["path"].removeprefix("build_artifacts/"),
            "size_bytes": item["size_bytes"],
            "sha256": item["sha256"],
        }
        for item in artifacts
        if str(item["path"]).startswith("build_artifacts/")
    ]
    if current_build != after:
        raise RuntimeError("Build provenance compile-output inventory differs")
    commands = payload.get("commands")
    if not isinstance(commands, Mapping) or set(commands) != {
        "frontend", "cores", "board", "compile", "upload"
    }:
        raise RuntimeError("Build provenance command ledger is incomplete")
    for name, command in commands.items():
        if (
            command.get("returncode") != 0
            or command.get("timed_out") is not False
            or _parse_timestamp(command["finished_utc"])
            < _parse_timestamp(command["started_utc"])
            or not isinstance(command.get("command"), list)
        ):
            raise RuntimeError(f"Build provenance command did not pass: {name}")
        log = command.get("log", {})
        if (
            not isinstance(log.get("path"), str)
            or log["path"] not in listed
            or (root / log["path"]).stat().st_size != log.get("size_bytes")
            or log.get("size_bytes", 0) <= 0
            or sha256_file(root / log["path"]) != log.get("sha256")
        ):
            raise RuntimeError(f"Build provenance command log differs: {name}")
    if not payload.get("physical_port") or not payload.get("physical_port_serial"):
        raise RuntimeError("Build provenance lacks physical-port identity")
    for field in ["model_sha256", "checkpoint_file_sha256", "export_id"]:
        if not _is_sha256(payload.get(field)):
            raise RuntimeError(f"Build provenance model identity is invalid: {field}")
    validate_host_environment(
        payload.get("host_environment", {}), source_root=host_source_root
    )
    runtime = parse_runtime_identity(str(payload.get("post_reset_runtime_identity")))
    for field, expected in {
        "bundle_id": payload.get("bundle_id"),
        "build_contract_id": payload.get("build_contract_id"),
        "board": payload.get("board"),
        "transport": payload.get("transport"),
        "student": payload.get("student"),
        "route": payload.get("route"),
        "export_id": payload.get("export_id"),
        "model_sha256": payload.get("model_sha256"),
    }.items():
        if runtime[field] != expected:
            raise RuntimeError(f"Provenance runtime identity differs for {field}")
    contract = validate_build_contract(payload.get("build_contract", {}))
    if contract["build_contract_id"] != payload.get("build_contract_id"):
        raise RuntimeError("Build provenance contract ID differs")
    recorded_bundle = payload.get("bundle_path_recorded")
    recorded_build = payload.get("build_dir_path_recorded")
    if not isinstance(recorded_bundle, str) or not isinstance(recorded_build, str):
        raise RuntimeError("Build provenance command paths are missing")
    cli = contract["compile_command"][0]
    expected_inspection = {
        "frontend": [cli, "version", "--format", "json"],
        "cores": [cli, "core", "list", "--format", "json"],
        "board": [
            cli,
            "board",
            "details",
            "--fqbn",
            contract["fqbn"],
            "--format",
            "json",
        ],
    }
    for name, expected_command in expected_inspection.items():
        if commands[name]["command"] != expected_command:
            raise RuntimeError(f"Build provenance inspection command differs: {name}")
    inspection_text = _validate_arduino_inspection(
        contract=contract,
        frontend_log=root / commands["frontend"]["log"]["path"],
        cores_log=root / commands["cores"]["log"]["path"],
        board_log=root / commands["board"]["log"]["path"],
    )
    compile_text = (root / commands["compile"]["log"]["path"]).read_text(
        encoding="utf-8", errors="strict"
    )
    expected_compile = _render_build_command(
        contract["compile_command"],
        bundle=recorded_bundle,
        fqbn=contract["fqbn"],
        port=payload["physical_port"],
        build_dir=recorded_build,
    )
    expected_upload = _render_build_command(
        contract["upload_command"],
        bundle=recorded_bundle,
        fqbn=contract["fqbn"],
        port=payload["physical_port"],
        build_dir=recorded_build,
    )
    if commands["compile"]["command"] != expected_compile:
        raise RuntimeError("Build provenance compile command binding differs")
    if commands["upload"]["command"] != expected_upload:
        raise RuntimeError("Build provenance upload command binding differs")
    previous_finished: datetime | None = None
    for name in ["frontend", "cores", "board", "compile", "upload"]:
        started = _parse_timestamp(commands[name]["started_utc"])
        finished = _parse_timestamp(commands[name]["finished_utc"])
        if previous_finished is not None and started < previous_finished:
            raise RuntimeError("Build provenance commands overlap or are reordered")
        previous_finished = finished
    if bundle_dir is not None:
        if expected_export is None:
            raise ValueError(
                "Bundle provenance verification requires an independently validated export"
            )
        bundle_root = bundle_dir.resolve()
        bundle = verify_final_bundle(
            bundle_root,
            expected_export=expected_export,
            **_bundle_source_options(host_source_root),
        )
        expected = {
            "bundle_manifest_sha256": bundle["_manifest_sha256"],
            "bundle_id": bundle["bundle_id"],
            "build_contract_id": bundle["build_contract_id"],
            "board": bundle["board"],
            "transport": bundle["transport"],
            "student": bundle["student"],
            "route": bundle["route"],
            "export_id": bundle["export_id"],
            "model_sha256": bundle["model_sha256"],
            "checkpoint_file_sha256": bundle["checkpoint_file_sha256"],
            "post_reset_runtime_identity": bundle["runtime_identity_response"],
        }
        for field, value in expected.items():
            if payload.get(field) != value:
                raise RuntimeError(f"Build provenance differs from bundle for {field}")
    return payload


def _expected_session_row_totals() -> dict[str, int]:
    totals = {
        "all_rows": 0,
        "timing_rows": 0,
        "full_rows": 0,
        "warmup_rows": 0,
        "smoke_rows": 0,
    }
    role_totals = {
        "balanced_timing_warmup": "warmup_rows",
        "full_replay_smoke_prefix": "smoke_rows",
        "balanced_timing": "timing_rows",
        "full_replay": "full_rows",
    }
    for stage in FINAL_STAGES:
        role = stage.get("input_role")
        if role not in role_totals:
            raise RuntimeError(f"Unknown final HIL input role: {role!r}")
        rows = int(stage["rows"])
        totals["all_rows"] += rows
        totals[role_totals[str(role)]] += rows
    return totals


def validate_session_stage_ledger(stages: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Pure six-stage gate used by completion and tamper tests."""

    expected_names = [stage["name"] for stage in FINAL_STAGES]
    observed_names = [str(stage.get("name")) for stage in stages]
    if observed_names != expected_names:
        raise RuntimeError(
            f"Session stage sequence is incomplete or reordered: {observed_names}"
        )
    attempt_ids: set[str] = set()
    previous_finished: datetime | None = None
    for observed, expected in zip(stages, FINAL_STAGES):
        if observed.get("status") != "passed":
            raise RuntimeError(f"Session contains an inadmissible stage: {expected['name']}")
        for field, value in expected.items():
            if observed.get(field) != value or (
                isinstance(value, bool) and observed.get(field) is not value
            ):
                raise RuntimeError(
                    f"Session stage contract differs: {expected['name']}:{field}"
                )
        attempt_id = observed.get("attempt_id")
        if (
            not isinstance(attempt_id, str)
            or len(attempt_id) != 32
            or any(character not in "0123456789ABCDEF" for character in attempt_id)
            or attempt_id in attempt_ids
        ):
            raise RuntimeError("Session has a missing or duplicate attempt ID")
        attempt_ids.add(attempt_id)
        started = _parse_timestamp(str(observed["started_utc"]))
        finished = _parse_timestamp(str(observed["finished_utc"]))
        if finished < started or (previous_finished is not None and started < previous_finished):
            raise RuntimeError("Session stages overlap or have reversed timestamps")
        previous_finished = finished
    return _expected_session_row_totals()


def complete_six_stage_session(
    *,
    attempt_dirs: Sequence[Path],
    export_dir: Path,
    cohort_dir: Path,
    bundle_dir: Path,
    provenance_json: Path,
    output_json: Path,
    verifier: Verifier | None = None,
) -> Path:
    if len(attempt_dirs) != len(FINAL_STAGES):
        raise RuntimeError("Six finalized attempts are required")
    cached_verifier = _memoized_export_verifier(verifier)
    export = validate_final_export(export_dir, verifier=cached_verifier)
    cohort = validate_balanced_cohort(
        cohort_dir,
        identities={export.model_key: export},
        reconstruct_sources=False,
        allow_identity_subset=True,
    )
    cohort_model = cohort.get("models", {}).get(export.model_key)
    if not isinstance(cohort_model, Mapping):
        raise RuntimeError("Balanced cohort does not contain this final model")
    for field in [
        "export_id",
        "trained_state_sha256",
        "full_replay_sha256",
        "full_reference_sha256",
    ]:
        if cohort_model.get(field) != getattr(export, field):
            raise RuntimeError(f"Balanced cohort differs from final export for {field}")
    bundle = verify_final_bundle(bundle_dir, expected_export=export)
    provenance = validate_build_upload_provenance(
        provenance_json,
        bundle_dir=bundle_dir,
        expected_export=export,
    )
    attempts = [
        verify_stage_attempt(
            path,
            export_dir=export_dir,
            cohort_dir=cohort_dir,
            bundle_dir=bundle_dir,
            verifier=cached_verifier,
        )
        for path in attempt_dirs
    ]
    if len({path.resolve() for path in attempt_dirs}) != len(FINAL_STAGES):
        raise RuntimeError("Six-stage completion contains a repeated attempt directory")
    attempt_paths = {
        attempt["attempt_id"]: path.resolve()
        for path, attempt in zip(attempt_dirs, attempts)
    }
    if len(attempt_paths) != len(FINAL_STAGES):
        raise RuntimeError("Six-stage completion contains a repeated attempt ID")
    attempts.sort(key=lambda item: int(item["stage"]["ordinal"]))
    campaign_sessions = {item["campaign_session_id"] for item in attempts}
    bundle_ids = {item["bundle_id"] for item in attempts}
    combinations = {
        (
            item["combination"]["student"],
            item["combination"]["route"],
            item["combination"]["board"],
            item["combination"]["transport"],
        )
        for item in attempts
    }
    if len(campaign_sessions) != 1 or bundle_ids != {bundle["bundle_id"]} or len(
        combinations
    ) != 1:
        raise RuntimeError("Six-stage attempts do not belong to one campaign session")
    wifi_binding = _validate_attempt_connection_set(attempts, bundle["transport"])
    require_session_id(next(iter(campaign_sessions)))
    stages = [
        {
            **dict(attempt["stage"]),
            "status": attempt["status"],
            "attempt_id": attempt["attempt_id"],
            "attempt_path_recorded": str(
                attempt_paths[attempt["attempt_id"]]
            ),
            "attempt_manifest_sha256": sha256_file(
                attempt_paths[attempt["attempt_id"]] / "final_attempt.json"
            ),
            "started_utc": attempt["started_utc"],
            "finished_utc": attempt["finished_utc"],
        }
        for attempt in attempts
    ]
    totals = validate_session_stage_ledger(stages)
    if _parse_timestamp(provenance["finished_utc"]) > _parse_timestamp(
        stages[0]["started_utc"]
    ):
        raise RuntimeError("Build/upload provenance overlaps the six-stage execution")
    student, route, board, transport = next(iter(combinations))
    physical_serials = {
        attempt["physical_identity"]["physical_port_serial"] for attempt in attempts
    }
    if physical_serials != {provenance["physical_port_serial"]}:
        raise RuntimeError("Build and execution physical-port identities differ")
    wifi_mac_reported = None
    if transport == "wifi_udp":
        wifi_macs = {
            attempt["physical_identity"].get("wifi_mac_reported")
            for attempt in attempts
        }
        if len(wifi_macs) != 1 or not next(iter(wifi_macs)):
            raise RuntimeError("Wi-Fi stages do not identify one physical radio")
        wifi_mac_reported = next(iter(wifi_macs))
    payload: dict[str, Any] = {
        "schema": SESSION_SCHEMA,
        "status": "passed",
        "campaign_session_id": next(iter(campaign_sessions)),
        "started_utc": stages[0]["started_utc"],
        "finished_utc": stages[-1]["finished_utc"],
        "combination_id": (
            f"student_{student}_{route}__{board}__{transport}"
        ),
        "model_key": f"student_{student}_{route}",
        "student": student,
        "route": route,
        "board": board,
        "transport": transport,
        "export_id": bundle["export_id"],
        "model_sha256": bundle["model_sha256"],
        "checkpoint_file_sha256": bundle["checkpoint_file_sha256"],
        "bundle_id": bundle["bundle_id"],
        "build_contract_id": bundle["build_contract_id"],
        "export_path_recorded": str(export_dir.resolve()),
        "export_manifest_sha256": export.manifest_sha256,
        "cohort_path_recorded": str(cohort_dir.resolve()),
        "cohort_manifest_sha256": sha256_file(
            cohort_dir.resolve() / "final_timing_cohort_manifest.json"
        ),
        "bundle_path_recorded": str(bundle_dir.resolve()),
        "bundle_manifest_sha256": bundle["_manifest_sha256"],
        "physical_port_serial": provenance["physical_port_serial"],
        "wifi_mac_reported": wifi_mac_reported,
        "provenance_id": provenance["provenance_id"],
        "provenance_path_recorded": str(provenance_json.resolve()),
        "provenance_sha256": sha256_file(provenance_json.resolve()),
        "stages": stages,
        "row_totals": totals,
        "warmup_excluded_from_reported_metrics": True,
        "smoke_excluded_from_reported_metrics": True,
        "timing_statistical_unit": TIMING_STATISTICAL_UNIT,
        **_session_connection_fields(wifi_binding),
    }
    connection = validate_session_connection_record(
        session=payload,
        attempts=attempts,
        bundle=bundle,
    )
    if connection is not None and _parse_timestamp(
        str(provenance["finished_utc"])
    ) > _parse_timestamp(str(connection["started_utc"])):
        raise RuntimeError("Build/upload overlaps Wi-Fi provisioning")
    payload = _sealed_payload(payload, "session_evidence_id")
    atomic_write_json(output_json.resolve(), payload)
    return output_json.resolve()


def validate_session_completion(
    path: Path,
    *,
    export_dir: Path | None = None,
    cohort_dir: Path | None = None,
    bundle_dir: Path | None = None,
    provenance_json: Path | None = None,
    connection_json: Path | None = None,
    attempt_dirs: Mapping[str, Path] | None = None,
    verifier: Verifier | None = None,
    host_source_root: Path | None = None,
) -> dict[str, Any]:
    source_path = Path(path)
    if source_path.is_symlink():
        raise RuntimeError("Session completion cannot be a symlink")
    payload = read_json(source_path.resolve())
    recorded = payload.get("session_evidence_id")
    copy = dict(payload)
    copy.pop("session_evidence_id", None)
    if recorded != canonical_json_sha256(copy):
        raise RuntimeError("Session completion ID is invalid")
    if payload.get("schema") != SESSION_SCHEMA or payload.get("status") != "passed":
        raise RuntimeError("Session completion is not passed final evidence")
    totals = validate_session_stage_ledger(payload.get("stages", []))
    if payload.get("row_totals") != totals:
        raise RuntimeError("Session completion row totals changed")
    if payload.get("warmup_excluded_from_reported_metrics") is not True:
        raise RuntimeError("Session completion includes warmup in reported metrics")
    if payload.get("smoke_excluded_from_reported_metrics") is not True:
        raise RuntimeError("Session completion includes smoke in reported metrics")
    if payload.get("timing_statistical_unit") != TIMING_STATISTICAL_UNIT:
        raise RuntimeError("Session timing statistical unit is invalid")
    require_session_id(str(payload.get("campaign_session_id", "")))
    expected_model_key = f"student_{payload.get('student')}_{payload.get('route')}"
    expected_combination = (
        f"{expected_model_key}__{payload.get('board')}__{payload.get('transport')}"
    )
    if (
        expected_model_key not in MODEL_KEYS
        or payload.get("board") not in BOARDS
        or payload.get("transport") not in TRANSPORTS
        or payload.get("model_key") != expected_model_key
        or payload.get("combination_id") != expected_combination
    ):
        raise RuntimeError("Session model/board/transport identity is invalid")
    if (
        payload.get("started_utc") != payload["stages"][0]["started_utc"]
        or payload.get("finished_utc") != payload["stages"][-1]["finished_utc"]
    ):
        raise RuntimeError("Session UTC interval differs from its stage ledger")
    selected_export = (
        Path(export_dir)
        if export_dir is not None
        else Path(payload["export_path_recorded"])
    )
    selected_cohort = (
        Path(cohort_dir)
        if cohort_dir is not None
        else Path(payload["cohort_path_recorded"])
    )
    selected_bundle = (
        Path(bundle_dir)
        if bundle_dir is not None
        else Path(payload["bundle_path_recorded"])
    )
    cached_verifier = _memoized_export_verifier(verifier)
    export = validate_final_export(selected_export, verifier=cached_verifier)
    cohort = validate_balanced_cohort(
        selected_cohort,
        identities={export.model_key: export},
        reconstruct_sources=False,
        allow_identity_subset=True,
    )
    bundle = verify_final_bundle(
        selected_bundle,
        expected_export=export,
        **_bundle_source_options(host_source_root),
    )
    if (
        export.manifest_sha256 != payload.get("export_manifest_sha256")
        or sha256_file(
            selected_cohort / "final_timing_cohort_manifest.json"
        )
        != payload.get("cohort_manifest_sha256")
        or bundle["_manifest_sha256"] != payload.get("bundle_manifest_sha256")
    ):
        raise RuntimeError("Session export/cohort/bundle manifest binding changed")
    cohort_model = cohort.get("models", {}).get(payload["model_key"])
    if not isinstance(cohort_model, Mapping):
        raise RuntimeError("Session model is absent from the balanced cohort")
    for field, value in {
        "export_id": export.export_id,
        "trained_state_sha256": export.trained_state_sha256,
        "full_replay_sha256": export.full_replay_sha256,
        "full_reference_sha256": export.full_reference_sha256,
    }.items():
        if cohort_model.get(field) != value:
            raise RuntimeError(f"Session cohort/export binding differs for {field}")
    for field, value in {
        "export_id": export.export_id,
        "model_sha256": export.trained_state_sha256,
        "checkpoint_file_sha256": export.checkpoint_file_sha256,
        "bundle_id": bundle["bundle_id"],
        "build_contract_id": bundle["build_contract_id"],
    }.items():
        if payload.get(field) != value:
            raise RuntimeError(f"Session identity differs from bound artifacts: {field}")
    expected_attempt_ids = {stage["attempt_id"] for stage in payload["stages"]}
    if attempt_dirs is not None and set(attempt_dirs) != expected_attempt_ids:
        raise RuntimeError("Session attempt override set is incomplete or unexpected")
    verified_attempts: list[dict[str, Any]] = []
    for stage in payload["stages"]:
        attempt_path = (
            Path(attempt_dirs.get(stage["attempt_id"]))
            if attempt_dirs is not None and stage["attempt_id"] in attempt_dirs
            else Path(stage["attempt_path_recorded"])
        )
        attempt_manifest = attempt_path / "final_attempt.json"
        if (
            not attempt_manifest.is_file()
            or sha256_file(attempt_manifest) != stage["attempt_manifest_sha256"]
        ):
            raise RuntimeError(f"Session attempt changed: {stage['name']}")
        attempt = verify_stage_attempt(
            attempt_path,
            export_dir=selected_export,
            cohort_dir=selected_cohort,
            bundle_dir=selected_bundle,
            verifier=cached_verifier,
            host_source_root=host_source_root,
        )
        verification = attempt.get("verification")
        if (
            attempt.get("status") != "passed"
            or attempt.get("attempt_id") != stage["attempt_id"]
            or attempt.get("campaign_session_id") != payload["campaign_session_id"]
            or attempt.get("bundle_id") != payload["bundle_id"]
            or attempt.get("build_contract_id") != payload["build_contract_id"]
            or attempt.get("combination")
            != {
                "student": payload["student"],
                "route": payload["route"],
                "board": payload["board"],
                "transport": payload["transport"],
            }
            or attempt.get("completed_rows") != stage["rows"]
            or attempt.get("stage", {}).get("name") != stage["name"]
            or not isinstance(verification, dict)
            or verification.get("status") != "passed"
            or verification.get("sequence_exact") is not True
            or verification.get("predictions_exact") is not True
            or verification.get("logits_exact") is not True
        ):
            raise RuntimeError(f"Session attempt is not exact passed evidence: {stage['name']}")
        if attempt.get("physical_identity", {}).get(
            "physical_port_serial"
        ) != payload.get("physical_port_serial"):
            raise RuntimeError(f"Session attempt uses another physical board: {stage['name']}")
        if payload["transport"] == "wifi_udp" and attempt.get(
            "physical_identity", {}
        ).get("wifi_mac_reported") != payload.get("wifi_mac_reported"):
            raise RuntimeError(f"Session attempt uses another Wi-Fi radio: {stage['name']}")
        if payload["transport"] == "wifi_udp" and not isinstance(
            verification.get("wifi_retry_reconciliation"), dict
        ):
            raise RuntimeError(f"Wi-Fi retry reconciliation is missing: {stage['name']}")
        if payload["transport"] == "usb_serial" and verification.get(
            "wifi_retry_reconciliation"
        ) is not None:
            raise RuntimeError(f"USB stage carries Wi-Fi retry evidence: {stage['name']}")
        verified_attempts.append(attempt)
    connection = validate_session_connection_record(
        session=payload,
        attempts=verified_attempts,
        bundle=bundle,
        connection_json=connection_json,
        host_source_root=host_source_root,
    )
    provenance = (
        Path(provenance_json)
        if provenance_json is not None
        else Path(payload["provenance_path_recorded"])
    )
    if not provenance.is_file() or sha256_file(provenance) != payload["provenance_sha256"]:
        raise RuntimeError("Session build/upload provenance changed")
    provenance_payload = validate_build_upload_provenance(
        provenance,
        bundle_dir=selected_bundle,
        expected_export=export,
        host_source_root=host_source_root,
    )
    if (
        provenance_payload.get("provenance_id") != payload.get("provenance_id")
        or provenance_payload.get("bundle_id") != payload.get("bundle_id")
        or provenance_payload.get("build_contract_id")
        != payload.get("build_contract_id")
        or provenance_payload.get("board") != payload.get("board")
        or provenance_payload.get("transport") != payload.get("transport")
        or provenance_payload.get("physical_port_serial")
        != payload.get("physical_port_serial")
        or provenance_payload.get("student") != payload.get("student")
        or provenance_payload.get("route") != payload.get("route")
        or provenance_payload.get("export_id") != payload.get("export_id")
        or provenance_payload.get("model_sha256") != payload.get("model_sha256")
        or provenance_payload.get("checkpoint_file_sha256")
        != payload.get("checkpoint_file_sha256")
    ):
        raise RuntimeError("Session build/upload provenance identity differs")
    if _parse_timestamp(str(provenance_payload["finished_utc"])) > _parse_timestamp(
        str(payload["stages"][0]["started_utc"])
    ):
        raise RuntimeError("Build/upload provenance overlaps the six-stage execution")
    if connection is not None and _parse_timestamp(
        str(provenance_payload["finished_utc"])
    ) > _parse_timestamp(str(connection["started_utc"])):
        raise RuntimeError("Build/upload overlaps Wi-Fi provisioning")
    return payload


def validate_common_board_build_contracts(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Require one exact FQBN/core/toolchain contract per physical board type."""

    result: dict[str, str] = {}
    for board in BOARDS:
        board_records = [record for record in records if record.get("board") == board]
        if not board_records:
            raise RuntimeError(f"No eligible build records exist for {board}")
        build_ids = {record.get("build_contract_id") for record in board_records}
        if len(build_ids) != 1 or not _is_sha256(next(iter(build_ids))):
            raise RuntimeError(
                f"Eligible {board} comparisons use different build contracts"
            )
        result[board] = str(next(iter(build_ids)))
    return result


def preflight_campaign(
    *,
    campaign_contract: Mapping[str, Any] | Path,
    cohort_dir: Path | None,
    export_dirs: Mapping[str, Path],
    bundle_dirs: Mapping[str, Path],
    output_json: Path | None = None,
    verifier: Verifier | None = None,
) -> dict[str, Any]:
    contract = validate_campaign_contract(campaign_contract)
    blockers: list[dict[str, str]] = []
    if contract["status"] == "blocked":
        blockers.extend(contract["blockers"])
    eligible_model_keys = [
        key for key in MODEL_KEYS if contract["models"][key].get("status") == "passed"
    ]
    cohort_sha256 = None
    validated_exports: dict[str, FinalExportIdentity] = {}
    cached_verifier = _memoized_export_verifier(verifier)
    if not blockers:
        if set(export_dirs) != set(eligible_model_keys):
            blockers.append(
                {
                    "scope": "exports",
                    "reason": "local final-export assignments differ from gate-eligible routes",
                }
            )
        else:
            try:
                for key in eligible_model_keys:
                    export = validate_final_export(
                        Path(export_dirs[key]), verifier=cached_verifier
                    )
                    if export.model_key != key:
                        raise RuntimeError(f"local final export identifies another model: {key}")
                    model = contract["models"][key]
                    for field, value in {
                        "export_id": export.export_id,
                        "trained_state_sha256": export.trained_state_sha256,
                        "checkpoint_file_sha256": export.checkpoint_file_sha256,
                        "manifest_sha256": export.manifest_sha256,
                    }.items():
                        if model.get(field) != value:
                            raise RuntimeError(
                                f"local final export differs from campaign for {key}:{field}"
                            )
                    validated_exports[key] = export
            except Exception as exc:
                blockers.append(
                    {
                        "scope": "exports",
                        "reason": f"{exc.__class__.__name__}: {exc}",
                    }
                )
    if not blockers:
        if cohort_dir is None:
            blockers.append({"scope": "cohort", "reason": "balanced cohort is missing"})
        else:
            try:
                cohort = validate_balanced_cohort(
                    cohort_dir,
                    identities=validated_exports,
                    reconstruct_sources=False,
                )
                if set(cohort.get("models", {})) != set(eligible_model_keys):
                    raise RuntimeError("cohort model set differs from gate-eligible routes")
                for key in eligible_model_keys:
                    model = contract["models"][key]
                    entry = cohort["models"][key]
                    for cohort_field, model_field in {
                        "export_id": "export_id",
                        "trained_state_sha256": "trained_state_sha256",
                        "full_replay_sha256": "full_replay_sha256",
                        "full_reference_sha256": "full_reference_sha256",
                    }.items():
                        if entry.get(cohort_field) != model.get(model_field):
                            raise RuntimeError(
                                f"cohort differs for {key}:{cohort_field}"
                            )
                first_model = contract["models"][eligible_model_keys[0]]
                if (
                    cohort.get("source_protocol_id") != first_model.get("protocol")
                    or cohort.get("dataset", {}).get("sha256")
                    != first_model.get("dataset_sha256")
                    or cohort.get("split", {}).get("split_indices_sha256")
                    != first_model.get("split_indices_sha256")
                    or cohort.get("split", {}).get("scaler_sha256")
                    != first_model.get("scaler_sha256")
                ):
                    raise RuntimeError("cohort preprocessing lineage differs from campaign")
                cohort_sha256 = sha256_file(
                    cohort_dir.resolve() / "final_timing_cohort_manifest.json"
                )
            except Exception as exc:
                blockers.append(
                    {
                        "scope": "cohort",
                        "reason": f"{exc.__class__.__name__}: {exc}",
                    }
                )
    bundle_ledger: dict[str, Any] = {}
    passed_bundle_records: list[dict[str, Any]] = []
    intended_ids = [item["combination_id"] for item in contract["combinations"]]
    expected_ids = [
        item["combination_id"] for item in contract["eligible_combinations"]
    ]
    for combination in contract["excluded_combinations"]:
        bundle_ledger[combination["combination_id"]] = {
            "status": "not_applicable",
            "model_status": combination["model_status"],
            "reason": combination["reason"],
        }
    for combination in contract["eligible_combinations"]:
        combination_id = combination["combination_id"]
        path = bundle_dirs.get(combination_id)
        if path is None:
            blockers.append(
                {"scope": combination_id, "reason": "final bundle is missing"}
            )
            bundle_ledger[combination_id] = {"status": "missing"}
            continue
        try:
            export = validated_exports.get(combination["model_key"])
            if export is None:
                raise RuntimeError("independently validated final export is unavailable")
            bundle = verify_final_bundle(path, expected_export=export)
            model = contract["models"][combination["model_key"]]
            expected = {
                "student": model.get("student"),
                "route": model.get("route"),
                "export_id": model.get("export_id"),
                "model_sha256": model.get("trained_state_sha256"),
                "checkpoint_file_sha256": model.get("checkpoint_file_sha256"),
                "board": combination["board"],
                "transport": combination["transport"],
            }
            for field, value in expected.items():
                if bundle.get(field) != value:
                    raise RuntimeError(f"bundle differs for {field}")
            bundle_ledger[combination_id] = {
                "status": "passed",
                "path_recorded": str(path.resolve()),
                "bundle_id": bundle["bundle_id"],
                "board": bundle["board"],
                "transport": bundle["transport"],
                "build_contract_id": bundle["build_contract_id"],
                "manifest_sha256": bundle["_manifest_sha256"],
            }
            passed_bundle_records.append(bundle_ledger[combination_id])
        except Exception as exc:
            reason = f"{exc.__class__.__name__}: {exc}"
            blockers.append({"scope": combination_id, "reason": reason})
            bundle_ledger[combination_id] = {"status": "invalid", "reason": reason}
    extras = set(bundle_dirs) - set(expected_ids)
    if extras:
        blockers.append(
            {"scope": "bundles", "reason": f"unexpected combinations: {sorted(extras)}"}
        )
    board_build_contracts = None
    if len(passed_bundle_records) == len(contract["eligible_combinations"]):
        try:
            board_build_contracts = validate_common_board_build_contracts(
                passed_bundle_records
            )
        except Exception as exc:
            blockers.append(
                {
                    "scope": "build_contracts",
                    "reason": f"{exc.__class__.__name__}: {exc}",
                }
            )
    payload: dict[str, Any] = {
        "schema": "cukd_final_hil_campaign_preflight_v1",
        "status": contract["status"] if not blockers else "blocked",
        "contract_id": contract["contract_id"],
        "intended_combination_count": contract["expected_combination_count"],
        "intended_combinations": intended_ids,
        "eligible_combination_count": contract[
            "expected_eligible_combination_count"
        ],
        "eligible_combinations": expected_ids,
        "excluded_combinations": [
            item["combination_id"] for item in contract["excluded_combinations"]
        ],
        "cohort_manifest_sha256": cohort_sha256,
        "bundles": bundle_ledger,
        "board_build_contracts": board_build_contracts,
        "blockers": blockers,
        "blocked_routes": contract["blocked_routes"],
        "intended_matrix_was_retained": len(intended_ids)
        == len(MODEL_KEYS) * len(BOARDS) * len(contract["transports"]),
        "execution_matrix_is_gate_derived": True,
    }
    payload = _sealed_payload(payload, "preflight_id")
    if output_json is not None:
        atomic_write_json(output_json.resolve(), payload)
    return payload


def validate_campaign_session_ledger(
    contract: Mapping[str, Any], sessions: Sequence[Mapping[str, Any]]
) -> dict[str, int]:
    """Pure exact-matrix gate used by campaign completion and tamper tests."""

    expected = [
        item["combination_id"] for item in contract["eligible_combinations"]
    ]
    observed = [str(item.get("combination_id")) for item in sessions]
    if len(observed) != len(set(observed)):
        raise RuntimeError("Campaign contains a duplicate combination")
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        unexpected = sorted(set(observed) - set(expected))
        raise RuntimeError(
            f"Campaign matrix is incomplete: missing={missing}, unexpected={unexpected}"
        )
    expected_session_rows = _expected_session_row_totals()
    for session in sessions:
        if session.get("row_totals") != expected_session_rows:
            raise RuntimeError(
                f"Campaign session row totals are invalid: {session.get('combination_id')}"
            )
        if len(session.get("stages", [])) != len(FINAL_STAGES):
            raise RuntimeError(
                f"Campaign session stage count is invalid: {session.get('combination_id')}"
            )
    totals = {
        "session_count": len(sessions),
        "stage_attempts": sum(len(item.get("stages", [])) for item in sessions),
        "balanced_timing_rows": sum(
            int(item["row_totals"]["timing_rows"]) for item in sessions
        ),
        "warmup_rows_excluded": sum(
            int(item["row_totals"]["warmup_rows"]) for item in sessions
        ),
        "smoke_rows_excluded": sum(
            int(item["row_totals"]["smoke_rows"]) for item in sessions
        ),
        "full_exact_replay_rows": sum(
            int(item["row_totals"]["full_rows"]) for item in sessions
        ),
        "all_device_inferences": sum(
            int(item["row_totals"]["all_rows"]) for item in sessions
        ),
    }
    expected_totals = {
        "session_count": contract["expected_eligible_combination_count"],
        "stage_attempts": contract["expected_eligible_stage_attempts"],
        "balanced_timing_rows": contract["expected_eligible_rows"][
            "balanced_timing"
        ],
        "warmup_rows_excluded": contract["expected_eligible_rows"][
            "warmup_excluded"
        ],
        "smoke_rows_excluded": contract["expected_eligible_rows"]["smoke"],
        "full_exact_replay_rows": contract["expected_eligible_rows"][
            "full_exact_replay"
        ],
        "all_device_inferences": contract["expected_eligible_rows"][
            "all_device_inferences"
        ],
    }
    if totals != expected_totals:
        raise RuntimeError(f"Campaign totals are invalid: {totals}")
    return totals


def verify_complete_campaign(
    *,
    campaign_contract: Mapping[str, Any] | Path,
    session_jsons: Sequence[Path],
    output_json: Path | None = None,
    verifier: Verifier | None = None,
    session_contexts: Mapping[str, Mapping[str, Any]] | None = None,
    host_source_root: Path | None = None,
) -> dict[str, Any]:
    contract = validate_campaign_contract(campaign_contract)
    if contract["status"] not in {"ready", "ready_with_blocked_routes"}:
        raise RuntimeError("Fatal campaign preflight blockers prevent completion")
    cached_verifier = _memoized_export_verifier(verifier)
    session_paths = [Path(path) for path in session_jsons]
    if any(path.is_symlink() for path in session_paths):
        raise RuntimeError("Campaign session completion cannot be a symlink")
    if len({path.resolve() for path in session_paths}) != len(session_paths):
        raise RuntimeError("Campaign contains a repeated session path")
    raw_sessions = [read_json(path) for path in session_paths]
    expected_combinations = {
        item["combination_id"] for item in contract["eligible_combinations"]
    }
    observed_combinations = [
        str(session.get("combination_id")) for session in raw_sessions
    ]
    if len(observed_combinations) != len(set(observed_combinations)):
        raise RuntimeError("Campaign contains a duplicate combination")
    if session_contexts is not None and set(session_contexts) != expected_combinations:
        raise RuntimeError(
            "Campaign session-context set is incomplete or unexpected"
        )

    sessions: list[dict[str, Any]] = []
    for path, raw in zip(session_paths, raw_sessions):
        combination_id = str(raw.get("combination_id"))
        if session_contexts is None:
            sessions.append(
                validate_session_completion(
                    path,
                    verifier=cached_verifier,
                    host_source_root=host_source_root,
                )
            )
            continue
        context = session_contexts.get(combination_id)
        if not isinstance(context, Mapping) or set(context) != {
            "export_dir",
            "cohort_dir",
            "bundle_dir",
            "provenance_json",
            "connection_json",
            "attempt_dirs",
        }:
            raise RuntimeError(
                f"Campaign session context is malformed: {combination_id}"
            )
        stage_records = raw.get("stages")
        stage_dirs = context.get("attempt_dirs")
        if not isinstance(stage_records, list) or not isinstance(stage_dirs, Mapping):
            raise RuntimeError(
                f"Campaign session attempt context is malformed: {combination_id}"
            )
        expected_stage_names = {stage["name"] for stage in FINAL_STAGES}
        if set(stage_dirs) != expected_stage_names:
            raise RuntimeError(
                f"Campaign session attempt context is incomplete: {combination_id}"
            )
        by_name = {
            str(stage.get("name")): stage
            for stage in stage_records
            if isinstance(stage, Mapping)
        }
        if set(by_name) != expected_stage_names:
            raise RuntimeError(
                f"Campaign session stage ledger is malformed: {combination_id}"
            )
        attempt_dirs = {
            str(by_name[name]["attempt_id"]): Path(stage_dirs[name])
            for name in expected_stage_names
        }
        sessions.append(
            validate_session_completion(
                path,
                export_dir=Path(context["export_dir"]),
                cohort_dir=Path(context["cohort_dir"]),
                bundle_dir=Path(context["bundle_dir"]),
                provenance_json=Path(context["provenance_json"]),
                connection_json=(
                    Path(context["connection_json"])
                    if context["connection_json"] is not None
                    else None
                ),
                attempt_dirs=attempt_dirs,
                verifier=cached_verifier,
                host_source_root=host_source_root,
            )
        )
    for field in [
        "session_evidence_id",
        "campaign_session_id",
        "bundle_id",
        "provenance_id",
    ]:
        values = [session.get(field) for session in sessions]
        if len(values) != len(set(values)):
            raise RuntimeError(f"Campaign contains a duplicate session identity: {field}")
    totals = validate_campaign_session_ledger(contract, sessions)
    board_build_contracts = validate_common_board_build_contracts(sessions)
    by_id = {session["combination_id"]: session for session in sessions}
    path_by_session_id = {
        session["session_evidence_id"]: path
        for session, path in zip(sessions, session_paths)
    }
    ordered = []
    for combination in contract["eligible_combinations"]:
        session = by_id[combination["combination_id"]]
        model = contract["models"][combination["model_key"]]
        expected = {
            "model_key": combination["model_key"],
            "board": combination["board"],
            "transport": combination["transport"],
            "export_id": model["export_id"],
            "model_sha256": model["trained_state_sha256"],
            "checkpoint_file_sha256": model["checkpoint_file_sha256"],
        }
        for field, value in expected.items():
            if session.get(field) != value:
                raise RuntimeError(
                    f"Campaign session {session['combination_id']} differs for {field}"
                )
        ordered.append(
            {
                "combination_id": session["combination_id"],
                "session_evidence_id": session["session_evidence_id"],
                "session_path_recorded": str(
                    path_by_session_id[session["session_evidence_id"]]
                ),
                "session_sha256": sha256_file(
                    path_by_session_id[session["session_evidence_id"]]
                ),
            }
        )
    specimens: dict[str, dict[str, Any]] = {}
    wifi_in_scope = "wifi_udp" in contract["transports"]
    for board in BOARDS:
        board_sessions = [session for session in sessions if session["board"] == board]
        serials = {session.get("physical_port_serial") for session in board_sessions}
        wifi_macs = {
            session.get("wifi_mac_reported")
            for session in board_sessions
            if session["transport"] == "wifi_udp"
        }
        if len(serials) != 1 or not next(iter(serials)):
            raise RuntimeError(f"Campaign uses ambiguous physical specimens for {board}")
        if wifi_in_scope and (len(wifi_macs) != 1 or not next(iter(wifi_macs))):
            raise RuntimeError(f"Campaign uses ambiguous Wi-Fi radios for {board}")
        wifi_mac = next(iter(wifi_macs)) if wifi_macs else None
        specimens[board] = {
            "physical_port_serial": next(iter(serials)),
            "wifi_mac_reported": wifi_mac,
            "session_count": len(board_sessions),
        }
    previous_finished: datetime | None = None
    for combination in contract["eligible_combinations"]:
        session = by_id[combination["combination_id"]]
        started = _parse_timestamp(session["started_utc"])
        finished = _parse_timestamp(session["finished_utc"])
        if finished < started or (
            previous_finished is not None and started < previous_finished
        ):
            raise RuntimeError("Campaign execution order overlaps or violates its contract")
        previous_finished = finished
    payload: dict[str, Any] = {
        "schema": CAMPAIGN_EVIDENCE_SCHEMA,
        "status": (
            "passed_with_blocked_routes"
            if contract["blocked_routes"]
            else "passed"
        ),
        "contract_id": contract["contract_id"],
        "sessions": ordered,
        "totals": totals,
        "all_four_models_retained": set(contract["models"]) == set(MODEL_KEYS),
        "all_gate_eligible_combinations_executed": True,
        "blocked_routes": contract["blocked_routes"],
        "excluded_combinations": contract["excluded_combinations"],
        "boards": list(BOARDS),
        "transports": list(contract["transports"]),
        "physical_specimens": specimens,
        "board_build_contracts": board_build_contracts,
        "claim_boundary": (
            "One seed-42 model specimen per fixed-point-gate-eligible route on one "
            "physical specimen of each board type. Gate-failed routes remain explicit "
            "non-deployment results. Exact replay and timing evidence do not establish "
            "multi-seed or multi-unit hardware variability, energy, or secure attestation."
        ),
    }
    payload = _sealed_payload(payload, "campaign_evidence_id")
    if output_json is not None:
        atomic_write_json(output_json.resolve(), payload)
    return payload
