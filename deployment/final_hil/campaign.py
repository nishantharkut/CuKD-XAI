"""Fail-closed execution controller for the sealed final HIL campaign."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import secrets
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    FINAL_STAGES,
    atomic_write_json,
    canonical_json_sha256,
    read_json,
    validate_campaign_contract,
)
from .evidence import (
    _memoized_export_verifier,
    complete_six_stage_session,
    preflight_campaign,
    record_build_upload_provenance,
    validate_session_completion,
    verify_complete_campaign,
)
from .runtime import (
    collect_host_environment,
    configure_final_wifi,
    execute_usb_stage,
    execute_wifi_stage,
    verify_stage_attempt,
)


def _emit(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def _safe_root(path: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(path))
    if any(item.is_symlink() for item in (absolute, *absolute.parents)):
        raise RuntimeError(f"{label} path cannot contain symlinks")
    if absolute.exists() and not absolute.is_dir():
        raise RuntimeError(f"{label} must be a directory")
    return absolute


def _safe_descendant(root: Path, path: Path, label: str) -> Path:
    selected_root = _safe_root(root, f"{label} root")
    selected = Path(os.path.abspath(path))
    try:
        relative = selected.relative_to(selected_root)
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes its allowed root") from exc
    current = selected_root
    if current.is_symlink():
        raise RuntimeError(f"{label} root cannot be a symlink")
    for part in relative.parts:
        current = current / part
        if current.exists() or current.is_symlink():
            if current.is_symlink():
                raise RuntimeError(f"{label} path cannot contain symlinks")
    return selected


@contextlib.contextmanager
def _exclusive_campaign_lock(evidence_root: Path):
    if os.name != "posix":
        raise RuntimeError("Final hardware campaign locking requires a POSIX host")
    import fcntl

    lock_path = _safe_descendant(
        evidence_root, evidence_root / ".campaign.lock", "campaign lock"
    )
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another final campaign controller is already running") from exc
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _source_freeze(repo_root: Path) -> dict[str, Any]:
    environment = collect_host_environment(repo_root)
    revision = environment.get("git_revision")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision.lower())
    ):
        raise RuntimeError("Final campaign requires an exact Git base revision")
    dependencies = environment.get("source_dependencies")
    if not isinstance(dependencies, list):
        raise RuntimeError("Final campaign host-source ledger is missing")
    return {
        "git_revision": revision,
        "machine": environment.get("machine"),
        "package_versions": environment.get("package_versions"),
        "platform": environment.get("platform"),
        "python": environment.get("python"),
        "python_executable": environment.get("python_executable"),
        "source_dependencies": dependencies,
        "source_dependencies_sha256": canonical_json_sha256(dependencies),
    }


def _require_source_freeze(repo_root: Path, expected: Mapping[str, Any]) -> None:
    if _source_freeze(repo_root) != dict(expected):
        raise RuntimeError("Final campaign host sources changed during execution")


def _load_wifi_credentials(path: Path, *, allowed_root: Path) -> tuple[str, str]:
    source = _safe_descendant(allowed_root, path, "Wi-Fi credential")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    details = None
    consumed = False
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode):
            raise RuntimeError("Wi-Fi credential source is not a regular file")
        if os.name == "posix":
            if details.st_uid != os.getuid():
                raise RuntimeError("Wi-Fi credential file is owned by another user")
            if stat.S_IMODE(details.st_mode) & 0o077:
                raise RuntimeError(
                    "Wi-Fi credential file must not be group/world accessible"
                )
        if details.st_size <= 0 or details.st_size > 4096:
            raise RuntimeError("Wi-Fi credential file size is invalid")
        raw = b""
        while len(raw) <= 4096:
            chunk = os.read(descriptor, 4097 - len(raw))
            if not chunk:
                break
            raw += chunk
        if len(raw) > 4096:
            raise RuntimeError("Wi-Fi credential file size is invalid")
        current = os.stat(source, follow_symlinks=False)
        if current.st_dev != details.st_dev or current.st_ino != details.st_ino:
            raise RuntimeError("Wi-Fi credential path changed while it was read")
        if os.name == "posix":
            source.unlink()
            if os.fstat(descriptor).st_nlink != 0:
                raise RuntimeError("Opened Wi-Fi credential inode remained linked")
            consumed = True
    finally:
        os.close(descriptor)
    if not consumed:
        assert details is not None
        current = os.stat(source, follow_symlinks=False)
        if current.st_dev != details.st_dev or current.st_ino != details.st_ino:
            raise RuntimeError("Wi-Fi credential path changed before deletion")
        source.unlink()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Wi-Fi credential file is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Wi-Fi credential file must contain one JSON object")
    if set(payload) != {"ssid", "password"}:
        raise RuntimeError("Wi-Fi credential file fields are invalid")
    ssid = payload["ssid"]
    password = payload["password"]
    if not isinstance(ssid, str) or not ssid or not isinstance(password, str) or not password:
        raise RuntimeError("Wi-Fi credentials must be nonempty strings")
    return ssid, password


def _validate_preflight(
    payload: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    result = dict(payload)
    recorded = result.pop("preflight_id", None)
    if recorded != canonical_json_sha256(result):
        raise RuntimeError("Execution preflight ID is invalid")
    expected_ids = [
        item["combination_id"] for item in contract["eligible_combinations"]
    ]
    if (
        payload.get("schema") != "cukd_final_hil_campaign_preflight_v1"
        or payload.get("status") != contract["status"]
        or payload.get("contract_id") != contract["contract_id"]
        or payload.get("blockers") != []
        or payload.get("eligible_combinations") != expected_ids
        or payload.get("eligible_combination_count") != len(expected_ids)
        or not isinstance(payload.get("board_build_contracts"), Mapping)
    ):
        raise RuntimeError("Execution preflight is not an exact ready campaign gate")
    return dict(payload)


def _existing_session_prefix(
    evidence_root: Path, sessions_dir: Path, expected_ids: Sequence[str]
) -> list[Path]:
    sessions_dir = _safe_descendant(evidence_root, sessions_dir, "session directory")
    sessions_dir.mkdir(parents=True, exist_ok=True)
    files = [
        _safe_descendant(evidence_root, path, "session evidence")
        for path in sorted(sessions_dir.glob("*.json"))
    ]
    observed = {path.stem: path for path in files}
    extras = set(observed) - set(expected_ids)
    if extras:
        raise RuntimeError(f"Unexpected completed sessions exist: {sorted(extras)}")
    prefix: list[Path] = []
    missing_seen = False
    for combination_id in expected_ids:
        path = observed.get(combination_id)
        if path is None:
            missing_seen = True
        elif missing_seen:
            raise RuntimeError("Completed sessions are not a campaign-order prefix")
        else:
            prefix.append(path)
    return prefix


def _artifact_paths(
    campaign_root: Path, contract: Mapping[str, Any]
) -> tuple[dict[str, Path], dict[str, Path], Path]:
    model_keys = {
        item["model_key"] for item in contract["eligible_combinations"]
    }
    exports = {key: campaign_root / "exports" / key for key in model_keys}
    bundles = {
        item["combination_id"]: campaign_root / "bundles" / item["combination_id"]
        for item in contract["eligible_combinations"]
    }
    return exports, bundles, campaign_root / "cohort"


def _verify_or_create_preflight(
    *,
    contract_path: Path,
    contract: Mapping[str, Any],
    cohort_dir: Path,
    export_dirs: Mapping[str, Path],
    bundle_dirs: Mapping[str, Path],
    output_path: Path,
    verifier: Any,
) -> dict[str, Any]:
    result = preflight_campaign(
        campaign_contract=contract_path,
        cohort_dir=cohort_dir,
        export_dirs=export_dirs,
        bundle_dirs=bundle_dirs,
        output_json=None,
        verifier=verifier,
    )
    validated = _validate_preflight(result, contract)
    if output_path.exists():
        if read_json(output_path) != validated:
            raise RuntimeError("Existing execution preflight differs from recomputation")
    else:
        atomic_write_json(output_path, validated)
    return validated


def _require_environment_freeze(
    environment: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    observed = {
        "git_revision": environment.get("git_revision"),
        "machine": environment.get("machine"),
        "package_versions": environment.get("package_versions"),
        "platform": environment.get("platform"),
        "python": environment.get("python"),
        "python_executable": environment.get("python_executable"),
        "source_dependencies": environment.get("source_dependencies"),
        "source_dependencies_sha256": canonical_json_sha256(
            environment.get("source_dependencies", [])
        ),
    }
    if observed != dict(expected):
        raise RuntimeError("Resumed evidence uses another host-source environment")


def _require_session_freeze(
    session: Mapping[str, Any], expected: Mapping[str, Any], evidence_root: Path
) -> None:
    provenance_path = _safe_descendant(
        evidence_root,
        Path(str(session["provenance_path_recorded"])),
        "session provenance",
    )
    provenance = read_json(provenance_path)
    _require_environment_freeze(provenance.get("host_environment", {}), expected)
    for stage in session["stages"]:
        attempt_dir = _safe_descendant(
            evidence_root,
            Path(str(stage["attempt_path_recorded"])),
            "session attempt",
        )
        attempt = read_json(attempt_dir / "final_attempt.json")
        _require_environment_freeze(attempt.get("host_environment", {}), expected)
    connection_path = session.get("connection_path_recorded")
    if connection_path is not None:
        connection = read_json(
            _safe_descendant(
                evidence_root, Path(str(connection_path)), "session connection"
            )
        )
        _require_environment_freeze(connection.get("host_environment", {}), expected)


def _validate_resumed_sessions(
    *,
    session_paths: Sequence[Path],
    expected_ids: Sequence[str],
    source_freeze: Mapping[str, Any],
    evidence_root: Path,
    repo_root: Path,
    ports: Mapping[str, tuple[str, str]],
    verifier: Any,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    sessions: list[dict[str, Any]] = []
    wifi_macs: dict[str, str] = {}
    for expected_id, path in zip(expected_ids, session_paths):
        session = validate_session_completion(
            path, verifier=verifier, host_source_root=repo_root
        )
        if path.stem != expected_id or session.get("combination_id") != expected_id:
            raise RuntimeError("Resumed session filename and sealed identity differ")
        board = str(session["board"])
        if session.get("physical_port_serial") != ports[board][1]:
            raise RuntimeError("Resumed session uses another physical board specimen")
        mac = session.get("wifi_mac_reported")
        if mac is not None:
            prior = wifi_macs.setdefault(board, str(mac))
            if prior != mac:
                raise RuntimeError("Resumed sessions use different Wi-Fi radios")
        _require_session_freeze(session, source_freeze, evidence_root)
        sessions.append(session)
    return sessions, wifi_macs


def _run_combination(
    *,
    combination: Mapping[str, Any],
    export_dir: Path,
    cohort_dir: Path,
    bundle_dir: Path,
    evidence_root: Path,
    repo_root: Path,
    source_freeze: Mapping[str, Any],
    port: str,
    physical_port_serial: str,
    wifi_credentials: tuple[str, str] | None,
    command_timeout_seconds: float,
    wifi_timeout_seconds: float,
    wifi_max_attempts: int,
    expected_wifi_mac: str | None,
    verifier: Any,
) -> Path:
    combination_id = str(combination["combination_id"])
    transport = str(combination["transport"])
    run_id = secrets.token_hex(8).upper()
    run_dir = _safe_descendant(
        evidence_root,
        evidence_root / "runs" / combination_id / run_id,
        "combination run",
    )
    run_dir.mkdir(parents=True)
    campaign_session_id = secrets.token_hex(16).upper()
    atomic_write_json(
        run_dir / "controller_contract.json",
        {
            "schema": "cukd_final_hil_controller_run_v1",
            "combination_id": combination_id,
            "campaign_session_id": campaign_session_id,
            "source_freeze": dict(source_freeze),
            "credentials_recorded": False,
            "transport_retry_policy": (
                "no request retry"
                if transport == "usb_serial"
                else (
                    "idempotent same-transaction retransmission with at most "
                    f"{wifi_max_attempts} attempts"
                )
            ),
            "failed_attempt_admission": "forbidden",
        },
    )
    _emit("combination_started", combination_id=combination_id, run_id=run_id)
    _require_source_freeze(repo_root, source_freeze)
    provenance = record_build_upload_provenance(
        export_dir=export_dir,
        bundle_dir=bundle_dir,
        physical_port=port,
        physical_port_serial=physical_port_serial,
        output_dir=run_dir / "provenance",
        command_timeout_seconds=command_timeout_seconds,
        verifier=verifier,
    )
    _require_source_freeze(repo_root, source_freeze)
    connection: Path | None = None
    if transport == "wifi_udp":
        if wifi_credentials is None:
            raise RuntimeError("Wi-Fi credentials are required for a pending Wi-Fi route")
        connection = configure_final_wifi(
            export_dir=export_dir,
            bundle_dir=bundle_dir,
            port=port,
            physical_port_serial=physical_port_serial,
            ssid=wifi_credentials[0],
            password=wifi_credentials[1],
            output_json=run_dir / "connection.json",
            verifier=verifier,
        )
        connection_payload = read_json(connection)
        if expected_wifi_mac is not None and connection_payload.get(
            "wifi_mac_reported"
        ) != expected_wifi_mac:
            raise RuntimeError("Wi-Fi provisioning reached another physical radio")
        _require_source_freeze(repo_root, source_freeze)

    attempt_dirs: list[Path] = []
    for stage in FINAL_STAGES:
        stage_name = str(stage["name"])
        output_root = run_dir / "attempts" / stage_name
        if transport == "usb_serial":
            final_attempt = execute_usb_stage(
                export_dir=export_dir,
                cohort_dir=cohort_dir,
                bundle_dir=bundle_dir,
                stage_name=stage_name,
                port=port,
                physical_port_serial=physical_port_serial,
                output_root=output_root,
                campaign_session_id=campaign_session_id,
                verifier=verifier,
            )
        else:
            assert connection is not None
            final_attempt = execute_wifi_stage(
                export_dir=export_dir,
                cohort_dir=cohort_dir,
                bundle_dir=bundle_dir,
                connection_json=connection,
                stage_name=stage_name,
                output_root=output_root,
                campaign_session_id=campaign_session_id,
                timeout_seconds=wifi_timeout_seconds,
                max_attempts=wifi_max_attempts,
                verifier=verifier,
            )
        attempt_dir = final_attempt.parent
        attempt = verify_stage_attempt(
            attempt_dir,
            export_dir=export_dir,
            cohort_dir=cohort_dir,
            bundle_dir=bundle_dir,
            verifier=verifier,
            host_source_root=repo_root,
        )
        if attempt.get("stage", {}).get("name") != stage_name:
            raise RuntimeError("Verified stage identity differs from execution order")
        attempt_dirs.append(attempt_dir)
        _require_source_freeze(repo_root, source_freeze)
        _emit(
            "stage_passed",
            combination_id=combination_id,
            stage=stage_name,
            completed_rows=attempt["completed_rows"],
            attempt_id=attempt["attempt_id"],
        )

    session_path = evidence_root / "sessions" / f"{combination_id}.json"
    complete_six_stage_session(
        attempt_dirs=attempt_dirs,
        export_dir=export_dir,
        cohort_dir=cohort_dir,
        bundle_dir=bundle_dir,
        provenance_json=provenance,
        output_json=session_path,
        verifier=verifier,
    )
    session_payload = read_json(session_path)
    attempt_overrides = {
        str(stage["attempt_id"]): path
        for stage, path in zip(session_payload["stages"], attempt_dirs)
    }
    session = validate_session_completion(
        session_path,
        export_dir=export_dir,
        cohort_dir=cohort_dir,
        bundle_dir=bundle_dir,
        provenance_json=provenance,
        connection_json=connection,
        attempt_dirs=attempt_overrides,
        verifier=verifier,
        host_source_root=repo_root,
    )
    if session.get("combination_id") != combination_id:
        raise RuntimeError("Completed session identifies another combination")
    _require_source_freeze(repo_root, source_freeze)
    _emit(
        "combination_passed",
        combination_id=combination_id,
        session_evidence_id=session["session_evidence_id"],
    )
    return session_path


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    campaign_root = _safe_root(args.campaign_root, "campaign root")
    evidence_root = _safe_root(args.evidence_root, "evidence root")
    evidence_root.mkdir(parents=True, exist_ok=True)
    with _exclusive_campaign_lock(evidence_root):
        return _run_campaign_locked(
            args=args,
            repo_root=repo_root,
            campaign_root=campaign_root,
            evidence_root=evidence_root,
        )


def _run_campaign_locked(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    campaign_root: Path,
    evidence_root: Path,
) -> dict[str, Any]:
    requested_contract = getattr(args, "contract", None)
    contract_path = _safe_descendant(
        campaign_root,
        (
            (
                Path(requested_contract)
                if Path(requested_contract).is_absolute()
                else campaign_root / Path(requested_contract)
            )
            if requested_contract is not None
            else campaign_root / "campaign" / "campaign_contract.json"
        ),
        "campaign contract",
    )
    contract = validate_campaign_contract(contract_path)
    if contract["status"] not in {"ready", "ready_with_blocked_routes"}:
        raise RuntimeError("Final campaign contract is not executable")
    expected_ids = [
        item["combination_id"] for item in contract["eligible_combinations"]
    ]
    export_dirs, bundle_dirs, cohort_dir = _artifact_paths(campaign_root, contract)
    source_freeze = _source_freeze(repo_root)
    verifier = _memoized_export_verifier(None)
    preflight = _verify_or_create_preflight(
        contract_path=contract_path,
        contract=contract,
        cohort_dir=cohort_dir,
        export_dirs=export_dirs,
        bundle_dirs=bundle_dirs,
        output_path=evidence_root / "execution_preflight.json",
        verifier=verifier,
    )
    _require_source_freeze(repo_root, source_freeze)
    _emit(
        "preflight_passed",
        preflight_id=preflight["preflight_id"],
        eligible_combinations=len(expected_ids),
    )

    ports = {
        "esp32c3": (args.esp32c3_port, args.esp32c3_serial),
        "arduino_r4": (args.arduino_r4_port, args.arduino_r4_serial),
    }
    sessions_dir = evidence_root / "sessions"
    session_paths = _existing_session_prefix(evidence_root, sessions_dir, expected_ids)
    resumed_sessions, wifi_macs = _validate_resumed_sessions(
        session_paths=session_paths,
        expected_ids=expected_ids[: len(session_paths)],
        source_freeze=source_freeze,
        evidence_root=evidence_root,
        repo_root=repo_root,
        ports=ports,
        verifier=verifier,
    )
    completed = {path.stem for path in session_paths}
    pending = [
        item
        for item in contract["eligible_combinations"]
        if item["combination_id"] not in completed
    ]
    wifi_credentials: tuple[str, str] | None = None
    if any(item["transport"] == "wifi_udp" for item in pending):
        if args.wifi_credentials_file is None:
            raise RuntimeError("A protected Wi-Fi credential file is required")
        wifi_credentials = _load_wifi_credentials(
            args.wifi_credentials_file, allowed_root=evidence_root
        )
    for combination in pending:
        combination_id = str(combination["combination_id"])
        model_key = str(combination["model_key"])
        port, physical_serial = ports[str(combination["board"])]
        session_path = _run_combination(
                combination=combination,
                export_dir=export_dirs[model_key],
                cohort_dir=cohort_dir,
                bundle_dir=bundle_dirs[combination_id],
                evidence_root=evidence_root,
                repo_root=repo_root,
                source_freeze=source_freeze,
                port=port,
                physical_port_serial=physical_serial,
                wifi_credentials=wifi_credentials,
                command_timeout_seconds=args.command_timeout,
                wifi_timeout_seconds=args.wifi_timeout,
                wifi_max_attempts=args.wifi_max_attempts,
                expected_wifi_mac=wifi_macs.get(str(combination["board"])),
                verifier=verifier,
            )
        session_paths.append(session_path)
        completed_session = validate_session_completion(
            session_path, verifier=verifier, host_source_root=repo_root
        )
        board = str(completed_session["board"])
        mac = completed_session.get("wifi_mac_reported")
        if mac is not None:
            wifi_macs.setdefault(board, str(mac))

    _require_source_freeze(repo_root, source_freeze)
    completion_path = evidence_root / "campaign_evidence.json"
    if completion_path.exists():
        computed = verify_complete_campaign(
            campaign_contract=contract_path,
            session_jsons=session_paths,
            verifier=verifier,
            host_source_root=repo_root,
        )
        if read_json(completion_path) != computed:
            raise RuntimeError("Existing campaign completion differs from recomputation")
        result = computed
    else:
        result = verify_complete_campaign(
            campaign_contract=contract_path,
            session_jsons=session_paths,
            output_json=completion_path,
            verifier=verifier,
            host_source_root=repo_root,
        )
    _require_source_freeze(repo_root, source_freeze)
    _emit(
        "campaign_passed",
        status=result["status"],
        campaign_evidence_id=result["campaign_evidence_id"],
        sessions=result["totals"]["session_count"],
        all_device_inferences=result["totals"]["all_device_inferences"],
    )
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--campaign-root", type=Path, required=True)
    result.add_argument(
        "--contract",
        type=Path,
        help="contract within the campaign root; defaults to campaign/campaign_contract.json",
    )
    result.add_argument("--evidence-root", type=Path, required=True)
    result.add_argument("--esp32c3-port", required=True)
    result.add_argument("--esp32c3-serial", required=True)
    result.add_argument("--arduino-r4-port", required=True)
    result.add_argument("--arduino-r4-serial", required=True)
    result.add_argument("--wifi-credentials-file", type=Path)
    result.add_argument("--command-timeout", type=float, default=1800.0)
    result.add_argument("--wifi-timeout", type=float, default=1.0)
    result.add_argument("--wifi-max-attempts", type=int, default=3)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command_timeout <= 0 or args.wifi_timeout <= 0:
        raise ValueError("Campaign timeouts must be positive")
    if not 1 <= args.wifi_max_attempts <= 3:
        raise ValueError("Wi-Fi max attempts must be between one and three")
    run_campaign(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
