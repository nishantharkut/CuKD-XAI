#!/usr/bin/env python3
"""Continue the interrupted full-route run with dtype-faithful verification.

The original runner's training path is retained exactly. This continuation
driver replaces only the persisted-prediction verifier in memory so completed
neural routes are checked in their executed float32 representation, while the
uncalibrated RF route retains its executed float32-to-float64 round trip. It
does not aggregate results; final aggregation remains a separate, explicit
step after all publication seeds are complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import finalize_fgds_full_routes as finalizer
from . import run_fgds_full_routes as runner


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_EVIDENCE_ROOT = (
    REPO_ROOT
    / "results/wsnds/evidence_completion_20260811"
    / "fgds_controlled_full_routes_10seed_v2_continuation"
)
DEFAULT_ARCHIVE_MANIFEST = (
    REPO_ROOT
    / "results/wsnds/evidence_completion_20260811"
    / "interrupted_attempts_20260812_0053_ist"
    / "artifact_manifest.json"
)
PROTOCOL_ID = "wsnds_fgds_full_routes_dtype_faithful_continuation_v1"
ARCHIVE_PROTOCOL_ID = "wsnds_interrupted_attempt_archive_v1"
POST_EVIDENCE_RECOVERY_PROTOCOL_ID = "wsnds_fgds_post_evidence_lock_recovery_v1"
LOCK_NAME = ".fgds_full_routes_dtype_faithful_continuation.lock"
STALE_LOCK_DIR_NAME = "failed_continuation_locks"
POST_EVIDENCE_LOCK_DIR_NAME = "post_evidence_stale_locks"
LOCK_CANDIDATE_MARKER = ".candidate-"
ATTEMPT_CONTRACT_NAME = "continuation_attempt_contract.json"
INTERRUPTION_CONTEXT_NAME = "continuation_interruption_context.json"
INTERRUPTION_MANIFEST_NAME = "continuation_interruption_manifest.json"
SOURCE_SNAPSHOTS = {
    "executed_continuation_source.py": SCRIPT_PATH,
    "bound_original_runner_source.py": finalizer.RUNNER_PATH,
    "bound_finalizer_source.py": finalizer.SCRIPT_PATH,
    "bound_common_source.py": finalizer.COMMON_PATH,
}


class ContinuationError(RuntimeError):
    """Raised when the interrupted run cannot be continued safely."""


def verification_override_contract() -> dict[str, Any]:
    return {
        "scope": "persisted prediction metric verification only",
        "training_changed": False,
        "model_selection_changed": False,
        "neural_probability_representation": "stored float32",
        "uncalibrated_rf_probability_representation": (
            "stored float32 promoted to float64, matching the executed route"
        ),
        "implementation": "corrected_metrics_from_npz_predictions",
        "original_runner_fallback_archive_disabled": True,
        "fallback_boundary_scope": "the continuation process",
        "fallback_boundary": (
            "A seed directory that appears after the continuation precheck causes an "
            "immediate failure before the original runner can move it or train."
        ),
        "cooperative_writer_boundary": (
            "The continuation and finalizer share the lifecycle lock. The unchanged "
            "original runner does not inspect that lock and must not be launched "
            "concurrently. Any detected artifact interference is rejected."
        ),
    }


def fingerprint_payload(payload: dict[str, Any], fingerprint_key: str) -> str:
    encoded = json.dumps(
        {key: value for key, value in payload.items() if key != fingerprint_key},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-csv", type=Path, default=runner.DEFAULT_DATASET)
    parser.add_argument("--base-root", type=Path, default=runner.DEFAULT_BASE)
    parser.add_argument("--output-root", type=Path, default=runner.DEFAULT_OUTPUT)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument(
        "--interrupted-archive-manifest",
        type=Path,
        default=DEFAULT_ARCHIVE_MANIFEST,
    )
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--confirm-continuation", action="store_true")
    mode.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--recover-stale-lock", action="store_true")
    return parser.parse_args()


def atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with source.open("rb") as source_handle, temporary.open("wb") as output_handle:
        shutil.copyfileobj(source_handle, output_handle)
        output_handle.flush()
        os.fsync(output_handle.fileno())
    os.replace(temporary, destination)
    fsync_existing_file(destination)
    fsync_directory(destination.parent)


def fsync_existing_file(path: Path) -> None:
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_existing_file(path)
    fsync_directory(path.parent)


def incomplete_seed_ids(output_root: Path) -> list[int]:
    incomplete: list[int] = []
    for seed in runner.PUBLICATION_SEEDS:
        seed_root = output_root / f"seed_{seed}"
        if seed_root.exists() and not (
            (seed_root / "seed_completion.json").is_file()
            and (seed_root / "artifact_manifest.json").is_file()
        ):
            incomplete.append(seed)
    return incomplete


def failed_attempt_directory_names(output_root: Path) -> set[str]:
    failed_root = output_root / "failed_seed_attempts"
    if not failed_root.exists():
        return set()
    if not failed_root.is_dir():
        raise ContinuationError("Failed-seed attempt path is not a directory")
    entries = list(failed_root.iterdir())
    if any(not path.is_dir() for path in entries):
        raise ContinuationError("Failed-seed attempt root contains unexpected files")
    return {path.name for path in entries}


class _RunnerOsGuard:
    def __init__(self, original_os: Any, protected_seed_root: Path, failed_root: Path):
        self._original_os = original_os
        self._protected_seed_root = self._lexical_path(protected_seed_root)
        self._failed_root = self._lexical_path(failed_root)

    @staticmethod
    def _lexical_path(path: Any) -> str:
        return os.path.normcase(os.path.abspath(os.fspath(path)))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original_os, name)

    def replace(self, source: Any, destination: Any) -> None:
        source_path = self._lexical_path(source)
        destination_path = self._lexical_path(destination)
        if (
            source_path == self._protected_seed_root
            and os.path.dirname(destination_path) == self._failed_root
        ):
            raise ContinuationError(
                "A target seed appeared concurrently; refusing the original runner "
                "fallback archive before training"
            )
        self._original_os.replace(source, destination)


def run_seed_without_fallback_archive(
    output_root: Path,
    base_root: Path,
    context: dict[str, Any],
    seed: int,
    device: Any,
    execution_contract_sha256: str,
) -> dict[str, Any]:
    original_runner_os = runner.os
    runner.os = _RunnerOsGuard(
        original_runner_os,
        output_root / f"seed_{seed}",
        output_root / "failed_seed_attempts",
    )
    try:
        return runner.run_seed(
            output_root,
            base_root,
            context,
            seed,
            device,
            execution_contract_sha256,
            True,
        )
    finally:
        runner.os = original_runner_os


def require_within_repo(path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ContinuationError(f"{label} is outside the repository: {resolved}") from exc
    return resolved


def paths_overlap(left: Path, right: Path) -> bool:
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def output_finalization_state(
    output_root: Path,
    verify_existing: bool,
    allow_partial_for_lock_recovery: bool = False,
) -> bool:
    aggregate_exists = (output_root / "aggregate_results.json").is_file()
    root_manifest_exists = (output_root / "artifact_manifest.json").is_file()
    if aggregate_exists != root_manifest_exists:
        if allow_partial_for_lock_recovery and aggregate_exists:
            return False
        raise ContinuationError(
            "Output root has an inconsistent partial-finalization state"
        )
    finalized = aggregate_exists and root_manifest_exists
    if finalized and not verify_existing:
        raise ContinuationError("Output root is already aggregated and finalized")
    return finalized


def validate_protected_paths(
    dataset_csv: Path,
    output_root: Path,
    base_root: Path,
    evidence_root: Path,
    archive_manifest: Path,
) -> None:
    exact_paths = {
        "dataset": (dataset_csv, runner.DEFAULT_DATASET.resolve()),
        "base evidence": (base_root, runner.DEFAULT_BASE.resolve()),
        "execution output": (output_root, runner.DEFAULT_OUTPUT.resolve()),
        "interrupted archive manifest": (
            archive_manifest,
            DEFAULT_ARCHIVE_MANIFEST.resolve(),
        ),
    }
    for label, (observed, expected) in exact_paths.items():
        if observed != expected:
            raise ContinuationError(f"{label} must be exactly {expected}")
    expected_evidence = DEFAULT_EVIDENCE_ROOT.resolve()
    if evidence_root != expected_evidence:
        raise ContinuationError(
            f"Continuation evidence root must be exactly {expected_evidence}"
        )
    archive_root = archive_manifest.parent
    protected = {
        "execution output": output_root,
        "base evidence": base_root,
        "interrupted archive": archive_root,
    }
    for label, path in protected.items():
        if paths_overlap(evidence_root, path):
            raise ContinuationError(
                f"Continuation evidence overlaps {label}: {evidence_root} / {path}"
            )


def process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        error_access_denied = 5
        error_invalid_parameter = 87
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(
            process_query_limited_information,
            False,
            pid,
        )
        if handle:
            kernel32.CloseHandle(handle)
            return True
        error = ctypes.get_last_error()
        if error == error_invalid_parameter:
            return False
        if error == error_access_denied:
            return True
        raise ContinuationError(
            f"Cannot determine whether lock process {pid} exists; Windows error {error}"
        )
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        raise ContinuationError(
            f"Cannot determine whether lock process {pid} exists: {exc}"
        ) from exc
    return True


def supplemental_environment_record() -> dict[str, Any]:
    torch = runner.torch
    device_index = int(torch.cuda.current_device())
    record = {
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cuda_device_index": device_index,
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_device_name": torch.cuda.get_device_name(device_index),
        "cuda_device_capability": list(torch.cuda.get_device_capability(device_index)),
    }
    required = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "deterministic_algorithms_enabled": True,
        "deterministic_algorithms_warn_only": False,
    }
    for key, expected in required.items():
        if record.get(key) != expected:
            raise ContinuationError(
                f"Continuation determinism control differs for {key}: "
                f"{record.get(key)!r} != {expected!r}"
            )
    return record


def validate_lock_payload(path: Path) -> dict[str, Any]:
    payload = finalizer.read_json(path)
    pid = payload.get("pid")
    started = payload.get("started_at_utc")
    owner_token = payload.get("owner_token")
    if (
        payload.get("protocol_id") != PROTOCOL_ID
        or not isinstance(pid, int)
        or pid <= 0
        or not isinstance(started, str)
        or not started
        or not isinstance(owner_token, str)
        or not re.fullmatch(r"[0-9a-f]{32}", owner_token)
        or payload.get("continuation_source_sha256")
        != finalizer.sha256_file(SCRIPT_PATH)
    ):
        raise ContinuationError(f"Continuation lock payload differs: {path}")
    return payload


def stale_lock_records(output_root: Path) -> list[dict[str, Any]]:
    archive_root = output_root / STALE_LOCK_DIR_NAME
    if not archive_root.exists():
        return []
    if not archive_root.is_dir():
        raise ContinuationError("Stale-lock archive is not a directory")
    records = []
    for path in sorted(archive_root.iterdir()):
        if not path.is_file():
            raise ContinuationError("Stale-lock archive contains a non-file")
        payload = validate_lock_payload(path)
        records.append(
            {
                "path": path.relative_to(output_root).as_posix(),
                "sha256": finalizer.sha256_file(path),
                "pid": payload["pid"],
                "started_at_utc": payload["started_at_utc"],
            }
        )
    return records


def preserve_post_evidence_stale_lock(
    lock_path: Path,
    output_root: Path,
    evidence_root: Path,
) -> Path:
    payload = validate_lock_payload(lock_path)
    if process_exists(int(payload["pid"])):
        raise ContinuationError(
            f"Continuation lock process is still active: {payload['pid']}"
        )
    evidence_contract = evidence_root / "continuation_contract.json"
    evidence_manifest = evidence_root / "artifact_manifest.json"
    if not evidence_contract.is_file() or not evidence_manifest.is_file():
        raise ContinuationError("Continuation evidence is incomplete before lock recovery")
    archive_parent = (
        output_root.parent / POST_EVIDENCE_LOCK_DIR_NAME / output_root.name
    )
    archive_parent.mkdir(parents=True, exist_ok=True)
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    recovery_root = archive_parent / f"recovery_{suffix}"
    if recovery_root.exists():
        raise ContinuationError(
            f"Post-evidence stale-lock destination exists: {recovery_root}"
        )
    recovery_root.mkdir()
    destination = recovery_root / "continuation_lock.json"
    os.replace(lock_path, destination)
    fsync_existing_file(destination)
    fsync_directory(destination.parent)
    root_manifest = output_root / "artifact_manifest.json"
    recovery = {
        "protocol_id": POST_EVIDENCE_RECOVERY_PROTOCOL_ID,
        "status": "complete",
        "recovered_at_utc": datetime.now(timezone.utc).isoformat(),
        "lock_file": destination.name,
        "lock_sha256": finalizer.sha256_file(destination),
        "lock_payload": payload,
        "output_root": str(output_root.relative_to(REPO_ROOT)),
        "output_root_manifest_sha256": (
            finalizer.sha256_file(root_manifest) if root_manifest.is_file() else None
        ),
        "continuation_evidence_root": str(evidence_root.relative_to(REPO_ROOT)),
        "continuation_contract_sha256": finalizer.sha256_file(evidence_contract),
        "continuation_evidence_manifest_sha256": finalizer.sha256_file(
            evidence_manifest
        ),
    }
    durable_atomic_write_json(
        recovery_root / "post_evidence_recovery.json",
        recovery,
    )
    durable_atomic_write_json(
        recovery_root / "artifact_manifest.json",
        runner.artifact_manifest(
            recovery_root,
            POST_EVIDENCE_RECOVERY_PROTOCOL_ID,
            "complete",
        ),
    )
    verify_post_evidence_recovery(recovery_root, output_root, evidence_root)
    return recovery_root


def verify_post_evidence_recovery(
    recovery_root: Path,
    output_root: Path,
    evidence_root: Path,
) -> dict[str, Any]:
    manifest_path = recovery_root / "artifact_manifest.json"
    verify_simple_manifest(
        recovery_root,
        manifest_path,
        POST_EVIDENCE_RECOVERY_PROTOCOL_ID,
    )
    record = finalizer.read_json(recovery_root / "post_evidence_recovery.json")
    lock_name = record.get("lock_file")
    if (
        record.get("protocol_id") != POST_EVIDENCE_RECOVERY_PROTOCOL_ID
        or record.get("status") != "complete"
        or not isinstance(record.get("recovered_at_utc"), str)
        or not record["recovered_at_utc"]
        or not isinstance(lock_name, str)
        or Path(lock_name).name != lock_name
    ):
        raise ContinuationError("Post-evidence lock recovery record is invalid")
    lock_file = recovery_root / lock_name
    lock_payload = validate_lock_payload(lock_file)
    root_manifest = output_root / "artifact_manifest.json"
    recorded_root_manifest = record.get("output_root_manifest_sha256")
    if recorded_root_manifest is not None and (
        not isinstance(recorded_root_manifest, str)
        or not root_manifest.is_file()
        or finalizer.sha256_file(root_manifest) != recorded_root_manifest
    ):
        raise ContinuationError("Recorded finalized-root manifest differs")
    expected = {
        "lock_sha256": finalizer.sha256_file(lock_file),
        "lock_payload": lock_payload,
        "output_root": str(output_root.relative_to(REPO_ROOT)),
        "output_root_manifest_sha256": recorded_root_manifest,
        "continuation_evidence_root": str(evidence_root.relative_to(REPO_ROOT)),
        "continuation_contract_sha256": finalizer.sha256_file(
            evidence_root / "continuation_contract.json"
        ),
        "continuation_evidence_manifest_sha256": finalizer.sha256_file(
            evidence_root / "artifact_manifest.json"
        ),
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ContinuationError(
                f"Post-evidence lock recovery differs for {key}"
            )
    return {
        "path": recovery_root.relative_to(output_root.parent).as_posix(),
        "manifest_sha256": finalizer.sha256_file(manifest_path),
        "lock_sha256": record["lock_sha256"],
        "recovered_at_utc": record["recovered_at_utc"],
    }


def post_evidence_recovery_records(
    output_root: Path,
    evidence_root: Path,
) -> list[dict[str, Any]]:
    archive_parent = (
        output_root.parent / POST_EVIDENCE_LOCK_DIR_NAME / output_root.name
    )
    if not archive_parent.exists():
        return []
    if not archive_parent.is_dir():
        raise ContinuationError("Post-evidence recovery archive is not a directory")
    records = []
    for recovery_root in sorted(archive_parent.iterdir()):
        if not recovery_root.is_dir():
            raise ContinuationError("Post-evidence recovery archive contains a non-directory")
        records.append(
            verify_post_evidence_recovery(
                recovery_root,
                output_root,
                evidence_root,
            )
        )
    return records


class ExclusiveRunLock:
    def __init__(
        self,
        path: Path,
        *,
        recover_stale: bool = False,
        stale_archive_root: Path | None = None,
    ) -> None:
        self.path = path
        self.recover_stale = recover_stale
        self.stale_archive_root = stale_archive_root
        self.owner_payload: dict[str, Any] | None = None
        self.candidate_path: Path | None = None

    def _candidate_path(self, pid: int, owner_token: str) -> Path:
        return self.path.with_name(
            f"{self.path.name}{LOCK_CANDIDATE_MARKER}{pid}-{owner_token}.json"
        )

    def _write_candidate(self, payload: dict[str, Any]) -> Path:
        candidate = self._candidate_path(payload["pid"], payload["owner_token"])
        descriptor = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            with os.fdopen(descriptor, "w", encoding="ascii") as handle:
                descriptor = -1
                json.dump(payload, handle, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        fsync_directory(candidate.parent)
        if validate_lock_payload(candidate) != payload:
            raise ContinuationError("Continuation lock candidate payload differs")
        return candidate

    def __enter__(self) -> "ExclusiveRunLock":
        owner_token = uuid.uuid4().hex
        payload = {
            "protocol_id": PROTOCOL_ID,
            "pid": os.getpid(),
            "owner_token": owner_token,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "continuation_source_sha256": finalizer.sha256_file(SCRIPT_PATH),
        }
        self.candidate_path = self._write_candidate(payload)
        claimed = False
        try:
            while True:
                try:
                    os.link(self.candidate_path, self.path)
                    fsync_directory(self.path.parent)
                    claimed = True
                    break
                except FileExistsError as exc:
                    if not self.recover_stale:
                        raise ContinuationError(
                            "Continuation lock exists; use --recover-stale-lock only after "
                            f"confirming its process is dead: {self.path}"
                        ) from exc
                    if self.stale_archive_root is None:
                        raise ContinuationError("Stale-lock archive root is absent") from exc
                    stale_payload = validate_lock_payload(self.path)
                    if process_exists(int(stale_payload["pid"])):
                        raise ContinuationError(
                            "Continuation lock process is still active: "
                            f"{stale_payload['pid']}"
                        ) from exc
                    self.stale_archive_root.mkdir(parents=True, exist_ok=True)
                    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
                    destination = self.stale_archive_root / f"lock_{suffix}.json"
                    if destination.exists():
                        raise ContinuationError(
                            f"Stale-lock archive destination exists: {destination}"
                        ) from exc
                    try:
                        os.replace(self.path, destination)
                        fsync_directory(destination.parent)
                    except FileNotFoundError:
                        continue
            if validate_lock_payload(self.path) != payload:
                raise ContinuationError("Published continuation lock payload differs")
        except Exception:
            if self.candidate_path is not None:
                try:
                    self.candidate_path.unlink()
                except FileNotFoundError:
                    pass
                self.candidate_path = None
            if claimed:
                # The published lock remains as fail-closed evidence if ownership
                # validation itself failed.
                fsync_directory(self.path.parent)
            raise
        self.owner_payload = payload
        try:
            self.candidate_path.unlink()
            self.candidate_path = None
        except FileNotFoundError:
            self.candidate_path = None
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        cleanup_error: Exception | None = None
        try:
            if self.owner_payload is None:
                raise ContinuationError("Continuation lock ownership was not established")
            observed = validate_lock_payload(self.path)
            if observed != self.owner_payload:
                raise ContinuationError(
                    "Continuation lock ownership changed; refusing to remove it"
                )
            self.path.unlink()
            fsync_directory(self.path.parent)
        except Exception as error:  # Keep an unverifiable lock for explicit recovery.
            cleanup_error = error
        if self.candidate_path is not None:
            try:
                self.candidate_path.unlink()
            except FileNotFoundError:
                pass
            except Exception as error:
                cleanup_error = cleanup_error or error
            self.candidate_path = None
        if cleanup_error is not None:
            if exc is not None and hasattr(exc, "add_note"):
                exc.add_note(f"Continuation lock cleanup failed: {cleanup_error}")
                return
            raise cleanup_error

    def assert_owned(self) -> None:
        if self.owner_payload is None:
            raise ContinuationError("Lifecycle lock ownership was not established")
        if validate_lock_payload(self.path) != self.owner_payload:
            raise ContinuationError("Lifecycle lock ownership changed during execution")


def completed_seed_records(
    output_root: Path,
    seeds: list[int] | tuple[int, ...] | None = None,
) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    selected = list(runner.PUBLICATION_SEEDS if seeds is None else seeds)
    for seed in selected:
        seed_root = output_root / f"seed_{seed}"
        completion = seed_root / "seed_completion.json"
        manifest = seed_root / "artifact_manifest.json"
        if not completion.is_file() or not manifest.is_file():
            raise ContinuationError(f"Seed {seed} is incomplete after continuation")
        records[str(seed)] = {
            "completion_sha256": finalizer.sha256_file(completion),
            "manifest_sha256": finalizer.sha256_file(manifest),
        }
    return records


def build_attempt_contract(
    output_root: Path,
    evidence_root: Path,
    execution_contract_sha256: str,
    archive_manifest: Path,
    archive_manifest_sha256: str,
    completed_before: list[int],
    continuation_environment: dict[str, Any],
) -> dict[str, Any]:
    target_seeds = [
        seed for seed in runner.PUBLICATION_SEEDS if seed not in completed_before
    ]
    payload = {
        "protocol_id": PROTOCOL_ID,
        "status": "started",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_execution_contract_sha256": execution_contract_sha256,
        "original_runner_sha256": finalizer.sha256_file(finalizer.RUNNER_PATH),
        "common_source_sha256": finalizer.sha256_file(finalizer.COMMON_PATH),
        "finalizer_source_sha256": finalizer.sha256_file(finalizer.SCRIPT_PATH),
        "continuation_source_sha256": finalizer.sha256_file(SCRIPT_PATH),
        "continuation_environment": continuation_environment,
        "output_root": str(output_root.relative_to(REPO_ROOT)),
        "continuation_evidence_root": str(evidence_root.relative_to(REPO_ROOT)),
        "completed_seeds_before_continuation": completed_before,
        "seed_records_before_continuation": completed_seed_records(
            output_root,
            completed_before,
        ),
        "target_seeds": target_seeds,
        "interrupted_attempt_archive_manifest": str(
            archive_manifest.relative_to(REPO_ROOT)
        ),
        "interrupted_attempt_archive_manifest_sha256": archive_manifest_sha256,
        "verification_override": verification_override_contract(),
        "aggregation_performed": False,
    }
    payload["attempt_fingerprint_sha256"] = fingerprint_payload(
        payload,
        "attempt_fingerprint_sha256",
    )
    return payload


def validate_attempt_contract(
    path: Path,
    output_root: Path,
    evidence_root: Path,
    execution_contract_sha256: str,
    archive_manifest: Path,
    archive_manifest_sha256: str,
    continuation_environment: dict[str, Any],
    current_completed: list[int],
) -> dict[str, Any]:
    payload = finalizer.read_json(path)
    required = {
        "protocol_id": PROTOCOL_ID,
        "status": "started",
        "original_execution_contract_sha256": execution_contract_sha256,
        "original_runner_sha256": finalizer.sha256_file(finalizer.RUNNER_PATH),
        "common_source_sha256": finalizer.sha256_file(finalizer.COMMON_PATH),
        "finalizer_source_sha256": finalizer.sha256_file(finalizer.SCRIPT_PATH),
        "continuation_source_sha256": finalizer.sha256_file(SCRIPT_PATH),
        "continuation_environment": continuation_environment,
        "output_root": str(output_root.relative_to(REPO_ROOT)),
        "continuation_evidence_root": str(evidence_root.relative_to(REPO_ROOT)),
        "interrupted_attempt_archive_manifest": str(
            archive_manifest.relative_to(REPO_ROOT)
        ),
        "interrupted_attempt_archive_manifest_sha256": archive_manifest_sha256,
        "verification_override": verification_override_contract(),
        "aggregation_performed": False,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ContinuationError(f"Continuation attempt differs for {key}")
    started = payload.get("started_at_utc")
    if not isinstance(started, str) or not started:
        raise ContinuationError("Continuation attempt start timestamp is absent")
    fingerprint = payload.get("attempt_fingerprint_sha256")
    if (
        not isinstance(fingerprint, str)
        or fingerprint != fingerprint_payload(payload, "attempt_fingerprint_sha256")
    ):
        raise ContinuationError("Continuation attempt fingerprint differs")
    completed_before = payload.get("completed_seeds_before_continuation")
    target_seeds = payload.get("target_seeds")
    if not isinstance(completed_before, list) or not isinstance(target_seeds, list):
        raise ContinuationError("Continuation attempt seed sets are invalid")
    publication_seeds = list(runner.PUBLICATION_SEEDS)
    if (
        completed_before != [
            seed for seed in publication_seeds if seed in completed_before
        ]
        or target_seeds != [
            seed for seed in publication_seeds if seed not in completed_before
        ]
        or set(completed_before).intersection(target_seeds)
        or completed_before + target_seeds != publication_seeds
    ):
        raise ContinuationError("Continuation attempt seed partition differs")
    if not set(completed_before).issubset(current_completed):
        raise ContinuationError("A pre-continuation completed seed is now absent")
    if not set(current_completed).issubset(publication_seeds):
        raise ContinuationError("Current completed seed set is outside the publication set")
    expected_records = completed_seed_records(output_root, completed_before)
    if payload.get("seed_records_before_continuation") != expected_records:
        raise ContinuationError("Pre-continuation seed records differ")
    return payload


def ensure_attempt_contract(
    path: Path,
    output_root: Path,
    evidence_root: Path,
    execution_contract_sha256: str,
    archive_manifest: Path,
    archive_manifest_sha256: str,
    completed_before: list[int],
    continuation_environment: dict[str, Any],
) -> dict[str, Any]:
    if not path.exists():
        durable_atomic_write_json(
            path,
            build_attempt_contract(
                output_root,
                evidence_root,
                execution_contract_sha256,
                archive_manifest,
                archive_manifest_sha256,
                completed_before,
                continuation_environment,
            ),
        )
    if not path.is_file():
        raise ContinuationError(f"Continuation attempt contract is not a file: {path}")
    return validate_attempt_contract(
        path,
        output_root,
        evidence_root,
        execution_contract_sha256,
        archive_manifest,
        archive_manifest_sha256,
        continuation_environment,
        completed_before,
    )


def interruption_manifest_payload(root: Path) -> dict[str, Any]:
    manifest_path = root / INTERRUPTION_MANIFEST_NAME
    manifest_temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    files = []
    for artifact in sorted(root.rglob("*")):
        if (
            not artifact.is_file()
            or artifact == manifest_path
            or artifact == manifest_temporary
        ):
            continue
        files.append(
            {
                "path": artifact.relative_to(root).as_posix(),
                "size_bytes": artifact.stat().st_size,
                "sha256": finalizer.sha256_file(artifact),
            }
        )
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "interrupted",
        "file_count_excluding_manifest": len(files),
        "files": files,
    }


def preserve_interrupted_manifest_temporary(root: Path) -> Path | None:
    manifest_path = root / INTERRUPTION_MANIFEST_NAME
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    if not temporary.exists():
        return None
    if not temporary.is_file():
        raise ContinuationError(
            f"Interrupted manifest temporary is not a file: {temporary}"
        )
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    destination = root / f"interrupted_manifest_write_{suffix}.tmp"
    if destination.exists():
        raise ContinuationError(
            f"Interrupted manifest archive destination exists: {destination}"
        )
    os.replace(temporary, destination)
    fsync_existing_file(destination)
    fsync_directory(destination.parent)
    return destination


def verify_interrupted_seed_archive(
    root: Path,
    output_root: Path,
    attempt_contract_sha256: str,
    expected_archive_path: str | None = None,
) -> dict[str, Any]:
    context_path = root / INTERRUPTION_CONTEXT_NAME
    manifest_path = root / INTERRUPTION_MANIFEST_NAME
    manifest_temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    if manifest_temporary.exists():
        raise ContinuationError(
            f"Interrupted continuation has an unpreserved manifest write: {root}"
        )
    context = finalizer.read_json(context_path)
    manifest = finalizer.read_json(manifest_path)
    seed = context.get("seed")
    if (
        context.get("protocol_id") != PROTOCOL_ID
        or context.get("status") != "interrupted"
        or context.get("continuation_attempt_contract_sha256")
        != attempt_contract_sha256
        or seed not in runner.PUBLICATION_SEEDS
        or context.get("source_path") != f"seed_{seed}"
        or context.get("deletion_performed") is not False
        or context.get("archive_path")
        != (
            root.relative_to(output_root).as_posix()
            if expected_archive_path is None
            else expected_archive_path
        )
    ):
        raise ContinuationError(f"Interrupted continuation context differs: {root}")
    archived_at = context.get("archived_at_utc")
    if not isinstance(archived_at, str) or not archived_at:
        raise ContinuationError(f"Interrupted continuation timestamp is absent: {root}")
    expected_manifest = interruption_manifest_payload(root)
    if manifest != expected_manifest:
        raise ContinuationError(f"Interrupted continuation manifest differs: {root}")
    return {
        "seed": int(seed),
        "archive_path": root.relative_to(output_root).as_posix(),
        "context_sha256": finalizer.sha256_file(context_path),
        "manifest_sha256": finalizer.sha256_file(manifest_path),
    }


def interrupted_seed_records(
    output_root: Path,
    attempt_contract_sha256: str,
) -> list[dict[str, Any]]:
    failed_root = output_root / "failed_seed_attempts"
    if not failed_root.exists():
        return []
    if not failed_root.is_dir():
        raise ContinuationError("Failed-seed attempt path is not a directory")
    unexpected_files = [path for path in failed_root.iterdir() if not path.is_dir()]
    if unexpected_files:
        raise ContinuationError("Failed-seed attempt root contains unexpected files")
    return [
        verify_interrupted_seed_archive(
            root,
            output_root,
            attempt_contract_sha256,
        )
        for root in sorted(failed_root.iterdir())
    ]


def archive_incomplete_seed(
    output_root: Path,
    seed: int,
    attempt_contract_sha256: str,
) -> Path:
    seed_root = (output_root / f"seed_{seed}").resolve()
    if seed_root.parent != output_root.resolve() or not seed_root.is_dir():
        raise ContinuationError(f"Incomplete seed path is invalid: {seed_root}")
    if (seed_root / "seed_completion.json").is_file() and (
        seed_root / "artifact_manifest.json"
    ).is_file():
        raise ContinuationError(f"Refusing to archive completed seed {seed}")
    failed_root = (output_root / "failed_seed_attempts").resolve()
    failed_root.mkdir(parents=True, exist_ok=True)
    try:
        failed_root.relative_to(output_root.resolve())
    except ValueError as exc:
        raise ContinuationError("Failed-seed archive escapes the output root") from exc

    context_path = seed_root / INTERRUPTION_CONTEXT_NAME
    manifest_path = seed_root / INTERRUPTION_MANIFEST_NAME
    if manifest_path.exists() and not context_path.is_file():
        raise ContinuationError(
            f"Interrupted seed {seed} has a manifest without context"
        )
    if context_path.exists():
        if not context_path.is_file():
            raise ContinuationError(
                f"Interrupted seed {seed} archival context is not a file"
            )
        context = finalizer.read_json(context_path)
        relative_destination = context.get("archive_path")
        if not isinstance(relative_destination, str):
            raise ContinuationError(f"Interrupted seed {seed} archive path is invalid")
        destination = (output_root / relative_destination).resolve()
        if not manifest_path.exists():
            preserve_interrupted_manifest_temporary(seed_root)
            durable_atomic_write_json(
                manifest_path,
                interruption_manifest_payload(seed_root),
            )
        if not manifest_path.is_file():
            raise ContinuationError(
                f"Interrupted seed {seed} archival manifest is not a file"
            )
        verify_interrupted_seed_archive(
            seed_root,
            output_root,
            attempt_contract_sha256,
            relative_destination,
        )
    else:
        suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        destination = (failed_root / f"seed_{seed}_{suffix}").resolve()
        context = {
            "protocol_id": PROTOCOL_ID,
            "status": "interrupted",
            "seed": seed,
            "archived_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_path": seed_root.relative_to(output_root).as_posix(),
            "archive_path": destination.relative_to(output_root).as_posix(),
            "continuation_attempt_contract_sha256": attempt_contract_sha256,
            "deletion_performed": False,
        }
        durable_atomic_write_json(context_path, context)
        preserve_interrupted_manifest_temporary(seed_root)
        durable_atomic_write_json(
            manifest_path,
            interruption_manifest_payload(seed_root),
        )
        verify_interrupted_seed_archive(
            seed_root,
            output_root,
            attempt_contract_sha256,
            destination.relative_to(output_root).as_posix(),
        )
    if destination.parent != failed_root or destination.exists():
        raise ContinuationError(f"Interrupted seed archive destination is invalid: {destination}")
    os.replace(seed_root, destination)
    fsync_directory(destination.parent)
    verify_interrupted_seed_archive(
        destination,
        output_root,
        attempt_contract_sha256,
    )
    return destination


def verify_archive_manifest(
    path: Path,
    expected_execution_contract_sha256: str,
) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    payload = finalizer.read_json(path)
    archive_root = path.parent
    files = payload.get("files")
    if (
        payload.get("protocol_id") != ARCHIVE_PROTOCOL_ID
        or payload.get("status") != "interrupted"
        or not isinstance(files, list)
        or payload.get("file_count_excluding_manifest") != len(files)
    ):
        raise ContinuationError("Interrupted-attempt archive manifest is invalid")
    declared: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise ContinuationError("Interrupted-attempt manifest item is invalid")
        relative = item.get("path")
        if not isinstance(relative, str) or not relative:
            raise ContinuationError("Interrupted-attempt manifest path is invalid")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ContinuationError("Interrupted-attempt manifest path escapes its root")
        normalized = relative_path.as_posix()
        if normalized in declared:
            raise ContinuationError("Interrupted-attempt manifest path is duplicated")
        declared.add(normalized)
        artifact = (archive_root / relative_path).resolve()
        try:
            artifact.relative_to(archive_root)
        except ValueError as exc:
            raise ContinuationError("Interrupted-attempt artifact escapes its root") from exc
        if (
            not artifact.is_file()
            or artifact.stat().st_size != item.get("size_bytes")
            or finalizer.sha256_file(artifact) != item.get("sha256")
        ):
            raise ContinuationError(
                f"Interrupted-attempt artifact differs from its manifest: {artifact}"
            )
    actual = {
        artifact.relative_to(archive_root).as_posix()
        for artifact in archive_root.rglob("*")
        if artifact.is_file() and artifact != path
    }
    if actual != declared:
        raise ContinuationError("Interrupted-attempt archive inventory differs")
    context_path = archive_root / "interruption_context.json"
    context = finalizer.read_json(context_path)
    expected_context = {
        "protocol_id": ARCHIVE_PROTOCOL_ID,
        "status": "interrupted",
        "full_route_execution_contract_sha256": expected_execution_contract_sha256,
        "full_route_output_root": (
            "results/wsnds/evidence_completion_20260811/"
            "fgds_controlled_full_routes_10seed_v2"
        ),
        "operations": {
            "deletion_performed": False,
            "full_route_partial_source": (
                "fgds_controlled_full_routes_10seed_v2/seed_8192"
            ),
            "full_route_archive": "full_route_seed_8192",
            "sensitivity_partial_source": (
                "fgds_rfkd_hyperparameter_sensitivity_10seed_v1/"
                "seed_3141/student_B/T2_alpha07"
            ),
            "sensitivity_archive": (
                "sensitivity_seed_3141_student_B_T2_alpha07"
            ),
        },
    }
    for key, expected in expected_context.items():
        if context.get(key) != expected:
            raise ContinuationError(
                f"Interrupted-attempt context differs for {key}"
            )
    return payload, finalizer.sha256_file(path)


def support_source_hashes() -> dict[str, str]:
    return {
        "runner": finalizer.sha256_file(finalizer.RUNNER_PATH),
        "common": finalizer.sha256_file(finalizer.COMMON_PATH),
        "finalizer": finalizer.sha256_file(finalizer.SCRIPT_PATH),
        "continuation": finalizer.sha256_file(SCRIPT_PATH),
    }


def assert_runtime_identity(
    *,
    dataset_csv: Path,
    base_root: Path,
    output_root: Path,
    archive_manifest: Path,
    expected_execution_contract: dict[str, Any],
    expected_execution_contract_sha256: str,
    expected_archive_manifest_sha256: str,
    expected_environment: dict[str, Any],
    expected_support_sources: dict[str, str],
    device: Any,
) -> None:
    observed_contract, observed_contract_sha256 = finalizer.validate_execution_identity(
        output_root
    )
    if (
        observed_contract != expected_execution_contract
        or observed_contract_sha256 != expected_execution_contract_sha256
    ):
        raise ContinuationError("Execution identity changed during continuation")
    _, observed_archive_sha256 = verify_archive_manifest(
        archive_manifest,
        expected_execution_contract_sha256,
    )
    if observed_archive_sha256 != expected_archive_manifest_sha256:
        raise ContinuationError("Interrupted-attempt archive changed during continuation")
    current_environment = {
        "execution_contract_environment": runner.environment_record(device),
        "supplemental_determinism": supplemental_environment_record(),
    }
    if current_environment != expected_environment:
        raise ContinuationError("Continuation environment changed during continuation")
    if support_source_hashes() != expected_support_sources:
        raise ContinuationError("A bound source changed during continuation")
    if (
        finalizer.sha256_file(dataset_csv)
        != expected_execution_contract.get("dataset_sha256")
    ):
        raise ContinuationError("Dataset changed during continuation")
    base_manifest = runner.verify_root_manifest(base_root, runner.BASE_PROTOCOL_ID)
    if base_manifest["sha256"] != expected_execution_contract.get(
        "base_root_manifest_sha256"
    ):
        raise ContinuationError("Base evidence changed during continuation")


def write_continuation_evidence(
    evidence_root: Path,
    output_root: Path,
    execution_contract_sha256: str,
    archive_manifest: Path,
    archive_manifest_sha256: str,
    attempt_contract_path: Path,
    attempt_contract: dict[str, Any],
    attempt_contract_sha256: str,
    interruption_records: list[dict[str, Any]],
    lock_recovery_records: list[dict[str, Any]],
    continuation_environment: dict[str, Any],
) -> None:
    if evidence_root.exists():
        raise ContinuationError(f"Refusing to overwrite continuation evidence: {evidence_root}")
    evidence_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{evidence_root.name}.staging-",
            dir=evidence_root.parent,
        )
    )
    expected_sources = {
        "runner": attempt_contract["original_runner_sha256"],
        "common": attempt_contract["common_source_sha256"],
        "finalizer": attempt_contract["finalizer_source_sha256"],
        "continuation": attempt_contract["continuation_source_sha256"],
    }
    if support_source_hashes() != expected_sources:
        raise ContinuationError("A bound source changed before evidence publication")
    for name, source in SOURCE_SNAPSHOTS.items():
        atomic_copy(source, staging / name)
    expected_snapshot_hashes = {
        "executed_continuation_source.py": expected_sources["continuation"],
        "bound_original_runner_source.py": expected_sources["runner"],
        "bound_finalizer_source.py": expected_sources["finalizer"],
        "bound_common_source.py": expected_sources["common"],
    }
    for name, expected_hash in expected_snapshot_hashes.items():
        if finalizer.sha256_file(staging / name) != expected_hash:
            raise ContinuationError(f"Continuation source snapshot differs: {name}")
    atomic_copy(attempt_contract_path, staging / ATTEMPT_CONTRACT_NAME)
    contract = {
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_execution_contract_sha256": execution_contract_sha256,
        "original_runner_sha256": expected_sources["runner"],
        "common_source_sha256": expected_sources["common"],
        "finalizer_source_sha256": expected_sources["finalizer"],
        "continuation_source_sha256": expected_sources["continuation"],
        "source_snapshots": expected_snapshot_hashes,
        "continuation_environment": continuation_environment,
        "output_root": str(output_root.relative_to(REPO_ROOT)),
        "attempt_contract": str(attempt_contract_path.relative_to(REPO_ROOT)),
        "attempt_contract_sha256": attempt_contract_sha256,
        "attempt_contract_snapshot_sha256": finalizer.sha256_file(
            staging / ATTEMPT_CONTRACT_NAME
        ),
        "attempt_fingerprint_sha256": attempt_contract[
            "attempt_fingerprint_sha256"
        ],
        "completed_seeds_before_continuation": attempt_contract[
            "completed_seeds_before_continuation"
        ],
        "seed_records_before_continuation": attempt_contract[
            "seed_records_before_continuation"
        ],
        "target_seeds": attempt_contract["target_seeds"],
        "completed_seeds_after_continuation": list(runner.PUBLICATION_SEEDS),
        "seed_records_after_continuation": completed_seed_records(output_root),
        "interrupted_continuation_attempts": interruption_records,
        "recovered_stale_locks": lock_recovery_records,
        "interrupted_attempt_archive_manifest": str(
            archive_manifest.relative_to(REPO_ROOT)
        ),
        "interrupted_attempt_archive_manifest_sha256": archive_manifest_sha256,
        "verification_override": verification_override_contract(),
        "aggregation_performed": False,
    }
    durable_atomic_write_json(staging / "continuation_contract.json", contract)
    durable_atomic_write_json(
        staging / "artifact_manifest.json",
        runner.artifact_manifest(staging, PROTOCOL_ID, "complete"),
    )
    verify_simple_manifest(staging, staging / "artifact_manifest.json")
    if support_source_hashes() != expected_sources:
        raise ContinuationError("A bound source changed during evidence publication")
    os.replace(staging, evidence_root)
    fsync_directory(evidence_root.parent)


def verify_simple_manifest(
    root: Path,
    manifest_path: Path,
    expected_protocol_id: str = PROTOCOL_ID,
) -> None:
    payload = finalizer.read_json(manifest_path)
    if (
        payload.get("protocol_id") != expected_protocol_id
        or payload.get("status") != "complete"
    ):
        raise ContinuationError("Continuation evidence manifest identity differs")
    files = payload.get("files")
    if not isinstance(files, list) or payload.get(
        "file_count_excluding_manifest"
    ) != len(files):
        raise ContinuationError("Continuation evidence manifest schema differs")
    declared: set[str] = set()
    for item in files:
        relative = item.get("path") if isinstance(item, dict) else None
        if not isinstance(relative, str) or not relative:
            raise ContinuationError("Continuation evidence manifest path is invalid")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ContinuationError("Continuation evidence manifest path escapes its root")
        normalized = relative_path.as_posix()
        artifact = (root / relative_path).resolve()
        try:
            artifact.relative_to(root.resolve())
        except ValueError as exc:
            raise ContinuationError("Continuation evidence path escapes its root") from exc
        if normalized in declared:
            raise ContinuationError("Continuation evidence path is duplicated")
        declared.add(normalized)
        if (
            not artifact.is_file()
            or artifact.stat().st_size != item.get("size_bytes")
            or finalizer.sha256_file(artifact) != item.get("sha256")
        ):
            raise ContinuationError(f"Continuation evidence artifact differs: {artifact}")
    actual = {
        artifact.relative_to(root).as_posix()
        for artifact in root.rglob("*")
        if artifact.is_file() and artifact != manifest_path
    }
    if actual != declared:
        raise ContinuationError("Continuation evidence inventory differs")


def verify_continuation_evidence(
    evidence_root: Path,
    output_root: Path,
    execution_contract_sha256: str,
    archive_manifest: Path,
    archive_manifest_sha256: str,
    attempt_contract_path: Path,
    attempt_contract: dict[str, Any],
    attempt_contract_sha256: str,
    interruption_records: list[dict[str, Any]],
    lock_recovery_records: list[dict[str, Any]],
    continuation_environment: dict[str, Any],
) -> None:
    manifest_path = evidence_root / "artifact_manifest.json"
    contract_path = evidence_root / "continuation_contract.json"
    if not manifest_path.is_file() or not contract_path.is_file():
        raise ContinuationError("Continuation evidence is incomplete")
    verify_simple_manifest(evidence_root, manifest_path)
    contract = finalizer.read_json(contract_path)
    required = {
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "original_execution_contract_sha256": execution_contract_sha256,
        "original_runner_sha256": finalizer.sha256_file(finalizer.RUNNER_PATH),
        "common_source_sha256": finalizer.sha256_file(finalizer.COMMON_PATH),
        "finalizer_source_sha256": finalizer.sha256_file(finalizer.SCRIPT_PATH),
        "continuation_source_sha256": finalizer.sha256_file(SCRIPT_PATH),
        "continuation_environment": continuation_environment,
        "output_root": str(output_root.relative_to(REPO_ROOT)),
        "attempt_contract": str(attempt_contract_path.relative_to(REPO_ROOT)),
        "attempt_contract_sha256": attempt_contract_sha256,
        "attempt_contract_snapshot_sha256": attempt_contract_sha256,
        "attempt_fingerprint_sha256": attempt_contract[
            "attempt_fingerprint_sha256"
        ],
        "completed_seeds_before_continuation": attempt_contract[
            "completed_seeds_before_continuation"
        ],
        "seed_records_before_continuation": attempt_contract[
            "seed_records_before_continuation"
        ],
        "target_seeds": attempt_contract["target_seeds"],
        "completed_seeds_after_continuation": list(runner.PUBLICATION_SEEDS),
        "seed_records_after_continuation": completed_seed_records(output_root),
        "interrupted_continuation_attempts": interruption_records,
        "recovered_stale_locks": lock_recovery_records,
        "interrupted_attempt_archive_manifest": str(
            archive_manifest.relative_to(REPO_ROOT)
        ),
        "interrupted_attempt_archive_manifest_sha256": archive_manifest_sha256,
        "aggregation_performed": False,
    }
    for key, expected in required.items():
        if contract.get(key) != expected:
            raise ContinuationError(f"Continuation evidence differs for {key}")
    generated = contract.get("generated_at_utc")
    if not isinstance(generated, str) or not generated:
        raise ContinuationError("Continuation evidence timestamp is absent")
    expected_snapshots = {
        name: finalizer.sha256_file(source)
        for name, source in SOURCE_SNAPSHOTS.items()
    }
    if contract.get("source_snapshots") != expected_snapshots:
        raise ContinuationError("Continuation source snapshot contract differs")
    for name, expected_hash in expected_snapshots.items():
        snapshot = evidence_root / name
        if not snapshot.is_file() or finalizer.sha256_file(snapshot) != expected_hash:
            raise ContinuationError(f"Continuation source snapshot differs: {snapshot}")
    attempt_snapshot = evidence_root / ATTEMPT_CONTRACT_NAME
    if (
        not attempt_snapshot.is_file()
        or finalizer.sha256_file(attempt_snapshot) != attempt_contract_sha256
    ):
        raise ContinuationError("Continuation attempt-contract snapshot differs")
    if contract.get("verification_override") != verification_override_contract():
        raise ContinuationError("Continuation verification boundary differs")


def verify_completed_continuation(
    output_root: Path,
    execution_contract_sha256: str,
    continuation_environment: dict[str, Any],
) -> dict[str, Any]:
    output_root = output_root.resolve()
    evidence_root = DEFAULT_EVIDENCE_ROOT.resolve()
    archive_manifest = DEFAULT_ARCHIVE_MANIFEST.resolve()
    validate_protected_paths(
        runner.DEFAULT_DATASET.resolve(),
        output_root,
        runner.DEFAULT_BASE.resolve(),
        evidence_root,
        archive_manifest,
    )
    completed = finalizer.complete_seed_ids(output_root)
    if completed != list(runner.PUBLICATION_SEEDS):
        raise ContinuationError("Continuation publication seed set is incomplete")
    _, archive_manifest_sha256 = verify_archive_manifest(
        archive_manifest,
        execution_contract_sha256,
    )
    attempt_contract_path = output_root / ATTEMPT_CONTRACT_NAME
    if not attempt_contract_path.is_file():
        raise ContinuationError("Continuation attempt contract is absent")
    attempt_contract = validate_attempt_contract(
        attempt_contract_path,
        output_root,
        evidence_root,
        execution_contract_sha256,
        archive_manifest,
        archive_manifest_sha256,
        continuation_environment,
        completed,
    )
    attempt_contract_sha256 = finalizer.sha256_file(attempt_contract_path)
    interruption_records = interrupted_seed_records(
        output_root,
        attempt_contract_sha256,
    )
    lock_recovery_records = stale_lock_records(output_root)
    verify_continuation_evidence(
        evidence_root,
        output_root,
        execution_contract_sha256,
        archive_manifest,
        archive_manifest_sha256,
        attempt_contract_path,
        attempt_contract,
        attempt_contract_sha256,
        interruption_records,
        lock_recovery_records,
        continuation_environment,
    )
    post_evidence_recovery_records(output_root, evidence_root)
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "evidence_root": str(evidence_root.relative_to(REPO_ROOT)),
        "continuation_contract_sha256": finalizer.sha256_file(
            evidence_root / "continuation_contract.json"
        ),
        "continuation_manifest_sha256": finalizer.sha256_file(
            evidence_root / "artifact_manifest.json"
        ),
        "attempt_contract_sha256": attempt_contract_sha256,
        "interrupted_archive_manifest_sha256": archive_manifest_sha256,
    }


def main() -> int:
    args = parse_args()
    if args.recover_stale_lock and not (
        args.confirm_continuation or args.verify_existing
    ):
        raise ContinuationError(
            "--recover-stale-lock requires --confirm-continuation or --verify-existing"
        )
    dataset_csv = require_within_repo(args.dataset_csv, "dataset")
    output_root = require_within_repo(args.output_root, "output root")
    base_root = require_within_repo(args.base_root, "base root")
    evidence_root = require_within_repo(args.evidence_root, "evidence root")
    lock_path = evidence_root.parent / LOCK_NAME
    archive_manifest = require_within_repo(
        args.interrupted_archive_manifest, "interrupted archive manifest"
    )
    validate_protected_paths(
        dataset_csv,
        output_root,
        base_root,
        evidence_root,
        archive_manifest,
    )
    if not output_root.is_dir():
        raise ContinuationError(f"Output root is absent: {output_root}")
    partial_finalization = (
        (output_root / "aggregate_results.json").is_file()
        != (output_root / "artifact_manifest.json").is_file()
    )
    output_is_finalized = output_finalization_state(
        output_root,
        args.verify_existing,
        allow_partial_for_lock_recovery=(
            args.verify_existing and args.recover_stale_lock
        ),
    )
    execution_contract, execution_contract_sha256 = (
        finalizer.validate_execution_identity(output_root)
    )
    _, archive_manifest_sha256 = verify_archive_manifest(
        archive_manifest, execution_contract_sha256
    )
    device = runner.resolve_device(args.device)
    runner.set_seed(runner.PUBLICATION_SEEDS[0])
    execution_environment = runner.environment_record(device)
    if execution_environment != execution_contract.get("environment"):
        raise ContinuationError("Continuation environment differs from execution contract")
    continuation_environment = {
        "execution_contract_environment": execution_environment,
        "supplemental_determinism": supplemental_environment_record(),
    }
    expected_support_sources = support_source_hashes()
    context = runner.load_context(dataset_csv, base_root)
    context["verified_base_seeds"] = {
        seed: runner.verify_base_seed(base_root, seed, context)
        for seed in runner.PUBLICATION_SEEDS
    }
    current_completed = finalizer.complete_seed_ids(output_root)
    finalizer.verify_seed_set(
        output_root, current_completed, execution_contract_sha256, context
    )
    attempt_contract_path = output_root / ATTEMPT_CONTRACT_NAME
    attempt_contract: dict[str, Any] | None = None
    attempt_contract_sha256: str | None = None
    interruption_records: list[dict[str, Any]] = []
    lock_recovery_records = stale_lock_records(output_root)
    if attempt_contract_path.exists():
        if not attempt_contract_path.is_file():
            raise ContinuationError(
                f"Continuation attempt contract is not a file: {attempt_contract_path}"
            )
        attempt_contract = validate_attempt_contract(
            attempt_contract_path,
            output_root,
            evidence_root,
            execution_contract_sha256,
            archive_manifest,
            archive_manifest_sha256,
            continuation_environment,
            current_completed,
        )
        attempt_contract_sha256 = finalizer.sha256_file(attempt_contract_path)
        interruption_records = interrupted_seed_records(
            output_root,
            attempt_contract_sha256,
        )
    else:
        failed_root = output_root / "failed_seed_attempts"
        if failed_root.exists() and any(failed_root.iterdir()):
            raise ContinuationError(
                "Unbound failed-seed attempts exist before the continuation contract"
            )
    if args.verify_existing:
        if current_completed != list(runner.PUBLICATION_SEEDS):
            raise ContinuationError("Publication seed set is incomplete")
        if attempt_contract is None or attempt_contract_sha256 is None:
            raise ContinuationError("Continuation attempt contract is absent")
        if output_is_finalized:
            finalizer.verify_finalized_root(
                output_root,
                execution_contract_sha256,
                context,
            )
        verify_continuation_evidence(
            evidence_root,
            output_root,
            execution_contract_sha256,
            archive_manifest,
            archive_manifest_sha256,
            attempt_contract_path,
            attempt_contract,
            attempt_contract_sha256,
            interruption_records,
            lock_recovery_records,
            continuation_environment,
        )
        post_evidence_recovery_records(output_root, evidence_root)
        if lock_path.exists():
            if not args.recover_stale_lock:
                raise ContinuationError(
                    "A post-evidence continuation lock remains; verify its process is "
                    "dead, then repeat --verify-existing with --recover-stale-lock"
                )
            preserved_lock = preserve_post_evidence_stale_lock(
                lock_path,
                output_root,
                evidence_root,
            )
            print(f"preserved post-evidence stale lock at {preserved_lock}", flush=True)
            post_evidence_recovery_records(output_root, evidence_root)
        if partial_finalization:
            print(
                "Partial finalization output remains; resume the finalizer after lock "
                "recovery.",
                flush=True,
            )
        print(evidence_root)
        return 0
    if evidence_root.exists():
        raise ContinuationError(
            "Continuation evidence already exists; use --verify-existing"
        )
    incomplete = incomplete_seed_ids(output_root)
    if incomplete and attempt_contract is None:
        raise ContinuationError(
            "Incomplete seeds predate the durable continuation contract: "
            f"{incomplete}"
        )
    attempt_targets = (
        attempt_contract["target_seeds"]
        if attempt_contract is not None
        else [
            seed for seed in runner.PUBLICATION_SEEDS if seed not in current_completed
        ]
    )
    if not set(incomplete).issubset(attempt_targets):
        raise ContinuationError(
            f"Incomplete seeds are outside the continuation target set: {incomplete}"
        )
    original_completed = (
        attempt_contract["completed_seeds_before_continuation"]
        if attempt_contract is not None
        else current_completed
    )
    report = {
        "protocol_id": PROTOCOL_ID,
        "completed_seeds_before_continuation": original_completed,
        "currently_completed_seeds": current_completed,
        "missing_seeds": [
            seed for seed in runner.PUBLICATION_SEEDS if seed not in current_completed
        ],
        "recoverable_incomplete_seeds": incomplete,
        "durable_attempt_contract_exists": attempt_contract is not None,
        "interrupted_archive_manifest_sha256": archive_manifest_sha256,
        "training_started": bool(args.confirm_continuation),
        "aggregation_performed": False,
    }
    print(json.dumps(report, indent=2), flush=True)
    if not args.confirm_continuation:
        print("Continuation was not started. Pass --confirm-continuation to run.")
        return 0

    with ExclusiveRunLock(
        lock_path,
        recover_stale=args.recover_stale_lock,
        stale_archive_root=output_root / STALE_LOCK_DIR_NAME,
    ) as active_lock:
        active_lock.assert_owned()
        output_finalization_state(output_root, False)
        if evidence_root.exists():
            raise ContinuationError(
                "Continuation evidence appeared before continuation acquired its lock"
            )
        locked_contract, locked_contract_sha256 = finalizer.validate_execution_identity(
            output_root
        )
        if (
            locked_contract != execution_contract
            or locked_contract_sha256 != execution_contract_sha256
        ):
            raise ContinuationError(
                "Execution identity changed before continuation acquired its lock"
            )
        _, locked_archive_sha256 = verify_archive_manifest(
            archive_manifest,
            execution_contract_sha256,
        )
        if locked_archive_sha256 != archive_manifest_sha256:
            raise ContinuationError(
                "Interrupted-attempt archive changed before continuation acquired its lock"
            )
        assert_runtime_identity(
            dataset_csv=dataset_csv,
            base_root=base_root,
            output_root=output_root,
            archive_manifest=archive_manifest,
            expected_execution_contract=execution_contract,
            expected_execution_contract_sha256=execution_contract_sha256,
            expected_archive_manifest_sha256=archive_manifest_sha256,
            expected_environment=continuation_environment,
            expected_support_sources=expected_support_sources,
            device=device,
        )
        locked_completed = finalizer.complete_seed_ids(output_root)
        if locked_completed != current_completed:
            raise ContinuationError(
                "Completed seed set changed before continuation acquired its lock"
            )
        finalizer.verify_seed_set(
            output_root,
            locked_completed,
            execution_contract_sha256,
            context,
        )
        locked_incomplete = incomplete_seed_ids(output_root)
        if attempt_contract is None and locked_incomplete:
            raise ContinuationError(
                "An incomplete seed appeared before the durable attempt contract"
            )
        attempt_contract = ensure_attempt_contract(
            attempt_contract_path,
            output_root,
            evidence_root,
            execution_contract_sha256,
            archive_manifest,
            archive_manifest_sha256,
            locked_completed,
            continuation_environment,
        )
        attempt_contract_sha256 = finalizer.sha256_file(attempt_contract_path)
        if not set(locked_incomplete).issubset(attempt_contract["target_seeds"]):
            raise ContinuationError(
                "An incomplete seed is outside the durable continuation target set"
            )
        for seed in locked_incomplete:
            archived = archive_incomplete_seed(
                output_root,
                seed,
                attempt_contract_sha256,
            )
            print(f"preserved interrupted seed={seed} at {archived}", flush=True)
        if incomplete_seed_ids(output_root):
            raise ContinuationError("Incomplete seed preservation did not finish")
        interruption_records = interrupted_seed_records(
            output_root,
            attempt_contract_sha256,
        )
        lock_recovery_records = stale_lock_records(output_root)
        original_verifier = runner.metrics_from_npz_predictions
        runner.metrics_from_npz_predictions = (
            finalizer.corrected_metrics_from_npz_predictions
        )
        try:
            for seed in runner.PUBLICATION_SEEDS:
                active_lock.assert_owned()
                completed_before_seed = set(finalizer.complete_seed_ids(output_root))
                missing_before_seed = seed not in completed_before_seed
                failed_before_seed = failed_attempt_directory_names(output_root)
                if missing_before_seed:
                    seed_root = output_root / f"seed_{seed}"
                    if seed_root.exists():
                        raise ContinuationError(
                            f"Target seed appeared outside the continuation lifecycle: {seed}"
                        )
                    assert_runtime_identity(
                        dataset_csv=dataset_csv,
                        base_root=base_root,
                        output_root=output_root,
                        archive_manifest=archive_manifest,
                        expected_execution_contract=execution_contract,
                        expected_execution_contract_sha256=execution_contract_sha256,
                        expected_archive_manifest_sha256=archive_manifest_sha256,
                        expected_environment=continuation_environment,
                        expected_support_sources=expected_support_sources,
                        device=device,
                    )
                completion = (
                    run_seed_without_fallback_archive(
                        output_root,
                        base_root,
                        context,
                        seed,
                        device,
                        execution_contract_sha256,
                    )
                    if missing_before_seed
                    else runner.run_seed(
                        output_root,
                        base_root,
                        context,
                        seed,
                        device,
                        execution_contract_sha256,
                        True,
                    )
                )
                if missing_before_seed:
                    failed_after_seed = failed_attempt_directory_names(output_root)
                    if failed_after_seed != failed_before_seed:
                        raise ContinuationError(
                            "The original runner fallback archive was triggered during "
                            f"continuation seed {seed}; refusing to accept the concurrent run"
                        )
                    assert_runtime_identity(
                        dataset_csv=dataset_csv,
                        base_root=base_root,
                        output_root=output_root,
                        archive_manifest=archive_manifest,
                        expected_execution_contract=execution_contract,
                        expected_execution_contract_sha256=execution_contract_sha256,
                        expected_archive_manifest_sha256=archive_manifest_sha256,
                        expected_environment=continuation_environment,
                        expected_support_sources=expected_support_sources,
                        device=device,
                    )
                completion = runner.verify_completed_seed_output(
                    output_root / f"seed_{seed}",
                    seed,
                    execution_contract_sha256,
                    context,
                )
                active_lock.assert_owned()
                print(
                    f"verified/completed seed={seed} "
                    f"wall_seconds={completion['wall_seconds']:.1f}",
                    flush=True,
                )
        finally:
            runner.metrics_from_npz_predictions = original_verifier

        finalizer.verify_seed_set(
            output_root,
            list(runner.PUBLICATION_SEEDS),
            execution_contract_sha256,
            context,
        )
        assert_runtime_identity(
            dataset_csv=dataset_csv,
            base_root=base_root,
            output_root=output_root,
            archive_manifest=archive_manifest,
            expected_execution_contract=execution_contract,
            expected_execution_contract_sha256=execution_contract_sha256,
            expected_archive_manifest_sha256=archive_manifest_sha256,
            expected_environment=continuation_environment,
            expected_support_sources=expected_support_sources,
            device=device,
        )
        active_lock.assert_owned()
        write_continuation_evidence(
            evidence_root,
            output_root,
            execution_contract_sha256,
            archive_manifest,
            archive_manifest_sha256,
            attempt_contract_path,
            attempt_contract,
            attempt_contract_sha256,
            interruption_records,
            lock_recovery_records,
            continuation_environment,
        )
        active_lock.assert_owned()
    print(evidence_root)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContinuationError, finalizer.FinalizationError) as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
