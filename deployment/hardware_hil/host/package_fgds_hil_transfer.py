"""Create a byte-preserving transfer archive for the FG-DS HIL rerun."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path

try:
    from .preflight_fgds_hil import BUNDLE_DIRS, EXPORT_DIRS, REPO_ROOT, run_preflight
except ImportError:
    from preflight_fgds_hil import BUNDLE_DIRS, EXPORT_DIRS, REPO_ROOT, run_preflight


HOST_FILES = [
    "deployment/firmware_export/wsnds_rfkd_hil/export_fgds_seed42_deployment.py",
    "deployment/hardware_hil/__init__.py",
    "deployment/hardware_hil/host/__init__.py",
    "deployment/hardware_hil/host/env_check.py",
    "deployment/hardware_hil/host/hil_common.py",
    "deployment/hardware_hil/host/preflight_fgds_hil.py",
    "deployment/hardware_hil/host/prepare_fgds_firmware_bundle.py",
    "deployment/hardware_hil/host/requirements.txt",
    "deployment/hardware_hil/host/stream_vectors.py",
    "deployment/hardware_hil/host/stream_vectors_fgds_strict.py",
    "deployment/hardware_hil/host/verify_results_fgds_strict.py",
    "deployment/hardware_hil/host/record_fgds_compile_evidence.py",
    "deployment/hardware_hil/host/generate_fgds_report.py",
    "deployment/hardware_hil/host/verify_fgds_transfer.py",
    "deployment/hardware_hil/scripts/run_fgds_hil.sh",
    "deployment/hardware_hil/docs/12_FGDS_SEED42_HIL_RUNBOOK.md",
    "tests/hardware_deployment_run/test_fgds_hil_contract.py",
]
COMPILE_EVIDENCE_ROOT = REPO_ROOT / "results" / "hardware_hil" / "fgds_seed42" / "compile_evidence"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def collect_files() -> list[Path]:
    files = [(REPO_ROOT / relative).resolve() for relative in HOST_FILES]
    for root in [*EXPORT_DIRS.values(), *BUNDLE_DIRS.values(), COMPILE_EVIDENCE_ROOT]:
        if not root.is_dir():
            raise FileNotFoundError(root)
        files.extend(path.resolve() for path in root.rglob("*") if path.is_file())
    unique = sorted(set(files), key=lambda path: path.relative_to(REPO_ROOT).as_posix())
    for path in unique:
        try:
            path.relative_to(REPO_ROOT)
        except ValueError as exc:
            raise RuntimeError(f"Transfer input escapes repository: {path}") from exc
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "dist" / "cukd_fgds_seed42_hil_ready.zip",
    )
    args = parser.parse_args()
    run_preflight()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite transfer archive: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Stale transfer temporary exists: {temporary}")

    files = collect_files()
    inventory = []
    snapshots: list[tuple[str, bytes]] = []
    for path in files:
        relative = path.relative_to(REPO_ROOT).as_posix()
        payload = path.read_bytes()
        snapshots.append((relative, payload))
        inventory.append({
            "path": relative,
            "size_bytes": len(payload),
            "sha256": sha256_bytes(payload),
        })
    manifest = {
        "protocol_id": "fgds_seed42_byte_preserving_transfer_v1",
        "file_count": len(inventory),
        "files": inventory,
        "extraction_instruction": "Extract at the repository root, then run verify_fgds_transfer.py.",
    }
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("_FGDS_TRANSFER_MANIFEST.json", manifest_bytes)
            for relative, payload in snapshots:
                archive.writestr(relative, payload)
        with zipfile.ZipFile(temporary, "r") as archive:
            if archive.testzip() is not None:
                raise RuntimeError("FG-DS transfer archive failed its CRC check")
            archived_manifest = json.loads(
                archive.read("_FGDS_TRANSFER_MANIFEST.json").decode("utf-8")
            )
            if archived_manifest != manifest:
                raise RuntimeError("FG-DS transfer manifest changed inside the archive")
            for item in inventory:
                if sha256_bytes(archive.read(item["path"])) != item["sha256"]:
                    raise RuntimeError(f"Archived payload hash differs: {item['path']}")
        os.replace(temporary, output)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    print(json.dumps({
        "status": "passed",
        "archive": str(output),
        "archive_sha256": sha256_bytes(output.read_bytes()),
        "file_count": len(inventory),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
