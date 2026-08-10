"""Verify a byte-preserving FG-DS HIL transfer after archive extraction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPO_ROOT / "_FGDS_TRANSFER_MANIFEST.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("protocol_id") != "fgds_seed42_byte_preserving_transfer_v1":
        raise RuntimeError("Unexpected FG-DS transfer protocol")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("FG-DS transfer manifest has no file inventory")
    seen: set[str] = set()
    for item in files:
        relative = Path(item.get("path", ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"Unsafe transfer path: {relative}")
        normalized = relative.as_posix()
        if normalized in seen:
            raise RuntimeError(f"Duplicate transfer path: {normalized}")
        seen.add(normalized)
        path = (REPO_ROOT / relative).resolve()
        try:
            path.relative_to(REPO_ROOT)
        except ValueError as exc:
            raise RuntimeError(f"Transfer path escapes repository: {relative}") from exc
        if (
            not path.is_file()
            or path.stat().st_size != item.get("size_bytes")
            or sha256_file(path) != item.get("sha256")
        ):
            raise RuntimeError(f"Transferred file is missing or changed: {relative}")
    if len(seen) != manifest.get("file_count"):
        raise RuntimeError("FG-DS transfer file count differs from its manifest")
    print(f"FG-DS transfer verified: {len(seen)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
