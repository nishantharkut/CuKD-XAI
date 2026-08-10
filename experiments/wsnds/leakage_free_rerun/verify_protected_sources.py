"""Record or verify the tracked-source and durable-manuscript baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_MANIFEST = SCRIPT_DIR / "protected_sources.sha256.json"
MANUSCRIPT_TRANSIENT_SUFFIXES = {
    ".aux",
    ".bbl",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_paths() -> list[Path]:
    proc = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    relative = [Path(item.decode("utf-8")) for item in proc.stdout.split(b"\0") if item]

    manuscript = REPO_ROOT / "manuscript"
    if manuscript.exists():
        relative.extend(
            path.relative_to(REPO_ROOT)
            for path in manuscript.rglob("*")
            if path.is_file() and path.suffix.lower() not in MANUSCRIPT_TRANSIENT_SUFFIXES
        )

    unique = sorted({path.as_posix(): path for path in relative}.values(), key=lambda p: p.as_posix())
    return [path for path in unique if (REPO_ROOT / path).is_file()]


def record(manifest_path: Path) -> int:
    entries = []
    for relative in protected_paths():
        absolute = REPO_ROOT / relative
        entries.append(
            {
                "path": relative.as_posix(),
                "size": absolute.stat().st_size,
                "sha256": sha256(absolute),
            }
        )
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "file_count": len(entries),
        "files": entries,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"recorded {len(entries)} protected files in {manifest_path}")
    return 0


def verify(manifest_path: Path) -> int:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for entry in payload["files"]:
        path = REPO_ROOT / entry["path"]
        if not path.is_file():
            failures.append(f"missing: {entry['path']}")
            continue
        size = path.stat().st_size
        digest = sha256(path)
        if size != entry["size"] or digest != entry["sha256"]:
            failures.append(f"changed: {entry['path']}")
    if failures:
        print("\n".join(failures))
        return 1
    print(f"verified {len(payload['files'])} protected files unchanged")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("record", "verify"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    return record(args.manifest) if args.mode == "record" else verify(args.manifest)


if __name__ == "__main__":
    raise SystemExit(main())
