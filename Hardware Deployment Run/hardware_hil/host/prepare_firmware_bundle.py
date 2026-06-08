"""Create Arduino-IDE-ready HIL firmware bundles for supported boards."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


HIL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = HIL_ROOT.parent
COMMON_DIR = HIL_ROOT / "firmware" / "common"

COMMON_FILES = [
    "cukd_model.h",
    "cukd_model.c",
    "cukd_preprocess.h",
    "cukd_preprocess.c",
    "cukd_protocol.h",
    "cukd_protocol.c",
]
GENERATED_FILES = [
    "model_weights.h",
    "preprocess_int_metadata.h",
]
BOARD_TEMPLATES = {
    "esp32c3": HIL_ROOT / "firmware" / "esp32c3" / "src" / "main.cpp",
    "arduino_r4": HIL_ROOT / "firmware" / "arduino_r4" / "cukd_hil_r4" / "cukd_hil_r4.ino",
}


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = (REPO_ROOT / resolved).resolve()
    return resolved


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_bundle(board: str, generated_dir: Path, output_dir: Path) -> dict[str, object]:
    if board not in BOARD_TEMPLATES:
        raise ValueError(f"unsupported board {board!r}; expected one of {sorted(BOARD_TEMPLATES)}")

    generated_dir = resolve_repo_path(generated_dir)
    output_dir = resolve_repo_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    sketch_name = f"{output_dir.name}.ino"
    copy_file(require_file(BOARD_TEMPLATES[board]), output_dir / sketch_name)
    copied.append(sketch_name)

    for name in GENERATED_FILES:
        copy_file(require_file(generated_dir / name), output_dir / name)
        copied.append(name)

    for name in COMMON_FILES:
        copy_file(require_file(COMMON_DIR / name), output_dir / name)
        copied.append(name)

    manifest = {
        "board": board,
        "generated_dir": str(generated_dir),
        "output_dir": str(output_dir),
        "sketch_file": sketch_name,
        "copied_files": copied,
        "serial_baud": 115200,
        "entrypoint": "Arduino setup()/loop(); no Wi-Fi, Bluetooth, filesystem, or cloud path",
        "claim_boundary": "USB serial replay of already extracted WSN-DS 17-feature records; not live WSN packet capture.",
    }
    (output_dir / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", choices=sorted(BOARD_TEMPLATES), required=True)
    parser.add_argument("--generated-dir", required=True, help="Directory from hardware_export/run_wsnds_student_a_rfkd_e2e.py")
    parser.add_argument("--output-dir", required=True, help="Arduino sketch folder to create")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_bundle(
        board=args.board,
        generated_dir=Path(args.generated_dir),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
