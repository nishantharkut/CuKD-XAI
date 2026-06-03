"""Prepare an Arduino/ESP32 sketch folder for CuKD-XAI hardware self-test."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HARDWARE_DIR = Path(__file__).resolve().parent
SKETCH_TEMPLATE_DIR = HARDWARE_DIR / "arduino_esp32_student_a_rfkd_self_test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy generated CuKD-XAI fixed-point artifacts into an Arduino/ESP32 sketch folder."
    )
    parser.add_argument(
        "--generated-dir",
        required=True,
        help="Directory created by run_wsnds_student_a_rfkd_e2e.py, containing model_weights.h and test_vectors.h.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(HARDWARE_DIR / "arduino_esp32_student_a_rfkd_package"),
        help="Arduino sketch output directory.",
    )
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_vector_count(test_vectors_h: Path) -> int | None:
    for line in test_vectors_h.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("#define CUKD_TEST_VECTOR_COUNT"):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    return int(parts[2])
                except ValueError:
                    return None
    return None


def copy_artifact(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> int:
    args = parse_args()
    generated_dir = Path(args.generated_dir)
    if not generated_dir.is_absolute():
        generated_dir = (REPO_ROOT / generated_dir).resolve()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    required = {
        "model_weights.h": require_file(generated_dir / "model_weights.h"),
        "test_vectors.h": require_file(generated_dir / "test_vectors.h"),
        "wsnds_student_a_rfkd_int8_inference.c": require_file(HARDWARE_DIR / "wsnds_student_a_rfkd_int8_inference.c"),
        "arduino_esp32_student_a_rfkd_self_test.ino": require_file(
            SKETCH_TEMPLATE_DIR / "arduino_esp32_student_a_rfkd_self_test.ino"
        ),
        "README.md": require_file(SKETCH_TEMPLATE_DIR / "README.md"),
    }

    sketch_name = output_dir.name
    ino_dst_name = f"{sketch_name}.ino"
    copy_artifact(required["arduino_esp32_student_a_rfkd_self_test.ino"], output_dir / ino_dst_name)
    for name in [
        "model_weights.h",
        "test_vectors.h",
        "wsnds_student_a_rfkd_int8_inference.c",
        "README.md",
    ]:
        copy_artifact(required[name], output_dir / name)

    vector_count = read_vector_count(required["test_vectors.h"])
    manifest = {
        "generated_dir": str(generated_dir),
        "output_dir": str(output_dir),
        "sketch_file": ino_dst_name,
        "vector_count": vector_count,
        "required_files": [
            ino_dst_name,
            "model_weights.h",
            "test_vectors.h",
            "wsnds_student_a_rfkd_int8_inference.c",
        ],
        "note": "Use 256 or 1000 vectors for small boards. The full 56200-vector header is mainly for host/toolchain evidence.",
    }
    (output_dir / "package_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
