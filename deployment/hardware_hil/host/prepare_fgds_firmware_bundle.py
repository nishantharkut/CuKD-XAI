"""Create a hash-bound Arduino bundle from a passed FG-DS deployment export."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

try:
    from .stream_vectors_fgds_strict import verify_export
except ImportError:
    from stream_vectors_fgds_strict import verify_export


HIL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = HIL_ROOT.parents[1]
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
    "cukd_export_identity.h",
]
BOARD_TEMPLATES = {
    "esp32c3": HIL_ROOT / "firmware" / "esp32c3" / "src" / "main.cpp",
    "arduino_r4": HIL_ROOT / "firmware" / "arduino_r4" / "cukd_hil_r4" / "cukd_hil_r4.ino",
}
PROCESS_SIGNATURES = {
    "esp32c3": "static void process_line(const char *line) {\n",
    "arduino_r4": "static void cukd_process_line(const char *line) {\n",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def read_passed_export(generated_dir: Path) -> dict[str, Any]:
    return verify_export(generated_dir.resolve())


def bundle_identity_payload(
    board: str,
    export_id: str,
    source_files: list[Path],
    transformed_sketch: str,
) -> dict[str, Any]:
    return {
        "board": board,
        "export_id": export_id,
        "source_files": [
            {"name": path.name, "sha256": sha256_file(path)} for path in source_files
        ],
        "transformed_sketch_sha256": hashlib.sha256(
            transformed_sketch.encode("utf-8")
        ).hexdigest(),
        "bundler_sha256": sha256_file(Path(__file__).resolve()),
    }


def compute_bundle_id(
    board: str,
    export_id: str,
    source_files: list[Path],
    transformed_sketch: str,
) -> str:
    payload = bundle_identity_payload(
        board, export_id, source_files, transformed_sketch
    )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def strict_sketch(source: str, board: str) -> str:
    source = source.replace("\r\n", "\n")
    if "\r" in source:
        raise RuntimeError("Board template contains unsupported mixed line endings")
    include_anchor = "#include <string.h>\n"
    identity_include = (
        "#include <string.h>\n\n"
        "#include \"cukd_export_identity.h\"\n"
        "#include \"cukd_bundle_identity.h\"\n"
    )
    if source.count(include_anchor) != 1:
        raise RuntimeError("Board template include anchor is missing or ambiguous")
    source = source.replace(include_anchor, identity_include, 1)

    function_anchor = PROCESS_SIGNATURES[board]
    identity_handler = function_anchor + (
        "    if (strcmp(line, \"CUKDID?\") == 0) {\n"
        "        Serial.print(\"CUKDBUILD,\");\n"
        "        Serial.print(CUKD_STUDENT_ID);\n"
        "        Serial.print(\",\");\n"
        "        Serial.print(CUKD_EXPORT_ID);\n"
        "        Serial.print(\",\");\n"
        "        Serial.println(CUKD_BUNDLE_ID);\n"
        "        return;\n"
        "    }\n"
    )
    if source.count(function_anchor) != 1:
        raise RuntimeError("Board template process-line anchor is missing or ambiguous")
    return source.replace(function_anchor, identity_handler, 1)


def prepare_staging(output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing bundle path: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / (
        f".{output_dir.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    if staging.exists():
        raise FileExistsError(f"Stale bundle staging path exists: {staging}")
    staging.mkdir()
    return staging


def discard_staging(staging: Path, parent: Path) -> None:
    resolved = staging.resolve()
    expected_parent = parent.resolve()
    if resolved.parent != expected_parent or not resolved.name.startswith(".") or ".tmp." not in resolved.name:
        raise RuntimeError(f"Refusing to remove an unsafe staging path: {resolved}")
    shutil.rmtree(resolved)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", choices=sorted(BOARD_TEMPLATES), required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    generated_dir = resolve(args.generated_dir)
    output_dir = resolve(args.output_dir)
    export_manifest = read_passed_export(generated_dir)
    export_report = json.loads(
        (generated_dir / "strict_export_report.json").read_text(encoding="utf-8")
    )
    tested_common = export_report.get("provenance", {}).get("firmware_common_files")
    expected_tested_names = {
        "cukd_model.c", "cukd_model.h", "cukd_preprocess.c", "cukd_preprocess.h"
    }
    if not isinstance(tested_common, dict) or set(tested_common) != expected_tested_names:
        raise RuntimeError("Strict export lacks the complete host-tested common-kernel identity")
    for name, expected_sha256 in tested_common.items():
        if sha256_file(COMMON_DIR / name) != expected_sha256:
            raise RuntimeError(
                f"Firmware common file changed after host equivalence testing: {name}"
            )
    try:
        output_dir.relative_to(generated_dir)
    except ValueError:
        pass
    else:
        raise RuntimeError("Firmware bundle output cannot be inside its strict export")
    template = BOARD_TEMPLATES[args.board]
    source_files = [
        template,
        *[COMMON_DIR / name for name in COMMON_FILES],
        *[generated_dir / name for name in GENERATED_FILES],
    ]
    for path in source_files:
        if not path.is_file():
            raise FileNotFoundError(path)
    source_snapshots = {path.name: path.read_bytes() for path in source_files}
    if len(source_snapshots) != len(source_files):
        raise RuntimeError("Strict bundle source filenames are not unique")
    export_hashes = {
        item["path"]: item["sha256"] for item in export_manifest.get("files", [])
    }
    for name in GENERATED_FILES:
        observed = hashlib.sha256(source_snapshots[name]).hexdigest()
        if observed != export_hashes.get(name):
            raise RuntimeError(f"Generated bundle input changed after export verification: {name}")
    for name, expected_sha256 in tested_common.items():
        observed = hashlib.sha256(source_snapshots[name]).hexdigest()
        if observed != expected_sha256:
            raise RuntimeError(f"Host-tested common bundle input changed: {name}")
    template_text = source_snapshots[template.name].decode("utf-8")
    transformed = strict_sketch(template_text, args.board)
    identity_payload = {
        "board": args.board,
        "export_id": export_manifest["export_id"],
        "source_files": [
            {
                "name": path.name,
                "sha256": hashlib.sha256(source_snapshots[path.name]).hexdigest(),
            }
            for path in source_files
        ],
        "transformed_sketch_sha256": hashlib.sha256(
            transformed.encode("utf-8")
        ).hexdigest(),
        "bundler_sha256": sha256_file(Path(__file__).resolve()),
    }
    bundle_id = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    staging = prepare_staging(output_dir)

    sketch_name = f"{output_dir.name}.ino"
    try:
        for name in [*COMMON_FILES, *GENERATED_FILES]:
            (staging / name).write_bytes(source_snapshots[name])
        (staging / "cukd_bundle_identity.h").write_text(
            "#ifndef CUKD_BUNDLE_IDENTITY_H\n"
            "#define CUKD_BUNDLE_IDENTITY_H\n"
            f"#define CUKD_BUNDLE_ID \"{bundle_id}\"\n"
            "#endif\n",
            encoding="ascii",
        )
        (staging / sketch_name).write_bytes(transformed.encode("utf-8"))

        manifest_path = staging / "strict_bundle_manifest.json"
        files = [
            {
                "path": path.relative_to(staging).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(staging.iterdir())
            if path.is_file() and path != manifest_path
        ]
        manifest = {
            "status": "passed",
            "board": args.board,
            "student": export_manifest["student"],
            "export_id": export_manifest["export_id"],
            "bundle_id": bundle_id,
            "bundle_identity_payload": identity_payload,
            "generated_dir_recorded": str(generated_dir),
            "strict_export_manifest_sha256": sha256_file(
                generated_dir / "strict_export_manifest.json"
            ),
            "base_template": str(template),
            "base_template_sha256": hashlib.sha256(
                source_snapshots[template.name]
            ).hexdigest(),
            "transformed_sketch_sha256": hashlib.sha256(
                transformed.encode("utf-8")
            ).hexdigest(),
            "bundler_sha256": identity_payload["bundler_sha256"],
            "sketch_file": sketch_name,
            "serial_identity_query": "CUKDID?",
            "serial_identity_response": (
                f"CUKDBUILD,{export_manifest['student']},"
                f"{export_manifest['export_id']},{bundle_id}"
            ),
            "file_count_excluding_manifest": len(files),
            "files": files,
            "claim_boundary": (
                "USB serial replay of extracted WSN-DS 17-feature records from "
                "the FG-DS seed-42 test partition; no live packet capture."
            ),
        }
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(temporary, manifest_path)
        os.replace(staging, output_dir)
    except Exception:
        if staging.exists():
            discard_staging(staging, output_dir.parent)
        raise
    print(output_dir / "strict_bundle_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
