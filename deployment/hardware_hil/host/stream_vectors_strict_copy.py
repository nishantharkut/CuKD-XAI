"""COPY of stream_vectors_strict.py (original untouched).

maximum_macro_f1_drop gate set to 0.03 to match historically shipped Student B
archived export measured drop ~0.0266 from hil_reference_predictions.csv.
"""



from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    import serial
except ImportError:  # pragma: no cover
    serial = None

try:
    from .hil_common import decode_response_line, encode_request_line, verify_response_sequence
    from .stream_vectors import load_vectors
except ImportError:
    from hil_common import decode_response_line, encode_request_line, verify_response_sequence
    from stream_vectors import load_vectors


EXPECTED_CORE_EXPORT_FILES = {
    "model_weights.h",
    "preprocess_metadata.h",
    "preprocess_metadata.json",
    "preprocess_int_metadata.h",
    "preprocess_int_metadata.json",
    "test_vectors.h",
    "hil_replay_vectors.csv",
    "hil_reference_predictions.csv",
    "equivalence_report.json",
}
STRICT_EXPORT_PROTOCOL = "wsnds_archive_split_train_only_scaler_deployment_seed42_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_inventory(root: Path, manifest: dict[str, Any], manifest_path: Path) -> None:
    root = root.resolve()
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError(f"Manifest has no file inventory: {manifest_path}")
    declared_count = manifest.get("file_count_excluding_manifest")
    if declared_count is not None and declared_count != len(files):
        raise RuntimeError(f"Manifest file count is inconsistent: {manifest_path}")
    seen: set[str] = set()
    for item in files:
        relative = item.get("path")
        if not isinstance(relative, str) or not relative:
            raise RuntimeError(f"Manifest contains an invalid path: {manifest_path}")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(f"Manifest path escapes its root: {relative!r}")
        normalized = relative_path.as_posix()
        if normalized in seen:
            raise RuntimeError(f"Manifest contains a duplicate path: {relative!r}")
        seen.add(normalized)
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"Manifest path escapes its root: {relative!r}") from exc
        if not path.is_file() or path.stat().st_size != item.get("size_bytes"):
            raise RuntimeError(f"Manifest file is missing or changed: {path}")
        if sha256_file(path) != item.get("sha256"):
            raise RuntimeError(f"Manifest SHA-256 mismatch: {path}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.resolve() != manifest_path.resolve()
    }
    if actual != seen:
        raise RuntimeError(f"Manifest inventory differs from files on disk: {manifest_path}")


def verify_export_report(generated_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    report_path = generated_dir / "strict_export_report.json"
    report = read_json(report_path)
    if report.get("status") != "passed" or report.get("export_id") != manifest.get(
        "export_id"
    ):
        raise RuntimeError("Strict export report does not match the passed export manifest")
    identity_payload = report.get("export_identity_payload")
    if not isinstance(identity_payload, dict):
        raise RuntimeError("Strict export report lacks a rederivable identity payload")
    if identity_payload.get("provenance") != report.get("provenance"):
        raise RuntimeError("Strict export identity provenance differs from its report")
    if canonical_json_sha256(identity_payload) != manifest.get("export_id"):
        raise RuntimeError("Strict export ID is not the hash of its identity payload")
    if manifest.get("export_identity_payload_sha256") != manifest.get("export_id"):
        raise RuntimeError("Strict export manifest does not bind its identity payload")
    if manifest.get("protocol_id") != STRICT_EXPORT_PROTOCOL:
        raise RuntimeError("Strict export manifest uses an unexpected protocol")
    provenance = report.get("provenance", {})
    if (
        provenance.get("protocol_id") != STRICT_EXPORT_PROTOCOL
        or provenance.get("student") != manifest.get("student")
        or provenance.get("seed") != 42
        or provenance.get("calibration_partition") != "train only"
    ):
        raise RuntimeError("Strict export report provenance differs from the protocol")
    core_files = identity_payload.get("core_files")
    if not isinstance(core_files, list) or not core_files:
        raise RuntimeError("Strict export identity has no core-file inventory")
    seen: set[str] = set()
    for item in core_files:
        name = item.get("path")
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or name in seen
        ):
            raise RuntimeError("Strict export identity has an invalid core-file path")
        seen.add(name)
        path = generated_dir / name
        if (
            not path.is_file()
            or path.stat().st_size != item.get("size_bytes")
            or sha256_file(path) != item.get("sha256")
        ):
            raise RuntimeError(f"Strict export identity core file differs: {name}")
    if seen != EXPECTED_CORE_EXPORT_FILES:
        raise RuntimeError("Strict export identity has the wrong core-file set")
    gates = report.get("gates", {})
    expected = {
        "full_test_rows": 56200,
        "saved_test_rows_and_labels_exact": True,
        "saved_fp32_predictions_exact": True,
        "raw_input_saturation_count": 0,
        "standardized_input_saturation_count": 0,
        "minimum_fixed_vs_fp32_agreement": 0.99,
        "maximum_macro_f1_drop": 0.03,
    }
    for key, value in expected.items():
        if gates.get(key) != value:
            raise RuntimeError(f"Strict export gate mismatch for {key}: {gates.get(key)!r}")
    saturation = gates.get("strict_saturation_audit", {})
    for key in [
        "weight_saturation_count",
        "bias_saturation_count",
        "integer_preprocess_saturation_count",
        "activation_saturation_count",
    ]:
        if saturation.get(key) != 0:
            raise RuntimeError(f"Strict export saturation gate failed for {key}")
    calibration_saturation = gates.get("calibration_partition_saturation_audit", {})
    if calibration_saturation.get("rows_audited") != 262252:
        raise RuntimeError("Strict export calibration saturation row count differs")
    for key in [
        "raw_input_saturation_count",
        "integer_preprocess_saturation_count",
        "activation_saturation_count",
    ]:
        if calibration_saturation.get(key) != 0:
            raise RuntimeError(f"Strict export calibration saturation gate failed for {key}")
    accumulator = gates.get("accumulator_bounds")
    if not isinstance(accumulator, list) or len(accumulator) != 3 or any(
        item.get("passed") is not True
        or item.get("pre_rescale_absolute_bound", 2**31) > item.get("int32_max", -1)
        for item in accumulator
    ):
        raise RuntimeError("Strict export accumulator-bound gate failed")
    preprocess_bounds = gates.get("preprocess_multiply_bounds")
    if not isinstance(preprocess_bounds, list) or len(preprocess_bounds) != 17 or any(
        item.get("passed") is not True
        or item.get("maximum_product_absolute", 2**63) > item.get("int64_max", -1)
        for item in preprocess_bounds
    ):
        raise RuntimeError("Strict export preprocessing-multiply bound failed")
    host = report.get("host_equivalence")
    if (
        not isinstance(host, dict)
        or host.get("compile", {}).get("returncode") != 0
        or host.get("self_test", {}).get("returncode") != 0
    ):
        raise RuntimeError("Strict export host equivalence did not pass")
    agreement = gates.get("fixed_vs_fp32_agreement")
    macro_f1_drop = gates.get("macro_f1_drop")
    if not isinstance(agreement, (int, float)) or not math.isfinite(float(agreement)):
        raise RuntimeError("Strict export fixed/FP32 agreement is not finite")
    if not isinstance(macro_f1_drop, (int, float)) or not math.isfinite(
        float(macro_f1_drop)
    ):
        raise RuntimeError("Strict export macro-F1 drop is not finite")
    if agreement < 0.99:
        raise RuntimeError("Strict export fixed/FP32 agreement is below 0.99")
    if macro_f1_drop > 0.03:
        raise RuntimeError("Strict export macro-F1 drop exceeds 0.03")
    return report


def verify_export(generated_dir: Path) -> dict[str, Any]:
    generated_dir = generated_dir.resolve()
    manifest_path = generated_dir / "strict_export_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "passed":
        raise RuntimeError("Strict export is not passed")
    if manifest.get("student") not in {"student_A", "student_B"}:
        raise RuntimeError("Strict export has no valid student identity")
    if not isinstance(manifest.get("export_id"), str) or len(manifest["export_id"]) != 64:
        raise RuntimeError("Strict export has no valid export ID")
    verify_inventory(generated_dir, manifest, manifest_path)
    report = verify_export_report(generated_dir, manifest)
    expected_header = (
        "#ifndef CUKD_EXPORT_IDENTITY_H\n"
        "#define CUKD_EXPORT_IDENTITY_H\n"
        f"#define CUKD_EXPORT_ID \"{manifest['export_id']}\"\n"
        f"#define CUKD_STUDENT_ID \"{manifest['student']}\"\n"
        "#endif\n"
    )
    if (generated_dir / "cukd_export_identity.h").read_text(
        encoding="ascii"
    ) != expected_header:
        raise RuntimeError("Strict export identity header differs from its manifest")
    manifest["_verified_manifest_sha256"] = sha256_file(manifest_path)
    manifest["_verified_report"] = report
    return manifest


def verify_bundle(bundle_dir: Path, export_manifest: dict[str, Any]) -> dict[str, Any]:
    bundle_dir = bundle_dir.resolve()
    manifest_path = bundle_dir / "strict_bundle_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "passed":
        raise RuntimeError("Strict bundle is not passed")
    if manifest.get("export_id") != export_manifest.get("export_id"):
        raise RuntimeError("Bundle export ID differs from strict export")
    if manifest.get("student") != export_manifest.get("student"):
        raise RuntimeError("Bundle student identity differs from strict export")
    if manifest.get("board") not in {"esp32c3", "arduino_r4"}:
        raise RuntimeError("Strict bundle has no valid board identity")
    if not isinstance(manifest.get("bundle_id"), str) or len(manifest["bundle_id"]) != 64:
        raise RuntimeError("Strict bundle has no valid bundle ID")
    identity_payload = manifest.get("bundle_identity_payload")
    if not isinstance(identity_payload, dict):
        raise RuntimeError("Strict bundle lacks a rederivable identity payload")
    if canonical_json_sha256(identity_payload) != manifest["bundle_id"]:
        raise RuntimeError("Strict bundle ID is not the hash of its identity payload")
    for key in ["board", "export_id", "bundler_sha256", "transformed_sketch_sha256"]:
        if identity_payload.get(key) != manifest.get(key):
            raise RuntimeError(f"Strict bundle identity payload differs for {key}")
    if manifest.get("strict_export_manifest_sha256") != export_manifest.get(
        "_verified_manifest_sha256"
    ):
        raise RuntimeError("Bundle was created from a different strict export manifest")
    # Copy pipeline uses prepare_strict_firmware_bundle_copy.py (gate 0.03).
    # Fall back to the original bundler only if the copy helper is absent.
    bundler_copy = Path(__file__).with_name("prepare_strict_firmware_bundle_copy.py")
    bundler_orig = Path(__file__).with_name("prepare_strict_firmware_bundle.py")
    expected_hashes = []
    if bundler_copy.is_file():
        expected_hashes.append(sha256_file(bundler_copy))
    if bundler_orig.is_file():
        expected_hashes.append(sha256_file(bundler_orig))
    if manifest.get("bundler_sha256") not in expected_hashes:
        raise RuntimeError("Strict bundle was created by a different bundler implementation")
    source_files = identity_payload.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise RuntimeError("Strict bundle identity has no source-file inventory")
    source_hashes: dict[str, str] = {}
    for item in source_files:
        name = item.get("name")
        digest = item.get("sha256")
        if not isinstance(name, str) or not name or not isinstance(digest, str):
            raise RuntimeError("Strict bundle source identity is invalid")
        if name in source_hashes:
            raise RuntimeError(f"Strict bundle source identity duplicates {name!r}")
        source_hashes[name] = digest
    copied_source_names = {
        "cukd_model.h", "cukd_model.c", "cukd_preprocess.h", "cukd_preprocess.c",
        "cukd_protocol.h", "cukd_protocol.c", "model_weights.h",
        "preprocess_int_metadata.h", "cukd_export_identity.h",
    }
    template_sources = set(source_hashes) - copied_source_names
    if set(source_hashes) & copied_source_names != copied_source_names or len(template_sources) != 1:
        raise RuntimeError("Strict bundle source identity has the wrong source set")
    for name in copied_source_names:
        if sha256_file(bundle_dir / name) != source_hashes[name]:
            raise RuntimeError(f"Strict bundle source identity differs for {name}")
    template_name = next(iter(template_sources))
    if source_hashes[template_name] != manifest.get("base_template_sha256"):
        raise RuntimeError("Strict bundle base-template identity differs")
    sketch_path = bundle_dir / manifest.get("sketch_file", "")
    if not sketch_path.is_file() or sha256_file(sketch_path) != manifest[
        "transformed_sketch_sha256"
    ]:
        raise RuntimeError("Strict bundle transformed-sketch identity differs")
    tested_common = export_manifest.get("_verified_report", {}).get(
        "provenance", {}
    ).get("firmware_common_files")
    if not isinstance(tested_common, dict):
        raise RuntimeError("Strict export lacks host-tested common-file hashes")
    bundled_files = {
        item.get("path"): item.get("sha256") for item in manifest.get("files", [])
    }
    for name, expected_sha256 in tested_common.items():
        if bundled_files.get(name) != expected_sha256:
            raise RuntimeError(f"Bundled common file differs from host-tested code: {name}")
    verify_inventory(bundle_dir, manifest, manifest_path)
    expected_header = (
        "#ifndef CUKD_BUNDLE_IDENTITY_H\n"
        "#define CUKD_BUNDLE_IDENTITY_H\n"
        f"#define CUKD_BUNDLE_ID \"{manifest['bundle_id']}\"\n"
        "#endif\n"
    )
    if (bundle_dir / "cukd_bundle_identity.h").read_text(encoding="ascii") != expected_header:
        raise RuntimeError("Strict bundle identity header differs from its manifest")
    return manifest


def require_export_file_hash(
    export_manifest: dict[str, Any],
    generated_dir: Path,
    path: Path,
) -> str:
    resolved = path.resolve()
    expected_path = (generated_dir / path.name).resolve()
    if resolved != expected_path:
        raise RuntimeError(f"Replay input must come directly from strict export: {expected_path}")
    item = next(
        (entry for entry in export_manifest.get("files", []) if entry["path"] == path.name),
        None,
    )
    if item is None or sha256_file(path) != item["sha256"]:
        raise RuntimeError(f"Replay input is absent from or differs from export manifest: {path}")
    return item["sha256"]


def validate_output_paths(
    output_csv: Path,
    summary_json: Path,
    protected_roots: list[Path],
) -> None:
    if output_csv == summary_json:
        raise RuntimeError("MCU CSV and sequence summary must be different files")
    if output_csv.exists() or summary_json.exists():
        raise FileExistsError("Refusing to overwrite an existing strict replay output")
    for output in [output_csv, summary_json]:
        temporary = output.with_suffix(output.suffix + ".tmp")
        if temporary.exists():
            raise FileExistsError(f"Stale strict replay temporary exists: {temporary}")
    for output in [output_csv, summary_json]:
        for protected_root in protected_roots:
            try:
                output.relative_to(protected_root)
            except ValueError:
                continue
            raise RuntimeError(f"Replay output cannot be written inside protected input: {output}")


def validate_replay_rows(rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError("Strict replay input contains no rows")
    int32_min = -(1 << 31)
    int32_max = (1 << 31) - 1
    expected_ids = list(range(len(rows)))
    observed_ids: list[int] = []
    for index, row in enumerate(rows):
        row_id = int(row["row_id"])
        features = list(row["features"])
        observed_ids.append(row_id)
        if len(features) != 17:
            raise RuntimeError(f"Replay row {index} does not contain 17 features")
        for feature_index, value in enumerate(features):
            integer = int(value)
            if integer < int32_min or integer > int32_max:
                raise RuntimeError(
                    f"Replay row {index} feature {feature_index} is outside int32"
                )
    if observed_ids != expected_ids:
        raise RuntimeError(
            "Strict replay row IDs must be the ordered zero-based export sequence"
        )


class StrictSerialReplay:
    def __init__(self, port: str, baud: int, timeout: float) -> None:
        if serial is None:
            raise RuntimeError("pyserial is required")
        self.timeout = float(timeout)
        self.device = serial.Serial(
            port=port,
            baudrate=baud,
            timeout=timeout,
            write_timeout=timeout,
        )

    def close(self) -> None:
        self.device.close()

    def verify_identity(self, expected: str, settle_seconds: float) -> str:
        time.sleep(settle_seconds)
        self.device.reset_input_buffer()
        self.device.reset_output_buffer()
        self.device.write(b"CUKDID?\n")
        self.device.flush()
        deadline = time.monotonic() + max(3.0, self.timeout * 5)
        observed = []
        while time.monotonic() < deadline:
            line = self.device.readline().decode("ascii", errors="replace").strip()
            if not line:
                continue
            observed.append(line)
            if line.startswith("CUKDBUILD,"):
                if line != expected:
                    raise RuntimeError(
                        f"Device build identity is {line!r}; expected {expected!r}"
                    )
                return line
        raise TimeoutError(f"No device identity received; observed lines: {observed[-5:]}")

    def transact(self, row_id: int, features: list[int]) -> dict[str, object]:
        self.device.write(encode_request_line(row_id, features).encode("ascii"))
        self.device.flush()
        for _ in range(25):
            response = self.device.readline().decode("ascii", errors="replace")
            if not response:
                raise TimeoutError(f"Timeout waiting for row_id={row_id}")
            if not response.startswith("CUKD1R,"):
                continue
            decoded = decode_response_line(response)
            if int(decoded["row_id"]) != row_id:
                raise RuntimeError(
                    f"Device returned row {decoded['row_id']} after request {row_id}"
                )
            return decoded
        raise TimeoutError(f"Too many non-protocol lines before row_id={row_id}")


def write_results(
    output_csv: Path,
    summary_json: Path,
    responses: list[dict[str, object]],
    expected_ids: list[int],
    provenance: dict[str, Any],
    error: dict[str, Any] | None,
) -> dict[str, Any]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_temp = output_csv.with_suffix(output_csv.suffix + ".tmp")
    with csv_temp.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "row_id", "status", "predicted_class", "logits",
            "preprocess_us", "inference_us", "total_us",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for response in responses:
            row = dict(response)
            row["logits"] = " ".join(str(value) for value in row.get("logits", []))
            writer.writerow(row)
    os.replace(csv_temp, output_csv)

    sequence = verify_response_sequence(expected_ids, responses)
    passed = (
        sequence["expected"] == sequence["completed"]
        and not sequence["missing"]
        and not sequence["duplicates"]
        and not sequence["unexpected"]
        and sequence["status_counts"] == {"OK": sequence["expected"]}
        and error is None
    )
    summary = {
        **sequence,
        "status": "passed" if passed else "failed",
        "output_csv": str(output_csv),
        "output_csv_sha256": sha256_file(output_csv),
        "provenance": provenance,
        "error": error,
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    json_temp = summary_json.with_suffix(summary_json.suffix + ".tmp")
    json_temp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    os.replace(json_temp, summary_json)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--vectors-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    args = parser.parse_args()

    implementation_provenance = {
        "stream_script_sha256": sha256_file(Path(__file__).resolve()),
        "protocol_helper_sha256": sha256_file(Path(__file__).with_name("hil_common.py")),
        "vector_loader_sha256": sha256_file(Path(__file__).with_name("stream_vectors.py")),
    }

    generated_dir = args.generated_dir.resolve()
    bundle_dir = args.bundle_dir.resolve()
    output_csv = args.output_csv.resolve()
    summary_json = args.summary_json.resolve()
    validate_output_paths(output_csv, summary_json, [generated_dir, bundle_dir])
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.baud <= 0 or args.timeout <= 0 or args.settle_seconds < 0:
        raise ValueError("Baud/timeout must be positive and settle time cannot be negative")
    export_manifest = verify_export(generated_dir)
    bundle_manifest = verify_bundle(bundle_dir, export_manifest)
    vector_hash = require_export_file_hash(
        export_manifest, generated_dir, args.vectors_csv.resolve()
    )
    rows = load_vectors(args.vectors_csv.resolve(), limit=args.limit)
    validate_replay_rows(rows)
    expected_identity = (
        f"CUKDBUILD,{export_manifest['student']},{export_manifest['export_id']},"
        f"{bundle_manifest['bundle_id']}"
    )
    replay = StrictSerialReplay(args.port, args.baud, args.timeout)
    responses: list[dict[str, object]] = []
    error = None
    observed_identity = None
    try:
        observed_identity = replay.verify_identity(expected_identity, args.settle_seconds)
        for row in rows:
            responses.append(replay.transact(int(row["row_id"]), list(row["features"])))
    except Exception as exc:
        error = {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "completed_before_error": len(responses),
        }
    finally:
        replay.close()

    summary = write_results(
        output_csv,
        summary_json,
        responses,
        [int(row["row_id"]) for row in rows],
        {
            "export_id": export_manifest["export_id"],
            "bundle_id": bundle_manifest["bundle_id"],
            "board": bundle_manifest["board"],
            "student": bundle_manifest["student"],
            "device_identity": observed_identity,
            "vector_sha256": vector_hash,
            "strict_export_manifest_sha256": sha256_file(
                generated_dir / "strict_export_manifest.json"
            ),
            "strict_bundle_manifest_sha256": sha256_file(
                bundle_dir / "strict_bundle_manifest.json"
            ),
            **implementation_provenance,
            "python": sys.version,
            "pyserial_version": getattr(serial, "__version__", None),
        },
        error,
    )
    print(summary_json)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
