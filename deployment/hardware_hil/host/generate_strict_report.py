"""Generate the final four-pair HIL table from passed strict verifications only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any


try:
    from .hil_common import compute_classification_metrics, summarize_latency
    from .record_compile_evidence import (
        BOARD_FQBN_PREFIXES,
        FLASH_PATTERN,
        RAM_PATTERN,
        parsed_match,
        validate_footprint,
        verify_binary_identity,
    )
    from .verify_results_strict import read_mcu, read_reference
except ImportError:
    from hil_common import compute_classification_metrics, summarize_latency
    from record_compile_evidence import (
        BOARD_FQBN_PREFIXES,
        FLASH_PATTERN,
        RAM_PATTERN,
        parsed_match,
        validate_footprint,
        verify_binary_identity,
    )
    from verify_results_strict import read_mcu, read_reference


EXPECTED_IDENTITIES = {
    "esp32c3_student_A": {"board": "esp32c3", "student": "student_A"},
    "arduino_r4_student_A": {"board": "arduino_r4", "student": "student_A"},
    "esp32c3_student_B": {"board": "esp32c3", "student": "student_B"},
    "arduino_r4_student_B": {"board": "arduino_r4", "student": "student_B"},
}
EXPECTED_LABELS = set(EXPECTED_IDENTITIES)
DEPLOYMENT_PROTOCOL = "wsnds_archive_split_train_only_scaler_deployment_seed42_v1"
HIL_PROTOCOL = "strict_hil_three_stage_v1"
REQUIRED_STAGES = {"smoke_10": 10, "validation_1000": 1000, "full_56200": 56200}
EXPECTED_CORE_EXPORT_FILES = {
    "model_weights.h", "preprocess_metadata.h", "preprocess_metadata.json",
    "preprocess_int_metadata.h", "preprocess_int_metadata.json", "test_vectors.h",
    "hil_replay_vectors.csv", "hil_reference_predictions.csv", "equivalence_report.json",
}


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("--run must be LABEL=PATH")
    return label, Path(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_portable_artifact(
    compile_path: Path,
    item: dict[str, Any],
) -> Path:
    base = compile_path.parent.resolve()
    path = (base / item["path"]).resolve()
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise RuntimeError(f"Compile artifact escapes evidence directory: {path}") from exc
    if not path.is_file() or sha256_file(path) != item["sha256"]:
        raise RuntimeError(f"Portable compile artifact is missing or changed: {path}")
    if "size_bytes" in item and path.stat().st_size != item["size_bytes"]:
        raise RuntimeError(f"Portable compile artifact size differs: {path}")
    return path


def validate_compile_identity(
    label: str,
    hil_evidence: dict[str, Any],
    compile_evidence: dict[str, Any],
    compile_path: Path | None = None,
) -> dict[str, Any]:
    identity = EXPECTED_IDENTITIES[label]
    student = identity["student"]
    expected = {
        "status": "passed",
        "student": student,
        "board": identity["board"],
        "export_id": hil_evidence["provenance"]["export_id"],
        "bundle_id": hil_evidence["provenance"]["bundle_id"],
    }
    for key, expected_value in expected.items():
        if compile_evidence.get(key) != expected_value:
            raise RuntimeError(
                f"Compile evidence identity mismatch for {label}: "
                f"{key}={compile_evidence.get(key)!r}, expected {expected_value!r}"
            )
    if hil_evidence["provenance"].get("board") != identity["board"]:
        raise RuntimeError(f"HIL board does not match run label {label}")
    if hil_evidence["provenance"].get("student") != student:
        raise RuntimeError(f"HIL student does not match run label {label}")
    for key in ["strict_export_manifest_sha256", "strict_bundle_manifest_sha256"]:
        if hil_evidence["provenance"].get(key) != compile_evidence.get(key):
            raise RuntimeError(f"HIL/compile manifest binding differs for {label}: {key}")
    if compile_path is not None:
        portable = compile_evidence.get("portable_artifacts", {})
        required = {
            "compile_log", "firmware_binary", "strict_export_manifest",
            "strict_export_report", "strict_bundle_manifest",
            "hil_reference_predictions",
        }
        if set(portable) != required:
            raise RuntimeError(f"Compile evidence portable artifact set is incomplete: {label}")
        paths = {
            key: resolve_portable_artifact(compile_path, portable[key])
            for key in required
        }
        if len(set(paths.values())) != len(paths):
            raise RuntimeError(f"Compile portable artifacts are not distinct: {label}")
        if portable["compile_log"]["sha256"] != compile_evidence["compile_log_sha256"]:
            raise RuntimeError(f"Portable compile log hash mismatch: {label}")
        if portable["firmware_binary"]["sha256"] != compile_evidence[
            "firmware_binary_sha256"
        ]:
            raise RuntimeError(f"Portable firmware binary hash mismatch: {label}")
        if portable["firmware_binary"].get("size_bytes") != compile_evidence.get(
            "firmware_binary_size_bytes"
        ):
            raise RuntimeError(f"Portable firmware binary size mismatch: {label}")
        if portable["compile_log"]["sha256"] == portable["firmware_binary"]["sha256"]:
            raise RuntimeError(f"Compile log and firmware binary are identical: {label}")
        export_manifest = json.loads(
            paths["strict_export_manifest"].read_text(encoding="utf-8")
        )
        export_report = json.loads(
            paths["strict_export_report"].read_text(encoding="utf-8")
        )
        bundle_manifest = json.loads(
            paths["strict_bundle_manifest"].read_text(encoding="utf-8")
        )
        expected_export = {
            "status": "passed",
            "student": student,
            "export_id": expected["export_id"],
        }
        expected_bundle = {
            "status": "passed",
            "student": student,
            "board": identity["board"],
            "export_id": expected["export_id"],
            "bundle_id": expected["bundle_id"],
        }
        for key, value in expected_export.items():
            if export_manifest.get(key) != value:
                raise RuntimeError(f"Copied export manifest mismatch for {label}: {key}")
        for key, value in expected_bundle.items():
            if bundle_manifest.get(key) != value:
                raise RuntimeError(f"Copied bundle manifest mismatch for {label}: {key}")
        if bundle_manifest.get("strict_export_manifest_sha256") != compile_evidence[
            "strict_export_manifest_sha256"
        ]:
            raise RuntimeError(f"Copied bundle/export binding mismatch: {label}")
        if paths["strict_export_manifest"].stat().st_size == 0 or sha256_file(
            paths["strict_export_manifest"]
        ) != compile_evidence["strict_export_manifest_sha256"]:
            raise RuntimeError(f"Copied export manifest hash mismatch: {label}")
        if sha256_file(paths["strict_bundle_manifest"]) != compile_evidence[
            "strict_bundle_manifest_sha256"
        ]:
            raise RuntimeError(f"Copied bundle manifest hash mismatch: {label}")
        report_item = next(
            (
                item for item in export_manifest.get("files", [])
                if item.get("path") == "strict_export_report.json"
            ),
            None,
        )
        if report_item is None or report_item.get("sha256") != sha256_file(
            paths["strict_export_report"]
        ):
            raise RuntimeError(f"Copied export report is not bound by its manifest: {label}")
        reference_item = next(
            (
                item for item in export_manifest.get("files", [])
                if item.get("path") == "hil_reference_predictions.csv"
            ),
            None,
        )
        if reference_item is None or reference_item.get("sha256") != sha256_file(
            paths["hil_reference_predictions"]
        ):
            raise RuntimeError(f"Copied HIL reference is not bound by its manifest: {label}")
        manifest_files = {
            item.get("path"): item for item in export_manifest.get("files", [])
        }
        replay_item = manifest_files.get("hil_replay_vectors.csv")
        if (
            replay_item is None
            or replay_item.get("sha256") != hil_evidence.get("provenance", {}).get(
                "vector_sha256"
            )
        ):
            raise RuntimeError(f"HIL replay-vector hash is not bound by export: {label}")
        if export_report.get("status") != "passed" or export_report.get(
            "export_id"
        ) != expected["export_id"]:
            raise RuntimeError(f"Copied strict export report identity mismatch: {label}")
        export_provenance = export_report.get("provenance", {})
        if export_provenance.get("student") != student:
            raise RuntimeError(f"Copied strict export report student mismatch: {label}")
        required_provenance = {
            "protocol_id": DEPLOYMENT_PROTOCOL,
            "student": student,
            "seed": 42,
            "calibration_partition": "train only",
        }
        for key, value in required_provenance.items():
            if export_provenance.get(key) != value:
                raise RuntimeError(f"Copied export provenance mismatch for {label}: {key}")
        for key in [
            "dataset_sha256", "split_hashes", "scaler_sha256", "model_file_sha256",
            "model_artifact_sha256", "execution_contract_sha256",
            "seed_completion_sha256", "feature_overlap_audit",
            "teacher_soft_target_provenance",
        ]:
            if not export_provenance.get(key):
                raise RuntimeError(f"Copied export provenance lacks {key}: {label}")
        export_identity_payload = export_report.get("export_identity_payload")
        if (
            not isinstance(export_identity_payload, dict)
            or export_identity_payload.get("provenance") != export_provenance
            or canonical_json_sha256(export_identity_payload) != expected["export_id"]
            or export_manifest.get("export_identity_payload_sha256") != expected["export_id"]
        ):
            raise RuntimeError(f"Copied strict export identity cannot be rederived: {label}")
        core_files = export_identity_payload.get("core_files", [])
        if not isinstance(core_files, list) or any(
            not isinstance(item, dict) for item in core_files
        ):
            raise RuntimeError(f"Copied strict export core inventory is malformed: {label}")
        if {
            item.get("path") for item in core_files
        } != EXPECTED_CORE_EXPORT_FILES:
            raise RuntimeError(f"Copied strict export core inventory differs: {label}")
        for item in core_files:
            declared = manifest_files.get(item["path"])
            if (
                declared is None
                or declared.get("sha256") != item.get("sha256")
                or declared.get("size_bytes") != item.get("size_bytes")
            ):
                raise RuntimeError(f"Copied export core file is not manifest-bound: {label}")
        bundle_identity_payload = bundle_manifest.get("bundle_identity_payload")
        if (
            not isinstance(bundle_identity_payload, dict)
            or canonical_json_sha256(bundle_identity_payload) != expected["bundle_id"]
            or bundle_identity_payload.get("board") != identity["board"]
            or bundle_identity_payload.get("export_id") != expected["export_id"]
        ):
            raise RuntimeError(f"Copied strict bundle identity cannot be rederived: {label}")
        if (
            bundle_identity_payload.get("bundler_sha256")
            != bundle_manifest.get("bundler_sha256")
            or bundle_identity_payload.get("transformed_sketch_sha256")
            != bundle_manifest.get("transformed_sketch_sha256")
        ):
            raise RuntimeError(f"Copied strict bundle implementation/sketch identity differs: {label}")
        bundle_file_items = {
            item.get("path"): item for item in bundle_manifest.get("files", [])
        }
        source_files = bundle_identity_payload.get("source_files")
        if not isinstance(source_files, list) or any(
            not isinstance(item, dict) for item in source_files
        ):
            raise RuntimeError(f"Copied strict bundle has no source inventory: {label}")
        source_hashes = {item.get("name"): item.get("sha256") for item in source_files}
        if (
            len(source_hashes) != len(source_files)
            or any(
                not isinstance(name, str)
                or not name
                or not isinstance(digest, str)
                or len(digest) != 64
                for name, digest in source_hashes.items()
            )
        ):
            raise RuntimeError(f"Copied strict bundle source inventory is malformed: {label}")
        copied_source_names = {
            "cukd_model.h", "cukd_model.c", "cukd_preprocess.h", "cukd_preprocess.c",
            "cukd_protocol.h", "cukd_protocol.c", "model_weights.h",
            "preprocess_int_metadata.h", "cukd_export_identity.h",
        }
        template_names = set(source_hashes) - copied_source_names
        if set(source_hashes) & copied_source_names != copied_source_names or len(template_names) != 1:
            raise RuntimeError(f"Copied strict bundle source inventory differs: {label}")
        for name in copied_source_names:
            if bundle_file_items.get(name, {}).get("sha256") != source_hashes[name]:
                raise RuntimeError(f"Copied strict bundle source is not manifest-bound: {label}/{name}")
        template_name = next(iter(template_names))
        if source_hashes[template_name] != bundle_manifest.get("base_template_sha256"):
            raise RuntimeError(f"Copied strict bundle template identity differs: {label}")
        sketch_file = bundle_manifest.get("sketch_file")
        if (
            not isinstance(sketch_file, str)
            or not sketch_file
            or compile_evidence.get("sketch_file") != sketch_file
            or bundle_file_items.get(sketch_file, {}).get("sha256")
            != bundle_manifest.get("transformed_sketch_sha256")
        ):
            raise RuntimeError(f"Copied strict bundle sketch binding differs: {label}")
        tested_common = export_report.get("provenance", {}).get("firmware_common_files")
        if not isinstance(tested_common, dict):
            raise RuntimeError(f"Copied export lacks host-tested common-file hashes: {label}")
        bundled_files = {
            item.get("path"): item.get("sha256")
            for item in bundle_manifest.get("files", [])
        }
        for name, expected_sha256 in tested_common.items():
            if bundled_files.get(name) != expected_sha256:
                raise RuntimeError(
                    f"Copied bundle differs from host-tested common file for {label}: {name}"
                )
        gates = export_report.get("gates", {})
        canonical_gates = {
            "full_test_rows": 56200,
            "saved_test_rows_and_labels_exact": True,
            "saved_fp32_predictions_exact": True,
            "raw_input_saturation_count": 0,
            "standardized_input_saturation_count": 0,
            "minimum_fixed_vs_fp32_agreement": 0.99,
            "maximum_macro_f1_drop": 0.01,
        }
        for key, value in canonical_gates.items():
            if gates.get(key) != value:
                raise RuntimeError(f"Copied strict export gate mismatch for {label}: {key}")
        agreement = gates.get("fixed_vs_fp32_agreement")
        macro_f1_drop = gates.get("macro_f1_drop")
        if (
            not isinstance(agreement, (int, float))
            or not math.isfinite(float(agreement))
            or float(agreement) < 0.99
        ):
            raise RuntimeError(f"Copied strict export agreement gate failed: {label}")
        if (
            not isinstance(macro_f1_drop, (int, float))
            or not math.isfinite(float(macro_f1_drop))
            or float(macro_f1_drop) > 0.01
        ):
            raise RuntimeError(f"Copied strict export macro-F1 gate failed: {label}")
        references = read_reference(paths["hil_reference_predictions"])
        if len(references) != 56200 or set(references) != set(range(56200)):
            raise RuntimeError(f"Copied strict reference row contract differs: {label}")
        ordered = [references[index] for index in range(56200)]
        source_row_indices = [item["source_row_index"] for item in ordered]
        if min(source_row_indices) < 0 or len(set(source_row_indices)) != 56200:
            raise RuntimeError(
                f"Copied strict reference source rows are not unique/non-negative: {label}"
            )
        for row_id, item in enumerate(ordered):
            if (
                not 0 <= item["true_label"] < 5
                or not 0 <= item["fp32_pred"] < 5
                or not 0 <= item["fixed_pred"] < 5
                or len(item["fixed_logits"]) != 5
                or any(value < -32768 or value > 32767 for value in item["fixed_logits"])
                or item["fixed_pred"]
                != max(range(5), key=lambda index: item["fixed_logits"][index])
            ):
                raise RuntimeError(f"Copied strict reference row is invalid: {label}/{row_id}")
        y_true = [item["true_label"] for item in ordered]
        fixed_predictions = [item["fixed_pred"] for item in ordered]
        fp32_predictions = [item["fp32_pred"] for item in ordered]
        derived_fixed = compute_classification_metrics(y_true, fixed_predictions, range(5))
        derived_fp32 = compute_classification_metrics(y_true, fp32_predictions, range(5))
        derived_agreement = sum(
            int(left == right) for left, right in zip(fixed_predictions, fp32_predictions)
        ) / len(ordered)
        derived_macro_drop = derived_fp32["macro_f1"] - derived_fixed["macro_f1"]
        for name, observed, expected_value in [
            ("fixed/FP32 agreement", agreement, derived_agreement),
            ("macro-F1 drop", macro_f1_drop, derived_macro_drop),
            (
                "fixed macro-F1",
                export_report.get("fixed_metrics", {}).get("macro_f1"),
                derived_fixed["macro_f1"],
            ),
            (
                "FP32 macro-F1",
                export_report.get("fp32_metrics", {}).get("macro_f1"),
                derived_fp32["macro_f1"],
            ),
        ]:
            if not isinstance(observed, (int, float)) or not math.isclose(
                float(observed), float(expected_value), rel_tol=0.0, abs_tol=1e-12
            ):
                raise RuntimeError(f"Copied export {name} differs from reference rows: {label}")
        saturation = gates.get("strict_saturation_audit", {})
        for key in [
            "weight_saturation_count", "bias_saturation_count",
            "integer_preprocess_saturation_count", "activation_saturation_count",
        ]:
            if saturation.get(key) != 0:
                raise RuntimeError(f"Copied strict export saturation mismatch: {label}/{key}")
        calibration_saturation = gates.get("calibration_partition_saturation_audit", {})
        if calibration_saturation.get("rows_audited") != 262252:
            raise RuntimeError(f"Calibration saturation audit row count differs: {label}")
        for key in [
            "raw_input_saturation_count",
            "integer_preprocess_saturation_count",
            "activation_saturation_count",
        ]:
            if calibration_saturation.get(key) != 0:
                raise RuntimeError(f"Calibration saturation gate failed: {label}/{key}")
        bounds = gates.get("accumulator_bounds")
        if not isinstance(bounds, list) or len(bounds) != 3:
            raise RuntimeError(f"Accumulator-bound audit is incomplete: {label}")
        for item in bounds:
            if (
                item.get("passed") is not True
                or item.get("pre_rescale_absolute_bound", 2**31) > item.get("int32_max", -1)
            ):
                raise RuntimeError(f"Accumulator-bound gate failed: {label}")
        preprocess_bounds = gates.get("preprocess_multiply_bounds")
        if not isinstance(preprocess_bounds, list) or len(preprocess_bounds) != 17:
            raise RuntimeError(f"Preprocessing-multiply audit is incomplete: {label}")
        for item in preprocess_bounds:
            if (
                item.get("passed") is not True
                or item.get("maximum_product_absolute", 2**63)
                > item.get("int64_max", -1)
            ):
                raise RuntimeError(f"Preprocessing-multiply gate failed: {label}")
        host = export_report.get("host_equivalence")
        if (
            not isinstance(host, dict)
            or host.get("compile", {}).get("returncode") != 0
            or host.get("self_test", {}).get("returncode") != 0
        ):
            raise RuntimeError(f"Host fixed-point equivalence did not pass: {label}")
        if not math.isclose(
            float(gates.get("fixed_vs_fp32_agreement", float("nan"))),
            float(hil_evidence.get("mcu_vs_fp32_agreement", float("nan"))),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"Export/HIL FP32 agreement differs: {label}")
        if not math.isclose(
            float(export_report.get("fixed_metrics", {}).get("accuracy", float("nan"))),
            float(hil_evidence.get("accuracy", float("nan"))),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"Export/HIL fixed accuracy differs: {label}")
        if not math.isclose(
            float(export_report.get("fixed_metrics", {}).get("macro_f1", float("nan"))),
            float(hil_evidence.get("macro_f1", float("nan"))),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"Export/HIL fixed macro-F1 differs: {label}")
        verify_binary_identity(
            paths["firmware_binary"], expected["export_id"], expected["bundle_id"]
        )
        compile_text = paths["compile_log"].read_text(
            encoding="utf-8", errors="replace"
        )
        if compile_evidence["sketch_file"] not in compile_text:
            raise RuntimeError(f"Copied compile log lacks strict sketch name: {label}")
        fqbn = compile_evidence.get("fqbn", "")
        if not any(
            fqbn == prefix or fqbn.startswith(prefix + ":")
            for prefix in BOARD_FQBN_PREFIXES[identity["board"]]
        ):
            raise RuntimeError(f"Copied compile FQBN does not match board: {label}")
        for key in ["fqbn", "board_core_version", "toolchain_version"]:
            if compile_evidence[key] not in compile_text:
                raise RuntimeError(f"Copied compile log lacks {key}: {label}")
        flash = parsed_match(FLASH_PATTERN, compile_text, "flash")
        ram = parsed_match(RAM_PATTERN, compile_text, "RAM")
        validate_footprint(flash, ram)
        if flash != compile_evidence["flash"] or ram != compile_evidence["ram"]:
            raise RuntimeError(f"Copied compile footprint differs from evidence: {label}")
        return {
            "student": student,
            "export_provenance": export_provenance,
            "export_gates": gates,
            "reference_csv_sha256": sha256_file(paths["hil_reference_predictions"]),
            "reference_row_contract_sha256": canonical_json_sha256({
                "source_row_indices": source_row_indices,
                "true_labels": y_true,
            }),
        }
    return {"student": student}


def derive_hil_stage(
    mcu_path: Path,
    references: dict[int, dict[str, Any]],
    expected_count: int,
    label: str,
) -> dict[str, Any]:
    rows = read_mcu(mcu_path)
    if [row["row_id"] for row in rows] != list(range(expected_count)):
        raise RuntimeError(f"MCU CSV is not the exact ordered stage prefix: {label}")
    y_true: list[int] = []
    y_mcu: list[int] = []
    y_fp32: list[int] = []
    for row in rows:
        reference = references[row["row_id"]]
        if (
            not 0 <= reference["true_label"] < 5
            or not 0 <= reference["fp32_pred"] < 5
            or len(reference["fixed_logits"]) != 5
            or any(value < -32768 or value > 32767 for value in reference["fixed_logits"])
        ):
            raise RuntimeError(f"Portable reference class/logit is invalid for {label}")
        reference_argmax = max(
            range(5), key=lambda index: reference["fixed_logits"][index]
        )
        if reference["fixed_pred"] != reference_argmax:
            raise RuntimeError(f"Portable reference prediction/logits differ for {label}")
        if (
            row["status"] != "OK"
            or len(row["logits"]) != 5
            or row["predicted_class"] != reference["fixed_pred"]
            or row["logits"] != reference["fixed_logits"]
            or row["predicted_class"]
            != max(range(5), key=lambda index: row["logits"][index])
            or row["preprocess_us"] < 0
            or row["inference_us"] < 0
            or row["total_us"] != row["preprocess_us"] + row["inference_us"]
        ):
            raise RuntimeError(f"MCU/reference row mismatch for {label}: {row['row_id']}")
        y_true.append(reference["true_label"])
        y_mcu.append(row["predicted_class"])
        y_fp32.append(reference["fp32_pred"])
    return {
        "metrics": compute_classification_metrics(y_true, y_mcu, range(5)),
        "mcu_vs_fp32_agreement": sum(
            int(left == right) for left, right in zip(y_mcu, y_fp32)
        ) / len(rows),
        "latency": {
            key: summarize_latency(row[key] for row in rows)
            for key in ["preprocess_us", "inference_us", "total_us"]
        },
    }


def validate_hil_source_evidence(
    label: str,
    metrics_path: Path,
    evidence: dict[str, Any],
    compile_path: Path,
    compile_evidence: dict[str, Any],
) -> dict[str, str]:
    result_root = metrics_path.parent.resolve()
    completion_path = result_root / "strict_hil_completion_manifest.json"
    if not completion_path.is_file():
        raise RuntimeError(f"HIL result directory has no completion manifest: {label}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    files = completion.get("files")
    if (
        completion.get("status") != "complete"
        or completion.get("protocol_id") != HIL_PROTOCOL
        or completion.get("required_stages") != REQUIRED_STAGES
        or not isinstance(files, list)
        or completion.get("file_count_excluding_manifest") != len(files)
    ):
        raise RuntimeError(f"HIL result completion manifest is invalid: {label}")
    declared: set[str] = set()
    for item in files:
        relative = item.get("path")
        if not isinstance(relative, str) or not relative:
            raise RuntimeError(f"HIL completion manifest path is invalid: {label}")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(f"HIL completion manifest path escapes its root: {label}")
        normalized = relative_path.as_posix()
        if normalized in declared:
            raise RuntimeError(f"HIL completion manifest path is duplicated: {label}")
        declared.add(normalized)
        artifact = (result_root / relative_path).resolve()
        try:
            artifact.relative_to(result_root)
        except ValueError as exc:
            raise RuntimeError(f"HIL completion artifact escapes its root: {label}") from exc
        if (
            not artifact.is_file()
            or artifact.stat().st_size != item.get("size_bytes")
            or sha256_file(artifact) != item.get("sha256")
        ):
            raise RuntimeError(f"HIL completion artifact is missing or changed: {label}")
    actual = {
        path.relative_to(result_root).as_posix()
        for path in result_root.iterdir()
        if path.is_file() and path != completion_path
    }
    if actual != declared:
        raise RuntimeError(f"HIL result inventory differs from completion manifest: {label}")
    expected_stage_files = {
        f"{stage}_{suffix}"
        for stage in REQUIRED_STAGES
        for suffix in ["mcu.csv", "sequence.json", "metrics.json"]
    }
    expected_stage_files.add("host_environment.json")
    if declared != expected_stage_files:
        raise RuntimeError(f"HIL result does not contain exactly three required stages: {label}")
    run_script = Path(__file__).resolve().parents[1] / "scripts" / "run_strict_hil.sh"
    current_run_script_sha256 = sha256_file(run_script)
    if (
        completion.get("run_script_sha256_at_start") != current_run_script_sha256
        or completion.get("run_script_sha256_at_completion") != current_run_script_sha256
    ):
        raise RuntimeError(f"HIL completion was produced by a different run script: {label}")
    if not completion.get("serial_endpoint_recorded"):
        raise RuntimeError(f"HIL completion lacks its operator-selected serial endpoint: {label}")
    environment_path = result_root / "host_environment.json"
    if completion.get("host_environment_sha256") != sha256_file(environment_path):
        raise RuntimeError(f"HIL host environment evidence differs: {label}")
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    if (
        not environment.get("timestamp_utc")
        or not environment.get("python")
        or not environment.get("platform")
        or completion["serial_endpoint_recorded"] not in environment.get("serial_ports", [])
    ):
        raise RuntimeError(f"HIL host environment does not bind the selected serial endpoint: {label}")
    portable = compile_evidence["portable_artifacts"]
    reference_path = resolve_portable_artifact(
        compile_path, portable["hil_reference_predictions"]
    )
    references = read_reference(reference_path)
    if len(references) != 56200 or set(references) != set(range(56200)):
        raise RuntimeError(f"Portable HIL reference is not rows 0..56199: {label}")
    source_row_indices = [references[index]["source_row_index"] for index in range(56200)]
    if min(source_row_indices) < 0 or len(set(source_row_indices)) != 56200:
        raise RuntimeError(f"Portable HIL reference source rows are invalid: {label}")
    implementation_hashes = {
        "stream_script_sha256": sha256_file(Path(__file__).with_name("stream_vectors_strict.py")),
        "protocol_helper_sha256": sha256_file(Path(__file__).with_name("hil_common.py")),
        "vector_loader_sha256": sha256_file(Path(__file__).with_name("stream_vectors.py")),
        "verification_script_sha256": sha256_file(Path(__file__).with_name("verify_results_strict.py")),
        "metric_helper_sha256": sha256_file(Path(__file__).with_name("hil_common.py")),
    }
    for required_stage, expected_count in REQUIRED_STAGES.items():
        stage_sequence_path = result_root / f"{required_stage}_sequence.json"
        stage_metrics_path = result_root / f"{required_stage}_metrics.json"
        stage_mcu_path = result_root / f"{required_stage}_mcu.csv"
        stage_sequence = json.loads(stage_sequence_path.read_text(encoding="utf-8"))
        stage_metrics = json.loads(stage_metrics_path.read_text(encoding="utf-8"))
        if (
            stage_sequence.get("status") != "passed"
            or stage_sequence.get("error") is not None
            or stage_sequence.get("expected") != expected_count
            or stage_sequence.get("completed") != expected_count
            or any(stage_sequence.get(key) for key in ["missing", "duplicates", "unexpected"])
            or stage_sequence.get("status_counts") != {"OK": expected_count}
            or stage_sequence.get("output_csv_sha256") != sha256_file(stage_mcu_path)
            or stage_metrics.get("status") != "passed"
            or stage_metrics.get("completed_vectors") != expected_count
            or stage_metrics.get("mcu_vs_fixed_reference_agreement") != 1.0
            or stage_metrics.get("exact_logit_agreement") != 1.0
            or stage_metrics.get("non_ok_status_count") != 0
        ):
            raise RuntimeError(f"Required HIL stage failed validation: {label}/{required_stage}")
        for key in ["export_id", "bundle_id", "board", "student"]:
            if stage_metrics.get("provenance", {}).get(key) != evidence.get(
                "provenance", {}
            ).get(key):
                raise RuntimeError(
                    f"Required HIL stage identity differs: {label}/{required_stage}/{key}"
                )
        if Path(stage_sequence.get("output_csv", "")).name != stage_mcu_path.name:
            raise RuntimeError(f"Required HIL stage records another MCU CSV: {label}/{required_stage}")
        stage_provenance = stage_metrics.get("provenance", {})
        expected_hashes = {
            "mcu_csv_sha256": sha256_file(stage_mcu_path),
            "sequence_json_sha256": sha256_file(stage_sequence_path),
            "reference_csv_sha256": sha256_file(reference_path),
        }
        for key, value in {**expected_hashes, **implementation_hashes}.items():
            if stage_provenance.get(key) != value:
                raise RuntimeError(
                    f"Required HIL stage provenance differs: {label}/{required_stage}/{key}"
                )
        for key in [
            "stream_script_sha256", "protocol_helper_sha256", "vector_loader_sha256"
        ]:
            if stage_sequence.get("provenance", {}).get(key) != implementation_hashes[key]:
                raise RuntimeError(
                    f"Required HIL sequence implementation differs: {label}/{required_stage}/{key}"
                )
        derived = derive_hil_stage(
            stage_mcu_path, references, expected_count, f"{label}/{required_stage}"
        )
        for key, value in derived["metrics"].items():
            if stage_metrics.get(key) != value:
                raise RuntimeError(
                    f"Required HIL stage metric differs from rows: {label}/{required_stage}/{key}"
                )
        if stage_metrics.get("mcu_vs_fp32_agreement") != derived["mcu_vs_fp32_agreement"]:
            raise RuntimeError(f"Required HIL stage FP32 agreement differs: {label}/{required_stage}")
        if stage_metrics.get("latency") != derived["latency"]:
            raise RuntimeError(f"Required HIL stage latency differs: {label}/{required_stage}")

    suffix = "_metrics"
    if not metrics_path.stem.endswith(suffix):
        raise RuntimeError(f"HIL metrics filename does not identify its source stage: {metrics_path}")
    stage = metrics_path.stem[:-len(suffix)]
    if stage != "full_56200":
        raise RuntimeError(f"Final HIL input must be the full_56200 metrics stage: {label}")
    mcu_path = metrics_path.with_name(f"{stage}_mcu.csv")
    sequence_path = metrics_path.with_name(f"{stage}_sequence.json")
    if not mcu_path.is_file() or not sequence_path.is_file():
        raise RuntimeError(f"HIL metrics source CSV/sequence is missing for {label}")

    provenance = evidence.get("provenance", {})
    hashes = {
        "mcu_csv_sha256": sha256_file(mcu_path),
        "sequence_json_sha256": sha256_file(sequence_path),
        "reference_csv_sha256": sha256_file(reference_path),
    }
    for key, value in hashes.items():
        if provenance.get(key) != value:
            raise RuntimeError(f"HIL metrics source hash mismatch for {label}: {key}")

    sequence = json.loads(sequence_path.read_text(encoding="utf-8"))
    if sequence.get("status") != "passed" or sequence.get("error") is not None:
        raise RuntimeError(f"HIL sequence is not passed: {label}")
    if sequence.get("expected") != 56200 or sequence.get("completed") != 56200:
        raise RuntimeError(f"HIL sequence is not the full 56,200 rows: {label}")
    if any(sequence.get(key) for key in ["missing", "duplicates", "unexpected"]):
        raise RuntimeError(f"HIL sequence has row-integrity failures: {label}")
    if sequence.get("status_counts") != {"OK": 56200}:
        raise RuntimeError(f"HIL sequence has non-OK responses: {label}")
    if sequence.get("output_csv_sha256") != hashes["mcu_csv_sha256"]:
        raise RuntimeError(f"HIL sequence does not bind its MCU CSV: {label}")
    if Path(sequence.get("output_csv", "")).name != mcu_path.name:
        raise RuntimeError(f"HIL sequence records a different MCU CSV filename: {label}")
    for key in [
        "export_id", "bundle_id", "board", "student", "device_identity",
        "vector_sha256", "strict_export_manifest_sha256",
        "strict_bundle_manifest_sha256", "stream_script_sha256",
        "protocol_helper_sha256", "vector_loader_sha256", "pyserial_version",
    ]:
        if sequence.get("provenance", {}).get(key) != provenance.get(key):
            raise RuntimeError(f"HIL sequence/metrics provenance differs for {label}: {key}")
    if sequence.get("provenance", {}).get("python") != provenance.get("stream_python"):
        raise RuntimeError(f"HIL sequence/metrics Python provenance differs for {label}")

    rows = read_mcu(mcu_path)
    if [row["row_id"] for row in rows] != list(range(56200)):
        raise RuntimeError(f"MCU CSV is not the exact ordered full-test sequence: {label}")

    y_true: list[int] = []
    y_mcu: list[int] = []
    y_fp32: list[int] = []
    for row in rows:
        reference = references[row["row_id"]]
        if (
            not 0 <= reference["true_label"] < 5
            or not 0 <= reference["fp32_pred"] < 5
            or any(value < -32768 or value > 32767 for value in reference["fixed_logits"])
        ):
            raise RuntimeError(f"Portable reference class/logit is invalid for {label}")
        reference_argmax = max(
            range(5), key=lambda index: reference["fixed_logits"][index]
        )
        if reference["fixed_pred"] != reference_argmax:
            raise RuntimeError(f"Portable reference prediction/logits differ for {label}")
        if (
            row["status"] != "OK"
            or len(row["logits"]) != 5
            or row["predicted_class"] != reference["fixed_pred"]
            or row["logits"] != reference["fixed_logits"]
            or row["predicted_class"]
            != max(range(5), key=lambda index: row["logits"][index])
            or row["preprocess_us"] < 0
            or row["inference_us"] < 0
            or row["total_us"] != row["preprocess_us"] + row["inference_us"]
        ):
            raise RuntimeError(f"MCU/reference row mismatch for {label}: {row['row_id']}")
        y_true.append(reference["true_label"])
        y_mcu.append(row["predicted_class"])
        y_fp32.append(reference["fp32_pred"])

    expected_metrics = compute_classification_metrics(y_true, y_mcu, range(5))
    for key, value in expected_metrics.items():
        if evidence.get(key) != value:
            raise RuntimeError(f"HIL derived metric differs from source rows for {label}: {key}")
    expected_fp32_agreement = sum(
        int(left == right) for left, right in zip(y_mcu, y_fp32)
    ) / len(rows)
    if evidence.get("mcu_vs_fp32_agreement") != expected_fp32_agreement:
        raise RuntimeError(f"HIL FP32 agreement differs from source rows for {label}")
    expected_latency = {
        key: summarize_latency(row[key] for row in rows)
        for key in ["preprocess_us", "inference_us", "total_us"]
    }
    if evidence.get("latency") != expected_latency:
        raise RuntimeError(f"HIL latency summary differs from source rows for {label}")
    return {
        "mcu_csv_path_recorded": str(mcu_path.resolve()),
        "mcu_csv_sha256": hashes["mcu_csv_sha256"],
        "sequence_json_path_recorded": str(sequence_path.resolve()),
        "sequence_json_sha256": hashes["sequence_json_sha256"],
        "portable_reference_sha256": hashes["reference_csv_sha256"],
        "result_completion_manifest_sha256": sha256_file(completion_path),
    }


def atomic_report_set(payloads: dict[Path, str]) -> None:
    if len(payloads) != 3:
        raise RuntimeError("Final report requires three distinct output files")
    if any(path.name == "final_report_manifest.json" for path in payloads):
        raise RuntimeError("A report payload cannot use the reserved manifest filename")
    parents = {path.parent.resolve() for path in payloads}
    if len(parents) != 1:
        raise RuntimeError("Final report outputs must share one dedicated directory")
    final_dir = parents.pop()
    if final_dir.exists():
        raise FileExistsError(f"Refusing to overwrite report directory: {final_dir}")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = final_dir.parent / f".{final_dir.name}.tmp"
    if temporary_dir.exists():
        raise FileExistsError(f"Stale temporary report directory exists: {temporary_dir}")
    temporary_dir.mkdir()
    try:
        inventory = []
        for path, payload in payloads.items():
            if path.parent.resolve() != final_dir:
                raise RuntimeError(f"Report output escapes its dedicated directory: {path}")
            temporary = temporary_dir / path.name
            temporary.write_text(payload, encoding="utf-8")
            inventory.append({
                "path": path.name,
                "size_bytes": temporary.stat().st_size,
                "sha256": sha256_file(temporary),
            })
        manifest = temporary_dir / "final_report_manifest.json"
        manifest.write_text(
            json.dumps({
                "status": "complete",
                "file_count_excluding_manifest": len(inventory),
                "files": sorted(inventory, key=lambda item: item["path"]),
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_dir, final_dir)
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--compile", action="append", type=parse_run, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    supplied = {label for label, _ in args.run}
    if supplied != EXPECTED_LABELS or len(args.run) != 4:
        raise RuntimeError(f"Expected exactly these four run labels: {sorted(EXPECTED_LABELS)}")
    supplied_compile = {label for label, _ in args.compile}
    if supplied_compile != EXPECTED_LABELS or len(args.compile) != 4:
        raise RuntimeError(
            f"Expected compile evidence for exactly these four labels: "
            f"{sorted(EXPECTED_LABELS)}"
        )
    run_paths = {label: path.resolve() for label, path in args.run}
    compile_paths = {label: path.resolve() for label, path in args.compile}
    if len(set(run_paths.values())) != 4:
        raise RuntimeError("Each board/student label must use a distinct HIL metrics file")
    if len(set(compile_paths.values())) != 4:
        raise RuntimeError("Each board/student label must use distinct compile evidence")
    rows: list[dict[str, Any]] = []
    export_ids: dict[str, set[str]] = {"student_A": set(), "student_B": set()}
    bundle_ids: set[str] = set()
    metric_hashes: set[str] = set()
    compile_hashes: set[str] = set()
    mcu_csv_hashes: set[str] = set()
    per_student_vector_hashes: dict[str, set[str]] = {"student_A": set(), "student_B": set()}
    per_student_reference_hashes: dict[str, set[str]] = {"student_A": set(), "student_B": set()}
    reference_row_contracts: set[str] = set()
    common_training_contracts: set[str] = set()
    per_board_compile_contracts: dict[str, set[tuple[Any, ...]]] = {
        "esp32c3": set(),
        "arduino_r4": set(),
    }
    for label in sorted(EXPECTED_LABELS):
        path = run_paths[label]
        evidence = json.loads(path.read_text(encoding="utf-8"))
        if evidence.get("status") != "passed":
            raise RuntimeError(f"HIL verification is not passed: {path}")
        if evidence.get("completed_vectors") != 56200:
            raise RuntimeError(f"HIL verification is not full-test: {path}")
        if evidence.get("mcu_vs_fixed_reference_agreement") != 1.0:
            raise RuntimeError(f"Prediction agreement is not exact: {path}")
        if evidence.get("exact_logit_agreement") != 1.0:
            raise RuntimeError(f"Logit agreement is not exact: {path}")
        if evidence.get("non_ok_status_count") != 0:
            raise RuntimeError(f"Non-OK response exists: {path}")
        agreement = evidence.get("mcu_vs_fp32_agreement")
        if (
            not isinstance(agreement, (int, float))
            or not math.isfinite(float(agreement))
            or float(agreement) < 0.99
        ):
            raise RuntimeError(f"MCU/FP32 agreement is below the strict gate: {path}")
        compile_path = compile_paths[label]
        compile_evidence = json.loads(compile_path.read_text(encoding="utf-8"))
        validated_compile = validate_compile_identity(
            label, evidence, compile_evidence, compile_path
        )
        student = validated_compile["student"]
        source_evidence = validate_hil_source_evidence(
            label, path, evidence, compile_path, compile_evidence
        )
        metrics_sha256 = sha256_file(path)
        compile_sha256 = sha256_file(compile_path)
        if metrics_sha256 in metric_hashes:
            raise RuntimeError("Two labels use identical HIL metrics evidence")
        if compile_sha256 in compile_hashes:
            raise RuntimeError("Two labels use identical compile evidence")
        metric_hashes.add(metrics_sha256)
        compile_hashes.add(compile_sha256)
        if source_evidence["mcu_csv_sha256"] in mcu_csv_hashes:
            raise RuntimeError("Two board/model labels use identical raw MCU result CSVs")
        mcu_csv_hashes.add(source_evidence["mcu_csv_sha256"])
        bundle_id = evidence["provenance"]["bundle_id"]
        if bundle_id in bundle_ids:
            raise RuntimeError("Two labels use the same strict firmware bundle")
        bundle_ids.add(bundle_id)
        export_ids[student].add(evidence["provenance"]["export_id"])
        per_student_vector_hashes[student].add(evidence["provenance"]["vector_sha256"])
        per_student_reference_hashes[student].add(
            validated_compile["reference_csv_sha256"]
        )
        reference_row_contracts.add(validated_compile["reference_row_contract_sha256"])
        export_provenance = validated_compile["export_provenance"]
        common_training_contracts.add(json.dumps({
            "protocol_id": export_provenance["protocol_id"],
            "seed": export_provenance["seed"],
            "dataset_sha256": export_provenance["dataset_sha256"],
            "split_hashes": export_provenance["split_hashes"],
            "scaler_sha256": export_provenance["scaler_sha256"],
            "execution_contract_sha256": export_provenance["execution_contract_sha256"],
            "seed_completion_sha256": export_provenance["seed_completion_sha256"],
            "calibration_partition": export_provenance["calibration_partition"],
            "feature_overlap_audit": export_provenance["feature_overlap_audit"],
            "teacher_soft_target_provenance": export_provenance[
                "teacher_soft_target_provenance"
            ],
        }, sort_keys=True, separators=(",", ":")))
        board = evidence["provenance"]["board"]
        per_board_compile_contracts[board].add((
            compile_evidence["fqbn"],
            compile_evidence["board_core_version"],
            compile_evidence["frontend_version"],
            compile_evidence["toolchain_version"],
            compile_evidence["flash"]["maximum"],
            compile_evidence["ram"]["maximum"],
        ))
        rows.append({
            "run": label,
            "board": evidence["provenance"]["board"],
            "student": student,
            "vectors": evidence["completed_vectors"],
            "mcu_vs_fixed": evidence["mcu_vs_fixed_reference_agreement"],
            "exact_logit_agreement": evidence["exact_logit_agreement"],
            "mcu_vs_fp32": evidence["mcu_vs_fp32_agreement"],
            "accuracy": evidence["accuracy"],
            "macro_f1": evidence["macro_f1"],
            "preprocess_mean_us": evidence["latency"]["preprocess_us"]["mean"],
            "inference_mean_us": evidence["latency"]["inference_us"]["mean"],
            "compute_total_mean_us": evidence["latency"]["total_us"]["mean"],
            "compute_total_p95_us": evidence["latency"]["total_us"]["p95"],
            "compute_total_p99_us": evidence["latency"]["total_us"]["p99"],
            "flash_used_bytes": compile_evidence["flash"]["used"],
            "flash_maximum_bytes": compile_evidence["flash"]["maximum"],
            "flash_percent_reported": compile_evidence["flash"]["percent"],
            "ram_used_bytes": compile_evidence["ram"]["used"],
            "ram_maximum_bytes": compile_evidence["ram"]["maximum"],
            "ram_percent_reported": compile_evidence["ram"]["percent"],
            "firmware_binary_size_bytes": compile_evidence["firmware_binary_size_bytes"],
            "fqbn": compile_evidence["fqbn"],
            "board_core_version": compile_evidence["board_core_version"],
            "frontend_version": compile_evidence["frontend_version"],
            "toolchain_version": compile_evidence["toolchain_version"],
            "export_id": evidence["provenance"]["export_id"],
            "bundle_id": bundle_id,
            "metrics_path_recorded": str(path),
            "metrics_sha256": metrics_sha256,
            "compile_evidence_path_recorded": str(compile_path),
            "compile_evidence_sha256": compile_sha256,
            **source_evidence,
        })
    if any(len(ids) != 1 for ids in export_ids.values()):
        raise RuntimeError("The two boards for one student do not share one export ID")
    if export_ids["student_A"] == export_ids["student_B"]:
        raise RuntimeError("Student A and Student B must use distinct strict exports")
    if any(len(values) != 1 for values in per_student_vector_hashes.values()):
        raise RuntimeError("The two boards for one student do not share one replay-vector hash")
    if any(len(values) != 1 for values in per_student_reference_hashes.values()):
        raise RuntimeError("The two boards for one student do not share one reference hash")
    if len(reference_row_contracts) != 1:
        raise RuntimeError("Student A/B references do not use identical test rows and labels")
    if len(common_training_contracts) != 1:
        raise RuntimeError("Student A/B exports do not share one training/data/teacher contract")
    if any(len(values) != 1 for values in per_board_compile_contracts.values()):
        raise RuntimeError("A board was compiled with different board/toolchain settings by student")

    rows.sort(key=lambda row: row["run"])
    json_payload = {
        "status": "passed",
        "run_count": 4,
        "full_test_board_predictions": 4 * 56200,
        "all_stage_board_inferences": 4 * sum(REQUIRED_STAGES.values()),
        "claim_boundary": (
            "Operator-selected serial-endpoint replay of extracted WSN-DS feature "
            "records with firmware identity verification; no cryptographic board attestation, "
            "live packet capture, on-board feature extraction, energy measurement, or "
            "TelosB execution. The archive-compatible random-row split contains exact "
            "feature groups that cross partitions, so these results establish execution "
            "fidelity rather than duplicate-free generalization."
        ),
        "compile_evidence_boundary": (
            "Flash and RAM values are Arduino frontend reports associated with a "
            "binary that embeds the strict export and bundle IDs. The sketch/log/"
            "binary association detects accidental mixing but is not cryptographic "
            "Arduino-build attestation. FQBN, board-core, and toolchain strings are "
            "checked in the verbose log; the frontend version is operator-recorded. "
            "These values are not measured energy or runtime heap usage."
        ),
        "latency_boundary": (
            "Board latency is preprocessing plus inference compute time measured by "
            "firmware micros(); it excludes USB serial transport and host overhead."
        ),
        "runs": rows,
    }
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

    lines = [
        "# Train-only-scaler WSN-DS HIL summary",
        "",
        "| Run | Vectors | MCU/fixed | Exact logits | MCU/FP32 | Accuracy | Macro-F1 | Mean compute (us) | Compute p95 (us) | Compute p99 (us) | Flash bytes | RAM bytes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run']} | {row['vectors']} | {row['mcu_vs_fixed']:.6f} | "
            f"{row['exact_logit_agreement']:.6f} | {row['mcu_vs_fp32']:.6f} | "
            f"{row['accuracy']:.6f} | {row['macro_f1']:.6f} | "
            f"{row['compute_total_mean_us']:.3f} | {row['compute_total_p95_us']} | "
            f"{row['compute_total_p99_us']} | {row['flash_used_bytes']} | "
            f"{row['ram_used_bytes']} |"
        )
    lines.extend([
        "",
        "## Claim boundary",
        "",
        json_payload["claim_boundary"],
        "",
        json_payload["compile_evidence_boundary"],
        "",
        json_payload["latency_boundary"],
        "",
    ])
    output_json = args.output_json.resolve()
    output_csv = args.output_csv.resolve()
    output_md = args.output_md.resolve()
    if len({output_json, output_csv, output_md}) != 3:
        raise RuntimeError("Final JSON, CSV, and Markdown outputs must be distinct")
    if any(path.name == "final_report_manifest.json" for path in [
        output_json, output_csv, output_md
    ]):
        raise RuntimeError("A report payload cannot use the reserved manifest filename")
    atomic_report_set({
        output_json: json.dumps(json_payload, indent=2) + "\n",
        output_csv: csv_buffer.getvalue(),
        output_md: "\n".join(lines),
    })
    print(output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
