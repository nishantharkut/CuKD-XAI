import csv
import json
import inspect
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deployment"))

from experiments.wsnds.leakage_free_rerun.run_tier15_confirmation import (
    EXPECTED_ACTIVE_V1_MANIFEST_SHA256,
    EXPECTED_ACTIVE_V1_SOURCE_SHA256,
    EXPECTED_DEPLOYMENT_RF_CACHE_SHA256,
    RF_CONFIG,
    StudentMLP,
    cpu_state_dict,
    fit_calibrated_rf,
    load_bound_deployment_cache,
    require_disjoint_output,
    set_seed,
    state_dict_sha256,
    verify_resume_root,
)
from experiments.wsnds.leakage_free_rerun.recover_active_v1_results import (
    EXPECTED_CONFIGS,
    require_disjoint_recovery_output,
    validate_metric_result,
)
from experiments.wsnds.leakage_free_rerun.tier15_common import feature_group_split
from firmware_export.wsnds_rfkd_hil.export_train_only_deployment import (
    MAXIMUM_MACRO_F1_DROP,
    MINIMUM_FIXED_FP32_AGREEMENT,
    accumulator_bounds,
    bind_replay_source_rows,
    load_verified_context,
    preprocess_multiply_bounds,
    resolve_manifest_member,
    saturation_audit,
    verify_manifest,
)
from hardware_hil.host.prepare_strict_firmware_bundle import (
    BOARD_TEMPLATES,
    compute_bundle_id,
    strict_sketch,
)
from hardware_hil.host.record_compile_evidence import (
    FLASH_PATTERN,
    RAM_PATTERN,
    parsed_match,
    validate_footprint,
    verify_binary_identity,
)
from hardware_hil.host.generate_strict_report import (
    HIL_PROTOCOL,
    REQUIRED_STAGES,
    atomic_report_set,
    derive_hil_stage,
    sha256_file,
    validate_compile_identity,
    validate_hil_source_evidence,
)
from hardware_hil.host.hil_common import (
    compute_classification_metrics,
    summarize_latency,
)
from hardware_hil.host.stream_vectors_strict import (
    EXPECTED_CORE_EXPORT_FILES,
    STRICT_EXPORT_PROTOCOL,
    canonical_json_sha256,
    validate_output_paths,
    validate_replay_rows,
    verify_export_report,
)


class TrainOnlyConfirmationContractTests(unittest.TestCase):
    def test_export_context_builder_has_no_preconstruction_context_reference(self):
        source = inspect.getsource(load_verified_context)
        self.assertNotIn('context["dataset"]', source)

    def test_feature_group_split_has_no_cross_partition_feature_rows(self):
        base = np.zeros((500, 17), dtype=np.float32)
        base[:, 0] = np.arange(500, dtype=np.float32)
        labels = np.arange(500, dtype=np.int64) % 5
        features = np.concatenate([base, base[:25]], axis=0)
        all_labels = np.concatenate([labels, labels[:25]], axis=0)

        split = feature_group_split(features, all_labels)

        audit = split["group_audit"]
        self.assertEqual(audit["train_validation_feature_overlap"], 0)
        self.assertEqual(audit["train_test_feature_overlap"], 0)
        self.assertEqual(audit["validation_test_feature_overlap"], 0)
        self.assertEqual(
            len(split["train_indices"])
            + len(split["validation_indices"])
            + len(split["test_indices"]),
            len(features),
        )

    def test_fixed_seed_recreates_exact_student_initialization(self):
        for hidden_dims in [(32, 16), (64, 32)]:
            set_seed(42)
            first = StudentMLP(17, hidden_dims, 5)
            set_seed(42)
            second = StudentMLP(17, hidden_dims, 5)
            self.assertEqual(
                state_dict_sha256(cpu_state_dict(first)),
                state_dict_sha256(cpu_state_dict(second)),
            )

    def test_grouped_rf_calibration_has_zero_group_overlap(self):
        features = np.zeros((75, 17), dtype=np.float32)
        features[:, 0] = np.arange(75, dtype=np.float32)
        labels = np.repeat(np.arange(5, dtype=np.int64), 15)
        groups = np.arange(75, dtype=np.int64)
        reduced = {**RF_CONFIG, "n_estimators": 2, "max_depth": 2}
        with patch.dict(RF_CONFIG, reduced, clear=True):
            _, audit = fit_calibrated_rf(features, labels, seed=42, groups=groups)
        self.assertEqual(audit["strategy"], "stratified_group_kfold")
        self.assertEqual(audit["group_overlap_per_fold"], [0, 0, 0])

    def test_accumulator_gate_accepts_safe_and_rejects_unsafe_layers(self):
        safe = [{
            "weight": np.ones((2, 17), dtype=np.int8),
            "bias": np.zeros(2, dtype=np.int32),
            "output_shift": 1,
        }]
        self.assertTrue(accumulator_bounds(safe)[0]["passed"])

        unsafe = [{
            "weight": np.full((1, 1000), 127, dtype=np.int8),
            "bias": np.zeros(1, dtype=np.int32),
            "output_shift": 0,
        }]
        with self.assertRaisesRegex(RuntimeError, "overflow"):
            accumulator_bounds(unsafe)

    def test_preprocess_multiply_gate_accepts_safe_and_rejects_int64_overflow(self):
        safe = {
            "scaler_mean_q": [0] * 17,
            "scaler_inv_scale_q": [1] * 17,
        }
        self.assertTrue(all(item["passed"] for item in preprocess_multiply_bounds(safe)))

        unsafe = {
            "scaler_mean_q": [0] * 17,
            "scaler_inv_scale_q": [np.iinfo(np.int64).max] + [1] * 16,
        }
        with self.assertRaisesRegex(RuntimeError, "int64 preprocessing multiply"):
            preprocess_multiply_bounds(unsafe)

    def test_saturation_audit_covers_parameters_preprocessing_and_activations(self):
        layers = [("net.0", np.asarray([[1.0]], dtype=np.float32), np.asarray([0.0]))]
        quantized = [{
            "weight": np.asarray([[1]], dtype=np.int8),
            "bias": np.asarray([0], dtype=np.int32),
            "weight_frac": 0,
            "accum_frac": 0,
            "output_shift": 0,
        }]
        metadata = {
            "scaler_mean_q": [0],
            "scaler_inv_scale_q": [1],
            "right_shift": 0,
        }
        audit, preprocessed, logits, predictions = saturation_audit(
            layers,
            quantized,
            np.asarray([[1], [-1]], dtype=np.int32),
            metadata,
        )
        self.assertEqual(audit["activation_saturation_count"], 0)
        np.testing.assert_array_equal(preprocessed[:, 0], [1, -1])
        np.testing.assert_array_equal(logits[:, 0], [1, -1])
        np.testing.assert_array_equal(predictions, [0, 0])

        unsafe_quantized = [{**quantized[0], "weight": np.asarray([[127]], dtype=np.int8)}]
        with self.assertRaisesRegex(RuntimeError, "saturation audit"):
            saturation_audit(
                layers,
                unsafe_quantized,
                np.asarray([[32767]], dtype=np.int32),
                metadata,
            )
        nonfinite_layers = [
            ("net.0", np.asarray([[np.nan]], dtype=np.float32), np.asarray([0.0]))
        ]
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            saturation_audit(
                nonfinite_layers,
                quantized,
                np.asarray([[1]], dtype=np.int32),
                metadata,
            )

    def test_export_manifest_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "artifact_manifest.json").write_text(json.dumps({
                "status": "complete",
                "file_count_excluding_manifest": 1,
                "files": [{
                    "path": "../outside.bin",
                    "size_bytes": 0,
                    "sha256": "0" * 64,
                }],
            }), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "escapes"):
                verify_manifest(root)

    def test_completion_artifact_resolution_rejects_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with self.assertRaisesRegex(RuntimeError, "escapes seed root"):
                resolve_manifest_member(root, {"files": []}, "../outside.bin", "0" * 64)

    def test_deployment_cache_rejects_any_file_other_than_preserved_seed_42(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache.npy"
            report = root / "preprocessing.json"
            manifest = root / "manifest.json"
            source = root / "source.py"
            cache.write_bytes(b"not-the-preserved-cache")
            for path in [report, manifest, source]:
                path.write_text("{}", encoding="utf-8")
            self.assertEqual(len(EXPECTED_DEPLOYMENT_RF_CACHE_SHA256), 64)
            with self.assertRaisesRegex(RuntimeError, "not the preserved seed-42"):
                load_bound_deployment_cache({}, cache, report, manifest, source)

    def test_active_v1_bindings_are_full_sha256_values(self):
        self.assertEqual(len(EXPECTED_ACTIVE_V1_SOURCE_SHA256), 64)
        self.assertEqual(len(EXPECTED_ACTIVE_V1_MANIFEST_SHA256), 64)

    def test_output_root_must_be_disjoint_from_sources_and_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            protected = Path(temporary).resolve() / "protected"
            protected.mkdir()
            with self.assertRaisesRegex(RuntimeError, "overlaps protected"):
                require_disjoint_output(protected / "new", [protected])
            with self.assertRaisesRegex(RuntimeError, "overlaps protected"):
                require_disjoint_output(protected.parent, [protected])

    def test_recovery_output_uses_supplied_active_root_as_protected_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            active_root = Path(temporary).resolve() / "custom_active"
            active_root.mkdir()
            with self.assertRaisesRegex(RuntimeError, "overlaps protected"):
                require_disjoint_recovery_output(
                    active_root / "recovery", active_root
                )

    def test_resume_rejects_execution_contract_drift_before_using_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "execution_contract.json").write_text("{}", encoding="utf-8")
            (root / "preprocessing_contract.json").write_text("{}", encoding="utf-8")
            (root / "split_indices.npz").write_bytes(b"x")
            (root / "scaler_parameters.npz").write_bytes(b"x")
            with self.assertRaisesRegex(RuntimeError, "execution contract differs"):
                verify_resume_root(root, {}, {"protocol_id": "expected"})

    def test_strict_bundle_injects_identity_query_once(self):
        for board, path in BOARD_TEMPLATES.items():
            output = strict_sketch(path.read_text(encoding="utf-8"), board)
            self.assertEqual(output.count('#include "cukd_export_identity.h"'), 1)
            self.assertEqual(output.count('#include "cukd_bundle_identity.h"'), 1)
            self.assertEqual(output.count('strcmp(line, "CUKDID?")'), 1)
            self.assertEqual(output.count("CUKDBUILD,"), 1)
            self.assertEqual(output.count("CUKD_STUDENT_ID"), 1)

    def test_bundle_id_changes_when_transformed_sketch_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.c"
            source.write_text("source", encoding="ascii")
            first = compute_bundle_id("esp32c3", "e" * 64, [source], "sketch-a")
            second = compute_bundle_id("esp32c3", "e" * 64, [source], "sketch-b")
            self.assertNotEqual(first, second)

    def test_compile_evidence_parser_uses_real_arduino_lines(self):
        text = (
            "Sketch uses 281192 bytes (21%) of program storage space. "
            "Maximum is 1310720 bytes.\n"
            "Global variables use 13556 bytes (4%) of dynamic memory, leaving "
            "314124 bytes for local variables. Maximum is 327680 bytes.\n"
        )
        self.assertEqual(parsed_match(FLASH_PATTERN, text, "flash")["used"], 281192)
        self.assertEqual(parsed_match(RAM_PATTERN, text, "RAM")["used"], 13556)
        validate_footprint(
            parsed_match(FLASH_PATTERN, text, "flash"),
            parsed_match(RAM_PATTERN, text, "RAM"),
        )

    def test_final_report_rejects_compile_evidence_from_another_bundle(self):
        hil = {
            "provenance": {
                "board": "esp32c3",
                "student": "student_A",
                "export_id": "export-a",
                "bundle_id": "bundle-a",
            }
        }
        compile_evidence = {
            "status": "passed",
            "student": "student_A",
            "board": "esp32c3",
            "export_id": "export-a",
            "bundle_id": "bundle-b",
        }
        with self.assertRaisesRegex(RuntimeError, "bundle_id"):
            validate_compile_identity(
                "esp32c3_student_A", hil, compile_evidence
            )

    def test_replay_outputs_cannot_overwrite_or_enter_protected_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            generated = root / "generated"
            bundle = root / "bundle"
            generated.mkdir()
            bundle.mkdir()
            with self.assertRaisesRegex(RuntimeError, "protected input"):
                validate_output_paths(
                    generated / "out.csv",
                    root / "summary.json",
                    [generated, bundle],
                )
            existing = root / "existing.csv"
            existing.write_text("x", encoding="ascii")
            with self.assertRaises(FileExistsError):
                validate_output_paths(
                    existing,
                    root / "summary.json",
                    [generated, bundle],
                )

    def test_strict_replay_rows_require_order_and_int32_features(self):
        valid = [
            {"row_id": index, "features": [index] * 17}
            for index in range(3)
        ]
        validate_replay_rows(valid)
        with self.assertRaisesRegex(RuntimeError, "ordered zero-based"):
            validate_replay_rows([valid[1], valid[0]])
        invalid = [{"row_id": 0, "features": [0] * 16 + [1 << 31]}]
        with self.assertRaisesRegex(RuntimeError, "outside int32"):
            validate_replay_rows(invalid)

    def test_final_report_set_is_directory_atomic_and_manifested(self):
        with tempfile.TemporaryDirectory() as temporary:
            report_dir = Path(temporary) / "reports"
            outputs = {
                report_dir / "summary.json": "{}\n",
                report_dir / "table.csv": "a\n",
                report_dir / "summary.md": "# report\n",
            }
            atomic_report_set(outputs)
            manifest = json.loads(
                (report_dir / "final_report_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["file_count_excluding_manifest"], 3)

    def test_final_report_set_rejects_reserved_manifest_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            report_dir = Path(temporary) / "reports"
            with self.assertRaisesRegex(RuntimeError, "reserved manifest"):
                atomic_report_set({
                    report_dir / "final_report_manifest.json": "{}\n",
                    report_dir / "table.csv": "a\n",
                    report_dir / "summary.md": "# report\n",
                })

    def test_hil_row_derivation_rejects_logit_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mcu.csv"
            path.write_text(
                "row_id,status,predicted_class,logits,preprocess_us,inference_us,total_us\n"
                '0,OK,0,"1 0 0 0 0",1,2,3\n',
                encoding="utf-8",
            )
            references = {
                0: {
                    "source_row_index": 100,
                    "true_label": 0,
                    "fixed_pred": 0,
                    "fp32_pred": 0,
                    "fixed_logits": [1, 0, 0, 0, 0],
                }
            }
            derive_hil_stage(path, references, 1, "valid")
            path.write_text(
                "row_id,status,predicted_class,logits,preprocess_us,inference_us,total_us\n"
                '0,OK,0,"2 0 0 0 0",1,2,3\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "MCU/reference row mismatch"):
                derive_hil_stage(path, references, 1, "tampered")

    def test_final_report_recomputes_full_hil_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result_root = root / "pair"
            portable_root = root / "compile_artifacts"
            result_root.mkdir()
            portable_root.mkdir()
            reference_path = portable_root / "hil_reference_predictions.csv"
            compile_path = root / "compile.json"
            environment_path = result_root / "host_environment.json"
            environment_path.write_text(json.dumps({
                "timestamp_utc": "2026-08-04T00:00:00+00:00",
                "python": "test-python",
                "platform": "test-platform",
                "serial_ports": ["/dev/test"],
            }), encoding="utf-8")

            with reference_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow([
                    "row_id", "source_row_index", "true_label", "fixed_pred", "fp32_pred",
                    *[f"fixed_logit_{index}" for index in range(5)],
                ])
                for row_id in range(56200):
                    writer.writerow([row_id, row_id + 100, 0, 0, 0, 1, 0, 0, 0, 0])

            base_provenance = {
                "export_id": "e" * 64,
                "bundle_id": "b" * 64,
                "board": "esp32c3",
                "student": "student_A",
                "device_identity": f"CUKDBUILD,student_A,{'e' * 64},{'b' * 64}",
                "vector_sha256": "v" * 64,
                "strict_export_manifest_sha256": "x" * 64,
                "strict_bundle_manifest_sha256": "y" * 64,
                "stream_script_sha256": sha256_file(
                    ROOT / "deployment" / "hardware_hil" / "host" / "stream_vectors_strict.py"
                ),
                "protocol_helper_sha256": sha256_file(
                    ROOT / "deployment" / "hardware_hil" / "host" / "hil_common.py"
                ),
                "vector_loader_sha256": sha256_file(
                    ROOT / "deployment" / "hardware_hil" / "host" / "stream_vectors.py"
                ),
                "pyserial_version": "test",
            }
            stage_metrics = {}
            for stage, count in REQUIRED_STAGES.items():
                mcu_path = result_root / f"{stage}_mcu.csv"
                sequence_path = result_root / f"{stage}_sequence.json"
                metrics_path = result_root / f"{stage}_metrics.json"
                with mcu_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow([
                        "row_id", "status", "predicted_class", "logits",
                        "preprocess_us", "inference_us", "total_us",
                    ])
                    for row_id in range(count):
                        writer.writerow([row_id, "OK", 0, "1 0 0 0 0", 1, 2, 3])
                sequence = {
                    "status": "passed",
                    "error": None,
                    "expected": count,
                    "completed": count,
                    "missing": [],
                    "duplicates": [],
                    "unexpected": [],
                    "status_counts": {"OK": count},
                    "output_csv": str(mcu_path.resolve()),
                    "output_csv_sha256": sha256_file(mcu_path),
                    "provenance": {**base_provenance, "python": "test-python"},
                }
                sequence_path.write_text(json.dumps(sequence), encoding="utf-8")
                labels = [0] * count
                metrics = compute_classification_metrics(labels, labels, range(5))
                metrics.update({
                    "status": "passed",
                    "completed_vectors": count,
                    "mcu_vs_fixed_reference_agreement": 1.0,
                    "mcu_vs_fp32_agreement": 1.0,
                    "exact_logit_agreement": 1.0,
                    "non_ok_status_count": 0,
                    "latency": {
                        "preprocess_us": summarize_latency([1] * count),
                        "inference_us": summarize_latency([2] * count),
                        "total_us": summarize_latency([3] * count),
                    },
                    "provenance": {
                        **base_provenance,
                        "stream_python": "test-python",
                        "mcu_csv_sha256": sha256_file(mcu_path),
                        "sequence_json_sha256": sha256_file(sequence_path),
                        "reference_csv_sha256": sha256_file(reference_path),
                        "verification_script_sha256": sha256_file(
                            ROOT / "deployment" / "hardware_hil" / "host" / "verify_results_strict.py"
                        ),
                        "metric_helper_sha256": sha256_file(
                            ROOT / "deployment" / "hardware_hil" / "host" / "hil_common.py"
                        ),
                    },
                })
                metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
                stage_metrics[stage] = metrics
            inventory = []
            for path in sorted(result_root.iterdir()):
                inventory.append({
                    "path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                })
            (result_root / "strict_hil_completion_manifest.json").write_text(
                json.dumps({
                    "status": "complete",
                    "protocol_id": HIL_PROTOCOL,
                    "required_stages": REQUIRED_STAGES,
                    "run_script_sha256_at_start": sha256_file(
                        ROOT / "deployment" / "hardware_hil" / "scripts" / "run_strict_hil.sh"
                    ),
                    "run_script_sha256_at_completion": sha256_file(
                        ROOT / "deployment" / "hardware_hil" / "scripts" / "run_strict_hil.sh"
                    ),
                    "host_environment_sha256": sha256_file(environment_path),
                    "serial_endpoint_recorded": "/dev/test",
                    "file_count_excluding_manifest": len(inventory),
                    "files": inventory,
                }),
                encoding="utf-8",
            )
            compile_path.write_text("{}", encoding="utf-8")
            compile_evidence = {
                "portable_artifacts": {
                    "hil_reference_predictions": {
                        "path": reference_path.relative_to(root).as_posix(),
                        "size_bytes": reference_path.stat().st_size,
                        "sha256": sha256_file(reference_path),
                    }
                }
            }
            source = validate_hil_source_evidence(
                "esp32c3_student_A",
                result_root / "full_56200_metrics.json",
                stage_metrics["full_56200"],
                compile_path,
                compile_evidence,
            )
            self.assertEqual(
                source["mcu_csv_sha256"],
                sha256_file(result_root / "full_56200_mcu.csv"),
            )

    def test_export_report_rejects_nonfinite_quality_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            core_files = []
            for name in sorted(EXPECTED_CORE_EXPORT_FILES):
                path = root / name
                path.write_bytes(b"")
                core_files.append({
                    "path": name,
                    "size_bytes": 0,
                    "sha256": sha256_file(path),
                })
            provenance = {
                "protocol_id": STRICT_EXPORT_PROTOCOL,
                "student": "student_A",
                "seed": 42,
                "calibration_partition": "train only",
            }
            identity_payload = {"provenance": provenance, "core_files": core_files}
            export_id = canonical_json_sha256(identity_payload)
            report = {
                "status": "passed",
                "export_id": export_id,
                "provenance": provenance,
                "export_identity_payload": identity_payload,
                "gates": {
                    "full_test_rows": 56200,
                    "saved_test_rows_and_labels_exact": True,
                    "saved_fp32_predictions_exact": True,
                    "raw_input_saturation_count": 0,
                    "standardized_input_saturation_count": 0,
                    "minimum_fixed_vs_fp32_agreement": MINIMUM_FIXED_FP32_AGREEMENT,
                    "maximum_macro_f1_drop": MAXIMUM_MACRO_F1_DROP,
                    "fixed_vs_fp32_agreement": float("nan"),
                    "macro_f1_drop": 0.0,
                    "strict_saturation_audit": {
                        "weight_saturation_count": 0,
                        "bias_saturation_count": 0,
                        "integer_preprocess_saturation_count": 0,
                        "activation_saturation_count": 0,
                    },
                    "calibration_partition_saturation_audit": {
                        "rows_audited": 262252,
                        "raw_input_saturation_count": 0,
                        "integer_preprocess_saturation_count": 0,
                        "activation_saturation_count": 0,
                    },
                    "accumulator_bounds": [{
                        "passed": True,
                        "pre_rescale_absolute_bound": 0,
                        "int32_max": int(np.iinfo(np.int32).max),
                    }] * 3,
                    "preprocess_multiply_bounds": [{
                        "passed": True,
                        "maximum_product_absolute": 0,
                        "int64_max": int(np.iinfo(np.int64).max),
                    }] * 17,
                },
                "host_equivalence": {
                    "compile": {"returncode": 0},
                    "self_test": {"returncode": 0},
                },
            }
            (root / "strict_export_report.json").write_text(
                json.dumps(report), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "not finite"):
                verify_export_report(root, {
                    "export_id": export_id,
                    "export_identity_payload_sha256": export_id,
                    "protocol_id": STRICT_EXPORT_PROTOCOL,
                    "student": "student_A",
                })

    def test_replay_binding_rejects_feature_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "replay.csv"
            columns = ["row_id", *[f"f{index}" for index in range(17)]]
            rows = np.zeros((2, 18), dtype=np.int64)
            rows[:, 0] = [0, 1]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(columns)
                writer.writerows(rows.tolist())
            expected = np.zeros((2, 17), dtype=np.int64)
            bind_replay_source_rows(path, np.asarray([10, 11]), expected)
            with path.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(next(csv.DictReader(handle))["source_row_index"], "10")

            drifted = Path(temporary) / "drifted.csv"
            rows[1, 1] = 1
            with drifted.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(columns)
                writer.writerows(rows.tolist())
            with self.assertRaisesRegex(RuntimeError, "features differ"):
                bind_replay_source_rows(drifted, np.asarray([10, 11]), expected)

    def test_recovery_metric_contract_uses_exact_configuration_family(self):
        self.assertEqual(len(EXPECTED_CONFIGS), 14)
        valid = {
            "accuracy": 1.0,
            "macro_precision": 1.0,
            "macro_recall": 1.0,
            "macro_f1": 1.0,
            "per_class_precision": [1.0] * 5,
            "per_class_recall": [1.0] * 5,
            "per_class_f1": [1.0] * 5,
            "confusion_matrix": np.diag([1507, 497, 2189, 51011, 996]).tolist(),
            "params": 1189,
            "model_size_kb": 4.64453125,
        }
        validate_metric_result(
            "D_Small_MLP", valid, Path("checkpoint.json"), student="A"
        )
        with self.assertRaisesRegex(RuntimeError, "outside"):
            validate_metric_result(
                "D_Small_MLP", {**valid, "macro_f1": 1.1}, Path("checkpoint.json")
            )
        corrupted = {
            **valid,
            "confusion_matrix": [row[:] for row in valid["confusion_matrix"]],
        }
        corrupted["confusion_matrix"][0][0] -= 1
        corrupted["confusion_matrix"][0][1] += 1
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            validate_metric_result("D_Small_MLP", corrupted, Path("checkpoint.json"))

    def test_compile_evidence_requires_both_ids_in_firmware_binary(self):
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / "firmware.bin"
            binary.write_bytes(b"prefix-export-a-middle-bundle-a-suffix")
            verify_binary_identity(binary, "export-a", "bundle-a")
            with self.assertRaisesRegex(RuntimeError, "bundle_id"):
                verify_binary_identity(binary, "export-a", "bundle-b")

    def test_training_mode_requires_explicit_confirmation(self):
        script = (
            ROOT
            / "experiments"
            / "wsnds"
            / "leakage_free_rerun"
            / "run_tier15_confirmation.py"
        )
        completed = subprocess.run(
            [sys.executable, str(script), "--mode", "deployment"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("--confirm-training", completed.stderr)


if __name__ == "__main__":
    unittest.main()
