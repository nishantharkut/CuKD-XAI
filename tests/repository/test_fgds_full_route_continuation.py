import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.wsnds.evidence_completion import continue_fgds_full_routes as continuation
from experiments.wsnds.evidence_completion import finalize_fgds_full_routes as finalizer
from experiments.wsnds.evidence_completion import run_fgds_full_routes as runner


def synthetic_probabilities() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(123)
    probabilities = rng.dirichlet(np.ones(5), size=100).astype(np.float32)
    labels = rng.randint(0, 5, size=100).astype(np.int64)
    indices = np.arange(100, dtype=np.int64)
    return probabilities, labels, indices


def metrics_for(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    metrics = runner.classification_metrics(labels, probabilities)
    metrics["ece_15_bin"] = runner.expected_calibration_error(probabilities, labels)
    return metrics


def write_predictions(
    path: Path,
    probabilities: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        source_row_index=indices,
        true_label=labels,
        probability=probabilities.astype(np.float32),
        predicted_label=probabilities.argmax(axis=1).astype(np.int64),
    )


class DtypeFaithfulMetricTests(unittest.TestCase):
    def test_neural_verification_retains_persisted_float32_ece(self):
        probabilities, labels, indices = synthetic_probabilities()
        expected = metrics_for(probabilities, labels)
        delta = abs(
            expected["ece_15_bin"]
            - runner.expected_calibration_error(probabilities.astype(np.float64), labels)
        )
        self.assertGreater(delta, finalizer.NEURAL_ECE_ATOL)

        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "student_A_test_predictions.npz"
            write_predictions(path, probabilities, labels, indices)

            with self.assertRaises(RuntimeError):
                runner.metrics_from_npz_predictions(path, indices, labels, expected)

            verified = finalizer.corrected_metrics_from_npz_predictions(
                path,
                indices,
                labels,
                expected,
            )

        self.assertEqual(verified.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(verified, probabilities)

    def test_uncalibrated_rf_verification_reproduces_executed_roundtrip(self):
        probabilities, labels, indices = synthetic_probabilities()
        executed = probabilities.astype(np.float32).astype(np.float64)
        expected = metrics_for(executed, labels)

        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / finalizer.RF_PREDICTION_NAME
            write_predictions(path, probabilities, labels, indices)
            verified = finalizer.corrected_metrics_from_npz_predictions(
                path,
                indices,
                labels,
                expected,
            )

        self.assertEqual(verified.dtype, np.dtype(np.float64))
        np.testing.assert_array_equal(verified, executed)


class ExactInferenceTests(unittest.TestCase):
    @staticmethod
    def _metric(value):
        return {
            "accuracy": value,
            "macro_precision": value,
            "macro_recall": value,
            "macro_f1": value,
            "ece_15_bin": value / 10.0,
            "per_class_f1": [value] * 5,
        }

    def test_exact_signed_rank_enumerates_all_nonzero_pair_signs(self):
        result = finalizer.exact_signed_rank([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertEqual(result["statistic"], 0.0)
        self.assertEqual(result["p_value_two_sided_exact"], 0.0625)
        self.assertEqual(result["enumerated_sign_assignments"], 32)
        self.assertEqual(result["positive_pairs"], 5)
        self.assertEqual(result["zero_pairs"], 0)

    def test_corrected_paired_test_drops_zero_pairs_and_reports_exact_method(self):
        result = finalizer.corrected_paired_test(
            [1.0, 2.0, 4.0],
            [1.0, 1.0, 2.0],
        )
        wilcoxon = result["wilcoxon"]
        self.assertEqual(wilcoxon["method"], "exact_signed_rank_enumeration")
        self.assertEqual(wilcoxon["zero_difference_count"], 1)
        self.assertEqual(wilcoxon["nonzero_difference_count"], 2)
        self.assertEqual(wilcoxon["enumerated_sign_assignments"], 4)
        self.assertEqual(wilcoxon["p_value_two_sided_exact"], 0.5)
        self.assertEqual(
            wilcoxon["p_value_two_sided"],
            wilcoxon["p_value_two_sided_exact"],
        )

    def test_exact_signed_rank_handles_ties_and_all_zero_pairs(self):
        tied = finalizer.exact_signed_rank([1.0, -1.0, 2.0])
        self.assertEqual(tied["p_value_two_sided_exact"], 0.75)
        self.assertTrue(tied["rank_ties_present"])
        self.assertEqual(tied["enumerated_sign_assignments"], 8)

        all_zero = finalizer.exact_signed_rank([0.0, 0.0, 0.0])
        self.assertEqual(all_zero["statistic"], 0.0)
        self.assertEqual(all_zero["p_value_two_sided_exact"], 1.0)
        self.assertEqual(all_zero["nonzero_pairs"], 0)
        self.assertEqual(all_zero["zero_pairs"], 3)
        self.assertEqual(all_zero["enumerated_sign_assignments"], 1)

    def test_holm_adjustment_is_step_down_and_monotonic(self):
        tests = {
            "first": {"p": 0.01},
            "second": {"p": 0.04},
            "third": {"p": 0.20},
        }
        runner.apply_holm(tests, lambda value: value["p"], "adjusted")
        self.assertAlmostEqual(tests["first"]["adjusted"], 0.03)
        self.assertAlmostEqual(tests["second"]["adjusted"], 0.08)
        self.assertAlmostEqual(tests["third"]["adjusted"], 0.20)

    def test_aggregation_restores_both_overrides_after_failure(self):
        original_verifier = runner.metrics_from_npz_predictions
        original_paired_test = runner.paired_test
        with mock.patch.object(
            runner,
            "aggregate",
            side_effect=RuntimeError("forced aggregation failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "forced aggregation failure"):
                finalizer.aggregate_with_corrections(
                    Path("unused"),
                    [42],
                    "execution-sha",
                    {},
                )
        self.assertIs(runner.metrics_from_npz_predictions, original_verifier)
        self.assertIs(runner.paired_test, original_paired_test)

    def test_exact_inference_reconstructs_vectors_from_seed_evidence(self):
        seeds = [1, 2]
        teacher_routes = ["left_teacher", "right_teacher"]
        student_routes = ["left_student", "right_student"]
        teacher_comparisons = [("left_teacher", "right_teacher")]
        student_comparisons = [("left_student", "right_student")]
        student_specs = {"student": (2, 1)}
        completions = {
            1: {
                "teacher_results": {
                    "left_teacher": {"metrics": self._metric(0.8)},
                    "right_teacher": {"metrics": self._metric(0.7)},
                },
                "student_results": {
                    "student": {
                        "left_student": {"metrics": self._metric(0.6)},
                        "right_student": {"metrics": self._metric(0.5)},
                    }
                },
            },
            2: {
                "teacher_results": {
                    "left_teacher": {"metrics": self._metric(0.9)},
                    "right_teacher": {"metrics": self._metric(0.75)},
                },
                "student_results": {
                    "student": {
                        "left_student": {"metrics": self._metric(0.65)},
                        "right_student": {"metrics": self._metric(0.55)},
                    }
                },
            },
        }
        teacher_aggregate = {
            route: runner.metric_aggregate(
                [completions[seed]["teacher_results"][route]["metrics"] for seed in seeds]
            )
            for route in teacher_routes
        }
        student_aggregate = {
            "student": {
                route: runner.metric_aggregate(
                    [
                        completions[seed]["student_results"]["student"][route]["metrics"]
                        for seed in seeds
                    ]
                )
                for route in student_routes
            }
        }
        tests = {
            "teacher:left_teacher_minus_right_teacher": {
                **finalizer.corrected_paired_test(
                    teacher_aggregate["left_teacher"]["macro_f1"]["values"],
                    teacher_aggregate["right_teacher"]["macro_f1"]["values"],
                ),
                "family": "teacher",
            },
            "student:left_student_minus_right_student": {
                **finalizer.corrected_paired_test(
                    student_aggregate["student"]["left_student"]["macro_f1"]["values"],
                    student_aggregate["student"]["right_student"]["macro_f1"]["values"],
                ),
                "family": "student",
            },
        }
        for family in ["teacher", "student"]:
            family_tests = {
                name: value for name, value in tests.items() if value["family"] == family
            }
            runner.apply_holm(
                family_tests,
                lambda value: value["wilcoxon"]["p_value_two_sided"],
                "holm_adjusted_wilcoxon_within_family_p",
            )
            runner.apply_holm(
                family_tests,
                lambda value: value["exact_sign_flip_mean_difference_p_two_sided"],
                "holm_adjusted_sign_flip_within_family_p",
            )
        runner.apply_holm(
            tests,
            lambda value: value["wilcoxon"]["p_value_two_sided"],
            "holm_adjusted_wilcoxon_global_p",
        )
        runner.apply_holm(
            tests,
            lambda value: value["exact_sign_flip_mean_difference_p_two_sided"],
            "holm_adjusted_sign_flip_global_p",
        )
        aggregate = {
            "protocol_id": runner.PROTOCOL_ID,
            "status": "complete",
            "seeds": seeds,
            "seed_count": len(seeds),
            "teacher_aggregate": teacher_aggregate,
            "student_aggregate": student_aggregate,
            "paired_route_tests": tests,
            "aliases_excluded_from_inference": {},
            "standard_deviation_definition": "sample SD across algorithmic run seeds (ddof=1)",
            "holm_families": ["teacher", "student"],
        }

        with tempfile.TemporaryDirectory() as temporary_dir, mock.patch.multiple(
            runner,
            PUBLICATION_SEEDS=seeds,
            TEACHER_ROUTES=teacher_routes,
            STUDENT_ROUTES=student_routes,
            TEACHER_COMPARISONS=teacher_comparisons,
            STUDENT_COMPARISONS=student_comparisons,
            STUDENT_SPECS=student_specs,
            ALIASES={},
        ):
            root = Path(temporary_dir)
            for seed, completion in completions.items():
                seed_root = root / f"seed_{seed}"
                seed_root.mkdir()
                (seed_root / "seed_completion.json").write_text(
                    json.dumps(completion), encoding="ascii"
                )
            aggregate_path = root / "aggregate_results.json"
            aggregate_path.write_text(json.dumps(aggregate), encoding="ascii")
            audit = finalizer.exact_inference_audit(root)
            self.assertEqual(audit["test_count"], 2)

            aggregate["paired_route_tests"][
                "teacher:left_teacher_minus_right_teacher"
            ]["left_values"][0] += 0.01
            aggregate_path.write_text(json.dumps(aggregate), encoding="ascii")
            with self.assertRaises(finalizer.FinalizationError):
                finalizer.exact_inference_audit(root)

    def test_seed_identity_detects_completion_or_manifest_changes(self):
        with tempfile.TemporaryDirectory() as temporary_dir, mock.patch.object(
            runner,
            "PUBLICATION_SEEDS",
            [42],
        ):
            root = Path(temporary_dir)
            seed_root = root / "seed_42"
            seed_root.mkdir()
            completion = seed_root / "seed_completion.json"
            manifest = seed_root / "artifact_manifest.json"
            completion.write_text('{"status":"complete"}\n', encoding="ascii")
            manifest.write_text('{"status":"complete"}\n', encoding="ascii")
            bound = finalizer.seed_manifest_records(root)
            finalizer.assert_seed_identity(root, bound)

            completion.write_text('{"status":"changed"}\n', encoding="ascii")
            with self.assertRaisesRegex(
                finalizer.FinalizationError,
                "Bound seed evidence changed",
            ):
                finalizer.assert_seed_identity(root, bound)

    def test_finalization_environment_is_bound_and_fail_closed(self):
        expected = {
            "device": "cuda",
            "deterministic_algorithms_enabled": True,
        }
        contract = {"environment": expected}
        cudnn = mock.Mock()
        cudnn.version.return_value = 90100
        cudnn.is_available.return_value = True
        cudnn.deterministic = True
        cudnn.benchmark = False
        with (
            mock.patch.object(runner, "set_seed"),
            mock.patch.object(runner, "environment_record", return_value=expected),
            mock.patch.object(
                runner.torch,
                "is_deterministic_algorithms_warn_only_enabled",
                return_value=False,
            ),
            mock.patch.object(runner.torch.cuda, "current_device", return_value=0),
            mock.patch.object(runner.torch.cuda, "device_count", return_value=2),
            mock.patch.object(
                runner.torch.cuda,
                "get_device_capability",
                return_value=(8, 9),
            ),
            mock.patch.object(runner.torch.backends, "cudnn", cudnn),
            mock.patch.dict(
                os.environ,
                {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
            ),
        ):
            observed = finalizer.validated_finalization_environment(contract)
        self.assertTrue(observed["execution_environment_exact_match"])
        self.assertEqual(
            observed["determinism_controls"]["cuda_device_capability"],
            [8, 9],
        )

        with (
            mock.patch.object(runner, "set_seed"),
            mock.patch.object(
                runner,
                "environment_record",
                return_value={**expected, "device": "cpu"},
            ),
        ):
            with self.assertRaisesRegex(
                finalizer.FinalizationError,
                "differs from execution environment",
            ):
                finalizer.validated_finalization_environment(contract)


class ContinuationSafetyTests(unittest.TestCase):
    def test_runner_guard_uses_lexical_identity_without_resolving_links(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir) / "output"
            protected = output / "seed_8192"
            failed = output / "failed_seed_attempts"
            guard = continuation._RunnerOsGuard(os, protected, failed)
            lexical_source = protected.parent / "." / protected.name
            with self.assertRaisesRegex(
                continuation.ContinuationError,
                "appeared concurrently",
            ):
                guard.replace(lexical_source, failed / "seed_8192_attempt")

    def test_empty_concurrent_seed_refuses_training_and_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir) / "output"
            seed_root = output / "seed_8192"
            seed_root.mkdir(parents=True)
            original_runner_os = runner.os

            with self.assertRaises(FileExistsError):
                continuation.run_seed_without_fallback_archive(
                    output,
                    Path(temporary_dir) / "base",
                    {},
                    8192,
                    None,
                    "execution-sha",
                )

            self.assertIs(runner.os, original_runner_os)
            self.assertTrue(seed_root.is_dir())
            self.assertEqual(list(seed_root.iterdir()), [])

    def test_missing_seed_guard_refuses_runner_fallback_before_training(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir) / "output"
            output.mkdir()
            seed_root = output / "seed_8192"
            seed_root.mkdir()
            partial = seed_root / "external_partial.bin"
            partial.write_bytes(b"external owner")
            original_runner_os = runner.os

            with self.assertRaisesRegex(
                continuation.ContinuationError,
                "appeared concurrently",
            ):
                continuation.run_seed_without_fallback_archive(
                    output,
                    Path(temporary_dir) / "base",
                    {},
                    8192,
                    None,
                    "execution-sha",
                )

            self.assertIs(runner.os, original_runner_os)
            self.assertEqual(partial.read_bytes(), b"external owner")
            self.assertEqual(list((output / "failed_seed_attempts").iterdir()), [])

    def test_finalizer_and_continuation_share_one_lifecycle_lock(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir) / "output"
            output.mkdir()
            lock_path = output.parent / continuation.LOCK_NAME
            with continuation.ExclusiveRunLock(lock_path):
                with self.assertRaisesRegex(
                    finalizer.FinalizationError,
                    "Lifecycle lock failed",
                ):
                    with finalizer.lifecycle_lock(output):
                        pass

    def test_post_evidence_stale_lock_is_preserved_outside_output_root(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary_dir:
            parent = Path(temporary_dir)
            output = parent / "output"
            output.mkdir()
            evidence = parent / "evidence"
            evidence.mkdir()
            (evidence / "continuation_contract.json").write_text(
                "{}\n",
                encoding="ascii",
            )
            (evidence / "artifact_manifest.json").write_text(
                "{}\n",
                encoding="ascii",
            )
            lock_path = parent / continuation.LOCK_NAME
            dead_pid = os.getpid() + 1_000_000
            while continuation.process_exists(dead_pid):
                dead_pid += 1
            lock_path.write_text(
                json.dumps(
                    {
                        "protocol_id": continuation.PROTOCOL_ID,
                        "pid": dead_pid,
                        "owner_token": "2" * 32,
                        "started_at_utc": "2026-08-12T00:00:00+00:00",
                        "continuation_source_sha256": finalizer.sha256_file(
                            continuation.SCRIPT_PATH
                        ),
                    }
                )
                + "\n",
                encoding="ascii",
            )

            preserved = continuation.preserve_post_evidence_stale_lock(
                lock_path,
                output,
                evidence,
            )

            self.assertFalse(lock_path.exists())
            self.assertTrue(preserved.is_dir())
            self.assertFalse(preserved.is_relative_to(output))
            self.assertFalse((output / continuation.POST_EVIDENCE_LOCK_DIR_NAME).exists())
            records = continuation.post_evidence_recovery_records(output, evidence)
            self.assertEqual(len(records), 1)
            self.assertTrue((preserved / "artifact_manifest.json").is_file())
            (output / "artifact_manifest.json").write_text(
                "{}\n",
                encoding="ascii",
            )
            self.assertEqual(
                len(continuation.post_evidence_recovery_records(output, evidence)),
                1,
            )

    def test_exclusive_lock_rejects_second_writer_and_releases(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary_dir:
            lock_path = Path(temporary_dir) / "continuation.lock"
            with continuation.ExclusiveRunLock(lock_path):
                self.assertTrue(lock_path.is_file())
                payload = continuation.validate_lock_payload(lock_path)
                self.assertRegex(payload["owner_token"], r"^[0-9a-f]{32}$")
                with self.assertRaises(continuation.ContinuationError):
                    with continuation.ExclusiveRunLock(lock_path):
                        pass
            self.assertFalse(lock_path.exists())
            with continuation.ExclusiveRunLock(lock_path):
                self.assertTrue(lock_path.is_file())

    def test_lock_is_published_only_from_a_complete_owner_record(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary_dir:
            lock_path = Path(temporary_dir) / "continuation.lock"
            observed = {}
            real_link = os.link

            def inspect_then_link(source, destination):
                observed.update(
                    continuation.validate_lock_payload(Path(source))
                )
                real_link(source, destination)

            with mock.patch.object(
                continuation.os,
                "link",
                side_effect=inspect_then_link,
            ):
                with continuation.ExclusiveRunLock(lock_path):
                    self.assertEqual(
                        continuation.validate_lock_payload(lock_path),
                        observed,
                    )
            self.assertFalse(lock_path.exists())

    def test_lock_cleanup_refuses_to_remove_a_replaced_owner(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary_dir:
            lock_path = Path(temporary_dir) / "continuation.lock"
            lock = continuation.ExclusiveRunLock(lock_path)
            lock.__enter__()
            replacement = dict(continuation.validate_lock_payload(lock_path))
            replacement["owner_token"] = "0" * 32
            lock_path.unlink()
            lock_path.write_text(json.dumps(replacement) + "\n", encoding="ascii")
            with self.assertRaises(continuation.ContinuationError):
                lock.__exit__(None, None, None)
            self.assertTrue(lock_path.exists())

    def test_finalization_state_allows_only_consistent_lifecycle_states(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            self.assertFalse(continuation.output_finalization_state(root, False))

            (root / "aggregate_results.json").write_text("{}\n", encoding="ascii")
            with self.assertRaises(continuation.ContinuationError):
                continuation.output_finalization_state(root, True)
            self.assertFalse(
                continuation.output_finalization_state(
                    root,
                    True,
                    allow_partial_for_lock_recovery=True,
                )
            )

            (root / "artifact_manifest.json").write_text("{}\n", encoding="ascii")
            with self.assertRaises(continuation.ContinuationError):
                continuation.output_finalization_state(root, False)
            self.assertTrue(continuation.output_finalization_state(root, True))

    def test_attempt_contract_preserves_original_seed_boundary_across_restart(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary_dir:
            root = Path(temporary_dir)
            output = root / "output"
            evidence = root / "evidence"
            archive = root / "archive" / "artifact_manifest.json"
            output.mkdir()
            archive.parent.mkdir()
            archive.write_text("{}\n", encoding="ascii")
            seed_42 = output / "seed_42"
            seed_42.mkdir()
            (seed_42 / "seed_completion.json").write_text("{}\n", encoding="ascii")
            (seed_42 / "artifact_manifest.json").write_text("{}\n", encoding="ascii")
            environment = {"device": "cuda"}
            contract = continuation.build_attempt_contract(
                output,
                evidence,
                "execution-sha",
                archive,
                "archive-sha",
                [42],
                environment,
            )
            contract_path = output / continuation.ATTEMPT_CONTRACT_NAME
            contract_path.write_text(
                json.dumps(contract, indent=2) + "\n",
                encoding="ascii",
            )

            seed_123 = output / "seed_123"
            seed_123.mkdir()
            (seed_123 / "seed_completion.json").write_text("{}\n", encoding="ascii")
            (seed_123 / "artifact_manifest.json").write_text("{}\n", encoding="ascii")
            observed = continuation.validate_attempt_contract(
                contract_path,
                output,
                evidence,
                "execution-sha",
                archive,
                "archive-sha",
                environment,
                [42, 123],
            )

        self.assertEqual(observed["completed_seeds_before_continuation"], [42])
        self.assertEqual(observed["target_seeds"][0], 123)

    def test_interrupted_seed_archive_preserves_every_file_and_is_verifiable(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary_dir:
            root = Path(temporary_dir)
            output = root / "output"
            output.mkdir()
            seed_root = output / "seed_8192"
            seed_root.mkdir()
            (seed_root / "partial.pt").write_bytes(b"checkpoint")
            (seed_root / "prediction.npz.tmp").write_bytes(b"partial prediction")
            manifest_temporary = seed_root / (
                continuation.INTERRUPTION_MANIFEST_NAME + ".tmp"
            )
            manifest_temporary.write_bytes(b"interrupted manifest write")

            archived = continuation.archive_incomplete_seed(
                output,
                8192,
                "attempt-sha",
            )
            records = continuation.interrupted_seed_records(
                output,
                "attempt-sha",
            )

            self.assertFalse(seed_root.exists())
            self.assertEqual((archived / "partial.pt").read_bytes(), b"checkpoint")
            self.assertEqual(
                (archived / "prediction.npz.tmp").read_bytes(),
                b"partial prediction",
            )
            manifest = json.loads(
                (archived / continuation.INTERRUPTION_MANIFEST_NAME).read_text(
                    encoding="utf-8"
                )
            )
            declared = {item["path"] for item in manifest["files"]}
            self.assertIn("partial.pt", declared)
            self.assertIn("prediction.npz.tmp", declared)
            preserved_manifest_writes = list(
                archived.glob("interrupted_manifest_write_*.tmp")
            )
            self.assertEqual(len(preserved_manifest_writes), 1)
            self.assertEqual(
                preserved_manifest_writes[0].read_bytes(),
                b"interrupted manifest write",
            )
            self.assertIn(preserved_manifest_writes[0].name, declared)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["seed"], 8192)

    def test_explicit_stale_lock_recovery_preserves_dead_process_lock(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir)
            lock_path = output / "continuation.lock"
            archive_root = output / continuation.STALE_LOCK_DIR_NAME
            dead_pid = os.getpid() + 1_000_000
            while continuation.process_exists(dead_pid):
                dead_pid += 1
            lock_path.write_text(
                json.dumps(
                    {
                        "protocol_id": continuation.PROTOCOL_ID,
                        "pid": dead_pid,
                        "owner_token": "1" * 32,
                        "started_at_utc": "2026-08-12T00:00:00+00:00",
                        "continuation_source_sha256": finalizer.sha256_file(
                            continuation.SCRIPT_PATH
                        ),
                    }
                )
                + "\n",
                encoding="ascii",
            )

            with continuation.ExclusiveRunLock(
                lock_path,
                recover_stale=True,
                stale_archive_root=archive_root,
            ):
                records = continuation.stale_lock_records(output)
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["pid"], dead_pid)
                self.assertTrue(lock_path.is_file())

            self.assertFalse(lock_path.exists())
            self.assertEqual(len(continuation.stale_lock_records(output)), 1)


if __name__ == "__main__":
    unittest.main()
