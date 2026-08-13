import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.wsnds.evidence_completion import run_fgds_controlled_xai_transfer as xai


class ControlledXAIContractTests(unittest.TestCase):
    def test_frozen_protocol_constants(self):
        self.assertEqual(xai.PROTOCOL_ID, "wsnds_fgds_controlled_xai_transfer_10seed_v1")
        self.assertEqual(xai.EXPECTED_SEEDS, [42, 123, 456, 789, 1001, 2024, 3141, 5678, 8192, 9999])
        self.assertEqual(xai.COHORT_SIZE, 250)
        self.assertEqual(xai.BACKGROUND_SIZE, 20)
        self.assertEqual(xai.KD_TEMPERATURE, 4.0)
        self.assertEqual(xai.PERMUTATION_REPEATS, 5)
        self.assertEqual(xai.ATTRIBUTION_NORM_EPSILON, 1e-6)
        self.assertEqual(xai.MIN_ELIGIBLE_PER_CLASS, 40)
        self.assertEqual(xai.RANDOMIZATION_SEEDS, {"student_A": 5042, "student_B": 6042})

    def test_temperature_softening_matches_power_transform(self):
        probabilities = np.array(
            [[0.70, 0.20, 0.05, 0.03, 0.02], [0.1, 0.1, 0.1, 0.1, 0.6]],
            dtype=np.float64,
        )
        observed = xai.soften_probabilities(probabilities)
        expected = probabilities ** 0.25
        expected /= expected.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-15)

    def test_student_source_replay_and_shap_temperatures_are_separate(self):
        model = xai.StudentMLP(17, (32, 16), 5).eval()
        values = np.arange(34, dtype=np.float32).reshape(2, 17) / 20.0
        with xai.torch.no_grad():
            logits = model(xai.torch.from_numpy(values))
            expected_t1 = xai.F.softmax(logits, dim=1).numpy()
            expected_t4 = xai.F.softmax(logits / 4.0, dim=1).numpy()
        observed_t1 = xai.student_predictor(model, temperature=1.0)(values)
        observed_t4 = xai.student_predictor(model)(values)
        np.testing.assert_allclose(observed_t1, expected_t1, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(observed_t4, expected_t4, rtol=0.0, atol=0.0)
        self.assertFalse(np.array_equal(observed_t1, observed_t4))

    def test_group_sampler_excludes_mixed_groups_and_uses_lowest_source_row(self):
        features = []
        labels = []
        source = []
        row = 100
        for class_index in range(5):
            for group_index in range(3):
                value = np.full(17, class_index * 10 + group_index + 1, dtype=np.float64)
                features.extend([value, value])
                labels.extend([class_index, class_index])
                source.extend([row + 9, row])
                row += 20
        mixed = np.full(17, 999.0, dtype=np.float64)
        features.extend([mixed, mixed])
        labels.extend([0, 1])
        source.extend([9991, 9990])
        features_array = np.asarray(features)
        labels_array = np.asarray(labels, dtype=np.int64)
        source_array = np.asarray(source, dtype=np.int64)
        result = xai.select_balanced_group_representatives(
            features_array,
            labels_array,
            source_array,
            groups_per_class=2,
            seed=2042,
            global_purity=xai.build_global_group_purity(features_array, labels_array),
        )
        np.testing.assert_array_equal(result["labels"], np.tile(np.arange(5), 2))
        self.assertEqual(len(np.unique(result["exact_feature_group_hashes"])), 10)
        self.assertNotIn(9990, result["source_row_indices"].tolist())
        for partition_index in result["partition_indices"]:
            same = np.all(features_array == features_array[partition_index], axis=1)
            self.assertEqual(source_array[partition_index], source_array[same].min())

    def test_group_sampler_enforces_purity_over_rows_outside_partition(self):
        global_features = []
        global_labels = []
        partition_features = []
        partition_labels = []
        partition_sources = []
        source_row = 100
        for class_index in range(5):
            for group_index in range(3):
                value = np.full(17, 100 * class_index + group_index + 1, dtype=np.float64)
                global_features.append(value)
                global_labels.append(class_index)
                partition_features.append(value)
                partition_labels.append(class_index)
                partition_sources.append(source_row)
                source_row += 1
        globally_mixed = np.full(17, 9999.0, dtype=np.float64)
        global_features.extend([globally_mixed, globally_mixed])
        global_labels.extend([0, 1])
        partition_features.append(globally_mixed)
        partition_labels.append(0)
        partition_sources.append(9999)
        result = xai.select_balanced_group_representatives(
            np.asarray(partition_features),
            np.asarray(partition_labels, dtype=np.int64),
            np.asarray(partition_sources, dtype=np.int64),
            groups_per_class=3,
            seed=2042,
            global_purity=xai.build_global_group_purity(
                np.asarray(global_features), np.asarray(global_labels, dtype=np.int64)
            ),
        )
        self.assertNotIn(9999, result["source_row_indices"].tolist())

    def test_rf_selected_output_extracts_one_feature_vector_per_row(self):
        values = np.arange(3 * 17 * 5, dtype=np.float64).reshape(3, 17, 5)
        old_size = xai.COHORT_SIZE
        try:
            xai.COHORT_SIZE = 3
            selected = xai.selected_attributions(values, np.array([0, 2, 4]))
        finally:
            xai.COHORT_SIZE = old_size
        expected = np.stack([values[0, :, 0], values[1, :, 2], values[2, :, 4]])
        np.testing.assert_array_equal(selected, expected)

    def test_cosine_marks_near_zero_rows_undefined_without_zero_imputation(self):
        left = np.array([[1.0, 0.0], [1e-8, 0.0], [1.0, 0.0]])
        right = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        values, eligible = xai.cosine_rows(left, right)
        np.testing.assert_array_equal(eligible, [True, False, True])
        self.assertEqual(values[0], 1.0)
        self.assertTrue(np.isnan(values[1]))
        self.assertEqual(values[2], 0.0)

    def test_primary_pair_metric_is_macro_class_kd_minus_scratch(self):
        labels = np.tile(np.arange(5, dtype=np.int64), 50)
        teacher_classes = np.zeros(250, dtype=np.int64)
        teacher = np.zeros((250, 17, 5), dtype=np.float64)
        scratch = np.zeros_like(teacher)
        kd = np.zeros_like(teacher)
        teacher[:, 0, 0] = 1.0
        scratch[:, 1, 0] = 1.0
        kd[:, 0, 0] = 1.0
        result = xai.pair_metrics(teacher, scratch, kd, teacher_classes, labels)
        self.assertEqual(result["primary"]["status"], "conclusive")
        self.assertEqual(result["eligibility"]["eligible_rows"], 250)
        self.assertAlmostEqual(result["primary"]["scratch_alignment_macro_class_mean"], 0.0)
        self.assertAlmostEqual(result["primary"]["rf_kd_alignment_macro_class_mean"], 1.0)
        self.assertAlmostEqual(result["primary"]["rf_kd_minus_scratch_alignment_gain"], 1.0)
        self.assertTrue(all(row["eligible_rows"] == 50 for row in result["per_class"]))

    def test_primary_pair_metric_persists_inconclusive_class_gate(self):
        labels = np.tile(np.arange(5, dtype=np.int64), 50)
        teacher_classes = np.zeros(250, dtype=np.int64)
        teacher = np.zeros((250, 17, 5), dtype=np.float64)
        scratch = np.zeros_like(teacher)
        kd = np.zeros_like(teacher)
        teacher[:, 0, 0] = 1.0
        scratch[:, 1, 0] = 1.0
        kd[:, 0, 0] = 1.0
        teacher[labels == 4, 0, 0] = 0.0
        result = xai.pair_metrics(teacher, scratch, kd, teacher_classes, labels)
        self.assertEqual(result["primary"]["status"], "inconclusive")
        self.assertEqual(result["per_class"][4]["eligible_rows"], 0)
        self.assertIn("minimum is 40", result["primary"]["inconclusive_reasons"][0])
        self.assertEqual(result["primary"]["descriptive_defined_class_count"], 4)
        self.assertAlmostEqual(result["primary"]["rf_kd_minus_scratch_alignment_gain"], 1.0)

    def test_exact_signed_rank_enumerates_all_sign_assignments(self):
        result = xai.exact_paired_signed_rank(np.arange(1.0, 11.0))
        self.assertEqual(result["p_value_two_sided"], 2.0 / 1024.0)
        self.assertEqual(result["enumerated_sign_assignments"], 1024)
        self.assertEqual(result["rank_tie_method"], "average")
        self.assertEqual(result["zero_method"], "wilcox")
        with_zero = xai.exact_paired_signed_rank(np.array([0.0, 1.0, -1.0]))
        self.assertEqual(with_zero["zero_difference_count"], 1)
        self.assertEqual(with_zero["enumerated_sign_assignments"], 4)

    def test_holm_adjustment_is_monotone(self):
        adjusted = xai.holm_adjust({"student_A": 0.01, "student_B": 0.03})
        self.assertEqual(adjusted, {"student_A": 0.02, "student_B": 0.03})

    def test_aggregate_uses_seed_as_unit_and_holm_family_of_two(self):
        completions = []
        for index, seed in enumerate(xai.EXPECTED_SEEDS):
            students = {}
            for student_key in xai.STUDENT_KEYS:
                gain = 0.1 + index * 0.001 if student_key == "student_A" else 0.05 + index * 0.001
                students[student_key] = {
                    "primary": {
                        "status": "conclusive",
                        "inconclusive_reasons": [],
                        "descriptive_defined_class_count": 5,
                        "scratch_alignment_macro_class_mean": 0.2,
                        "rf_kd_alignment_macro_class_mean": 0.2 + gain,
                        "rf_kd_minus_scratch_alignment_gain": gain,
                    },
                    "eligibility": {"eligible_rows": 250, "undefined_rows": 0},
                    "secondary": {"placeholder": float(index)},
                }
            completions.append(
                {
                    "seed": seed,
                    "metrics": {
                        "students": students,
                        "randomization_sanity": {} if seed == 42 else {},
                    },
                }
            )
        result, frame = xai.aggregate_results(completions)
        self.assertEqual(len(frame), 20)
        for student_key in xai.STUDENT_KEYS:
            test = result["primary_tests"][student_key]
            self.assertEqual(test["holm_family_size"], 2)
            self.assertEqual(test["exact_paired_wilcoxon"]["p_value_two_sided"], 2.0 / 1024.0)
            self.assertEqual(test["positive_seed_count"], 10)

    def test_aggregate_withholds_entire_holm_family_when_one_seed_is_inconclusive(self):
        completions = []
        for index, seed in enumerate(xai.EXPECTED_SEEDS):
            students = {}
            for student_key in xai.STUDENT_KEYS:
                inconclusive = student_key == "student_B" and seed == 9999
                students[student_key] = {
                    "primary": {
                        "status": "inconclusive" if inconclusive else "conclusive",
                        "inconclusive_reasons": ["Normal has 39 eligible rows; minimum is 40"] if inconclusive else [],
                        "descriptive_defined_class_count": 5,
                        "scratch_alignment_macro_class_mean": 0.2,
                        "rf_kd_alignment_macro_class_mean": 0.3,
                        "rf_kd_minus_scratch_alignment_gain": 0.1,
                    },
                    "eligibility": {"eligible_rows": 249 if inconclusive else 250, "undefined_rows": 1 if inconclusive else 0},
                    "secondary": {"placeholder": float(index)},
                }
            completions.append(
                {
                    "seed": seed,
                    "metrics": {
                        "students": students,
                        "randomization_sanity": {},
                    },
                }
            )
        result, _ = xai.aggregate_results(completions)
        self.assertEqual(result["status"], "complete_primary_inconclusive")
        self.assertEqual(result["primary_family_status"], "inconclusive")
        for student_key in xai.STUDENT_KEYS:
            test = result["primary_tests"][student_key]
            self.assertEqual(test["status"], "inconclusive")
            self.assertIsNone(test["exact_paired_wilcoxon"])
            self.assertIsNone(test["holm_adjusted_p"])
            self.assertIsNone(test["reject_holm_alpha_0_05"])

    def test_random_controls_have_unambiguous_non_route_keys(self):
        self.assertEqual(
            xai.RANDOM_CONTROL_KEYS,
            [
                "control_student_A_fully_reinitialized",
                "control_student_B_fully_reinitialized",
            ],
        )
        self.assertFalse(set(xai.RANDOM_CONTROL_KEYS) & set(xai.SUBJECT_KEYS))
        source = inspect.getsource(xai.reconstruct_seed_subjects)
        self.assertIn('"trained_route": False', source)
        self.assertIn('"must_not_be_interpreted_as"', source)

    def test_new_pipeline_does_not_bind_historical_shap_or_serialize_rf(self):
        source = xai.SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertNotIn("run_fgds_exact_teacher_shap", source)
        self.assertNotIn("fgds_seed42_reconstructed_teacher_shap", source)
        self.assertNotIn("joblib.dump", source)
        self.assertIn("reconstructed RF objects are not serialized", source)
        self.assertIn("--confirm-explanations", source)
        self.assertEqual(
            set(xai.SOURCE_SNAPSHOTS),
            {"executed_controlled_xai_source.py", "bound_common_source.py"},
        )


class ControlledXAIAtomicityTests(unittest.TestCase):
    def make_contract(self):
        value = {
            "software": {
                "source_snapshots": {
                    name: xai.sha256_file(path) for name, path in xai.SOURCE_SNAPSHOTS.items()
                }
            }
        }
        value["execution_contract_id"] = xai.canonical_sha256(value)
        return value

    def make_sampling(self):
        return {
            "cohort_partition_indices": np.arange(10, dtype=np.int64),
            "cohort_labels": np.arange(10, dtype=np.int64) % 5,
        }

    def test_fresh_root_is_initialized_atomically_and_resume_checks_snapshots(self):
        contract = self.make_contract()
        sampling = self.make_sampling()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "xai_output"
            state = xai.prepare_output_root(output, contract, sampling, resume=False)
            self.assertEqual(state, "fresh")
            xai.verify_inventory(output, {"running"})
            xai.verify_sampling(output / "sampling_contract.npz", sampling)
            state = xai.prepare_output_root(output, contract, sampling, resume=True)
            self.assertEqual(state, "resume")
            (output / "bound_common_source.py").write_text("tampered\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                xai.prepare_output_root(output, contract, sampling, resume=True)

    def test_existing_output_requires_resume(self):
        contract = self.make_contract()
        sampling = self.make_sampling()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "xai_output"
            xai.prepare_output_root(output, contract, sampling, resume=False)
            with self.assertRaises(RuntimeError):
                xai.prepare_output_root(output, contract, sampling, resume=False)

    def test_resume_rejects_live_executing_dependency_changed_after_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            live = root / "live.py"
            output = root / "output"
            output.mkdir()
            live.write_text("original\n", encoding="utf-8")
            (output / "executing.py").write_text("original\n", encoding="utf-8")
            expected_hash = xai.sha256_file(live)
            contract = {"software": {"source_snapshots": {"executing.py": expected_hash}}}
            live.write_text("changed\n", encoding="utf-8")
            with mock.patch.object(xai, "SOURCE_SNAPSHOTS", {"executing.py": live}):
                with self.assertRaises(RuntimeError):
                    xai.verify_source_snapshots(output, contract)


class ControlledXAIPersistedIntegrationTests(unittest.TestCase):
    def test_final_source_lineage_and_sampling_preflight(self):
        context = xai.reconstruct_context(xai.DEFAULT_DATASET, xai.DEFAULT_SOURCE_ROOT)
        sampling = xai.build_sampling(context)
        contract = xai.build_contract(xai.DEFAULT_DATASET, xai.DEFAULT_SOURCE_ROOT, sampling)
        self.assertEqual(contract["source_protocol_id"], xai.SOURCE_PROTOCOL_ID)
        self.assertEqual(contract["seeds"], xai.EXPECTED_SEEDS)
        self.assertEqual(
            contract["software"]["common_source_sha256"],
            context["source_execution"]["common_module_sha256"],
        )
        self.assertIn("recorded_training_runner_sha256", contract["source_run"])
        self.assertIn("executed_runner_snapshot_available", contract["source_run"])
        self.assertNotIn("current_runner_path", contract["source_run"])
        self.assertNotIn("current_runner_sha256", contract["source_run"])
        self.assertNotIn("current_runner_matches_recorded_hash", contract["source_run"])
        self.assertIn("excluded from this execution contract", contract["source_run"]["provenance_boundary"])
        self.assertEqual(len(sampling["cohort_partition_indices"]), 250)
        self.assertEqual(len(sampling["background_partition_indices"]), 20)
        np.testing.assert_array_equal(
            sampling["cohort_labels"], np.tile(np.arange(5, dtype=np.int64), 50)
        )
        np.testing.assert_array_equal(
            sampling["background_labels"], np.tile(np.arange(5, dtype=np.int64), 4)
        )
        self.assertEqual(len(np.unique(sampling["cohort_exact_feature_group_hashes"])), 250)
        self.assertEqual(len(np.unique(sampling["background_exact_feature_group_hashes"])), 20)

    def test_persisted_output_verifies_when_present(self):
        if not xai.DEFAULT_OUTPUT.exists():
            self.skipTest("Controlled XAI output has not been generated")
        result = xai.verify_existing(xai.DEFAULT_OUTPUT)
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["verified_seeds"], 10)


if __name__ == "__main__":
    unittest.main()
