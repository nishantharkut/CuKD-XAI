import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.wsnds.evidence_completion import analyze_fgds_behavioral_transfer_logits as logits_analysis


analysis = logits_analysis
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def skip_without_hydrated_dataset(test_case: unittest.TestCase) -> None:
    dataset = logits_analysis.DEFAULT_DATASET
    if not dataset.is_file():
        test_case.skipTest("WSN-DS dataset is not available")
    with dataset.open("rb") as handle:
        if handle.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX:
            test_case.skipTest("WSN-DS Git LFS object is not hydrated")


class BehavioralTransferUnitTests(unittest.TestCase):
    def test_softening_matches_kd_probability_transform(self):
        probabilities = np.array(
            [[0.70, 0.20, 0.05, 0.03, 0.02], [0.1, 0.1, 0.1, 0.1, 0.6]],
            dtype=np.float64,
        )
        observed = analysis.soften(probabilities, 4.0)
        expected = probabilities ** 0.25
        expected /= expected.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-15)

    def test_identical_probabilities_have_zero_distance_and_full_overlap(self):
        probabilities = np.array(
            [[0.70, 0.20, 0.05, 0.03, 0.02], [0.1, 0.1, 0.1, 0.1, 0.6]],
            dtype=np.float64,
        )
        softened = analysis.soften(probabilities, 4.0)
        old_rows = analysis.EXPECTED_TEST_ROWS
        try:
            analysis.EXPECTED_TEST_ROWS = 2
            metrics = analysis.per_row_metrics(
                probabilities,
                probabilities,
                softened,
                softened,
            )
        finally:
            analysis.EXPECTED_TEST_ROWS = old_rows
        np.testing.assert_allclose(metrics["kl_teacher_to_student_T4"], 0.0, atol=1e-15)
        np.testing.assert_allclose(metrics["js_T4"], 0.0, atol=1e-15)
        np.testing.assert_allclose(metrics["l1_T4"], 0.0, atol=1e-15)
        np.testing.assert_array_equal(metrics["hard_agreement_T1"], 1.0)
        np.testing.assert_array_equal(metrics["top2_overlap_T1"], 1.0)

    def test_group_balanced_weights_assign_one_total_unit_per_group(self):
        groups = np.array([10, 10, 10, 20, 30, 30], dtype=np.uint64)
        labels = np.array([0, 0, 0, 1, 2, 2], dtype=np.int64)
        old_rows = analysis.EXPECTED_TEST_ROWS
        old_groups = analysis.EXPECTED_TEST_GROUPS
        try:
            analysis.EXPECTED_TEST_ROWS = 6
            analysis.EXPECTED_TEST_GROUPS = 3
            context = analysis.exact_group_context(groups, labels)
        finally:
            analysis.EXPECTED_TEST_ROWS = old_rows
            analysis.EXPECTED_TEST_GROUPS = old_groups
        weights = context["group_balanced_weights"]
        self.assertAlmostEqual(weights[:3].sum(), 1.0)
        self.assertAlmostEqual(weights[3:4].sum(), 1.0)
        self.assertAlmostEqual(weights[4:].sum(), 1.0)

    def test_paired_summary_uses_positive_gain_for_lower_kl(self):
        records = []
        for seed_index, seed in enumerate(analysis.EXPECTED_SEEDS):
            scratch = 0.20 + seed_index * 0.001
            kd = scratch - 0.05
            for route, value in (("scratch", scratch), ("rf_kd", kd)):
                records.append(
                    {
                        "seed": seed,
                        "student": "student_A",
                        "route": route,
                        "weighting": "exact_group_balanced",
                        analysis.PRIMARY_METRIC: value,
                    }
                )
        result = analysis.paired_summary(
            pd.DataFrame(records),
            "student_A",
            "exact_group_balanced",
            analysis.PRIMARY_METRIC,
        )
        self.assertAlmostEqual(result["transfer_gain_mean"], 0.05)
        self.assertEqual(result["positive_seed_count"], 10)
        self.assertEqual(
            result["exact_paired_wilcoxon"]["p_value_two_sided"], 2.0 / 1024.0
        )


class BehavioralTransferIntegrationTests(unittest.TestCase):
    def test_persisted_output_verifies_when_present(self):
        if not analysis.DEFAULT_OUTPUT.exists():
            self.skipTest("Behavioral-transfer output has not been generated")
        skip_without_hydrated_dataset(self)
        result = analysis.verify_existing(analysis.DEFAULT_OUTPUT)
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["protocol_id"], analysis.PROTOCOL_ID)


class CheckpointLogitBehavioralTransferTests(unittest.TestCase):
    def test_checkpoint_t4_is_direct_softmax_of_logits(self):
        torch.manual_seed(17)
        model = logits_analysis.StudentMLP(17, (32, 16), 5).eval()
        values = np.arange(34, dtype=np.float32).reshape(2, 17) / 11.0
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.pt"
            torch.save(model.state_dict(), path)
            old_rows = logits_analysis.EXPECTED_TEST_ROWS
            try:
                logits_analysis.EXPECTED_TEST_ROWS = 2
                replay, observed_hash = logits_analysis.checkpoint_probabilities(
                    path, (32, 16), values
                )
            finally:
                logits_analysis.EXPECTED_TEST_ROWS = old_rows
        with torch.no_grad():
            model_logits = model(torch.from_numpy(values))
            expected_t1 = F.softmax(model_logits, dim=1).numpy()
            expected_t4 = F.softmax(model_logits.double() / 4.0, dim=1).numpy()
        np.testing.assert_allclose(replay["probabilities_t1"], expected_t1, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(replay["probabilities_t4"], expected_t4, rtol=0.0, atol=0.0)
        self.assertEqual(observed_hash, logits_analysis.state_content_sha256(model.state_dict()))

    def test_checkpoint_t4_avoids_float32_softmax_underflow(self):
        model = logits_analysis.StudentMLP(17, (32, 16), 5).eval()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.net[4].bias.copy_(
                torch.tensor([-213.5, 213.5, 0.0, 1.0, -1.0], dtype=torch.float32)
            )
        values = np.zeros((2, 17), dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.pt"
            torch.save(model.state_dict(), path)
            old_rows = logits_analysis.EXPECTED_TEST_ROWS
            try:
                logits_analysis.EXPECTED_TEST_ROWS = 2
                replay, _ = logits_analysis.checkpoint_probabilities(
                    path, (32, 16), values
                )
            finally:
                logits_analysis.EXPECTED_TEST_ROWS = old_rows
        with torch.no_grad():
            logits = model(torch.from_numpy(values))
            self.assertGreater(int((F.softmax(logits / 4.0, dim=1) == 0).sum()), 0)
            expected = F.softmax(logits.double() / 4.0, dim=1).numpy()
        self.assertTrue(np.all(replay["probabilities_t4"] > 0.0))
        np.testing.assert_allclose(
            replay["probabilities_t4"], expected, rtol=0.0, atol=0.0
        )

    def test_v2_contract_separates_teacher_floor_from_student_logits(self):
        self.assertEqual(
            logits_analysis.PROTOCOL_ID,
            "wsnds_fgds_behavioral_transfer_logits_10seed_v5",
        )
        source = logits_analysis.SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn("softmax(saved_checkpoint_logits / 4)", source)
        self.assertIn('"student_probability_floor": None', source)
        self.assertIn("paired training-run/model seed", source)

    def test_canonical_json_hash_is_stable_across_json_key_round_trip(self):
        in_memory = {42: "seed-42", 1001: "seed-1001"}
        persisted = {"42": "seed-42", "1001": "seed-1001"}
        self.assertEqual(
            logits_analysis.canonical_json_sha256(in_memory),
            logits_analysis.canonical_json_sha256(persisted),
        )

    def test_v2_persisted_output_verifies_when_present(self):
        if not logits_analysis.DEFAULT_OUTPUT.exists():
            self.skipTest("Checkpoint-logit behavioral output has not been generated")
        skip_without_hydrated_dataset(self)
        result = logits_analysis.verify_existing(logits_analysis.DEFAULT_OUTPUT)
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["verified_seeds"], 10)


if __name__ == "__main__":
    unittest.main()
