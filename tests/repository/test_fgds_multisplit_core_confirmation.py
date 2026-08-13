import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.wsnds.evidence_completion import run_fgds_multisplit_core_confirmation as run


class MultiSplitCoreUnitTests(unittest.TestCase):
    def test_split_is_deterministic_and_group_disjoint(self):
        features = np.array(
            [[group, group % 3] for group in range(200) for _ in range(2)],
            dtype=np.float32,
        )
        labels = np.array([group % 5 for group in range(200) for _ in range(2)], dtype=np.int64)
        original = (
            run.EXPECTED_GROUP_COUNT,
            run.EXPECTED_MIXED_LABEL_GROUPS,
            run.EXPECTED_MIXED_LABEL_ROWS,
        )
        try:
            run.EXPECTED_GROUP_COUNT = 200
            run.EXPECTED_MIXED_LABEL_GROUPS = 0
            run.EXPECTED_MIXED_LABEL_ROWS = 0
            table = run.prepare_group_table(features, labels)
            first = run.make_split(features, labels, table, 123)
            second = run.make_split(features, labels, table, 123)
        finally:
            (
                run.EXPECTED_GROUP_COUNT,
                run.EXPECTED_MIXED_LABEL_GROUPS,
                run.EXPECTED_MIXED_LABEL_ROWS,
            ) = original
        for partition in ("train", "validation", "test"):
            np.testing.assert_array_equal(
                first[f"{partition}_indices"], second[f"{partition}_indices"]
            )
        for value in first["group_audit"].values():
            if isinstance(value, int):
                self.assertGreaterEqual(value, 0)
        self.assertEqual(first["group_audit"]["train_test_feature_overlap"], 0)

    def test_split_seed_changes_assignment(self):
        features = np.array(
            [[group, group % 3] for group in range(200) for _ in range(2)],
            dtype=np.float32,
        )
        labels = np.array([group % 5 for group in range(200) for _ in range(2)], dtype=np.int64)
        original = (
            run.EXPECTED_GROUP_COUNT,
            run.EXPECTED_MIXED_LABEL_GROUPS,
            run.EXPECTED_MIXED_LABEL_ROWS,
        )
        try:
            run.EXPECTED_GROUP_COUNT = 200
            run.EXPECTED_MIXED_LABEL_GROUPS = 0
            run.EXPECTED_MIXED_LABEL_ROWS = 0
            table = run.prepare_group_table(features, labels)
            first = run.make_split(features, labels, table, 42)
            second = run.make_split(features, labels, table, 123)
        finally:
            (
                run.EXPECTED_GROUP_COUNT,
                run.EXPECTED_MIXED_LABEL_GROUPS,
                run.EXPECTED_MIXED_LABEL_ROWS,
            ) = original
        self.assertFalse(np.array_equal(first["test_indices"], second["test_indices"]))

    def test_contract_has_distinct_split_and_optimizer_seeds(self):
        self.assertEqual(len(run.SPLIT_SEEDS), 10)
        self.assertEqual(len(run.OPTIMIZER_SEEDS), 2)
        self.assertEqual(len(set(run.SPLIT_SEEDS)), 10)
        self.assertEqual(len(set(run.OPTIMIZER_SEEDS)), 2)
        self.assertEqual(run.RF_SEED, 42)
        self.assertEqual(run.PROTOCOL_ID, "wsnds_fgds_multisplit_core_10x2_v2")

    def test_probability_gate_rejects_nonfinite_and_non_simplex_values(self):
        valid = np.full((3, len(run.CLASS_NAMES)), 1.0 / len(run.CLASS_NAMES))
        np.testing.assert_array_equal(
            run.validate_probability_matrix(valid, 3, "valid"), valid
        )
        nonfinite = valid.copy()
        nonfinite[0, 0] = np.nan
        with self.assertRaisesRegex(RuntimeError, "NaN or infinity"):
            run.validate_probability_matrix(nonfinite, 3, "nonfinite")
        nonsimplex = valid.copy()
        nonsimplex[0, 0] += 0.1
        with self.assertRaisesRegex(RuntimeError, "do not sum to one"):
            run.validate_probability_matrix(nonsimplex, 3, "nonsimplex")

    def test_hard_label_metric_reconstruction_uses_all_five_classes(self):
        labels = np.arange(len(run.CLASS_NAMES), dtype=np.uint8)
        metrics = run.metrics_from_labels(labels, labels)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["per_class_support"], [1] * len(run.CLASS_NAMES))
        self.assertEqual(np.asarray(metrics["confusion_matrix"]).trace(), len(run.CLASS_NAMES))
        changed = json.loads(json.dumps(metrics))
        changed["macro_f1"] = 0.9
        with self.assertRaisesRegex(RuntimeError, "differs for macro_f1"):
            run.require_metrics_equal(changed, metrics, "changed")

    def test_multisplit_aggregate_is_explicitly_descriptive(self):
        completions = []
        for split_index, split_seed in enumerate(run.SPLIT_SEEDS):
            rows = []
            for optimizer_seed in run.OPTIMIZER_SEEDS:
                for student_name in run.STUDENT_SPECS:
                    scratch = 0.90 + split_index * 0.001
                    delta = 0.002 if student_name == "student_A" else -0.001
                    rows.append(
                        {
                            "student": student_name,
                            "optimizer_seed": optimizer_seed,
                            "scratch": {"macro_f1": scratch},
                            "rf_kd": {"macro_f1": scratch + delta},
                            "rf_kd_minus_scratch_macro_f1": delta,
                        }
                    )
            completions.append(
                {
                    "split_seed": split_seed,
                    "teacher_metrics": {"per_class_support": [1, 1, 1, 1, 1]},
                    "student_results": rows,
                }
            )
        result, table = run.aggregate_results(completions)
        self.assertFalse(result["formal_hypothesis_test_performed"])
        self.assertNotIn("tests", result)
        self.assertEqual(len(table), 2 * len(run.SPLIT_SEEDS))
        self.assertIn("not as independent replications", result["claim_boundary"])

    def test_source_snapshot_contract_covers_all_executable_dependencies(self):
        self.assertEqual(
            set(run.SOURCE_SNAPSHOTS),
            {"executed_multisplit_source.py", "bound_common_source.py", "bound_rf_source.py"},
        )
        for source in run.SOURCE_SNAPSHOTS.values():
            self.assertTrue(source.is_file())

    def test_empty_output_root_and_resume_snapshot_preflight(self):
        snapshots = {
            name: run.sha256_file(source) for name, source in run.SOURCE_SNAPSHOTS.items()
        }
        contract = {
            "execution_contract_id": "test-contract",
            "software": {
                "source_snapshots": snapshots,
                "executed_source_sha256": snapshots["executed_multisplit_source.py"],
                "common_source_sha256": snapshots["bound_common_source.py"],
                "rf_source_sha256": snapshots["bound_rf_source.py"],
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "already_empty"
            output.mkdir()
            self.assertEqual(run.prepare_output_root(output, contract, False), "fresh")
            self.assertEqual(run.prepare_output_root(output, contract, True), "resume")
            (output / "bound_common_source.py").write_text("corrupt", encoding="ascii")
            with self.assertRaisesRegex(RuntimeError, "Source snapshot mismatch"):
                run.prepare_output_root(output, contract, True)

    def test_initial_source_sealing_is_atomic_at_output_boundary(self):
        snapshots = {
            name: run.sha256_file(source) for name, source in run.SOURCE_SNAPSHOTS.items()
        }
        contract = {
            "execution_contract_id": "test-contract",
            "software": {
                "source_snapshots": snapshots,
                "executed_source_sha256": snapshots["executed_multisplit_source.py"],
                "common_source_sha256": snapshots["bound_common_source.py"],
                "rf_source_sha256": snapshots["bound_rf_source.py"],
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "atomic_output"
            output.mkdir()
            with patch.object(run.shutil, "copy2", side_effect=RuntimeError("copy failed")):
                with self.assertRaisesRegex(RuntimeError, "copy failed"):
                    run.prepare_output_root(output, contract, False)
            self.assertTrue(output.is_dir())
            self.assertFalse(any(output.iterdir()))
            self.assertEqual(run.prepare_output_root(output, contract, False), "fresh")
            run.verify_inventory(output, {"running"})

    def test_live_verifier_must_match_sealed_sources(self):
        snapshots = {
            name: run.sha256_file(source) for name, source in run.SOURCE_SNAPSHOTS.items()
        }
        contract = {
            "software": {
                "source_snapshots": snapshots,
                "executed_source_sha256": snapshots["executed_multisplit_source.py"],
                "common_source_sha256": snapshots["bound_common_source.py"],
                "rf_source_sha256": snapshots["bound_rf_source.py"],
            }
        }
        run.verify_live_sources_match_contract(contract)
        contract["software"]["source_snapshots"]["bound_common_source.py"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "Live executable source differs"):
            run.verify_live_sources_match_contract(contract)

    def test_resume_quarantines_semantically_invalid_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split_root = root / "split_42"
            split_root.mkdir()
            (split_root / "corrupt.txt").write_text("corrupt", encoding="ascii")
            with patch.object(
                run,
                "verify_split_completion",
                side_effect=RuntimeError("semantic corruption"),
            ):
                with self.assertRaises(KeyError):
                    run.run_split(
                        root,
                        {"execution_contract_id": "test-contract"},
                        {},
                        {},
                        42,
                        run.torch.device("cpu"),
                        True,
                    )
            quarantined = list((root / "failed_split_attempts").glob("split_42_*"))
            self.assertEqual(len(quarantined), 1)
            reason = json.loads(
                (quarantined[0] / "quarantine_reason.json").read_text(encoding="utf-8")
            )
            self.assertEqual(reason["reason_type"], "RuntimeError")
            self.assertIn("semantic corruption", reason["reason"])
            run.verify_inventory(quarantined[0], {"failed"})

    def test_manifest_rejects_unlisted_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "declared.txt").write_text("declared", encoding="ascii")
            run.atomic_write_json(
                root / "artifact_manifest.json", run.build_inventory(root, "running")
            )
            run.verify_inventory(root, {"running"})
            (root / "stale.txt").write_text("stale", encoding="ascii")
            with self.assertRaisesRegex(RuntimeError, "does not exactly cover"):
                run.verify_inventory(root, {"running"})

    def test_failed_semantic_record_cannot_claim_passed(self):
        contract = {"execution_contract_id": "test-contract"}
        record = run.semantic_verification_record(
            contract, "failed", RuntimeError("intentional failure")
        )
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["verified_split_count"], 0)
        self.assertEqual(record["failure_type"], "RuntimeError")
        self.assertIn("intentional failure", record["failure_message"])


class MultiSplitCoreIntegrationTests(unittest.TestCase):
    def test_persisted_output_verifies_when_present(self):
        manifest_path = run.DEFAULT_OUTPUT / "artifact_manifest.json"
        if not manifest_path.is_file():
            self.skipTest("Multi-split output has not been generated")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            self.skipTest("Multi-split output is still running")
        result = run.verify_existing(run.DEFAULT_OUTPUT)
        self.assertEqual(result["status"], "verified")


if __name__ == "__main__":
    unittest.main()
