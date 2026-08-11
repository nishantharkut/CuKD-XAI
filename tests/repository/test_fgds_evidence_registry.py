import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.evidence import build_fgds_evidence_registry as registry


class EvidenceRegistryUnitTests(unittest.TestCase):
    def test_recorded_file_accepts_only_exact_or_crlf_checkout_bytes(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "evidence.json"
            canonical = b'{\n  "status": "passed"\n}\n'
            path.write_bytes(canonical)
            expected = hashlib.sha256(canonical).hexdigest()
            self.assertEqual(
                registry.verify_recorded_file(path, expected, len(canonical)),
                "exact_bytes",
            )

            path.write_bytes(canonical.replace(b"\n", b"\r\n"))
            self.assertEqual(
                registry.verify_recorded_file(path, expected, len(canonical)),
                "working_tree_crlf_normalized_to_recorded_lf",
            )

            path.write_bytes(canonical.replace(b"passed", b"failed"))
            with self.assertRaises(RuntimeError):
                registry.verify_recorded_file(path, expected, len(canonical))

    def test_exact_signed_rank_enumerates_all_five_seed_signs(self):
        result = registry.exact_signed_rank([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertEqual(result["enumerated_sign_assignments"], 32)
        self.assertEqual(result["positive_pairs"], 5)
        self.assertEqual(result["negative_pairs"], 0)
        self.assertEqual(result["p_value_two_sided_exact"], 0.0625)


class EvidenceRegistryIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        required = [
            registry.TEN_SEED_ROOT,
            registry.DEPLOYMENT_ROOT,
            registry.RUNTIME_ROOT,
            registry.USB_REPORT_ROOT,
            registry.WIRELESS_REPORT_ROOT,
            registry.EDGE_ROOT,
        ]
        if any(not path.exists() for path in required):
            raise unittest.SkipTest("Complete local evidence set is not present")
        cls.result = registry.build_registry()

    def test_primary_protocol_and_statistics_are_frozen(self):
        result = self.result["primary_wsnds_multi_seed"]
        self.assertEqual(result["protocol_id"], registry.TEN_SEED_PROTOCOL)
        self.assertEqual(result["seeds"], registry.EXPECTED_SEEDS)
        self.assertEqual(result["test_rows_per_model_seed"], 56_301)
        self.assertAlmostEqual(
            result["routes"]["student_A_rf_kd"]["macro_f1_mean"],
            0.9137807036842556,
        )
        self.assertAlmostEqual(
            result["routes"]["student_B_rf_kd"]["macro_f1_mean"],
            0.9321418755524803,
        )
        self.assertFalse(
            result["paired_tests"]["student_A"]["reject_holm_alpha_0_05"]
        )
        self.assertFalse(
            result["paired_tests"]["student_B"]["reject_holm_alpha_0_05"]
        )

    def test_runtime_usb_and_wireless_share_exact_export_ids(self):
        deployment = self.result["fixed_deployment_specimen"]
        hardware = self.result["hardware_execution"]
        self.assertEqual(hardware["usb_full_predictions"], 225_204)
        self.assertEqual(hardware["wireless_full_predictions"], 225_204)
        for label, row in hardware["rows"].items():
            self.assertEqual(
                row["export_id"], deployment["exports"][row["student"]]["export_id"]
            )
            self.assertEqual(row["mcu_vs_fixed_reference_agreement"], 1.0)
            self.assertEqual(row["exact_logit_agreement"], 1.0)

    def test_deployment_checkpoint_is_not_substituted_from_local_run(self):
        deployment = self.result["fixed_deployment_specimen"]
        self.assertEqual(
            deployment["checkpoints_byte_identical"],
            {"student_A": False, "student_B": False},
        )
        self.assertTrue(
            deployment["preprocessing_equivalence_to_ten_seed"][
                "all_semantic_fields_equal"
            ]
        )

    def test_edge_contract_keeps_residual_encoded_overlap_visible(self):
        edge = self.result["secondary_edge_iiotset"]
        self.assertEqual(edge["input_dim"], 40)
        self.assertEqual(edge["pre_encode_group_overlap"], 0)
        self.assertEqual(
            edge["encoded_exact_row_overlap"],
            {"train_test": 163, "train_validation": 157, "validation_test": 26},
        )

    def test_persisted_registry_matches_current_registry_id(self):
        path = registry.DEFAULT_OUTPUT / "evidence_registry.json"
        persisted = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(persisted["status"], "passed")
        self.assertEqual(persisted["registry_id"], self.result["registry_id"])


if __name__ == "__main__":
    unittest.main()
