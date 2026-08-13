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
            registry.FULL_ROUTE_ROOT,
            registry.SENSITIVITY_ROOT,
            registry.GROUP_BALANCED_ROOT,
            registry.SHAP_ROOT,
            registry.FIXED_POINT_REFINEMENT_ROOT,
            registry.MSP430_STATIC_ROOT,
            registry.BEHAVIORAL_TRANSFER_ROOT,
            registry.MULTISPLIT_ROOT,
            registry.ALL_SEED_FIXED_POINT_ROOT,
            registry.FINAL_HIL_REPORT_ROOT,
            registry.PREDECESSOR_REGISTRY_ROOT,
            registry.FINAL_HIL_CAMPAIGN_ROOT,
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

    def test_completed_route_and_sensitivity_evidence_is_current(self):
        routes = self.result["controlled_full_route_matrix"]
        self.assertEqual(routes["status"], "current_controlled_route_evidence")
        self.assertEqual(routes["seeds"], registry.EXPECTED_SEEDS)
        self.assertEqual(routes["paired_test_count"], 26)
        self.assertEqual(routes["finalization"]["neural_artifacts_replayed_exactly"], 180)
        self.assertEqual(routes["finalization"]["rf_artifacts_replayed_exactly"], 10)
        self.assertAlmostEqual(
            routes["student_routes"]["student_A"]["J_CoDistill_RF_CL"]["macro_f1"]["mean"],
            0.917462515299208,
        )

        sensitivity = self.result["rfkd_hyperparameter_sensitivity"]
        self.assertEqual(sensitivity["training_jobs"], 180)
        self.assertFalse(sensitivity["selection_performed"])
        self.assertFalse(sensitivity["primary_result_replaced"])
        comparison = sensitivity["selected_comparisons"][
            "student_A:T1_alpha03_minus_persisted_scratch_test_macro_f1"
        ]
        self.assertAlmostEqual(comparison["difference"]["mean"], 0.009276490601257492)
        self.assertAlmostEqual(comparison["holm_global_p"], 0.10546875)

    def test_repeated_pattern_views_do_not_replace_primary_inference(self):
        result = self.result["repeated_pattern_sensitivity"]
        self.assertEqual(result["test_group_summary"]["test_rows"], 56_301)
        self.assertEqual(result["test_group_summary"]["test_exact_feature_groups"], 54_174)
        self.assertEqual(result["test_group_summary"]["mixed_label_groups"], 0)
        self.assertEqual(
            result["within_group_prediction_probability_audit"]["route_seed_count"],
            240,
        )
        self.assertLessEqual(
            result["within_group_prediction_probability_audit"][
                "global_max_probability_abs_delta"
            ],
            result["within_group_prediction_probability_audit"][
                "probability_max_abs_delta_tolerance"
            ],
        )
        self.assertAlmostEqual(
            result["rf_kd_minus_scratch"]["inverse_test_group_size"]["student_A"][
                "differences"
            ]["mean"],
            0.0012090345885727,
        )

    def test_current_shap_is_exact_and_v2_is_excluded(self):
        shap = self.result["current_exact_lineage_xai"]
        self.assertEqual(shap["protocol_id"], registry.SHAP_PROTOCOL)
        self.assertEqual(shap["explained_test_rows"], 500)
        self.assertEqual(shap["estimator_replicates"], 3)
        self.assertLessEqual(
            shap["global_max_local_accuracy_residual"], shap["local_accuracy_atol"]
        )
        self.assertAlmostEqual(
            shap["condition_summaries"]["kd_softened_probabilities_T4"]["student_B"][
                "global_spearman_rho"
            ]["mean"],
            0.553921568627451,
        )
        partial = self.result["excluded_incomplete"]["shap_v2_partial_attempt"]
        self.assertEqual(partial["status"], "incomplete_excluded")
        self.assertGreater(partial["file_count"], 0)

    def test_behavioral_transfer_is_post_hoc_and_directionally_consistent(self):
        result = self.result["behavioral_transfer"]
        self.assertEqual(result["protocol_id"], registry.BEHAVIORAL_TRANSFER_PROTOCOL)
        self.assertEqual(result["status"], "post_hoc_secondary_evidence")
        self.assertEqual(result["seeds"], registry.EXPECTED_SEEDS)
        for student in ("student_A", "student_B"):
            test = result["primary_tests"][student]
            self.assertEqual(test["positive_seed_count"], 10)
            self.assertTrue(test["reject_holm_alpha_0_05"])
            self.assertLessEqual(test["holm_adjusted_p"], 0.05)

    def test_multisplit_confirmation_remains_descriptive(self):
        result = self.result["multisplit_core_confirmation"]
        self.assertEqual(result["protocol_id"], registry.MULTISPLIT_PROTOCOL)
        self.assertEqual(result["split_count"], 10)
        self.assertEqual(result["optimizer_seeds_per_split"], [42, 123])
        self.assertEqual(result["training_jobs"], 80)
        self.assertFalse(result["formal_hypothesis_test_performed"])
        self.assertEqual(
            result["descriptive_summaries"]["student_A"]["positive_split_count"],
            10,
        )
        self.assertEqual(
            result["descriptive_summaries"]["student_B"]["positive_split_count"],
            5,
        )
        self.assertEqual(
            result["descriptive_summaries"]["student_B"]["negative_split_count"],
            5,
        )

    def test_all_seed_fixed_point_failures_are_retained(self):
        result = self.result["all_seed_software_fixed_point_audit"]
        self.assertEqual(result["protocol_id"], registry.ALL_SEED_FIXED_POINT_PROTOCOL)
        self.assertEqual(result["model_count"], 40)
        self.assertEqual(result["status_counts"], {"passed": 26, "gate_failed": 14})
        self.assertIn("not 40 independent data splits", result["statistical_unit_disclosure"])

    def test_final_usb_campaign_is_exact_and_keeps_blocked_route_visible(self):
        result = self.result["final_usb_hardware_campaign"]
        self.assertEqual(result["status"], "passed_with_blocked_routes")
        self.assertEqual(result["accepted_models"], [
            "student_A_rf_kd",
            "student_A_scratch",
            "student_B_rf_kd",
        ])
        self.assertEqual(result["blocked_models"], ["student_B_scratch"])
        self.assertEqual(result["session_count"], 6)
        self.assertEqual(result["full_exact_replay_rows"], 337_806)
        self.assertEqual(result["all_device_inferences"], 355_926)
        self.assertEqual(
            result["source_predictive_protocol"], registry.TEN_SEED_PROTOCOL
        )
        self.assertEqual(len(result["sessions"]), 6)
        for session in result["sessions"]:
            self.assertEqual(session["fidelity"]["rows"], 56_301)
            self.assertEqual(
                session["fidelity"]["mcu_vs_fixed_reference_agreement"], 1.0
            )
            self.assertEqual(
                session["fidelity"]["mcu_fixed_logits_exact_fraction"], 1.0
            )
        self.assertEqual(
            result["external_archive"]["compressed_sha256"],
            "0361f70877b00a27df5e7c559d178a9f4fbdd37136c05dc7d68fdce0b4c79561",
        )

    def test_registry_does_not_hide_unexecuted_planned_work(self):
        self.assertEqual(self.result["status"], "passed_with_open_planned_work")
        predecessor = self.result["predecessor_registry"]
        self.assertEqual(predecessor["registry_id"], registry.PREDECESSOR_REGISTRY_ID)
        open_work = self.result["open_planned_work"]
        self.assertEqual(open_work["status"], "open_planned_work")
        self.assertEqual(
            open_work["controlled_ten_seed_xai"]["status"], "not_executed"
        )
        self.assertEqual(open_work["final_lineage_wifi"]["status"], "not_executed")
        self.assertEqual(
            open_work["final_lineage_wifi"]["planned_full_replay_rows"], 337_806
        )

    def test_software_refinement_and_msp430_boundaries_are_enforced(self):
        deployment = self.result["fixed_deployment_specimen"]
        refinement = self.result["software_only_fixed_point_refinement"]
        self.assertEqual(refinement["status"], "software_only_candidate_evidence")
        self.assertIn("has not been strictly exported", refinement["boundary"])
        for student in ("student_A", "student_B"):
            self.assertEqual(
                refinement["students"][student]["source_export_id"],
                deployment["exports"][student]["export_id"],
            )
            self.assertTrue(
                refinement["students"][student]["zero_saturation_and_no_overflow_gate"]
            )

        msp430 = self.result["current_msp430_static"]
        self.assertEqual(msp430["status"], "current_static_cross_compile_evidence")
        self.assertIn("No physical TelosB execution", msp430["claim_boundary"])
        self.assertEqual(msp430["students"]["student_A"]["static_flash_load_bytes"], 2846)
        self.assertEqual(msp430["students"]["student_B"]["static_flash_load_bytes"], 5196)
        for student in ("student_A", "student_B"):
            self.assertEqual(
                msp430["students"][student]["export_id"],
                deployment["exports"][student]["export_id"],
            )

    def test_persisted_registry_matches_current_registry_id(self):
        path = registry.DEFAULT_OUTPUT / "evidence_registry.json"
        persisted = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(persisted["status"], "passed_with_open_planned_work")
        self.assertEqual(persisted["registry_id"], self.result["registry_id"])
        inventory = registry.verify_existing_registry(registry.DEFAULT_OUTPUT)
        self.assertEqual(inventory["verified_files"], 4)

    def test_writer_refuses_to_overwrite_any_existing_directory(self):
        with tempfile.TemporaryDirectory(dir=registry.DEFAULT_OUTPUT.parent) as temporary:
            with self.assertRaisesRegex(RuntimeError, "Refusing to overwrite"):
                registry.write_registry(Path(temporary), self.result)


if __name__ == "__main__":
    unittest.main()
