"""Integrity tests for paper-strength E2E freeze package."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "results" / "paper_strength_e2e"
FG = (
    ROOT
    / "results/wsnds/confirmation_runs_v2/remote_winterfell_feature_group_5seed_20260805/feature_group_5seed"
)


class PaperStrengthE2ETests(unittest.TestCase):
    def test_package_files_exist(self):
        for name in [
            "01_dual_identity_freeze.json",
            "02_protocol_ladder.json",
            "03_per_class_kd_minus_scratch.csv",
            "04_gate_policy_freeze.json",
            "05_edge_duplicate_disclosure.csv",
            "06_claim_freeze.json",
            "README.md",
        ]:
            self.assertTrue((PKG / name).is_file(), name)

    def test_dual_identity_z_scores(self):
        dual = json.loads((PKG / "01_dual_identity_freeze.json").read_text(encoding="utf-8"))
        ck = dual["units"]["multi_seed_pipeline_seed42"]["student_A_rf_kd"]
        dep = dual["units"]["deployment_clean_seed42"]["student_A_rf_kd"]
        self.assertLess(abs(ck["z_vs_10seed_mean"]), 2.0)
        self.assertGreater(dep["z_vs_10seed_mean"], 3.5)
        self.assertTrue(dual["units"]["deployment_clean_seed42"]["soft_targets_match_main_10seed"])
        self.assertAlmostEqual(ck["macro_f1"], 0.924859, places=5)
        self.assertAlmostEqual(dep["macro_f1"], 0.948509, places=5)

    def test_fg_mean_matches_values(self):
        ladder = json.loads((PKG / "02_protocol_ladder.json").read_text(encoding="utf-8"))
        row = next(
            r
            for r in ladder
            if r["student"] == "A" and "feature_group" in r["protocol"]
        )
        vals = row["fg_per_seed_rf_kd"]
        self.assertEqual(len(vals), 5)
        self.assertAlmostEqual(float(np.mean(vals)), row["rf_kd_macro_f1_mean"], places=12)
        # recompute one seed from CSV
        pred = pd.read_csv(FG / "seed_42/student_A_KD_from_RF_test_predictions.csv")
        f1 = f1_score(pred.true_label, pred.predicted_label, average="macro")
        self.assertAlmostEqual(f1, vals[0], places=6)

    def test_fg_kd_scratch_collapse(self):
        ladder = json.loads((PKG / "02_protocol_ladder.json").read_text(encoding="utf-8"))
        row = next(
            r
            for r in ladder
            if r["student"] == "A" and "feature_group" in r["protocol"]
        )
        self.assertLess(abs(row["kd_minus_scratch_mean_paired"]), 0.001)
        self.assertGreater(row["paired_t_p"], 0.5)

    def test_trainonly_a_kd_edge_borderline(self):
        ladder = json.loads((PKG / "02_protocol_ladder.json").read_text(encoding="utf-8"))
        row = next(
            r
            for r in ladder
            if r["student"] == "A" and r["protocol"] == "train_only_scaler_random_row_10seed"
        )
        self.assertGreater(row["kd_minus_scratch_mean_paired"], 0.005)
        self.assertLess(row["paired_t_p"], 0.06)

    def test_gate_policy_b(self):
        g = json.loads((PKG / "04_gate_policy_freeze.json").read_text(encoding="utf-8"))
        self.assertEqual(g["decision"], "B_measured_conversion_bound")
        self.assertFalse(g["students"]["A"]["passes_0_01"])
        self.assertTrue(g["students"]["A"]["passes_0_03"])
        self.assertFalse(g["students"]["B"]["passes_0_01"])
        self.assertTrue(g["students"]["B"]["passes_0_03"])

    def test_forbidden_claims_present(self):
        c = json.loads((PKG / "06_claim_freeze.json").read_text(encoding="utf-8"))
        ids = {x["id"] for x in c["forbidden_or_retired_claims"]}
        self.assertIn("X1", ids)
        self.assertIn("X2", ids)
        self.assertIn("X3", ids)

    def test_shap_deployment_results(self):
        path = PKG / "shap_train_only_deployment" / "shap_results.json"
        self.assertTrue(path.is_file(), "SHAP results missing — run run_shap_train_only_deployment.py")
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data.get("subject"), "deployment_clean_seed42_rf_kd")
        for stu in ("A", "B"):
            row = data["students"][stu]
            self.assertLess(abs(row["ranking_agreement_spearman"]), 0.7)
            self.assertGreater(row["ranking_agreement_p"], 0.05)
            self.assertEqual(len(row["bootstrap_spearman_values"]), 5)
            self.assertTrue((PKG / f"shap_train_only_deployment/student_{stu}_feature_ranks.csv").is_file())
        ids = {x["id"] for x in json.loads((PKG / "06_claim_freeze.json").read_text(encoding="utf-8"))["allowed_primary_claims"]}
        self.assertIn("C6_xai_deployment", ids)


if __name__ == "__main__":
    unittest.main()
