"""Integrity tests for train-only research packages (no hardware).

Fails if incomplete n=2 summaries are treated as full 10-seed evidence,
or if four-pair HIL / runtime contracts regress.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FULL_AGG = ROOT / "results/wsnds/leakage_free_rerun/main_10seed_full_aggregate_copy"
INCOMPLETE = ROOT / "results/wsnds/leakage_free_rerun/main_10seed"
HIL = ROOT / "results/hardware_hil/train_only_scaler_copy"
RUNTIME = ROOT / "results/runtime/onnx_openvino/wsnds/train_only_seed42_copy"
SEEDS = {42, 123, 456, 789, 1001, 2024, 3141, 5678, 8192, 9999}


class TrainOnlyResearchIntegrityTests(unittest.TestCase):
    def test_full_aggregate_is_ten_seeds(self):
        for student in ("A", "B"):
            path = FULL_AGG / f"wsnds_results_student_{student}_10seed.csv"
            self.assertTrue(path.is_file(), path)
            df = pd.read_csv(path)
            self.assertTrue((df["n_seeds"] == 10).all(), path)
            rf = df[df["Config"] == "E_KD_from_RF"]
            self.assertEqual(len(rf), 1)
            self.assertGreater(float(rf.iloc[0]["MacroF1_mean"]), 0.9)

    def test_incomplete_summary_csv_is_not_ten_seeds(self):
        """Guardrail: original merge files must not be mistaken for full 10-seed."""
        for student in ("A", "B"):
            path = INCOMPLETE / f"wsnds_results_student_{student}.csv"
            self.assertTrue(path.is_file(), path)
            df = pd.read_csv(path)
            # Documented incomplete residue
            self.assertTrue((df["n_seeds"] == 2).all(), f"{path} unexpectedly full")

    def test_twenty_checkpoints_present(self):
        paths = list((INCOMPLETE).glob("checkpoint_student_*_seed_*.json"))
        self.assertEqual(len(paths), 20)
        seen = set()
        for p in paths:
            data = json.loads(p.read_text(encoding="utf-8"))
            seen.add((str(data.get("student")).replace("student_", ""), int(data["seed"])))
        for student in ("A", "B"):
            for seed in SEEDS:
                self.assertIn((student, seed), seen)

    def test_no_j_codistill_in_train_only_checkpoints(self):
        sample = json.loads(
            (INCOMPLETE / "checkpoint_student_A_seed_42.json").read_text(encoding="utf-8")
        )
        keys = set(sample["results"].keys())
        self.assertFalse(any(k.startswith("J_") or "CoDistill" in k for k in keys))

    def test_four_pair_hil_contract(self):
        summary = json.loads((HIL / "four_pair_summary.json").read_text(encoding="utf-8"))
        self.assertTrue(summary.get("matrix_complete") or summary.get("esp32c3"))
        for folder in (
            "pi5_arduino_r4_student_A",
            "pi5_arduino_r4_student_B",
            "pi5_esp32c3_student_A",
            "pi5_esp32c3_student_B",
        ):
            metrics = json.loads(
                (HIL / folder / "full_56200_metrics.json").read_text(encoding="utf-8")
            )
            seq = json.loads(
                (HIL / folder / "full_56200_sequence.json").read_text(encoding="utf-8")
            )
            self.assertEqual(seq.get("expected"), 56200)
            self.assertEqual(seq.get("completed"), 56200)
            self.assertEqual(seq.get("status_counts", {}).get("OK"), 56200)
            self.assertEqual(metrics.get("mcu_vs_fixed_reference_agreement"), 1.0)
            self.assertEqual(metrics.get("n"), 56200)

    def test_runtime_onnx_fp32_perfect_agreement(self):
        data = json.loads(
            (RUNTIME / "train_only_runtime_results.json").read_text(encoding="utf-8")
        )
        rows = data.get("onnx_runtime_rows") or []
        fp32 = [r for r in rows if r.get("variant") == "onnx_fp32"]
        self.assertEqual(len(fp32), 2)
        for row in fp32:
            self.assertEqual(row.get("prediction_agreement_vs_pytorch_fp32"), 1.0)

    def test_openvino_present_and_agrees(self):
        path = RUNTIME / "train_only_openvino_results.json"
        self.assertTrue(path.is_file(), "OpenVINO results missing")
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data.get("rows") or []:
            self.assertEqual(row.get("prediction_agreement_vs_pytorch_fp32"), 1.0)
            self.assertEqual(row.get("prediction_agreement_vs_onnx_runtime"), 1.0)

    def test_research_completion_status_exists(self):
        path = ROOT / "results/RESEARCH_COMPLETION_STATUS.md"
        self.assertTrue(path.is_file())
        text = path.read_text(encoding="utf-8")
        self.assertIn("Manuscript policy", text)
        self.assertIn("NOT complete", text)


if __name__ == "__main__":
    unittest.main()
