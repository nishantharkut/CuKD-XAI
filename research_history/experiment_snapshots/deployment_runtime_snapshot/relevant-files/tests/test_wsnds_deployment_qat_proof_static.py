import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
SOURCE = BASE_DIR / "cukd_xai_wsnds_deployment_qat_proof.py"
NOTEBOOK = BASE_DIR / "cukd_xai_wsnds_deployment_qat_proof.ipynb"
GUIDE = BASE_DIR / "WSNDS_DEPLOYMENT_QAT_PROOF_GUIDE.md"


class TestWSNDSDeploymentQATProofStatic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_text = SOURCE.read_text(encoding="utf-8") if SOURCE.exists() else ""

    def test_files_exist(self):
        self.assertTrue(SOURCE.exists(), "Source file missing")
        self.assertTrue(NOTEBOOK.exists(), "Notebook file missing")
        self.assertTrue(GUIDE.exists(), "Guide file missing")

    def test_core_definitions_present(self):
        self.assertIn("class TeacherMLP", self.source_text)
        self.assertIn("class StudentMLP", self.source_text)
        self.assertIn("def train_standard", self.source_text)
        self.assertIn("def train_kd", self.source_text)

    def test_students_and_teachers_present(self):
        self.assertIn("STUDENT_A_HIDDEN", self.source_text)
        self.assertIn("STUDENT_B_HIDDEN", self.source_text)
        self.assertIn("RandomForestClassifier", self.source_text)
        self.assertIn("B_Full_MLP", self.source_text)

    def test_proof_students_present(self):
        for token in [
            "D_student_A_scratch",
            "E_student_A_KD_from_RF",
            "D_student_B_scratch",
            "E_student_B_KD_from_RF",
        ]:
            self.assertIn(token, self.source_text)

    def test_dynamic_int8_present(self):
        self.assertIn("quantize_dynamic", self.source_text)
        self.assertIn("qint8", self.source_text)

    def test_qat_present(self):
        for token in [
            "QuantStub",
            "DeQuantStub",
            "prepare_qat",
            "convert",
            "fuse_modules",
        ]:
            self.assertIn(token, self.source_text)

    def test_no_disallowed_tokens(self):
        lowered = self.source_text.lower()
        for token in ["edge-iiot", "ciciot", "codistill"]:
            self.assertNotIn(token, lowered)

    def test_no_tmp_usage(self):
        self.assertNotIn("/tmp", self.source_text)

    def test_no_original_full_blocks(self):
        lowered = self.source_text.lower()
        for token in ["run_all_configs", "wilcoxon", "grid search", "shap", "all_seed_results"]:
            self.assertNotIn(token, lowered)

    def test_output_columns_present(self):
        required_columns = [
            "latency_p50_ms_b1",
            "latency_p95_ms_b1",
            "latency_p99_ms_b1",
            "latency_p50_ms_b64",
            "latency_p95_ms_b64",
            "latency_p99_ms_b64",
            "throughput_samples_per_s_b1",
            "throughput_samples_per_s_b64",
            "params",
            "flops_per_sample",
            "serialized_size_kb",
            "compression_ratio_vs_rf",
            "compression_ratio_vs_full_mlp_fp32",
            "macro_f1",
            "macro_f1_delta_vs_fp32",
        ]
        for col in required_columns:
            self.assertIn(col, self.source_text)

    def test_notebook_regenerated(self):
        text = NOTEBOOK.read_text(encoding="utf-8")
        self.assertIn("PROOF_SEED", text)
        self.assertIn("wsnds_deployment_qat_outputs", text)


if __name__ == "__main__":
    unittest.main()
