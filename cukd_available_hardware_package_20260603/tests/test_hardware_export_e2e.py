import tempfile
import unittest
from pathlib import Path

from hardware_export import export_wsnds_student_a_rfkd_int8 as exp


class HardwareExportE2ETests(unittest.TestCase):
    def test_build_preprocessing_metadata_records_v23_contract(self):
        metadata = exp.build_preprocessing_metadata(
            target_col="Attack type",
            feature_names=["feat_b", "feat_a"],
            class_names=["Blackhole", "Flooding", "Normal"],
            scaler_mean=[15.0, 1.0],
            scaler_scale=[3.415650255, 0.3415650255],
            split_sizes={"train": 4, "val": 1, "test": 1},
        )

        self.assertEqual(metadata["target_col"], "Attack type")
        self.assertEqual(metadata["feature_names"], ["feat_b", "feat_a"])
        self.assertEqual(metadata["class_names"], ["Blackhole", "Flooding", "Normal"])
        self.assertEqual(metadata["scaler"]["mean"], [15.0, 1.0])
        self.assertEqual(metadata["scaler"]["scale"], [3.415650255, 0.3415650255])
        self.assertEqual(metadata["split_sizes"], {"train": 4, "val": 1, "test": 1})
        self.assertIn("v2.3", metadata["preprocessing_contract"])


    def test_write_preprocessing_header_exports_feature_order_and_scaler_constants(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "preprocess_metadata.h"
            metadata = exp.build_preprocessing_metadata(
                target_col="Attack type",
                feature_names=["feat_b", "feat_a"],
                class_names=["Blackhole", "Flooding"],
                scaler_mean=[15.0, 1.0],
                scaler_scale=[3.5, 0.25],
                split_sizes={"train": 4, "val": 1, "test": 1},
            )

            summary = exp.write_preprocessing_header(output_path, metadata)

            text = output_path.read_text(encoding="ascii")
            self.assertIn("#define CUKD_PREPROCESS_INPUT_DIM 2", text)
            self.assertIn("cukd_scaler_mean", text)
            self.assertIn("15.000000000f", text)
            self.assertIn("cukd_feature_names", text)
            self.assertEqual(summary["input_dim"], 2)
            self.assertEqual(summary["num_classes"], 2)

    def test_build_integer_preprocessing_metadata_exports_fixed_point_scaler_cost(self):
        metadata = exp.build_preprocessing_metadata(
            target_col="Attack type",
            feature_names=["feat_b", "feat_a"],
            class_names=["Blackhole", "Flooding"],
            scaler_mean=[10.0, -2.0],
            scaler_scale=[2.0, 0.5],
            split_sizes={"train": 4, "val": 1, "test": 1},
        )

        fixed = exp.build_integer_preprocessing_metadata(
            metadata,
            output_q_frac=8,
            raw_q_frac=4,
            inv_scale_q_frac=10,
        )

        self.assertEqual(fixed["input_dim"], 2)
        self.assertEqual(fixed["raw_q_frac"], 4)
        self.assertEqual(fixed["output_q_frac"], 8)
        self.assertEqual(fixed["right_shift"], 6)
        self.assertEqual(fixed["scaler_mean_q"], [160, -32])
        self.assertEqual(fixed["scaler_inv_scale_q"], [512, 2048])
        self.assertEqual(fixed["operation_counts"]["subtracts"], 2)
        self.assertEqual(fixed["operation_counts"]["multiplies"], 2)
        self.assertEqual(fixed["operation_counts"]["shifts"], 2)
        self.assertIn("StandardScaler", fixed["scope"])

    def test_write_integer_preprocessing_header_contains_no_float_scaler_constants(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "preprocess_int_metadata.h"
            fixed = {
                "input_dim": 2,
                "raw_q_frac": 4,
                "inv_scale_q_frac": 10,
                "output_q_frac": 8,
                "right_shift": 6,
                "scaler_mean_q": [160, -32],
                "scaler_inv_scale_q": [512, 2048],
                "operation_counts": {
                    "features": 2,
                    "subtracts": 2,
                    "multiplies": 2,
                    "shifts": 2,
                    "saturations": 2,
                },
            }

            summary = exp.write_integer_preprocessing_header(output_path, fixed)

            text = output_path.read_text(encoding="ascii")
            self.assertIn("#define CUKD_PREPROCESS_RAW_Q_FRAC 4", text)
            self.assertIn("#define CUKD_PREPROCESS_OUTPUT_Q_FRAC 8", text)
            self.assertIn("static const int32_t cukd_scaler_mean_q", text)
            self.assertIn("static const int32_t cukd_scaler_inv_scale_q", text)
            self.assertNotIn("float", text.lower())
            self.assertEqual(summary["input_dim"], 2)
            self.assertEqual(summary["integer_preprocess_ops_per_sample"], 8)

    def test_integer_preprocessing_c_core_compiles_with_generated_metadata(self):
        import shutil
        import subprocess

        gcc = shutil.which("gcc")
        if gcc is None:
            self.skipTest("gcc is not available")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fixed = {
                "input_dim": 2,
                "raw_q_frac": 4,
                "inv_scale_q_frac": 10,
                "output_q_frac": 8,
                "right_shift": 6,
                "scaler_mean_q": [160, -32],
                "scaler_inv_scale_q": [512, 2048],
                "operation_counts": {
                    "features": 2,
                    "subtracts": 2,
                    "multiplies": 2,
                    "shifts": 2,
                    "saturations": 2,
                },
            }
            exp.write_integer_preprocessing_header(tmp_path / "preprocess_int_metadata.h", fixed)
            source = Path(__file__).resolve().parents[1] / "hardware_export" / "wsnds_preprocess_int16.c"

            proc = subprocess.run(
                [
                    gcc,
                    "-std=c99",
                    "-Wall",
                    "-Wextra",
                    "-Os",
                    "-I",
                    str(tmp_path),
                    "-c",
                    str(source),
                    "-o",
                    str(tmp_path / "preprocess.o"),
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_write_test_vectors_header_contains_q15_inputs_predictions_and_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "test_vectors.h"
            q_inputs = [[0, 32767, -32768], [1, 2, 3]]
            labels = [2, 0]
            fp32_preds = [2, 2]
            fixed_preds = [2, 1]
            fixed_logits = [[10, 20, 30], [30, 20, 10]]

            summary = exp.write_test_vectors_header(
                output_path,
                q_inputs,
                labels,
                fp32_preds,
                fixed_preds,
                fixed_logits,
            )

            text = output_path.read_text(encoding="ascii")
            self.assertIn("#define CUKD_TEST_VECTOR_COUNT 2", text)
            self.assertIn("#define CUKD_TEST_INPUT_DIM 3", text)
            self.assertIn("cukd_test_inputs_q15", text)
            self.assertIn("cukd_test_expected_fixed_pred", text)
            self.assertEqual(summary["num_test_vectors"], 2)
            self.assertEqual(summary["input_dim"], 3)
            self.assertEqual(summary["fixed_vs_fp32_agreement"], 0.5)
            self.assertEqual(summary["fixed_accuracy_on_vectors"], 0.5)

    def test_choose_q_frac_prevents_direct_q15_saturation_for_large_standardized_values(self):
        frac = exp.choose_q_frac_for_range(43.54)

        self.assertLess(frac, 15)
        self.assertEqual(frac, 9)

    def test_compute_output_shift_tracks_input_weight_and_output_fracs(self):
        metadata = exp.compute_fixed_point_scale_metadata(
            input_frac=9,
            weight_frac=7,
            output_frac=12,
        )

        self.assertEqual(metadata["input_frac"], 9)
        self.assertEqual(metadata["weight_frac"], 7)
        self.assertEqual(metadata["output_frac"], 12)
        self.assertEqual(metadata["accum_frac"], 16)
        self.assertEqual(metadata["output_shift"], 4)

    def test_build_equivalence_report_keeps_fixed_and_fp32_metrics_separate(self):
        labels = [0, 1, 1, 2]
        fp32_preds = [0, 1, 2, 2]
        fixed_preds = [0, 0, 2, 2]
        fixed_logits = [
            [100, 0, 0],
            [100, 50, 0],
            [0, 10, 20],
            [0, 0, 30],
        ]

        report = exp.build_equivalence_report(
            labels=labels,
            fp32_preds=fp32_preds,
            fixed_preds=fixed_preds,
            fixed_logits=fixed_logits,
            metadata={"split": "test"},
        )

        self.assertEqual(report["metadata"], {"split": "test"})
        self.assertEqual(report["num_vectors"], 4)
        self.assertEqual(report["fp32_accuracy_on_vectors"], 0.75)
        self.assertEqual(report["fixed_accuracy_on_vectors"], 0.5)
        self.assertEqual(report["fixed_vs_fp32_agreement"], 0.75)
        self.assertEqual(report["fixed_logit_min"], 0)
        self.assertEqual(report["fixed_logit_max"], 100)


if __name__ == "__main__":
    unittest.main()
