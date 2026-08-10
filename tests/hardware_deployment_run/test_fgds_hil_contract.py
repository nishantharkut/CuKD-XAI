import inspect
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deployment"))

from deployment.hardware_hil.host import generate_fgds_report
from deployment.hardware_hil.host import stream_vectors_fgds_strict
from deployment.hardware_hil.host import verify_results_fgds_strict
from deployment.hardware_hil.host.preflight_fgds_hil import (
    BUNDLE_DIRS,
    EXPORT_DIRS,
    run_preflight,
)


class FgdsHilContractTests(unittest.TestCase):
    def test_protocol_constants_are_the_feature_group_contract(self):
        self.assertEqual(stream_vectors_fgds_strict.FULL_TEST_ROWS, 56301)
        self.assertEqual(stream_vectors_fgds_strict.CALIBRATION_ROWS, 262197)
        self.assertEqual(stream_vectors_fgds_strict.MAXIMUM_MACRO_F1_DROP, 0.03)
        self.assertEqual(generate_fgds_report.FULL_TEST_ROWS, 56301)
        self.assertEqual(
            generate_fgds_report.REQUIRED_STAGES,
            {"smoke_10": 10, "validation_1000": 1000, "full_56301": 56301},
        )

    def test_fgds_tools_do_not_reuse_random_row_counts_or_protocol(self):
        modules = [
            stream_vectors_fgds_strict,
            verify_results_fgds_strict,
            generate_fgds_report,
        ]
        combined = "\n".join(inspect.getsource(module) for module in modules)
        self.assertNotIn("56200", combined)
        self.assertNotIn("strict_train_only_seed42_export_v1", combined)
        self.assertIn("fgds", combined.lower())

    def test_preflight_verifies_all_generated_exports_and_bundles(self):
        missing = [
            path
            for path in [*EXPORT_DIRS.values(), *BUNDLE_DIRS.values()]
            if not path.exists()
        ]
        if missing:
            self.skipTest(f"Generated FG-DS assets are not present: {missing}")
        result = run_preflight()
        self.assertEqual(result["status"], "ready_for_compile_flash_and_hil")
        self.assertEqual(result["full_test_rows_per_board_model_pair"], 56301)
        self.assertEqual(result["planned_full_test_board_predictions"], 225204)
        self.assertEqual(set(result["exports"]), {"student_A", "student_B"})
        self.assertEqual(len(result["bundles"]), 4)
        self.assertEqual(
            len({item["bundle_id"] for item in result["bundles"].values()}), 4
        )


if __name__ == "__main__":
    unittest.main()
