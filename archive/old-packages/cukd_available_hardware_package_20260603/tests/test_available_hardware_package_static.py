import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "hardware_export" / "AVAILABLE_HARDWARE_TEST_PLAN.md"
SKETCH = (
    ROOT
    / "hardware_export"
    / "arduino_esp32_student_a_rfkd_self_test"
    / "arduino_esp32_student_a_rfkd_self_test.ino"
)
SKETCH_README = (
    ROOT
    / "hardware_export"
    / "arduino_esp32_student_a_rfkd_self_test"
    / "README.md"
)
PACKAGER = ROOT / "hardware_export" / "prepare_arduino_esp32_package.py"


class AvailableHardwarePackageStaticTests(unittest.TestCase):
    def test_hardware_plan_exists_and_is_conservative(self):
        text = PLAN.read_text(encoding="utf-8")
        for token in [
            "ESP32",
            "Arduino",
            "Raspberry Pi",
            "microcontroller proxy",
            "not a TelosB deployment claim",
            "WSN mote",
            "256",
            "1,000",
        ]:
            self.assertIn(token, text)

    def test_arduino_esp32_sketch_uses_fixed_point_core_and_serial_report(self):
        text = SKETCH.read_text(encoding="utf-8")
        for token in [
            '#include "model_weights.h"',
            '#include "test_vectors.h"',
            "extern \"C\"",
            "cukd_forward_q15",
            "cukd_predict_q15",
            "CUKD_TEST_VECTOR_COUNT",
            "Serial.begin",
            "micros()",
            "prediction_failures",
            "logit_failures",
            "predict_wrapper_failures",
        ]:
            self.assertIn(token, text)
        self.assertNotIn("float", text)
        self.assertNotIn("double", text)

    def test_packager_copies_required_artifacts(self):
        source = PACKAGER.read_text(encoding="utf-8")
        compile(source, str(PACKAGER), "exec")
        for token in [
            "model_weights.h",
            "test_vectors.h",
            "wsnds_student_a_rfkd_int8_inference.c",
            "arduino_esp32_student_a_rfkd_self_test.ino",
            "--generated-dir",
            "--output-dir",
        ]:
            self.assertIn(token, source)

    def test_packager_creates_arduino_valid_sketch_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            generated = tmp_path / "generated"
            output = tmp_path / "cukd_hw_sketch"
            generated.mkdir()
            (generated / "model_weights.h").write_text(
                "#define CUKD_INPUT_DIM 17\n#define CUKD_OUTPUT_DIM 5\n",
                encoding="ascii",
            )
            (generated / "test_vectors.h").write_text(
                "#define CUKD_TEST_VECTOR_COUNT 256\n",
                encoding="ascii",
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    str(PACKAGER),
                    "--generated-dir",
                    str(generated),
                    "--output-dir",
                    str(output),
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue((output / "cukd_hw_sketch.ino").exists())
            self.assertTrue((output / "model_weights.h").exists())
            self.assertTrue((output / "test_vectors.h").exists())
            self.assertTrue((output / "wsnds_student_a_rfkd_int8_inference.c").exists())
            manifest = json.loads((output / "package_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["sketch_file"], "cukd_hw_sketch.ino")
            self.assertEqual(manifest["vector_count"], 256)

    def test_sketch_readme_documents_run_steps(self):
        text = SKETCH_README.read_text(encoding="utf-8")
        for token in [
            "Arduino IDE",
            "ESP32",
            "Serial Monitor",
            "115200",
            "prediction_failures = 0",
            "logit_failures = 0",
            "predict_wrapper_failures = 0",
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
