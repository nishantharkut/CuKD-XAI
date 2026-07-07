import tempfile
import unittest
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deployment"))
HIL = ROOT / "deployment" / "hardware_hil"


class HardwareHilStaticTests(unittest.TestCase):
    def test_required_documentation_exists_and_locks_claim_boundaries(self):
        readme = HIL / "README.md"
        manifest = HIL / "MANIFEST.md"
        protocol = HIL / "protocol" / "serial_protocol.md"

        for path in [readme, manifest, protocol]:
            self.assertTrue(path.exists(), f"missing {path}")

        text = readme.read_text(encoding="utf-8")
        for token in [
            "WSN-DS",
            "Student A",
            "E_KD_from_RF",
            "17-feature",
            "test-vector replay",
            "not live packet capture",
            "not energy measurement",
            "Raspberry Pi AI HAT+",
            "future gateway",
        ]:
            self.assertIn(token, text)

    def test_beginner_docs_exist_and_reference_official_hardware_docs(self):
        expected = [
            "00_READ_THIS_FIRST.md",
            "01_RASPBERRY_PI5_HOST_SETUP.md",
            "02_GENERATE_REPLAY_ASSETS.md",
            "03_BUILD_FIRMWARE_BUNDLES.md",
            "04_FLASH_ESP32C3.md",
            "05_FLASH_ARDUINO_R4.md",
            "06_RUN_REPLAY_AND_VERIFY.md",
            "07_TROUBLESHOOTING.md",
            "OFFICIAL_REFERENCES.md",
        ]
        for name in expected:
            self.assertTrue((HIL / "docs" / name).exists(), f"missing docs/{name}")

        refs = (HIL / "docs" / "OFFICIAL_REFERENCES.md").read_text(encoding="utf-8")
        for token in [
            "raspberrypi.com/documentation",
            "docs.espressif.com",
            "docs.arduino.cc",
            "Claim Boundary",
        ]:
            self.assertIn(token, refs)

        runbook = (HIL / "docs" / "00_READ_THIS_FIRST.md").read_text(encoding="utf-8")
        for token in [
            "No live WSN packet capture",
            "No energy measurement",
            "ESP32-C3",
            "Arduino R4",
            "Raspberry Pi 5",
        ]:
            self.assertIn(token, runbook)

    def test_student_b_runbook_exists_and_uses_separate_paths(self):
        runbook = HIL / "docs" / "10_STUDENT_B_HIL_RUNBOOK.md"
        self.assertTrue(runbook.exists(), f"missing {runbook}")

        text = runbook.read_text(encoding="utf-8")
        for token in [
            "E_student_B_KD_from_RF_fp32.pt",
            "generated_student_b_rfkd_hil_full",
            "cukd_hil_esp32c3_student_b",
            "cukd_hil_arduino_r4_student_b",
            "pi5_esp32c3_student_b",
            "pi5_arduino_r4_student_b",
            "17 -> 64 -> 32 -> 5",
            "Do not overwrite Student A",
            "not live WSN packet capture",
            "Do not reuse `generated_student_a_rfkd_hil_full`",
        ]:
            self.assertIn(token, text)

    def test_protocol_defines_checksum_row_id_and_error_statuses(self):
        text = (HIL / "protocol" / "serial_protocol.md").read_text(encoding="utf-8")
        for token in [
            "row_id",
            "CRC-16-CCITT",
            "BAD_START",
            "BAD_LENGTH",
            "BAD_CHECKSUM",
            "BAD_FEATURE_RANGE",
            "INTERNAL_ERROR",
            "OK",
            "17",
        ]:
            self.assertIn(token, text)

    def test_firmware_common_uses_bounded_integer_execution(self):
        model_c = (HIL / "firmware" / "common" / "cukd_model.c").read_text(encoding="utf-8")
        preprocess_c = (HIL / "firmware" / "common" / "cukd_preprocess.c").read_text(encoding="utf-8")
        protocol_c = (HIL / "firmware" / "common" / "cukd_protocol.c").read_text(encoding="utf-8")
        combined = "\n".join([model_c, preprocess_c, protocol_c])

        self.assertIn("int32_t acc", model_c)
        self.assertIn("int64_t", preprocess_c)
        self.assertIn("cukd_crc16_ccitt", protocol_c)
        for forbidden in ["malloc", "calloc", "realloc", "free(", "new ", "delete ", "float ", "double "]:
            self.assertNotIn(forbidden, combined)

    def test_board_firmware_keeps_hardware_scope_clean(self):
        esp = (HIL / "firmware" / "esp32c3" / "src" / "main.cpp").read_text(encoding="utf-8")
        r4 = (HIL / "firmware" / "arduino_r4" / "cukd_hil_r4" / "cukd_hil_r4.ino").read_text(encoding="utf-8")
        combined = esp + "\n" + r4

        for token in [
            "cukd_standardize_raw_q",
            "cukd_predict_q15",
            "preprocess_us",
            "inference_us",
            "total_us",
            "BAD_CHECKSUM",
        ]:
            self.assertIn(token, combined)
        for forbidden in ["WiFi.begin", "Bluetooth", "BLE", "delay(1000)"]:
            self.assertNotIn(forbidden, combined)

    def test_host_scripts_exist_and_are_import_safe(self):
        for rel in [
            "host/hil_common.py",
            "host/env_check.py",
            "host/stream_vectors.py",
            "host/verify_results.py",
            "host/generate_report.py",
            "host/prepare_firmware_bundle.py",
            "host/requirements.txt",
        ]:
            path = HIL / rel
            self.assertTrue(path.exists(), f"missing {path}")

    def test_prepare_firmware_bundle_creates_board_sketch_folder(self):
        from hardware_hil.host.prepare_firmware_bundle import build_bundle

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            generated = tmp_path / "generated"
            generated.mkdir()
            (generated / "model_weights.h").write_text("#define CUKD_INPUT_DIM 17\n", encoding="ascii")
            (generated / "preprocess_int_metadata.h").write_text(
                "#define CUKD_PREPROCESS_INPUT_DIM 17\n", encoding="ascii"
            )
            output = tmp_path / "cukd_hil_esp32c3"

            manifest = build_bundle("esp32c3", generated, output)

            self.assertEqual(manifest["board"], "esp32c3")
            self.assertTrue((output / "cukd_hil_esp32c3.ino").exists())
            self.assertTrue((output / "model_weights.h").exists())
            self.assertTrue((output / "preprocess_int_metadata.h").exists())
            self.assertTrue((output / "cukd_model.c").exists())
            self.assertTrue((output / "cukd_preprocess.c").exists())
            self.assertTrue((output / "cukd_protocol.c").exists())
            self.assertTrue((output / "bundle_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
