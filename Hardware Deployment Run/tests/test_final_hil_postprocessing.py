import importlib.util
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "hardware_hil"
    / "host"
    / "analyze_final_hil_evidence.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("analyze_final_hil_evidence", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FinalHilPostprocessingTest(unittest.TestCase):
    def test_parse_arduino_compile_log_extracts_flash_and_ram(self):
        module = load_module()
        log = """Sketch uses 281192 bytes (21%) of program storage space. Maximum is 1310720 bytes.
Global variables use 13556 bytes (4%) of dynamic memory, leaving 314124 bytes for local variables. Maximum is 327680 bytes.
"""
        parsed = module.parse_compile_log(log)
        self.assertEqual(parsed["program_bytes"], 281192)
        self.assertEqual(parsed["program_percent"], 21)
        self.assertEqual(parsed["program_max_bytes"], 1310720)
        self.assertEqual(parsed["global_bytes"], 13556)
        self.assertEqual(parsed["global_percent"], 4)
        self.assertEqual(parsed["global_max_bytes"], 327680)

    def test_cycles_per_mac_uses_inference_latency_not_total_latency(self):
        module = load_module()
        result = module.compute_efficiency(
            mean_inference_us=112.31238434163701,
            mean_total_us=118.403256227758,
            clock_mhz=160.0,
            macs=1136,
        )
        self.assertEqual(round(result["inference_cycles"]), 17970)
        self.assertEqual(round(result["cycles_per_mac"], 2), 15.82)
        self.assertEqual(round(result["total_throughput_per_s"], 1), 8445.7)

    def test_missing_hil_metric_is_recorded_in_traceability(self):
        module = load_module()
        root = Path('/tmp/cukd_empty_final_hil_root')
        if root.exists():
            import shutil
            shutil.rmtree(root)
        root.mkdir(parents=True)
        analysis = module.build_analysis(root)
        statuses = {row["claim"]: row["status"] for row in analysis["evidence_rows"]}
        self.assertEqual(
            statuses["Student B RF-KD ESP32-C3 DevKitM-1 HIL metrics"],
            "missing_required",
        )
        self.assertEqual(len(analysis["compile_rows"]), 4)

    def test_unparseable_compile_log_is_recorded_as_parse_failed(self):
        module = load_module()
        root = Path('/tmp/cukd_parse_fail_final_hil_root')
        if root.exists():
            import shutil
            shutil.rmtree(root)
        metrics_dir = root / "hardware_hil" / "results" / "pi5_esp32c3"
        metrics_dir.mkdir(parents=True)
        (metrics_dir / "full_56200_metrics.json").write_text(
            '{"accuracy": 1, "macro_f1": 1, "weighted_f1": 1, "completed_vectors": 1, '
            '"mcu_vs_fixed_reference_agreement": 1, "mcu_vs_fp32_agreement": 1, '
            '"latency": {"total_us": {"mean": 1, "p99": 1}, "inference_us": {"mean": 1}}}',
            encoding="utf-8",
        )
        log_dir = root / "hardware_hil" / "compile_logs"
        log_dir.mkdir(parents=True)
        (log_dir / "esp32c3_student_a_compile.txt").write_text("bad log", encoding="utf-8")
        analysis = module.build_analysis(root)
        statuses = {row["claim"]: row["status"] for row in analysis["evidence_rows"]}
        self.assertEqual(
            statuses["Student A RF-KD ESP32-C3 DevKitM-1 compile summary"],
            "parse_failed",
        )

    def test_serial_baseline_compile_log_is_recorded_in_traceability(self):
        module = load_module()
        root = Path('/tmp/cukd_baseline_trace_final_hil_root')
        if root.exists():
            import shutil
            shutil.rmtree(root)
        log_dir = root / "hardware_hil" / "compile_logs"
        log_dir.mkdir(parents=True)
        (log_dir / "esp32c3_serial_baseline_compile.txt").write_text(
            "Sketch uses 250000 bytes (19%) of program storage space. Maximum is 1310720 bytes.\n"
            "Global variables use 12000 bytes (3%) of dynamic memory, leaving 0 bytes for local variables. Maximum is 327680 bytes.\n",
            encoding="utf-8",
        )
        analysis = module.build_analysis(root)
        statuses = {row["claim"]: row["status"] for row in analysis["evidence_rows"]}
        self.assertEqual(
            statuses["ESP32-C3 DevKitM-1 serial baseline compile summary"],
            "present",
        )
        self.assertEqual(
            statuses["Arduino R4 WiFi serial baseline compile summary"],
            "missing_optional",
        )


if __name__ == "__main__":
    unittest.main()
