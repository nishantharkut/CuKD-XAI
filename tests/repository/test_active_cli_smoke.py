import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

CLI_HELP_SCRIPTS = [
    "deployment/firmware_export/wsnds_rfkd_hil/export_wsnds_student_a_rfkd_int8.py",
    "deployment/firmware_export/wsnds_rfkd_hil/run_wsnds_student_a_rfkd_e2e.py",
    "deployment/firmware_export/wsnds_rfkd_hil/prepare_arduino_esp32_package.py",
    "deployment/msp430/export_wsnds_student_a_rfkd_int8.py",
    "deployment/msp430/run_wsnds_student_a_rfkd_e2e.py",
    "deployment/hardware_hil/host/analyze_final_hil_evidence.py",
    "deployment/hardware_hil/host/env_check.py",
    "deployment/hardware_hil/host/prepare_firmware_bundle.py",
    "deployment/hardware_hil/host/stream_vectors.py",
    "deployment/hardware_hil/host/verify_results.py",
    "deployment/hardware_hil/host/generate_report.py",
    "experiments/edge_iiot/literature_comparable/edgeiiot_literature_metric_gap_analysis.py",
]

REQUIRED_EVIDENCE_FILES = [
    "README.md",
    "ARTIFACT.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "LICENSE",
    "NOTICE.md",
    "requirements.txt",
    "docs/research/PROJECT_TECHNICAL_BRIEF.md",
    "docs/research/RESULTS_AND_EVIDENCE.md",
    "docs/repository/REPOSITORY_MAP.md",
    "docs/repository/PATH_REFERENCE_AUDIT.md",
    "results/hardware_hil/reports/final_postprocessing/final_postprocessing_analysis.md",
    "results/hardware_hil/reports/final_postprocessing/hil_fidelity.csv",
    "results/hardware_hil/reports/final_postprocessing/compile_framework_baseline.csv",
    "results/wsnds/final_results/2026-05-30-10seed-plus-j/wsnds_results_student_A.csv",
    "results/wsnds/final_results/2026-05-30-10seed-plus-j/wsnds_results_student_B.csv",
    "results/edge_iiot/literature_metric_gap/edgeiiot_literature_metric_comparison.md",
]


class ActiveCliSmokeTests(unittest.TestCase):
    def test_active_cli_scripts_show_help(self):
        failures = []
        for rel_script in CLI_HELP_SCRIPTS:
            script = ROOT / rel_script
            proc = subprocess.run(
                [sys.executable, str(script), "--help"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            if proc.returncode != 0:
                failures.append(
                    f"{rel_script}: rc={proc.returncode}; stderr={proc.stderr[:500]}"
                )
                continue
            output = (proc.stdout + proc.stderr).lower()
            if "usage:" not in output:
                failures.append(f"{rel_script}: help output did not contain usage")
        self.assertEqual(failures, [])

    def test_required_research_evidence_files_exist(self):
        missing = [
            rel_path
            for rel_path in REQUIRED_EVIDENCE_FILES
            if not (ROOT / rel_path).is_file()
        ]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
