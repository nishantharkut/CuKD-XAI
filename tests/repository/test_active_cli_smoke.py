import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

CLI_HELP_SCRIPTS = [
    "experiments/wsnds/leakage_free_rerun/run_feature_group_10seed_confirmation.py",
    "experiments/wsnds/leakage_free_rerun/analyze_feature_group_confirmation.py",
    "experiments/wsnds/leakage_free_rerun/run_leftover_e2e_closure.py",
    "experiments/edge_iiot/audit_edgeiiot_split_duplicates.py",
    "deployment/firmware_export/wsnds_rfkd_hil/export_fgds_runtime.py",
]

CLI_HELP_MODULES = [
    "experiments.wsnds.evidence_completion.run_fgds_full_routes",
    "experiments.wsnds.evidence_completion.continue_fgds_full_routes",
    "experiments.wsnds.evidence_completion.finalize_fgds_full_routes",
    "experiments.wsnds.evidence_completion.analyze_fgds_group_balanced_routes",
    "experiments.wsnds.evidence_completion.analyze_fgds_behavioral_transfer_logits",
    "experiments.wsnds.evidence_completion.run_fgds_multisplit_core_confirmation",
    "experiments.wsnds.evidence_completion.run_fgds_rfkd_hyperparameter_sensitivity",
    "experiments.wsnds.evidence_completion.run_fgds_exact_teacher_shap",
    "experiments.wsnds.evidence_completion.run_fgds_controlled_xai_transfer",
    "experiments.wsnds.evidence_completion.run_fgds_fixed_point_refinement",
    "experiments.evidence.build_fgds_evidence_registry",
    "deployment.firmware_export.wsnds_final_hil.export_final_seed42",
    "deployment.firmware_export.wsnds_final_hil.audit_all_seeds",
    "deployment.final_hil",
]

REQUIRED_EVIDENCE_FILES = [
    "README.md",
    "ARTIFACT.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "LICENSE",
    "NOTICE.md",
    "requirements.txt",
    "docs/repository/REPOSITORY_MAP.md",
    "research_history/documentation_snapshots/repository_restructure/PATH_REFERENCE_AUDIT.md",
    "results/evidence_registry/fgds_20260814_current/EVIDENCE_REGISTRY.md",
    "results/evidence_registry/fgds_20260814_current/evidence_registry.json",
    "results/evidence_registry/fgds_20260814_current/claim_boundaries.csv",
    "results/evidence_registry/fgds_20260814_current/artifact_manifest.json",
    "results/wsnds/confirmation_runs_v2/local_feature_group_10seed_20260811/feature_group_10seed_analysis/feature_group_10seed_analysis.json",
    "results/wsnds/evidence_completion_20260811/fgds_controlled_full_routes_10seed_v2/aggregate_results.json",
    "results/wsnds/evidence_completion_20260812/fgds_behavioral_transfer_logits_10seed_v5/behavioral_transfer_summary.json",
    "results/wsnds/evidence_completion_20260811/fgds_seed42_reconstructed_teacher_shap_v3/shap_report.json",
    "results/runtime/onnx_openvino/wsnds/fgds_seed42_exact/runtime_report.json",
    "results/wsnds/evidence_completion_20260813/fgds_all_seed_fixed_point_audit_v1/all_seed_fixed_point_report.json",
    "results/hardware_hil/final_fgds_seed42_v1/final_campaign_usb_v1/final_hil_summary.json",
    "deployment/msp430/current_fgds_static/artifacts/msp430_static_summary.json",
    "results/leftover_e2e_closure/04_edge_group_aware/edge_group_aware_summary.json",
]


class ActiveCliSmokeTests(unittest.TestCase):
    def test_active_cli_scripts_show_help(self):
        failures = []
        commands = [
            (rel_script, [sys.executable, str(ROOT / rel_script), "--help"])
            for rel_script in CLI_HELP_SCRIPTS
        ]
        commands.extend(
            (module, [sys.executable, "-m", module, "--help"])
            for module in CLI_HELP_MODULES
        )
        for target, command in commands:
            proc = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=60,
                check=False,
            )
            if proc.returncode != 0:
                failures.append(
                    f"{target}: rc={proc.returncode}; stderr={proc.stderr[:500]}"
                )
                continue
            output = (proc.stdout + proc.stderr).lower()
            if "usage:" not in output:
                failures.append(f"{target}: help output did not contain usage")
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
