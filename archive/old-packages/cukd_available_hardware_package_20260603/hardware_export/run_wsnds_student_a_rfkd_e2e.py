#!/usr/bin/env python3
"""Run the complete WSN-DS Student A RF-KD fixed-point export proof.

This orchestrates the full software E2E path:
  1. export model_weights.h from the trained FP32 PyTorch state_dict
  2. reproduce WSN-DS v2.3 preprocessing and write scaler metadata
  3. generate representative calibrated-int16 test vectors
  4. compile the dependency-free integer preprocessing and inference sources
  5. run the generated self-test and write e2e_run_report.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def run_command(cmd: list[str], cwd: Path) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-dict",
        default="origin/main:Final/wsnds_deployment_qat_outputs/tmp/E_student_A_KD_from_RF_fp32.pt",
        help="Local .pt path or git object path for the trained Student A RF-KD state_dict.",
    )
    parser.add_argument(
        "--dataset-csv",
        default="WSN-DS.csv",
        help="WSN-DS CSV path used to reproduce preprocessing and test-vector generation.",
    )
    parser.add_argument(
        "--output-dir",
        default="hardware_export/generated_student_a_rfkd_e2e",
        help="Directory for generated headers, reports, binary, and run report.",
    )
    parser.add_argument("--num-test-vectors", type=int, default=256)
    parser.add_argument("--test-vector-seed", type=int, default=42)
    parser.add_argument("--cc", default="gcc", help="C compiler for host self-test, e.g. gcc or clang.")
    parser.add_argument("--skip-compile", action="store_true", help="Only export artifacts; skip C compile and self-test.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    exporter = script_dir / "export_wsnds_student_a_rfkd_int8.py"
    export_cmd = [
        sys.executable,
        str(exporter),
        "--state-dict",
        args.state_dict,
        "--output-dir",
        str(output_dir),
        "--dataset-csv",
        args.dataset_csv,
        "--num-test-vectors",
        str(args.num_test_vectors),
        "--test-vector-seed",
        str(args.test_vector_seed),
    ]

    report: dict[str, Any] = {
        "output_dir": str(output_dir),
        "export": run_command(export_cmd, repo_root),
        "compile": None,
        "self_test": None,
        "status": "failed",
    }

    if report["export"]["returncode"] != 0:
        (output_dir / "e2e_run_report.json").write_text(json.dumps(report, indent=2), encoding="ascii")
        print(json.dumps(report, indent=2))
        return report["export"]["returncode"]

    if not args.skip_compile:
        exe_path = output_dir / "cukd_student_a_rfkd_self_test"
        compile_cmd = [
            args.cc,
            "-std=c99",
            "-Wall",
            "-Wextra",
            "-Os",
            "-I",
            str(output_dir),
            str(script_dir / "wsnds_preprocess_int16.c"),
            str(script_dir / "wsnds_student_a_rfkd_int8_inference.c"),
            str(script_dir / "wsnds_student_a_rfkd_self_test.c"),
            "-o",
            str(exe_path),
        ]
        report["compile"] = run_command(compile_cmd, repo_root)
        if report["compile"]["returncode"] == 0:
            report["self_test"] = run_command([str(exe_path)], repo_root)

    compile_ok = args.skip_compile or (report["compile"] and report["compile"]["returncode"] == 0)
    self_test_ok = args.skip_compile or (report["self_test"] and report["self_test"]["returncode"] == 0)
    report["status"] = "passed" if compile_ok and self_test_ok else "failed"

    (output_dir / "e2e_run_report.json").write_text(json.dumps(report, indent=2), encoding="ascii")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
