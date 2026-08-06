#!/usr/bin/env python3
"""Validate manuscript coverage and source-derived numerical anchors."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT = ROOT / "manuscript"
MAIN = MANUSCRIPT / "main.tex"
BIB = MANUSCRIPT / "references.bib"
GENERATED = MANUSCRIPT / "generated"
TRACEABILITY = MANUSCRIPT / "CLAIM_TRACEABILITY.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def row_for(rows: list[dict[str, str]], config: str) -> dict[str, str]:
    return next(row for row in rows if row["Config"] == config)


def require(text: str, token: str, category: str) -> None:
    if token not in text:
        raise RuntimeError(f"Missing {category}: {token}")


def validate_citations(main: str, bib: str) -> tuple[int, int]:
    cited: set[str] = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", main):
        cited.update(key.strip() for key in group.split(","))
    available = set(re.findall(r"@\w+\{([^,]+),", bib))
    missing = sorted(cited - available)
    if missing:
        raise RuntimeError(f"Citations missing from references.bib: {missing}")
    return len(cited), len(available)


def validate_traceability_paths() -> int:
    text = TRACEABILITY.read_text(encoding="utf-8")
    checked: set[str] = set()
    for token in re.findall(r"`([^`]+)`", text):
        if "/" not in token and "\\" not in token:
            continue
        candidate = Path(token)
        if token.startswith(("generated/", "scripts/")):
            candidate = MANUSCRIPT / candidate
        else:
            candidate = ROOT / candidate
        if not candidate.exists():
            raise RuntimeError(f"Traceability path does not exist: {token}")
        checked.add(token)
    return len(checked)


def main() -> int:
    subprocess.run([sys.executable, str(Path(__file__).with_name("build_evidence.py"))], check=True)

    source = MAIN.read_text(encoding="utf-8")
    bibliography = BIB.read_text(encoding="utf-8")
    traceability_path_count = validate_traceability_paths()

    required_sections = [
        "Introduction",
        "Related Work",
        "System Overview and Research Design",
        "Experimental Evaluation",
        "Results",
        "Comparison with Prior Work",
        "Discussion",
        "Threats to Validity",
        "Reproducibility and Artifact Traceability",
        "Conclusion",
        "Complete Result Matrices",
    ]
    for section in required_sections:
        require(source, section, "section")

    required_generated_inputs = [
        "wsn_student_a_all_rows.tex",
        "wsn_student_b_all_rows.tex",
        "wsn_selected_per_class_rows.tex",
        "wsn_codistill_wilcoxon_rows.tex",
        "shap_rank_rows.tex",
        "qat_all_rows.tex",
        "runtime_all_rows.tex",
        "hil_fidelity_rows.tex",
        "hil_cycles_rows.tex",
        "hil_compile_rows.tex",
        "hil_model_drift_rows.tex",
        "edge_strict_all_rows.tex",
        "edge_comparable_all_rows.tex",
    ]
    for name in required_generated_inputs:
        require(source, f"generated/{name}", "generated evidence input")
        if not (GENERATED / name).is_file():
            raise RuntimeError(f"Generated evidence file does not exist: {name}")

    wsn_dir = ROOT / "results" / "wsnds" / "final_results" / "2026-05-30-10seed-plus-j"
    wsn_a = read_csv(wsn_dir / "wsnds_results_student_A.csv")
    wsn_b = read_csv(wsn_dir / "wsnds_results_student_B.csv")
    a_rfkd = row_for(wsn_a, "E_KD_from_RF")
    b_codistill = row_for(wsn_b, "J_CoDistill_RF_CL")
    require(source, f"{float(a_rfkd['MacroF1_mean']):.5f}", "Student A RF-KD macro-F1")
    require(source, f"{float(b_codistill['MacroF1_mean']):.5f}", "Student B co-distillation macro-F1")

    shap = json.loads((wsn_dir / "cukd_xai_results_with_J.json").read_text(encoding="utf-8"))["shap_results"]
    require(source, "\\ShapRho", "SHAP rank agreement macro")
    require(source, "\\ShapP", "SHAP rank p-value macro")
    if len(shap["student_global_importance"]) != 17 or len(shap["teacher_global_importance"]) != 17:
        raise RuntimeError("The archived SHAP audit does not contain all 17 features")

    wilcoxon_tex = (GENERATED / "wsn_codistill_wilcoxon_rows.tex").read_text(encoding="ascii")
    wilcoxon_rows = [line for line in wilcoxon_tex.splitlines() if line.strip()]
    if len(wilcoxon_rows) != 10:
        raise RuntimeError(f"Expected 10 unique co-distillation comparisons, found {len(wilcoxon_rows)}")
    for adjusted_p in ("0.0195", "0.0352", "0.0781"):
        require(wilcoxon_tex, adjusted_p, "Holm-adjusted Wilcoxon anchor")
    require(source, "p_{\\mathrm{Holm}}", "Holm-adjusted inference label")

    hil_rows = read_csv(
        ROOT / "results" / "hardware_hil" / "reports" / "final_postprocessing" / "hil_fidelity.csv"
    )
    latencies = [float(row["mean_total_us"]) for row in hil_rows]
    require(source, f"{min(latencies):.2f}--{max(latencies):.2f}", "four-pair HIL latency range")

    runtime_dir = ROOT / "results" / "runtime" / "onnx_openvino" / "wsnds"
    deployment_results = json.loads((runtime_dir / "wsnds_deployment_results.json").read_text(encoding="utf-8"))
    proof_seed = int(deployment_results["proof_seed"])
    required_rfkd_models = {"E_student_A_KD_from_RF", "E_student_B_KD_from_RF"}
    proof_rows = {
        row["model_name"]: row
        for row in deployment_results["results"]
        if row["model_name"] in required_rfkd_models and row["variant"] == "fp32"
    }
    if set(proof_rows) != required_rfkd_models or any(int(row["proof_seed"]) != proof_seed for row in proof_rows.values()):
        raise RuntimeError("RF-KD deployment artifacts do not share the recorded proof seed")
    export_root = ROOT / "deployment" / "firmware_export" / "wsnds_rfkd_hil"
    export_dirs = [
        export_root / "generated_student_a_rfkd_hil_full",
        export_root / "generated_student_b_rfkd_hil_full",
    ]
    expected_sources = ["E_student_A_KD_from_RF_fp32.pt", "E_student_B_KD_from_RF_fp32.pt"]
    for export_dir, expected_source in zip(export_dirs, expected_sources, strict=True):
        export = json.loads((export_dir / "export_summary.json").read_text(encoding="utf-8"))
        if Path(export["source"]).name != expected_source:
            raise RuntimeError(f"Unexpected HIL model source: {export['source']}")
        preprocess = json.loads((export_dir / "preprocess_metadata.json").read_text(encoding="utf-8"))
        if "seed-42 stratified 70/15/15 split" not in preprocess["preprocessing_contract"]:
            raise RuntimeError("HIL preprocessing metadata does not record the seed-42 split")
        metadata = export["e2e"]["equivalence_report"]["metadata"]
        indices = metadata["selected_indices"]
        if int(metadata["test_vector_seed"]) != 42:
            raise RuntimeError("Unexpected HIL test-vector seed")
        if indices != list(range(56200)):
            raise RuntimeError("Full HIL export does not contain ordered indices 0--56,199")
    require(source, f"model seed {proof_seed}", "HIL model proof seed")
    require(source, "test-vector seed 42", "HIL exporter seed")
    require(source, "0--56,199", "full HIL replay index range")

    strict = read_csv(ROOT / "results" / "edge_iiot" / "strict_generalization" / "edgeiiot_v23_config_rankings.csv")
    comparable = read_csv(
        ROOT / "results" / "edge_iiot" / "literature_comparable" / "edgeiiot_v23_config_rankings.csv"
    )
    strict_best = max(float(row["macro_f1_mean"]) for row in strict if row["params"])
    comparable_best = max(float(row["macro_f1_mean"]) for row in comparable if row["params"])
    require(source, f"{strict_best:.4f}", "strict Edge-IIoT compact macro-F1")
    require(source, f"{comparable_best:.4f}", "literature-oriented Edge-IIoT compact macro-F1")

    for forbidden in ("Lorem ipsum", "safe paper claim", "TODO", "FIXME"):
        if forbidden.lower() in source.lower():
            raise RuntimeError(f"Draft placeholder or meta-writing phrase remains: {forbidden}")

    cited_count, bibliography_count = validate_citations(source, bibliography)
    manifest = json.loads((GENERATED / "evidence_manifest.json").read_text(encoding="ascii"))
    report = {
        "status": "passed",
        "manuscript": str(MAIN.relative_to(ROOT)).replace("\\", "/"),
        "sections_checked": len(required_sections),
        "generated_inputs_checked": len(required_generated_inputs),
        "traceability_paths_checked": traceability_path_count,
        "citations_used": cited_count,
        "bibliography_entries": bibliography_count,
        "evidence_sources_hashed": len(manifest["sources"]),
        "source_invariants": manifest["invariants"],
        "numeric_anchors": {
            "wsn_student_a_rfkd_macro_f1": float(a_rfkd["MacroF1_mean"]),
            "wsn_student_b_codistill_macro_f1": float(b_codistill["MacroF1_mean"]),
            "hil_latency_min_us": min(latencies),
            "hil_latency_max_us": max(latencies),
            "hil_model_proof_seed": proof_seed,
            "hil_test_vector_seed": 42,
            "edge_strict_best_compact_macro_f1": strict_best,
            "edge_literature_best_compact_macro_f1": comparable_best,
            "shap_features": len(shap["student_global_importance"]),
            "holm_adjusted_comparisons": len(wilcoxon_rows),
        },
    }
    (GENERATED / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    print(GENERATED / "validation_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
