# CuKD-XAI Manuscript

This directory contains the evidence-backed IEEE journal manuscript for the complete CuKD-XAI research program. The paper covers the primary WSN-DS experiments, explanation audit, software conversion, fixed-point deployment, four hardware-in-the-loop replays, MSP430F1611 cross-compilation, and two Edge-IIoTset robustness protocols. It is not a summary assembled from manually copied headline values: result tables and plots are generated from the repository's CSV and JSON evidence.

## Manuscript Contents

| Component | Coverage |
|---|---|
| Research design | Five research questions spanning predictive compression, training routes, explanation fidelity, executable deployment, and cross-dataset robustness |
| WSN-DS modeling | Full ten-seed Student A and Student B route matrices, per-class F1, model size, compression ratios, and Holm-adjusted paired Wilcoxon analyses |
| Explainability | All 17 RF-reference/student SHAP feature ranks, Spearman agreement, and repeated-subsample sensitivity |
| Software deployment | All stored PyTorch QAT, ONNX Runtime FP32/dynamic-INT8, and OpenVINO rows |
| Fixed-point deployment | Model-only footprint, quantization drift, cycles/MAC, throughput, compiler footprint, and full-test-set HIL fidelity |
| Hardware targets | ESP32-C3 and Arduino UNO R4 WiFi for Student A and Student B; MSP430F1611 cross-compile feasibility for Student A |
| Robustness | Complete strict 43-feature and literature-oriented 49-feature Edge-IIoTset matrices |
| Prior work | Dataset-specific performance and deployment positioning with metric and protocol boundaries preserved |
| Validity | Pre-split WSN-DS scaling, fixed-split seed interpretation, low-powered adjusted statistics, single-pair SHAP scope, and replay-only HIL scope |

## Files

| Path | Role |
|---|---|
| `main.tex` | IEEEtran manuscript source |
| `main.pdf` | Compiled manuscript |
| `references.bib` | Bibliography used by the manuscript |
| `CLAIM_TRACEABILITY.md` | Claim-to-artifact ledger |
| `scripts/build_evidence.py` | Deterministically builds manuscript tables, figures, and a SHA-256 source manifest |
| `scripts/validate_manuscript.py` | Checks result-family coverage, numerical anchors, citations, HIL sequence integrity, and compiler-log consistency |
| `generated/` | Machine-generated TeX rows, figures, evidence manifest, and validation report |

## Rebuild and Validate

Run from the repository root:

```powershell
python manuscript/scripts/validate_manuscript.py

Push-Location manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
Pop-Location
```

The validation command first regenerates all evidence-derived content. It fails if any of the following changes unexpectedly:

- WSN-DS result-row counts;
- QAT or ONNX/OpenVINO result-row counts;
- strict or literature-oriented Edge-IIoTset result-row counts;
- any full HIL replay's expected/completed count, sequence integrity, or status count;
- the proof-run model seed, exporter seed, and full ordered HIL index range;
- Arduino compiler footprint values or their serial-baseline deltas;
- all ten unique co-distillation comparisons and selected Holm-adjusted anchors;
- citation keys, required result sections, generated table inputs, or selected numerical anchors.

The resulting provenance records are:

- `generated/evidence_manifest.json`: byte size and SHA-256 digest for every source artifact consumed by the paper;
- `generated/validation_report.json`: checked invariants, citation counts, and source-derived numerical anchors.

Do not manually edit files under `generated/`. Change the source artifact or generator and rerun validation.

## Interpretation Boundaries

The paper distinguishes five evidence levels:

1. ten-seed predictive evaluation on archived tabular splits;
2. post-hoc explanation comparison for one preserved Student A curriculum-KD/RF-reference pair;
3. software-format conversion and host-runtime benchmarking for preserved artifacts;
4. fixed-point model-core execution through USB replay on two MCU boards;
5. MSP430 memory feasibility by cross-compilation.

These levels do not establish live WSN packet capture, packet-to-feature extraction, radio-stack integration, physical TelosB execution, or measured board energy. The WSN-DS code audit found that `StandardScaler` was fitted before the split for the archived multi-seed tables; the manuscript labels that lineage explicitly. A train-only-scaler seed-42 RF-KD confirmation package now regenerates deployment weights, ONNX host runtime, fixed-point exports, and four full-test HIL pairs (see `CLAIM_TRACEABILITY.md` and `results/runtime/onnx_openvino/wsnds/train_only_seed42_copy/train_only_tier15_completeness.json`). That confirmation is single-seed and does not replace the ten-seed distribution.

## Submission Metadata

The current author block contains only the identity and affiliation available in repository metadata. Author order, affiliations, corresponding-author designation, venue-specific formatting, and acknowledgments must be confirmed by the research team before submission. No coauthor metadata has been inferred.
