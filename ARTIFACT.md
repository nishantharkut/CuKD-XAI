# Artifact Guide

This guide describes how to review the repository as a research artifact during
paper writing. It is not a claim that any external artifact badge has already
been awarded.

## Artifact Scope

The artifact supports these evidence categories:

- WSN-DS predictive-compression results.
- Co-distillation and model-size trade-off evidence.
- SHAP-based explanation-rank audit.
- ONNX, ONNX Runtime, OpenVINO, and dynamic INT8 runtime evidence.
- Fixed-point C export and integer preprocessing metadata.
- ESP32-C3 and Arduino R4 hardware-in-loop replay logs.
- MSP430F1611/TelosB-class memory-feasibility compile evidence.
- Edge-IIoT supporting robustness and literature-comparison evidence.

## Review Entry Points

| Review question | Start here |
|---|---|
| What is the project and complete evidence chain? | `docs/research/PROJECT_TECHNICAL_BRIEF.md` |
| What exact claims are supported by files? | `docs/research/RESULTS_AND_EVIDENCE.md` |
| What are the final WSN-DS tables and figures? | `results/wsnds/final_results/2026-05-30-10seed-plus-j/` |
| What did hardware replay prove? | `results/hardware_hil/reports/final_postprocessing/final_postprocessing_analysis.md` |
| What did Edge-IIoT add? | `results/edge_iiot/literature_metric_gap/edgeiiot_literature_metric_comparison.md` |
| How is the repository organized? | `docs/repository/REPOSITORY_MAP.md` |

## Reproducibility Status

| Level | Status |
|---|---|
| Evidence inspection | Supported by tracked reports, CSVs, JSON files, and figures. |
| Smoke testing | Supported by `pytest` and Python compile checks. |
| Post-processing regeneration | Supported for HIL evidence and Edge-IIoT metric-gap tables. |
| Full training rerun | Code is preserved, but full reruns are compute- and data-dependent. |
| Hardware replay rerun | Requires ESP32-C3 or Arduino R4, flashed firmware, and USB serial access. |
| Live WSN deployment | Not included in this artifact. |

## Minimal Review Commands

```powershell
py -3.11 -m pytest -q
py -3.11 -m compileall -q experiments deployment tests
```

The repository intentionally does not include a Dockerfile. The active artifact
contains notebooks, Python scripts, fixed-point firmware exports, generated
hardware bundles, and measured outputs rather than a single service container.

## Claim Boundary

Use this artifact to support paper writing and reviewer discussion around
resource-aware explainable IDS compression. Do not use it to claim live WSN
packet capture, energy measurement, physical TelosB deployment, or full
packet-to-feature extraction.
