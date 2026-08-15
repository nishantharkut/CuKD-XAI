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
| What is the current evidence chain? | `results/evidence_registry/fgds_20260814_current/EVIDENCE_REGISTRY.md` |
| What exact claims and exclusions apply? | `results/evidence_registry/fgds_20260814_current/claim_boundaries.csv` |
| What are the primary WSN-DS results? | `results/wsnds/confirmation_runs_v2/local_feature_group_10seed_20260811/feature_group_10seed_analysis/feature_group_10seed_analysis.json` |
| What did final hardware replay prove? | `results/hardware_hil/final_fgds_seed42_v1/final_campaign_usb_v1/final_hil_summary.json` |
| What did Edge-IIoTset add? | `results/leftover_e2e_closure/04_edge_group_aware/edge_group_aware_summary.json` |
| How can the evidence be verified or regenerated? | `REPRODUCIBILITY.md` |
| How is the repository organized? | `docs/repository/REPOSITORY_MAP.md` |

## Reproducibility Status

| Level | Status |
|---|---|
| Evidence inspection | Supported by the tracked registry, reports, CSVs, JSON files, and figures. |
| Registry verification | Supported by hash-bound manifests after required Git LFS objects are hydrated. |
| Smoke testing | Supported by current CLI, repository, export, and HIL host-side tests. |
| Primary training rerun | Supported by the ten-seed feature-group-disjoint driver and analysis program. |
| Extended WSN-DS rerun | Supported by separate full-route, sensitivity, behavioral, XAI, and fixed-point programs. |
| Hardware replay rerun | Requires ESP32-C3 and Arduino R4 boards, flashed firmware, toolchains, and serial access. |
| Live WSN deployment | Not included in this artifact. |

## Minimal Review Commands

```powershell
git lfs install
py -3.11 -m pytest tests/repository/test_active_cli_smoke.py tests/repository/test_repository_structure_smoke.py -q
py -3.11 -m pytest tests/hardware tests/hardware_deployment_run -q
py -3.11 -m compileall -q experiments deployment tests
```

Deep registry verification requires hydrated LFS objects:

```powershell
py -3.11 -m experiments.evidence.build_fgds_evidence_registry `
    --output-dir results/evidence_registry/fgds_20260814_current `
    --verify-existing
```

The repository intentionally does not include a Dockerfile. The active artifact
contains notebooks, Python scripts, fixed-point firmware exports, generated
hardware bundles, and measured outputs rather than a single service container.

## Claim Boundary

Use this artifact to support paper writing and reviewer discussion around
resource-aware explainable IDS compression. Do not use it to claim live WSN
packet capture, energy measurement, physical TelosB deployment, or full
packet-to-feature extraction. The current registry also records two unexecuted
planned items: ten-seed scratch-controlled XAI and final-lineage Wi-Fi HIL.
