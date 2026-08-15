# CuKD-XAI Repository Map

This map identifies the current research, deployment, evidence, and historical surfaces. The current claim source is the evidence registry, while earlier result lineages remain preserved for traceability.

## Top-Level Layout

| Path | Role |
|---|---|
| `README.md` | Professional project entry point. |
| `REPRODUCIBILITY.md` | Reviewer verification, regeneration, and environment guide. |
| `ARTIFACT.md` | Artifact scope and evidence-review order. |
| `experiments/` | Runnable or inspectable research pipelines. |
| `deployment/` | Firmware export, hardware-in-loop, and MSP430 deployment evidence. |
| `results/` | Generated outputs, result tables, runtime evidence, and historical results. |
| `data/` | Dataset files and preserved duplicate copies from historical runs. |
| `docs/` | Research briefs, literature, reproduction guides, and repository documentation. |
| `tests/` | Preserved static, export, and hardware checks. |
| `research_history/` | Historical runs, old packages, update packs, and scratch files. |

## Experiments

| Path | Contents |
|---|---|
| `experiments/wsnds/leakage_free_rerun/` | Current feature-group-disjoint primary driver, analysis, and preserved predecessor protocols. |
| `experiments/wsnds/evidence_completion/` | Controlled full-route, sensitivity, behavioral-transfer, XAI, and fixed-point analyses. |
| `experiments/evidence/` | Current evidence-registry builder and verifier. |
| `experiments/wsnds/main/` | Original WSN-DS notebook and Python implementation retained as an earlier lineage. |
| `experiments/wsnds/codistillation/` | Original co-distillation implementation and preserved pilot outputs. |
| `experiments/wsnds/deployment_runtime/` | ONNX/OpenVINO and dynamic-INT8 runtime/export scripts. |
| `experiments/edge_iiot/strict_generalization/` | Strict Edge-IIoT generalization implementation. |
| `experiments/edge_iiot/literature_comparable/` | Edge-IIoT literature-comparable implementation and metric-gap analysis script. |
| `experiments/edge_iiot/audit_edgeiiot_split_duplicates.py` | Edge-IIoT split-overlap audit. |

## Deployment

| Path | Contents |
|---|---|
| `deployment/final_hil/` | Current fail-closed final campaign controller and archive verifier. |
| `deployment/hardware_hil/` | USB HIL host tools, firmware templates, and lineage-specific runbooks. |
| `deployment/wireless_hil/` | Wi-Fi UDP HIL tooling and its separate campaign lineage. |
| `deployment/firmware_export/wsnds_final_hil/` | Current seed-42 final exporter and 40-instance software audit. |
| `deployment/firmware_export/wsnds_rfkd_hil/` | FG-DS runtime and earlier RF-KD firmware exports. |
| `deployment/msp430/current_fgds_static/` | Current FG-DS MSP430F1611 static compile evidence. |
| `deployment/msp430/` | Earlier MSP430 export and compile material retained for traceability. |

## Results

| Path | Contents |
|---|---|
| `results/evidence_registry/fgds_20260814_current/` | Authoritative current registry, claim boundaries, and manifest. |
| `results/wsnds/confirmation_runs_v2/local_feature_group_10seed_20260811/` | Primary ten-seed FG-DS run and paired analysis. |
| `results/wsnds/evidence_completion_20260811/` | Current full-route, sensitivity, SHAP, and refinement evidence. |
| `results/wsnds/evidence_completion_20260812/` | Current behavioral-transfer and multi-split evidence. |
| `results/wsnds/evidence_completion_20260813/` | Current all-seed fixed-point audit. |
| `results/runtime/onnx_openvino/wsnds/fgds_seed42_exact/` | Current seed-42 runtime conversion evidence. |
| `results/hardware_hil/final_fgds_seed42_v1/` | Final USB campaign contracts, bundles, sessions, and report. |
| `results/leftover_e2e_closure/04_edge_group_aware/` | Secondary group-aware Edge-IIoTset evidence. |
| `results/wsnds/final_results/` and `results/wsnds/legacy_runs/` | Earlier WSN-DS lineages retained for historical comparison only. |
| `results/edge_iiot/` | Earlier strict and literature-style Edge-IIoTset outputs. |

## Documentation

| Path | Contents |
|---|---|
| `docs/research/` | Historical project synthesis and related-work comparison; current claims live in the evidence registry. |
| `docs/publication/` | Manuscript preparation and writing guidance. |
| `docs/literature/papers/` | Source-paper corpus and associated review artifacts. |
| `docs/literature/comparison_tables/` | Generated comparison tables for related work. |
| `docs/reproduction/wsnds/` | WSN-DS reproduction guides. |
| `docs/reproduction/edge_iiot/` | Edge-IIoT reproduction guide. |
| `research_history/documentation_snapshots/updates/` | Historical planning/update material. |
| `docs/repository/` | Repository restructuring plan, README blueprint, path-reference audit, inventories, and this map. |

## Data

| Path | Contents |
|---|---|
| `data/wsnds/WSN-DS.csv` | Main WSN-DS CSV copy used by the restructured repository. |
| `data/wsnds/copies/` | Preserved duplicate dataset copies from historical runs. |

## Research History

| Path | Contents |
|---|---|
| `research_history/experiment_snapshots/` | Old notebooks, previous project runs, and preserved final-file duplicates. |
| `research_history/software_snapshots/` | Preserved software and hardware package snapshots. |
| `research_history/documentation_snapshots/` | Superseded planning notes and project updates. |
| `research_history/development_records/` | Scratch logs, extracted paper text, notebook checkpoints, and local IDE files. |

## Evidence Entry Points

Start from these files when reviewing the research:

| Question | File |
|---|---|
| What is the current project result? | `results/evidence_registry/fgds_20260814_current/EVIDENCE_REGISTRY.md` |
| What are the exact claim boundaries? | `results/evidence_registry/fgds_20260814_current/claim_boundaries.csv` |
| How can a reviewer reproduce it? | `REPRODUCIBILITY.md` |
| What did final hardware HIL prove? | `results/hardware_hil/final_fgds_seed42_v1/final_campaign_usb_v1/final_hil_summary.json` |
| What did secondary Edge-IIoTset testing show? | `results/leftover_e2e_closure/04_edge_group_aware/edge_group_aware_summary.json` |
| What was changed in the repository structure? | `research_history/documentation_snapshots/repository_restructure/REPOSITORY_RESTRUCTURE_PLAN.md` |
| Were old path references audited? | `research_history/documentation_snapshots/repository_restructure/PATH_REFERENCE_AUDIT.md` |
