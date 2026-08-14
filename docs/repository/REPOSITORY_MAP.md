# CuKD-XAI Repository Map

This map explains the restructured repository after the no-delete organization pass. Historical files were moved with `git mv`; content was preserved rather than deleted.

## Top-Level Layout

| Path | Role |
|---|---|
| `README.md` | Professional project entry point. |
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
| `experiments/wsnds/main/` | Main WSN-DS notebook and Python implementation. |
| `experiments/wsnds/codistillation/` | Co-distillation implementation and preserved pilot outputs. |
| `experiments/wsnds/deployment_runtime/` | ONNX/OpenVINO/dynamic-INT8 runtime/export scripts. |
| `experiments/edge_iiot/strict_generalization/` | Strict Edge-IIoT generalization implementation. |
| `experiments/edge_iiot/literature_comparable/` | Edge-IIoT literature-comparable implementation and metric-gap analysis script. |

## Deployment

| Path | Contents |
|---|---|
| `deployment/hardware_hil/` | HIL host tools, firmware bundles, board runbooks, results, and reports. |
| `deployment/firmware_export/wsnds_rfkd_hil/` | Generated C/header exports and replay vectors for Student A/B RF-KD HIL. |
| `deployment/msp430/` | MSP430F1611 memory-feasibility export and compile evidence. |

## Results

| Path | Contents |
|---|---|
| `results/wsnds/final_results/2026-05-30-10seed-plus-j/` | Main WSN-DS final 10-seed plus co-distillation outputs. |
| `results/wsnds/legacy_runs/` | Historical WSN-DS result folders. |
| `results/runtime/onnx_openvino/wsnds/` | ONNX, dynamic INT8, OpenVINO-related runtime evidence and summaries. |
| `results/edge_iiot/strict_generalization/` | Strict Edge-IIoT generated outputs. |
| `results/edge_iiot/literature_comparable/` | Selected-capacity Edge-IIoT outputs. |
| `results/edge_iiot/literature_metric_gap/` | Edge-IIoT literature metric-gap tables. |

## Documentation

| Path | Contents |
|---|---|
| `docs/research/` | Technical brief, results evidence ledger, and related-work comparison. |
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
| What is the whole project? | `docs/research/PROJECT_TECHNICAL_BRIEF.md` |
| What are the exact result claims? | `docs/research/RESULTS_AND_EVIDENCE.md` |
| What did hardware HIL prove? | `results/hardware_hil/reports/final_postprocessing/final_postprocessing_analysis.md` |
| What did Edge-IIoT prove? | `docs/literature/comparison_tables/EDGEIIOT_LITERATURE_COMPARISON.md` |
| What was changed in the repository structure? | `research_history/documentation_snapshots/repository_restructure/REPOSITORY_RESTRUCTURE_PLAN.md` |
| Were old path references audited? | `research_history/documentation_snapshots/repository_restructure/PATH_REFERENCE_AUDIT.md` |
