# CuKD-XAI Repository Restructure Plan

> Branch prepared for planning: `repo/professional-structure-plan`

## Goal

Make the repository look like a professional research/engineering evidence package without deleting any file. The restructure must preserve all existing evidence, old runs, datasets, notebooks, hardware outputs, generated firmware bundles, papers, and scratch files. The only allowed destructive-looking operation is `git mv`, which records a rename but keeps content in history and in the working tree.

## Non-Negotiable Guardrails

- Do not delete files.
- Do not use `Remove-Item`, `rm`, `git rm`, `git clean`, `git reset --hard`, or `git checkout --`.
- Use `git mv` for tracked files and directories.
- Keep every historical result somewhere discoverable.
- Separate move-only changes from content-edit changes.
- Keep the current evidence chain intact: WSN-DS results, Edge-IIoT results, hardware HIL, MSP430, papers, and research documentation.
- After every restructuring batch, run deletion checks before continuing.

## Current Top-Level Problem

The repository currently mixes active code, final evidence files, old notebooks, old result folders, datasets, hardware work, paper PDFs, scratch logs, and research-facing notes at the root. That makes the project look less focused than the research actually is.

Examples of root-level items that should be organized:

| Current item | Problem |
|---|---|
| `cukd_xai_colab.py`, `cukd_xai_colab.ipynb` | Main WSN-DS pipeline is at root without context. |
| `Final/` | Important, but name is vague and contains mixed code, evidence files, and outputs. |
| `Hardware Deployment Run/` | Important, but space-heavy name and mixed export/HIL/results layout. |
| `Edge-IIOT-run/` | Important secondary dataset route, but not grouped with experiments. |
| `Results - ...` folders | Many historical outputs at root. |
| `Update - 12 april 2026/`, `Archive*`, `V4/` | Old planning and historical versions at root. |
| `Papers/` | Literature should live under docs/literature. |
| `WSN-DS.csv` | Dataset at root makes the repo look messy and heavy. |
| `*.su`, `gpu_temp.log`, `cleanup_untrack_paths.txt` | Scratch/build outputs at root. |

## Recommended Target Structure

```text
CuKD-XAI/
  README.md
  CITATION.cff
  LICENSE
  pyproject.toml or requirements.txt

  src/
    cukd_xai/
      README.md

  experiments/
    wsnds/
      main/
      codistillation/
      deployment_runtime/
    edge_iiot/
      strict_generalization/
      literature_comparable/

  deployment/
    hardware_hil/
    firmware_export/
    msp430/

  results/
    wsnds/
      final_results/
      legacy_runs/
    edge_iiot/
      final_results/
    hardware/
      hil_reports/
      generated_bundles/
    runtime/
      onnx_openvino/

  data/
    wsnds/
    edge_iiot/

  docs/
    research/
    publication/
    literature/
      papers/
      comparison_tables/
    repository/
    archive/

  tests/
    hardware/
    export/
    static/

  archive/
    old_runs/
    old_updates/
    old_packages/
    scratch/
```

## Naming Rules

- Use lowercase folder names with hyphens or underscores consistently.
- Prefer no spaces in new folder names.
- Preserve exact filenames where they are evidence files unless the filename itself is misleading.
- Use `archive/` for historical material, not `delete`.
- Use `results/` for generated outputs/results.
- Use `experiments/` for runnable pipelines and notebooks.
- Use `deployment/` for firmware, HIL, MSP430, ONNX/OpenVINO, and deployment-oriented code.
- Use `docs/` for human-facing explanation, research briefs, publication material, and literature.

## Proposed Move Map

This is the high-level mapping. It should be implemented with `git mv` in small batches.

| Current path | Target path | Reason |
|---|---|---|
| `cukd_xai_colab.py` | `experiments/wsnds/main/cukd_xai_colab.py` | Main WSN-DS pipeline should be grouped under experiments. |
| `cukd_xai_colab.ipynb` | `experiments/wsnds/main/cukd_xai_colab.ipynb` | Notebook companion to main pipeline. |
| `Codistillation/` | `experiments/wsnds/codistillation/` | Keeps co-distillation work near WSN-DS experiments. |
| `Final/cukd_xai_wsnds_*.py` | `experiments/wsnds/deployment_runtime/` or `experiments/wsnds/final_routes/` | These are final runnable routes, not root files. |
| `Final/wsnds_deployment_qat_outputs/` | `results/runtime/onnx_openvino/wsnds/` | Generated software runtime evidence. |
| `Final/edgeiiot_v23_literature_comparable_selected_capacity_outputs/` | `results/edge_iiot/literature_comparable/` | Edge-IIoT generated outputs. |
| `Edge-IIOT-run/` | `experiments/edge_iiot/` | Secondary dataset route. |
| `Hardware Deployment Run/hardware_hil/` | `deployment/hardware_hil/` | HIL code, firmware, results, docs. |
| `Hardware Deployment Run/hardware_export/` | `deployment/firmware_export/wsnds_rfkd/` | Fixed-point C export pipeline. |
| `hardware_export/` | `deployment/msp430/` | MSP430 and older fixed-point export evidence. |
| `Papers/` | `docs/literature/papers/` | Literature belongs under docs. |
| `PROFESSOR_RESULTS_COMPARISON.md` | `docs/research/RELATED_WORK_RESULTS_COMPARISON.md` | Related-work evidence and comparison. |
| `ACCURACY_IMPROVEMENT_PLAN.md` | `docs/paper/ACCURACY_IMPROVEMENT_PLAN.md` | Research planning material. |
| `WORK_DIVISION.md` | `docs/project/WORK_DIVISION.md` | Project management material. |
| `WORK_DIVISION_FLOW.md` | `docs/project/WORK_DIVISION_FLOW.md` | Project management material. |
| `Claude-.md` | `docs/archive/assistant-notes/Claude-.md` | Historical assistant note. |
| `Update - 12 april 2026/` | `docs/archive/updates/2026-04-12/` | Historical planning/update pack. |
| `Archive/`, `Archive_v2/`, `archive-v3-27 may/`, `V4/` | `archive/old-runs/` | Historical old versions. |
| `Results - ...` folders | `results/wsnds/legacy_runs/` | Historical WSN-DS result outputs. |
| `WSN-DS.csv` | `data/wsnds/WSN-DS.csv` | Dataset should not sit at root. |
| `Edge-IIOT-run/WSN-DS.csv` | `data/wsnds/edge_iiot_copy/WSN-DS.csv` or archive with note | Preserve duplicate exactly, but document why it exists. |
| `*.su`, `gpu_temp.log`, `cleanup_untrack_paths.txt` | `archive/scratch/` | Preserve scratch/build outputs without cluttering root. |
| `test_hardware_export_e2e.py` | `tests/hardware/test_hardware_export_e2e.py` | Tests should live under `tests/`. |
| `tmp/` | `archive/scratch/tmp/` | Preserve temporary files without root clutter. |

## Implementation Phases

### Phase 0: Safety Snapshot

- [ ] Confirm branch:
  ```powershell
  git branch --show-current
  ```
  Expected: `repo/professional-structure-plan` or the final restructure branch.

- [ ] Save tracked-file inventory:
  ```powershell
  git ls-files | Sort-Object | Set-Content docs/repository/tracked_files_before_restructure.txt
  ```

- [ ] Save top-level inventory:
  ```powershell
  Get-ChildItem -Force | Select-Object Mode,Length,Name |
      Format-Table -AutoSize |
      Out-String -Width 240 |
      Set-Content docs/repository/top_level_before_restructure.txt
  ```

### Phase 1: Add Professional Documentation Shell

- [ ] Add `docs/repository/REPOSITORY_RESTRUCTURE_PLAN.md`.
- [ ] Add `docs/repository/README_REWRITE_BLUEPRINT.md`.
- [ ] Add `docs/repository/REPOSITORY_MAP.md` after the move map is finalized.
- [ ] Do not move any existing files in this phase.
- [ ] Check:
  ```powershell
  git diff --name-status
  ```
  Expected: only added docs in `docs/repository/`.

### Phase 2: Create Canonical Directories

- [ ] Create empty target directories with `.gitkeep` only where needed:
  ```powershell
  New-Item -ItemType Directory -Force experiments, deployment, results, data, tests, archive | Out-Null
  ```
- [ ] Do not move files yet.
- [ ] Check:
  ```powershell
  git status --short
  ```

### Phase 3: Move Old and Historical Material First

Move historical folders first because they are least likely to break imports.

- [ ] Move old archive folders:
  ```powershell
  git mv Archive archive/old-runs/Archive
  git mv Archive_v2 archive/old-runs/Archive_v2
  git mv "archive-v3-27 may" "archive/old-runs/archive-v3-27 may"
  git mv V4 archive/old-runs/V4
  ```

- [ ] Move old updates:
  ```powershell
  git mv "Update - 12 april 2026" "docs/archive/updates/2026-04-12"
  ```

- [ ] Move root scratch/build leftovers:
  ```powershell
  git mv gpu_temp.log archive/scratch/gpu_temp.log
  git mv cleanup_untrack_paths.txt archive/scratch/cleanup_untrack_paths.txt
  git mv msp430_smoke_main.su archive/scratch/msp430_smoke_main.su
  git mv wsnds_preprocess_int16.su archive/scratch/wsnds_preprocess_int16.su
  git mv wsnds_student_a_rfkd_int8_inference.su archive/scratch/wsnds_student_a_rfkd_int8_inference.su
  ```

- [ ] Verify no deletes:
  ```powershell
  git diff --name-status | Select-String '^D'
  ```
  Expected: no output.

### Phase 4: Move Literature and Research Materials

- [ ] Move papers:
  ```powershell
  git mv Papers docs/literature/papers
  ```

- [ ] Move related-work comparison:
  ```powershell
  git mv PROFESSOR_RESULTS_COMPARISON.md docs/research/RELATED_WORK_RESULTS_COMPARISON.md
  ```

- [ ] Move planning docs:
  ```powershell
  git mv ACCURACY_IMPROVEMENT_PLAN.md docs/paper/ACCURACY_IMPROVEMENT_PLAN.md
  git mv WORK_DIVISION.md docs/project/WORK_DIVISION.md
  git mv WORK_DIVISION_FLOW.md docs/project/WORK_DIVISION_FLOW.md
  git mv Claude-.md docs/archive/assistant-notes/Claude-.md
  ```

- [ ] Verify no deletes:
  ```powershell
  git diff --name-status | Select-String '^D'
  ```
  Expected: no output.

### Phase 5: Move WSN-DS Main Experiments and Results

- [ ] Move main WSN-DS code:
  ```powershell
  git mv cukd_xai_colab.py experiments/wsnds/main/cukd_xai_colab.py
  git mv cukd_xai_colab.ipynb experiments/wsnds/main/cukd_xai_colab.ipynb
  ```

- [ ] Move WSN-DS results:
  ```powershell
  git mv "Results - 10 Seed Run 30 may" "results/wsnds/legacy_runs/2026-05-30-10seed"
  git mv "Results - 10 Seed Run 30 may + J" "results/wsnds/final_results/2026-05-30-10seed-plus-j"
  git mv "Results - 12 April 2026" "results/wsnds/legacy_runs/2026-04-12"
  git mv "Results - 8 may QuickModeOff (v3)" "results/wsnds/legacy_runs/2026-05-08-v3-quickmode-off"
  git mv "Results - 8 may QuickmodeON (v3)" "results/wsnds/legacy_runs/2026-05-08-v3-quickmode-on"
  git mv "Results - quickModeOFF (v2.3)" "results/wsnds/legacy_runs/v2.3-quickmode-off"
  git mv "Results - QuickModeON (v2.3)" "results/wsnds/legacy_runs/v2.3-quickmode-on"
  ```

- [ ] Move WSN-DS dataset:
  ```powershell
  git mv WSN-DS.csv data/wsnds/WSN-DS.csv
  ```

- [ ] Verify no deletes:
  ```powershell
  git diff --name-status | Select-String '^D'
  ```
  Expected: no output.

### Phase 6: Move Edge-IIoT Experiments and Artifacts

- [ ] Move Edge-IIoT strict route:
  ```powershell
  git mv Edge-IIOT-run experiments/edge_iiot
  ```

- [ ] Move final literature-comparable Edge-IIoT outputs from `Final/` into results:
  ```powershell
  git mv "Final/edgeiiot_v23_literature_comparable_selected_capacity_outputs" "results/edge_iiot/literature_comparable"
  ```

- [ ] Verify no deletes:
  ```powershell
  git diff --name-status | Select-String '^D'
  ```
  Expected: no output.

### Phase 7: Move Deployment and Hardware Evidence

- [ ] Move HIL package:
  ```powershell
  git mv "Hardware Deployment Run/hardware_hil" deployment/hardware_hil
  ```

- [ ] Move hardware export package:
  ```powershell
  git mv "Hardware Deployment Run/hardware_export" deployment/firmware_export/wsnds_rfkd
  ```

- [ ] Move root MSP430/export package:
  ```powershell
  git mv hardware_export deployment/msp430
  ```

- [ ] Move remaining `Hardware Deployment Run` docs or tests into archive if any files remain:
  ```powershell
  Get-ChildItem "Hardware Deployment Run" -Recurse -Force
  ```
  Only move remaining tracked files with `git mv`; do not delete the directory manually.

- [ ] Verify no deletes:
  ```powershell
  git diff --name-status | Select-String '^D'
  ```
  Expected: no output.

### Phase 8: Move Runtime Artifacts and Final Routes

- [ ] Move final WSN-DS runtime scripts:
  ```powershell
  git mv Final/cukd_xai_wsnds_deployment_qat_proof.py experiments/wsnds/deployment_runtime/cukd_xai_wsnds_deployment_qat_proof.py
  git mv Final/cukd_xai_wsnds_runtime_from_existing.py experiments/wsnds/deployment_runtime/cukd_xai_wsnds_runtime_from_existing.py
  git mv Final/cukd_xai_wsnds_j_only_merge.py experiments/wsnds/codistillation/cukd_xai_wsnds_j_only_merge.py
  git mv Final/cukd_xai_edgeiiot_v23_literature_comparable.py experiments/edge_iiot/literature_comparable/cukd_xai_edgeiiot_v23_literature_comparable.py
  ```

- [ ] Move runtime outputs:
  ```powershell
  git mv Final/wsnds_deployment_qat_outputs results/runtime/onnx_openvino/wsnds
  ```

- [ ] Move remaining `Final/` contents into `archive/old-runs/Final` or a more precise artifact folder after inspecting:
  ```powershell
  git ls-files Final
  ```

- [ ] Verify no deletes:
  ```powershell
  git diff --name-status | Select-String '^D'
  ```
  Expected: no output.

### Phase 9: Tests and Import Path Repair

This phase may require content edits because moved paths can break tests or README links.

- [ ] Move root test:
  ```powershell
  git mv test_hardware_export_e2e.py tests/hardware/test_hardware_export_e2e.py
  ```

- [ ] Search stale paths:
  ```powershell
  rg -n "Hardware Deployment Run|Edge-IIOT-run|Results -|Update - 12 april 2026|Papers/|Final/" .
  ```

- [ ] Update only documentation/path references needed after moves.
- [ ] Avoid changing algorithm code in this PR unless a moved relative path breaks a test.

### Phase 10: Professional README Rewrite

- [ ] Replace the current quick-start README with a polished research README.
- [ ] Include badges, table of contents, project diagram, core results table, repository map, setup, reproduction, hardware boundary, citation, and claim boundaries.
- [ ] Keep claims consistent with `docs/research/RESULTS_AND_EVIDENCE.md`.

### Phase 11: Verification

- [ ] Confirm no deletions:
  ```powershell
  git diff --name-status main...HEAD | Select-String '^D'
  ```
  Expected: no output.

- [ ] Confirm every original tracked file still exists somewhere:
  ```powershell
  git ls-files | Sort-Object | Set-Content docs/repository/tracked_files_after_restructure.txt
  ```
  Then compare count:
  ```powershell
  (Get-Content docs/repository/tracked_files_before_restructure.txt).Count
  (git ls-files).Count
  ```

- [ ] Run hardware/export tests from the new paths:
  ```powershell
  python -m pytest tests -q
  ```

- [ ] Run targeted HIL static tests if still under hardware package:
  ```powershell
  python -m pytest deployment/hardware_hil/tests -q
  ```
  If the test folder is moved differently, use the new exact path.

- [ ] Run link/path scan:
  ```powershell
  rg -n "README|docs/research|docs/publication|deployment/hardware_hil|experiments/wsnds|results/wsnds" README.md docs experiments deployment
  ```

## PR Strategy

Use multiple commits or multiple PRs. The safest path is:

1. PR 1: add restructuring plan and README blueprint only.
2. PR 2: move archive/history/scratch/literature files.
3. PR 3: move active WSN-DS, Edge-IIoT, and deployment paths.
4. PR 4: rewrite README and update references.

This avoids one huge, hard-to-review PR and makes it easy to prove that no content was deleted.

## Final Review Checklist

- [ ] Root contains only professional entry points and top-level domains.
- [ ] No important evidence file is hidden without a map.
- [ ] README explains the research in 60 seconds.
- [ ] README links to exact evidence docs.
- [ ] README has Mermaid diagrams.
- [ ] README has reproducibility commands.
- [ ] README has clear claim boundaries.
- [ ] `git diff --name-status main...HEAD` contains no `D` records.
- [ ] Tests either pass or failures are documented as pre-existing/path-migration issues.
