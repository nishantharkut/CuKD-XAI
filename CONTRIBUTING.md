# Contributing

This repository is currently in paper-writing and evidence-freezing mode.
Changes should preserve traceability and avoid weakening the claim boundary.

## Ground Rules

- Do not remove evidence files. Move files only when preserving them under a
  clearer location.
- Do not rewrite historical material in `archive/` unless the change is itself
  documented as an archival correction.
- Do not add new claims unless they are backed by a result file, report, or
  reproducible script in the repository.
- Keep generated outputs and scripts consistent: if an active script output
  changes, stage the regenerated evidence and explain why.
- Keep README-level text concise. Detailed discussion belongs in `docs/`.

## Before Opening A Change

Run the smoke checks from the repository root:

```powershell
py -3.11 -m pytest -q
py -3.11 -m compileall -q experiments deployment tests
```

For hardware-evidence changes, also run:

```powershell
py -3.11 deployment\hardware_hil\host\analyze_final_hil_evidence.py `
    --project-root . `
    --output-dir results\hardware_hil\reports\final_postprocessing
```

For Edge-IIoT literature-metric changes, also run:

```powershell
py -3.11 experiments\edge_iiot\literature_comparable\edgeiiot_literature_metric_gap_analysis.py `
    --repo-root . `
    --artifact-dir results\edge_iiot\literature_comparable `
    --output-dir results\edge_iiot\literature_metric_gap
```

## Review Checklist

- The active path-reference audit has no `active_review` rows.
- `git diff --cached --name-status --find-renames=1%` shows no pure deletes
  unless the removal is deliberate and explained.
- README claims are backed by files under `results/`, `docs/`, `deployment/`,
  or `experiments/`.
- Empty old folders are not treated as data loss; Git tracks files, not empty
  directories.
