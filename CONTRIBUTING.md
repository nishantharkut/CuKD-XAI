# Contributing

This repository is currently in paper-writing and evidence-freezing mode.
Changes should preserve traceability and avoid weakening the claim boundary.

## Ground Rules

- Do not remove evidence files. Move files only when preserving them under a
  clearer location.
- Do not rewrite historical material in `research_history/` unless the change is itself
  documented as an archival correction.
- Do not add new claims unless they are backed by a result file, report, or
  reproducible script in the repository.
- Keep generated outputs and scripts consistent: if an active script output
  changes, stage the regenerated evidence and explain why.
- Keep README-level text concise. Detailed discussion belongs in `docs/`.

## Before Opening A Change

Run the smoke checks from the repository root:

```powershell
py -3.11 -m pytest tests/repository/test_active_cli_smoke.py tests/repository/test_repository_structure_smoke.py -q
py -3.11 -m pytest tests/hardware tests/hardware_deployment_run -q
py -3.11 -m compileall -q experiments deployment tests
```

The full suite requires all relevant Git LFS objects and is intentionally not
the quick pre-change check:

```powershell
git lfs fsck
py -3.11 -m pytest -q
```

For current evidence-registry changes, run the read-only verifier:

```powershell
py -3.11 -m experiments.evidence.build_fgds_evidence_registry `
    --output-dir results/evidence_registry/fgds_20260814_current `
    --verify-existing
```

Use `REPRODUCIBILITY.md` for experiment-specific regeneration commands. Never
write a rerun into a sealed evidence directory.

The protected-source verifier is a local restructuring baseline over hydrated
working-tree bytes. Run it only in the original preservation workspace; it is
not a portable clean-clone check.

## Review Checklist

- The active path-reference audit has no `active_review` rows.
- `git diff --cached --name-status --find-renames=1%` shows no pure deletes
  unless the removal is deliberate and explained.
- README claims are backed by files under `results/`, `docs/`, `deployment/`,
  or `experiments/`.
- Empty old folders are not treated as data loss; Git tracks files, not empty
  directories.
