# Path Reference Audit

This audit records the repository-wide old-path reference scan performed after the no-delete restructure.

## Scope

The scan searched the working tree, excluding `.git`, for old repository-location tokens such as:

- `Final/`
- `Hardware Deployment Run`
- `Edge-IIOT-run`
- `Codistillation/`
- `Results -`
- `hardware_export/`
- old HIL result/report/compile-log locations

The compact line-numbered index is stored in:

- `docs/repository/path_reference_audit.csv`
- `docs/repository/path_reference_audit_summary.csv`

## Classification

| Category | Count | Decision |
|---|---:|---|
| `audit_self_reference` | 6 | Preserved because this audit names the old tokens it searched for. |
| `generated_or_provenance` | 267 | Preserved unless the file is regenerated; these entries are generated outputs, notebooks, CSV/JSON evidence, or build products. |
| `historical_archive` | 6237 | Preserved because archived material intentionally records older structure. |
| `repository_inventory` | 687 | Preserved because before/after inventory files intentionally record paths. |
| `repository_smoke_test` | 4 | Preserved because the smoke test contains old-path tokens as a negative scan pattern. |
| `restructure_plan` | 39 | Preserved because the plan documents old-to-new mapping. |
| `active_review` | 0 | No active unclassified old-path references remain after the migration pass. |

## Policy

Active code, active runbooks, and active repository-facing docs should use the new structure. Historical archives, generated reports, model provenance strings, notebook output cells, and before/after inventories are not silently rewritten because that can blur evidence provenance.

## Follow-Up Rule

When adding new scripts or docs, use the current top-level structure:

- `experiments/` for research pipelines
- `deployment/` for firmware/HIL/MSP430 implementation
- `results/` for generated evidence and tables
- `data/` for dataset copies
- `docs/` for papers, reproduction, and professor-facing material
- `archive/` only for preserved historical material
