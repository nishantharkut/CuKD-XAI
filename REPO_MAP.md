# Repository Map

This repository is organized into active experiment/evidence folders, documentation, and preserved archive material.

## Active / Current

- `Final/`
  Final model training and evaluation evidence.

- `Edge-IIOT-run/`
  Edge-IIoT experiments, selected-capacity runs, and literature comparison analysis.

- `Hardware Deployment Run/`
  Hardware export, fixed-point firmware, HIL host tools, ESP32-C3 / Arduino R4 results, compile logs, and final hardware reports.

- `hardware_export/`
  Older root-level hardware export area retained for compatibility with earlier commands and reports. Audit before moving.

- `WSN-DS.csv`
  Dataset file used by current WSN-DS export/replay workflows.

## Documentation

- `docs/papers/`
  Related papers and literature material.

- `docs/professor/`
  Professor-facing summaries and comparison notes.

- `docs/manuscript/`
  Draft planning, work division, and manuscript-support notes.

## Preserved Archive

- `Repository_Archive/old_runs/`
  Older WSN-DS/Edge-IIoT runs and previous experiment folders.

- `Repository_Archive/old_updates/`
  Older update folders preserved for traceability.

- `Repository_Archive/old_packages/`
  Older generated packages.

- `Repository_Archive/root_scratch/`
  Root-level scratch logs and compiler side outputs moved out of the main working area.

## Hygiene Rule

Do not delete old evidence. Move obsolete or duplicate material into `Repository_Archive/`.
Keep current reproducibility paths stable unless a later audit updates all references.
