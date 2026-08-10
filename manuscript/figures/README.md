# Manuscript figures — professional stack

## Tools installed / used

| Layer | Tool | Role |
|---|---|---|
| **Plots** | matplotlib + **seaborn** | Pareto, per-class bars, future SHAP/HIL charts |
| **Flow diagrams** | **Graphviz** (`dot`) | System/HIL, dual-identity, protocol ladder (vector PDF/SVG) |
| **LaTeX diagrams** | **TikZ** (MiKTeX) | In-source IEEE figures (`fig_system_hil.tex`, etc.) |

## Setup (once)

```powershell
# 1) Graphviz system binary
winget install Graphviz.Graphviz

# 2) Python packages in project venv
& "experiments\wsnds\leakage_free_rerun\.venv\Scripts\python.exe" -m pip install -r manuscript\figures\tools\requirements-figures.txt
```

If `dot` is not found after install, either open a **new** shell or rely on `tools/graphviz_env.py` (auto-detects `C:\Program Files\Graphviz\bin`).

## Build all figures

```powershell
cd "C:\N Drive\Research\Cukd-XAI\CuKD-XAI"
$env:Path = "C:\Program Files\Graphviz\bin;" + $env:Path
& "experiments\wsnds\leakage_free_rerun\.venv\Scripts\python.exe" manuscript\figures\tools\build_all_figures.py
```

Outputs (PDF + PNG + SVG for DOT):

- `fig_pareto_train_only.pdf` — seaborn/matplotlib  
- `fig_perclass_delta.pdf` — seaborn/matplotlib  
- `fig_system_hil_dot.pdf` — Graphviz  
- `fig_dual_identity_dot.pdf` — Graphviz  
- `fig_protocol_ladder_dot.pdf` — Graphviz  

TikZ sources (compile with the paper):

- `fig_system_hil.tex`  
- `fig_dual_identity.tex`  

## Style rules (IEEE-like)

- Serif 7–8 pt; thin black strokes; light gray fills only  
- No neon, no 3D stock IoT art, no AI decorative images  
- Always ship **PDF vector** for the paper; PNG is preview only  
- Numbers only from freeze / aggregate CSVs  

## Which format to include in LaTeX

| Figure | Preferred include |
|---|---|
| Pareto / bars | `\includegraphics{figures/fig_pareto_train_only.pdf}` |
| System HIL | TikZ `\input{figures/fig_system_hil}` **or** Graphviz PDF |
| Dual identity | TikZ **or** `fig_dual_identity_dot.pdf` |
| Protocol ladder | `fig_protocol_ladder_dot.pdf` |
