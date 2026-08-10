# draw.io (diagrams.net) setup for CuKD figures

## Installed

- **draw.io Desktop** `JGraph.Draw` v31.1.5 via winget  
- Executable: `%LOCALAPPDATA%\Programs\draw.io\draw.io.exe`

## Project files

| File | Purpose |
|---|---|
| `system_hil.drawio` | System / HIL architecture (a–c) |
| `dual_identity.drawio` | U-MS vs U-DC evaluation units |
| `export/` | PDF/PNG/SVG exports for LaTeX |

## Open a diagram

```powershell
& "$env:LOCALAPPDATA\Programs\draw.io\draw.io.exe" "C:\N Drive\Research\Cukd-XAI\CuKD-XAI\manuscript\figures\drawio\system_hil.drawio"
```

Or: double-click the `.drawio` file in Explorer.

## Export for the paper (CLI)

```powershell
cd "C:\N Drive\Research\Cukd-XAI\CuKD-XAI"
.\manuscript\figures\tools\export_drawio.ps1
```

Exports each `.drawio` in this folder to `export/*.pdf` and `export/*.png`.

Manual (one file):

```powershell
$draw = "$env:LOCALAPPDATA\Programs\draw.io\draw.io.exe"
& $draw --export --format pdf --output "manuscript\figures\drawio\export\system_hil.pdf" "manuscript\figures\drawio\system_hil.drawio"
& $draw --export --format png --output "manuscript\figures\drawio\export\system_hil.png" "manuscript\figures\drawio\system_hil.drawio"
```

## Style guide (IEEE-like, not AI slop)

- Black strokes, white / light-gray fills only  
- No neon, 3D, clipart, or drop shadows  
- Helvetica / sans, 10–12 pt labels  
- Put **quantitative numbers only if copied from** `_provenance.json` / freeze / CSV  
- Architecture figures: prefer structure over metrics  

## Grounding

`system_hil.drawio` uses **N = 56200** only as the documented test-set size from  
`results/runtime/onnx_openvino/wsnds/train_only_seed42_copy/train_only_runtime_results.json` → `n_test`.

For dual-identity **macro-F1 values**, do not hard-type in draw.io; either:

1. Leave qualitative labels and put numbers in the LaTeX caption from provenance, or  
2. After `build_all_figures.py`, paste exact rounded values from `_provenance.json`.

## Also available

| Tool | Where |
|---|---|
| Graphviz | `figures/tools/build_all_figures.py` |
| TikZ | `figures/fig_*.tex` |
| seaborn/matplotlib | same build script |
| Product photos | `figures/hardware/` |

Use **draw.io** when you want interactive layout control; use **Graphviz/TikZ** for fully reproducible text-first diagrams.
