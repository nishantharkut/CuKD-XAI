# CuKD-XAI v3.0 — Complete Setup & Execution Guide

**Project:** CuKD-XAI (Curriculum-Guided Knowledge Distillation for WSN Intrusion Detection)
**Author:** Nishant Harkut (2023IMG-040), ABV-IIITM Gwalior
**Hardware target:** ASUS TUF A15 FA507NU (Ryzen 7 7735HS / RTX 4050 Laptop 6GB / 16GB RAM)
**v3 code created:** May 2026
**This guide covers:** Environment setup → smoke test → full 10-seed publication run → result analysis

---

## Table of Contents

1. [What you currently have](#1-what-you-currently-have)
2. [What you still need to set up](#2-what-you-still-need-to-set-up)
3. [Folder structure (final state)](#3-folder-structure-final-state)
4. [Environment setup — Windows step-by-step](#4-environment-setup--windows-step-by-step)
5. [v3 architecture overview](#5-v30-architecture-overview)
6. [v3 toggle flags explained](#6-v30-toggle-flags-explained)
7. [Pre-flight verification](#7-pre-flight-verification)
8. [Smoke test (run first)](#8-smoke-test-30-min-run-this-first)
9. [Phase-by-phase progressive testing](#9-phase-by-phase-progressive-testing)
10. [Full publication run (10 seeds)](#10-full-publication-run-10-seeds)
11. [Expected outputs](#11-expected-outputs)
12. [Result interpretation guide](#12-result-interpretation-guide)
13. [Common errors and fixes](#13-common-errors-and-fixes)
14. [Performance expectations](#14-performance-expectations-on-asus-tuf-a15)
15. [Comparison with v2.3 baseline](#15-comparison-with-v23-baseline)
16. [Roadmap to paper submission](#16-roadmap-to-paper-submission)
17. [When to ask for help](#17-when-to-ask-claude-for-help)
18. [Appendix — Restoring v2.3 baseline](#appendix-a--restoring-v23-baseline)

---

## 1. What you currently have

You said: *"I have only downloaded the new code files that you have made currently. Nothing more than that."*

So at this moment on your Windows machine you have:

| File | What it is | Approximate size |
|---|---|---|
| `cukd_xai_colab.py` | v3.0 Python source — 3,551 lines, all 9 phases applied | 153 KB |
| `cukd_xai_colab.ipynb` | v3.0 Jupyter notebook — 46 cells (22 code + 24 markdown) | 211 KB |

**Note:** You downloaded these via the cloudflare tunnel earlier today. Confirm both files are present before continuing.

### What's IN the v3 code (vs v2.3 baseline)

The v3 file is the v2.3 file **plus** these additions (none of the original v2.3 code was removed — everything is additive):

| Phase | Workstream | What was added |
|---|---|---|
| 1 | A — Feature engineering | Domain-informed features (`drop_rate`, `forward_ratio`, etc.) — INPUT_DIM goes 17 → 23 |
| 2 | D1 — Decoupled KD | DKD loss + 3 new configs (`E_KD_from_RF_dkd`, `E2_KD_from_MLP_dkd`, `F_KD_from_CL_MLP_ext_dkd`) |
| 3 | F — Training tricks | EMA + label smoothing + grad clipping + 4 new configs (`B_Full_MLP_tricks`, `D_Small_MLP_tricks`, `E_KD_from_RF_tricks`, `F_KD_from_CL_MLP_ext_tricks`) |
| 5 | QAT | Quantization-Aware Training with auto-derive of hidden_dims |
| 6 | CICIoT2023 v3 | Streams 63 MERGED_CSV files; 8-class taxonomy; class-grounded CL ordering; CICIoT-specific engineered features |
| 7 | 10-seed extension | `RUN_FULL_10_SEEDS = True` extends to 10 seeds for tighter Wilcoxon CIs |
| 8 | Enhanced Wilcoxon | Bootstrap CIs + Holm-Bonferroni correction + 11 v3 comparisons |
| 9 | C — Architecture sweep | KD-from-RF at 5 student sizes (XS/S/M/L/XL) with TelosB-fit flag |

---

## 2. What you still need to set up

### 2A. Required project files (besides the .py / .ipynb)

| File | Purpose | Where to get |
|---|---|---|
| `WSN-DS.csv` | Primary dataset (374,661 records, 19 columns) | Already in your GitHub repo `nishantharkut/CuKD-XAI` (26 MB) — re-download if needed |
| `MERGED_CSV/Merged01.csv ... Merged63.csv` | CICIoT2023 dataset (already on your laptop per your earlier comment at `C:\N Drive\Acads\6th SEM\WCT\CuKD-XAI\MERGED_CSV\MERGED_CSV\`) | Official UNB CIC site if missing |
| (optional) `cukd_xai_colab.v2.3.backup.py` and `.ipynb` | For revert if v3 misbehaves | Re-clone from GitHub (commit `6250c37`) |

### 2B. Python packages (your existing `cukd_env` likely has most)

You confirmed: `python 3.11.9`, `torch 2.5.1+cu121`. You already have most packages from v2.3.

Verify by running this in PowerShell from the project folder (with your env active):

```powershell
python -c "import torch, numpy, pandas, sklearn, shap, matplotlib, seaborn, scipy, imblearn, copy, json, os, time; print('All v2.3 packages OK')"
```

### 2C. NEW packages or features required by v3

The v3 code adds **NO new Python packages** beyond what v2.3 needed. All v3 features use:
- `torch.ao.quantization` (built into PyTorch 2.5.1)
- `scipy.stats.wilcoxon` (already used in v2.3)
- `numpy`, `pandas` (already used in v2.3)

**No `pip install` is needed for v3 features.**

### 2D. Folder layout your laptop should end up with

```
C:\N Drive\Acads\6th SEM\WCT\CuKD-XAI\
├── cukd_xai_colab.py              ← v3.0 file you just downloaded
├── cukd_xai_colab.ipynb           ← v3.0 file you just downloaded
├── WSN-DS.csv                      ← from your GitHub repo (26 MB)
├── MERGED_CSV\                     ← already on your laptop
│   └── MERGED_CSV\
│       ├── Merged01.csv
│       ├── Merged02.csv
│       ├── …
│       └── Merged63.csv
├── ACCURACY_IMPROVEMENT_PLAN.md   (already in repo)
├── README.md                       (already in repo)
└── results\                        (will be created by the run)
    ├── cukd_xai_results.json
    ├── wsnds_results_student_A.csv
    ├── wsnds_results_student_B.csv
    ├── per_class_f1.png
    ├── confusion_matrix_E.png
    ├── confusion_matrix_F.png
    ├── pareto_frontier.png         ← v3 will overlay 5-size sweep
    ├── shap_summary_student.png
    └── loss_curves_B_vs_C.png
```

---

## 3. Folder structure (final state)

Open PowerShell in your project folder and verify the layout:

```powershell
cd "C:\N Drive\Acads\6th SEM\WCT\CuKD-XAI"
Get-ChildItem | Format-Table Name, Length
```

You should see at minimum:
- `cukd_xai_colab.py` (~153 KB)
- `cukd_xai_colab.ipynb` (~211 KB)
- `WSN-DS.csv` (~26 MB)
- `MERGED_CSV\` directory

If `WSN-DS.csv` is missing:
```powershell
Invoke-WebRequest -Uri "https://github.com/nishantharkut/CuKD-XAI/raw/main/WSN-DS.csv" -OutFile "WSN-DS.csv"
```

---

## 4. Environment setup — Windows step-by-step

### 4A. Pre-flight checklist (do BEFORE running anything)

| Check | What to do |
|---|---|
| 1. RAM free ≥ 6 GB | Close Chrome, Edge, Discord, VS Code. Run: `Get-CimInstance Win32_OperatingSystem \| Select-Object @{Name='FreeGB';Expression={[math]::Round($_.FreePhysicalMemory/1024/1024,2)}}` |
| 2. Plugged in (charger connected) | Battery operation throttles RTX 4050 by ~50% |
| 3. Power plan = Best Performance | Settings → System → Power & battery → Power mode → "Best performance" |
| 4. Armoury Crate = Turbo mode | Open Armoury Crate → Operating Mode → Turbo (or Performance if Turbo is unavailable on AC) |
| 5. Laptop on a hard surface | Not bed/blanket — thermal throttle on a multi-hour run is real |
| 6. NVIDIA driver active | Run `nvidia-smi` — should show RTX 4050 with CUDA 12.1+ |
| 7. Disable Windows Update during run | Settings → Windows Update → Pause for 1 week |

### 4B. Activate your Python environment

You said earlier the env is named `cukd_env`. Activate it:

```powershell
cd "C:\N Drive\Acads\6th SEM\WCT\CuKD-XAI"
.\cukd_env\Scripts\Activate.ps1
```

You should see `(cukd_env)` prefix in your prompt.

### 4C. Verify PyTorch sees the GPU

```powershell
python -c "import torch; print('CUDA OK:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Expected output:
```
CUDA OK: True
Device: NVIDIA GeForce RTX 4050 Laptop GPU
```

If CUDA is not available, your training will run on CPU and be ~10× slower. Reinstall the CUDA-enabled PyTorch:
```powershell
pip uninstall -y torch torchvision torchaudio
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

### 4D. Verify v3 .py file syntax (sanity check)

```powershell
python -c "import ast; ast.parse(open('cukd_xai_colab.py', encoding='utf-8').read()); print('PY OK')"
```

Expected: `PY OK`. If you see a SyntaxError, the download was corrupted — re-download.

### 4E. Open the notebook

```powershell
jupyter notebook cukd_xai_colab.ipynb
```

OR open it in VS Code with the Jupyter extension.

---

## 5. v3.0 architecture overview

When you open the notebook, you'll see 46 cells in this order:

| # | Cell | What it does |
|---|---|---|
| 0-2 | Markdown intro + status banner | Read this first |
| 3 | CELL 1 | Install dependencies (already commented out) |
| 4-5 | CELL 2 | **Imports + ALL v3 toggle flags** ← you'll edit this |
| 6-7 | CELL 3 | Load WSN-DS.csv |
| 8-9 | CELL 4 | Preprocess WSN-DS |
| 10-11 | **CELL 4b (NEW v3)** | Domain-informed feature engineering — augments X_all if `USE_ENGINEERED_FEATURES=True` |
| 12-13 | CELL 5 | TeacherMLP + StudentMLP architectures |
| 14-15 | CELL 6 | All training/eval helpers (now includes `train_kd_dkd`, `train_standard_v3`, `train_kd_v3`, EMAWrapper, QAT helpers, architecture-sweep helper) |
| 16-17 | CELL 7 | Difficulty scoring (loss-based + domain-based) |
| 18-19 | CELL 8 | KD hyperparameter grid search |
| 20-21 | CELL 9 | Single-seed `run_all_configs()` — now includes 7 new v3 config keys (DKD + tricks variants) |
| 22-23 | CELL 10 | Multi-seed driver (uses `SEEDS_V3` if `RUN_FULL_10_SEEDS=True`) |
| 24-25 | **CELL 10b (NEW v3)** | Architecture sweep XS/S/M/L/XL — runs after CELL 10 |
| 26-27 | CELL 11 | Aggregate + standard Wilcoxon |
| 28-29 | **CELL 11b (NEW v3)** | Enhanced Wilcoxon with bootstrap CIs + Holm-Bonferroni |
| 30-31 | CELL 12 | SHAP analysis (DeepExplainer + TreeExplainer) |
| 32-33 | CELL 13 | INT8 dynamic quantization (kept from v2.3) |
| 34-35 | **CELL 13b (NEW v3)** | Quantization-Aware Training (QAT) |
| 36-37 | CELL 14 | Inference latency benchmarks |
| 38-39 | CELL 15 | Visualizations — Pareto plot now overlays 5-size architecture sweep |
| 40-41 | **CELL 16 (REPLACED, v3)** | CICIoT2023 generalization with MERGED_CSV streaming |
| 42-43 | CELL 17 | Save all results to JSON + CSVs |
| 44-45 | CELL 18 | Final summary printout |

**Critical concept:** v3 cells are clearly marked with `(v3.0)` or `(NEW, v3.0)` in their header. To revert any v3 feature, just toggle its flag in CELL 2 (no code edits needed).

---

## 6. v3.0 toggle flags explained

All v3 toggles are at the top of CELL 2 (the second code cell, around line 70-150 in the .py). Look for the section labeled `# v3.0 — ...`.

| Flag | Default | What it does | When to set False |
|---|---|---|---|
| `RUN_FULL_10_SEEDS` | `True` | Use 10 seeds (publication run, ~6-8 hr) instead of 5 (~3-4 hr) | Smoke testing |
| `QUICK_MODE` | `False` | If True, uses only 1 seed and skips KD grid search | Pure dry-run only |
| `USE_ENGINEERED_FEATURES` | `True` | Adds 6 engineered features to X_all (drop_rate, forward_ratio, etc.). INPUT_DIM 17→23. | If you want pure v2.3 reproduction |
| `ENGINEERED_FEATURE_TIER` | `2` | 1 (drop-rate trio only), 2 (+control ratios), 3 (+topology context) | Tier 1 for safest baseline |
| `USE_DKD` | `True` | Adds 3 DKD config variants (E_dkd, E2_dkd, F_ext_dkd) | If DKD seems to hurt |
| `DKD_ALPHA` | `1.0` | TCKD weight in DKD loss | Per Zhao 2022 paper |
| `DKD_BETA` | `8.0` | NCKD weight (paper recommends 4-16) | Tune up if DKD underperforms |
| `USE_TRAINING_TRICKS` | `True` | Adds 4 tricks-augmented configs (B_tricks, D_tricks, E_tricks, F_ext_tricks) | If tricks seem to hurt |
| `TRICK_LABEL_SMOOTHING` | `0.05` | Label smoothing on CE term | Disable with 0.0 |
| `TRICK_EMA_DECAY` | `0.999` | EMA decay rate for shadow weights | Disable with 0.0 |
| `TRICK_GRAD_CLIP` | `1.0` | Max gradient norm | Disable with 0.0 |
| `USE_QAT` | `True` | Runs QAT pipeline in CELL 13b after fp32 training | False if QAT hits PyTorch errors |
| `QAT_FT_EPOCHS` | `10` | QAT fine-tuning epochs from fp32 weights | Increase if int8 F1 drops too much |
| `QAT_FT_LR` | `1e-4` | Low LR for stable QAT fine-tuning | Don't change unless needed |
| `RUN_CICIOT` | `True` | Runs CICIoT2023 generalization (CELL 16) | **False for first smoke test** |
| `CICIOT_MERGED_DIR` | `'/content/MERGED_CSV'` | **MUST CHANGE for local run** — see below | n/a |
| `CICIOT_CAP_PER_CLASS` | `50_000` | Per-class sample cap (Web/BruteForce will be smaller floors) | Reduce for faster smoke test |
| `USE_ARCH_SWEEP` | `True` | Runs the 5-size architecture sweep in CELL 10b | False to skip Pareto curve |
| `ARCH_SWEEP_VARIANTS` | XS/S/M/L/XL list | The 5 student architectures | Edit if you want different sizes |

### Critical: change `CICIOT_MERGED_DIR` for local Windows run

In CELL 2, find this line:
```python
CICIOT_MERGED_DIR = '/content/MERGED_CSV'   # Colab default; set absolute path locally
```

**Change it to (use raw string `r"..."`):**
```python
CICIOT_MERGED_DIR = r'C:\N Drive\Acads\6th SEM\WCT\CuKD-XAI\MERGED_CSV\MERGED_CSV'
```

Note the **double `MERGED_CSV\MERGED_CSV`** — the inner folder is where the actual `Merged01.csv ... Merged63.csv` live, based on your earlier folder scan.

---

## 7. Pre-flight verification

Before running anything heavy, verify the v3 setup with these quick checks:

### 7A. Verify v3 markers exist in the notebook

In Jupyter, search (Ctrl+F) for these strings — each should be found:

- `USE_ENGINEERED_FEATURES`
- `def _dkd_loss`
- `class EMAWrapper`
- `def train_qat_from_pretrained`
- `def map_ciciot_label_to_8class`
- `def run_architecture_sweep`
- `def bootstrap_paired_diff`
- `def holm_bonferroni`
- `RUN_FULL_10_SEEDS`

If any are missing, the .ipynb you downloaded is incomplete — re-fetch.

### 7B. Verify the WSN-DS file loads

Run only **CELL 1, 2, 3** of the notebook. Expected output:
```
Shape: (374661, 19)
Columns: [' id', ' Time', ' Is_CH', …]
First row: …
```

If the file is missing, you'll see `FileNotFoundError`. Place `WSN-DS.csv` in the project folder.

### 7C. Verify CICIoT2023 path is correct

In a Python prompt:
```powershell
python -c "import os; p = r'C:\N Drive\Acads\6th SEM\WCT\CuKD-XAI\MERGED_CSV\MERGED_CSV'; print('Exists:', os.path.isdir(p)); print('Files:', len([f for f in os.listdir(p) if f.startswith('Merged')]) if os.path.isdir(p) else 'n/a')"
```

Expected:
```
Exists: True
Files: 63
```

If `Files: 0` — your CICIoT2023 files are at a different path; update `CICIOT_MERGED_DIR` accordingly.

### 7D. Verify CUDA is selected as device

After running CELL 2 (imports), look for:
```
Device: cuda
PyTorch: 2.5.1+cu121
Seeds: [42, 123, 456, 789, 1001, 2024, 3141, 5678, 8192, 9999]
[v3] Engineered features: True (tier 2)
[v3] DKD: True (α=1.0, β=8.0)
[v3] Training tricks: True
[v3] QAT: True
```

If `Device: cpu`, your CUDA is broken — fix per step 4C above.

---

## 8. Smoke test (30 min — RUN THIS FIRST)

**Purpose:** Verify the entire pipeline runs end-to-end on a tiny budget before committing 6-8 hours.

### 8A. Configure for smoke test

In **CELL 2**, temporarily set:
```python
RUN_FULL_10_SEEDS = False    # 5 seeds will be auto-truncated
QUICK_MODE = True             # forces SEEDS = SEEDS[:1] = [42]
USE_QAT = False               # skip QAT for first run (heaviest extra)
RUN_CICIOT = False            # skip CICIoT for first run
USE_ARCH_SWEEP = False        # skip architecture sweep for first run
USE_ENGINEERED_FEATURES = True   # keep v3 feature engineering
USE_DKD = True                # keep v3 DKD configs
USE_TRAINING_TRICKS = True    # keep v3 tricks configs
```

**Translation:** "Run the full v3 codepath but on 1 seed only, skipping the slow optional cells (QAT, CICIoT, arch sweep)."

### 8B. Run cells 1 → 18 in order

In Jupyter: `Cell → Run All`. Or run them one at a time to catch errors early.

### 8C. Expected smoke-test output

| Stage | Expected | Time |
|---|---|---|
| CELL 3 — Load WSN-DS | `Shape: (374661, 19)` | < 5 sec |
| CELL 4 — Preprocess | `Input dim: 17` then `Class distribution: {…}` | < 2 sec |
| CELL 4b — Feature engineering | `New INPUT_DIM: 23` + 6 engineered feature names | < 5 sec |
| CELL 8 — KD grid search | `Best KD hyperparameters: T=2, alpha=0.5` (or similar — depends on seed) | ~3 min |
| CELL 9-10 — Single-seed run on Student A | Per-config F1 printout for ~14 configs | ~30 min |
| CELL 11 — Aggregate + Wilcoxon | DataFrame printout | < 30 sec |
| CELL 11b — v3 Wilcoxon | Bootstrap CIs printout for 11 comparisons | < 30 sec |
| CELL 12 — SHAP | Feature importance printout + saved PNG | ~2 min |
| CELL 13 — INT8 quantization | Per-config size + F1 deltas | < 1 min |
| CELL 15 — Figures | "Saved …" messages | < 30 sec |
| CELL 17 — Save | "Saved cukd_xai_results.json" | < 2 sec |
| CELL 18 — Summary | Final aggregate table | instant |

**Total smoke-test time:** ~35-45 min (including warmup).

### 8D. What to check after smoke test

```powershell
# Run after smoke test completes
python -c "import json; r=json.load(open('cukd_xai_results.json')); print('Top-level keys:', list(r.keys()))"
```

Expected keys (must include the v3 keys):
- `wsn_ds_multi_seed_student_A`
- `wsn_ds_multi_seed_student_B` (empty — only Student A ran in QUICK_MODE)
- `aggregate_student_A`
- `kd_hyperparameters`
- `shap_results`
- `quantization`
- **`qat_results`** (v3, will be `{}` since `USE_QAT=False`)
- **`arch_sweep_results`** (v3, will be `{}` since `USE_ARCH_SWEEP=False`)
- **`arch_sweep_aggregate`** (v3, will be `{}`)
- `inference_benchmarks`
- `ciciot_results` (v3, will be `{}` since `RUN_CICIOT=False`)
- `wilcoxon_results` (will include `v3_student_A` sub-key)
- **`v3_flags`** (v3, snapshot of which toggles were active)
- `seeds`
- `class_names`
- `feature_names` (with the 6 engineered features added if Tier 2)

If all those keys are present and the JSON is valid → **v3 codepath is working**.

---

## 9. Phase-by-phase progressive testing

After the smoke test passes, enable v3 features one at a time to verify each works on its own.

### Step 1: Enable QAT only

In CELL 2:
```python
USE_QAT = True
QUICK_MODE = True       # still 1 seed
```

Re-run only CELL 13b (QAT) — should produce QAT int8 metrics for ~5 configs.

**Expected runtime:** Additional 15-25 min (QAT is CPU-only).
**Expected output:** Per-config printout like `[QAT] E_KD_from_RF (derived hidden_dims=(32, 16)) ... fp32 F1: 0.92 → QAT int8 F1: 0.91 (Δ -1.x%)`

### Step 2: Enable architecture sweep

```python
USE_QAT = True
USE_ARCH_SWEEP = True
QUICK_MODE = True
```

Re-run CELL 10b — should produce 5 student variants per seed.

**Expected runtime:** Additional 15-20 min on RTX 4050.
**Expected output:** Per-variant printout: `arch_XS F1=0.xx params=565 int8=0.55 KB (✓ fits TelosB)`.

### Step 3: Enable CICIoT2023

This is the heaviest test — it streams through the 13 GB MERGED_CSV folder.

```python
RUN_CICIOT = True
QUICK_MODE = True
CICIOT_CAP_PER_CLASS = 5_000   # smoke test cap, not 50K
```

Re-run CELL 16. Expected behavior:
- **Loader phase** (5-15 min): streams Merged01.csv ... onwards, accumulates per-class up to 5,000 each. Stops early.
- **Training phase**: ~15-25 min for 1 seed × 9 configs.

**If loader fails:** check `CICIOT_MERGED_DIR` path. The error message will tell you.

### Step 4: Full smoke test with all v3 features enabled

```python
RUN_FULL_10_SEEDS = False    # still 5 seeds (not 10)
QUICK_MODE = False            # use full seed list (5 seeds)
USE_ENGINEERED_FEATURES = True
USE_DKD = True
USE_TRAINING_TRICKS = True
USE_QAT = True
USE_ARCH_SWEEP = True
RUN_CICIOT = True
CICIOT_CAP_PER_CLASS = 50_000  # full cap
```

This is the **5-seed full v3 run**. Expected total time: ~5-7 hours on RTX 4050.

If this completes cleanly, you're ready for the publication run.

---

## 10. Full publication run (10 seeds)

### 10A. Configuration

In CELL 2:
```python
RUN_FULL_10_SEEDS = True   # 10 seeds for tighter Wilcoxon CIs
QUICK_MODE = False
USE_ENGINEERED_FEATURES = True
ENGINEERED_FEATURE_TIER = 2
USE_DKD = True
USE_TRAINING_TRICKS = True
USE_QAT = True
USE_ARCH_SWEEP = True
RUN_CICIOT = True
CICIOT_MERGED_DIR = r'C:\N Drive\Acads\6th SEM\WCT\CuKD-XAI\MERGED_CSV\MERGED_CSV'
CICIOT_CAP_PER_CLASS = 50_000
```

### 10B. Pre-run checklist (do these EVERY time before a long run)

1. ✓ Charger plugged in
2. ✓ Armoury Crate Turbo mode confirmed
3. ✓ ≥ 8 GB free RAM (close everything except this Jupyter window)
4. ✓ Laptop on hard cooled surface
5. ✓ Screen sleep disabled (`Settings → System → Power & battery → Screen and sleep → Never on AC`)
6. ✓ Windows Update paused for 1 week
7. ✓ ≥ 50 GB free disk space (`(Get-PSDrive C).Free / 1GB`)
8. ✓ No other GPU jobs running (`nvidia-smi` should show only your Python process)

### 10C. Estimated runtime on ASUS TUF A15

| Stage | Expected | Notes |
|---|---|---|
| WSN-DS Student A multi-seed (10 seeds × 14 configs + 7 v3) | ~3.5-4.5 hr | Most compute |
| WSN-DS Student B multi-seed | ~3.5-4.5 hr | Roughly same |
| Architecture sweep (10 seeds × 5 variants) | ~1-1.5 hr | Could be longer if RF teacher slow |
| QAT pipeline | ~1 hr | CPU-only |
| CICIoT2023 (10 seeds × 9 configs, ~287K rows) | ~4-6 hr | Slowest |
| SHAP + figures + saving | ~30 min | |
| **TOTAL** | **~14-18 hours** | Plan to run overnight + half-day |

**Recommendation:** Start the run at 10 PM, let it run overnight + full next day. Monitor via `nvidia-smi -l 5` from a separate terminal.

### 10D. Monitoring during the run

In a SEPARATE PowerShell window:
```powershell
# GPU utilization
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv -l 30

# RAM
Get-CimInstance Win32_OperatingSystem | Select-Object @{Name='FreeGB';Expression={[math]::Round($_.FreePhysicalMemory/1024/1024,2)}}

# Watch the notebook output — Jupyter will keep printing
```

### 10E. Recovery if Jupyter crashes mid-run

The seed-by-seed loop saves intermediate state in memory only — if Jupyter crashes, you lose everything from that run. **Mitigation:**

After every 2-3 seeds, manually run **CELL 17** (save results) to dump partial state to `cukd_xai_results.json`. You can then resume from where it crashed by editing the seed list:

```python
# in CELL 2 if recovering
SEEDS_V3 = [3141, 5678, 8192, 9999]  # only the seeds that didn't run
```

Then re-run CELL 10 onward.

---

## 11. Expected outputs

After a successful full publication run, your project folder will contain:

### 11A. JSON results file

`cukd_xai_results.json` (~500 KB - 2 MB)

Top-level keys:
| Key | Contents |
|---|---|
| `wsn_ds_multi_seed_student_A` | Per-seed × per-config metrics for Student A (32-16 architecture) |
| `wsn_ds_multi_seed_student_B` | Same for Student B (64-32 architecture) |
| `aggregate_student_A` | DataFrame-style mean ± std per config |
| `aggregate_student_B` | (v3 NEW) Same for Student B |
| `kd_hyperparameters` | Best T, alpha from grid search |
| `shap_results` | Feature importance, Spearman ρ between student/teacher |
| `quantization` | PT dynamic INT8 results per config |
| `qat_results` | (v3 NEW) QAT INT8 results per config — **fp32 vs QAT int8 macro F1 deltas** |
| `arch_sweep_results` | (v3 NEW) Per-seed metrics for each of 5 student sizes |
| `arch_sweep_aggregate` | (v3 NEW) Mean ± std per architecture variant |
| `inference_benchmarks` | GPU/CPU latency per config |
| `ciciot_results` | (v3 RECONSTRUCTED) Per-seed × per-config CICIoT2023 metrics + aggregate |
| `wilcoxon_results` | Standard Wilcoxon comparisons + (v3) `v3_student_A`, `v3_student_B` with bootstrap CIs |
| `v3_flags` | (v3 NEW) Snapshot of which v3 toggles were active |
| `seeds` | List of seeds used (10 if RUN_FULL_10_SEEDS=True) |
| `class_names` | `['Blackhole', 'Flooding', 'Grayhole', 'Normal', 'TDMA']` |
| `feature_names` | 23 names (17 raw + 6 engineered) if Tier 2 |

### 11B. CSV files

- `wsnds_results_student_A.csv` — aggregate metrics table for Student A
- `wsnds_results_student_B.csv` — aggregate metrics table for Student B

### 11C. PNG figures

- `per_class_f1.png` — per-class F1 across configs (mean ± std)
- `confusion_matrix_E.png` — Config E (KD from RF) confusion matrix
- `confusion_matrix_F.png` — Config F confusion matrix
- `confusion_matrix_F_fair.png` and `_ext.png` — F variants
- `pareto_frontier.png` — **v3 ENHANCED** — now includes the 5-size architecture sweep curve (XS/S/M/L/XL) overlaid on the existing per-config points, with TelosB 10 KB region shaded in green
- `shap_summary_student.png` — SHAP feature importance for the student
- `loss_curves_B_vs_C.png` — training loss curves comparing CL teacher (Config C) to baseline (Config B)

### 11D. Saved model checkpoints

The current code does NOT save model weights. If you want this for the paper, manually save after CELL 10:

```python
# Add to CELL 17 if you want weights saved
import pickle
with open('final_models.pkl', 'wb') as f:
    pickle.dump({k: v.state_dict() for k, v in final_models.items()}, f)
```

---

## 12. Result interpretation guide

After the run finishes, interpret the JSON output to assess what worked. Use this Python snippet (run in a new cell or separate script):

```python
import json
import numpy as np

with open('cukd_xai_results.json') as f:
    r = json.load(f)

# 1. Headline aggregate
print("=" * 70)
print("WSN-DS aggregate (Student A, mean ± std across seeds)")
print("=" * 70)
configs_to_check = [
    'A_RF_500',                       # teacher
    'B_Full_MLP',                     # baseline teacher
    'D_Small_MLP',                    # scratch student
    'E_KD_from_RF',                   # v2.3 main result
    'E_KD_from_RF_dkd',               # v3 DKD variant
    'E_KD_from_RF_tricks',            # v3 tricks variant
    'F_KD_from_CL_MLP_ext',           # CL student
    'F_KD_from_CL_MLP_ext_dkd',       # v3 DKD on CL
    'F_KD_from_CL_MLP_ext_tricks',    # v3 tricks on CL
]
for cfg in configs_to_check:
    seed_data = r['wsn_ds_multi_seed_student_A']
    f1s = [seed_data[s][cfg]['macro_f1'] for s in seed_data if cfg in seed_data[s]]
    if f1s:
        print(f"  {cfg:34s} F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}  (n={len(f1s)})")

# 2. Did v3 features help?
print("\n" + "=" * 70)
print("v3 deltas (positive = v3 helped)")
print("=" * 70)
for v3, base in [('E_KD_from_RF_dkd', 'E_KD_from_RF'),
                 ('E_KD_from_RF_tricks', 'E_KD_from_RF'),
                 ('F_KD_from_CL_MLP_ext_dkd', 'F_KD_from_CL_MLP_ext'),
                 ('F_KD_from_CL_MLP_ext_tricks', 'F_KD_from_CL_MLP_ext')]:
    sd = r['wsn_ds_multi_seed_student_A']
    f1_v3 = [sd[s][v3]['macro_f1'] for s in sd if v3 in sd[s]]
    f1_b = [sd[s][base]['macro_f1'] for s in sd if base in sd[s]]
    if f1_v3 and f1_b:
        delta = np.mean(f1_v3) - np.mean(f1_b)
        sign = '+' if delta >= 0 else ''
        print(f"  {v3:34s} vs {base:30s} Δ = {sign}{delta:.4f}")

# 3. Wilcoxon significance
print("\n" + "=" * 70)
print("v3 Wilcoxon (Student A) — significant after Holm-Bonferroni?")
print("=" * 70)
v3w = r['wilcoxon_results'].get('v3_student_A', {})
for key, w in v3w.items():
    sig = '✓ sig' if w.get('significant_after_correction') else '  ns'
    p = w.get('wilcoxon_p_holm_bonferroni')
    print(f"  {key:55s} p_adj = {p:.4f}  {sig}")

# 4. QAT delta (v3)
print("\n" + "=" * 70)
print("QAT int8 vs fp32 macro F1 delta (closer to 0 is better)")
print("=" * 70)
for cfg, qres in r.get('qat_results', {}).items():
    if isinstance(qres, dict) and 'f1_delta_pct' in qres:
        print(f"  {cfg:34s} fp32 = {qres['fp32_macro_f1']:.4f} → QAT int8 = {qres['qat_int8_macro_f1']:.4f}  (Δ {qres['f1_delta_pct']:+.2f}%)")

# 5. Architecture sweep — Pareto frontier
print("\n" + "=" * 70)
print("Architecture sweep — student size vs accuracy")
print("=" * 70)
asa = r.get('arch_sweep_aggregate', {})
for cfg, a in asa.items():
    fits = '✓ fits TelosB' if a.get('fits_telosb_10kb') else '✗ exceeds TelosB'
    print(f"  {cfg:10s} F1 = {a['macro_f1_mean']:.4f} ± {a['macro_f1_std']:.4f}  params = {a['params']:>6d}  int8 = {a['size_kb_int8_theoretical']:6.2f} KB  ({fits})")

# 6. CICIoT2023 generalization
print("\n" + "=" * 70)
print("CICIoT2023 generalization — aggregate")
print("=" * 70)
cic_agg = r.get('ciciot_results', {}).get('aggregate', {})
for cfg, a in cic_agg.items():
    print(f"  {cfg:30s} F1 = {a['macro_f1_mean']:.4f} ± {a['macro_f1_std']:.4f}  acc = {a['accuracy_mean']:.4f}")
```

### 12B. What outcomes mean for your paper

| Result pattern | What it means | Paper framing |
|---|---|---|
| `E_dkd > E` significantly | DKD is a real improvement | "DKD improves tree-to-neural KD on WSN-DS by X.X pp (p<0.05)" |
| `E_dkd ≈ E` | DKD neutral; honest negative | "DKD does not improve on vanilla KD in this setting (p>0.05)" — still a valid contribution |
| `E_tricks > E` significantly | Tricks help | "Training tricks bundle adds X.X pp" |
| `Engineered features help` (E with FE > E without) | Workstream A succeeded | "Domain-informed features close X.X pp of the teacher gap" |
| `QAT int8 within 1% of fp32` | QAT works | "QAT enables int8 deployment with <1% F1 degradation, vs 2-12% for PT dynamic quantization" |
| `Architecture sweep shows knee` | Pareto frontier is clear | "Below 1,000 params, F1 drops sharply; above 4,000, returns diminish" |
| `CICIoT2023 F1 reasonable` | Generalization holds | "Pipeline generalizes to a second IoT IDS dataset" |

### 12C. Compare against your prof's 4% gap target

Prof asked Student F1 ≥ **0.9391** (4% gap from RF teacher 0.9790).

**Check: did any v3 config reach ≥ 0.9391 on Student A?**
- v2.3 baseline: Student A E_KD_from_RF = 0.9205 (gap 5.85% — short)
- v3 target: any of E_dkd / E_tricks / F_ext_tricks ≥ 0.9391

If yes → declare success.
If no → Student B at 0.9345 (3.32 KB) is your best bet; pivot the paper to "Student B at 3.32 KB INT8 with 4.5% gap" + "Student A at 1.16 KB INT8 with 6% gap" as Pareto positioning.

---

## 13. Common errors and fixes

| Error / Symptom | Likely cause | Fix |
|---|---|---|
| `FileNotFoundError: WSN-DS.csv` | File not in project folder | `Invoke-WebRequest -Uri "https://github.com/nishantharkut/CuKD-XAI/raw/main/WSN-DS.csv" -OutFile "WSN-DS.csv"` |
| `KeyError: 'DATA_R'` in CELL 4b | Column name mismatch after `.str.strip()` | Verify CELL 3 ran successfully and printed the columns |
| `CUDA out of memory` | 6 GB VRAM exceeded | Reduce `TRAIN_CONFIG['batch_size']` from 256 to 128 in CELL 2 |
| `RuntimeError: ... in [some QAT op]` | PyTorch QAT bug | Set `USE_QAT = False`. The fp32 results are still valid for the paper. |
| `FileNotFoundError: No Merged*.csv in {dir}` | `CICIOT_MERGED_DIR` path wrong | Verify path with the Python check in step 7C above |
| `ValueError: 'Label' column missing` | Reading wrong CSV files | Make sure you're pointing at MERGED_CSV (not the per-attack CSV folders) |
| Notebook hangs on a cell | Likely RF teacher training (single-threaded sklearn) | Wait — RF on 287K rows takes 5-15 min per seed |
| Python kernel crashes | RAM OOM | Close all other apps; reduce `CICIOT_CAP_PER_CLASS` to 25,000 |
| Tunnel-downloaded file fails to parse | Download corruption | Verify SHA256: `Get-FileHash cukd_xai_colab.py -Algorithm SHA256` should match `5f2425c51df5f904b4ab226012dfa92c907bc260867c7b3fc3471226b56d44cf` |
| `nvidia-smi: command not found` | NVIDIA driver not installed | Download the latest Studio/Game Ready driver from nvidia.com |
| Battery drains during run | Charger came unplugged | Replug; if Jupyter is still running, accept the temporary throttle |
| Thermal throttling (temp >95°C) | Insufficient cooling | Use a laptop cooling pad; stop run if temp >100°C |

---

## 14. Performance expectations on ASUS TUF A15

### 14A. Per-seed runtimes (rough estimates)

| Stage | Per-seed time | Notes |
|---|---|---|
| RF (500 trees, calibrated, on 262K rows) | 5-10 min | CPU-bound, single-threaded sklearn |
| Teacher MLP (40 epochs, 262K rows, batch 256) | 2-3 min | RTX 4050 |
| 2 CL teachers (fair + ext budget) | 4-6 min | Same as B |
| 7 student configs (E, E2, F variants, etc.) | 5-7 min | Each <1 min |
| 3 v3 DKD variants | 3-4 min | Same as KD |
| 4 v3 tricks variants (with EMA) | 5-7 min | Slightly slower due to EMA shadow updates |
| **Per seed total (Student A)** | **~25-35 min** | |
| **Student B (same)** | **~25-35 min** | |

For 10 seeds × 2 students = ~10-13 hours just for WSN-DS multi-seed loop.

### 14B. Architecture sweep (CELL 10b)

5 variants × 10 seeds × 1 config (KD-from-RF only) = 50 trainings + 50 RF teachers (one per seed, reused across variants) = ~3-5 hours.

### 14C. CICIoT2023 (CELL 16)

- Loader: 5-15 min (one-time)
- Per seed (9 configs): ~25-35 min
- 10 seeds: ~4-6 hours

### 14D. QAT (CELL 13b)

- Per config: ~5-8 min CPU
- 6 configs: ~30-50 min total

### 14E. Cumulative

**Full v3 run on RTX 4050 ≈ 14-18 hours.**

If budget is tighter, switch to lab GPU (e.g., RTX 3060 Ti or A100 if available) — runtime will be ~3-4× faster.

---

## 15. Comparison with v2.3 baseline

If you want to compare v3 against v2.3 to defend the v3 changes in your paper:

### 15A. Re-clone the v2.3 baseline

```powershell
git clone https://github.com/nishantharkut/CuKD-XAI cukd_v23_baseline
cd cukd_v23_baseline
git checkout 6250c37  # the commit you pushed before v3 modifications
# This gives you the original v2.3 .py and .ipynb
```

### 15B. Run v2.3 with the same 5 seeds

In `cukd_v23_baseline/cukd_xai_colab.py`, ensure:
```python
SEEDS = [42, 123, 456, 789, 1001]   # same as v3 first 5 seeds
QUICK_MODE = False
RUN_CICIOT = False
```

Then run end-to-end. Save as `cukd_xai_results_v23.json`.

### 15C. Side-by-side comparison

```python
import json
v23 = json.load(open('cukd_v23_baseline/cukd_xai_results_v23.json'))
v3 = json.load(open('cukd_xai_results.json'))

# Compare E_KD_from_RF (the headline config) on Student A across same 5 seeds
for s in ['42', '123', '456', '789', '1001']:
    f1_v23 = v23['wsn_ds_multi_seed_student_A'][s]['E_KD_from_RF']['macro_f1']
    f1_v3 = v3['wsn_ds_multi_seed_student_A'][s]['E_KD_from_RF']['macro_f1']
    print(f"Seed {s}: v2.3 = {f1_v23:.4f}  v3 = {f1_v3:.4f}  Δ = {f1_v3 - f1_v23:+.4f}")
```

If `Δ > 0` consistently → v3 is a real improvement.
If `Δ ≈ 0` → v3 features didn't help on standard configs (but still added new configs E_dkd, E_tricks, etc. for ablation).

---

## 16. Roadmap to paper submission

Once the run completes and v3 results are in:

### 16A. Week 1 — Verify and analyze
- [ ] Run the result-interpretation Python snippet (Section 12A)
- [ ] Confirm v3 toggles are saved correctly in `v3_flags`
- [ ] Identify which v3 configs improved over baseline
- [ ] Confirm at least Student B reached your prof's 4% gap target

### 16B. Week 2 — Generate paper-ready figures
- [ ] Pareto frontier with 5-architecture sweep + TelosB region (already in v3)
- [ ] Per-class F1 bars for the winning config
- [ ] Bland-Altman plots if reporting QAT comparison
- [ ] Confusion matrix for the winning config
- [ ] Statistical significance heatmap from `v3_student_A` Wilcoxon

### 16C. Week 3 — Write paper draft
- Use the comparison table from `btech_research_topics.md` for related work positioning
- Section 1 (Intro): cite Ghadi 2024 base paper, Talukder 2025 SOTA
- Section 3 (Methods): describe the v3 pipeline (engineered features → DKD → tricks → QAT → architecture sweep)
- Section 4 (Results): WSN-DS 10-seed table + CICIoT2023 generalization + Pareto frontier
- Section 5 (Discussion): honest negative results (e.g., DKD neutral) + ethical/limitations

### 16D. Week 4 — Submit

Target venues (in priority order):
1. **IEEE Journal of Biomedical and Health Informatics** (Q1) — IF ~7
2. **Computers in Biology and Medicine** (Q1, Elsevier) — IF ~7
3. **Biomedical Signal Processing and Control** (Q1, Elsevier) — IF ~5.1
4. **IEEE Sensors Journal** (Q1) — for the deployment angle
5. **MDPI Sensors** (Q2) — backup

For an undergrad-led submission targeting Q1, **start at Computers in Biology and Medicine** — its review cycle is faster (~3 months) and acceptance rate is friendlier than IEEE TMI.

---

## 17. When to ask Claude for help

The deeper-issue scenarios where you should come back and ask:

| Situation | What to share with Claude |
|---|---|
| Smoke test passes, but full 10-seed run fails mid-way | The full Jupyter cell output where it crashed, RAM/GPU usage at crash time |
| `cukd_xai_results.json` produced, but result interpretation seems off | Paste the JSON's top-level summary; ask for sanity check |
| QAT int8 F1 drops more than 5% | Per-config delta from CELL 13b; ask whether to tune `QAT_FT_EPOCHS` or skip |
| CICIoT2023 loader times out | Loader's print output; ask whether to reduce `CICIOT_CAP_PER_CLASS` |
| Some configs perform much worse than expected | Per-config table from result-interpretation snippet; ask whether it's a real problem or seed variance |
| You want to add a new config or workstream | Describe the change; I'll write the targeted edit to CELL 9 |
| Paper reviewer asks for a specific ablation | Reviewer's exact question; I'll write the cell that produces the answer |

**Don't ask for help with:**
- Routine debugging (`KeyError`, missing imports) — fix per Section 13
- Setting up Python on Windows — Section 4 covers this
- Running Jupyter — standard tutorials online

---

## 18. Final pre-flight checklist before starting the publication run

Print this out and tick each item:

```
[ ] cukd_xai_colab.py present in project folder
[ ] cukd_xai_colab.ipynb present in project folder
[ ] WSN-DS.csv present (~26 MB)
[ ] MERGED_CSV folder accessible at the configured path
[ ] cukd_env Python environment activated
[ ] PyTorch sees CUDA (RTX 4050)
[ ] CICIOT_MERGED_DIR set to your local Windows absolute path with raw string r"..."
[ ] All v3 toggles set as desired (RUN_FULL_10_SEEDS=True, etc.)
[ ] Charger plugged in
[ ] Armoury Crate Turbo mode
[ ] ≥ 8 GB free RAM
[ ] ≥ 50 GB free disk space
[ ] Laptop on hard cooled surface
[ ] Windows Update paused for 1 week
[ ] Screen sleep disabled while plugged in
[ ] Discord/Slack/Edge/Chrome closed
[ ] Smoke test (Section 8) completed and passed
[ ] Phase-by-phase progressive tests (Section 9) all passed
[ ] Set start time before sleeping (e.g., 10 PM)
[ ] Plan to check progress at 9 AM next morning
[ ] cukd_xai_results.json renamed to add timestamp BEFORE re-running (so you don't overwrite a previous good run)
```

---

## Appendix A — Restoring v2.3 baseline

If v3 misbehaves and you want to revert to v2.3:

### Option 1: Re-clone GitHub
```powershell
git clone https://github.com/nishantharkut/CuKD-XAI cukd_v23_recovery
# Use cukd_v23_recovery/cukd_xai_colab.py and .ipynb as your baseline
```

### Option 2: Git history
If you've been version-controlling locally:
```powershell
git log --oneline cukd_xai_colab.py
git checkout <commit-hash> cukd_xai_colab.py
```

### Option 3: Toggle all v3 flags off (effective revert without file changes)
Set in CELL 2:
```python
USE_ENGINEERED_FEATURES = False
USE_DKD = False
USE_TRAINING_TRICKS = False
USE_QAT = False
USE_ARCH_SWEEP = False
RUN_CICIOT = False
RUN_FULL_10_SEEDS = False
```

This makes the v3 file behave functionally like v2.3 without removing any code.

---

## Appendix B — File hashes for verification

After downloading, verify with PowerShell:

```powershell
Get-FileHash cukd_xai_colab.py -Algorithm SHA256
# Expected: 5f2425c51df5f904b4ab226012dfa92c907bc260867c7b3fc3471226b56d44cf
# (this hash is for the v3.0 file as of May 2026 build)

Get-FileHash cukd_xai_colab.ipynb -Algorithm SHA256
# Expected: ae474a08ccfecd6b28d9a2ca9c5031d3613f60ae602ba895ef3e93053eb6fc8e
```

If the hash doesn't match → re-download from the cloudflare tunnel (or ask Claude to host again).

---

## Appendix C — Quick-reference command cheatsheet

| Task | Command |
|---|---|
| Activate env | `.\cukd_env\Scripts\Activate.ps1` |
| Verify CUDA | `python -c "import torch; print(torch.cuda.is_available())"` |
| Verify .py syntax | `python -c "import ast; ast.parse(open('cukd_xai_colab.py', encoding='utf-8').read())"` |
| Open notebook | `jupyter notebook cukd_xai_colab.ipynb` |
| GPU monitor (separate window) | `nvidia-smi -l 30` |
| Free RAM check | `Get-CimInstance Win32_OperatingSystem \| Select-Object @{Name='FreeGB';Expression={[math]::Round($_.FreePhysicalMemory/1024/1024,2)}}` |
| Free disk check | `(Get-PSDrive C).Free / 1GB` |
| Backup current results before re-run | `Copy-Item cukd_xai_results.json "cukd_xai_results.$(Get-Date -Format 'yyyyMMdd_HHmm').json"` |

---

**End of guide.** If anything is unclear, ask before running — a misconfiguration costs hours, not minutes.
