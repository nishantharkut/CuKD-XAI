# CuKD-XAI Colab Notebook — Quick Start

## Files

- **`cukd_xai_colab.ipynb`** — Main Jupyter notebook (upload to Colab)
- **`cukd_xai_colab.py`** — Same code as .py (reference only)
- **`CuKD_XAI_MASTER_DOCUMENT.md`** — Full research context

## How to run (10 min setup)

### Step 1: Open in Colab
1. Go to https://colab.research.google.com
2. File → Upload notebook → select `cukd_xai_colab.ipynb`
3. Runtime → Change runtime type → **GPU (T4)**

### Step 2: Get WSN-DS dataset
**Option A (easiest):** Upload manually
1. Click the Files icon (📁) in left sidebar
2. Download WSN-DS from https://www.kaggle.com/datasets/bassamkasasbeh1/wsnds
3. Upload `WSN-DS.csv` to Colab root

**Option B:** Kaggle API
```
!pip install -q kaggle
!mkdir -p ~/.kaggle
!echo '{"username":"YOUR_USERNAME","key":"YOUR_KEY"}' > ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d bassamkasasbeh1/wsnds
!unzip -q wsnds.zip
```

### Step 3: Install deps
In Cell 1, uncomment and run:
```
!pip install -q shap scikit-learn pandas numpy matplotlib seaborn torch
```
(Most are preinstalled on Colab — only `shap` usually needs installing.)

### Step 4: Run all cells
Runtime → Run all (or Ctrl+F9)

Expected total runtime on Colab T4 GPU: **~15-25 minutes** for single-seed run of all 10 configs.

## What the notebook does

| Cell | Config | What it trains |
|------|--------|----------------|
| 7 | A | Random Forest baseline (500 trees) |
| 8 | B | Full MLP teacher (~70K params) |
| 9 | D | Small MLP from scratch (1,189 params) |
| 10 | — | Difficulty scoring for CL |
| 11 | C | CL-trained MLP teacher |
| 13 | E | KD: calibrated RF → student |
| 14 | E2 | KD: standard MLP → student |
| 15 | F | **KD: CL-MLP → student (CORE CLAIM)** |
| 16 | G | KD: random-pacing teacher → student (control) |
| 17 | — | Results summary table |
| 18 | H | SHAP explainability analysis |
| 19 | — | Comparison figures |

## Key outputs

After running:
- `results.json` — all metrics for all configs
- `shap_summary.png` — per-attack feature importance
- `per_class_f1.png` — comparison figure

## Core comparisons to look for

1. **Does CL help the teacher?** Compare `C_CL_MLP` vs `B_Full_MLP` macro F1
2. **Does KD beat scratch?** Compare `E2_KD_from_MLP` vs `D_Small_MLP_scratch`
3. **Does CL improve KD? (CORE CLAIM)** Compare `F_KD_from_CL_MLP` vs `E2_KD_from_MLP`
4. **Is it curriculum or just pacing?** Compare `F_KD_from_CL_MLP` vs `G_KD_random_pacing`

## After first run

Once the single-seed run completes successfully:
1. Wrap the whole pipeline in a loop for 5 different random seeds
2. Compute mean ± std
3. Run Wilcoxon signed-rank tests for F vs E2, F vs D, C vs B
4. Then run on CICIoT2023 for generalizability

## Troubleshooting

- **OOM on Colab:** Reduce batch_size in train_model from 256 to 128
- **SHAP error:** `pip install shap --upgrade` — DeepExplainer sometimes breaks with newer PyTorch
- **Kaggle download fails:** Use Option A (manual upload)
- **Column name mismatch:** WSN-DS on Kaggle sometimes has `' Attack type'` (with leading space) — the notebook already handles this with `str.strip()`

## Verified architecture numbers

| Model | Params | Size fp32 | Size INT8 | TelosB fit? |
|-------|--------|-----------|-----------|-------------|
| RF (500 trees) | N/A | 5-50 MB | 5-50 MB | ❌ |
| Teacher MLP | ~69,893 | ~273 KB | ~68 KB | ❌ |
| Student (32-16) | 1,189 | 4.64 KB | 1.16 KB | ✅ |
| Student (64-32) | 3,397 | 13.27 KB | 3.32 KB | ✅ |

## Hyperparameters (verified from Benaddi et al. 2025)

| Param | Value | Source |
|-------|-------|--------|
| Temperature T | 4.0 | Benaddi grid {2,3,4} |
| Alpha (KD weight) | 0.7 | Benaddi grid {0.5,0.7,0.9} |
| Optimizer | AdamW | Benaddi |
| LR | 1e-3 | Benaddi |
| Weight decay | 1e-3 | Benaddi |
| Dropout (teacher) | 0.2-0.3 | Benaddi |
| Batch size | 256 | Standard |
| Epochs (teacher) | 25 (3 CL stages × 7/7/11) | Our design |
| Epochs (student KD) | 30 | Standard |
| Early stopping patience | 8 | Benaddi |
