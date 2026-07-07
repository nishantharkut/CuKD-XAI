# Running CuKD-XAI Notebook Locally

**Applies to:** `cukd_xai_colab.ipynb` (v2.3 or later)
**Why this file exists:** You ran the v2.0 experiment locally (not on Colab/Kaggle). This document captures the setup steps so you can do it again for v2.3.
**Last updated:** April 11, 2026
**Your primary GPU (confirmed):** NVIDIA RTX 4050 laptop (6 GB VRAM, Ada Lovelace, CUDA 12.x). Integrated AMD Radeon will NOT be used by PyTorch.

---

## 0. START HERE — Which path are you on?

Answer these three questions first:

### Question 1: Do you have an NVIDIA GPU?
- **YES** → follow the **GPU path** (Sections 2-5 have specific instructions marked [GPU])
- **NO (CPU only, or Mac with M1/M2/M3, or AMD GPU)** → follow the **CPU path** (Section 1.5 + CPU-marked instructions)
- **Not sure?** Run `nvidia-smi` in a terminal. If it prints a table with your GPU name → YES. If "command not found" → NO.

### Question 2: Windows, Linux, or Mac?
Different activation commands and path conventions. Each section shows all three.

### Question 3: First time running, or re-running?
- **First time** → start from Section 1 (Prerequisites)
- **Re-running** (environment from v2.0 still exists) → go to Section 0.5 (Quick restart)

---

## 0.5 Quick restart (if your environment from the previous run still exists)

If you already ran v2.0 on this machine, your venv and dataset are probably still in place.

1. `cd` into the project folder
2. Activate your venv (see Section 2)
3. **Back up previous results first** (the re-run will overwrite them):
   ```bash
   # Linux / macOS / WSL
   mkdir -p backups/v2.0_results_$(date +%Y%m%d)
   cp -v results_download/cukd_xai_results.json results_download/*.png results_download/*.csv backups/v2.0_results_$(date +%Y%m%d)/ 2>/dev/null
   ```
   ```powershell
   # Windows PowerShell
   $stamp = Get-Date -Format "yyyyMMdd"
   New-Item -ItemType Directory -Force "backups\v2.0_results_$stamp"
   Copy-Item results_download\*.json, results_download\*.png, results_download\*.csv "backups\v2.0_results_$stamp\" -ErrorAction SilentlyContinue
   ```
4. Verify PyTorch still detects the right device:
   ```bash
   python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('MPS:', getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU/MPS')"
   ```
5. Go to Section 10 (QUICK_MODE validation run first, then full run)

If the venv is gone, start fresh from Section 2.

---

## 1.5 CPU-only path (no NVIDIA GPU) — SELF-CONTAINED SECTION

**Teammates without an RTX/GTX GPU: this section is your complete guide. Read it end-to-end, then skip directly to Section 4 (dataset) and Section 5 (run). You can IGNORE Sections 1, 3's GPU instructions, and the GPU-specific troubleshooting.**

**Nishant (RTX 4050 user): skip this section and go to Section 1.**

### CPU hardware guidance

| CPU | Expected time for 5-seed full run (Student A only) |
|---|---|
| Recent laptop i5/Ryzen 5 (8 cores) | 8-14 hours |
| Recent desktop i7/Ryzen 7 (12-16 cores) | 4-8 hours |
| Low-end / older CPU | 12-24 hours (consider reducing scope — see below) |
| Apple M1/M2/M3 (with MPS backend) | 2-4 hours (surprisingly fast; see "Mac MPS" below) |

**CPU users: strongly recommended scope reductions.** These cut runtime in half to two-thirds:

In Cell 2 (`Imports and global config`), set:

```python
# CPU-friendly configuration
SEEDS = [42, 123]          # 2 seeds instead of 5 (still enough for mean±std, weak Wilcoxon)
QUICK_MODE = False          # keep the grid search + Student B
RUN_CICIOT = False          # skip CICIoT2023 (adds 1-3 more hours on CPU)

TRAIN_CONFIG = {
    'epochs': 20,           # reduced from 30 (fewer full passes)
    'batch_size': 512,      # LARGER batch on CPU = fewer python loop iterations = faster
    'lr': 1e-3,
    'weight_decay': 1e-3,
    'patience': 6,          # early-stop sooner to save time
}

CL_STAGES_FAIR = [(0.33, 2), (0.66, 2), (1.0, 16)]   # 20 total, matches reduced B budget
CL_STAGES_EXT  = [(0.33, 3), (0.66, 3), (1.0, 20)]   # 26 total, still extended
```

And in Cell 9 (`run_all_configs`), optionally skip Config I (SMOTE) by commenting out the `# ----- Config I:` block — SMOTE expands training data by 4.5× and Config I alone can take 2-4 hours on CPU.

**These reductions are OPTIONAL but practical.** The core comparisons (F_fair vs E2, C_fair vs B, SHAP Spearman) all still work with 2 seeds and 20 epochs. If you have time, keep the full config.

### CPU PyTorch install

```bash
# Linux / macOS / WSL
pip install numpy pandas scikit-learn scipy matplotlib seaborn shap imbalanced-learn jupyter ipykernel
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

```powershell
# Windows PowerShell
pip install numpy pandas scikit-learn scipy matplotlib seaborn shap imbalanced-learn jupyter ipykernel
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Do NOT install the default `pip install torch`** on Windows/Linux if you want CPU-only — it'll try to grab ~2 GB of CUDA libraries you won't use. Use the `/whl/cpu` index URL to get a ~200 MB CPU-only wheel.

### Verify PyTorch is actually using CPU

```bash
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Threads:', torch.get_num_threads())"
```

Expected output on CPU-only:
```
Torch: 2.x.x+cpu
CUDA: False
Threads: <number of your CPU cores>
```

If `Torch: 2.x.x+cu121` or similar — you accidentally installed the GPU version. Reinstall with the `+cpu` URL.

### Mac Apple Silicon (M1/M2/M3) specific — MPS backend

Recent PyTorch supports Apple's Metal GPU via the MPS backend. **Our notebook's device detection already handles this** — if MPS is available, it uses GPU, not CPU. So Mac users should install the standard PyTorch:

```bash
pip install torch
```

Then in the notebook's Cell 2, the device detection line:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

has a subtle issue: it will always pick `cpu` on a Mac, ignoring the available MPS backend. **Mac users should manually change this to:**
```python
if torch.cuda.is_available():
    device = torch.device('cuda')
elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
```

Expected runtime on M1 Pro / M2 / M3: ~2-4 hours for the full 5-seed Student A run (comparable to a low-end NVIDIA GPU).

**Known MPS caveats:**
- Some SHAP DeepExplainer internals fall back to CPU (slow). Consider skipping SHAP on MPS and running it separately on CPU afterwards if it hangs.
- `torch.quantization.quantize_dynamic` produces CPU-only quantized models (not MPS). This is fine — our quant section already moves to CPU.

### AMD dedicated GPU on Linux (ROCm) — rarely applicable

If someone has a dedicated AMD Radeon RX 6xxx/7xxx card on Linux with ROCm installed, PyTorch supports it:

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
```

But: integrated AMD, Windows AMD, and anything older than RX 6xxx are NOT supported. If in doubt, use the CPU path.

---

## 1. Prerequisites

### For all paths

| Item | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11 |
| RAM | 8 GB (skip Config I and reduce batch) | 16 GB+ (everything works) |
| Disk | 2 GB free | 5 GB |
| OS | Linux / macOS / Windows | any |

### For NVIDIA GPU path (your primary setup — RTX 4050 laptop)

| Item | Requirement |
|---|---|
| GPU VRAM | 4 GB minimum (your RTX 4050 has 6 GB — plenty) |
| CUDA | 12.1 for RTX 40 series (11.8 works for RTX 20/30 series) |
| NVIDIA driver | 525+ for CUDA 12.1. Verify with `nvidia-smi` |

### For CPU path (teammates without GPU — see Section 1.5)

| Item | Requirement |
|---|---|
| CPU | Any modern x86-64 or Apple Silicon |
| RAM | **16 GB strongly recommended** (SMOTE peaks ~5 GB) |
| Patience | 4-14 hours depending on CPU power |

**Training time estimates (full 5-seed run, v2.3, Student A + B, no CICIoT):**

| Hardware | Full run | QUICK_MODE (1 seed, Student A) |
|---|---|---|
| RTX 4090 / H100 | ~45-60 min | ~10 min |
| RTX 4050 laptop (your machine) | ~75-105 min | ~15-20 min |
| Colab T4 (reference point) | ~90-120 min | ~20 min |
| Apple M2 / M3 with MPS | ~120-240 min | ~30-50 min |
| Modern desktop CPU (16 cores, no GPU) | ~4-8 hours | ~1 hour |
| Modern laptop CPU (8 cores, no GPU) | ~8-14 hours | ~1.5-2 hours |
| Older/low-end CPU | ~12-24 hours | ~2-4 hours |

Add ~60-90 min (GPU) or ~3-5 hours (CPU) if `RUN_CICIOT = True`.

**CPU users should strongly consider the scope reductions in Section 1.5** — they cut runtime ~60% without losing the core research questions.

**VRAM usage on RTX 4050 (6 GB):**
- Teacher MLP forward pass: ~50-80 MB peak
- SMOTE-expanded dataset (Config I): ~100-200 MB
- Peak during training: ~1-2 GB
- **Plenty of headroom** — 6 GB is more than enough for this notebook

**Integrated AMD Radeon note:** PyTorch does not support integrated AMD GPUs. Even on Linux with ROCm, only dedicated AMD GPUs (Vega, RDNA discrete cards) are supported. Your integrated Radeon is irrelevant — the RTX 4050 handles all GPU work.

---

## 2. Environment setup (applies to BOTH GPU and CPU paths)

### Option A — `venv` (simplest, Python standard library)

**Linux / macOS / WSL2:**
```bash
cd /path/to/wct
python3 -m venv cukd_env
source cukd_env/bin/activate
pip install --upgrade pip
# DON'T pip install -r requirements.txt yet — see Section 3 for RTX 4050 GPU install order
```

**Windows PowerShell (native, not WSL):**
```powershell
cd C:\path\to\wct
python -m venv cukd_env
.\cukd_env\Scripts\Activate.ps1
# If you get a policy error, run this once as admin:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
python -m pip install --upgrade pip
```

**Windows Command Prompt (cmd.exe):**
```cmd
cd C:\path\to\wct
python -m venv cukd_env
cukd_env\Scripts\activate.bat
python -m pip install --upgrade pip
```

### Option B — `conda` (if you already use Anaconda / Miniconda)

```bash
conda create -n cukd python=3.11 -y
conda activate cukd
pip install -r requirements.txt
```

### Option C — System Python (not recommended, but quickest if you already have everything)

```bash
pip install --user torch scikit-learn pandas numpy matplotlib seaborn shap imbalanced-learn scipy jupyter
```

---

## 3. Dependencies — GPU path

> **CPU-only users: skip this section.** You already installed everything in Section 1.5. Go to Section 4.

A ready-to-use `requirements.txt` already exists in the project folder. Contents:

```
numpy>=1.24,<2.0
pandas>=2.0
scikit-learn>=1.3,<1.6
scipy>=1.11
matplotlib>=3.7
seaborn>=0.12
shap>=0.44,<0.46
imbalanced-learn>=0.11
torch>=2.0,<2.5
jupyter>=1.0
ipykernel>=6.25
```

### For your RTX 4050 laptop — exact installation commands

**Step 1: Install non-Torch dependencies first (they're the same for CPU/GPU):**

```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn shap imbalanced-learn jupyter ipykernel
```

**Step 2: Install Torch with CUDA 12.1 support (for RTX 40 series):**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

(Do NOT use `pip install torch` from regular PyPI — you'll get CPU-only torch even if you have a GPU.)

**Step 3: Verify GPU detection:**

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('CUDA version:', torch.version.cuda); print('Torch version:', torch.__version__)"
```

Expected output on your machine:
```
CUDA available: True
Device: NVIDIA GeForce RTX 4050 Laptop GPU
CUDA version: 12.1
Torch version: 2.x.x+cu121
```

If `CUDA available: False`:
- Check `nvidia-smi` works in a terminal → if it doesn't, install/update the NVIDIA driver (minimum 525 for CUDA 12.1)
- On Windows: make sure you rebooted after the NVIDIA driver install
- On WSL2: install the NVIDIA CUDA WSL driver from https://developer.nvidia.com/cuda/wsl (install it on the Windows host, NOT inside WSL)
- Reinstall torch: `pip uninstall torch -y && pip install torch --index-url https://download.pytorch.org/whl/cu121`

### Alternative: single install-from-requirements command

If you prefer, after editing `requirements.txt` to comment out the `torch` line, run:

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 4. Get the WSN-DS dataset

The notebook expects `WSN-DS.csv` in the same directory as the notebook.

**Option 1 — Manual download from Kaggle (simplest):**
1. Go to https://www.kaggle.com/datasets/bassamkasasbeh1/wsnds
2. Click "Download" (requires a free Kaggle account)
3. Unzip and move `WSN-DS.csv` into the project folder (same directory as `cukd_xai_colab.ipynb`)

**Option 2 — Kaggle CLI:**
```bash
pip install kaggle
# Place your kaggle.json API key in ~/.kaggle/kaggle.json (Linux/Mac)
# or %USERPROFILE%\.kaggle\kaggle.json (Windows)
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d bassamkasasbeh1/wsnds
unzip wsnds.zip -d .
```

**Option 3 — If you already have `WSN-DS.csv` somewhere else on disk:**
Edit Cell 2 of the notebook and change:
```python
WSNDS_PATH = 'WSN-DS.csv'
```
to your actual path, e.g.:
```python
WSNDS_PATH = '/home/you/datasets/WSN-DS.csv'
```

**Expected file properties after download:**
- File name: `WSN-DS.csv`
- Size: ~63 MB
- Rows: 374,661 (including header)
- Columns: 19

---

## 5. Run the notebook

### Option A — Jupyter in a browser

```bash
cd /path/to/wct
source cukd_env/bin/activate
jupyter notebook cukd_xai_colab.ipynb
```

A browser window will open. Click **Kernel → Restart & Run All**.

### Option B — VS Code

1. Open the folder in VS Code
2. Install the Python + Jupyter extensions (if not already installed)
3. Open `cukd_xai_colab.ipynb`
4. Select the Python interpreter from `cukd_env` (bottom-right corner)
5. Click **Run All** at the top

### Option C — JupyterLab

```bash
pip install jupyterlab
jupyter lab cukd_xai_colab.ipynb
```

### Option D — Run as a .py script (skipping Jupyter entirely)

```bash
cd /path/to/wct
source cukd_env/bin/activate
python3 cukd_xai_colab.py
```

This works but:
- You lose the Status Banner display (it's a markdown cell)
- Any cell-by-cell debugging is harder
- All `print()` output streams to terminal
- Generated figures are saved as PNGs in the current directory (not inline)

Useful if running in a headless server or tmux session.

---

## 6. Comment out Colab-specific lines (if any are present)

v2.3 of the notebook no longer has `from google.colab import drive` or similar, but older versions did. If you see errors like `ModuleNotFoundError: No module named 'google.colab'`, remove the offending lines. They're not needed for local running.

Cell 1 (`!pip install ...`) is a comment in v2.3; uncomment only if you need to install deps from inside the notebook. Generally do the install outside the notebook (step 3 above) and leave this cell commented.

---

## 7. Common errors and fixes

### `FileNotFoundError: WSN-DS.csv`
- The dataset file isn't where the notebook expects it. Either:
  - Move `WSN-DS.csv` into the same folder as `cukd_xai_colab.ipynb`
  - Or edit `WSNDS_PATH` in Cell 2 to point to the actual location

### `ModuleNotFoundError: No module named 'torch'`
- Dependencies not installed. Activate your venv and run `pip install -r requirements.txt`

### `RuntimeError: CUDA out of memory`
- GPU VRAM exhausted. Happens with small GPUs (<8 GB) and the large training set.
- Fix: In Cell 2, reduce batch size:
  ```python
  TRAIN_CONFIG = {
      'epochs': 30,
      'batch_size': 128,   # reduced from 256
      'lr': 1e-3,
      'weight_decay': 1e-3,
      'patience': 8,
  }
  ```

### `MemoryError` or `killed (OOM)` during SMOTE
- Config I (SMOTE-trained teacher) expands the training set from 262K to ~1.2M samples. Needs ~5 GB RAM.
- Fix: Skip Config I on low-memory machines. In `run_all_configs` in Cell 9, comment out the `# ----- Config I:` block, or set it to pass silently.

### SHAP DeepExplainer throws `AttributeError` or hook errors
- Known compatibility issue between PyTorch versions and SHAP versions.
- Fix: `pip install --upgrade shap`. If that fails, try `pip install shap==0.44.0` (a known-good version).

### `sklearn.calibration.CalibratedClassifierCV` takes forever
- Isotonic calibration refits RF 3 times (cv=3). This is slow on the full dataset.
- Fix: In Cell 9 Config E section, change `cv=3` to `cv=2` for faster calibration (slightly less reliable soft targets but significantly faster).

### Training is slower than expected on GPU
- Check that PyTorch is actually using the GPU:
  ```python
  import torch
  print(torch.cuda.is_available())          # Should be True
  print(torch.cuda.current_device())
  print(torch.cuda.get_device_name(0))
  ```
- If CUDA is not available but you have a GPU, you installed the CPU-only PyTorch. Reinstall with the CUDA index URL (see step 3).

### Notebook kernel dies silently
- Usually RAM exhaustion. Check system monitor during the run.
- Fix: Reduce batch size, skip Config I, or run `QUICK_MODE = True` with fewer seeds.

### Matplotlib errors / "no display" / Tkinter errors
- Happens when running as a script on a headless machine or via SSH without X11 forwarding.
- Fix: Set the backend to `Agg` (file-only, no display needed) before importing matplotlib:
  ```bash
  export MPLBACKEND=Agg      # Linux / macOS / WSL
  $env:MPLBACKEND = "Agg"    # Windows PowerShell
  ```
  Then run the notebook. Figures save to PNG files as expected.
- All plot code in this notebook uses `plt.savefig()` + `plt.close()` and never calls `plt.show()`, so `Agg` works fine.

### `nvidia-smi` works but PyTorch still can't see the GPU (RTX 4050 laptop specific)
- NVIDIA Optimus / hybrid graphics is switching between the integrated Radeon and dedicated RTX 4050. Python might be launched on the integrated GPU.
- Fix (Windows): Right-click the Python/Jupyter shortcut → "Run with graphics processor" → "High-performance NVIDIA processor"
- Fix (Linux laptop): Install `nvidia-prime` and run `prime-select nvidia` before starting Python, OR use `__NV_PRIME_RENDER_OFFLOAD=1` prefix when launching
- Or permanently set GPU preference in BIOS/UEFI (look for "hybrid graphics" / "dGPU only")

### Laptop thermal throttling during long runs
- The RTX 4050 laptop cooler isn't designed for 90-minute sustained loads. Expect ~5-15% slowdown as clocks throttle.
- Fix: Elevate the laptop for airflow, use a cooling pad, or set Windows power plan to "Best Performance" / run `sudo cpupower frequency-set -g performance` on Linux
- Not a correctness issue — just longer wall-clock time.

### [CPU path] Accidentally installed CUDA PyTorch but I don't have an NVIDIA GPU
- Symptom: `pip install torch` downloaded ~2 GB and `torch.cuda.is_available()` returns False but torch version says `+cu121` or similar.
- Fix: uninstall and reinstall from the CPU wheel:
  ```bash
  pip uninstall torch -y
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```
- Saves ~2 GB disk and startup time. No functional difference.

### [CPU path] My laptop is unusable — all CPU cores at 100%, browser/editor freezes
- PyTorch defaults to using ALL available CPU cores for training. On a laptop with 8 cores you can leave 2 cores free for OS/browser.
- Fix: In Cell 2 after `import torch`, add:
  ```python
  torch.set_num_threads(6)  # leave 2 cores free on an 8-core laptop
  ```
- Reduces CPU utilization from 100% to ~75% and keeps the laptop responsive. ~10% slower overall but worth it.

### [CPU path] Training looks frozen — no output for 10+ minutes
- On CPU, a single training epoch on the full WSN-DS (262K samples × batch 256) takes 60-180 seconds depending on CPU. If you see "Seed 42" printed and then nothing for 10 minutes, that's normal — it's training silently.
- To see progress: In Cell 9 `run_all_configs`, change `verbose=False` to `verbose=True` (or the individual `train_standard(..., verbose=True)` calls). You'll see per-epoch prints.
- Sanity check it's not actually frozen: open Task Manager / `htop` — CPU should be at high utilization (60-100% on multiple cores).

### [CPU path] Runtime is unacceptably long
- Apply the scope reductions in Section 1.5 (fewer seeds, smaller epochs, skip Config I).
- With scope reductions: ~3-6 hours instead of 8-14 hours on a laptop.
- If even that's too long: run only the absolutely essential configs manually. In `run_all_configs`, comment out configs C2, G, I. Keep A, B, C_fair, D, E, E2, F_fair. That's 7 configs instead of 10, cutting ~30% more time.

---

## 8. Outputs produced by a successful run

After `Runtime → Run all` finishes, these files should appear in the project directory:

```
cukd_xai_results.json              # main results (~100 KB)
wsnds_results_student_A.csv        # aggregate metrics for Student A
wsnds_results_student_B.csv        # aggregate metrics for Student B (if Student B ran)
per_class_f1.png                   # per-class F1 bar chart (8 configs)
confusion_matrix_E.png             # Config E confusion matrix
confusion_matrix_F.png             # Config F (canonical alias = fair) confusion
confusion_matrix_F_fair.png        # Config F fair-budget confusion
confusion_matrix_F_ext.png         # Config F extended-budget confusion
pareto_frontier.png                # Model size vs macro F1 (both student sizes)
shap_summary_student.png           # SHAP global feature importance
loss_curves_B_vs_C.png             # B vs C_fair vs C_ext training dynamics
```

---

## 9. Save your results safely

After a successful run, **copy the output files to a dated backup folder** before you accidentally re-run and overwrite them:

```bash
mkdir -p backups/run_$(date +%Y%m%d_%H%M%S)
cp cukd_xai_results.json wsnds_results_student_A.csv *.png backups/run_$(date +%Y%m%d_%H%M%S)/
```

Or if you're on Windows PowerShell:
```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
New-Item -ItemType Directory -Path "backups\run_$stamp"
Copy-Item cukd_xai_results.json, wsnds_results_student_A.csv, *.png "backups\run_$stamp\"
```

---

## 10. Running `QUICK_MODE` first (STRONGLY recommended)

Before committing to a full 90+ minute run, validate the pipeline in ~20 minutes:

In **Cell 2 (Imports and global config)**, set:

```python
SEEDS = [42]
QUICK_MODE = True
RUN_CICIOT = False
```

Then run all cells. If the single-seed run completes without error, flip back to:

```python
SEEDS = [42, 123, 456, 789, 1001]
QUICK_MODE = False
```

and re-run for the full 5-seed experiment.

---

## 11. What NOT to do

- **Do not run `python3 make_notebook.py`** — it regenerates the .ipynb from the .py and wipes the Status Banner. It now has a confirmation prompt, but still: don't run it unless you have a backup of the .ipynb.
- **Do not run multiple Python processes training simultaneously on the same GPU** — they'll contend for memory and everything will OOM.
- **Do not commit `cukd_xai_results.json` to git without checking the file size first** — it can be >1 MB with bootstrap SHAP values.
- **Do not modify `CL_STAGES_FAIR` or `CL_STAGES_EXT` without understanding the compute-budget fairness argument** — the fair budget exactly matches Config B; changing it invalidates the comparison.

---

## 12. Feeling lost? Start here

1. `RESUME_HERE.md` — 2-minute project orientation
2. **This file** — 5-minute local setup
3. `cukd_xai_colab.ipynb` Cell 1 — 3-minute version history + critical findings
4. `CuKD_XAI_MASTER_DOCUMENT.md` Parts 17-19 — deep dive on experimental results and paper narrative
5. `CuKD_XAI_EXECUTION_PLAN.md` Part 10 — day-by-day next actions

Total cold-start to decision-ready: ~20 minutes.

---

*If anything in this file is wrong or missing, update it based on what actually worked on your machine so the next time you come back (or a teammate joins), the instructions match reality.*
