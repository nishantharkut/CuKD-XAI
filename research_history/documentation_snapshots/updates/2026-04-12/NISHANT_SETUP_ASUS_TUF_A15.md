# Nishant's Setup Guide — ASUS TUF Gaming A15 FA507NU

**Machine:** ASUS TUF Gaming A15 FA507NU_FA577NU
**OS:** Windows 11 Home Single Language (Build 26200)
**CPU:** AMD Ryzen 7 (Family 25 = Zen 3+/Rembrandt, ~1978 MHz base) — likely Ryzen 7 7735HS (8C/16T)
**GPU (dedicated):** NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB VRAM
**GPU (integrated):** AMD Radeon 680M (iGPU in Ryzen 7735HS) — will NOT be used by PyTorch
**RAM:** 16 GB (15,611 MB visible)
**Storage:** NVMe SSD (standard for this model)
**Last updated:** April 11, 2026

This is a **walkthrough tailored to your exact machine**. Teammates on different hardware should use `LOCAL_RUN_SETUP.md` instead.

---

## 0. Why a separate file for you?

Your ASUS TUF A15 has three specific quirks that affect ML training:

1. **Hybrid graphics (NVIDIA Optimus)** — Python might launch on the integrated AMD Radeon by default, missing the RTX 4050 entirely. Needs explicit GPU selection.
2. **Virtualization-Based Security (VBS/HVCI) is running** — Confirmed from your `systeminfo` output. VBS adds ~5-15% CPU overhead on Windows 11. Not a blocker but worth knowing.
3. **ASUS Armoury Crate power profiles** — The laptop has three modes (Silent / Performance / Turbo) that dramatically affect GPU and CPU clocks. Default "Silent" on battery will halve training speed.

---

## 1. Pre-flight checks (5 minutes, do BEFORE anything else)

### Check 1: RAM availability

Your `systeminfo` showed **only 4,767 MB available** out of 16 GB — that means ~11 GB was already in use by other apps (Chrome, Edge, VS Code, etc.).

**Close unnecessary apps before the run.** For the full 5-seed experiment you need at least **6-8 GB free**.

To check current free RAM in PowerShell:
```powershell
Get-CimInstance Win32_OperatingSystem | Select-Object @{Name='FreeGB';Expression={[math]::Round($_.FreePhysicalMemory/1024/1024,2)}}
```

Target: **FreeGB ≥ 6.0** before starting the notebook.

### Check 2: Laptop is plugged in

The RTX 4050 laptop GPU runs at significantly lower power on battery. Plug in the charger. Check battery icon is in "charging" state.

### Check 3: Power plan = Best Performance

1. Settings → System → Power & battery → Power mode → **"Best performance"**
2. Or via PowerShell:
   ```powershell
   powercfg /setactive SCHEME_MIN
   ```
3. Verify: `powercfg /getactivescheme`

### Check 4: ASUS Armoury Crate = Turbo mode

1. Open **Armoury Crate** (pre-installed ASUS app)
2. Navigate to **Operating Mode**
3. Select **Turbo** (if available on AC power)
4. If no Turbo, select **Performance**
5. **Do not run in Silent mode** — it throttles both CPU and GPU.

If Armoury Crate is missing, you can download it from the ASUS support page for FA507NU, or skip this check and just use Windows power plan.

### Check 5: NVIDIA driver version

Open Command Prompt or PowerShell and run:
```cmd
nvidia-smi
```

You should see a table with:
- `NVIDIA-SMI 525.xx` or higher (any driver ≥ 525 works for CUDA 12.1)
- `NVIDIA GeForce RTX 4050 Laptop GPU`
- `Driver Version: 525.xx+` (current as of April 2026 is 555+)
- `CUDA Version: 12.1` (or higher)

**If `nvidia-smi` says "command not found":**
1. Download latest NVIDIA Studio or Game Ready driver from https://www.nvidia.com/Download/index.aspx
2. Select: RTX 40 Series → GeForce RTX 4050 → Laptop → Windows 11 64-bit
3. Install and reboot

**If `nvidia-smi` works but shows a different GPU or "no devices found":**
- Your laptop is in iGPU-only mode. Fix in Check 6 below.

### Check 6: Force NVIDIA dGPU for Python

Windows 11 has a per-app GPU preference. Python processes might default to the iGPU, making PyTorch's `torch.cuda.is_available()` return False even though you have an RTX 4050.

**Set Python to always use the dGPU:**

1. Settings → System → Display → **Graphics**
2. Click **"Browse"**
3. Navigate to your Python installation (e.g., `C:\Users\nhnis\AppData\Local\Programs\Python\Python311\python.exe`)
4. Select it and click Add
5. Click python.exe → **Options**
6. Choose **"High performance (NVIDIA)"**
7. Click Save
8. **Repeat for every Python executable you use** — include `py.exe`, `pythonw.exe`, and any venv Python (`C:\path\to\cukd_env\Scripts\python.exe`)

For Jupyter specifically: also add the Jupyter launcher (wherever you invoke `jupyter notebook` from) to the graphics preferences.

**Alternative (blunter but effective):** Disable the integrated AMD Radeon entirely in Device Manager. This forces all GPU work to the RTX 4050 but breaks some Windows features (display refresh, power saving). Not recommended — use the per-app preference instead.

### Check 7: VBS/HVCI status (optional)

Your `systeminfo` confirms **Virtualization-Based Security is running** with **Hypervisor enforced Code Integrity**. This adds 5-15% CPU overhead but is Windows 11's security default.

**For maximum training speed, you could disable VBS** (trade-off: reduced security):
```powershell
# Check current VBS status
Get-CimInstance -ClassName Win32_DeviceGuard -Namespace root\Microsoft\Windows\DeviceGuard
# If VirtualizationBasedSecurityStatus is 2, VBS is running

# Disable via Group Policy Editor (Pro only) or registry:
# gpedit.msc → Computer Configuration → Administrative Templates → System → Device Guard → Turn On VBS → Disabled
```

**I recommend leaving VBS enabled.** The 5-15% slowdown adds maybe 10-15 minutes to an 80-minute run — not worth weakening security for.

---

## 2. Install Python + dependencies

### Step 1: Python

Check if Python 3.11 is already installed:
```cmd
python --version
```

If it prints `Python 3.11.x`, skip to Step 2.

If not installed or wrong version:
1. Go to https://www.python.org/downloads/windows/
2. Download **Python 3.11.9** (or latest 3.11.x — NOT 3.12 yet, some packages lag)
3. Run the installer
4. **IMPORTANT: Check "Add Python to PATH"** at the bottom of the first installer screen
5. Click "Install Now"
6. After install, open a NEW PowerShell window (so PATH takes effect) and verify:
   ```cmd
   python --version
   pip --version
   ```

### Step 2: Create the virtual environment

Open **PowerShell** (not Command Prompt — PowerShell handles activation scripts better):

```powershell
cd C:\Users\nhnis\Downloads\wct   # or wherever the wct folder is
python -m venv cukd_env
```

First-time activation policy fix (only needed once):
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Press Y when prompted
```

Activate:
```powershell
.\cukd_env\Scripts\Activate.ps1
```

Your prompt should now show `(cukd_env)` prefix. If yes, venv is active.

Upgrade pip inside the venv:
```powershell
python -m pip install --upgrade pip
```

### Step 3: Install non-Torch dependencies

```powershell
pip install numpy pandas scikit-learn scipy matplotlib seaborn shap imbalanced-learn jupyter ipykernel
```

Expected install time on your RTX 4050 laptop with wired internet: ~2-4 minutes.

### Step 4: Install PyTorch with CUDA 12.1 (critical for RTX 4050)

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

This downloads ~2 GB of CUDA libraries. Expected: 3-8 minutes depending on internet speed.

**Do NOT use plain `pip install torch`** — that pulls the CPU-only version which won't use your RTX 4050.

### Step 5: Verify GPU detection

```powershell
python -c "import torch; print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

**Expected output on your machine:**
```
Torch: 2.x.x+cu121
CUDA available: True
CUDA version: 12.1
Device: NVIDIA GeForce RTX 4050 Laptop GPU
```

**If `CUDA available: False`:**
1. Go back to Check 6 (force dGPU) — this is the most common cause
2. Check `nvidia-smi` still works in a terminal
3. Reinstall torch: `pip uninstall torch -y && pip install torch --index-url https://download.pytorch.org/whl/cu121`
4. If still failing, restart Windows (sometimes required after driver changes)

---

## 3. Get the WSN-DS dataset

You probably already have `WSN-DS.csv` from the v2.0 run. If yes, skip to Section 4.

If not:
1. Go to https://www.kaggle.com/datasets/bassamkasasbeh1/wsnds
2. Sign in (create free account if needed)
3. Click **Download** — gets `archive.zip`
4. Extract `WSN-DS.csv` from the zip
5. Move `WSN-DS.csv` into your `wct` folder (same place as `cukd_xai_colab.ipynb`)

Verify:
```powershell
dir WSN-DS.csv
```
Should show: ~63 MB file.

---

## 4. Run the notebook

### Recommended: Jupyter in browser

With the venv still activated:

```powershell
cd C:\Users\nhnis\Downloads\wct   # or your actual path
.\cukd_env\Scripts\Activate.ps1    # if not already activated
jupyter notebook cukd_xai_colab.ipynb
```

Jupyter opens in your default browser. You should see:
- Cell 0: Main title
- Cell 1: "🚨 IF YOU'RE OPENING THIS AFTER THE MAY EXAMS" banner
- Cell 2: "STATUS BANNER"
- Cell 3 onwards: Install dependencies, imports, etc.

### Alternative: VS Code

1. Open VS Code
2. File → Open Folder → select the `wct` folder
3. Install Python + Jupyter extensions if prompted
4. Open `cukd_xai_colab.ipynb`
5. Click the kernel selector in top-right → choose **cukd_env (Python 3.11.x)**
6. Click **Run All** at the top

### First run: QUICK_MODE validation (~20 minutes)

Before a full run, validate the pipeline:

In **Cell 4 (Imports and global config)** of the notebook, find and edit:
```python
SEEDS = [42]           # was [42, 123, 456, 789, 1001]
QUICK_MODE = True      # was False
```

Then Runtime → Run all (Jupyter) or Run All (VS Code).

**Expected duration on your machine: 15-20 minutes.**

During the run:
- You'll see training progress printed for each config
- GPU utilization should be 30-70% (check in Task Manager → Performance → GPU)
- RAM usage should be 6-10 GB total (system + Python)
- CPU temp may reach 85-95°C (normal for a laptop under training load)
- Fan noise will be loud — this is expected

If the run completes, you're ready for the full 5-seed experiment.

### Full run (~80-100 minutes)

After QUICK_MODE succeeds, edit Cell 4 again:
```python
SEEDS = [42, 123, 456, 789, 1001]
QUICK_MODE = False
```

Runtime → Run all.

**During the 80-100 minute run:**
- Don't close the laptop lid (the notebook will suspend)
- Don't unplug (battery mode throttles GPU)
- You can minimize Jupyter but don't close the browser tab
- **Do not run other GPU-heavy apps** (games, video encoding, etc.) — they'll fight for VRAM
- Chrome / Edge / VS Code are fine to use for light browsing

### Output files (appear in `wct/` folder after successful run)

```
cukd_xai_results.json              (~100 KB)
wsnds_results_student_A.csv
wsnds_results_student_B.csv
per_class_f1.png
confusion_matrix_E.png
confusion_matrix_F.png
confusion_matrix_F_fair.png
confusion_matrix_F_ext.png
pareto_frontier.png
shap_summary_student.png
loss_curves_B_vs_C.png
```

---

## 5. Save results safely

As soon as the run completes, back up the output files. PowerShell:

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backup = "backups\v2.3_run_$stamp"
New-Item -ItemType Directory -Force $backup | Out-Null
Copy-Item cukd_xai_results.json, *.csv, *.png $backup
Write-Host "Backed up to $backup"
```

This copies all the output files into a timestamped `backups/` subfolder. If you re-run later, the new run will overwrite the latest files but your backup is safe.

---

## 6. Common ASUS TUF A15 + Windows 11 issues

### `CUDA available: False` even after Check 6

- **Cause:** Windows 11 sometimes reverts Graphics preferences after driver updates.
- **Fix:** Re-do Check 6 (force High Performance NVIDIA for python.exe). Reboot.
- **Prevention:** After any NVIDIA driver update, re-verify with the GPU detection snippet.

### `ImportError: DLL load failed while importing _C` (PyTorch)

- **Cause:** Missing Visual C++ Redistributable on Windows 11. Rare but happens.
- **Fix:** Download and install https://aka.ms/vs/17/release/vc_redist.x64.exe, then restart.

### Training pauses randomly for 5-10 seconds

- **Cause:** Windows Defender scanning the output files or Python processes.
- **Fix:** Add the `wct` folder and `cukd_env` folder to Windows Defender exclusions:
  Settings → Privacy & security → Windows Security → Virus & threat protection → Manage settings → Exclusions → Add an exclusion → Folder → select each folder.

### Fan maxes out / laptop becomes very hot

- **Cause:** Sustained GPU + CPU load. Normal for a laptop.
- **Fix:** Elevate the laptop (even two books under the back) improves airflow ~10-15°C. Consider a laptop cooling pad for long runs.
- **Not a bug:** The notebook will still complete successfully.

### Battery drain warning during training

- **Shouldn't happen** if plugged in. If you see this, Windows is detecting the power draw exceeds the 180W charger's capacity — happens briefly during peak load. Ignore unless laptop actually runs down.

### PowerShell says "`.\cukd_env\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled on this system`"

- **Cause:** Default Windows execution policy.
- **Fix:** Run once (as your user, not admin): `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- Press Y. This allows locally-created scripts to run.

### Config I (SMOTE) crashes with OOM or kills kernel

- **Cause:** SMOTE expands training data from 262K to ~1.2M samples. With only 16 GB RAM and Windows+VS Code already eating ~5 GB, this can hit the limit.
- **Fix 1:** Close ALL other apps before running. Target: 10+ GB free RAM.
- **Fix 2:** Skip Config I. In Cell 9 `run_all_configs`, comment out the `# ----- Config I:` block.
- **Fix 3:** Increase Windows page file. Settings → System → About → Advanced system settings → Performance Settings → Advanced → Change virtual memory → set to "Custom: initial 16384 MB, max 32768 MB".

### SHAP DeepExplainer hangs or errors

- **Known issue** with some PyTorch/SHAP version combos.
- **Fix:** `pip install shap==0.44.1` (a known-good version for PyTorch 2.x)
- If still failing: the rest of the notebook will still run — SHAP is wrapped in try/except.

---

## 7. After the run

1. **Check the 4 critical numbers** printed at the end:
   - `C_CL_MLP_loss_fair` macro F1 → does CL help teacher at equal budget?
   - `C_CL_MLP_loss_ext` macro F1 → does CL help teacher with more budget?
   - `F_KD_from_CL_MLP_fair` macro F1 → does CL cascade to KD at equal budget?
   - `F_KD_from_CL_MLP_ext` macro F1 → does CL cascade to KD with more budget?
2. Compare against `B_Full_MLP` (~0.921 in v2.0) and `E2_KD_from_MLP` (~0.911 in v2.0)
3. See `CuKD_XAI_MASTER_DOCUMENT.md` Part 19 for the decision tree on what paper narrative to commit to

---

## 8. If something goes wrong and you need help

1. Don't panic
2. Screenshot the error message (or copy/paste the stack trace)
3. Read `LOCAL_RUN_SETUP.md` Section 7 (general troubleshooting) and Section 6 of THIS file (Windows-specific)
4. If still stuck: open a new Claude session, share this file and the error, Claude should be able to diagnose
5. Worst case: comment out the failing cell and continue — most sections are independent. You'll lose data for that config but the rest runs.

---

## 9. Quick reference card (print / pin this)

```
VENV ACTIVATE:    .\cukd_env\Scripts\Activate.ps1
RUN NOTEBOOK:     jupyter notebook cukd_xai_colab.ipynb
CHECK GPU:        python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
CHECK FREE RAM:   Get-CimInstance Win32_OperatingSystem | Select FreePhysicalMemory
KILL ZOMBIE PY:   Get-Process python | Stop-Process -Force
VENV DEACTIVATE:  deactivate

Run order for re-run after May exams:
  1. Plug in laptop
  2. Close Chrome/Edge/other apps
  3. Open Armoury Crate → Turbo mode
  4. cd to wct folder
  5. Activate venv
  6. Verify GPU with torch.cuda.is_available()
  7. Back up previous results (backups/)
  8. Open notebook
  9. Set QUICK_MODE=True for validation run (~20 min)
 10. If successful, set QUICK_MODE=False for full run (~90 min)
```

---

*This file is specific to Nishant's ASUS TUF A15. For teammates on different hardware, see `LOCAL_RUN_SETUP.md`. Last updated April 11, 2026.*
