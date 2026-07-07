# 🚨 RESUME HERE — Read This First

**Last updated:** April 11, 2026
**Next expected session:** After May 4 exams
**Project:** CuKD-XAI research paper (WSN intrusion detection)
**Status:** Paused for exams. One decision gate blocked on a 15-minute Colab re-run.

---

## The 60-second catch-up

You were working on a research paper that compresses a Random Forest (~85 MB) into a tiny MLP (1.16 KB INT8) for wireless sensor network intrusion detection using knowledge distillation. You ran the first full experiment on April 10-11 and got these results:

**Good news:**
- The core compression story works: KD from a calibrated Random Forest teacher gets macro F1 = 0.9219 with 1,189 params (18,335× compression)
- You found a novel finding: compressed student and teacher SHAP rankings are uncorrelated (Spearman ρ ≈ 0) — this is a genuine contribution to the literature
- All 5-seed statistical validation done for Student A

**Bad news:**
- The original CL-guided KD hypothesis FAILED: Config F (KD from CL teacher) = 0.8908, Config E2 (KD from standard MLP) = 0.9114. CL hurt the student by 2%.
- The CL teacher itself (Config C = 0.8607) is 6% WORSE than the standard MLP (Config B = 0.9211). CL hurt the teacher.

**The pending decision:**
Is the CL failure a **bug** (Stage 3 under-training) or a **fundamental finding** (CL doesn't help tabular IDS, per Wu et al. ICLR 2021)?

You already applied the bug-fix hypothesis to the notebook (v2.2 with `CL_STAGES = [(0.33,5),(0.66,5),(1.0,30)]` and per-stage fresh optimizer). You need to run it once to see if Config C now reaches ≥ 0.91.

---

## Your first action when you return (~20 minutes on your ASUS TUF A15)

**⚠️ NOT Colab — you run locally on Windows.**

For your ASUS TUF Gaming A15 FA507NU (Windows 11 + RTX 4050 + Ryzen + 16 GB), the complete step-by-step is in:
**`NISHANT_SETUP_ASUS_TUF_A15.md`** ← read this first

Fast path (if the venv from your v2.0 run still exists):
1. Plug laptop into charger, open Armoury Crate → set to **Turbo** mode
2. Close Chrome/Edge/other apps (need 8+ GB free RAM)
3. Open **PowerShell**, `cd` to the wct folder
4. Back up previous results: see Section 0.5 of `NISHANT_SETUP_ASUS_TUF_A15.md`
5. Activate venv: `.\cukd_env\Scripts\Activate.ps1`
6. Verify GPU: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` — must print `True NVIDIA GeForce RTX 4050 Laptop GPU`
7. Launch notebook: `jupyter notebook cukd_xai_colab.ipynb`
8. In **Cell 2 (Imports and global config)**, set:
   ```python
   SEEDS = [42]
   QUICK_MODE = True
   ```
9. Runtime → Run All → wait ~20 minutes
10. Check Config C macro F1:
    - **≥ 0.91** → CL fix worked → set SEEDS back to 5 and QUICK_MODE=False for full run
    - **Still ~0.86** → pivot to backup narrative (see Master Doc Part 19)

**If the venv is gone**, follow `NISHANT_SETUP_ASUS_TUF_A15.md` from Section 2 (full reinstall, ~15 minutes extra).

**Teammates without an RTX GPU** → they should read `LOCAL_RUN_SETUP.md` Section 0 to pick their path (GPU / CPU / Mac MPS).

Then send yourself this note (or tell Claude) which outcome you got, and the next step is obvious from the decision tree in `CuKD_XAI_EXECUTION_PLAN.md` Part 6.

---

## Which file answers which question

| Question | File |
|---|---|
| "What am I doing right now?" | **This file** (`RESUME_HERE.md`) |
| "How do I run the notebook on **my ASUS TUF A15 Windows laptop**?" | **`NISHANT_SETUP_ASUS_TUF_A15.md`** ← primary runner guide for your machine |
| "How does a teammate without an NVIDIA GPU run it?" | `LOCAL_RUN_SETUP.md` Section 1.5 (CPU path) |
| "How does a teammate on Mac / other GPU / Linux run it?" | `LOCAL_RUN_SETUP.md` Section 0 (decision tree) |
| "What's the next step of this week/day?" | `CuKD_XAI_EXECUTION_PLAN.md` Part 5 + Part 10 |
| "What's the full research context?" | `CuKD_XAI_MASTER_DOCUMENT.md` v3.0 (Parts 17-19 are current; Parts 1-16 are historical design) |
| "What are the exact results from the last run?" | `results_download/cukd_xai_results.json` + `STATE_SNAPSHOT_2026_04_11.md` |
| "Which competitor papers must I cite?" | `CuKD_XAI_EXECUTION_PLAN.md` Part 8 + Part 11 |
| "What does this notebook cell do?" | `cukd_xai_colab.ipynb` Cell 1 (STATUS BANNER) |
| "Was a claim verified?" | `CuKD_XAI_EXECUTION_PLAN.md` Part 12 (Verification Audit Log) |

---

## Critical warnings

### ⚠️ DO NOT run `python3 make_notebook.py`

This script regenerates `cukd_xai_colab.ipynb` from `cukd_xai_colab.py` and **will overwrite the Status Banner cell** at the top of the notebook. The Status Banner is your primary orientation tool inside Colab and is NOT stored in the .py file.

If you accidentally run it, restore the banner by asking Claude to regenerate it, or by re-reading this file.

### ⚠️ The notebook has 6 applied bug fixes from v2.1 → v2.2

Look for `# FIXED 2026-04-11:` comments in `cukd_xai_colab.py`. These are:
1. `CL_STAGES` changed from `[(0.33,7),(0.66,7),(1.0,11)]` to `[(0.33,5),(0.66,5),(1.0,30)]`
2. `train_with_curriculum` now uses fresh per-stage optimizer+scheduler
3. Wilcoxon results now persisted to JSON (previously only printed)
4. INT8 quantization now sweeps all 6 student configs (previously only Config F)
5. Inference benchmarks now cover 7 models (previously 3)
6. Per-class SHAP Spearman correlation added

**These fixes are NOT in the previous run's results** (which used the buggy v2.0 code). When you re-run the notebook, the new results will be different from what's in `results_download/`.

### ⚠️ The memory file is OUTSIDE the repo

The Claude memory file at `~/.claude/projects/-home-ubuntu-nishant-workspace-local-new-feat-wct/memory/project_apr11_results_and_competitors.md` contains detailed competitor analysis. A copy has been dumped into this repo as `STATE_SNAPSHOT_2026_04_11.md` for portability. If you restart Claude or switch machines, you still have this file.

---

## The 4 things that are NOT done yet (in priority order)

### 1. Run the CL fix validation (~15 min) — BLOCKING

The single blocking decision. Until this runs, you don't know which paper narrative to commit to. Do this first.

### 2. Run the full 5-seed experiment with fixes (~60-90 min)

Only do this AFTER step 1 confirms the CL fix direction. Re-runs all 10 configs × 5 seeds with the corrected CL + all 6 fixes above.

### 3. Run CICIoT2023 generalization (~60-90 min)

Non-negotiable for publication. Reviewers will reject a single-dataset paper. Upload `CICIoT2023.csv` (sampled to 400K rows) and set `RUN_CICIOT = True`.

### 4. Run Student B (64-32-5) multi-seed (~30-40 min)

For the Pareto frontier figure. Less critical than CICIoT but still important.

---

## The 6 competitors you MUST cite (verified April 11, 2026)

These weren't in the original plan and you discovered them during the April 11 literature search:

1. **[Alfarra & AbuSamra 2025](https://doi.org/10.37936/ecti-cit.2025194.263081)** — Pruning + INT8 quantization + hybrid CNN-LSTM on WSN-DS (**4 classes only**, drops TDMA). 0.93 macro-F1. PDF in `alfarra_2025.pdf`. **This is the closest competitor — you MUST explain how your approach differs.**

2. **[Vidhya & Varunadevi 2026](https://doi.org/10.1002/dac.70277)** — Binarized Simplicial CNN on WSN-DS (5 classes). PDF not accessible (Wiley paywall).

3. **[Xiao & Duan 2025](https://doi.org/10.1007/s11760-025-04880-4)** — DNN + CatBoost + metaheuristic optimization + Cosine Amplitude Method (NOT SHAP) on WSN-DS. 95.62% accuracy.

4. **[Birahim et al. 2025](https://doi.org/10.1109/access.2025.3528341)** — PSO + SMOTE-Tomek + ensemble + SHAP + LIME on WSN-DS. 99.73% accuracy. **This invalidates any "first XAI on WSN-DS" claim.**

5. **[Benaddi et al. 2025](https://arxiv.org/abs/2512.19488)** — SHAP + KD + Kronecker networks on **TON_IoT, not WSN-DS**. PDF in `benaddi_2025.pdf`. Closest methodological analogue.

6. **[Talukder et al. 2025](https://www.nature.com/articles/s41598-025-87028-1)** — Current SOTA on WSN-DS: KMS-SMOTE + PCA + RF, 99.94% macro-F1. PDF in `sota_wsn_ds_2025.pdf`.

Full differentiation table: `CuKD_XAI_EXECUTION_PLAN.md` Part 8.

---

## Paper narrative options (depends on CL fix outcome)

**If CL fix works (Config C ≥ 0.91):**
> "CuKD-XAI: Curriculum-Guided Knowledge Distillation with Explainable Feature Analysis for Lightweight WSN Intrusion Detection"

Core claim: CL-guided KD improves compression for tabular IDS. Feature alignment gap is a secondary novel finding.

**If CL fix fails (Config C still < 0.91):**
> "Tree-to-Neural Knowledge Distillation for Full-Class WSN Intrusion Detection: 18,000× Compression with a Feature-Alignment Gap"

Core claim: 18,335× compression of RF → tiny MLP. Novel finding: feature alignment gap (Spearman ρ ≈ 0). CL investigated and reported as a rigorous negative result (consistent with Wu et al. ICLR 2021).

**Both versions are publishable.** The decision gate is the 15-minute Colab re-run.

Full decision tree: `CuKD_XAI_MASTER_DOCUMENT.md` Part 19.

---

## Where everything is

```
/home/ubuntu/nishant_workspace/local/new_feat/wct/
├── RESUME_HERE.md                          ← You are here
├── STATE_SNAPSHOT_2026_04_11.md            ← Portable state dump (from memory)
├── CuKD_XAI_MASTER_DOCUMENT.md             ← Master research doc v3.0
├── CuKD_XAI_EXECUTION_PLAN.md              ← Day-by-day execution plan
├── CUKD_XAI_README.md                      ← Notebook run instructions
├── cukd_xai_colab.ipynb                    ← Colab notebook v2.2 (THE THING TO RUN)
├── cukd_xai_colab.py                       ← Python source (don't regenerate .ipynb from this!)
├── make_notebook.py                        ← ⚠️ DO NOT RUN (wipes Status Banner)
├── alfarra_2025.pdf                        ← Competitor paper (full text)
├── benaddi_2025.pdf                        ← Competitor paper (full text)
├── sota_wsn_ds_2025.pdf                    ← Current WSN-DS SOTA paper
├── base_paper.pdf                          ← Original base paper (Ghadi 2024)
└── results_download/
    ├── cukd_xai_results.json               ← Previous run output (v2.0 code — buggy CL)
    ├── wsnds_results_student_A.csv         ← Aggregate CSV
    ├── per_class_f1.png                    ← Per-class F1 bar chart
    ├── confusion_matrix_F.png              ← Config F confusion matrix
    ├── pareto_frontier.png                 ← Model size vs macro F1
    ├── shap_summary_student.png            ← SHAP feature importance
    └── loss_curves_B_vs_C.png              ← The figure that revealed the CL bug
```

---

## If you're lost, read these 3 documents in order

1. **This file** (2 minutes) — you're here
2. **`cukd_xai_colab.ipynb` Cell 1 (STATUS BANNER)** (3 minutes) — version history + what's fixed
3. **`CuKD_XAI_EXECUTION_PLAN.md` Part 10** (3 minutes) — immediate next action

Total orientation time: **8 minutes**. Then you're ready to run the notebook.

---

## If Claude memory is lost

If you open a new Claude session and it doesn't remember anything about this project, tell it:

> "Read `/home/ubuntu/nishant_workspace/local/new_feat/wct/RESUME_HERE.md` and `STATE_SNAPSHOT_2026_04_11.md` to get up to speed on the CuKD-XAI paper project. We stopped before the CL-fix validation run."

That should be enough context to continue.

---

*Last edit: April 11, 2026 (end of day). Next expected touch: post-May 4 exams. Good luck with exams. 📚*
