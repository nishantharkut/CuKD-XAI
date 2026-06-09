# CuKD-XAI: Complete Execution Plan

**Owner:** Nishant Harkut (2023IMG-040)
**Last updated:** April 11, 2026
**Purpose:** Single source of truth for what's done, what's left, and the step-by-step plan to get the paper submission-ready.

---

## Part 1 — What Has Been Run (from `cukd_xai_results.json`)

Results downloaded from the Colab run shared in Drive folder:
https://drive.google.com/drive/folders/1DfaboxajQaoVNI-DFc7vS-zCC4VNFzRV

| Notebook section | Run status | Evidence |
|---|---|---|
| **Cell 3-4** — Data loading + preprocessing | ✅ Done | `class_names`, `feature_names`, data arrays all populated in results |
| **Cell 5** — Architecture definitions (Teacher, Student A, Student B) | ✅ Done | All models instantiated correctly |
| **Cell 6** — Training utilities | ✅ Done | `train_standard`, `train_with_curriculum`, `train_kd`, `evaluate_model` all worked |
| **Cell 7** — Difficulty scoring (loss-based + domain-based) | ✅ Done | Both Config C and C2 have results |
| **Cell 8** — KD hyperparameter grid search | ⚠️ **UNCLEAR** | `kd_hyperparameters` = `{T: 4, α: 0.7}` — either grid search ran and returned defaults, or grid search was skipped and defaults were used. Cannot distinguish from JSON alone. |
| **Cell 9** — `run_all_configs` function defined | ✅ Done | Function executed correctly 5 times |
| **Cell 10 — Student A multi-seed loop** | ✅ **COMPLETE** | 5 seeds × 10 configs = 50 runs. Seeds: [42, 123, 456, 789, 1001] |
| **Cell 10 — Student B multi-seed loop** | ❌ **NOT RUN** | `wsn_ds_multi_seed_student_B` is empty |
| **Cell 10 — Final re-run to capture Student A models** | ✅ Done | Enabled SHAP/INT8/benchmarks downstream |
| **Cell 11** — Aggregate stats + Wilcoxon tests | ⚠️ **Partially Done** | `aggregate_student_A` has 10 entries (verified). But Wilcoxon test results are **NOT saved to JSON** — only printed to stdout. No `wilcoxon_*` key in `cukd_xai_results.json`. Need to re-run or manually parse from Colab stdout. |
| **Cell 12** — SHAP (DeepExplainer + TreeExplainer + Spearman) | ✅ Done | `shap_results` has 5 keys |
| **Cell 13** — INT8 quantization | ✅ Done **(Config F only)** | `quantization` has 7 keys |
| **Cell 14** — Inference benchmarks | ✅ Done **(3 models)** | `Teacher_MLP`, `Student_scratch`, `Student_KD_F` benchmarked |
| **Cell 15** — Visualizations | ✅ Done | All 5 figures present in Drive folder |
| **Cell 16** — CICIoT2023 generalizability | ❌ **NOT RUN** | `ciciot_results` is empty (flag `RUN_CICIOT = False`) |
| **Cell 17** — Save JSON | ✅ Done | `cukd_xai_results.json` exists |
| **Cell 18** — Final summary print | ✅ Done | Terminal output captured in Drive screenshots |

### Configs run for Student A (all 5 seeds)
All 10 configurations executed successfully:
1. **A_RF_500** — Random Forest (500 trees) baseline
2. **B_Full_MLP** — Full MLP teacher baseline (no CL)
3. **C_CL_MLP_loss** — CL teacher with loss-based difficulty
4. **C2_CL_MLP_domain** — CL teacher with domain-based difficulty
5. **D_Small_MLP** — Small MLP from scratch (no KD)
6. **E_KD_from_RF** — KD from calibrated RF
7. **E2_KD_from_MLP** — KD from standard MLP (no CL)
8. **F_KD_from_CL_MLP** — KD from CL-MLP (CORE CLAIM)
9. **G_KD_random_pacing** — KD with random pacing control
10. **I_KD_from_SMOTE_MLP** — KD from SMOTE-trained teacher

---

## Part 2 — What Is Left to Run (Within the Existing Notebook)

### Must run (notebook has the code, just needs execution)

**2.1 Student B (64-32-5) multi-seed loop**
- **Status:** Code present in Cell 10 but skipped (`all_seed_results_B = {}`)
- **Effort:** ~30-40 minutes on Colab T4
- **Produces:** Second Pareto point, direct Student A vs B comparison
- **Needed because:** Paper currently has zero evidence from a second student size. Reviewer will ask "why 32-16-5 and not something else?"
- **How to run:** Set `QUICK_MODE = False`, ensure the Student B loop in Cell 10 is not commented out, run Cell 10 onwards

**2.2 CICIoT2023 generalization (Cell 16)**
- **Status:** Code present, gated behind `RUN_CICIOT = False`
- **Effort:** ~60-90 minutes on Colab T4 (dataset loading + preprocessing + 4-5 configs)
- **Produces:** Cross-dataset validation of the KD approach
- **Needed because:** Non-negotiable for publication — reviewers will reject "tested on one dataset only"
- **How to run:**
  1. Upload CICIoT2023.csv to Colab (download from https://www.kaggle.com/datasets/himadri07/ciciot2023, then sample to 400K rows)
  2. Set `RUN_CICIOT = True` in Cell 2
  3. Set `CICIOT_PATH = '/content/CICIoT2023.csv'` in Cell 2
  4. Run Cell 16

### Should run (small modifications needed)

**2.3 Extended KD hyperparameter grid search**
- **Status:** Current grid `T ∈ {2,3,4,5}`, `α ∈ {0.5,0.7,0.9}` = 12 combos. Results say best is `T=4, α=0.7` which equals defaults (ambiguous whether search actually ran).
- **Effort:** ~15-20 minutes
- **Produces:** Hyperparameter sensitivity heatmap for the paper
- **Needed because:** Reviewer question "did you tune hyperparameters?" — the paper should have a sensitivity analysis
- **How to run:** In Cell 2, change `KD_T_GRID = [2, 3, 4, 5, 6]` and `KD_ALPHA_GRID = [0.3, 0.5, 0.7, 0.9]`. Save hyperparameter search results to a separate JSON.

**2.4 INT8 quantization sweep (all students, not just Config F)**
- **Status:** Cell 13 only quantizes Config F. Needs to loop over D, E, E2, F, G, I
- **Effort:** ~10 minutes
- **Produces:** Quantization-vs-architecture ablation table
- **Needed because:** The 3% F1 drop on Config F is large and deserves investigation. Is it unique to Config F (because Config F's teacher is broken) or does all-config quantization show the same issue?
- **Modification:** Wrap the existing quantization block in a `for cfg_name in ['D', 'E', 'E2', 'F', 'G', 'I']:` loop

**2.5 Inference benchmarks for ALL students**
- **Status:** Cell 14 only benchmarks Teacher + Student_scratch + Student_KD_F
- **Effort:** ~5 minutes
- **Produces:** Complete latency table for paper
- **Modification:** Add `'Student_KD_E': final_models['E_KD_from_RF']` and similar for other configs to the `candidate_models` list

### Nice to have (small additions to notebook)

**2.6 Per-class SHAP Spearman correlation**
- **Status:** Current `ranking_agreement_spearman = -0.039` is a single global number
- **Effort:** ~10 minutes
- **Produces:** Per-attack teacher-student feature alignment table
- **Why:** Strengthens the "feature alignment gap" novel finding. Does alignment differ between easy (Blackhole) and hard (TDMA) attacks?
- **Code:** After computing `student_shap_list` and `rf_shap_list`, loop over classes and compute Spearman per class

**2.7 SHAP bootstrap stability test**
- **Status:** Not in notebook
- **Effort:** ~15 minutes
- **Produces:** Bootstrap confidence interval on Spearman correlation
- **Why:** Defends the novel "ρ ≈ 0" finding against "that's just noise"
- **Code:**
```python
spearman_bootstrap = []
for bs_seed in range(5):
    rng = np.random.RandomState(bs_seed)
    bg_idx_bs = rng.choice(len(X_train_shap_t), 100, replace=False)
    # ... recompute SHAP with this background ...
    # ... append new rho to spearman_bootstrap ...
print(f"Spearman ρ: {np.mean(spearman_bootstrap):.3f} ± {np.std(spearman_bootstrap):.3f}")
```

**2.8 LIME comparison**
- **Status:** Not in notebook
- **Effort:** ~20 minutes
- **Produces:** SHAP vs LIME feature ranking agreement
- **Why:** Defends against "SHAP is unreliable" reviewer critique (feature alignment gap could be SHAP's fault, not compression's fault)
- **Setup:** `!pip install lime` then `from lime.lime_tabular import LimeTabularExplainer`

---

## Part 3 — Critical Bug to Fix Before Any Additional Runs

### The CL under-training hypothesis (Cell 11 / `train_with_curriculum`)

**Symptom (VERIFIED from JSON):** Config C (CL teacher, loss-based) aggregate macro F1 = 0.8607 ± 0.0025, which is **6.04% WORSE** than Config B (standard MLP) at 0.9211 ± 0.0037. Individual seed 42 shows C=0.8576, B=0.9155, confirming the gap is consistent across seeds.

**Hypothesis (INFERRED from `loss_curves_B_vs_C.png`, not yet tested):**
- Stage 1 (epochs 1-7): trained on only 33% of data (easy samples) — validation F1 stuck low
- Stage 2 (epochs 8-14): trained on 66% of data — still low because hard classes not seen
- Stage 3 (epochs 15-25): full data — validation F1 jumps to ~0.9 but only has 11 epochs
- Meanwhile Config B gets 30 full epochs on all data (~2.7× more exposure to full distribution)
- Net effect: Config C may be under-trained on the full class distribution

**Caveat:** This hypothesis is plausible but NOT the only possible cause. Other possibilities:
- CL may genuinely not help for well-separated tabular features (consistent with Wu et al., ICLR 2021)
- The scheduler (cosine annealing over 25 total epochs) may have decayed the LR before Stage 3 could benefit
- The "best val F1" model selection may have captured a transient peak rather than converged weights

**Proposed fix (NOT yet tested — verify empirically):**

**Fix (in Cell 2):**
```python
# Change from:
CL_STAGES = [(0.33, 7), (0.66, 7), (1.0, 11)]  # total 25 epochs, only 11 on full
# To:
CL_STAGES = [(0.33, 5), (0.66, 5), (1.0, 30)]  # total 40 epochs, 30 on full (matches B)
```

**Additional fix (in Cell 6 / `train_with_curriculum` function):**
The optimizer and cosine scheduler currently persist across stages. By the time Stage 3 begins, the learning rate is already decayed. We should either:
- **Option A:** Reset the optimizer at each stage transition
- **Option B:** Use a per-stage cosine scheduler (each stage gets its own full cosine annealing)
- **Option C:** Disable scheduler entirely and use constant LR

Simplest fix (Option C):
```python
def train_with_curriculum(model, X, y, difficulty_order, X_val, y_val,
                          stages=CL_STAGES, class_weights=None,
                          batch_size=256, lr=1e-3, weight_decay=1e-3, ...):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # REMOVED: scheduler = ...CosineAnnealingLR(optimizer, T_max=total_epochs)
    ...
    # REMOVED: scheduler.step() after epoch
```

**Decision gate:** Run the fixed CL training on Config C alone (1 seed, ~10 min). Check if C's macro F1 now reaches ~0.92 (matching B) or even exceeds it.

- **If C ≥ B:** CL is fixed. Re-run the full experiment. Paper narrative: "CL-guided KD improves IDS compression" (original hypothesis confirmed).
- **If C < B still:** CL fundamentally fails for tabular IDS. Commit to pivot narrative: "Tree-to-Neural KD with feature alignment gap." Report CL as a rigorous negative result.

---

## Part 4 — What the Paper Needs BEYOND the Notebook

These are not code tasks — they're research/writing tasks required for publication.

### 4.1 Literature review (MUST DO before writing)

All of these competitor papers must be cited and properly differentiated:

**Must-cite WSN-DS papers (verified April 11 2026):**
- Almomani 2016 (original dataset)
- Talukder 2025 (SOTA, KMS+PCA+RF, 99.94%)
- Alfarra & AbuSamra 2025 (closest compression competitor — pruning+quantization CNN-LSTM, 4 classes)
- Xiao 2025 (metaheuristic ensemble + sensitivity analysis on WSN-DS, 5 classes)
- Vidhya & Varunadevi 2026 (binarized simplicial CNN on WSN-DS, 5 classes)
- Birahim 2025 (PSO+SMOTE+ensemble+SHAP+LIME on WSN-DS — closest XAI competitor)
- Rana 2024 (Springer ML benchmarking on WSN-DS)
- Akhtar 2024 (MLSTL-WSN, SMOTE-TomekLink)
- Salmi & Oughdir 2022 (CNN-LSTM for DoS in WSN)
- GSWO-CatBoost (Sensors 2024)
- Tabu search RF (Scientific Reports 2025)

**Must-cite KD-IDS papers (none on WSN-DS):**
- Benaddi 2025 (arXiv:2512.19488) — SHAP+KD+Kronecker on TON_IoT (closest methodological analogue)
- AL-Nomasy 2025 (DistillGuard, Computers & Security) — Transformer KD + XAI for IDS
- Yong 2025 (MDPI Computers) — KD for IoT/UAV IDS
- Wang 2024 (Applied Soft Computing) — CNN + self-adaptive KD
- DTKD-IDS 2025 (Ad Hoc Networks) — Dual-teacher KD
- DNN-KDQ 2025 (J. Cloud Computing) — KD + quantization

**Methodology foundations:**
- Bengio 2009 (curriculum learning)
- Hinton 2015 (knowledge distillation)
- Lundberg & Lee 2017 (SHAP)
- Wu et al. ICLR 2021 ("When Do Curricula Work?")

### 4.2 Paper sections to draft

| Section | Depends on experiments? | Estimated pages (IEEE double-column) |
|---|---|---|
| 1. Introduction | No | ~1 |
| 2. Related Work | No | ~1.5 |
| 3. Methodology | No | ~2 |
| 4. Experimental Setup | Yes (needs config + hyperparameter info) | ~1 |
| 5. Results and Discussion | Yes (needs final results) | ~2-3 |
| 6. Conclusion | Yes (depends on final results) | ~0.5 |
| References | No | ~1 |

Sections 1, 2, 3 can be drafted BEFORE any more experiments run.

### 4.3 Target venue selection

Given the current state of results (empirical compression paper, negative results on CL, novel Spearman finding):

**Recommended Tier 1 targets (rolling submission, empirical fit):**
- **IEEE Access** (IF 3.4, 2-3 month review, Scopus+SCI indexed) — best match
- **MDPI Sensors** — Special Issue "Securing the IoT" (~2 month review)
- **Scientific Reports** (IF 3.8) — same journal as Talukder SOTA paper

**Conference fallbacks (deadline-driven):**
- IEEE CONECCT 2026 — April 18 deadline (tight)
- IEEE INDICON 2026
- IEEE TENCON 2026

**Not realistic:** NeurIPS/ICML/ICLR (wrong venue), USENIX Security/IEEE S&P (too competitive for first paper).

---

## Part 5 — Step-by-Step Execution Order

### Phase A — Unblock the narrative (1 day, ~2-3 hours of active work)

**A1. Fix CL training and validate** *(~30 min, DECISION GATE)*
1. Open notebook in Colab
2. In Cell 2, change `CL_STAGES = [(0.33, 5), (0.66, 5), (1.0, 30)]`
3. In Cell 6, remove scheduler from `train_with_curriculum` (or reset at each stage)
4. Set `SEEDS = [42]` and `QUICK_MODE = True` (for quick test)
5. Run Cell 1-11 (single seed, all configs)
6. Check: is Config C's macro F1 now ≥ 0.91?

**A2. Decide paper narrative based on A1 result**
- **If CL works:** Original story "CL+KD for tabular IDS". Paper becomes much stronger.
- **If CL still fails:** Pivot to "Tree-to-Neural KD + feature alignment gap" story. The feature alignment finding (Spearman ρ ≈ 0) becomes the headline novelty.

### Phase B — Complete the missing notebook sections (1-2 days, ~3 hours of active work)

**B1. Re-run full 5-seed experiment with fixed CL** *(~60-90 min)*
- `SEEDS = [42, 123, 456, 789, 1001]`
- `QUICK_MODE = False`
- All 10 configs for Student A
- Saves new `cukd_xai_results.json` with corrected CL data

**B2. Run Student B (64-32-5)** *(~30-40 min)*
- Uncomment Student B loop in Cell 10
- Same 5 seeds
- Populates `wsn_ds_multi_seed_student_B`

**B3. Run CICIoT2023 generalization** *(~60-90 min)*
- Download CICIoT2023 from Kaggle
- Upload to Colab, set `CICIOT_PATH` accordingly
- Set `RUN_CICIOT = True`
- Run Cell 16

### Phase C — Strengthen the novel contribution (1 day, ~2 hours active work)

**C1. Per-class SHAP Spearman** *(~10 min + code edit)*
- Add loop in Cell 12 for per-class correlation
- Result: 5 Spearman values instead of 1 — tests if alignment varies by attack

**C2. SHAP bootstrap stability test** *(~15 min + code)*
- 5 different random backgrounds
- Compute mean ± std of Spearman
- Result: confidence interval on feature alignment finding

**C3. LIME comparison** *(~20 min + code)*
- Install LIME
- Run on same 500 test samples
- Compute top-5 feature agreement between SHAP and LIME
- Result: confidence that the alignment gap isn't a SHAP-specific artifact

**C4. INT8 quantization sweep** *(~15 min + code)*
- Apply `quantize_dynamic_int8()` to all 6 student configs (D, E, E2, F, G, I)
- Build a table: config → fp32 size, int8 size, fp32 F1, int8 F1
- Investigate why the 3% F1 drop on F is worse than expected

**C5. Hyperparameter sensitivity heatmap** *(~20 min + code)*
- Extended grid search: `T ∈ {2,3,4,5,6}`, `α ∈ {0.3,0.5,0.7,0.9}` = 20 combos
- Only need to run this for Config F (core claim) to produce a heatmap figure
- Matplotlib `imshow` → publication figure

### Phase D — Paper writing (4-7 days)

**D1. Update `CuKD_XAI_MASTER_DOCUMENT.md`** *(~1 hour)*
- Incorporate all April 11 findings
- Add Alfarra/Xiao/Vidhya/Birahim to competitor analysis
- Correct all invalidated "first X" claims
- Add the Spearman ρ ≈ 0 finding as a new core contribution

**D2. Draft Sections 1 (Intro), 2 (Related Work), 3 (Methodology)** *(~2-3 days)*
- Use IEEE double-column template (available on Overleaf)
- These don't depend on the remaining experiments — can start now
- Review with professor before moving on

**D3. Draft Sections 4 (Setup), 5 (Results), 6 (Conclusion)** *(~2-3 days)*
- After Phase B and C are complete
- Populate with final numbers + figures from the notebook

**D4. Polish figures for publication** *(~1 day)*
- Current PNG figures are functional but not polished
- Re-generate with larger fonts, higher DPI, consistent color scheme
- Tables should be formatted for IEEE style

**D5. Professor review + revision** *(~2-3 days)*
- Submit draft to professor
- Incorporate feedback

**D6. Final proofread + submit** *(~1 day)*
- Grammar, typos, reference consistency
- Choose final venue
- Submit

---

## Part 6 — Decision Tree for Paper Narrative

```
Start: Fix CL bug and re-run Config C (~10 min)
  │
  ├─ Config C macro F1 ≥ 0.91? 
  │   │
  │   ├─ YES: Original story works
  │   │   │
  │   │   ├─ Run full 5-seed re-experiment
  │   │   ├─ Check if F now beats E2 (Wilcoxon test)
  │   │   │
  │   │   ├─ F > E2 significantly?
  │   │   │   ├─ YES: 🎉 "CL-guided KD for WSN IDS" paper
  │   │   │   │    Core claim: CL improves KD for tabular IDS
  │   │   │   │    Title: "CuKD-XAI: Curriculum-Guided Knowledge Distillation 
  │   │   │   │            for Lightweight WSN Intrusion Detection"
  │   │   │   │
  │   │   │   └─ NO: CL helps teacher but doesn't cascade
  │   │   │         Paper: "KD compression for WSN IDS" with CL as ablation
  │   │   │         Still publishable with feature alignment as main novelty
  │   │   │
  │   └─ NO: CL still fails for tabular
  │       Commit to pivot: "Tree-to-Neural KD + Feature Alignment Gap"
  │       Report CL as rigorous negative result
  │       Core novelty: Spearman ρ ≈ 0 finding
  │       Title: "Tree-to-Neural Knowledge Distillation for Full-Class WSN 
  │               Intrusion Detection: 18,000× Compression with a 
  │               Feature-Alignment Gap"
```

---

## Part 7 — Current Novelty Claims (Post-Verification)

After the April 11 literature search, these are our defensible claims:

✅ **SAFE:**
- First knowledge distillation applied to WSN-DS
- First teacher-student distillation for WSN security
- First tree-to-neural (RF → MLP) KD for WSN IDS
- First curriculum-learning ablation on tabular WSN IDS
- First KD benchmark on the full 5-class WSN-DS (Alfarra uses only 4 classes)
- First systematic feature-alignment analysis of compressed IDS (Spearman ρ ≈ 0 finding)
- First SHAP analysis of a KD-compressed WSN-DS model

⚠️ **NEEDS CAREFUL PHRASING:**
- "Smallest reported KD-compressed model for WSN-DS" (true so far — 1.16 KB int8, no other KD on WSN-DS exists)
- "Deepest compression ratio reported on WSN-DS at time of submission" — need to verify against Vidhya 2026 binarized CNN exact size

❌ **CANNOT CLAIM (invalidated by literature search):**
- ~~First XAI/SHAP on WSN-DS~~ — Birahim 2025 has SHAP+LIME
- ~~First compression on WSN-DS~~ — Alfarra 2025 (pruning+quantization), Vidhya 2026 (binarization)
- ~~First lightweight WSN-DS IDS~~ — many prior lightweight efforts
- ~~First explainable + resource-aware WSN IDS~~ — Xiao 2025 claims this with metaheuristic ensemble + sensitivity analysis

---

## Part 8 — Must-Cite Competitors (Verified April 11, 2026)

Each entry lists the verification method used: **PDF** = full PDF read directly, **CrossRef** = CrossRef API metadata, **OpenAlex** = OpenAlex API metadata, **abstract** = publisher-page abstract extracted via curl, **agent** = literature-search agent report (less reliable).

### Direct compression competitors on WSN-DS

**[C1] Alfarra & AbuSamra 2025** — "Energy-Efficient Hybrid Learning for Secure Wireless Sensor Networks"
- Venue: ECTI Transactions on Computer and Information Technology, Vol. 19, No. 4, October 2025
- DOI: [10.37936/ecti-cit.2025194.263081](https://doi.org/10.37936/ecti-cit.2025194.263081)
- URL: https://ph01.tci-thaijo.org/index.php/ecticit/article/view/263081
- Code: https://github.com/abalfattah-alfarra/WSNIDS
- PDF on disk: `alfarra_2025.pdf`
- **Verification level:** PDF read in full (pages 1-10) + GitHub code inspected
- **Method:** 2-stage hybrid IDS. Stage 1 (sensor MCU): integer-only rule filter (Shannon entropy + SoC rule + timing rule), ≤0.05 mJ/packet. Stage 2 (edge gateway): CNN-LSTM (Conv2D(16,3×3) → Conv2D(32,3×3) → LSTM(64, return_sequences) → LSTM(64) → Dense(4, softmax)) with 50% structured pruning + INT8 TFLite quantization
- **Dataset:** WSN-DS, augmented with NS-3 simulations (15K packets/attack added to balance minority classes), 70/15/15 stratified split
- **Classes:** Explicitly 4 classes (Normal, Flood, Black, Gray). Paper states "The tiny TDMA class (only two windows after windowing) is effectively ignored; this is acceptable because TDMA spoofing is treated as an open-set event in the threat model."
- **Results:** 98% accuracy, 0.93 macro-F1 (over 4 classes), Blackhole 0.95 / Grayhole 0.90 / Flooding 0.84 recalls, 69-day T50 network lifetime (+82% vs GRU-on-node baseline)
- **Statistical rigor:** 10 random seeds, paired Wilcoxon (α=0.05), 95% CI bootstrap (1K resamples)
- **Model size (estimated from architecture, NOT directly measured):** base ~150K params → ~75K after 50% structured pruning → ~75-100 KB TFLite file. Exact final size not reported in the paper.
- **No KD, no SHAP, no curriculum learning.** Uses pruning + quantization + rule-based filtering.

**[C2] Vidhya & Varunadevi 2026** — "Efficient Detection of Denial-of-Service Attacks in Wireless Sensor Networks Depending on Binarized Simplicial Convolutional Neural Networks for Enhanced Security"
- Venue: International Journal of Communication Systems (Wiley), Issue 1, print date 2026-01-10 (online first 2025-12-01). Volume number not confirmed via CrossRef.
- DOI: [10.1002/dac.70277](https://doi.org/10.1002/dac.70277)
- **Verification level:** CrossRef + OpenAlex metadata (full PDF not accessible — Wiley Cloudflare block; abstract extracted)
- **Authors:** R. Vidhya (first, Hindusthan College of Engineering and Technology, Coimbatore, India), S. Varunadevi
- **Method (ED-DoS-WSN-BSCNN pipeline, from abstract):**
  - Preprocessing: Adaptive two-stage unscented Kalman filter (ATSUKF)
  - Classifier: Binarized Simplicial Convolutional Neural Network (BSCNN) — 1-bit weight quantization
  - Optimizer: Arctic tern optimizer (ATO) for hyperparameter tuning
- **Dataset:** WSN-DS, **all 5 classes** (normal, blackhole, grayhole, flooded, TDMA)
- **Baselines compared:** CNN-DoS-WSN, KNN-DoS-WSN, RT-DoS-WSN
- **Results reported in abstract:** 4.05% / 7.52% / 2.91% higher accuracy vs those three baselines respectively (no absolute number given in abstract)
- **Exact model size NOT verifiable without full PDF.** Binarized weights at 1 bit per weight are theoretically ~8× smaller than INT8 for the same architecture, but the BSCNN architecture size is unknown.
- **No KD, no SHAP, no curriculum learning.**

### Direct XAI competitors on WSN-DS

**[C3] Birahim et al. 2025** — "Intrusion Detection for Wireless Sensor Network Using Particle Swarm Optimization Based Explainable Ensemble Machine Learning Approach"
- Venue: IEEE Access, 2025
- DOI: [10.1109/ACCESS.2025.3528341](https://doi.org/10.1109/ACCESS.2025.3528341)
- IEEE Xplore: https://ieeexplore.ieee.org/document/10836702/
- **Verification level:** OpenAlex metadata + search-agent abstract (PDF not read directly)
- **Method (from agent report):** PSO + SMOTE-Tomek Link + ensemble (Random Forest + Decision Tree + KNN) + SHAP + LIME
- **Dataset:** WSN-DS, 5 classes
- **Accuracy reported:** 99.73%
- **No compression, no KD.** This is the closest direct competitor on the "XAI for WSN-DS" axis.

**[C4] Xiao & Duan 2025** — "Metaheuristically optimized deep soft-voting ensemble for explainable and resource-aware signal processing in wireless sensor network intrusion detection"
- Venue: Signal, Image and Video Processing (Springer London), Vol. 19, Issue 15, December 2025
- DOI: [10.1007/s11760-025-04880-4](https://doi.org/10.1007/s11760-025-04880-4)
- **Verification level:** CrossRef metadata + Springer page abstract extracted
- **Authors (verified from CrossRef):** Xiongzhi Xiao, Wenzhou Duan (The Affiliated Guangdong Second Provincial General Hospital of Jinan University, Information Department, Guangzhou, China)
- **Method:** DNN + CatBoost soft-voting ensemble with Quadratic Interpolation Optimization (QIO) + Osprey Optimization Algorithm (OOA) for hyperparameter tuning
- **Dataset:** WSN-DS, 23 features, 5 classes (Blackhole, Grayhole, Flooding, TDMA Scheduling, Normal)
- **Accuracy reported:** 95.62% (DCQI — QIO-enabled variant)
- **XAI method:** Cosine Amplitude Method (CAM) for feature sensitivity analysis — **NOT SHAP, NOT LIME**
- **Top features identified:** Is_CH, Who_CH, Dist_to_CH (similar to our SHAP top-3 on Config F: Is_CH, Time, dist_CH_To_BS)
- **No KD, no pruning, no quantization.** Claims "explainable and resource-aware" but has no real compression — just metaheuristic hyperparameter tuning.

### Closest methodological analogue (different dataset)

**[C5] Benaddi, Jouhari, Laamech, Motii, Ibrahimi 2025** — "Lightweight Intrusion Detection in IoT via SHAP-Guided Feature Pruning and Knowledge-Distilled Kronecker Networks"
- Venue: Proceedings of the 2025 8th International Conference on Advanced Communication Technologies and Networking (CommNet 2025) / IEEE
- arXiv: [arXiv:2512.19488](https://arxiv.org/abs/2512.19488)
- Code: https://github.com/mohammed-jouhari/iot-ids-lightweight
- PDF on disk: `benaddi_2025.pdf`
- **Verification level:** PDF read in full (6 pages)
- **Method:** SHAP-guided feature pruning + knowledge distillation from MLP teacher (512-256-128, 769,922 params per Table II) to Kronecker-factored student (3,042 params per Table II). Loss: L = (1−α) L_CE + α T² KL(softmax(z_t/T) ∥ softmax(z_s/T)). Grid search: T ∈ {2,3,4}, α ∈ {0.5, 0.7, 0.9}.
- **Dataset:** TON_IoT (21.98M flow records, 42 features → 1,624 after one-hot encoding, 9 attack types). **NOT WSN-DS.**
- **Results:** Teacher accuracy 0.9989 / macro-F1 0.9955. Student fp32 accuracy 0.9968 / macro-F1 0.9863. Student INT8 accuracy 0.9969 / macro-F1 0.9867.
- **Compression:** ~253× parameter reduction (769,922 → 3,042). Teacher model 3021.53 KB → student 22.29 KB fp32, 22.44 KB INT8.
- **No curriculum learning.** This is the closest methodological analogue (SHAP + KD, small student), but on a different dataset and with Kronecker student rather than plain MLP.

### SOTA baseline on WSN-DS

**[C6] Talukder, Khalid, Sultana 2025** — "A hybrid machine learning model for intrusion detection in wireless sensor networks leveraging data balancing and dimensionality reduction"
- Venue: Scientific Reports (Nature), 2025
- DOI: [10.1038/s41598-025-87028-1](https://doi.org/10.1038/s41598-025-87028-1)
- URL: https://www.nature.com/articles/s41598-025-87028-1
- PDF on disk: `sota_wsn_ds_2025.pdf`
- **Verification level:** PDF read in full (pages 1-17)
- **Method:** StandardScaler + Label Encoding → KMeans-SMOTE (KMS) for data balancing → PCA with Kaiser criterion (retains 5 components) → Random Forest Classifier (500 trees). Also tested: XGBoost, LightGBM, HistGradientBoosting, DecisionTree, CatBoost.
- **Dataset:** WSN-DS + TON-IoT (both evaluated), all 5 classes on WSN-DS
- **Evaluation:** 5-fold cross-validation (NOT 80/20 split)
- **Results on WSN-DS (KMS+PCA+RFC):** Accuracy 99.94%, F1 99.94%, ROC-AUC 99.99%, Precision 99.94%, Sensitivity 99.94%
- **Per-class F1 (Table 10):** Normal 99.87, Grayhole 99.97, Blackhole 99.99, TDMA 99.93, Flooding 99.96
- **Model size:** NOT explicitly reported. Full RF with 500 trees serialized via joblib is ~5-50 MB.
- **No compression, no KD, no SHAP.** Pure optimization of the ML pipeline itself. This is the current macro-F1 SOTA on WSN-DS.

---

## Part 9 — File Inventory

### On local disk (`/home/ubuntu/nishant_workspace/local/new_feat/wct/`)

**Code:**
- `cukd_xai_colab.py` — Python source (1,646 lines)
- `cukd_xai_colab.ipynb` — Jupyter notebook (18 code cells)
- `make_notebook.py` — Script to regenerate .ipynb from .py

**Papers (full PDFs read):**
- `alfarra_2025.pdf` — Alfarra & AbuSamra 2025 (closest compression competitor)
- `benaddi_2025.pdf` — Benaddi 2025 (closest KD+SHAP competitor)
- `sota_wsn_ds_2025.pdf` — Talukder 2025 (SOTA baseline)
- `base_paper.pdf` — Ghadi 2024 (original base paper)

**Results:**
- `results_download/cukd_xai_results.json` — Full run results
- `results_download/wsnds_results_student_A.csv` — Aggregate CSV
- `results_download/per_class_f1.png` — Per-class F1 comparison
- `results_download/confusion_matrix_F.png` — Config F confusion matrix
- `results_download/pareto_frontier.png` — Accuracy vs size
- `results_download/shap_summary_student.png` — SHAP feature importance
- `results_download/loss_curves_B_vs_C.png` — CL vs no-CL convergence
- 4 screenshot PNGs from Colab terminal output

**Documentation:**
- `CuKD_XAI_MASTER_DOCUMENT.md` — Master research doc (needs April 11 update)
- `CuKD_XAI_EXECUTION_PLAN.md` — This file
- `CUKD_XAI_README.md` — Notebook run instructions
- `CuKD_XAI_COMPLETE_DESIGN.md` — Original Phase 2 design doc
- `CuKD_XAI_FINAL_HONEST_ASSESSMENT.md` — Phase 2 risk assessment

**Memory files (in `~/.claude/projects/.../memory/`):**
- `project_research_paper.md` — Main project memory (needs update)
- `project_apr11_results_and_competitors.md` — Fresh April 11 findings
- `reference_key_papers.md` — Paper references
- `user_profile.md`
- `feedback_research_style.md`

---

## Part 10 — Immediate Next Single Action

**Do this right now:**

Open `cukd_xai_colab.ipynb` in Google Colab. In Cell 2:

```python
# Change this line:
CL_STAGES = [(0.33, 7), (0.66, 7), (1.0, 11)]

# To this:
CL_STAGES = [(0.33, 5), (0.66, 5), (1.0, 30)]
```

In Cell 6, find `train_with_curriculum` and remove the scheduler lines:

```python
# Delete or comment out:
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
# scheduler.step()
```

In Cell 2 also set:
```python
SEEDS = [42]          # single seed for quick test
QUICK_MODE = True     # skip grid search + Student B + CICIoT
```

Run all cells. Takes ~15-20 min. Check the printed `Config C Macro F1` number.

**Then send me:**
- The macro F1 of Config C (CL teacher)
- The macro F1 of Config F (KD from CL teacher)
- The macro F1 of Config B (reference baseline)
- Whether `loss_curves_B_vs_C.png` now shows different behavior

Based on those numbers, I'll tell you which of the two paper narratives we commit to, and then we execute Phases B → C → D in order.

---

---

## Part 11 — Complete Reference List (with verification status)

Each reference indicates how it was verified: **[PDF]** = full text read, **[CrossRef/OpenAlex]** = metadata API, **[abstract]** = publisher-page abstract only, **[agent]** = from literature-search agent (lowest reliability).

### Primary sources (directly read in full)

1. **[PDF]** Almomani, I., Al-Kasasbeh, B., & Al-Akhras, M. (2016). "WSN-DS: A Dataset for Intrusion Detection Systems in Wireless Sensor Networks." *Journal of Sensors*, 2016, Article 4731953. DOI: [10.1155/2016/4731953](https://doi.org/10.1155/2016/4731953)

2. **[PDF]** Ghadi, Y. Y., et al. (2024). "Machine Learning Solutions for the Security of Wireless Sensor Networks: A Review." *IEEE Access*, 12. DOI: [10.1109/ACCESS.2024.3355312](https://doi.org/10.1109/ACCESS.2024.3355312)
   - Our base paper. Identifies lightweight ML as open issue.

3. **[PDF]** Talukder, M. A., Khalid, M., & Sultana, N. (2025). "A hybrid machine learning model for intrusion detection in wireless sensor networks leveraging data balancing and dimensionality reduction." *Scientific Reports*, 15(1), 4617. DOI: [10.1038/s41598-025-87028-1](https://doi.org/10.1038/s41598-025-87028-1)
   - Current macro-F1 SOTA on WSN-DS: 99.94%. PDF on disk: `sota_wsn_ds_2025.pdf`.

4. **[PDF]** Benaddi, H., Jouhari, M., Laamech, N., Motii, A., & Ibrahimi, K. (2025). "Lightweight Intrusion Detection in IoT via SHAP-Guided Feature Pruning and Knowledge-Distilled Kronecker Networks." In *Proc. CommNet 2025*. arXiv: [arXiv:2512.19488](https://arxiv.org/abs/2512.19488). Code: https://github.com/mohammed-jouhari/iot-ids-lightweight
   - Closest methodological analogue (SHAP + KD, tiny student). On TON_IoT, not WSN-DS. PDF on disk: `benaddi_2025.pdf`.

5. **[PDF]** Alfarra, A. M., & AbuSamra, A. A. (2025). "Energy-Efficient Hybrid Learning for Secure Wireless Sensor Networks." *ECTI Transactions on Computer and Information Technology*, 19(4). DOI: [10.37936/ecti-cit.2025194.263081](https://doi.org/10.37936/ecti-cit.2025194.263081). Code: https://github.com/abalfattah-alfarra/WSNIDS
   - Closest compression competitor on WSN-DS. 4-class classifier (treats TDMA as open-set). PDF on disk: `alfarra_2025.pdf`.

### Competitor papers on WSN-DS (verified via metadata API)

6. **[CrossRef+OpenAlex]** Vidhya, R., & Varunadevi, S. (2026). "Efficient Detection of Denial-of-Service Attacks in Wireless Sensor Networks Depending on Binarized Simplicial Convolutional Neural Networks for Enhanced Security." *International Journal of Communication Systems* (Wiley), Issue 1, online 2025-12-01, print 2026-01-10. DOI: [10.1002/dac.70277](https://doi.org/10.1002/dac.70277)
   - Binarized Simplicial CNN + Arctic tern optimizer on WSN-DS (5 classes). Closed access, full PDF not read.

7. **[CrossRef+abstract]** Xiao, X., & Duan, W. (2025). "Metaheuristically optimized deep soft-voting ensemble for explainable and resource-aware signal processing in wireless sensor network intrusion detection." *Signal, Image and Video Processing*, 19(15). DOI: [10.1007/s11760-025-04880-4](https://doi.org/10.1007/s11760-025-04880-4)
   - DNN + CatBoost + QIO + OOA. Uses Cosine Amplitude Method (not SHAP). 95.62% accuracy. Verified author list via CrossRef.

8. **[OpenAlex+agent]** Birahim, S. A., Paul, A., Rahman, F., Islam, Y., Roy, T., & Hasan, Md. R. (2025). "Intrusion Detection for Wireless Sensor Network Using Particle Swarm Optimization Based Explainable Ensemble Machine Learning Approach." *IEEE Access*. DOI: [10.1109/ACCESS.2025.3528341](https://doi.org/10.1109/ACCESS.2025.3528341). IEEE Xplore: https://ieeexplore.ieee.org/document/10836702/
   - PSO + SMOTE-Tomek + ensemble + SHAP + LIME. 99.73% accuracy. Closest XAI competitor on WSN-DS. **Not directly read — verify before citing.**

9. **[abstract]** Rana, A., Prajapat, S., Kumar, P., & Kumar, K. (2024). "Performance Evaluation of Machine Learning Models for Intrusion Detection in Wireless Sensor Networks: A Case Study Using the WSN DS Dataset." In *Machine Intelligence for Research and Innovations*, Springer. DOI: [10.1007/978-981-99-8129-8_15](https://doi.org/10.1007/978-981-99-8129-8_15)
   - Standard ML benchmarking paper on WSN-DS. Verified title and year (2024) from Springer page metadata.

10. **[agent]** Akhtar, M. S., Islam, H. M. A., Alam, M. M., Ahmed, M. S., & Ahmed, M. M. (2024). "MLSTL-WSN: Machine learning-based intrusion detection using SMOTETomek in WSNs." *International Journal of Information Security*. DOI: [10.1007/s10207-024-00833-z](https://doi.org/10.1007/s10207-024-00833-z)
   - MLP + SMOTE-Tomek Link on WSN-DS. 98.80% (multiclass) / 99.37% (binary). Verify author list before citing.

11. **[agent]** Pandey (or Ali et al., verify) (2025). "Enhancing intrusion detection in wireless sensor networks using a Tabu search based optimized random forest." *Scientific Reports*. Available at PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC12119870/. DOI: [10.1038/s41598-025-03498-3](https://doi.org/10.1038/s41598-025-03498-3)
   - Tabu Search RF baseline on WSN-DS. 99.67% accuracy. Author list needs verification.

12. **[agent]** Hameed, S. A., et al. (2024). "Enhancing Intrusion Detection in Wireless Sensor Networks Using a GSWO-CatBoost Approach." *Sensors (MDPI)*, 24(11), 3339. DOI: [10.3390/s24113339](https://doi.org/10.3390/s24113339). URL: https://www.mdpi.com/1424-8220/24/11/3339
   - GSWO-optimized CatBoost, 99.65% accuracy.

13. **[agent]** Salmi, S., & Oughdir, L. (2022). "CNN-LSTM Based Approach for DoS Attacks Detection in Wireless Sensor Networks." *International Journal of Advanced Computer Science and Applications (IJACSA)*, 13(4).
   - CNN-LSTM on WSN-DS, 97% accuracy. One of the earlier deep-learning benchmarks.

### KD/XAI for IDS on other datasets (must cite as related work)

14. **[agent]** AL-Nomasy, N., Alamri, Y., Aljuhani, A., & Kumar, R. (2025). "Transformer-based knowledge distillation for explainable intrusion detection system (DistillGuard)." *Computers & Security*, 154. DOI via URL: https://www.sciencedirect.com/science/article/pii/S0167404825001063
   - Transformer teacher + Selective Gradient-Based KD + XAI. Not WSN-DS. Verify before citing.

15. **[agent]** Yong / Ogbodo, E. U., et al. (2025). "A Lightweight Intrusion Detection System for IoT and UAV Using Deep Neural Networks with Knowledge Distillation." *Computers (MDPI)*, 14(7), 291. URL: https://www.mdpi.com/2073-431X/14/7/291
   - KD for IoT/UAV. NSL-KDD, UNSW-NB15, CIC-IDS2017, IoTID20. Not WSN-DS.

16. **[agent]** Wang, Y., Dai, X., & Chen, J. (2024). "Lightweight intrusion detection model based on CNN and knowledge distillation." *Applied Soft Computing*, 165. DOI via URL: https://www.sciencedirect.com/science/article/abs/pii/S1568494624008925
   - 8-layer CNN teacher → 1-layer student with self-adaptive temperature KD.

17. **[agent]** Almabdy, S., & Ullah, A. (2025). "Optimising Intrusion Detection Systems in Cloud-Edge Continuum with Knowledge Distillation for Privacy-Preserving and Efficient Communication." arXiv: [arXiv:2504.10698](https://arxiv.org/abs/2504.10698)
   - Federated + KD on Edge-IIoTset.

18. **[agent]** Hossain, M. A., Saif, S., & Md, S. I. (2025). "A novel federated learning approach for IoT botnet intrusion detection using SHAP-based knowledge distillation." *Complex & Intelligent Systems*. DOI: [10.1007/s40747-025-02001-9](https://doi.org/10.1007/s40747-025-02001-9)
   - Federated + SHAP + KD for IoT botnet. N-BaIoT dataset.

19. **[agent]** Xie, B., et al. (2025). "DTKD-IDS: Dual-Teacher Knowledge Distillation for Intrusion Detection." *Ad Hoc Networks*. DOI via URL: https://www.sciencedirect.com/science/article/abs/pii/S1570870525001179
   - Dual-teacher KD for NSL-KDD, X-IIoTID.

20. **[agent]** Peng, H., Wu, C., & Xiao, Y. (2025). "FD-IDS: Federated Learning with Knowledge Distillation for Intrusion Detection in Non-IID IoT Environments." *Sensors (MDPI)*, 25(14), 4309. URL: https://www.mdpi.com/1424-8220/25/14/4309
   - Federated + KD for Edge-IIoT, N-BaIoT.

### Methodology foundations (well-known, cite from memory)

21. Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). "Curriculum learning." *ICML 2009*. DOI: [10.1145/1553374.1553380](https://doi.org/10.1145/1553374.1553380)

22. Hinton, G., Vinyals, O., & Dean, J. (2015). "Distilling the Knowledge in a Neural Network." arXiv: [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)

23. Lundberg, S. M., & Lee, S.-I. (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS 2017*. arXiv: [arXiv:1705.07874](https://arxiv.org/abs/1705.07874)

24. Wu, X., Dyer, E., & Neyshabur, B. (2021). "When Do Curricula Work?" *ICLR 2021*. OpenReview: [tW4QEInpni](https://openreview.net/forum?id=tW4QEInpni)
   - Key citation for CL negative results. Our findings align with theirs.

25. Stanton, S., Izmailov, P., Kirichenko, P., Alemi, A. A., & Wilson, A. G. (2021). "Does Knowledge Distillation Really Work?" *NeurIPS 2021*. arXiv: [arXiv:2106.05945](https://arxiv.org/abs/2106.05945)
   - Important caveat: KD doesn't always reproduce teacher behavior — supports our Spearman ≈ 0 finding.

### Curriculum learning related (for CL negative result context)

26. **[agent]** Narkedimilli, S., Raghavaraju, R., Uppalapu, M., & Mahabusarakam, N. (2025). "Enhancing IoT Network Security through Adaptive Curriculum Learning and XAI." arXiv: [arXiv:2501.11618](https://arxiv.org/abs/2501.11618)
   - Only prior CL-for-IDS paper. On Edge-IIoT, CIC-IoV, CIC-APT-IIoT — not WSN-DS. Uses LIME (not SHAP) and no KD.

27. **[agent]** Liu, Y., et al. (2025). "CLIMB: Class-Imbalanced Learning Benchmark on Tabular Data." arXiv: [arXiv:2505.17451](https://arxiv.org/abs/2505.17451)
   - Benchmark for imbalanced tabular learning. CL not included in their 29-method comparison — supports "CL is not standard for tabular imbalance."

### Datasets cited

28. WSN-DS on Kaggle: https://www.kaggle.com/datasets/bassamkasasbeh1/wsnds
29. CICIoT2023 on Kaggle: https://www.kaggle.com/datasets/himadri07/ciciot2023
30. TON_IoT: https://research.unsw.edu.au/projects/toniot-datasets

---

## Part 12 — Verification Audit Log

This section enumerates every factual claim in this document that has been independently verified, along with the method of verification. Use this to flag any unverified assertions before citing in the paper.

| Claim | Verified? | Method |
|---|---|---|
| `cukd_xai_colab.py` has 1,646 lines | ✅ | `wc -l` |
| `cukd_xai_colab.ipynb` has 18 code cells + 19 markdown cells | ✅ | JSON parse of notebook |
| Student A multi-seed has 5 seeds [42, 123, 456, 789, 1001] | ✅ | JSON key inspection |
| Student A has all 10 configs per seed | ✅ | JSON key inspection (A, B, C, C2, D, E, E2, F, G, I) |
| Student B multi-seed is empty | ✅ | `len(wsn_ds_multi_seed_student_B) == 0` |
| CICIoT results are empty | ✅ | `len(ciciot_results) == 0` |
| Wilcoxon tests NOT saved to JSON | ✅ | No `wilcoxon` key in JSON top-level |
| Config C aggregate F1 = 0.8607 ± 0.0025 | ✅ | Computed from seed results in JSON |
| Config B aggregate F1 = 0.9211 ± 0.0037 | ✅ | Computed from seed results |
| Config E aggregate F1 = 0.9219 ± 0.0137 | ✅ | Computed from seed results |
| Config F aggregate F1 = 0.8908 ± 0.0034 | ✅ | Computed from seed results |
| SHAP Spearman ρ = −0.039, p = 0.881 | ✅ | `shap_results` key in JSON |
| fp32 → INT8 F1 drop = 0.8945 → 0.8656 (3%) on Config F | ✅ | `quantization` key in JSON |
| INT8 size reduction = 18.77% (7.68 → 6.24 KB on disk) | ✅ | `quantization` key in JSON |
| Teacher MLP has 69,893 params (excluding BN) or 70,917 (with BN) | ⚠️ Estimated | Calculated from architecture, not measured. BN params contribute 1,024 additional params not counted in `count_params`. |
| Student A has 1,189 params | ✅ | Calculated + matches JSON `params` field |
| 1,189 params × 1 byte = 1.16 KB INT8 | ✅ | Arithmetic, theoretical only (on-disk is 6.24 KB with metadata) |
| Alfarra 2025 uses only 4 classes (treats TDMA as open-set) | ✅ | PDF read directly, Section 3.3 Table 1 + narrative |
| Alfarra 2025 macro-F1 = 0.93 over 4 classes | ✅ | PDF Table 7 |
| Alfarra 2025 model size ~75-100 KB | ⚠️ Estimated | Back-of-envelope from architecture; not reported in paper |
| Alfarra 2025 uses 10 seeds + Wilcoxon + 95% CI bootstrap | ✅ | PDF Section 3.6 and 4.6 |
| Vidhya 2026 uses all 5 WSN-DS classes (including TDMA) | ✅ | CrossRef abstract explicitly lists: "normal, blackhole, grayhole, flooded, and TDMA" |
| Vidhya 2026 model size | ❌ NOT VERIFIED | Closed access, no PDF |
| Xiao 2025 has 2 authors: Xiongzhi Xiao, Wenzhou Duan | ✅ | CrossRef author list |
| Xiao 2025 Volume 19, Issue 15, Dec 2025 | ✅ | CrossRef metadata |
| Xiao 2025 accuracy = 95.62% | ✅ | Springer page abstract |
| Xiao 2025 XAI uses Cosine Amplitude Method, not SHAP | ✅ | Abstract explicitly states "Cosine Amplitude Method" |
| Birahim 2025 DOI 10.1109/access.2025.3528341 | ✅ | OpenAlex search result |
| Birahim 2025 uses SHAP + LIME on WSN-DS | ⚠️ Agent-reported | Title confirmed via OpenAlex; method details from literature search agent only — not directly verified from paper |
| Benaddi 2025 student has 3,042 params per Table II (or 1,282 per text — paper inconsistency) | ✅ | PDF read, flagged internal inconsistency |
| Benaddi 2025 macro-F1 = 0.9863 (fp32) / 0.9867 (int8) | ✅ | PDF Table II |
| "First KD on WSN-DS" | ⚠️ Best-effort | Comprehensive search found no prior work; cannot prove absence definitively. Recommend "to the best of our knowledge" phrasing. |
| "First CL on WSN-DS" | ⚠️ Best-effort | Same caveat. Only prior CL-for-IDS work is Narkedimilli 2025 on IoT (not WSN-DS). |
| "Spearman ρ ≈ 0 is novel" | ⚠️ Unverified | Cannot confirm no prior work on feature alignment in KD-compressed IDS without exhaustive search. Recommend "to the best of our knowledge, the first reported measurement". |

---

*End of execution plan. Last updated April 11, 2026.*
*Every competitor citation above has a verification-method tag: **[PDF]** (full-text read), **[CrossRef]/[OpenAlex]** (metadata API), **[abstract]** (publisher abstract extract), **[agent]** (literature-search-agent report — lowest reliability).*
*Before using any **[agent]**-tagged citation in the paper, fetch and read the actual paper first.*

