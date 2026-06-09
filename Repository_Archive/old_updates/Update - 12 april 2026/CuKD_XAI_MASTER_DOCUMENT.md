# CuKD-XAI: MASTER DOCUMENT FOR PUBLICATION

**Paper Title (v1, may change):** "CuKD-XAI: Curriculum-Guided Knowledge Distillation with Explainable Feature Analysis for Lightweight Intrusion Detection in Wireless Sensor Networks"

**Paper Title (v2 pivot candidate, if CL fix fails):** "Tree-to-Neural Knowledge Distillation for Full-Class WSN Intrusion Detection: 18,000× Compression with a Feature-Alignment Gap"

**Author:** Nishant Harkut, ABV-IIITM Gwalior (2023IMG-040)
**Supervisor:** Professor (IEEE Editor)
**Document Version:** v3.0
**Last Updated:** April 11, 2026
**Status:** Phase 3 (Implementation) PARTIALLY DONE — Student A complete, Student B + CICIoT2023 pending. CL core claim failed as originally formulated; awaiting CL-fix re-run decision.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| **v3.0** | **2026-04-11** | **Major update after first full Colab run + thorough literature search.** Added verified experimental results (Student A 5-seed macro F1). Added 4 new verified competitor papers (Alfarra 2025, Vidhya 2026, Xiao 2025, Birahim 2025). Invalidated 3 "first X" claims (first SHAP, first compression, first lightweight). Added novel Spearman ρ ≈ 0 finding as alternative headline contribution. Diagnosed CL under-training hypothesis. This is now the authoritative source — older sections below (Parts 2-16) retain original text from v2.0 and should be cross-referenced against Part 18 (Correction Log) for invalidated claims. |
| v2.0 | 2026-04-07 | Consolidated research, all 16 parts written. Based on pre-experiment design. All "first X" claims assumed valid at time of writing. |
| v1.0 | 2026-03-25 | Initial design document: approach, experiments, risks. |

## CRITICAL — Read This Before Trusting Anything Below

**Parts 2-16 of this document were written on April 7, 2026 BEFORE any experiments were run.** Many claims (including the "first X" novelty claims and the predicted curriculum order) have been either invalidated or modified by the April 11 experimental run and literature search.

**Read Parts 17-19 FIRST:**
- **Part 17:** April 11 experimental results (what we actually measured)
- **Part 18:** Correction log (what claims from Parts 2-16 are now invalidated)
- **Part 19:** Revised paper narrative and next-action decision tree

Treat Parts 2-16 as the **historical design document**. For current state of facts, use the Execution Plan (`CuKD_XAI_EXECUTION_PLAN.md`) and Parts 17-19 of this document.

---

# PART 1: PROBLEM STATEMENT & MOTIVATION

## 1.1 The Problem

Wireless Sensor Networks (WSNs) use tiny sensor nodes (e.g., TelosB mote: MSP430 CPU at 8MHz, 10KB RAM, 48KB flash) running the LEACH routing protocol. These nodes are vulnerable to four specific attacks: Blackhole, Grayhole, Flooding, and Scheduling (TDMA manipulation).

Current SOTA intrusion detection on WSN-DS achieves 99.94% accuracy using a Random Forest with 500 trees + KMeans-SMOTE + PCA (Talukder et al., Scientific Reports 2025). But this RF model is 5-50MB in size and cannot run on a sensor node with only 10KB RAM.

**The gap:** No one has compressed a high-accuracy IDS model into something small enough to deploy on actual WSN sensor nodes. Specifically:
- Zero papers apply Knowledge Distillation to WSN-DS (verified April 2026, Semantic Scholar search)
- Zero papers apply Curriculum Learning to WSN-DS (verified)
- Zero papers combine CL + KD for any IDS task (verified)

## 1.2 What We Propose

CuKD-XAI: A three-stage pipeline:
1. **Curriculum Learning (CL):** Train a large teacher model easy-to-hard (Blackhole first → Scheduling last)
2. **Knowledge Distillation (KD):** Compress teacher into a tiny student MLP (~1.2K-3.4K params, ~1.2-3.3KB INT8)
3. **SHAP Explainability:** Explain which WSN features matter for each attack type

## 1.3 Why This Matters (From Base Paper)

Our base paper — Ghadi et al., "ML Solutions for Security of WSNs: A Review," IEEE Access 2024 (DOI: 10.1109/ACCESS.2024.3355312) — is a pure review paper (no experiments, no code, no new method). It identified 4 open issues:
- (A) Training location
- (B) **Lightweight ML algorithms** ← we address this
- (C) Privacy
- (D) Trust domain

We also address the **interpretability gap** not mentioned in the base paper but critical for trust in deployed IDS.

---

# PART 2: DATASET — WSN-DS (COMPLETE SPECIFICATION)

**Source:** Almomani et al., "WSN-DS: A Dataset for IDS in Wireless Sensor Networks," Journal of Sensors, 2016.
- **DOI:** 10.1155/2016/4731953
- **Paper:** https://onlinelibrary.wiley.com/doi/10.1155/2016/4731953
- **Kaggle:** https://www.kaggle.com/datasets/bassamkasasbeh1/wsnds
- **Generated with:** NS-2 simulator, LEACH routing protocol

## 2.1 Features (19 columns)

| # | Feature Name | Description | Type |
|---|---|---|---|
| 1 | `Id` | Node identifier | Integer (**DROP — non-informative**) |
| 2 | `Time` | Timestamp of the record | Float |
| 3 | `Is_CH` | Whether the node is a Cluster Head | Binary (0/1) |
| 4 | `who_CH` | ID of the node's Cluster Head | Integer |
| 5 | `Dist_To_CH` | Distance to the Cluster Head | Float |
| 6 | `ADV_S` | Advertisement messages sent | Integer |
| 7 | `ADV_R` | Advertisement messages received | Integer |
| 8 | `JOIN_S` | Join request messages sent | Integer |
| 9 | `JOIN_R` | Join request messages received | Integer |
| 10 | `ADV_SCH_S` | Advertisement schedule messages sent | Integer |
| 11 | `ADV_SCH_R` | Advertisement schedule messages received | Integer |
| 12 | `Rank` | Node rank in cluster hierarchy | Float |
| 13 | `DATA_S` | Data packets sent | Integer |
| 14 | `DATA_R` | Data packets received | Integer |
| 15 | `Data_Sent_BS` | Data sent to Base Station | Integer |
| 16 | `Dist_CH_BS` | Distance from Cluster Head to Base Station | Float |
| 17 | `Send_code` | Send code/type | Integer |
| 18 | `Consumed_Energy` | Energy consumed by the node | Float |
| 19 | `Attack_Type` | Class label (target variable) | Categorical (5 classes) |

**After preprocessing:** Drop `Id` → **17 usable features + 1 target = 18 columns**

## 2.2 Class Distribution (VERIFIED across 6+ sources including SOTA paper Table 3)

| Class | Count | Percentage | Imbalance Ratio (vs Normal) |
|---|---|---|---|
| Normal | 340,066 | 90.77% | 1.0x |
| Grayhole | 14,596 | 3.90% | 23.3x |
| Blackhole | 10,049 | 2.68% | 33.8x |
| Scheduling (TDMA) | 6,638 | 1.77% | 51.2x |
| Flooding | 3,312 | 0.88% | 102.7x |
| **Total** | **374,661** | **100%** | — |

## 2.3 Attack Difficulty Ranking (VERIFIED from per-class F1 across 5 papers 2019-2025)

| Attack | Per-Class F1 Range (without oversampling) | Difficulty | Why |
|---|---|---|---|
| Blackhole | 0.98-0.99 | **Easiest** | Drops ALL forwarded packets — obvious anomaly in DATA_R |
| Grayhole | 0.95-0.99 | **Medium** | Selectively drops — solved by modern feature engineering |
| Flooding | 0.95-0.97 | **Medium** | Generates excess traffic — high recall but precision issues |
| Scheduling (TDMA) | 0.93-0.96 | **Hardest** | Manipulates time slots — mimics legitimate CH behavior |

**Sources for difficulty ranking:**
- GXGBoost (Sensors 2019): Scheduling recall 0.928 (lowest) — https://www.mdpi.com/1424-8220/19/20/4383
- MC-GRU (2022): Scheduling 93.0% (lowest) — https://onlinelibrary.wiley.com/doi/10.1155/2022/2448010
- Tabu RF (Scientific Reports 2025): Scheduling recall 0.93 (lowest) — https://pmc.ncbi.nlm.nih.gov/articles/PMC12119870/
- MLSTL-WSN (2024): Scheduling F1 96.1% (lowest) — https://arxiv.org/html/2402.13277v2

**IMPORTANT NOTE:** In the original 2016 paper (basic ANN), Grayhole was hardest at 75.6%. Every modern model since 2019 shows Scheduling as hardest. Our curriculum design uses the CORRECTED modern ranking.

**NOTE ON SOTA PER-CLASS (with KMeans-SMOTE oversampling):** From SOTA paper Table 10, RFC with KMS+PCA achieves near-perfect on ALL classes: Normal 99.87, Grayhole 99.97, Blackhole 99.99, TDMA 99.93, Flooding 99.96. The oversampling essentially eliminates the class difficulty problem. Our CL approach offers an alternative to oversampling that doesn't require synthetic data generation.

## 2.4 PCA on WSN-DS

From the SOTA paper (Fig. 2), PCA with Kaiser criterion (eigenvalues > 1) retains **5 principal components** from the 17 features, preserving >95% of variance.

**Impact on our design:** If we apply PCA, input dimension drops from 17 to 5. We should test BOTH:
- Without PCA: 17 input features (more interpretable for SHAP)
- With PCA: 5 input features (more compressed, better for deployment, but SHAP less interpretable)

## 2.5 Second Dataset: CICIoT2023

- **Source:** https://www.kaggle.com/datasets/himadri07/ciciot2023
- **Records:** ~46 million (we sample to ~400K for tractability)
- **Features:** 47
- **Purpose:** Generalizability validation — proves our method works beyond WSN-DS
- **Note:** Different attack types (DDoS, Mirai, etc.) — curriculum stages must be redesigned for this dataset

---

# PART 3: CURRENT STATE OF THE ART (VERIFIED)

## 3.1 WSN-DS Leaderboard (from verified papers)

| Rank | Method | Accuracy | F1 | Year | Source |
|------|--------|----------|-----|------|--------|
| 1 | **KMS+PCA+RFC** | **99.94%** | **99.94%** | 2025 | Scientific Reports (https://www.nature.com/articles/s41598-025-87028-1) |
| 2 | XGBC+KMS+PCA | 99.93% | 99.93% | 2025 | Same paper |
| 3 | RF+SMOTE-TomekLink | 99.92% | 99.92% | 2024 | Springer (https://link.springer.com/article/10.1007/s10207-024-00833-z) |
| 4 | Optimized CNN+IGOA | 99.94% | — | 2024 | SRJECS |
| 5 | Tabu RF (TS-RF) | 99.67% | 99.67% | 2025 | Scientific Reports (https://pmc.ncbi.nlm.nih.gov/articles/PMC12119870/) |
| 6 | GSWO-CatBoost | 99.65% | — | 2024 | Sensors (https://www.mdpi.com/1424-8220/24/11/3339) |
| 7 | Standard RF | 99.66-99.72% | ~99.67% | various | Multiple |
| 8 | MLP+SMOTE-TomekLink | 98.80% | 98.80% | 2024 | Springer |
| 9 | CNN-LSTM | 97% | — | 2022 | IJACSA |
| 10 | ANN (original baseline) | ~96.6% | — | 2016 | J. of Sensors |

**KEY INSIGHT:** Accuracy is SATURATED at 99.94%. Our paper CANNOT be about improving accuracy. It MUST be framed about:
1. **Compression** (50MB RF → 1-3KB MLP student)
2. **Deployability** (fits on TelosB mote)
3. **Explainability** (SHAP per-attack analysis)
4. **Methodology** (CL as alternative to SMOTE for handling imbalance)

## 3.2 SOTA Paper Details (VERIFIED from actual PDF)

**Paper:** Talukder et al., "A hybrid machine learning model for intrusion detection in wireless sensor networks leveraging data balancing and dimensionality reduction," Scientific Reports, 2025.

- Uses KMeans-SMOTE to balance ALL classes to ~340,066 samples each (Flooding goes from 3,312 to 340,066 = 102x synthetic oversampling)
- Uses PCA with Kaiser criterion → 5 components
- Uses Random Forest Classifier as final model
- Evaluation: **5-fold cross-validation** (NOT 80/20 split)
- Preprocessing: StandardScaler + Label Encoding
- Also tested on TON-IoT dataset (99.97% accuracy)
- Compared against DTC, CBC, XGBC, LGBC, HGBC

**Per-class results (RFC, KMS+PCA, from Table 10):**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Normal | 99.85 | 99.88 | 99.87 |
| Grayhole | 99.96 | 99.97 | 99.97 |
| Blackhole | 99.98 | 99.99 | 99.99 |
| TDMA (Scheduling) | 100.00 | 99.87 | 99.93 |
| Flooding | 99.92 | 99.99 | 99.96 |

**Confusion Matrix (Table 8, KMS+PCA+RFC on WSN-DS):**

|  | Normal | Grayhole | Blackhole | TDMA | Flooding |
|---|---|---|---|---|---|
| Normal | 68,114 | 21 | 2 | 2 | 56 |
| Grayhole | 10 | 67,575 | 9 | 0 | 0 |
| Blackhole | 0 | 5 | 68,249 | 0 | 0 |
| TDMA | 86 | 2 | 0 | 68,188 | 0 |
| Flooding | 4 | 0 | 0 | 0 | 67,743 |

## 3.3 KD-IDS Landscape (2024-2026) — KD for IDS is ACTIVE but none on WSN-DS

| Paper | Year | Compression | Dataset | Innovation |
|---|---|---|---|---|
| **Benaddi et al.** | 2025 | ~253x params | TON_IoT | SHAP+KD+Kronecker |
| DTKD-IDS | 2025 | 250x size | NSL-KDD, X-IIoTID | Dual-teacher KD |
| QuCAD-IDS | 2026 | 98% params | NSL-KDD, UNSW-NB15 | Cross-domain KD |
| DNN-KDQ | 2025 | 10x size | CICIDS2017 | KD+Quantization |
| Cloud-Edge KD | 2025 | 74.5% params | Edge-IIoTset | Federated+KD |
| DistillGuard | 2025 | — | — | Transformer+KD+XAI |
| IoT/UAV KD | 2025 | 92-95% params | Multiple | Dense teacher→student |
| CP-LSKD | 2025 | ~80% | IoT datasets | Ladder-structured KD |
| Hossain et al. | 2025 | — | N-BaIoT | Federated SHAP+KD |

**CRITICAL: KD for IDS is NOT novel by itself (9+ papers). What IS novel is KD on WSN-DS specifically.**

---

# PART 4: NOVELTY VERIFICATION (Updated April 7, 2026)

## 4.1 Novelty Claims

| Gap | Status | How Verified | Date |
|---|---|---|---|
| CL + KD combined for any IDS | **ZERO papers** | Semantic Scholar search "curriculum learning knowledge distillation intrusion detection" → top results are all federated+KD, not CL+KD | Apr 7, 2026 |
| KD on WSN-DS | **ZERO papers** | Semantic Scholar exact search "WSN-DS" + "knowledge distillation" → 36 results, NONE about WSN-DS IDS dataset. Also confirmed by SOTA paper's related works review | Apr 7, 2026 |
| CL on WSN-DS | **ZERO papers** | Only 2 CL-IDS papers exist (Narkedimilli 2025a, 2025b), neither uses WSN-DS | Mar 25, 2026 |
| CL + KD + SHAP anywhere | **ZERO papers** | Cross-verified across multiple search angles | Mar 25, 2026 |

## 4.2 Three Closest Competitors (Must Cite and Differentiate)

### Competitor 1: Narkedimilli et al. 2025a
- **Title:** "Enhancing IoT Network Security through Adaptive Curriculum Learning and XAI"
- **Venue:** IEEE FNWF / arXiv:2501.11618
- **URL:** https://arxiv.org/abs/2501.11618
- **What they do:** CL with staged training (Normal → Simple → Medium → Complex attacks) + GRU+LSTM+Attention + LIME + ensemble (RF/XGBoost)
- **Datasets:** CIC-IoV-2024, CIC-APT-IIoT-2024, EDGE-IIoT
- **Results:** 97-98% accuracy, CL alone contributed +6% (but baseline only saw Normal data — unfair comparison)
- **Architecture:** 94,051 parameters
- **Difficulty metric:** Human-defined attack category groupings (NOT quantitative)
- **What they DON'T do:** No KD, no SHAP, no WSN-DS
- **Our differentiation:** We add KD compression, use SHAP (not LIME), target WSN-DS

### Competitor 2: Narkedimilli et al. 2025b
- **Title:** "Curriculum Learning with Image Transformation and Explainable AI for Improved Network Intrusion Detection"
- **Venue:** ICISS 2025, Springer LNCS vol. 16380
- **URL:** https://link.springer.com/chapter/10.1007/978-3-032-13714-2_18
- **What they do:** CL + image transformation + SHAP + ensemble stacking
- **Datasets:** CIC-APT-IIoT, Edge-IIoT, CIC-IoV-2024
- **What they DON'T do:** No KD, no WSN-DS

### Competitor 3: Benaddi et al. 2025 (VERIFIED FROM ACTUAL PDF)
- **Title:** "Lightweight Intrusion Detection in IoT via SHAP-Guided Feature Pruning and Knowledge-Distilled Kronecker Networks"
- **Venue:** CommNet 2025 (IEEE), arXiv:2512.19488
- **URL:** https://arxiv.org/abs/2512.19488
- **Code:** https://github.com/mohammed-jouhari/iot-ids-lightweight
- **Dataset:** TON_IoT (21.98M records, 42 features, 9 attack types)
- **Teacher:** MLP with hidden layers 512, 256, 128. **769,922 params, 3,021.53 KB** (from Table II)
- **Student:** Kronecker-factored layers. **3,042 params, 22.29 KB fp32** (from Table II; paper text says 1,282 — internal inconsistency)
- **Student int8:** 3,042 params, 22.44 KB
- **Compression:** ~253x parameter reduction (769,922 / 3,042)
- **Teacher performance:** Accuracy 0.9989, Macro-F1 0.9955
- **Student fp32 performance:** Accuracy 0.9968, Macro-F1 0.9863
- **Student int8 performance:** Accuracy 0.9969, Macro-F1 0.9867
- **Latency:** Teacher 1963ms → Student 303ms (mean, batch 1024)
- **KD Loss:** L_tot = (1-α)L_CE + αL_KD (Eq. 4)
- **KD Hyperparameters:** Grid search T ∈ {2,3,4}, α ∈ {0.5, 0.7, 0.9}
- **SHAP:** K=32 features retained from 1,624 (after one-hot encoding)
- **Training:** Google Colab, PyTorch, batch 1024, lr 1e-3, AdamW, cosine scheduler, dropout 0.3, patience 8, weighted CE
- **What they DON'T do:** No CL, no WSN-DS, no WSN-specific attacks
- **Our differentiation:** We add curriculum training, target WSN-DS with LEACH attacks, use standard MLP student (not Kronecker)

### Additional SHAP+KD Competitor: Hossain et al. 2025
- **Title:** "A novel federated learning approach for IoT botnet intrusion detection using SHAP-based knowledge distillation"
- **Venue:** Complex & Intelligent Systems, Springer, 2025
- **URL:** https://link.springer.com/article/10.1007/s40747-025-02001-9
- **What they do:** Federated learning + SHAP + KD for IoT botnet detection
- **What they DON'T do:** No CL, no WSN-DS

---

# PART 5: CORRECTED ARCHITECTURE DETAILS

**ALL PARAM COUNTS VERIFIED BY CALCULATION ON APRIL 7, 2026**

## 5.1 Teacher MLP: 128-256-128-5

```python
Teacher = nn.Sequential(
    nn.Linear(17, 128),    # 17*128 + 128 = 2,304
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 256),   # 128*256 + 256 = 33,024
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),   # 256*128 + 128 = 32,896
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 5)      # 128*5 + 5 = 645
)
```

**Linear layers:** 2,304 + 33,024 + 32,896 + 645 = **68,869 params**
**BatchNorm layers:** BatchNorm1d(128) + BatchNorm1d(256) + BatchNorm1d(128) = 256 + 512 + 256 = **1,024 params** (gamma + beta per feature)
**Total:** 68,869 + 1,024 = **69,893 params**
**Size:** 69,893 × 4 bytes = **~273 KB fp32**

**CORRECTION:** Previous documents said "~200K params" — this was WRONG. Actual is ~70K.

## 5.2 Student MLP Option A: 32-16-5

```python
StudentA = nn.Sequential(
    nn.Linear(17, 32),     # 17*32 + 32 = 576
    nn.ReLU(),
    nn.Linear(32, 16),     # 32*16 + 16 = 528
    nn.ReLU(),
    nn.Linear(16, 5)       # 16*5 + 5 = 85
)
```

**Total:** 576 + 528 + 85 = **1,189 params**
**Size fp32:** 1,189 × 4 = 4,756 bytes = **4.64 KB**
**Size INT8:** 1,189 × 1 = 1,189 bytes = **1.16 KB**
**Fits on TelosB (10KB RAM):** YES (1.16 KB << 10 KB)
**Compression vs Teacher:** 69,893 / 1,189 = **~59x parameter reduction**

**CORRECTION:** Previous documents said "~3.3K params, ~3.3KB INT8" — this was WRONG for 32-16-5. That number actually corresponds to 64-32-5 below.

## 5.3 Student MLP Option B: 64-32-5

```python
StudentB = nn.Sequential(
    nn.Linear(17, 64),     # 17*64 + 64 = 1,152
    nn.ReLU(),
    nn.Linear(64, 32),     # 64*32 + 32 = 2,080
    nn.ReLU(),
    nn.Linear(32, 5)       # 32*5 + 5 = 165
)
```

**Total:** 1,152 + 2,080 + 165 = **3,397 params**
**Size fp32:** 3,397 × 4 = 13,588 bytes = **13.27 KB**
**Size INT8:** 3,397 × 1 = 3,397 bytes = **3.32 KB**
**Fits on TelosB (10KB RAM):** YES (3.32 KB < 10 KB, but leaves only ~6.7 KB for buffers/code)
**Compression vs Teacher:** 69,893 / 3,397 = **~21x parameter reduction**

## 5.4 If PCA Applied (5 input features instead of 17)

| Architecture | With 17 features | With 5 features (PCA) |
|---|---|---|
| Student 32-16-5 | 1,189 params | 5×32+32 + 32×16+16 + 16×5+5 = 192+528+85 = **805 params (0.79 KB INT8)** |
| Student 64-32-5 | 3,397 params | 5×64+64 + 64×32+32 + 32×5+5 = 384+2080+165 = **2,629 params (2.57 KB INT8)** |

## 5.5 Teacher Random Forest: 500 trees

- **Params:** Not directly comparable (tree structure, not weights)
- **Size on disk:** Typically 5-50 MB serialized (sklearn joblib)
- **Inference:** Traverses 500 trees, O(depth × 500) per sample
- **Fits on TelosB:** IMPOSSIBLE (5-50 MB >> 10 KB RAM)
- **predict_proba():** Needs isotonic regression calibration before use as KD soft labels
- **Calibration:** `sklearn.calibration.CalibratedClassifierCV(rf, method='isotonic', cv=5)`

## 5.6 Hardware Feasibility Mapping

| Model | Size (INT8) | TelosB (10KB RAM) | MICAz (4KB RAM) | ESP32 (520KB RAM) |
|---|---|---|---|---|
| RF (500 trees) | 5-50 MB | **Impossible** | **Impossible** | Possible |
| Teacher MLP (~70K params) | ~68 KB | **Impossible** | **Impossible** | Yes |
| Student B (3,397 params) | 3.32 KB | **Yes** | Barely | Yes |
| Student A (1,189 params) | 1.16 KB | **Yes** | **Yes** | Yes |

**Source for TelosB/MICAz specs:** https://www.researchgate.net/figure/DZ50-MicaZ-and-TelosB-hardware-specifications_tbl1_265788738
**Source for MLP inference on MSP430:** Capuchin framework — https://github.com/leleonardzhang/Capuchin

## 5.7 Recommendation

**Use BOTH student sizes in the paper.** Report Student A (1,189 params) as the ultra-compact option and Student B (3,397 params) as the balanced option. This gives readers deployment choices.

---

# PART 6: COMPLETE METHODOLOGY

## 6.1 Curriculum Learning

### Difficulty Scoring Methods (Report BOTH in paper)

**Method 1: Loss-Based Difficulty (Recommended — most principled)**
Based on DIHCL (NeurIPS 2020): https://proceedings.neurips.cc/paper/2020/file/62000dee5a05a6a71de3a6127a68778a-Paper.pdf

1. Train a small probe model (same architecture as teacher) for 2-3 epochs on full dataset
2. Record per-sample cross-entropy loss
3. Low loss = easy (model already handles it), High loss = hard
4. Rank all samples by loss
5. Divide into 3 difficulty tiers based on percentiles

**Method 2: Domain-Knowledge Difficulty**
Based on verified attack difficulty ranking:

| Tier | Classes Included | Rationale |
|---|---|---|
| Tier 1 (Easy) | Normal + Blackhole | Blackhole has F1 0.98-0.99, most obvious anomaly |
| Tier 2 (Medium) | + Grayhole + Flooding | F1 0.95-0.97, more subtle |
| Tier 3 (Hard) | + Scheduling (TDMA) | F1 0.93-0.96, mimics normal CH behavior |

### Pacing Strategy

```
Stage 1 (Epochs 1-7):   Train on Tier 1 only (easiest 33% of data)
Stage 2 (Epochs 8-14):  Train on Tier 1 + Tier 2 (easiest 66%)
Stage 3 (Epochs 15-25): Train on ALL data
```

**Hyperparameters to search:**
- Starting fraction p0 ∈ {0.2, 0.33, 0.5}
- Number of stages ∈ {2, 3, 4}
- Epochs per stage ∈ {5, 7, 10}

### CL Evidence & Risks

**Evidence FOR CL on IDS:**
- Narkedimilli 2025: +6% accuracy on IoT IDS (arXiv:2501.11618)
- DDCL 2024: 2-10% improvement on tabular medical data (IEEE Access)
- CL helps specifically for noisy/imbalanced data (Wu et al., ICLR 2021)
- WSN-DS IS noisy (simulated) and imbalanced (90.77% Normal)

**Evidence AGAINST CL on tabular:**
- ICLR 2021 "When Do Curricula Work?": CL marginal in standard settings — https://openreview.net/forum?id=tW4QEInpni
- CLIMB benchmark (NeurIPS 2025): Tested 29 methods for imbalanced tabular — CL not even included — https://arxiv.org/abs/2505.17451
- Difficulty scoring paper (arXiv 2024): "No general advantage of CL over uniform sampling" — https://arxiv.org/abs/2411.00973
- Narkedimilli's +6% may be artifact of unfair baseline (baseline only saw Normal data)

**Our defense:** We frame CL as a research hypothesis being tested, not a proven improvement. If CL doesn't help, we report it honestly — negative results about when CL fails for tabular IDS is itself a contribution.

## 6.2 Knowledge Distillation

### Loss Function (VERIFIED — standard Hinton formulation, confirmed by Benaddi 2025)

```
L_total = (1 - alpha) * L_CE(y_true, softmax(z_student))
        + alpha * T^2 * KL_Div(softmax(z_teacher/T) || softmax(z_student/T))
```

Where:
- `z_student`, `z_teacher` = logits (pre-softmax outputs)
- `T` = temperature (softens probability distribution, T > 1)
- `alpha` = mixing weight between hard labels and soft labels
- `T^2` = gradient magnitude correction factor (from Hinton 2015)

### Hyperparameters (VERIFIED from Benaddi 2025 and literature)

| Parameter | Starting Value | Grid Search Range | Source |
|---|---|---|---|
| Temperature T | 4 | {2, 3, 4, 5} | Benaddi 2025 searched {2,3,4}; Hinton 2015 suggests T=2-4 for small students on tabular |
| Alpha | 0.7 | {0.5, 0.7, 0.9} | Benaddi 2025 used exact same grid; higher alpha for accurate teachers |
| Optimizer | AdamW | — | Used by Benaddi 2025 |
| Learning rate | 1e-3 | {5e-4, 1e-3, 2e-3} | Benaddi 2025 used 1e-3 |
| Weight decay | 1e-3 | — | Benaddi 2025 used 1e-3 |
| Batch size | 256 | {64, 128, 256, 1024} | Benaddi used 1024; for WSN-DS (smaller), 256 may suffice |
| Epochs | 50 | {30, 50, 100} | With early stopping (patience 8) |
| Scheduler | Cosine annealing | — | Benaddi 2025 used step 10, gamma 0.5 |
| Dropout | 0.3 | — | Benaddi 2025 used 0.3 |
| Loss weighting | Weighted CE (inverse class freq) | — | Benaddi 2025 found weighted CE > focal loss |

### RF Teacher: Probability Calibration

RF `predict_proba()` outputs are fraction of trees voting for each class. These are known to be poorly calibrated — they cluster near 0 and 1, which undermines KD's "dark knowledge" mechanism.

**Solution:** Apply isotonic regression calibration before extracting soft labels.

```python
from sklearn.calibration import CalibratedClassifierCV
rf = RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42)
rf.fit(X_train, y_train)
calibrated_rf = CalibratedClassifierCV(rf, method='isotonic', cv=5)
calibrated_rf.fit(X_train, y_train)
soft_labels = calibrated_rf.predict_proba(X_train)  # Use these for KD
```

**Alternative for MLP teacher:** MLP softmax outputs are naturally better suited for KD. Temperature scaling of logits is the standard approach.

```python
with torch.no_grad():
    teacher_logits = teacher_model(X)
    soft_targets = F.softmax(teacher_logits / T, dim=1)
```

### Expected KD Performance (from verified literature)

| Metric | Typical Range in KD-IDS papers | Source |
|---|---|---|
| Accuracy loss after KD | **0.5-3%** | Multiple papers (Benaddi: 0.21% loss; Cloud-Edge: 0.68% loss) |
| Macro-F1 loss | 0.5-2% | Benaddi: 0.9955 → 0.9863 = 0.92% loss |
| Compression ratio (params) | 20-250x | Benaddi: 253x; DTKD: 20x; Our plan: 20-58x |

**CORRECTION:** Previous documents claimed "26 KD-IDS papers show <2% loss" — this was not verified and too strong. Softened to "0.5-3% typical based on verified papers."

## 6.3 SHAP Explainability

### Implementation

```python
import shap

# For student MLP:
background = X_train[:200]  # 200 background samples
explainer = shap.DeepExplainer(student_model, background)
shap_values = explainer.shap_values(X_test[:1000])

# For RF teacher:
explainer_rf = shap.TreeExplainer(rf_model)
shap_values_rf = explainer_rf.shap_values(X_test[:1000])
```

### SHAP Outputs for the Paper

1. **Global feature importance bar plot** — which features matter most overall
2. **Per-attack-type feature importance** — which features distinguish each attack
3. **Summary (beeswarm) plot** — feature importance + value distribution
4. **Teacher vs Student feature ranking comparison** — does compression preserve reasoning?
5. **Force plots for case studies** — individual prediction explanations for Scheduling (hardest) attack

### SHAP Risks

- SHAP can be unreliable: "The Inadequacy of Shapley Values for Explainability" (arXiv 2023) — https://arxiv.org/pdf/2302.08160
- SHAP is sensitive to feature representation (ACM FAccT 2025) — https://dl.acm.org/doi/10.1145/3715275.3732105
- **Mitigation:** Run stability tests (bootstrap SHAP on 5 subsets), validate against domain knowledge, add LIME comparison if time permits

---

# PART 7: EXPERIMENT DESIGN (9 CONFIGURATIONS)

## 7.1 Configuration Table (CORRECTED param counts)

| Config | Description | CL | KD | SHAP | Teacher | Student | Purpose |
|---|---|---|---|---|---|---|---|
| **A** | RF baseline | No | No | No | RF (500 trees) | N/A | SOTA baseline (~99.7% without SMOTE) |
| **B** | Full MLP baseline | No | No | No | MLP (68,869 params) | N/A | Neural baseline |
| **C** | CL-trained full MLP | **Yes** | No | No | MLP (68,869 params) | N/A | Does CL improve teacher? |
| **D** | Small MLP from scratch | No | No | No | N/A | MLP (1,189 or 3,397 params) | Scratch student baseline |
| **E** | KD from calibrated RF | No | **Yes** | No | Calibrated RF | MLP student | Tree-to-MLP KD |
| **E2** | KD from standard MLP (no CL) | No | **Yes** | No | Standard MLP | MLP student | MLP-to-MLP KD baseline (needed to isolate CL contribution) |
| **F** | KD from CL-trained MLP | **Yes** | **Yes** | No | CL-MLP | MLP student | **CORE CLAIM: CL+KD** |
| **G** | KD with random pacing | Random | **Yes** | No | Random-pacing MLP | MLP student | Control: is ordering important? |
| **H** | Full CuKD-XAI | **Yes** | **Yes** | **Yes** | CL-MLP | MLP student | Complete pipeline with SHAP |
| **I** | KD from SMOTE-teacher | SMOTE | **Yes** | No | SMOTE-trained MLP | MLP student | CL vs SMOTE comparison |

## 7.2 What Each Comparison Proves

| Comparison | What It Tests | If better | If same/worse |
|---|---|---|---|
| **C vs B** | Does CL improve teacher? | CL helps training | CL doesn't help on tabular |
| **F vs E2** | Does CL-teacher produce better KD student? (SAME teacher arch, only CL differs) | **Core claim validated** | CL doesn't cascade |
| **E vs E2** | RF teacher vs MLP teacher for KD? | Tree KD different from MLP KD | Similar distillation quality |
| **F vs D** | Does KD beat scratch-trained student? | KD adds value | KD unnecessary |
| **F vs G** | Is difficulty ORDER important, or just pacing? | Curriculum ordering matters | Just pacing helps |
| **E vs D** | Does tree-to-MLP KD work at all? | Tree KD validated | KD doesn't help |
| **H vs F** | Does SHAP feature selection help? | Feature pruning adds value | All features needed |
| **F vs I** | CL vs SMOTE for handling imbalance in KD? | CL better than SMOTE | SMOTE still needed |
| **A vs F** | Accuracy vs model size tradeoff | Student competitive with RF | Gap quantified |

## 7.3 Metrics

| Category | Specific Metrics |
|---|---|
| **Classification** | Accuracy, Precision, Recall, F1 (macro + per-class, especially Scheduling) |
| **Compression** | Parameter count, Model size (KB fp32 and INT8), FLOPs per inference |
| **Deployment** | Inference time (ms), Fits on TelosB? (yes/no) |
| **Statistical** | Mean ± std over 5 runs, Wilcoxon signed-rank test (p < 0.05) |
| **Teacher quality** | Expected Calibration Error (ECE) — does CL improve teacher calibration? |
| **Convergence** | Training loss curves — does CL improve convergence speed? |

## 7.4 Data Split

- **70% train / 15% validation / 15% test** — stratified to preserve class distribution
- Apply SMOTE **only on training set** for Config I (not on val/test)
- For CL configs: curriculum applies to training set only
- **Alternative:** 5-fold CV (matching SOTA paper). Report both if possible.

## 7.5 Statistical Protocol

- **5 independent runs** per configuration (different random seeds)
- Report mean ± standard deviation for all metrics
- **Wilcoxon signed-rank test** (p < 0.05) for key comparisons: F vs D, F vs E, C vs B
- **Effect size:** Cohen's d for practical significance

---

# PART 8: IMPLEMENTATION GUIDE

## 8.1 Environment

- **Platform:** Google Colab (free tier T4 GPU, 16GB VRAM, 12.7GB RAM)
- **Framework:** PyTorch (matching Benaddi 2025 for comparability)
- **Libraries:** scikit-learn, shap, pandas, numpy, matplotlib, seaborn
- **Compute estimate:** Full pipeline (all 9 configs, 5 runs each) = ~2-4 hours on Colab T4

## 8.2 Data Pipeline

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load
df = pd.read_csv('WSN-DS.csv')

# Drop Id column
df = df.drop('Id', axis=1, errors='ignore')

# Encode labels
le = LabelEncoder()
df['Attack_Type'] = le.fit_transform(df['Attack_Type'])
# Mapping: Blackhole=0, Flooding=1, Grayhole=2, Normal=3, TDMA=4
# (alphabetical order by default)

# Features and target
X = df.drop('Attack_Type', axis=1).values
y = df['Attack_Type'].values

# Standardize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split: 70/15/15 stratified
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)
```

## 8.3 Difficulty Scoring (for CL)

```python
# Method 1: Loss-based difficulty scoring
# Train probe model for 2 epochs, record per-sample loss

probe = nn.Sequential(
    nn.Linear(17, 64), nn.ReLU(),
    nn.Linear(64, 32), nn.ReLU(),
    nn.Linear(32, 5)
)

# Train for 2 epochs with CE loss (no reduction)
criterion_unreduced = nn.CrossEntropyLoss(reduction='none')

# After 2 epochs:
with torch.no_grad():
    logits = probe(X_train_tensor)
    per_sample_loss = criterion_unreduced(logits, y_train_tensor)

# Sort: low loss = easy, high loss = hard
difficulty_order = torch.argsort(per_sample_loss)  # ascending: easy first
```

## 8.4 KD Training Loop

```python
def kd_loss(student_logits, teacher_logits, labels, T, alpha):
    """Standard Hinton KD loss"""
    soft_student = F.log_softmax(student_logits / T, dim=1)
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    
    kd_term = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T * T)
    ce_term = F.cross_entropy(student_logits, labels)
    
    return alpha * kd_term + (1 - alpha) * ce_term
```

## 8.5 SHAP Analysis

```python
import shap

# Student model SHAP
background = torch.tensor(X_train[:200], dtype=torch.float32)
explainer = shap.DeepExplainer(student_model, background)

test_sample = torch.tensor(X_test[:1000], dtype=torch.float32)
shap_values = explainer.shap_values(test_sample)

# Feature names
feature_names = ['Time', 'Is_CH', 'who_CH', 'Dist_To_CH', 'ADV_S', 'ADV_R',
                 'JOIN_S', 'JOIN_R', 'ADV_SCH_S', 'ADV_SCH_R', 'Rank',
                 'DATA_S', 'DATA_R', 'Data_Sent_BS', 'Dist_CH_BS',
                 'Send_code', 'Consumed_Energy']

# Global importance
shap.summary_plot(shap_values, X_test[:1000], feature_names=feature_names)

# Per-class importance (one plot per attack type)
class_names = ['Blackhole', 'Flooding', 'Grayhole', 'Normal', 'TDMA']
for i, name in enumerate(class_names):
    shap.summary_plot(shap_values[i], X_test[:1000], feature_names=feature_names, title=name)
```

---

# PART 9: PAPER STRUCTURE (IEEE FORMAT)

## Recommended structure for IEEE conference paper (6 pages) or journal (10-15 pages)

### Section 1: Introduction (~1 page)
- WSN security challenge (resource-constrained nodes)
- IDS accuracy is saturated (99.94%) but models are too large for deployment
- Gap: no KD on WSN-DS, no CL+KD for IDS
- Our contribution: CuKD-XAI — CL + KD + SHAP for lightweight WSN IDS
- Organization of paper

### Section 2: Related Work (~1.5 pages)
- 2.1 IDS for WSN (cite SOTA paper, base paper, 3-4 WSN-DS papers)
- 2.2 Knowledge Distillation for IDS (cite Benaddi, DTKD, DistillGuard, DNN-KDQ — differentiate: none on WSN-DS)
- 2.3 Curriculum Learning for Security (cite Narkedimilli 2025a, 2025b — differentiate: no KD, no WSN-DS)
- 2.4 Explainability in IDS (cite SHAP papers, PSO+SHAP on WSN-DS)
- Table summarizing gap (CL × KD × SHAP × WSN-DS = 0 papers)

### Section 3: Methodology (~2 pages)
- 3.1 Overview (architecture diagram)
- 3.2 Difficulty Scoring (loss-based + domain-knowledge)
- 3.3 Curriculum-Guided Teacher Training (pacing strategy, stage design)
- 3.4 Knowledge Distillation (loss function, hyperparameters, RF calibration)
- 3.5 SHAP Feature Analysis (DeepExplainer, per-attack importance)
- 3.6 INT8 Quantization for Deployment

### Section 4: Experimental Setup (~1 page)
- 4.1 Datasets (WSN-DS, CICIoT2023)
- 4.2 Baselines and Configurations (Table of 9 configs)
- 4.3 Metrics
- 4.4 Implementation Details (Colab, PyTorch, hyperparameters)

### Section 5: Results and Discussion (~2-3 pages)
- 5.1 Baseline Results (Table: all 9 configs, accuracy/F1/params/size)
- 5.2 CL Analysis (C vs B: does CL help teacher?)
- 5.3 KD Analysis (F vs D vs E: does KD work? does CL improve KD?)
- 5.4 SHAP Analysis (per-attack feature importance, teacher vs student comparison)
- 5.5 Hardware Feasibility (model size table, TelosB fit analysis)
- 5.6 Statistical Significance (Wilcoxon test results)
- 5.7 Generalizability (CICIoT2023 results, if completed)

### Section 6: Conclusion (~0.5 pages)
- Summary of contributions
- Key finding: whether CL helps KD for tabular IDS (honest report either way)
- Limitations (WSN-DS is simulated, CL evidence is thin)
- Future work (real-world WSN data, online CL, federated deployment)

### References (~30-40 citations)

---

# PART 10: FOUR PAPER CONTRIBUTIONS

**C1 — First KD study on WSN-DS:**
We present the first knowledge distillation benchmark for wireless sensor network intrusion detection, compressing a ~70K-parameter teacher MLP (~273 KB) into a 1,189-parameter student (1.16 KB INT8), achieving [XX]x compression with [XX]% accuracy retention. The student model fits within the 10 KB RAM constraint of TelosB sensor nodes. (No prior KD paper on WSN-DS — verified via Semantic Scholar, April 2026)

**C2 — First CL investigation for tabular IDS:**
We investigate curriculum learning as an alternative to synthetic oversampling (SMOTE) for handling the severe class imbalance in WSN-DS (90.77% Normal, 0.88% Flooding). We design domain-informed difficulty curricula based on verified per-class F1 rankings across 5 published studies, and compare against loss-based difficulty scoring. We report CL's effectiveness honestly, including null results. (No prior CL paper on WSN-DS — verified)

**C3 — CL-guided KD (CuKD) — a novel paradigm for IDS compression:**
We test the hypothesis that curriculum-trained teachers produce better soft labels for knowledge distillation, resulting in higher-quality compressed students. This CL+KD combination exists in computer vision (TC3KD 2022, DCKD 2026) but has never been tested for intrusion detection. (Zero CL+KD papers for IDS — verified)

**C4 — Per-attack SHAP analysis on compressed WSN IDS model:**
We provide SHAP-based per-attack-type feature importance analysis on both the teacher and the compressed student, revealing which LEACH protocol features (energy, routing, cluster-head metrics) are most discriminative for each WSN attack type, and whether compression preserves the teacher's reasoning patterns.

---

# PART 11: RISK ASSESSMENT & MITIGATION

| Risk | Severity | Probability | Mitigation | Backup |
|---|---|---|---|---|
| CL doesn't improve tabular IDS | HIGH | 50-60% | Frame as research hypothesis; report honestly; negative result is still a contribution | Paper becomes "KD+SHAP on WSN-DS with CL investigation" |
| KD student doesn't beat scratch-trained | MEDIUM-HIGH | 30% | Include Config D vs E comparison; focus on soft-label inter-class knowledge | If gap is small, emphasize deployment and interpretability |
| WSN-DS is "solved" — reviewers reject | CRITICAL | 40% | Frame as compression/deployability paper, NOT accuracy; add CICIoT2023 | Emphasize 58x compression, TelosB deployment, SHAP insights |
| "Just combining techniques" criticism | HIGH | 50% | Ablation proves synergy (F vs E must be significant); cite CL+KD evidence from CV/NLP | Focus on the WSN-specific contributions and SHAP analysis |
| 1,189-param student is too small | MEDIUM | 30% | Test both 1,189 and 3,397 param students; report Pareto frontier | Use 3,397-param student as primary if 1,189 underperforms |
| SHAP is unstable/decorative | MEDIUM | 20% | Bootstrap stability tests; domain knowledge validation; LIME comparison | Focus on aggregate feature rankings, not individual explanations |
| RF calibration fails for KD | LOW-MEDIUM | 20% | Use isotonic regression; also have MLP teacher as backup | Config F (MLP teacher) is the primary approach; Config E (RF teacher) is supplementary |

---

# PART 12: 12 REVIEWER ATTACK VECTORS & DEFENSES

| # | Attack | Damage | Defense |
|---|---|---|---|
| 1 | "WSN-DS already solved at 99.94%" | 9/10 | Reframe as deployability paper; RF is 5-50MB, student is 1.16KB; add CICIoT2023 |
| 2 | "Just combining existing techniques" | 8.5/10 | Ablation proves synergy: Config F vs E with Wilcoxon test. CL+KD synergy hypothesis is non-obvious and testable |
| 3 | "KD unnecessary — small model works fine" | 8/10 | Config D vs E comparison. If gap is small, report honestly |
| 4 | "SHAP is unreliable/decorative" | 7.5/10 | DeepExplainer + stability tests + domain knowledge validation + LIME comparison |
| 5 | "CL has no proven tabular benefit" | 7/10 | Include random-pacing baseline (Config G); measure teacher ECE; cite ICLR 2021 exception for noisy/imbalanced data |
| 6 | "Student can't actually run on WSN nodes" | 7/10 | INT8: 1.16KB fits in 10KB RAM; cite Capuchin (MSP430 MLP inference); clarify deployment on cluster heads |
| 7 | "RF is already lightweight AND interpretable" | 6.5/10 | Measure actual RF size (5-50MB); 500-tree ensemble is NOT interpretable; student has fixed inference time |
| 8 | "WSN-DS is outdated/simulated" | 6/10 | Only LEACH benchmark (verified); validate on CICIoT2023; call for real-world datasets in future work |
| 9 | "CL overhead is unjustified" | 5.5/10 | Report wall-clock time for every config; CL cost is training-only, zero inference overhead |
| 10 | "Compression ratio is misleading" | 5/10 | Report params, size (KB), FLOPs, inference time separately; plot Pareto frontier |
| 11 | "KD propagates teacher bias" | 4.5/10 | CL as teacher debiasing mechanism; compare per-class confusion matrices teacher vs student |
| 12 | "'First to apply X' claims unverifiable" | 4/10 | Use "to the best of our knowledge"; document search methodology in paper |

---

# PART 13: NOVELTY JUSTIFICATION PARAGRAPH (For Cover Letter)

"Unlike Narkedimilli et al. (2025), who applied curriculum learning to IoT intrusion detection without model compression, and Benaddi et al. (2025), who combined SHAP-guided feature pruning with knowledge distillation on the TON_IoT dataset, our work introduces CuKD-XAI — the first framework that unifies curriculum-guided training, teacher-student knowledge distillation, and SHAP-based explainability for intrusion detection in wireless sensor networks. Applied to the WSN-DS benchmark, CuKD-XAI produces a model that is (a) accurate on hard-to-detect LEACH protocol attacks through curriculum-informed training, (b) lightweight through knowledge distillation (1,189–3,397 parameters, 1.16–3.32 KB INT8, suitable for TelosB sensor nodes with 10 KB RAM), and (c) interpretable through per-attack-type SHAP analysis revealing which WSN features drive detection of each attack. This triple contribution addresses the lightweight ML and interpretability gaps identified in recent WSN security surveys (Ghadi et al., IEEE Access 2024), while introducing curriculum learning as a principled alternative to synthetic oversampling for handling the severe class imbalance (90.77% Normal, 0.88% Flooding) inherent in WSN intrusion datasets."

---

# PART 14: COMPLETE VERIFIED REFERENCE LIST

## Core References

| # | Citation | URL | Verified? |
|---|---|---|---|
| 1 | Ghadi et al., "ML Solutions for Security of WSNs: A Review," IEEE Access, 2024. DOI: 10.1109/ACCESS.2024.3355312 | — | Yes (base paper PDF read) |
| 2 | Almomani et al., "WSN-DS: A Dataset for IDS in WSNs," J. Sensors, 2016 | https://onlinelibrary.wiley.com/doi/10.1155/2016/4731953 | Yes |
| 3 | Talukder et al., "Hybrid ML for IDS in WSN (KMS+PCA+RFC)," Scientific Reports, 2025 | https://www.nature.com/articles/s41598-025-87028-1 | **Yes (PDF read, numbers verified from Table 6, 8, 10)** |
| 4 | Benaddi et al., "SHAP-Guided KD Kronecker Networks," CommNet 2025 | https://arxiv.org/abs/2512.19488 | **Yes (PDF read, numbers verified from Table II)** |
| 5 | Narkedimilli et al., "CL + XAI for IoT IDS," arXiv:2501.11618, 2025 | https://arxiv.org/abs/2501.11618 | Yes (arXiv verified) |
| 6 | Narkedimilli et al., "CL + Image Transform for NIDS," ICISS/Springer, 2025 | https://link.springer.com/chapter/10.1007/978-3-032-13714-2_18 | Yes (Springer verified) |
| 7 | Hinton et al., "Distilling the Knowledge in a Neural Network," arXiv, 2015 | https://arxiv.org/abs/1503.02531 | Yes |
| 8 | Bengio et al., "Curriculum Learning," ICML, 2009 | https://dl.acm.org/doi/10.1145/1553374.1553380 | Yes |
| 9 | Lundberg & Lee, "SHAP," NeurIPS, 2017 | https://arxiv.org/abs/1705.07874 | Yes |
| 10 | Wu et al., "When Do Curricula Work?" ICLR, 2021 | https://openreview.net/forum?id=tW4QEInpni | Yes |
| 11 | Stanton et al., "Does KD Really Work?" NeurIPS, 2021 | https://arxiv.org/pdf/2106.05945 | Yes |

## WSN-DS Studies

| # | Citation | URL |
|---|---|---|
| 12 | Ali et al., "Tabu Search RF for WSN-DS," Scientific Reports, 2025 | https://pmc.ncbi.nlm.nih.gov/articles/PMC12119870/ |
| 13 | Hameed et al., "GSWO-CatBoost for WSN IDS," Sensors (MDPI), 2024 | https://www.mdpi.com/1424-8220/24/11/3339 |
| 14 | Akhtar et al., "MLSTL-WSN," Int. J. Information Security, 2024 | https://link.springer.com/article/10.1007/s10207-024-00833-z |
| 15 | Birahim et al., "PSO+SHAP+LIME on WSN-DS," IEEE Access, 2025 | https://ieeexplore.ieee.org/document/10836702/ |
| 16 | Salmi & Oughdir, "CNN-LSTM for DoS in WSN," J. Big Data, 2023 | https://journalofbigdata.springeropen.com/articles/10.1186/s40537-023-00692-w |
| 17 | Baniata et al., "CNN-GCN-BiLSTM for WSN IDS," Future Internet, 2026 | https://www.mdpi.com/1999-5903/18/1/5 |

## KD-IDS Papers (none on WSN-DS)

| # | Citation | URL |
|---|---|---|
| 18 | Xie et al., "DTKD-IDS: Dual-Teacher KD," Ad Hoc Networks, 2025 | https://www.sciencedirect.com/science/article/abs/pii/S1570870525001179 |
| 19 | AL-Nomasy et al., "DistillGuard: Transformer KD for IDS," Computers & Security, 2025 | https://www.sciencedirect.com/science/article/pii/S0167404825001063 |
| 20 | Zhang et al., "DNN-KDQ," J. Cloud Computing, 2025 | https://link.springer.com/article/10.1186/s13677-025-00762-9 |
| 21 | Li et al., "CP-LSKD: Ladder KD for IoT IDS," Cluster Computing, 2025 | https://link.springer.com/article/10.1007/s10586-025-05597-2 |
| 22 | Albaseer et al., "Lightweight IDS for IoT/UAV with KD," Computers (MDPI), 2025 | https://www.mdpi.com/2073-431X/14/7/291 |
| 23 | Almabdy & Ullah, "Cloud-Edge KD-IDS," arXiv:2504.10698, 2025 | https://arxiv.org/abs/2504.10698 |
| 24 | Hossain et al., "Federated SHAP+KD for IoT Botnet," Complex & Intelligent Systems, 2025 | https://link.springer.com/article/10.1007/s40747-025-02001-9 |
| 25 | Okey et al., "RAID-KL," Expert Systems w/ Applications, 2026 | https://www.sciencedirect.com/science/article/pii/S0957417425040758 |

## CL References

| # | Citation | URL |
|---|---|---|
| 26 | Soviany et al., "Curriculum Learning: A Survey," IJCV, 2022 | https://link.springer.com/article/10.1007/s11263-022-01611-x |
| 27 | Wang et al., "Dynamic CL for Imbalanced Data," ICCV, 2019 | https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Dynamic_Curriculum_Learning_for_Imbalanced_Data_Classification_ICCV_2019_paper.pdf |
| 28 | Liu et al., "CLIMB: Imbalanced Learning Benchmark," arXiv, 2025 | https://arxiv.org/abs/2505.17451 |
| 29 | Zhou et al., "DIHCL: Dynamic Instance Hardness CL," NeurIPS, 2020 | https://proceedings.neurips.cc/paper/2020/file/62000dee5a05a6a71de3a6127a68778a-Paper.pdf |
| 30 | Czarnowski, "DDCL: Data Distribution CL," IEEE Access, 2024 | https://arxiv.org/html/2402.07352 |
| 31 | "Does the Definition of Difficulty Matter?" arXiv, 2024 | https://arxiv.org/abs/2411.00973 |

## CL+KD in Other Domains (Not IDS)

| # | Citation | URL |
|---|---|---|
| 32 | TC3KD, "Teacher-Student Cooperative Curriculum," Neurocomputing, 2022 | https://www.sciencedirect.com/science/article/abs/pii/S0925231222009146 |
| 33 | Han et al., "DCKD: Dynamic Curriculum KD," Multimedia Systems, 2026 | https://link.springer.com/article/10.1007/s00530-025-02117-5 |
| 34 | "Curriculum Temperature KD," AAAI, 2023 | https://arxiv.org/abs/2211.16231 |

## XAI for IDS

| # | Citation | URL |
|---|---|---|
| 35 | Dawoud et al., "XAI-Based IDS Survey," Information (MDPI), 2025 | https://www.mdpi.com/2078-2489/16/12/1036 |
| 36 | Adeyemi et al., "Systematic Review on XAI in IDS," Frontiers in AI, 2025 | https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1526221/full |
| 37 | Eke et al., "SHAP vs LIME in IDS Models," Applied Sciences, 2025 | https://www.mdpi.com/2076-3417/15/13/7329 |

## Supporting Technical References

| # | Citation | URL |
|---|---|---|
| 38 | Sarridis et al., "KD Comprehensive Survey," TMLR, 2025 | https://arxiv.org/abs/2503.12067 |
| 39 | "Position: Embracing Negative Results in ML," ICML, 2024 | https://arxiv.org/abs/2406.03980 |
| 40 | Capuchin: MLP inference on MSP430 | https://github.com/leleonardzhang/Capuchin |
| 41 | "Dynamic Temperature Scheduler for KD," arXiv, 2025 | https://arxiv.org/html/2511.13767 |

**Total: 41 verified references with URLs**

---

# PART 15: CORRECTION LOG

All errors discovered and corrected during the research process:

| Date | Error | Old Value | Corrected Value | How Discovered |
|---|---|---|---|---|
| Mar 25 | Attack difficulty ranking | Grayhole = hardest | Scheduling = hardest | Per-class F1 from 5 papers |
| Mar 25 | Class counts | Grayhole/Flooding swapped | Fixed from 6+ sources | Cross-referencing |
| Mar 25 | CNN-BiLSTM-Attention as novel | Assumed novel | Already done (Baniata 2026) | Web search |
| Mar 25 | CL evidence +3.8% on imbalanced | Presented as strong | Image-only data; ICLR 2021 shows marginal | Literature check |
| Mar 25 | RF calibration | Not mentioned | Added isotonic regression requirement | KD literature |
| Apr 7 | Teacher params | ~200K | **68,869** | Manual calculation |
| Apr 7 | Student 32-16-5 params | ~3.3K | **1,189** | Manual calculation |
| Apr 7 | Student 64-32-5 params | Not tracked | **3,397** (the actual 3.3K match) | Manual calculation |
| Apr 7 | Benaddi student params | 1,282 (from agent) | **3,042** (from Table II of actual PDF) | Read actual PDF |
| Apr 7 | Benaddi compression | 600x (from agent) | **~253x** (769,922/3,042) | Read actual PDF |
| Apr 7 | SOTA evaluation method | Assumed 80/20 | **5-fold CV** | Read actual PDF |
| Apr 7 | PCA on WSN-DS | Not mentioned | **Kaiser criterion → 5 components** | Read actual PDF |
| Apr 7 | "26 KD-IDS papers <2% loss" | Claimed as fact | **Softened to "0.5-3% typical"** — not all 26 verified | Honesty |
| Apr 7 | KD for IDS novelty | Implied novel | **NOT novel** (9+ papers 2024-2025). KD on WSN-DS IS novel | Web search |
| Apr 7 | "DART" paper by Li et al. | Cited as real | **Does not exist** — was hallucinated by agent | Web search |
| Apr 7 | Experiment table param counts | Used old wrong numbers | Updated with verified calculations | Manual calculation |

---

# PART 16: IMPLEMENTATION TIMELINE

| Day | Task | Output |
|---|---|---|
| 1 | Data pipeline + Baselines (A, B, D) | WSN-DS loaded, preprocessed, RF/MLP baselines with metrics |
| 2 | KD experiments (E, F) | KD from RF teacher and MLP teacher, compare vs scratch student (D) |
| 2-3 | CL experiments (C, G, I) | CL teacher training, random pacing control, SMOTE comparison |
| 3-4 | SHAP analysis (H) | Per-attack feature importance, teacher vs student comparison |
| 4-5 | Statistical validation | 5 runs of all configs, Wilcoxon tests, results tables |
| 5-8 | Paper writing | IEEE format, all sections, figures, tables |
| 8+ | Professor review + submission | Adapt to venue requirements |

---

# PART 17: BACKUP PLAN

## If CL Doesn't Work (50-60% probability)

**Option A (Recommended): Keep CL as ablation study**
- Report: "We investigated whether CL improves KD for tabular IDS. Our results show marginal/no improvement, consistent with Wu et al. (ICLR 2021). This finding saves future researchers from pursuing this direction."
- Paper becomes: "KD + SHAP on WSN-DS, with CL investigation"
- This ADDS rigor — shows honesty and thoroughness
- Still 3 strong contributions: first KD on WSN-DS + SHAP analysis + CL investigation

**Option B: Drop CL entirely**
- Publish as: "Lightweight Explainable IDS for WSN via Knowledge Distillation and SHAP Analysis"
- Weaker novelty (Benaddi 2025 did SHAP+KD on TON_IoT)
- Must emphasize WSN-specific contributions (LEACH attacks, TelosB deployment, per-attack SHAP)

**The paper survives either way.** CL is high-risk/high-reward. KD + SHAP is the guaranteed floor.

---

*Document generated: April 7, 2026*
*Incorporates verified research from actual PDFs (Benaddi 2025, SOTA WSN-DS 2025)*
*All corrections logged. All references verified with URLs.*
*Every claim marked with verification status.*

---

# === v3.0 ADDITIONS (April 11, 2026) ===

The sections below supersede any conflicting claims in Parts 1-16 above.

---

# PART 17: EXPERIMENTAL RESULTS (Verified from `cukd_xai_results.json`)

## 17.1 Run metadata

- **Notebook executed:** `cukd_xai_colab.ipynb` v2.0 (pre-tagged version)
- **Configuration:** `QUICK_MODE = False`, 5 seeds, Student A only, no CICIoT2023
- **Seeds:** [42, 123, 456, 789, 1001]
- **KD hyperparameters used:** T=4, α=0.7 (either grid search confirmed these OR defaults were used — cannot disambiguate from JSON alone)
- **Results saved:** `results_download/cukd_xai_results.json`, `wsnds_results_student_A.csv`, 5 PNG figures

## 17.2 Aggregate Macro F1 (mean ± std across 5 seeds)

| Config | Description | Macro F1 | Accuracy | Params | Size |
|--------|-------------|----------|----------|--------|------|
| A_RF_500 | Random Forest baseline | 0.9791 ± 0.0003 | 0.9966 ± 0.00005 | N/A | ~85 MB |
| B_Full_MLP | Full MLP teacher (no CL) | 0.9211 ± 0.0037 | 0.9870 ± 0.0006 | 69,893 | 273 KB |
| C_CL_MLP_loss | CL teacher (loss-based diff) | **0.8607 ± 0.0025** | 0.9726 ± 0.0011 | 69,893 | 273 KB |
| C2_CL_MLP_domain | CL teacher (domain-based diff) | **0.8618 ± 0.0025** | 0.9738 ± 0.0011 | 69,893 | 273 KB |
| D_Small_MLP | Small MLP from scratch | 0.9130 ± 0.0043 | 0.9850 ± 0.0014 | 1,189 | 4.64 KB |
| **E_KD_from_RF** | **KD from calibrated RF** | **0.9219 ± 0.0137** | **0.9871 ± 0.0022** | **1,189** | **4.64 KB** |
| E2_KD_from_MLP | KD from standard MLP (no CL) | 0.9114 ± 0.0031 | 0.9852 ± 0.0004 | 1,189 | 4.64 KB |
| F_KD_from_CL_MLP | **KD from CL-MLP (original CORE CLAIM)** | **0.8908 ± 0.0034** | 0.9778 ± 0.0013 | 1,189 | 4.64 KB |
| G_KD_random_pacing | KD with random pacing control | 0.9120 ± 0.0026 | 0.9851 ± 0.0004 | 1,189 | 4.64 KB |
| I_KD_from_SMOTE_MLP | KD from SMOTE-trained teacher | 0.9091 ± 0.0058 | 0.9849 ± 0.0011 | 1,189 | 4.64 KB |

## 17.3 Interpretation

**Original core claim FAILED:** F (0.8908) was 2.06% WORSE than E2 (0.9114). CL-trained teachers produced WORSE KD students than standard teachers.

**Unexpected winner:** E (KD from calibrated RF) at 0.9219 macro F1 — the best student configuration, achieving ~94.2% of RF teacher's 0.9791 F1 at a **18,335× compression ratio** (~85 MB RF to 4.64 KB fp32 MLP, or 1.16 KB INT8).

**CL hurts teacher:** C (0.8607) is 6.04% WORSE than B (0.9211). Two hypotheses:
- **H1 (under-training):** Stage 3 only has 11 epochs of full-data exposure vs B's 30. Fix: longer Stage 3.
- **H2 (CL intrinsically doesn't help):** Consistent with Wu et al. ICLR 2021. Fix: not possible, pivot narrative.

**Random pacing ≈ standard training:** G (0.9120) ≈ E2 (0.9114) confirms that curriculum *ordering* does not matter — only pacing does (and pacing doesn't help either).

**SMOTE teacher doesn't help:** I (0.9091) ≈ E2 (0.9114). SMOTE-based oversampling does not improve KD teacher quality for this dataset.

## 17.4 SHAP Feature Alignment — NOVEL FINDING

From `shap_results.ranking_agreement_spearman`:

**Spearman ρ = −0.039, p = 0.881**

The feature importance rankings of the compressed student (Config F) and the RF teacher (Config A) are essentially uncorrelated. This means:
- The student achieves ~94% of teacher macro F1
- But uses completely different features to make its decisions
- The common assumption "distillation preserves teacher reasoning" is challenged on tabular IDS data

**This is a genuine novel contribution to the KD/XAI-IDS literature.** To the best of our knowledge (post-April 11 literature search), no prior IDS paper has reported this finding.

Student top-3 features (by mean |SHAP|): **Is_CH, Time, dist_CH_To_BS**
Teacher top-3 features: (stored in `shap_results.teacher_global_importance`, not extracted in summary)

Notably, the **metaheuristic competitor Xiao 2025** reports top features as Is_CH, Who_CH, Dist_to_CH — overlapping two of our three top student features but using a different XAI method (Cosine Amplitude Method instead of SHAP).

## 17.5 INT8 Quantization Results (Config F only)

From `quantization` key in JSON:

- fp32 on-disk size: 7.68 KB (torch.save with metadata)
- int8 on-disk size: 6.24 KB
- Size reduction: **only 18.77%** (expected 75%)
- fp32 macro F1: 0.8945
- int8 macro F1: 0.8656
- F1 drop from quantization: **−2.89%**

**This is worse than expected.** PyTorch's `quantize_dynamic` doesn't efficiently handle tiny MLPs; the TFLite metadata overhead dominates the on-disk size for models under ~10 KB. The 3% F1 drop suggests quantization-aware training (QAT) may be needed.

## 17.6 Confusion Matrix Insight (Config F)

From `F_KD_from_CL_MLP.confusion_matrix` in seed 42:

```
                Predicted
                Bl    Fl    Gr    No    TD
True Bl       1394     0  113    0    0
True Fl          0   497    0    0    0
True Gr        397     0 1791    0    1
True No          3    69  514 50341   84
True TD          0     0    1   69  926
```

Hardest confusions:
1. **Grayhole → Blackhole (397 errors)** — same mechanism (dropping packets), different rates
2. **Normal → Grayhole (514 errors)** — routing noise near legitimate behavior
3. **Blackhole → Grayhole (113 errors)** — reverse of #1

**Scheduling/TDMA is NOT the hardest class** — 93% of TDMA samples are classified correctly. Our pre-experiment prediction "TDMA is hardest" was wrong. The real difficulty is Grayhole ↔ Blackhole confusion.

---

# PART 18: CORRECTION LOG (claims from Parts 1-16 that are now invalidated or modified)

| Original Claim (Parts 1-16) | Status | Correction |
|-----------------------------|--------|------------|
| "Zero papers apply Knowledge Distillation to WSN-DS" | ✅ Still holds | Verified via April 11 literature search (CrossRef, OpenAlex, web search) |
| "Zero papers apply Curriculum Learning to WSN-DS" | ✅ Still holds | Verified |
| "Zero papers combine CL + KD for any IDS task" | ✅ Still holds | Verified |
| **"First XAI analysis on WSN-DS"** | ❌ **INVALID** | Birahim et al. 2025 (IEEE Access, DOI 10.1109/access.2025.3528341) has PSO + SMOTE-Tomek + ensemble + **SHAP + LIME** on WSN-DS with 99.73% accuracy |
| **"First compression on WSN-DS"** | ❌ **INVALID** | Alfarra & AbuSamra 2025 (ECTI-CIT, DOI 10.37936/ecti-cit.2025194.263081) apply 50% structured pruning + INT8 TFLite quantization to CNN-LSTM on WSN-DS. Vidhya & Varunadevi 2026 (IJCS, DOI 10.1002/dac.70277) apply binarized simplicial CNN (1-bit weights) on WSN-DS. |
| **"First lightweight WSN-DS IDS"** | ❌ **INVALID** | Multiple prior lightweight efforts on WSN-DS |
| "TDMA/Scheduling is the hardest class" | ❌ **PARTIALLY INVALID** | Our confusion matrix shows **Grayhole ↔ Blackhole** is the hardest confusion (397 + 113 errors). TDMA itself is detected with 93% per-class F1. The curriculum design (easy→Blackhole→Scheduling) was based on prior-literature F1 averages, not on our model's actual error modes. |
| "CL will improve KD student (core hypothesis)" | ❌ **FAILED (as run)** | F (0.8908) < E2 (0.9114) by 2%. Possibly fixable via Stage 3 over-training fix; awaiting re-run. |
| "CL improves teacher training" | ❌ **FAILED (as run)** | C (0.8607) < B (0.9211) by 6%. Same fix-or-pivot decision as above. |
| "Compression preserves teacher reasoning" (unstated assumption) | ❌ **INVALIDATED** | Spearman ρ = −0.039 between compressed student and RF teacher SHAP rankings. This became the paper's **novel alternative contribution**. |
| "Teacher MLP has ~200K params" (earlier estimate) | ❌ | Actual: **69,893 params** (verified calculation). Corrected in Part 5 of v3.0. |
| "Student MLP 32-16-5 has ~3.3K params" (earlier estimate) | ❌ | Actual: **1,189 params** at 17 input features, or 805 params with PCA to 5 features. Verified calculation. |
| "Benaddi 2025 compresses by 600×" (from literature-search agent) | ❌ | Actual from Benaddi PDF Table II: **~253× compression** (769,922 → 3,042 params, not 1,282 as paper text claims). |
| "CICIoT2023 experiments complete" (planned) | ❌ Still pending | `ciciot_results = {}` in JSON. Must run to rebut "single-dataset" reviewer critique. |
| "5-run statistical validation with Wilcoxon test" (planned) | ⚠️ Partial | 5 seeds ran, but Wilcoxon results only printed to stdout, not saved to JSON. Fix the notebook to persist these. |
| "Student B (64-32-5) results" (planned) | ❌ Not run | Student B multi-seed loop was skipped in the Colab run. Must re-run for Pareto analysis. |

---

# PART 19: REVISED PAPER NARRATIVE (post-April 11)

## 19.1 Decision tree (repeated here for accessibility)

```
Fix CL under-training hypothesis: change CL_STAGES to [(0.33,5),(0.66,5),(1.0,30)]
                                    + remove cosine scheduler from train_with_curriculum
                                    + re-run Config C on seed 42 (~15 min)
                                    
                                    │
                        ┌───────────┴───────────┐
                        │                       │
              Config C F1 ≥ 0.91?       Config C F1 still < 0.91?
                        │                       │
                        ▼                       ▼
              ORIGINAL NARRATIVE         PIVOT NARRATIVE
              CL+KD+SHAP story           Tree-to-Neural KD
              (see 19.2)                  + Feature Gap story
                                         (see 19.3)
```

## 19.2 Narrative A — Original "CL-guided KD for WSN IDS" (if CL fix works)

**Title:** "CuKD-XAI: Curriculum-Guided Knowledge Distillation with Explainable Feature Analysis for Lightweight WSN Intrusion Detection"

**Contributions:**
1. First knowledge distillation benchmark on WSN-DS (full 5-class)
2. First curriculum-learning-guided KD for tabular IDS
3. First per-attack SHAP analysis of a KD-compressed IDS model
4. 18,000× compression (calibrated RF teacher → 1.16 KB INT8 MLP student) with <2% macro F1 loss
5. Systematic ablation: CL, random-pacing control, SMOTE comparison, 5-seed Wilcoxon validation

**Strength:** The full original story. Strongest paper if CL is fixable.

## 19.3 Narrative B — Pivot "Tree-to-Neural KD + Feature Alignment Gap" (if CL fix fails)

**Title:** "Tree-to-Neural Knowledge Distillation for Full-Class WSN Intrusion Detection: 18,000× Compression with a Feature-Alignment Gap"

**Contributions:**
1. First knowledge distillation benchmark on WSN-DS (full 5-class, unlike Alfarra's 4-class)
2. **18,335× compression ratio** — 85 MB calibrated RF ensemble → 1.16 KB INT8 MLP, retaining 94% macro F1
3. **Novel feature alignment analysis** — Spearman ρ ≈ 0 between teacher and compressed student SHAP rankings. This challenges the "distillation preserves reasoning" assumption for tabular IDS.
4. **Rigorous negative result on curriculum learning** for tabular IDS — Under standard pacing configurations, CL neither improves the teacher nor produces better KD students, consistent with Wu et al. (ICLR 2021).
5. **Negative result on SMOTE-guided KD** — SMOTE-trained teachers do not improve distilled students beyond standard-trained teachers.
6. Explicit comparison with Alfarra 2025 (4-class pruning+quantization) and Xiao 2025 (metaheuristic ensemble), positioning as complementary compression techniques.

**Strength:** This is the paper our data actually supports. The Spearman finding is genuinely novel.

## 19.4 Differentiation from verified competitors (both narratives)

| Competitor | What they do | Our differentiation |
|------------|-------------|---------------------|
| Alfarra 2025 (ECTI-CIT) | Pruning + INT8 quantization on CNN-LSTM, 4-class, hybrid sensor+gateway, ~0.93 macro F1 on 4 classes | Full 5-class, single-stage on-sensor deployment, ~13-16× smaller model, KD instead of pruning, includes SHAP analysis |
| Vidhya 2026 (IJCS) | Binarized simplicial CNN on WSN-DS, 5-class | Different compression mechanism (distillation vs binarization), SHAP analysis, teacher-student framework, systematic ablation |
| Xiao 2025 (SIVP) | DNN+CatBoost ensemble + metaheuristic optimization + CAM sensitivity, 95.62% accuracy | Real compression (not just hyperparameter tuning), actual SHAP (not CAM), teacher-student framework, feature alignment finding |
| Birahim 2025 (IEEE Access) | PSO + SMOTE-Tomek + ensemble + SHAP + LIME, 99.73% accuracy, no compression | Real model compression, negative result on SMOTE, feature alignment gap finding, systematic KD ablation |
| Benaddi 2025 (CommNet) | SHAP-guided feature pruning + KD + Kronecker student, TON_IoT | Different dataset (WSN-DS), different student architecture (MLP vs Kronecker), tree-based teacher (RF vs MLP), feature alignment finding, 5-class (vs 9-class IoT) |
| Talukder 2025 (Sci Reports) | KMS-SMOTE + PCA + RF, 99.94% F1 — SOTA on WSN-DS | We distill FROM this SOTA-class teacher (RF) rather than compete on raw accuracy. Our compression ratio is the novelty, not our F1 |

## 19.5 Next action

Regardless of which narrative we commit to, the immediate next action is the **CL fix re-run** (see Execution Plan Part 10 and the Status Banner in `cukd_xai_colab.ipynb`).

**This is the single decision gate that determines everything downstream.** Do it first, before any paper writing or additional experiments.

---

*Appendix A: See `CuKD_XAI_EXECUTION_PLAN.md` for the day-by-day execution plan.*
*Appendix B: See `project_apr11_results_and_competitors.md` in the memory folder for verified-citation details.*

*Document v3.0 generated: April 11, 2026*
*Updated in response to experimental results from Colab run + thorough literature search*
