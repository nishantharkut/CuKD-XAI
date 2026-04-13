# Accuracy Improvement R&D Plan — CuKD-XAI (Post-Exam, Unconstrained)

**Date:** 2026-04-13
**Status:** Post-presentation, professor requested accuracy improvement toward 3–4% gap from teacher
**Constraints:** None (time ~4–6 weeks post-May-4, institute GPUs available, parameter budget 1,189 preserved)
**Target:** Student (Config E) macro-F1 ≥ **0.9391** (current 0.9219, gap currently 5.73%)
**Headroom needed:** +1.73 percentage points on macro-F1

---

## 1. Problem statement and diagnostic

### 1.1 The headline gap is hiding a very specific failure mode

The 5.73% macro-F1 gap between the Random Forest teacher and the distilled student is **not spread across classes**. It is almost entirely concentrated in two classes — Blackhole and Grayhole — which are the two attacks that share the most similar feature signatures.

| Class | RF Teacher F1 | Student E F1 | Gap | Student std | KD lift (E − D) |
|---|---|---|---|---|---|
| Normal | 0.9984 | 0.9979 | 0.0005 | 0.0002 | +0.0014 |
| Flooding | 0.9645 | 0.9547 | 0.0098 | 0.0004 | **+0.0187** |
| TDMA | 0.9590 | 0.9551 | 0.0039 | 0.0018 | **+0.0255** |
| **Blackhole** | 0.9902 | **0.8422** | **0.1480** | **0.0376** | **−0.0122** ⚠️ |
| **Grayhole** | 0.9834 | **0.8594** | **0.1240** | **0.0328** | +0.0109 |

Four critical diagnostics from this table:

1. **Normal, Flooding, and TDMA are already at teacher parity.** There is almost zero headroom on these three classes. Any macro-F1 improvement must come from Blackhole and Grayhole.

2. **Grayhole and Blackhole have 180× higher student variance than Normal.** These classes are not merely *lower* accuracy — they are *unstable*. Different seeds produce materially different Blackhole F1 scores (0.80–0.88 range). This is the signature of a decision boundary the student is struggling to find, not just a class it is poor at.

3. **KD is actively *hurting* the student on Blackhole** (−1.22 pp vs the no-KD baseline). The RF's Blackhole soft labels, which are nearly one-hot at 0.99, are providing a signal the student cannot usefully learn from. The student ends up worse than if we had never distilled at all on this class. This is unusual and important.

4. **KD helps on the other four classes.** The distillation framework is working — it is only one specific class (Blackhole) where the teacher-student gradient becomes adversarial.

### 1.2 Root cause hypothesis

Blackhole drops 100% of forwarded packets. Grayhole drops a selective fraction. They differ on a single continuous quantity: **the ratio of data-packets-forwarded to data-packets-received**. In WSN-DS terminology, this is approximately:

```
drop_rate = 1 − Data_Sent_To_BS / max(DATA_R, 1)
```

This quantity is not one of the 17 raw features. It exists only as an **interaction** between three raw features (`DATA_R`, `DATA_S`, `Data_Sent_To_BS`).

A 500-tree Random Forest, with depth unlimited and ~70,000 total split nodes, can reconstruct this interaction implicitly — each split probes one of the underlying features and the ensemble aggregates the partial signals. A **2-layer 32-16 ReLU MLP with 1,189 parameters** plausibly cannot carve a stable decision boundary over this interaction because:

- It has only two non-linearities (two ReLU layers)
- Its hidden bottleneck at 16 neurons restricts the rank of internal representations
- It sees the 17 raw features once and must build the interaction from scratch inside its first hidden layer, which is only 32-dimensional

The RF teacher's soft labels for Blackhole are sharp — `predict_proba` returns close to `[0, 0, 0.99, 0, 0]` on clean Blackhole samples — but this sharp-target training signal does not help the student because **the student cannot represent the decision surface that would justify that sharpness**. Gradient descent on a representable surface that approximates the RF boundary is worse than gradient descent on the hard labels alone.

### 1.3 What this implies for the research plan

The standard KD improvement literature (FitNets, RKD, CRD, TAKD) targets optimization-side problems: gradient alignment, feature-map matching, capacity gap smoothing. These help when the student *can* represent the target but is struggling to *find* it. Our problem is the opposite — the student cannot represent the target. Optimization tricks will give diminishing returns.

**Effective interventions must attack capacity in one of two ways:**

1. **Increase the information density of each input sample** (feature engineering, augmentation, tree embeddings). Gives the student richer raw input so less has to be learned inside the hidden layers.
2. **Reallocate the 1,189-parameter budget toward shapes that better serve fine-grained boundary learning** (wider first layer, activation-function changes, residual connections, gated units).

**Secondary interventions** (still worth testing, lower expected yield):
- Better teacher quality (XGBoost, TabNet, stacked ensembles)
- Alternative distillation losses (Decoupled KD, Contrastive Representation Distillation)
- Training procedure (SAM, SWA, EMA, OneCycleLR)
- Curriculum learning *specifically on the Blackhole-Grayhole subset*

We rank these in Section 3 and schedule them in Section 4.

---

## 2. The realistic ceiling

Before planning, be honest about what is reachable.

| Scenario | Student macro-F1 | Gap | Publishability |
|---|---|---|---|
| Current baseline (Config E, vanilla KD) | 0.9219 | 5.73% | Already publishable as-is |
| Conservative (feature engineering only) | 0.9300 – 0.9350 | 4.4–4.9% | Same story, cleaner gap |
| **Target (prof's ask)** | **0.9391+** | **≤4%** | **Strong story** |
| Aspirational (multiple workstreams combine) | 0.9450 – 0.9500 | 2.9–3.5% | Strong story + SHAP faithfulness finding |
| Ceiling (student matches teacher within seed noise) | 0.9700 + | ~1% | Unlikely at 1,189 params |
| Hard ceiling (teacher itself) | 0.9791 | 0% | Impossible without changing teacher |

**The 0.9391 target requires closing 30% of the current gap.** That is not trivial at a fixed parameter budget. It is also not unreasonable — our diagnostic shows the gap is concentrated, not diffuse, which gives us leverage. A single high-impact intervention that unlocks Blackhole-Grayhole stability could move us most of the way there.

**The 0.9500 aspirational ceiling requires closing ~50% of the gap** and would require multiple interventions to combine positively without canceling each other. This is research, not engineering — there is real risk it does not materialize even with weeks of effort.

Plan accordingly: treat 0.9391 as the success criterion, 0.9450 as a stretch goal, and 0.9500 as an upside scenario that would justify a premier venue submission.

---

## 3. Workstreams ranked by (expected gain × confidence × cost)

Each workstream below has: target mechanism, literature anchor, expected gain, confidence, compute cost, kill criterion.

### Workstream A — Domain-informed feature engineering ⭐ HIGHEST PRIORITY

**Mechanism:** Pre-compute the feature interactions that the 500-tree Random Forest learns implicitly, and hand them to the student as additional input columns. The student's input dimension grows from 17 → 21-25, and the first-layer parameter count grows by ~50-200 params, keeping us under the 1,300 budget.

**Why it directly addresses our root cause:** The diagnostic in Section 1.2 identified `drop_rate` as the interaction most likely driving the Blackhole-Grayhole confusion. If we hand this quantity to the student as a literal input feature, the student no longer needs to reconstruct it inside a 32-neuron layer.

**Candidate engineered features (ranked by mechanism):**

```python
# TIER 1 — directly target Blackhole/Grayhole
drop_rate          = 1 - Data_Sent_To_BS / max(DATA_R, 1e-6)
forward_ratio      = Data_Sent_To_BS / max(DATA_S, 1e-6)
data_drop_count    = DATA_R - Data_Sent_To_BS  # absolute count

# TIER 2 — targets Flooding and control-plane attacks
control_to_data_ratio  = (ADV_S + JOIN_S) / max(DATA_S + 1e-6)
adv_imbalance          = abs(ADV_S - ADV_R) / max(ADV_S + ADV_R, 1e-6)
join_imbalance         = abs(JOIN_S - JOIN_R) / max(JOIN_S + JOIN_R, 1e-6)

# TIER 3 — cluster-head and topology context
ch_distance_normalized = Dist_To_CH / max(dist_CH_To_BS, 1e-6)
energy_per_packet      = Expaned_Energy / max(DATA_S + DATA_R, 1e-6)

# TIER 4 — experimental (include if TIER 1-3 lift is insufficient)
data_sch_ratio         = DATA_S / max(SCH_S + 1e-6)  # TDMA-relevant
rank_deviation         = Rank - Rank.median()
```

**Literature anchors:**
- Kadra et al. 2021 *"Well-tuned simple nets excel on tabular datasets"* — shows that feature engineering + regularization often beats more complex architectures on tabular data
- Gorishniy et al. 2021 *"Revisiting Deep Learning Models for Tabular Data"* — feature preprocessing is a larger effect than architecture choice
- Prokhorenkova et al. 2018 *"CatBoost: unbiased boosting with categorical features"* — demonstrates how interaction features can close the gap between linear and tree models

**Expected gain:** **+1.0 to +3.0 pp macro-F1**, concentrated on Blackhole and Grayhole specifically. Moderate to high confidence (about 75%) because the mechanism directly attacks the diagnosed root cause.

**Compute:** ~1 GPU-hour for full 5-seed re-run.

**Kill criterion:** If after adding tier-1 features the Blackhole student F1 does not exceed 0.87 (currently 0.8422), the capacity ceiling hypothesis is wrong and we should pivot to architecture search.

**Secondary payoff:** The engineered features are *reusable* for the paper's reproducibility section and make the work more convincing to reviewers — "we analyzed the gap, identified the root cause, and engineered a targeted fix" reads better than "we tried a dozen KD variants."

---

### Workstream B — Tree embeddings as student input features

**Mechanism:** For each training sample, run the sample through every tree in the Random Forest and record the leaf index it lands in. Convert these leaf indices into a sparse one-hot representation, then compress to a dense low-dimensional embedding via PCA or a learned projection. Append this embedding to the student's input.

This technique is sometimes called "stacking with tree features" or "tree embeddings" — it is standard in the pre-deep-learning Kaggle tabular toolbox and has appeared in recent NeurIPS work on hybrid tree-neural models.

**Why it matters for us:** A tree embedding is a *direct transfer* of the teacher's structural knowledge. It does not rely on the student reconstructing anything — it simply gives the student access to a compact summary of which tree-leaf the teacher assigned to each sample. The student then learns a small linear projection from leaf-summary → class, which is an easy learning problem.

**Literature anchors:**
- Moosmann et al. 2007 *"Fast Discriminative Visual Codebooks using Randomized Clustering Forests"* — early tree-embedding usage
- Vens & Costa 2011 *"Random Forest Based Feature Induction"*
- He et al. 2014 (Facebook) *"Practical Lessons from Predicting Clicks on Ads at Facebook"* — Facebook's canonical stacking paper, used 500-tree embeddings feeding a logistic regression

**Expected gain:** **+0.5 to +1.5 pp.** Moderate confidence (about 60%). Adds meaningful new signal but may overlap with Workstream A if both engineered features and tree embeddings encode similar information.

**Compute:** ~30 min for embedding generation + 1 GPU-hour re-run.

**Compatibility:** Combines well with Workstream A (different information sources). Combines poorly with Workstream D4 (KD temperature annealing) only because of debugging complexity, not mechanism.

**Kill criterion:** If leaf embeddings increase student input dim above ~25 without measurable Blackhole F1 improvement, the compression narrative weakens. Cap at 8 leaf-embedding dimensions.

---

### Workstream C — Architecture search within the 1,189-param budget

**Mechanism:** The current `17 → 32 → 16 → 5` shape is one arbitrary choice from a much larger design space. Sweep width-vs-depth trade-offs at a fixed parameter budget and select the best shape.

**Candidate architectures:**

| Shape | Params | Parameter distribution | Hypothesis |
|---|---|---|---|
| 17 → 32 → 16 → 5 (current) | 1,189 | 48% / 44% / 7% | Balanced |
| 17 → 48 → 8 → 5 | 1,301 | 66% / 31% / 3% | Wide first, narrow second |
| 17 → 28 → 20 → 5 | 1,189 (exact) | 42% / 49% / 9% | Wider second layer |
| 17 → 24 → 24 → 5 | 1,145 | 42% / 52% / 10% | Equal hidden layers |
| 17 → 40 → 12 → 5 | 1,277 | 55% / 39% / 5% | Intermediate |
| 17 → 56 → 5 (shallow) | 1,293 | 100% / 0% / 0% | Single hidden layer, wide |

**Plus activation function sweep on the winner:**
- ReLU (current) — baseline
- GELU — smoother, slightly better gradient flow in small networks
- Mish — Misra 2019, empirically competitive
- SiLU/Swish — Ramachandran 2017, useful for shallow networks

**Literature anchors:**
- Wu et al. 2019 *"Wider or Deeper: Revisiting the ResNet Model for Visual Recognition"* — width-vs-depth trade-off at fixed compute
- Tan & Le 2019 *"EfficientNet"* — compound scaling principles
- Kadra et al. 2021 (again) — the "well-tuned simple MLP" paper. Their best shapes for tabular are wide and shallow, not deep and narrow. Specifically they recommend single-hidden-layer MLPs with 3–10× more hidden units than input features, matching our `17 → 56 → 5` candidate.

**Expected gain:** **+0.3 to +1.0 pp**, with the single-hidden-layer wide shape being the most promising candidate based on Kadra et al.'s tabular findings. Medium confidence (50%). 

**Compute:** ~3 GPU-hours for full sweep (6 architectures × 3 seeds each, then 5-seed re-run on the winner).

**Interaction note:** If a wider-first architecture wins, it should combine multiplicatively with Workstream A (engineered features feed directly into a wider first layer and give the student more capacity to use the new signal).

**Kill criterion:** If no architecture beats the current shape by ≥0.3 pp on the full 5-seed run, stop the sweep and move on.

---

### Workstream D — Advanced distillation losses

Instead of vanilla Hinton KD (KL divergence on soft labels + CE on hard labels), test alternative distillation formulations that may transfer the teacher's knowledge more effectively for our specific problem.

#### D1 — Decoupled Knowledge Distillation (DKD, Zhao et al. CVPR 2022) ⭐

**Mechanism:** Hinton KD's KL divergence on soft labels treats all classes equivalently. Zhao et al. prove that the standard KD loss can be algebraically decomposed into two terms:

```
KD_loss = TCKD (target-class KD) + NCKD (non-target-class KD)
```

where TCKD captures how well the student matches the teacher on the correct class, and NCKD captures how the student distributes probability mass among the *wrong* classes. Zhao et al. show that NCKD is the larger source of useful signal, but standard KD under-weights it.

DKD introduces two independent weights (α for TCKD, β for NCKD) and finds that β should typically be much larger than α. On CIFAR-100 they report 0.5–2% gains.

**Why this matters for us:** Our problem is confusion between Blackhole and Grayhole — a *non-target-class* problem. The student is getting the top-1 class mostly right, but distributing its remaining probability mass poorly between these two similar attacks. NCKD directly attacks this failure mode.

**Literature anchor:** Zhao, B. et al. 2022 *"Decoupled Knowledge Distillation"* CVPR 2022. Strong paper, well-cited, widely reproduced.

**Expected gain:** **+0.5 to +1.0 pp.** Medium-high confidence (60%) because the theoretical analysis directly maps to our diagnostic.

**Compute:** ~1 GPU-hour (two hyperparameters α, β to tune, ~6-point grid).

---

#### D2 — Contrastive Representation Distillation (CRD, Tian et al. ICLR 2020)

**Mechanism:** Instead of matching output logits, CRD matches *representations* via a contrastive InfoNCE loss. For each sample, the student's representation is pulled close to the teacher's representation of the same sample and pushed away from teacher representations of other samples. This forces the student to learn a representation space that preserves the teacher's similarity structure.

**Adaptation for our tabular setting:** The student has no meaningful intermediate features on a 32-16 MLP — the only layer with useful dimensionality is the 16-neuron pre-output layer. Attach a projection head that maps the 16-d student layer to the teacher's final representation dimension, train with InfoNCE.

The problem: RF does not have a natural "representation layer." We would need to use the raw probability vector `predict_proba` (dim 5) as the teacher representation, which is degenerate.

**Workaround:** Train an intermediate MLP distilled from the RF, use its pre-output 128-d layer as the "teacher representation," then apply CRD from that MLP to the tiny student. This is essentially TAKD + CRD combined.

**Literature anchor:** Tian, Y. et al. 2020 *"Contrastive Representation Distillation"* ICLR 2020.

**Expected gain:** **+0.3 to +0.8 pp.** Medium confidence (45%). The tabular adaptation is nontrivial and CRD gains on tabular are under-studied.

**Compute:** ~4 GPU-hours (more complex training loop, hyperparameters to tune, intermediate teacher to train).

---

#### D3 — Relation-based KD (RKD, Park et al. CVPR 2019)

**Mechanism:** Instead of matching per-sample logits, match *pairwise* relationships between samples. For a minibatch, compute the pairwise distance matrix in the teacher's representation space and in the student's, then penalize their difference. Forces the student to preserve the teacher's global sample-similarity structure even if individual predictions diverge.

**Literature anchor:** Park, W. et al. 2019 *"Relational Knowledge Distillation"* CVPR 2019.

**Expected gain:** **+0.2 to +0.6 pp.** Lower confidence (35%). RKD's gains on simple tabular problems are typically small because pairwise relationships are less informationally rich than for image tasks with high-dimensional features.

**Compute:** ~2 GPU-hours.

**Kill criterion:** If DKD (D1) succeeds, skip RKD entirely. They are redundant in expectation.

---

#### D4 — Temperature annealing schedule

**Mechanism:** Instead of fixed temperature T=4 throughout training, start with a high temperature (T=6 or 8) for the first 10 epochs, then cosine-anneal T down to 2 over the remaining 20 epochs. High T early gives the student soft, exploratory targets that emphasize relative class ordering. Low T late sharpens the targets once the student has a basic representation.

**Literature anchor:** Mentioned in passing in many KD papers; Li et al. 2023 *"Curriculum Temperature for Knowledge Distillation"* at AAAI formalized it.

**Expected gain:** **+0.1 to +0.4 pp.** Low confidence (30%). Cheap to try.

**Compute:** ~30 min.

---

### Workstream E — Teacher quality upgrade

Current teacher: Random Forest with 500 trees, isotonic calibrated, 0.9791 macro-F1. Alternative teachers to test:

| Teacher | Expected teacher F1 | Notes |
|---|---|---|
| RF 500 (current) | 0.9791 | Baseline |
| RF 1000, depth-30 | ~0.980 | Marginal improvement |
| XGBoost (default) | ~0.982 | Often beats RF on tabular |
| LightGBM (default) | ~0.982 | Fast, similar quality to XGBoost |
| CatBoost | ~0.983 | Category-aware gradient boosting |
| TabNet | ~0.975 | Attention-based, may have transferable representations |
| FT-Transformer | ~0.978 | Feature Tokenizer + Transformer for tabular |
| Ensemble stack (RF + XGB + LGBM) | 0.984-0.986 | Averaged soft labels; known to give 0.1-0.3% over best individual |

**Key question:** Does a better teacher produce a better student through KD, *or* is the student bottleneck independent of teacher quality?

Our current data gives a preliminary answer. Config E2 (KD from MLP teacher, macro-F1 0.9211) produces student 0.9114. Config E (KD from RF teacher, macro-F1 0.9791) produces student 0.9219. That is a **+0.58 pp teacher lift that translates to +1.05 pp student lift**. Ratio of ~1.8×. That means teacher quality *does* transmit to the student, but with leverage — a 0.1 pp teacher improvement could yield ~0.18 pp student improvement.

If the XGBoost/LightGBM ensemble reaches teacher F1 ≈ 0.985 (+0.6 pp over RF), expected student lift: **+1.0 pp** at the current KD configuration. Combined with Workstream A and D1, this stacks.

**Literature anchors:**
- Chen & Guestrin 2016 *"XGBoost"* — the tabular workhorse
- Ke et al. 2017 *"LightGBM"* — equivalent quality, faster
- Arik & Pfister 2019 *"TabNet"* — attention-based tabular
- Gorishniy et al. 2021 *"FT-Transformer"*
- Wolpert 1992 *"Stacked generalization"* — ensemble teacher theory

**Expected gain:** **+0.3 to +1.0 pp** depending on teacher choice and ensembling. Medium confidence (55%). The mechanism is well-established; the question is whether WSN-DS specifically rewards stacked ensembles.

**Compute:** ~4 GPU-hours (train XGBoost, LightGBM, possibly TabNet; then re-run KD with each as teacher on 5 seeds).

**Risk:** If the best teacher is not calibrated, the soft-label distribution becomes pathological (near one-hot) and KD transfer weakens. Calibrate all candidate teachers with isotonic regression before distillation.

---

### Workstream F — Training-procedure improvements

These are cheaper interventions that typically add 0.1–0.5 pp each. None is a silver bullet but they combine additively.

| Technique | Mechanism | Expected | Cost |
|---|---|---|---|
| **SAM (Sharpness-Aware Minimization)** — Foret 2021 | Optimizes for flat minima; empirically better generalization | +0.2 to +0.5 pp | +50% training time |
| **SWA (Stochastic Weight Averaging)** — Izmailov 2018 | Average weights over the final epochs; gives a single model | +0.1 to +0.3 pp | Almost free |
| **EMA (Exponential Moving Average)** | Maintain a rolling average of weights during training; use averaged weights at inference | +0.1 to +0.3 pp | Nearly free |
| **Label smoothing on the CE term** | Replaces one-hot labels with slightly-soft labels; regularizes | +0.1 to +0.3 pp | Free |
| **Mixup for tabular** — Zhang 2018 | Train on convex combinations of (x, y) pairs; regularization + data augmentation | +0.2 to +0.5 pp | Small |
| **OneCycleLR** — Smith 2019 | Cyclical learning rate schedule; converges faster with better final accuracy | +0.1 to +0.3 pp | Free |
| **Weight decay sweep** | Try `wd ∈ {1e-4, 1e-3, 1e-2}` | +0.0 to +0.4 pp | Free |
| **Gradient clipping** | Stabilizes training on minority classes | +0.0 to +0.2 pp | Free |

**Literature anchors:**
- Foret, P. et al. 2021 *"Sharpness-Aware Minimization"* ICLR
- Izmailov, P. et al. 2018 *"Averaging Weights Leads to Wider Optima and Better Generalization"*
- Zhang, H. et al. 2018 *"Mixup: Beyond Empirical Risk Minimization"* ICLR
- Smith, L. N. 2018 *"A disciplined approach to neural network hyper-parameters"*

**Expected combined gain:** **+0.5 to +1.0 pp** if most tricks stack positively. Medium confidence (50%).

**Compute:** ~2 GPU-hours for a quick ablation of which tricks help, then integrate winners.

**Strategy:** Bundle all the "free" tricks (SWA, EMA, label smoothing, gradient clipping, wd sweep) into one run — they almost certainly stack additively. Then test SAM and Mixup individually if headroom remains.

---

### Workstream G — Subset curriculum on Blackhole-Grayhole

**Mechanism:** Our curriculum learning attempt (Config C) failed at the macro scale. But the diagnostic says the *only* hard part of the problem is distinguishing Blackhole from Grayhole. We can build a curriculum *specifically for this sub-problem*:

1. **Phase 1** — train the student on the four "easy" classes (Normal, Flooding, TDMA, and either Blackhole or Grayhole but not both) for 10 epochs
2. **Phase 2** — add the remaining attack class, train 10 more epochs
3. **Phase 3** — train on all 5 classes with standard KD for 30 epochs

This is a *data curriculum* rather than a *difficulty curriculum* — the distinction avoids Wu et al.'s ICLR 2021 critique because we are not ordering samples by difficulty, we are ordering *subtasks* by complexity.

**Literature anchors:**
- Bengio et al. 2009 *"Curriculum Learning"* (base reference, dated)
- Soviany et al. 2022 *"Curriculum Learning: A Survey"* — systematic review
- Jiang et al. 2015 *"Self-Paced Curriculum Learning"*

**Expected gain:** **+0.2 to +0.8 pp.** Low-medium confidence (40%). Has not been tried and the mechanism is plausible, but curricula on small MLPs are historically fickle.

**Compute:** ~1 GPU-hour.

**Kill criterion:** If the subset curriculum does not improve Blackhole or Grayhole specifically, abandon — the technique is only useful if it helps where the diagnostic says the problem is.

---

### Workstream H — Two-stage classifier (architectural pivot)

**Mechanism:** Instead of a single 5-way classifier, train **two models** that together fit within the 1,189-parameter budget:

- **Stage 1** (tiny): binary `Normal-vs-Attack` classifier. This is the easy subproblem — 0.998 F1 already achievable at ~200 params.
- **Stage 2** (remaining budget): 4-way attack classifier that only sees samples Stage 1 flagged as attacks. Can be ~900 params, and only needs to distinguish Blackhole, Grayhole, Flooding, TDMA.

**Why this might win:** Stage 2 has a concentrated parameter budget on the hard sub-problem. No capacity is wasted learning Normal — which we already nail at the teacher level. All 900 params go toward the Blackhole-Grayhole-Flooding-TDMA boundary.

**Trade-offs:**
- Cascaded error: if Stage 1 misclassifies an attack as Normal, Stage 2 never sees it. Stage 1 must therefore be tuned for high recall on the attack class.
- Inference time is two forward passes instead of one (but still sub-millisecond on a mote).
- Changes the paper narrative: it is no longer a single-model story. Needs to be framed as "modular lightweight IDS" rather than "compressed RF."

**Literature anchors:**
- Viola & Jones 2001 *"Rapid Object Detection using a Boosted Cascade"* — cascade classifier foundations
- Angelova et al. 2015 *"Real-Time Pedestrian Detection With Deep Network Cascades"* — modern cascades
- Panda et al. 2016 *"Conditional Deep Learning for Energy-Efficient and Enhanced Pattern Recognition"* — energy-motivated cascades on constrained devices

**Expected gain:** **+0.5 to +1.5 pp.** Medium-high confidence (60%) for the mechanism, but lower confidence (40%) that the paper narrative survives the restructure.

**Compute:** ~3 GPU-hours.

**Strategic note:** This is the highest-variance bet in the plan. It could either be the single biggest win or a total dead-end that also requires rewriting half the paper. Treat as workstream 2 or 3 priority, not 1.

---

### Workstream I — Parameter-efficient capacity tricks

These expand the effective capacity of the student without literally increasing parameter count.

#### I1 — Gated Linear Units (GLU) in the first layer

**Mechanism:** Replace the first `Linear(17 → 32)` with a `GLU(Linear(17 → 64))` where half the output channels act as multiplicative gates on the other half. Net output dimension is still 32, but the gating mechanism doubles the effective nonlinear capacity of the first layer.

**Cost:** Double the first-layer parameters (from 576 to 1,152). The budget must be redistributed — e.g., shrink the second layer from `32 → 16` to `32 → 8` or drop the output layer bias. Total param count stays near 1,189.

**Literature anchor:** Dauphin et al. 2017 *"Language Modeling with Gated Convolutional Networks"* — GLU formulation. Shazeer 2020 *"GLU Variants Improve Transformer"* — extended analysis.

**Expected gain:** **+0.3 to +0.8 pp.** Medium confidence (50%) because GLU has not been tested on tabular small nets specifically.

**Compute:** ~1 GPU-hour.

#### I2 — Weight sharing across layers

**Mechanism:** Share the `32 → 16` second-layer weights across a "residual block" that runs twice (input → 32 → 16 → 32 → 16). This doubles depth without doubling parameters.

**Literature anchor:** Lan et al. 2020 *"ALBERT"* — weight sharing in BERT reduced params from 108M to 12M with comparable accuracy. Principle transfers to tiny MLPs.

**Expected gain:** **+0.1 to +0.4 pp.** Low confidence (30%). Depth gains at this scale are typically small.

**Compute:** ~30 min.

---

## 4. Execution timeline (6 weeks post-exam, assuming ~10 GPU-hours per week)

### Week 1 — Baseline + Workstream A (feature engineering)
- Day 1: Reproduce current Config E with v2.3 fixes (CL-fair, all 5 seeds). Confirm 0.9219 baseline.
- Days 2–3: Implement tier-1 and tier-2 engineered features. Single-seed ablation of each to identify which features actually lift Blackhole/Grayhole.
- Days 4–5: Full 5-seed run with the chosen feature set. **Decision point: does Workstream A alone close the gap to <4.5%?**
- If yes → proceed to Week 2 for additive gains
- If no → re-diagnose root cause, possibly pivot

### Week 2 — Workstream D1 (DKD) + Workstream F (training tricks)
- Days 1–2: Implement DKD loss, tune α/β hyperparameters on one seed
- Days 3–4: Bundle the "free" training tricks (SWA, EMA, label smoothing, OneCycleLR) into the training loop
- Day 5: Full 5-seed run combining Workstream A + D1 + F. **Decision point: Is macro-F1 ≥ 0.9391?**
- If yes → stop further experiments, write paper
- If no → proceed to Week 3

### Week 3 — Workstream B (tree embeddings) + Workstream E (better teacher)
- Days 1–2: Generate tree-leaf embeddings from the 500-tree RF; integrate as additional student features
- Days 3–4: Train XGBoost and LightGBM alternative teachers; run KD from each
- Day 5: Full 5-seed run with the best combo so far

### Week 4 — Workstream C (architecture search) + Workstream I (capacity tricks)
- Days 1–2: Sweep 6 architectures on 3 seeds each; pick winner
- Day 3: GLU first-layer variant on the winning architecture
- Days 4–5: Full 5-seed run on the best configuration found

### Week 5 — High-risk workstream: Workstream H (two-stage classifier)
- Only execute if the previous 4 weeks did not close the gap to 4%
- Days 1–3: Implement cascaded binary → 4-class pipeline
- Days 4–5: Evaluation, decision on whether to commit this to the paper narrative

### Week 6 — Ablation freeze, final 10-seed run, paper update
- Day 1–2: Lock the final configuration
- Days 3–4: Run 10 seeds (not 5) for publication-grade confidence intervals
- Day 5: Update the paper draft, regenerate figures, finalize slide deck for internal review

---

## 5. Combined expected outcomes

**If Workstream A alone succeeds (feature engineering only):**
- Macro-F1: ~0.93–0.935
- Gap: 4.4–4.9%
- Status: Slightly above target. Stop early. Safe outcome.

**If Workstream A + D1 (DKD) + F (training tricks) combine:**
- Macro-F1: ~0.938–0.945
- Gap: 3.4–4.1%
- Status: **Target achieved.** Paper-ready.

**If Workstream A + D1 + F + E (better teacher) combine:**
- Macro-F1: ~0.943–0.950
- Gap: 2.9–3.6%
- Status: **Stretch goal achieved.** Strong submission to a top venue.

**If Workstream H (cascade) succeeds:**
- Macro-F1 could reach 0.950+
- But requires paper restructure
- Worth the gamble only if weeks 1–4 underperform

**If nothing meaningfully improves:**
- Macro-F1: ~0.922–0.928 (small incremental gains only)
- Gap: 5.1–5.7%
- Status: Still above the prof's 5–6% tolerance. Lean on the SHAP faithfulness novelty for the paper.

---

## 6. Risks and failure modes

### Risk 1 — The engineered features do not generalize
If `drop_rate` is genuinely the discriminating quantity between Blackhole and Grayhole but one specific seed happens to split samples unluckily, the student may overfit to the feature on training and underperform on test. **Mitigation:** 5-seed validation always. Variance across seeds is the tell.

### Risk 2 — Workstreams interfere with each other
Example: adding engineered features (A) + switching to DKD (D1) + SAM optimizer (F) may destabilize training. **Mitigation:** Always re-run vanilla baseline with each new technique *alone* first, then combine. Document which combinations cancel.

### Risk 3 — Chasing teacher F1 without student benefit
A better teacher (Workstream E) that the student cannot learn from is wasted compute. **Mitigation:** Always measure teacher→student lift ratio. If a teacher has higher F1 but lower student lift, it is a worse teacher for our purpose.

### Risk 4 — Paper narrative drift
Some workstreams (H, two-stage classifier) fundamentally change what CuKD-XAI is. If we end up with a cascade instead of a single distilled MLP, we need to rewrite slides 5–7 of the presentation and sections 3–4 of the paper. **Mitigation:** Decide in Week 5 whether to commit to the restructure, based on whether the simpler approaches cleared the target.

### Risk 5 — Overfitting to the 5-seed validation
With enough tries, someone will find a configuration that happens to win on the 5 seeds tested. **Mitigation:** Run 10 seeds on the final chosen configuration as a held-out confirmation. The 10-seed result is what appears in the paper; the 5-seed is for iteration.

### Risk 6 — Compute budget creep
Unconstrained experiments can consume weeks of GPU time without converging. **Mitigation:** Enforce the 6-week timeline strictly. Kill any workstream that does not produce a win within its allocated days.

---

## 7. Paper narrative implications

Depending on which workstreams succeed, the paper title and framing change. Prepare three candidate titles:

**Scenario A — Feature engineering dominates:**
> *"Domain-Informed Knowledge Distillation: Closing the Fine-Grained Attack Discrimination Gap in Compressed WSN Intrusion Detection"*
>
> Core claim: engineered drop-rate and forwarding-ratio features, combined with standard KD, enable a 1,189-parameter student to match a Random Forest teacher within 4% on WSN-DS across all 5 classes.

**Scenario B — Multiple workstreams combine:**
> *"CuKD-XAI: A Multi-Lever Approach to Lightweight Explainable Intrusion Detection for Wireless Sensor Networks"*
>
> Core claim: the combination of domain features, decoupled distillation, and ensemble teachers closes the compression gap to 3–4%. SHAP faithfulness finding is the second contribution.

**Scenario C — Cascade pivot wins:**
> *"A Two-Stage Cascade for KB-Scale Wireless Sensor Network Intrusion Detection"*
>
> Core claim: splitting the detection task into a binary attack-vs-normal stage and a specialized 4-way attack classifier outperforms single-model distillation at fixed parameter budget.

---

## 8. Literature reading list (for workstream implementation)

Before starting each workstream, read these in order:

**Workstream A (feature engineering):**
- Kadra et al. 2021 — *Well-tuned simple nets excel on tabular datasets* (NeurIPS)
- Gorishniy et al. 2021 — *Revisiting Deep Learning Models for Tabular Data* (NeurIPS)
- He et al. 2014 (Facebook) — *Practical Lessons from Predicting Clicks on Ads*

**Workstream B (tree embeddings):**
- Moosmann et al. 2007 — tree embedding theory
- Vens & Costa 2011 — *Random Forest Based Feature Induction*

**Workstream D1 (DKD):**
- Zhao et al. 2022 — *Decoupled Knowledge Distillation* (CVPR) — MUST-READ
- Hinton et al. 2015 — *Distilling the Knowledge in a Neural Network* (NIPS-WS) — base reference

**Workstream D2 (CRD):**
- Tian et al. 2020 — *Contrastive Representation Distillation* (ICLR)
- Oord et al. 2018 — *Representation Learning with Contrastive Predictive Coding* (for InfoNCE background)

**Workstream E (alternative teachers):**
- Chen & Guestrin 2016 — *XGBoost: A Scalable Tree Boosting System*
- Ke et al. 2017 — *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*
- Arik & Pfister 2019 — *TabNet: Attentive Interpretable Tabular Learning*
- Gorishniy et al. 2021 — *FT-Transformer*

**Workstream F (training tricks):**
- Foret et al. 2021 — *Sharpness-Aware Minimization* (ICLR)
- Izmailov et al. 2018 — *Averaging Weights Leads to Wider Optima*
- Zhang et al. 2018 — *Mixup: Beyond Empirical Risk Minimization* (ICLR)
- Smith 2018 — *Disciplined approach to neural network hyper-parameters* (a.k.a. OneCycleLR)

**Workstream G (curriculum):**
- Soviany et al. 2022 — *Curriculum Learning: A Survey* (IJCV)
- Wu et al. 2021 — *When Do Curricula Work?* (ICLR) — counter-evidence, essential reading

**Workstream H (cascade):**
- Viola & Jones 2001 — *Rapid Object Detection using a Boosted Cascade*
- Panda et al. 2016 — *Conditional Deep Learning for Energy-Efficient IoT*

**Workstream I (capacity tricks):**
- Dauphin et al. 2017 — *Language Modeling with Gated Convolutional Networks* (GLU)
- Lan et al. 2020 — *ALBERT: A Lite BERT* (weight sharing)
- Shazeer 2020 — *GLU Variants Improve Transformer*

---

## 9. Execution checklist per experiment

For every run, log:
- [ ] Baseline config reproduced (Config E, v2.3 fixes, 5 seeds) → 0.9219 ± noise
- [ ] Change applied (one variable at a time)
- [ ] Training hyperparameters documented
- [ ] 5-seed run completed
- [ ] Macro-F1 aggregate + per-class F1 + per-class std recorded
- [ ] Wilcoxon p-value vs baseline computed (one-sided, pre-registered direction)
- [ ] SHAP audit repeated to confirm Spearman novelty still holds (do not break the secondary contribution)
- [ ] Results written to `results_download/cukd_xai_results_vN.json` (increment N per experiment)
- [ ] Short 3-sentence summary added to `experiment_log.md`: what changed, result, next action

This discipline is what separates R&D that converges from R&D that wanders.

---

## 10. Minimum-viable path (if time is tighter than expected)

If exam prep eats into post-exam time and the 6-week plan becomes 3 weeks, execute only:

1. **Workstream A** (feature engineering) — Week 1
2. **Workstream D1** (DKD) — Week 2
3. **Workstream F** (free training tricks) — Week 2 (parallel with D1)
4. **5-seed confirmation run on the combined best** — Week 3

This minimum path is high-confidence and should close the gap to ~4.2–4.5%. That is the floor we should aim to guarantee. Everything else is upside.

---

## 11. Summary

The 5.73% gap is concentrated on Blackhole and Grayhole, which are fundamentally similar attacks distinguished only by drop rate — a feature interaction the 1,189-param student cannot represent but the Random Forest can. The entire improvement plan is therefore oriented around **giving the student access to that interaction** rather than asking it to learn it from scratch.

The highest-confidence intervention is domain-informed feature engineering (Workstream A). The most theoretically promising KD improvement is Decoupled KD (Workstream D1), which decomposes the KL loss in a way that specifically targets non-target-class confusion — exactly our failure mode. The cheapest wins come from Workstream F (SWA, EMA, label smoothing, OneCycleLR).

Target 0.9391 is realistic on the 3-workstream minimum path. Target 0.9450 is achievable if the full plan executes cleanly. Target 0.9500 is a stretch that would require either the cascade pivot (H) or the full 6-workstream combination stacking positively.

**The plan is structured so that each workstream has a clean kill criterion.** Stop any intervention that does not produce a measurable win within its allocated days. The worst outcome is to spend weeks on a technique that produces 0.1 pp gains and then discover the time was wasted.

---

*End of plan. Revise after exams when actual execution begins.*
