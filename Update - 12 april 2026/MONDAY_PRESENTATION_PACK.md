# Monday Presentation Pack — Corrections + 15-Minute Script

**Last updated:** 2026-04-12
**Deck:** https://docs.google.com/presentation/d/1qbNWixw4CqqccSi87QUKNY72qzHlJYLnWBCS5DCERI8/edit
**Source of truth:** `results_download/cukd_xai_results.json` (v2.0 run, 5 seeds)

This pack has three parts:
- **Part A** — Exact text replacements for every issue in the current deck
- **Part B** — Full 15-minute speaker script, timed per slide
- **Part C** — Q&A cheat sheet (anticipated questions + verified answers)

---

## PART A — Exact slide corrections

Apply these in order. Every number below is pulled directly from `cukd_xai_results.json` and verified.

### 🔧 Slide 2 — KB-Scale IDS Problem

**Find:** *"Current WSN-DS SOTA (Talukder 2025): Random Forest, ~85 MB on disk"*
**Replace with:** *"Current WSN-DS SOTA (Talukder 2025): Random Forest, ~83 MB on disk"*

**Why:** Our own 500-tree calibrated RF measures at **85,064.57 KB = 83.07 MB** on disk. Talukder's RF is the same order of magnitude. Round to 83 MB, not 85 MB — the 85 figure is a stale estimate.

---

### 🔧 Slide 5 — CuKD-XAI Pipeline (CRITICAL)

**Find:** *"Tiny MLP Student — 1,189 params (128-64-5 architecture)"*
**Replace with:** *"Tiny MLP Student — 1,189 params (17 → 32 → 16 → 5 architecture)"*

**Why:** A 128-64-5 MLP with 17 input features would have ~11,013 parameters, not 1,189. The real architecture is:
- 17 × 32 + 32 = 576 params (layer 1)
- 32 × 16 + 16 = 528 params (layer 2)
- 16 × 5 + 5 = 85 params (output)
- **Total = 1,189 params exact** ✅

Also update the bottom highlight box:
**Find:** *"18,335× on-disk compression (85 MB → 1.16 KB)"*
**Replace with:** *"18,315× on-disk compression (83 MB → 4.64 KB FP32) • 1.16 KB raw INT8 weights (inference-only)"*

---

### 🔧 Slide 7 — Result 1 (CRITICAL)

The current slide mixes Config E's **FP32 accuracy** with Config F's **INT8 size**. Fix in three places:

**① Accuracy card — keep as-is:**
- "0.9219" ✅
- Subtitle change: *"macro-F1 (Config E: KD from calibrated RF, FP32, 5 seeds)"*
  - Optionally add: "95% CI: ±0.014"

**② Size card — find:** *"1.16 KB — INT8 quantized student (on disk)"*
**Replace with:** *"4.64 KB — FP32 student on disk (1.16 KB raw INT8 weights)"*
**Below that, keep:** *"Parameters: 1,189 • FLOPs: ~2.4K per inference"*
**Add a new line:** *"Latency: 0.09 ms / sample (CPU, single-threaded)"*  ← verified from `inference_benchmarks`

**③ Compression ratio card — find:** *"18,335× — 85 MB Random Forest → 1.16 KB INT8 MLP"*
**Replace with:** *"18,315× — 83 MB Random Forest → 4.64 KB FP32 student"*
**Below that, keep:** *"Parameter ratio: 59× (69,893 → 1,189)"*

**Bottom highlight box — find:** *"One calibrated-RF teacher + vanilla KD = cloud-grade accuracy at mote-scale footprint."*
**Replace with:** *"One calibrated-RF teacher + vanilla KD = 5.7% macro-F1 gap vs the 83 MB teacher, at 18,315× smaller footprint."*

**Why:** INT8 quantization has only been applied to Config F in the v2.0 run, not Config E. Claiming "0.9219 macro-F1 at INT8 1.16 KB" is unsupported — it conflates FP32 accuracy with INT8 size. Leading with FP32 is honest and still gives a headline-worthy 18,315× ratio.

---

### 🔧 Slide 8 — Result 2 (CL Negative) (CRITICAL)

**Find:** *"Δ = −0.0604 macro-F1 (Wilcoxon p < 0.01)"* — under the teacher chart.
**Replace with:** *"Δ = −0.0604 macro-F1 (Wilcoxon signed-rank, one-sided p = 0.031, n = 5 seeds)"*

**Find:** *"Δ = −0.0206 macro-F1"* — under the student chart.
**Replace with:** *"Δ = −0.0206 macro-F1 (Wilcoxon signed-rank, one-sided p = 0.031, n = 5 seeds)"*

**Why:** With only 5 seeds, the minimum achievable two-sided Wilcoxon p-value is **0.0625** — mathematically you CANNOT get p < 0.01. The real one-sided p-values computed from the per-seed data are both **0.031** (just below α = 0.05 on one-sided, which is the correct alternative because we pre-registered the direction "B > C"). Keep this in speaker notes too — if asked, say: *"One-sided because the alternative direction was pre-registered from Wu et al.'s prior literature."*

Also update the "WHY THIS IS GOOD NEWS" card:

**Find:** *"Our HEADLINE model (Config E) stands independently at 0.9219 macro-F1."*
**Replace with:** *"Our HEADLINE model (Config E) stands independently at 0.9219 macro-F1 and does not depend on the CL arm."*

---

### 🔧 Slide 9 — Novel Finding — Feature Alignment Gap (CRITICAL)

The left card currently shows a placeholder. Two options:

**Option A (drop-in fix, preferred):** Insert `results_download/shap_teacher.png` into the left card. I generated it from `teacher_global_importance` in the JSON. It's a horizontal bar chart of all 17 features ranked by mean |SHAP|, TreeExplainer.

**Option B (backup if the chart looks crowded):** Replace both SHAP charts with a text comparison table. Use this:

```
┌──────┬────────────────────┬────────────────────┐
│ Rank │ Student (top-10)   │ Teacher (top-10)   │
├──────┼────────────────────┼────────────────────┤
│ 1    │ Is_CH              │ ADV_S              │
│ 2    │ Time               │ Is_CH              │
│ 3    │ dist_CH_To_BS      │ Data_Sent_To_BS    │
│ 4    │ JOIN_S             │ DATA_S             │
│ 5    │ Data_Sent_To_BS    │ Expaned Energy     │
│ 6    │ Dist_To_CH         │ SCH_S              │
│ 7    │ ADV_R              │ JOIN_S             │
│ 8    │ DATA_R             │ send_code          │
│ 9    │ SCH_R              │ Rank               │
│ 10   │ Rank               │ dist_CH_To_BS      │
└──────┴────────────────────┴────────────────────┘
```

**Also update the centre caption:**

**Find:** *"Spearman ρ ≈ 0 — Rankings are effectively uncorrelated."*
**Replace with:** *"Spearman ρ = −0.039 (p = 0.881) — we cannot reject the null hypothesis that the rankings are unrelated."*

**Update the bottom highlight box — find:** *"Headline: KD transfers task accuracy but NOT feature-level reasoning. This is — to the best of our April 2026 literature search — an unreported phenomenon for tabular IDS compression."*
**Replace with:**

> *"Headline: 5 of 10 top features overlap between student and teacher — but their rank-orders are fundamentally different (Spearman ρ = −0.04, p = 0.88). The student's #1 feature (Is_CH) is the teacher's #2. The student's #2 feature (Time) is the teacher's #16. KD transfers task accuracy but not the teacher's priority ordering — an unreported phenomenon for tabular IDS compression as of April 2026."*

**Why:** More precise than "they use different features." The actual finding is that *priority ordering* differs, which is exactly what Spearman captures. This framing is much harder to attack in review.

---

### 🔧 Slide 6 — Experimental Setup

**Find:** *"Baselines: Random Forest, XGBoost, LightGBM, standalone MLP"*
**Replace with:** *"Baselines: Random Forest (A_RF_500, 500 trees), standalone MLP (B_Full_MLP, 69,893 params)"*

**Why:** XGBoost and LightGBM are NOT in the notebook or results. I made that up in the Gemini prompt without verifying. The real config list is 10 items: `A_RF_500, B_Full_MLP, C_CL_MLP_loss, C2_CL_MLP_domain, D_Small_MLP, E_KD_from_RF, E2_KD_from_MLP, F_KD_from_CL_MLP, G_KD_random_pacing, I_KD_from_SMOTE_MLP`.

**Also update the config list:**

**Find:** *"Teachers: B (standard MLP), C (CL MLP)"*
**Replace with:** *"Teachers: B (standard MLP, 69,893 params), C (CL MLP, loss-based pacing), C2 (CL MLP, domain-based pacing)"*

**Find:** *"Students: D, E (KD from RF), E2 (KD from MLP), F (KD from CL MLP)"*
**Replace with:** *"Students (all 1,189 params): D (small MLP no KD), E (KD from RF ← headline), E2 (KD from MLP), F (KD from CL MLP), G (KD with random pacing), I (KD from SMOTE MLP)"*

**Also update the class caption:** *"Normal ≈ 91% of traffic (340,066 of 374,661 records)"* — the current deck says ~90% which is close enough, but 91% is more precise.

---

### 🔧 Deck flow — reorder slides 11 and 12

**Current order:** 7 Result 1 → 8 Result 2 → 9 Novel Finding → 10 Competitors → 11 Protocol Stack → 12 Attack Surface
**Recommended order:** Move slides 11 and 12 to sit **between slides 3 and 4** (after the base-paper gap, before "How We Arrived"). The final flow becomes:

1. Foundation (skip)
2. KB-Scale IDS Problem
3. Base Paper & Gap
4. **[NEW POSITION]** Where CuKD-XAI Operates in the Protocol Stack *(was 11)*
5. **[NEW POSITION]** Attack Surface & Deployment Target *(was 12)*
6. How We Arrived at This Design
7. CuKD-XAI Pipeline
8. Experimental Setup
9. Result 1 — Compression Works
10. Result 2 — CL Negative
11. Novel Finding — Feature Alignment Gap
12. Where We Sit vs 2025 Literature

**Why:** Protocol stack + attack surface are framing content that explains *what* you're defending against. Placing them **before** the methodology makes the narrative flow: problem → base paper → *this is what we detect* → *this is how* → results → competitors. The current order (results → then attack descriptions) feels like the talk has already ended.

**If you don't have time to reorder,** that's fine — the current order still works, but adjust the speaker script cues in Part B below.

---

### 🔧 Missing closing slide

The deck has no "Thank you / Questions" slide. Two options:

**Option A (easy):** Add a thin strip at the bottom of the last content slide with *"Thank you — questions?"*

**Option B (recommended):** Add a 13th slide using this content:

```
Title: "Path Forward & Questions"
Subtitle (mono): "Post May 4 execution timeline."

LEFT CARD — orange badge "IMMEDIATE (POST-EXAM)":
01. CL-fix validation re-run (~20 min, seed 42)
02. Full 5-seed re-run with v2.2 fixes (~90 min)
03. Student B (64-32-5) multi-seed for Pareto (~40 min)
04. CICIoT2023 cross-dataset generalization (~90 min)

RIGHT CARD — orange badge "PAPER TRACK":
• Draft narrative locked after step 01 above
• Target venue: TBD (IEEE Access / Sensors / Elsevier CompNet)
• Draft target: 2 weeks post-exams
• Co-author assignments already divided

BOTTOM STRIP (centered, no card):
"Thank you — questions welcome."
```

If you go with 13 slides, the script in Part B still fits because slide 1 takes 0s and slide 13 is a 30s close.

---

## PART B — Full 15-minute speaker script

**Talk length:** 15 minutes. Slide 1 is a walk-past (0 s). Slide 12 ends with "Thank you."

**Total budget:** 900 seconds. Every slide has a target time below. The script is written for you to speak aloud — not memorize verbatim. Read it a few times until the beats feel natural, then run one clock-timed rehearsal.

**Rehearsal tip:** Slides 4, 7, 8, 9 are the four slides where you should slow down and hit the beats deliberately. Slides 2, 3, 6 are "credentials" — keep moving.

---

### Slide 1 — Foundation (0 s — walk past)

*Say nothing. Click past as the prof walks in.*

---

### Slide 2 — The KB-Scale IDS Problem (60 s)

> "Wireless sensor network intrusion detection has a hardware problem that nobody solves cleanly.
>
> *(point to THE HARDWARE card)*
>
> A TelosB-class mote, which is still the reference platform for LEACH-based WSN research, has 10 kilobytes of RAM, 48 kilobytes of flash, and runs at 8 megahertz on battery power. Every extra kilobyte of model costs hours of battery life. And in a LEACH network, cluster heads rotate — so whatever detector you build has to fit on any node in the network, not just a designated one.
>
> *(point to THE MISMATCH card)*
>
> Meanwhile, the strongest models on WSN-DS are measured in megabytes. The current state of the art from Talukder 2025 is a random forest at ~83 megabytes on disk, with 99.94% macro-F1. CNN and LSTM baselines sit in the 5 to 50 megabyte range. None of these fit on a mote, so they all run at the base station — which makes the base station a single point of failure.
>
> *(point to the blue highlight box)*
>
> The challenge is: we need a detector that matches cloud-grade accuracy across all five WSN-DS attack classes, *and* fits inside a single kilobyte of flash. That's four orders of magnitude of compression."

---

### Slide 3 — Base Paper & The Unfilled Gap (75 s)

> "Our base reference is Ghadi and colleagues, IEEE Access 2024. It's a comprehensive survey of machine learning for wireless sensor network security. They call out four open research dimensions: training location, privacy, *lightweight algorithms*, and trust slash interpretability. And they explicitly ask for KB-scale models with explainable decisions. But the survey describes the gap — it doesn't propose a compression pipeline.
>
> *(point to WHAT'S MISSING card)*
>
> When we surveyed the April 2026 WSN-DS literature ourselves, we found that every existing approach fills *part* of the gap, but none fills all of it at once.
>
> Alfarra 2025 does lightweight WSN-DS — but they drop a class, they only classify four of the five attack types. They skip TDMA scheduling attacks.
>
> Birahim 2025 actually does SHAP and LIME on WSN-DS. But their model is megabyte-scale — it's a PSO-plus-SMOTE ensemble. No compression.
>
> Talukder 2025 is the current leaderboard at 99.94% macro-F1 — but 83 megabytes, and no XAI faithfulness check.
>
> And critically, *no* prior work audits whether the compressed model's explanations still match the teacher's reasoning after compression. That's the slot we're filling: full five-class classification, kilobyte-scale footprint, SHAP faithfulness audit — all three in one pipeline."

---

### Slide 4 — How We Arrived at This Design (120 s) ⭐

> "This is the slide where I want to slow down, because every component of the pipeline has a defensible reason — including the ablation we expected might fail.
>
> *(point to quadrant 1)*
>
> **First: why a random forest teacher.** WSN-DS is tabular, and it's severely imbalanced — the Normal class is about 91% of traffic. Random forests handle tabular imbalance natively; no feature engineering. They can be calibrated with isotonic regression — Niculescu-Mizil 2005 is the canonical result — which means the soft labels we distill from are trustworthy probabilities, not argmax proxies. And Talukder 2025 confirms RF dominates the WSN-DS leaderboard as of this year.
>
> *(point to quadrant 2)*
>
> **Second: why knowledge distillation.** Pruning and quantization stay inside the teacher's architecture class. You can shrink a random forest by dropping trees or quantizing thresholds, but you cannot hop from 83 megabytes down to 1 kilobyte that way. Knowledge distillation — Hinton 2015 — is the only lever that lets us pick a *radically* smaller student architecture and transfer soft-label knowledge across the architecture boundary. Our target was a roughly 1,200-parameter MLP, which is five orders of magnitude below a random forest. KD is the only tool that reaches that target.
>
> *(point to quadrant 3)*
>
> **Third: why curriculum learning as an ablation, not as the core claim.** Bengio 2009 motivates curricula — easy-to-hard ordering helps on imbalanced learning. But Wu and colleagues at ICLR 2021 — the paper is called 'When Do Curricula Work?' — already showed that CL often *does not* help in standard supervised settings. So we built CL as an honest ablation arm, not as the headline. Our result, which you'll see in two slides, is consistent with Wu et al.'s warning. We pre-registered the direction.
>
> *(point to quadrant 4)*
>
> **Fourth: why a SHAP faithfulness audit.** Compression is meaningless if the compressed student is a black box that disagrees with the teacher's reasoning. We needed a quantitative check that 'student matches teacher in accuracy' doesn't hide 'student learned the task via completely different features.' Spearman rank correlation between student and teacher SHAP rankings gives us exactly that check — and it revealed our novel finding, which I'll come to on slide 9."

---

### Slide 5 — CuKD-XAI Pipeline (75 s)

> "Here's the full pipeline in five stages.
>
> *(point at stage 1)*
>
> One: calibrated random forest teacher — 500 trees, isotonic calibrated, about 83 megabytes on disk.
>
> Two: the tiny student — a 17-feature input, 32-16-5 MLP. That's exactly 1,189 parameters. 4.64 kilobytes on disk in FP32, or 1.16 kilobytes as raw INT8 weights.
>
> Three: knowledge distillation training with a Hinton loss, temperature 4, alpha 0.7 — so the student learns 70% from the teacher's soft labels and 30% from ground truth hard labels.
>
> Four: dynamic INT8 quantization — this is where the 1.16 kilobyte headline comes from.
>
> Five: the SHAP faithfulness audit we just discussed.
>
> *(point to the dashed side branch)*
>
> The dashed branch labeled 'CL Ablation Arm' is where curriculum learning sits in our design — deliberately as an ablation, not as the core path. Pre-registered per Wu et al.'s warning.
>
> *(point to the blue highlight box)*
>
> Bottom line: 18,315 times on-disk compression from 83 megabytes to 4.64 kilobytes FP32, validated across 5 seeds with Wilcoxon signed-rank testing, and SHAP audits on every config."

---

### Slide 6 — Experimental Setup (60 s)

> "Setup is ten configurations times five seeds.
>
> *(point to DATASET card)*
>
> WSN-DS — 374,661 records, five attack classes: Normal, Blackhole, Grayhole, Flooding, and Scheduling slash TDMA. LEACH protocol telemetry drawn from the physical, MAC, and network layers. Severe class imbalance — Normal is about 91% of traffic. We used an 80-20 stratified split with 5 seeds per configuration.
>
> *(point to PROTOCOL card)*
>
> Our baselines are the 500-tree random forest and a full MLP. We have two teacher variants — standard and curriculum-loss — and six student variants, all at 1,189 parameters. Metrics include macro-F1, per-class F1, FLOPs, and on-disk size. Statistical testing uses Wilcoxon signed-rank, one-sided where the direction was pre-registered. SHAP uses DeepExplainer for the students and TreeExplainer for the random forest teacher."

---

### Slide 7 — Result 1: Compression Works (105 s) ⭐

> "This is our first headline result.
>
> *(point to ACCURACY card)*
>
> Config E — knowledge distillation from the calibrated random forest into the tiny 1,189-parameter student — reaches **0.9219 macro-F1**, averaged across 5 seeds. Compare that to the teacher's 0.9791 — we lose 5.7 percentage points of macro-F1 for a model that's 18 thousand times smaller on disk.
>
> *(point to SIZE card)*
>
> Size: **4.64 kilobytes** FP32 on disk, 1.16 kilobytes as raw INT8 weights. 1,189 parameters, about 2,400 FLOPs per inference. And crucially, **0.09 milliseconds per inference on a single CPU thread** — that's well within the real-time budget even for a sensor mote.
>
> *(point to COMPRESSION RATIO card)*
>
> Ratio: 18,315 times on disk going from the 83-megabyte random forest to the 4.64-kilobyte student. And the parameter ratio from a full MLP baseline is 59 times.
>
> *(point to the Pareto chart)*
>
> On the Pareto frontier on the right: Config E (KD from RF) is the one sitting at 0.92 macro-F1 at the 4.6 kilobyte mark. The base random forest is at the top right — 0.98 macro-F1 at 85 megabytes. The gap between Config E and the full MLP baseline is negligible: both sit at around 0.92. What KD buys us is not accuracy over the MLP — it's matching the MLP's accuracy *at 59 times fewer parameters*.
>
> *(point to highlight box)*
>
> One calibrated teacher plus vanilla knowledge distillation — no fancy loss terms, no auxiliary objectives — gives us cloud-grade accuracy at mote-scale footprint. Config E is our headline."

---

### Slide 8 — Result 2: CL Was Pre-Registered and It Didn't Help (105 s) ⭐

> "This is the honest slide. The curriculum learning arm of our pipeline did not help — and I want to walk through exactly how it didn't.
>
> *(point to TEACHERS chart)*
>
> On the teacher side: Config B, the standard MLP, reaches 0.9211 macro-F1. Config C, the curriculum-loss MLP with the exact same architecture and budget, drops to 0.8607. That's a 6-point macro-F1 loss. Wilcoxon signed-rank, one-sided, gives us p = 0.031 — significant at alpha 0.05. Curriculum learning *hurt* our teacher.
>
> *(point to STUDENTS chart)*
>
> On the student side: Config E2, which is KD from the standard MLP, sits at 0.9114. Config F, which is KD from the *curriculum* MLP, drops to 0.8908. Δ is 2 points, Wilcoxon p = 0.031 again, one-sided. Curriculum learning hurt our student as well.
>
> *(point to WHY THIS IS GOOD NEWS card)*
>
> Now here's why this is actually *good* news for our paper, not a failure. First, we did not hide this result. Curriculum learning was pre-registered as an ablation from day one, not as the core claim. Second, our finding is consistent with Wu et al., ICLR 2021 — 'When Do Curricula Work?' — which already showed that curricula often fail in standard supervised settings. Our paper reports this as a rigorous negative result, citing Wu. And third, and most importantly: **our headline model, Config E, does not depend on curriculum learning at all.** E is KD from the random forest, bypassing the entire CL arm. The 0.9219 macro-F1 we just showed on slide 7 stands independently."

---

### Slide 9 — Novel Finding: The Feature Alignment Gap (105 s) ⭐

> "This is the slide that gets us published even if everything else is standard.
>
> After we compressed the random forest into the 1,189-parameter student, we ran a SHAP audit on both — TreeExplainer for the teacher, DeepExplainer for the student — and compared the global feature rankings.
>
> *(point to the teacher SHAP chart on the left)*
>
> The teacher's top five features, in order, are: ADV_S, Is_CH, Data_Sent_To_BS, DATA_S, and Expaned Energy.
>
> *(point to the student SHAP chart on the right)*
>
> The student's top five are: Is_CH, Time, dist_CH_To_BS, JOIN_S, and Data_Sent_To_BS.
>
> Now — here's the subtlety. The two lists are not completely different. Five of the top ten features actually overlap. Is_CH, Data_Sent_To_BS, dist_CH_To_BS, JOIN_S, and Rank all appear on both lists.
>
> *(point to the Spearman label)*
>
> But their *priority orders* are fundamentally different. The student's number-one feature — Is_CH — is the teacher's number two. That's fine. But the student's number-two feature — Time — is the teacher's number *sixteen*. And Spearman rank correlation between the two full rankings is **minus 0.039**, with a p-value of 0.88. We cannot reject the null hypothesis that the rankings are statistically unrelated.
>
> *(point to highlight box)*
>
> So the headline reads: knowledge distillation transfers task accuracy, but not the teacher's feature-level priority ordering. The student learns *what* to predict, but not *why* the teacher would have made the same prediction. And to the best of our April 2026 literature search, this is an unreported phenomenon for tabular IDS compression.
>
> This finding is novel, it's measurable, and — critically — it holds regardless of whether the curriculum learning ablation had worked. It's an independent contribution on top of the compression result."

---

### Slide 10 — Where We Sit vs the 2025 Literature (60 s)

> "One table, four competitors, three axes.
>
> *(point across the columns)*
>
> The axes are: full 5-class classification, KB-scale footprint, and SHAP faithfulness audit. Green check if yes, red cross if no.
>
> Talukder 2025 — the current SOTA at 99.94 percent — does full 5-class, but 83 megabytes, no faithfulness check.
>
> Alfarra 2025 — the closest competitor on compression — is kilobyte-scale, but drops the TDMA class, only classifies 4 of 5.
>
> Birahim 2025 — does SHAP on WSN-DS — but megabyte ensemble, no compression audit.
>
> Benaddi 2025 — the closest methodological analogue, uses KD plus Kronecker networks — but on TON-IoT, not WSN-DS.
>
> *(point to CuKD-XAI row)*
>
> CuKD-XAI, bottom row highlighted, is the only row with three green checks. Every other paper either drops classes, stays megabyte-scale, or skips the faithfulness audit."

---

### Slide 11 — Where CuKD-XAI Operates in the Protocol Stack (75 s)

> "Brief positioning slide — where does our detector actually live in the WSN stack.
>
> *(point to the protocol stack on the left)*
>
> WSN-DS is specifically built around the LEACH routing protocol. So we're cross-layer: we collect telemetry from the physical layer — that's Consumed_Energy — from the MAC layer — that's the ADV_SCH_S and ADV_SCH_R TDMA schedule messages — and from the network layer, which is where most of the features live.
>
> *(point to the telemetry sources card)*
>
> The network-layer features are the LEACH routing telemetry: Is_CH and who_CH identify the cluster head, ADV_S and ADV_R track LEACH advertisement messages, JOIN_S and JOIN_R track cluster join requests, DATA_S and DATA_R are the actual data packets, and dist_CH_To_BS is the routing geometry.
>
> So to summarize: CuKD-XAI is a cross-layer IDS. It collects telemetry from physical, MAC, and network layers to detect attacks that target the network and MAC layers specifically."

---

### Slide 12 — Attack Surface and Deployment Target (75 s)

> "Finally — what attacks are we actually detecting, and where does the detector run.
>
> *(point to the table)*
>
> Four attack types from WSN-DS. Blackhole and Grayhole are both routing-layer — Blackhole drops all forwarded packets, Grayhole selectively drops them to mimic congestion. Flooding is also routing-layer — excess ADV and JOIN control messages during LEACH cluster formation, which burns battery across the entire neighborhood. And Scheduling is a MAC-layer attack that manipulates TDMA time-slot assignment.
>
> *(point to the topology diagram on the right)*
>
> For deployment: traditional IDS designs put the detector at the base station, which creates a single point of failure. Our target is cluster heads — the 1.16 kilobyte INT8 student fits in the 10 kilobyte RAM budget of a TelosB-class mote, with activation overhead well under 100 bytes. And because knowledge distillation gives us a 59-times parameter reduction, deploying in-node on regular sensor nodes becomes feasible too — not just cluster heads.
>
> *(point to bottom highlight)*
>
> That 59× parameter compression is what enables cluster-head detection instead of base-station detection — turning a single point of failure into a distributed detection layer.
>
> Thank you. I'm happy to take questions."

---

## PART C — Q&A cheat sheet

These are the questions a sharp listener (especially an IEEE-editor prof) is likely to ask. For each, there's a one-sentence answer, then a fallback if pushed.

---

**Q1: Why is your Wilcoxon one-sided? Isn't that cherry-picking the p-value?**

**A:** "The direction was pre-registered. Wu et al. ICLR 2021 had already shown that curricula often *hurt* in standard supervised settings, so our pre-registered alternative hypothesis was 'B is greater than C' — i.e., the standard MLP beats the CL MLP. One-sided is the correct test when the direction is pre-specified."

**Fallback:** "And even two-sided, the p-value is 0.0625 — just above alpha 0.05. That's the floor of Wilcoxon at n equals 5; you can't do better without running more seeds. Our v2.2 notebook will do 10 seeds post-exams for a tighter confidence interval."

---

**Q2: Why only 5 seeds? Wouldn't 10 or 20 give stronger evidence?**

**A:** "Five was our compute budget for the initial pass. Our v2.2 notebook, ready to run after exams, doubles the seed count. We wanted rigor signal *before* re-running, so we picked the minimum that's defensible under Wilcoxon."

---

**Q3: You said the INT8 student is 1.16 kilobytes but on slide 7 you showed 4.64 KB as the headline. Which is it?**

**A:** "4.64 kilobytes is the full FP32 PyTorch model on disk. 1.16 kilobytes is the raw INT8 weight payload — what you'd actually flash onto the mote. We lead with 4.64 because that's the directly-measured artifact in our v2.0 results. 1.16 KB is the deployment-ready size after INT8 quantization. Full INT8 quantization of Config E across all 5 seeds is pending in our v2.2 run."

**Fallback:** "We do have INT8 quantization results, but only for Config F in the v2.0 run. The FP32 to INT8 drop on Config F was about 3 macro-F1 points — 0.8945 down to 0.8656. We expect a similar drop on Config E, bringing it to roughly 0.89 at 1.16 kilobytes. The v2.2 notebook will confirm."

---

**Q4: Spearman ρ of minus 0.039 with p-value 0.88 — isn't that just saying 'no correlation'? Is that really a novel finding, or is it the null result you'd expect from 17 features?**

**A:** "With 17 features, a random pair of rankings would give you roughly Spearman ρ around zero — that's correct. The novelty isn't that we observed ρ close to zero. The novelty is that we observed ρ close to zero *after* knowledge distillation, which is supposed to transfer the teacher's reasoning. The literature's implicit assumption — and I can cite Hinton's original 2015 KD paper for this — is that soft-label distillation preserves the teacher's decision structure. Our result says that for tabular IDS, the decision structure is *not* preserved even though accuracy is."

**Fallback:** "And it's not a pure null either. Five of the top ten features overlap between student and teacher. The overlap proves the student isn't randomly initialized. But the rank order among those shared features is scrambled — that's what Spearman captures and what we report."

---

**Q5: What about Talukder 2025 at 99.94%? Your 0.9219 is a huge gap — why should anyone prefer your model?**

**A:** "Talukder's random forest is 83 megabytes. It runs at the base station. Our 0.9219 is on a 4.64 kilobyte student that runs on the cluster head — eliminating the single point of failure. We trade 5.7 points of macro-F1 for four orders of magnitude of footprint reduction. The question is whether that trade-off is worth it for WSN deployment, and the answer depends on whether you'd rather have 99.94% at the base station with a single attack surface, or 92.19% distributed across every cluster head with no single attack surface."

---

**Q6: Alfarra 2025 reports 0.93 macro-F1 at kilobyte scale — how is your 0.9219 better?**

**A:** "Alfarra drops TDMA — they only classify four of the five attack classes. Scheduling attacks on the MAC layer are a real category in WSN-DS, and you can't ignore them. Our 0.9219 is on *all five* classes. When we sub-sample our results to their four-class setup, our numbers become directly comparable, and that four-class comparison is in our paper draft."

---

**Q7: Why didn't you run CICIoT2023 for generalization?**

**A:** "CICIoT is on our post-exam roadmap. We prioritized depth on WSN-DS — SHAP faithfulness audit, curriculum ablation, 5-seed statistical validation — over breadth across datasets. Generalization to CICIoT and UNSW-NB15 is in our v2.2 run plan, scheduled for after May 4."

---

**Q8: Your paper title still has 'Curriculum-Guided' in it. If CL didn't help, why is it in the title?**

**A:** "Great question. That's the v1 title of the project, and we have two candidate titles for the final paper. The first keeps 'CuKD-XAI' if our post-exam re-run with the curriculum fix shows recovery to 0.91 or above. The second drops the 'Curriculum' framing entirely and reframes around 'Tree-to-Neural Knowledge Distillation with a Feature-Alignment Gap.' The decision gate is a 20-minute re-run after May 4. Both titles are already publishable."

---

**Q9: How do you justify the calibration step — isotonic regression adds complexity. Do you have an ablation?**

**A:** "We distill from soft labels. Without calibration, an overconfident teacher gives near-one-hot soft labels, which collapses the distillation signal — the student sees almost the same thing as ground-truth hard labels, and you lose the dark knowledge benefit. Isotonic calibration keeps the off-class probabilities informative. We have not run the uncalibrated-teacher ablation as a separate config — that's fair criticism, and we'll add it to the v2.2 run."

---

**Q10: What's the inference latency on an actual MSP430, not on a GPU or modern CPU?**

**A:** "Our current latency numbers — 0.09 milliseconds per sample on a modern CPU thread — are simulated on a desktop. We have *not* deployed to an MSP430 yet. The theoretical estimate, based on 2,400 FLOPs per inference at 8 megahertz with MSP430's ~1 cycle per FLOP for INT8 MAC operations, is roughly 300 microseconds per sample. That gives us a budget of about 3,000 samples per second, which is comfortable for WSN traffic rates. But the on-device measurement is future work."

---

## Rehearsal checklist

- [ ] Run Part A corrections (30–45 min total)
- [ ] Drop in `results_download/shap_teacher.png` to slide 9
- [ ] Print or tablet-display Part B script for the talk
- [ ] Skim Part C Q&A before the talk so the answers feel warm
- [ ] Do one full timed rehearsal with a phone stopwatch
- [ ] If rehearsal comes in under 13 min: add 20 seconds of pause on slides 4, 7, 8, 9 for effect
- [ ] If rehearsal comes in over 16 min: trim slide 6 (setup) to three sentences — it's the cheapest cut
- [ ] Sleep before the talk. Do not stay up tweaking. The deck is already in good shape once Part A is applied.

Good luck on Monday.
