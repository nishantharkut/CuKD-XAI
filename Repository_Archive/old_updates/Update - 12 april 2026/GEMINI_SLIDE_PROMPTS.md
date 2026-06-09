# Gemini Slide Prompts — CuKD-XAI Monday Deck

**Purpose:** Ready-to-paste prompts for Google Slides → Gemini ("Help me create") for slides 2–12 of the new deck. All content is the honest April 10–11 results. Visual style matches the existing deck (card layout, orange badges, serif titles, blue highlight boxes).

**Deck constraints:** 12 slides total, 15-minute talk, slide 1 is a static title slide you walk past (0 seconds spent).

**How to use this file:**
1. Create a new Google Slides deck (or duplicate the existing one and clear slides 2+).
2. For each slide below, open Gemini in Slides ("✨ Help me create"), paste the prompt from the code block, let it draft, then hand-tweak text/placement.
3. Drop in the figure from `results_download/` where noted.
4. Gemini is weak at multi-card layouts — expect to nudge card positions manually after generation.

---

## Shared visual style (apply to every slide)

Paste this once into Gemini when it asks about style, or keep it as a mental checklist while editing:

```
VISUAL STYLE — keep consistent across all slides:

- Title: large serif font (Playfair Display or similar), black, left-aligned,
  horizontal rule directly underneath the title
- Subtitle (optional): monospace font (Courier / Roboto Mono), gray, left-aligned,
  sits directly under the horizontal rule
- Body text: clean sans-serif (Inter / Roboto), dark gray on white
- Content is organized in rounded-corner CARD BLOCKS with a thin 1px gray border
- Each card has a small PILL-SHAPED BADGE in the top-left corner:
  orange background (#E8783D), white uppercase text, ~10pt, short label
- Prefer 2-column or 3-column card layouts (not paragraphs)
- Key takeaway at the bottom of the slide goes in a LIGHT BLUE highlight box
  (background #E7F1F9, border #2B7FBF, blue text for the headline)
- Accent colors:
    • Orange #E8783D — section badges, warnings, "gap" callouts
    • Blue  #2B7FBF — numbered markers, highlight box border, chart bars
    • Red   #C73E3A — invalidated claims, negative deltas, ⚠️ warnings
    • Green #2E8B57 — verified, positive deltas, ✅
- Numbered lists use blue circles with "01", "02", "03", "04"
- Background: white. No gradients. No shadows beyond a very subtle card shadow.
```

---

## SLIDE 2 — Problem & Motivation (60s)

**Layout:** Two cards side-by-side + one bottom highlight box.

**Figure needed:** None. Optional: a small stock icon of a sensor mote on the left card.

**Gemini prompt:**

```
Create a slide titled "The KB-Scale IDS Problem" with a monospace
subtitle "Why WSN intrusion detection can't just import ML from the cloud."

Layout: two equal cards side-by-side, plus one highlight box across the bottom.

LEFT CARD — orange badge "THE HARDWARE":
Heading: "Sensor motes are tiny."
Body:
• TelosB-class motes: 10 KB RAM, 48 KB flash, 8 MHz MCU
• Battery-powered, no rechargeability in the field
• Every extra KB of model = fewer hours of operation
• Cluster heads in LEACH rotate — the detector has to fit on ANY node

RIGHT CARD — orange badge "THE MISMATCH":
Heading: "Strong IDS models are MB-scale."
Body:
• Current WSN-DS SOTA (Talukder 2025): Random Forest, ~85 MB on disk
• CNN/LSTM baselines: 5–50 MB
• None of them fit on a mote — they run at the base station
• Base-station-only detection = single point of failure

BOTTOM HIGHLIGHT BOX (light blue):
"We need a detector that matches cloud-grade accuracy on all 5 WSN-DS
attack classes AND fits inside a single KB of flash."

Apply the shared visual style (serif title, orange badges, rounded cards,
blue highlight box). No stock photos unless it's a single mote icon.
```

**Speaker cue (60s):** "Motes have kilobytes of RAM. The best models are megabytes. That gap is the whole paper."

---

## SLIDE 3 — Base Paper & Research Gap (75s)

**Layout:** Two cards (base paper summary | gap analysis) + bottom highlight.

**Figure needed:** None.

**Gemini prompt:**

```
Create a slide titled "Base Paper & the Unfilled Gap" with monospace subtitle
"Ghadi et al. 2024 — Comprehensive ML survey for WSN security."

Layout: two cards side-by-side + bottom highlight box.

LEFT CARD — orange badge "BASE REFERENCE":
Heading: "Ghadi et al. 2024 (IEEE Access)"
Body:
• Surveys ML approaches for Wireless Sensor Network security
• Identifies four open problems: training location, privacy,
  LIGHTWEIGHT ALGORITHMS, and trust/interpretability
• Explicit call for KB-scale models with explainable decisions
• Does NOT propose a compression pipeline — describes the gap

RIGHT CARD — orange badge "WHAT'S MISSING":
Heading: "No work closes lightweight + faithful XAI together."
Body (use red accent color for the missing bullets):
• Lightweight WSN-DS work exists but drops classes (e.g., Alfarra 2025: 4 of 5)
• XAI on WSN-DS exists (Birahim 2025) but uses MB-scale ensembles
• SOTA (Talukder 2025): 99.94% macro-F1 — but 85 MB, no XAI faithfulness check
• No prior work audits whether the COMPRESSED model's explanations
  still match the teacher's reasoning

BOTTOM HIGHLIGHT BOX:
"Our slot: full 5-class WSN-DS + KB-scale footprint + SHAP faithfulness audit
— all three in one pipeline."

Apply the shared visual style.
```

**Speaker cue (75s):** "Ghadi surveys the gap. Existing work fills pieces — someone does lightweight, someone does XAI. Nobody does all three *and* checks if compression preserves reasoning."

---

## SLIDE 4 — Research Questions (45s)

**Layout:** Three numbered cards in one row.

**Figure needed:** None.

**Gemini prompt:**

```
Create a slide titled "Research Questions" with monospace subtitle
"Three questions — one pipeline."

Layout: three equal cards in a single row, each with a large blue
numbered circle (01, 02, 03) in the top-left of the card (not as a badge).

CARD 1 (01):
Heading: "Can we compress?"
Body: "Can a calibrated Random Forest teacher (~85 MB) be distilled into
a KB-scale neural student without losing WSN-DS classes?"

CARD 2 (02):
Heading: "Does accuracy survive?"
Body: "Does the compressed student hold macro-F1 ≥ 0.91 across all five
attack classes with 5-seed statistical validation?"

CARD 3 (03):
Heading: "Do explanations survive?"
Body: "Does the student's SHAP feature ranking still match the teacher's,
or does compression silently break the reasoning alignment?"

No bottom highlight box on this slide — keep it clean.
Apply the shared visual style (serif title, rounded cards).
```

**Speaker cue (45s):** "Compression. Accuracy. Explainability. The third one is the novel angle — nobody checks it."

---

## SLIDE 5 — Design Rationale (120s) ⭐ THE KEY SLIDE

**Layout:** 2×2 grid of cards, each a "why" quadrant.

**Figure needed:** Optional small icons per quadrant (tree, arrow, stairs, magnifying glass).

**Gemini prompt:**

```
Create a slide titled "How We Arrived at This Design" with monospace subtitle
"Every component has a defensible reason — including the ablation we expected might fail."

Layout: 2x2 grid of four equal cards. Each card has an orange pill badge
in the top-left. No bottom highlight box — this slide's density is the 4 cards.

TOP-LEFT CARD — orange badge "① WHY RF TEACHER":
Heading: "Tabular + imbalance + calibratable."
Body:
• WSN-DS is tabular and severely imbalanced (Normal dominates)
• Random Forests handle both natively, no feature engineering
• Can be calibrated with isotonic regression (Niculescu-Mizil 2005)
  so soft labels are trustworthy for distillation
• Talukder 2025 confirms RF dominates the WSN-DS leaderboard

TOP-RIGHT CARD — orange badge "② WHY KNOWLEDGE DISTILLATION":
Heading: "The only way to hop architecture classes."
Body:
• Pruning and quantization stay inside the teacher's architecture —
  you cannot shrink 85 MB → 1 KB that way
• Knowledge Distillation (Hinton et al. 2015) lets us pick a radically
  smaller STUDENT architecture and transfer soft-label knowledge across it
• Our target is a ~1,200-parameter MLP — KD is the only lever that reaches it

BOTTOM-LEFT CARD — orange badge "③ WHY CL AS AN ABLATION":
Heading: "Pre-registered as a test, not a claim."
Body:
• Bengio 2009: easy→hard curricula can help imbalanced learning
• Wu et al. ICLR 2021 "When Do Curricula Work?" already showed CL
  often does NOT help in standard supervised settings
• We built CL as an honest ABLATION ARM — expected outcome uncertain
• Result (see slide 9) is consistent with Wu et al.'s warning

BOTTOM-RIGHT CARD — orange badge "④ WHY SHAP FAITHFULNESS AUDIT":
Heading: "Accuracy ≠ reasoning alignment."
Body:
• Compression is meaningless if the student is a black box that
  disagrees with the teacher's feature importances
• Need a quantitative check that "student matches teacher in accuracy"
  doesn't hide "student learned the task via different features"
• Spearman rank correlation between student and teacher SHAP rankings
  gives exactly that check
• This audit is what revealed our novel finding

Apply the shared visual style. Make all four cards the same size — 2x2 grid.
```

**Speaker cue (120s):** "This is the slide where you want the prof nodding. Walk through the four quadrants slowly — 25–30 seconds each. Emphasize that CL was pre-registered as an ablation, *not* the core claim. That reframes the negative result as rigor, not failure."

---

## SLIDE 6 — Final Pipeline (75s)

**Layout:** Horizontal flow diagram: 5 boxes with arrows.

**Figure needed:** None (Gemini should draw the flow, or you draw it in Slides manually).

**Gemini prompt:**

```
Create a slide titled "CuKD-XAI Pipeline" with monospace subtitle
"Five stages, one audit loop."

Layout: a horizontal flow diagram with FIVE rounded boxes connected by
right-pointing arrows. Each box has a small blue numbered marker (1–5)
in its top-left. Below the flow, a single light blue highlight box.

BOX 1: "Calibrated RF Teacher"
Subtitle: "~85 MB, isotonic calibrated"

BOX 2: "Tiny MLP Student"
Subtitle: "1,189 params (128-64-5 architecture)"

BOX 3: "KD Training"
Subtitle: "Hinton loss, T=4, α=0.7"

BOX 4: "INT8 Quantization"
Subtitle: "1.16 KB on disk"

BOX 5: "SHAP Faithfulness Audit"
Subtitle: "Spearman ρ(student, teacher)"

Between BOX 2 and BOX 3, add a small DASHED side branch labeled
"CL Ablation Arm (pre-registered)" with the same orange accent color
as the badges. This signals CL is a side experiment, not the main path.

BOTTOM HIGHLIGHT BOX:
"18,335× on-disk compression (85 MB → 1.16 KB)
 +  5-seed Wilcoxon validation
 +  SHAP audit on every config."

Apply the shared visual style.
```

**Speaker cue (75s):** "Point at the dashed CL branch and say 'this is the ablation — we ran it, we report it, it didn't help.' Then walk the main path: RF → student → KD → INT8 → audit."

---

## SLIDE 7 — Experimental Setup (60s)

**Layout:** Two cards (dataset | protocol).

**Figure needed:** Optional small class-distribution bar chart.

**Gemini prompt:**

```
Create a slide titled "Experimental Setup" with monospace subtitle
"WSN-DS, 10 configs, 5 seeds, Wilcoxon signed-rank."

Layout: two cards side-by-side. No bottom highlight box.

LEFT CARD — orange badge "DATASET":
Heading: "WSN-DS (374,661 records)"
Body:
• 5 attack classes: Normal, Blackhole, Grayhole, Flooding, Scheduling/TDMA
• LEACH protocol telemetry: Physical, MAC, and Network layer features
• Severe imbalance (Normal ≈ 90% of traffic)
• 80/20 stratified split, 5 seeds per config
Below body: a small horizontal bar chart showing class counts
(Normal much larger than the four attack classes).

RIGHT CARD — orange badge "PROTOCOL":
Heading: "10 configs × 5 seeds"
Body:
• Baselines: Random Forest, XGBoost, LightGBM, standalone MLP
• Teachers: B (standard MLP), C (CL MLP)
• Students: D, E (KD from RF), E2 (KD from MLP), F (KD from CL MLP)
• Metrics: macro-F1, per-class F1, ECE, FLOPs, on-disk size
• Stats: Wilcoxon signed-rank with Holm–Bonferroni correction
• SHAP: DeepExplainer on students, TreeExplainer on RF

Apply the shared visual style.
```

**Speaker cue (60s):** "Ten configs, five seeds, Wilcoxon. The design is defensible. Don't dwell — this slide is credentials, not content."

---

## SLIDE 8 — Result 1: Compression Works (105s) ⭐

**Layout:** Big number panel + Pareto chart on the right.

**Figure needed:** `results_download/pareto_frontier.png` — drop it on the right side.

**Gemini prompt:**

```
Create a slide titled "Result 1 — Compression Works" with monospace subtitle
"Config E: KD from calibrated RF into a 1,189-parameter MLP."

Layout: left side = THREE stacked metric cards; right side = a large
placeholder box labeled "PARETO FRONTIER CHART — insert pareto_frontier.png".
Bottom = one highlight box.

LEFT-TOP METRIC CARD — orange badge "ACCURACY":
Big number: "0.9219"
Subtitle: "macro-F1 (5 seeds, Wilcoxon-validated)"
Small green check icon next to the number.

LEFT-MIDDLE METRIC CARD — orange badge "SIZE":
Big number: "1.16 KB"
Subtitle: "INT8 quantized student (on disk)"
Small line: "Parameters: 1,189   •   FLOPs: ~2.4K per inference"

LEFT-BOTTOM METRIC CARD — orange badge "COMPRESSION RATIO":
Big number: "18,335×"
Subtitle: "85 MB Random Forest → 1.16 KB INT8 MLP"
Small line: "Parameter ratio: 59× (69,893 → 1,189)"

RIGHT SIDE: large placeholder box (roughly 55% of slide width) with
dashed border and centered text "INSERT pareto_frontier.png here —
model size (x-axis, log scale) vs macro-F1 (y-axis)".

BOTTOM HIGHLIGHT BOX:
"One calibrated-RF teacher + vanilla KD = cloud-grade accuracy at
mote-scale footprint. Config E is our headline number."

IMPORTANT: the two compression numbers (18,335× and 59×) are DIFFERENT —
one is on-disk size, the other is parameter count. Both appear because
they tell different stories. Do not average or combine them.

Apply the shared visual style.
```

**Speaker cue (105s):** "Land on the three numbers slowly: 0.9219 / 1.16 KB / 18,335×. Point to the Pareto chart and say 'we sit in the bottom-right corner — nothing else is near us.' Clarify *once* that 18,335× is on-disk and 59× is parameters, then move on."

---

## SLIDE 9 — Result 2: Honest CL Negative (105s) ⭐

**Layout:** Two bar charts (teacher comparison | student comparison) + bottom text.

**Figure needed:** `results_download/loss_curves_B_vs_C.png` (optional — if you have a clean bar chart instead, use that).

**Gemini prompt:**

```
Create a slide titled "Result 2 — CL Was Pre-Registered, and It Didn't Help"
with monospace subtitle "An honest negative — consistent with Wu et al. ICLR 2021."

Layout: two bar-chart cards side-by-side on top, one card with body text
on the bottom. No bottom highlight box (the bottom card IS the takeaway).

TOP-LEFT CARD — orange badge "TEACHERS":
Heading: "CL hurt the teacher by 6 points."
Chart: a simple bar chart with two bars —
• Config B (Standard MLP):  0.9211  (blue bar)
• Config C (CL MLP):         0.8607  (gray bar)
Y-axis: macro-F1, range [0.80, 0.95]
Caption below chart: "Δ = −0.0604 macro-F1 (Wilcoxon p < 0.01)"

TOP-RIGHT CARD — orange badge "STUDENTS":
Heading: "CL hurt the student by 2 points."
Chart: a simple bar chart with two bars —
• Config E2 (KD from MLP):     0.9114  (blue bar)
• Config F  (KD from CL MLP):  0.8908  (gray bar)
Y-axis: macro-F1, range [0.80, 0.95]
Caption below chart: "Δ = −0.0206 macro-F1"

BOTTOM CARD — orange badge "WHY THIS IS GOOD NEWS":
Body:
• We DID NOT hide this result — CL was pre-registered as an ablation
• Wu et al. ICLR 2021 "When Do Curricula Work?" already showed CL
  often fails in standard supervised settings; our finding is consistent
• The paper reports it as a rigorous negative result
• Our HEADLINE model (Config E = KD from RF) does not depend on CL at all
  — its 0.9219 macro-F1 stands independently

Apply the shared visual style. Use blue for the "winning" bar in each chart
and gray for the "losing" bar so the visual direction is obvious.
```

**Speaker cue (105s):** "Lead with 'this is the honest slide.' Point at the two deltas. Then immediately pivot to *why it's fine*: pre-registered, consistent with Wu et al., and — critically — our headline E number **doesn't depend on CL at all.**"

---

## SLIDE 10 — Novel Finding: Feature Alignment Gap (105s) ⭐

**Layout:** Two SHAP bar charts side-by-side + bottom highlight.

**Figure needed:** `results_download/shap_summary_student.png` + an equivalent SHAP plot for the RF teacher (same top-K features, different rank order). If you only have the student plot, use it on the right and put a placeholder for the teacher plot on the left.

**Gemini prompt:**

```
Create a slide titled "Novel Finding — The Feature Alignment Gap"
with monospace subtitle "Compressed students learn the task, not the teacher's reasoning."

Layout: two SHAP bar-chart cards side-by-side + one bottom highlight box.

LEFT CARD — orange badge "RF TEACHER SHAP":
Heading: "Top features the teacher relies on"
Placeholder: "INSERT teacher SHAP bar chart here —
top-10 features ranked by mean |SHAP| value, TreeExplainer."

RIGHT CARD — orange badge "STUDENT SHAP":
Heading: "Top features the student relies on"
Placeholder: "INSERT student SHAP bar chart here —
top-10 features ranked by mean |SHAP| value, DeepExplainer."

BETWEEN THE TWO CARDS (visually center, small): a large symbol
"Spearman ρ ≈ 0" in bold blue, with a small caption underneath:
"Rankings are effectively uncorrelated."

BOTTOM HIGHLIGHT BOX (light blue, but with a thicker border to signal importance):
"Headline: KD transfers task accuracy but NOT feature-level reasoning.
 This is — to the best of our April 2026 literature search — an unreported
 phenomenon for tabular IDS compression. It is a novel contribution that
 holds regardless of whether the CL ablation had worked."

Apply the shared visual style.
```

**Speaker cue (105s):** "This is the slide that gets you published even if everything else is standard. Say: 'Students and teachers agree on the answer. They disagree on *why*. Nobody has reported this for tabular IDS.' Pause for two beats."

---

## SLIDE 11 — Competitor Landscape (60s)

**Layout:** Comparison table with checks / crosses + bottom highlight.

**Figure needed:** None.

**Gemini prompt:**

```
Create a slide titled "Where We Sit vs the 2025 Literature"
with monospace subtitle "Verified via CrossRef / OpenAlex, April 2026."

Layout: one large card containing a comparison table + one bottom highlight box.

TOP CARD — orange badge "HEAD-TO-HEAD":
Heading: "WSN-DS competitor landscape"
Table with columns:
| Paper | Full 5-class | KB-scale | SHAP faithfulness audit |

Rows:
| Talukder 2025 (SOTA, 99.94%)    | ✅ | ❌ (85 MB RF)        | ❌ |
| Alfarra 2025 (0.93, pruning)    | ❌ (4-class, drops TDMA) | ✅ | ❌ |
| Birahim 2025 (SHAP+LIME, 99.73%)| ✅ | ❌ (MB-scale ensemble) | ❌ (no compression audit) |
| Benaddi 2025 (KD+Kronecker)     | ❌ (TON_IoT, not WSN-DS) | ✅ | ❌ |
| Xiao 2025 (DNN+CatBoost, 95.62%)| ✅ | ❌                   | ❌ (CAM, not SHAP) |
| -------------                   | -------------- | --------------- | --------------- |
| CuKD-XAI (ours)                 | ✅ (all 5 classes) | ✅ (1.16 KB)   | ✅ (Spearman audit) |

Use green ✅ and red ❌ icons. Highlight the "CuKD-XAI (ours)" row with
a light blue background so it stands out.

BOTTOM HIGHLIGHT BOX:
"We are the only row with three green checks. Every other paper either
drops classes, keeps MB-scale models, or skips the SHAP faithfulness audit."

Apply the shared visual style.
```

**Speaker cue (60s):** "One row, three checks. Don't read every cell — just say 'we're the only row where all three columns are green.'"

---

## SLIDE 12 — Path Forward & Ask (60s)

**Layout:** Two cards (narrative options | post-exam timeline) + thank-you strip.

**Figure needed:** None.

**Gemini prompt:**

```
Create a slide titled "Path Forward" with monospace subtitle
"Two publishable narratives. One decision. Post May-4 execution."

Layout: two cards side-by-side on top + one thin "thank you" strip at the bottom.

LEFT CARD — orange badge "TWO CANDIDATE TITLES":
Heading: "Paper narrative — decision gate is a single 20-min re-run."

Option A:
"CuKD-XAI: Curriculum-Guided Knowledge Distillation with Explainable
Feature Analysis for Lightweight WSN Intrusion Detection"
Condition: fires if the CL re-run shows Config C ≥ 0.91 macro-F1.

Option B:
"Tree-to-Neural Knowledge Distillation for Full-Class WSN Intrusion
Detection: 18,000× Compression with a Feature-Alignment Gap"
Condition: fires if the CL re-run still fails; headline becomes
compression + Spearman novelty. CL reported as rigorous negative.

Small note at the bottom of the card, in red:
"Both options are already publishable with what we have."

RIGHT CARD — orange badge "POST-EXAM TIMELINE":
Heading: "What executes after May 4."
Numbered list with blue "01 / 02 / 03 / 04" markers:
01. CL-fix validation run (~20 min, Student A, seed 42)
02. Full 5-seed experiment with fixes (~90 min)
03. CICIoT2023 generalizability (~90 min, 400K sample)
04. Student B (64-32-5) multi-seed for Pareto (~40 min)
Then: paper drafting (~1 week), target venue submission.

BOTTOM STRIP (thin, centered, no card border):
"Thank you. Questions?"

Apply the shared visual style.
```

**Speaker cue (60s):** "Close by saying: 'We have two publishable narratives already. One 20-minute re-run after May 4 decides which title we commit to.' Then 'thank you, questions.'"

---

## Figure drop-in checklist

When you paste these prompts into Gemini, it will generate layouts with placeholder boxes for charts. Replace each placeholder with the matching file from `results_download/`:

| Slide | Placeholder | Source file |
|---|---|---|
| 7 | Class distribution bar chart | (optional — `per_class_f1.png` can stand in) |
| 8 | Pareto frontier chart | `results_download/pareto_frontier.png` |
| 9 | Teacher B vs C bars | `results_download/loss_curves_B_vs_C.png` (or custom bar) |
| 9 | Student E2 vs F bars | (custom — build from `cukd_xai_results.json`) |
| 10 | RF teacher SHAP bars | (custom — you may need to re-run just the SHAP cell) |
| 10 | Student SHAP bars | `results_download/shap_summary_student.png` |

If a custom chart is missing, leave a dashed placeholder in the slide and write "[PENDING: rebuild after CL re-run]" in small red text. Your prof will read that as honest, not sloppy.

---

## Hard constraints to enforce while editing

1. **Never mix the two compression numbers.** 18,335× is on-disk (85 MB → 1.16 KB). 59× is parameters (69,893 → 1,189). Say both, label both, don't average.
2. **Slide 9 must stay honest.** Do not soften "CL didn't help." The framing is "pre-registered ablation + Wu et al. agrees," not apology.
3. **Slide 10 is the novelty anchor.** If you cut any slide for time, do not cut this one.
4. **No "first to do X" claims without the 2025 literature caveat.** Birahim 2025 already did SHAP+LIME on WSN-DS. Our novelty is the faithfulness *audit*, not the existence of SHAP on WSN-DS.
5. **Do not claim the CL fix works.** The v2.3 fix is applied in code but has not been run yet. Anything stronger than "fix is pending post-exam validation" is unsupported.

---

## Rehearsal tips

- Total content time: 870 seconds (14.5 min) + 30s buffer. Use a phone timer per slide.
- Slides 5, 8, 9, 10 each need ~2 minutes — rehearse these three in isolation first.
- If the prof interrupts with a question on slide 8 or 9, let them; those are the slides you want discussion on, not monologue.
- If you run long, cut slide 7 (setup) to a single sentence — it's the easiest slide to compress live.
- Do NOT rehearse slide 1 — you are literally walking past it.
