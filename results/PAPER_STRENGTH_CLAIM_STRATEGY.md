# Paper strength strategy (honest, tier-1 framed)

**Not a finished manuscript.** Research claim architecture for a stronger paper.

---

## 1. Settle Claude’s two “unresolved” items (with source checks)

### 1.1 FG Student A mean 0.91411 — **not a broken aggregator**

| Seed | A RF-KD macro-F1 (CSV recompute = seed_completion) |
|-----:|----------------------------------------------------:|
| 42 | 0.916076 |
| 123 | 0.921058 |
| 456 | 0.910198 |
| 789 | 0.904245 |
| 1001 | 0.918986 |
| **mean** | **0.914112** |

`aggregate_results.json` mean equals `numpy.mean(values)` exactly.  
The list that averages to ~0.91618 is **not in the FG package**.  
Test size under FG = **56,301** (not 56,200).

→ No silent poison in FG aggregation. Keep 0.9141 as the correct FG A RF-KD mean.

### 1.2 Seed-42 deployment 0.9485 — **explained, not an evaluation bug**

| Unit | A RF-KD F1 | Role |
|------|------------|------|
| main_10seed **checkpoint** seed 42 | **0.9249** (z≈+0.69) | Member of multi-seed distribution |
| **deployment** seed 42 | **0.9485** (z≈+4.3 vs 10-seed mean) | Clean `set_seed(42)` + RF-KD only |
| 10-seed mean | 0.9203 ± 0.0065 | Statistical primary for random-row train-only |

Same test n=56,200; same soft targets (byte-identical).  
Difference = **training path / RNG**: multi-config pipeline vs single-route deployment.

→ Hardware on deployment weights is valid for **deployment route**.  
→ Never claim deployment 0.9485 is “the seed-42 multi-seed point.”

### 1.3 Fixed-point gates

Not a measurement bug. Deployment PTQ drop ≈ **0.024 / 0.027**.  
Strict 0.01 fails; copy 0.03 is operational.  
Paper must either improve PTQ or own a **measured conversion-loss** claim.

---

## 2. What held vs what must change in the story

### Held (compression / systems spine)

- RF teacher ~0.979; Full MLP ~0.923 — stable across scaler correction.
- Ultra-small students still ~0.91–0.94 macro-F1 in ~4.6–13 KB FP32.
- MCU vs fixed reference agreement **1.0** on full 56,200 (both lineages).
- ONNX/OpenVINO FP32 agreement 1.0 on train-only deploy graphs.
- Student B weak KD benefit already foreshadowed by archived Wilcoxon.

### Actually weakened (must rewrite, not caveat)

| Claim | Archived | Train-only 10-seed | FG 5-seed (group-disjoint) |
|-------|----------|--------------------|----------------------------|
| A RF-KD − scratch (macro-F1) | +0.0077 (mean of means) | **+0.0094** paired; t=2.29, p≈0.048; Wilcoxon p≈0.049 | **+0.0002**; t=0.07, p≈0.95; Wilcoxon p=1.0 |
| B RF-KD − scratch | +0.0006 means | +0.0078 paired; t=1.85, p≈0.097; n.s. | **−0.0017**; n.s. |

**Interpretation:** RF-KD’s Student A edge is **real under random-row** (including train-only scaler) but **vanishes under exact-feature-group disjoint splits**. That is a **leakage-sensitivity finding**, not a null paper.

### Per-class costs (do not hide behind macro-F1)

**Archived B:** RF-KD already trades Blackhole (0.873 vs scratch 0.891) and Grayhole (0.889 vs 0.898) for Flooding/TDMA gains.

**FG B (clearer):** RF-KD vs scratch mean per-class Δ  
Blackhole **−0.023**, Grayhole **−0.016**, Flooding **+0.010**, TDMA **+0.020**.

**Train-only 10-seed A:** RF-KD vs scratch  
Blackhole **−0.010**, Grayhole **+0.016**, Flooding **+0.019**, TDMA **+0.021**.

Macro-F1 “flat/up” can hide **minority-class redistribution**. Paper must show per-class tables for deployment claims.

---

## 3. Stronger paper architecture (recommended)

### Core thesis (defensible)

> **Kilobyte-scale students retain high multi-class F1 and exact integer-board fidelity; teacher choice and split construction dominate “KD benefit”; compression and conversion must be audited under leakage-aware protocols.**

Not: “RF-KD is universally the strongest ultra-small route.”

### Research questions (tier-1 clean)

1. **RQ1 Compression:** Can 1.2k–3.4k-param students retain ≥0.91 macro-F1?  
   → Yes across archived + train-only + FG absolute levels.

2. **RQ2 Route effects:** Does RF-KD beat scratch/curriculum under controlled seeds?  
   → Random-row: modest A benefit (borderline at n=10).  
   → Group-disjoint: **no** A benefit.  
   → B: no reliable KD benefit; minority-class tradeoffs.

3. **RQ3 Explanation:** Does student SHAP match RF ranks?  
   → Archived audit: no (keep, narrow scope).  
   → Optional: SHAP on deploy RF-KD or explicitly “not claimed for deployment student.”

4. **RQ4 Numerical deployment:** Does fixed-point MCU match integer reference?  
   → Yes (1.0). Report FP32 gap and **measured** F1 drop.

5. **RQ5 Robustness / protocol sensitivity:** Edge protocols + WSN split construction.  
   → Edge literature has **17% of test rows** in cross-partition feature groups; strict ~4.4%.  
   → WSN FG is the key sensitivity experiment.

### Abstract/conclusion rewrite skeleton

**Drop:** “RF-KD is the strongest ultra-small route” as unqualified headline.

**Replace with something like:**

- Ultra-small RF-distilled students reach ~0.92 macro-F1 (train-only 10-seed) in 4.64 KB.  
- Under random-row splits, A RF-KD edges scratch by ~0.009 macro-F1 (paired p≈0.05).  
- Under exact-feature-group disjoint splits that edge **disappears** (Δ≈0, n.s.).  
- B shows no stable KD gain; minority Blackhole/Grayhole F1 can fall when macro-F1 looks flat.  
- On two MCUs, fixed-point RF-KD matches the integer reference on all 56,200 test vectors; float→fixed incurs ~2.4–2.7% macro-F1 drop on the deployment artifact.  
- Therefore the contribution is a **multi-layer audit** (prediction, leakage, conversion, board), not a single “best KD” slogan.

---

## 4. Evidence objects to freeze for the stronger paper

| Layer | Use as | Numbers |
|-------|--------|---------|
| Train-only 10-seed full aggregate | **Primary multi-seed** | A RF-KD 0.9203±0.0065; B 0.9392±0.0129 |
| Archived 10-seed | Historical / optional appendix | Disclose pre-split scaler |
| FG 5-seed | **Leakage sensitivity** | A KD−scratch +0.0002; B −0.0017 |
| Deployment seed-42 | **Artifact route only** | F1 0.9485/0.9449; not multi-seed seed 42 |
| Checkpoint seed-42 | If citing 10-seed seed 42 | F1 0.9249 |
| HIL four-pair | Deployment artifact fidelity | agree_fixed 1.0; F1 fixed 0.9244/0.9180 |
| ONNX/OV | Conversion fidelity | agree 1.0 FP32 |
| Edge + dup audit | Protocol sensitivity | lit 17% test rows cross-partition |

---

## 5. What makes the paper *stronger* (not weaker)

Leakage killing a KD delta is **a result**, if framed as:

> Prior compact-KD gains on WSN-style tables can be **split-construction sensitive**. We measure that sensitivity and still show **deployable** integer cores with exact board fidelity.

That is more publishable than defending a fragile +0.008 mean under one split.

**Add (software, high value):**

1. One **protocol ladder figure**: absolute F1 + (RF-KD−scratch) across Archived / Train-only RR / FG.  
2. **Per-class Δ heatmap** RF-KD vs scratch (train-only 10 + FG).  
3. Explicit **two-model-identity** box: multi-seed student vs deployment student (RNG).  
4. Fixed-point: report drop + minority drift, not only agreement.  
5. Edge literature: disclose 17% test cross-partition row share; retrain only if claiming SOTA on that protocol.

**Do not spend cycles on:** seed-5678 CL-ext collapse (already known instability).

---

## 6. Hardware when available (only if needed)

| Goal | Need HW? |
|------|----------|
| Keep claims on **deployment** artifact | No re-HIL required (already 1.0) |
| HIL must equal **10-seed checkpoint** seed 42 | Yes: re-export that state dict + HIL |
| Pass strict 0.01 drop | Likely yes after better PTQ |

---

## 7. Honest bottom line

| Claude summary | Our audit |
|----------------|-----------|
| Compression holds | **Agree** |
| A KD benefit dies under group-disjoint | **Agree** (and train-only RR still shows borderline +0.009) |
| B flat macro hides minority cost | **Agree** (already visible archived; stronger under FG) |
| FG aggregate still buggy | **Disagree** — source mean is correct |
| Seed-42 deploy implausible | **Agree as multi-seed member** — explained as **different training unit** |
| Hardware chain not clean | **Clean for deployment identity**; **unclean if conflated with multi-seed seed 42** |

**Stronger paper = multi-protocol honesty + systems fidelity, not a single KD slogan.**
