## Question 1: Sampling strategy for CICIoT2023

### What published papers actually do (verified)
                                                                                                                                                                                                                  
  ┌────────────────────┬────────────┬──────────────────────────────────────────┬──────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────┐    
  │       Paper        │  Sample    │                 Strategy                 │     Taxonomy     │                                                  Source                                                  │    
  │                    │    size    │                                          │                  │                                                                                                          │    
  ├────────────────────┼────────────┼──────────────────────────────────────────┼──────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────┤    
  │ Neto et al. 2023   │ ~48M       │ Proportional, no rebalancing             │ All 3            │ MDPI Sensors 23:5941 (https://www.mdpi.com/1424-8220/23/13/5941)                                         │    
  │ (original)         │ (full)     │                                          │ (binary/8/34)    │                                                                                                          │    
  ├────────────────────┼────────────┼──────────────────────────────────────────┼──────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────┤    
  │ Susilo et al. 2025 │ 4.7M       │ Proportional 10% + SMOTE on minorities   │ 8-class          │ MDPI Sensors 25:580 (https://www.mdpi.com/1424-8220/25/2/580)                                            │    
  ├────────────────────┼────────────┼──────────────────────────────────────────┼──────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────┤    
  │ Alharby 2025       │ 1.21M      │ Balanced 302K/class — but dropped Web,   │ 4-class only     │ Sci Reports (https://www.nature.com/articles/s41598-025-23711-7)                                         │
  │                    │            │ BruteForce, Recon, Spoofing              │                  │                                                                                                          │    
  ├────────────────────┼────────────┼──────────────────────────────────────────┼──────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────┤    
  │ Khanday et al.     │ 661K       │ SMOTE-balanced after subset              │ 34-class         │ IJMEMS                                                                                                   │    
  │ 2024               │            │                                          │                  │ (https://www.ijmems.in/cms/storage/app/public/uploads/volumes/10-IJMEMS-23-0442-9-1-188-204-2024.pdf)    │
  ├────────────────────┼────────────┼──────────────────────────────────────────┼──────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Adewole et al.     │ 21M        │ Near-full but binary only                │ Binary           │ PMC (https://pmc.ncbi.nlm.nih.gov/articles/PMC11945305/)                                                 │
  │ 2025               │ dedup'd    │                                          │                  │                                                                                                          │    
  ├────────────────────┼────────────┼──────────────────────────────────────────┼──────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Houichi et al.     │ Stratified │ Class-weighted loss instead of           │ 8-class          │ IET Smart Cities (https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/smc2.70014)               │    
  │ 2025               │            │ resampling                               │                  │                                                                                                          │    
  └────────────────────┴────────────┴──────────────────────────────────────────┴──────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────┘

### Recommendation: Cap-at-50K per class, floor-to-available ✅

For each of 8 classes:

    sample N = min(50,000, all_available_records_in_class)

### Expected totals:

    Benign     : 50,000 (cap from ~1M)
    DDoS       : 50,000 (cap from ~33M)
    DoS        : 50,000 (cap from ~8M)
    Mirai      : 50,000 (cap from ~2.6M)
    Recon      : 50,000 (cap from ~354K)
    Spoofing   : 50,000 (cap from ~485K)
    Web        : 24,000 (floor — all available)
    BruteForce : 13,000 (floor — all available)

    TOTAL ≈ 287,000 records, 8 classes

### Why this:

- Closest published precedent is Alharby 2025 (equal-per-class) but they cheated by dropping minority classes. We keep all 8 — defensible.
- Avoids SMOTE — synthetic samples distort teacher-student divergence, which is bad for KD experiments specifically (per agent's note)
- Workable on a laptop, replicable across 5 seeds

**Honest caveat:** With BruteForce at 13K, every seed sees the same BruteForce records → seed variance on that class will under-report. State this explicitly in the paper.

**Split:** 70/15/15 stratified, OR 80/20 with k-fold CV inside training. Both have published precedent.

---

## Question 2: CL difficulty ordering for CICIoT2023

This is where the research yielded something really useful — your CL ordering is essentially solved by the published per-class F1 evidence.

### Why your WSN-DS CL failed (now diagnosed)                                                                                                                                                                       
                                                                                                                                                                                                                  
### Why your WSN-DS CL failed (now diagnosed)

Hand-crafted "easy → hard" ordering has weak empirical support (Wu, Dyer & Neyshabur, ICLR 2021 (https://openreview.net/forum?id=tW4QEInpni) — "When Do Curricula Work?"). Your WSN-DS choice (Normal+Blackhole → +Grayhole+Flooding → +Scheduling) was domain intuition, not data-grounded. The Narkedimilli 2025 papers we cite use the same hand-crafted approach without quantitative justification — exactly your failure mode.

### The fix: Transfer-teacher class-level CL

Use the RF teacher's per-class F1 as the difficulty signal. This is the class-level realization of Hacohen & Weinshall, ICML 2019 (https://proceedings.mlr.press/v97/hacohen19a.html) — currently the strongest empirical CL method.

### Difficulty ranking from literature (5+ papers agree)

Aggregated across Neto 2023, Susilo 2025, Fares 2025, Narayan 2023, Houichi 2025:                                                                                                                               
  
  ┌──────┬────────────┬─────────┬────────────┬───────────────────────────────────────────────┐                                                                                                                    
  │ Rank │   Class    │ Mean F1 │ Confidence │                    Reason                     │
  ├──────┼────────────┼─────────┼────────────┼───────────────────────────────────────────────┤                                                                                                                    
  │ 1    │ Mirai      │ 0.999   │ very high  │ Distinct propagation signatures               │
  ├──────┼────────────┼─────────┼────────────┼───────────────────────────────────────────────┤
  │ 2    │ DoS        │ 0.997   │ very high  │ Single-source flood patterns                  │                                                                                                                    
  ├──────┼────────────┼─────────┼────────────┼───────────────────────────────────────────────┤                                                                                                                    
  │ 3    │ DDoS       │ 0.959   │ very high  │ High-volume flood signatures                  │                                                                                                                    
  ├──────┼────────────┼─────────┼────────────┼───────────────────────────────────────────────┤                                                                                                                    
  │ 4    │ Benign     │ 0.93    │ high       │ Bleeds into Recon/Spoofing                    │
  ├──────┼────────────┼─────────┼────────────┼───────────────────────────────────────────────┤                                                                                                                    
  │ 5    │ Recon      │ 0.59    │ high       │ Looks like benign scanning                    │
  ├──────┼────────────┼─────────┼────────────┼───────────────────────────────────────────────┤                                                                                                                    
  │ 6    │ Spoofing   │ 0.56    │ high       │ Resembles benign + recon                      │
  ├──────┼────────────┼─────────┼────────────┼───────────────────────────────────────────────┤                                                                                                                    
  │ 7    │ BruteForce │ 0.35    │ very high  │ Tiny class (13K), 5.6% RF baseline F1         │
  ├──────┼────────────┼─────────┼────────────┼───────────────────────────────────────────────┤                                                                                                                    
  │ 8    │ Web        │ 0.30    │ very high  │ Smallest (24K), heterogeneous (6 sub-attacks) │
  └──────┴────────────┴─────────┴────────────┴───────────────────────────────────────────────┘                                                                                                                    

### Recommended CL stages
### Recommended CL stages

- **Stage 1 (easy):** Mirai, DoS, DDoS [F1 > 0.95 across all baselines]
- **Stage 2 (medium):** + Benign, Recon, Spoofing [F1 ~0.55–0.93]
- **Stage 3 (hard):** + BruteForce, Web [F1 ~0.07–0.35]

### Stage epoch allocation (your WSN-DS fix already planned)

```
CL_STAGES = [(stage1, 5 epochs), (stage2, 5 epochs), (stage3, 30 epochs)]
```

Stage 3 gets the most epochs because it has the hardest classes that need the most training.

### Required ablations (defenses against reviewers)

Per the research, you must include these to satisfy Wu et al. 2021's critique:

1. No-CL baseline (your existing Config E)
2. Random class ordering (control)
3. Anti-curriculum (hardest-first — sometimes wins on long-tail per Wang et al. ICCV 2019 DCL)
4. Hand-crafted intuition ordering (what you did on WSN-DS)
5. Transfer-teacher data-grounded ordering (the proposed method)

If transfer-teacher beats random AND hand-crafted, that's your headline result and it explains the WSN-DS failure.

### Optional add-on: SuperLoss wrapper

Castells, Weinzaepfel & Revaud, NeurIPS 2020 (https://proceedings.neurips.cc/paper/2020/hash/2cfa8f9e50e0f510ede9d12338a5f564-Abstract.html) — drop-in replacement for cross-entropy, adds sample-level robustness against noisy borderline flows (Benign vs slow-rate attacks). One extra hyperparameter (τ). Worth trying as a cheap addition.

---

### Honest uncertainty flags (must be in your paper)

1. No paper says "rank classes by baseline-F1 and curriculum by that" — we're framing this as a class-level reduction of Hacohen & Weinshall transfer-teacher scoring. Defensible, but reviewers may push.
2. Tabular CL evidence is thin overall. Most CL is CV/NLP. Frame this as "extending CL methods originally validated on CV/NLP to tabular IDS."
3. Wu et al. ICLR 2021 says CL helps mainly under noisy labels and limited budgets — both describe IDS, so this is actually a pro, but engage with it explicitly.
4. Mirai punches above its weight (rank 1 despite less data than DDoS/DoS) — interesting finding to highlight; suggests signature-richness > sample size.
5. Susilo 2025 deviation: their hybrid pushed Web to 0.74 F1 while crashing Spoofing to 0.10 — minority oversampling has trade-offs. Don't blindly target high Web F1.

---

### What I'd save to memory if you confirm direction

This is solid enough that we should persist it. Want me to update the project memory with:
- Final sampling strategy decision (cap-50K with floors)
- CL ordering grounded in transfer-teacher + per-class F1
- The 5-paper ranking evidence
- The required ablation list

Or do you want to discuss/adjust before locking it in?