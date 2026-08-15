# CuKD-XAI manuscript positioning (freeze-safe)

> **Historical positioning record:** The authority below predates registry
> `cukd_fgds_evidence_registry_20260814_v3` and must not drive current result
> claims.

**Historical authority:** `results/paper_strength_e2e/06_claim_freeze.json`
**Lit evidence:** `docs/literature/papers/reviews/`  
**Do not use:** marketing claims from the invalidated Claude draft abstract.

---

## 1. One-sentence paper claim (allowed)

> Under **train-only** evaluation, RF knowledge distillation yields ultra-small neural students with competitive macro-F1 on WSN-DS, but the KD benefit is **protocol-sensitive**; deployment-clean students achieve **dual-MCU full-test integer agreement** with a **measured** fixed-point macro-F1 cost, while global SHAP rank agreement with the RF teacher is **low and non-significant**.

If a sentence cannot be traced to C1–C10, cut it.

---

## 2. What this paper is *not* (avoid reviewer traps)

| Trap claim | Why it fails | Who already did something close |
|---|---|---|
| “First KD for IDS” | Many exist | Yang 2023, Wisan 2025, Yagiz 2025, Benaddi 2025, Peng 2025 |
| “First MCU IDS” | Trees already on ESP32 | **Javed 2024** CatBoost thermostat, µs inference |
| “First SHAP+KD” | Used for pruning / FL | Benaddi 2025, Hossain 2025 |
| “SHAP alignment success” | **X5 retired** | Measured ρ≈0.24 n.s. (C6) |
| “KD always helps ultra-small students” | **X1 retired** | FG: A Δ≈0 (C2) |
| “Deploy 0.9485 = multi-seed seed-42” | **X2 retired** | Dual identity (C9) |
| “PTQ drop <0.01” | **X3 retired** | Drop ≈0.024/0.027 (C4) |
| “Co-distill best under train-only” | **X4 retired** | J underperforms RF-KD (C7) |

---

## 3. Real contribution stack (map to freeze)

### Pillar A — Compression under honest protocol (C1, C2, C3, C7, C8, C9)
- Ultra-small RF-KD students: A **0.9203±0.0065** (1189 params), B **0.9392±0.0129** (3397) under train-only 10-seed (C1)  
- KD−scratch significant for A under random-row (+0.0094, p≈0.048) but **vanishes under FG** (+0.0002, p≈0.95) (C2)  
- Macro-F1 hides minority costs (Blackhole/Grayhole) (C3)  
- Co-distill J **not** superior under train-only (C7)  
- Curriculum-ext collapse at 5678 is **pipeline RNG contingency**, recoverable under clean seed (C8)  
- Pipeline mean ≠ per-route clean seed mean (C9 dual identity)

### Pillar B — Deployment fidelity (C4, C5)
- Deploy-clean seed-42 unit: **agree=1.0** vs fixed-point ref on **56,200** vectors, **ESP32-C3 and Arduino R4**  
- Float→fixed macro-F1 drop **~0.024 / ~0.027** (gate policy 0.03, not 0.01)  
- Host ONNX/OpenVINO FP32 agree 1.0; dynamic INT8 costly (C5)

### Pillar C — XAI as transfer diagnostic (C6)
- On **deployment** RF-KD units: global TreeExplainer–DeepExplainer Spearman **ρ≈0.24 / 0.23, n.s.**  
- Frame as: **distillation does not preserve RF global feature ranking** — consistent with Krishna disagreement literature, opposite of “alignment win”

### Pillar D — Secondary dataset leakage honesty (C10)
- Edge-IIoT group-aware: ~17% cross-partition exposure under random-row; GA scores lower absolute F1 with positive KDΔ  

---

## 4. Related-work structure (use this section order)

### 4.1 WSN-DS and classical WSN IDS
- Almomani 2016 (dataset; Grayhole hard)  
- Later WSN ML (Talukder, Pandey, Nguyen GSWO-CatBoost, Alqahtani GXGBoost, Birahim PSO-ensemble)  
→ **Gap they leave:** little ultra-small **distilled NN** + MCU dual-board fidelity on this corpus

### 4.2 Knowledge distillation for IDS / edge
- Hinton 2015; Stanton 2021 (fidelity≠generalization — *conceptual* backbone of C2)  
- Yang 2023 LNet-SKD (self-KD, ~5k params software)  
- Wisan 2025 (DNN→shallow KD, multi-dataset, **workstation only**)  
- Benaddi 2025 (SHAP prune + Kronecker KD, TON_IoT, ms software)  
- Peng 2025 FD-IDS (FL+KD Non-IID)  
- Yagiz 2025 LENS-XAI (KD+VAE+attribution, multi-dataset software)  
→ **Gap:** RF teacher + train-only protocol ladder + dual identity + dual MCU integer HIL on WSN-DS

### 4.3 Hardware-aware / on-device IDS
- Jacob 2018 (integer quant theory)  
- Diab 2025 (HW-aware trees/CNN on **Pi 3 B+**, Edge-IIoT; no KD)  
- Javed 2024 (**ESP32 CatBoost** thermostat; IDSH; µs; no KD/NN student)  
- Alfarra 2025 (WSN-DS hybrid: **rule on-node + CNN-LSTM gateway**)  
- Misrak 2025 (QAT hybrid IDS software size claims)  
→ **Gap:** distilled NN students with **dual-board full-test integer agreement** and honest PTQ F1 cost

### 4.4 XAI on IDS and explanation disagreement
- Lundberg 2017 SHAP; Krishna 2022 disagreement; Adebayo 2018 sanity  
- Nugraha 2025 SHAP feature selection IDS  
- Benaddi/Hossain SHAP for selection or FL features  
→ **Gap:** quantifying **post-KD** RF TreeExplainer vs student DeepExplainer **global rank** on **deployed** weights (and reporting failure)

---

## 5. Differentiation table (camera-ready style)

| Work | Domain | KD | On-device | Dual board | Protocol ablations | XAI metric |
|---|---|---|---|---|---|---|
| Yang 2023 | NSL/CIC | Self-KD | No | No | Limited | No |
| Wisan 2025 | Multi IDS | DNN→shallow | No (PC+GPU) | No | α,T fixed | No |
| Benaddi 2025 | TON_IoT | Kronecker KD | Software latency | No | α,T grid | SHAP **prune** |
| Yagiz 2025 | Multi + Edge-IIoT | KD+VAE | No | No | — | Attribution XAI |
| Peng 2025 | Edge-IIoT FL | FL+KD | No | No | Non-IID θ | No |
| Diab 2025 | Edge-IIoT | No | Pi gateway | No | HW budgets | No |
| Javed 2024 | IDSH home IoT | No (trees) | **ESP32** | No | FS / depth | No |
| Alfarra 2025 | WSN-DS | No | Hybrid node+**gateway** | No | Energy | No |
| **CuKD-XAI** | **WSN-DS (+Edge GA)** | **RF→tiny NN** | **ESP32-C3 + R4** | **Yes, agree=1.0** | **Train-only / FG / dual ID / J** | **Spearman ρ (fails)** |

---

## 6. Abstract skeleton (freeze-safe; fill numbers only from freeze)

1. Problem: WSN routing-attack IDS needs tiny models; literature either software KD or non-KD MCU trees/hybrids.  
2. Method: RF teacher → students A/B; train-only protocol; deployment-clean export; dual MCU HIL; SHAP rank diagnostic.  
3. Result compression (C1).  
4. Result protocol sensitivity (C2) + minority costs (C3).  
5. Result dual identity (C9) + HIL agree + PTQ drop (C4).  
6. Result XAI non-preservation (C6).  
7. Implication: claim KD carefully; validate deployment unit separately; treat XAI rank transfer as empirical, not assumed.

---

## 7. Writing rules for “perfect”

1. Every quantitative sentence → table/figure ID + freeze claim ID.  
2. Never mix pipeline 10-seed cells with deployment-clean 0.9485.  
3. Prefer **macro-F1 ± std, n seeds, paired tests** over single-run Acc.  
4. Discuss **limitations** in body (not only appendix): FG null KD, SHAP n.s., PTQ 0.024, J underperforms, CLEXT RNG.  
5. Related work cites **Javed and Wisan** in first two paragraphs of hardware/KD sections — reviewers will know them.  
6. Tone: systems + measurement paper, not “SOTA crush.”
