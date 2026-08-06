# Manual PDF drop list (Option B)

## Ingest status (2026-08-06)

**Source folder:** `docs/literature/papers/e2e_pdfs/nishant downloaded/`  
**Canonical copies:** `docs/literature/papers/e2e_pdfs/<id>.pdf`  
**Rasterized:** all 16 → `e2e_pages/<id>/` (**16/16 OK**, ~303 new pages; corpus now **45** page dirs / **817** PNGs)

| ID | Status |
|---|---|
| javed2024thermostat | PDF + pages; card PARTIAL |
| wisanwanichthan2025kd | PDF + pages; card PARTIAL |
| peng2025fdids | PDF + pages; card PARTIAL |
| almomani2016wsnds | PDF + pages; card PARTIAL |
| nguyen2024gswo | PDF + pages; card pending |
| alqahtani2019gxgboost | PDF + pages; card pending |
| birahim2025pso | PDF + pages; card pending |
| ghadi2024review | PDF + pages; card pending |
| alshehri2024sadcnn | PDF + pages; card pending |
| seyedkolaei2025cnn | PDF + pages; card pending |
| gao2026lightweight | PDF + pages; card pending |
| adjewa2026seed | PDF + pages; card pending |
| ishtiaq2025cstafnet | PDF + pages; card pending |
| salmi2022cnnlstm | PDF + pages; card pending |
| chawla2002smote | PDF + pages (35/37); card pending |
| ticnna_hybrid_iot | bonus PDF + pages; card pending |

---

**Your job (original):** download PDFs you can access (institution VPN / personal login) and put them here:

```
C:\N Drive\Research\Cukd-XAI\CuKD-XAI\docs\literature\papers\e2e_pdfs\
```

**Filename rule:** use the exact `id` below + `.pdf`  
Example: `javed2024thermostat.pdf`

**After drop, tell me the IDs.** I will rasterize + visual-card only those files. I will **not** invent content for missing papers.

---

## Priority 1 — drop if you will cite (CuKD-critical gaps)

| Save as | Full title | Why | Link |
|---|---|---|---|
| `javed2024thermostat.pdf` | **Embedding Tree-Based Intrusion Detection System in Smart Thermostats for Enhanced IoT Security** (Javed et al., *Sensors* 2024, 24, 7320) | Tree IDS on **ESP32** smart thermostat | https://www.mdpi.com/1424-8220/24/22/7320 |
| `wisanwanichthan2025kd.pdf` | **A Lightweight Intrusion Detection System for IoT and UAV Using Deep Neural Networks with Knowledge Distillation** (Wisanwanichthan & Thammawichai, *Computers* 2025, 14, 291) | **KD** for IoT/UAV | https://www.mdpi.com/2073-431X/14/7/291 |
| `peng2025fdids.pdf` | **FD-IDS: Federated Learning with Knowledge Distillation for Intrusion Detection in Non-IID IoT Environments** (Peng et al., *Sensors* 2025, 25, 4309) | Federated **KD** IDS | https://www.mdpi.com/1424-8220/25/14/4309 |
| `almomani2016wsnds.pdf` | **WSN-DS: A Dataset for Intrusion Detection Systems in Wireless Sensor Networks** (Almomani et al., *Journal of Sensors* 2016, 4731953) | **WSN-DS** original dataset paper | https://www.hindawi.com/journals/js/2016/4731953/ (or ResearchGate) |

## Priority 2 — useful if claiming WSN-ML / survey coverage

| Save as | Full title | Link |
|---|---|---|
| `nguyen2024gswo.pdf` | **Enhancing Intrusion Detection in Wireless Sensor Networks Using a GSWO-CatBoost Approach** (Nguyen et al., *Sensors* 2024, 24, 3339) | https://www.mdpi.com/1424-8220/24/11/3339 |
| `alqahtani2019gxgboost.pdf` | **A Genetic-Based Extreme Gradient Boosting Model for Detecting Intrusions in Wireless Sensor Networks** (Alqahtani et al., *Sensors* 2019, 19, 4383) | https://www.mdpi.com/1424-8220/19/20/4383 |
| `birahim2025pso.pdf` | **Intrusion Detection for Wireless Sensor Network Using Particle Swarm Optimization Based Explainable Ensemble Machine Learning Approach** (Birahim et al., *IEEE Access* 2025) | https://ieeexplore.ieee.org/document/10836702 (arnumber 10844145 in stamp URL) |
| `ghadi2024review.pdf` | **Machine Learning Solutions for the Security of Wireless Sensor Networks: A Review** (Ghadi et al., *IEEE Access* 2024) | https://ieeexplore.ieee.org/document/10401918 |
| `alshehri2024sadcnn.pdf` | Self-attention DCNN for IIoT intrusion detection (*IEEE Access* ~2024; arnumber 10478858) — confirm exact title on IEEE when downloading | https://ieeexplore.ieee.org/document/10478858 |

## Priority 3 — only if you need them

| Save as | Full title (as targeted) | Note |
|---|---|---|
| `seyedkolaei2025cnn.pdf` | CNN multiclass IoT (*Future Internet* 2025, 17, 230) | MDPI: https://www.mdpi.com/1999-5903/17/6/230 |
| `gao2026lightweight.pdf` | Lightweight multi-class edge IoT (*Electronics* 2026, 15, 938) | MDPI: https://www.mdpi.com/2079-9292/15/5/938 |
| `adjewa2026seed.pdf` | Edge transformer offload (*Sensors* 2026, 26, 356) | MDPI: https://www.mdpi.com/1424-8220/26/2/356 |
| `ishtiaq2025cstafnet.pdf` | **CST-AFNet** Edge-IIoT (ScienceDirect; paywall likely) | library / SciDirect |
| `salmi2022cnnlstm.pdf` | **CNN-LSTM Based Approach for DoS Attacks Detection in Wireless Sensor Networks** (or similar title; old IJACSA PDF 404) | search title on Google Scholar |
| `chawla2002smote.pdf` | **SMOTE: Synthetic Minority Over-sampling Technique** (Chawla et al., *JAIR* 2002) | optional classic cite |

## Do **not** re-drop (already on disk)

These PDFs already exist under `e2e_pdfs/`. I will review them without waiting for you:

yang, stanton, yagiz, diab, benaddi, jacob, krishna, adhane, alfarra, hossain, misrak, nugraha, cantini, ferrag, xiao, talukder*, hinton, bengio, lundberg, adebayo, guo*, banbury, sze, hasan, pandey, …

Note: `talukder2024mlstl` / `talukder2025hybrid` failed **re-rasterize** once but **pages already exist** — no re-download needed.

## Naming checklist when you drop

```
docs/literature/papers/e2e_pdfs/javed2024thermostat.pdf
docs/literature/papers/e2e_pdfs/wisanwanichthan2025kd.pdf
docs/literature/papers/e2e_pdfs/peng2025fdids.pdf
docs/literature/papers/e2e_pdfs/almomani2016wsnds.pdf
...
```

Optional: also drop into a folder `docs/literature/papers/manual_inbox/` if you prefer; I will copy+rename.

---

## My side while you hunt PDFs

Continue visual evidence cards for **on-disk** papers only → `docs/literature/papers/reviews/`.  
No manuscript abstract until cards cover Tier A.
