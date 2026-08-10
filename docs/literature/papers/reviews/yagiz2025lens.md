# Review card: yagiz2025lens

**Status:** PASS1B+PASS2_TEXT+PASS2_VISUAL_TITLE OK
**PDF text pages extracted:** 52
**Ground truth extract:** `_extract/yagiz2025lens.full.txt`
**Evidence JSON:** `_pass1b_evidence/yagiz2025lens.json`
**Generated:** 2026-08-06

## Identity (page-1 text)
- **Title:** LENS-XAI: Redefining Lightweight and Explainable Network Security through Knowledge Distillation and
- **Tags:** KD, XAI

## Abstract (extracted)
_Not auto-detected; open full extract._

## Table headers present in PDF text (exact lines)
- `Table 1: Software and hardware configurations for experimentation.`
- `Table 2: Comparison of Edge-IIoTset, UKM20, CTU-13, and NSL-KDD datasets.`
- `Table 3 provides a comprehensive comparison of the proposed LENS-XAI`
- `Table 3: Performance metrics comparison for multi-class classification on the Edge-IIoTset`
- `Table 5 provides a comparison of the performance metrics for the LENS-`
- `Table 4: Performance metrics comparison for the LENS-XAI framework on the Edge-`
- `Table 5: Evaluation metrics for multi-class classification on the UKM20 dataset.`
- `Table 6:`
- `Table 7 presents a comparative analysis of the proposed LENS-XAI frame-`
- `Table 7: Classification achievements of the proposed scheme on the CTU-13 dataset.`
- `Table 8: Comparative performance metrics for multi-class classification on the NSL-KDD`
- `Table 9: Performance metrics comparison for the LENS-XAI framework on the NSL-KDD`

## Metric-bearing lines (exact PDF lines; PASS2-verified)
- `environments. Gaspar et al. [8] explored the integration of SHAP (SHapley`
- `2.4. Knowledge Distillation & VAEs`
- `tion accuracy. Sindiramutty et al. [15] effectively combined these techniques`
- `recall, precision, and F1 scores, while maintaining a lightweight computa-`
- `3.4. Knowledge Distillation for Model Optimization`
- `mance, we employ Knowledge Distillation [32], transferring the “knowledge”`
- `3.4.1. Distillation Setup`
- `3.4.2. Distillation Loss`
- `Ldistill(θs) = (1 −α) LCE(y, ˆys)`
- `6: Verify local accuracy:`
- `of 0.001, ensuring robust latent representations. Knowledge distillation is`
- `conducted with a temperature parameter T = 2 and a weighting coefficient`
- `13: Step 3: Knowledge Distillation`
- `16: Train student model fs to minimize the distillation loss:`
- `Ldistill = (1 −α) LCE + α T 2 DKL`
- `uated using metrics such as accuracy, precision, recall, and F1-score.`
- `precision (96.75%) and recall (95.34%) compared to NIDS-BAI (94.7%`
- `precision, 94.8% recall) [38]. Similarly, the Student model main-`
- `tained competitive precision (95.74%) and recall (95.31%), showcas-`
- `• F1-Measure: Both models achieved balanced performance, with the`
- `Teacher model scoring an F1-measure of 95.09% and the Student`
- `F1-Score (%)`
- `accuracy (99.64%–100.00%). For Backdoor, the models perform strongly`
- `imating the Teacher’s accuracy. Notably, it correctly classifies 61,156 DDoS`
- `F1 Score (%)`
- `accuracy of 99.92%, outperforming the Student model’s accuracy of`
- `sion and recall values, with the Teacher model achieving 99.92% for`
- `• F1-Measure: The Teacher model demonstrated an F1-measure of`
- `99.92%, indicating a balanced performance across precision and re-`
- `call. The Student model also performed remarkably, achieving an F1-`

## CuKD freeze notes (non-numeric)
- KD neighborhood → compare to C1/C2; do not claim novelty of KD-for-IDS alone.
- XAI neighborhood → do not invent Spearman ρ; C6 is CuKD measurement.
- **Numbers only from exact lines above.**

## Verification
- PASS1B: evidence JSON written with exact lines from extract
- PASS2: all `42` quoted lines re-found in full extract (by construction)
- PASS2_TEXT: OK; PASS2_VISUAL_TITLE: OK

## PASS2 independent text re-check (2026-08-06)
- **PASS2_OK** — 42/42 quotes exact-match in full extract

## PASS2 visual title check (2026-08-06)
- page_001.png exists: True
- title word hit ratio vs page-1 text: 1.000
- **PASS2_VISUAL_TITLE_OK** (title words supported by page-1 text + PNG on disk)
