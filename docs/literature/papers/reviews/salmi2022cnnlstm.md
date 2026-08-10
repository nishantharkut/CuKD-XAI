# Review card: salmi2022cnnlstm

**Status:** PASS1B+PASS2_TEXT+PASS2_VISUAL_TITLE OK
**PDF text pages extracted:** 6
**Ground truth extract:** `_extract/salmi2022cnnlstm.full.txt`
**Evidence JSON:** `_pass1b_evidence/salmi2022cnnlstm.json`
**Generated:** 2026-08-06

## Identity (page-1 text)
- **Title:** CNN-LSTM based Approach for DDoS Detection Tahani Alasmari
- **Tags:** XAI

## Abstract (extracted)
> Distributed Denial of Service (DDoS) attacks have become increasingly common, causing financial and reputational losses for organizations. Despite the existence of numerous conventional detection solutions, DDoS attacks continue to rise in frequency, demanding effective models to detect and prevent them. This paper focuses on developing a machine learning- based approach for DDoS attack detection. By leveraging the power of machine learning, we aim to overcome the limitations of existing methods and propose a novel solution. Our work emphasizes the importance of exploring advanced models and techniques to enhance detection accuracy and efficiency. Through rigorous experimentation, we demonstrate the effectiveness of our approach in proactive defense against real-world DDoS attacks.

## Table headers present in PDF text (exact lines)
- `Table 2 shows the difference in the accuracy of`

## Metric-bearing lines (exact PDF lines; PASS2-verified)
- `reached an accuracy of 99% in detecting DDOS attacks.`
- `96.43% accuracy rate.`
- `98.9% accuracy rate. Moreover, a machine learning-`
- `the required shape. Our input data has 67 features. So,`
- `the reshape layer makes an input shape of (67, 1), where`
- `(2*2*64) = 256 trainable parameters. And each of them`
- `Figure 5 shows the accuracy of all machine learning`
- `Fig. 5. Comparison of the accuracy of different ML algorithms.`
- `Figure 6 shows the accuracy of all machine learning`
- `Fig. 6. Comparison of the inaccuracy of different ML algorithms.`
- `Fig. 7. Comparison of the accuracy of different ML algorithms with`
- `Table 2 shows the difference in the accuracy of`

## CuKD freeze notes (non-numeric)
- XAI neighborhood → do not invent Spearman ρ; C6 is CuKD measurement.
- **Numbers only from exact lines above.**

## Verification
- PASS1B: evidence JSON written with exact lines from extract
- PASS2: all `13` quoted lines re-found in full extract (by construction)
- PASS2_TEXT: OK; PASS2_VISUAL_TITLE: OK

## PASS2 independent text re-check (2026-08-06)
- **PASS2_OK** — 13/13 quotes exact-match in full extract

## PASS2 visual title check (2026-08-06)
- page_001.png exists: True
- title word hit ratio vs page-1 text: 1.000
- **PASS2_VISUAL_TITLE_OK** (title words supported by page-1 text + PNG on disk)
