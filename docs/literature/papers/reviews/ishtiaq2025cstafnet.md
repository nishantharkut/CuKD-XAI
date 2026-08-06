# Review card: ishtiaq2025cstafnet

**Status:** PASS1B+PASS2_TEXT+PASS2_VISUAL_TITLE OK
**PDF text pages extracted:** 9
**Ground truth extract:** `_extract/ishtiaq2025cstafnet.full.txt`
**Evidence JSON:** `_pass1b_evidence/ishtiaq2025cstafnet.json`
**Generated:** 2026-08-06

## Identity (page-1 text)
- **Title:** CST-AFNet: A dual attention-based deep learning framework for intrusion detection in IoT networks
- **Tags:** XAI

## Abstract (extracted)
_Not auto-detected; open full extract._

## Table headers present in PDF text (exact lines)
- `Table 1`
- `Table 2`
- `Table 3`
- `Table 4 shows perfect performance of the proposed model in binary`
- `Table 4`
- `Table 5`

## Metric-bearing lines (exact PDF lines; PASS2-verified)
- `an outstanding accuracy with 15 attack types and benign traffic. CST-AFNet model achieves 99.97 % accuracy.`
- `Moreover, this model demonstrates an exceptional accuracy with macro-averaged precision, recall, and F1-score`
- `all above 99.3 %. Experimental results demonstrate that CST-AFNet achieves superior detection accuracy,`
- `[14]. Consequently, many proposed models demonstrate high accuracy`
- `yielding improved detection accuracy [21–23]. However, these methods`
- `Reshape to (60, 1)`
- `processing) is reshaped into a 2D tensor for 1D convolution processing:`
- `Where, W1 and W2 are trainable parameters, σ is the sigmoid activation`
- `trained parameters with their values are presented Table 2.`
- `key indicators including precision, recall, and F1-score for each class,`
- `Using precision in Fig. 4, recall in Fig. 3, and F1-score in Fig. 5 as`
- `precision, recall, and F1-score for several critical classes such as`
- `Here, fingerprinting showed relatively lower F1-scores compared to`
- `(0.9337), recall (0.9150), and F1-score (0.9242) are slightly reduced,`
- `model achieves an exceptional overall accuracy of 99.97 %, with macro-`
- `averaged precision, recall, and F1-score all above 99.3 %. The macro and`
- `Fig. 8 shows, the training and validation accuracy of the CST-AFNet`
- `the first few epochs, stabilizing above 99 % accuracy. This indicates that`
- `Fig. 3. Recall for each class in the multi-class classification.`
- `Fig. 4. Precision for each class in the multi-class intrusion detection.`
- `Fig. 5. F1-score for each class in the multi-class intrusion detection.`
- `F1-Score`
- `classification, achieving 100 % precision, recall, and F1-score for both`
- `training and validation accuracy rapidly reach 100 %. In this research,`
- `the validation accuracy remains relentlessly flat at 1.0 across all epochs.`
- `outstanding performance, achieving up to 100 % accuracy in binary`
- `Fig. 8. Accuracy and Loss curve against number of epochs for multi class`
- `F1-Score`
- `Fig. 10. Accuracy and Loss curve against number of epochs for binary class`

## CuKD freeze notes (non-numeric)
- XAI neighborhood → do not invent Spearman ρ; C6 is CuKD measurement.
- **Numbers only from exact lines above.**

## Verification
- PASS1B: evidence JSON written with exact lines from extract
- PASS2: all `35` quoted lines re-found in full extract (by construction)
- PASS2_TEXT: OK; PASS2_VISUAL_TITLE: OK

## PASS2 independent text re-check (2026-08-06)
- **PASS2_OK** — 30/30 quotes exact-match in full extract

## PASS2 visual title check (2026-08-06)
- page_001.png exists: True
- title word hit ratio vs page-1 text: 1.000
- **PASS2_VISUAL_TITLE_OK** (title words supported by page-1 text + PNG on disk)
