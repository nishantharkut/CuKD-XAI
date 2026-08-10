# Review card: talukder2025hybrid

**Status:** PASS1B+PASS2_TEXT+PASS2_VISUAL_TITLE OK
**PDF text pages extracted:** 23
**Ground truth extract:** `_extract/talukder2025hybrid.full.txt`
**Evidence JSON:** `_pass1b_evidence/talukder2025hybrid.json`
**Generated:** 2026-08-06

## Identity (page-1 text)
- **Title:** A hybrid machine learning model for intrusion detection in wireless sensor networks leveraging data
- **Tags:** WSN

## Abstract (extracted)
_Not auto-detected; open full extract._

## Table headers present in PDF text (exact lines)
- `Table 2.  Category-to-label mapping.`
- `Table 1.  Category-to-label mapping.`
- `Table 3 highlights the impact of data balancing techniques-STL, GAN, and KMS-on the WSN-DS dataset. It`
- `Table 4 demonstrates the effects of data balancing techniques-STL, GAN, and KMS-on the TON-IoT network`
- `Table 3.  Before and after data balancing techniques on WSN-DS.`
- `Table 4.  Before and after data balancing techniques on TON-IoT network.`
- `Table 5.  Confusion matrix.`
- `Table 6 shows the performance of various ML models on the WSN dataset across three experimental setups:`
- `Table 7 illustrates the performance analysis of various hybrid machine learning models applied to the TON-`
- `Table 7.  Performance analysis of ML models on TON-IoT network dataset. Significant values are in bold.`
- `Table 6.  Performance analysis of ML models on WSN dataset. Significant values are in bold.`
- `Table 9.  Confusion matrix of the proposed (KMS+PCA+RFC) model on TON-IoT-network.`

## Metric-bearing lines (exact PDF lines; PASS2-verified)
- `hybrid (KMS + PCA + RFC) approach achieves remarkable performance, with an accuracy of 99.94% and`
- `an f1-score of 99.94% on the WSN-DS dataset. For the TON-IoT dataset, it achieves 99.97% accuracy`
- `and an f1-score of 99.97%, outperforming traditional SMOTE TomekLink and Generative Adversarial`
- `of accuracy, precision, recall, and f1-score, thereby providing a robust framework for securing these critical`
- `accuracy, precision, recall, and f1-score, in multiclass scenarios, demonstrating its effectiveness in identifying`
- `Grey Wolf Optimizer (GWO), achieved an impressive performance with 99.34% accuracy, 98.36% recall and`
- `96.67% f1-score. These results demonstrate the FA-ML method’s effectiveness as a robust security solution for`
- `the models tested, LOA-Cb-C achieved the highest accuracy of 99.66% and 99% of precision, recall and f1-score`
- `and the lowest error rate of 0.34%, significantly improving WSN-IoT security by enhancing detection accuracy`
- `precision, 85.29% recall, 82.34% f1-score, and 96.86% accuracy.`
- `local optima. The experimental results showed that the model on WSN-DS achieved an accuracy of 94.76%,`
- `precision of 86%, recall of 80%, and an F-score of 80.06%.`
- `with Min-Max Scaling achieved the highest accuracy of 99.70%, outperforming the other approaches.`
- `accurac, precision, recall, and F1 score respectively.`
- `methods were tested on benchmark IDS datasets. The proposed usfAD model achieved 99.43% accuracy, 98.39%`
- `precision, and 99.37% f1-score on the TON-IoT Network dataset. Results demonstrated that the OCC model,`
- `errors. The ensemble voting model outperformed individual classifiers, achieving 96.32% accuracy, 93.11%`
- `precision, 84.55% recall, and 88.63% f1-score, demonstrating its effectiveness in detecting IoT cyber threats.`
- `time. Decision trees (DT), using PCA with 33 features, achieved 77.62% accuracy, 67.34% precision, 48.67%`
- `recall, and a 45.53% f1-score. The research highlighted the strengths and limitations of each approach, offering`
- `results with an accuracy of 99.63%, precision of 96.48%, recall of 93.18%, and an f1-score of 94.45%. These`
- `framework demonstrated strong performance, achieving an accuracy and f1-score of 98.94% on the TON-IoT`
- `rics such as Accuracy, Precision, Recall, f1-score, Kappa, MCC, ROC-AUC. MAE, MSE, RMSE, Confusion`
- `and false negatives. This analysis typically includes metrics such as accuracy, precision, recall, f1-score, and area`
- `•	 F1-Score:`
- `F1-score = 2 · Precision · Recall`
- `such as accuracy, precision, sensitivity, specificity, F1 score, kappa, MCC, ROC-AUC, MAE, MSE, and RMSE`
- `achieving the highest metrics across all categories, including accuracy (99.94%), precision (99.94%), sensitivity`
- `(99.94%), specificity (99.99%), f1-score (99.94%), Kappa (99.93%), MCC (99.93%), and ROC-AUC (99.99%).`
- `99.97%, precision of 99.97%, sensitivity of 99.97%, specificity of 100.0%, F1 score of 99.97%, kappa of 99.97%,`

## CuKD freeze notes (non-numeric)
- WSN neighborhood → Almomani WSN-DS lineage.
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
