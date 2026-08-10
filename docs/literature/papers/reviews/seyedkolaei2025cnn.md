# Review card: seyedkolaei2025cnn

**Status:** PASS1B+PASS2_TEXT+PASS2_VISUAL_TITLE OK
**PDF text pages extracted:** 21
**Ground truth extract:** `_extract/seyedkolaei2025cnn.full.txt`
**Evidence JSON:** `_pass1b_evidence/seyedkolaei2025cnn.json`
**Generated:** 2026-08-06

## Identity (page-1 text)
- **Title:** Tolulope Odeyomi and Revised: 20 May 2025 Seyedkolaei, A.A.;
- **Tags:** EdgeIIoT, quant, gateway

## Abstract (extracted)
> The rapid expansion of the Internet of Things (IoT) and industrial Internet of Things (IIoT) ecosystems has introduced new security challenges, particularly the need for robust intrusion detection systems (IDSs) capable of adapting to increasingly sophisticated cyberattacks. In this study, we propose a novel intrusion detection approach based on convolutional neural networks (CNNs), designed to automatically extract spatial patterns from network traffic data. Leveraging the DNN-EdgeIIoT dataset, which includes a wide range of attack types and traffic scenarios, we conduct comprehensive experiments to compare the CNN-based model against traditional machine learning techniques, including decision trees, random forests, support vector machines, and K-nearest neighbors. Our approach consistently outperforms baseline models across multiple performance metrics— such as F1 score, precision, and recall—in both binary (benign vs. attack) and multiclass settings (6-class and 15-class classification). The CNN model achieves F1 scores of 1.00, 0.994, and 0.946, respectively, highlighting its strong generalization ability across diverse attack categories. These results demonstrate the effectiveness of deep-learning-based IDSs in enhancing the security posture of IoT and IIoT infrastructures, paving the way for intelligent, adaptive, and scalable threat detection systems.

## Table headers present in PDF text (exact lines)
- `Table 1. Distribution of 2-class samples.`
- `Table 2. Distribution of 6-class samples.`
- `Table 3. Distribution of 15-class samples.`
- `Table 4. Key parameter values for the proposed CNN model.`
- `Table 5. Number of samples selected for training and test subsets.`
- `Table 6. Classification results for 2-class.`
- `Table 7. Classification results for 6-class.`
- `Table 8. Classification results for 15-class.`
- `Table 9. Performance of the CNN model for intrusion detection in a binary classification scenario.`
- `Table 10. Results of ML and DL methods for 2-class classification.`
- `Table 11. Results of ML and DL methods for 6-class classification.`
- `Table 12. Results of ML and DL methods for 15-class classification.`

## Metric-bearing lines (exact PDF lines; PASS2-verified)
- `such as F1 score, precision, and recall—in both binary (benign vs. attack) and multiclass`
- `settings (6-class and 15-class classification). The CNN model achieves F1 scores of 1.00,`
- `accuracy, precision, recall, and F1 score, comparing it with other state-of-the-art algorithms.`
- `F1 scores of 99.9% and 99.87% across both datasets, demonstrating its strong performance`
- `Table 4. Key parameter values for the proposed CNN model.`
- `as model pruning (reducing parameters by 30–40% without accuracy loss) and quantiza-`
- `across all classes. The calculation of accuracy is shown in Equation (4).`
- `recall is shown in Equation (5).`
- `F1 Score: It serves as a suitable criterion for assessing detector performance, as it`
- `integrates both precision and recall. Equation (7) outlines the expression for F1 score`
- `F1 Score = 2 ∗Precision ∗Recall`
- `Classical performance metrics (accuracy, precision, recall, and F1 score) were evaluated`
- `In Table 6, the binary classification results (2-class) indicate an accuracy of 100% and a`
- `has not misclassified any attacks as normal. Additionally, the model’s precision is 100%,`
- `reflecting its high ability to accurately predict attacks, and the F1 score is 1.00. These results`
- `model achieved strong weighted values, with a precision of 99.57%, recall of 99.48%,`
- `and an F1 score of 0.994. However, for specific attack types, some misclassifications`
- `samples, yet misclassified 577 as injection-based attacks, resulting in a high F1 score of`
- `but 1565 samples were incorrectly categorized as this attack type, yielding an F1 score`
- `of 0.929. Similarly, MITM-based attacks achieved an F1 score of 0.971, whereas malware-`
- `based attacks had the lowest classification performance, with an F1 score of 0.649. Overall,`
- `reaching a precision of 95.02%, a recall of 94.72%, and an F1 score of 0.946. For normal`
- `rates (F1 scores of 0.999). It is noticeable that some attacks with low representation in`
- `proposed method. On the other hand, attacks such as XSS-based (F1 = 0.428), password-`
- `based (F1 = 0.448), uploading-based (F1 = 0.561), and SQL_injection-based (F1 = 0.567)`
- `the model’s learning. Furthermore, validation accuracy reached 1.0, indicating effective`
- `KNN) across binary, 6-class, and 15-class classifications. The CNN achieved 100% accuracy`
- `Figure 5. Comparison of the proposed method’s accuracy with other methods.`
- `The CNN’s accuracy on Raspberry Pi 4 decreased to 98.7% due to hardware constraints`
- `parameter reduction) and INT8 quantization. Class imbalance (e.g., rare attacks like XSS)`

## CuKD freeze notes (non-numeric)
- Edge-IIoT neighborhood → C10 group-aware discussion.
- Quantization neighborhood → Jacob/C4 PTQ honesty.
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
