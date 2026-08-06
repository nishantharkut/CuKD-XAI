# Review card: gao2026lightweight

**Status:** PASS1B+PASS2_TEXT+PASS2_VISUAL_TITLE OK
**PDF text pages extracted:** 20
**Ground truth extract:** `_extract/gao2026lightweight.full.txt`
**Evidence JSON:** `_pass1b_evidence/gao2026lightweight.json`
**Generated:** 2026-08-06

## Identity (page-1 text)
- **Title:** Revised: 12 February 2026 distributed under the terms and conditions of the Creative Commons
- **Tags:** EdgeIIoT

## Abstract (extracted)
> Intrusion detection aims to effectively detect abnormal attacks in Internet of Things (IoT) networks, which is crucial for cybersecurity. However, it is difficult for traditional in- trusion detection methods to effectively extract data features from traffic data, and most existing models are too complex to be deployed on edge servers. Addressing this need, this paper proposes a hybrid feature selection method and a lightweight deep learning intrusion detection model. Firstly, the data feature space is reduced using variance filter- ing, mutual information, and the Pearson Correlation Coefficient, thereby reducing the computational cost of subsequent model training. Then, an intrusion detection model based on a Temporal Convolutional Network (TCN) is constructed. This model utilizes dilated causal convolutions to effectively capture long-term temporal dependencies in network traffic. Simultaneously, the residual connections are used to mitigate the vanishing gradient problem, making the model easier to train and converge. Finally, experiments are conducted on the newly released Edge-IIoTset dataset. The results show that the proposed feature selection algorithm maintains good detection performance despite a significant reduction in feature dimensionality. Furthermore, compared with other models, the proposed TCN-based approach achieves higher classification accuracy with lower com- putational overhead, demonstrating its suitability for deployment in resource-constrained edge computing environments.

## Table headers present in PDF text (exact lines)
- `Table 1. Model hyperparameter configuration.`
- `Table 2. Detection results of different feature dimensions (values are in %).`
- `Table 3 shows that the TCN training model achieved the best accuracy, recall, and F1`
- `Table 3. Test results for different setting of parameters d and w.`
- `Table 4. Comparison of performance of different methods.`
- `Table 5. Detailed classification report.`

## Metric-bearing lines (exact PDF lines; PASS2-verified)
- `accuracy by 1.5% and recall by 4% on NSL-KDD, with only a marginal increase of 0.1 ms in`
- `f1, . . . fp`
- `f1, f2, . . . fp′`
- `4.2. Hyperparameters for the Model’s Training`
- `Table 1. Model hyperparameter configuration.`
- `recall, and F1 score as evaluation metrics for the model training results. The specific`
- `F1 = 2 × Precision × Recall`
- `Table 3 shows that the TCN training model achieved the best accuracy, recall, and F1`
- `Table 3. Test results for different setting of parameters d and w.`
- `recall (Rc.), and F1 score, as well as the number of model parameters, memory usage, and`
- `highest metrics: 93.79% accuracy, 93.33% precision, 93.79% recall, and a 93.13% F1 score,`
- `Figure 8 shows the accuracy of each model on the validation set as a function of epochs.`
- `Figure 8. Comparison of validation accuracy across different models.`
- `shown in Table 5, the model’s accuracy in identifying Normal, DDoS_UDP, DDoS_ICMP,`
- `F1 Score`

## CuKD freeze notes (non-numeric)
- Edge-IIoT neighborhood → C10 group-aware discussion.
- **Numbers only from exact lines above.**

## Verification
- PASS1B: evidence JSON written with exact lines from extract
- PASS2: all `21` quoted lines re-found in full extract (by construction)
- PASS2_TEXT: OK; PASS2_VISUAL_TITLE: OK

## PASS2 independent text re-check (2026-08-06)
- **PASS2_OK** — 21/21 quotes exact-match in full extract

## PASS2 visual title check (2026-08-06)
- page_001.png exists: True
- title word hit ratio vs page-1 text: 1.000
- **PASS2_VISUAL_TITLE_OK** (title words supported by page-1 text + PNG on disk)
