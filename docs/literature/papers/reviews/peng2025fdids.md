# Review card: peng2025fdids

**Status:** PASS1B+PASS2_TEXT+PASS2_VISUAL_TITLE OK
**PDF text pages extracted:** 31
**Ground truth extract:** `_extract/peng2025fdids.full.txt`
**Evidence JSON:** `_pass1b_evidence/peng2025fdids.json`
**Generated:** 2026-08-06

## Identity (page-1 text)
- **Title:** Revised: 22 June 2025 FD-IDS: Federated Learning with Knowledge Distillation for Intrusion
- **Tags:** KD, EdgeIIoT, FL

## Abstract (extracted)
> With the rapid advancement of Internet of Things (IoT) technology, intrusion detection systems (IDSs) have become pivotal in ensuring network security. However, the data produced by IoT devices is typically sensitive and tends to display non-independent and identically distributed (Non-IID) properties. These factors impose significant limitations on the application of traditional centralized learning. In response to these issues, this study introduces a novel IDS framework grounded in federated learning and knowledge distillation (KD), termed FD-IDS. The proposed FD-IDS aims to tackle issues related to safe- guarding data privacy and distributed heterogeneity. FD-IDS employs mutual information for feature selection to enhance training efficiency. For Non-IID data scenarios, the system combines a proximal term with KD. The proximal term restricts the deviation between local and global models, while KD utilizes the global model to steer the training process of local models. Together, these mechanisms effectively alleviate the problem of model drift. Experiments conducted on both the Edge-IIoT and N-BaIoT datasets demonstrate that FD-IDS achieves promising detection performance across multiple evaluation metrics.

## Table headers present in PDF text (exact lines)
- `Table 1. Summary of the recent research on IDSs.`
- `Table 2. Hyperparameter search space.`
- `Table 3. Hyperparameter settings.`
- `Table 4. Confusion matrix.`
- `Table 5. Performance evaluation of FD-IDS in different Non-IID scenarios.`
- `Table 5, the results on this dataset exhibit a degree of consistency with those obtained`
- `Table 6 presents the experimental results that were obtained using Edge-IIoT. In the`
- `Table 7 presents the experimental results that were obtained using N-BaIoT. When Î¸ = 1,`
- `Table 6. Performance comparison of the different KD intervals in different Non-IID scenarios (Edge-`
- `Table 7. Performance comparison of the different KD intervals in different Non-IID scenarios (N-`
- `Table 8. Performance comparison of an IDS with and without distillation under high- and low-Non-`
- `Table 9. Performance comparison of an IDS with and without distillation under high- and low-Non-`

## Metric-bearing lines (exact PDF lines; PASS2-verified)
- `accuracy on the UNSW-NB15 dataset, while BiLSTM attained 96.41%. On the BoT-IoT`
- `dataset, both models excelled, achieving a remarkable 99.99% accuracy. Fatani et al. [25]`
- `data. Using three clients, the RNN model achieved a global detection accuracy of 91.87%.`
- `a fifteen-class task on Edge-IIoTset, the global model achieved 89.91% accuracy after 10`
- `training rounds. On the InSDN dataset, F-BIDS achieved a global model accuracy of 99.91%`
- `after 50 rounds, with the lowest client accuracy exceeding 99.70%. Nobakht et al. [29] pro-`
- `performance degradation caused by data heterogeneity, achieving an F1 score improvement`
- `and CNN-LSTM. In the FL scenario, the highest accuracy for the 15-class classification task`
- `the model parameters, the Adam optimizer was chosen with a learning rate of 0.001.`
- `3.5. Federated Learning Process with Knowledge Distillation`
- `client devices. The central server initializes the parameters W0`
- `their local objective functions to generate updated model parameters wi+1`
- `algorithm to generate the global model parameters wi+1`
- `2 represents the squared Euclidean distance between the parameters of the`
- `3.5.2. Federated Learning Optimization Based on Knowledge Distillation`
- `data imbalances. The distillation process is illustrated in Figure 5.`
- `Figure 5. Distillation process.`
- `between their respective outputs [43]. The distillation loss is defined as follows:`
- `Ldistill = Î» Â· Lhard + (1 âˆ’Î») Â· Lsoft,`
- `Ldistill = Î» Â· Lhard + (1 âˆ’Î») Â· Lsoft + Î² Â· Lproximal,`
- `Ldistill = Î» Â· Lhard + (1 âˆ’Î») Â· Lsoft + Î² Â· Lproximal`
- `where Î¸ > 0 denotes the concentration parameter, which controls the degree of similarity`
- `existing studies [18,27,29,34] and established a reasonable set of hyperparameter candidates`
- `is presented in Table 2. After hyperparameter optimization, the final configuration adopted`
- `Table 2. Hyperparameter search space.`
- `Table 3. Hyperparameter settings.`
- `F1 âˆ’score = 2 Ã— Precision Ã— Recall`
- `global model accuracy was 85.27%. Despite the relatively low degree of data heterogeneity,`
- `performing client achieved an accuracy of 86.08%, whereas the worst-performing client`
- `had an accuracy of only 7.32%. The global model accuracy in this scenario was 77.12%.`

## CuKD freeze notes (non-numeric)
- KD neighborhood â†’ compare to C1/C2; do not claim novelty of KD-for-IDS alone.
- Edge-IIoT neighborhood â†’ C10 group-aware discussion.
- Federated setting â†’ distinct from single-node MCU HIL.
- **Numbers only from exact lines above.**

## Verification
- PASS1B: evidence JSON written with exact lines from extract
- PASS2: all `42` quoted lines re-found in full extract (by construction)
- PASS2_TEXT: OK; PASS2_VISUAL_TITLE: OK

## PASS2 independent text re-check (2026-08-06)
- **PASS2_OK** â€” 42/42 quotes exact-match in full extract

## PASS2 visual title check (2026-08-06)
- page_001.png exists: True
- title word hit ratio vs page-1 text: 1.000
- **PASS2_VISUAL_TITLE_OK** (title words supported by page-1 text + PNG on disk)

## DEEP_VISUAL (manual image pages 001,017)

- Table 6 Edge-IIoT: Round-wise KD Acc 94.82 (low Non-IID) / 93.86 (high Non-IID)
- FL+KD setting

