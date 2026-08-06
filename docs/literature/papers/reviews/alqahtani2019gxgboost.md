# Review card: alqahtani2019gxgboost

**Status:** PASS1B+PASS2_TEXT+PASS2_VISUAL_TITLE OK
**PDF text pages extracted:** 20
**Ground truth extract:** `_extract/alqahtani2019gxgboost.full.txt`
**Evidence JSON:** `_pass1b_evidence/alqahtani2019gxgboost.json`
**Generated:** 2026-08-06

## Identity (page-1 text)
- **Title:** A Genetic-Based Extreme Gradient Boosting Model for Detecting Intrusions in Wireless Sensor Networks
- **Tags:** WSN

## Abstract (extracted)
> An Intrusion detection system is an essential security tool for protecting services and infrastructures of wireless sensor networks from unseen and unpredictable attacks. Few works of machine learning have been proposed for intrusion detection in wireless sensor networks and that have achieved reasonable results. However, these works still need to be more accurate and eﬃcient against imbalanced data problems in network traﬃc. In this paper, we proposed a new model to detect intrusion attacks based on a genetic algorithm and an extreme gradient boosting (XGBoot) classiﬁer, called GXGBoost model. The latter is a gradient boosting model designed for improving the performance of traditional models to detect minority classes of attacks in the highly imbalanced data traﬃc of wireless sensor networks. A set of experiments were conducted on wireless sensor network-detection system (WSN-DS) dataset using holdout and 10 fold cross validation techniques. The results of 10 fold cross validation tests revealed that the proposed approach outperformed the state-of-the-art approaches and other ensemble learning classiﬁers with high detection rates of 98.2%, 92.9%, 98.9%, and 99.5% for ﬂooding, scheduling, grayhole, and blackhole attacks, respectively, in addition to 99.9% for normal traﬃc.

## Table headers present in PDF text (exact lines)
- `Table 1. Extracted features of the wireless sensor network-detection system (WSN-DS) Dataset.`
- `Table 1. Cont.`
- `Table 2. Data samples from the WSN-DS dataset [46].`
- `Table 2. Cont.`
- `Table 3. The dataset separated 60% training set and 40% testing set using holdout method.`
- `Table 4. Precision results of the 10 fold cross validation.`
- `Table 5. Recall results of the 10 fold cross validation.`
- `Table 6. F1-score results of the 10 fold cross validation.`
- `Table 7. Positive and negative rates results of the 10 folds cross validation.`
- `Table 8. Average results of precision, recall, and F1-score, and their weighted average for the 10 fold`
- `Table 9 lists the true positive, true negative, false positive, and false negative rates results of the`
- `Table 9. Positive and negative rates results of the 10 fold cross validation.`

## Metric-bearing lines (exact PDF lines; PASS2-verified)
- `in WSNs. The proposed classiﬁer attains best F1-score results are 96%, 99%, 98%, 96% and 100% for`
- `the study showed that the accuracy of using random forest classiﬁer was 92.39% and the accuracy of`
- `using SMOTE has increased the accuracy to 92.57%.`
- `max_accuracy = 0`
- `model1_validation_accuracy = population_validation_accuracy [pop_index]`
- `model2_validation_accuracy= max_accuracy`
- `new_model = crossover_GXGBoost (model1, model1_validation_accuracy, model2,`
- `model2_validation_accuracy)`
- `A set of evaluation metrics including the accuracy (ACC), precision (PR), recall (RE), and f1-score`
- `F1-Score = 2*((Precision * Recall)/(Precision + Recall))`
- `Table 4. Precision results of the 10 fold cross validation.`
- `Table 5. Recall results of the 10 fold cross validation.`
- `Table 6. F1-score results of the 10 fold cross validation.`
- `Table 8. Average results of precision, recall, and F1-score, and their weighted average for the 10 fold`
- `F1-Score`
- `GXGBoost model using the holdout method. While, the precision, recall, and F1-score results and their`
- `Table 10. Precision, recall, and F1-score results of the holdout method.`
- `F1-Score`

## CuKD freeze notes (non-numeric)
- WSN neighborhood → Almomani WSN-DS lineage.
- **Numbers only from exact lines above.**

## Verification
- PASS1B: evidence JSON written with exact lines from extract
- PASS2: all `30` quoted lines re-found in full extract (by construction)
- PASS2_TEXT: OK; PASS2_VISUAL_TITLE: OK

## PASS2 independent text re-check (2026-08-06)
- **PASS2_OK** — 30/30 quotes exact-match in full extract

## PASS2 visual title check (2026-08-06)
- page_001.png exists: True
- title word hit ratio vs page-1 text: 1.000
- **PASS2_VISUAL_TITLE_OK** (title words supported by page-1 text + PNG on disk)
