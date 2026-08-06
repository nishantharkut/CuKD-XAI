# Review card: xiao2025local

**Status:** PASS1B+PASS2_TEXT+PASS2_VISUAL_TITLE OK
**PDF text pages extracted:** 15
**Ground truth extract:** `_extract/xiao2025local.full.txt`
**Evidence JSON:** `_pass1b_evidence/xiao2025local.json`
**Generated:** 2026-08-06

## Identity (page-1 text)
- **Title:** Signal, Image and Video Processing (2025) 19:1335 Metaheuristically optimized deep soft-voting ensemble
- **Tags:** WSN

## Abstract (extracted)
> Wireless Sensor Networks (WSN) are widely employed in various sensitive areas; however, due to their limited computing power, energy, and memory, these networks are exposed to signal-level anomalies as well as cyber-attacks. This study aims to introduce a metaheuristically optimized deep soft-voting ensemble that integrates Deep Neural Networks and CATBoost classiﬁers to enable secure intrusion detection. To achieve a seamless integration between convergence accuracy and explo- ration divergence, the authors have employed two complementary metaheuristics, Quadratic Interpolation Optimization (QIO) and Osprey Optimization Algorithm (OOA), for tuning the hyperparameters and decision thresholds. The system derives 23 operational and topological signal features that represent normal trafﬁc and four major attack types, i.e., Blackhole, Grayhole, Flooding, and TDMA Scheduling. Rigorous preprocessing, such as Variance Inﬂation Factor analysis, supports that the fea- tures are independent and stable. The metrics used include accuracy, precision, recall, F1-score, speciﬁcity, and AUC, with the QIO-enabled model (DCQI) leading the way with the highest test accuracy of 95.62%.DCQI outperforms the baseline (DNCA) and the OOA-enhanced models, and thus, both class-level detection and balanced performance can be achieved. The feature sensitivity analysis based on the Cosine Amplitude Method and ranking both distinctly and consistently indicates Is CH, Who CH, and Dist to CH as the most inﬂuential predictors, which not only strengthens the understanding of WSNs but also shows the computational efﬁciency of the adopted security.

## Table headers present in PDF text (exact lines)
- `Table 1 Comparison of state-of-the-art IDS approaches in WSNs and the research gap addressed by this study`
- `Table 3 presents the description of features alongside their`
- `Table 2 Description of used attributes`
- `Table 3 Description of features and determination of selected features`
- `Table 4 Hyperparameters and architectural details for reproducibility`
- `Table 6 visually summarizes the quantitative evaluation`
- `Table 5 Class-wise Sample`
- `Table 6 Evaluation metrics were`
- `Table 7 breaks down the performance of the models across`
- `Table 7 The models’`

## Metric-bearing lines (exact PDF lines; PASS2-verified)
- `and Osprey Optimization Algorithm (OOA), for tuning the hyperparameters and decision thresholds. The system derives 23`
- `tures are independent and stable. The metrics used include accuracy, precision, recall, F1-score, speciﬁcity, and AUC, with`
- `the QIO-enabled model (DCQI) leading the way with the highest test accuracy of 95.62%.DCQI outperforms the baseline`
- `ment across accuracy, precision, recall, F1-score, speci-`
- `F1 Score is one of the measures that combines the fea-`
- `F1 −score = 2 × Recall × Precision`
- `hyperparameter, and an iteration limit of 200. The stopping`
- `actions between precision, recall, and F1-score for different`
- `Table 4 Hyperparameters and architectural details for reproducibility`
- `F1-score, and speciﬁcity, measured in both training and`
- `framework, with an accuracy of 0.9218 (train) and 0.9217`
- `(test), combined with high precision (0.9580/0.9557) and`
- `rates effectively. However, the recall values (0.9218/0.9217)`
- `mance level, reaching maximum training accuracy (0.9545)`
- `and testing accuracy (0.9562), as well as a balanced and`
- `raised F1-score (0.9593/0.9607), thus providing evidence`
- `F1-scores (0.9530/0.9527). This implies the effectiveness of`
- `ofhighaccuracy,balancedF1-scores,andsuperiorspeciﬁcity`
- `almost perfect precision for all models (> 0.998) with high`
- `recall (> 0.92), showing their ability to detect benign traf-`
- `the best F1-score value (0.9764) among all the others. This`
- `low (0.2942), whereas its recall was quite high (0.9218). It`
- `highest F1-score (0.5969) as a result. Additionally, not only`
- `the F1-scores over DNCA, which is the key to the role that`
- `features (precision: 0.6993, F1-score: 0.8060), while DCOA`
- `F1 score`
- `cision, recall, F1-score, speciﬁcity, and AUC) as evidence`

## CuKD freeze notes (non-numeric)
- WSN neighborhood → Almomani WSN-DS lineage.
- **Numbers only from exact lines above.**

## Verification
- PASS1B: evidence JSON written with exact lines from extract
- PASS2: all `37` quoted lines re-found in full extract (by construction)
- PASS2_TEXT: OK; PASS2_VISUAL_TITLE: OK

## PASS2 independent text re-check (2026-08-06)
- **PASS2_OK** — 37/37 quotes exact-match in full extract

## PASS2 visual title check (2026-08-06)
- page_001.png exists: True
- title word hit ratio vs page-1 text: 1.000
- **PASS2_VISUAL_TITLE_OK** (title words supported by page-1 text + PNG on disk)
