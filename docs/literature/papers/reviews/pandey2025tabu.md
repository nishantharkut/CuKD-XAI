# Review card: pandey2025tabu

**Status:** PASS1B+PASS2_TEXT+PASS2_VISUAL_TITLE OK
**PDF text pages extracted:** 21
**Ground truth extract:** `_extract/pandey2025tabu.full.txt`
**Evidence JSON:** `_pass1b_evidence/pandey2025tabu.json`
**Generated:** 2026-08-06

## Identity (page-1 text)
- **Title:** Enhancing intrusion detection in wireless sensor networks using a Tabu search based optimized
- **Tags:** WSN

## Abstract (extracted)
_Not auto-detected; open full extract._

## Table headers present in PDF text (exact lines)
- `Table 1.  Literature review summary of ML-based intrusion detection systems in WSN/IoT.`
- `Table 2.  Performance metrics comparison between initial and optimized RF models over WSN-DS.`
- `Table 4 shows that the model scored 99% accuracy over 74,933 instances, whereby Normal traffic (67,965 cases;`
- `Table 5.  Classification report of the optimized RF model over WSN-DS.`
- `Table 4.  Classification report of the initial RF model over WSN-DS.`
- `Table 3.  The results for the initial and optimized CICIDS2017 and CIC-IoT-2023 models.`
- `Table 6.  Performance comparison of the proposed model against state-of-the-art techniques.`

## Metric-bearing lines (exact PDF lines; PASS2-verified)
- `like precision, recall, F1-score, Cohen’s Kappa, and ROC AUC. Detection of Blackhole and Gray Hole`
- `environmental or physical parameter3. In WSN, several sensor nodes collect the data from the environment and`
- `such as RF6. Distributed wireless sensors monitor the environmental and physical parameters in WSN. Thus,`
- `than on tuning the parameters used10.`
- `or accuracy, precision, recall, ROC AUC, Cohen’s kappa, and F1-score values.`
- `In the literature, SVM-based IDS has achieved 90.8% accuracy using this model15. Among the energy`
- `learning, 100% accuracy for binary classification and 99.95% for multiclass has been achieved to classify`
- `outstanding performance at 99.7 percent compared to NB, which reached a reasonable accuracy of 97.9 percent;`
- `quantitative metrics comprising accuracy, precision, recall, F1-score, and MCC22,23. The performance results`
- `of the machine learning-based IDS models mostly depicted high accuracy values of over 99%, especially when`
- `been proposed that achieved high performance in intrusion detection. Results showed 99.7% accuracy, 99.8%`
- `precision, and 97.8% recall on the NSL-KDD dataset26. The same high outcomes were seen on other datasets,`
- `such as UNSW-NB15 and CICIDS 2017, with the model’s efficacy in improving detection accuracy through`
- `algorithm. It achieved 99% detection accuracy against DoS attacks28. Using traditional kNN, this design enhanced`
- `the detection accuracy by 10%.It realized a balance between lightweight computation and high intrusion`
- `rate reached up to 98.8% by using techniques of KNN and boosting. Precision and recall rates had reached up to`
- `97%.In boosting and stacking ensemble methods, the F1 scores reached 96% and 97%, respectively. This model`
- `19 hyperparameters, designing an ideal model manually with the correct number of hyperparameters will be`
- `have many parameters, known as hyperparameters (as many as 19), that are hard to tune because they create a`
- `A RF classifier is implemented with initial hyperparametersn_estimators = 100, max_depth = 10, and was trained`
- `The model was assessed using the metrics mentioned: accuracy, precision, recall, F1 Score, Cohen’s Kappa, and`
- `Achieved 90.8% accuracy`
- `100% accuracy for binary classification;`
- `RNN: 99.7% accuracy; NB: 97.9% accuracy`
- `> 99% accuracy with ensemble methods`
- `99.7% accuracy, 99.8% precision, 97.8%`
- `99% detection accuracy against DoS`
- `precision, recall, F1 score, and support.`
- `R (θ) is the recall can be represented as TP/TP + FN; F1 (θ) is the F1 score, balancing both precision and`
- `recall as 2(P ∗R)/P + R; K (θ) is Cohen’s Kappa, which measures the agreement between prediction and`

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
