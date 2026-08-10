# Review card: nguyen2024gswo

**Status:** PASS1B+PASS2_TEXT+PASS2_VISUAL_TITLE OK
**PDF text pages extracted:** 26
**Ground truth extract:** `_extract/nguyen2024gswo.full.txt`
**Evidence JSON:** `_pass1b_evidence/nguyen2024gswo.json`
**Generated:** 2026-08-06

## Identity (page-1 text)
- **Title:** Yoo, M. Enhancing Intrusion Detection in Wireless Sensor Networks Using a GSWO-CatBoost
- **Tags:** XAI, WSN, quant

## Abstract (extracted)
> Intrusion detection systems (IDSs) in wireless sensor networks (WSNs) rely heavily on effective feature selection (FS) for enhanced efficacy. This study proposes a novel approach called Genetic Sacrificial Whale Optimization (GSWO) to address the limitations of conventional methods. GSWO combines a genetic algorithm (GA) and whale optimization algorithms (WOA) modified by applying a new three-population division strategy with a proposed conditional inherited choice (CIC) to overcome premature convergence in WOA. The proposed approach achieves a balance between exploration and exploitation and enhances global search abilities. Additionally, the CatBoost model is employed for classification, effectively handling categorical data with complex patterns. A new technique for fine-tuning CatBoost’s hyperparameters is introduced, using effective quantization and the GSWO strategy. Extensive experimentation on various datasets demonstrates the superiority of GSWO-CatBoost, achieving higher accuracy rates on the WSN-DS, WSNBFSF, NSL-KDD, and CICIDS2017 datasets than the existing approaches. The comprehensive evaluations highlight the real-time applicability and accuracy of the proposed method across diverse data sources, including specialized WSN datasets and established benchmarks. Specifically, our GSWO-CatBoost method has an inference time nearly 100 times faster than deep learning methods while achieving high accuracy rates of 99.65%, 99.99%, 99.76%, and 99.74% for WSN-DS, WSNBFSF, NSL-KDD, and CICIDS2017, respectively.

## Table headers present in PDF text (exact lines)
- `Table 1. Some recent works about intrusion detection in WSN.`
- `Table 2. NSL-KDD Dataset description [48].`
- `Table 3. Summarizing the distribution of training and testing set in this study.`
- `Table 4. Data distribution of relabeled attack.`
- `Table 3.`
- `Table 5. Features selected after FS technology.`
- `Table 6. Comparison of performance metrics of FS algorithms.`
- `Table 7. Results of hyperparameter optimization techniques.`
- `Table 8. Comparison of performance metrics of hyperparameter optimization techniques.`
- `Table 9. Validation of the proposed method with the 10-fold cross-validation technique.`
- `Table 10. Comparison of performance metrics of existing ML techniques.`

## Metric-bearing lines (exact PDF lines; PASS2-verified)
- `accuracy rates of 99.65%, 99.99%, 99.76%, and 99.74% for WSN-DS, WSNBFSF, NSL-KDD, and`
- `performance and predictive accuracy of these models [20]. This process involves adjusting`
- `accuracy rate of 98.79%. Notably, compared to ML models, deep learning models introduce`
- `performance in terms of the F1-score for the four types of network attacks compared with`
- `PSO and achieved an overall accuracy of 98%, according to the experimental evaluation`
- `nary and multi-class attacks with notable accuracy, precision, recall, and F1-score metrics.`
- `networks. The GA-RF approach attained a test accuracy of 87.61% and an Area Under`
- `2.3. Fine-Tuning Hyperparameters for Machine Learning Model`
- `SHERPA [36], introduced by Hertel et al., is an advanced hyperparameter tuning`
- `for deep learning models. Akiba et al. [37] proposed a new hyperparameter optimiza-`
- `termed BSOXGB [38] (BorutaShap feature selection combined with Optuna hyperparam-`
- `eter tuning of eXtremely Gradient Boost), achieving a notable accuracy of 97.70% in the`
- `Equation (9). Here, the parameter b represents a constant indicative of the logarithmic`
- `spiral’s shape, and l constitutes a random number within the specified range of [−1, 1].`
- `4.5. Applying GSWO for Hyperparameter Optimization`
- `Similarly, hyperparameters such as depth, random strength, bagging temperature, and L2`
- `Figure 7. Through comprehensive experiments for fine-tuning hyperparameters of the`
- `Figure 7. An example of a candidate whale for hyperparameter optimization.`
- `Figure 8. An example of an input whale population for hyperparameter optimization.`
- `Figure 9. An example of a result of GSWO for hyperparameter optimization.`
- `implementation. This dataset has 41 unique parameters, including the content type, core`
- `Equation (21) formulates the accuracy parameter as the ratio of all correctly classi-`
- `(TP) and erroneously classified normal samples (FP). Equation (23) computes recall as the`
- `and inaccurate predictions. The F1-score (Equation (24)) serves as a balanced measure of`
- `model performance, relying on the harmonic mean of sensitivity and recall. The F1-score,`
- `F1-score = 2 ∗Precision ∗Recall`
- `search, grid search, and Optuna. Table 7 presents the values of CatBoost’s hyperparameters,`
- `Table 7. Results of hyperparameter optimization techniques.`
- `Table 8. Comparison of performance metrics of hyperparameter optimization techniques.`
- `recall, F1-score, and inference time. This approach leverages a GSWO-based technique for`

## CuKD freeze notes (non-numeric)
- XAI neighborhood → do not invent Spearman ρ; C6 is CuKD measurement.
- WSN neighborhood → Almomani WSN-DS lineage.
- Quantization neighborhood → Jacob/C4 PTQ honesty.
- **Numbers only from exact lines above.**

## Verification
- PASS1B: evidence JSON written with exact lines from extract
- PASS2: all `41` quoted lines re-found in full extract (by construction)
- PASS2_TEXT: OK; PASS2_VISUAL_TITLE: OK

## PASS2 independent text re-check (2026-08-06)
- **PASS2_OK** — 41/41 quotes exact-match in full extract

## PASS2 visual title check (2026-08-06)
- page_001.png exists: True
- title word hit ratio vs page-1 text: 1.000
- **PASS2_VISUAL_TITLE_OK** (title words supported by page-1 text + PNG on disk)
