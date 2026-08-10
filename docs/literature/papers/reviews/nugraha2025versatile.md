# Review card: nugraha2025versatile

**Status:** PASS1B+PASS2_TEXT+PASS2_VISUAL_TITLE OK
**PDF text pages extracted:** 26
**Ground truth extract:** `_extract/nugraha2025versatile.full.txt`
**Evidence JSON:** `_pass1b_evidence/nugraha2025versatile.json`
**Generated:** 2026-08-06

## Identity (page-1 text)
- **Title:** Annals of Telecommunications (2025) 80:1095–1120 A versatile XAI-based framework for efﬁcient and explainable intrusion
- **Tags:** XAI

## Abstract (extracted)
> Artiﬁcialintelligence(AI)-basedintrusiondetectionsystems(IDSs)markedlyadvancenetworksecuritybyleveragingmachine learning (ML) and deep learning (DL) models for accurate, adaptive threat detection. Their main drawback, however, is an inherent “black-box” character that impedes trust, traceability, and regulatory compliance. To overcome this limitation, we propose an efﬁcient explainable-AI (XAI) framework that enhances both robustness and interpretability. The two-stage process ﬁrst couples a statistical selector (ANOVA) with global SHAP scores to retain only the ten most informative features, an approximately 70% dimensionality reduction, then retrains a lightweight XGBoost detector whose decisions are explained locally by SHAP and LIME. Cross-validating the two explanation modalities adds a reliability check absent from earlier hybrids, while the inclusion of a time-efﬁciency evaluation for explanation generation provides a new performance dimension that prior XAI-IDS studies have not addressed. To our knowledge, this is the ﬁrst framework to jointly apply dual-stage statistical–model-based feature selection and SHAP–LIME cross-validation in IDS, enabling near-real-time explainability without sacriﬁcing accuracy. Comprehensive experiments on three representative traces, CIC-DDoS2019 (legacy IP DDoS), CICIoT2023 (IoT malware), and 5G PFCP (control-plane attacks), conﬁrm the framework’s versatility: it sustains an F1 Score of at least 99 % while accelerating LIME explanation time from 36 to 4.9 s, an 87 % speed-up. These results demonstrate that high detection accuracy and transparent, near-real-time interpretability can be achieved simultaneously in modern IDS deployments.

## Table headers present in PDF text (exact lines)
- `Table 1 Comparison of recent explainable IDS frameworks`
- `Table 2 Comparison of recent`
- `Table 3 Summary of the`
- `Table 4 Parameter grid used for XGBoost hyperparameter tuning with`
- `Table 5 Most important`
- `Table 6 Most important features as identiﬁed in the ﬁrst stage (continued)`
- `Table 7 Performance metric`
- `Table 8 Performance metric`
- `Table 9 Performance metric`
- `Table 10 Performance metric`
- `Table 15 complements the XGBoost results in Table9 with`
- `Table 12 Confusion matrix - SYN Flood (Top-10 features)`

## Metric-bearing lines (exact PDF lines; PASS2-verified)
- `without sacriﬁcing accuracy. Comprehensive experiments on three representative traces, CIC-DDoS2019 (legacy IP DDoS),`
- `CICIoT2023 (IoT malware), and 5G PFCP (control-plane attacks), conﬁrm the framework’s versatility: it sustains an F1 Score`
- `In the ﬁrst stage, ANOVA [7] with SHAP Global Expla-`
- `3. An extended efﬁciency evaluation that measures SHAP`
- `high precision and recall on SCVIC-APT-2021, though the`
- `evaluation, yielding a DNN with marginal F1 gains but only`
- `synthesizes prior studies. Mane and Rao [16] applied SHAP`
- `ability and accuracy. Mahbooba et al. [17] employed decision`
- `et al. [18] integrated CNN and LSTM with SHAP and LIME`
- `explanations. Marcilio-Jr and Eler [22] utilized SHAP for`
- `latency is not analyzed [24]. Most recently, a collabora-`
- `CIC-IoT 2023, achieving high accuracy while preserving`
- `as Fast TreeSHAP by Yang [31], accelerate SHAP calcu-`
- `[34] expanded the capabilities of SHAP to compute higher-`
- `iment, PCA and MI top-10 ﬁlters achieved higher F1 Score`
- `(MI); ANOVA reached a respectable F1 Score of 0.997`
- `and global SHAP values to reduce the input space to the 10`
- `F1 Score ≥99% across CIC-DDoS2019, CICIoT2023, and`
- `2: Output: Trained model M, explanations (eSHAP, eLIME), and con-`
- `9: Obtain SHAP importances from`
- `10: Normalize and combine (40% ANOVA + 60% SHAP) scores`
- `(40% ANOVA + 60% SHAP), producing a ranked list from`
- `about 2% of ﬂows require analyst review. Because SHAP`
- `3.2.1 SHAP`
- `SHAP, introduced by Lundberg and Lee [3], employs Shap-`
- `[3]. SHAP’s advantages lie in its game-theoretical founda-`
- `ANOVA and SHAP Global Explanation, assigning a 60%`
- `weight to SHAP results and 40% to ANOVA results. This`
- `accuracy. Preliminary trials with adjacent ratios (50:50 and`
- `for each parameter is summarized in Table 4, with the ﬁnal`

## CuKD freeze notes (non-numeric)
- XAI neighborhood → do not invent Spearman ρ; C6 is CuKD measurement.
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
