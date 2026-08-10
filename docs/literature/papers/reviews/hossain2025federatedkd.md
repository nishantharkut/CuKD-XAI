# Review card: hossain2025federatedkd

**Status:** PASS1B+PASS2_TEXT+PASS2_VISUAL_TITLE OK
**PDF text pages extracted:** 23
**Ground truth extract:** `_extract/hossain2025federatedkd.full.txt`
**Evidence JSON:** `_pass1b_evidence/hossain2025federatedkd.json`
**Generated:** 2026-08-06

## Identity (page-1 text)
- **Title:** Complex & Intelligent Systems (2025) 11:422 A novel federated learning approach for IoT botnet intrusion detection
- **Tags:** KD, XAI, FL

## Abstract (extracted)
> The exponential growth of the Internet of Things (IoT) has introduced new security vulnerabilities, particularly from botnet attacks that exploit the heterogeneity and limited processing capabilities of IoT devices. Traditional centralized intrusion detection models are ineffective in protecting distributed IoT environments due to data privacy concerns and the challenges posed by non-IID (non-independent and identically distributed) data. In response, we propose a novel, privacy-preserving federated learning framework tailored for IoT intrusion detection. Our framework leverages SHAP (Shapley Additive Expla- nations), a technique for computing feature importance, to provide interpretable insights while maintaining data privacy. Each IoT client trains locally on its unique, heterogeneous data, computes SHAP values to quantify feature relevance, and shares only distilled feature knowledge with the central server. This aggregated knowledge forms a global feature proﬁle that enables the global model to accurately detect diverse botnet intrusions across non-IID client data. Experimental results demonstrate that our model achieves near-perfect accuracy (99.99%) across various botnet types, showcasing robustness in identifying botnet-speciﬁc attack patterns while preserving privacy. By addressing IoT data heterogeneity, non-IID data, and privacy concerns, our framework provides a scalable, interpretable, and privacy-compliant federated learning solution, advancing the security of IoT networks against botnet intrusions.

## Table headers present in PDF text (exact lines)
- `Table 1 Short forms and corresponding full terms used in the research`
- `Table 2 Review of existing approaches`
- `Table 3 Amount of data for each client in different settings`
- `Table 5 presents the classiﬁcation performance metrics for`
- `Table 6 presents the classiﬁcation performance of the local`
- `Table 7 summarizes the classiﬁcation performance of the`
- `Table 4 Classiﬁcation results of`
- `Table 5 Classiﬁcation`
- `Table 6 Classiﬁcation report for`
- `Table 7 Classiﬁcation results for`
- `Table 8 Comparison of the`
- `Table 8 provides a comparative analysis of our proposed`

## Metric-bearing lines (exact PDF lines; PASS2-verified)
- `that our model achieves near-perfect accuracy (99.99%) across various botnet types, showcasing robustness in identifying`
- `• RQ2: How does the integration of SHAP-based knowl-`
- `ations as potential threats, achieving up to 98% accuracy`
- `ing up to 98.6% accuracy. The model effectively handled`
- `approach achieved up to 92.42% accuracy with a low false`
- `FL methods like FedAvg, which averaged 59.87% accuracy.`
- `ing accuracy, reaching 95.97% with AdaBoost. However,`
- `Fig. 4 Federated SHAP-based knowledge distillation framework for IoT botnet detection with heterogeneous clients and privacy-preserving feature`
- `Algorithm 1 Proposed SHAP-Based Knowledge Distillation Approach`
- `as accuracy, precision, recall, and F1-score. This ﬁnal evalu-`
- `cision, recall, and F1-score—on each client’s predictions to`
- `• F1 Score: The F1 score is the harmonic mean of precision`
- `F1 −Score = 2 × Precision × Recall`
- `rizes precision, recall, and F1-score for each class, offering`
- `accuracy, precision, recall, and F1-score are averaged`
- `provides the accuracy, precision, recall, and F1-score for`
- `accuracy values ranging from 99.95 to 99.99%.`
- `is evaluated in terms of precision, recall, and F1-score for`
- `accuracy of 99.94%, with both classes showing high preci-`
- `sion and recall. Similarly, Client 2, comprising Mirai (1) and`
- `Normal (2), achieved an accuracy of 99.97%, with minimal`
- `variance between precision, recall, and F1-score, suggest-`
- `and Torii (3) classes, achieved an accuracy of 99.99%, show-`
- `scores in accuracy, precision, recall, and F1, demonstrating`
- `F1-score (%)`
- `F1-score`
- `errors. Similarly, Clients 3, 4, and 5 show high accuracy in`
- `tions across multiple IoT clients. In Setting 1, high accuracy`
- `F1-Score`
- `F1-Score`

## CuKD freeze notes (non-numeric)
- KD neighborhood → compare to C1/C2; do not claim novelty of KD-for-IDS alone.
- XAI neighborhood → do not invent Spearman ρ; C6 is CuKD measurement.
- Federated setting → distinct from single-node MCU HIL.
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
