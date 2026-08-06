# Review card: alshehri2024sadcnn

**Status:** PASS1B+PASS2_TEXT+PASS2_VISUAL_TITLE OK
**PDF text pages extracted:** 11
**Ground truth extract:** `_extract/alshehri2024sadcnn.full.txt`
**Evidence JSON:** `_pass1b_evidence/alshehri2024sadcnn.json`
**Generated:** 2026-08-06

## Identity (page-1 text)
- **Title:** Digital Object Identifier 10.1109/ACCESS.2024.3380816 A Self-Attention-Based Deep Convolutional
- **Tags:** EdgeIIoT

## Abstract (extracted)
> The Industrial Internet of Things (IIoT) comprises a variety of systems, smart devices, and an extensive range of communication protocols. Hence, these systems face susceptibility to privacy and security challenges, making them prime targets for malicious attacks that can result in harm to the overall system. Privacy breach issues are a notable concern within the realm of IIoT. Various intrusion detection systems based on machine learning (ML) and deep learning (DL) have been introduced to detect malicious activities within these networks and identify attacks. However, traditional ML and DL models encounter significant hurdles when faced with highly imbalanced training data and repetitive patterns within network datasets, hampering their performance in distinguishing between various classes of attacks. To overcome the challenges inherent in existing systems, this paper presents a self-attention-based deep convolutional neural network (SA-DCNN) model designed for monitoring the IIoT networks and detecting malicious activities. The SA mechanism computes the significance value for each input feature, and the DCNN processes these parameters to detect IIoT network behavior. Additionally, a two-step cleaning method has been implemented to eliminate redundancy within the training data, considering both intra-class and cross- class samples. Furthermore, to tackle the issue of underfitting, we have employed a mutual information- based feature filtering method. This method ranks all the features in descending order based on their mutual information and subsequently removes the features with negative impact from the dataset. The performance of the SA-DCNN model is assessed using IoTID20 and Edge-IIoTset datasets. Moreover, the proposed study is demonstrated through a comprehensive

## Table headers present in PDF text (exact lines)
- `TABLE 1. Performance assessment with various layers combination on IoTID20 category.`
- `TABLE 2. Performance assessment with various layers combination on IoTID20 sub-category.`
- `TABLE 3. Performance assessment with various layers combination on Edge-IIoTset category.`
- `TABLE 4. Performance assessment with various layers combination on Edge-IIoTset sub-category.`
- `TABLE 5. Results comparison with other models on IoTID20 category.`
- `TABLE 6. Results comparison with other models on IoTID20 sub-category.`
- `TABLE 7. Results comparison with other models on Edge-IIoTset category.`
- `TABLE 8. Results comparison with other models on Edge-IIoTset`
- `TABLE 9. Performance comparison with related articles.`

## Metric-bearing lines (exact PDF lines; PASS2-verified)
- `attribute [22], and DCNN processes these parameters to`
- `parameters [23], [24]. This process enhances detection per-`
- `evaluation metrics, including precision, recall, F1-score,`
- `accuracy of 77.55% in detecting malicious activities.`
- `98.69% accuracy in classifying attacks.`
- `2018, and Edge-IIoTset datasets. They achieved accuracy`
- `98.88% accuracy in the classification of multi-class sub-`
- `accuracy, precision, recall, and F1-score.`
- `96.89% accuracy, 92.39% precision, 87.83% recall, and`
- `90.05% F1 score on the IoTID20 dataset. Additionally,`
- `it achieves 99.95% accuracy, 99.46% precision, 99.61%`
- `recall, and a 99.53% F1 score on the Edge-IIoTset dataset.`

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
