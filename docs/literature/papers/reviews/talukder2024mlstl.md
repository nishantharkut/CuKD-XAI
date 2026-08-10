# Review card: talukder2024mlstl

**Status:** PASS1B+PASS2_TEXT+PASS2_VISUAL_TITLE OK
**PDF text pages extracted:** 20
**Ground truth extract:** `_extract/talukder2024mlstl.full.txt`
**Evidence JSON:** `_pass1b_evidence/talukder2024mlstl.json`
**Generated:** 2026-08-06

## Identity (page-1 text)
- **Title:** International Journal of Information Security (2024) 23:2139–2158 REGULAR CONTRIBUTION
- **Tags:** WSN

## Abstract (extracted)
> In the domain of cyber-physical systems, wireless sensor networks (WSNs) play a pivotal role as infrastructures, encompassing both stationary and mobile sensors. These sensors self-organize and establish multi-hop connections for communication, collectively sensing, gathering, processing, and transmitting data about their surroundings. Despite their signiﬁcance, WSNs face rapid and detrimental attacks that can disrupt functionality. Existing intrusion detection methods for WSNs encounter challenges such as low detection rates, computational overhead, and false alarms. These issues stem from sensor node resource constraints, data redundancy, and high correlation within the network. To address these challenges, we propose an innovative intrusion detection approach that integrates machine learning (ML) techniques with the Synthetic Minority Oversampling Technique Tomek Link (SMOTE-TomekLink) algorithm. This blend synthesizes minority instances and eliminates Tomek links, resulting in a balanced dataset that signiﬁcantly enhances detection accuracy in WSNs. Additionally, we incorporate feature scaling through standardization to render input features consistent and scalable, facilitating more precise training and detection. To counteract imbalanced WSN datasets, we employ the SMOTE-Tomek resampling technique, mitigating overﬁtting and underﬁtting issues. Our comprehensive evaluation, using the wireless sensor network dataset (WSN-DS) containing 374,661 records, identiﬁes the optimal model for intrusion detection in WSNs. The standout outcome of our research is the remarkable performance of our model. In binary classiﬁcation scenarios, it achieves an accuracy rate of 99.78%, and in multiclass classiﬁcation scenarios, it attains an exceptional accuracy rate of 99.92%. These ﬁn

## Table headers present in PDF text (exact lines)
- `Table 1`
- `Table 2 Label encoding for`
- `Table 3 Binary classiﬁcation`
- `Table 4 Multiclass distribution without and with SMOTE-TomekLink`
- `Table 5`
- `Table 6 Binary performance`
- `Table 7 reveals robust performance across all techniques.`
- `Table 7`
- `Table 8 presents a comprehensive comparison of various`
- `Table 8 Comparison analysis of wireless sensor intrusion detection models in WSN-DS`
- `Table 9 Time complexity of ML models in IDS`

## Metric-bearing lines (exact PDF lines; PASS2-verified)
- `research is the remarkable performance of our model. In binary classiﬁcation scenarios, it achieves an accuracy rate of 99.78%,`
- `and in multiclass classiﬁcation scenarios, it attains an exceptional accuracy rate of 99.92%. These ﬁndings underscore the`
- `accuracy, precision, recall, and F1 score. Through exten-`
- `an accuracy of 92.39%, surpassing other algorithms in com-`
- `that ANN achieved the highest accuracy at 98.56%, followed`
- `Ratio (PDR). Evaluation yielded a 98.29% accuracy, surpass-`
- `impressive performance scores, including a 100% precision`
- `accuracy rate of 97%, surpassing existing solutions. Their`
- `accuracy of 99.95%. Speciﬁcally, it achieved impressive`
- `accuracy rates: 99.99% for Normal, 99.96% for Grayhole,`
- `hole attacks, with 99% accuracy for normal trafﬁc. These`
- `accuracy rate of 97.95% for the gas pipeline data and 97.62%`
- `impressive 96.90% accuracy on the WSN-DS dataset, sur-`
- `outstanding accuracy of 94.55%, highlighting the system’s`
- `sion, recall, and F1-score, were employed to provide a`
- `• F1-Score:`
- `F1-score = 2 · Precision · Recall`
- `note that a higher accuracy, precision, recall, and F1 score`
- `sion, recall, and f1-score performances of various ML models`
- `and 99.72%, respectively. The corresponding precision val-`
- `99.44%, respectively. The corresponding recall values for`
- `and98.93%,respectively.ThecorrespondingF1-scorevalues`
- `Tomek (WiSTL), accuracy, precision, recall, and f1-score`
- `respectively. The corresponding precision values of 99.65%,`
- `The corresponding recall values for these models are 99.65%,`
- `The corresponding F1-score values of 99.65%, 99.78%,`
- `Figure2 compares the accuracy in graphical form of var-`
- `F1-score`
- `Fig. 2 Binary accuracy analysis of ML model for WSNs`
- `outperformed DT with an accuracy rate of 99.69%. This`

## CuKD freeze notes (non-numeric)
- WSN neighborhood → Almomani WSN-DS lineage.
- **Numbers only from exact lines above.**

## Verification
- PASS1B: evidence JSON written with exact lines from extract
- PASS2: all `41` quoted lines re-found in full extract (by construction)
- PASS2_TEXT: OK; PASS2_VISUAL_TITLE: OK

## PASS2 independent text re-check (2026-08-06)
- **PASS2_OK** — 38/38 quotes exact-match in full extract

## PASS2 visual title check (2026-08-06)
- page_001.png exists: True
- title word hit ratio vs page-1 text: 1.000
- **PASS2_VISUAL_TITLE_OK** (title words supported by page-1 text + PNG on disk)
