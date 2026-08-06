# Review card: wisanwanichthan2025kd

**Status:** PASS1B+PASS2_TEXT+PASS2_VISUAL_TITLE OK
**PDF text pages extracted:** 25
**Ground truth extract:** `_extract/wisanwanichthan2025kd.full.txt`
**Evidence JSON:** `_pass1b_evidence/wisanwanichthan2025kd.json`
**Generated:** 2026-08-06

## Identity (page-1 text)
- **Title:** Revised: 15 July 2025 Thammawichai, M. A Lightweight Intrusion Detection System for IoT and
- **Tags:** KD, FL

## Abstract (extracted)
> Deep neural networks (DNNs) are highly effective for intrusion detection systems (IDS) due to their ability to learn complex patterns and detect potential anomalies within the systems. However, their high resource consumption requirements including memory and computation make them difficult to deploy on low-powered platforms. This study explores the possibility of using knowledge distillation (KD) to reduce constraints such as power and hardware consumption and improve real-time inference speed but maintain high detection accuracy in IDS across all attack types. The technique utilizes the transfer of knowledge from DNNs (teacher) models to more lightweight shallow neural network (student) models. KD has been proven to achieve significant parameter reduction (92â€“95%) and faster inference speed (7â€“11%) while improving overall detection performance (up to 6.12%). Experimental results on datasets such as NSL-KDD, UNSW-NB15, CIC-IDS2017, IoTID20, and UAV IDS demonstrate DNN with KDâ€™s effectiveness in achieving high accuracy, precision, F1 score, and area under the curve (AUC) metrics. These findings confirm KDâ€™s ability as a potential edge computing strategy for IoT and UAV devices, which are suitable for resource-constrained environments and lead to real-time anomaly detection for next-generation distributed systems.

## Table headers present in PDF text (exact lines)
- `Table 1. Comparison of key KD-based IDS studies and this work.`
- `Table 1. Cont.`
- `Table 2. NSL-KDD [32] class descriptions and data distribution.`
- `Table 3. UNSW-NB15 [34] class descriptions and data distribution.`
- `Table 4. CIC-IDS2017 [9] class descriptions and data distribution.`
- `Table 5. IoTID20 [37] class descriptions and data distribution.`
- `Table 6. UAV IDS [40] class descriptions and data distribution.`
- `Table 7. Evaluation Metrics used in this Study.`
- `Table 8. Comparison of teacher and student models on the NSL-KDD dataset.`
- `Table 9. Comparison of teacher and student models on the UNSW-NB15 dataset.`
- `Table 10. Comparison of teacher and student models on the CIC-IDS2017 dataset.`
- `Table 11. Comparison of teacher and student models on the IoTID20 dataset.`

## Metric-bearing lines (exact PDF lines; PASS2-verified)
- `(student) models. KD has been proven to achieve significant parameter reduction (92â€“95%)`
- `accuracy, precision, F1 score, and area under the curve (AUC) metrics. These findings`
- `accuracy and its adaptability can be one of the best choices for anomaly-based IDS [9].`
- `(3) show that the distilled student models maintain high detection accuracy while reducing`
- `2.1. Knowledge Distillation-Based IDS`
- `model size and computation cost by 99% and has achieved state-of-the-art accuracy of 94.3%`
- `size by 86% with only 0.4% loss of accuracy and showed better performance than state-of-`
- `labeled and unlabeled data in training. The framework resulted in 98.49% parameter`
- `models while only slightly decreasing in accuracy of around 1â€“2%. Zhu et al. [19] proposed`
- `to obtain accuracy above 98% on the ToN-IoT and IoT-23 datasets while reducing the model`
- `parameters to less than 1% from the teacher model, thus proving ideal for deployment at`
- `model can reduce the number of parameters from around 1.7 million to only one thousand,`
- `while maintaining overall accuracy in every tested dataset. Abbasi et al. [21] demonstrated`
- `KD achieved similar accuracy to larger teacher models from 93.57% to 92.45% but with`
- `2.2. Federated Learning with Knowledge Distillation`
- `(SSFL) scheme with KD. This achieved a precision of 91.73% on IoT traffic datasets com-`
- `FLEKD, which achieved 99.8% accuracy in CIC-IDS2019 and further improved the detec-`
- `They achieved an accuracy of 79% in cases of poisoning attacks on the N-BaIoT dataset and`
- `and Abualkibash [25] proposed KDDT, which is a knowledge distillation-empowered`
- `KD with a pre-trained variational autoencoder, the system achieved F1 scores of 93.1% and`
- `IoT and IIoT networks. Their lightweight method maintained high accuracy, above 95% in`
- `berry Pi in achieving F1-scores up to 0.96, AUC = 0.98, and only 1â€“4% CPU/RAM overhead`
- `On the UAVIDS dataset of real UAV traffic, the FFCNN model attains 98.23% accuracy. He`
- `2.5. Self-Knowledge Distillation and Optimization-Based Distillation`
- `et al. [30] proposed a lightweight self-knowledge distillation model, LNet-SKD, that accom-`
- `plished high accuracy while reducing computation overhead. Li and Yao [15] designed a`
- `achieve up to 99.95% in accuracy in binary classification on KDD CUP99. Wang el al. [31]`
- `Realized 98.49% parameter`
- `achieving > 90% parameter`
- `3.3.2. Knowledge Distillation (KD)`

## CuKD freeze notes (non-numeric)
- KD neighborhood â†’ compare to C1/C2; do not claim novelty of KD-for-IDS alone.
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

## DEEP_VISUAL (manual image pages 001,012,014,015)

- Teacher 256-128-64-32; student 32-16; α=0.4 T=3
- Table 8 NSL-KDD (image): Teacher 224657 params Acc 77.22 F1 73.05; Student+KD 13649 Acc 76.73 F1 71.83
- Workstation RTX 2070 only — not MCU HIL

