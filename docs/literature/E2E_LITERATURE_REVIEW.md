# E2E Literature Review (double-verified corpus)

**Generated:** 2026-08-06
**Method:** For each on-disk PDF: (1) full text extract → exact metric/table lines → review card; (2) independent re-check that every quoted line appears in extract; (3) page-1 title word check + page_001.png present.
**PASS counts:** VERIFIED_2PASS=42; INVALID_PDF=1
**Claim authority for CuKD results:** `results/paper_strength_e2e/06_claim_freeze.json` only.
**Manuscript map:** `docs/literature/MANUSCRIPT_POSITIONING.md`
**Per-paper cards:** `docs/literature/papers/reviews/<id>.md`
**Extracts:** `docs/literature/papers/_extract/<id>.full.txt`

## Integrity rules used
1. No numeric claim in a card unless the **exact line** exists in the PDF text extract.
2. Full pipeline run **twice** (PASS1B+PASS2, then re-run PASS1B+PASS2).
3. Wrong PDF detected and **excluded** (ferrag2022edgeiiot).
4. CuKD experimental numbers only from freeze — never from literature cards.

## Corpus matrix

| ID | Status | Pages | #metric lines | #table headers | Tags | Title (extracted) |
|---|---|---:|---:|---:|---|---|
| `adebayo2018sanity` | VERIFIED_2PASS | 30 | 12 | 0 | XAI | Sanity Checks for Saliency Maps Julius Adebayo∗, Justin Gilmer♯, Michael Muelly♯, Ian Good |
| `adhane2025explainkd` | VERIFIED_2PASS | 10 | 14 | 2 | KD | On Explaining Knowledge Distillation: Measuring and Visualising the Knowledge Transfer Pro |
| `adjewa2026seed` | VERIFIED_2PASS | 22 | 30 | 8 | XAI,quant | Revised: 23 December 2025 distributed under the terms and conditions of the Creative Commo |
| `alfarra2025local` | VERIFIED_2PASS | 14 | 30 | 12 | MCU,WSN,gateway | cient Hybrid Learning for Secure Wireless Sensor Networks ECTI Transactions on Computer an |
| `almomani2016wsnds` | VERIFIED_2PASS | 16 | 8 | 12 | WSN | WSN-DS: A Dataset for Intrusion Detection Systems in Wireless Sensor Networks Iman Almoman |
| `alqahtani2019gxgboost` | VERIFIED_2PASS | 20 | 18 | 12 | WSN | A Genetic-Based Extreme Gradient Boosting Model for Detecting Intrusions in Wireless Senso |
| `alshehri2024sadcnn` | VERIFIED_2PASS | 11 | 12 | 9 | EdgeIIoT | Digital Object Identifier 10.1109/ACCESS.2024.3380816 A Self-Attention-Based Deep Convolut |
| `banbury2021tinyml` | VERIFIED_2PASS | 8 | 1 | 4 | MCU | BENCHMARKING TINYML SYSTEMS: CHALLENGES AND DIRECTION Colby R. Banbury 1 Vijay Janapa Redd |
| `benaddi2025arxiv` | VERIFIED_2PASS | 6 | 30 | 0 | KD,XAI,quant | Lightweight Intrusion Detection in IoT via SHAP-Guided Feature Pruning and Knowledge-Disti |
| `bengio2009curriculum` | VERIFIED_2PASS | 8 | 8 | 0 | XAI | Curriculum Learning J´erˆome Louradour1,2 Ronan Collobert3 |
| `birahim2025pso` | VERIFIED_2PASS | 20 | 30 | 7 | XAI,WSN | Digital Object Identifier 10.1109/ACCESS.2025.3528341 Intrusion Detection for Wireless Sen |
| `cantini2024dixtill` | VERIFIED_2PASS | 17 | 30 | 4 | KD,XAI,quant | © The Author(s) 2024, corrected publication 2024. Open Access  This article is licensed un |
| `chawla2002smote` | VERIFIED_2PASS | 37 | 13 | 7 |  | SMOTE: Synthetic Minority Over-sampling Technique Nitesh V. Chawla |
| `diab2025hardware` | VERIFIED_2PASS | 6 | 30 | 0 | EdgeIIoT,gateway | Intrusion Detection on Resource-Constrained IoT Devices with Hardware-Aware ML and DL |
| `ferrag2022edgeiiot` | INVALID_PDF | 6 | 0 | 0 |  | Radical-induced Hetero-Nuclear Mixing and Low-ﬁeld 13C Relaxation in Solid Pyruvic Acid |
| `gao2026lightweight` | VERIFIED_2PASS | 20 | 15 | 6 | EdgeIIoT | Revised: 12 February 2026 distributed under the terms and conditions of the Creative Commo |
| `ghadi2024review` | VERIFIED_2PASS | 21 | 4 | 8 | WSN | Digital Object Identifier 10.1109/ACCESS.2024.3355312 Machine Learning Solutions for the S |
| `guo2017calibration` | VERIFIED_2PASS | 14 | 8 | 2 | KD | On Calibration of Modern Neural Networks Chuan Guo * 1 Geoff Pleiss * 1 Yu Sun * 1 Kilian  |
| `guo2024whykd` | VERIFIED_2PASS | 29 | 24 | 1 | KD,XAI | Why does Knowledge Distillation Work? Rethink its Attention and Fidelity Mechanism |
| `hasan2025autoencoder` | VERIFIED_2PASS | 8 | 17 | 9 | EdgeIIoT | Enhanced Intrusion Detection in IIoT Networks: A Lightweight Approach with Autoencoder-Bas |
| `hinton2015distilling` | VERIFIED_2PASS | 9 | 21 | 6 | KD | arXiv:1503.02531v1  [stat.ML]  9 Mar 2015 Distilling the Knowledge in a Neural Network |
| `hossain2025federatedkd` | VERIFIED_2PASS | 23 | 30 | 12 | KD,XAI,FL | Complex & Intelligent Systems (2025) 11:422 A novel federated learning approach for IoT bo |
| `ishtiaq2025cstafnet` | VERIFIED_2PASS | 9 | 29 | 6 | XAI | CST-AFNet: A dual attention-based deep learning framework for intrusion detection in IoT n |
| `jacob2018integer` | VERIFIED_2PASS | 10 | 30 | 9 | quant | Quantization and Training of Neural Networks for Efﬁcient Integer-Arithmetic-Only Inferenc |
| `javed2024thermostat` | VERIFIED_2PASS | 15 | 30 | 10 | XAI,MCU,gateway | Qureshi, A.-u.-H.; Jawad, M.; Arshad, J.; Larijani, H. Embedding Tree-Based Intrusion Dete |
| `krishna2022disagreement` | VERIFIED_2PASS | 34 | 26 | 7 | XAI | The Disagreement Problem in Explainable Machine Learning: A Practitioner’s Perspective |
| `lundberg2017shap` | VERIFIED_2PASS | 10 | 30 | 0 | XAI | A Uniﬁed Approach to Interpreting Model Scott M. Lundberg Paul G. Allen School of Computer |
| `misrak2025quantization` | VERIFIED_2PASS | 34 | 30 | 9 | quant | © The Author(s) 2025. Open Access  This article is licensed under a Creative Commons Attri |
| `nguyen2024gswo` | VERIFIED_2PASS | 26 | 30 | 11 | XAI,WSN,quant | Yoo, M. Enhancing Intrusion Detection in Wireless Sensor Networks Using a GSWO-CatBoost |
| `nugraha2025versatile` | VERIFIED_2PASS | 26 | 30 | 12 | XAI | Annals of Telecommunications (2025) 80:1095–1120 A versatile XAI-based framework for efﬁci |
| `pandey2025tabu` | VERIFIED_2PASS | 21 | 30 | 7 | WSN | Enhancing intrusion detection in wireless sensor networks using a Tabu search based optimi |
| `peng2025fdids` | VERIFIED_2PASS | 31 | 30 | 12 | KD,EdgeIIoT,FL | Revised: 22 June 2025 FD-IDS: Federated Learning with Knowledge Distillation for Intrusion |
| `salmi2022cnnlstm` | VERIFIED_2PASS | 6 | 12 | 1 | XAI | CNN-LSTM based Approach for DDoS Detection Tahani Alasmari |
| `seyedkolaei2025cnn` | VERIFIED_2PASS | 21 | 30 | 12 | EdgeIIoT,quant,gateway | Tolulope Odeyomi and Revised: 20 May 2025 Seyedkolaei, A.A.; |
| `stanton2021kd` | VERIFIED_2PASS | 21 | 30 | 2 | KD | Does Knowledge Distillation Really Work? Polina Kirichenko Alexander A. Alemi |
| `sze2017efficient` | VERIFIED_2PASS | 32 | 30 | 0 | XAI,quant | Efﬁcient Processing of Deep Neural Networks: A Tutorial and Survey Vivienne Sze, Senior Me |
| `talukder2024mlstl` | VERIFIED_2PASS | 20 | 30 | 11 | WSN | International Journal of Information Security (2024) 23:2139–2158 REGULAR CONTRIBUTION |
| `talukder2025hybrid` | VERIFIED_2PASS | 23 | 30 | 12 | WSN | A hybrid machine learning model for intrusion detection in wireless sensor networks levera |
| `ticnna_hybrid_iot` | VERIFIED_2PASS | 15 | 30 | 12 | XAI | Digital Object Identifier 10.1109/ACCESS.2026.3663379 TICNN—A Hybrid Light-Weight CNN for  |
| `wisanwanichthan2025kd` | VERIFIED_2PASS | 25 | 30 | 12 | KD,FL | Revised: 15 July 2025 Thammawichai, M. A Lightweight Intrusion Detection System for IoT an |
| `xiao2025local` | VERIFIED_2PASS | 15 | 27 | 10 | WSN | Signal, Image and Video Processing (2025) 19:1335 Metaheuristically optimized deep soft-vo |
| `yagiz2025lens` | VERIFIED_2PASS | 52 | 30 | 12 | KD,XAI | LENS-XAI: Redefining Lightweight and Explainable Network Security through Knowledge Distil |
| `yang2023selfkd` | VERIFIED_2PASS | 6 | 16 | 0 | KD | A Lightweight Approach for Network Intrusion Detection based on Self-Knowledge Distillatio |

## Cluster map for manuscript related work

### KD / distillation
`adhane2025explainkd`, `benaddi2025arxiv`, `cantini2024dixtill`, `guo2017calibration`, `guo2024whykd`, `hinton2015distilling`, `hossain2025federatedkd`, `peng2025fdids`, `stanton2021kd`, `wisanwanichthan2025kd`, `yagiz2025lens`, `yang2023selfkd`

### XAI / SHAP
`adebayo2018sanity`, `adjewa2026seed`, `benaddi2025arxiv`, `bengio2009curriculum`, `birahim2025pso`, `cantini2024dixtill`, `guo2024whykd`, `hossain2025federatedkd`, `ishtiaq2025cstafnet`, `javed2024thermostat`, `krishna2022disagreement`, `lundberg2017shap`, `nguyen2024gswo`, `nugraha2025versatile`, `salmi2022cnnlstm`, `sze2017efficient`, `ticnna_hybrid_iot`, `yagiz2025lens`

### MCU / embedded
`alfarra2025local`, `banbury2021tinyml`, `javed2024thermostat`

### WSN
`alfarra2025local`, `almomani2016wsnds`, `alqahtani2019gxgboost`, `birahim2025pso`, `ghadi2024review`, `nguyen2024gswo`, `pandey2025tabu`, `talukder2024mlstl`, `talukder2025hybrid`, `xiao2025local`

### Edge-IIoT
`alshehri2024sadcnn`, `diab2025hardware`, `gao2026lightweight`, `hasan2025autoencoder`, `peng2025fdids`, `seyedkolaei2025cnn`

### Quantization
`adjewa2026seed`, `benaddi2025arxiv`, `cantini2024dixtill`, `jacob2018integer`, `misrak2025quantization`, `nguyen2024gswo`, `seyedkolaei2025cnn`, `sze2017efficient`

### Federated
`hossain2025federatedkd`, `peng2025fdids`, `wisanwanichthan2025kd`

### Other / foundational
`chawla2002smote`

## Freeze-safe positioning (summary)
See `MANUSCRIPT_POSITIONING.md` for full rules. Short form:
- **Do not claim first KD-for-IDS** (Wisan, Yang, Yagiz, Benaddi, Peng present).
- **Do not claim first MCU IDS** (Javed CatBoost on ESP32 present).
- **Do claim** RF→tiny-NN on WSN-DS + train-only/FG protocol ladder + dual identity + dual-board integer HIL + honest PTQ drop + failed SHAP rank (C1–C10 / not X1–X5).

## High-value verified neighbors (read full card before citing numbers)
### `javed2024thermostat`
- Title: Qureshi, A.-u.-H.; Jawad, M.; Arshad, J.; Larijani, H. Embedding Tree-Based Intrusion Detection System in Smart
- Tags: XAI, MCU, gateway
- Abstract (trunc): IoT devices with limited resources, and in the absence of gateways, become vulnerable to various attacks, such as denial of service (DoS) and man-in-the-middle (MITM) attacks. Intrusion detection systems (IDS) are designed to detect and respond to these threats in IoT environments. While machine learning-based IDS have typically been deployed at the edge (gateways) or in the cloud, in the absence …
- Sample exact metric lines:
  - `intrusion detection. The results demonstrated that the IDS achieved an accuracy of 98.71% for binary`
  - `classification with an inference time of 276 microseconds, and an accuracy of 97.51% for multi-`
  - `XGBoost-based IDS for binary classification in an ESP32 (https://www.espressif.com/en/`
  - `products/socs/esp32 (accessed on 6 October 2024))-powered smart thermostat. The smart`
  - `com/espressif/esp-lwip (accessed on 6 October 2024)) library for the ESP32 provides`
  - `the novel implementation of CatBoost-based IDS on the ESP32 microcontroller. CatBoost`
  - `capabilities of devices like the ESP32.`
  - `credibility index. The experimental results showed that the accuracy of PCIDS was 94.7%.`
- Card: `reviews/javed2024thermostat.md`

### `wisanwanichthan2025kd`
- Title: Revised: 15 July 2025 Thammawichai, M. A Lightweight Intrusion Detection System for IoT and
- Tags: KD, FL
- Abstract (trunc): Deep neural networks (DNNs) are highly effective for intrusion detection systems (IDS) due to their ability to learn complex patterns and detect potential anomalies within the systems. However, their high resource consumption requirements including memory and computation make them difficult to deploy on low-powered platforms. This study explores the possibility of using knowledge distillation (KD)…
- Sample exact metric lines:
  - `(student) models. KD has been proven to achieve significant parameter reduction (92–95%)`
  - `accuracy, precision, F1 score, and area under the curve (AUC) metrics. These findings`
  - `accuracy and its adaptability can be one of the best choices for anomaly-based IDS [9].`
  - `(3) show that the distilled student models maintain high detection accuracy while reducing`
  - `2.1. Knowledge Distillation-Based IDS`
  - `model size and computation cost by 99% and has achieved state-of-the-art accuracy of 94.3%`
  - `size by 86% with only 0.4% loss of accuracy and showed better performance than state-of-`
  - `labeled and unlabeled data in training. The framework resulted in 98.49% parameter`
- Card: `reviews/wisanwanichthan2025kd.md`

### `peng2025fdids`
- Title: Revised: 22 June 2025 FD-IDS: Federated Learning with Knowledge Distillation for Intrusion
- Tags: KD, EdgeIIoT, FL
- Abstract (trunc): With the rapid advancement of Internet of Things (IoT) technology, intrusion detection systems (IDSs) have become pivotal in ensuring network security. However, the data produced by IoT devices is typically sensitive and tends to display non-independent and identically distributed (Non-IID) properties. These factors impose significant limitations on the application of traditional centralized learn…
- Sample exact metric lines:
  - `accuracy on the UNSW-NB15 dataset, while BiLSTM attained 96.41%. On the BoT-IoT`
  - `dataset, both models excelled, achieving a remarkable 99.99% accuracy. Fatani et al. [25]`
  - `data. Using three clients, the RNN model achieved a global detection accuracy of 91.87%.`
  - `a fifteen-class task on Edge-IIoTset, the global model achieved 89.91% accuracy after 10`
  - `training rounds. On the InSDN dataset, F-BIDS achieved a global model accuracy of 99.91%`
  - `after 50 rounds, with the lowest client accuracy exceeding 99.70%. Nobakht et al. [29] pro-`
  - `performance degradation caused by data heterogeneity, achieving an F1 score improvement`
  - `and CNN-LSTM. In the FL scenario, the highest accuracy for the 15-class classification task`
- Card: `reviews/peng2025fdids.md`

### `benaddi2025arxiv`
- Title: Lightweight Intrusion Detection in IoT via SHAP-Guided Feature Pruning and Knowledge-Distilled Kronecker Networks
- Tags: KD, XAI, quant
- Abstract (trunc): The widespread deployment of Internet of Things (IoT) devices requires intrusion detection systems (IDS) with high accuracy while operating under strict resource constraints. Con- ventional deep learning IDS are often too large and computation- ally intensive for edge deployment. We propose a lightweight IDS that combines SHAP-guided feature pruning with knowledge- distilled Kronecker networks. A …
- Sample exact metric lines:
  - `teacher yet sustains macro-F1 above 0.986 with millisecond-level`
  - `sumption without compromising accuracy [1].`
  - `tectures capable of balancing efficiency and accuracy [2]. Ap-`
  - `[4], leaving the interaction among latency, scalability, and`
  - `efficiency, and accuracy [2].`
  - `dimensional convolutional detector achieves near-perfect F1-`
  - `orders of magnitude and still delivers accuracy above 99%`
  - `The “SHAP Feature Ranking & Pruning” module in Fig. 1`
- Card: `reviews/benaddi2025arxiv.md`

### `yang2023selfkd`
- Title: A Lightweight Approach for Network Intrusion Detection based on Self-Knowledge Distillation
- Tags: KD
- Abstract (trunc): Network Intrusion Detection (NID) works as a ker- nel technology for the security network environment, obtaining extensive research and application. Despite enormous efforts by researchers, NID still faces challenges in deploying on resource- constrained devices. To improve detection accuracy while reduc- ing computational costs and model storage simultaneously, we propose a lightweight intrusion …
- Sample exact metric lines:
  - `Fig. 1. F1 score v.s. the number of model parameters on NSL-KDD datasets.`
  - `trade-off between efficiency and accuracy (See Fig. 1). Our`
  - `distillation. Wang et al. [14] proposed a knowledge distillation`
  - `parameters, which is less than classical ones Ci × (K2 × Co)`
  - `Fig. 3. The impact of hyper-parameters.`
  - `Accuracy, Precision, Recall, and F1 score are used to eval-`
  - `FLOPs of 194.58K (↓63.1%). Meanwhile, our self-distillation`
  - `proving the accuracy of LNet by 2.1% / 1.6% on NSL-KDD`
- Card: `reviews/yang2023selfkd.md`

### `diab2025hardware`
- Title: Intrusion Detection on Resource-Constrained IoT Devices with Hardware-Aware ML and DL
- Tags: EdgeIIoT, gateway
- Abstract (trunc): This paper proposes a hardware-aware intrusion detection system (IDS) for Internet of Things (IoT) and Industrial IoT (IIoT) networks; it targets scenarios where classification is essential for fast, privacy-preserving, and resource-efficient threat detection. The goal is to optimize both tree-based machine learning (ML) models and compact deep neural networks (DNNs) within strict edge-device cons…
- Sample exact metric lines:
  - `flash, RAM, and compute limits: LightGBM achieves 95.3%`
  - `accuracy using 75 KB flash and 1.2 K operations, while the HW-`
  - `NAS–optimized CNN reaches 97.2% with 190 KB flash and 840 K`
  - `within 30 ms and that CNNs remain suitable when accuracy`
  - `reporting satisfactory accuracy on public benchmarks [4]–`
  - `vary with hyperparameter configurations [8]. Deep models,`
  - `accurate models: LightGBM achieves 95.3% accuracy`
  - `using just 75 KB of flash and 1.2 K operations, while the`
- Card: `reviews/diab2025hardware.md`

### `alfarra2025local`
- Title: cient Hybrid Learning for Secure Wireless Sensor Networks ECTI Transactions on Computer and Information Technology
- Tags: MCU, WSN, gateway
- Abstract (trunc): Article information: Wireless Sensor Networks (WSNs) power critical applications from envi- ronmental monitoring to Internet-of-Medical-Things healthcare yet their tiny batteries and low-end microcontrollers leave them exposed to net- work-layer Denial-of-Service (DoS) attacks such as Blackhole, Grayhole, Flooding and TDMA scheduling. Signature IDSs miss zero-day variants and shallow machine-learn…
- Sample exact metric lines:
  - `work, show that the scheme achieves 98 % accuracy, 0.93 macro-F1 and`
  - `minority-class recalls of 0.840.95 while extending network lifetime (T50)`
  - `% overall accuracy [2, 7, 9]. Never-theless, their false-`
  - `accuracy on the WSN-DS benchmark [3, 9, 12]. The`
  - `to suppress false positives yet preserve recall [5, 7,`
  - `timeout elapses, keeping latency well below the 250`
  - `∼96 % accuracy on the WSN-DS dataset while reduc-`
  - `LSTM achieves >95 % detection accuracy for packet-`
- Card: `reviews/alfarra2025local.md`

### `almomani2016wsnds`
- Title: WSN-DS: A Dataset for Intrusion Detection Systems in Wireless Sensor Networks Iman Almomani,1,2 Bassam Al-Kasasbeh,2 and Mousa AL-Akhras2,3
- Tags: WSN
- Sample exact metric lines:
  - `DS improved the ability of IDS to achieve higher classification accuracy rate. WEKA toolbox was used with holdout and 10-Fold`
  - `Table 5: Ns-2 simulation parameters.`
  - `Simulation parameters are summarized in Table 5.`
  - `Table 7: Parameters for MLP neural network classifier.`
  - `Table 7 shows the parameters and the values used in`
  - `hidden layer, an overall classification accuracy of 97.5431%`
  - `From Table 9, it can be concluded that the accuracy`
  - `architecture. From Table 10, it can be shown that the accuracy`
- Card: `reviews/almomani2016wsnds.md`

### `stanton2021kd`
- Title: Does Knowledge Distillation Really Work? Polina Kirichenko Alexander A. Alemi
- Tags: KD
- Abstract (trunc): Knowledge distillation is a popular technique for training a small student network to emulate a larger teacher model, such as an ensemble of networks. We show that while knowledge distillation can improve student generalization, it does not typically work as it is commonly understood: there often remains a surprisingly large discrepancy between the predictive distributions of the teacher and the s…
- Sample exact metric lines:
  - `distillation [20] argues that Bucil˘a et al. [5] “demonstrate convincingly that the knowledge acquired`
  - `Conversely, in Figure 1 we show that with modern architectures knowledge distillation can lead to`
  - `distillation dataset. In Section 6 we investigate the hypothesis that low ﬁdelity is an optimization`
  - `Figure 1: Evaluating the ﬁdelity of knowledge distillation. The effect of enlarging the CIFAR-100`
  - `ResNet-56 networks. Student ﬁdelity increases as the dataset grows, but test accuracy decreases.`
  - `Knowledge distillation can improve model efﬁciency [34, 40], unsupervised domain adaptation [33],`
  - `early work proposed distilling ensembles of shallow networks into a single network [49], an idea`
  - `which resonates with more recent work on the distillation of deep ensembles [2, 7, 41, 45, 47].`
- Card: `reviews/stanton2021kd.md`

### `yagiz2025lens`
- Title: LENS-XAI: Redefining Lightweight and Explainable Network Security through Knowledge Distillation and
- Tags: KD, XAI
- Sample exact metric lines:
  - `environments. Gaspar et al. [8] explored the integration of SHAP (SHapley`
  - `2.4. Knowledge Distillation & VAEs`
  - `tion accuracy. Sindiramutty et al. [15] effectively combined these techniques`
  - `recall, precision, and F1 scores, while maintaining a lightweight computa-`
  - `3.4. Knowledge Distillation for Model Optimization`
  - `mance, we employ Knowledge Distillation [32], transferring the “knowledge”`
  - `3.4.1. Distillation Setup`
  - `3.4.2. Distillation Loss`
- Card: `reviews/yagiz2025lens.md`

### `nguyen2024gswo`
- Title: Yoo, M. Enhancing Intrusion Detection in Wireless Sensor Networks Using a GSWO-CatBoost
- Tags: XAI, WSN, quant
- Abstract (trunc): Intrusion detection systems (IDSs) in wireless sensor networks (WSNs) rely heavily on effective feature selection (FS) for enhanced efficacy. This study proposes a novel approach called Genetic Sacrificial Whale Optimization (GSWO) to address the limitations of conventional methods. GSWO combines a genetic algorithm (GA) and whale optimization algorithms (WOA) modified by applying a new three-popu…
- Sample exact metric lines:
  - `accuracy rates of 99.65%, 99.99%, 99.76%, and 99.74% for WSN-DS, WSNBFSF, NSL-KDD, and`
  - `performance and predictive accuracy of these models [20]. This process involves adjusting`
  - `accuracy rate of 98.79%. Notably, compared to ML models, deep learning models introduce`
  - `performance in terms of the F1-score for the four types of network attacks compared with`
  - `PSO and achieved an overall accuracy of 98%, according to the experimental evaluation`
  - `nary and multi-class attacks with notable accuracy, precision, recall, and F1-score metrics.`
  - `networks. The GA-RF approach attained a test accuracy of 87.61% and an Area Under`
  - `2.3. Fine-Tuning Hyperparameters for Machine Learning Model`
- Card: `reviews/nguyen2024gswo.md`

### `birahim2025pso`
- Title: Digital Object Identifier 10.1109/ACCESS.2025.3528341 Intrusion Detection for Wireless Sensor Network
- Tags: XAI, WSN
- Abstract (trunc): Wireless Sensor Networks (WSN) play a pivotal role in various domains, including monitoring, security, and data transmission. However, their susceptibility to intrusions poses a significant challenge. This paper proposes a novel Intrusion Detection System (IDS) leveraging Particle Swarm Optimization (PSO) and an ensemble machine learning approach combining Random Forest (RF), Decision Tree (DT), a…
- Sample exact metric lines:
  - `improvements in detection accuracy, precision, recall, and F1 score while providing clear, interpretable`
  - `accuracy of 99.73%, with precision, recall, and F1 score values of 99.72% each, outperforming existing`
  - `resulting in an accuracy of 99.16% with both the LightGBM`
  - `technique and got an accuracy of 92.39%. After applying`
  - `SMOTE, the accuracy improved slightly to 92.57%, under-`
  - `[15]. The system’s accuracy and efficiency were improved`
  - `requirements, achieving an accuracy of 98.29% in preventing`
  - `identify intrusions. With an accuracy of 95.53%, the system`
- Card: `reviews/birahim2025pso.md`

### `salmi2022cnnlstm`
- Title: CNN-LSTM based Approach for DDoS Detection Tahani Alasmari
- Tags: XAI
- Abstract (trunc): Distributed Denial of Service (DDoS) attacks have become increasingly common, causing financial and reputational losses for organizations. Despite the existence of numerous conventional detection solutions, DDoS attacks continue to rise in frequency, demanding effective models to detect and prevent them. This paper focuses on developing a machine learning- based approach for DDoS attack detection.…
- Sample exact metric lines:
  - `reached an accuracy of 99% in detecting DDOS attacks.`
  - `96.43% accuracy rate.`
  - `98.9% accuracy rate. Moreover, a machine learning-`
  - `the required shape. Our input data has 67 features. So,`
  - `the reshape layer makes an input shape of (67, 1), where`
  - `(2*2*64) = 256 trainable parameters. And each of them`
  - `Figure 5 shows the accuracy of all machine learning`
  - `Fig. 5. Comparison of the accuracy of different ML algorithms.`
- Card: `reviews/salmi2022cnnlstm.md`

### `hossain2025federatedkd`
- Title: Complex & Intelligent Systems (2025) 11:422 A novel federated learning approach for IoT botnet intrusion detection
- Tags: KD, XAI, FL
- Abstract (trunc): The exponential growth of the Internet of Things (IoT) has introduced new security vulnerabilities, particularly from botnet attacks that exploit the heterogeneity and limited processing capabilities of IoT devices. Traditional centralized intrusion detection models are ineffective in protecting distributed IoT environments due to data privacy concerns and the challenges posed by non-IID (non-inde…
- Sample exact metric lines:
  - `that our model achieves near-perfect accuracy (99.99%) across various botnet types, showcasing robustness in identifying`
  - `• RQ2: How does the integration of SHAP-based knowl-`
  - `ations as potential threats, achieving up to 98% accuracy`
  - `ing up to 98.6% accuracy. The model effectively handled`
  - `approach achieved up to 92.42% accuracy with a low false`
  - `FL methods like FedAvg, which averaged 59.87% accuracy.`
  - `ing accuracy, reaching 95.97% with AdaBoost. However,`
  - `Fig. 4 Federated SHAP-based knowledge distillation framework for IoT botnet detection with heterogeneous clients and privacy-preserving feature`
- Card: `reviews/hossain2025federatedkd.md`

### `jacob2018integer`
- Title: Quantization and Training of Neural Networks for Efﬁcient Integer-Arithmetic-Only Inference
- Tags: quant
- Abstract (trunc): The rising popularity of intelligent mobile devices and the daunting computational cost of deep learning-based models call for efﬁcient and accurate on-device inference schemes. We propose a quantization scheme that allows inference to be carried out using integer-only arithmetic, which can be implemented more efﬁciently than ﬂoating point inference on commonly available integer-only hard- ware. W…
- Sample exact metric lines:
  - `[29], are all over-parameterized by design in order to extract`
  - `Top 1 Accuracy`
  - `the fast integer-arithmetic circuits in common CPUs to deliver an improved latency-vs-accuracy tradeoff (section 4). The ﬁgure compares`
  - `and just a few parameters (bias vectors) as 32-bit integers.`
  - `relative accuracy. Multiplication by M0 can thus be imple-`
  - `to preserve good end-to-end neural network accuracy6.`
  - `smoothing parameter being close to 1 so that observed`
  - `a much higher range and precision compared to the 8 bit`
- Card: `reviews/jacob2018integer.md`

### `krishna2022disagreement`
- Title: The Disagreement Problem in Explainable Machine Learning: A Practitioner’s Perspective
- Tags: XAI
- Sample exact metric lines:
  - `et al., 2016b), SHAP (Lundberg & Lee, 2017)) and gradient-based methods (e.g., Gradient times Input`
  - `terms of accuracy (Ribeiro et al., 2016b). This has led to significant interest in post hoc explanation methods,`
  - `SHAP, and gradient-based methods) in their day-to-day workflow. 19 participants (76%) were male, and 6`
  - `(2016b) and KernelShap Lundberg & Lee (2017)), and four gradient-based explanation methods (Vanilla`
  - `Figure 4 that LIME exhibits higher agreement with other explanation methods compared to KernelSHAP`
  - `0.273, as opposed to 0.113 in case of KernelSHAP). This finding is consistent with the insights we observed`
  - `point generated using two different explanation methods (e.g., LIME and KernelSHAP in Figure 5). The`
  - `chosen, finding that indeed, certain methods were favored over others. While KernelSHAP was chosen 66.7%`
- Card: `reviews/krishna2022disagreement.md`

## Open manual items
1. **ferrag2022edgeiiot**: drop correct IEEE Access PDF (DOI 10.1109/ACCESS.2022.3165809); re-run extract+cards.
2. Optional: re-rasterize `yagiz2025lens` beyond 35 pages (PDF has 52 text pages).
3. Image-dense table transcription: for camera-ready, open `e2e_pages/<id>/` for any number not in text layer.

## Verification artifacts
- `_pass2_verify/pass2b_results.json` — quote re-check
- `_pass2_verify/pass2_visual_title.json` — title vs page1
- `_extract/*.full.txt` — ground truth text
