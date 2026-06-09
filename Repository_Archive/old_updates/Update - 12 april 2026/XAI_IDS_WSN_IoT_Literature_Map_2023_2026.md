# Comprehensive Literature Map: Explainable AI (SHAP/LIME/XAI) + Intrusion Detection for WSN/IoT (2023--2026)

**Compiled: March 2026 | All claims web-search verified**

---

## SECTION A: COMPLETE PAPER CATALOG

### Paper #1 -- SHAP + Knowledge Distillation + Kronecker Networks (Lightweight IoT IDS)
| Field | Detail |
|---|---|
| **Title** | Lightweight Intrusion Detection in IoT via SHAP-Guided Feature Pruning and Knowledge-Distilled Kronecker Networks |
| **Authors** | Hafsa Benaddi, Mohammed Jouhari, Nouha Laamech, Anas Motii, Khalil Ibrahimi |
| **Venue** | 2025 8th Int'l Conf. on Advanced Communication Technologies and Networking (CommNet) |
| **Year** | 2025 |
| **XAI Method** | SHAP (global feature importance for pruning) |
| **ML Model** | Teacher-Student: Teacher (large DNN, 769,922 params) -> Student (Kronecker-structured NN, 3,042 params) via KD |
| **Datasets** | TON_IoT |
| **Results** | Student macro-F1 > 0.986; ~250x parameter reduction; ~135x size reduction (3,021 KB -> 22 KB); mean latency 1.29 ms; 6.5x throughput gain |
| **DOI/URL** | [10.1109/CommNet68224.2025.11288887](https://arxiv.org/abs/2512.19488) |

---

### Paper #2 -- Federated Learning + SHAP-Based Knowledge Distillation (Botnet IDS)
| Field | Detail |
|---|---|
| **Title** | A Novel Federated Learning Approach for IoT Botnet Intrusion Detection Using SHAP-Based Knowledge Distillation |
| **Authors** | Md. Alamgir Hossain, Md. Saiful Islam |
| **Venue** | Complex & Intelligent Systems (Springer) |
| **Year** | 2025 |
| **XAI Method** | SHAP (feature importance for knowledge distillation across federated clients) |
| **ML Model** | Federated learning with local classifiers; SHAP-based feature knowledge sharing |
| **Datasets** | IoT botnet datasets (multiple botnet types) |
| **Results** | Accuracy 99.99% across various botnet types; privacy-preserving |
| **DOI/URL** | [10.1007/s40747-025-02001-9](https://link.springer.com/article/10.1007/s40747-025-02001-9) |

---

### Paper #3 -- Self-Attention + SHAP/LIME (IoT IDS)
| Field | Detail |
|---|---|
| **Title** | Interpretable Intrusion Detection for IoT Environments Using a Self-Attention-Based Explainable AI Framework |
| **Authors** | (Published in Scientific Reports, Nov 2025) |
| **Venue** | Scientific Reports (Nature) |
| **Year** | 2025 |
| **XAI Method** | SHAP + LIME (post-hoc interpretability) |
| **ML Model** | Self-Attention Deep Neural Network (SA-DNN) with Learnable Feature Gating (LFG) |
| **Datasets** | BoT-IoT, N-BaIoT, UNSW-NB15 |
| **Results** | 99.3% accuracy (BoT-IoT), 99.6% accuracy (N-BaIoT) |
| **DOI/URL** | [s41598-025-23750-0](https://www.nature.com/articles/s41598-025-23750-0) |

---

### Paper #4 -- SHAP-Enhanced NGBoost (IoT IDS)
| Field | Detail |
|---|---|
| **Title** | Interpretable Intrusion Detection for IoT Security: A SHAP-Enhanced NGBoost Model |
| **Authors** | Jingnan Dong, Haolei Chen, Shigen Shen, Huibin Xu, Zhiquan Liu |
| **Venue** | Computer Networks (Elsevier), Vol. 275 |
| **Year** | 2026 (February) |
| **XAI Method** | SHAP (interpretation framework for feature contributions) |
| **ML Model** | NGBoost (Natural Gradient Boosting) |
| **Datasets** | UNSW-NB15, CICIDS2017, N-BaIoT |
| **Results** | Outperforms mainstream baselines in detection and interpretability |
| **DOI** | [10.1016/j.comnet.2025.111937](https://www.sciencedirect.com/science/article/abs/pii/S1389128625009028) |

---

### Paper #5 -- SHAP + Quantum Neural Networks (DDoS in IoT)
| Field | Detail |
|---|---|
| **Title** | SHAP-Based Intrusion Detection in IoT Networks Using Quantum Neural Networks on IonQ Hardware |
| **Authors** | K. Rajkumar, S. Mercy Shalinie |
| **Venue** | Journal of Parallel and Distributed Computing (Elsevier) |
| **Year** | 2025 (October) |
| **XAI Method** | SHAP (post-processing to interpret QNN outputs) |
| **ML Model** | Quantum Neural Networks (QNN) on IonQ via Amazon Braket |
| **Datasets** | CIC-IoT2022, SDN-DDoS24 |
| **Results** | 0.98 expectation value; 113 ms latency; superior DDoS detection |
| **DOI** | [10.1016/j.jpdc.2025.105133](https://www.sciencedirect.com/science/article/abs/pii/S0743731525001005) |

---

### Paper #6 -- MLSTL-WSN: SHAP + LightGBM on WSN-DS
| Field | Detail |
|---|---|
| **Title** | MLSTL-WSN: Machine Learning-Based Intrusion Detection Using SMOTETomek in WSNs |
| **Authors** | Md. Alamin Talukder, Selina Sharmin, Md Ashraf Uddin, Md Manowarul Islam, Sunil Aryal |
| **Venue** | International Journal of Information Security (Springer) |
| **Year** | 2024 |
| **XAI Method** | SHAP (feature selection via SHAP analysis + Recursive Feature Elimination) |
| **ML Model** | LightGBM (iterative tree model, Optuna-optimized) |
| **Datasets** | **WSN-DS** (374,661 records) |
| **Results** | Binary: 99.78% accuracy; Multiclass: 99.92% accuracy; 46% reduction in modeling time; >99% detection for all attack types |
| **DOI** | [10.1007/s10207-024-00833-z](https://link.springer.com/article/10.1007/s10207-024-00833-z) |

---

### Paper #7 -- PSO + Ensemble ML + SHAP/LIME on WSN-DS
| Field | Detail |
|---|---|
| **Title** | Intrusion Detection for Wireless Sensor Network Using Particle Swarm Optimization Based Explainable Ensemble Machine Learning Approach |
| **Authors** | S. A. Birahim, A. Paul, F. Rahman, Y. Islam, T. Roy, M. A. Hasan, F. Haque, M. E. Chowdhury |
| **Venue** | IEEE Access, Vol. 13, pp. 13711--13730 |
| **Year** | 2025 |
| **XAI Method** | SHAP + LIME |
| **ML Model** | PSO-optimized Ensemble (Random Forest + Decision Tree + K-NN) |
| **Datasets** | **WSN-DS** |
| **Results** | Accuracy 99.73%; Precision 99.72%; Recall 99.72%; F1 99.72% |
| **DOI** | [10.1109/ACCESS.2025.3528341](https://ieeexplore.ieee.org/document/10836702/) |

---

### Paper #8 -- Versatile XAI Framework (ANOVA + SHAP + LIME)
| Field | Detail |
|---|---|
| **Title** | A Versatile XAI-Based Framework for Efficient and Explainable Intrusion Detection Systems |
| **Authors** | (Published in Annals of Telecomm., Sept 2025) |
| **Venue** | Annals of Telecommunications (Springer) |
| **Year** | 2025 |
| **XAI Method** | SHAP (global) + LIME (local) with SHAP-LIME cross-validation |
| **ML Model** | XGBoost (lightweight detector) |
| **Datasets** | CIC-DDoS2019, CICIoT2023, 5G PFCP |
| **Results** | F1 >= 99% across all datasets; LIME explanation time reduced 87% (36s -> 4.9s); ~70% dimensionality reduction |
| **DOI** | [10.1007/s12243-025-01118-9](https://link.springer.com/article/10.1007/s12243-025-01118-9) |

---

### Paper #9 -- XAI for IoT Data Streams (CNN/DNN/TabNet + SHAP/LIME)
| Field | Detail |
|---|---|
| **Title** | An Intrusion Detection System over the IoT Data Streams Using eXplainable Artificial Intelligence (XAI) |
| **Authors** | Adel Alabbadi, Fuad Bajaber |
| **Venue** | Sensors (MDPI), Vol. 25(3), 847 |
| **Year** | 2025 (January) |
| **XAI Method** | SHAP + LIME |
| **ML Model** | 1D-CNN, DNN, TabNet (pre-trained) |
| **Datasets** | TON_IoT (7 sub-datasets: Network, Fridge, Thermostat, MotionLight, Garage, Modbus, GPS) |
| **Results** | Network: 99.24% (CNN); IoT avg: 99.96%; 100% on Fridge/Thermostat/Modbus/GPS |
| **DOI** | [10.3390/s25030847](https://pmc.ncbi.nlm.nih.gov/articles/PMC11820747/) |

---

### Paper #10 -- Explainable AI-Based IDS in IoT Systems
| Field | Detail |
|---|---|
| **Title** | Explainable AI-Based Intrusion Detection in IoT Systems |
| **Authors** | Sarah Bin Hulayyil, Shancang Li, Neetesh Saxena |
| **Venue** | Internet of Things (Elsevier) |
| **Year** | 2025 (May) |
| **XAI Method** | XAI techniques (integrated for interpretable IDS decisions) |
| **ML Model** | Binary and multi-classification models |
| **Datasets** | CUSmartHome, IoT23 |
| **Results** | Demonstrated efficiency in detecting IoT vulnerabilities |
| **DOI** | [10.1016/j.iot.2025.101589](https://www.sciencedirect.com/science/article/abs/pii/S2542660525001027) |

---

### Paper #11 -- RAID-KL: SHAP + Knowledge Distillation + Adaptive Loss
| Field | Detail |
|---|---|
| **Title** | Explainable Resource-Aware IoT Security Model via Knowledge Distillation and Adaptive Loss Function Optimization |
| **Authors** | Ogobuchi Daniel Okey, Sajjad Dadkhah, Joao Henrique Kleinschmidt, et al. |
| **Venue** | Expert Systems with Applications (Elsevier), Vol. 302 |
| **Year** | 2026 |
| **XAI Method** | SHAP (feature contribution analysis) |
| **ML Model** | 1D-CNN teacher-student via KD with hybrid KL-JS divergence loss |
| **Datasets** | CICIoT2023, CICIoMT2024, NIMSLABIoT2025 |
| **Results** | 91.24% model compression; 11.3% CPU reduction; 64.33% memory reduction; accuracy maintained |
| **DOI** | [10.1016/j.eswa.2025.130460](https://www.sciencedirect.com/science/article/pii/S0957417425040758) |

---

### Paper #12 -- GA-Optimized LSTM/GRU + LIME (IoT IDS)
| Field | Detail |
|---|---|
| **Title** | GA-Optimized Deep Learning Intrusion Detection Framework with LIME Explainability for IoT Networks |
| **Authors** | (Published in Evolutionary Intelligence, Springer, 2025) |
| **Venue** | Evolutionary Intelligence (Springer) |
| **Year** | 2025 |
| **XAI Method** | LIME |
| **ML Model** | GA-optimized LSTM and GRU |
| **Datasets** | IoT network datasets |
| **Results** | 99.84% accuracy; model size reduced to 108.42 KB via dynamic quantization |
| **DOI** | [10.1007/s12065-025-01133-8](https://link.springer.com/article/10.1007/s12065-025-01133-8) |

---

### Paper #13 -- Evaluating ML-Based IDS with XAI
| Field | Detail |
|---|---|
| **Title** | Evaluating Machine Learning-Based Intrusion Detection Systems with Explainable AI: Enhancing Transparency and Interpretability |
| **Authors** | Vincent Zibi Mohale, Ibidun Christiana Obagbuwa |
| **Venue** | Frontiers in Computer Science |
| **Year** | 2025 |
| **XAI Method** | LIME, SHAP, ELI5 |
| **ML Model** | Decision Trees, MLP, XGBoost, Random Forest, CatBoost, Logistic Regression, Gaussian NB |
| **Datasets** | UNSW-NB15 |
| **Results** | Best accuracy 87% (XGBoost, CatBoost); Key features: sttl, ct_srv_dst |
| **DOI** | [10.3389/fcomp.2025.1520741](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2025.1520741/full) |

---

### Paper #14 -- Explainable AI for Forensic IDS (SHAP vs LIME)
| Field | Detail |
|---|---|
| **Title** | Explainable AI for Forensic Analysis: A Comparative Study of SHAP and LIME in Intrusion Detection Models |
| **Authors** | Pamela Hermosilla, Sebastian Berrios, Hector Allende-Cid |
| **Venue** | Applied Sciences (MDPI), Vol. 15(13), 7329 |
| **Year** | 2025 (June) |
| **XAI Method** | SHAP vs LIME (comparative evaluation: fidelity, consistency, Jaccard similarity, stability) |
| **ML Model** | XGBoost, TabNet |
| **Datasets** | UNSW-NB15 |
| **Results** | XGBoost: 97.8% accuracy; superior explanation stability and coherence vs TabNet |
| **DOI** | [10.3390/app15137329](https://www.mdpi.com/2076-3417/15/13/7329) |

---

### Paper #15 -- Explainable AI for IDS on MLP (IEEE Access)
| Field | Detail |
|---|---|
| **Title** | Explainable AI for Intrusion Detection Systems: LIME and SHAP Applicability on Multi-Layer Perceptron |
| **Authors** | Gaspar, Silva et al. |
| **Venue** | IEEE Access, Vol. 12 |
| **Year** | 2024 |
| **XAI Method** | SHAP + LIME (with perturbation analysis) |
| **ML Model** | Multi-Layer Perceptron (MLP) |
| **Datasets** | IoT datasets |
| **Results** | Demonstrated applicability of SHAP/LIME on MLP for IoT IDS |
| **DOI** | [10.1109/ACCESS.2024.3368685](https://ieeexplore.ieee.org/document/10440604/) |

---

### Paper #16 -- Explainable AI-Based Innovative Models for Big Data WSN IDS
| Field | Detail |
|---|---|
| **Title** | Explainable AI-Based Innovative Models for Intrusion Detection in Big Data WSN |
| **Authors** | Naima Samout, Thouraya Gouasmi, Nejah Nasri |
| **Venue** | Procedia Computer Science (Elsevier), Vol. 270, pp. 5065--5072 |
| **Year** | 2025 |
| **XAI Method** | SHAP + LIME |
| **ML Model** | Decision Trees, SVMs, Ensemble methods |
| **Datasets** | KDD Cup 99 |
| **Results** | High accuracy, precision, recall, F1 with decision transparency for agricultural IoT monitoring |
| **DOI** | [10.1016/j.procs.2025.09.633](https://www.sciencedirect.com/science/article/pii/S187705092503306X) |

---

### Paper #17 -- Rule-Induction Explainable IDS for IoT
| Field | Detail |
|---|---|
| **Title** | Intrusion Detection Framework for Internet of Things with Rule Induction for Model Explanation |
| **Authors** | (Published in Sensors, MDPI, March 2025) |
| **Venue** | Sensors (MDPI), Vol. 25(6), 1845 |
| **Year** | 2025 |
| **XAI Method** | Rule induction (inherently interpretable) |
| **ML Model** | Ensemble: RF, AdaBoost, XGBoost, LightGBM, CatBoost |
| **Datasets** | CIC-IDS2017, CICIoT2023 |
| **Results** | XGBoost: 99.91% accuracy, 99.88% AUC-ROC (CIC-IDS2017); 98.54% accuracy, 93.06% AUC-ROC (CICIoT2023) |
| **DOI** | [10.3390/s25061845](https://www.mdpi.com/1424-8220/25/6/1845) |

---

### Paper #18 -- Trustworthy Adaptive AI for IIoT IDS
| Field | Detail |
|---|---|
| **Title** | Trustworthy Adaptive AI for Real-Time Intrusion Detection in Industrial IoT Security |
| **Authors** | (Published in MDPI, Sept 2025) |
| **Venue** | MDPI Journal (2624-831X), Vol. 6(3), 53 |
| **Year** | 2025 |
| **XAI Method** | SHAP |
| **ML Model** | Ensemble of online learning models |
| **Datasets** | Industrial IoT network traffic |
| **Results** | 96.4% accuracy; 2.1% FPR; 35 ms average detection time on edge devices |
| **DOI/URL** | [MDPI 2624-831X/6/3/53](https://www.mdpi.com/2624-831X/6/3/53) |

---

### Paper #19 -- Transformer-Based KD for Explainable IDS
| Field | Detail |
|---|---|
| **Title** | Transformer-Based Knowledge Distillation for Explainable Intrusion Detection System |
| **Authors** | Nadiah AL-Nomasy, Abdulelah Alamri, Ahamed Aljuhani, Prabhat Kumar |
| **Venue** | Computers & Security (Elsevier), Vol. 154 |
| **Year** | 2025 (July) |
| **XAI Method** | Gradient contribution heatmaps, layer-wise contribution, gradient selection impact analysis |
| **ML Model** | Transformer (MHSA + cross-attention) teacher -> lightweight student via SG-KD |
| **Datasets** | Not specified |
| **Results** | Superior accuracy and efficiency vs state-of-the-art IDS |
| **DOI** | [10.1016/j.cose.2025.104417](https://www.sciencedirect.com/science/article/pii/S0167404825001063) |

---

### Paper #20 -- SHAP + Transformer for VANET IDS
| Field | Detail |
|---|---|
| **Title** | A Novel Transformer-Based Explainable AI Approach Using SHAP for Intrusion Detection in Vehicular Ad Hoc Networks |
| **Authors** | Waqar Khan, Jawad Ahmad, Nada Alasbali, Alanoud Al Mazroa, Mohammed S. Alshehri, Muhammad Shahbaz Khan |
| **Venue** | Computer Networks (Elsevier) |
| **Year** | 2025 |
| **XAI Method** | SHAP |
| **ML Model** | Transformer with multi-head attention |
| **Datasets** | VeReMi extension dataset |
| **Results** | Multi-class: 96.15%; Binary: 98.28% |
| **DOI** | [10.1016/j.comnet.2025.111575](https://www.sciencedirect.com/science/article/pii/S1389128625005420) |

---

### Paper #21 -- L-XAIDS: LIME-based IDS Framework
| Field | Detail |
|---|---|
| **Title** | L-XAIDS: A LIME-Based eXplainable AI Framework for Intrusion Detection Systems |
| **Authors** | Aoun E Muhammad et al. |
| **Venue** | Cluster Computing (Springer) |
| **Year** | 2025 |
| **XAI Method** | LIME + ELI5 |
| **ML Model** | Ensemble techniques |
| **Datasets** | Network intrusion datasets |
| **Results** | Higher detection rate, lower FPR vs 3 traditional approaches |
| **DOI/URL** | [10.1007/s10586-025-05326-9](https://link.springer.com/article/10.1007/s10586-025-05326-9) |

---

### Paper #22 -- CNN + SHAP/LIME for IoT IDS (Cybersecurity Journal)
| Field | Detail |
|---|---|
| **Title** | Intrusion Detection in the Internet of Things Using Convolutional Neural Networks: An Explainable AI Approach |
| **Authors** | (Published in Cybersecurity, Springer, Sept 2025) |
| **Venue** | Cybersecurity (SpringerOpen) |
| **Year** | 2025 |
| **XAI Method** | SHAP (global + local) + LIME (local) |
| **ML Model** | Lightweight 1D-CNN |
| **Datasets** | TON-IoT |
| **Results** | Resource-efficient; SHAP used for feature selection phase |
| **DOI** | [10.1186/s42400-025-00369-2](https://cybersecurity.springeropen.com/articles/10.1186/s42400-025-00369-2) |

---

### Paper #23 -- Tabu Search + Random Forest on WSN-DS (non-XAI but WSN-DS benchmark)
| Field | Detail |
|---|---|
| **Title** | Enhancing Intrusion Detection in Wireless Sensor Networks Using a Tabu Search Based Optimized Random Forest |
| **Authors** | Vivek Kumar Pandey, Shiv Prakash, Tarun Kumar Gupta, Priyanshu Sinha, Tiansheng Yang, Rajkumar Singh Rathore, Lu Wang, Sabeen Tahir, Sheikh Tahir Bakhsh |
| **Venue** | Scientific Reports (Nature) |
| **Year** | 2025 |
| **XAI Method** | None (no SHAP/LIME used) |
| **ML Model** | Tabu Search-optimized Random Forest |
| **Datasets** | **WSN-DS**, CICIDS 2017, CIC-IoT 2023 |
| **Results** | WSN-DS: Accuracy 99.67%, F1 99.67%; Blackhole F1=0.99, Flooding F1=0.96, Grayhole F1=0.98, TDMA F1=0.96 |
| **DOI** | [10.1038/s41598-025-03498-3](https://www.nature.com/articles/s41598-025-03498-3) |

---

### Paper #24 -- KD for Lightweight + Explainable IDS (IEEE Consumer Electronics)
| Field | Detail |
|---|---|
| **Title** | Knowledge Distillation for Lightweight and Explainable Intrusion Detection in Resource-Constrained Consumer Devices |
| **Authors** | (IEEE, 2025) |
| **Venue** | IEEE Transactions on Consumer Electronics |
| **Year** | 2025 |
| **XAI Method** | Explainability via KD transparency |
| **ML Model** | Teacher-student KD |
| **Datasets** | N-BaIoT, CIC-IDS 2023 |
| **Results** | Student accuracy: 99.87% (N-BaIoT), 98.71% (CIC-IDS2023); 263k -> 120k params; 1.00 MB -> 471.67 KB |
| **DOI/URL** | [IEEE 11133481](https://ieeexplore.ieee.org/iel8/30/11306167/11133481.pdf) |

---

## SURVEY PAPERS (2025--2026)

### Survey #1 -- XAI-IDS for Industry 5.0
| Field | Detail |
|---|---|
| **Title** | Explainable AI-Based Intrusion Detection Systems for Industry 5.0 and Adversarial XAI: A Systematic Review |
| **Venue** | Information (MDPI), Vol. 16(12), 1036 |
| **Year** | 2025 (November) |
| **Scope** | PRISMA-guided analysis of 135 studies; XAI technique distributions; adversarial XAI (Adv-XIDS) |
| **Key Finding** | Documents XAI's dual nature: IDS enhancement AND adversarial attack vector via SHAP/LIME/gradient methods |
| **URL** | [MDPI Information](https://www.mdpi.com/2078-2489/16/12/1036) |

### Survey #2 -- Explainable DL-Based IDS for IoT (Systematic Review)
| Field | Detail |
|---|---|
| **Title** | Performance Analysis of Explainable Deep Learning-Based Intrusion Detection Systems for IoT Networks: A Systematic Review |
| **Venue** | Sensors (MDPI), Vol. 26(2), 363 |
| **Year** | 2026 (January) |
| **Scope** | 129 peer-reviewed studies (2018--2025); PRISMA methodology |
| **Key Finding** | Proposes UXIEF framework modeling trilemma: detection performance vs resource efficiency vs explanation quality |
| **URL** | [MDPI Sensors](https://www.mdpi.com/1424-8220/26/2/363) |

### Survey #3 -- Systematic Review of XAI in IDS (Frontiers)
| Field | Detail |
|---|---|
| **Title** | A Systematic Review on the Integration of Explainable Artificial Intelligence in Intrusion Detection Systems to Enhancing Transparency and Interpretability in Cybersecurity |
| **Venue** | Frontiers in Artificial Intelligence |
| **Year** | 2025 |
| **Key Finding** | Rule-based and tree-based XAI models preferred for interpretability; trade-offs with detection accuracy remain challenging |
| **URL** | [Frontiers in AI](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1526221/full) |

### Survey #4 -- Comprehensive Review of XAI in Cybersecurity
| Field | Detail |
|---|---|
| **Title** | A Comprehensive Review of Explainable AI in Cybersecurity: Decoding the Black Box |
| **Venue** | ScienceDirect |
| **Year** | 2025 |
| **Key Finding** | Taxonomy of XAI techniques for cybersecurity; emphasizes significance of explainability in boosting trust |
| **URL** | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2405959525001584) |

---

## SECTION B: CRITICAL ANALYSIS

### B1. Papers Using SHAP on WSN-DS Specifically

From all searches conducted, the following papers specifically use **SHAP on the WSN-DS dataset**:

| # | Paper | SHAP Role | Year |
|---|---|---|---|
| 1 | **MLSTL-WSN** (Talukder et al.) | SHAP for feature selection + RFE on WSN-DS | 2024 |
| 2 | **PSO-based Ensemble** (Birahim et al.) | SHAP + LIME for explainability on WSN-DS | 2025 |

**Only 2 papers** explicitly combine SHAP with the WSN-DS dataset. Paper #23 (Pandey et al.) uses WSN-DS but does NOT employ SHAP. Paper #16 (Samout et al.) addresses WSN but uses KDD Cup 99, not WSN-DS.

**Assessment:** The intersection of SHAP + WSN-DS is remarkably sparse. This represents a **clear and exploitable research gap**.

---

### B2. Is "SHAP for IDS" Saturated or Still Novel?

**Verdict: SHAP for IDS is widespread but NOT saturated -- specific niches remain open.**

Evidence:
- The 2026 survey (Sensors, 129 papers) and 2025 survey (Information, 135 studies) confirm SHAP is the **most popular** post-hoc XAI method in IDS research.
- However, SHAP is predominantly used on:
  - General network datasets (UNSW-NB15, CICIDS2017, CIC-DDoS2019)
  - IoT-specific datasets (TON_IoT, BoT-IoT, N-BaIoT, CICIoT2023)
- SHAP on **WSN-specific datasets (WSN-DS)** = only 2 papers (see B1 above)
- SHAP combined with **model compression/KD for WSN** = essentially zero on WSN-DS
- SHAP combined with **lightweight/edge-deployable models** is emerging but early-stage (only 3--4 papers: Benaddi et al., Okey et al., AL-Nomasy et al.)

**Where novelty still exists:**
1. SHAP + WSN-DS (only 2 prior papers)
2. SHAP + knowledge distillation targeting WSN constraints specifically
3. SHAP-guided feature selection for WSN-specific attack types (Blackhole, Grayhole, Flooding, TDMA)
4. SHAP + federated learning for WSN (no existing work on WSN-DS)
5. SHAP for WSN in agricultural/healthcare IoT domains (one paper: Samout et al. on KDD99, but not WSN-DS)

---

### B3. XAI Methods Beyond SHAP Gaining Traction

| XAI Method | Example Papers | Trend |
|---|---|---|
| **LIME** | Papers #3, #7, #8, #9, #12, #13, #14, #15, #21, #22 | Co-dominant with SHAP; often used together |
| **ELI5** | Paper #13 (Mohale & Obagbuwa), Paper #21 (L-XAIDS) | Niche; for RF feature weights |
| **Rule Induction** | Paper #17 (Sensors 2025) | Rising; inherently interpretable for stakeholder adoption |
| **Attention Mechanisms** (self-attention, multi-head) | Papers #3, #19, #20 | Growing; attention weights serve as built-in explanations |
| **Gradient-Based Methods** (Grad-CAM, gradient heatmaps) | Paper #19 (AL-Nomasy et al.) | Emerging for Transformer/DL-based IDS |
| **Swin Vision Transformer** | VANET paper (Wiley 2025) | Very new for network security |
| **Adversarial XAI** | Survey #1 (Adv-XIDS) | New concern: SHAP/LIME as attack vectors |

**Key Insight:** LIME is nearly as prevalent as SHAP but used differently (local vs global). Attention-based and gradient-based methods are the **fastest-growing alternatives**, especially with Transformer architectures entering IDS. Rule induction is gaining traction where stakeholder trust and regulatory compliance matter.

---

### B4. Has Anyone Combined SHAP with Model Compression/KD for WSN?

**For IoT generally: YES (3 papers found)**

| Paper | SHAP + KD Combination | Target | On WSN-DS? |
|---|---|---|---|
| **Benaddi et al. (2025)** | SHAP-guided feature pruning + Kronecker KD | IoT (TON_IoT) | **No** |
| **Hossain & Islam (2025)** | SHAP-based federated KD for feature knowledge sharing | IoT botnets | **No** |
| **Okey et al. (2026)** | SHAP + 1D-CNN KD with adaptive KL-JS loss | IoT (CICIoT2023, CICIoMT2024) | **No** |

**For WSN-DS specifically: NO -- zero papers found.**

This is a **significant gap**. No existing work combines:
- SHAP explainability
- Knowledge distillation / model compression
- WSN-DS dataset or WSN-specific attack types

---

## SECTION C: GAP SUMMARY TABLE

| Research Dimension | Coverage (2023--2026) | Gap Level |
|---|---|---|
| SHAP + general IoT IDS | ~15+ papers | Saturated |
| SHAP + LIME comparative studies | ~5 papers | Moderate |
| SHAP on WSN-DS | 2 papers | **Large Gap** |
| SHAP + KD for IoT | 3 papers | Open |
| SHAP + KD for WSN/WSN-DS | **0 papers** | **Major Gap** |
| LIME-only IDS frameworks | 3--4 papers | Open |
| Rule induction for IoT IDS | 1--2 papers | Open |
| Attention-based XAI for IDS | 3--4 papers | Emerging |
| Adversarial XAI for IDS | 1 survey + emerging | Wide open |
| SHAP + federated learning for WSN | 0 on WSN-DS | **Major Gap** |

---

## SECTION D: FULL SOURCE LIST

1. [Benaddi et al. -- SHAP+KD Kronecker (arXiv)](https://arxiv.org/abs/2512.19488)
2. [Hossain & Islam -- FL+SHAP KD (Springer)](https://link.springer.com/article/10.1007/s40747-025-02001-9)
3. [SA-DNN Framework (Nature Scientific Reports)](https://www.nature.com/articles/s41598-025-23750-0)
4. [Dong et al. -- NGBoost+SHAP (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S1389128625009028)
5. [Rajkumar & Shalinie -- QNN+SHAP (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0743731525001005)
6. [Talukder et al. -- MLSTL-WSN (Springer)](https://link.springer.com/article/10.1007/s10207-024-00833-z)
7. [Birahim et al. -- PSO Ensemble WSN-DS (IEEE Access)](https://ieeexplore.ieee.org/document/10836702/)
8. [Versatile XAI Framework (Annals of Telecomm.)](https://link.springer.com/article/10.1007/s12243-025-01118-9)
9. [Alabbadi & Bajaber -- IoT Streams (Sensors)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11820747/)
10. [Bin Hulayyil et al. -- XAI IoT IDS (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S2542660525001027)
11. [Okey et al. -- RAID-KL (Expert Systems)](https://www.sciencedirect.com/science/article/pii/S0957417425040758)
12. [GA-Optimized LSTM+LIME (Springer)](https://link.springer.com/article/10.1007/s12065-025-01133-8)
13. [Mohale & Obagbuwa -- XAI Evaluation (Frontiers)](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2025.1520741/full)
14. [Hermosilla et al. -- Forensic SHAP vs LIME (Applied Sciences)](https://www.mdpi.com/2076-3417/15/13/7329)
15. [Gaspar & Silva -- MLP+SHAP/LIME (IEEE Access)](https://ieeexplore.ieee.org/document/10440604/)
16. [Samout et al. -- Big Data WSN (Procedia CS)](https://www.sciencedirect.com/science/article/pii/S187705092503306X)
17. [Rule Induction IoT IDS (Sensors)](https://www.mdpi.com/1424-8220/25/6/1845)
18. [Trustworthy Adaptive AI IIoT (MDPI)](https://www.mdpi.com/2624-831X/6/3/53)
19. [AL-Nomasy et al. -- Transformer KD (Computers & Security)](https://www.sciencedirect.com/science/article/pii/S0167404825001063)
20. [Khan et al. -- Transformer VANET SHAP (Computer Networks)](https://www.sciencedirect.com/science/article/pii/S1389128625005420)
21. [L-XAIDS (Cluster Computing)](https://link.springer.com/article/10.1007/s10586-025-05326-9)
22. [1D-CNN+SHAP IoT (Cybersecurity/Springer)](https://cybersecurity.springeropen.com/articles/10.1186/s42400-025-00369-2)
23. [Pandey et al. -- Tabu Search WSN-DS (Scientific Reports)](https://www.nature.com/articles/s41598-025-03498-3)
24. [KD for Consumer Devices (IEEE TCE)](https://ieeexplore.ieee.org/iel8/30/11306167/11133481.pdf)
25. [XAI-IDS Industry 5.0 Survey (Information MDPI)](https://www.mdpi.com/2078-2489/16/12/1036)
26. [Explainable DL IDS IoT Survey (Sensors 2026)](https://www.mdpi.com/1424-8220/26/2/363)
27. [XAI+IDS Systematic Review (Frontiers in AI)](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1526221/full)
28. [XAI Cybersecurity Review (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S2405959525001584)
