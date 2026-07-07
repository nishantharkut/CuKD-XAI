# Complete Landscape: Concept Drift in Intrusion Detection Systems (2023-2026)

**Generated:** March 25, 2026
**Scope:** All web-verified papers on concept drift + intrusion detection from 2023 through early 2026

---

## TABLE OF CONTENTS
1. [Comprehensive Paper Catalog](#1-comprehensive-paper-catalog)
2. [Critical Questions Answered](#2-critical-questions-answered)
3. [Drift Detection Methods Taxonomy for IDS](#3-drift-detection-methods-taxonomy-for-ids)
4. [Research Gap Analysis](#4-research-gap-analysis)

---

## 1. COMPREHENSIVE PAPER CATALOG

### Paper #1: Evolving Cybersecurity Frontiers (SURVEY)
- **Title:** "Evolving cybersecurity frontiers: A comprehensive survey on concept drift and feature dynamics aware machine and deep learning in intrusion detection systems"
- **Authors:** Methaq A. Shyaa, Noor Farizah Ibrahim, Zurinahni Zainol, Rosni Abdullah, Mohammed Anbar, Laith Alzubaidi (corresponding)
- **Venue:** Engineering Applications of Artificial Intelligence, Vol. 137, Part A, 2024
- **DOI:** 10.1016/j.engappai.2024.109143
- **Datasets:** Survey paper (analyzes works from 2019-2024)
- **Drift Detection:** Surveys all major methods; proposes taxonomy covering signature-based drift (protocol mutation, signature variations) and anomaly-based drift (user behavior changes, infrastructure changes)
- **Adaptation:** Proposes novel framework with dynamic feature selection, adaptive learning algorithms, reinforcement learning, and continuous monitoring
- **Key Findings:** Identifies critical gap: no unified integration of concept drift AND feature drift within IDS. Experimental evaluation planned for future work.
- **URL:** https://www.sciencedirect.com/science/article/pii/S0952197624013010

---

### Paper #2: EMNCD - IoT Data Streams Concept Drift Localization
- **Title:** "Intrusion detection in the IoT data streams using concept drift localization"
- **Authors:** Renjie Chu, Peiyuan Jin, Hanli Qiao, Quanxi Feng
- **Venue:** AIMS Mathematics, Vol. 9, Issue 1, pp. 1535-1561, 2024
- **DOI:** 10.3934/math.2024076
- **Datasets:** Synthetic (SEA, Stagger, Rotating Hyperplane); Real-world: Edge-Industrial IoTset (IIoTset)
- **Drift Detection:** EMNCD (Ensemble Multiple Non-parametric Concept localization Detectors) - combines Kolmogorov-Smirnov test, Wilcoxon rank sum test, Mann-Kendall test in ensemble with sliding windows
- **Adaptation:** Isolation Forest (iForest) for online anomaly detection; WOA-XGBoost (Whale Optimization Algorithm-XGBoost) hybrid that dynamically updates using detected drift points
- **Key Results:** Superior performance over classical methods on artificial datasets; effective concept drift localization and intrusion detection
- **URL:** https://www.aimspress.com/article/doi/10.3934/math.2024076

---

### Paper #3: GPC Variants with Incremental Learning
- **Title:** "Enhanced Intrusion Detection with Data Stream Classification and Concept Drift Guided by the Incremental Learning Genetic Programming Combiner"
- **Authors:** Methaq A. Shyaa, Zurinahni Zainol, Rosni Abdullah, Mohammed Anbar, Laith Alzubaidi, Jose Santamaria
- **Venue:** Sensors, Vol. 23, Issue 7, Article 3736, 2023
- **DOI:** 10.3390/s23073736
- **Datasets:** KDD Cup '99, CICIDS-2017, CSE-CIC-IDS-2018, ISCX '12
- **Drift Detection:** Monitors classifier performance degradation across data stream batches
- **Adaptation:** Extended GPC (Genetic Programming Combiner) with three OSELM variants: GPC-KOS (Knowledge-Preserving OSELM), GPC-FOS (Feature-Adaptive OSELM), GPC-OS (OSELM). Includes data balancing and classifier update components.
- **Key Results:** GPC-KOS achieves 100% accuracy; GPC-FOS achieves 98% accuracy; both outperform traditional GPC and state-of-the-art
- **URL:** https://www.mdpi.com/1424-8220/23/7/3736

---

### Paper #4: ReCDA (KDD 2024)
- **Title:** "ReCDA: Concept Drift Adaptation with Representation Enhancement for Network Intrusion Detection"
- **Authors:** (CityU Hong Kong group including Edith C. H. Ngai)
- **Venue:** ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2024)
- **DOI:** 10.1145/3637528.3672007
- **Datasets:** Several benchmark NID datasets under varying degrees of concept drift (including CIC-IDS variants)
- **Drift Detection:** Implicit (drift-aware perturbation mechanism)
- **Adaptation:** Two-stage: (1) Self-supervised representation enhancement with drift-aware perturbation + representation alignment (drift-aware and drift-invariant perspectives); (2) Weakly-supervised classifier tuning with instructive sampling strategy and robust representation constraint
- **Key Results:** Superior adaptability and robustness over existing methods; requires minimal labeled drifting samples
- **URL:** https://dl.acm.org/doi/10.1145/3637528.3672007

---

### Paper #5: DDM-ORF (Jemili et al.)
- **Title:** "Intrusion detection based on concept drift detection and online incremental learning"
- **Authors:** Farah Jemili, Khaoula Jouini, Ouajdi Korbaa
- **Venue:** International Journal of Pervasive Computing and Communications, Vol. 21, Issue 1, pp. 81+, 2025
- **DOI:** 10.1108/IJPCC-12-2023-0358
- **Datasets:** MAWILab dataset (Fukuda Laboratory, University of Tokyo)
- **Drift Detection:** DDM (Drift Detection Method)
- **Adaptation:** ORF (Online Random Forest) for incremental learning; implemented on Apache Spark for distributed computing
- **Key Results:** 99.96% accuracy; superior anomalous/suspicious detection rates vs. CNN-based model; high processing rates on Spark
- **URL:** https://www.emerald.com/insight/content/doi/10.1108/ijpcc-12-2023-0358/full/html

---

### Paper #6: Managing Concept Drift in Online IDS (ITASEC 2025)
- **Title:** "Managing Concept Drift in Online Intrusion Detection"
- **Authors:** Francesco Camarda, Alessandra De Paola, Salvatore Drago, Pierluca Ferraro, Giuseppe Lo Re
- **Venue:** Joint National Conference on Cybersecurity (ITASEC & SERICS 2025), February 2025, Bologna, Italy. CEUR-WS Vol-3962, Paper 42.
- **Datasets:** Labeled benign and malicious traffic compiled by domain experts
- **Drift Detection:** Monitors statistical distribution changes in network traffic
- **Adaptation:** Online adaptive learning (active learning component)
- **Key Results:** Addresses the open environment challenge where network traffic distribution changes post-deployment
- **URL:** https://ceur-ws.org/Vol-3962/paper42.pdf

---

### Paper #7: Concept Drift Stream Generator for IDS (Ital-IA 2025)
- **Title:** "A Concept Drift Stream Generator for Intrusion Detection Systems"
- **Authors:** Gabriele Nicolo Costa, Alessandra De Paola, Salvatore Drago, Pierluca Ferraro
- **Venue:** Ital-IA 2025: 5th National Conference on Artificial Intelligence, June 23-24, 2025, Trieste, Italy. CEUR-WS Vol-4121.
- **Datasets:** Generates synthetic IDS data streams with controlled drift
- **Drift Detection:** N/A (tool paper - generates streams with abrupt/gradual drift)
- **Adaptation:** N/A (provides drift benchmarking tool)
- **Key Results:** Provides a stream generator to simulate abrupt and gradual concept drift for IDS evaluation. Directly addresses the lack of drift-aware IDS benchmarks.
- **URL:** https://ceur-ws.org/Vol-4121/Ital-IA_2025_paper_94.pdf

---

### Paper #8: REACT-D3QN (2026)
- **Title:** "React-D3QN: Resilient Adaptive Concept-Drift-Aware Dueling Double Deep Q-Network for Robust Edge-Centric Intrusion Detection"
- **Authors:** (Not fully extracted - Springer Cybersecurity journal)
- **Venue:** Cybersecurity (Springer Nature), 2026
- **DOI:** 10.1186/s42400-026-00558-7
- **Datasets:** CIC-IDS 2017
- **Drift Detection:** Implicit in DRL framework (detects "Recall Collapse" under drift)
- **Adaptation:** Dueling architecture decouples state-value estimation from action advantages; 10:1 risk-aware penalty reward architecture; designed for resource-constrained edge devices
- **Key Results:** Addresses "Recall Collapse" and "Information Poverty" in edge-centric IDS; maintains detection integrity during cross-dataset migration
- **URL:** https://link.springer.com/article/10.1186/s42400-026-00558-7

---

### Paper #9: Self-Supervised Adaptation (IEEE TDSC 2025)
- **Title:** "Self-Supervised Adaptation Method to Concept Drift for Network Intrusion Detection"
- **Authors:** Yang et al.
- **Venue:** IEEE Transactions on Dependable and Secure Computing, August 2025
- **DOI:** (IEEE Xplore Document 11125964)
- **Datasets:** Network intrusion detection benchmarks
- **Drift Detection:** Detects distributional deviation between arriving traffic flows and historical empirical distributions
- **Adaptation:** Self-supervised learning approach; does not require significant labeled drifting samples
- **Key Results:** Addresses labor-intensive labeling and label noise risks of existing approaches
- **URL:** https://ieeexplore.ieee.org/document/11125964/

---

### Paper #10: Dynamic IDS with Concept Drift (2026)
- **Title:** "A Dynamic Intrusion Detection System Integrating Concept Drift Detection and Incremental Learning"
- **Authors:** Zhuoqi Liao (Zhejiang Normal University)
- **Venue:** Journal of Computer, Signal, and System Research, Vol. 3, No. 1, 2026
- **DOI:** 10.71222/bvjsn889
- **Datasets:** Not specified
- **Drift Detection:** Concept drift detection mechanisms (distributional change identification)
- **Adaptation:** Incremental learning strategies; models update knowledge without retraining from scratch
- **Key Results:** Balances detection accuracy with computational efficiency under continuously changing conditions
- **URL:** https://www.gbspress.com/index.php/JCSSR/article/view/553

---

### Paper #11: DWOIDS (ICCS 2025)
- **Title:** "Dual Adaptive Windows Toward Concept-Drift in Online Network Intrusion Detection"
- **Authors:** Hu, X., Ma, D., Wang, W., Liu, F.
- **Venue:** Computational Science - ICCS 2025 (Springer)
- **DOI:** 10.1007/978-3-031-97626-1_15
- **Datasets:** Network intrusion detection datasets (not explicitly named)
- **Drift Detection:** Dual adaptive windows monitoring prediction error of the classifier
- **Adaptation:** Hoeffding tree classifier with dual adaptive windows; continuously refines accuracy when drift occurs
- **Key Results:** Addresses accuracy decrease in online IDS caused by concept drift in dynamic data streams
- **URL:** https://link.springer.com/chapter/10.1007/978-3-031-97626-1_15

---

### Paper #12: DyNA-IDS (2026)
- **Title:** "Revealing the Frailty of Static Benchmarks: The DyNA-IDS Framework for Concept Drift Adaptation in Time-Series Network Intrusion Detection"
- **Authors:** Jia, K., Gao, H., Chen, J., Qi, J.
- **Venue:** Springer (2026)
- **DOI:** 10.1007/978-981-95-6203-9_18
- **Datasets:** Network intrusion detection benchmarks (time-series)
- **Drift Detection:** Framework-integrated drift detection
- **Adaptation:** End-to-end framework for concept drift adaptation in time-series NIDS
- **Key Results:** Demonstrates frailty of static benchmarks; provides robust framework for dynamic environments. Code: https://github.com/GZHU-Innovation-Intersection-Lab/DyNA-IDS
- **URL:** https://link.springer.com/chapter/10.1007/978-981-95-6203-9_18

---

### Paper #13: NetGuard (KDD 2025)
- **Title:** "Generative Active Adaptation for Drifting and Imbalanced Network Intrusion Detection"
- **Authors:** Ragini Gupta, Shinan Liu, Ruixiao Zhang, Xinyue Hu, Xiaoyang Wang, Hadjer Benkraouda, Pranav Kommaraju, Phuong Cao, Nick Feamster, Klara Nahrstedt
- **Venue:** arXiv:2503.03022 (March 2025), accepted at KDD'25
- **Datasets:** CIC-IDS 2018 and real-world ISP network traffic data
- **Drift Detection:** Implicit (addresses drift through generative adaptation)
- **Adaptation:** Density-aware dataset prior selection for informative sample identification; deep generative models for conditional sample synthesis; dataset augmentation
- **Key Results:** F1-score improved from 0.60 to 0.86; rare attacks: Infiltration (0.001->0.30), Web Attack (0.04->0.50), FTP-BruteForce (0.00->0.71)
- **URL:** https://arxiv.org/abs/2503.03022

---

### Paper #14: Drift Adaptive Online DDoS (2024)
- **Title:** "Drift Adaptive Online DDoS Attack Detection Framework for IoT System"
- **Authors:** Yonas Kibret Beshah, Surafel Lemma Abebe, Henock Mulugeta Melaku
- **Venue:** Electronics, Vol. 13, Issue 6, Article 1004, 2024
- **DOI:** 10.3390/electronics13061004
- **Datasets:** IOTID20, CICIoT2023
- **Drift Detection:** ADWIN (Adaptive Windowing), DDM (Drift Detection Method)
- **Adaptation:** Four base learners: ARF-ADWIN, ARF-DDM, SRPs-DDM, KNN-ADWIN for DDoS detection
- **Key Results:** Effective DDoS attack detection with adaptive drift handling in IoT
- **URL:** https://www.mdpi.com/2079-9292/13/6/1004

---

### Paper #15: PWPAE (2021, foundational)
- **Title:** "PWPAE: An Ensemble Framework for Concept Drift Adaptation in IoT Data Streams"
- **Authors:** Li Yang, Abdallah Shami
- **Venue:** IEEE GlobeCom 2021
- **DOI:** 10.1109/GlobeCom46510.2021.9685338
- **Datasets:** IoTID20, CICIDS2017
- **Drift Detection:** ADWIN, DDM
- **Adaptation:** Performance Weighted Probability Averaging Ensemble (PWPAE) - dynamic weight assignment to base learners (ARF-ADWIN, ARF-DDM, SRP-ADWIN, SRP-DDM)
- **Key Results:** 99.16% accuracy; outperforms individual base learners (97.85%-98.99%)
- **URL:** https://ieeexplore.ieee.org/document/9685338/

---

### Paper #16: OASW (2021, foundational)
- **Title:** "A Lightweight Concept Drift Detection and Adaptation Framework for IoT Data Streams"
- **Authors:** Li Yang, Abdallah Shami
- **Venue:** IEEE Internet of Things Magazine, June 2021
- **Datasets:** IoTID20, NSL-KDD
- **Drift Detection:** OASW (Online Adaptive Sliding Window) - combines sliding window, adaptive window, and performance-based methods
- **Adaptation:** Lightweight model retraining on Raspberry Pi 3
- **Key Results:** 99.92% accuracy on IoTID20; prediction time 7.8ms/9.1ms on Raspberry Pi; smallest memory usage among compared methods
- **URL:** https://arxiv.org/abs/2104.10529

---

### Paper #17: OASIS (2025)
- **Title:** "OASIS: Online Adaptive Ensembles for Drift Adaptation on Evolving IoT Data Streams"
- **Authors:** (Not fully extracted)
- **Venue:** Internet of Things (Elsevier), 2025
- **DOI:** 10.1016/j.iot.2025.101523 (approx.)
- **Datasets:** EDGE-IIoT, IoTID20, Aalto, RT-IoT 2022
- **Drift Detection:** EDDM (Early Drift Detection Method), HDDM-A, ADWIN
- **Adaptation:** Six distinct online learners with three drift adaptation algorithms; feature selection via PSO, DA, GWO, GA, WOA; light curve-based optimal sliding window size selection
- **Key Results:** 98.98% binary / 99.92% multiclass on EDGE-IIoT; 99.94% binary / 100% multiclass on IoTID20; 99.99% on Aalto; 96.52% on RT-IoT 2022
- **URL:** https://www.sciencedirect.com/science/article/abs/pii/S2542660525000587

---

### Paper #18: Online Deep Learning for IoT Under Drift (2022, foundational)
- **Title:** "Intrusion Detection in the IoT Under Data and Concept Drifts: Online Deep Learning Approach"
- **Authors:** (Polytechnique Montreal group)
- **Venue:** IEEE Internet of Things Journal, Vol. 9, Issue 20, October 2022
- **DOI:** 10.1109/JIOT.2022.3167005
- **Datasets:** IoT intrusion detection data streams
- **Drift Detection:** PCA-based variance change monitoring across features in data streams
- **Adaptation:** Online outlier detection (divergence from historical and temporally close data); online DNN with Hedge weighting mechanism for dynamic hidden layer sizing
- **Key Results:** Reduces false positives by ~6% and false negatives by ~4.5%; stabilizes IDS performance on both training/testing vs. static DNN
- **URL:** https://ieeexplore.ieee.org/document/9755949/

---

### Paper #19: SSID (IEEE TIFS 2024)
- **Title:** "Online Self-Supervised Deep Learning for Intrusion Detection Systems"
- **Authors:** Mert Nakip, Erol Gelenbe
- **Venue:** IEEE Transactions on Information Forensics and Security, 2024
- **DOI:** 10.1109/TIFS.2024.3402148
- **Datasets:** Public IDS datasets (not specified by name in abstracts)
- **Drift Detection:** Statistical trustworthiness estimation
- **Adaptation:** Auto-Associative Deep Random Neural Network; fully online learning with zero human intervention or prior offline training
- **Key Results:** Competitive with well-known ML/DL models while requiring no human labeling or offline training
- **URL:** https://ieeexplore.ieee.org/document/10531267/

---

### Paper #20: Model Retraining with Isolation Forest (2025)
- **Title:** "Model Retraining upon Concept Drift Detection in Network Traffic Big Data"
- **Authors:** Sikha S. Bagui, Mohammad Pale Khan, Chedlyne Valmyr, Subhash C. Bagui
- **Venue:** Future Internet, Vol. 17, Issue 8, Article 328, 2025
- **DOI:** (MDPI Future Internet)
- **Datasets:** UWF-ZeekDataFall22 (Zeek Connection Logs via Security Onion 2, labeled with MITRE ATT&CK)
- **Drift Detection:** Isolation Forest for anomaly-based drift detection; detects distribution changes even with small batch sizes
- **Adaptation:** Random Forest model retraining triggered upon drift detection
- **Key Results:** Identifies retraining trigger points; early retraining keeps model synchronized with latest data
- **URL:** https://www.mdpi.com/1999-5903/17/8/328

---

### Paper #21: MEDA (IEEE 2025)
- **Title:** "MEDA: MoE-based Concept Drift Adaptation for In-vehicle Network Intrusion Detection"
- **Authors:** (Not fully extracted)
- **Venue:** IEEE Conference Publication, 2025
- **DOI:** (IEEE Xplore Document 10946581)
- **Datasets:** In-vehicle network data
- **Drift Detection:** Integrated drift detection within MoE framework
- **Adaptation:** Mixture of Experts (MoE) method for real-time self-updating IDS
- **Key Results:** Real-time IDS adaptation for in-vehicle networks under evolving traffic patterns
- **URL:** https://ieeexplore.ieee.org/document/10946581/

---

### Paper #22: GPrUO-TD2NN (2026)
- **Title:** "GPrUO-TD2NN: Optimized Drift-Enabled Deep Learning Framework for Intrusion Detection in Wireless Sensor Networks"
- **Authors:** D. M. Gohil
- **Venue:** International Journal of Communication Systems, Vol. 39, No. 5, 2026
- **DOI:** 10.1002/dac.70440
- **Datasets:** "Data sharing not applicable" stated; uses WSN intrusion detection benchmarks (not explicitly named)
- **Drift Detection:** Internal covariance analysis of data distributions (drift mechanism)
- **Adaptation:** Transfer learning (GPrUO-TD2NN); grouped probabilistic update optimization (GPrUO) for parameter tuning; GPrUS2M for data balancing
- **Key Results:** 96.92% precision, 97.29% recall, 97.16% accuracy, 97.10% F1-score (90% training)
- **URL:** https://onlinelibrary.wiley.com/doi/10.1002/dac.70440

---

### Paper #23: Online Ensemble for WSN (2023)
- **Title:** "An Online Ensemble Learning Model for Detecting Attacks in Wireless Sensor Networks"
- **Authors:** Hiba Tabbaa, Samir Ifzarne, Imad Hafidi
- **Venue:** Computing and Informatics, Vol. 42, No. 4, 2023
- **Datasets:** **WSN-DS** (Blackhole, Grayhole, Flooding, Scheduling attacks)
- **Drift Detection:** Implicit via ARF's built-in ADWIN detector
- **Adaptation:** Heterogeneous ensemble (ARF + HAT); Homogeneous ensemble (10x HAT)
- **Key Results:** Heterogeneous ARF+HAT: 96.84%; Homogeneous 10xHAT: 97.2%; efficient for concept drift in resource-constrained WSNs
- **URL:** https://www.cai.sk/ojs/index.php/cai/article/view/2023_4_1013

---

### Paper #24: HATS-RL for Fog Networks (2026)
- **Title:** "A Comparative Analysis of Self-Aware Reinforcement Learning Models for Real-Time Intrusion Detection in Fog Networks"
- **Authors:** (Not fully extracted)
- **Venue:** Future Internet, Vol. 18, Issue 2, Article 100, 2026
- **Datasets:** Realistic fog network simulator with streaming traffic, multi-type attack bursts and gradual concept drift
- **Drift Detection:** Online uncertainty estimation and concept-drift detection integrated in RL model
- **Adaptation:** HATS-RL (Hierarchical Adaptive Thompson Sampling-Reinforcement Learning); F-HATS-RL (Federated variant)
- **Key Results:** F-HATS-RL: AUROC 0.933, AUPR 0.857, latency 0.27ms, energy 0.0137 mJ
- **URL:** https://www.mdpi.com/1999-5903/18/2/100

---

### Paper #25: CAEAID (2025)
- **Title:** "CAEAID: An Incremental Contrast Learning-Based Intrusion Detection Framework for IoT Networks"
- **Authors:** (Not fully extracted)
- **Venue:** Computer Networks, Vol. 262, 2025
- **DOI:** 10.1016/j.comnet.2025.111161
- **Datasets:** IoTID20, CICIDS2018
- **Drift Detection:** Improved extreme value theory for adaptive threshold setting
- **Adaptation:** Contrastive autoencoder for low-dimensional latent representations; incremental learning with pseudo-labels from detection results; consistency analysis
- **Key Results:** Accuracy improvement >1.15% (IoTID20) and >1.72% (CICIDS2018) over SOTA; F1-score improved >0.74% and >1.90%
- **URL:** https://www.sciencedirect.com/science/article/abs/pii/S138912862500129X

---

### Paper #26: DRL from Drifting Environments (2026)
- **Title:** "Deep Reinforcement Learning from Drifting Network Environments in Anomaly Detection"
- **Authors:** Zheng, J., Wang, S., Xu, Z., Zhang, X., Cheng, H.
- **Venue:** Springer, 2026
- **DOI:** 10.1007/978-3-031-94445-1_17
- **Datasets:** Two public datasets exhibiting real-world drift
- **Drift Detection:** Integrated within DRL framework
- **Adaptation:** Deep reinforcement learning-based autonomous adaptation
- **Key Results:** F1 score of 0.93, outperforming state-of-the-art methods
- **URL:** https://link.springer.com/chapter/10.1007/978-3-031-94445-1_17

---

### Paper #27: FORT-IDS (2025)
- **Title:** "FORT-IDS: A Federated, Optimized, Robust and Trustworthy Intrusion Detection System for IIoT Security"
- **Authors:** Alanoud Al Mazroa
- **Venue:** Scientific Reports, December 2025
- **DOI:** 10.1038/s41598-025-31025-x
- **Datasets:** Multiple IIoT benchmarks (cross-domain)
- **Drift Detection:** Heterogeneous feature space alignment through lightweight mapping/normalization
- **Adaptation:** FedAvg-based federated learning with K=20 clients; SMOTE for class balancing
- **Key Results:** FedAvg accuracy converging to 0.934 by round 5
- **URL:** https://www.nature.com/articles/s41598-025-31025-x

---

### Paper #28: LQB-IDS (2025)
- **Title:** "An adaptive intrusion detection system for the internet of things using large language models and post-quantum-secure blockchain"
- **Authors:** (Not fully extracted)
- **Venue:** Computer Networks (Elsevier), 2025
- **DOI:** 10.1016/j.comnet.2025.111856 (approx.)
- **Datasets:** IoT traffic datasets
- **Drift Detection:** LLM-based analysis for unknown attack patterns
- **Adaptation:** Dual-switch: lightweight ML for known attacks + LLM for unknown attacks; credit scoring with PQ-Secure Blockchain database
- **Key Results:** 0.98 detection rate; 0.97 average F1-score without labeled data
- **URL:** https://www.sciencedirect.com/science/article/abs/pii/S1389128625007856

---

### Paper #29: Incremental FL for IoT IDS (ICISSP 2026)
- **Title:** "Incremental Federated Learning for Intrusion Detection in IoT Networks"
- **Authors:** (Not fully extracted)
- **Venue:** 12th International Conference on Information Systems Security and Privacy (ICISSP 2026), Vol. 1, pp. 97-107
- **Datasets:** KDDCup99 (referenced)
- **Drift Detection:** Data and model-based measures against catastrophic learning under drift
- **Adaptation:** LSTM models in federated learning setting with incremental learning
- **Key Results:** Comprehensive performance analysis of incremental FL for non-stationary IDS
- **URL:** https://www.scitepress.org/Papers/2026/144759/144759.pdf

---

### Paper #30: NDSS 2025 - Malware Concept Drift
- **Title:** "Revisiting Concept Drift in Windows Malware Detection: Adaptation to Real Drifted Malware with Minimal Samples"
- **Authors:** Adrian Shuai Li, Arun Iyengar, Ashish Kundu, Elisa Bertino
- **Venue:** NDSS Symposium 2025
- **Datasets:** Real-world malware data (March-September 2024); MalwareDrift dataset
- **Drift Detection:** Observes features inconsistent across pre-drift and post-drift data
- **Adaptation:** Focuses on consistent features across drift boundary; minimal sample adaptation
- **Key Results:** Matches supervised training accuracy with 5X less labeling; improves family-level classification by 9-14% with just 10 new samples per family
- **URL:** https://www.ndss-symposium.org/ndss-paper/revisiting-concept-drift-in-windows-malware-detection-adaptation-to-real-drifted-malware-with-minimal-samples/
- **Code:** https://github.com/gloryer/malware-detection-concept-drift

---

### Paper #31: WHTE for IoMT (2022, foundational)
- **Title:** "WHTE: Weighted Hoeffding Tree Ensemble for Network Attack Detection at Fog-IoMT"
- **Authors:** (ResearchGate reference)
- **Venue:** (Published 2022)
- **Datasets:** IoMT network traffic
- **Drift Detection:** Built-in Hoeffding tree drift detection mechanisms
- **Adaptation:** Weighted Hoeffding Tree Ensemble (WHTE) - multiple incremental classifiers including IKNN, INB, HTMC, HTNB, HTNBA
- **Key Results:** ~100% accuracy; <6 MiB memory; WHTE more accurate and less sensitive to concept drift
- **URL:** https://www.researchgate.net/publication/363081849

---

### Paper #32: Hoeffding Tree for IoT DDoS (2023)
- **Title:** "Applying Hoeffding Tree Algorithms for Effective Stream Learning in IoT DDoS Detection"
- **Authors:** (Not fully extracted)
- **Venue:** IEEE Conference Publication, 2023
- **DOI:** (IEEE Xplore Document 10361862)
- **Datasets:** IoT DDoS traffic datasets
- **Drift Detection:** Built-in Hoeffding tree mechanisms
- **Adaptation:** Stream learning via Hoeffding Tree algorithms without storing all traffic
- **Key Results:** Effective DDoS detection with stream learning; adaptable to concept drift
- **URL:** https://ieeexplore.ieee.org/document/10361862/

---

## 2. CRITICAL QUESTIONS ANSWERED

### Q1: Has concept drift been studied on WSN-DS?

**ANSWER: Partially YES, but no dedicated concept drift study exists.**

**Evidence:**
- **Paper #23 (Tabbaa et al., 2023)** is the CLOSEST: They applied online ensemble learning (ARF + HAT) on WSN-DS and stated their models are "efficient and effective in dealing with concept drift while taking into account the resource constraints of WSNs." However, they did NOT explicitly inject or simulate concept drift. They used ARF which has built-in ADWIN drift detection, but the study was primarily a classification evaluation, not a concept drift investigation.
- **Paper #22 (Gohil, 2026, GPrUO-TD2NN)** addresses drift-enabled IDS for WSNs but does NOT explicitly name WSN-DS as its dataset (the paper says "data sharing not applicable").
- **No paper found explicitly and primarily studies concept drift ON the WSN-DS dataset as its central research question.**

### Q2: Has anyone simulated drift by varying WSN attack distributions?

**ANSWER: NO explicit evidence found for WSN-DS specifically. YES for IDS in general.**

**Evidence:**
- **Paper #7 (Costa et al., 2025)** created a "Concept Drift Stream Generator for Intrusion Detection Systems" that can simulate abrupt and gradual drift, but this was for general IDS data, NOT specifically WSN-DS.
- **Paper #24 (HATS-RL, 2026)** used a "realistic fog network simulator with multi-type attack bursts and gradual concept drift" but not on WSN-DS.
- **Paper #2 (Chu et al., 2024)** simulated drift on synthetic datasets (SEA, Stagger, Rotating Hyperplane) and the IIoTset dataset, NOT WSN-DS.
- **No paper was found that specifically varies attack distributions (Blackhole, Grayhole, Flooding, Scheduling ratios) on WSN-DS to simulate concept drift.**

### Q3: What drift detection methods exist for IDS?

**Comprehensive taxonomy verified from search results:**

| Method | Type | Used In (from catalog) |
|--------|------|----------------------|
| **ADWIN** (Adaptive Windowing) | Statistical/Window-based | Papers #14, #15, #17, #23 |
| **DDM** (Drift Detection Method) | Performance/Error-rate | Papers #5, #14, #15 |
| **EDDM** (Early Drift Detection Method) | Performance/Error-distance | Paper #17 |
| **HDDM-A** (Hoeffding Drift Detection Method) | Statistical/Bound-based | Paper #17 |
| **PCA-based variance monitoring** | Distribution/Feature-based | Paper #18 |
| **Isolation Forest** | Anomaly-based | Paper #20 |
| **EMNCD** (Ensemble Non-parametric: KS + Wilcoxon + Mann-Kendall) | Statistical ensemble | Paper #2 |
| **OASW** (Online Adaptive Sliding Window) | Hybrid window+performance | Paper #16 |
| **Drift-aware perturbation** (implicit) | Representation-based | Paper #4 (ReCDA) |
| **Internal covariance of data distributions** | Distribution-based | Paper #22 (GPrUO-TD2NN) |
| **LLM-based dynamic analysis** | Semantic/AI-based | Paper #28 (LQB-IDS) |
| **PHT** (Page-Hinkley Test) | Statistical/Sequential | Referenced in surveys |
| **STEPD** (Statistical Test of Equal Proportions) | Statistical | Referenced in surveys |

### Q4: What is the CLOSEST paper to "concept drift on WSN-DS"?

**ANSWER: Paper #23 - Tabbaa et al. (2023) "An Online Ensemble Learning Model for Detecting Attacks in Wireless Sensor Networks"**

**Why it is closest:**
1. It is the ONLY paper found that uses BOTH WSN-DS AND online/streaming learning methods with implicit concept drift handling (ARF with built-in ADWIN).
2. It explicitly mentions "concept drift" as a challenge it addresses.
3. It uses ARF (Adaptive Random Forest) + HAT (Hoeffding Adaptive Tree) which are standard concept drift-aware classifiers.
4. Results on WSN-DS: ARF+HAT heterogeneous ensemble = 96.84%, homogeneous HAT = 97.2%.

**Why it is NOT a full concept drift study on WSN-DS:**
- No explicit drift injection/simulation on WSN-DS.
- No before/after drift performance analysis.
- No drift point detection or localization on WSN-DS.
- WSN-DS is treated as a static classification benchmark, not a streaming/evolving dataset.

**Second closest: Paper #22 - Gohil (2026) GPrUO-TD2NN:**
- Addresses drift in WSN IDS.
- Uses transfer learning + drift mechanism.
- But does NOT explicitly use WSN-DS; dataset not named.

---

## 3. DRIFT DETECTION METHODS TAXONOMY FOR IDS

### 3.1 Performance-Based (Supervised)
- **DDM:** Monitors error rate; triggers when error significantly exceeds minimum + threshold
- **EDDM:** Monitors distance between consecutive misclassifications
- **HDDM:** Uses Hoeffding bounds; monitors false positive/negative rates

### 3.2 Window/Distribution-Based
- **ADWIN:** Adaptive sliding window; compares sub-window means; no pre-set window size
- **OASW:** Combines sliding + adaptive + performance ideas; lightweight for IoT
- **Dual Adaptive Windows (DWOIDS):** Two windows monitoring prediction error for Hoeffding tree

### 3.3 Statistical Test-Based
- **EMNCD:** Ensemble of KS, Wilcoxon, Mann-Kendall non-parametric tests
- **PHT (Page-Hinkley):** Sequential analysis; monitors cumulative mean deviation
- **PCA-based:** Monitors variance changes across feature streams

### 3.4 Anomaly/Isolation-Based
- **Isolation Forest:** Detects anomalous distribution changes; triggers retraining

### 3.5 Representation/AI-Based (Emerging 2024-2026)
- **ReCDA drift-aware perturbation:** Self-supervised; no labeled drift samples needed
- **DRL-based implicit detection:** Detects "Recall Collapse" (REACT-D3QN)
- **LLM-based semantic analysis:** Detects unknown attacks dynamically (LQB-IDS)
- **MoE-based (MEDA):** Mixture of experts adapts to distribution changes

---

## 4. RESEARCH GAP ANALYSIS

### Confirmed Research Gaps (as of March 2026):

1. **NO dedicated concept drift study on WSN-DS exists.** The WSN-DS dataset has been used exclusively for static classification tasks. No paper simulates, detects, or adapts to concept drift specifically on WSN-DS.

2. **NO paper varies WSN attack distributions to simulate drift.** While Costa et al. (2025) built a general IDS stream generator and others used synthetic drift datasets, no work specifically manipulates the Blackhole/Grayhole/Flooding/Scheduling attack ratios in WSN-DS to create realistic drift scenarios.

3. **WSN + concept drift gap persists.** Concept drift research in IDS heavily focuses on enterprise network datasets (CIC-IDS, NSL-KDD, IoTID20). WSN-specific drift studies are virtually absent.

4. **Stream learning on WSN-DS is minimal.** Only Tabbaa et al. (2023) applied streaming algorithms (ARF, HAT) to WSN-DS, and even they did not conduct a proper drift analysis.

5. **Lightweight drift detection for resource-constrained WSN nodes is underexplored.** While OASW (Yang & Shami, 2021) demonstrated lightweight detection on Raspberry Pi, this was not tested on WSN-specific attack patterns.

### Summary Table: Dataset Coverage in Concept Drift + IDS Papers

| Dataset | Papers Using It | Drift Study? |
|---------|----------------|-------------|
| CIC-IDS 2017/2018 | #3, #4, #8, #13, #15 | YES (multiple) |
| IoTID20 | #14, #15, #16, #17, #25 | YES (multiple) |
| NSL-KDD | #3, #16 | YES |
| MAWILab | #5 | YES |
| Edge-IIoTset | #2, #17 | YES |
| KDD Cup '99 | #3, #29 | YES |
| UWF-ZeekDataFall22 | #20 | YES |
| **WSN-DS** | **#23 only** | **NO (static only)** |

---

## Sources

- [Shyaa et al. 2024 - Evolving Cybersecurity Frontiers Survey](https://www.sciencedirect.com/science/article/pii/S0952197624013010)
- [Chu et al. 2024 - EMNCD IoT Data Streams](https://www.aimspress.com/article/doi/10.3934/math.2024076)
- [Shyaa et al. 2023 - GPC Variants](https://www.mdpi.com/1424-8220/23/7/3736)
- [ReCDA - KDD 2024](https://dl.acm.org/doi/10.1145/3637528.3672007)
- [Jemili et al. 2025 - DDM-ORF](https://www.emerald.com/insight/content/doi/10.1108/ijpcc-12-2023-0358/full/html)
- [Camarda et al. 2025 - Managing Concept Drift](https://ceur-ws.org/Vol-3962/paper42.pdf)
- [Costa et al. 2025 - Drift Stream Generator](https://ceur-ws.org/Vol-4121/Ital-IA_2025_paper_94.pdf)
- [REACT-D3QN 2026](https://link.springer.com/article/10.1186/s42400-026-00558-7)
- [Yang et al. 2025 - Self-Supervised Adaptation IEEE TDSC](https://ieeexplore.ieee.org/document/11125964/)
- [Liao 2026 - Dynamic IDS](https://www.gbspress.com/index.php/JCSSR/article/view/553)
- [Hu et al. 2025 - DWOIDS ICCS](https://link.springer.com/chapter/10.1007/978-3-031-97626-1_15)
- [Jia et al. 2026 - DyNA-IDS](https://link.springer.com/chapter/10.1007/978-981-95-6203-9_18)
- [Gupta et al. 2025 - NetGuard](https://arxiv.org/abs/2503.03022)
- [Beshah et al. 2024 - Drift Adaptive DDoS](https://www.mdpi.com/2079-9292/13/6/1004)
- [Yang & Shami 2021 - PWPAE](https://ieeexplore.ieee.org/document/9685338/)
- [Yang & Shami 2021 - OASW](https://arxiv.org/abs/2104.10529)
- [OASIS 2025](https://www.sciencedirect.com/science/article/abs/pii/S2542660525000587)
- [IoT Under Data and Concept Drifts 2022](https://ieeexplore.ieee.org/document/9755949/)
- [Nakip & Gelenbe 2024 - SSID](https://ieeexplore.ieee.org/document/10531267/)
- [Bagui et al. 2025 - Isolation Forest Retraining](https://www.mdpi.com/1999-5903/17/8/328)
- [MEDA 2025 - MoE for In-vehicle IDS](https://ieeexplore.ieee.org/document/10946581/)
- [Gohil 2026 - GPrUO-TD2NN WSN](https://onlinelibrary.wiley.com/doi/10.1002/dac.70440)
- [Tabbaa et al. 2023 - Online Ensemble WSN-DS](https://www.cai.sk/ojs/index.php/cai/article/view/2023_4_1013)
- [HATS-RL 2026 - Fog Networks](https://www.mdpi.com/1999-5903/18/2/100)
- [CAEAID 2025](https://www.sciencedirect.com/science/article/abs/pii/S138912862500129X)
- [Zheng et al. 2026 - DRL Drifting Networks](https://link.springer.com/chapter/10.1007/978-3-031-94445-1_17)
- [Al Mazroa 2025 - FORT-IDS](https://www.nature.com/articles/s41598-025-31025-x)
- [LQB-IDS 2025](https://www.sciencedirect.com/science/article/abs/pii/S1389128625007856)
- [Incremental FL ICISSP 2026](https://www.scitepress.org/Papers/2026/144759/144759.pdf)
- [Li et al. NDSS 2025 - Malware Drift](https://www.ndss-symposium.org/ndss-paper/revisiting-concept-drift-in-windows-malware-detection-adaptation-to-real-drifted-malware-with-minimal-samples/)
- [WHTE 2022 - IoMT](https://www.researchgate.net/publication/363081849)
- [Hoeffding Tree IoT DDoS 2023](https://ieeexplore.ieee.org/document/10361862/)
- [WSN-DS Original Dataset Paper - Almomani 2016](https://onlinelibrary.wiley.com/doi/full/10.1155/2016/4731953)
