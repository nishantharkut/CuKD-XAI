# Review card: javed2024thermostat

**Status:** PASS1B+PASS2_TEXT+PASS2_VISUAL_TITLE OK
**PDF text pages extracted:** 15
**Ground truth extract:** `_extract/javed2024thermostat.full.txt`
**Evidence JSON:** `_pass1b_evidence/javed2024thermostat.json`
**Generated:** 2026-08-06

## Identity (page-1 text)
- **Title:** Qureshi, A.-u.-H.; Jawad, M.; Arshad, J.; Larijani, H. Embedding Tree-Based Intrusion Detection System in Smart
- **Tags:** XAI, MCU, gateway

## Abstract (extracted)
> IoT devices with limited resources, and in the absence of gateways, become vulnerable to various attacks, such as denial of service (DoS) and man-in-the-middle (MITM) attacks. Intrusion detection systems (IDS) are designed to detect and respond to these threats in IoT environments. While machine learning-based IDS have typically been deployed at the edge (gateways) or in the cloud, in the absence of gateways, the IDS must be embedded within the sensor nodes themselves. Available datasets mainly contain features extracted from network traffic at the edge (e.g., Raspberry Pi/computer) or cloud servers. We developed a unique dataset, named as Intrusion Detection in the Smart Homes (IDSH) dataset, which is based on features retrievable from microcontroller-based IoT devices. In this work, a Tree-based IDS is embedded into a smart thermostat for real-time intrusion detection. The results demonstrated that the IDS achieved an accuracy of 98.71% for binary classification with an inference time of 276 microseconds, and an accuracy of 97.51% for multi- classification with an inference time of 273 microseconds. Real-time testing showed that the smart thermostat is capable of detecting DoS and MITM attacks without relying on a gateway or cloud.

## Table headers present in PDF text (exact lines)
- `Table 1. The timestamp, source IP, and destination IP were excluded to prevent overfitting.`
- `Table 1. Inputs and target of IDS.`
- `Table 2. Selected features.`
- `Table 3. Performance of CatBoost for binary classification.`
- `Table 4. Performance of CatBoost for binary classification using feature selection.`
- `Table 5. Performance of CatBoost for multi-classification without using feature selection.`
- `Table 5. Cont.`
- `Table 6. Performance of CatBoost for multi-classification with feature selection.`
- `Table 7. Implementation of IDS on the smart thermostat for binary and multi-classification.`
- `Table 7. Cont.`

## Metric-bearing lines (exact PDF lines; PASS2-verified)
- `intrusion detection. The results demonstrated that the IDS achieved an accuracy of 98.71% for binary`
- `classification with an inference time of 276 microseconds, and an accuracy of 97.51% for multi-`
- `XGBoost-based IDS for binary classification in an ESP32 (https://www.espressif.com/en/`
- `products/socs/esp32 (accessed on 6 October 2024))-powered smart thermostat. The smart`
- `com/espressif/esp-lwip (accessed on 6 October 2024)) library for the ESP32 provides`
- `the novel implementation of CatBoost-based IDS on the ESP32 microcontroller. CatBoost`
- `capabilities of devices like the ESP32.`
- `credibility index. The experimental results showed that the accuracy of PCIDS was 94.7%.`
- `The Deep-IDS was trained using the CIC-IDS2017 dataset and achieved an accuracy of`
- `The results showed that DT outperformed other techniques with an accuracy of 99%.`
- `The authors in [30] improved the precision of IDS using the Shapley (SHAP) value-`
- `accuracy for binary classification and 97.06% for multi-classification. The authors in [32]`
- `technique achieved an excellent accuracy of 99.80%.`
- `developed an IDSH dataset [23] that was captured on the ESP32 without using packet`
- `The block diagram of the testbed is shown in Figure 2 and the parameters collected at`
- `In this study, XGBoost and CatBoost models are deployed on an ESP32-based smart`
- `processing capabilities of devices like the ESP32.`
- `stat built with an ESP32 microcontroller. The ESP32 microcontroller has 440 KB of ROM`
- `discuss the implementation of CatBoost IDS on an ESP32-based smart thermostat and`
- `0.3]. The performance of CatBoost was evaluated in terms of accuracy, precision, recall,`
- `and F1-score. The testing results are shown in Table 3. The highest accuracy of 99.03% was`
- `F1-Score`
- `show that the CatBoost model with 10 features and a depth of 10 gives the highest accuracy`
- `F1-Score`
- `learning rate, regularization parameter (L2_Leaf_reg), and the number of trees. The range`
- `terms of accuracy, precision, recall, and F1-score. The testing results are shown in Table 5.`
- `The highest accuracy of 98.15% was achieved at a depth of seven.`
- `F1-Score`
- `F1-Score`
- `seven features, the accuracy of IDS dropped below 90%. The performance of CatBoost was`

## CuKD freeze notes (non-numeric)
- XAI neighborhood â†’ do not invent Spearman Ï; C6 is CuKD measurement.
- MCU/embedded neighborhood â†’ compare to C4 dual-board RF-KD HIL; Javed is tree-on-ESP32 prior.
- **Numbers only from exact lines above.**

## Verification
- PASS1B: evidence JSON written with exact lines from extract
- PASS2: all `40` quoted lines re-found in full extract (by construction)
- PASS2_TEXT: OK; PASS2_VISUAL_TITLE: OK

## PASS2 independent text re-check (2026-08-06)
- **PASS2_OK** â€” 40/40 quotes exact-match in full extract

## PASS2 visual title check (2026-08-06)
- page_001.png exists: True
- title word hit ratio vs page-1 text: 1.000
- **PASS2_VISUAL_TITLE_OK** (title words supported by page-1 text + PNG on disk)

## DEEP_VISUAL (manual image pages 001,008,010,011)

Verified from page images (not only text layer):
- On-device Table 7: CatBoost binary best **98.71%** @ **276 µs**, 9 features, Trees=200 Depth=6
- Multi-class CatBoost **97.51%** @ **267–273 µs**
- XGBoost prior binary 97.66% @ 3515 µs
- Quantization float32 on ESP32; Fig.3 shows quant vs non-quant agreement on samples
- Dataset IDSH (not WSN-DS); DoS+MITM

