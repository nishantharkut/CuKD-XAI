# Edge-IIoTset Literature Metric Comparison

This report is generated from completed Edge-IIoTset confusion matrices. It does not retrain any model.

- Protocol: `literature_comparable`
- Experiment scope: `selected_capacity`
- Seeds: `[42, 123, 456, 789, 1001]`
- Classes: `15`
- Input dimension: `49`

## Key Interpretation

- F1 is compared apples-to-apples only when the paper states the averaging type: macro-to-macro, weighted-to-weighted, or micro-to-micro.
- If a paper reports plain `F1-score` without stating macro/weighted/micro, the matched F1 gap is `NR`; the table separately shows an explicitly non-apples-to-apples best-F1 reference gap.
- Use accuracy and deployment footprint for the broad literature comparison; use macro-F1 only for minority-class robustness and only against papers that explicitly report macro-F1.
- No training rerun is needed for this metric recalculation because confusion matrices are already present in the completed results.

## Literature Comparison

| Paper / Work | Year | Paper Model | Paper Acc. (%) | Paper F1 (%) | Paper F1 Type | Paper Footprint | Our Matched Model | Our Acc. (%) | Our Macro-F1 (%) | Our Weighted-F1 (%) | Our Micro-F1 (%) | Our Matched F1 (%) | Our Best-F1 Ref. (%) | Best-F1 Type | Our Size (KB) | Acc. Gap (pts) | Matched F1 Gap (pts) | Best-F1 Ref. Gap (pts) | F1 Basis | Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ferrag et al., Edge-IIoTset | 2022 | DNN | 94.67 | NR | not_reported | NR | Student C RF-KD | 96.85 | 82.44 | 96.86 | 96.85 | NR | NR | NR | 61.06 | 2.18 | NR | NR | not comparable: paper F1 not reported | We exceed the original 15-class DNN accuracy with a compressed student. |
| Ferrag et al., Edge-IIoTset | 2022 | RF | 80.83 | NR | not_reported | NR | Student C RF-KD | 96.85 | 82.44 | 96.86 | 96.85 | NR | NR | NR | 61.06 | 16.02 | NR | NR | not comparable: paper F1 not reported | Large accuracy improvement over the original RF baseline. |
| Ferrag et al., Edge-IIoTset | 2022 | SVM | 77.61 | NR | not_reported | NR | Student C RF-KD | 96.85 | 82.44 | 96.86 | 96.85 | NR | NR | NR | 61.06 | 19.24 | NR | NR | not comparable: paper F1 not reported | Large accuracy improvement over the original SVM baseline. |
| Ferrag et al., Edge-IIoTset | 2022 | KNN | 79.18 | NR | not_reported | NR | Student C RF-KD | 96.85 | 82.44 | 96.86 | 96.85 | NR | NR | NR | 61.06 | 17.67 | NR | NR | not comparable: paper F1 not reported | Large accuracy improvement over the original KNN baseline. |
| Ferrag et al., Edge-IIoTset | 2022 | DT | 67.11 | NR | not_reported | NR | Student C RF-KD | 96.85 | 82.44 | 96.86 | 96.85 | NR | NR | NR | 61.06 | 29.74 | NR | NR | not comparable: paper F1 not reported | Large accuracy improvement over the original DT baseline. |
| Diab et al., Hardware-Aware ML/DL | 2025 | LightGBM | 95.25 | 94.74 | unspecified | 74.93 KB flash, 1.13 KB RAM, 1.20K ops | Student C RF-KD | 96.85 | 82.44 | 96.86 | 96.85 | NR | 96.86 | weighted | 61.06 | 1.60 | NR | 2.12 | not comparable: paper F1 averaging not specified | Accuracy and storage are competitive; F1 type is unspecified, so the best-F1 reference is only optimistic context. |
| Diab et al., Hardware-Aware ML/DL | 2025 | XGBoost | 95.11 | 94.42 | unspecified | 266.59 KB flash, 0.51 KB RAM, 4.27K ops | Student C RF-KD | 96.85 | 82.44 | 96.86 | 96.85 | NR | 96.86 | weighted | 61.06 | 1.74 | NR | 2.44 | not comparable: paper F1 averaging not specified | Accuracy is higher and storage is smaller; F1 type is unspecified, so the best-F1 reference is only optimistic context. |
| Diab et al., Hardware-Aware ML/DL | 2025 | RF | 94.12 | 93.21 | unspecified | 211.22 KB flash, 4.61 KB RAM, 3.38K ops | Student C RF-KD | 96.85 | 82.44 | 96.86 | 96.85 | NR | 96.86 | weighted | 61.06 | 2.73 | NR | 3.65 | not comparable: paper F1 averaging not specified | Accuracy is higher and storage is smaller; F1 type is unspecified, so the best-F1 reference is only optimistic context. |
| Diab et al., Hardware-Aware ML/DL | 2025 | HW-NAS 1D-CNN | 96.73 | 97.24 | unspecified | 190.34 KB flash, 6.89 KB RAM, 838.89K FLOPs | Student C RF-KD | 96.85 | 82.44 | 96.86 | 96.85 | NR | 96.86 | weighted | 61.06 | 0.12 | NR | -0.38 | not comparable: paper F1 averaging not specified | Accuracy is essentially matched with smaller storage; CNN F1 remains stronger. |
| Abualhassan et al., IIoT-TinyDNN | 2025 | TinyDNN | 92.99 | NR | not_reported | 2,255 params, 4.38K FLOPs | Student A LightGBM-KD | 96.61 | 81.43 | 96.60 | 96.61 | NR | NR | NR | 9.31 | 3.62 | NR | NR | not comparable: paper F1 not reported | Similar parameter scale and higher accuracy. |
| Abualhassan et al., IIoT-TinyDNN | 2025 | TinyCNN | 95.55 | NR | not_reported | 5,967 params, 32.13K FLOPs | Student B LightGBM-KD | 96.80 | 82.20 | 96.79 | 96.80 | NR | NR | NR | 22.56 | 1.25 | NR | NR | not comparable: paper F1 not reported | Similar parameter scale, higher accuracy, and lower FLOPs. |
| Hasan et al., Autoencoder Feature Learning | 2025 | AE + DT/XGB/LGBM/LDA/TabNet/LSTM | 99.94 | 99.94 | unspecified | NR, Jetson Nano inference reported | Student C RF-KD | 96.85 | 82.44 | 96.86 | 96.85 | NR | 96.86 | weighted | 61.06 | -3.09 | NR | -3.08 | not comparable: paper F1 averaging not specified | Their metrics are higher; our defense is smaller compressed-student deployability. |
| Abdi et al., CNN Multiclass Attack Classification | 2025 | CNN | 95.50 | 94.60 | unspecified | NR | Student C RF-KD | 96.85 | 82.44 | 96.86 | 96.85 | NR | 96.86 | weighted | 61.06 | 1.35 | NR | 2.26 | not comparable: paper F1 averaging not specified | Our compressed student is higher in accuracy; their F1 type is unspecified and their CNN is not presented as an ultra-small deployment artifact. |
| Yagiz and Goktas, LENS-XAI | 2025 | LENS-XAI Student | 95.31 | 95.36 | unspecified | NR, KD/VAE/XAI with 10% training-data claim | Student C RF-KD | 96.85 | 82.44 | 96.86 | 96.85 | NR | 96.86 | weighted | 61.06 | 1.54 | NR | 1.50 | not comparable: paper F1 averaging not specified | Accuracy is higher; their F1 type is unspecified, while our macro-F1 remains lower because it stresses minority classes. |
| Salehiyan et al., Transformer-GAN-AE | 2025 | Transformer-GAN-AE | 98.63 | NR | not_reported | RTX 3090 workstation; footprint NR | Student C RF-KD | 96.85 | 82.44 | 96.86 | 96.85 | NR | NR | NR | 61.06 | -1.78 | NR | NR | not comparable: paper F1 not reported | Their heavy hybrid DL model has higher accuracy; our contribution is much smaller compressed-student deployment. |
| WO-XGB Feature-Level Ensemble | 2025 | WO-XGB | 99.98 | 99.97 | unspecified | NR, XGBoost ensemble | Student C RF-KD | 96.85 | 82.44 | 96.86 | 96.85 | NR | 96.86 | weighted | 61.06 | -3.13 | NR | -3.11 | not comparable: paper F1 averaging not specified | Their ensemble is much stronger in raw metrics; not an ultra-small neural student comparison. |
| Ishtiaq et al., CST-AFNet | 2025 | CST-AFNet CNN-BiGRU dual attention | 99.97 | 99.30 | unspecified | NR, multi-scale CNN + BiGRU + dual attention | Student C RF-KD | 96.85 | 82.44 | 96.86 | 96.85 | NR | 96.86 | weighted | 61.06 | -3.12 | NR | -2.44 | not comparable: paper F1 averaging not specified | Their deep attention model is raw-metric superior but not comparable to a 61 KB compressed MLP target. |
| Neuro-Symbolic Edge IDS | 2026 | Neuro-symbolic KD framework | 94.30 | 93.50 | macro | 37% memory reduction, 54% latency reduction; absolute size NR | Student C RF-KD | 96.85 | 82.44 | 96.86 | 96.85 | 82.44 | NR | NR | 61.06 | 2.55 | -11.06 | NR | macro-to-macro | Our accuracy is higher; their explicitly reported macro-F1 and interpretability claims remain stronger. |
| Hybrid LLM/HGB IDS | 2026 | BERT embeddings + RF selection + HGB | 98.19 | 98.19 | weighted | Frozen BERT + HGB; footprint NR | Student C RF-KD | 96.85 | 82.44 | 96.86 | 96.85 | 96.86 | NR | NR | 61.06 | -1.34 | -1.33 | NR | weighted-to-weighted | Their hybrid representation is stronger in macro/weighted metrics; our model is far smaller and standalone. |
| Abdulkareem et al., FI-SEL | 2024 | Feature-importance stacked ensemble | 87.37 | 80.88 | unspecified | 8 features; absolute model size NR | Student A LightGBM-KD | 96.61 | 81.43 | 96.60 | 96.61 | NR | 96.61 | micro | 9.31 | 9.24 | NR | 15.73 | not comparable: paper F1 averaging not specified | Our smallest student is much stronger in accuracy; F1 type is unspecified, so the best-F1 reference is only optimistic context. |
| Qathrady et al., SACNN-IDS | 2024 | Self-attention CNN IDS | 99.95 | 99.79 | unspecified | NR, deep self-attention CNN | Student C RF-KD | 96.85 | 82.44 | 96.86 | 96.85 | NR | 96.86 | weighted | 61.06 | -3.10 | NR | -2.93 | not comparable: paper F1 averaging not specified | Raw metrics are much higher; this is a heavy accuracy benchmark, not a compression-equivalent model. |
| Alshehri et al., SA-DCNN | 2024 | Self-attention DCNN | 99.96 | 99.81 | unspecified | NR, self-attention DCNN | Student C RF-KD | 96.85 | 82.44 | 96.86 | 96.85 | NR | 96.86 | weighted | 61.06 | -3.11 | NR | -2.95 | not comparable: paper F1 averaging not specified | Raw metrics are much higher; not comparable to our compressed 61 KB student objective. |
| Cao et al., FedDynST | 2025 | FedDynST | 97.28 | 97.62 | unspecified | NR, federated/dynamic model | Student C RF-KD | 96.85 | 82.44 | 96.86 | 96.85 | NR | 96.86 | weighted | 61.06 | -0.43 | NR | -0.76 | not comparable: paper F1 averaging not specified | Accuracy is close but lower than their reported value; their model is not a tiny standalone student. |
| Gao et al., Lightweight TCN | 2026 | TCN, 22 features | 93.79 | 93.13 | unspecified | 16.25 KB | Student A LightGBM-KD | 96.61 | 81.43 | 96.60 | 96.61 | NR | 96.61 | micro | 9.31 | 2.82 | NR | 3.48 | not comparable: paper F1 averaging not specified | Accuracy and size are stronger; F1 type is unspecified, so the best-F1 reference is only optimistic context. |
| Gao et al., Lightweight TCN | 2026 | TCN, all features | 94.24 | 93.71 | unspecified | 25.37 KB | Student B LightGBM-KD | 96.80 | 82.20 | 96.79 | 96.80 | NR | 96.80 | micro | 22.56 | 2.56 | NR | 3.09 | not comparable: paper F1 averaging not specified | Accuracy and size are stronger; F1 type is unspecified, so the best-F1 reference is only optimistic context. |
| SEED: Edge Transformer to IoT Decisions | 2026 | EdgeBERT + IoT classifier | 99.99 | NR | not_reported | 137 KB IoT classifier + 40.6 MB EdgeBERT | Student C RF-KD | 96.85 | 82.44 | 96.86 | 96.85 | NR | NR | NR | 61.06 | -3.14 | NR | NR | not comparable: paper F1 not reported | Not a direct standalone tiny-student comparison because their result uses edge offloading. |

## Sources Used

- Ferrag et al., Edge-IIoTset: https://doi.org/10.1109/ACCESS.2022.3165809
- Diab et al., Hardware-Aware ML/DL: https://arxiv.org/abs/2512.02272
- Abualhassan et al., IIoT-TinyDNN: https://doi.org/10.1109/CSCN67557.2025.11230733
- Hasan et al., Autoencoder Feature Learning: https://doi.org/10.5220/0013203700003944
- Abdi et al., CNN multiclass classification: https://doi.org/10.3390/fi17060230
- Yagiz and Goktas, LENS-XAI: https://arxiv.org/abs/2501.00790
- Salehiyan et al., Transformer-GAN-AE: https://doi.org/10.3390/fi17070279
- WO-XGB feature-level ensemble: https://link.springer.com/article/10.1007/s43926-025-00185-7
- CST-AFNet: https://doi.org/10.1016/j.array.2025.100501
- Neuro-symbolic edge IDS: https://link.springer.com/article/10.1007/s44397-026-00047-z
- Hybrid LLM/HGB IDS: https://www.mdpi.com/1424-8220/26/4/1231
- Gao et al., lightweight TCN: https://doi.org/10.3390/electronics15050938
- Rows for FI-SEL, SACNN-IDS, SA-DCNN, and FedDynST are included from the comparison table reported in the Hybrid LLM/HGB IDS paper above when primary metrics were not directly accessible in full text.
