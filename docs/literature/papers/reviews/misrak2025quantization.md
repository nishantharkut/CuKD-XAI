# Review card: misrak2025quantization

**Status:** PASS1B+PASS2_TEXT+PASS2_VISUAL_TITLE OK
**PDF text pages extracted:** 34
**Ground truth extract:** `_extract/misrak2025quantization.full.txt`
**Evidence JSON:** `_pass1b_evidence/misrak2025quantization.json`
**Generated:** 2026-08-06

## Identity (page-1 text)
- **Title:** © The Author(s) 2025. Open Access  This article is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International
- **Tags:** quant

## Abstract (extracted)
_Not auto-detected; open full extract._

## Table headers present in PDF text (exact lines)
- `Table 1  Summary of research publications related to Lightweight Intrusion Detection Systems`
- `Table 2  CIC-IDS2017 dataset attack distribution`
- `Table 3  CIC-IoT2023 dataset attack distribution`
- `Table 4  Comparison of several models’ multiclassification accuracy on CIC-IDS2017 dataset`
- `Table 5. The experimental findings substantiate that the proposed DNN-BiLSTMQ`
- `Table 5  Comparison of several models’ multiclassification accuracy on CIC-IoT2023 dataset`
- `Table 6  Comparison of Model Lightweightness on Different Datasets`
- `Table 7  Comparison of models’ multiclass classification effectiveness`
- `Table 8  Comparison of models’ lightweightness`

## Metric-bearing lines (exact PDF lines; PASS2-verified)
- `IDS2017 dataset, a detection accuracy of 99.73% is achieved with a model size of`
- `just 25.6 KB, while on the CIC-IoT2023 dataset, the achieved a detection accuracy`
- `Achieved up to 99.73% accuracy with model sizes under 32 KB, which could be`
- `accuracy rates of 97.75% in detecting malicious cyber attacks on the CSE-CIC-IDS2018`
- `cation, they get 98.12%, and for multiclassification of 77.0% accuracy.`
- `show almost 93% detection accuracy and 92% f1-score [24].`
- `dataset. Through their research, [28] showcased the accuracy, sensitivity, and efficacy of`
- `95% accuracy with a reasonable detection time, utilizing relatively small training data`
- `parameter optimization, as detailed in Sect. 3.7, to achive the best accuracy and while`
- `model parameters from 32-bit floating-point to achieve better resolution and efficiency.`
- `Train the DNN-BiLSTM model using full-precision (FP32) weights.`
- `Apply quantization-aware training (QAT) with low-precision (e.g., INT8) effects`
- `sification accuracy. Optuna, introduced by [45], represents a contemporary approach to`
- `with quantization-aware training. Algorithm  2 outlines the Optuna hyperparameter`
- `Algorithm 2  Optuna Hyperparameter Optimization with RLA-MIFS + two stage IPCA Pseudocode`
- `The evaluation framework includes metrics like precision, recall, and F1-score, assess­`
- `mizes false alarms, while recall identifies genuine attacks. The F1-score, harmonizing`
- `F1 =2 × Precision × Recall`
- `achieves an accuracy of 99.753%, outperforming single stage RAL-MIFS + IPCA tech­`
- `precision, accuracy, recall, and F1-score. A comprehensive performance comparison`
- `F1-score, time of training and time of inference. The results of these comparative experi­`
- `batch_size: 148, lr: 0.0018895 weight_decay: 1.44357 × 10−5. These hyperparameters`
- `tently produced the highest F1-score and lowest validation loss across multiple trials.`
- `For accuracy, the DNN-BiLSTMQ model achieves 99.73%, which is 0.06%, 0.22%,`
- `For precision, the DNN-BiLSTMQ model obtains 99.57%, which is 0.04%, 0.10%,`
- `For recall, the DNN-BiLSTMQ model achieves 99.73%, which is 0.06%, 0.22%, and`
- `For the F1-score, the DNN-BiLSTMQ model obtains 99.64%, which is 0.05%, 0.16%,`
- `strating unparalleled performance across all evaluation metrics, with accuracy, F1-score,`
- `recall, and precision surpassing the 99.57% threshold. This exceptional performance is`
- `Table 4  Comparison of several models’ multiclassification accuracy on CIC-IDS2017 dataset`

## CuKD freeze notes (non-numeric)
- Quantization neighborhood → Jacob/C4 PTQ honesty.
- **Numbers only from exact lines above.**

## Verification
- PASS1B: evidence JSON written with exact lines from extract
- PASS2: all `39` quoted lines re-found in full extract (by construction)
- PASS2_TEXT: OK; PASS2_VISUAL_TITLE: OK

## PASS2 independent text re-check (2026-08-06)
- **PASS2_OK** — 39/39 quotes exact-match in full extract

## PASS2 visual title check (2026-08-06)
- page_001.png exists: True
- title word hit ratio vs page-1 text: 1.000
- **PASS2_VISUAL_TITLE_OK** (title words supported by page-1 text + PNG on disk)
