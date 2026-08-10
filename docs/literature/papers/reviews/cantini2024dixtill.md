# Review card: cantini2024dixtill

**Status:** PASS1B+PASS2_TEXT+PASS2_VISUAL_TITLE OK
**PDF text pages extracted:** 17
**Ground truth extract:** `_extract/cantini2024dixtill.full.txt`
**Evidence JSON:** `_pass1b_evidence/cantini2024dixtill.json`
**Generated:** 2026-08-06

## Identity (page-1 text)
- **Title:** © The Author(s) 2024, corrected publication 2024. Open Access  This article is licensed under a Creative Commons Attribution 4.0 Interna-
- **Tags:** KD, XAI, quant

## Abstract (extracted)
> Large Language Models (LLMs) are characterized by their inherent memory inef- ficiency and compute-intensive nature, making them impractical to run on low- resource devices and hindering their applicability in edge AI contexts. To address this issue, Knowledge Distillation approaches have been adopted to transfer knowledge from a complex model, referred to as the teacher, to a more compact, computationally efficient one, known as the student. The aim is to retain the performance of the original model while substantially reducing computational requirements. However, traditional knowledge distillation methods may struggle to effectively transfer crucial explainable knowledge from an LLM teacher to the student, potentially leading to explanation inconsistencies and decreased performance. This paper presents DiXtill, a method based on a novel approach to distilling knowledge from LLMs into lightweight neural architectures. The main idea is to leverage local explanations provided by an eXplain- able Artificial Intelligence (XAI) method to guide the cross-architecture distillation of a teacher LLM into a self-explainable student, specifically a bi-directional LSTM network.Experimental results show that our XAI-driven distillation method allows the teacher explanations to be effectively transferred to the student, resulting in bet- ter agreement compared to classical distillation methods,thus enhancing the student interpretability. Furthermore, it enables the student to achieve comparable perfor- mance to the teacher LLM while also delivering a significantly higher compression ratio and speedup compared to other techniques such as post-training quantiza- tion and pruning, which paves the way for more efficient and sustainable edge AI applications

## Table headers present in PDF text (exact lines)
- `Table 1  Classification performance comparison of different knowledge distillation methods`
- `Table 2  Performance achieved by DiXtill at different temperature values`
- `Table 3  Comparison of compression ratio and speedup between DiXtill, PTQ, and AHP models,`
- `Table 4  Comparison of classification performance between DiXtill, PTQ, and AHP models`

## Metric-bearing lines (exact PDF lines; PASS2-verified)
- `ers) encompasses 110 million parameters in its base version, while the GPT-3 contains`
- `175 billion parameters, requiring at least 320 gigabytes of storage in half-precision (i.e.,`
- `edge Distillation (KD) [7, 8]. In particular, knowledge distillation has been leveraged to`
- `the literature [16]. Knowledge distillation has been explored to reduce the size of LLMs`
- `tilling BERT base with 40% fewer parameters than the original BERT. Other attempts`
- `tectures, using a cross-architecture knowledge distillation process  [18]. For example,`
- `Tang et al. [15] proposed a method to distill knowledge from BERT into a single-layer`
- `ing the precision of model parameters (e.g., weights and activations from 32-bit floating-`
- `(SHapley Additive exPlanation) [30]. The method computes the contribution of each fea-`
- `Bahdanau-like attention mechanism [37], implemented by a parameterized feed-for-`
- `Fig. 1  Distillation process in DiXtill. The Explainer, the Teacher LLM, and the Student are indicated in red, green,`
- `cross-entropy loss term LCE . Finally, the temperature value τ was set to 5 for all distilla-`
- `knowledge distillation [8] and matching logits with MSE [14, 15], respectively; (iii)`
- `For each model, we evaluated the accuracy, macro F1 score, Matthews correlation,`
- `bound for performance, reaching an accuracy of 85.5% and a macro F1 score of 81%.`
- `is a noticeable decrease in accuracy to 82.7% and in macro F1 score to 76%. Simi-`
- `further reduces the accuracy to 81.6% and macro F1 to 75.2%. As a baseline, train-`
- `racy of 80.2%. These results highlight the challenges of distilling an LLM into a small`
- `the performance achieved by using DiXtill, which shows an accuracy of 84.3% and a`
- `macro F1 score of 78.9%, indicates that incorporating local explanations during distil-`
- `remarkable reduction of the number of parameters, decreasing from 0.11 billion to`
- `Table 1  Classification performance comparison of different knowledge distillation methods`
- `Macro F1`
- `Fig. 2  Classification performance comparison with other knowledge distillation methods. Dotted lines`
- `Macro F1`
- `The dynamically int8 quantized model maintains a high accuracy of 85.2% , com-`
- `Macro F1`
- `depicts the evolution of the macro F1 score as the number of pruned heads increases`
- `reaching a macro F1 score of 0.4 when employing only 1 head. In line with the results`
- `distilled students S (i.e., ES(x) ). We used the following metrics [40]:`

## CuKD freeze notes (non-numeric)
- KD neighborhood → compare to C1/C2; do not claim novelty of KD-for-IDS alone.
- XAI neighborhood → do not invent Spearman ρ; C6 is CuKD measurement.
- Quantization neighborhood → Jacob/C4 PTQ honesty.
- **Numbers only from exact lines above.**

## Verification
- PASS1B: evidence JSON written with exact lines from extract
- PASS2: all `34` quoted lines re-found in full extract (by construction)
- PASS2_TEXT: OK; PASS2_VISUAL_TITLE: OK

## PASS2 independent text re-check (2026-08-06)
- **PASS2_OK** — 34/34 quotes exact-match in full extract

## PASS2 visual title check (2026-08-06)
- page_001.png exists: True
- title word hit ratio vs page-1 text: 1.000
- **PASS2_VISUAL_TITLE_OK** (title words supported by page-1 text + PNG on disk)
