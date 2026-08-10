# Review card: benaddi2025arxiv

**Status:** PASS1B+PASS2_TEXT+PASS2_VISUAL_TITLE OK
**PDF text pages extracted:** 6
**Ground truth extract:** `_extract/benaddi2025arxiv.full.txt`
**Evidence JSON:** `_pass1b_evidence/benaddi2025arxiv.json`
**Generated:** 2026-08-06

## Identity (page-1 text)
- **Title:** Lightweight Intrusion Detection in IoT via SHAP-Guided Feature Pruning and Knowledge-Distilled Kronecker Networks
- **Tags:** KD, XAI, quant

## Abstract (extracted)
> The widespread deployment of Internet of Things (IoT) devices requires intrusion detection systems (IDS) with high accuracy while operating under strict resource constraints. Con- ventional deep learning IDS are often too large and computation- ally intensive for edge deployment. We propose a lightweight IDS that combines SHAP-guided feature pruning with knowledge- distilled Kronecker networks. A high-capacity teacher model identifies the most relevant features through SHAP explanations, and a compressed student leverages Kronecker-structured layers to minimize parameters while preserving discriminative inputs. Knowledge distillation transfers softened decision boundaries from teacher to student, improving generalization under com- pression. Experiments on the TON IOT dataset show that the student is nearly three orders of magnitude smaller than the teacher yet sustains macro-F1 above 0.986 with millisecond-level inference latency. The results demonstrate that explainability- driven pruning and structured compression can jointly enable scalable, low-latency, and energy-efficient IDS for heterogeneous IoT environments.

## Table headers present in PDF text (exact lines)
_None detected (image-only tables possible)._

## Metric-bearing lines (exact PDF lines; PASS2-verified)
- `teacher yet sustains macro-F1 above 0.986 with millisecond-level`
- `sumption without compromising accuracy [1].`
- `tectures capable of balancing efficiency and accuracy [2]. Ap-`
- `[4], leaving the interaction among latency, scalability, and`
- `efficiency, and accuracy [2].`
- `dimensional convolutional detector achieves near-perfect F1-`
- `orders of magnitude and still delivers accuracy above 99%`
- `The â€œSHAP Feature Ranking & Pruningâ€ module in Fig. 1`
- `corresponds to lines 3â€“4 in Algorithm 1. SHAP values provide`
- `preserved macro-F1 within 2% of the full model, making it`
- `linear head. This design reduces the parameter count to 1,282,`
- `The â€œKnowledge Distillationâ€ component in Fig. 1 is re-`
- `maximizing validation macro-F1.`
- `3: Compute SHAP attributions and rank features.`
- `4: Select smallest K preserving macro-F1.`
- `Validate macro-F1 and retain best (T, Î±).`
- `lation, the student is optionally quantized to INT8 precision,`
- `1,282 parameters, the student model is lightweight enough`
- `As shown in Fig. 4, SHAP highlights the traffic volume`
- `tical recall with no false negatives. Specificity reaches 0.9966`
- `and precision 0.9497 for fp32, with minor improvements for`
- `int8. Reported accuracy (â‰ˆ0.997) and macro-F1 (â‰ˆ0.987)`
- `COMPARISON OF TEACHER AND STUDENT MODELS IN TERMS OF ACCURACY, MACRO-F1, LATENCY, AND COMPRESSION.`
- `Macro-F1`
- `Fig. 4. Global ranking of discriminative features via mean absolute SHAP.`
- `nearly 1/250 while maintaining macro-F1 above 0.986. Both`
- `latency of 1.29 ms and p95 latency of 1.62 ms, enabling real-`
- `time inference. The int8 version slightly increases latency but`
- `Confusion matrices of the distilled student under fp32 and int8`
- `while mean latency falls from 1963 ms for the teacher to`

## CuKD freeze notes (non-numeric)
- KD neighborhood â†’ compare to C1/C2; do not claim novelty of KD-for-IDS alone.
- XAI neighborhood â†’ do not invent Spearman Ï; C6 is CuKD measurement.
- Quantization neighborhood â†’ Jacob/C4 PTQ honesty.
- **Numbers only from exact lines above.**

## Verification
- PASS1B: evidence JSON written with exact lines from extract
- PASS2: all `30` quoted lines re-found in full extract (by construction)
- PASS2_TEXT: OK; PASS2_VISUAL_TITLE: OK

## PASS2 independent text re-check (2026-08-06)
- **PASS2_OK** â€” 30/30 quotes exact-match in full extract

## PASS2 visual title check (2026-08-06)
- page_001.png exists: True
- title word hit ratio vs page-1 text: 1.000
- **PASS2_VISUAL_TITLE_OK** (title words supported by page-1 text + PNG on disk)

## DEEP_VISUAL (manual image pages 001,003,004,005,006)

- Table II: Teacher 769922 params macro-F1 0.9955; Student fp32 3042 params macro-F1 0.9863 mean lat 1.29 ms
- SHAP for pruning not Tree-Deep rank

