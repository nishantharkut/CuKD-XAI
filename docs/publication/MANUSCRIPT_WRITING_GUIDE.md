# CuKD-XAI Manuscript Writing Guide

This document extracts structural conventions from the selected Mamba/channel-prediction reference paper:

`docs/literature/reference_papers/lightweight_accurate_channel_prediction_using_mamba.pdf`

That paper is titled:

> Lightweight and Accurate Channel Prediction Using Mamba in High-Mobility Wireless Networks

The goal here is not to copy wording. The goal is to understand the style: how the paper frames a problem, introduces a method, claims contributions, organizes experiments, and presents results. This document maps that style onto CuKD-XAI.

## 1. Reference Paper Style

### 1.1 Title Style

His title is direct and benefit-led:

`Lightweight and Accurate Channel Prediction Using Mamba in High-Mobility Wireless Networks`

Pattern:

```text
[Benefit 1] and [Benefit 2] [Task] Using [Method] in [Target Domain]
```

For CuKD-XAI, the matching title style should emphasize:

- lightweight or compressed,
- accurate or resource-aware,
- explainable or explanation-faithful,
- WSN intrusion detection,
- knowledge distillation.

Possible CuKD-XAI titles in this style:

1. `Lightweight and Explainable Intrusion Detection Using Knowledge Distillation in Wireless Sensor Networks`
2. `Compressed and Deployable Knowledge Distillation for Explainable WSN Intrusion Detection`
3. `CuKD-XAI: Resource-Aware Knowledge Distillation and Explanation Auditing for WSN Intrusion Detection`
4. `Lightweight and Hardware-Aware WSN Intrusion Detection via Knowledge Distillation`

Best safe title for discussion:

> Compressed and Deployable Knowledge Distillation for Explainable WSN Intrusion Detection

This is strong but still honest. It does not claim best accuracy or physical TelosB deployment.

## 2. Abstract Pattern

The Mamba paper abstract follows this structure:

1. Broad importance of the domain.
2. Specific challenge.
3. Why existing methods are limited.
4. Proposed method.
5. Experimental setting.
6. Quantified results.
7. Final impact statement.

### 2.1 CuKD-XAI Abstract Skeleton

Use this skeleton when explaining how the paper should be written:

```text
Wireless sensor network intrusion detection requires accurate multiclass detection under severe memory and deployment constraints. Existing WSN-DS studies often report high accuracy using tree ensembles, optimized classifiers, or deep models, but these models are not always suitable for constrained edge or mote-class deployment and do not necessarily analyze whether compression preserves explanation behavior.

This work presents CuKD-XAI, a resource-aware knowledge-distillation framework for compressing high-performing WSN-DS intrusion detectors into compact neural students. A Random Forest teacher is distilled into small multilayer perceptrons with two capacity points: an ultra-small Student A and a stronger Student B. The framework evaluates scratch learning, curriculum variants, RF-based knowledge distillation, and co-distillation, then audits explanation transfer using SHAP teacher-student rank alignment.

Across 10 WSN-DS seeds, Student A RF-KD reaches 0.9869 accuracy and 0.9200 macro-F1 with only 4.64 KB FP32 storage, while Student B reaches up to 0.9891 accuracy and 0.9335 macro-F1 with 13.27 KB storage. Compared with the 85064.54 KB RF teacher, these students are about 18315x and 6411x smaller, respectively. Fixed-point C exports are further validated through MCU replay on ESP32-C3 and Arduino R4, and MSP430F1611 cross-compilation supports model-core memory feasibility.

The results show that useful WSN-DS detection performance can be retained under aggressive compression, while SHAP rank analysis indicates that predictive distillation does not automatically preserve teacher explanation structure.
```

### 2.2 Why This Matches His Style

| Mamba paper style | CuKD-XAI equivalent |
|---|---|
| "High-mobility environments need accurate CSI." | "WSN IDS needs accurate detection under constrained deployment." |
| "Existing CNN/RNN/attention methods have latency or robustness issues." | "Existing WSN-DS models are accurate but often large, ensemble-based, or not deployment-audited." |
| "We adapt Mamba for channel prediction." | "We use KD to compress RF behavior into KB-scale students." |
| "Experiments under 3GPP CDL-A." | "Experiments on WSN-DS, with Edge-IIoT as stress/generalization evidence." |
| "Up to 10 dB lower error, 15% faster convergence." | "18315x and 6411x compression, 1.0 MCU-vs-fixed agreement." |

## 3. Introduction Pattern

The Mamba paper introduction works like this:

1. Start with a broad research need.
2. Explain why the target environment is hard.
3. Review existing approaches.
4. Identify a gap.
5. Introduce the proposed method.
6. List contributions.

### 3.1 CuKD-XAI Introduction Flow

Use this sequence during manuscript planning and technical review:

```text
Paragraph 1: WSNs are used in critical environments, so intrusion detection must be accurate and reliable.

Paragraph 2: WSN devices are constrained. Accuracy alone is not enough because large models may not fit or may be impractical for deployment.

Paragraph 3: WSN-DS literature already has strong RF, LightGBM, CatBoost, and ensemble results. These papers are important but mostly emphasize raw detection performance.

Paragraph 4: The gap is compressed, deployment-oriented, explanation-audited WSN-DS IDS. We need to know whether a large accurate teacher can be distilled into a KB-scale student and whether that student preserves teacher reasoning.

Paragraph 5: CuKD-XAI compresses RF teacher behavior into small MLP students, evaluates multiple training routes, audits SHAP alignment, and validates fixed-point execution through HIL replay.
```

### 3.2 Strong Opening Paragraph in His Style

```text
Wireless sensor networks are increasingly used in distributed monitoring, industrial automation, and IoT-enabled sensing, where intrusion detection must operate under strict memory and computation constraints. Although recent WSN-DS studies report strong multiclass detection performance using optimized tree ensembles and deep learning models, the resulting models are often difficult to interpret, compress, or validate for constrained deployment. This motivates a resource-aware intrusion detection pipeline that treats model size, firmware feasibility, and explanation consistency as first-class evaluation targets rather than secondary implementation details.
```

## 4. Contribution Bullets in His Style

The Mamba paper uses a direct "main contributions are as follows" block. For CuKD-XAI, use this:

```text
The main contributions are as follows:

1. We propose CuKD-XAI, a resource-aware knowledge-distillation framework that compresses a high-performing WSN-DS Random Forest teacher into KB-scale neural students for multiclass WSN intrusion detection.

2. We conduct a 10-seed WSN-DS evaluation across scratch training, curriculum learning, RF-KD, MLP-KD, KD from curriculum teachers, and co-distillation, identifying Student A as an ultra-small deployment point and Student B as a stronger accuracy-compression point.

3. We quantify the compression trade-off: Student A RF-KD reaches 0.9869 accuracy and 0.9200 macro-F1 at 4.64 KB, while Student B reaches up to 0.9891 accuracy and 0.9335 macro-F1 at 13.27 KB, compared with an 85064.54 KB RF teacher.

4. We audit explanation transfer using SHAP rank alignment and show that predictive distillation does not necessarily preserve the teacher's feature-importance ordering.

5. We validate the deployment path through ONNX/OpenVINO runtime checks, fixed-point C export, MCU-class HIL replay on ESP32-C3 and Arduino R4, and MSP430F1611 target-toolchain memory-feasibility evidence.
```

If the contribution list is too long, combine items 2 and 3.

## 5. Methodology Structure

His Mamba paper uses equations, an algorithm block, an architecture figure, and parameter tables. CuKD-XAI should mirror this with the following structure.

### 5.1 Recommended Method Section

```text
II. Methodology

A. WSN-DS Preprocessing and Problem Formulation
   - 17-feature input vector.
   - 5-class multiclass IDS target.
   - Train-only preprocessing and scaling.

B. Teacher and Student Models
   - RF teacher.
   - Full MLP baseline.
   - Student A: 17-32-16-5.
   - Student B: 17-64-32-5.

C. Knowledge Distillation Objective
   - Hard-label classification loss.
   - Teacher-soft target loss.
   - Temperature/alpha if used in the final code path.

D. Curriculum and Co-Distillation Variants
   - Explain that these are tested training routes.
   - Do not overstate them as the main success.

E. Explanation-Faithfulness Audit
   - Student SHAP ranking.
   - Teacher SHAP ranking.
   - Spearman rank agreement.

F. Deployment-Oriented Export
   - ONNX/OpenVINO.
   - Fixed-point C export.
   - HIL serial replay protocol.
   - MSP430 cross-compile footprint.
```

### 5.2 Algorithm Block to Propose

This is the kind of algorithm block that matches the reference paper's style:

```text
Algorithm 1: CuKD-XAI Compression and Validation Pipeline

Require: WSN-DS feature matrix X, labels y, teacher model T, student architecture S
Ensure: trained student, compression metrics, SHAP alignment, deployment evidence

1. Split WSN-DS into train/validation/test partitions.
2. Fit preprocessing on training data and transform all partitions.
3. Train RF teacher T on the processed training set.
4. Train small student S using hard-label loss and teacher-guided KD loss.
5. Evaluate accuracy, macro-F1, weighted-F1, and per-class F1 on the test set.
6. Compute serialized size and compression ratio relative to the RF teacher.
7. Compute teacher and student SHAP rankings and Spearman rank agreement.
8. Export selected RF-KD students to ONNX and fixed-point C.
9. Replay held-out WSN-DS vectors through MCU firmware over serial.
10. Verify MCU predictions against fixed-point and FP32 references.
```

## 6. Results Section in His Style

The reference paper reports results with specific, quantified comparisons. For CuKD-XAI, the results should be organized around claims, not just many tables.

### 6.1 Result Claim 1: Teacher Accuracy vs Size

```text
The RF teacher achieves the strongest detection performance, reaching 0.9966 accuracy and 0.9789 macro-F1. However, its serialized size is approximately 85064.54 KB, motivating compression into compact neural students.
```

Source: `results/wsnds/final_results/2026-05-30-10seed-plus-j/wsnds_results_student_A.csv`.

### 6.2 Result Claim 2: Student A Ultra-Small Point

```text
Student A RF-KD reaches 0.9869 accuracy and 0.9200 macro-F1 with only 1189 parameters and 4.64 KB FP32 storage, corresponding to approximately 18315x size reduction relative to the RF teacher.
```

Source: `results/wsnds/final_results/2026-05-30-10seed-plus-j/wsnds_results_student_A.csv`.

### 6.3 Result Claim 3: Student B Accuracy-Compression Point

```text
Student B reaches up to 0.9891 accuracy and 0.9335 macro-F1 at 13.27 KB. This model is approximately 6411x smaller than the RF teacher while improving over the smaller Student A in macro-F1.
```

Source: `results/wsnds/final_results/2026-05-30-10seed-plus-j/wsnds_results_student_B.csv`.

### 6.4 Result Claim 4: Explanation-Faithfulness Gap

```text
The teacher-student SHAP rank Spearman correlation is near zero, indicating that predictive distillation does not automatically preserve the teacher's global feature-importance ordering.
```

Source: `docs/research/RELATED_WORK_RESULTS_COMPARISON.md`, `results/wsnds/legacy_runs/2026-05-30-10seed/cukd_xai_results.json`.

### 6.5 Result Claim 5: Hardware-Path Validation

```text
In HIL replay, Student A and Student B RF-KD fixed-point firmware matched the generated fixed-point reference with 1.0 agreement across all 56200 test vectors on both ESP32-C3 and Arduino R4.
```

Source: `results/hardware_hil/reports/final_postprocessing/hil_fidelity.csv`.

## 7. Figure and Table Plan

The reference paper uses a figure/table-heavy IEEE structure. CuKD-XAI should have a similar visual plan.

### 7.1 Must-Have Figures

| Figure | Purpose | Existing or needed |
|---|---|---|
| Overall CuKD-XAI pipeline | Shows data -> teachers -> students -> SHAP -> deployment | Mermaid diagram in `docs/research/PROJECT_TECHNICAL_BRIEF.md` |
| Model compression structure | Shows RF teacher to Student A/B to fixed-point firmware | Mermaid diagram in the technical brief |
| Pareto frontier | Shows accuracy/macro-F1 vs size | Existing: `results/wsnds/final_results/2026-05-30-10seed-plus-j/pareto_frontier_with_J.png` |
| Per-class F1 | Shows minority-class behavior | Existing: `results/wsnds/final_results/2026-05-30-10seed-plus-j/per_class_f1_student_A_with_J.png`, `per_class_f1_student_B_with_J.png` |
| SHAP summary or rank comparison | Shows explanation audit | Existing: `results/wsnds/final_results/2026-05-30-10seed-plus-j/shap_summary_student.png`; rank audit in JSON/docs |
| Hardware HIL flow | Shows host serial replay and MCU verification | Mermaid diagram in the technical brief |

### 7.2 Must-Have Tables

| Table | Purpose | Source |
|---|---|---|
| WSN-DS 10-seed Student A results | Ultra-small comparison | `docs/research/RESULTS_AND_EVIDENCE.md` |
| WSN-DS 10-seed Student B results | Accuracy-compression comparison | `docs/research/RESULTS_AND_EVIDENCE.md` |
| Compression table | RF vs students vs fixed-point | `docs/research/RESULTS_AND_EVIDENCE.md` |
| Runtime/deployment table | ONNX/OpenVINO evidence | `results/runtime/onnx_openvino/wsnds/runtime_from_existing_outputs/wsnds_existing_artifact_runtime_summary.csv` |
| HIL table | ESP32-C3 and Arduino R4 validation | `results/hardware_hil/reports/final_postprocessing/hil_fidelity.csv` |
| Related work table | Honest comparison with WSN-DS and Edge-IIoT papers | `docs/research/RELATED_WORK_RESULTS_COMPARISON.md`, `docs/literature/comparison_tables/EDGEIIOT_LITERATURE_COMPARISON.md` |

## 8. How to Talk to Him in His Writing Language

If he asks "What is the paper angle?", say:

```text
Sir, I think the paper should be framed like a resource-aware compression paper, not like an accuracy-SOTA paper. The gap is that WSN-DS already has strong accuracy results, but compact deployable and explanation-audited students are not the main focus of those works. Our strongest angle is that we compress an 85 MB RF teacher into 4.64 KB and 13.27 KB students, validate fixed-point MCU replay, and show that SHAP explanation ranking is not automatically preserved after KD.
```

If he asks "What is the main result?", say:

```text
The main result is the size-performance trade-off. Student A RF-KD gives about 98.69% accuracy and 92.00% macro-F1 at 4.64 KB, around 18315x smaller than the RF teacher. Student B reaches about 98.91% accuracy and 93.35% macro-F1 at 13.27 KB, around 6411x smaller. So we lose some macro-F1 compared with RF, but gain massive compression and firmware validation.
```

If he asks "How is it novel?", say:

```text
The novelty is not that KD or SHAP individually are new. The novelty is the combination and evidence chain for WSN-DS: KB-scale RF-to-student compression, multi-seed evaluation, SHAP teacher-student explanation-faithfulness audit, ONNX/OpenVINO export, fixed-point C export, MCU HIL replay, and MSP430 memory-feasibility evidence.
```

If he asks "What should be the contribution bullets?", use the contribution block in Section 4.

## 9. Paper Section Outline

This section order matches his IEEE-style writing:

```text
I. Introduction
   - WSN IDS importance.
   - Resource constraints.
   - Existing WSN-DS accuracy papers.
   - Gap: compression, deployability, explanation-faithfulness.
   - Contributions.

II. Related Work
   - WSN-DS IDS.
   - Lightweight/edge IDS.
   - KD in IDS.
   - XAI and SHAP in IDS.
   - Hardware-aware IDS.

III. CuKD-XAI Methodology
   - Dataset and preprocessing.
   - Teacher and student models.
   - KD and co-distillation.
   - SHAP alignment audit.
   - Deployment export and HIL protocol.

IV. Experimental Setup
   - Datasets.
   - Seeds.
   - Metrics.
   - Hardware/software environment.
   - Baselines.

V. Results and Discussion
   - WSN-DS performance.
   - Compression.
   - Explanation alignment.
   - Deployment/HIL.
   - Edge-IIoT generalization.
   - Related-work comparison.

VI. Limitations
   - Not accuracy SOTA.
   - No physical TelosB energy/latency yet.
   - No live packet feature extraction.
   - INT8 speedup not supported.

VII. Conclusion
   - Compressed students preserve useful detection performance.
   - Deployment path is validated up to fixed-point MCU replay and MSP430 memory feasibility.
   - Explanation transfer remains an open issue.
```

## 10. Wording That Fits His Style

Use these phrases:

- "This work addresses the deployment gap by..."
- "The main contributions are as follows..."
- "To evaluate robustness, we further..."
- "The results demonstrate that..."
- "These findings highlight..."
- "Compared with the large teacher..."
- "The proposed framework achieves..."
- "The analysis reveals that..."

Avoid these phrases:

- "This is the first..."
- "This is perfect..."
- "This beats all papers..."
- "Hardware deployment is done..."
- "INT8 is faster..."
- "SHAP proves the model is explainable..."

Better safe versions:

| Unsafe | Safe |
|---|---|
| "We beat WSN-DS SOTA." | "We target compression rather than pure accuracy SOTA." |
| "We deployed on TelosB." | "We provide MSP430F1611 target-toolchain memory-feasibility evidence." |
| "INT8 improves speed." | "Dynamic INT8 reduced artifact size but did not support a speedup claim." |
| "SHAP proves explainability." | "SHAP rank analysis audits explanation transfer after compression." |
| "Co-distillation is always best." | "Co-distillation is useful, but RF-KD is simpler and often nearly as strong." |

## 11. Meeting Checklist

Before a technical review, open these files:

1. `docs/research/PROJECT_TECHNICAL_BRIEF.md`
2. `docs/research/RESULTS_AND_EVIDENCE.md`
3. `docs/publication/MANUSCRIPT_WRITING_GUIDE.md`
4. `docs/research/RELATED_WORK_RESULTS_COMPARISON.md`
5. `results/hardware_hil/reports/final_postprocessing/final_postprocessing_analysis.md`
6. `deployment/msp430/MSP430_CROSS_COMPILE_REPORT.md`
7. `docs/literature/comparison_tables/EDGEIIOT_LITERATURE_COMPARISON.md`

Start with this:

```text
Sir, I studied the way your Mamba paper is framed. It starts with a domain need, then the technical gap, then a compact method, then quantified results. I think CuKD-XAI should be framed in the same style: not as another WSN-DS accuracy paper, but as a compressed, deployable, explanation-audited WSN IDS paper.
```

Then continue with:

```text
The exact gap is that WSN-DS already has high-accuracy papers, but there is still room for a resource-aware study that compresses an accurate teacher into KB-scale students, validates fixed-point execution, and checks whether explanation behavior transfers after distillation.
```



