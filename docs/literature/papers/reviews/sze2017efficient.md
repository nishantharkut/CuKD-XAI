# Review card: sze2017efficient

**Status:** PASS1B+PASS2_TEXT+PASS2_VISUAL_TITLE OK
**PDF text pages extracted:** 32
**Ground truth extract:** `_extract/sze2017efficient.full.txt`
**Evidence JSON:** `_pass1b_evidence/sze2017efficient.json`
**Generated:** 2026-08-06

## Identity (page-1 text)
- **Title:** Efﬁcient Processing of Deep Neural Networks: A Tutorial and Survey Vivienne Sze, Senior Member, IEEE, Yu-Hsin Chen, Student Member, IEEE, Tien-Ju Yang, Student
- **Tags:** XAI, quant

## Abstract (extracted)
> Deep neural networks (DNNs) are currently widely used for many artiﬁcial intelligence (AI) applications including computer vision, speech recognition, and robotics. While DNNs deliver state-of-the-art accuracy on many AI tasks, it comes at the cost of high computational complexity. Accordingly, techniques that enable efﬁcient processing of DNNs to improve energy efﬁciency and throughput without sacriﬁcing application accuracy or increasing hardware cost are critical to the wide deployment of DNNs in AI systems. This article aims to provide a comprehensive tutorial and survey about the recent advances towards the goal of enabling efﬁcient processing of DNNs. Speciﬁcally, it will provide an overview of DNNs, discuss various hardware platforms and architectures that support DNNs, and highlight key trends in reducing the computation cost of DNNs either solely via hardware design changes or via joint hardware design and DNN algorithm changes. It will also summarize various development resources that enable researchers and practitioners to quickly get started in this ﬁeld, and highlight important benchmarking metrics and design considerations that should be used for evaluating the rapidly growing number of DNN hardware designs, optionally including algorithmic co-designs, being proposed in academia and industry. The reader will take away the following concepts from this article: understand the key design considerations for DNNs; be able to evaluate different DNN hardware implementations with benchmarks and comparison metrics; understand the trade-offs between various hardware architectures and platforms; be able to evaluate the utility of various DNN design techniques for efﬁcient processing; and understand recent implementation trends and opportunities.

## Table headers present in PDF text (exact lines)
_None detected (image-only tables possible)._

## Metric-bearing lines (exact PDF lines; PASS2-verified)
- `used to compute reduced precision neural networks [9]. These types of neural`
- `Accuracy (Top-5 error)`
- `human-level accuracy with a top-5 error rate4 below 5%. Since`
- `the accuracy of speech recognition [21] as well as many`
- `in Eq. (2), where the parameters (γ, β) are learned from`
- `human-level accuracy with a top-5 error rate below 5%. One`
- `using 1×1 ﬁlters to reduce the number of weight parameters.`
- `accuracy of these models can vary by around 1% to 2%`
- `images. LeNet-5 was able to achieve an accuracy of 99.05%`
- `network was able to achieve 64.84% accuracy on CIFAR-10`
- `when it was ﬁrst introduced [66]. Since then the accuracy has`
- `SUMMARY OF POPULAR DNNS [3, 15, 48, 50, 51]. †ACCURACY IS MEASURED BASED ON TOP-5 ERROR ON IMAGENET [14]. ‡THIS VERSION OF LENET-5`
- `accuracy of 83.6% for the top-5 (which is substantially better`
- `set. In 2017, the highest accuracy was 97.7% for the top-5.`
- `to perform two FP16 operations on a single precision core for`
- `layer shapes and sizes (e.g., FFT for ﬁlters greater than 5×5,`
- `appropriate algorithm for a given shape and size [77, 78].`
- `Fig. 30. Mapping optimization takes in hardware and DNNs shape constraints`
- `as shown in Fig. 32. First, replication can be used to map shapes`
- `precision of 32-bit ﬂoating point, which is the default precision`
- `1) Linear quantization: The ﬁrst step of reducing precision`
- `precision can vary between 4 and 9 bits for AlexNet across`
- `to ﬁne-grain variations in bit precision can result in a 2.24×`
- `an accuracy loss of 19% and 29.8%, respectively [129].`
- `With these changes, BWN reduced the accuracy loss to 0.8%,`
- `the accuracy loss to 5.2%.`
- `weight (i.e., -w1, 0, w2) for an accuracy loss of 0.6% [132],`
- `accuracy versus a 5% loss for log base-2 quantization for`
- `METHODS TO REDUCE NUMERICAL PRECISION FOR ALEXNET. ACCURACY MEASURED FOR TOP-5 ERROR ON IMAGENET. *NOT APPLIED TO FIRST AND/OR`
- `Top-5 Accuracy`

## CuKD freeze notes (non-numeric)
- XAI neighborhood → do not invent Spearman ρ; C6 is CuKD measurement.
- Quantization neighborhood → Jacob/C4 PTQ honesty.
- **Numbers only from exact lines above.**

## Verification
- PASS1B: evidence JSON written with exact lines from extract
- PASS2: all `30` quoted lines re-found in full extract (by construction)
- PASS2_TEXT: OK; PASS2_VISUAL_TITLE: OK

## PASS2 independent text re-check (2026-08-06)
- **PASS2_OK** — 30/30 quotes exact-match in full extract

## PASS2 visual title check (2026-08-06)
- page_001.png exists: True
- title word hit ratio vs page-1 text: 1.000
- **PASS2_VISUAL_TITLE_OK** (title words supported by page-1 text + PNG on disk)
