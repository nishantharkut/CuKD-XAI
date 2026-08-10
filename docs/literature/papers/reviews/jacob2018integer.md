# Review card: jacob2018integer

**Status:** PASS1B+PASS2_TEXT+PASS2_VISUAL_TITLE OK
**PDF text pages extracted:** 10
**Ground truth extract:** `_extract/jacob2018integer.full.txt`
**Evidence JSON:** `_pass1b_evidence/jacob2018integer.json`
**Generated:** 2026-08-06

## Identity (page-1 text)
- **Title:** Quantization and Training of Neural Networks for Efﬁcient Integer-Arithmetic-Only Inference
- **Tags:** quant

## Abstract (extracted)
> The rising popularity of intelligent mobile devices and the daunting computational cost of deep learning-based models call for efﬁcient and accurate on-device inference schemes. We propose a quantization scheme that allows inference to be carried out using integer-only arithmetic, which can be implemented more efﬁciently than ﬂoating point inference on commonly available integer-only hard- ware. We also co-design a training procedure to preserve end-to-end model accuracy post quantization. As a result, the proposed quantization scheme improves the tradeoff be- tween accuracy and on-device latency. The improvements are signiﬁcant even on MobileNets, a model family known for run-time efﬁciency, and are demonstrated in ImageNet classiﬁcation and COCO detection on popular CPUs.

## Table headers present in PDF text (exact lines)
- `Table 4.1: ResNet on ImageNet: Floating-point vs quantized net-`
- `Table 4.2: ResNet on ImageNet: Accuracy under various quan-`
- `table 4.1. Accuracies of integer-only quantized networks`
- `Table 4.3 shows that 7-bit quantized training produces`
- `Table 4.3: Inception v3 on ImageNet: Accuracy and recall 5 com-`
- `Table 4.4 shows the latency-vs-accuracy tradeoff be-`
- `Table 4.4: Object detection speed and accuracy on COCO dataset`
- `Table 4.5: Face detection accuracy of ﬂoating point and integer-`
- `Table 4.6: Face attributes: relative average category precision of`

## Metric-bearing lines (exact PDF lines; PASS2-verified)
- `[29], are all over-parameterized by design in order to extract`
- `Top 1 Accuracy`
- `the fast integer-arithmetic circuits in common CPUs to deliver an improved latency-vs-accuracy tradeoff (section 4). The ﬁgure compares`
- `and just a few parameters (bias vectors) as 32-bit integers.`
- `relative accuracy. Multiplication by M0 can thus be imple-`
- `to preserve good end-to-end neural network accuracy6.`
- `smoothing parameter being close to 1 so that observed`
- `a much higher range and precision compared to the 8 bit`
- `Here γ is the batch normalization’s scale parameter, σ2`
- `Table 4.2: ResNet on ImageNet: Accuracy under various quan-`
- `ing, and quantized models with ReLU6 have less accuracy`
- `recall 5`
- `Table 4.3: Inception v3 on ImageNet: Accuracy and recall 5 com-`
- `Top 1 Accuracy`
- `Figure 4.1: Latency-vs-accuracy tradeoff of ﬂoat vs. integer-only`
- `Top 1 Accuracy`
- `Figure 4.2: Latency-vs-accuracy tradeoff of ﬂoat vs. integer-only`
- `time budget. The accuracy gap is quite substantial (∼10%)`
- `for Snapdragon 835 LITTLE cores at the 33ms latency`
- `Table 4.4 shows the latency-vs-accuracy tradeoff be-`
- `minimal loss in accuracy (−1.8% relative).`
- `Table 4.4: Object detection speed and accuracy on COCO dataset`
- `Table 4.5: Face detection accuracy of ﬂoating point and integer-`
- `correct detection, for x in {0.5, 0.55, . . . , 0.95}. Latency (ms) of`
- `pendix D.1, quantization provides close to a 2× latency`
- `the latency of quantized models. Table Appendix D.1 shows`
- `Figure 4.3 shows the latency-vs-`
- `Table 4.6: Face attributes: relative average category precision of`
- `Figure 4.3: Latency-vs-accuracy tradeoff of ﬂoat vs. integer-only`
- `with 50x fewer parameters and¡ 1mb model size.`

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
