# Review card: guo2017calibration

**Status:** PASS1B+PASS2_TEXT+PASS2_VISUAL_TITLE OK
**PDF text pages extracted:** 14
**Ground truth extract:** `_extract/guo2017calibration.full.txt`
**Evidence JSON:** `_pass1b_evidence/guo2017calibration.json`
**Generated:** 2026-08-06

## Identity (page-1 text)
- **Title:** On Calibration of Modern Neural Networks Chuan Guo * 1 Geoff Pleiss * 1 Yu Sun * 1 Kilian Q. Weinberger 1
- **Tags:** KD

## Abstract (extracted)
> Conﬁdence calibration – the problem of predict- ing probability estimates representative of the true correctness likelihood – is important for classiﬁcation models in many applications. We discover that modern neural networks, unlike those from a decade ago, are poorly calibrated. Through extensive experiments, we observe that depth, width, weight decay, and Batch Normal- ization are important factors inﬂuencing calibra- tion. We evaluate the performance of various post-processing calibration methods on state-of- the-art architectures with image and document classiﬁcation datasets. Our analysis and exper- iments not only offer insights into neural net- work learning, but also provide a simple and straightforward recipe for practical settings: on most datasets, temperature scaling – a single- parameter variant of Platt Scaling – is surpris- ingly effective at calibrating predictions.

## Table headers present in PDF text (exact lines)
- `Table 1. ECE (%) (with M = 15 bins) on standard vision and NLP datasets before calibration and with various calibration methods.`
- `Table 1 displays model calibration,`

## Metric-bearing lines (exact PDF lines; PASS2-verified)
- `beneﬁcial to classiﬁcation accuracy. On CIFAR-100, test`
- `parameters of a binning scheme are θ1, . . . , θM. Under this`
- `The parameters θ1, . . . , θM can be viewed as parameters of`
- `2005), Platt scaling learns scalar parameters a, b ∈R and`
- `scaling, uses a single scalar parameter T > 0 for all classes.`
- `knowledge distillation (Hinton et al., 2015) and statistical`
- `accuracy (see Section S3). Histogram binning, the simplest`
- `whose parameter is 1/T. We set T =1 during training, and`

## CuKD freeze notes (non-numeric)
- KD neighborhood → compare to C1/C2; do not claim novelty of KD-for-IDS alone.
- **Numbers only from exact lines above.**

## Verification
- PASS1B: evidence JSON written with exact lines from extract
- PASS2: all `10` quoted lines re-found in full extract (by construction)
- PASS2_TEXT: OK; PASS2_VISUAL_TITLE: OK

## PASS2 independent text re-check (2026-08-06)
- **PASS2_OK** — 10/10 quotes exact-match in full extract

## PASS2 visual title check (2026-08-06)
- page_001.png exists: True
- title word hit ratio vs page-1 text: 1.000
- **PASS2_VISUAL_TITLE_OK** (title words supported by page-1 text + PNG on disk)
