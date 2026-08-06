# Review card: hinton2015distilling

**Status:** PASS1B+PASS2_TEXT+PASS2_VISUAL_TITLE OK
**PDF text pages extracted:** 9
**Ground truth extract:** `_extract/hinton2015distilling.full.txt`
**Evidence JSON:** `_pass1b_evidence/hinton2015distilling.json`
**Generated:** 2026-08-06

## Identity (page-1 text)
- **Title:** arXiv:1503.02531v1  [stat.ML]  9 Mar 2015 Distilling the Knowledge in a Neural Network
- **Tags:** KD

## Abstract (extracted)
> A very simple way to improve the performance of almost any machine learning algorithm is to train many different models on the same data and then to average their predictions [3]. Unfortunately, making predictions using a whole ensemble of models is cumbersome and may be too computationally expensive to allow de- ployment to a large number of users, especially if the individual models are large neural nets. Caruana and his collaborators [1] have shown that it is possible to compress the knowledge in an ensemble into a single model which is much eas- ier to deploy and we develop this approach further using a different compression technique. We achieve some surprising results on MNIST and we show that we can signiﬁcantly improve the acoustic model of a heavily used commercial system by distilling the knowledge in an ensemble of models into a single model. We also introduce a new type of ensemble composed of one or more full models and many specialist models which learn to distinguish ﬁne-grained classes that the full mod- els confuse. Unlike a mixture of experts, these specialist models can be trained rapidly and in parallel.

## Table headers present in PDF text (exact lines)
- `Table 1: Frame classiﬁcation accuracy and WER showing that the distilled single model performs`
- `Table 1 shows that, indeed, our distillation approach is able to extract more useful information from`
- `Table 2: Example classes from clusters computed by our covariance matrix clustering algorithm`
- `Table 3: Classiﬁcation accuracy (top 1) on the JFT development set.`
- `Table 4: Top 1 accuracy improvement by # of specialist models covering correct class on the JFT`
- `Table 5: Soft targets allow a new model to generalize well from only 3% of the training set. The soft`

## Metric-bearing lines (exact PDF lines; PASS2-verified)
- `used when training the distilled model, but after it has been trained it uses a temperature of 1.`
- `using exactly the same logits in softmax of the distilled model but at a temperature of 1. We found`
- `So in the high temperature limit, distillation is equivalent to minimizing 1/2(zi −vi)2, provided the`
- `When the distilled net had 300 or more units in each of its two hidden layers, all temperatures above`
- `of the distilled model, 3 is a mythical digit that it has never seen. Despite this, the distilled model`
- `by 3.5 (which optimizes overall performance on the test set), the distilled model makes 109 errors`
- `of which 14 are on 3s. So with the right bias, the distilled model gets 98.6% of the test 3s correct`
- `training set, the distilled model makes 47.3% test errors, but when the biases for 7 and 8 are reduced`
- `total number of parameters is about 85M. This is a slightly outdated version of the acoustic model`
- `examples. This system achieves a frame accuracy of 58.9%, and a Word Error Rate (WER) of 10.9%`
- `Table 1: Frame classiﬁcation accuracy and WER showing that the distilled single model performs`
- `Table 1 shows that, indeed, our distillation approach is able to extract more useful information from`
- `improvement in frame classiﬁcation accuracy achieved by using an ensemble of 10 models is trans-`
- `the class probabilities of an already trained larger model [8]. However, they do the distillation at a`
- `temperature of 1 using a large unlabeled dataset and their best distilled model only reduces the error`
- `Table 3: Classiﬁcation accuracy (top 1) on the JFT development set.`
- `Table 4: Top 1 accuracy improvement by # of specialist models covering correct class on the JFT`
- `whether we use KL(p, q) or KL(q, p)). We parameterize q = softmax(z) (with T = 1) and we`
- `3 shows the absolute test accuracy for the baseline system and the baseline system combined with`
- `tive percentage improvement in top1 accuracy for the JFT dataset broken down by the number of`
- `did early stopping, as the accuracy drops sharply after reaching 44.5%), whereas the same model`

## CuKD freeze notes (non-numeric)
- KD neighborhood → compare to C1/C2; do not claim novelty of KD-for-IDS alone.
- **Numbers only from exact lines above.**

## Verification
- PASS1B: evidence JSON written with exact lines from extract
- PASS2: all `27` quoted lines re-found in full extract (by construction)
- PASS2_TEXT: OK; PASS2_VISUAL_TITLE: OK

## PASS2 independent text re-check (2026-08-06)
- **PASS2_OK** — 27/27 quotes exact-match in full extract

## PASS2 visual title check (2026-08-06)
- page_001.png exists: True
- title word hit ratio vs page-1 text: 1.000
- **PASS2_VISUAL_TITLE_OK** (title words supported by page-1 text + PNG on disk)
