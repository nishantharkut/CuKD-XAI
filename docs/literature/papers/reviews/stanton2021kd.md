# Review card: stanton2021kd

**Status:** PASS1B+PASS2_TEXT+PASS2_VISUAL_TITLE OK
**PDF text pages extracted:** 21
**Ground truth extract:** `_extract/stanton2021kd.full.txt`
**Evidence JSON:** `_pass1b_evidence/stanton2021kd.json`
**Generated:** 2026-08-06

## Identity (page-1 text)
- **Title:** Does Knowledge Distillation Really Work? Polina Kirichenko Alexander A. Alemi
- **Tags:** KD

## Abstract (extracted)
> Knowledge distillation is a popular technique for training a small student network to emulate a larger teacher model, such as an ensemble of networks. We show that while knowledge distillation can improve student generalization, it does not typically work as it is commonly understood: there often remains a surprisingly large discrepancy between the predictive distributions of the teacher and the student, even in cases when the student has the capacity to perfectly match the teacher. We identify difﬁculties in optimization as a key reason for why the student is unable to match the teacher. We also show how the details of the dataset used for distillation play a role in how closely the student matches the teacher — and that more closely matching the teacher paradoxically does not always lead to better student generalization.

## Table headers present in PDF text (exact lines)
- `Table 1: We examine whether ﬁdelity can be improved in the context of ResNet-20 self-distillation on`
- `Table 2: Distillation results when the dataset is varied. All metrics are computed on the test set. We`

## Metric-bearing lines (exact PDF lines; PASS2-verified)
- `distillation [20] argues that Bucil˘a et al. [5] “demonstrate convincingly that the knowledge acquired`
- `Conversely, in Figure 1 we show that with modern architectures knowledge distillation can lead to`
- `distillation dataset. In Section 6 we investigate the hypothesis that low ﬁdelity is an optimization`
- `Figure 1: Evaluating the ﬁdelity of knowledge distillation. The effect of enlarging the CIFAR-100`
- `ResNet-56 networks. Student ﬁdelity increases as the dataset grows, but test accuracy decreases.`
- `Knowledge distillation can improve model efﬁciency [34, 40], unsupervised domain adaptation [33],`
- `early work proposed distilling ensembles of shallow networks into a single network [49], an idea`
- `which resonates with more recent work on the distillation of deep ensembles [2, 7, 41, 45, 47].`
- `Recently Fakoor et al. [12] developed a data-augmentation scheme for the distillation of large`
- `student top-1 accuracy. In this paper we investigate many of the same prescriptions, including`
- `Hinton et al. [20] proposed a simple approach to knowledge distillation. The student minimizes a`
- `both scaled by a temperature hyperparameter τ > 0. If τ = 1 then LKD is similarly equivalent to the`
- `To measure generalization, we report top-1 accuracy, negative log-likelihood (NLL) and expected`
- `metrics requires some care. For example, suppose we have three independent models: f1, f2, and f3`
- `that respectively achieve 55%, 75%, and 95% test accuracy. f1 and f3 can agree on at most 60% of`
- `claim about f2 being a better distillation of f3 since each model was trained completely independently.`
- `In this section, we present evidence that we are not able to distill large networks such as a ResNet-56`
- `We ﬁrst consider the easy task of distilling a LeNet-5 teacher into an identical student network as`
- `training set for 100 epochs, resulting in a 84% to 86% teacher test accuracy across different subsets.2`
- `We then distill the teacher using the full MNIST train dataset with 60,000 examples, as well as 25%,`
- `In Figure 2 we see that knowledge distillation works as expected. With enough examples the student`
- `Figure 2: LeNet-5 self-distillation on`
- `many more parameters than a LeNet-5, it is possible that`
- `We have seen in Figure 1(a) that with self-distillation the student can exceed the teacher performance,`
- `In Figure 1(b) we see that if we move from self-distillation to the distillation of a 3 ResNet-56 teacher`
- `would achieve over 99% test accuracy.`
- `Figure 3: Data augmentation and distillation: Test accuracy and teacher-student agreement when`
- `distilling a 5-component ResNet-56 teacher ensemble into a ResNet-56 student on CIFAR-100 with`
- `gap in ﬁdelity, even after the distillation set is enlarged with 50k GAN samples. In practice, the`
- `controllers [e.g., 10, 19, 24, 46, 48]. While in self-distillation generalization and ﬁdelity are in`

## CuKD freeze notes (non-numeric)
- KD neighborhood → compare to C1/C2; do not claim novelty of KD-for-IDS alone.
- **Numbers only from exact lines above.**

## Verification
- PASS1B: evidence JSON written with exact lines from extract
- PASS2: all `32` quoted lines re-found in full extract (by construction)
- PASS2_TEXT: OK; PASS2_VISUAL_TITLE: OK

## PASS2 independent text re-check (2026-08-06)
- **PASS2_OK** — 32/32 quotes exact-match in full extract

## PASS2 visual title check (2026-08-06)
- page_001.png exists: True
- title word hit ratio vs page-1 text: 1.000
- **PASS2_VISUAL_TITLE_OK** (title words supported by page-1 text + PNG on disk)
