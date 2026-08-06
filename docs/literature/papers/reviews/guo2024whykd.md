# Review card: guo2024whykd

**Status:** PASS1B+PASS2_TEXT+PASS2_VISUAL_TITLE OK
**PDF text pages extracted:** 29
**Ground truth extract:** `_extract/guo2024whykd.full.txt`
**Evidence JSON:** `_pass1b_evidence/guo2024whykd.json`
**Generated:** 2026-08-06

## Identity (page-1 text)
- **Title:** Why does Knowledge Distillation Work? Rethink its Attention and Fidelity Mechanism
- **Tags:** KD, XAI

## Abstract (extracted)
_Not auto-detected; open full extract._

## Table headers present in PDF text (exact lines)
- `Table 1: Affinity, and Validation Accuracy (Val-Acc) of models with various data aug-`

## Metric-bearing lines (exact PDF lines; PASS2-verified)
- `Knowledge Distillation (KD) (Hinton et al, 2015) is renowned for its effec-`
- `in (Stanton et al, 2021), which empirically shows that good student accuracy`
- `plays a crucial role in shaping KD (Stanton et al, 2021).`
- `zs and the one-hot labels ys. LKD1,2 is the added knowledge distillation term`
- `of 100, and their balanced counterparts. Hyperparameters remain consistent`
- `of top-1 accuracy and overfitting during both the training and validation`
- `Table 1: Affinity, and Validation Accuracy (Val-Acc) of models with various data aug-`
- `for all the others. The student model of ResNet18 is distilled for: 200 epochs`
- `and 165 epochs for ImageNet-LT dataset, when their validation accuracy`
- `Hyper-parameters, including temperatures of τ = 10, hard label weight`
- `and Tmax = 60, etamin = 0 for student distillation.`
- `with the following hyperparameters: step1 = 25, step2 = 40, step3 = 60`
- `and step1 = 35, step2 = 50 for ImageNet-LT. During student distillation,`
- `with the following hyperparameters: step1 = 190, step2 = 195 for CIFAR-`
- `attention maps. In this experiment, two ViT-b32 teachers are distilled on`
- `In the main text, we focused on Knowledge Distillation (KD) with 2`
- `(2020) and DMAE Bai et al (2023), focusing on the top-1 validation accuracy.`
- `distilled to one ResNet18 student model, with TwSs. Ours(2T) is refferred to`
- `the KD with two ResNet50 teacher models distilled to one ResNet18 student`
- `teacher models distilled to one ResNet18 student model, with T1sT2wT3wSs.`
- `validation accuracy gap. Figure H.9 Bottom further compares the model cal-`
- `Asif, U., Tang, J., Harrer, S., 2019. Ensemble Knowledge Distillation for`
- `C., 2023. Masked Autoencoders Enable Efficient Knowledge Distillers, in:`
- `Hinton, G., Vinyals, O., Dean, J., 2015. Distilling the Knowledge in a Neural`

## CuKD freeze notes (non-numeric)
- KD neighborhood → compare to C1/C2; do not claim novelty of KD-for-IDS alone.
- XAI neighborhood → do not invent Spearman ρ; C6 is CuKD measurement.
- **Numbers only from exact lines above.**

## Verification
- PASS1B: evidence JSON written with exact lines from extract
- PASS2: all `25` quoted lines re-found in full extract (by construction)
- PASS2_TEXT: OK; PASS2_VISUAL_TITLE: OK

## PASS2 independent text re-check (2026-08-06)
- **PASS2_OK** — 25/25 quotes exact-match in full extract

## PASS2 visual title check (2026-08-06)
- page_001.png exists: True
- title word hit ratio vs page-1 text: 1.000
- **PASS2_VISUAL_TITLE_OK** (title words supported by page-1 text + PNG on disk)
