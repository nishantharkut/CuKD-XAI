# Review card: adhane2025explainkd

**Status:** PASS1B+PASS2_TEXT+PASS2_VISUAL_TITLE OK
**PDF text pages extracted:** 10
**Ground truth extract:** `_extract/adhane2025explainkd.full.txt`
**Evidence JSON:** `_pass1b_evidence/adhane2025explainkd.json`
**Generated:** 2026-08-06

## Identity (page-1 text)
- **Title:** On Explaining Knowledge Distillation: Measuring and Visualising the Knowledge Transfer Process
- **Tags:** KD

## Abstract (extracted)
> Knowledge distillation (KD) remains challenging due to the opaque nature of the knowledge transfer process from a Teacher to a Student, making it difficult to address certain issues related to KD. To address this, we proposed UniCAM, a novel gradient-based visual explanation method, which effectively interprets the knowledge learned during KD. Our experimental results demonstrate that with the guidance of the Teacher’s knowledge, the Student model becomes more efficient, learning more relevant features while discarding those that are not relevant. We refer to the features learned with the Teacher’s guidance as distilled features and the features irrelevant to the task and ignored by the Student as residual features. Distilled features focus on key aspects of the input, such as textures and parts of objects. In con- trast, residual features demonstrate more diffused attention, often targeting irrelevant areas, including the backgrounds of the target objects. In addition, we proposed two novel metrics: the feature similarity score (FSS) and the rele- vance score (RS), which quantify the relevance of the dis- tilled knowledge. Experiments on the CIFAR10, ASIRRA, and Plant Disease datasets demonstrate that UniCAM and the two metrics offer valuable insights to explain the KD process.

## Table headers present in PDF text (exact lines)
- `Table 1. Relevance of features (RS) learned by Student (ResNet-`
- `Table 2. RS of Students trained using Response-based KD with`

## Metric-bearing lines (exact PDF lines; PASS2-verified)
- `performance of the Student [25], adapting the distillation`
- `6: Extract the distilled features by adjusting for the mutual influence:`
- `8: Compute the importance of the distilled features:`
- `Figure 1. Residual and distilled features after perturbation.`
- `tures. This is illustrated in Fig. 5, where the distilled fea-`
- `the plant disease classification3, distilled features accurately`
- `Figure 5. Sample visualisation of Distilled and residual features`
- `distilled features learned by two Students: ResNet-18 di-`
- `rectly distilled from ResNet-101 (R18-R101) and ResNet-`
- `18 distilled from ResNet-101 through Teacher assistant`
- `Fig. 8 shows that the saliency maps of the distilled features`
- `the model directly distilled from ResNet-101 and the Base`
- `Figure 8. Comparison of distilled and residual features between`
- `[23] Geoffrey Hinton, Oriol Vinyals, Jeff Dean, et al. Distilling`

## CuKD freeze notes (non-numeric)
- KD neighborhood → compare to C1/C2; do not claim novelty of KD-for-IDS alone.
- **Numbers only from exact lines above.**

## Verification
- PASS1B: evidence JSON written with exact lines from extract
- PASS2: all `16` quoted lines re-found in full extract (by construction)
- PASS2_TEXT: OK; PASS2_VISUAL_TITLE: OK

## PASS2 independent text re-check (2026-08-06)
- **PASS2_OK** — 16/16 quotes exact-match in full extract

## PASS2 visual title check (2026-08-06)
- page_001.png exists: True
- title word hit ratio vs page-1 text: 1.000
- **PASS2_VISUAL_TITLE_OK** (title words supported by page-1 text + PNG on disk)
