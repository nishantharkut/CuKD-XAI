# Review card: yang2023selfkd

**Status:** PASS1B+PASS2_TEXT+PASS2_VISUAL_TITLE OK
**PDF text pages extracted:** 6
**Ground truth extract:** `_extract/yang2023selfkd.full.txt`
**Evidence JSON:** `_pass1b_evidence/yang2023selfkd.json`
**Generated:** 2026-08-06

## Identity (page-1 text)
- **Title:** A Lightweight Approach for Network Intrusion Detection based on Self-Knowledge Distillation
- **Tags:** KD

## Abstract (extracted)
> Network Intrusion Detection (NID) works as a ker- nel technology for the security network environment, obtaining extensive research and application. Despite enormous efforts by researchers, NID still faces challenges in deploying on resource- constrained devices. To improve detection accuracy while reduc- ing computational costs and model storage simultaneously, we propose a lightweight intrusion detection approach based on self-knowledge distillation, namely LNet-SKD, which achieves the trade-off between accuracy and efficiency. Specifically, we carefully design the DeepMax block to extract compact repre- sentation efficiently and construct the LNet by stacking DeepMax blocks. Furthermore, considering compensating for performance degradation caused by the lightweight network, we adopt batch- wise self-knowledge distillation to provide the regularization of training consistency. Experiments on benchmark datasets demonstrate the effectiveness of our proposed LNet-SKD, which outperforms existing state-of-the-art techniques with fewer pa- rameters and lower computation loads.

## Table headers present in PDF text (exact lines)
_None detected (image-only tables possible)._

## Metric-bearing lines (exact PDF lines; PASS2-verified)
- `Fig. 1. F1 score v.s. the number of model parameters on NSL-KDD datasets.`
- `trade-off between efficiency and accuracy (See Fig. 1). Our`
- `distillation. Wang et al. [14] proposed a knowledge distillation`
- `parameters, which is less than classical ones Ci Ã— (K2 Ã— Co)`
- `Fig. 3. The impact of hyper-parameters.`
- `Accuracy, Precision, Recall, and F1 score are used to eval-`
- `FLOPs of 194.58K (â†“63.1%). Meanwhile, our self-distillation`
- `proving the accuracy of LNet by 2.1% / 1.6% on NSL-KDD`
- `F1-score. From the Table III, we observe that the DeepMax`
- `block only reduces accuracy by 1.89% and 1.57% on two`
- `mance in terms of accuracy and F1 score on both datasets with`
- `only 4.94K and 5K parameters, respectively. Compared with`
- `size by about 62% with a slight improvement in accuracy`
- `and F1 score. Furthermore, LNet-SKD outperforms baseline`
- `[12] G. Hinton, O. Vinyals, J. Dean et al., â€œDistilling the knowledge in a`
- `[15] Y. Shen, L. Xu, Y. Yang, Y. Li, and Y. Guo, â€œSelf-distillation from the`

## CuKD freeze notes (non-numeric)
- KD neighborhood â†’ compare to C1/C2; do not claim novelty of KD-for-IDS alone.
- **Numbers only from exact lines above.**

## Verification
- PASS1B: evidence JSON written with exact lines from extract
- PASS2: all `16` quoted lines re-found in full extract (by construction)
- PASS2_TEXT: OK; PASS2_VISUAL_TITLE: OK

## PASS2 independent text re-check (2026-08-06)
- **PASS2_OK** â€” 16/16 quotes exact-match in full extract

## PASS2 visual title check (2026-08-06)
- page_001.png exists: True
- title word hit ratio vs page-1 text: 1.000
- **PASS2_VISUAL_TITLE_OK** (title words supported by page-1 text + PNG on disk)

## DEEP_VISUAL (manual image pages 001,004,005)

- Table III NSL-KDD LNet-SKD Acc 98.66 F1 89.03 Para 4.94K FLOPs 194.58K

