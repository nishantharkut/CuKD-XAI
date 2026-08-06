# Review card: krishna2022disagreement

**Status:** PASS1B+PASS2_TEXT+PASS2_VISUAL_TITLE OK
**PDF text pages extracted:** 34
**Ground truth extract:** `_extract/krishna2022disagreement.full.txt`
**Evidence JSON:** `_pass1b_evidence/krishna2022disagreement.json`
**Generated:** 2026-08-06

## Identity (page-1 text)
- **Title:** The Disagreement Problem in Explainable Machine Learning: A Practitioner’s Perspective
- **Tags:** XAI

## Abstract (extracted)
_Not auto-detected; open full extract._

## Table headers present in PDF text (exact lines)
- `Table 1: Themes summarizing how participants decided between explanations when faced with disagreement`
- `Table 2: Disagreement on ImageNet between LIME and KernelSHAP`
- `Table 3: Reasons participants chose the top four most favored explainability algorithms (KernelSHAP,`
- `Table 4: Reasons people answered "It depends" after being asked to choose between disagreements`
- `Table 5: Representative quotes highlighting themes of how participants address the disagreement problem in`
- `Table 6: Themes summarizing how academic participants decided between explanations when faced with`
- `Table 7: Themes summarizing how industry participants decided between explanations when faced with`

## Metric-bearing lines (exact PDF lines; PASS2-verified)
- `et al., 2016b), SHAP (Lundberg & Lee, 2017)) and gradient-based methods (e.g., Gradient times Input`
- `terms of accuracy (Ribeiro et al., 2016b). This has led to significant interest in post hoc explanation methods,`
- `SHAP, and gradient-based methods) in their day-to-day workflow. 19 participants (76%) were male, and 6`
- `(2016b) and KernelShap Lundberg & Lee (2017)), and four gradient-based explanation methods (Vanilla`
- `Figure 4 that LIME exhibits higher agreement with other explanation methods compared to KernelSHAP`
- `0.273, as opposed to 0.113 in case of KernelSHAP). This finding is consistent with the insights we observed`
- `point generated using two different explanation methods (e.g., LIME and KernelSHAP in Figure 5). The`
- `chosen, finding that indeed, certain methods were favored over others. While KernelSHAP was chosen 66.7%`
- `3. LIME and SHAP are better because the COMPAS dataset comprises tabular data (23%):`
- `3. LIME/SHAP are`
- `model performs with 90.67% accuracy. The architecture comprises an embedding layer of dimension 300,`
- `% on Accuracy@1 and Accuracy@5 metrics6, respectively.`
- `for 7,600 samples in the test set. For LIME and KernelSHAP, we follow the convergence analysis described`
- `perturbations for LIME and KernelSHAP. Integrated Gradients explanations were generated using 500`
- `LIME and KernelSHAP, we chose 100 perturbations to train the surrogate model as we did not notice any`
- `significant changes in attributions beyond 50 perturbations. KernelSHAP and LIME were used to compute`
- `Table 2: Disagreement on ImageNet between LIME and KernelSHAP`
- `6. Which explainability methods do you use in your day to day workflow? (eg: LIME, KernelSHAP,`
- `• [36%] SHAP is better for tabular data ("SHAP is more commonly used`
- `• [25%] SHAP is more familiar ("More information present + more`
- `• [14%] SHAP is a better algorithm overall ("SHAP seems more me-`
- `Table 3: Reasons participants chose the top four most favored explainability algorithms (KernelSHAP,`
- `algorithms such as KernelSHAP were favored over other algorithms. In Table 3, we list the top reasons the`
- `that 14 of 19 participants used LIME, 14 of 19 participants used SHAP, and 13 of 19 participants used some`
- `both LIME and SHAP, with another 3 of 19 participants stating LIME only. We showcase some intriguing`
- `always prefer SHAP. Figure 21 shows that while both groups prefer SHAP in case of the disagreement, this`

## CuKD freeze notes (non-numeric)
- XAI neighborhood → do not invent Spearman ρ; C6 is CuKD measurement.
- **Numbers only from exact lines above.**

## Verification
- PASS1B: evidence JSON written with exact lines from extract
- PASS2: all `33` quoted lines re-found in full extract (by construction)
- PASS2_TEXT: OK; PASS2_VISUAL_TITLE: OK

## PASS2 independent text re-check (2026-08-06)
- **PASS2_OK** — 33/33 quotes exact-match in full extract

## PASS2 visual title check (2026-08-06)
- page_001.png exists: True
- title word hit ratio vs page-1 text: 1.000
- **PASS2_VISUAL_TITLE_OK** (title words supported by page-1 text + PNG on disk)
