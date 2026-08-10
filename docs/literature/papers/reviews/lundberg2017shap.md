# Review card: lundberg2017shap

**Status:** PASS1B+PASS2_TEXT+PASS2_VISUAL_TITLE OK
**PDF text pages extracted:** 10
**Ground truth extract:** `_extract/lundberg2017shap.full.txt`
**Evidence JSON:** `_pass1b_evidence/lundberg2017shap.json`
**Generated:** 2026-08-06

## Identity (page-1 text)
- **Title:** A Uniﬁed Approach to Interpreting Model Scott M. Lundberg Paul G. Allen School of Computer Science
- **Tags:** XAI

## Abstract (extracted)
> Understanding why a model makes a certain prediction can be as crucial as the prediction’s accuracy in many applications. However, the highest accuracy for large modern datasets is often achieved by complex models that even experts struggle to interpret, such as ensemble or deep learning models, creating a tension between accuracy and interpretability. In response, various methods have recently been proposed to help users interpret the predictions of complex models, but it is often unclear how these methods are related and when one method is preferable over another. To address this problem, we present a uniﬁed framework for interpreting predictions, SHAP (SHapley Additive exPlanations). SHAP assigns each feature an importance value for a particular prediction. Its novel components include: (1) the identiﬁcation of a new class of additive feature importance measures, and (2) theoretical results showing there is a unique solution in this class with a set of desirable properties. The new class uniﬁes six existing methods, notable because several recent methods in the class lack the proposed desirable properties. Based on insights from this uniﬁcation, we present new methods that show improved computational performance and/or better consistency with human intuition than previous approaches.

## Table headers present in PDF text (exact lines)
_None detected (image-only tables possible)._

## Metric-bearing lines (exact PDF lines; PASS2-verified)
- `1https://github.com/slundberg/shap`
- `additive feature attribution methods (Section 3) and propose SHAP values as a uniﬁed measure of`
- `3. We propose new SHAP value estimation methods and demonstrate that they are better aligned`
- `of model predictions: Shapley regression values [4], Shapley sampling values [9], and Quantitative`
- `For Shapley regression values, hx maps 1 or 0 to the original input space, where 1 indicates the input`
- `Shapley regression values match Equation 1 and are hence an additive feature attribution method.`
- `Shapley sampling values are meant to explain any model by: (1) applying sampling approximations`
- `than 2|F | differences to be computed. Since the explanation model form of Shapley sampling values`
- `Property 1 (Local accuracy)`
- `as Shapley values [6]. Young (1985) demonstrated that Shapley values are the only set of values`
- `redundant in this setting (see Supplementary Material). Property 2 is required to adapt the Shapley`
- `values violate local accuracy and/or consistency (methods in Section 2 already respect missingness).`
- `Figure 1: SHAP (SHapley Additive exPlanation) values attribute to each feature the change in the`
- `Based on Sections 2 and 3, SHAP values provide the unique additive feature importance measure that`
- `in [9, 5, 7, 3], then SHAP values can be estimated directly using the Shapley sampling values method`
- `approximation of a permutation version of the classic Shapley value equations (Equation 8). Separate`
- `original model to obtain similar approximation accuracy (Section 5).`
- `very different from the classical Shapley value formulation of Equation 8. However, since linear`
- `solution to Equation 2 that satisﬁes Properties 1-3 – local accuracy, missingness and consistency. A`
- `choices for these parameters are made heuristically; using these choices, Equation 2 does not recover`
- `Below we show how to avoid heuristically choosing the parameters in Equation 2 and how to ﬁnd the`
- `Theorem 2 (Shapley kernel) Under Deﬁnition 1, the speciﬁc forms of πx′, L, and Ωthat make`
- `that is equivalent to the approximation of the SHAP mapping given in Equation 12, this enables`
- `The intuitive connection between linear regression and Shapley values is that Equation 8 is a difference`
- `For linear models, if we assume input feature independence (Equation 11), SHAP values can be`
- `Corollary 1 (Linear SHAP) Given a linear model f(x) = PM`
- `minimization formulation of Shapley values proposed in econometrics [2].`
- `Figure 2: (A) The Shapley kernel weighting is symmetric when all possible z′ vectors are ordered`
- `lets us compute the Shapley values of a max function with M inputs in O(M 2) time instead of`
- `connection between Shapley values and DeepLIFT [8]. If we interpret the reference value in Equation`

## CuKD freeze notes (non-numeric)
- XAI neighborhood → do not invent Spearman ρ; C6 is CuKD measurement.
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
