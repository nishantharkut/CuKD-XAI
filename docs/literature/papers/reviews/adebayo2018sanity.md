# Review card: adebayo2018sanity

**Status:** PASS1B+PASS2_TEXT+PASS2_VISUAL_TITLE OK
**PDF text pages extracted:** 30
**Ground truth extract:** `_extract/adebayo2018sanity.full.txt`
**Evidence JSON:** `_pass1b_evidence/adebayo2018sanity.json`
**Generated:** 2026-08-06

## Identity (page-1 text)
- **Title:** Sanity Checks for Saliency Maps Julius Adebayo∗, Justin Gilmer♯, Michael Muelly♯, Ian Goodfellow♯, Moritz Hardt♯†, Been Kim♯
- **Tags:** XAI

## Abstract (extracted)
> Saliency methods have emerged as a popular tool to highlight features in an input deemed relevant for the prediction of a learned model. Several saliency methods have been proposed, often guided by visual appeal on image data. In this work, we propose an actionable methodology to evaluate what kinds of explanations a given method can and cannot provide. We ﬁnd that reliance, solely, on visual assessment can be misleading. Through extensive experiments we show that some existing saliency methods are independent both of the model and of the data generating process. Consequently, methods that fail the proposed tests are inadequate for tasks that are sensitive to either data or model, such as, ﬁnding outliers in the data, explaining the relationship between inputs and outputs that the model learned, and debugging the model. We interpret our ﬁndings through an analogy with edge detection in images, a technique that requires neither training data nor model. Theory in the case of a linear model and a single-layer convolutional neural network supports our experimental ﬁndings2.

## Table headers present in PDF text (exact lines)
_None detected (image-only tables possible)._

## Metric-bearing lines (exact PDF lines; PASS2-verified)
- `methods are equivalent to the input ⊙gradient. Similarly, Lundberg and Lee [18] proposed SHAP`
- `x ∈R2. A gradient-based explanation for the model’s behavior for input x is given by the parameter`
- `model to greater than 95% training set accuracy. Note that the test accuracy is never better than`
- `(Recall that σ′(x) = 0 if x < 0 and 1 otherwise). This implies that the 3 × 3 activation pattern local`
- `networks lack sensitivity to parameter values. 2018.`
- `Randomization Tests We perform 2 types of randomizations. For the model parameter randomization tests, we`
- `model to greater than 95 percent training set accuracy. As expected the performance of these models on the tests`
- `master/research/slim#Pretrained. This model has a 93.9 top-5 accuracy on the ImageNet test set. For`
- `apply weight decay (penalty 0.001) to the weights of the network. The ﬁnal test set accuracy of this model is`
- `99.2 percent. For model parameter randomization test, we reinitialize each layer successively or independently`
- `with the ADAM optimizer for 20 thousand iterations. All non-linearities used are Relu. The ﬁnal test set accuracy`
- `America. This inception v4 model was trained retained the standard original parameters except it was trained`

## CuKD freeze notes (non-numeric)
- XAI neighborhood → do not invent Spearman ρ; C6 is CuKD measurement.
- **Numbers only from exact lines above.**

## Verification
- PASS1B: evidence JSON written with exact lines from extract
- PASS2: all `12` quoted lines re-found in full extract (by construction)
- PASS2_TEXT: OK; PASS2_VISUAL_TITLE: OK

## PASS2 independent text re-check (2026-08-06)
- **PASS2_OK** — 12/12 quotes exact-match in full extract

## PASS2 visual title check (2026-08-06)
- page_001.png exists: True
- title word hit ratio vs page-1 text: 1.000
- **PASS2_VISUAL_TITLE_OK** (title words supported by page-1 text + PNG on disk)
