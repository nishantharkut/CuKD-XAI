# Review card: chawla2002smote

**Status:** PASS1B+PASS2_TEXT+PASS2_VISUAL_TITLE OK
**PDF text pages extracted:** 37
**Ground truth extract:** `_extract/chawla2002smote.full.txt`
**Evidence JSON:** `_pass1b_evidence/chawla2002smote.json`
**Generated:** 2026-08-06

## Identity (page-1 text)
- **Title:** SMOTE: Synthetic Minority Over-sampling Technique Nitesh V. Chawla
- **Tags:** n/a

## Abstract (extracted)
> An approach to the construction of classiﬁers from imbalanced datasets is described. A dataset is imbalanced if the classiﬁcation categories are not approximately equally rep- resented. Often real-world data sets are predominately composed of “normal” examples with only a small percentage of “abnormal” or “interesting” examples. It is also the case that the cost of misclassifying an abnormal (interesting) example as a normal example is often much higher than the cost of the reverse error. Under-sampling of the majority (nor- mal) class has been proposed as a good means of increasing the sensitivity of a classiﬁer to the minority class. This paper shows that a combination of our method of over-sampling the minority (abnormal) class and under-sampling the majority (normal) class can achieve better classiﬁer performance (in ROC space) than only under-sampling the majority class. This paper also shows that a combination of our method of over-sampling the minority class and under-sampling the majority class can achieve better classiﬁer performance (in ROC space) than varying the loss ratios in Ripper or class priors in Naive Bayes. Our method of over-sampling the minority class involves creating synthetic minority class examples. Experiments are performed using C4.5, Ripper and a Naive Bayes classiﬁer. The method is evaluated using the area under the Receiver Operating Characteristic curve (AUC) and the ROC convex hull strategy.

## Table headers present in PDF text (exact lines)
- `Table 1: Example of generation of synthetic examples (SMOTE).`
- `Table 2: Dataset distribution`
- `Table 3: AUC’s [C4.5 as the base classiﬁer] with the best highlighted in bold.`
- `Table 4: Cross-validation results (Kubat et al., 1998)`
- `Table 5: Cross-validation results for SMOTE at 500% SMOTE on the Oil data set.`
- `Table 6: Example of nearest neighbor computation for SMOTE-NC.`
- `Table 7: Example of SMOTE-N`

## Metric-bearing lines (exact PDF lines; PASS2-verified)
- `performance metric. Error rate is 1 −Accuracy. In the presence of imbalanced datasets`
- `f1 1 = 6 f2 1 = 4 f2 1 - f1 1 = -2`
- `f1 2 = 4 f2 2 = 3 f2 2 - f1 2 = -1`
- `(f1’,f2’) = (6,4) + rand(0-1) * (-2,-1)`
- `for this case, the default accuracy would be 97.68% when every sample is labeled non-`
- `results from (Kubat et al., 1998). Acc+ is the accuracy on positive (minority) examples and`
- `Acc−is the accuracy on the negative (majority) examples. Figure 25 shows the trend for`
- `F1 = 1 2 3 A B C [Let this be the sample for which we are computing nearest`
- `So, Euclidean Distance between F2 and F1 would be:`
- `which diﬀer for the two feature vectors: F1 and F2.`
- `Let F1 = A B C D E be the feature vector under consideration`
- `Provost, F., Fawcett, T., & Kohavi, R. (1998). The Case Against Accuracy Estimation`
- `Swets, J. (1988). Measuring the Accuracy of Diagnostic Systems. Science, 240, 1285–1293.`

## CuKD freeze notes (non-numeric)
- Background/foundational cite.
- **Numbers only from exact lines above.**

## Verification
- PASS1B: evidence JSON written with exact lines from extract
- PASS2: all `20` quoted lines re-found in full extract (by construction)
- PASS2_TEXT: OK; PASS2_VISUAL_TITLE: OK

## PASS2 independent text re-check (2026-08-06)
- **PASS2_OK** — 20/20 quotes exact-match in full extract

## PASS2 visual title check (2026-08-06)
- page_001.png exists: True
- title word hit ratio vs page-1 text: 1.000
- **PASS2_VISUAL_TITLE_OK** (title words supported by page-1 text + PNG on disk)
