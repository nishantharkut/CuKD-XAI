# Review card: ticnna_hybrid_iot

**Status:** PASS1B+PASS2_TEXT+PASS2_VISUAL_TITLE OK
**PDF text pages extracted:** 15
**Ground truth extract:** `_extract/ticnna_hybrid_iot.full.txt`
**Evidence JSON:** `_pass1b_evidence/ticnna_hybrid_iot.json`
**Generated:** 2026-08-06

## Identity (page-1 text)
- **Title:** Digital Object Identifier 10.1109/ACCESS.2026.3663379 TICNN—A Hybrid Light-Weight CNN for Large
- **Tags:** XAI

## Abstract (extracted)
> The swift growth in ubiquitous connectivity, such as 5G and 6G, cloud computing, which enables data-driven automation and increases efficiency across industries, gave rise to the Internet of Things. This proliferation significantly expands the network attack surface, making IoT-based networks increasingly vulnerable to a multitude of attacks. To mitigate the attacks, Intrusion detection systems are developed. While developing the IDS, a primary challenge is the data collection of attack patterns and their training. The data collected is often massive, heterogeneous, and, critically, imbalanced. Training conventional models on such imbalanced data typically leads to poor performance, particularly in overlooking minority attack classes during multi-class classification. Our proposed system directly addresses this challenge of data imbalance. We introduce a hybrid technique, the Transformed Image Generative Adversarial Network (TIGAN), that generates high-fidelity attack samples and is then used to augment underrepresented attack classes, complemented by downsampling of overrepresented ones. The balanced data allows our lightweight deep learning model, Transformed Image Convolutional Neural Network (TICNN), to achieve robust attack detection. The experimental TICNN-based IDS delivers a faster detection time and detection accuracy of 99.56% at the macro level, which showcases its strong capability to identify diverse attacks, sporadic minority classes, in challenging IoT settings.

## Table headers present in PDF text (exact lines)
- `TABLE 1. Dataset details.`
- `TABLE 2. CNN architecture for TICNN.`
- `TABLE 3. Performance comparison of VGG16, VGG19, Xception, and TICNN models before and after performing TIGAN for CICIoT2023 (TICNN Enhanced.`
- `TABLE 7 shows our TICNN model achieves accuracy`
- `TABLE 4. Performance comparison of VGG16, VGG19, Xception, and TICNN models before and after performing TIGAN for CICIDS2017 (TICNN Enhanced).`
- `TABLE 5. Comparison of major and minor class attacks metrics before and after performing TIGAN for CICIoT2023 and CICIDS2017.`
- `TABLE 6. Computational complexity comparison: TICNN vs. Baselines.`
- `TABLE 7. Comparison of CICIDS2017 and CICIoT2023 with state of the art`
- `TABLE 8. Combined performance metrics across image sizes for CICIDS2017 and CICIoT2023.`
- `TABLE 9. Comparison of SMOTE, ADASYN, and TIGAN sampling methods`
- `TABLE 10. TIGAN performance in different sampling scenarios.`
- `TABLE 11. Computational feasibility comparison: Tabular sampling vs.`

## Metric-bearing lines (exact PDF lines; PASS2-verified)
- `to achieve 1.81 million parameters while attaining 99.56%`
- `DL models to improve detection accuracy [14]. These`
- `I ←reshape(V, (R, C, 3))`
- `I ←reshape(V, (R, C, 3))`
- `Mi denotes the metric (precision, recall, or F1-score) for`
- `Xception had the lowest macro average precision at 0.722.`
- `achieving a precision, recall, and F1-score of 0.998. Its macro`
- `average precision was 0.855, the highest among the models,`
- `weighted average precision of 0.997, recall of 0.995, and`
- `F1-score of 0.996. Its macro average precision and recall also`
- `substantial weighted average precision, recall, and F1-scores,`
- `along with overall accuracy, primarily in the 0.984 to`
- `Macro Precision and F1-Score) to 0.755 (TICNN’s Macro`
- `average precision, recall, and overall accuracy in the 0.980 to`
- `0.989 range, their F1-scores and macro average scores`
- `precision, recall, and F1 score reveal the model’s accuracy in`
- `actual attacks. The F1 score balances both metrics.`
- `very low performance, with precision, recall, and F1 scores`
- `to recalls of 0.962 and 0.961, respectively. Balancing the`
- `with scores near zero across precision, recall, and F1 metrics.`
- `instance, Backdoor Malware improved in recall from 0.500 to`
- `previously undetected (with recalls of 0.000), saw a recall`
- `TABLE 7 shows our TICNN model achieves accuracy`
- `Cross-Validation in Section IV-D.3. The reported accuracy of`
- `the baseline architectures. Table 6 details the Parameter`
- `VGG19 rely on massive parameter sets (approx. 138M`
- `only 1.81 million parameters. This represents a model size`
- `CICIDS2017). For both datasets, the accuracy obtained`
- `duced 96.5% and 98.0% accuracy for the CICIoT2023 and`
- `highest accuracy of 99.56% for CICIoT2023 and 99.98%`

## CuKD freeze notes (non-numeric)
- XAI neighborhood → do not invent Spearman ρ; C6 is CuKD measurement.
- **Numbers only from exact lines above.**

## Verification
- PASS1B: evidence JSON written with exact lines from extract
- PASS2: all `42` quoted lines re-found in full extract (by construction)
- PASS2_TEXT: OK; PASS2_VISUAL_TITLE: OK

## PASS2 independent text re-check (2026-08-06)
- **PASS2_OK** — 42/42 quotes exact-match in full extract

## PASS2 visual title check (2026-08-06)
- page_001.png exists: True
- title word hit ratio vs page-1 text: 1.000
- **PASS2_VISUAL_TITLE_OK** (title words supported by page-1 text + PNG on disk)
