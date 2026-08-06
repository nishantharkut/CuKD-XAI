# Review card: diab2025hardware

**Status:** PASS1B+PASS2_TEXT+PASS2_VISUAL_TITLE OK
**PDF text pages extracted:** 6
**Ground truth extract:** `_extract/diab2025hardware.full.txt`
**Evidence JSON:** `_pass1b_evidence/diab2025hardware.json`
**Generated:** 2026-08-06

## Identity (page-1 text)
- **Title:** Intrusion Detection on Resource-Constrained IoT Devices with Hardware-Aware ML and DL
- **Tags:** EdgeIIoT, gateway

## Abstract (extracted)
> This paper proposes a hardware-aware intrusion detection system (IDS) for Internet of Things (IoT) and Industrial IoT (IIoT) networks; it targets scenarios where classification is essential for fast, privacy-preserving, and resource-efficient threat detection. The goal is to optimize both tree-based machine learning (ML) models and compact deep neural networks (DNNs) within strict edge-device constraints. This allows for a fair comparison and reveals trade-offs between model families. We apply constrained grid search for tree-based classifiers and hardware-aware neural architecture search (HW-NAS) for 1D convolutional neural networks (1D-CNNs). Evaluation on the Edge-IIoTset benchmark shows that selected models meet tight flash, RAM, and compute limits: LightGBM achieves 95.3% accuracy using 75 KB flash and 1.2 K operations, while the HW- NASâ€“optimized CNN reaches 97.2% with 190 KB flash and 840 K floating-point operations (FLOPs). We deploy the full pipeline on a Raspberry Pi 3 B+, confirming that tree-based models operate within 30 ms and that CNNs remain suitable when accuracy outweighs latency. These results highlight the practicality of hardware-constrained model design for real-time IDS at the edge.

## Table headers present in PDF text (exact lines)
_None detected (image-only tables possible)._

## Metric-bearing lines (exact PDF lines; PASS2-verified)
- `flash, RAM, and compute limits: LightGBM achieves 95.3%`
- `accuracy using 75 KB flash and 1.2 K operations, while the HW-`
- `NASâ€“optimized CNN reaches 97.2% with 190 KB flash and 840 K`
- `within 30 ms and that CNNs remain suitable when accuracy`
- `reporting satisfactory accuracy on public benchmarks [4]â€“`
- `vary with hyperparameter configurations [8]. Deep models,`
- `accurate models: LightGBM achieves 95.3% accuracy`
- `using just 75 KB of flash and 1.2 K operations, while the`
- `optimized CNN reaches 97.2% accuracy within 190 KB`
- `performance with an accuracy of 80.83%, while [7] evaluated`
- `accuracy. In [11], the authors have evaluated several tree-`
- `achieved the best performance with 89.09% accuracy using`
- `accuracy. In [6], the authors have proposed an attention-based`
- `with an attention mechanism, achieving 99.95% accuracy.`
- `proaches still focus mainly on accuracy [8], [9], overlooking`
- `inference latency and energy consumption [17], while Flash`
- `berry Pi 3 B+) and the need for stable, low-latency operation`
- `budgets of â‰¤300 KB flash, â‰¤50 KB RAM, and â‰¤1.5Ã—106`
- `show the average accuracy (Acc.) and F1-score (F1) in %,`
- `LightGBM achieves the highest accuracy (95.25%) and F1-`
- `score (94.74%) with the smallest flash (74.93 KB) and RAM`
- `delivers comparable accuracy (95.11%) but incurs a flash size`
- `Forest (RF) shows slightly lower accuracy (94.12%) and F1`
- `columns report accuracy (Acc.) and F1-score (F1) (in %),`
- `The proposed model achieves 96.73% accuracy and 97.24%`
- `F1-score with minimal variation, indicating strong perfor-`
- `computation, with flash sizes exceeding 1 MB and FLOPs`
- `of flash, 6.89 KB of RAM, and 838.89 K FLOPs.`
- `with only 52.03 K FLOPs but suffers from a low F1-score of`
- `plied method in [1] reports 94.67% accuracy, and the proposed`

## CuKD freeze notes (non-numeric)
- Edge-IIoT neighborhood â†’ C10 group-aware discussion.
- **Numbers only from exact lines above.**

## Verification
- PASS1B: evidence JSON written with exact lines from extract
- PASS2: all `30` quoted lines re-found in full extract (by construction)
- PASS2_TEXT: OK; PASS2_VISUAL_TITLE: OK

## PASS2 independent text re-check (2026-08-06)
- **PASS2_OK** â€” 30/30 quotes exact-match in full extract

## PASS2 visual title check (2026-08-06)
- page_001.png exists: True
- title word hit ratio vs page-1 text: 1.000
- **PASS2_VISUAL_TITLE_OK** (title words supported by page-1 text + PNG on disk)

## DEEP_VISUAL (manual image pages 001,003,005)

- LightGBM 95.25 Acc / 94.74 F1 / ~75KB flash; Pi 3 B+ deploy; no KD

