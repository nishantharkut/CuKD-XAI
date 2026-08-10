# Review card: banbury2021tinyml

**Status:** PASS1B+PASS2_TEXT+PASS2_VISUAL_TITLE OK
**PDF text pages extracted:** 8
**Ground truth extract:** `_extract/banbury2021tinyml.full.txt`
**Evidence JSON:** `_pass1b_evidence/banbury2021tinyml.json`
**Generated:** 2026-08-06

## Identity (page-1 text)
- **Title:** BENCHMARKING TINYML SYSTEMS: CHALLENGES AND DIRECTION Colby R. Banbury 1 Vijay Janapa Reddi 1 Max Lam 1 William Fu 1 Amin Fazel 2 Jeremy Holleman 3 4
- **Tags:** MCU

## Abstract (extracted)
> Recent advancements in ultra-low-power machine learning (TinyML) hardware promises to unlock an entirely new class of smart applications. However, continued progress is limited by the lack of a widely accepted benchmark for these systems. Benchmarking allows us to measure and thereby systematically compare, evaluate, and improve the performance of systems and is therefore fundamental to a ﬁeld reaching maturity. In this position paper, we present the current landscape of TinyML and discuss the challenges and direction towards developing a fair and useful hardware benchmark for TinyML workloads. Furthermore, we present our four benchmarks and discuss our selection methodology. Our viewpoints reﬂect the collective thoughts of the TinyMLPerf working group that is comprised of over 30 organizations.

## Table headers present in PDF text (exact lines)
- `Table 3 lists common model types for TinyML use cases.`
- `Table 1. Survey of TinyML Use Cases, Models, and Datasets`
- `Table 2. Existing Benchmarks`
- `Table 3. TinyMLPerf Benchmarking Suite`

## Metric-bearing lines (exact PDF lines; PASS2-verified)
- `1-16b precision reconﬁgurable digital in-memory com-`

## CuKD freeze notes (non-numeric)
- MCU/embedded neighborhood → compare to C4 dual-board RF-KD HIL; Javed is tree-on-ESP32 prior.
- **Numbers only from exact lines above.**

## Verification
- PASS1B: evidence JSON written with exact lines from extract
- PASS2: all `5` quoted lines re-found in full extract (by construction)
- PASS2_TEXT: OK; PASS2_VISUAL_TITLE: OK

## PASS2 independent text re-check (2026-08-06)
- **PASS2_OK** — 5/5 quotes exact-match in full extract

## PASS2 visual title check (2026-08-06)
- page_001.png exists: True
- title word hit ratio vs page-1 text: 1.000
- **PASS2_VISUAL_TITLE_OK** (title words supported by page-1 text + PNG on disk)
