# Review card: almomani2016wsnds

**Status:** PASS1B+PASS2_TEXT+PASS2_VISUAL_TITLE OK
**PDF text pages extracted:** 16
**Ground truth extract:** `_extract/almomani2016wsnds.full.txt`
**Evidence JSON:** `_pass1b_evidence/almomani2016wsnds.json`
**Generated:** 2026-08-06

## Identity (page-1 text)
- **Title:** WSN-DS: A Dataset for Intrusion Detection Systems in Wireless Sensor Networks Iman Almomani,1,2 Bassam Al-Kasasbeh,2 and Mousa AL-Akhras2,3
- **Tags:** WSN

## Abstract (extracted)
_Not auto-detected; open full extract._

## Table headers present in PDF text (exact lines)
- `Table 1: Comparison between the mathematical model and simulation results.`
- `Table 2: Applying Theorem 3 equation to round 1 of simulation round.`
- `Table 1 shows 100% match between the mathemati-`
- `Table 3: Observations for five different simulation scenarios (Aâ€“E) when determining the number of nodes monitored by each node.`
- `Table 4: Sample from WSN-DS dataset.`
- `Table 5: Ns-2 simulation parameters.`
- `Table 6: Dataset separated 60% training set and 40 testing sets using`
- `Table 6 shows data separation using holdout method.`
- `Table 7: Parameters for MLP neural network classifier.`
- `Table 7 shows the parameters and the values used in`
- `Table 8 shows the Confusion matrix for this method.`
- `Table 9 shows the results of the remaining metrics for the`

## Metric-bearing lines (exact PDF lines; PASS2-verified)
- `DS improved the ability of IDS to achieve higher classification accuracy rate. WEKA toolbox was used with holdout and 10-Fold`
- `Table 5: Ns-2 simulation parameters.`
- `Simulation parameters are summarized in Table 5.`
- `Table 7: Parameters for MLP neural network classifier.`
- `Table 7 shows the parameters and the values used in`
- `hidden layer, an overall classification accuracy of 97.5431%`
- `From Table 9, it can be concluded that the accuracy`
- `architecture. From Table 10, it can be shown that the accuracy`

## CuKD freeze notes (non-numeric)
- WSN neighborhood â†’ Almomani WSN-DS lineage.
- **Numbers only from exact lines above.**

## Verification
- PASS1B: evidence JSON written with exact lines from extract
- PASS2: all `20` quoted lines re-found in full extract (by construction)
- PASS2_TEXT: OK; PASS2_VISUAL_TITLE: OK

## PASS2 independent text re-check (2026-08-06)
- **PASS2_OK** â€” 20/20 quotes exact-match in full extract

## PASS2 visual title check (2026-08-06)
- page_001.png exists: True
- title word hit ratio vs page-1 text: 1.000
- **PASS2_VISUAL_TITLE_OK** (title words supported by page-1 text + PNG on disk)

## DEEP_VISUAL (manual image page 001 + text conclusions)

- WSN-DS foundational; Grayhole ~75.6% class acc in ANN baseline

