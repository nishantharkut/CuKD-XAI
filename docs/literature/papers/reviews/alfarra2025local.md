# Review card: alfarra2025local

**Status:** PASS1B+PASS2_TEXT+PASS2_VISUAL_TITLE OK
**PDF text pages extracted:** 14
**Ground truth extract:** `_extract/alfarra2025local.full.txt`
**Evidence JSON:** `_pass1b_evidence/alfarra2025local.json`
**Generated:** 2026-08-06

## Identity (page-1 text)
- **Title:** cient Hybrid Learning for Secure Wireless Sensor Networks ECTI Transactions on Computer and Information Technology
- **Tags:** MCU, WSN, gateway

## Abstract (extracted)
> Article information: Wireless Sensor Networks (WSNs) power critical applications from envi- ronmental monitoring to Internet-of-Medical-Things healthcare yet their tiny batteries and low-end microcontrollers leave them exposed to net- work-layer Denial-of-Service (DoS) attacks such as Blackhole, Grayhole, Flooding and TDMA scheduling. Signature IDSs miss zero-day variants and shallow machine-learning detectors produce many false alarms, while running monolithic deep-learning models on every node exhaust energy re- serve. We introduce a two-stage hybrid IDS in which each sensor executes an integer-only rule lter that costs ≤0.05 mJ per packet and discards ≈95% of benign tra c, forwarding only agged ows over BLE/LoRa to an edge gateway. There, a 50 %-pruned, 8-bit CNN-LSTM processes 32-window batches in 28 mJ and ≈42 ms. Ex-periments on the public WSN-DS corpus, augmented by ns-3 simulations of a 50-node LoRa net- work, show that the scheme achieves 98 % accuracy, 0.93 macro-F1 and minority-class recalls of 0.840.95 while extending network lifetime (T50) to 69 days an 82 % gain over on-node GRU and 35 % over a signature IDS. Removing the rule lter erases most of the lifetime bene t without aecting accuracy, con rming that local triage, not downsized deep models, is the key to energy e ciency. The evaluation answers four research questions covering optimal hybrid architecture, rule- lter tuning, node-level energy overhead, and performance trade-os against traditional ML and stan- dalone DL baselines. These ndings demonstrate that intelligent workload partitioning can deliver deep-learning-level security without shortening the lifetime of resource-constrained WSN deployments.

## Table headers present in PDF text (exact lines)
- `Table 1:`
- `Table 2:`
- `Table 3:`
- `Table 4.`
- `Table 4:`
- `Table 5: Attack Scenarios Injected in the ns-3 Sim-`
- `Table 6.`
- `Table 6:`
- `Table 7:`
- `Table 8:`
- `Table 9. show that removing any single optimiza-`
- `Table 9: Ablation Study of Hybrid IDS Components.`

## Metric-bearing lines (exact PDF lines; PASS2-verified)
- `work, show that the scheme achieves 98 % accuracy, 0.93 macro-F1 and`
- `minority-class recalls of 0.840.95 while extending network lifetime (T50)`
- `% overall accuracy [2, 7, 9]. Never-theless, their false-`
- `accuracy on the WSN-DS benchmark [3, 9, 12]. The`
- `to suppress false positives yet preserve recall [5, 7,`
- `timeout elapses, keeping latency well below the 250`
- `∼96 % accuracy on the WSN-DS dataset while reduc-`
- `LSTM achieves >95 % detection accuracy for packet-`
- `overall accuracy but reveal low F1-scores for imbal-`
- `anced attacks (e.g., Grayhole F1 ≈0.59). Our hybrid`
- `that SVM yields ∼92 % accuracy but consumes ∼120`
- `uses ∼50 mJ with ∼85 % accuracy. These ﬁndings`
- `nected layers achieves ∼98 % accuracy across Black-`
- `tection latency on a microcontroller reduces by ∼60`
- `• Result: 95.4 % macro-F1, –65 % MACs, –50 %`
- `NS-3 Simulation Parameters and Experi-`
- `F1, AUC.`
- `16 ﬂagged windows (or 2 s) balances latency (∼42`
- `Sensor mock-up: LoPy4 ESP32 running CMSIS-`
- `Detection Accuracy, Precision, Recall, Macro-F1,`
- `25 of 50 nodes deplete to 0 % SoC. Latency Sensor`
- `5.1 Detection Accuracy (Figure 3)`
- `achieves near-perfect overall accuracy (98 %) while`
- `macro-F1 of 0.93 conﬁrms that high accuracy is not`
- `Macro-F1`
- `Macro-F1`
- `5.3 Latency`
- `(§4.4), end-to-end decision latency averages 42 ms`
- `Macro F1`
- `Highest F1,`

## CuKD freeze notes (non-numeric)
- MCU/embedded neighborhood → compare to C4 dual-board RF-KD HIL; Javed is tree-on-ESP32 prior.
- WSN neighborhood → Almomani WSN-DS lineage.
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
