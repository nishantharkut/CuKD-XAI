# Final Evidence Completion Plan

## Scope

This plan completes the current CuKD-XAI research without replacing or relabeling historical artifacts. New evidence is written to versioned, non-overwriting directories. Edge-IIoTset remains secondary robustness evidence under its recorded protocol; no new Edge-IIoTset training is part of this completion cycle.

## Frozen Primary Contract

| Item | Contract |
|---|---|
| Dataset | WSN-DS, SHA-256 `c65d05b983a85753bd62b6f76c5739fc52fe0c14cbb7644255cee4742f5ff7c9` |
| Inputs | 17 recorded tabular features |
| Primary split | Seed-42 exact-feature-group-disjoint split |
| Primary split sizes | 262,197 train, 56,163 validation, 56,301 test |
| Preprocessing | `StandardScaler` fit on train only |
| Model seeds | 42, 123, 456, 789, 1001, 2024, 3141, 5678, 8192, 9999 |
| Students | A: 17-32-16-5; B: 17-64-32-5 |
| Core routes | Matched scratch and calibrated RF-KD |
| KD settings | Historical fixed values `T=4`, `alpha=0.7`; no test retuning |
| Primary statistical unit | Paired optimizer seed on one fixed split |
| Hardware specimen seed | 42, fixed a priori; not selected by test performance |

## Execution Order

### 1. Response-Transfer Evidence

Status: complete.

- Read the finalized ten-seed prediction artifacts without refitting models.
- Primary transfer metric: exact-group-balanced `KL(RF_T4 || student_T4)`.
- Compare RF-KD with its matched scratch route within each optimizer seed.
- Apply Holm correction across Student A and Student B.
- Keep row-weighted and per-class results as sensitivity analyses.
- Do not interpret this as mechanism, explanation, or off-manifold boundary equivalence.

Output: `results/wsnds/evidence_completion_20260812/fgds_behavioral_transfer_logits_10seed_v5/`

### 2. Split-Robustness Confirmation

Status: complete under protocol `wsnds_fgds_multisplit_core_10x2_v2`.

- Generate ten distinct exact-feature-group-disjoint holdout assignments using the publication seed list as split seeds.
- Hold the RF seed fixed at 42 across splits.
- Hold student optimizer seeds fixed at 42 and 123 across splits.
- Fit the scaler and calibrated RF inside each training partition.
- Train only scratch and RF-KD for both students.
- Average the two paired optimizer-seed differences within each split.
- Report the ten split-level means as descriptive split-sensitivity evidence.
- Do not treat overlapping repeated holdouts from one dataset as independent inferential units and do not calculate multi-split hypothesis-test p-values.
- Retain split indices, true labels, predicted labels, metrics, hashes, and manifests. Do not retain RF objects, training probabilities, copied data, or deployment checkpoints.

Output: `results/wsnds/evidence_completion_20260812/fgds_multisplit_core_10x2_v2/`

The incomplete v1 contract was stopped after its first split when dependency-sealing and semantic-verification gaps were found. It is retained only as `fgds_multisplit_core_10x2_v1_aborted_unsealed_20260812T2113` and is excluded from evidence synthesis.

### 3. Controlled XAI and Boundary Audit

Status: not executed. The existing seed-42 reconstructed-teacher SHAP audit is
valid single-specimen evidence, but it does not satisfy this ten-seed contract.

- Use the finalized ten-seed checkpoints, not the historical five-seed deployment checkpoints.
- Reconstruct each calibrated RF deterministically and require exact agreement with preserved train and test probabilities before explanation.
- Use one fixed 250-record cohort containing 50 non-conflicting unique exact feature groups per class, selected without replacement with RNG seed `2042`, and one fixed 20-record training background containing four such groups per class with RNG seed `1042`. Use the lowest source-row index as each group's representative and interleave classes `0,1,2,3,4`.
- Explain all five `T=4` probability outputs for the RF, scratch A/B, and RF-KD A/B with the same exact-invariance Independent masker, permutation seed `3042`, five permutation repeats, and `1e-6` local-accuracy tolerance.
- For each record, compare the 17 signed attributions for the RF-predicted output class. The primary value is RF-KD-to-RF cosine minus scratch-to-RF cosine. Average within true class and then across the five classes to obtain one paired value per model seed.
- Freeze the near-zero attribution L2 threshold at `1e-6` probability units. A record is eligible only when RF, scratch, and RF-KD norms all exceed it; never replace undefined cosines with zero. Mark a seed inconclusive if any class retains fewer than 40 of 50 records.
- Secondary XAI outputs: rank correlation, top-feature overlap, thresholded sign agreement, global absolute-importance rank, local-accuracy residual, and class-conditional summaries.
- Run a descriptive seed-42 full-parameter-reinitialization sanity control for Student A with seed `5042` and Student B with seed `6042`, using the identical cohort, background, selected outputs, and SHAP estimator.
- Treat model seed as the inferential unit. Use exact two-sided paired Wilcoxon signed-rank tests with average ranks, Wilcox zero handling, and explicit sign enumeration; Holm-adjust only the two primary Student A/B tests.
- Do not perform synthetic feature interpolation or label-randomization retraining in this completion cycle.

### 4. Software Fixed-Point Audit

Status: complete with retained failures. Of 40 model-seed instances, 26 passed
all software gates and 14 failed at least one frozen gate. Failed instances were
retained and were not advanced to native C equivalence.

- Quantize all ten seeds for scratch A/B and RF-KD A/B in memory using one frozen procedure and arithmetic contract. Calibration values are learned per model from the frozen non-test calibration partition; one numeric scale is not shared across different model states.
- Persist compact per-seed metrics and gate outcomes only.
- Check input and activation saturation, accumulator bounds, fixed versus FP32 agreement, macro-F1 change, class-wise change, and exact C versus Python fixed-reference equivalence.
- Freeze the acceptance gates before execution: fixed-versus-FP32 prediction agreement must be at least `0.99`, absolute macro-F1 degradation must not exceed `0.015`, all audited input/activation/weight/bias saturation counts must be zero, and accumulator bounds must remain within the implemented integer contract. These gates are deployment checks, not model-selection objectives; retain and report every result whether it passes or fails.
- Generate full portable artifacts only for the four final seed-42 variants.
- Never select a model or quantization setting using test results.

### 5. Final Hardware Campaign

Status: USB complete for all six eligible model-board combinations. The six
final-lineage Wi-Fi combinations have not been executed. The older completed
Wi-Fi campaign belongs to a distinct five-seed deployment lineage and cannot
substitute for this work.

Intended model variants:

1. Student A scratch, seed 42
2. Student A RF-KD, seed 42
3. Student B scratch, seed 42
4. Student B RF-KD, seed 42

The immutable contract retains all four variants. The frozen fixed-point gate
passed Student A scratch, Student A RF-KD, and Student B RF-KD. Student B
scratch is excluded before hardware execution because fixed-versus-FP32
agreement is `0.989574` (required: at least `0.99`) and the absolute macro-F1
drop is `0.020193` (allowed: at most `0.015`). This leaves 12 gate-eligible
board-model-transport combinations and four explicit exclusions.

Required targets and transports:

| Target | USB serial | Wi-Fi UDP |
|---|---:|---:|
| ESP32-C3 | Yes | Yes |
| Arduino UNO R4 WiFi | Yes | Yes |

Run policy:

- Record USB serial numbers, firmware hashes, export IDs, bundle IDs, host revision, environment, and session ID before every run.
- Include student, route, seed, model hash, export ID, bundle ID, board, transport, and protocol ID in every device identity.
- Run targets sequentially for timing evidence to prevent host and network contention.
- Counterbalance model and transport order across targets.
- Build one deterministic timing cohort from 200 non-conflicting unique exact feature groups per class, with one representative row per group and interleaved class order.
- For every board-model-transport combination: boot identity check, 10-vector untimed warm-up, 10-vector smoke, three repeated balanced 1,000-vector timing sessions, and one complete test replay.
- Full fidelity scope: `2 boards x 3 eligible models x 2 transports x 56,301 = 675,612` test-record executions.
- Timing-repeat scope: `2 x 3 x 2 x 3 x 1,000 = 36,000` additional executions.
- Require zero missing, duplicate, unexpected, or non-OK sequence records.
- Require exact MCU versus fixed-reference prediction and logit agreement.
- Report on-device preprocessing, inference, and total compute separately from host round-trip and Wi-Fi transport timing.
- Define inference timing identically on USB and Wi-Fi as fixed-point forward pass plus argmax. Exclude parsing, formatting, transport, and host work.
- Retain full per-row MCU CSVs only until final verification. Archive one compressed canonical copy per full session and reference it by hash from reports.

Energy measurement is conditional on a named, calibrated measurement instrument. Record instrument model, sampling rate, resolution, supply voltage, firmware state, idle window, active window, repetitions, and uncertainty. Without that record, report no energy claim.

### 6. Final Integration

Status: current evidence registry prepared with open planned work recorded.
Manuscript integration remains separate.

- Build one new immutable evidence registry after all software and hardware gates pass.
- Reference existing evidence by path and SHA-256; do not copy historical result trees.
- Correct active claim language to "no statistically detectable difference" where null tests do not reject.
- Keep historical five-seed hardware and XAI lineages explicitly separate from the final ten-seed lineage.
- Update README, result tables, figures, and manuscript only from the final registry.
- Compile and visually inspect the manuscript at desktop and print scale.
- Run numerical cross-checks, source-to-table traceability, statistical review, hardware-lineage review, and claim-boundary review before submission.

## Artifact Policy

- No copied datasets.
- No repository-local replacement environment.
- No overwrite of completed output directories.
- No copied historical result tree in a new registry.
- No per-row probability duplication when compact statistics are sufficient.
- No retained temporary RF reconstruction or explainer cache after its manifest-backed summary is sealed.
- No generated previews, caches, executables, or logs added to Git unless they are required evidence.

## Stop Conditions

Stop a stage immediately if any dataset, split, scaler, model, source, export, firmware, or session hash differs from its execution contract; if a partition overlaps by exact raw-feature group; if an expected paired route is missing; if a probability or fixed-point arithmetic gate fails; or if a hardware sequence contains a missing, duplicate, unexpected, or non-OK record. Preserve the failed attempt separately and do not merge it into final evidence.
