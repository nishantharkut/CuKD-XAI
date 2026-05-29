# CuKD-XAI v2.3 Co-distillation Add-on Guide

Created: 2026-05-29

Use this file pair for the strict v2.3 add-on experiment:

- `cukd_xai_v23_codistill_pilot.py`
- `cukd_xai_v23_codistill_pilot.ipynb`

This file is meant to be a near-copy of `cukd_xai_colab.py`, with only the useful co-distillation add-on added and the weak/unneeded configs removed from the execution path.

## Diff Contract

Cells 1-8 match `cukd_xai_colab.py` exactly. The add-on starts at `CELL 8B`, where the co-distillation helper is added. The original `run_all_configs`, multi-seed loop, Student B sweep, Wilcoxon, SHAP, quantization, latency, figures, JSON export, CSV export, and final summary structure are preserved.

Removed from this codistill route:

- domain-difficulty CL `C2_CL_MLP_domain`
- extended CL teacher `C_CL_MLP_loss_ext`
- extended-F student `F_KD_from_CL_MLP_ext`
- random pacing `G_KD_random_pacing`
- SMOTE teacher/student `I_KD_from_SMOTE_MLP`

Kept and run:

- `A_RF_500`
- `B_Full_MLP`
- `C_CL_MLP_loss_fair`
- `D_Small_MLP`
- `E_KD_from_RF`
- `E2_KD_from_MLP`
- `F_KD_from_CL_MLP_fair`
- `F_KD_from_CL_MLP` alias
- `J_CoDistill_RF_CL`

Student A and Student B both run through the kept configs.

## Co-distillation Algorithm

`J_CoDistill_RF_CL` trains one student with three loss terms:

```text
loss =
  CE_weight * CrossEntropy(student, labels)
  + RF_weight * KL(student/T, RF_teacher/T)
  + CL_weight * KL(student/T, curriculum_teacher/T)
```

Defaults:

```python
CODISTILL_CE_WEIGHT = 0.30
CODISTILL_RF_WEIGHT = 0.40
CODISTILL_CL_WEIGHT = 0.30
CODISTILL_EPOCHS = 40
CODISTILL_LR = 7e-4
CODISTILL_PATIENCE = 10
```

The KD temperature `T` comes from the same `BEST_T`/`kd_T` route used by v2.3 KD configs. RF probabilities reuse the same calibrated RF predict_proba soft labels used by v2.3 Config E, then are temperature-softened through pseudo-logits. The CL teacher uses logits directly with temperature softmax. KL terms use the standard `T*T` multiplier; this is intentionally retained from v2.3 `train_kd` because it corrects the gradient scale introduced by temperature softening, while the normalized weights control the CE-vs-KD balance. Config J alone uses a slightly slower schedule, 40 epochs at lr=7e-4 with patience 10, to reduce under-training risk from two teacher signals.

## Outputs

The add-on preserves the original v2.3 output style:

- `cukd_xai_results.json`
- `wsnds_results_student_A.csv`
- `wsnds_results_student_B.csv`
- `per_class_f1.png`
- `confusion_matrix_E.png`
- `confusion_matrix_F.png`
- `confusion_matrix_F_fair.png`
- `confusion_matrix_J_codistill.png`
- `pareto_frontier.png`
- `shap_summary_student.png`
- `loss_curves_B_vs_C.png`

SHAP uses `J_CoDistill_RF_CL` as the student and `A_RF_500` as the RF reference. Quantization and benchmark sweeps include `J_CoDistill_RF_CL`.

## Runtime Expectation

This route removes weak configs but keeps Student A, Student B, SHAP, quantization, latency, figures, and statistics. On the RTX 1000 Ada plus i9-13950HX system, expect roughly 7-11 hours for 3 seeds with normal RAM headroom, and 11-15 hours if available RAM remains very low or CPU contention is high.

## Decision Rule

Continue beyond 3 seeds only if `J_CoDistill_RF_CL` improves over the best of `E`, `E2`, and `F` by about `0.003` macro-F1 or more, without a per-class collapse.
