# Edge-IIoT v2.3 Generalization Guide

This is the corrected secondary-dataset route for CuKD-XAI. It uses the same v2.3 model/training/KD logic from `cukd_xai_colab.py`; only the Edge-IIoT adapter, run orchestration, and result export are dataset-specific.

## Files

- `cukd_xai_edgeiiot_v23_generalization.py`: source file to inspect/edit.
- `cukd_xai_edgeiiot_v23_generalization.ipynb`: notebook generated from the source file.
- `edgeiiot_v23_generalization_outputs/edgeiiot_v23_results.json`: full output after the run.
- `edgeiiot_v23_generalization_outputs/edgeiiot_v23_results_student_A.csv`: aggregated results for Student A.
- `edgeiiot_v23_generalization_outputs/edgeiiot_v23_results_student_B.csv`: aggregated results for Student B.
- `edgeiiot_v23_generalization_outputs/edgeiiot_v23_metadata.json`: dataset and run metadata.
- `edgeiiot_v23_generalization_outputs/edgeiiot_v23_config_rankings.csv`: config ranking by mean Macro F1, then accuracy.

## What It Runs

Run All executes 5 seeds by default: `[42, 123, 456, 789, 1001]`.

For each seed, it runs the v2.3-comparable configs:

- `A_RF_500`: Random Forest teacher, 500 trees, max depth 15.
- `B_Full_MLP`: v2.3 full MLP teacher.
- `C_CL_MLP_loss_fair`: v2.3 loss-based curriculum teacher with the fair 30-epoch schedule.
- `D_Small_MLP`: plain student.
- `E_KD_from_RF`: student distilled from calibrated RF probabilities.
- `E2_KD_from_MLP`: student distilled from the full MLP teacher.
- `F_KD_from_CL_MLP`: student distilled from the curriculum MLP teacher.
- `J_CoDistill_RF_CL`: student trained with hard labels plus calibrated RF and curriculum-MLP soft targets.

Both students are included:

- `student_A_32_16`
- `student_B_64_32`

KD uses `T=2, alpha=0.5`, matching the completed non-quick v2.3 WSN-DS runs. Config `J_CoDistill_RF_CL` uses the same co-distillation schedule as the WSN pilot: CE/RF/CL weights `0.30/0.40/0.30`, 40 epochs, learning rate `7e-4`, and patience 10. The Edge-IIoT secondary run is therefore a transfer/generalization check, not a separate hyperparameter tuning exercise.

## Edge-IIoT Adapter

The Edge-IIoT adapter removes identifier/source/payload leakage columns, keeps `Attack_type` as the multiclass target, performs stratified train/validation/test splitting, fits categorical policy and numeric imputation only on the training split, and scales only continuous columns. Dummy columns are not standardized.

This is intentionally not a SHAP, Wilcoxon, QAT, or deployment-proof file. Those belong to the WSN-DS primary route and deployment evidence, not the secondary generalization run. Config J is included only as the selected co-distillation generalization check.

## How To Run

1. Put `ML-EdgeIIoT-dataset.csv` beside the notebook, keep the original extracted `Edge-IIoTset dataset/Selected dataset for ML and DL/` folder structure, or set `EDGEIIOT_ML_PATH` to the CSV path before Run All.
2. Open `cukd_xai_edgeiiot_v23_generalization.ipynb` in Jupyter.
3. Clear kernel and all outputs.
4. Click Run All.
5. Wait for `Edge-IIoT v2.3 generalization complete.` at the end.
6. Read `edgeiiot_v23_results.json`, the two student CSVs, and `edgeiiot_v23_config_rankings.csv` from `edgeiiot_v23_generalization_outputs/`.

## Time Estimate

On a laptop-class workstation, expect roughly 10-20+ hours for the final 5 seeds because each seed trains RF, calibrated RF, full MLP, curriculum MLP, and five student runs across two student sizes. If available RAM is low, runtime can increase because the calibrated RF and encoded Edge-IIoT feature matrix are CPU/RAM-heavy.

For a syntax/data-loader smoke check only, set `EDGEIIOT_RUN_MODE = 'edgeiiot_quick'`; final paper evidence should use the default 5 seeds.

## Defensible Claim

Use this file to claim that CuKD-XAI was tested beyond WSN-DS using Edge-IIoTset while preserving the same v2.3 model/training/KD logic. Report the mean/std metrics, per-class F1, class names, feature count, leakage columns removed, seed list, and J-vs-E2/F/D deltas from `edgeiiot_v23_results.json` and `edgeiiot_v23_metadata.json`.
