# WSN-DS Group-Balanced Route Sensitivity

## Scope

The analysis validates and post-processes 56,301 rows from one fixed feature-group-disjoint test split across ten paired algorithmic seeds. No model is fitted.

Primary inference uses row-level metrics, exhaustive paired Wilcoxon tests, and Holm adjustment within the predeclared teacher, Student A, or Student B route family. View-global Holm values and the two group-balanced views are stricter multiplicity and repeated-pattern sensitivity analyses.

The exact p-values reported here supersede the source runner's approximate Wilcoxon fields. Model artifacts, predictions, and performance metrics are unchanged.

## Test-Pattern Accounting

| Quantity | Count |
|---|---:|
| Exact feature groups | 54,174 |
| Singleton groups | 52,675 |
| Repeated groups | 1,499 |
| Rows in repeated groups | 3,626 |
| Mixed-label groups | 0 |
| Mixed-label rows | 0 |
| Label-pure representative rows | 54,174 |

The inverse-size view retains mixed-label rows with their recorded labels. The representative view excludes every mixed-label group and row. Neither view assigns a majority label.

## Route Macro-F1

| View | Category | Student | Route | Mean | Sample SD |
|---|---|---|---|---:|---:|
| inverse_test_group_size | student | student_A | D_Small_MLP | 0.914092 | 0.005455 |
| inverse_test_group_size | student | student_A | E2_KD_from_MLP | 0.914408 | 0.002340 |
| inverse_test_group_size | student | student_A | E_KD_from_RF | 0.915301 | 0.002227 |
| inverse_test_group_size | student | student_A | F_KD_from_CL_MLP_ext | 0.912826 | 0.002650 |
| inverse_test_group_size | student | student_A | F_KD_from_CL_MLP_fair | 0.914959 | 0.002402 |
| inverse_test_group_size | student | student_A | G_KD_random_pacing | 0.915404 | 0.002921 |
| inverse_test_group_size | student | student_A | I_KD_from_SMOTE_MLP | 0.913227 | 0.005531 |
| inverse_test_group_size | student | student_A | J_CoDistill_RF_CL | 0.917861 | 0.005908 |
| inverse_test_group_size | student | student_B | D_Small_MLP | 0.931977 | 0.005612 |
| inverse_test_group_size | student | student_B | E2_KD_from_MLP | 0.921584 | 0.002877 |
| inverse_test_group_size | student | student_B | E_KD_from_RF | 0.931773 | 0.010206 |
| inverse_test_group_size | student | student_B | F_KD_from_CL_MLP_ext | 0.921019 | 0.004116 |
| inverse_test_group_size | student | student_B | F_KD_from_CL_MLP_fair | 0.919470 | 0.004908 |
| inverse_test_group_size | student | student_B | G_KD_random_pacing | 0.921005 | 0.004161 |
| inverse_test_group_size | student | student_B | I_KD_from_SMOTE_MLP | 0.919674 | 0.004508 |
| inverse_test_group_size | student | student_B | J_CoDistill_RF_CL | 0.926796 | 0.007437 |
| inverse_test_group_size | teacher | - | A_RF_500_uncalibrated | 0.978528 | 0.000361 |
| inverse_test_group_size | teacher | - | A_calibrated_RF_KD_teacher | 0.978439 | 0.000430 |
| inverse_test_group_size | teacher | - | B_Full_MLP | 0.921388 | 0.002011 |
| inverse_test_group_size | teacher | - | C2_CL_MLP_domain | 0.914801 | 0.002295 |
| inverse_test_group_size | teacher | - | C_CL_MLP_loss_ext | 0.918458 | 0.002809 |
| inverse_test_group_size | teacher | - | C_CL_MLP_loss_fair | 0.914307 | 0.003150 |
| inverse_test_group_size | teacher | - | G_random_pacing_teacher | 0.919568 | 0.003831 |
| inverse_test_group_size | teacher | - | I_SMOTE_MLP_teacher | 0.949913 | 0.009569 |
| pure_group_representative | student | student_A | D_Small_MLP | 0.914092 | 0.005455 |
| pure_group_representative | student | student_A | E2_KD_from_MLP | 0.914408 | 0.002340 |
| pure_group_representative | student | student_A | E_KD_from_RF | 0.915301 | 0.002227 |
| pure_group_representative | student | student_A | F_KD_from_CL_MLP_ext | 0.912826 | 0.002650 |
| pure_group_representative | student | student_A | F_KD_from_CL_MLP_fair | 0.914959 | 0.002402 |
| pure_group_representative | student | student_A | G_KD_random_pacing | 0.915404 | 0.002921 |
| pure_group_representative | student | student_A | I_KD_from_SMOTE_MLP | 0.913227 | 0.005531 |
| pure_group_representative | student | student_A | J_CoDistill_RF_CL | 0.917861 | 0.005908 |
| pure_group_representative | student | student_B | D_Small_MLP | 0.931977 | 0.005612 |
| pure_group_representative | student | student_B | E2_KD_from_MLP | 0.921584 | 0.002877 |
| pure_group_representative | student | student_B | E_KD_from_RF | 0.931773 | 0.010206 |
| pure_group_representative | student | student_B | F_KD_from_CL_MLP_ext | 0.921019 | 0.004116 |
| pure_group_representative | student | student_B | F_KD_from_CL_MLP_fair | 0.919470 | 0.004908 |
| pure_group_representative | student | student_B | G_KD_random_pacing | 0.921005 | 0.004161 |
| pure_group_representative | student | student_B | I_KD_from_SMOTE_MLP | 0.919674 | 0.004508 |
| pure_group_representative | student | student_B | J_CoDistill_RF_CL | 0.926796 | 0.007437 |
| pure_group_representative | teacher | - | A_RF_500_uncalibrated | 0.978528 | 0.000361 |
| pure_group_representative | teacher | - | A_calibrated_RF_KD_teacher | 0.978439 | 0.000430 |
| pure_group_representative | teacher | - | B_Full_MLP | 0.921388 | 0.002011 |
| pure_group_representative | teacher | - | C2_CL_MLP_domain | 0.914801 | 0.002295 |
| pure_group_representative | teacher | - | C_CL_MLP_loss_ext | 0.918458 | 0.002809 |
| pure_group_representative | teacher | - | C_CL_MLP_loss_fair | 0.914307 | 0.003150 |
| pure_group_representative | teacher | - | G_random_pacing_teacher | 0.919568 | 0.003831 |
| pure_group_representative | teacher | - | I_SMOTE_MLP_teacher | 0.949913 | 0.009569 |
| row_level | student | student_A | D_Small_MLP | 0.914792 | 0.005658 |
| row_level | student | student_A | E2_KD_from_MLP | 0.914321 | 0.003204 |
| row_level | student | student_A | E_KD_from_RF | 0.913781 | 0.004546 |
| row_level | student | student_A | F_KD_from_CL_MLP_ext | 0.912296 | 0.003330 |
| row_level | student | student_A | F_KD_from_CL_MLP_fair | 0.915357 | 0.003084 |
| row_level | student | student_A | G_KD_random_pacing | 0.914970 | 0.003614 |
| row_level | student | student_A | I_KD_from_SMOTE_MLP | 0.911577 | 0.006552 |
| row_level | student | student_A | J_CoDistill_RF_CL | 0.917463 | 0.006553 |
| row_level | student | student_B | D_Small_MLP | 0.932867 | 0.005727 |
| row_level | student | student_B | E2_KD_from_MLP | 0.922492 | 0.003731 |
| row_level | student | student_B | E_KD_from_RF | 0.932142 | 0.010930 |
| row_level | student | student_B | F_KD_from_CL_MLP_ext | 0.922113 | 0.005211 |
| row_level | student | student_B | F_KD_from_CL_MLP_fair | 0.920344 | 0.005458 |
| row_level | student | student_B | G_KD_random_pacing | 0.921007 | 0.003971 |
| row_level | student | student_B | I_KD_from_SMOTE_MLP | 0.919619 | 0.005092 |
| row_level | student | student_B | J_CoDistill_RF_CL | 0.927575 | 0.007727 |
| row_level | teacher | - | A_RF_500_uncalibrated | 0.979477 | 0.000277 |
| row_level | teacher | - | A_calibrated_RF_KD_teacher | 0.979500 | 0.000489 |
| row_level | teacher | - | B_Full_MLP | 0.921951 | 0.001703 |
| row_level | teacher | - | C2_CL_MLP_domain | 0.915146 | 0.002516 |
| row_level | teacher | - | C_CL_MLP_loss_ext | 0.919417 | 0.003050 |
| row_level | teacher | - | C_CL_MLP_loss_fair | 0.915163 | 0.003085 |
| row_level | teacher | - | G_random_pacing_teacher | 0.920555 | 0.003991 |
| row_level | teacher | - | I_SMOTE_MLP_teacher | 0.951188 | 0.010474 |

## Paired Route Tests

| View | Family | Contrast | Mean delta | Exact signed-rank p | Family Holm p | View-global Holm p |
|---|---|---|---:|---:|---:|---:|
| inverse_test_group_size | student_A | E2_KD_from_MLP - D_Small_MLP | 0.000316 | 1.000000 | 1.000000 | 1.000000 |
| inverse_test_group_size | student_A | E_KD_from_RF - D_Small_MLP | 0.001209 | 0.431641 | 1.000000 | 1.000000 |
| inverse_test_group_size | student_A | E_KD_from_RF - E2_KD_from_MLP | 0.000893 | 0.492188 | 1.000000 | 1.000000 |
| inverse_test_group_size | student_A | F_KD_from_CL_MLP_ext - E2_KD_from_MLP | -0.001582 | 0.160156 | 1.000000 | 1.000000 |
| inverse_test_group_size | student_A | F_KD_from_CL_MLP_fair - D_Small_MLP | 0.000867 | 0.769531 | 1.000000 | 1.000000 |
| inverse_test_group_size | student_A | F_KD_from_CL_MLP_fair - E2_KD_from_MLP | 0.000551 | 0.625000 | 1.000000 | 1.000000 |
| inverse_test_group_size | student_A | F_KD_from_CL_MLP_fair - G_KD_random_pacing | -0.000445 | 0.845703 | 1.000000 | 1.000000 |
| inverse_test_group_size | student_A | F_KD_from_CL_MLP_fair - I_KD_from_SMOTE_MLP | 0.001731 | 0.322266 | 1.000000 | 1.000000 |
| inverse_test_group_size | student_A | I_KD_from_SMOTE_MLP - E2_KD_from_MLP | -0.001181 | 0.492188 | 1.000000 | 1.000000 |
| inverse_test_group_size | student_A | J_CoDistill_RF_CL - E2_KD_from_MLP | 0.003453 | 0.130859 | 1.000000 | 1.000000 |
| inverse_test_group_size | student_A | J_CoDistill_RF_CL - E_KD_from_RF | 0.002560 | 0.105469 | 1.000000 | 1.000000 |
| inverse_test_group_size | student_A | J_CoDistill_RF_CL - F_KD_from_CL_MLP_fair | 0.002902 | 0.322266 | 1.000000 | 1.000000 |
| inverse_test_group_size | student_B | E2_KD_from_MLP - D_Small_MLP | -0.010394 | 0.001953 | 0.023438 | 0.050781 |
| inverse_test_group_size | student_B | E_KD_from_RF - D_Small_MLP | -0.000204 | 1.000000 | 1.000000 | 1.000000 |
| inverse_test_group_size | student_B | E_KD_from_RF - E2_KD_from_MLP | 0.010190 | 0.037109 | 0.333984 | 0.779297 |
| inverse_test_group_size | student_B | F_KD_from_CL_MLP_ext - E2_KD_from_MLP | -0.000564 | 0.556641 | 1.000000 | 1.000000 |
| inverse_test_group_size | student_B | F_KD_from_CL_MLP_fair - D_Small_MLP | -0.012507 | 0.001953 | 0.023438 | 0.050781 |
| inverse_test_group_size | student_B | F_KD_from_CL_MLP_fair - E2_KD_from_MLP | -0.002113 | 0.556641 | 1.000000 | 1.000000 |
| inverse_test_group_size | student_B | F_KD_from_CL_MLP_fair - G_KD_random_pacing | -0.001535 | 0.322266 | 1.000000 | 1.000000 |
| inverse_test_group_size | student_B | F_KD_from_CL_MLP_fair - I_KD_from_SMOTE_MLP | -0.000204 | 0.921875 | 1.000000 | 1.000000 |
| inverse_test_group_size | student_B | I_KD_from_SMOTE_MLP - E2_KD_from_MLP | -0.001909 | 0.375000 | 1.000000 | 1.000000 |
| inverse_test_group_size | student_B | J_CoDistill_RF_CL - E2_KD_from_MLP | 0.005213 | 0.048828 | 0.390625 | 0.976562 |
| inverse_test_group_size | student_B | J_CoDistill_RF_CL - E_KD_from_RF | -0.004977 | 0.193359 | 1.000000 | 1.000000 |
| inverse_test_group_size | student_B | J_CoDistill_RF_CL - F_KD_from_CL_MLP_fair | 0.007326 | 0.019531 | 0.195312 | 0.429688 |
| inverse_test_group_size | teacher | C_CL_MLP_loss_ext - B_Full_MLP | -0.002930 | 0.009766 | 0.009766 | 0.224609 |
| inverse_test_group_size | teacher | C_CL_MLP_loss_fair - B_Full_MLP | -0.007081 | 0.001953 | 0.003906 | 0.050781 |
| pure_group_representative | student_A | E2_KD_from_MLP - D_Small_MLP | 0.000316 | 1.000000 | 1.000000 | 1.000000 |
| pure_group_representative | student_A | E_KD_from_RF - D_Small_MLP | 0.001209 | 0.431641 | 1.000000 | 1.000000 |
| pure_group_representative | student_A | E_KD_from_RF - E2_KD_from_MLP | 0.000893 | 0.492188 | 1.000000 | 1.000000 |
| pure_group_representative | student_A | F_KD_from_CL_MLP_ext - E2_KD_from_MLP | -0.001582 | 0.160156 | 1.000000 | 1.000000 |
| pure_group_representative | student_A | F_KD_from_CL_MLP_fair - D_Small_MLP | 0.000867 | 0.769531 | 1.000000 | 1.000000 |
| pure_group_representative | student_A | F_KD_from_CL_MLP_fair - E2_KD_from_MLP | 0.000551 | 0.625000 | 1.000000 | 1.000000 |
| pure_group_representative | student_A | F_KD_from_CL_MLP_fair - G_KD_random_pacing | -0.000445 | 0.845703 | 1.000000 | 1.000000 |
| pure_group_representative | student_A | F_KD_from_CL_MLP_fair - I_KD_from_SMOTE_MLP | 0.001731 | 0.322266 | 1.000000 | 1.000000 |
| pure_group_representative | student_A | I_KD_from_SMOTE_MLP - E2_KD_from_MLP | -0.001181 | 0.492188 | 1.000000 | 1.000000 |
| pure_group_representative | student_A | J_CoDistill_RF_CL - E2_KD_from_MLP | 0.003453 | 0.130859 | 1.000000 | 1.000000 |
| pure_group_representative | student_A | J_CoDistill_RF_CL - E_KD_from_RF | 0.002560 | 0.105469 | 1.000000 | 1.000000 |
| pure_group_representative | student_A | J_CoDistill_RF_CL - F_KD_from_CL_MLP_fair | 0.002902 | 0.322266 | 1.000000 | 1.000000 |
| pure_group_representative | student_B | E2_KD_from_MLP - D_Small_MLP | -0.010394 | 0.001953 | 0.023438 | 0.050781 |
| pure_group_representative | student_B | E_KD_from_RF - D_Small_MLP | -0.000204 | 1.000000 | 1.000000 | 1.000000 |
| pure_group_representative | student_B | E_KD_from_RF - E2_KD_from_MLP | 0.010190 | 0.037109 | 0.333984 | 0.779297 |
| pure_group_representative | student_B | F_KD_from_CL_MLP_ext - E2_KD_from_MLP | -0.000564 | 0.556641 | 1.000000 | 1.000000 |
| pure_group_representative | student_B | F_KD_from_CL_MLP_fair - D_Small_MLP | -0.012507 | 0.001953 | 0.023438 | 0.050781 |
| pure_group_representative | student_B | F_KD_from_CL_MLP_fair - E2_KD_from_MLP | -0.002113 | 0.556641 | 1.000000 | 1.000000 |
| pure_group_representative | student_B | F_KD_from_CL_MLP_fair - G_KD_random_pacing | -0.001535 | 0.322266 | 1.000000 | 1.000000 |
| pure_group_representative | student_B | F_KD_from_CL_MLP_fair - I_KD_from_SMOTE_MLP | -0.000204 | 0.921875 | 1.000000 | 1.000000 |
| pure_group_representative | student_B | I_KD_from_SMOTE_MLP - E2_KD_from_MLP | -0.001909 | 0.375000 | 1.000000 | 1.000000 |
| pure_group_representative | student_B | J_CoDistill_RF_CL - E2_KD_from_MLP | 0.005213 | 0.048828 | 0.390625 | 0.976562 |
| pure_group_representative | student_B | J_CoDistill_RF_CL - E_KD_from_RF | -0.004977 | 0.193359 | 1.000000 | 1.000000 |
| pure_group_representative | student_B | J_CoDistill_RF_CL - F_KD_from_CL_MLP_fair | 0.007326 | 0.019531 | 0.195312 | 0.429688 |
| pure_group_representative | teacher | C_CL_MLP_loss_ext - B_Full_MLP | -0.002930 | 0.009766 | 0.009766 | 0.224609 |
| pure_group_representative | teacher | C_CL_MLP_loss_fair - B_Full_MLP | -0.007081 | 0.001953 | 0.003906 | 0.050781 |
| row_level | student_A | E2_KD_from_MLP - D_Small_MLP | -0.000471 | 0.921875 | 1.000000 | 1.000000 |
| row_level | student_A | E_KD_from_RF - D_Small_MLP | -0.001012 | 0.556641 | 1.000000 | 1.000000 |
| row_level | student_A | E_KD_from_RF - E2_KD_from_MLP | -0.000540 | 0.845703 | 1.000000 | 1.000000 |
| row_level | student_A | F_KD_from_CL_MLP_ext - E2_KD_from_MLP | -0.002025 | 0.193359 | 1.000000 | 1.000000 |
| row_level | student_A | F_KD_from_CL_MLP_fair - D_Small_MLP | 0.000565 | 0.845703 | 1.000000 | 1.000000 |
| row_level | student_A | F_KD_from_CL_MLP_fair - E2_KD_from_MLP | 0.001036 | 0.492188 | 1.000000 | 1.000000 |
| row_level | student_A | F_KD_from_CL_MLP_fair - G_KD_random_pacing | 0.000387 | 0.769531 | 1.000000 | 1.000000 |
| row_level | student_A | F_KD_from_CL_MLP_fair - I_KD_from_SMOTE_MLP | 0.003780 | 0.160156 | 1.000000 | 1.000000 |
| row_level | student_A | I_KD_from_SMOTE_MLP - E2_KD_from_MLP | -0.002744 | 0.275391 | 1.000000 | 1.000000 |
| row_level | student_A | J_CoDistill_RF_CL - E2_KD_from_MLP | 0.003142 | 0.275391 | 1.000000 | 1.000000 |
| row_level | student_A | J_CoDistill_RF_CL - E_KD_from_RF | 0.003682 | 0.105469 | 1.000000 | 1.000000 |
| row_level | student_A | J_CoDistill_RF_CL - F_KD_from_CL_MLP_fair | 0.002106 | 0.625000 | 1.000000 | 1.000000 |
| row_level | student_B | E2_KD_from_MLP - D_Small_MLP | -0.010375 | 0.001953 | 0.023438 | 0.050781 |
| row_level | student_B | E_KD_from_RF - D_Small_MLP | -0.000725 | 0.845703 | 1.000000 | 1.000000 |
| row_level | student_B | E_KD_from_RF - E2_KD_from_MLP | 0.009650 | 0.037109 | 0.371094 | 0.816406 |
| row_level | student_B | F_KD_from_CL_MLP_ext - E2_KD_from_MLP | -0.000380 | 0.625000 | 1.000000 | 1.000000 |
| row_level | student_B | F_KD_from_CL_MLP_fair - D_Small_MLP | -0.012523 | 0.001953 | 0.023438 | 0.050781 |
| row_level | student_B | F_KD_from_CL_MLP_fair - E2_KD_from_MLP | -0.002148 | 0.275391 | 1.000000 | 1.000000 |
| row_level | student_B | F_KD_from_CL_MLP_fair - G_KD_random_pacing | -0.000663 | 0.769531 | 1.000000 | 1.000000 |
| row_level | student_B | F_KD_from_CL_MLP_fair - I_KD_from_SMOTE_MLP | 0.000725 | 0.921875 | 1.000000 | 1.000000 |
| row_level | student_B | I_KD_from_SMOTE_MLP - E2_KD_from_MLP | -0.002873 | 0.232422 | 1.000000 | 1.000000 |
| row_level | student_B | J_CoDistill_RF_CL - E2_KD_from_MLP | 0.005083 | 0.083984 | 0.671875 | 1.000000 |
| row_level | student_B | J_CoDistill_RF_CL - E_KD_from_RF | -0.004567 | 0.193359 | 1.000000 | 1.000000 |
| row_level | student_B | J_CoDistill_RF_CL - F_KD_from_CL_MLP_fair | 0.007231 | 0.037109 | 0.371094 | 0.816406 |
| row_level | teacher | C_CL_MLP_loss_ext - B_Full_MLP | -0.002534 | 0.019531 | 0.019531 | 0.449219 |
| row_level | teacher | C_CL_MLP_loss_fair - B_Full_MLP | -0.006789 | 0.001953 | 0.003906 | 0.050781 |

## Claim Boundaries

- This is post-processing of one fixed 56,301-row WSN-DS test partition; no new split or model fit is performed.
- The inferential unit is the paired algorithmic run seed, with ten seeds for every unique route.
- Primary inference uses the row-level view, the exact paired Wilcoxon signed-rank test, and Holm adjustment within each predeclared teacher or student route family.
- View-global Holm results are reported as a stricter multiplicity sensitivity analysis; they do not replace the predeclared family-wise primary policy.
- The inverse-size view retains every row and assigns each exact test feature group total weight one; mixed-label rows keep their recorded labels.
- The representative view keeps the smallest source-row index from each label-pure exact feature group and excludes every mixed-label group and row.
- No mixed-label feature group is assigned a majority label in either sensitivity view.
- The sensitivity views quantify within-test repeated-pattern weighting and do not estimate performance on a new dataset, new partition, live traffic, or independent network events.
- Holm families are declared separately by view for teachers, Student A routes, and Student B routes; view-global adjustments are also reported.
- Alias route names are excluded from inference because they duplicate an already included route exactly.
- Per-class route deltas are descriptive because no per-class hypothesis-test family is performed.
- Predicted classes must agree exactly within every exact-feature group. Persisted probability vectors are audited with a maximum absolute tolerance of 2e-6 for float32 and decimal serialization effects; group weighting does not use probabilities.
- The exact paired p-values in this analysis supersede the full-route runner's approximate Wilcoxon fields; saved models, predictions, and performance metrics are unchanged.
