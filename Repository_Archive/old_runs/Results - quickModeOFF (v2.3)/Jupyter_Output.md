# 1 
nothing

---

# 2 - 

Device: cuda
PyTorch: 2.5.1+cu121
Seeds: [42, 123, 456, 789, 1001]
Quick mode: False

---

# 3 - 

Shape: (374661, 19)
Columns: [' id', ' Time', ' Is_CH', ' who CH', ' Dist_To_CH', ' ADV_S', ' ADV_R', ' JOIN_S', ' JOIN_R', ' SCH_S', ' SCH_R', 'Rank', ' DATA_S', ' DATA_R', ' Data_Sent_To_BS', ' dist_CH_To_BS', ' send_code ', 'Expaned Energy', 'Attack type']

First row:
       id  Time  Is_CH  who CH  Dist_To_CH  ADV_S  ADV_R  JOIN_S  JOIN_R  SCH_S  SCH_R  Rank  DATA_S  DATA_R  Data_Sent_To_BS  dist_CH_To_BS  send_code  Expaned Energy Attack type
0  101000    50      1  101000         0.0      1      0       0      25      1      0     0       0    1200               48      130.08535          0          2.4694      Normal

---

# 4 - 

Target column: Attack type
Dropped id
Classes: ['Blackhole', 'Flooding', 'Grayhole', 'Normal', 'TDMA']
Mapping: {'Blackhole': 0, 'Flooding': 1, 'Grayhole': 2, 'Normal': 3, 'TDMA': 4}
Input dim: 17
Class distribution: {'Blackhole': 10049, 'Flooding': 3312, 'Grayhole': 14596, 'Normal': 340066, 'TDMA': 6638}

---

# 5 - 

Teacher MLP: 69893 params (273.02 KB fp32, 68.25 KB int8)
Student A (32, 16): 1189 params (4.64 KB fp32, 1.16 KB int8)
Student B (64, 32): 3397 params (13.27 KB fp32, 3.32 KB int8)
Student A FLOPs: 2373
Student B FLOPs: 6789

---

# 10 - 

Train: (262252, 17), Val: (56209, 17), Test: (56200, 17)

>>> KD hyperparameter grid search (seed 42, Student A)
  T=2, alpha=0.5: val_f1=0.9113
  T=2, alpha=0.7: val_f1=0.9055
  T=2, alpha=0.9: val_f1=0.9027
  T=3, alpha=0.5: val_f1=0.9050
  T=3, alpha=0.7: val_f1=0.9089
  T=3, alpha=0.9: val_f1=0.9098
  T=4, alpha=0.5: val_f1=0.9084
  T=4, alpha=0.7: val_f1=0.9061
  T=4, alpha=0.9: val_f1=0.9060
  T=5, alpha=0.5: val_f1=0.9069
  T=5, alpha=0.7: val_f1=0.9096
  T=5, alpha=0.9: val_f1=0.9082

Best KD hyperparameters: T=2, alpha=0.5 (val F1 0.9113)

>>> Running 5 seeds with Student A (32, 16)

============================================================
Seed 42 — Student (32, 16)
============================================================
[A] RF baseline...
[B] Full MLP baseline...
[Difficulty] Loss-based scoring...
[C_fair] CL-trained MLP (loss-based, fair budget)...
[C_ext] CL-trained MLP (loss-based, extended budget)...
[C2] CL-trained MLP (domain-based, fair budget)...
[D] Small MLP (32, 16) from scratch...
[E] KD from calibrated RF...
[E2] KD from standard MLP teacher...
[F_fair] KD from fair-budget CL-MLP...
[F_ext] KD from extended-budget CL-MLP...
[G] KD from random-pacing MLP (control)...
[I] KD from SMOTE-trained MLP teacher...

Seed 42 summary (macro F1):
  A_RF_500                  0.9794
  B_Full_MLP                0.9268
  C_CL_MLP_loss_fair        0.9141
  C_CL_MLP_loss_ext         0.9173
  C_CL_MLP_loss             0.9141
  C2_CL_MLP_domain          0.9147
  D_Small_MLP               0.9111
  E_KD_from_RF              0.9168
  E2_KD_from_MLP            0.9198
  F_KD_from_CL_MLP_fair     0.9098
  F_KD_from_CL_MLP_ext      0.9177
  F_KD_from_CL_MLP          0.9098
  G_KD_random_pacing        0.9164
  I_KD_from_SMOTE_MLP       0.9110

============================================================
Seed 123 — Student (32, 16)
============================================================
[A] RF baseline...
[B] Full MLP baseline...
[Difficulty] Loss-based scoring...
[C_fair] CL-trained MLP (loss-based, fair budget)...
[C_ext] CL-trained MLP (loss-based, extended budget)...
[C2] CL-trained MLP (domain-based, fair budget)...
[D] Small MLP (32, 16) from scratch...
[E] KD from calibrated RF...
[E2] KD from standard MLP teacher...
[F_fair] KD from fair-budget CL-MLP...
[F_ext] KD from extended-budget CL-MLP...
[G] KD from random-pacing MLP (control)...
[I] KD from SMOTE-trained MLP teacher...

Seed 123 summary (macro F1):
  A_RF_500                  0.9786
  B_Full_MLP                0.9201
  C_CL_MLP_loss_fair        0.9172
  C_CL_MLP_loss_ext         0.9202
  C_CL_MLP_loss             0.9172
  C2_CL_MLP_domain          0.9125
  D_Small_MLP               0.9200
  E_KD_from_RF              0.9248
  E2_KD_from_MLP            0.9160
  F_KD_from_CL_MLP_fair     0.9116
  F_KD_from_CL_MLP_ext      0.9137
  F_KD_from_CL_MLP          0.9116
  G_KD_random_pacing        0.9127
  I_KD_from_SMOTE_MLP       0.9097

============================================================
Seed 456 — Student (32, 16)
============================================================
[A] RF baseline...
[B] Full MLP baseline...
[Difficulty] Loss-based scoring...
[C_fair] CL-trained MLP (loss-based, fair budget)...
[C_ext] CL-trained MLP (loss-based, extended budget)...
[C2] CL-trained MLP (domain-based, fair budget)...
[D] Small MLP (32, 16) from scratch...
[E] KD from calibrated RF...
[E2] KD from standard MLP teacher...
[F_fair] KD from fair-budget CL-MLP...
[F_ext] KD from extended-budget CL-MLP...
[G] KD from random-pacing MLP (control)...
[I] KD from SMOTE-trained MLP teacher...

Seed 456 summary (macro F1):
  A_RF_500                  0.9789
  B_Full_MLP                0.9253
  C_CL_MLP_loss_fair        0.9181
  C_CL_MLP_loss_ext         0.9215
  C_CL_MLP_loss             0.9181
  C2_CL_MLP_domain          0.9133
  D_Small_MLP               0.9125
  E_KD_from_RF              0.9193
  E2_KD_from_MLP            0.9164
  F_KD_from_CL_MLP_fair     0.9124
  F_KD_from_CL_MLP_ext      0.9137
  F_KD_from_CL_MLP          0.9124
  G_KD_random_pacing        0.9108
  I_KD_from_SMOTE_MLP       0.9164

============================================================
Seed 789 — Student (32, 16)
============================================================
[A] RF baseline...
[B] Full MLP baseline...
[Difficulty] Loss-based scoring...
[C_fair] CL-trained MLP (loss-based, fair budget)...
[C_ext] CL-trained MLP (loss-based, extended budget)...
[C2] CL-trained MLP (domain-based, fair budget)...
[D] Small MLP (32, 16) from scratch...
[E] KD from calibrated RF...
[E2] KD from standard MLP teacher...
[F_fair] KD from fair-budget CL-MLP...
[F_ext] KD from extended-budget CL-MLP...
[G] KD from random-pacing MLP (control)...
[I] KD from SMOTE-trained MLP teacher...

Seed 789 summary (macro F1):
  A_RF_500                  0.9788
  B_Full_MLP                0.9216
  C_CL_MLP_loss_fair        0.9110
  C_CL_MLP_loss_ext         0.9159
  C_CL_MLP_loss             0.9110
  C2_CL_MLP_domain          0.9109
  D_Small_MLP               0.9136
  E_KD_from_RF              0.9194
  E2_KD_from_MLP            0.9123
  F_KD_from_CL_MLP_fair     0.9149
  F_KD_from_CL_MLP_ext      0.9110
  F_KD_from_CL_MLP          0.9149
  G_KD_random_pacing        0.9130
  I_KD_from_SMOTE_MLP       0.9204

============================================================
Seed 1001 — Student (32, 16)
============================================================
[A] RF baseline...
[B] Full MLP baseline...
[Difficulty] Loss-based scoring...
[C_fair] CL-trained MLP (loss-based, fair budget)...
[C_ext] CL-trained MLP (loss-based, extended budget)...
[C2] CL-trained MLP (domain-based, fair budget)...
[D] Small MLP (32, 16) from scratch...
[E] KD from calibrated RF...
[E2] KD from standard MLP teacher...
[F_fair] KD from fair-budget CL-MLP...
[F_ext] KD from extended-budget CL-MLP...
[G] KD from random-pacing MLP (control)...
[I] KD from SMOTE-trained MLP teacher...

Seed 1001 summary (macro F1):
  A_RF_500                  0.9791
  B_Full_MLP                0.9267
  C_CL_MLP_loss_fair        0.9212
  C_CL_MLP_loss_ext         0.9152
  C_CL_MLP_loss             0.9212
  C2_CL_MLP_domain          0.9209
  D_Small_MLP               0.9031
  E_KD_from_RF              0.9222
  E2_KD_from_MLP            0.9196
  F_KD_from_CL_MLP_fair     0.9126
  F_KD_from_CL_MLP_ext      0.9195
  F_KD_from_CL_MLP          0.9126
  G_KD_random_pacing        0.9199
  I_KD_from_SMOTE_MLP       0.9102

>>> Running 5 seeds with Student B (64, 32)

>>> Re-running final seed to capture models for SHAP/benchmarks...

---

# 11 - 

============================================================
MULTI-SEED AGGREGATE RESULTS (Student A)
============================================================
               Config  Accuracy_mean  Accuracy_std  MacroF1_mean  MacroF1_std  n_seeds
             A_RF_500       0.996619      0.000046      0.978967     0.000289        5
           B_Full_MLP       0.987516      0.000448      0.924084     0.002748        5
     C2_CL_MLP_domain       0.985815      0.000524      0.914472     0.003465        5
        C_CL_MLP_loss       0.986142      0.000587      0.916353     0.003490        5
    C_CL_MLP_loss_ext       0.986441      0.000420      0.918036     0.002443        5
   C_CL_MLP_loss_fair       0.986142      0.000587      0.916353     0.003490        5
          D_Small_MLP       0.984822      0.002011      0.912039     0.005412        5
       E2_KD_from_MLP       0.986206      0.000444      0.916829     0.002755        5
         E_KD_from_RF       0.986968      0.000482      0.920490     0.002759        5
     F_KD_from_CL_MLP       0.985434      0.000247      0.912259     0.001646        5
 F_KD_from_CL_MLP_ext       0.985900      0.000505      0.915104     0.003047        5
F_KD_from_CL_MLP_fair       0.985434      0.000247      0.912259     0.001646        5
   G_KD_random_pacing       0.985804      0.000460      0.914557     0.003217        5
  I_KD_from_SMOTE_MLP       0.985598      0.000702      0.913549     0.004180        5

============================================================
KEY WILCOXON COMPARISONS (Student A)
============================================================
Does CL help teacher at FAIR budget? (C_fair vs B)
  C_CL_MLP_loss_fair: 0.9164
  B_Full_MLP: 0.9241
  diff: -0.0077  |  p=0.0625  |  not significant

Does CL help teacher at EXT budget? (C_ext vs B)
  C_CL_MLP_loss_ext: 0.9180
  B_Full_MLP: 0.9241
  diff: -0.0060  |  p=0.125  |  not significant

Does CL cascade at FAIR budget? (F_fair vs E2)
  F_KD_from_CL_MLP_fair: 0.9123
  E2_KD_from_MLP: 0.9168
  diff: -0.0046  |  p=0.125  |  not significant

Does CL cascade at EXT budget? (F_ext vs E2)
  F_KD_from_CL_MLP_ext: 0.9151
  E2_KD_from_MLP: 0.9168
  diff: -0.0017  |  p=0.0625  |  not significant

Does KD beat scratch? (F vs D)
  F_KD_from_CL_MLP: 0.9123
  D_Small_MLP: 0.9120
  diff: +0.0002  |  p=1.0  |  not significant

Does KD work at all? (E2 vs D)
  E2_KD_from_MLP: 0.9168
  D_Small_MLP: 0.9120
  diff: +0.0048  |  p=0.3125  |  not significant

Order vs random pacing? (F vs G)
  F_KD_from_CL_MLP: 0.9123
  G_KD_random_pacing: 0.9146
  diff: -0.0023  |  p=0.625  |  not significant

CL vs SMOTE teacher? (F vs I)
  F_KD_from_CL_MLP: 0.9123
  I_KD_from_SMOTE_MLP: 0.9135
  diff: -0.0013  |  p=0.625  |  not significant

RF teacher vs MLP teacher? (E vs E2)
  E_KD_from_RF: 0.9205
  E2_KD_from_MLP: 0.9168
  diff: +0.0037  |  p=0.3125  |  not significant

---

# 12 - 

============================================================
SHAP ANALYSIS
============================================================
Computing SHAP for student (DeepExplainer)...

Student top-10 features (global):
        feature  student_shap
          Is_CH      2.195449
           Time      1.406792
Data_Sent_To_BS      1.160995
         JOIN_S      0.841460
  dist_CH_To_BS      0.686562
         DATA_R      0.604351
          SCH_S      0.556257
         who CH      0.520036
           Rank      0.500557
          ADV_R      0.500505

Student per-class top-3:
  Blackhole: [('Is_CH', 2.6465), ('dist_CH_To_BS', 2.3454), ('Data_Sent_To_BS', 2.2101)]
  Flooding: [('Time', 3.2326), ('ADV_S', 1.0044), ('SCH_S', 0.6612)]
  Grayhole: [('Is_CH', 2.8031), ('Data_Sent_To_BS', 1.4489), ('Time', 1.2365)]
  Normal: [('Is_CH', 3.0628), ('Time', 1.543), ('JOIN_S', 1.0192)]
  TDMA: [('Is_CH', 1.912), ('Data_Sent_To_BS', 0.8555), ('JOIN_S', 0.7725)]

Computing SHAP for RF teacher (TreeExplainer)...

Teacher top-10 features (global):
        feature  teacher_shap
          ADV_S      0.022460
          Is_CH      0.018910
Data_Sent_To_BS      0.011235
         DATA_S      0.010423
 Expaned Energy      0.008432
          SCH_S      0.006787
         JOIN_S      0.006140
      send_code      0.006098
           Rank      0.005533
  dist_CH_To_BS      0.005397

Feature ranking agreement (Spearman): rho=0.1176, p=6.5293e-01
Interpretation: Student diverges from teacher reasoning
  Blackhole    rho=+0.3725  p=1.4084e-01
  Flooding     rho=+0.2770  p=2.8184e-01
  Grayhole     rho=+0.0074  p=9.7766e-01
  Normal       rho=+0.0515  p=8.4447e-01
  TDMA         rho=+0.1593  p=5.4136e-01
C:\Users\nhnis\AppData\Local\Temp\ipykernel_24160\1396050619.py:113: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(
Saved shap_summary_student.png

Bootstrap SHAP stability (5 different backgrounds)...
  bootstrap 1/5: rho=+0.1176  p=6.5293e-01
  bootstrap 2/5: rho=+0.1789  p=4.9202e-01
  bootstrap 3/5: rho=+0.1348  p=6.0597e-01
  bootstrap 4/5: rho=+0.2770  p=2.8184e-01
  bootstrap 5/5: rho=+0.1250  p=6.3264e-01

Bootstrap Spearman (mean ± std): +0.1667 ± 0.0591
95% bootstrap CI (approx): [+0.0508, +0.2825]

---

# 13 -

============================================================
INT8 QUANTIZATION SWEEP — all student configs
============================================================
  [D_Small_MLP] fp32 7.29KB F1=0.9031 → int8 5.85KB F1=0.8690 (F1 -3.404%)
  [E_KD_from_RF] fp32 7.29KB F1=0.9222 → int8 5.85KB F1=0.8995 (F1 -2.266%)
  [E2_KD_from_MLP] fp32 7.29KB F1=0.9196 → int8 5.85KB F1=0.8065 (F1 -11.315%)
  [F_KD_from_CL_MLP_fair] fp32 7.29KB F1=0.9126 → int8 5.85KB F1=0.8730 (F1 -3.956%)
  [F_KD_from_CL_MLP_ext] fp32 7.29KB F1=0.9195 → int8 5.85KB F1=0.8491 (F1 -7.031%)
  [F_KD_from_CL_MLP] fp32 7.29KB F1=0.9126 → int8 5.85KB F1=0.8730 (F1 -3.956%)
  [G_KD_random_pacing] fp32 7.29KB F1=0.9199 → int8 5.85KB F1=0.8708 (F1 -4.910%)
  [I_KD_from_SMOTE_MLP] fp32 7.29KB F1=0.9102 → int8 5.85KB F1=0.7879 (F1 -12.231%)

---

# 14 - 

============================================================
INFERENCE TIME BENCHMARKS
============================================================
Teacher_MLP:
  Params:     69893
  Size fp32:  273.02 KB
  Size int8:  68.25 KB
  GPU latency (batch=1): 0.791 ms
  CPU latency (batch=1): 0.171 ms
Student_D_scratch:
  Params:     1189
  Size fp32:  4.64 KB
  Size int8:  1.16 KB
  GPU latency (batch=1): 0.241 ms
  CPU latency (batch=1): 0.050 ms
Student_E_KD_RF:
  Params:     1189
  Size fp32:  4.64 KB
  Size int8:  1.16 KB
  GPU latency (batch=1): 0.241 ms
  CPU latency (batch=1): 0.049 ms
Student_E2_KD_MLP:
  Params:     1189
  Size fp32:  4.64 KB
  Size int8:  1.16 KB
  GPU latency (batch=1): 0.236 ms
  CPU latency (batch=1): 0.050 ms
Student_F_KD_CL:
  Params:     1189
  Size fp32:  4.64 KB
  Size int8:  1.16 KB
  GPU latency (batch=1): 0.248 ms
  CPU latency (batch=1): 0.051 ms
Student_F_KD_CL_fair:
  Params:     1189
  Size fp32:  4.64 KB
  Size int8:  1.16 KB
  GPU latency (batch=1): 0.238 ms
  CPU latency (batch=1): 0.049 ms
Student_F_KD_CL_ext:
  Params:     1189
  Size fp32:  4.64 KB
  Size int8:  1.16 KB
  GPU latency (batch=1): 0.249 ms
  CPU latency (batch=1): 0.051 ms
Student_G_rand_pacing:
  Params:     1189
  Size fp32:  4.64 KB
  Size int8:  1.16 KB
  GPU latency (batch=1): 0.238 ms
  CPU latency (batch=1): 0.055 ms
Student_I_KD_SMOTE:
  Params:     1189
  Size fp32:  4.64 KB
  Size int8:  1.16 KB
  GPU latency (batch=1): 0.237 ms
  CPU latency (batch=1): 0.051 ms

---

# 15 - 

============================================================
GENERATING FIGURES
============================================================
Saved per_class_f1.png
Saved confusion_matrix_E.png
Saved confusion_matrix_F.png
Saved confusion_matrix_F_fair.png
Saved confusion_matrix_F_ext.png
Saved pareto_frontier.png
Saved loss_curves_B_vs_C.png


---

# 18 - 

============================================================
FINAL SUMMARY — CuKD-XAI ON WSN-DS
============================================================

Seeds: [42, 123, 456, 789, 1001]  |  KD: T=2, alpha=0.5
Student architecture: MLP(32, 16) → 5 classes

Compression:
  Teacher: 69893 params (273.02 KB fp32)
  Student: 1189 params (4.64 KB fp32, 1.16 KB int8)
  Ratio:   58.8x parameter reduction

Key metrics (mean over seeds):
               Config  MacroF1_mean  MacroF1_std  Accuracy_mean  Accuracy_std  params      size_kb
             A_RF_500      0.978967     0.000289       0.996619      0.000046     NaN 85064.536133
           B_Full_MLP      0.924084     0.002748       0.987516      0.000448 69893.0   273.019531
     C2_CL_MLP_domain      0.914472     0.003465       0.985815      0.000524 69893.0   273.019531
        C_CL_MLP_loss      0.916353     0.003490       0.986142      0.000587 69893.0   273.019531
    C_CL_MLP_loss_ext      0.918036     0.002443       0.986441      0.000420 69893.0   273.019531
   C_CL_MLP_loss_fair      0.916353     0.003490       0.986142      0.000587 69893.0   273.019531
          D_Small_MLP      0.912039     0.005412       0.984822      0.002011  1189.0     4.644531
       E2_KD_from_MLP      0.916829     0.002755       0.986206      0.000444  1189.0     4.644531
         E_KD_from_RF      0.920490     0.002759       0.986968      0.000482  1189.0     4.644531
     F_KD_from_CL_MLP      0.912259     0.001646       0.985434      0.000247  1189.0     4.644531
 F_KD_from_CL_MLP_ext      0.915104     0.003047       0.985900      0.000505  1189.0     4.644531
F_KD_from_CL_MLP_fair      0.912259     0.001646       0.985434      0.000247  1189.0     4.644531
   G_KD_random_pacing      0.914557     0.003217       0.985804      0.000460  1189.0     4.644531
  I_KD_from_SMOTE_MLP      0.913549     0.004180       0.985598      0.000702  1189.0     4.644531

============================================================
IMPLEMENTATION COMPLETE
============================================================
Outputs:
  cukd_xai_results.json — all results
  wsnds_results_student_A.csv — aggregate metrics
  per_class_f1.png — per-class F1 comparison
  confusion_matrix_F.png — Config F confusion matrix
  pareto_frontier.png — size vs accuracy trade-off
  shap_summary_student.png — SHAP feature importance
  loss_curves_B_vs_C.png — CL convergence comparison