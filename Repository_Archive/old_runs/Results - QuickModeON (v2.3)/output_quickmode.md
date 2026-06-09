# 1

- nothing

---


# 2

Device: cuda
PyTorch: 2.5.1+cu121
Seeds: [42]
Quick mode: True
QUICK MODE: running single seed only

---


# 3 - load wsn dataset
Shape: (374661, 19)
Columns: [' id', ' Time', ' Is_CH', ' who CH', ' Dist_To_CH', ' ADV_S', ' ADV_R', ' JOIN_S', ' JOIN_R', ' SCH_S', ' SCH_R', 'Rank', ' DATA_S', ' DATA_R', ' Data_Sent_To_BS', ' dist_CH_To_BS', ' send_code ', 'Expaned Energy', 'Attack type']

First row:
       id  Time  Is_CH  who CH  Dist_To_CH  ADV_S  ADV_R  JOIN_S  JOIN_R  SCH_S  SCH_R  Rank  DATA_S  DATA_R  Data_Sent_To_BS  dist_CH_To_BS  send_code  Expaned Energy Attack type
0  101000    50      1  101000         0.0      1      0       0      25      1      0     0       0    1200               48      130.08535          0          2.4694      Normal

---


# 4 - preprocess wsn dataset
Target column: Attack type
Dropped id
Classes: ['Blackhole', 'Flooding', 'Grayhole', 'Normal', 'TDMA']
Mapping: {'Blackhole': 0, 'Flooding': 1, 'Grayhole': 2, 'Normal': 3, 'TDMA': 4}
Input dim: 17
Class distribution: {'Blackhole': 10049, 'Flooding': 3312, 'Grayhole': 14596, 'Normal': 340066, 'TDMA': 6638}

---

# 5 - Model architectures
Teacher MLP: 69893 params (273.02 KB fp32, 68.25 KB int8)
Student A (32, 16): 1189 params (4.64 KB fp32, 1.16 KB int8)
Student B (64, 32): 3397 params (13.27 KB fp32, 3.32 KB int8)
Student A FLOPs: 2373
Student B FLOPs: 6789

---

# 6 - training, evaluation, and measurement utilities

nothing

---

# 7 - Difficulty scoring - both methods 

nothing

---

# 8 - KD hyperparameter grid search (run once on seed 42)

nothing

---

# 9 - Single-seed experiment runner — ALL configs A-I + H¶

nothing

---

# 10 Run WSN-DS experiments over multiple seeds

Train: (262252, 17), Val: (56209, 17), Test: (56200, 17)

>>> Running 1 seeds with Student A (32, 16)

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
  E_KD_from_RF              0.9193
  E2_KD_from_MLP            0.9117
  F_KD_from_CL_MLP_fair     0.9048
  F_KD_from_CL_MLP_ext      0.9122
  F_KD_from_CL_MLP          0.9048
  G_KD_random_pacing        0.9072
  I_KD_from_SMOTE_MLP       0.9128

>>> Re-running final seed to capture models for SHAP/benchmarks...

---

# 11 -Aggregate multi-seed statistics + Wilcoxon tests

```
============================================================
MULTI-SEED AGGREGATE RESULTS (Student A)
============================================================
               Config  Accuracy_mean  Accuracy_std  MacroF1_mean  MacroF1_std  n_seeds
             A_RF_500       0.996673           0.0      0.979410          0.0        1
           B_Full_MLP       0.987936           0.0      0.926796          0.0        1
     C2_CL_MLP_domain       0.985872           0.0      0.914721          0.0        1
        C_CL_MLP_loss       0.985996           0.0      0.914145          0.0        1
    C_CL_MLP_loss_ext       0.986281           0.0      0.917323          0.0        1
   C_CL_MLP_loss_fair       0.985996           0.0      0.914145          0.0        1
          D_Small_MLP       0.985107           0.0      0.911100          0.0        1
       E2_KD_from_MLP       0.985302           0.0      0.911680          0.0        1
         E_KD_from_RF       0.986868           0.0      0.919297          0.0        1
     F_KD_from_CL_MLP       0.984146           0.0      0.904804          0.0        1
 F_KD_from_CL_MLP_ext       0.985356           0.0      0.912220          0.0        1
F_KD_from_CL_MLP_fair       0.984146           0.0      0.904804          0.0        1
   G_KD_random_pacing       0.984555           0.0      0.907194          0.0        1
  I_KD_from_SMOTE_MLP       0.985516           0.0      0.912773          0.0        1

============================================================
KEY WILCOXON COMPARISONS (Student A)
============================================================
Does CL help teacher at FAIR budget? (C_fair vs B)
  C_CL_MLP_loss_fair: 0.9141
  B_Full_MLP: 0.9268
  diff: n/a  |  p=None  |  insufficient data

Does CL help teacher at EXT budget? (C_ext vs B)
  C_CL_MLP_loss_ext: 0.9173
  B_Full_MLP: 0.9268
  diff: n/a  |  p=None  |  insufficient data

Does CL cascade at FAIR budget? (F_fair vs E2)
  F_KD_from_CL_MLP_fair: 0.9048
  E2_KD_from_MLP: 0.9117
  diff: n/a  |  p=None  |  insufficient data

Does CL cascade at EXT budget? (F_ext vs E2)
  F_KD_from_CL_MLP_ext: 0.9122
  E2_KD_from_MLP: 0.9117
  diff: n/a  |  p=None  |  insufficient data

Does KD beat scratch? (F vs D)
  F_KD_from_CL_MLP: 0.9048
  D_Small_MLP: 0.9111
  diff: n/a  |  p=None  |  insufficient data

Does KD work at all? (E2 vs D)
  E2_KD_from_MLP: 0.9117
  D_Small_MLP: 0.9111
  diff: n/a  |  p=None  |  insufficient data

Order vs random pacing? (F vs G)
  F_KD_from_CL_MLP: 0.9048
  G_KD_random_pacing: 0.9072
  diff: n/a  |  p=None  |  insufficient data

CL vs SMOTE teacher? (F vs I)
  F_KD_from_CL_MLP: 0.9048
  I_KD_from_SMOTE_MLP: 0.9128
  diff: n/a  |  p=None  |  insufficient data

RF teacher vs MLP teacher? (E vs E2)
  E_KD_from_RF: 0.9193
  E2_KD_from_MLP: 0.9117
  diff: n/a  |  p=None  |  insufficient data
```

---

# 12 - SHAP analysis — DeepExplainer on student + TreeExplainer on RF teacher

============================================================
SHAP ANALYSIS
============================================================
Computing SHAP for student (DeepExplainer)...

Student top-10 features (global):
        feature  student_shap
         JOIN_S      1.909443
           Time      1.407187
          Is_CH      1.391150
Data_Sent_To_BS      0.978766
  dist_CH_To_BS      0.961578
         DATA_R      0.560115
          ADV_R      0.462843
          ADV_S      0.421603
      send_code      0.405828
 Expaned Energy      0.325797

Student per-class top-3:
  Blackhole: [('dist_CH_To_BS', 3.3602), ('JOIN_S', 1.6169), ('Is_CH', 1.2541)]
  Flooding: [('Time', 2.5651), ('ADV_S', 1.1678), ('JOIN_S', 1.0138)]
  Grayhole: [('JOIN_S', 2.1322), ('Time', 1.5975), ('Data_Sent_To_BS', 1.5743)]
  Normal: [('JOIN_S', 2.6997), ('Is_CH', 2.0854), ('Time', 1.8614)]
  TDMA: [('JOIN_S', 2.0846), ('Is_CH', 1.6782), ('Data_Sent_To_BS', 0.7539)]

Computing SHAP for RF teacher (TreeExplainer)...

Teacher top-10 features (global):
        feature  teacher_shap
          Is_CH      0.022170
          ADV_S      0.019786
Data_Sent_To_BS      0.010833
         DATA_S      0.008980
           Rank      0.008360
 Expaned Energy      0.008258
          SCH_S      0.006782
  dist_CH_To_BS      0.005664
      send_code      0.005289
         JOIN_S      0.004979

Feature ranking agreement (Spearman): rho=0.0980, p=7.0815e-01
Interpretation: Student diverges from teacher reasoning
  Blackhole    rho=+0.0343  p=8.9598e-01
  Flooding     rho=+0.6176  p=8.2423e-03
  Grayhole     rho=-0.0147  p=9.5533e-01
  Normal       rho=+0.2500  p=3.3317e-01
  TDMA         rho=+0.0515  p=8.4447e-01
C:\Users\nhnis\AppData\Local\Temp\ipykernel_25244\1396050619.py:113: FutureWarning: The NumPy global RNG was seeded by calling `np.random.seed`. In a future version this function will no longer use the global RNG. Pass `rng` explicitly to opt-in to the new behaviour and silence this warning.
  shap.summary_plot(
Saved shap_summary_student.png

Bootstrap SHAP stability (5 different backgrounds)...
  bootstrap 1/5: rho=+0.0980  p=7.0815e-01
  bootstrap 2/5: rho=+0.1054  p=6.8727e-01
  bootstrap 3/5: rho=+0.0098  p=9.7021e-01
  bootstrap 4/5: rho=+0.1250  p=6.3264e-01
  bootstrap 5/5: rho=+0.1520  p=5.6042e-01

Bootstrap Spearman (mean ± std): +0.0980 ± 0.0479
95% bootstrap CI (approx): [+0.0041, +0.1919]

---

# 13 - Actual INT8 quantization experiment — SWEEP over all student configs

============================================================
INT8 QUANTIZATION SWEEP — all student configs
============================================================
  [D_Small_MLP] fp32 7.29KB F1=0.9111 → int8 5.85KB F1=0.8337 (F1 -7.744%)
  [E_KD_from_RF] fp32 7.29KB F1=0.9193 → int8 5.85KB F1=0.8521 (F1 -6.722%)
  [E2_KD_from_MLP] fp32 7.29KB F1=0.9117 → int8 5.85KB F1=0.8716 (F1 -4.005%)
  [F_KD_from_CL_MLP_fair] fp32 7.29KB F1=0.9048 → int8 5.85KB F1=0.8550 (F1 -4.983%)
  [F_KD_from_CL_MLP_ext] fp32 7.29KB F1=0.9122 → int8 5.85KB F1=0.8652 (F1 -4.706%)
  [F_KD_from_CL_MLP] fp32 7.29KB F1=0.9048 → int8 5.85KB F1=0.8550 (F1 -4.983%)
  [G_KD_random_pacing] fp32 7.29KB F1=0.9072 → int8 5.85KB F1=0.8491 (F1 -5.810%)
  [I_KD_from_SMOTE_MLP] fp32 7.29KB F1=0.9128 → int8 5.85KB F1=0.7847 (F1 -12.807%)


---

# 14 - Inference time & throughput benchmarks

============================================================
INFERENCE TIME BENCHMARKS
============================================================
Teacher_MLP:
  Params:     69893
  Size fp32:  273.02 KB
  Size int8:  68.25 KB
  GPU latency (batch=1): 0.873 ms
  CPU latency (batch=1): 0.168 ms
Student_D_scratch:
  Params:     1189
  Size fp32:  4.64 KB
  Size int8:  1.16 KB
  GPU latency (batch=1): 0.226 ms
  CPU latency (batch=1): 0.050 ms
Student_E_KD_RF:
  Params:     1189
  Size fp32:  4.64 KB
  Size int8:  1.16 KB
  GPU latency (batch=1): 0.222 ms
  CPU latency (batch=1): 0.049 ms
Student_E2_KD_MLP:
  Params:     1189
  Size fp32:  4.64 KB
  Size int8:  1.16 KB
  GPU latency (batch=1): 0.218 ms
  CPU latency (batch=1): 0.048 ms
Student_F_KD_CL:
  Params:     1189
  Size fp32:  4.64 KB
  Size int8:  1.16 KB
  GPU latency (batch=1): 0.228 ms
  CPU latency (batch=1): 0.048 ms
Student_F_KD_CL_fair:
  Params:     1189
  Size fp32:  4.64 KB
  Size int8:  1.16 KB
  GPU latency (batch=1): 0.218 ms
  CPU latency (batch=1): 0.048 ms
Student_F_KD_CL_ext:
  Params:     1189
  Size fp32:  4.64 KB
  Size int8:  1.16 KB
  GPU latency (batch=1): 0.217 ms
  CPU latency (batch=1): 0.047 ms
Student_G_rand_pacing:
  Params:     1189
  Size fp32:  4.64 KB
  Size int8:  1.16 KB
  GPU latency (batch=1): 0.229 ms
  CPU latency (batch=1): 0.050 ms
Student_I_KD_SMOTE:
  Params:     1189
  Size fp32:  4.64 KB
  Size int8:  1.16 KB
  GPU latency (batch=1): 0.231 ms
  CPU latency (batch=1): 0.048 ms


---

# 15 - Visualization — per-class F1, confusion matrix, Pareto, loss curves

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

# 16 - CICIOT2023

NOT RAN

---

# 17 - Final summary for paper

============================================================
FINAL SUMMARY — CuKD-XAI ON WSN-DS
============================================================

Seeds: [42]  |  KD: T=4, alpha=0.7
Student architecture: MLP(32, 16) → 5 classes

Compression:
  Teacher: 69893 params (273.02 KB fp32)
  Student: 1189 params (4.64 KB fp32, 1.16 KB int8)
  Ratio:   58.8x parameter reduction

Key metrics (mean over seeds):
               Config  MacroF1_mean  MacroF1_std  Accuracy_mean  Accuracy_std  params      size_kb
             A_RF_500      0.979410          0.0       0.996673           0.0     NaN 85064.536133
           B_Full_MLP      0.926796          0.0       0.987936           0.0 69893.0   273.019531
     C2_CL_MLP_domain      0.914721          0.0       0.985872           0.0 69893.0   273.019531
        C_CL_MLP_loss      0.914145          0.0       0.985996           0.0 69893.0   273.019531
    C_CL_MLP_loss_ext      0.917323          0.0       0.986281           0.0 69893.0   273.019531
   C_CL_MLP_loss_fair      0.914145          0.0       0.985996           0.0 69893.0   273.019531
          D_Small_MLP      0.911100          0.0       0.985107           0.0  1189.0     4.644531
       E2_KD_from_MLP      0.911680          0.0       0.985302           0.0  1189.0     4.644531
         E_KD_from_RF      0.919297          0.0       0.986868           0.0  1189.0     4.644531
     F_KD_from_CL_MLP      0.904804          0.0       0.984146           0.0  1189.0     4.644531
 F_KD_from_CL_MLP_ext      0.912220          0.0       0.985356           0.0  1189.0     4.644531
F_KD_from_CL_MLP_fair      0.904804          0.0       0.984146           0.0  1189.0     4.644531
   G_KD_random_pacing      0.907194          0.0       0.984555           0.0  1189.0     4.644531
  I_KD_from_SMOTE_MLP      0.912773          0.0       0.985516           0.0  1189.0     4.644531

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