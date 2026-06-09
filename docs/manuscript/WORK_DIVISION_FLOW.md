# The Pipeline Architecture

```text
Stage 0: Shared data prep (everyone runs this, ~2 min)
    ↓
Stage 1 (Person 1): Teachers → teachers/*.pt on Drive
    ↓
Stage 2 (Person 2): Students via KD → students/*.pt on Drive
    ↓              ↓
Stage 3 (Person 3)  Stage 4 (Person 4)
(evaluation)       (SHAP + quant + viz)
    ↓              ↓
    → results/*.json on Drive
```

Stages 3 and 4 run in parallel once Stage 2 finishes.

## Shared Drive structure (set up before anyone starts)

```text
/MyDrive/cukd_shared/
    /shared/          # Stage 0 outputs
    /teachers/        # Stage 1 outputs (Person 1)
    /students/        # Stage 2 outputs (Person 2)
    /results/         # Stage 3 + 4 outputs (Persons 3, 4)
```

One person creates this folder and shares it with the other 3.

---

## Stage 0 — Shared setup (~2 min, everyone runs)

Every person's notebook starts with this cell:

```python
from google.colab import drive
drive.mount('/content/drive')

SHARED = '/content/drive/MyDrive/cukd_shared'
import os
for sub in ['shared', 'teachers', 'students', 'results']:
    os.makedirs(f'{SHARED}/{sub}', exist_ok=True)

# Only runs once — if the file exists, skip
import pickle
DATA_PATH = f'{SHARED}/shared/data_arrays.pkl'
if not os.path.exists(DATA_PATH):
    # ... data loading, preprocessing, stratified split ...
    # (copy Cells 3, 4 from the main notebook)
    with open(DATA_PATH, 'wb') as f:
        pickle.dump({
            'X_train': X_train_np, 'y_train': y_train_np,
            'X_val': X_val_np, 'y_val': y_val_np,
            'X_test': X_test_np, 'y_test': y_test_np,
            'feature_names': FEATURE_NAMES,
            'class_names': CLASS_NAMES,
        }, f)

with open(DATA_PATH, 'rb') as f:
    data = pickle.load(f)
X_train_np, y_train_np = data['X_train'], data['y_train']
X_val_np, y_val_np = data['X_val'], data['y_val']
X_test_np, y_test_np = data['X_test'], data['y_test']
FEATURE_NAMES = data['feature_names']
CLASS_NAMES = data['class_names']
INPUT_DIM = X_train_np.shape[1]
NUM_CLASSES = len(CLASS_NAMES)
```

---

## Stage 1 — Person 1: Teacher Training

Ownership: Cells 5, 6, 7, 9 (architecture, training utilities, difficulty scoring, Configs A/B/C/C2/G teachers)

Trains: 6 teachers
- teachers/rf_500.pkl — Config A (Random Forest, 500 trees)
- teachers/mlp_standard.pt — Config B (Full MLP, no CL)
- teachers/mlp_cl_loss.pt — Config C (CL teacher, loss-based difficulty)
- teachers/mlp_cl_domain.pt — Config C2 (CL teacher, domain-based difficulty)
- teachers/mlp_random_pacing.pt — Config G teacher (random pacing control)
- teachers/mlp_smote.pt — Config I teacher (SMOTE-trained)

Runtime: ~15-20 min

Output also saves evaluation metrics: teachers/teacher_metrics.json (accuracy, F1, ECE for each teacher)

Skeleton:

```python
# (after Stage 0)
# ... copy architecture classes (TeacherMLP, StudentMLP) ...
# ... copy train_standard, train_with_curriculum, evaluate_model ...
# ... copy compute_difficulty_loss_based, compute_difficulty_domain_based ...

# Train Config A
rf = RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train_np, y_train_np)
with open(f'{SHARED}/teachers/rf_500.pkl', 'wb') as f:
    pickle.dump(rf, f)

# Train calibrated RF for later KD use
rf_calib = CalibratedClassifierCV(
    RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1),
    method='isotonic', cv=3
)
rf_calib.fit(X_train_np, y_train_np)
with open(f'{SHARED}/teachers/rf_calibrated.pkl', 'wb') as f:
    pickle.dump(rf_calib, f)

# Train Config B (standard MLP)
teacher_b = TeacherMLP(INPUT_DIM, NUM_CLASSES)
teacher_b = train_standard(teacher_b, X_train_t, y_train_t, X_val_t, y_val_t, ...)
torch.save(teacher_b.state_dict(), f'{SHARED}/teachers/mlp_standard.pt')

# Train Config C (CL loss-based)
loss_order = compute_difficulty_loss_based(...)
teacher_c = TeacherMLP(INPUT_DIM, NUM_CLASSES)
teacher_c = train_with_curriculum(teacher_c, ..., loss_order, ...)
torch.save(teacher_c.state_dict(), f'{SHARED}/teachers/mlp_cl_loss.pt')

# ... same for C2, G, I_teacher ...

# Save metrics JSON
import json
teacher_metrics = {
    'A_RF': evaluate_rf(rf, X_test_np, y_test_np),
    'B_MLP': evaluate_model(teacher_b, X_test_t, y_test_t),
    'C_MLP_cl_loss': evaluate_model(teacher_c, X_test_t, y_test_t),
    # ... etc
}
with open(f'{SHARED}/teachers/teacher_metrics.json', 'w') as f:
    json.dump(teacher_metrics, f, indent=2, default=str)
```

---

## Stage 2 — Person 2: Student Training via KD

Ownership: KD loss function + Configs D, E, E2, F, G_student, I

Depends on: Stage 1 outputs

Trains: 6 students (all 32-16-5 architecture)
- students/scratch.pt — Config D (no KD)
- students/kd_from_rf.pt — Config E (KD from calibrated RF)
- students/kd_from_mlp.pt — Config E2 (KD from standard MLP)
- students/kd_from_cl_mlp.pt — Config F (KD from CL-MLP — CORE CLAIM)
- students/kd_random_pacing.pt — Config G student
- students/kd_from_smote.pt — Config I student

Runtime: ~8-12 min (KD is faster since students are small)

Output also saves: students/student_metrics.json

Skeleton:

```python
# (after Stage 0)
# ... copy StudentMLP class, train_kd function ...

# Load teachers
with open(f'{SHARED}/teachers/rf_calibrated.pkl', 'rb') as f:
    rf_calib = pickle.load(f)

teacher_b = TeacherMLP(INPUT_DIM, NUM_CLASSES)
teacher_b.load_state_dict(torch.load(f'{SHARED}/teachers/mlp_standard.pt'))
teacher_b.to(device).eval()

teacher_c = TeacherMLP(INPUT_DIM, NUM_CLASSES)
teacher_c.load_state_dict(torch.load(f'{SHARED}/teachers/mlp_cl_loss.pt'))
teacher_c.to(device).eval()

# ... load teacher_g, teacher_i ...

# Config D: scratch student
student_d = StudentMLP(INPUT_DIM, (32, 16), NUM_CLASSES)
student_d = train_standard(student_d, ...)
torch.save(student_d.state_dict(), f'{SHARED}/students/scratch.pt')

# Config E: KD from RF
rf_soft = torch.tensor(rf_calib.predict_proba(X_train_np), dtype=torch.float32)
student_e = StudentMLP(INPUT_DIM, (32, 16), NUM_CLASSES)
student_e = train_kd(student_e, rf_soft, ...)
torch.save(student_e.state_dict(), f'{SHARED}/students/kd_from_rf.pt')

# Config E2: KD from standard MLP
student_e2 = StudentMLP(INPUT_DIM, (32, 16), NUM_CLASSES)
student_e2 = train_kd(student_e2, teacher_b, ...)
torch.save(student_e2.state_dict(), f'{SHARED}/students/kd_from_mlp.pt')

# Config F: KD from CL-MLP (CORE CLAIM)
student_f = StudentMLP(INPUT_DIM, (32, 16), NUM_CLASSES)
student_f = train_kd(student_f, teacher_c, ...)
torch.save(student_f.state_dict(), f'{SHARED}/students/kd_from_cl_mlp.pt')

# ... same for G_student, I ...

# Save metrics
student_metrics = {...}
with open(f'{SHARED}/students/student_metrics.json', 'w') as f:
    json.dump(student_metrics, f, indent=2, default=str)
```

---

## Stage 3 — Person 3: Evaluation + Statistics

Ownership: Aggregation, Wilcoxon tests, result tables, per-class F1 figure

Depends on: Stages 1 and 2

Runtime: ~5-10 min

Outputs:
- results/metrics_table.csv — all configs, all metrics, paper-ready
- results/wilcoxon_comparisons.json
- results/per_class_f1.png
- results/confusion_matrix_F.png

Skeleton:

```python
# (after Stage 0)
# Load teacher metrics + student metrics
with open(f'{SHARED}/teachers/teacher_metrics.json') as f:
    teacher_metrics = json.load(f)
with open(f'{SHARED}/students/student_metrics.json') as f:
    student_metrics = json.load(f)

# Combine into one DataFrame
all_metrics = {**teacher_metrics, **student_metrics}
rows = []
for cfg, m in all_metrics.items():
    rows.append({
        'Config': cfg,
        'Accuracy': m['accuracy'],
        'Macro_F1': m['macro_f1'],
        'Normal_F1': m['per_class_f1'][3],  # assuming alphabetical
        'Blackhole_F1': m['per_class_f1'][0],
        # ... etc
    })
df = pd.DataFrame(rows)
df.to_csv(f'{SHARED}/results/metrics_table.csv', index=False)

# Wilcoxon requires multiple seeds — if only 1 seed, compute t-test on paired test samples
# For the demo, just report mean differences
# OR: each person runs stage with different seed and Person 3 aggregates

# Per-class F1 comparison plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 6))
# ... plot per-class F1 for all configs
plt.savefig(f'{SHARED}/results/per_class_f1.png', dpi=150, bbox_inches='tight')

# Confusion matrix heatmap for Config F
# Load student_f, run predictions, plot confusion matrix
student_f = StudentMLP(INPUT_DIM, (32, 16), NUM_CLASSES)
student_f.load_state_dict(torch.load(f'{SHARED}/students/kd_from_cl_mlp.pt'))
# ... confusion matrix code
```

---

## Stage 4 — Person 4: SHAP + INT8 + Benchmarks + Visualization

Ownership: Explainability, hardware feasibility, inference time, Pareto plot

Depends on: Stage 2 (runs in parallel with Stage 3)

Runtime: ~20-25 min

Outputs:
- results/shap_student.json — DeepExplainer feature importance
- results/shap_teacher.json — TreeExplainer feature importance
- results/shap_correlation.json — Spearman correlation + per-attack top features
- results/quantization_results.json — INT8 accuracy delta
- results/inference_benchmarks.json
- results/pareto_frontier.png
- results/shap_summary.png

Skeleton:

```python
# (after Stage 0)
import shap
from scipy.stats import spearmanr

# Load Config F student + RF teacher
student_f = StudentMLP(INPUT_DIM, (32, 16), NUM_CLASSES)
student_f.load_state_dict(torch.load(f'{SHARED}/students/kd_from_cl_mlp.pt'))
student_f.to(device).eval()

with open(f'{SHARED}/teachers/rf_500.pkl', 'rb') as f:
    rf = pickle.load(f)

# DeepExplainer on student
bg_idx = np.random.RandomState(42).choice(len(X_train_np), 100, replace=False)
explain_idx = np.random.RandomState(42).choice(len(X_test_np), 500, replace=False)
background = torch.tensor(X_train_np[bg_idx], dtype=torch.float32).to(device)
to_explain = torch.tensor(X_test_np[explain_idx], dtype=torch.float32).to(device)

student_explainer = shap.DeepExplainer(student_f, background)
student_shap = student_explainer.shap_values(to_explain)
# ... handle both list and 3D array formats ...

# TreeExplainer on RF
rf_explainer = shap.TreeExplainer(rf)
rf_shap = rf_explainer.shap_values(X_test_np[explain_idx])

# Spearman correlation between rankings
student_global = np.abs(np.stack(student_shap_list)).mean(axis=(0, 1))
rf_global = np.abs(np.stack(rf_shap_list)).mean(axis=(0, 1))
rho, rho_p = spearmanr(student_global, rf_global)
# ... save to JSON ...

# INT8 quantization experiment
student_int8 = torch.quantization.quantize_dynamic(
    student_f.cpu().eval(), {nn.Linear}, dtype=torch.qint8
)
# ... evaluate on CPU, save results ...

# Inference benchmarks
# ... copy measure_inference_time_ms function, call on all loaded models ...

# Pareto frontier: need data from Stage 3's metrics_table.csv
# Wait for Stage 3 to finish OR read the individual metric files
df = pd.read_csv(f'{SHARED}/results/metrics_table.csv')
# ... scatter plot size vs accuracy ...
```

---

## Run order and timing

```text
┌─────────┬──────────────────┬────────────────────────┬──────────────────────────────┬─────────────────────────────┐
│ t (min) │     Person 1     │        Person 2        │           Person 3           │          Person 4           │
├─────────┼──────────────────┼────────────────────────┼──────────────────────────────┼─────────────────────────────┤
│ 0-2     │ Stage 0          │ Stage 0                │ Stage 0                      │ Stage 0                     │
├─────────┼──────────────────┼────────────────────────┼──────────────────────────────┼─────────────────────────────┤
│ 2-20    │ Stage 1 training │ Prep code, read papers │ Prep code, write aggregation │ Prep SHAP code              │
├─────────┼──────────────────┼────────────────────────┼──────────────────────────────┼─────────────────────────────┤
│ 20-30   │ Help debug       │ Stage 2 training       │ Continue prep                │ Continue prep               │
├─────────┼──────────────────┼────────────────────────┼──────────────────────────────┼─────────────────────────────┤
│ 30-40   │ Help debug       │ Help debug             │ Stage 3                      │ Stage 4 (first half — SHAP) │
├─────────┼──────────────────┼────────────────────────┼──────────────────────────────┼─────────────────────────────┤
│ 40-55   │ Review results   │ Review results         │ Done                         │ Stage 4 (second half)       │
└─────────┴──────────────────┴────────────────────────┴──────────────────────────────┴─────────────────────────────┘
```

Total clock time: ~55 min (vs ~25 min if 1 person ran everything, but 4 people each produce a distinct artifact).
