# The 4-way split

```
┌────────┬─────────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────┬─────────────────────────────────────────┬──────────┐
│ Person │                         Cell 2 changes                          │                      What runs                       │                Produces                 │ Runtime  │
├────────┼─────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼─────────────────────────────────────────┼──────────┤
│ 1      │ SEEDS=[42], STUDENT_A_HIDDEN=(32,16), QUICK_MODE=False but skip │ Full pipeline on Student A, seed 42                  │ Main results JSON + all figures + saves │ ~25 min  │
│        │  Student B loop                                                 │                                                      │  student_f.pt to Drive                  │          │
├────────┼─────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼─────────────────────────────────────────┼──────────┤
│ 2      │ SEEDS=[123], STUDENT_A_HIDDEN=(32,16), skip Student B           │ Full pipeline on Student A, seed 123                 │ Second-seed JSON (for cross-validation  │ ~25 min  │
│        │                                                                 │                                                      │ with Person 1)                          │          │
├────────┼─────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼─────────────────────────────────────────┼──────────┤
│ 3      │ SEEDS=[42], STUDENT_A_HIDDEN=(64,32) (so it's actually Student  │ Full pipeline but as if Student A is (64,32)         │ Student B results JSON                  │ ~25 min  │
│        │ B), skip Student B loop below                                   │                                                      │                                         │          │
├────────┼─────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼─────────────────────────────────────────┼──────────┤
│ 4      │ KD_T_GRID=[2,3,4,5,6], KD_ALPHA_GRID=[0.3,0.5,0.7,0.9], skips   │ Only Cell 8 (extended grid search, 20 combos) + SHAP │ Hyperparameter heatmap + extended SHAP  │ ~30-40   │
│        │ main multi-seed loop                                            │  deep dive loading Person 1's saved model            │ analysis                                │ min      │
└────────┴─────────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────┴─────────────────────────────────────────┴──────────┘
```

## Specific Cell 2 edits for each person

### Person 1 — Main Track

```
SEEDS = [42]
QUICK_MODE = False
RUN_CICIOT = False
STUDENT_A_HIDDEN = (32, 16)
SAVE_MODELS_TO_DRIVE = True   # new flag
```

Then in Cell 10 after the final re-run, add:

```
from google.colab import drive
drive.mount('/content/drive')
import torch
torch.save(final_models['F_KD_from_CL_MLP'].state_dict(), '/content/drive/MyDrive/cukd_shared/student_f.pt')
torch.save(final_models['B_Full_MLP'].state_dict(), '/content/drive/MyDrive/cukd_shared/teacher_b.pt')
torch.save(final_models['C_CL_MLP_loss'].state_dict(), '/content/drive/MyDrive/cukd_shared/teacher_c.pt')
# Also save the RF
import pickle
with open('/content/drive/MyDrive/cukd_shared/rf_a.pkl', 'wb') as f:
    pickle.dump(final_models['A_RF_500'], f)
```

Also: in Cell 10 right before the Student B loop, add all_seed_results_B = {} and comment out the for seed in SEEDS Student B block.

### Person 2 — Second-seed Cross-validation

Same as Person 1 but change:

```
SEEDS = [123]
```

and do NOT save models (avoid overwriting Person 1's). Just collects the second-seed JSON.

### Person 3 — Student B Track

```
SEEDS = [42]
QUICK_MODE = False
STUDENT_A_HIDDEN = (64, 32)   # trick: put Student B dims where Student A goes
```

Again skip the Student B explicit loop (since STUDENT_A_HIDDEN is now Student B's dims). Rename output files in the save cells:

```
agg_A.to_csv('wsnds_results_student_B_seed42.csv', index=False)
```

### Person 4 — Grid Search + SHAP Deep Dive

Person 4 runs the notebook only through Cell 8 with extended grid, then skips the multi-seed loops entirely and jumps to SHAP. Add this at the top of Cell 10 to replace the multi-seed section:

```
# Person 4: skip multi-seed training, load Person 1's models from Drive
from google.colab import drive
drive.mount('/content/drive')

final_models = {}
student_f = StudentMLP(INPUT_DIM, STUDENT_A_HIDDEN, NUM_CLASSES)
student_f.load_state_dict(torch.load('/content/drive/MyDrive/cukd_shared/student_f.pt'))
student_f.to(device).eval()
final_models['F_KD_from_CL_MLP'] = student_f

import pickle
with open('/content/drive/MyDrive/cukd_shared/rf_a.pkl', 'rb') as f:
    final_models['A_RF_500'] = pickle.load(f)

# Skip multi-seed loops entirely
all_seed_results_A = {}
all_seed_results_B = {}
agg_A = pd.DataFrame()
BEST_T, BEST_ALPHA = 4, 0.7  # default
final_results = {'F_KD_from_CL_MLP': {'accuracy': 0, 'macro_f1': 0}}  # placeholder
```

Then in Cell 8 (grid search), extend the grid:

```
KD_T_GRID = [2, 3, 4, 5, 6]
KD_ALPHA_GRID = [0.3, 0.5, 0.7, 0.9]
```

Person 4 then runs Cell 8 (grid search on a fresh teacher they train locally) + Cell 12 (SHAP) + builds a grid heatmap.

Person 4 also adds SHAP extensions:

```
# Bootstrap SHAP stability — run 5 times with different backgrounds
spearman_rhos = []
for bootstrap_i in range(5):
    rng_bs = np.random.RandomState(bootstrap_i)
    bg = X_train_shap_t[rng_bs.choice(len(X_train_shap_t), 100, replace=False)].to(device)
    explainer_bs = shap.DeepExplainer(student_for_shap, bg)
    shap_vals_bs = explainer_bs.shap_values(to_explain)
    # ... compute ranking correlation
```

## Coordination — single shared Drive folder

Before anyone starts, one person creates a Drive folder /MyDrive/cukd_shared/ and shares it with the other 3. That's the only coordination needed.

## Start order

1. Person 1 starts first (anyone else can still start in parallel, they just won't have Person 1's saved models yet)
2. Persons 2 and 3 start at the same time as Person 1 — they don't need anyone's models
3. Person 4 waits ~15 minutes for Person 1's models to appear in Drive, then starts. During that wait, Person 4 prepares their grid search extensions in a scratch cell
4. All 4 finish roughly together at ~30-40 min

## Outputs Monday will have

```
┌──────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│   From   │                                                        What you show                                                         │
├──────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Person 1 │ Main results (Student A, seed 42), all figures, the core Wilcoxon comparisons (limited to 1 seed → use paired-t placeholder) │
├──────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Person 2 │ Second-seed confirmation of Person 1's numbers — sanity check                                                                │
├──────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Person 3 │ Student B comparison → Pareto point (smaller vs larger student)                                                              │
├──────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Person 4 │ Hyperparameter heatmap + SHAP stability analysis                                                                             │
└──────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

This gives you 4 genuinely distinct demo artifacts for Monday, not 4 people looking at the same output.

## One honest caveat

Person 4's grid search uses a standard MLP teacher (not CL-trained), and Person 1 already does a similar grid search in Cell 8 (just with smaller range). So there's ~5 minutes of duplicated teacher training. Acceptable overhead.

Want me to produce 4 pre-modified notebooks, one per person, with the edits already applied? That way nobody has to manually edit cells — they just download their assigned notebook and run it.
