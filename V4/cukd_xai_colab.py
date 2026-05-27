# ============================================================================
#  !!!! WARNING — READ BEFORE EDITING !!!!
#
#  This .py file is the Python SOURCE for the notebook. The canonical runnable
#  artifact is `cukd_xai_colab.ipynb`, which contains a STATUS BANNER cell at
#  the top that is NOT present in this .py file.
#
#  DO NOT run `python3 make_notebook.py` after editing this file — it will
#  regenerate the .ipynb and OVERWRITE the Status Banner. You will lose your
#  primary orientation tool inside Colab.
#
#  If you need to propagate a code edit, do it by editing the .ipynb directly
#  (cell-by-cell) or ask Claude to re-apply the Status Banner after regenerating.
#
#  Bug fixes applied April 11, 2026 are marked `# FIXED 2026-04-11:` inline.
#  See RESUME_HERE.md for full project context.
# ============================================================================

# ============================================================================
# CuKD-XAI: Curriculum-Guided Knowledge Distillation with Explainability
# for Lightweight WSN Intrusion Detection
#
# COMPLETE IMPLEMENTATION (v2)
# - 10 experiment configurations (A, B, C, D, E, E2, F, G, H, I)
# - 5-seed statistical validation with Wilcoxon signed-rank test
# - Both difficulty scoring methods (loss-based + domain-knowledge)
# - Both student sizes (32-16-5 and 64-32-5)
# - T/alpha grid search for KD
# - DeepExplainer (student) + TreeExplainer (teacher) SHAP comparison
# - Actual INT8 quantization via torch.quantization
# - FLOPs calculation + real inference time benchmarks
# - Expected Calibration Error (ECE) for teacher quality
# - Training loss curves + Pareto frontier + confusion matrix heatmaps
# - CICIoT2023 generalizability (conditional, gated behind flag)
#
# Author: Nishant Harkut (2023IMG-040), ABV-IIITM Gwalior
# ============================================================================

# ============================================================================
# CELL 1: Install dependencies
# ============================================================================
# !pip install -q shap scikit-learn pandas numpy matplotlib seaborn torch imbalanced-learn

# ============================================================================
# CELL 2: Imports and global config
# ============================================================================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              f1_score, classification_report, confusion_matrix)
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import time
import json
import os
import tempfile

# ----------------------------------------------------------------------------
# EXPERIMENT CONFIGURATION
# ----------------------------------------------------------------------------
SEEDS_V2_FINAL5 = [42, 123, 456, 789, 1001]
SEEDS_PUBLICATION10 = [42, 123, 456, 789, 1001, 2024, 3141, 5678, 8192, 9999]
RUN_PUBLICATION_10_SEEDS = False          # Set True only for final WSN-DS paper run
SEEDS = SEEDS_PUBLICATION10 if RUN_PUBLICATION_10_SEEDS else SEEDS_V2_FINAL5
N_SEEDS = len(SEEDS)                      # Set to 1 for quick debugging runs
QUICK_MODE = False                        # True = single seed, no grid search
RUN_CICIOT = False                        # True = also run CICIoT2023 (slow)
CICIOT_PATH = 'CICIoT2023.csv'            # Path if using CICIoT2023
WSNDS_PATH = 'WSN-DS.csv'                 # Path to WSN-DS
RUN_LEGACY_V23_EXPERIMENTS = False     # Set True only to reproduce old v2.3 full notebook outputs

# Final-project paths and run guards. Defaults are deliberately off so a
# normal Run All loads data/functions without launching long legacy/final runs.
FINAL_RUN_ROOT = 'final_runs'
RESUME_FINAL_RUNS = True
FINAL_RUN_MODE = 'none'                 # Run All selector: none, edge_smoke, edge_final, wsnds_final, qat_profile, shap_alignment
RUN_EDGEIIOT_ML_SMOKE = False
RUN_EDGEIIOT_ML_5SEED = False
RUN_WSNDS_RESUME_FINAL = False
RUN_QAT_FOR_BEST_STUDENTS = False
RUN_DEPLOYMENT_PROFILE = False
RUN_SHAP_ALIGNMENT_FINAL = False
RUN_WSNDS_KD_GRID_FINAL = True
RUN_EDGEIIOT_KD_GRID_FINAL = True
EDGEIIOT_KD_GRID_RF_TREES = 300
EDGEIIOT_KD_GRID_EPOCHS = 15
EDGEIIOT_ML_PATH = os.environ.get(
    'EDGEIIOT_ML_PATH',
    'Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv'
)
EDGEIIOT_TARGET_COL = 'Attack_type'        # 15-class multiclass task
EDGEIIOT_SEEDS = SEEDS_V2_FINAL5

# KD hyperparameter grid (from Benaddi et al. 2025)
KD_T_GRID = [2, 3, 4, 5]
KD_ALPHA_GRID = [0.5, 0.7, 0.9]
KD_T_DEFAULT = 4
KD_ALPHA_DEFAULT = 0.7

# Training hyperparameters (from Benaddi et al. 2025)
TRAIN_CONFIG = {
    'epochs': 30,
    'batch_size': 256,
    'lr': 1e-3,
    'weight_decay': 1e-3,
    'patience': 8,
}
# Teacher dropout is set in TeacherMLP constructor, not passed through TRAIN_CONFIG

# CL pacing stages: list of (fraction, epochs)
#
# FIXED 2026-04-11 (v2.3): Two variants are now tested in parallel to rule out
# compute-budget unfairness as a confounder.
#
#   FAIR   — 3+3+24 = 30 total epochs, matches Config B's budget exactly.
#            Fair comparison: "does CL help when we hold total compute constant?"
#   EXT    — 5+5+30 = 40 total epochs, gives CL extra training time.
#            Generous comparison: "does CL help when we give it a larger budget?"
#
# Original (v2.0, broken) was [(0.33,7),(0.66,7),(1.0,11)] = 25 total with
# only 11 epochs on the full distribution, which badly under-trained Stage 3.
CL_STAGES_FAIR = [(0.33, 3), (0.66, 3), (1.0, 24)]       # 30 total, matches B
CL_STAGES_EXT  = [(0.33, 5), (0.66, 5), (1.0, 30)]       # 40 total, extended
CL_STAGES = CL_STAGES_FAIR  # default when called with no explicit stages arg

# Student architectures to test
STUDENT_A_HIDDEN = (32, 16)   # Ultra-compact: 1,189 params
STUDENT_B_HIDDEN = (64, 32)   # Balanced: 3,397 params

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"PyTorch: {torch.__version__}")
print(f"Seeds: {SEEDS}")
print(f"Quick mode: {QUICK_MODE}")
print(f"Legacy v2.3 full run: {RUN_LEGACY_V23_EXPERIMENTS}")
print(f"Final Run All mode: {FINAL_RUN_MODE}")

if QUICK_MODE:
    SEEDS = SEEDS[:1]
    N_SEEDS = 1
    print("QUICK MODE: running single seed only")

# ============================================================================
# CELL 3: Load WSN-DS dataset
# ============================================================================
# Upload WSN-DS.csv manually via Colab Files panel before running this cell.
# Or use kaggle API:
#   !pip install -q kaggle
#   !mkdir -p ~/.kaggle && cp /content/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
#   !kaggle datasets download -d bassamkasasbeh1/wsnds && unzip -q wsnds.zip

df = pd.read_csv(WSNDS_PATH)
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
df.columns = df.columns.str.strip()
print(f"\nFirst row:\n{df.head(1).to_string()}")

# ============================================================================
# CELL 4: Preprocess WSN-DS
# ============================================================================
# Identify target column (handle common variations)
target_candidates = ['Attack type', 'Attack_Type', 'attack_type', 'Attack Type', 'class']
target_col = None
for cand in target_candidates:
    if cand in df.columns:
        target_col = cand
        break
if target_col is None:
    target_col = df.columns[-1]
print(f"Target column: {target_col}")

# Drop Id column (non-informative, can bias the model)
for id_col in ['id', 'Id', 'ID']:
    if id_col in df.columns:
        df = df.drop(id_col, axis=1)
        print(f"Dropped {id_col}")
        break

# Clean target values
df[target_col] = df[target_col].astype(str).str.strip()

# Label encode (alphabetical)
le = LabelEncoder()
df[target_col] = le.fit_transform(df[target_col])
CLASS_NAMES = le.classes_.tolist()
NUM_CLASSES = len(CLASS_NAMES)
print(f"Classes: {CLASS_NAMES}")
print(f"Mapping: {dict(zip(CLASS_NAMES, range(NUM_CLASSES)))}")

# Features / target
X_all = df.drop(target_col, axis=1).values.astype(np.float32)
y_all = df[target_col].values.astype(np.int64)
FEATURE_NAMES = df.drop(target_col, axis=1).columns.tolist()
INPUT_DIM = X_all.shape[1]
print(f"Input dim: {INPUT_DIM}")
print(f"Class distribution: {dict(zip(CLASS_NAMES, np.bincount(y_all).tolist()))}")

# Standardize (fit on all data, then split — consistent with SOTA paper)
scaler = StandardScaler()
X_all_std = scaler.fit_transform(X_all)

# ============================================================================
# CELL 5: Model architectures
# ============================================================================
class TeacherMLP(nn.Module):
    """128-256-128-5 MLP. ~69,893 params with BatchNorm."""
    def __init__(self, input_dim: int = 17, num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.67),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StudentMLP(nn.Module):
    """Configurable small student MLP. No BatchNorm (INT8-friendly)."""
    def __init__(self, input_dim: int = 17, hidden_dims: tuple = (32, 16),
                 num_classes: int = 5):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_kb(model: nn.Module, dtype_bytes: int = 4) -> float:
    return count_params(model) * dtype_bytes / 1024


def compute_flops_mlp(input_dim: int, hidden_dims: tuple, num_classes: int) -> int:
    """Compute FLOPs for a linear-ReLU MLP (one forward pass per sample).

    Each linear layer: 2 * in * out (multiply + add), ReLU is ~out ops.
    """
    dims = [input_dim] + list(hidden_dims) + [num_classes]
    flops = 0
    for i in range(len(dims) - 1):
        flops += 2 * dims[i] * dims[i + 1]  # MAC operations
        flops += dims[i + 1]                # bias
        if i < len(dims) - 2:
            flops += dims[i + 1]            # ReLU
    return flops


# Verify architecture param counts
_t = TeacherMLP(INPUT_DIM, NUM_CLASSES)
_sa = StudentMLP(INPUT_DIM, STUDENT_A_HIDDEN, NUM_CLASSES)
_sb = StudentMLP(INPUT_DIM, STUDENT_B_HIDDEN, NUM_CLASSES)
print(f"Teacher MLP: {count_params(_t)} params ({model_size_kb(_t):.2f} KB fp32, {model_size_kb(_t, 1):.2f} KB int8)")
print(f"Student A {STUDENT_A_HIDDEN}: {count_params(_sa)} params ({model_size_kb(_sa):.2f} KB fp32, {model_size_kb(_sa, 1):.2f} KB int8)")
print(f"Student B {STUDENT_B_HIDDEN}: {count_params(_sb)} params ({model_size_kb(_sb):.2f} KB fp32, {model_size_kb(_sb, 1):.2f} KB int8)")
print(f"Student A FLOPs: {compute_flops_mlp(INPUT_DIM, STUDENT_A_HIDDEN, NUM_CLASSES)}")
print(f"Student B FLOPs: {compute_flops_mlp(INPUT_DIM, STUDENT_B_HIDDEN, NUM_CLASSES)}")
del _t, _sa, _sb

# ============================================================================
# CELL 6: Training, evaluation, and measurement utilities
# ============================================================================
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _batched_predict(model: nn.Module, X: torch.Tensor, batch_size: int = 4096):
    """Memory-safe batched inference returning predicted class indices."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size].to(device)
            preds.append(model(batch).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def _batched_probs(model: nn.Module, X: torch.Tensor, batch_size: int = 4096):
    """Memory-safe batched inference returning class probabilities."""
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size].to(device)
            probs.append(F.softmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(probs)


def evaluate_model(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> dict:
    """Return dict of evaluation metrics.

    FIXED 2026-04-11 (v2.3): Now also returns per-class precision and recall
    (previously only per-class F1). Useful for analyzing the Grayhole<->Blackhole
    confusion pattern seen in the v2.0 run.
    """
    preds = _batched_predict(model, X)
    y_np = y.cpu().numpy() if torch.is_tensor(y) else np.asarray(y)

    acc = accuracy_score(y_np, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_np, preds, average='macro', zero_division=0
    )
    per_class_prec, per_class_rec, per_class_f1_arr, _ = precision_recall_fscore_support(
        y_np, preds, average=None, zero_division=0
    )
    cm = confusion_matrix(y_np, preds)
    return {
        'accuracy': float(acc),
        'macro_precision': float(prec),
        'macro_recall': float(rec),
        'macro_f1': float(f1),
        'per_class_precision': per_class_prec.tolist(),
        'per_class_recall': per_class_rec.tolist(),
        'per_class_f1': per_class_f1_arr.tolist(),
        'confusion_matrix': cm.tolist(),
    }


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray,
                                n_bins: int = 15) -> float:
    """Compute Expected Calibration Error (ECE) for a probabilistic classifier."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if mask.sum() > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += (mask.sum() / len(y_true)) * abs(bin_acc - bin_conf)
    return float(ece)


def measure_inference_time_ms(model: nn.Module, X: torch.Tensor,
                                batch_size: int = 1, n_runs: int = 500) -> dict:
    """Measure per-sample inference latency (ms) on CPU and GPU."""
    model.eval()
    X_sample = X[:batch_size].to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(20):
            _ = model(X_sample)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # GPU timing
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = model(X_sample)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        gpu_time = (time.perf_counter() - start) / n_runs * 1000  # ms per batch

    # CPU timing
    model_cpu = copy.deepcopy(model).cpu()
    X_cpu = X[:batch_size].cpu()
    model_cpu.eval()
    with torch.no_grad():
        for _ in range(10):
            _ = model_cpu(X_cpu)
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = model_cpu(X_cpu)
        cpu_time = (time.perf_counter() - start) / n_runs * 1000

    return {
        'gpu_ms_per_batch': gpu_time,
        'cpu_ms_per_batch': cpu_time,
        'batch_size': batch_size,
    }


def train_standard(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor,
                   X_val: torch.Tensor, y_val: torch.Tensor,
                   class_weights: torch.Tensor = None,
                   epochs: int = 30, batch_size: int = 256, lr: float = 1e-3,
                   weight_decay: float = 1e-3, patience: int = 8,
                   return_loss_curve: bool = False, verbose: bool = False):
    """Standard supervised training with cosine LR schedule and early stopping."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_ds = TensorDataset(X_train.to(device), y_train.to(device))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    X_val_d = X_val.to(device)
    y_val_np = y_val.cpu().numpy()

    loss_curve = []
    val_curve = []
    best_val = 0.0
    best_state = None
    bad = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        nb = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            nb += 1
        scheduler.step()
        epoch_loss /= max(nb, 1)
        loss_curve.append(epoch_loss)

        preds = _batched_predict(model, X_val_d)
        val_f1 = f1_score(y_val_np, preds, average='macro')
        val_curve.append(val_f1)

        if val_f1 > best_val:
            best_val = val_f1
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

        if verbose:
            print(f"Epoch {epoch+1}: loss={epoch_loss:.4f} val_f1={val_f1:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    if return_loss_curve:
        return model, {'loss': loss_curve, 'val_f1': val_curve}
    return model


def train_with_curriculum(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                          difficulty_order: np.ndarray,
                          X_val: torch.Tensor, y_val: torch.Tensor,
                          stages=CL_STAGES,
                          class_weights: torch.Tensor = None,
                          batch_size: int = 256, lr: float = 1e-3,
                          weight_decay: float = 1e-3,
                          patience: int = 8,
                          return_loss_curve: bool = False,
                          verbose: bool = False):
    """Curriculum learning with discrete stage pacing.

    stages: list of (fraction, epochs). Samples with difficulty_order[:n]
    are used in each stage (easy-first if difficulty_order is loss-ascending).
    """
    # FIXED 2026-04-11 (v2.2): Previously one global optimizer + cosine scheduler
    # spanning all stages. With CL_STAGES = [(0.33,7),(0.66,7),(1.0,11)] the LR
    # was nearly half-decayed by the time Stage 3 reached the full dataset. We
    # now create a FRESH optimizer + per-stage cosine schedule at each stage
    # transition, so Stage 3 gets a full cosine cycle on the full data.
    #
    # FIXED 2026-04-11 (v2.3): Added early stopping with `patience` parameter
    # (defaults to 8, same as train_standard). Without this, CL got unlimited
    # training time while train_standard (Config B) had early stopping —
    # an unfair compute comparison. Patience is GLOBAL across stages so the
    # two functions have symmetric compute budgets.
    model = model.to(device)

    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    X_d = X.to(device)
    y_d = y.to(device)
    X_val_d = X_val.to(device)
    y_val_np = y_val.cpu().numpy()

    loss_curve = []
    val_curve = []
    best_val = 0.0
    best_state = None
    bad = 0
    stopped_early = False
    n_total = len(X)

    for stage_idx, (frac, n_epochs) in enumerate(stages):
        if stopped_early:
            break
        n_use = int(n_total * frac)
        idx = torch.tensor(np.asarray(difficulty_order[:n_use]),
                           dtype=torch.long, device=device)
        stage_ds = TensorDataset(X_d[idx], y_d[idx])
        stage_loader = DataLoader(stage_ds, batch_size=batch_size, shuffle=True)

        # Per-stage fresh optimizer + cosine schedule scoped to this stage's epochs
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        if verbose:
            print(f"  Stage {stage_idx+1}: {n_use}/{n_total} samples, {n_epochs} epochs, fresh optimizer+scheduler")

        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0.0
            nb = 0
            for xb, yb in stage_loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                nb += 1
            scheduler.step()
            epoch_loss /= max(nb, 1)
            loss_curve.append(epoch_loss)

            preds = _batched_predict(model, X_val_d)
            val_f1 = f1_score(y_val_np, preds, average='macro')
            val_curve.append(val_f1)

            if val_f1 > best_val:
                best_val = val_f1
                best_state = copy.deepcopy(model.state_dict())
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    if verbose:
                        print(f"  Early stopping at stage {stage_idx+1}, epoch {epoch+1}")
                    stopped_early = True
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    if return_loss_curve:
        return model, {'loss': loss_curve, 'val_f1': val_curve}
    return model


def train_kd(student: nn.Module,
             teacher_source,
             X_train: torch.Tensor, y_train: torch.Tensor,
             X_val: torch.Tensor, y_val: torch.Tensor,
             T: float = KD_T_DEFAULT, alpha: float = KD_ALPHA_DEFAULT,
             class_weights: torch.Tensor = None,
             epochs: int = 30, batch_size: int = 256,
             lr: float = 1e-3, weight_decay: float = 1e-3,
             patience: int = 8, verbose: bool = False):
    """Knowledge distillation training.

    teacher_source may be:
      - a callable / nn.Module that produces logits (MLP teacher)
      - a raw probability tensor of shape (N, num_classes) (e.g., RF)

    Loss = alpha * T^2 * KL(softmax(student/T) || softmax(teacher/T))
         + (1 - alpha) * CE(student, y)
    """
    student = student.to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if class_weights is not None:
        class_weights = class_weights.to(device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    X_d = X_train.to(device)
    y_d = y_train.to(device)
    X_val_d = X_val.to(device)
    y_val_np = y_val.cpu().numpy()

    # Precompute soft targets (temperature-softened, matches student side)
    if callable(teacher_source):
        teacher_source.eval()
        soft_list = []
        with torch.no_grad():
            for i in range(0, len(X_d), 4096):
                logits = teacher_source(X_d[i:i + 4096])
                soft_list.append(F.softmax(logits / T, dim=1).detach())
        soft_targets = torch.cat(soft_list, dim=0)
    else:
        # Raw probability tensor (e.g., RF predict_proba). Convert to
        # pseudo-logits via log, then temperature-soften to match student side.
        raw = teacher_source.to(device)
        pseudo_logits = torch.log(raw.clamp(min=1e-8))
        soft_targets = F.softmax(pseudo_logits / T, dim=1).detach()

    ds = TensorDataset(X_d, y_d, soft_targets)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    best_val = 0.0
    best_state = None
    bad = 0

    for epoch in range(epochs):
        student.train()
        for xb, yb, sb in loader:
            optimizer.zero_grad()
            logits = student(xb)
            log_soft_s = F.log_softmax(logits / T, dim=1)
            kd_term = F.kl_div(log_soft_s, sb, reduction='batchmean') * (T * T)
            ce_term = ce_loss(logits, yb)
            loss = alpha * kd_term + (1 - alpha) * ce_term
            loss.backward()
            optimizer.step()
        scheduler.step()

        preds = _batched_predict(student, X_val_d)
        val_f1 = f1_score(y_val_np, preds, average='macro')
        if val_f1 > best_val:
            best_val = val_f1
            best_state = copy.deepcopy(student.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose:
            print(f"KD epoch {epoch+1}: val_f1={val_f1:.4f}")

    if best_state is not None:
        student.load_state_dict(best_state)
    return student


def quantize_dynamic_int8(model: nn.Module) -> nn.Module:
    """Apply PyTorch dynamic INT8 quantization to Linear layers."""
    model_cpu = copy.deepcopy(model).cpu().eval()
    try:
        quantized = torch.quantization.quantize_dynamic(
            model_cpu, {nn.Linear}, dtype=torch.qint8
        )
        return quantized
    except Exception as e:
        print(f"Dynamic INT8 quantization failed: {e}")
        return model_cpu


def model_size_on_disk_kb(model: nn.Module) -> float:
    """Measure serialized state_dict size without using a shared temp path."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix='cukd_xai_model_', suffix='.pt', delete=False) as tmp:
            tmp_path = tmp.name
        torch.save(model.state_dict(), tmp_path)
        return os.path.getsize(tmp_path) / 1024
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ============================================================================
# CELL 7: Difficulty scoring — both methods
# ============================================================================
def compute_difficulty_loss_based(X_train: torch.Tensor, y_train: torch.Tensor,
                                   input_dim: int, num_classes: int,
                                   seed: int = 42) -> np.ndarray:
    """Loss-based difficulty: train probe for 3 epochs, sort by per-sample loss."""
    set_seed(seed)
    probe = StudentMLP(input_dim, (64, 32), num_classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)

    X_d = X_train.to(device)
    y_d = y_train.to(device)
    ds = TensorDataset(X_d, y_d)
    loader = DataLoader(ds, batch_size=512, shuffle=True)

    # Train probe
    ce = nn.CrossEntropyLoss()
    probe.train()
    for _ in range(3):
        for xb, yb in loader:
            opt.zero_grad()
            ce(probe(xb), yb).backward()
            opt.step()

    # Per-sample loss
    ce_none = nn.CrossEntropyLoss(reduction='none')
    probe.eval()
    losses = []
    with torch.no_grad():
        for i in range(0, len(X_d), 4096):
            logits = probe(X_d[i:i + 4096])
            losses.append(ce_none(logits, y_d[i:i + 4096]).cpu().numpy())
    per_sample_loss = np.concatenate(losses)
    return np.argsort(per_sample_loss)  # ascending: easy first


def compute_difficulty_domain_based(y_train: np.ndarray,
                                     class_names: list) -> np.ndarray:
    """Domain-knowledge difficulty for WSN-DS LEACH attacks.

    Per verified per-class F1 across 5 published WSN-DS papers:
      Tier 1 (easiest): Normal + Blackhole (F1 0.98-0.99)
      Tier 2 (medium):  Grayhole + Flooding (F1 0.95-0.97)
      Tier 3 (hardest): Scheduling / TDMA   (F1 0.93-0.96)
    """
    tier_map = {
        'Normal':   1,
        'Blackhole': 1,
        'Grayhole': 2,
        'Flooding': 2,
        'TDMA':     3,
        'Scheduling': 3,
    }
    # Default tier 2 for unknown labels
    sample_tier = np.array(
        [tier_map.get(class_names[int(lbl)], 2) for lbl in y_train]
    )
    # Sort by tier (ascending) — easy first
    # Shuffle within tier for randomness
    rng = np.random.RandomState(42)
    order = np.lexsort((rng.rand(len(sample_tier)), sample_tier))
    return order


# ============================================================================
# CELL 8: KD hyperparameter grid search (run once on seed 42)
# ============================================================================
def kd_grid_search(teacher: nn.Module, student_hidden: tuple,
                   X_train: torch.Tensor, y_train: torch.Tensor,
                   X_val: torch.Tensor, y_val: torch.Tensor,
                   class_weights: torch.Tensor,
                   T_grid=KD_T_GRID, alpha_grid=KD_ALPHA_GRID) -> dict:
    """Grid search over KD hyperparameters. Returns best (T, alpha, val_f1)."""
    best = {'T': KD_T_DEFAULT, 'alpha': KD_ALPHA_DEFAULT, 'val_f1': 0.0}
    results = []
    for T in T_grid:
        for a in alpha_grid:
            student = StudentMLP(INPUT_DIM, student_hidden, NUM_CLASSES)
            student = train_kd(
                student, teacher, X_train, y_train, X_val, y_val,
                T=T, alpha=a, class_weights=class_weights,
                epochs=20,  # shorter for grid search
            )
            val_metrics = evaluate_model(student, X_val, y_val)
            val_f1 = val_metrics['macro_f1']
            results.append({'T': T, 'alpha': a, 'val_f1': val_f1})
            print(f"  T={T}, alpha={a}: val_f1={val_f1:.4f}")
            if val_f1 > best['val_f1']:
                best = {'T': T, 'alpha': a, 'val_f1': val_f1}
    return {'best': best, 'all': results}


# ============================================================================
# CELL 9: Single-seed experiment runner — ALL configs A-I + H
# ============================================================================
def run_all_configs(seed: int,
                    X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray,
                    student_hidden: tuple = STUDENT_A_HIDDEN,
                    kd_T: float = KD_T_DEFAULT,
                    kd_alpha: float = KD_ALPHA_DEFAULT,
                    return_models: bool = False,
                    verbose: bool = True) -> dict:
    """Run all 10 configurations for one seed. Returns metrics dict."""
    set_seed(seed)
    if verbose:
        print(f"\n{'='*60}\nSeed {seed} — Student {student_hidden}\n{'='*60}")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    # Class weights (inverse frequency)
    counts = np.bincount(y_train, minlength=NUM_CLASSES)
    class_weights = torch.tensor(
        len(y_train) / (NUM_CLASSES * np.maximum(counts, 1)), dtype=torch.float32
    )

    results = {}
    models = {}

    # ----- Config A: RF baseline -----
    if verbose: print("[A] RF baseline...")
    t0 = time.perf_counter()
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=15, random_state=seed, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_time = time.perf_counter() - t0

    rf_preds = rf.predict(X_test)
    rf_probs_test = rf.predict_proba(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_prec, rf_rec, rf_f1, _ = precision_recall_fscore_support(
        y_test, rf_preds, average='macro', zero_division=0
    )
    rf_per_class = f1_score(y_test, rf_preds, average=None, zero_division=0)
    import pickle
    rf_size_kb = len(pickle.dumps(rf)) / 1024
    results['A_RF_500'] = {
        'accuracy': float(rf_acc),
        'macro_f1': float(rf_f1),
        'macro_precision': float(rf_prec),
        'macro_recall': float(rf_rec),
        'per_class_f1': rf_per_class.tolist(),
        'ece': expected_calibration_error(rf_probs_test, y_test),
        'model_size_kb': rf_size_kb,
        'train_time_sec': rf_time,
    }
    models['A_RF_500'] = rf

    # ----- Config B: Full MLP baseline -----
    if verbose: print("[B] Full MLP baseline...")
    t0 = time.perf_counter()
    teacher_b = TeacherMLP(INPUT_DIM, NUM_CLASSES)
    teacher_b, b_curve = train_standard(
        teacher_b, X_train_t, y_train_t, X_val_t, y_val_t,
        class_weights=class_weights, return_loss_curve=True,
        **TRAIN_CONFIG
    )
    b_time = time.perf_counter() - t0
    m_b = evaluate_model(teacher_b, X_test_t, y_test_t)
    m_b['ece'] = expected_calibration_error(
        _batched_probs(teacher_b, X_test_t), y_test
    )
    m_b['params'] = count_params(teacher_b)
    m_b['model_size_kb'] = model_size_kb(teacher_b)
    m_b['train_time_sec'] = b_time
    m_b['loss_curve'] = b_curve
    results['B_Full_MLP'] = m_b
    models['B_Full_MLP'] = teacher_b

    # ----- Difficulty scoring -----
    if verbose: print("[Difficulty] Loss-based scoring...")
    loss_order = compute_difficulty_loss_based(
        X_train_t, y_train_t, INPUT_DIM, NUM_CLASSES, seed=seed
    )
    domain_order = compute_difficulty_domain_based(y_train, CLASS_NAMES)

    # FIXED 2026-04-11 (v2.3): Train TWO CL variants side-by-side.
    # C_fair   — CL with fair compute budget (matches Config B exactly)
    # C_ext    — CL with extended compute budget (+33% total epochs)
    # This lets us separate "CL doesn't help" from "CL needs more compute".
    # Config F (KD student) is similarly forked into F_fair and F_ext.

    # ----- Config C (fair): CL teacher, loss-based difficulty, matched budget -----
    if verbose: print("[C_fair] CL-trained MLP (loss-based, fair budget)...")
    t0 = time.perf_counter()
    teacher_c_fair = TeacherMLP(INPUT_DIM, NUM_CLASSES)
    teacher_c_fair, c_fair_curve = train_with_curriculum(
        teacher_c_fair, X_train_t, y_train_t, loss_order, X_val_t, y_val_t,
        stages=CL_STAGES_FAIR, class_weights=class_weights,
        return_loss_curve=True
    )
    c_fair_time = time.perf_counter() - t0
    m_c_fair = evaluate_model(teacher_c_fair, X_test_t, y_test_t)
    m_c_fair['ece'] = expected_calibration_error(
        _batched_probs(teacher_c_fair, X_test_t), y_test
    )
    m_c_fair['params'] = count_params(teacher_c_fair)
    m_c_fair['model_size_kb'] = model_size_kb(teacher_c_fair)
    m_c_fair['train_time_sec'] = c_fair_time
    m_c_fair['loss_curve'] = c_fair_curve
    results['C_CL_MLP_loss_fair'] = m_c_fair
    models['C_CL_MLP_loss_fair'] = teacher_c_fair

    # ----- Config C (ext): CL teacher, loss-based difficulty, extended budget -----
    if verbose: print("[C_ext] CL-trained MLP (loss-based, extended budget)...")
    t0 = time.perf_counter()
    teacher_c_ext = TeacherMLP(INPUT_DIM, NUM_CLASSES)
    teacher_c_ext, c_ext_curve = train_with_curriculum(
        teacher_c_ext, X_train_t, y_train_t, loss_order, X_val_t, y_val_t,
        stages=CL_STAGES_EXT, class_weights=class_weights,
        return_loss_curve=True
    )
    c_ext_time = time.perf_counter() - t0
    m_c_ext = evaluate_model(teacher_c_ext, X_test_t, y_test_t)
    m_c_ext['ece'] = expected_calibration_error(
        _batched_probs(teacher_c_ext, X_test_t), y_test
    )
    m_c_ext['params'] = count_params(teacher_c_ext)
    m_c_ext['model_size_kb'] = model_size_kb(teacher_c_ext)
    m_c_ext['train_time_sec'] = c_ext_time
    m_c_ext['loss_curve'] = c_ext_curve
    results['C_CL_MLP_loss_ext'] = m_c_ext
    models['C_CL_MLP_loss_ext'] = teacher_c_ext

    # FIXED 2026-04-11 (v2.3): alias C_CL_MLP_loss DETERMINISTICALLY to the FAIR
    # variant (not data-dependent). A data-dependent alias breaks aggregation
    # across seeds because different seeds might pick different underlying variants,
    # turning the aggregate into a chimera. The "fair" variant is also the more
    # reviewer-defensible primary result, so we use it as the canonical CL teacher.
    teacher_c = teacher_c_fair
    results['C_CL_MLP_loss'] = {**m_c_fair, '_source': 'fair (alias)'}
    models['C_CL_MLP_loss'] = teacher_c

    # ----- Config C2: CL teacher with domain-based difficulty (fair budget) -----
    if verbose: print("[C2] CL-trained MLP (domain-based, fair budget)...")
    teacher_c2 = TeacherMLP(INPUT_DIM, NUM_CLASSES)
    teacher_c2 = train_with_curriculum(
        teacher_c2, X_train_t, y_train_t, domain_order, X_val_t, y_val_t,
        stages=CL_STAGES_FAIR, class_weights=class_weights,
    )
    m_c2 = evaluate_model(teacher_c2, X_test_t, y_test_t)
    m_c2['ece'] = expected_calibration_error(
        _batched_probs(teacher_c2, X_test_t), y_test
    )
    m_c2['params'] = count_params(teacher_c2)
    m_c2['model_size_kb'] = model_size_kb(teacher_c2)
    results['C2_CL_MLP_domain'] = m_c2
    models['C2_CL_MLP_domain'] = teacher_c2

    # ----- Config D: Small MLP from scratch -----
    if verbose: print(f"[D] Small MLP {student_hidden} from scratch...")
    t0 = time.perf_counter()
    student_d = StudentMLP(INPUT_DIM, student_hidden, NUM_CLASSES)
    student_d = train_standard(
        student_d, X_train_t, y_train_t, X_val_t, y_val_t,
        class_weights=class_weights, **TRAIN_CONFIG
    )
    d_time = time.perf_counter() - t0
    m_d = evaluate_model(student_d, X_test_t, y_test_t)
    m_d['params'] = count_params(student_d)
    m_d['model_size_kb'] = model_size_kb(student_d)
    m_d['model_size_kb_int8'] = model_size_kb(student_d, 1)
    m_d['flops'] = compute_flops_mlp(INPUT_DIM, student_hidden, NUM_CLASSES)
    m_d['train_time_sec'] = d_time
    results['D_Small_MLP'] = m_d
    models['D_Small_MLP'] = student_d

    # ----- Config E: KD from calibrated RF -----
    if verbose: print("[E] KD from calibrated RF...")
    t0 = time.perf_counter()
    rf_calib = CalibratedClassifierCV(
        RandomForestClassifier(
            n_estimators=500, max_depth=15, random_state=seed, n_jobs=-1
        ),
        method='isotonic', cv=3
    )
    rf_calib.fit(X_train, y_train)
    rf_soft = torch.tensor(
        rf_calib.predict_proba(X_train), dtype=torch.float32
    )

    student_e = StudentMLP(INPUT_DIM, student_hidden, NUM_CLASSES)
    student_e = train_kd(
        student_e, rf_soft, X_train_t, y_train_t, X_val_t, y_val_t,
        T=kd_T, alpha=kd_alpha, class_weights=class_weights,
        epochs=TRAIN_CONFIG['epochs'], batch_size=TRAIN_CONFIG['batch_size']
    )
    e_time = time.perf_counter() - t0
    m_e = evaluate_model(student_e, X_test_t, y_test_t)
    m_e['params'] = count_params(student_e)
    m_e['model_size_kb'] = model_size_kb(student_e)
    m_e['flops'] = compute_flops_mlp(INPUT_DIM, student_hidden, NUM_CLASSES)
    m_e['ece'] = expected_calibration_error(
        _batched_probs(student_e, X_test_t), y_test
    )
    m_e['train_time_sec'] = e_time
    results['E_KD_from_RF'] = m_e
    models['E_KD_from_RF'] = student_e

    # ----- Config E2: KD from standard MLP teacher (no CL) -----
    if verbose: print("[E2] KD from standard MLP teacher...")
    student_e2 = StudentMLP(INPUT_DIM, student_hidden, NUM_CLASSES)
    student_e2 = train_kd(
        student_e2, teacher_b, X_train_t, y_train_t, X_val_t, y_val_t,
        T=kd_T, alpha=kd_alpha, class_weights=class_weights,
        epochs=TRAIN_CONFIG['epochs'], batch_size=TRAIN_CONFIG['batch_size']
    )
    m_e2 = evaluate_model(student_e2, X_test_t, y_test_t)
    m_e2['params'] = count_params(student_e2)
    m_e2['model_size_kb'] = model_size_kb(student_e2)
    m_e2['flops'] = compute_flops_mlp(INPUT_DIM, student_hidden, NUM_CLASSES)
    m_e2['ece'] = expected_calibration_error(
        _batched_probs(student_e2, X_test_t), y_test
    )
    results['E2_KD_from_MLP'] = m_e2
    models['E2_KD_from_MLP'] = student_e2

    # FIXED 2026-04-11 (v2.3): Config F is now two configs — F_fair and F_ext —
    # distilling from the two CL teacher variants. Comparing F_fair vs E2 tells
    # us whether CL helps at equal budget. F_ext vs E2 tells us whether CL helps
    # with extra budget.

    # ----- Config F_fair: KD from CL-trained MLP (fair budget) -----
    if verbose: print("[F_fair] KD from fair-budget CL-MLP...")
    student_f_fair = StudentMLP(INPUT_DIM, student_hidden, NUM_CLASSES)
    student_f_fair = train_kd(
        student_f_fair, teacher_c_fair, X_train_t, y_train_t, X_val_t, y_val_t,
        T=kd_T, alpha=kd_alpha, class_weights=class_weights,
        epochs=TRAIN_CONFIG['epochs'], batch_size=TRAIN_CONFIG['batch_size']
    )
    m_f_fair = evaluate_model(student_f_fair, X_test_t, y_test_t)
    m_f_fair['params'] = count_params(student_f_fair)
    m_f_fair['model_size_kb'] = model_size_kb(student_f_fair)
    m_f_fair['flops'] = compute_flops_mlp(INPUT_DIM, student_hidden, NUM_CLASSES)
    m_f_fair['ece'] = expected_calibration_error(
        _batched_probs(student_f_fair, X_test_t), y_test
    )
    results['F_KD_from_CL_MLP_fair'] = m_f_fair
    models['F_KD_from_CL_MLP_fair'] = student_f_fair

    # ----- Config F_ext: KD from CL-trained MLP (extended budget) -----
    if verbose: print("[F_ext] KD from extended-budget CL-MLP...")
    student_f_ext = StudentMLP(INPUT_DIM, student_hidden, NUM_CLASSES)
    student_f_ext = train_kd(
        student_f_ext, teacher_c_ext, X_train_t, y_train_t, X_val_t, y_val_t,
        T=kd_T, alpha=kd_alpha, class_weights=class_weights,
        epochs=TRAIN_CONFIG['epochs'], batch_size=TRAIN_CONFIG['batch_size']
    )
    m_f_ext = evaluate_model(student_f_ext, X_test_t, y_test_t)
    m_f_ext['params'] = count_params(student_f_ext)
    m_f_ext['model_size_kb'] = model_size_kb(student_f_ext)
    m_f_ext['flops'] = compute_flops_mlp(INPUT_DIM, student_hidden, NUM_CLASSES)
    m_f_ext['ece'] = expected_calibration_error(
        _batched_probs(student_f_ext, X_test_t), y_test
    )
    results['F_KD_from_CL_MLP_ext'] = m_f_ext
    models['F_KD_from_CL_MLP_ext'] = student_f_ext

    # FIXED 2026-04-11 (v2.3): Alias F_KD_from_CL_MLP DETERMINISTICALLY to F_fair.
    # Same reasoning as C_CL_MLP_loss above: a data-dependent alias breaks
    # multi-seed aggregation. F_fair is also the reviewer-defensible primary.
    results['F_KD_from_CL_MLP'] = {**m_f_fair, '_source': 'fair (alias)'}
    models['F_KD_from_CL_MLP'] = student_f_fair

    # ----- Config G: KD from random-pacing MLP (control, fair budget) -----
    if verbose: print("[G] KD from random-pacing MLP (control)...")
    random_order = np.random.RandomState(seed).permutation(len(X_train))
    teacher_g = TeacherMLP(INPUT_DIM, NUM_CLASSES)
    teacher_g = train_with_curriculum(
        teacher_g, X_train_t, y_train_t, random_order, X_val_t, y_val_t,
        stages=CL_STAGES_FAIR, class_weights=class_weights,
    )
    # Compute ECE for Config G teacher
    student_g = StudentMLP(INPUT_DIM, student_hidden, NUM_CLASSES)
    student_g = train_kd(
        student_g, teacher_g, X_train_t, y_train_t, X_val_t, y_val_t,
        T=kd_T, alpha=kd_alpha, class_weights=class_weights,
        epochs=TRAIN_CONFIG['epochs'], batch_size=TRAIN_CONFIG['batch_size']
    )
    m_g = evaluate_model(student_g, X_test_t, y_test_t)
    m_g['params'] = count_params(student_g)
    m_g['model_size_kb'] = model_size_kb(student_g)
    m_g['flops'] = compute_flops_mlp(INPUT_DIM, student_hidden, NUM_CLASSES)
    m_g['ece'] = expected_calibration_error(
        _batched_probs(student_g, X_test_t), y_test
    )
    # Also save teacher G ECE for the "CL improves calibration" analysis
    m_g['teacher_ece'] = expected_calibration_error(
        _batched_probs(teacher_g, X_test_t), y_test
    )
    results['G_KD_random_pacing'] = m_g
    models['G_KD_random_pacing'] = student_g

    # ----- Config I: KD from SMOTE-trained MLP teacher -----
    if verbose: print("[I] KD from SMOTE-trained MLP teacher...")
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=seed, k_neighbors=3)
        X_tr_smote, y_tr_smote = smote.fit_resample(X_train, y_train)
        X_tr_smote_t = torch.tensor(X_tr_smote, dtype=torch.float32)
        y_tr_smote_t = torch.tensor(y_tr_smote, dtype=torch.long)

        teacher_i = TeacherMLP(INPUT_DIM, NUM_CLASSES)
        teacher_i = train_standard(
            teacher_i, X_tr_smote_t, y_tr_smote_t, X_val_t, y_val_t,
            class_weights=None,  # SMOTE balances; no class weights needed
            **TRAIN_CONFIG
        )
        student_i = StudentMLP(INPUT_DIM, student_hidden, NUM_CLASSES)
        student_i = train_kd(
            student_i, teacher_i, X_train_t, y_train_t, X_val_t, y_val_t,
            T=kd_T, alpha=kd_alpha, class_weights=class_weights,
            epochs=TRAIN_CONFIG['epochs'], batch_size=TRAIN_CONFIG['batch_size']
        )
        m_i = evaluate_model(student_i, X_test_t, y_test_t)
        m_i['params'] = count_params(student_i)
        m_i['model_size_kb'] = model_size_kb(student_i)
        m_i['flops'] = compute_flops_mlp(INPUT_DIM, student_hidden, NUM_CLASSES)
        m_i['ece'] = expected_calibration_error(
            _batched_probs(student_i, X_test_t), y_test
        )
        m_i['teacher_ece'] = expected_calibration_error(
            _batched_probs(teacher_i, X_test_t), y_test
        )
        results['I_KD_from_SMOTE_MLP'] = m_i
        models['I_KD_from_SMOTE_MLP'] = student_i
    except ImportError:
        print("  imblearn not installed, skipping Config I")
    except Exception as ex:
        print(f"  Config I failed: {ex}")

    # Print seed summary
    if verbose:
        print(f"\nSeed {seed} summary (macro F1):")
        for cfg, m in results.items():
            print(f"  {cfg:25s} {m['macro_f1']:.4f}")

    if return_models:
        return results, models
    return results


# ============================================================================
# CELL 10: Run WSN-DS experiments over multiple seeds
# ============================================================================
# First split (same split for all seeds to ensure comparability)
X_trainval, X_test_np, y_trainval, y_test_np = train_test_split(
    X_all_std, y_all, test_size=0.15, random_state=42, stratify=y_all
)
X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
    X_trainval, y_trainval, test_size=0.1765, random_state=42, stratify=y_trainval
)
# Final split: ~70 / 15 / 15 stratified
print(f"Train: {X_train_np.shape}, Val: {X_val_np.shape}, Test: {X_test_np.shape}")

# Optional: run KD hyperparameter grid search once on seed 42.
# Disabled by default for the final project route so a top-to-bottom notebook
# execution does not waste hours on the historical v2.3 sweep before the final
# resumable entrypoints are available.
BEST_T, BEST_ALPHA = KD_T_DEFAULT, KD_ALPHA_DEFAULT
grid_result = None
all_seed_results_A = {}
all_seed_results_B = {}
final_results, final_models = {}, {}

if RUN_LEGACY_V23_EXPERIMENTS:
    if not QUICK_MODE:
        print("\n>>> KD hyperparameter grid search (seed 42, Student A)")
        set_seed(42)
        _Xtr = torch.tensor(X_train_np, dtype=torch.float32)
        _ytr = torch.tensor(y_train_np, dtype=torch.long)
        _Xv = torch.tensor(X_val_np, dtype=torch.float32)
        _yv = torch.tensor(y_val_np, dtype=torch.long)
        _cw = torch.tensor(
            len(y_train_np) / (NUM_CLASSES * np.maximum(np.bincount(y_train_np, minlength=NUM_CLASSES), 1)),
            dtype=torch.float32,
        )

        # Need a teacher for grid search — use a quick standard MLP
        _t_search = TeacherMLP(INPUT_DIM, NUM_CLASSES)
        _t_search = train_standard(
            _t_search, _Xtr, _ytr, _Xv, _yv,
            class_weights=_cw, epochs=15, batch_size=256, lr=1e-3
        )

        grid_result = kd_grid_search(
            _t_search, STUDENT_A_HIDDEN, _Xtr, _ytr, _Xv, _yv, _cw
        )
        BEST_T = grid_result['best']['T']
        BEST_ALPHA = grid_result['best']['alpha']
        print(f"\nBest KD hyperparameters: T={BEST_T}, alpha={BEST_ALPHA} "
              f"(val F1 {grid_result['best']['val_f1']:.4f})")
        del _t_search, _Xtr, _ytr, _Xv, _yv, _cw

    # Multi-seed runs — Student A (32-16-5)
    print(f"\n>>> Running {N_SEEDS} seeds with Student A {STUDENT_A_HIDDEN}")
    for seed in SEEDS:
        all_seed_results_A[seed] = run_all_configs(
            seed,
            X_train_np, y_train_np,
            X_val_np, y_val_np,
            X_test_np, y_test_np,
            student_hidden=STUDENT_A_HIDDEN,
            kd_T=BEST_T, kd_alpha=BEST_ALPHA,
            verbose=True,
        )

    # Optional: Student B (64-32-5) — for Pareto analysis
    if not QUICK_MODE:
        print(f"\n>>> Running {N_SEEDS} seeds with Student B {STUDENT_B_HIDDEN}")
        for seed in SEEDS:
            all_seed_results_B[seed] = run_all_configs(
                seed,
                X_train_np, y_train_np,
                X_val_np, y_val_np,
                X_test_np, y_test_np,
                student_hidden=STUDENT_B_HIDDEN,
                kd_T=BEST_T, kd_alpha=BEST_ALPHA,
                verbose=False,
            )

    # Keep models from the last seed of Student A for SHAP / inference benchmarks
    print("\n>>> Re-running final seed to capture models for SHAP/benchmarks...")
    final_results, final_models = run_all_configs(
        SEEDS[-1],
        X_train_np, y_train_np,
        X_val_np, y_val_np,
        X_test_np, y_test_np,
        student_hidden=STUDENT_A_HIDDEN,
        kd_T=BEST_T, kd_alpha=BEST_ALPHA,
        return_models=True,
        verbose=False,
    )
else:
    print("\nSkipping legacy v2.3 multi-seed run. Use the final resumable entrypoints below.")

# ============================================================================
# CELL 11: Aggregate multi-seed statistics + Wilcoxon tests
# ============================================================================
def aggregate_multi_seed(seed_results: dict) -> pd.DataFrame:
    """Build DataFrame of mean ± std per config across seeds.

    Handles the case where a config may be missing from some seeds
    (e.g., Config I if SMOTE failed on one seed).
    """
    configs = set()
    for r in seed_results.values():
        configs.update(r.keys())
    configs = sorted(configs)

    rows = []
    for cfg in configs:
        accs = [seed_results[s][cfg]['accuracy'] for s in seed_results
                if cfg in seed_results[s]]
        f1s = [seed_results[s][cfg]['macro_f1'] for s in seed_results
               if cfg in seed_results[s]]
        per_class = [seed_results[s][cfg]['per_class_f1'] for s in seed_results
                     if cfg in seed_results[s]]

        if len(accs) == 0:
            continue  # Config missing from every seed

        per_class_arr = np.array(per_class)  # (n_seeds, n_classes)

        row = {
            'Config': cfg,
            'Accuracy_mean': float(np.mean(accs)),
            'Accuracy_std': float(np.std(accs)),
            'MacroF1_mean': float(np.mean(f1s)),
            'MacroF1_std': float(np.std(f1s)),
            'n_seeds': len(accs),
        }
        for i, name in enumerate(CLASS_NAMES):
            row[f'{name}_F1_mean'] = float(per_class_arr[:, i].mean())
            row[f'{name}_F1_std'] = float(per_class_arr[:, i].std())

        # Find a seed that has this config (not guaranteed to be the first)
        first_hit = next(
            (seed_results[s][cfg] for s in seed_results if cfg in seed_results[s]),
            {}
        )
        row['params'] = first_hit.get('params', None)
        row['size_kb'] = first_hit.get('model_size_kb', None)
        rows.append(row)
    return pd.DataFrame(rows)


def wilcoxon_test(seed_results: dict, cfg_a: str, cfg_b: str,
                  metric: str = 'macro_f1') -> dict:
    """Paired Wilcoxon signed-rank test between two configurations."""
    vals_a = [seed_results[s][cfg_a][metric] for s in seed_results
              if cfg_a in seed_results[s]]
    vals_b = [seed_results[s][cfg_b][metric] for s in seed_results
              if cfg_b in seed_results[s]]
    if len(vals_a) != len(vals_b) or len(vals_a) < 2:
        return {'stat': None, 'p': None, 'diff': None,
                'verdict': 'insufficient data'}
    diffs = np.array(vals_a) - np.array(vals_b)
    if np.all(diffs == 0):
        return {'stat': 0.0, 'p': 1.0, 'diff': 0.0,
                'verdict': 'identical'}
    try:
        stat, p = wilcoxon(vals_a, vals_b, zero_method='wilcox')
    except Exception:
        stat, p = None, None
    verdict = '—'
    if p is not None:
        if p < 0.01:
            verdict = '** p<0.01'
        elif p < 0.05:
            verdict = '* p<0.05'
        else:
            verdict = 'not significant'
    return {
        'stat': float(stat) if stat is not None else None,
        'p': float(p) if p is not None else None,
        'diff_mean': float(diffs.mean()),
        'verdict': verdict,
    }


agg_A = pd.DataFrame()
wilcoxon_results = {}
if RUN_LEGACY_V23_EXPERIMENTS and all_seed_results_A:
    print("\n" + "=" * 60)
    print("MULTI-SEED AGGREGATE RESULTS (Student A)")
    print("=" * 60)
    agg_A = aggregate_multi_seed(all_seed_results_A)
    print(agg_A[['Config', 'Accuracy_mean', 'Accuracy_std',
                 'MacroF1_mean', 'MacroF1_std', 'n_seeds']].to_string(index=False))

    print("\n" + "=" * 60)
    print("KEY WILCOXON COMPARISONS (Student A)")
    print("=" * 60)
    key_comparisons = [
        # Teacher-level CL question
        ('C_CL_MLP_loss_fair', 'B_Full_MLP', "Does CL help teacher at FAIR budget? (C_fair vs B)"),
        ('C_CL_MLP_loss_ext',  'B_Full_MLP', "Does CL help teacher at EXT budget? (C_ext vs B)"),
        # Student-level CL question (the core claim)
        ('F_KD_from_CL_MLP_fair', 'E2_KD_from_MLP', "Does CL cascade at FAIR budget? (F_fair vs E2)"),
        ('F_KD_from_CL_MLP_ext',  'E2_KD_from_MLP', "Does CL cascade at EXT budget? (F_ext vs E2)"),
        # KD-effectiveness question
        ('F_KD_from_CL_MLP', 'D_Small_MLP', "Does KD beat scratch? (F vs D)"),
        ('E2_KD_from_MLP',   'D_Small_MLP', "Does KD work at all? (E2 vs D)"),
        # Difficulty-ordering vs pacing
        ('F_KD_from_CL_MLP', 'G_KD_random_pacing', "Order vs random pacing? (F vs G)"),
        # CL vs SMOTE alternative
        ('F_KD_from_CL_MLP', 'I_KD_from_SMOTE_MLP', "CL vs SMOTE teacher? (F vs I)"),
        # Tree vs NN teacher
        ('E_KD_from_RF', 'E2_KD_from_MLP',    "RF teacher vs MLP teacher? (E vs E2)"),
    ]
    # FIXED 2026-04-11: Wilcoxon results were previously only printed to stdout.
    # Now we persist them to a dict so they can be saved to the final JSON output.
    wilcoxon_results = {}
    for a, b, desc in key_comparisons:
        if a not in agg_A['Config'].values or b not in agg_A['Config'].values:
            print(f"{desc}: one or both configs missing, skipping")
            wilcoxon_results[f"{a}_vs_{b}"] = {"status": "skipped", "desc": desc}
            continue
        w = wilcoxon_test(all_seed_results_A, a, b)
        a_mean = float(agg_A[agg_A['Config']==a]['MacroF1_mean'].iloc[0])
        b_mean = float(agg_A[agg_A['Config']==b]['MacroF1_mean'].iloc[0])
        print(f"{desc}")
        print(f"  {a}: {a_mean:.4f}")
        print(f"  {b}: {b_mean:.4f}")
        print(f"  diff: {w['diff_mean']:+.4f}  |  p={w['p']}  |  {w['verdict']}\n")
        wilcoxon_results[f"{a}_vs_{b}"] = {
            "desc": desc,
            "a_config": a,
            "b_config": b,
            "a_macro_f1_mean": a_mean,
            "b_macro_f1_mean": b_mean,
            "diff_mean": w['diff_mean'],
            "stat": w['stat'],
            "p": w['p'],
            "verdict": w['verdict'],
        }

else:
    print("\nSkipping legacy v2.3 aggregation/Wilcoxon block.")

# ============================================================================
# CELL 12: Legacy v2.3 reporting block (guarded)
# ============================================================================
if RUN_LEGACY_V23_EXPERIMENTS and final_models:
    # ============================================================================
    # CELL 12: SHAP analysis — DeepExplainer on student + TreeExplainer on RF teacher
    # ============================================================================
    print("\n" + "=" * 60)
    print("SHAP ANALYSIS")
    print("=" * 60)

    shap_results = {}
    try:
        import shap
        from scipy.stats import spearmanr

        # Use Config F student + Config A RF teacher from final_models
        if 'F_KD_from_CL_MLP' not in final_models or 'A_RF_500' not in final_models:
            raise RuntimeError("Required models missing from final_models dict")
        student_for_shap = final_models['F_KD_from_CL_MLP']
        rf_for_shap = final_models['A_RF_500']

        X_train_shap_t = torch.tensor(X_train_np, dtype=torch.float32)
        X_test_shap_t = torch.tensor(X_test_np, dtype=torch.float32)

        rng = np.random.RandomState(42)
        bg_idx = rng.choice(len(X_train_shap_t), 100, replace=False)
        explain_idx = rng.choice(len(X_test_shap_t), 500, replace=False)

        # ---- Student: DeepExplainer ----
        print("Computing SHAP for student (DeepExplainer)...")
        student_for_shap.eval()
        background = X_train_shap_t[bg_idx].to(device)
        to_explain = X_test_shap_t[explain_idx].to(device)

        student_explainer = shap.DeepExplainer(student_for_shap, background)
        student_shap_values = student_explainer.shap_values(to_explain)

        # shap_values: list of (n_samples, n_features) arrays, one per class
        # (or a single 3D array in newer SHAP versions)
        if isinstance(student_shap_values, np.ndarray) and student_shap_values.ndim == 3:
            # Shape: (n_samples, n_features, n_classes) — transpose
            student_shap_list = [student_shap_values[:, :, i] for i in range(NUM_CLASSES)]
        else:
            student_shap_list = student_shap_values

        student_global = np.abs(np.stack(student_shap_list)).mean(axis=(0, 1))
        student_imp_df = pd.DataFrame({
            'feature': FEATURE_NAMES,
            'student_shap': student_global,
        }).sort_values('student_shap', ascending=False)

        print("\nStudent top-10 features (global):")
        print(student_imp_df.head(10).to_string(index=False))

        # Per-class top features for student
        print("\nStudent per-class top-3:")
        student_per_class = {}
        for i, name in enumerate(CLASS_NAMES):
            class_imp = np.abs(student_shap_list[i]).mean(axis=0)
            top_idx = np.argsort(class_imp)[::-1][:3]
            top_feats = [(FEATURE_NAMES[j], float(class_imp[j])) for j in top_idx]
            student_per_class[name] = top_feats
            print(f"  {name}: {[(f, round(v, 4)) for f, v in top_feats]}")

        # ---- RF Teacher: TreeExplainer ----
        print("\nComputing SHAP for RF teacher (TreeExplainer)...")
        rf_explainer = shap.TreeExplainer(rf_for_shap)
        # TreeExplainer on 500 samples is fast
        rf_shap_values = rf_explainer.shap_values(X_test_np[explain_idx])

        if isinstance(rf_shap_values, np.ndarray) and rf_shap_values.ndim == 3:
            rf_shap_list = [rf_shap_values[:, :, i] for i in range(NUM_CLASSES)]
        else:
            rf_shap_list = rf_shap_values

        rf_global = np.abs(np.stack(rf_shap_list)).mean(axis=(0, 1))
        teacher_imp_df = pd.DataFrame({
            'feature': FEATURE_NAMES,
            'teacher_shap': rf_global,
        }).sort_values('teacher_shap', ascending=False)

        print("\nTeacher top-10 features (global):")
        print(teacher_imp_df.head(10).to_string(index=False))

        # ---- Compare rankings ----
        compare_df = pd.DataFrame({
            'feature': FEATURE_NAMES,
            'student_shap': student_global,
            'teacher_shap': rf_global,
        })
        # Rank correlation (spearman) between student and teacher importance
        student_ranks = pd.Series(student_global).rank(ascending=False)
        teacher_ranks = pd.Series(rf_global).rank(ascending=False)
        rho, rho_p = spearmanr(student_ranks, teacher_ranks)
        print(f"\nFeature ranking agreement (Spearman): rho={rho:.4f}, p={rho_p:.4e}")
        print(f"Interpretation: {'Student preserves teacher reasoning' if rho > 0.7 else 'Student diverges from teacher reasoning'}")

        # FIXED 2026-04-11: Per-class Spearman correlation — strengthens the novel
        # "feature alignment gap" finding by showing whether misalignment is uniform
        # across attack classes or concentrated in specific classes.
        per_class_spearman = {}
        for class_idx, class_name in enumerate(CLASS_NAMES):
            student_class_imp = np.abs(student_shap_list[class_idx]).mean(axis=0)
            rf_class_imp = np.abs(rf_shap_list[class_idx]).mean(axis=0)
            s_ranks = pd.Series(student_class_imp).rank(ascending=False)
            t_ranks = pd.Series(rf_class_imp).rank(ascending=False)
            try:
                class_rho, class_p = spearmanr(s_ranks, t_ranks)
            except Exception:
                class_rho, class_p = float('nan'), float('nan')
            per_class_spearman[class_name] = {
                'rho': float(class_rho) if class_rho == class_rho else None,
                'p': float(class_p) if class_p == class_p else None,
            }
            print(f"  {class_name:12s} rho={class_rho:+.4f}  p={class_p:.4e}")

        # Save student SHAP summary plot
        try:
            shap.summary_plot(
                student_shap_list, X_test_np[explain_idx],
                feature_names=FEATURE_NAMES, show=False,
                class_names=CLASS_NAMES, plot_size=(10, 6)
            )
            plt.tight_layout()
            plt.savefig('shap_summary_student.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("Saved shap_summary_student.png")
        except Exception as e:
            print(f"Failed to save student summary plot: {e}")

        # FIXED 2026-04-11 (v2.3): Bootstrap SHAP stability test. Repeatedly compute
        # the global Spearman with different random background samples. This gives a
        # confidence interval on rho, defending against "your ~0 correlation might
        # just be SHAP sampling noise".
        print("\nBootstrap SHAP stability (5 different backgrounds)...")
        bootstrap_rhos = []
        bootstrap_ps = []
        for bs_i in range(5):
            bs_rng = np.random.RandomState(42 + bs_i * 37)
            bs_bg_idx = bs_rng.choice(len(X_train_shap_t), 100, replace=False)
            bs_explain_idx = bs_rng.choice(len(X_test_shap_t), 500, replace=False)

            try:
                bs_bg = X_train_shap_t[bs_bg_idx].to(device)
                bs_expl = X_test_shap_t[bs_explain_idx].to(device)
                bs_explainer = shap.DeepExplainer(student_for_shap, bs_bg)
                bs_shap_vals = bs_explainer.shap_values(bs_expl)
                if isinstance(bs_shap_vals, np.ndarray) and bs_shap_vals.ndim == 3:
                    bs_shap_list = [bs_shap_vals[:, :, i] for i in range(NUM_CLASSES)]
                else:
                    bs_shap_list = bs_shap_vals
                bs_student_global = np.abs(np.stack(bs_shap_list)).mean(axis=(0, 1))

                bs_rf_shap = rf_explainer.shap_values(X_test_np[bs_explain_idx])
                if isinstance(bs_rf_shap, np.ndarray) and bs_rf_shap.ndim == 3:
                    bs_rf_shap_list = [bs_rf_shap[:, :, i] for i in range(NUM_CLASSES)]
                else:
                    bs_rf_shap_list = bs_rf_shap
                bs_rf_global = np.abs(np.stack(bs_rf_shap_list)).mean(axis=(0, 1))

                bs_s_ranks = pd.Series(bs_student_global).rank(ascending=False)
                bs_t_ranks = pd.Series(bs_rf_global).rank(ascending=False)
                bs_rho, bs_p = spearmanr(bs_s_ranks, bs_t_ranks)
                bootstrap_rhos.append(float(bs_rho))
                bootstrap_ps.append(float(bs_p))
                print(f"  bootstrap {bs_i+1}/5: rho={bs_rho:+.4f}  p={bs_p:.4e}")
            except Exception as ex:
                print(f"  bootstrap {bs_i+1}/5 failed: {ex}")

        if bootstrap_rhos:
            bs_rho_mean = float(np.mean(bootstrap_rhos))
            bs_rho_std = float(np.std(bootstrap_rhos))
            print(f"\nBootstrap Spearman (mean ± std): {bs_rho_mean:+.4f} ± {bs_rho_std:.4f}")
            print(f"95% bootstrap CI (approx): [{bs_rho_mean - 1.96*bs_rho_std:+.4f}, {bs_rho_mean + 1.96*bs_rho_std:+.4f}]")
        else:
            bs_rho_mean, bs_rho_std = None, None

        shap_results = {
            'student_global_importance': student_imp_df.to_dict('records'),
            'teacher_global_importance': teacher_imp_df.to_dict('records'),
            'student_per_class_top3': student_per_class,
            'ranking_agreement_spearman': float(rho),
            'ranking_agreement_p': float(rho_p),
            'per_class_spearman': per_class_spearman,
            'bootstrap_spearman_values': bootstrap_rhos,
            'bootstrap_spearman_ps': bootstrap_ps,
            'bootstrap_spearman_mean': bs_rho_mean,
            'bootstrap_spearman_std': bs_rho_std,
        }
    except ImportError:
        print("shap not installed. Install with: !pip install shap")
    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        import traceback
        traceback.print_exc()

    # ============================================================================
    # CELL 13: Actual INT8 quantization experiment — SWEEP over all student configs
    # ============================================================================
    # FIXED 2026-04-11: Previously quantized only Config F. Now sweeps over every
    # student config present in final_models, so we can see whether the ~3% F1 drop
    # we saw on F is unique to F (because its teacher was broken) or systematic
    # across all students.
    print("\n" + "=" * 60)
    print("INT8 QUANTIZATION SWEEP — all student configs")
    print("=" * 60)

    # FIXED 2026-04-11 (v2.3): Quantize both F_fair and F_ext separately so the paper
    # can report INT8 quantization effect on each CL budget variant independently.
    # We keep F_KD_from_CL_MLP in the list for backward-compat with tooling that
    # reads the canonical name, but its values duplicate F_fair.
    STUDENT_CONFIGS_TO_QUANTIZE = [
        'D_Small_MLP',
        'E_KD_from_RF',
        'E2_KD_from_MLP',
        'F_KD_from_CL_MLP_fair',
        'F_KD_from_CL_MLP_ext',
        'F_KD_from_CL_MLP',   # alias — duplicates fair, keeps tooling compatibility
        'G_KD_random_pacing',
        'I_KD_from_SMOTE_MLP',
    ]

    quant_results = {}  # keyed by config name

    X_test_cpu = torch.tensor(X_test_np, dtype=torch.float32)

    for cfg_name in STUDENT_CONFIGS_TO_QUANTIZE:
        if cfg_name not in final_models:
            print(f"  [{cfg_name}] skipped — not in final_models")
            continue
        try:
            m_fp32 = final_models[cfg_name]
            fp32_size = model_size_on_disk_kb(m_fp32)
            m_int8 = quantize_dynamic_int8(m_fp32)
            int8_size = model_size_on_disk_kb(m_int8)

            m_int8.eval()
            with torch.no_grad():
                preds_int8 = []
                for i in range(0, len(X_test_cpu), 4096):
                    logits = m_int8(X_test_cpu[i:i + 4096])
                    preds_int8.append(logits.argmax(dim=1).numpy())
            preds_int8 = np.concatenate(preds_int8)
            acc_int8 = float(accuracy_score(y_test_np, preds_int8))
            f1_int8 = float(f1_score(y_test_np, preds_int8, average='macro', zero_division=0))

            acc_fp32 = float(final_results[cfg_name]['accuracy'])
            f1_fp32 = float(final_results[cfg_name]['macro_f1'])

            quant_results[cfg_name] = {
                'fp32_size_kb': fp32_size,
                'int8_size_kb': int8_size,
                'size_reduction_pct': (1 - int8_size / fp32_size) * 100,
                'fp32_accuracy': acc_fp32,
                'int8_accuracy': acc_int8,
                'fp32_macro_f1': f1_fp32,
                'int8_macro_f1': f1_int8,
                'acc_delta_pct': (acc_int8 - acc_fp32) * 100,
                'f1_delta_pct': (f1_int8 - f1_fp32) * 100,
            }
            print(f"  [{cfg_name}] fp32 {fp32_size:.2f}KB F1={f1_fp32:.4f} → "
                  f"int8 {int8_size:.2f}KB F1={f1_int8:.4f} "
                  f"(F1 {((f1_int8 - f1_fp32)*100):+.3f}%)")
        except Exception as e:
            print(f"  [{cfg_name}] INT8 quantization failed: {e}")
            quant_results[cfg_name] = {'error': str(e)}

    # ============================================================================
    # CELL 14: Inference time & throughput benchmarks
    # ============================================================================
    print("\n" + "=" * 60)
    print("INFERENCE TIME BENCHMARKS")
    print("=" * 60)

    X_bench = torch.tensor(X_test_np[:1024], dtype=torch.float32)
    bench_results = {}

    # FIXED 2026-04-11 (v2.3): Benchmarks now include both F_fair and F_ext variants
    # in addition to the canonical F alias, so we can report latency for each CL
    # budget variant. All students share the same architecture (student_hidden),
    # so they have identical latency, but we still run all to verify consistency.
    candidate_models = [
        ('Teacher_MLP', 'B_Full_MLP'),
        ('Student_D_scratch', 'D_Small_MLP'),
        ('Student_E_KD_RF', 'E_KD_from_RF'),
        ('Student_E2_KD_MLP', 'E2_KD_from_MLP'),
        ('Student_F_KD_CL', 'F_KD_from_CL_MLP'),
        ('Student_F_KD_CL_fair', 'F_KD_from_CL_MLP_fair'),
        ('Student_F_KD_CL_ext', 'F_KD_from_CL_MLP_ext'),
        ('Student_G_rand_pacing', 'G_KD_random_pacing'),
        ('Student_I_KD_SMOTE', 'I_KD_from_SMOTE_MLP'),
    ]
    models_to_bench = {
        name: final_models[key] for name, key in candidate_models if key in final_models
    }

    for name, m in models_to_bench.items():
        timing = measure_inference_time_ms(m, X_bench, batch_size=1, n_runs=200)
        params = count_params(m)
        print(f"{name}:")
        print(f"  Params:     {params}")
        print(f"  Size fp32:  {model_size_kb(m):.2f} KB")
        print(f"  Size int8:  {model_size_kb(m, 1):.2f} KB")
        print(f"  GPU latency (batch=1): {timing['gpu_ms_per_batch']:.3f} ms")
        print(f"  CPU latency (batch=1): {timing['cpu_ms_per_batch']:.3f} ms")
        bench_results[name] = {
            'params': params,
            'size_kb_fp32': model_size_kb(m),
            'size_kb_int8': model_size_kb(m, 1),
            **timing,
        }

    # ============================================================================
    # CELL 15: Visualization — per-class F1, confusion matrix, Pareto, loss curves
    # ============================================================================
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)

    # --- Per-class F1 comparison (mean ± std across seeds) ---
    # FIXED 2026-04-11 (v2.3): Use explicit fair/ext CL variants (not the alias) so
    # the bar chart shows genuinely distinct configurations with no duplicates.
    configs_to_plot = [
        'B_Full_MLP',
        'C_CL_MLP_loss_fair',
        'C_CL_MLP_loss_ext',
        'D_Small_MLP',
        'E_KD_from_RF',
        'E2_KD_from_MLP',
        'F_KD_from_CL_MLP_fair',
        'F_KD_from_CL_MLP_ext',
    ]
    configs_present = [c for c in configs_to_plot if c in agg_A['Config'].values]

    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(NUM_CLASSES)
    width = 0.13
    colors = plt.cm.tab10(np.linspace(0, 1, len(configs_present)))
    for i, cfg in enumerate(configs_present):
        means = [agg_A[agg_A['Config'] == cfg][f'{name}_F1_mean'].iloc[0] for name in CLASS_NAMES]
        stds = [agg_A[agg_A['Config'] == cfg][f'{name}_F1_std'].iloc[0] for name in CLASS_NAMES]
        ax.bar(x_pos + i * width, means, width, yerr=stds, label=cfg,
               color=colors[i], capsize=2)
    ax.set_xticks(x_pos + width * (len(configs_present) - 1) / 2)
    ax.set_xticklabels(CLASS_NAMES, rotation=15)
    ax.set_ylabel('Per-class Macro F1')
    ax.set_title(f'Per-class F1 across configurations (mean ± std, {N_SEEDS} seeds)')
    ax.legend(fontsize=8, loc='lower right', ncol=2)
    ax.set_ylim(0.80, 1.005)
    plt.tight_layout()
    plt.savefig('per_class_f1.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved per_class_f1.png")

    # FIXED 2026-04-11 (v2.3): Generate confusion matrices for BOTH Config E (winner)
    # and Config F (core claim). Previous version only had F.
    for cfg_name, fig_suffix in [('E_KD_from_RF', 'E'),
                                  ('F_KD_from_CL_MLP', 'F'),
                                  ('F_KD_from_CL_MLP_fair', 'F_fair'),
                                  ('F_KD_from_CL_MLP_ext', 'F_ext')]:
        if cfg_name in final_results and 'confusion_matrix' in final_results[cfg_name]:
            cm = np.array(final_results[cfg_name]['confusion_matrix'])
            fig, ax = plt.subplots(figsize=(7, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(f'Confusion Matrix — {cfg_name}')
            plt.tight_layout()
            plt.savefig(f'confusion_matrix_{fig_suffix}.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved confusion_matrix_{fig_suffix}.png")

    # FIXED 2026-04-11 (v2.3): Pareto frontier now includes BOTH student sizes
    # (A=32-16 and B=64-32) when both agg tables are available.
    fig, ax = plt.subplots(figsize=(10, 6))

    def _plot_agg(agg_df, marker_style, size_label):
        for _, row in agg_df.iterrows():
            cfg = row['Config']
            size_kb = row['size_kb'] if row['size_kb'] else 1.0
            acc = row['MacroF1_mean']
            acc_err = row['MacroF1_std']
            ax.errorbar(size_kb, acc, yerr=acc_err, fmt=marker_style,
                        markersize=8, capsize=3, label=f'{cfg} ({size_label})' if cfg.startswith(('D_', 'E_', 'E2_', 'F_', 'G_', 'I_')) else None)
            ax.annotate(cfg.replace('_', ' '), (size_kb, acc),
                        xytext=(6, 4), textcoords='offset points', fontsize=7)

    _plot_agg(agg_A, 'o', 'Student A 32-16')
    # If Student B was also run, overlay its points
    if len(all_seed_results_B) > 0:
        agg_B_local = aggregate_multi_seed(all_seed_results_B)
        _plot_agg(agg_B_local, 's', 'Student B 64-32')

    ax.set_xscale('log')
    ax.set_xlabel('Model size (KB, fp32) — log scale')
    ax.set_ylabel('Macro F1 (test)')
    ax.set_title('Model size vs. Macro F1 (Pareto frontier, both student sizes)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pareto_frontier.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved pareto_frontier.png")

    # --- Training loss curves: Config B vs C_fair vs C_ext (non-CL vs both CL budgets) ---
    # FIXED 2026-04-11 (v2.3): Now plots all three for direct compute-budget comparison.
    curve_configs = [
        ('B_Full_MLP', 'B (no CL)', 'C0'),
        ('C_CL_MLP_loss_fair', 'C_fair (CL, fair budget)', 'C1'),
        ('C_CL_MLP_loss_ext', 'C_ext (CL, +33% budget)', 'C2'),
    ]
    curves_available = [(cfg, label, color) for cfg, label, color in curve_configs
                        if cfg in final_results and 'loss_curve' in final_results.get(cfg, {})]

    if len(curves_available) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        for cfg, label, color in curves_available:
            curve = final_results[cfg]['loss_curve']
            axes[0].plot(curve['loss'], label=label, color=color)
            axes[1].plot(curve['val_f1'], label=label, color=color)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Training loss')
        axes[0].set_title('Training loss curves — B vs C_fair vs C_ext')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Validation macro F1')
        axes[1].set_title('Validation F1 — B vs C_fair vs C_ext')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('loss_curves_B_vs_C.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved loss_curves_B_vs_C.png")

    # ============================================================================
    # CELL 16: CICIoT2023 generalizability (optional)
    # ============================================================================
    if RUN_CICIOT and os.path.exists(CICIOT_PATH):
        print("\n" + "=" * 60)
        print("CICIoT2023 GENERALIZABILITY EXPERIMENT")
        print("=" * 60)

        ciciot_results = {}
        try:
            dfc = pd.read_csv(CICIOT_PATH)
            dfc.columns = dfc.columns.str.strip()

            # Sample to ~400K rows for tractability
            if len(dfc) > 400_000:
                dfc = dfc.sample(400_000, random_state=42).reset_index(drop=True)
            print(f"CICIoT shape: {dfc.shape}")

            # Identify target column
            target_candidates_c = ['Label', 'label', 'Class', 'class', 'attack']
            tgt_c = None
            for cand in target_candidates_c:
                if cand in dfc.columns:
                    tgt_c = cand
                    break
            if tgt_c is None:
                tgt_c = dfc.columns[-1]

            # Drop obviously non-numeric columns (e.g., flow IDs)
            dfc = dfc.select_dtypes(include=[np.number, object])
            # Encode target
            dfc[tgt_c] = dfc[tgt_c].astype(str).str.strip()
            le_c = LabelEncoder()
            dfc[tgt_c] = le_c.fit_transform(dfc[tgt_c])
            CLASS_NAMES_C = le_c.classes_.tolist()
            NUM_CLASSES_C = len(CLASS_NAMES_C)
            print(f"CICIoT classes: {NUM_CLASSES_C}")

            # Keep only numeric features
            X_c = dfc.drop(tgt_c, axis=1).select_dtypes(include=[np.number]).values.astype(np.float32)
            y_c = dfc[tgt_c].values.astype(np.int64)
            # Handle NaN / inf
            X_c = np.nan_to_num(X_c, nan=0.0, posinf=1e6, neginf=-1e6)

            scaler_c = StandardScaler()
            X_c = scaler_c.fit_transform(X_c)

            X_ctr, X_cte, y_ctr, y_cte = train_test_split(
                X_c, y_c, test_size=0.15, random_state=42, stratify=y_c
            )
            X_ctr, X_cv, y_ctr, y_cv = train_test_split(
                X_ctr, y_ctr, test_size=0.1765, random_state=42, stratify=y_ctr
            )
            INPUT_DIM_C = X_ctr.shape[1]

            # Run a compact version: Config B, D, E2, F only
            set_seed(42)
            Xtr_t = torch.tensor(X_ctr, dtype=torch.float32)
            ytr_t = torch.tensor(y_ctr, dtype=torch.long)
            Xv_t = torch.tensor(X_cv, dtype=torch.float32)
            yv_t = torch.tensor(y_cv, dtype=torch.long)
            Xte_t = torch.tensor(X_cte, dtype=torch.float32)
            yte_t = torch.tensor(y_cte, dtype=torch.long)

            cw_c = torch.tensor(
                len(y_ctr) / (NUM_CLASSES_C * np.maximum(np.bincount(y_ctr, minlength=NUM_CLASSES_C), 1)),
                dtype=torch.float32,
            )

            # Config B
            print("[CICIoT-B] Full MLP baseline...")
            t_b = TeacherMLP(INPUT_DIM_C, NUM_CLASSES_C)
            t_b = train_standard(t_b, Xtr_t, ytr_t, Xv_t, yv_t,
                                  class_weights=cw_c, **TRAIN_CONFIG)
            ciciot_results['B_MLP'] = evaluate_model(t_b, Xte_t, yte_t)

            # Loss-based difficulty + CL teacher
            print("[CICIoT-C] CL teacher (loss-based)...")
            order_c = compute_difficulty_loss_based(
                Xtr_t, ytr_t, INPUT_DIM_C, NUM_CLASSES_C, seed=42
            )
            t_c = TeacherMLP(INPUT_DIM_C, NUM_CLASSES_C)
            t_c = train_with_curriculum(
                t_c, Xtr_t, ytr_t, order_c, Xv_t, yv_t,
                stages=CL_STAGES, class_weights=cw_c
            )
            ciciot_results['C_CL_MLP'] = evaluate_model(t_c, Xte_t, yte_t)

            # Config D
            print("[CICIoT-D] Student scratch...")
            s_d = StudentMLP(INPUT_DIM_C, STUDENT_A_HIDDEN, NUM_CLASSES_C)
            s_d = train_standard(s_d, Xtr_t, ytr_t, Xv_t, yv_t,
                                  class_weights=cw_c, **TRAIN_CONFIG)
            ciciot_results['D_Student_scratch'] = evaluate_model(s_d, Xte_t, yte_t)

            # Config E2
            print("[CICIoT-E2] KD from standard MLP...")
            s_e2 = StudentMLP(INPUT_DIM_C, STUDENT_A_HIDDEN, NUM_CLASSES_C)
            s_e2 = train_kd(s_e2, t_b, Xtr_t, ytr_t, Xv_t, yv_t,
                             T=BEST_T, alpha=BEST_ALPHA, class_weights=cw_c,
                             epochs=TRAIN_CONFIG['epochs'])
            ciciot_results['E2_KD_from_MLP'] = evaluate_model(s_e2, Xte_t, yte_t)

            # Config F
            print("[CICIoT-F] KD from CL-MLP (CORE)...")
            s_f = StudentMLP(INPUT_DIM_C, STUDENT_A_HIDDEN, NUM_CLASSES_C)
            s_f = train_kd(s_f, t_c, Xtr_t, ytr_t, Xv_t, yv_t,
                            T=BEST_T, alpha=BEST_ALPHA, class_weights=cw_c,
                            epochs=TRAIN_CONFIG['epochs'])
            ciciot_results['F_KD_from_CL_MLP'] = evaluate_model(s_f, Xte_t, yte_t)

            print("\nCICIoT2023 summary:")
            for cfg, m in ciciot_results.items():
                print(f"  {cfg:25s} acc={m['accuracy']:.4f}, f1={m['macro_f1']:.4f}")
        except Exception as ex:
            print(f"CICIoT2023 pipeline failed: {ex}")
            import traceback
            traceback.print_exc()
            ciciot_results = {}
    else:
        if RUN_CICIOT:
            print(f"CICIoT2023 not found at {CICIOT_PATH} — skipping")
        ciciot_results = {}

    # ============================================================================
    # CELL 17: Save all results and print final summary
    # ============================================================================
    def json_convert(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, pd.DataFrame):
            return o.to_dict('records')
        return str(o)

    final_output = {
        'wsn_ds_multi_seed_student_A': all_seed_results_A,
        'wsn_ds_multi_seed_student_B': all_seed_results_B,
        'aggregate_student_A': agg_A.to_dict('records'),
        'kd_hyperparameters': {'T': BEST_T, 'alpha': BEST_ALPHA},
        'shap_results': shap_results,
        'quantization': quant_results,
        'inference_benchmarks': bench_results,
        'ciciot_results': ciciot_results,
        'wilcoxon_results': wilcoxon_results if 'wilcoxon_results' in dir() else {},
        'seeds': SEEDS,
        'class_names': CLASS_NAMES,
        'feature_names': FEATURE_NAMES,
    }

    with open('cukd_xai_results.json', 'w') as f:
        json.dump(final_output, f, indent=2, default=json_convert)
    print("\nSaved cukd_xai_results.json")

    # Save aggregate CSV for the paper
    agg_A.to_csv('wsnds_results_student_A.csv', index=False)
    print("Saved wsnds_results_student_A.csv")
    if len(all_seed_results_B) > 0:
        agg_B = aggregate_multi_seed(all_seed_results_B)
        agg_B.to_csv('wsnds_results_student_B.csv', index=False)
        print("Saved wsnds_results_student_B.csv")

    # ============================================================================
    # CELL 18: Final summary for paper
    # ============================================================================
    print("\n" + "=" * 60)
    print("FINAL SUMMARY — CuKD-XAI ON WSN-DS")
    print("=" * 60)
    print(f"\nSeeds: {SEEDS}  |  KD: T={BEST_T}, alpha={BEST_ALPHA}")
    print(f"Student architecture: MLP{STUDENT_A_HIDDEN} → {NUM_CLASSES} classes")

    # Compute teacher-student compression ratios using Config F student
    if 'B_Full_MLP' in final_models and 'F_KD_from_CL_MLP' in final_models:
        teacher_m = final_models['B_Full_MLP']
        student_m = final_models['F_KD_from_CL_MLP']
        teacher_params = count_params(teacher_m)
        student_params = count_params(student_m)
        print(f"\nCompression:")
        print(f"  Teacher: {teacher_params} params ({model_size_kb(teacher_m):.2f} KB fp32)")
        print(f"  Student: {student_params} params ({model_size_kb(student_m):.2f} KB fp32, "
              f"{model_size_kb(student_m, 1):.2f} KB int8)")
        print(f"  Ratio:   {teacher_params / student_params:.1f}x parameter reduction")
    else:
        print("Teacher or student model missing from final_models; skipping compression summary")

    print("\nKey metrics (mean over seeds):")
    print(agg_A[['Config', 'MacroF1_mean', 'MacroF1_std',
                 'Accuracy_mean', 'Accuracy_std',
                 'params', 'size_kb']].to_string(index=False))

    print("\n" + "=" * 60)
    print("IMPLEMENTATION COMPLETE")
    print("=" * 60)
    print("Outputs:")
    print("  cukd_xai_results.json — all results")
    print("  wsnds_results_student_A.csv — aggregate metrics")
    print("  per_class_f1.png — per-class F1 comparison")
    print("  confusion_matrix_F.png — Config F confusion matrix")
    print("  pareto_frontier.png — size vs accuracy trade-off")
    print("  shap_summary_student.png — SHAP feature importance")
    print("  loss_curves_B_vs_C.png — CL convergence comparison")


else:
    shap_results = {}
    quant_results = {}
    bench_results = {}
    ciciot_results = {}
    final_output = {}
    print("\nSkipping legacy v2.3 SHAP/quantization/figures/save/summary block.")
    print("Use the final route entrypoints below for WSN-DS/Edge-IIoTset final runs.")

# ============================================================================
# CELL 19: Final publication route roadmap and run guards
# ============================================================================
# FINAL 2026-05-27:
# The validated v2.3 WSN-DS path above remains the primary evidence base.
# The remaining work is appended below as an explicit final-project layer:
#   1. WSN-DS 10-seed final validation.
#   2. Edge-IIoTset 15-class generalization on the selected ML CSV.
#   3. Crash-resumable per-seed saving for long Jupyter runs.
#   4. Focused post-training INT8 + QAT for best students only.
#   5. Latency / FLOPs / parameter / activation-memory profiling.
#   6. SHAP teacher-student alignment on both datasets.
#   7. Paper-ready guide and limitations.
#
# Nothing in this section runs a long experiment unless FINAL_RUN_MODE or
# a backwards-compatible run guard in CELL 2 explicitly selects it.


FINAL_RUN_MODES = {
    'none',
    'edge_smoke',
    'edge_final',
    'wsnds_final',
    'qat_profile',
    'shap_alignment',
}


def validate_final_run_mode(mode: str = FINAL_RUN_MODE) -> str:
    if mode not in FINAL_RUN_MODES:
        raise ValueError(f"FINAL_RUN_MODE must be one of {sorted(FINAL_RUN_MODES)}, got {mode!r}")
    return mode


def final_mode_enabled(mode_name: str, legacy_flag: bool = False) -> bool:
    """Return True for one final mode; explicit FINAL_RUN_MODE wins over legacy flags."""
    mode = validate_final_run_mode(FINAL_RUN_MODE)
    if mode != 'none':
        return mode == mode_name
    return bool(legacy_flag)

FINAL_PROJECT_LEFTOVER_WORK = [
    "WSN-DS 10-seed clean final using the v2.3 core.",
    "Edge-IIoTset ML 15-class multiclass generalization with leakage-safe preprocessing.",
    "Optional DNN-EdgeIIoTset sampled/full follow-up only after the ML path is stable.",
    "Per-seed checkpointing so Jupyter power/network failures lose at most one seed.",
    "Post-training dynamic INT8 and focused QAT for best compact students only.",
    "Deployment-feasibility profiling: params, fp32/int8 size, FLOPs, activation RAM, p50/p95 latency.",
    "SHAP teacher-student feature-alignment audit on WSN-DS and Edge-IIoTset.",
    "Paper narrative: RF/tree-ensemble-to-MLP KD + KB-scale compression + XAI alignment gap.",
]


def print_final_project_leftover_work() -> None:
    print("\n" + "=" * 72)
    print("FINAL CuKD-XAI LEFTOVER WORK")
    print("=" * 72)
    for idx, item in enumerate(FINAL_PROJECT_LEFTOVER_WORK, 1):
        print(f"{idx}. {item}")
    print("\nDefault guards are OFF. Set only the exact final run you want to execute.")
    print(f"FINAL_RUN_MODE = {FINAL_RUN_MODE}")
    print(f"RUN_LEGACY_V23_EXPERIMENTS = {RUN_LEGACY_V23_EXPERIMENTS}")
    print(f"RUN_PUBLICATION_10_SEEDS = {RUN_PUBLICATION_10_SEEDS}")
    print(f"RUN_EDGEIIOT_ML_SMOKE    = {RUN_EDGEIIOT_ML_SMOKE}")
    print(f"RUN_EDGEIIOT_ML_5SEED    = {RUN_EDGEIIOT_ML_5SEED}")
    print(f"RUN_WSNDS_RESUME_FINAL   = {RUN_WSNDS_RESUME_FINAL}")
    print(f"RUN_QAT_FOR_BEST_STUDENTS = {RUN_QAT_FOR_BEST_STUDENTS}")
    print(f"RUN_DEPLOYMENT_PROFILE   = {RUN_DEPLOYMENT_PROFILE}")
    print(f"RUN_SHAP_ALIGNMENT_FINAL = {RUN_SHAP_ALIGNMENT_FINAL}")


print_final_project_leftover_work()


# ============================================================================
# CELL 20: Crash-resume helpers for final Jupyter runs
# ============================================================================
def final_json_convert(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return o.to_dict('records')
    return str(o)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def write_json_atomic(path: str, payload: dict) -> None:
    ensure_dir(os.path.dirname(path) or '.')
    tmp_path = f"{path}.tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, default=final_json_convert)
    os.replace(tmp_path, path)


def read_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def seed_dir(out_dir: str, seed: int, student_name: str) -> str:
    return os.path.join(out_dir, f"seed_{seed}", student_name)


def seed_metrics_path(out_dir: str, seed: int, student_name: str) -> str:
    return os.path.join(seed_dir(out_dir, seed, student_name), 'metrics.json')


def seed_done(out_dir: str, seed: int, student_name: str) -> bool:
    return os.path.exists(seed_metrics_path(out_dir, seed, student_name))


def save_seed_metrics(out_dir: str, seed: int, student_name: str, metrics: dict) -> None:
    write_json_atomic(seed_metrics_path(out_dir, seed, student_name), metrics)


def load_seed_metrics(out_dir: str, seed: int, student_name: str) -> dict:
    return read_json(seed_metrics_path(out_dir, seed, student_name))



def normalized_run_config(config: dict) -> dict:
    return json.loads(json.dumps(config, sort_keys=True, default=final_json_convert))


def attach_run_config(metrics: dict, run_config: dict) -> dict:
    payload = copy.deepcopy(metrics)
    payload['_run_config'] = normalized_run_config(run_config)
    return payload


def metrics_match_run_config(metrics: dict, run_config: dict) -> bool:
    return metrics.get('_run_config') == normalized_run_config(run_config)


def save_model_state(out_dir: str, seed: int, student_name: str,
                     models: dict, model_keys=('D_Small_MLP', 'E_KD_from_RF')) -> None:
    model_dir = ensure_dir(os.path.join(seed_dir(out_dir, seed, student_name), 'models'))
    for key in model_keys:
        model = models.get(key)
        if isinstance(model, nn.Module):
            torch.save(model.cpu().state_dict(), os.path.join(model_dir, f"{key}.pt"))
            model.to(device)


def aggregate_seed_metrics(seed_results: dict, class_names: list) -> pd.DataFrame:
    configs = sorted({cfg for per_seed in seed_results.values() for cfg in per_seed.keys()})
    rows = []
    for cfg in configs:
        accs, f1s, per_class = [], [], []
        params, size_kb, flops = None, None, None
        for seed, result in seed_results.items():
            if cfg not in result:
                continue
            metric = result[cfg]
            accs.append(metric.get('accuracy', np.nan))
            f1s.append(metric.get('macro_f1', np.nan))
            if 'per_class_f1' in metric:
                per_class.append(metric['per_class_f1'])
            params = metric.get('params', params)
            size_kb = metric.get('model_size_kb', metric.get('size_kb', size_kb))
            flops = metric.get('flops', flops)
        if not f1s:
            continue
        row = {
            'Config': cfg,
            'Accuracy_mean': float(np.nanmean(accs)),
            'Accuracy_std': float(np.nanstd(accs)),
            'MacroF1_mean': float(np.nanmean(f1s)),
            'MacroF1_std': float(np.nanstd(f1s)),
            'n_seeds': len(f1s),
            'params': params,
            'size_kb': size_kb,
            'flops': flops,
        }
        if per_class:
            arr = np.asarray(per_class, dtype=float)
            for i, name in enumerate(class_names):
                if i < arr.shape[1]:
                    row[f'{name}_F1_mean'] = float(np.nanmean(arr[:, i]))
                    row[f'{name}_F1_std'] = float(np.nanstd(arr[:, i]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values('MacroF1_mean', ascending=False)


# ============================================================================
# CELL 21: Edge-IIoTset leakage-safe preprocessing
# ============================================================================
EDGEIIOT_LEAKAGE_DROP_COLUMNS = [
    'frame.time', 'ip.src_host', 'ip.dst_host',
    'arp.src.proto_ipv4', 'arp.dst.proto_ipv4',
    'http.file_data', 'http.request.full_uri', 'icmp.transmit_timestamp',
    'http.request.uri.query', 'tcp.options', 'tcp.payload',
    'tcp.srcport', 'tcp.dstport', 'udp.port', 'mqtt.msg',
]

EDGEIIOT_KNOWN_ZERO_VARIANCE_COLUMNS = [
    'icmp.unused', 'http.tls_port', 'dns.qry.type',
    'dns.retransmit_request_in', 'mqtt.msg_decoded_as',
    'mbtcp.len', 'mbtcp.trans_id', 'mbtcp.unit_id',
]


def load_edgeiiot_dataframe(path: str = EDGEIIOT_ML_PATH,
                            max_rows: int = None) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Edge-IIoTset CSV not found at {path!r}. Set EDGEIIOT_ML_PATH to the extracted ML CSV."
        )
    return pd.read_csv(path, low_memory=False, nrows=max_rows)


def prepare_edgeiiot_dataframe(df_edge: pd.DataFrame,
                               target_col: str = EDGEIIOT_TARGET_COL,
                               drop_duplicates: bool = True,
                               verbose: bool = True) -> dict:
    """Leakage-safe Edge-IIoTset preprocessing for 15-class experiments.

    Uses Attack_type by default. Attack_label is kept only for optional binary
    supplementary work and is never allowed into the feature matrix.
    """
    df_edge = df_edge.copy()
    df_edge.columns = df_edge.columns.str.strip()
    if target_col not in df_edge.columns:
        raise ValueError(f"target_col={target_col!r} not found. Columns: {df_edge.columns.tolist()}")

    original_shape = df_edge.shape
    drop_existing = [c for c in EDGEIIOT_LEAKAGE_DROP_COLUMNS if c in df_edge.columns]
    df_edge = df_edge.drop(columns=drop_existing)
    df_edge = df_edge.dropna(axis=0, how='any')
    if drop_duplicates:
        df_edge = df_edge.drop_duplicates(keep='first')

    zero_existing = [c for c in EDGEIIOT_KNOWN_ZERO_VARIANCE_COLUMNS if c in df_edge.columns]
    dynamic_zero = [
        c for c in df_edge.columns
        if c not in ('Attack_label', 'Attack_type') and df_edge[c].nunique(dropna=False) <= 1
    ]
    zero_to_drop = sorted(set(zero_existing + dynamic_zero))
    df_edge = df_edge.drop(columns=zero_to_drop)

    y_raw = df_edge[target_col].astype(str).str.strip()
    label_encoder = LabelEncoder()
    y_edge = label_encoder.fit_transform(y_raw).astype(np.int64)
    class_names = label_encoder.classes_.tolist()

    feature_df = df_edge.drop(columns=[c for c in ('Attack_label', 'Attack_type') if c in df_edge.columns])
    categorical_cols = feature_df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        feature_df = pd.get_dummies(feature_df, columns=categorical_cols, dtype=np.float32)

    for col in feature_df.columns:
        feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    X_edge = feature_df.values.astype(np.float32)
    metadata = {
        'target_col': target_col,
        'original_shape': original_shape,
        'shape_after_cleaning': df_edge.shape,
        'dropped_leakage_columns': drop_existing,
        'dropped_zero_variance_columns': zero_to_drop,
        'categorical_columns_one_hot': categorical_cols,
        'feature_names': feature_df.columns.tolist(),
        'class_names': class_names,
        'class_counts': y_raw.value_counts().to_dict(),
        'n_features': int(X_edge.shape[1]),
        'n_rows': int(X_edge.shape[0]),
    }
    if verbose:
        print("\nEdge-IIoTset preprocessing summary")
        print(f"  original shape: {original_shape}")
        print(f"  cleaned shape:  {df_edge.shape}")
        print(f"  X shape:        {X_edge.shape}")
        print(f"  classes:        {class_names}")
        print(f"  dropped leakage columns: {drop_existing}")
        print(f"  one-hot columns: {categorical_cols}")
    return {'X': X_edge, 'y': y_edge, 'metadata': metadata, 'label_encoder': label_encoder}


def split_and_scale_arrays(X: np.ndarray, y: np.ndarray,
                           split_seed: int = 42,
                           scale_scope: str = 'train') -> dict:
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=split_seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1765,
        random_state=split_seed, stratify=y_trainval
    )
    scaler = StandardScaler()
    if scale_scope == 'global':
        scaler.fit(X)
    elif scale_scope == 'train':
        scaler.fit(X_train)
    else:
        raise ValueError("scale_scope must be 'train' or 'global'")
    return {
        'X_train': scaler.transform(X_train).astype(np.float32),
        'X_val': scaler.transform(X_val).astype(np.float32),
        'X_test': scaler.transform(X_test).astype(np.float32),
        'y_train': y_train.astype(np.int64),
        'y_val': y_val.astype(np.int64),
        'y_test': y_test.astype(np.int64),
        'scaler': scaler,
        'split_seed': split_seed,
        'scale_scope': scale_scope,
        'split_counts': {
            'train': int(len(y_train)),
            'val': int(len(y_val)),
            'test': int(len(y_test)),
        },
    }


# ============================================================================
# CELL 22: Compact final runner for Edge-IIoTset and paper-grade reruns
# ============================================================================
def _class_weights_np(y_train: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=num_classes)
    return torch.tensor(
        len(y_train) / (num_classes * np.maximum(counts, 1)),
        dtype=torch.float32,
    )


def run_compact_kd_configs(seed: int,
                           X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray,
                           input_dim: int,
                           num_classes: int,
                           student_hidden: tuple,
                           kd_T: float = 2.0,
                           kd_alpha: float = 0.5,
                           rf_trees: int = 500,
                           rf_max_depth: int = 15,
                           return_models: bool = False,
                           verbose: bool = True) -> dict:
    """Focused final runner: RF, full MLP, scratch student, KD from RF.

    This is intentionally smaller than run_all_configs(). It is meant for
    Edge-IIoTset and final validation runs where CL/DKD/v3 variants are not the
    headline. The full v2.3 WSN-DS runner above remains available for historical
    CL ablations.
    """
    set_seed(seed)
    if verbose:
        print(f"\n{'=' * 60}\nCompact seed {seed} — Student {student_hidden}\n{'=' * 60}")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    class_weights = _class_weights_np(y_train, num_classes)

    results, models = {}, {}

    if verbose: print("[A] RF teacher...")
    t0 = time.perf_counter()
    rf = RandomForestClassifier(
        n_estimators=rf_trees, max_depth=rf_max_depth,
        random_state=seed, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_time = time.perf_counter() - t0
    rf_preds = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)
    rf_prec, rf_rec, rf_f1, _ = precision_recall_fscore_support(
        y_test, rf_preds, average='macro', zero_division=0
    )
    results['A_RF'] = {
        'accuracy': float(accuracy_score(y_test, rf_preds)),
        'macro_precision': float(rf_prec),
        'macro_recall': float(rf_rec),
        'macro_f1': float(rf_f1),
        'per_class_f1': f1_score(y_test, rf_preds, average=None, zero_division=0).tolist(),
        'ece': expected_calibration_error(rf_probs, y_test),
        'train_time_sec': rf_time,
    }
    models['A_RF'] = rf

    if verbose: print("[B] Full MLP teacher...")
    t0 = time.perf_counter()
    teacher_b = TeacherMLP(input_dim, num_classes)
    teacher_b = train_standard(
        teacher_b, X_train_t, y_train_t, X_val_t, y_val_t,
        class_weights=class_weights, **TRAIN_CONFIG
    )
    m_b = evaluate_model(teacher_b, X_test_t, y_test_t)
    m_b['params'] = count_params(teacher_b)
    m_b['model_size_kb'] = model_size_kb(teacher_b)
    m_b['train_time_sec'] = time.perf_counter() - t0
    results['B_Full_MLP'] = m_b
    models['B_Full_MLP'] = teacher_b

    if verbose: print("[D] Student from scratch...")
    t0 = time.perf_counter()
    student_d = StudentMLP(input_dim, student_hidden, num_classes)
    student_d = train_standard(
        student_d, X_train_t, y_train_t, X_val_t, y_val_t,
        class_weights=class_weights, **TRAIN_CONFIG
    )
    m_d = evaluate_model(student_d, X_test_t, y_test_t)
    m_d['params'] = count_params(student_d)
    m_d['model_size_kb'] = model_size_kb(student_d)
    m_d['model_size_kb_int8'] = model_size_kb(student_d, 1)
    m_d['flops'] = compute_flops_mlp(input_dim, student_hidden, num_classes)
    m_d['train_time_sec'] = time.perf_counter() - t0
    results['D_Small_MLP'] = m_d
    models['D_Small_MLP'] = student_d

    if verbose: print("[E] KD from calibrated RF...")
    t0 = time.perf_counter()
    rf_calib = CalibratedClassifierCV(
        RandomForestClassifier(
            n_estimators=rf_trees, max_depth=rf_max_depth,
            random_state=seed, n_jobs=-1
        ),
        method='isotonic', cv=3
    )
    rf_calib.fit(X_train, y_train)
    rf_soft = torch.tensor(rf_calib.predict_proba(X_train), dtype=torch.float32)
    student_e = StudentMLP(input_dim, student_hidden, num_classes)
    student_e = train_kd(
        student_e, rf_soft, X_train_t, y_train_t, X_val_t, y_val_t,
        T=kd_T, alpha=kd_alpha, class_weights=class_weights,
        epochs=TRAIN_CONFIG['epochs'], batch_size=TRAIN_CONFIG['batch_size']
    )
    m_e = evaluate_model(student_e, X_test_t, y_test_t)
    m_e['params'] = count_params(student_e)
    m_e['model_size_kb'] = model_size_kb(student_e)
    m_e['model_size_kb_int8'] = model_size_kb(student_e, 1)
    m_e['flops'] = compute_flops_mlp(input_dim, student_hidden, num_classes)
    m_e['ece'] = expected_calibration_error(_batched_probs(student_e, X_test_t), y_test)
    m_e['train_time_sec'] = time.perf_counter() - t0
    results['E_KD_from_RF'] = m_e
    models['E_KD_from_RF'] = student_e

    if verbose:
        print("Seed summary (macro F1):")
        for cfg, metric in results.items():
            print(f"  {cfg:18s} {metric['macro_f1']:.4f}")
    if return_models:
        return results, models
    return results


def run_multiseed_compact_with_resume(out_dir: str,
                                      seeds: list,
                                      student_name: str,
                                      student_hidden: tuple,
                                      arrays: dict,
                                      input_dim: int,
                                      num_classes: int,
                                      class_names: list,
                                      kd_T: float = 2.0,
                                      kd_alpha: float = 0.5,
                                      rf_trees: int = 500,
                                      rf_max_depth: int = 15,
                                      resume: bool = True,
                                      save_models: bool = True) -> dict:
    ensure_dir(out_dir)
    run_config = {
        'runner': 'compact_kd',
        'student_name': student_name,
        'student_hidden': list(student_hidden),
        'input_dim': input_dim,
        'num_classes': num_classes,
        'kd_T': kd_T,
        'kd_alpha': kd_alpha,
        'rf_trees': rf_trees,
        'rf_max_depth': rf_max_depth,
        'class_names': class_names,
    }
    seed_results = {}
    last_models = {}
    for seed in seeds:
        if resume and seed_done(out_dir, seed, student_name):
            cached_metrics = load_seed_metrics(out_dir, seed, student_name)
            if metrics_match_run_config(cached_metrics, {**run_config, 'seed': seed}):
                print(f"[resume] loading {student_name} seed {seed}")
                seed_results[seed] = cached_metrics
                continue
            print(f"[stale] rerunning {student_name} seed {seed}; saved run config differs")
        metrics, models = run_compact_kd_configs(
            seed,
            arrays['X_train'], arrays['y_train'],
            arrays['X_val'], arrays['y_val'],
            arrays['X_test'], arrays['y_test'],
            input_dim=input_dim,
            num_classes=num_classes,
            student_hidden=student_hidden,
            kd_T=kd_T,
            kd_alpha=kd_alpha,
            rf_trees=rf_trees,
            rf_max_depth=rf_max_depth,
            return_models=True,
            verbose=True,
        )
        metrics = attach_run_config(metrics, {**run_config, 'seed': seed})
        seed_results[seed] = metrics
        save_seed_metrics(out_dir, seed, student_name, metrics)
        if save_models:
            save_model_state(out_dir, seed, student_name, models)
        last_models = models
        agg = aggregate_seed_metrics(seed_results, class_names)
        agg.to_csv(os.path.join(out_dir, f'{student_name}_aggregate_so_far.csv'), index=False)
    return {'seed_results': seed_results, 'last_models': last_models}


# ============================================================================
# CELL 23: Focused INT8 and QAT deployment-feasibility utilities
# ============================================================================
class QATStudentMLP(nn.Module):
    """Student MLP wrapper with QuantStub/DeQuantStub for QAT."""
    def __init__(self, input_dim: int, hidden_dims: tuple, num_classes: int):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.net = StudentMLP(input_dim, hidden_dims, num_classes).net
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.net(x)
        return self.dequant(x)


def evaluate_cpu_model(model: nn.Module, X_np: np.ndarray, y_np: np.ndarray,
                       batch_size: int = 4096) -> dict:
    model = model.cpu().eval()
    X_t = torch.tensor(X_np, dtype=torch.float32)
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            preds.append(model(X_t[i:i + batch_size]).argmax(dim=1).numpy())
    pred = np.concatenate(preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_np, pred, average='macro', zero_division=0
    )
    return {
        'accuracy': float(accuracy_score(y_np, pred)),
        'macro_precision': float(prec),
        'macro_recall': float(rec),
        'macro_f1': float(f1),
        'per_class_f1': f1_score(y_np, pred, average=None, zero_division=0).tolist(),
    }


def select_quantized_backend(preferred=('fbgemm', 'x86', 'qnnpack')) -> str:
    """Choose a supported PyTorch quantization backend for CPU QAT."""
    if not hasattr(torch.backends, 'quantized'):
        raise RuntimeError('torch.backends.quantized is unavailable in this PyTorch build')
    supported = list(getattr(torch.backends.quantized, 'supported_engines', []))
    usable = [engine for engine in supported if engine and engine != 'none']
    if not usable:
        raise RuntimeError(f'No usable quantized backend found. supported_engines={supported}')
    for engine in preferred:
        if engine in usable:
            torch.backends.quantized.engine = engine
            return engine
    engine = usable[0]
    torch.backends.quantized.engine = engine
    return engine


def train_qat_from_fp32_student(fp32_student: nn.Module,
                                arrays: dict,
                                input_dim: int,
                                hidden_dims: tuple,
                                num_classes: int,
                                epochs: int = 10,
                                lr: float = 1e-4) -> dict:
    if not hasattr(torch, 'ao') or not hasattr(torch.ao, 'quantization'):
        raise RuntimeError('torch.ao.quantization is unavailable in this PyTorch build')

    qat_device = torch.device('cpu')
    qat_backend = select_quantized_backend()
    qat_model = QATStudentMLP(input_dim, hidden_dims, num_classes).to(qat_device)
    qat_model.net.load_state_dict(copy.deepcopy(fp32_student.cpu().net.state_dict()))
    qat_model.qconfig = torch.ao.quantization.get_default_qat_qconfig(qat_backend)
    qat_model = torch.ao.quantization.prepare_qat(qat_model, inplace=False)

    X_train_t = torch.tensor(arrays['X_train'], dtype=torch.float32)
    y_train_t = torch.tensor(arrays['y_train'], dtype=torch.long)
    X_val_t = torch.tensor(arrays['X_val'], dtype=torch.float32)
    y_val_np = arrays['y_val']
    class_weights = _class_weights_np(arrays['y_train'], num_classes).to(qat_device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=TRAIN_CONFIG['batch_size'], shuffle=True)
    opt = torch.optim.AdamW(qat_model.parameters(), lr=lr, weight_decay=TRAIN_CONFIG['weight_decay'])

    best_state, best_val = None, -1.0
    for epoch in range(epochs):
        qat_model.train()
        if epoch == max(3, epochs // 2):
            qat_model.apply(torch.ao.quantization.disable_observer)
        for xb, yb in loader:
            opt.zero_grad()
            loss = ce_loss(qat_model(xb), yb)
            loss.backward()
            opt.step()
        qat_model.eval()
        with torch.no_grad():
            val_preds = qat_model(X_val_t).argmax(dim=1).numpy()
        val_f1 = f1_score(y_val_np, val_preds, average='macro', zero_division=0)
        if val_f1 > best_val:
            best_val = val_f1
            best_state = copy.deepcopy(qat_model.state_dict())
        print(f"QAT epoch {epoch + 1}/{epochs}: val_macro_f1={val_f1:.4f}")

    if best_state is not None:
        qat_model.load_state_dict(best_state)
    qat_model.eval()
    int8_model = torch.ao.quantization.convert(qat_model, inplace=False)
    return {
        'qat_int8_model': int8_model,
        'metrics': evaluate_cpu_model(int8_model, arrays['X_test'], arrays['y_test']),
        'on_disk_kb': model_size_on_disk_kb(int8_model),
    }


def profile_student_for_deployment(model: nn.Module,
                                   X_sample_np: np.ndarray,
                                   input_dim: int,
                                   hidden_dims: tuple,
                                   num_classes: int,
                                   n_runs: int = 1000) -> dict:
    timing = measure_cpu_latency_distribution(
        model, X_sample_np, batch_size=1, n_runs=n_runs
    )
    activation_floats = sum(hidden_dims) + num_classes
    return {
        'params': count_params(model),
        'size_kb_fp32_weights': model_size_kb(model, 4),
        'size_kb_int8_weights_theoretical': model_size_kb(model, 1),
        'flops_per_inference': compute_flops_mlp(input_dim, hidden_dims, num_classes),
        'activation_ram_bytes_fp32_estimate': int(activation_floats * 4),
        'activation_ram_bytes_int8_estimate': int(activation_floats),
        **timing,
    }




# Final-route KD tuning helpers. These prevent the final route from silently
# using arbitrary/default KD settings after the legacy v2.3 sweep is disabled.
def run_rf_kd_grid_search_arrays(X_train: np.ndarray, y_train: np.ndarray,
                                 X_val: np.ndarray, y_val: np.ndarray,
                                 input_dim: int,
                                 num_classes: int,
                                 student_hidden: tuple,
                                 seed: int = 42,
                                 rf_trees: int = EDGEIIOT_KD_GRID_RF_TREES,
                                 rf_max_depth: int = 15,
                                 epochs: int = EDGEIIOT_KD_GRID_EPOCHS,
                                 T_grid=KD_T_GRID,
                                 alpha_grid=KD_ALPHA_GRID) -> dict:
    """Tune RF-to-student KD hyperparameters on the validation split."""
    set_seed(seed)
    print("\n>>> RF-KD hyperparameter grid search")
    rf_calib = CalibratedClassifierCV(
        RandomForestClassifier(
            n_estimators=rf_trees, max_depth=rf_max_depth,
            random_state=seed, n_jobs=-1,
        ),
        method='isotonic', cv=3,
    )
    rf_calib.fit(X_train, y_train)
    rf_soft = torch.tensor(rf_calib.predict_proba(X_train), dtype=torch.float32)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    class_weights = _class_weights_np(y_train, num_classes)

    best = {'T': KD_T_DEFAULT, 'alpha': KD_ALPHA_DEFAULT, 'val_f1': -np.inf}
    results = []
    for T in T_grid:
        for alpha in alpha_grid:
            student = StudentMLP(input_dim, student_hidden, num_classes)
            student = train_kd(
                student, rf_soft, X_train_t, y_train_t, X_val_t, y_val_t,
                T=T, alpha=alpha, class_weights=class_weights,
                epochs=epochs, batch_size=TRAIN_CONFIG['batch_size'],
                lr=TRAIN_CONFIG['lr'], weight_decay=TRAIN_CONFIG['weight_decay'],
                patience=TRAIN_CONFIG['patience'],
            )
            val_metrics = evaluate_model(student, X_val_t, y_val_t)
            val_f1 = float(val_metrics['macro_f1'])
            row = {'T': T, 'alpha': alpha, 'val_f1': val_f1}
            results.append(row)
            print(f"  T={T}, alpha={alpha}: val_f1={val_f1:.4f}")
            if val_f1 > best['val_f1']:
                best = row.copy()
    return {
        'teacher': 'calibrated_random_forest',
        'student_hidden': list(student_hidden),
        'rf_trees': rf_trees,
        'rf_max_depth': rf_max_depth,
        'epochs': epochs,
        'best': best,
        'all': results,
    }


def resolve_edgeiiot_kd_hyperparams(out_dir: str,
                                    arrays: dict,
                                    input_dim: int,
                                    num_classes: int,
                                    class_names: list,
                                    smoke: bool,
                                    student_hidden: tuple = STUDENT_A_HIDDEN,
                                    resume: bool = RESUME_FINAL_RUNS) -> dict:
    """Return KD hyperparameters for Edge-IIoTset without wasting smoke runs."""
    if smoke or not RUN_EDGEIIOT_KD_GRID_FINAL:
        return {
            'best': {'T': KD_T_DEFAULT, 'alpha': KD_ALPHA_DEFAULT, 'val_f1': None},
            'all': [],
            'mode': 'default_for_smoke_or_disabled_grid',
        }
    cache_path = os.path.join(out_dir, 'edgeiiot_kd_grid_search.json')
    if resume and os.path.exists(cache_path):
        print(f"[resume] loading Edge-IIoTset KD grid search from {cache_path}")
        return read_json(cache_path)
    grid = run_rf_kd_grid_search_arrays(
        arrays['X_train'], arrays['y_train'], arrays['X_val'], arrays['y_val'],
        input_dim=input_dim,
        num_classes=num_classes,
        student_hidden=student_hidden,
        seed=42,
        rf_trees=EDGEIIOT_KD_GRID_RF_TREES,
        rf_max_depth=15,
        epochs=EDGEIIOT_KD_GRID_EPOCHS,
    )
    grid['mode'] = 'rf_kd_validation_grid'
    grid['class_names'] = class_names
    write_json_atomic(cache_path, grid)
    return grid

# ============================================================================
# CELL 24: Edge-IIoTset ML run entrypoint (disabled by default)
# ============================================================================
def run_edgeiiot_ml_entrypoint(smoke: bool = True) -> dict:
    df_edge = load_edgeiiot_dataframe(EDGEIIOT_ML_PATH)
    prepared = prepare_edgeiiot_dataframe(df_edge, target_col=EDGEIIOT_TARGET_COL)
    arrays = split_and_scale_arrays(prepared['X'], prepared['y'], split_seed=42, scale_scope='train')
    class_names = prepared['metadata']['class_names']
    input_dim = prepared['metadata']['n_features']
    num_classes = len(class_names)
    seeds = EDGEIIOT_SEEDS[:1] if smoke else EDGEIIOT_SEEDS
    out_dir = ensure_dir(os.path.join(
        FINAL_RUN_ROOT,
        'edgeiiot_ml_smoke' if smoke else 'edgeiiot_ml_5seed'
    ))
    write_json_atomic(os.path.join(out_dir, 'preprocessing_metadata.json'), prepared['metadata'])
    kd_grid = resolve_edgeiiot_kd_hyperparams(
        out_dir, arrays, input_dim, num_classes, class_names,
        smoke=smoke, student_hidden=STUDENT_A_HIDDEN, resume=RESUME_FINAL_RUNS,
    )
    edge_kd_T = float(kd_grid['best']['T'])
    edge_kd_alpha = float(kd_grid['best']['alpha'])
    print(f"Edge-IIoTset KD hyperparameters: T={edge_kd_T}, alpha={edge_kd_alpha}")

    results = {}
    for student_name, hidden in [('student_A_32_16', STUDENT_A_HIDDEN), ('student_B_64_32', STUDENT_B_HIDDEN)]:
        results[student_name] = run_multiseed_compact_with_resume(
            out_dir=out_dir,
            seeds=seeds,
            student_name=student_name,
            student_hidden=hidden,
            arrays=arrays,
            input_dim=input_dim,
            num_classes=num_classes,
            class_names=class_names,
            kd_T=edge_kd_T,
            kd_alpha=edge_kd_alpha,
            rf_trees=100 if smoke else 500,
            rf_max_depth=15,
            resume=RESUME_FINAL_RUNS,
            save_models=not smoke,
        )
    final_payload = {
        'dataset': 'Edge-IIoTset ML selected CSV',
        'smoke': smoke,
        'seeds': seeds,
        'class_names': class_names,
        'preprocessing_metadata': prepared['metadata'],
        'kd_hyperparameters': {'T': edge_kd_T, 'alpha': edge_kd_alpha},
        'kd_grid_search': kd_grid,
        'student_A': results['student_A_32_16']['seed_results'],
        'student_B': results['student_B_64_32']['seed_results'],
    }
    write_json_atomic(os.path.join(out_dir, 'edgeiiot_results.json'), final_payload)
    return final_payload


if final_mode_enabled('edge_smoke', RUN_EDGEIIOT_ML_SMOKE):
    edgeiiot_smoke_results = run_edgeiiot_ml_entrypoint(smoke=True)
elif final_mode_enabled('edge_final', RUN_EDGEIIOT_ML_5SEED):
    edgeiiot_final_results = run_edgeiiot_ml_entrypoint(smoke=False)
else:
    print("\nEdge-IIoTset final cells loaded. For Run All, set FINAL_RUN_MODE=edge_smoke or FINAL_RUN_MODE=edge_final.")


# ============================================================================
# CELL 25: Final WSN-DS resume runner
# ============================================================================
def _default_wsnds_arrays() -> dict:
    required = [
        'X_train_np', 'y_train_np', 'X_val_np', 'y_val_np',
        'X_test_np', 'y_test_np', 'CLASS_NAMES', 'INPUT_DIM', 'NUM_CLASSES'
    ]
    missing = [name for name in required if name not in globals()]
    if missing:
        raise RuntimeError(
            f"WSN-DS arrays are not available yet: {missing}. Run the WSN-DS setup cells first."
        )
    return {
        'X_train': X_train_np,
        'y_train': y_train_np,
        'X_val': X_val_np,
        'y_val': y_val_np,
        'X_test': X_test_np,
        'y_test': y_test_np,
        'class_names': CLASS_NAMES,
        'input_dim': INPUT_DIM,
        'num_classes': NUM_CLASSES,
    }




def run_wsnds_kd_grid_search_final(out_dir: str = None,
                                   resume: bool = RESUME_FINAL_RUNS,
                                   force: bool = False) -> dict:
    """Run/cache the same v2.3 WSN-DS KD grid before publication seeds."""
    arrays = _default_wsnds_arrays()
    out_dir = ensure_dir(out_dir or os.path.join(FINAL_RUN_ROOT, 'wsnds_10seed_resume'))
    cache_path = os.path.join(out_dir, 'wsnds_kd_grid_search.json')
    if resume and not force and os.path.exists(cache_path):
        print(f"[resume] loading WSN-DS KD grid search from {cache_path}")
        return read_json(cache_path)

    print("\n>>> WSN-DS KD hyperparameter grid search (seed 42, Student A)")
    set_seed(42)
    Xtr = torch.tensor(arrays['X_train'], dtype=torch.float32)
    ytr = torch.tensor(arrays['y_train'], dtype=torch.long)
    Xv = torch.tensor(arrays['X_val'], dtype=torch.float32)
    yv = torch.tensor(arrays['y_val'], dtype=torch.long)
    class_weights = torch.tensor(
        len(arrays['y_train']) / (
            arrays['num_classes'] * np.maximum(
                np.bincount(arrays['y_train'], minlength=arrays['num_classes']), 1
            )
        ),
        dtype=torch.float32,
    )
    teacher = TeacherMLP(arrays['input_dim'], arrays['num_classes'])
    teacher = train_standard(
        teacher, Xtr, ytr, Xv, yv,
        class_weights=class_weights, epochs=15, batch_size=256, lr=1e-3,
    )
    grid = kd_grid_search(
        teacher, STUDENT_A_HIDDEN, Xtr, ytr, Xv, yv, class_weights,
    )
    grid['mode'] = 'wsnds_v23_mlp_teacher_grid'
    grid['student_hidden'] = list(STUDENT_A_HIDDEN)
    grid['class_names'] = arrays['class_names']
    write_json_atomic(cache_path, grid)
    return grid


def run_wsnds_final_with_resume(out_dir: str = None,
                                seeds: list = None,
                                run_student_b: bool = True,
                                resume: bool = RESUME_FINAL_RUNS,
                                save_models: bool = True,
                                tune_kd: bool = RUN_WSNDS_KD_GRID_FINAL,
                                kd_T: float = None,
                                kd_alpha: float = None) -> dict:
    """Run WSN-DS final seeds with per-seed checkpointing.

    This is the resumable final-run path. It keeps the historical v2.3
    run_all_configs() behavior but writes each completed seed to disk before
    starting the next one.
    """
    arrays = _default_wsnds_arrays()
    out_dir = ensure_dir(out_dir or os.path.join(FINAL_RUN_ROOT, 'wsnds_10seed_resume'))
    seeds = list(seeds or SEEDS_PUBLICATION10)
    kd_grid = None
    if (kd_T is None or kd_alpha is None) and tune_kd:
        kd_grid = run_wsnds_kd_grid_search_final(out_dir=out_dir, resume=resume)
        kd_T = kd_T if kd_T is not None else float(kd_grid['best']['T'])
        kd_alpha = kd_alpha if kd_alpha is not None else float(kd_grid['best']['alpha'])
    else:
        kd_T = BEST_T if kd_T is None and 'BEST_T' in globals() else (kd_T or KD_T_DEFAULT)
        kd_alpha = BEST_ALPHA if kd_alpha is None and 'BEST_ALPHA' in globals() else (kd_alpha or KD_ALPHA_DEFAULT)
    print(f"WSN-DS KD hyperparameters: T={kd_T}, alpha={kd_alpha}")

    student_specs = [('student_A_32_16', STUDENT_A_HIDDEN)]
    if run_student_b:
        student_specs.append(('student_B_64_32', STUDENT_B_HIDDEN))

    all_results = {}
    for student_name, hidden in student_specs:
        run_config = {
            'runner': 'wsnds_run_all_configs',
            'student_name': student_name,
            'student_hidden': list(hidden),
            'kd_T': kd_T,
            'kd_alpha': kd_alpha,
            'input_dim': arrays['input_dim'],
            'num_classes': arrays['num_classes'],
            'class_names': arrays['class_names'],
        }
        student_results = {}
        for seed in seeds:
            if resume and seed_done(out_dir, seed, student_name):
                cached_metrics = load_seed_metrics(out_dir, seed, student_name)
                if metrics_match_run_config(cached_metrics, {**run_config, 'seed': seed}):
                    print(f"[resume] loading WSN-DS {student_name} seed {seed}")
                    metrics = cached_metrics
                else:
                    print(f"[stale] rerunning WSN-DS {student_name} seed {seed}; saved run config differs")
                    metrics = None
            else:
                metrics = None
            if metrics is None:
                print(f"[run] WSN-DS {student_name} seed {seed}")
                metrics, models = run_all_configs(
                    seed,
                    arrays['X_train'], arrays['y_train'],
                    arrays['X_val'], arrays['y_val'],
                    arrays['X_test'], arrays['y_test'],
                    student_hidden=hidden,
                    kd_T=kd_T,
                    kd_alpha=kd_alpha,
                    return_models=True,
                    verbose=True,
                )
                metrics = attach_run_config(metrics, {**run_config, 'seed': seed})
                save_seed_metrics(out_dir, seed, student_name, metrics)
                if save_models:
                    save_model_state(out_dir, seed, student_name, models)
            student_results[seed] = metrics
            agg = aggregate_seed_metrics(student_results, arrays['class_names'])
            agg.to_csv(os.path.join(out_dir, f'{student_name}_aggregate_so_far.csv'), index=False)
        all_results[student_name] = student_results

    payload = {
        'dataset': 'WSN-DS',
        'seeds': seeds,
        'kd_T': kd_T,
        'kd_alpha': kd_alpha,
        'kd_grid_search': kd_grid,
        'student_A': all_results.get('student_A_32_16', {}),
        'student_B': all_results.get('student_B_64_32', {}),
    }
    write_json_atomic(os.path.join(out_dir, 'wsnds_results.json'), payload)
    return payload


# ============================================================================
# CELL 26: Final summary, INT8, QAT, and deployment profile entrypoints
# ============================================================================
def write_final_summary_tables(out_dir: str, tables: dict) -> dict:
    """Write named result tables as CSV files plus a small manifest."""
    ensure_dir(out_dir)
    manifest = {}
    for name, rows in tables.items():
        if isinstance(rows, pd.DataFrame):
            df_out = rows
        elif isinstance(rows, dict):
            df_out = pd.DataFrame.from_dict(rows, orient='index').reset_index().rename(columns={'index': 'item'})
        else:
            df_out = pd.DataFrame(rows)
        path = os.path.join(out_dir, f'{name}.csv')
        df_out.to_csv(path, index=False)
        manifest[name] = path
    write_json_atomic(os.path.join(out_dir, 'summary_manifest.json'), manifest)
    return manifest


def select_best_kd_seed_from_results(seed_results: dict,
                                     config_name: str = 'E_KD_from_RF') -> tuple:
    """Return (seed, metric_dict) for the best macro-F1 seed of a config."""
    best_seed, best_metric, best_f1 = None, None, -np.inf
    for seed_key, per_seed in seed_results.items():
        if not isinstance(per_seed, dict) or config_name not in per_seed:
            continue
        metric = per_seed[config_name]
        f1 = metric.get('macro_f1')
        if f1 is None:
            continue
        if float(f1) > best_f1:
            best_seed = int(seed_key)
            best_metric = metric
            best_f1 = float(f1)
    if best_seed is None:
        raise ValueError(f"No usable {config_name!r} metrics found in seed results")
    return best_seed, best_metric


def safe_torch_load_state_dict(model_path: str) -> dict:
    """Load a state_dict while supporting both new and older PyTorch versions."""
    try:
        return torch.load(model_path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(model_path, map_location='cpu')


def load_student_state_for_profile(model_path: str,
                                   input_dim: int,
                                   hidden_dims: tuple,
                                   num_classes: int) -> nn.Module:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Student checkpoint not found: {model_path}")
    model = StudentMLP(input_dim, hidden_dims, num_classes)
    state = safe_torch_load_state_dict(model_path)
    model.load_state_dict(state)
    return model.cpu().eval()


def measure_cpu_latency_distribution(model: nn.Module,
                                     X_np: np.ndarray,
                                     batch_size: int = 1,
                                     n_runs: int = 500,
                                     warmup: int = 20) -> dict:
    model = copy.deepcopy(model).cpu().eval()
    X_t = torch.tensor(X_np[:batch_size], dtype=torch.float32)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(X_t)
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(X_t)
            times.append((time.perf_counter() - start) * 1000.0)
    arr = np.asarray(times, dtype=float)
    return {
        'cpu_latency_batch_size': batch_size,
        'cpu_latency_runs': n_runs,
        'cpu_latency_mean_ms': float(np.mean(arr)),
        'cpu_latency_p50_ms': float(np.percentile(arr, 50)),
        'cpu_latency_p95_ms': float(np.percentile(arr, 95)),
    }


def run_dynamic_int8_eval(model: nn.Module,
                          X_test_np_local: np.ndarray,
                          y_test_np_local: np.ndarray) -> dict:
    fp32_model = copy.deepcopy(model).cpu().eval()
    int8_model = quantize_dynamic_int8(fp32_model)
    fp32_metrics = evaluate_cpu_model(fp32_model, X_test_np_local, y_test_np_local)
    int8_metrics = evaluate_cpu_model(int8_model, X_test_np_local, y_test_np_local)
    return {
        'fp32_metrics': fp32_metrics,
        'int8_metrics': int8_metrics,
        'fp32_on_disk_kb': model_size_on_disk_kb(fp32_model),
        'int8_on_disk_kb': model_size_on_disk_kb(int8_model),
        'macro_f1_delta': float(int8_metrics['macro_f1'] - fp32_metrics['macro_f1']),
        'int8_model': int8_model,
    }


def _edgeiiot_arrays_for_final() -> dict:
    df_edge = load_edgeiiot_dataframe(EDGEIIOT_ML_PATH)
    prepared = prepare_edgeiiot_dataframe(df_edge, target_col=EDGEIIOT_TARGET_COL, verbose=False)
    arrays = split_and_scale_arrays(prepared['X'], prepared['y'], split_seed=42, scale_scope='train')
    return {
        **arrays,
        'class_names': prepared['metadata']['class_names'],
        'input_dim': prepared['metadata']['n_features'],
        'num_classes': len(prepared['metadata']['class_names']),
        'metadata': prepared['metadata'],
    }


def _student_final_specs() -> list:
    return [
        ('student_A_32_16', 'student_A', STUDENT_A_HIDDEN),
        ('student_B_64_32', 'student_B', STUDENT_B_HIDDEN),
    ]


def _qat_profile_one_student(model: nn.Module,
                             arrays: dict,
                             input_dim: int,
                             hidden_dims: tuple,
                             num_classes: int,
                             run_qat: bool,
                             run_profile: bool,
                             latency_runs: int) -> dict:
    dyn = run_dynamic_int8_eval(model, arrays['X_test'], arrays['y_test'])
    row = {
        'fp32_macro_f1': dyn['fp32_metrics']['macro_f1'],
        'dynamic_int8_macro_f1': dyn['int8_metrics']['macro_f1'],
        'dynamic_int8_f1_delta': dyn['macro_f1_delta'],
        'fp32_on_disk_kb': dyn['fp32_on_disk_kb'],
        'dynamic_int8_on_disk_kb': dyn['int8_on_disk_kb'],
    }
    if run_profile:
        row.update(profile_student_for_deployment(
            model, arrays['X_test'], input_dim, hidden_dims, num_classes, n_runs=latency_runs
        ))
        int8_latency = measure_cpu_latency_distribution(
            dyn['int8_model'], arrays['X_test'], batch_size=1, n_runs=latency_runs
        )
        row.update({f'dynamic_int8_{k}': v for k, v in int8_latency.items()})
    if run_qat:
        qat = train_qat_from_fp32_student(
            model, arrays, input_dim, hidden_dims, num_classes, epochs=10, lr=1e-4
        )
        row.update({
            'qat_int8_macro_f1': qat['metrics']['macro_f1'],
            'qat_int8_accuracy': qat['metrics']['accuracy'],
            'qat_int8_on_disk_kb': qat['on_disk_kb'],
            'qat_int8_f1_delta': float(qat['metrics']['macro_f1'] - dyn['fp32_metrics']['macro_f1']),
        })
    return row


def run_edgeiiot_qat_and_profile_entrypoint(run_qat: bool = RUN_QAT_FOR_BEST_STUDENTS,
                                            run_profile: bool = RUN_DEPLOYMENT_PROFILE,
                                            source_run: str = 'edgeiiot_ml_5seed',
                                            latency_runs: int = 500) -> dict:
    source_dir = os.path.join(FINAL_RUN_ROOT, source_run)
    results_path = os.path.join(source_dir, 'edgeiiot_results.json')
    if not os.path.exists(results_path):
        raise FileNotFoundError(
            f"Missing {results_path}. Run RUN_EDGEIIOT_ML_5SEED=True before final QAT/profile."
        )
    edge_results = read_json(results_path)
    arrays = _edgeiiot_arrays_for_final()
    out_dir = ensure_dir(os.path.join(FINAL_RUN_ROOT, 'edgeiiot_qat_profile'))
    rows, details = [], {}

    for student_name, payload_key, hidden in _student_final_specs():
        seed, metric = select_best_kd_seed_from_results(edge_results[payload_key], 'E_KD_from_RF')
        model_path = os.path.join(source_dir, f'seed_{seed}', student_name, 'models', 'E_KD_from_RF.pt')
        model = load_student_state_for_profile(model_path, arrays['input_dim'], hidden, arrays['num_classes'])
        row = _qat_profile_one_student(
            model, arrays, arrays['input_dim'], hidden, arrays['num_classes'],
            run_qat=run_qat, run_profile=run_profile, latency_runs=latency_runs
        )
        row.update({
            'dataset': 'Edge-IIoTset ML',
            'student': student_name,
            'selected_seed': seed,
            'selected_seed_macro_f1': metric.get('macro_f1'),
        })
        rows.append(row)
        details[student_name] = row

    manifest = write_final_summary_tables(out_dir, {'edgeiiot_qat_profile_summary': rows})
    payload = {'rows': rows, 'manifest': manifest, 'metadata': arrays['metadata']}
    write_json_atomic(os.path.join(out_dir, 'edgeiiot_qat_profile_results.json'), payload)
    return payload


def run_wsnds_qat_and_profile_entrypoint(run_qat: bool = RUN_QAT_FOR_BEST_STUDENTS,
                                         run_profile: bool = RUN_DEPLOYMENT_PROFILE,
                                         source_run: str = 'wsnds_10seed_resume',
                                         latency_runs: int = 500) -> dict:
    arrays = _default_wsnds_arrays()
    source_dir = os.path.join(FINAL_RUN_ROOT, source_run)
    results_path = os.path.join(source_dir, 'wsnds_results.json')
    if not os.path.exists(results_path):
        raise FileNotFoundError(
            f"Missing {results_path}. Run RUN_WSNDS_RESUME_FINAL=True before WSN-DS final QAT/profile."
        )
    wsnds_results = read_json(results_path)
    out_dir = ensure_dir(os.path.join(FINAL_RUN_ROOT, 'wsnds_qat_profile'))
    rows = []

    for student_name, payload_key, hidden in _student_final_specs():
        seed, metric = select_best_kd_seed_from_results(wsnds_results[payload_key], 'E_KD_from_RF')
        model_path = os.path.join(source_dir, f'seed_{seed}', student_name, 'models', 'E_KD_from_RF.pt')
        model = load_student_state_for_profile(model_path, arrays['input_dim'], hidden, arrays['num_classes'])
        row = _qat_profile_one_student(
            model, arrays, arrays['input_dim'], hidden, arrays['num_classes'],
            run_qat=run_qat, run_profile=run_profile, latency_runs=latency_runs
        )
        row.update({
            'dataset': 'WSN-DS',
            'student': student_name,
            'selected_seed': seed,
            'selected_seed_macro_f1': metric.get('macro_f1'),
        })
        rows.append(row)

    manifest = write_final_summary_tables(out_dir, {'wsnds_qat_profile_summary': rows})
    payload = {'rows': rows, 'manifest': manifest}
    write_json_atomic(os.path.join(out_dir, 'wsnds_qat_profile_results.json'), payload)
    return payload


# ============================================================================
# CELL 27: Final SHAP alignment entrypoints for RF-to-KD story
# ============================================================================
def _as_class_shap_list(shap_values, num_classes: int) -> list:
    if isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3 and shap_values.shape[-1] == num_classes:
            return [shap_values[:, :, i] for i in range(num_classes)]
        if shap_values.ndim == 3 and shap_values.shape[0] == num_classes:
            return [shap_values[i] for i in range(num_classes)]
        if shap_values.ndim == 2:
            return [shap_values]
    return list(shap_values)


def compute_shap_alignment(student_model: nn.Module,
                           rf_teacher,
                           X_background_np: np.ndarray,
                           X_explain_np: np.ndarray,
                           feature_names: list,
                           class_names: list,
                           seed: int = 42,
                           max_background: int = 100,
                           max_explain: int = 500) -> dict:
    """Compute global and per-class SHAP rank alignment between student and RF."""
    import shap
    from scipy.stats import spearmanr

    rng = np.random.RandomState(seed)
    bg_n = min(max_background, len(X_background_np))
    ex_n = min(max_explain, len(X_explain_np))
    bg_idx = rng.choice(len(X_background_np), bg_n, replace=False)
    ex_idx = rng.choice(len(X_explain_np), ex_n, replace=False)

    X_bg_t = torch.tensor(X_background_np[bg_idx], dtype=torch.float32).to(device)
    X_ex_t = torch.tensor(X_explain_np[ex_idx], dtype=torch.float32).to(device)
    student_model = copy.deepcopy(student_model).to(device).eval()

    student_explainer = shap.DeepExplainer(student_model, X_bg_t)
    student_shap = _as_class_shap_list(student_explainer.shap_values(X_ex_t), len(class_names))
    student_global = np.abs(np.stack(student_shap)).mean(axis=(0, 1))

    rf_explainer = shap.TreeExplainer(rf_teacher)
    rf_shap = _as_class_shap_list(rf_explainer.shap_values(X_explain_np[ex_idx]), len(class_names))
    rf_global = np.abs(np.stack(rf_shap)).mean(axis=(0, 1))

    student_ranks = pd.Series(student_global).rank(ascending=False)
    teacher_ranks = pd.Series(rf_global).rank(ascending=False)
    rho, rho_p = spearmanr(student_ranks, teacher_ranks)

    per_class = {}
    for class_idx, class_name in enumerate(class_names):
        if class_idx >= len(student_shap) or class_idx >= len(rf_shap):
            continue
        s_imp = np.abs(student_shap[class_idx]).mean(axis=0)
        t_imp = np.abs(rf_shap[class_idx]).mean(axis=0)
        class_rho, class_p = spearmanr(
            pd.Series(s_imp).rank(ascending=False),
            pd.Series(t_imp).rank(ascending=False),
        )
        per_class[class_name] = {
            'rho': float(class_rho) if class_rho == class_rho else None,
            'p': float(class_p) if class_p == class_p else None,
        }

    compare_df = pd.DataFrame({
        'feature': feature_names,
        'student_shap': student_global,
        'teacher_shap': rf_global,
    }).sort_values('teacher_shap', ascending=False)
    return {
        'ranking_agreement_spearman': float(rho) if rho == rho else None,
        'ranking_agreement_p': float(rho_p) if rho_p == rho_p else None,
        'per_class_spearman': per_class,
        'top_teacher_features': compare_df.head(20).to_dict('records'),
        'top_student_features': compare_df.sort_values('student_shap', ascending=False).head(20).to_dict('records'),
        'n_background': int(bg_n),
        'n_explain': int(ex_n),
    }


def run_wsnds_rf_kd_shap_alignment_entrypoint(max_background: int = 100,
                                              max_explain: int = 500) -> dict:
    arrays = _default_wsnds_arrays()
    out_dir = ensure_dir(os.path.join(FINAL_RUN_ROOT, 'shap_alignment'))

    if 'final_models' in globals() and 'E_KD_from_RF' in final_models:
        student = final_models['E_KD_from_RF']
        seed = SEEDS[-1]
    else:
        source_dir = os.path.join(FINAL_RUN_ROOT, 'wsnds_10seed_resume')
        wsnds_results = read_json(os.path.join(source_dir, 'wsnds_results.json'))
        seed, _ = select_best_kd_seed_from_results(wsnds_results['student_A'], 'E_KD_from_RF')
        model_path = os.path.join(source_dir, f'seed_{seed}', 'student_A_32_16', 'models', 'E_KD_from_RF.pt')
        student = load_student_state_for_profile(model_path, arrays['input_dim'], STUDENT_A_HIDDEN, arrays['num_classes'])

    if 'final_models' in globals() and 'A_RF_500' in final_models:
        rf_teacher = final_models['A_RF_500']
    else:
        rf_teacher = RandomForestClassifier(n_estimators=500, max_depth=15, random_state=seed, n_jobs=-1)
        rf_teacher.fit(arrays['X_train'], arrays['y_train'])

    alignment = compute_shap_alignment(
        student, rf_teacher,
        arrays['X_train'], arrays['X_test'],
        FEATURE_NAMES, arrays['class_names'],
        seed=seed,
        max_background=max_background,
        max_explain=max_explain,
    )
    alignment['dataset'] = 'WSN-DS'
    alignment['student'] = 'student_A_32_16_E_KD_from_RF'
    alignment['seed'] = seed
    write_json_atomic(os.path.join(out_dir, 'wsnds_rf_kd_shap_alignment.json'), alignment)
    return alignment


def run_edgeiiot_shap_alignment_entrypoint(source_run: str = 'edgeiiot_ml_5seed',
                                           max_background: int = 100,
                                           max_explain: int = 500,
                                           rf_trees: int = 500) -> dict:
    source_dir = os.path.join(FINAL_RUN_ROOT, source_run)
    edge_results = read_json(os.path.join(source_dir, 'edgeiiot_results.json'))
    arrays = _edgeiiot_arrays_for_final()
    out_dir = ensure_dir(os.path.join(FINAL_RUN_ROOT, 'shap_alignment'))

    candidates = []
    for student_name, payload_key, hidden in _student_final_specs():
        seed, metric = select_best_kd_seed_from_results(edge_results[payload_key], 'E_KD_from_RF')
        candidates.append((float(metric['macro_f1']), student_name, payload_key, hidden, seed))
    _, student_name, _, hidden, seed = sorted(candidates, reverse=True)[0]

    model_path = os.path.join(source_dir, f'seed_{seed}', student_name, 'models', 'E_KD_from_RF.pt')
    student = load_student_state_for_profile(model_path, arrays['input_dim'], hidden, arrays['num_classes'])
    rf_teacher = RandomForestClassifier(n_estimators=rf_trees, max_depth=15, random_state=seed, n_jobs=-1)
    rf_teacher.fit(arrays['X_train'], arrays['y_train'])

    alignment = compute_shap_alignment(
        student, rf_teacher,
        arrays['X_train'], arrays['X_test'],
        arrays['metadata']['feature_names'], arrays['class_names'],
        seed=seed,
        max_background=max_background,
        max_explain=max_explain,
    )
    alignment['dataset'] = 'Edge-IIoTset ML'
    alignment['student'] = f'{student_name}_E_KD_from_RF'
    alignment['seed'] = seed
    write_json_atomic(os.path.join(out_dir, 'edgeiiot_rf_kd_shap_alignment.json'), alignment)
    return alignment


# ============================================================================
# CELL 28: Final guarded execution layer
# ============================================================================
if final_mode_enabled('wsnds_final', RUN_WSNDS_RESUME_FINAL):
    wsnds_resume_results = run_wsnds_final_with_resume()

if final_mode_enabled('qat_profile', RUN_QAT_FOR_BEST_STUDENTS or RUN_DEPLOYMENT_PROFILE):
    final_qat_profile_outputs = {}
    try:
        final_qat_profile_outputs['wsnds'] = run_wsnds_qat_and_profile_entrypoint(
            run_qat=RUN_QAT_FOR_BEST_STUDENTS or FINAL_RUN_MODE == 'qat_profile',
            run_profile=RUN_DEPLOYMENT_PROFILE or FINAL_RUN_MODE == 'qat_profile',
        )
    except Exception as e:
        print(f"WSN-DS QAT/profile skipped: {e}")
    try:
        final_qat_profile_outputs['edgeiiot'] = run_edgeiiot_qat_and_profile_entrypoint(
            run_qat=RUN_QAT_FOR_BEST_STUDENTS or FINAL_RUN_MODE == 'qat_profile',
            run_profile=RUN_DEPLOYMENT_PROFILE or FINAL_RUN_MODE == 'qat_profile',
        )
    except Exception as e:
        print(f"Edge-IIoTset QAT/profile skipped: {e}")
    if final_qat_profile_outputs:
        write_json_atomic(
            os.path.join(FINAL_RUN_ROOT, 'final_qat_profile_outputs.json'),
            final_qat_profile_outputs,
        )

if final_mode_enabled('shap_alignment', RUN_SHAP_ALIGNMENT_FINAL):
    final_shap_alignment_outputs = {}
    try:
        final_shap_alignment_outputs['wsnds'] = run_wsnds_rf_kd_shap_alignment_entrypoint()
    except Exception as e:
        print(f"WSN-DS SHAP alignment skipped: {e}")
    try:
        final_shap_alignment_outputs['edgeiiot'] = run_edgeiiot_shap_alignment_entrypoint()
    except Exception as e:
        print(f"Edge-IIoTset SHAP alignment skipped: {e}")
    if final_shap_alignment_outputs:
        write_json_atomic(
            os.path.join(FINAL_RUN_ROOT, 'final_shap_alignment_outputs.json'),
            final_shap_alignment_outputs,
        )
