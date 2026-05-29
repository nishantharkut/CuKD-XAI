# ============================================================================
#  !!!! WARNING - READ BEFORE EDITING !!!!
#
#  This .py file is the Python source for the matching notebook. Make source
#  edits here, then regenerate the notebook with make_notebook_preserve_banner.py.
#
#  Root route:
#      python3 make_notebook_preserve_banner.py
#
#  v2.3-plus route:
#      python3 make_notebook_preserve_banner.py cukd_xai_v23_plus_addons.py cukd_xai_v23_plus_addons.ipynb
#
#  Bug fixes applied April 11, 2026 are marked `# FIXED 2026-04-11:` inline.
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
#
# Author: Nishant Harkut (2023IMG-040), ABV-IIITM Gwalior
# ============================================================================

# ============================================================================
# CELL 1: Install dependencies
# ============================================================================
import importlib.util
import subprocess
import sys

_REQUIRED_PACKAGES = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'torch': 'torch',
    'sklearn': 'scikit-learn',
    'scipy': 'scipy',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'shap': 'shap',
    'imblearn': 'imbalanced-learn',
}
_missing_packages = [pkg for module, pkg in _REQUIRED_PACKAGES.items()
                     if importlib.util.find_spec(module) is None]
if _missing_packages:
    print(f"Installing missing packages: {_missing_packages}")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *_missing_packages])
else:
    print('All required packages already installed.')

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

# ----------------------------------------------------------------------------
# EXPERIMENT CONFIGURATION
# ----------------------------------------------------------------------------
SEEDS_V2_FINAL5 = [42, 123, 456, 789, 1001]
SEEDS_PUBLICATION10 = [42, 123, 456, 789, 1001, 2024, 3141, 5678, 8192, 9999]
FINAL_RUN_MODE = 'wsnds_final'              # final route: original v2.3 with 10 publication seeds
SEEDS = SEEDS_PUBLICATION10 if FINAL_RUN_MODE == 'wsnds_final' else SEEDS_V2_FINAL5
N_SEEDS = len(SEEDS)
QUICK_MODE = False                        # final route default; set True only for a reduced sanity run
WSNDS_PATH = 'WSN-DS.csv'                 # Path to WSN-DS

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
print(f"Final run mode: {FINAL_RUN_MODE}")
print(f"Seeds: {SEEDS}")
print(f"Quick mode: {QUICK_MODE}")

if QUICK_MODE:
    SEEDS = SEEDS[:1]
    N_SEEDS = len(SEEDS)
    print("QUICK_MODE=True: reduced sanity run only; final publication outputs require QUICK_MODE=False.")

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
    """Measure serialized model size by saving to disk."""
    tmp = '/tmp/_tmp_model.pt'
    torch.save(model.state_dict(), tmp)
    size = os.path.getsize(tmp) / 1024
    os.remove(tmp)
    return size


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
    except ImportError as ex:
        raise RuntimeError(
            "imbalanced-learn is required for Config I. Install imbalanced-learn before running the notebook."
        ) from ex
    except Exception as ex:
        raise RuntimeError(
            "Config I failed; stopping so required outputs are not silently missing."
        ) from ex

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

# Optional: run KD hyperparameter grid search once on seed 42
BEST_T, BEST_ALPHA = KD_T_DEFAULT, KD_ALPHA_DEFAULT
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
all_seed_results_A = {}
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
all_seed_results_B = {}
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
            raise RuntimeError(f'Config has no seed results: {cfg}')

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
        raise RuntimeError(f'Insufficient paired data for Wilcoxon: {cfg_a} vs {cfg_b}')
    diffs = np.array(vals_a) - np.array(vals_b)
    if np.all(diffs == 0):
        return {'stat': 0.0, 'p': 1.0, 'diff': 0.0,
                'verdict': 'identical'}
    try:
        stat, p = wilcoxon(vals_a, vals_b, zero_method='wilcox')
    except Exception as ex:
        raise RuntimeError(f'Wilcoxon test failed for {cfg_a} vs {cfg_b}') from ex
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
        raise RuntimeError(f"Required Wilcoxon configs missing for comparison: {desc}")
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
        raise RuntimeError(f"Failed to save student summary plot: {e}") from e

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
except ImportError as ex:
    raise RuntimeError("shap is required for Cell 12. Install shap before running the notebook.") from ex
except Exception as e:
    raise RuntimeError("SHAP analysis failed; stopping so required SHAP outputs are not silently missing.") from e

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
        raise RuntimeError(f"Required quantization model missing: {cfg_name}")
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
missing_bench_models = [key for _, key in candidate_models if key not in final_models]
if missing_bench_models:
    raise RuntimeError(f"Required benchmark models missing: {missing_bench_models}")
models_to_bench = {name: final_models[key] for name, key in candidate_models}

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
# Student B must be present for the final run.
if len(all_seed_results_B) == 0:
    raise RuntimeError('Student B results are missing. Keep QUICK_MODE=False for final runs.')
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
# CELL 16: Save all results and print final summary
# ============================================================================
def json_convert(o):
    if isinstance(o, (np.integer, np.int_)):
        return int(o)
    if isinstance(o, (np.floating, np.float_)):
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
if len(all_seed_results_B) == 0:
    raise RuntimeError('Student B results are missing. Keep QUICK_MODE=False for final runs.')
agg_B = aggregate_multi_seed(all_seed_results_B)
agg_B.to_csv('wsnds_results_student_B.csv', index=False)
print("Saved wsnds_results_student_B.csv")

# ============================================================================
# CELL 17: Final summary for paper
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
    raise RuntimeError("Teacher or student model missing from final_models; cannot compute compression summary")

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

# ============================================================================
# CELL 18: Add-on configuration and shared utilities
# ============================================================================
# This file is a COPY of v2.3 plus complete post-run add-ons.
# The root v2.3 notebook is not required to use this section.
# Add-ons are controlled by these flags. The complete plus route runs all
# four add-ons after EDGEIIOT_ML_PATH points to the correct CSV.

RUN_EDGEIIOT_GENERALIZATION_ADDON = True
RUN_QAT_ADDON = True
RUN_DEPLOYMENT_PROOF_ADDON = True
RUN_RF_KD_SHAP_ALIGNMENT_ADDON = True

ADDON_OUTPUT_DIR = 'v23_plus_addon_outputs'
EDGEIIOT_ML_PATH = 'Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv'
EDGEIIOT_TARGET_COL = 'Attack_type'
EDGEIIOT_SEEDS = [42, 123, 456, 789, 1001]
EDGEIIOT_RF_TREES = 300
EDGEIIOT_MAX_CATEGORICAL_CARDINALITY = 64
EDGEIIOT_RARE_CATEGORY_MIN_COUNT = 10
EDGEIIOT_RF_CALIBRATION_METHOD = "sigmoid"
EDGEIIOT_RF_CALIBRATION_CV = 3
EDGEIIOT_STUDENT_SPECS = [
    ('student_A_32_16', STUDENT_A_HIDDEN),
    ('student_B_64_32', STUDENT_B_HIDDEN),
]
EDGEIIOT_LEAKAGE_COLUMNS = [
    'frame.time', 'ip.src_host', 'ip.dst_host',
    'arp.src.proto_ipv4', 'arp.dst.proto_ipv4',
    'http.file_data', 'http.request.full_uri', 'icmp.transmit_timestamp',
    'http.request.uri.query', 'tcp.options', 'tcp.payload',
    'tcp.srcport', 'tcp.dstport', 'udp.port', 'mqtt.msg',
]
EDGEIIOT_AUXILIARY_TARGET_COLUMNS = [
    'Attack_label', 'attack_label', 'Attack Label', 'Label', 'label',
    'class', 'Class', 'Attack_type', 'attack_type', 'Attack Type',
]


def ensure_addon_dir(path: str = ADDON_OUTPUT_DIR) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def addon_json_convert(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return o.to_dict('records')
    return str(o)


def addon_write_json(path: str, payload: dict) -> None:
    ensure_addon_dir(os.path.dirname(path) or '.')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, default=addon_json_convert)


def addon_class_weights(y_np: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y_np, minlength=num_classes)
    return torch.tensor(len(y_np) / (num_classes * np.maximum(counts, 1)), dtype=torch.float32)


def sklearn_metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, class_names: list) -> dict:
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    pc_prec, pc_rec, pc_f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    return {
        'accuracy': float(acc),
        'macro_precision': float(prec),
        'macro_recall': float(rec),
        'macro_f1': float(f1),
        'per_class_precision': pc_prec.tolist(),
        'per_class_recall': pc_rec.tolist(),
        'per_class_f1': pc_f1.tolist(),
        'class_names': class_names,
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }



def _batched_predict_cpu_addon(model: nn.Module, X, batch_size: int = 4096):
    model_cpu = model.cpu().eval()
    X_cpu = X if torch.is_tensor(X) else torch.tensor(X, dtype=torch.float32)
    X_cpu = X_cpu.cpu().float()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_cpu), batch_size):
            preds.append(model_cpu(X_cpu[i:i + batch_size]).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def evaluate_model_cpu_addon(model: nn.Module, X, y, batch_size: int = 4096) -> dict:
    y_np = y.cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
    preds = _batched_predict_cpu_addon(model, X, batch_size=batch_size)
    acc = accuracy_score(y_np, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_np, preds, average='macro', zero_division=0)
    pc_prec, pc_rec, pc_f1, _ = precision_recall_fscore_support(y_np, preds, average=None, zero_division=0)
    return {
        'accuracy': float(acc),
        'macro_precision': float(prec),
        'macro_recall': float(rec),
        'macro_f1': float(f1),
        'per_class_precision': pc_prec.tolist(),
        'per_class_recall': pc_rec.tolist(),
        'per_class_f1': pc_f1.tolist(),
        'confusion_matrix': confusion_matrix(y_np, preds).tolist(),
    }


def train_standard_cpu_addon(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor,
                             X_val: torch.Tensor, y_val: torch.Tensor,
                             class_weights: torch.Tensor = None,
                             epochs: int = 8, batch_size: int = 256,
                             lr: float = 1e-4, weight_decay: float = 1e-3,
                             patience: int = 4):
    model = model.cpu()
    X_train = X_train.cpu().float()
    y_train = y_train.cpu().long()
    X_val = X_val.cpu().float()
    y_val_np = y_val.cpu().numpy()
    if class_weights is not None:
        class_weights = class_weights.cpu().float()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    best_val, best_state, bad = 0.0, None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()
        preds = _batched_predict_cpu_addon(model, X_val)
        val_f1 = f1_score(y_val_np, preds, average='macro')
        if val_f1 > best_val:
            best_val = val_f1
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu().eval()

def train_kd_cpu_addon(student: nn.Module,
                       teacher_probs,
                       X_train: torch.Tensor, y_train: torch.Tensor,
                       X_val: torch.Tensor, y_val: torch.Tensor,
                       T: float = KD_T_DEFAULT, alpha: float = KD_ALPHA_DEFAULT,
                       class_weights: torch.Tensor = None,
                       epochs: int = 8, batch_size: int = 256,
                       lr: float = 1e-4, weight_decay: float = 1e-3,
                       patience: int = 4):
    student = student.cpu()
    X_train = X_train.cpu().float()
    y_train = y_train.cpu().long()
    X_val = X_val.cpu().float()
    y_val_np = y_val.cpu().numpy()
    probs = teacher_probs if torch.is_tensor(teacher_probs) else torch.tensor(teacher_probs, dtype=torch.float32)
    probs = probs.cpu().float()
    pseudo_logits = torch.log(probs.clamp(min=1e-8))
    soft_targets = F.softmax(pseudo_logits / T, dim=1).detach()

    if class_weights is not None:
        class_weights = class_weights.cpu().float()
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    loader = DataLoader(TensorDataset(X_train, y_train, soft_targets), batch_size=batch_size, shuffle=True)
    best_val, best_state, bad = 0.0, None, 0

    for _ in range(epochs):
        student.train()
        for xb, yb, sb in loader:
            optimizer.zero_grad()
            logits = student(xb)
            log_soft_s = F.log_softmax(logits / T, dim=1)
            kd_term = F.kl_div(log_soft_s, sb, reduction="batchmean") * (T * T)
            ce_term = ce_loss(logits, yb)
            loss = alpha * kd_term + (1 - alpha) * ce_term
            loss.backward()
            optimizer.step()
        scheduler.step()
        preds = _batched_predict_cpu_addon(student, X_val)
        val_f1 = f1_score(y_val_np, preds, average="macro")
        if val_f1 > best_val:
            best_val = val_f1
            best_state = copy.deepcopy(student.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        student.load_state_dict(best_state)
    return student.cpu().eval()


# ============================================================================
# CELL 19: Edge-IIoTset generalization add-on
# ============================================================================
def edgeiiot_normalize_categorical_addon(series: pd.Series) -> pd.Series:
    values = series.where(series.notna(), "__MISSING__").astype(str).str.strip()
    return values.mask(values == "", "__EMPTY__")


def edgeiiot_fit_category_caps_addon(X_train_raw: pd.DataFrame, categorical_cols: list) -> tuple:
    allowed_categories = {}
    category_stats = {}
    for col in categorical_cols:
        values = edgeiiot_normalize_categorical_addon(X_train_raw[col])
        counts = values.value_counts(dropna=False)
        retained = counts[counts >= EDGEIIOT_RARE_CATEGORY_MIN_COUNT].head(
            EDGEIIOT_MAX_CATEGORICAL_CARDINALITY
        ).index.tolist()
        if not retained and len(counts) > 0:
            retained = counts.head(min(EDGEIIOT_MAX_CATEGORICAL_CARDINALITY, len(counts))).index.tolist()
        allowed_categories[col] = set(retained)
        category_stats[col] = {
            "raw_cardinality": int(len(counts)),
            "retained_categories": int(len(retained)),
            "other_rows_in_train": int((~values.isin(retained)).sum()),
        }
    return allowed_categories, category_stats


def edgeiiot_encode_split_addon(X_raw: pd.DataFrame, numeric_cols: list, categorical_cols: list,
                                allowed_categories: dict, dummy_columns: list = None) -> pd.DataFrame:
    parts = []
    if numeric_cols:
        X_numeric = X_raw[numeric_cols].apply(pd.to_numeric, errors="coerce")
        X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
        parts.append(X_numeric)
    if categorical_cols:
        X_cat = pd.DataFrame(index=X_raw.index)
        for col in categorical_cols:
            values = edgeiiot_normalize_categorical_addon(X_raw[col])
            retained = allowed_categories.get(col, set())
            X_cat[col] = values.where(values.isin(retained), "__OTHER__")
        X_dummy = pd.get_dummies(X_cat, columns=categorical_cols, dummy_na=False, dtype=np.float32)
        if dummy_columns is not None:
            X_dummy = X_dummy.reindex(columns=dummy_columns, fill_value=0.0).astype(np.float32)
        parts.append(X_dummy)
    if not parts:
        raise ValueError("Edge-IIoTset preprocessing produced no usable features")
    return pd.concat(parts, axis=1)


def prepare_edgeiiot_ml_for_addon(csv_path: str = EDGEIIOT_ML_PATH,
                                  target_col: str = EDGEIIOT_TARGET_COL) -> dict:
    df_edge = pd.read_csv(csv_path)
    df_edge.columns = df_edge.columns.str.strip()
    if target_col not in df_edge.columns:
        raise ValueError(f"{target_col!r} not found. Available columns: {df_edge.columns.tolist()}")

    rows_raw, cols_raw = df_edge.shape
    candidate_leakage = EDGEIIOT_LEAKAGE_COLUMNS + [
        c for c in EDGEIIOT_AUXILIARY_TARGET_COLUMNS if c != target_col
    ]
    removed_leakage = [c for c in dict.fromkeys(candidate_leakage) if c in df_edge.columns]
    df_edge = df_edge.drop(columns=removed_leakage)
    df_edge = df_edge.dropna().drop_duplicates().reset_index(drop=True)

    y_labels = df_edge[target_col].astype(str).str.strip()
    X_df = df_edge.drop(columns=[target_col])
    zero_var = [c for c in X_df.columns if X_df[c].nunique(dropna=False) <= 1]
    X_df = X_df.drop(columns=zero_var)
    categorical_cols = X_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = [c for c in X_df.columns if c not in categorical_cols]

    le_edge = LabelEncoder()
    y = le_edge.fit_transform(y_labels).astype(np.int64)

    X_trainval_raw, X_test_raw, y_trainval, y_test = train_test_split(
        X_df, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_trainval_raw, y_trainval, test_size=0.1765, random_state=42, stratify=y_trainval
    )

    allowed_categories, category_stats = edgeiiot_fit_category_caps_addon(X_train_raw, categorical_cols)
    X_train_encoded = edgeiiot_encode_split_addon(X_train_raw, numeric_cols, categorical_cols, allowed_categories)
    dummy_columns = [c for c in X_train_encoded.columns if c not in numeric_cols]
    X_val_encoded = edgeiiot_encode_split_addon(
        X_val_raw, numeric_cols, categorical_cols, allowed_categories, dummy_columns=dummy_columns
    )
    X_test_encoded = edgeiiot_encode_split_addon(
        X_test_raw, numeric_cols, categorical_cols, allowed_categories, dummy_columns=dummy_columns
    )

    constant_encoded_cols = [c for c in X_train_encoded.columns if X_train_encoded[c].nunique(dropna=False) <= 1]
    if constant_encoded_cols:
        X_train_encoded = X_train_encoded.drop(columns=constant_encoded_cols)
        X_val_encoded = X_val_encoded.drop(columns=constant_encoded_cols)
        X_test_encoded = X_test_encoded.drop(columns=constant_encoded_cols)

    continuous_cols = [c for c in numeric_cols if c in X_train_encoded.columns]
    scaler_edge = StandardScaler()
    if continuous_cols:
        X_train_encoded.loc[:, continuous_cols] = scaler_edge.fit_transform(X_train_encoded[continuous_cols]).astype(np.float32)
        X_val_encoded.loc[:, continuous_cols] = scaler_edge.transform(X_val_encoded[continuous_cols]).astype(np.float32)
        X_test_encoded.loc[:, continuous_cols] = scaler_edge.transform(X_test_encoded[continuous_cols]).astype(np.float32)

    X_train = X_train_encoded.to_numpy(dtype=np.float32, copy=True)
    X_val = X_val_encoded.to_numpy(dtype=np.float32, copy=True)
    X_test = X_test_encoded.to_numpy(dtype=np.float32, copy=True)
    feature_names = X_train_encoded.columns.tolist()

    metadata = {
        "rows_raw": int(rows_raw),
        "cols_raw": int(cols_raw),
        "rows_after_cleaning": int(len(df_edge)),
        "removed_leakage_columns": removed_leakage,
        "removed_zero_variance_columns": zero_var,
        "removed_constant_encoded_columns": constant_encoded_cols,
        "categorical_columns": categorical_cols,
        "numeric_columns_scaled": continuous_cols,
        "categorical_cardinality_policy": {
            "max_categories_per_column": int(EDGEIIOT_MAX_CATEGORICAL_CARDINALITY),
            "rare_category_min_count": int(EDGEIIOT_RARE_CATEGORY_MIN_COUNT),
        },
        "categorical_cardinality_stats": category_stats,
        "n_features_after_encoding": int(X_train.shape[1]),
        "class_names": le_edge.classes_.tolist(),
        "class_distribution": dict(zip(le_edge.classes_.tolist(), np.bincount(y).astype(int).tolist())),
    }
    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "feature_names": feature_names,
        "class_names": le_edge.classes_.tolist(),
        "input_dim": int(X_train.shape[1]),
        "num_classes": int(len(le_edge.classes_)),
        "metadata": metadata,
    }


def edgeiiot_predict_proba_aligned_addon(model, X_np: np.ndarray, num_classes: int) -> np.ndarray:
    raw = model.predict_proba(X_np)
    classes = getattr(model, "classes_", np.arange(raw.shape[1]))
    aligned = np.zeros((len(X_np), num_classes), dtype=np.float32)
    for idx, cls in enumerate(classes):
        cls_idx = int(cls)
        if 0 <= cls_idx < num_classes:
            aligned[:, cls_idx] = raw[:, idx]
    row_sums = aligned.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0) or not np.all(np.isfinite(aligned)):
        raise RuntimeError("Edge-IIoTset RF teacher produced invalid probability rows")
    return aligned / row_sums


def fit_edgeiiot_rf_addon(arrays: dict, seed: int,
                          rf_trees: int = EDGEIIOT_RF_TREES):
    base_rf = RandomForestClassifier(
        n_estimators=rf_trees, max_depth=15, random_state=seed, n_jobs=-1
    )
    class_counts = np.bincount(arrays["y_train"], minlength=arrays["num_classes"])
    min_class_count = int(class_counts.min()) if len(class_counts) else 0
    cv = min(int(EDGEIIOT_RF_CALIBRATION_CV), min_class_count)
    if cv >= 2:
        rf = CalibratedClassifierCV(
            base_rf,
            method=EDGEIIOT_RF_CALIBRATION_METHOD,
            cv=cv,
        )
        calibration_info = {
            "method": EDGEIIOT_RF_CALIBRATION_METHOD,
            "cv": int(cv),
            "min_train_class_count": min_class_count,
        }
    else:
        rf = base_rf
        calibration_info = {
            "method": "none",
            "cv": 0,
            "min_train_class_count": min_class_count,
        }
    rf.fit(arrays["X_train"], arrays["y_train"])
    rf_proba = edgeiiot_predict_proba_aligned_addon(rf, arrays["X_test"], arrays["num_classes"])
    rf_pred = np.argmax(rf_proba, axis=1)
    rf_metrics = sklearn_metrics_dict(arrays["y_test"], rf_pred, arrays["class_names"])
    rf_metrics["calibration"] = calibration_info
    return rf, rf_metrics

def run_edgeiiot_one_seed_addon(arrays: dict, seed: int, student_hidden: tuple,
                                rf_trees: int = EDGEIIOT_RF_TREES,
                                kd_T: float = KD_T_DEFAULT,
                                kd_alpha: float = KD_ALPHA_DEFAULT,
                                rf_model=None, rf_metrics: dict = None) -> dict:
    set_seed(seed)
    X_train, y_train = arrays['X_train'], arrays['y_train']
    X_val, y_val = arrays['X_val'], arrays['y_val']
    X_test, y_test = arrays['X_test'], arrays['y_test']
    input_dim, num_classes = arrays['input_dim'], arrays['num_classes']
    class_names = arrays['class_names']

    if rf_model is None or rf_metrics is None:
        rf_model, rf_metrics = fit_edgeiiot_rf_addon(arrays, seed, rf_trees)
    rf = rf_model

    Xtr_t = torch.tensor(X_train, dtype=torch.float32)
    ytr_t = torch.tensor(y_train, dtype=torch.long)
    Xv_t = torch.tensor(X_val, dtype=torch.float32)
    yv_t = torch.tensor(y_val, dtype=torch.long)
    Xte_t = torch.tensor(X_test, dtype=torch.float32)
    yte_t = torch.tensor(y_test, dtype=torch.long)
    cw = addon_class_weights(y_train, num_classes)

    scratch = StudentMLP(input_dim, student_hidden, num_classes)
    scratch = train_standard_cpu_addon(
        scratch, Xtr_t, ytr_t, Xv_t, yv_t, class_weights=cw,
        epochs=TRAIN_CONFIG["epochs"], batch_size=TRAIN_CONFIG["batch_size"],
        lr=TRAIN_CONFIG["lr"], weight_decay=TRAIN_CONFIG["weight_decay"],
        patience=TRAIN_CONFIG["patience"],
    )
    scratch_metrics = evaluate_model_cpu_addon(scratch, Xte_t, yte_t)
    scratch_metrics.update({
        'params': count_params(scratch),
        'model_size_kb': model_size_kb(scratch),
        'model_size_kb_int8_theoretical': model_size_kb(scratch, 1),
        'flops_per_sample': compute_flops_mlp(input_dim, student_hidden, num_classes),
    })

    rf_soft = edgeiiot_predict_proba_aligned_addon(rf, X_train, num_classes)
    kd_student = StudentMLP(input_dim, student_hidden, num_classes)
    kd_student = train_kd_cpu_addon(
        kd_student, rf_soft, Xtr_t, ytr_t, Xv_t, yv_t,
        T=kd_T, alpha=kd_alpha, class_weights=cw,
        epochs=TRAIN_CONFIG['epochs'], batch_size=TRAIN_CONFIG['batch_size'],
        lr=TRAIN_CONFIG['lr'], weight_decay=TRAIN_CONFIG['weight_decay'],
        patience=TRAIN_CONFIG['patience'],
    )
    kd_metrics = evaluate_model_cpu_addon(kd_student, Xte_t, yte_t)
    kd_metrics.update({
        'params': count_params(kd_student),
        'model_size_kb': model_size_kb(kd_student),
        'model_size_kb_int8_theoretical': model_size_kb(kd_student, 1),
        'flops_per_sample': compute_flops_mlp(input_dim, student_hidden, num_classes),
    })
    return {'A_RF_calibrated': rf_metrics, 'D_student_scratch': scratch_metrics, 'E_KD_from_RF': kd_metrics}


def aggregate_addon_seed_metrics(seed_results: dict, config_name: str) -> dict:
    f1s, accs = [], []
    for per_seed in seed_results.values():
        if config_name in per_seed:
            f1s.append(per_seed[config_name]['macro_f1'])
            accs.append(per_seed[config_name]['accuracy'])
    return {
        'macro_f1_mean': float(np.mean(f1s)) if f1s else None,
        'macro_f1_std': float(np.std(f1s)) if f1s else None,
        'accuracy_mean': float(np.mean(accs)) if accs else None,
        'accuracy_std': float(np.std(accs)) if accs else None,
        'n_seeds': len(f1s),
    }


def run_edgeiiot_generalization_addon(csv_path: str = EDGEIIOT_ML_PATH,
                                      seeds: list = EDGEIIOT_SEEDS) -> dict:
    out_dir = ensure_addon_dir(os.path.join(ADDON_OUTPUT_DIR, 'edgeiiot_generalization'))
    arrays = prepare_edgeiiot_ml_for_addon(csv_path)
    payload = {'metadata': arrays['metadata'], 'students': {}, 'seeds': seeds}
    summary_rows = []
    student_seed_results = {student_name: {} for student_name, _ in EDGEIIOT_STUDENT_SPECS}
    for seed in seeds:
        print(f'[Edge-IIoTset] RF teacher seed {seed}')
        rf_model, rf_metrics = fit_edgeiiot_rf_addon(arrays, seed)
        for student_name, hidden in EDGEIIOT_STUDENT_SPECS:
            print(f'[Edge-IIoTset] {student_name} seed {seed}')
            student_seed_results[student_name][seed] = run_edgeiiot_one_seed_addon(
                arrays, seed, hidden, rf_model=rf_model, rf_metrics=copy.deepcopy(rf_metrics)
            )
    for student_name, _ in EDGEIIOT_STUDENT_SPECS:
        seed_results = student_seed_results[student_name]
        payload['students'][student_name] = seed_results
        for config_name in ['A_RF_calibrated', 'D_student_scratch', 'E_KD_from_RF']:
            row = {'student': student_name, 'config': config_name, **aggregate_addon_seed_metrics(seed_results, config_name)}
            summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, 'edgeiiot_generalization_summary.csv'), index=False)
    addon_write_json(os.path.join(out_dir, 'edgeiiot_generalization_results.json'), payload)
    return {'payload': payload, 'summary': summary_df}


def infer_student_mlp_io_addon(model: nn.Module) -> tuple:
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not linears:
        raise RuntimeError("Could not infer Linear input/output dimensions from model")
    return int(linears[0].in_features), int(linears[-1].out_features)


def validate_feature_matrix_for_model_addon(model: nn.Module, X_np: np.ndarray, context: str) -> tuple:
    input_dim, num_classes = infer_student_mlp_io_addon(model)
    if X_np.shape[1] != input_dim:
        raise RuntimeError(
            f"{context} feature dimension mismatch: model expects {input_dim}, got {X_np.shape[1]}"
        )
    return input_dim, num_classes


# ============================================================================
# CELL 20: QAT add-on
# ============================================================================
class AddonQATStudentMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple, num_classes: int):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.net = StudentMLP(input_dim, hidden_dims, num_classes).net
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        return self.dequant(self.net(self.quant(x)))

    def fuse_model(self):
        fuse_pairs = []
        names = list(self.net._modules.keys())
        for left, right in zip(names[::2], names[1::2]):
            if isinstance(self.net._modules[left], nn.Linear) and isinstance(self.net._modules[right], nn.ReLU):
                fuse_pairs.append([left, right])
        if fuse_pairs:
            fuse_fn = getattr(torch.ao.quantization, "fuse_modules_qat", torch.ao.quantization.fuse_modules)
            fuse_fn(self.net, fuse_pairs, inplace=True)


def copy_student_weights_into_qat_model(fp32_model: nn.Module, qat_model: nn.Module) -> None:
    qat_linears = [m for m in qat_model.modules() if isinstance(m, nn.Linear)]
    fp32_linears = [m for m in fp32_model.modules() if isinstance(m, nn.Linear)]
    if len(qat_linears) != len(fp32_linears):
        raise ValueError(f'QAT layer count mismatch: {len(qat_linears)} vs {len(fp32_linears)}')
    for q_layer, f_layer in zip(qat_linears, fp32_linears):
        q_layer.weight.data.copy_(f_layer.weight.data.cpu())
        q_layer.bias.data.copy_(f_layer.bias.data.cpu())


def select_quant_backend_addon() -> str:
    if 'fbgemm' in torch.backends.quantized.supported_engines:
        return 'fbgemm'
    if 'qnnpack' in torch.backends.quantized.supported_engines:
        return 'qnnpack'
    return torch.backends.quantized.supported_engines[0]


def train_qat_student_addon(fp32_model: nn.Module, hidden_dims: tuple,
                            X_train_arr: np.ndarray = None, y_train_arr: np.ndarray = None,
                            X_val_arr: np.ndarray = None, y_val_arr: np.ndarray = None,
                            X_test_arr: np.ndarray = None, y_test_arr: np.ndarray = None,
                            epochs: int = 8, lr: float = 1e-4) -> dict:
    X_train_arr = X_train_np if X_train_arr is None else X_train_arr
    y_train_arr = y_train_np if y_train_arr is None else y_train_arr
    X_val_arr = X_val_np if X_val_arr is None else X_val_arr
    y_val_arr = y_val_np if y_val_arr is None else y_val_arr
    X_test_arr = X_test_np if X_test_arr is None else X_test_arr
    y_test_arr = y_test_np if y_test_arr is None else y_test_arr

    input_dim, num_classes = validate_feature_matrix_for_model_addon(fp32_model, X_train_arr, "QAT train")
    validate_feature_matrix_for_model_addon(fp32_model, X_val_arr, "QAT validation")
    validate_feature_matrix_for_model_addon(fp32_model, X_test_arr, "QAT test")

    backend = select_quant_backend_addon()
    torch.backends.quantized.engine = backend
    qat_model = AddonQATStudentMLP(input_dim, hidden_dims, num_classes)
    copy_student_weights_into_qat_model(fp32_model.cpu().eval(), qat_model)
    qat_model.train()
    qat_model.fuse_model()
    qat_model.qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
    torch.ao.quantization.prepare_qat(qat_model, inplace=True)

    Xtr_t = torch.tensor(X_train_arr, dtype=torch.float32)
    ytr_t = torch.tensor(y_train_arr, dtype=torch.long)
    Xv_t = torch.tensor(X_val_arr, dtype=torch.float32)
    yv_t = torch.tensor(y_val_arr, dtype=torch.long)
    cw = addon_class_weights(y_train_arr, num_classes)
    qat_model = train_standard_cpu_addon(
        qat_model, Xtr_t, ytr_t, Xv_t, yv_t, class_weights=cw,
        epochs=epochs, batch_size=256, lr=lr,
        weight_decay=TRAIN_CONFIG["weight_decay"], patience=4,
    )
    qat_model.cpu().eval()
    converted = torch.ao.quantization.convert(qat_model, inplace=False)
    metrics = evaluate_model_cpu_addon(converted, X_test_arr, y_test_arr)
    return {"backend": backend, "metrics": metrics, "serialized_qat_int8_kb": model_size_on_disk_kb(converted)}


def run_qat_addon_for_final_models() -> dict:
    out_dir = ensure_addon_dir(os.path.join(ADDON_OUTPUT_DIR, 'qat'))
    targets = [
        ('student_A_D_scratch', 'D_Small_MLP', STUDENT_A_HIDDEN),
        ('student_A_E_KD_from_RF', 'E_KD_from_RF', STUDENT_A_HIDDEN),
        ('student_A_F_KD_from_CL', 'F_KD_from_CL_MLP', STUDENT_A_HIDDEN),
    ]
    rows = []
    details = {}
    for label, model_key, hidden in targets:
        if model_key not in final_models:
            raise RuntimeError(f'[QAT] required model missing: {model_key}')
        fp32_model = final_models[model_key].cpu().eval()
        validate_feature_matrix_for_model_addon(fp32_model, X_test_np, "QAT fp32 evaluation")
        fp32_metrics = evaluate_model_cpu_addon(fp32_model, X_test_np, y_test_np)
        dynamic_model = quantize_dynamic_int8(fp32_model)
        dynamic_metrics = evaluate_model_cpu_addon(dynamic_model, X_test_np, y_test_np)
        qat = train_qat_student_addon(
            fp32_model, hidden,
            X_train_np, y_train_np, X_val_np, y_val_np, X_test_np, y_test_np,
        )
        row = {
            'model': label,
            'fp32_macro_f1': fp32_metrics['macro_f1'],
            'dynamic_int8_macro_f1': dynamic_metrics['macro_f1'],
            'qat_int8_macro_f1': qat['metrics']['macro_f1'],
            'dynamic_int8_f1_delta': dynamic_metrics['macro_f1'] - fp32_metrics['macro_f1'],
            'qat_int8_f1_delta': qat['metrics']['macro_f1'] - fp32_metrics['macro_f1'],
            'serialized_fp32_kb': model_size_on_disk_kb(fp32_model),
            'serialized_dynamic_int8_kb': model_size_on_disk_kb(dynamic_model),
            'serialized_qat_int8_kb': qat['serialized_qat_int8_kb'],
            'quant_backend': qat['backend'],
        }
        rows.append(row)
        details[label] = {'summary': row, 'qat_metrics': qat['metrics']}
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'qat_summary.csv'), index=False)
    addon_write_json(os.path.join(out_dir, 'qat_results.json'), details)
    return {'rows': rows, 'details': details}


# ============================================================================
# CELL 21: Deployment proof add-on
# ============================================================================
def latency_distribution_addon(model: nn.Module, X_np: np.ndarray,
                               batch_size: int = 1, n_runs: int = 1000,
                               warmup: int = 50,
                               expected_input_dim: int = None) -> dict:
    if expected_input_dim is not None and X_np.shape[1] != expected_input_dim:
        raise RuntimeError(
            f"Latency feature dimension mismatch: model expects {expected_input_dim}, got {X_np.shape[1]}"
        )
    model_cpu = copy.deepcopy(model).cpu().eval()
    X_cpu = torch.tensor(X_np[:batch_size], dtype=torch.float32)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model_cpu(X_cpu)
        samples = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model_cpu(X_cpu)
            samples.append((time.perf_counter() - start) * 1000.0)
    arr = np.asarray(samples, dtype=float)
    return {
        'latency_batch_size': batch_size,
        'latency_runs': n_runs,
        'latency_mean_ms': float(np.mean(arr)),
        'latency_p50_ms': float(np.percentile(arr, 50)),
        'latency_p95_ms': float(np.percentile(arr, 95)),
        'latency_p99_ms': float(np.percentile(arr, 99)),
        'latency_std_ms': float(np.std(arr)),
        'throughput_samples_per_sec': float(batch_size * 1000.0 / max(np.mean(arr), 1e-12)),
    }


def deployment_environment_addon() -> dict:
    import platform
    import sklearn
    env = {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'processor': platform.processor(),
        'torch': torch.__version__,
        'sklearn': sklearn.__version__,
        'device_selected_by_v23': str(device),
        'quantized_engines': list(torch.backends.quantized.supported_engines),
    }
    try:
        import psutil
        env['cpu_physical_cores'] = psutil.cpu_count(logical=False)
        env['cpu_logical_cores'] = psutil.cpu_count(logical=True)
        env['ram_total_gb'] = round(psutil.virtual_memory().total / (1024 ** 3), 3)
    except Exception as exc:
        env['psutil_unavailable'] = str(exc)
    return env


def activation_ram_bytes_estimate_addon(hidden_dims: tuple, num_classes: int, dtype_bytes: int = 4) -> int:
    return int((sum(hidden_dims) + num_classes) * dtype_bytes)


DEPLOYMENT_PROOF_KEY_COLUMNS = [
    'model', 'config_key', 'params', 'flops_per_sample',
    'serialized_fp32_kb', 'serialized_dynamic_int8_kb',
    'fp32_macro_f1', 'dynamic_int8_macro_f1', 'dynamic_int8_f1_delta',
    'fp32_latency_p50_ms', 'fp32_latency_p95_ms', 'fp32_latency_p99_ms',
    'dynamic_int8_latency_p50_ms', 'dynamic_int8_latency_p95_ms', 'dynamic_int8_latency_p99_ms',
    'fp32_batch_latency_p95_ms', 'dynamic_int8_batch_latency_p95_ms',
    'fp32_throughput_samples_per_sec', 'dynamic_int8_throughput_samples_per_sec',
    'activation_ram_bytes_estimate', 'activation_ram_bytes_int8_estimate',
]


def run_deployment_proof_addon() -> dict:
    out_dir = ensure_addon_dir(os.path.join(ADDON_OUTPUT_DIR, 'deployment_proof'))
    candidates = [
        ('student_A_D_scratch', 'D_Small_MLP', STUDENT_A_HIDDEN),
        ('student_A_E_KD_from_RF', 'E_KD_from_RF', STUDENT_A_HIDDEN),
        ('student_A_F_KD_from_CL', 'F_KD_from_CL_MLP', STUDENT_A_HIDDEN),
    ]
    rows = []
    for label, model_key, hidden in candidates:
        if model_key not in final_models:
            raise RuntimeError(f'[Deployment Proof] required model missing: {model_key}')
        model = final_models[model_key].cpu().eval()
        input_dim, num_classes = validate_feature_matrix_for_model_addon(
            model, X_test_np, "Deployment proof"
        )
        dynamic_model = quantize_dynamic_int8(model)
        fp32_macro_f1 = final_results.get(model_key, {}).get('macro_f1')
        fp32_accuracy = final_results.get(model_key, {}).get('accuracy')
        dynamic_metrics = evaluate_model_cpu_addon(dynamic_model, X_test_np, y_test_np)
        row = {
            'model': label,
            'config_key': model_key,
            'params': count_params(model),
            'flops_per_sample': compute_flops_mlp(input_dim, hidden, num_classes),
            'weights_fp32_kb_theoretical': model_size_kb(model, 4),
            'weights_int8_kb_theoretical': model_size_kb(model, 1),
            'serialized_fp32_kb': model_size_on_disk_kb(model),
            'serialized_dynamic_int8_kb': model_size_on_disk_kb(dynamic_model),
            'activation_ram_bytes_estimate': activation_ram_bytes_estimate_addon(hidden, num_classes, 4),
            'activation_ram_bytes_int8_estimate': activation_ram_bytes_estimate_addon(hidden, num_classes, 1),
            'fp32_macro_f1': fp32_macro_f1,
            'fp32_accuracy': fp32_accuracy,
            'dynamic_int8_macro_f1': dynamic_metrics['macro_f1'],
            'dynamic_int8_accuracy': dynamic_metrics['accuracy'],
            'dynamic_int8_f1_delta': (dynamic_metrics['macro_f1'] - fp32_macro_f1) if fp32_macro_f1 is not None else None,
            'dynamic_int8_accuracy_delta': (dynamic_metrics['accuracy'] - fp32_accuracy) if fp32_accuracy is not None else None,
        }
        row.update({f'fp32_{k}': v for k, v in latency_distribution_addon(model, X_test_np, batch_size=1, n_runs=1000, expected_input_dim=input_dim).items()})
        row.update({f'dynamic_int8_{k}': v for k, v in latency_distribution_addon(dynamic_model, X_test_np, batch_size=1, n_runs=1000, expected_input_dim=input_dim).items()})
        batch64 = latency_distribution_addon(model, X_test_np, batch_size=min(64, len(X_test_np)), n_runs=300, expected_input_dim=input_dim)
        dynamic_batch64 = latency_distribution_addon(dynamic_model, X_test_np, batch_size=min(64, len(X_test_np)), n_runs=300, expected_input_dim=input_dim)
        row.update({f'fp32_batch_{k}': v for k, v in batch64.items()})
        row.update({f'dynamic_int8_batch_{k}': v for k, v in dynamic_batch64.items()})
        rows.append(row)
    summary = pd.DataFrame(rows)
    if not summary.empty:
        key_cols = [c for c in DEPLOYMENT_PROOF_KEY_COLUMNS if c in summary.columns]
        summary = summary[key_cols + [c for c in summary.columns if c not in key_cols]]
    summary.to_csv(os.path.join(out_dir, 'deployment_proof_summary.csv'), index=False)
    payload = {'environment': deployment_environment_addon(), 'rows': rows}
    addon_write_json(os.path.join(out_dir, 'deployment_proof_results.json'), payload)
    return payload


# ============================================================================
# CELL 22: RF-KD SHAP alignment add-on
# ============================================================================
def rank_alignment_addon(a: np.ndarray, b: np.ndarray) -> dict:
    from scipy.stats import spearmanr, kendalltau
    a_rank = pd.Series(a).rank(ascending=False)
    b_rank = pd.Series(b).rank(ascending=False)
    rho, rho_p = spearmanr(a_rank, b_rank)
    tau, tau_p = kendalltau(a_rank, b_rank)
    return {
        'spearman_rho': float(rho) if rho == rho else None,
        'spearman_p': float(rho_p) if rho_p == rho_p else None,
        'kendall_tau': float(tau) if tau == tau else None,
        'kendall_p': float(tau_p) if tau_p == tau_p else None,
    }


def run_rf_kd_shap_alignment_addon(max_background: int = 100, max_explain: int = 500) -> dict:
    out_dir = ensure_addon_dir(os.path.join(ADDON_OUTPUT_DIR, 'rf_kd_shap_alignment'))
    try:
        import shap
    except Exception as exc:
        raise RuntimeError('Install shap before running this add-on') from exc
    if 'E_KD_from_RF' not in final_models:
        raise RuntimeError('E_KD_from_RF missing from final_models; finish v2.3 run first')
    rf_teacher = final_models.get('A_RF_500')
    if rf_teacher is None:
        rf_teacher = RandomForestClassifier(n_estimators=500, max_depth=15, random_state=SEEDS[-1], n_jobs=-1)
        rf_teacher.fit(X_train_np, y_train_np)

    rng = np.random.RandomState(42)
    bg_idx = rng.choice(len(X_train_np), min(max_background, len(X_train_np)), replace=False)
    ex_idx = rng.choice(len(X_test_np), min(max_explain, len(X_test_np)), replace=False)
    student = copy.deepcopy(final_models['E_KD_from_RF']).to(device).eval()
    background = torch.tensor(X_train_np[bg_idx], dtype=torch.float32).to(device)
    explain_t = torch.tensor(X_test_np[ex_idx], dtype=torch.float32).to(device)

    student_explainer = shap.DeepExplainer(student, background)
    student_values = student_explainer.shap_values(explain_t)
    if isinstance(student_values, np.ndarray) and student_values.ndim == 3:
        student_list = [student_values[:, :, i] for i in range(NUM_CLASSES)]
    else:
        student_list = student_values
    student_global = np.abs(np.stack(student_list)).mean(axis=(0, 1))

    rf_explainer = shap.TreeExplainer(rf_teacher)
    rf_values = rf_explainer.shap_values(X_test_np[ex_idx])
    if isinstance(rf_values, np.ndarray) and rf_values.ndim == 3:
        rf_list = [rf_values[:, :, i] for i in range(NUM_CLASSES)]
    else:
        rf_list = rf_values
    rf_global = np.abs(np.stack(rf_list)).mean(axis=(0, 1))

    top_table = pd.DataFrame({
        'feature': FEATURE_NAMES,
        'student_abs_shap': student_global,
        'rf_abs_shap': rf_global,
    })
    top_table['student_rank'] = top_table['student_abs_shap'].rank(ascending=False)
    top_table['rf_rank'] = top_table['rf_abs_shap'].rank(ascending=False)
    top_table = top_table.sort_values('rf_abs_shap', ascending=False)
    top_table.to_csv(os.path.join(out_dir, 'rf_kd_shap_feature_alignment.csv'), index=False)
    payload = {
        'global_rank_alignment': rank_alignment_addon(student_global, rf_global),
        'top_rf_features': top_table.head(20).to_dict('records'),
        'top_student_features': top_table.sort_values('student_abs_shap', ascending=False).head(20).to_dict('records'),
        'n_background': int(len(bg_idx)),
        'n_explain': int(len(ex_idx)),
    }
    addon_write_json(os.path.join(out_dir, 'rf_kd_shap_alignment_results.json'), payload)
    return payload


# ============================================================================
# CELL 23: Execute selected add-ons
# ============================================================================
if RUN_EDGEIIOT_GENERALIZATION_ADDON:
    edgeiiot_generalization_addon_results = run_edgeiiot_generalization_addon()

if RUN_QAT_ADDON:
    qat_addon_results = run_qat_addon_for_final_models()

if RUN_DEPLOYMENT_PROOF_ADDON:
    deployment_proof_addon_results = run_deployment_proof_addon()

if RUN_RF_KD_SHAP_ALIGNMENT_ADDON:
    rf_kd_shap_alignment_addon_results = run_rf_kd_shap_alignment_addon()

