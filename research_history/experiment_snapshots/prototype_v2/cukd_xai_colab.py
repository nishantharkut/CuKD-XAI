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

# ----------------------------------------------------------------------------
# EXPERIMENT CONFIGURATION
# ----------------------------------------------------------------------------
SEEDS = [42, 123, 456, 789, 1001]        # 5 seeds for statistical validation
N_SEEDS = len(SEEDS)                      # Set to 1 for quick debugging runs
QUICK_MODE = False                        # True = single seed, no grid search
RUN_CICIOT = False                        # True = also run CICIoT2023 (slow)
CICIOT_PATH = 'CICIoT2023.csv'            # Path if using CICIoT2023
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
CL_STAGES = [(0.33, 7), (0.66, 7), (1.0, 11)]

# Student architectures to test
STUDENT_A_HIDDEN = (32, 16)   # Ultra-compact: 1,189 params
STUDENT_B_HIDDEN = (64, 32)   # Balanced: 3,397 params

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"PyTorch: {torch.__version__}")
print(f"Seeds: {SEEDS}")
print(f"Quick mode: {QUICK_MODE}")

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
    """Return dict of evaluation metrics."""
    preds = _batched_predict(model, X)
    y_np = y.cpu().numpy() if torch.is_tensor(y) else np.asarray(y)

    acc = accuracy_score(y_np, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_np, preds, average='macro', zero_division=0
    )
    per_class_f1 = f1_score(y_np, preds, average=None, zero_division=0)
    cm = confusion_matrix(y_np, preds)
    return {
        'accuracy': float(acc),
        'macro_precision': float(prec),
        'macro_recall': float(rec),
        'macro_f1': float(f1),
        'per_class_f1': per_class_f1.tolist(),
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
                          return_loss_curve: bool = False,
                          verbose: bool = False):
    """Curriculum learning with discrete stage pacing.

    stages: list of (fraction, epochs). Samples with difficulty_order[:n]
    are used in each stage (easy-first if difficulty_order is loss-ascending).
    """
    model = model.to(device)
    total_epochs = sum(s[1] for s in stages)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

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
    n_total = len(X)

    for stage_idx, (frac, n_epochs) in enumerate(stages):
        n_use = int(n_total * frac)
        idx = torch.tensor(np.asarray(difficulty_order[:n_use]),
                           dtype=torch.long, device=device)
        stage_ds = TensorDataset(X_d[idx], y_d[idx])
        stage_loader = DataLoader(stage_ds, batch_size=batch_size, shuffle=True)

        if verbose:
            print(f"  Stage {stage_idx+1}: {n_use}/{n_total} samples, {n_epochs} epochs")

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

    # ----- Config C: CL-trained MLP teacher (loss-based difficulty) -----
    if verbose: print("[C] CL-trained MLP (loss-based)...")
    t0 = time.perf_counter()
    teacher_c = TeacherMLP(INPUT_DIM, NUM_CLASSES)
    teacher_c, c_curve = train_with_curriculum(
        teacher_c, X_train_t, y_train_t, loss_order, X_val_t, y_val_t,
        stages=CL_STAGES, class_weights=class_weights,
        return_loss_curve=True
    )
    c_time = time.perf_counter() - t0
    m_c = evaluate_model(teacher_c, X_test_t, y_test_t)
    m_c['ece'] = expected_calibration_error(
        _batched_probs(teacher_c, X_test_t), y_test
    )
    m_c['params'] = count_params(teacher_c)
    m_c['model_size_kb'] = model_size_kb(teacher_c)
    m_c['train_time_sec'] = c_time
    m_c['loss_curve'] = c_curve
    results['C_CL_MLP_loss'] = m_c
    models['C_CL_MLP_loss'] = teacher_c

    # ----- Config C2: CL teacher with domain-based difficulty -----
    if verbose: print("[C2] CL-trained MLP (domain-based)...")
    teacher_c2 = TeacherMLP(INPUT_DIM, NUM_CLASSES)
    teacher_c2 = train_with_curriculum(
        teacher_c2, X_train_t, y_train_t, domain_order, X_val_t, y_val_t,
        stages=CL_STAGES, class_weights=class_weights,
    )
    m_c2 = evaluate_model(teacher_c2, X_test_t, y_test_t)
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
    results['E2_KD_from_MLP'] = m_e2
    models['E2_KD_from_MLP'] = student_e2

    # ----- Config F: KD from CL-trained MLP (CORE CLAIM) -----
    if verbose: print("[F] KD from CL-MLP (CORE CLAIM)...")
    student_f = StudentMLP(INPUT_DIM, student_hidden, NUM_CLASSES)
    student_f = train_kd(
        student_f, teacher_c, X_train_t, y_train_t, X_val_t, y_val_t,
        T=kd_T, alpha=kd_alpha, class_weights=class_weights,
        epochs=TRAIN_CONFIG['epochs'], batch_size=TRAIN_CONFIG['batch_size']
    )
    m_f = evaluate_model(student_f, X_test_t, y_test_t)
    m_f['params'] = count_params(student_f)
    m_f['model_size_kb'] = model_size_kb(student_f)
    results['F_KD_from_CL_MLP'] = m_f
    models['F_KD_from_CL_MLP'] = student_f

    # ----- Config G: KD from random-pacing MLP (control) -----
    if verbose: print("[G] KD from random-pacing MLP (control)...")
    random_order = np.random.RandomState(seed).permutation(len(X_train))
    teacher_g = TeacherMLP(INPUT_DIM, NUM_CLASSES)
    teacher_g = train_with_curriculum(
        teacher_g, X_train_t, y_train_t, random_order, X_val_t, y_val_t,
        stages=CL_STAGES, class_weights=class_weights,
    )
    student_g = StudentMLP(INPUT_DIM, student_hidden, NUM_CLASSES)
    student_g = train_kd(
        student_g, teacher_g, X_train_t, y_train_t, X_val_t, y_val_t,
        T=kd_T, alpha=kd_alpha, class_weights=class_weights,
        epochs=TRAIN_CONFIG['epochs'], batch_size=TRAIN_CONFIG['batch_size']
    )
    m_g = evaluate_model(student_g, X_test_t, y_test_t)
    m_g['params'] = count_params(student_g)
    m_g['model_size_kb'] = model_size_kb(student_g)
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
    ('C_CL_MLP_loss', 'B_Full_MLP',     "Does CL help teacher? (C vs B)"),
    ('F_KD_from_CL_MLP', 'E2_KD_from_MLP', "Does CL cascade through KD? (F vs E2, CORE)"),
    ('F_KD_from_CL_MLP', 'D_Small_MLP', "Does KD beat scratch? (F vs D)"),
    ('F_KD_from_CL_MLP', 'G_KD_random_pacing', "Order vs random pacing? (F vs G)"),
    ('F_KD_from_CL_MLP', 'I_KD_from_SMOTE_MLP', "CL vs SMOTE teacher? (F vs I)"),
    ('E2_KD_from_MLP', 'D_Small_MLP',     "Does KD work at all? (E2 vs D)"),
    ('E_KD_from_RF', 'E2_KD_from_MLP',    "RF teacher vs MLP teacher? (E vs E2)"),
]
for a, b, desc in key_comparisons:
    if a not in agg_A['Config'].values or b not in agg_A['Config'].values:
        print(f"{desc}: one or both configs missing, skipping")
        continue
    w = wilcoxon_test(all_seed_results_A, a, b)
    print(f"{desc}")
    print(f"  {a}: {agg_A[agg_A['Config']==a]['MacroF1_mean'].iloc[0]:.4f}")
    print(f"  {b}: {agg_A[agg_A['Config']==b]['MacroF1_mean'].iloc[0]:.4f}")
    print(f"  diff: {w['diff_mean']:+.4f}  |  p={w['p']}  |  {w['verdict']}\n")

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

    shap_results = {
        'student_global_importance': student_imp_df.to_dict('records'),
        'teacher_global_importance': teacher_imp_df.to_dict('records'),
        'student_per_class_top3': student_per_class,
        'ranking_agreement_spearman': float(rho),
        'ranking_agreement_p': float(rho_p),
    }
except ImportError:
    print("shap not installed. Install with: !pip install shap")
except Exception as e:
    print(f"SHAP analysis failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# CELL 13: Actual INT8 quantization experiment
# ============================================================================
print("\n" + "=" * 60)
print("INT8 QUANTIZATION EXPERIMENT (Student F)")
print("=" * 60)

quant_results = {}
try:
    if 'F_KD_from_CL_MLP' not in final_models:
        raise RuntimeError("Config F model missing from final_models")
    student_to_quantize = final_models['F_KD_from_CL_MLP']
    student_fp32_size = model_size_on_disk_kb(student_to_quantize)
    print(f"fp32 student size on disk: {student_fp32_size:.2f} KB")

    student_int8 = quantize_dynamic_int8(student_to_quantize)
    student_int8_size = model_size_on_disk_kb(student_int8)
    print(f"int8 student size on disk: {student_int8_size:.2f} KB")

    # Evaluate int8 student on CPU
    X_test_cpu = torch.tensor(X_test_np, dtype=torch.float32)
    student_int8.eval()
    with torch.no_grad():
        preds_int8 = []
        for i in range(0, len(X_test_cpu), 4096):
            logits = student_int8(X_test_cpu[i:i + 4096])
            preds_int8.append(logits.argmax(dim=1).numpy())
    preds_int8 = np.concatenate(preds_int8)
    acc_int8 = accuracy_score(y_test_np, preds_int8)
    f1_int8 = f1_score(y_test_np, preds_int8, average='macro', zero_division=0)
    print(f"int8 student accuracy: {acc_int8:.4f}, macro F1: {f1_int8:.4f}")

    acc_fp32 = final_results['F_KD_from_CL_MLP']['accuracy']
    f1_fp32 = final_results['F_KD_from_CL_MLP']['macro_f1']
    print(f"fp32 → int8 accuracy delta: {(acc_int8 - acc_fp32) * 100:+.3f}%")
    print(f"fp32 → int8 macro F1 delta: {(f1_int8 - f1_fp32) * 100:+.3f}%")

    quant_results = {
        'fp32_size_kb': student_fp32_size,
        'int8_size_kb': student_int8_size,
        'size_reduction_pct': (1 - student_int8_size / student_fp32_size) * 100,
        'fp32_accuracy': float(acc_fp32),
        'int8_accuracy': float(acc_int8),
        'fp32_macro_f1': float(f1_fp32),
        'int8_macro_f1': float(f1_int8),
    }
except Exception as e:
    print(f"INT8 quantization / evaluation failed: {e}")
    import traceback
    traceback.print_exc()
    quant_results = {}

# ============================================================================
# CELL 14: Inference time & throughput benchmarks
# ============================================================================
print("\n" + "=" * 60)
print("INFERENCE TIME BENCHMARKS")
print("=" * 60)

X_bench = torch.tensor(X_test_np[:1024], dtype=torch.float32)
bench_results = {}

candidate_models = [
    ('Teacher_MLP', 'B_Full_MLP'),
    ('Student_scratch', 'D_Small_MLP'),
    ('Student_KD_F', 'F_KD_from_CL_MLP'),
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
configs_to_plot = ['B_Full_MLP', 'C_CL_MLP_loss', 'D_Small_MLP',
                   'E_KD_from_RF', 'E2_KD_from_MLP', 'F_KD_from_CL_MLP']
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

# --- Confusion matrix heatmap for Config F ---
f_cm = np.array(final_results['F_KD_from_CL_MLP']['confusion_matrix'])
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(f_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix — Config F (CL+KD Student)')
plt.tight_layout()
plt.savefig('confusion_matrix_F.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved confusion_matrix_F.png")

# --- Pareto frontier: size vs accuracy ---
fig, ax = plt.subplots(figsize=(9, 6))
for _, row in agg_A.iterrows():
    cfg = row['Config']
    size = row['size_kb'] if row['size_kb'] else 1.0
    acc = row['MacroF1_mean']
    acc_err = row['MacroF1_std']
    ax.errorbar(size, acc, yerr=acc_err, fmt='o', markersize=8, capsize=3)
    ax.annotate(cfg.replace('_', ' '), (size, acc),
                xytext=(6, 4), textcoords='offset points', fontsize=8)
ax.set_xscale('log')
ax.set_xlabel('Model size (KB, fp32) — log scale')
ax.set_ylabel('Macro F1 (test)')
ax.set_title('Model size vs. Macro F1 (Pareto frontier)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pareto_frontier.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved pareto_frontier.png")

# --- Training loss curves: Config B vs C (non-CL vs CL) ---
if 'loss_curve' in final_results.get('B_Full_MLP', {}) and \
   'loss_curve' in final_results.get('C_CL_MLP_loss', {}):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    b_curve = final_results['B_Full_MLP']['loss_curve']
    c_curve = final_results['C_CL_MLP_loss']['loss_curve']

    axes[0].plot(b_curve['loss'], label='B (no CL)', color='C0')
    axes[0].plot(c_curve['loss'], label='C (CL)', color='C1')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training loss')
    axes[0].set_title('Training loss — B vs C')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(b_curve['val_f1'], label='B (no CL)', color='C0')
    axes[1].plot(c_curve['val_f1'], label='C (CL)', color='C1')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation macro F1')
    axes[1].set_title('Validation F1 — B vs C')
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
    'ciciot_results': ciciot_results,
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
