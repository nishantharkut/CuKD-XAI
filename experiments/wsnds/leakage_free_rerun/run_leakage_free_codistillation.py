# ============================================================================
#  !!!! WARNING - READ BEFORE EDITING !!!!
#
#  This .py file is the Python source for the matching notebook. Make source
#  edits here, then regenerate the notebook with make_notebook_preserve_banner.py.
#
#  J-only merge route:
#      python3 make_notebook_preserve_banner.py experiments/wsnds/codistillation/cukd_xai_wsnds_j_only_merge.py experiments/wsnds/codistillation/cukd_xai_wsnds_j_only_merge.ipynb
#
#  This route reads the completed v2.3 10-seed JSON and runs only Config J.
#
#  Bug fixes applied April 11, 2026 are marked `# FIXED 2026-04-11:` inline.
# ============================================================================

# ============================================================================
# CuKD-XAI WSN-DS J-only Co-distillation Merge Route
# for Lightweight WSN Intrusion Detection
#
# FOCUSED ADD-ON IMPLEMENTATION
# - Loads existing v2.3 10-seed results
# - Runs only Config J over the same seeds
# - Reuses the exact v2.3 split, seeds, model classes, and training helpers
# - Trains only the RF soft-label teacher, nominal-schedule CL teacher, and J students
# - Recomputes aggregate tables, Wilcoxon comparisons, and J-specific figures
# - Writes all merged outputs to a new folder without modifying the original run
# - Co-distillation add-on config J (RF + nominal-schedule CL teacher)
#
# Author: Nishant Harkut (2023IMG-040), ABV-IIITM Gwalior
# ============================================================================

# ============================================================================
# CELL 1: Validate dependencies
# ============================================================================
import importlib.util

_REQUIRED_PACKAGES = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'torch': 'torch',
    'sklearn': 'scikit-learn',
    'scipy': 'scipy',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
}
_missing_packages = [pkg for module, pkg in _REQUIRED_PACKAGES.items()
                     if importlib.util.find_spec(module) is None]
if _missing_packages:
    raise RuntimeError(
        'Missing required packages. Install the pinned rerun requirements '
        f'before execution: {_missing_packages}'
    )
else:
    print('All required packages already installed.')

# ============================================================================
# CELL 2: Imports and global config
# ============================================================================
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              f1_score, classification_report, confusion_matrix)
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import hashlib
import time
import json
import os
from pathlib import Path

# ----------------------------------------------------------------------------
# EXPERIMENT CONFIGURATION
# ----------------------------------------------------------------------------
SEEDS_V2_FINAL5 = [42, 123, 456, 789, 1001]
SEEDS_PUBLICATION10 = [42, 123, 456, 789, 1001, 2024, 3141, 5678, 8192, 9999]
FINAL_RUN_MODE = 'wsnds_final'              # final route: original v2.3 with 10 publication seeds
SEEDS = SEEDS_PUBLICATION10 if FINAL_RUN_MODE == 'wsnds_final' else SEEDS_V2_FINAL5
N_SEEDS = len(SEEDS)
QUICK_MODE = os.environ.get('CUKD_QUICK_MODE', '0') == '1'
GRID_TUNING_SEED = 42
BASE_PROTOCOL_ID = 'archive_random_split_train_scaler_controlled_tuning_v2'
J_PROTOCOL_ID = 'archive_random_split_train_scaler_codistillation_v2'
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
WSNDS_PATH = str(Path(os.environ.get(
    'WSNDS_PATH',
    str(REPO_ROOT / 'data' / 'wsnds' / 'WSN-DS.csv'),
)).resolve())

# KD hyperparameter grid (from Benaddi et al. 2025)
KD_T_GRID = [2, 3, 4, 5]
KD_ALPHA_GRID = [0.5, 0.7, 0.9]
KD_T_DEFAULT = 4
KD_ALPHA_DEFAULT = 0.7
RF_CALIBRATION_CONFIG = {
    'n_estimators': 500,
    'max_depth': 15,
    'calibration_method': 'isotonic',
    'calibration_cv': 3,
}

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
# Two historical variants are tested in parallel. Their nominal epoch totals do
# not establish compute matching because early stages use dataset subsets and
# global early stopping may terminate before the final stage.
#
#   FAIR   — historical name for 3+3+24 staged nominal epochs.
#            It is not matched to 30 full-dataset epochs.
#   EXT    — 5+5+30 = 40 total epochs, gives CL extra training time.
#            Generous comparison: "does CL help when we give it a larger budget?"
#
# Original (v2.0, broken) was [(0.33,7),(0.66,7),(1.0,11)] = 25 total with
# only 11 epochs on the full distribution, which badly under-trained Stage 3.
CL_STAGES_FAIR = [(0.33, 3), (0.66, 3), (1.0, 24)]       # historical label only
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

# Scaling is intentionally deferred until after the raw seed-42 split.
scaler = None

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


def _iter_shuffled_batches(*tensors: torch.Tensor, batch_size: int):
    """Yield DataLoader-compatible shuffled batches without Python collation."""
    if not tensors:
        return
    size = len(tensors[0])
    if any(len(tensor) != size for tensor in tensors):
        raise ValueError('All batch tensors must have the same first dimension.')
    torch.empty((), dtype=torch.int64).random_()
    sampler_seed = int(torch.empty((), dtype=torch.int64).random_().item())
    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(sampler_seed)
    order = torch.randperm(size, generator=sampler_generator).to(tensors[0].device)
    for start in range(0, size, batch_size):
        index = order[start:start + batch_size]
        yield tuple(tensor[index] for tensor in tensors)


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

    X_train_d = X_train.to(device)
    y_train_d = y_train.to(device)
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
        for xb, yb in _iter_shuffled_batches(X_train_d, y_train_d, batch_size=batch_size):
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
    # Early stopping uses the same nominal patience as train_standard. The
    # staged subsets and global stopping rule still prevent a compute-matched
    # interpretation against full-dataset training.
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
        X_stage = X_d[idx]
        y_stage = y_d[idx]

        # Per-stage fresh optimizer + cosine schedule scoped to this stage's epochs
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        if verbose:
            print(f"  Stage {stage_idx+1}: {n_use}/{n_total} samples, {n_epochs} epochs, fresh optimizer+scheduler")

        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0.0
            nb = 0
            for xb, yb in _iter_shuffled_batches(X_stage, y_stage, batch_size=batch_size):
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

    Loss = alpha * T^2 * KL(teacher_T || student_T)
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

    best_val = 0.0
    best_state = None
    bad = 0

    for epoch in range(epochs):
        student.train()
        for xb, yb, sb in _iter_shuffled_batches(
            X_d, y_d, soft_targets, batch_size=batch_size
        ):
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
    import io
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return len(buffer.getbuffer()) / 1024


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
    # Train probe
    ce = nn.CrossEntropyLoss()
    probe.train()
    for _ in range(3):
        for xb, yb in _iter_shuffled_batches(X_d, y_d, batch_size=512):
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
            set_seed(GRID_TUNING_SEED)
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
# CELL 8B: Co-distillation helper for Config J
# ============================================================================
CODISTILL_CE_WEIGHT = 0.30
CODISTILL_RF_WEIGHT = 0.40
CODISTILL_CL_WEIGHT = 0.30
CODISTILL_EPOCHS = 40
CODISTILL_LR = 7e-4
CODISTILL_PATIENCE = 10


def soften_probability_targets(raw_probs: torch.Tensor, T: float) -> torch.Tensor:
    pseudo_logits = torch.log(raw_probs.clamp(min=1e-8))
    return F.softmax(pseudo_logits / T, dim=1).detach()


def mlp_teacher_soft_targets(teacher: nn.Module, X_d: torch.Tensor, T: float,
                             batch_size: int = 4096) -> torch.Tensor:
    teacher.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(X_d), batch_size):
            logits = teacher(X_d[i:i + batch_size])
            chunks.append(F.softmax(logits / T, dim=1).detach())
    return torch.cat(chunks, dim=0)


def train_codistill_rf_cl(student: nn.Module,
                          rf_probs: torch.Tensor,
                          cl_teacher: nn.Module,
                          X_train: torch.Tensor, y_train: torch.Tensor,
                          X_val: torch.Tensor, y_val: torch.Tensor,
                          T: float = KD_T_DEFAULT,
                          ce_weight: float = CODISTILL_CE_WEIGHT,
                          rf_weight: float = CODISTILL_RF_WEIGHT,
                          cl_weight: float = CODISTILL_CL_WEIGHT,
                          class_weights: torch.Tensor = None,
                          epochs: int = 30, batch_size: int = 256,
                          lr: float = 1e-3, weight_decay: float = 1e-3,
                          patience: int = 8, verbose: bool = False):
    """Train one student with hard labels plus RF and CL-teacher soft targets.

    The T*T factor follows the original v2.3 train_kd implementation and the
    standard KD gradient correction. The normalized weights control the
    hard-label versus teacher-target balance.
    """
    total_weight = ce_weight + rf_weight + cl_weight
    if total_weight <= 0:
        raise ValueError('Co-distillation weights must have positive sum')
    ce_weight = ce_weight / total_weight
    rf_weight = rf_weight / total_weight
    cl_weight = cl_weight / total_weight

    student = student.to(device)
    cl_teacher = cl_teacher.to(device).eval()
    X_d = X_train.to(device)
    y_d = y_train.to(device)
    X_val_d = X_val.to(device)
    y_val_np = y_val.cpu().numpy()

    if class_weights is not None:
        class_weights = class_weights.to(device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    rf_soft_targets = soften_probability_targets(rf_probs.to(device).float(), T)
    cl_soft_targets = mlp_teacher_soft_targets(cl_teacher, X_d, T)

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = 0.0
    best_state = None
    bad = 0
    loss_curve = []
    val_curve = []

    for epoch in range(epochs):
        student.train()
        epoch_loss = 0.0
        nb = 0
        for xb, yb, rf_sb, cl_sb in _iter_shuffled_batches(
            X_d, y_d, rf_soft_targets, cl_soft_targets, batch_size=batch_size
        ):
            optimizer.zero_grad()
            logits = student(xb)
            log_soft_s = F.log_softmax(logits / T, dim=1)
            ce_term = ce_loss(logits, yb)
            rf_kd_term = F.kl_div(log_soft_s, rf_sb, reduction='batchmean') * (T * T)
            cl_kd_term = F.kl_div(log_soft_s, cl_sb, reduction='batchmean') * (T * T)
            loss = ce_weight * ce_term + rf_weight * rf_kd_term + cl_weight * cl_kd_term
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            nb += 1
        scheduler.step()
        epoch_loss /= max(nb, 1)
        loss_curve.append(epoch_loss)

        preds = _batched_predict(student, X_val_d)
        val_f1 = f1_score(y_val_np, preds, average='macro')
        val_curve.append(float(val_f1))
        if val_f1 > best_val:
            best_val = val_f1
            best_state = copy.deepcopy(student.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose:
            print(f"CoDistill epoch {epoch + 1}: loss={epoch_loss:.4f} val_f1={val_f1:.4f}")

    if best_state is not None:
        student.load_state_dict(best_state)
    return student, {'loss': loss_curve, 'val_f1': val_curve,
                     'weights': {'ce': ce_weight, 'rf': rf_weight, 'cl': cl_weight, 'T': T},
                     'schedule': {'epochs': epochs, 'lr': lr, 'patience': patience}}

# ============================================================================
# CELL 9: Existing 10-seed results resolver and compatibility checks
# ============================================================================
from pathlib import Path
import shutil

EXISTING_RESULTS_PATH = os.environ.get(
    'EXISTING_RESULTS_PATH',
    str(REPO_ROOT / 'results' / 'wsnds' / 'leakage_free_rerun' / 'main_10seed_v2' / 'cukd_xai_results.json'),
)
J_MERGE_OUTPUT_DIR = os.environ.get(
    'J_MERGE_OUTPUT_DIR',
    str(REPO_ROOT / 'results' / 'wsnds' / 'leakage_free_rerun' / 'main_10seed_v2_plus_j'),
)
RF_SOFT_CACHE_DIR = Path(os.environ.get(
    'CUKD_RF_SOFT_CACHE_DIR',
    str(REPO_ROOT / 'results' / 'wsnds' / 'leakage_free_rerun' / 'main_10seed_v2'),
)).resolve()
J_ONLY_QUICK_MODE = QUICK_MODE
ALLOW_EXISTING_J_IN_BASE = False

STUDENT_SPECS_FOR_J = {
    'student_A_32_16': STUDENT_A_HIDDEN,
    'student_B_64_32': STUDENT_B_HIDDEN,
}
RESULT_BLOCK_BY_STUDENT = {
    'student_A_32_16': 'wsn_ds_multi_seed_student_A',
    'student_B_64_32': 'wsn_ds_multi_seed_student_B',
}
REQUIRED_EXISTING_CONFIGS = [
    'A_RF_500',
    'B_Full_MLP',
    'C_CL_MLP_loss_fair',
    'D_Small_MLP',
    'E_KD_from_RF',
    'E2_KD_from_MLP',
    'F_KD_from_CL_MLP_fair',
    'F_KD_from_CL_MLP',
]
J_CONFIG_NAME = 'J_CoDistill_RF_CL'


def json_convert(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return o.to_dict('records')
    if isinstance(o, Path):
        return str(o)
    return str(o)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, default=json_convert)
        handle.write('\n')
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode('ascii'))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _strip_single_parent(path: Path) -> Path:
    parts = path.parts
    if parts and parts[0] == "..":
        return Path(*parts[1:]) if len(parts) > 1 else Path(".")
    return path


def _candidate_paths(path_text: str) -> list:
    configured = Path(os.path.expanduser(str(path_text)))
    candidates = []
    if configured.is_absolute():
        candidates.append(configured)
    else:
        stripped = _strip_single_parent(configured)
        cwd = Path.cwd()
        candidates.append(cwd / configured)
        candidates.append(cwd / stripped)
        try:
            script_dir = Path(__file__).resolve().parent
            repo_root = script_dir.parents[2]
            candidates.append(script_dir / configured)
            candidates.append(script_dir / stripped)
            candidates.append(script_dir.parent / stripped)
            candidates.append(repo_root / configured)
            candidates.append(repo_root / stripped)
        except NameError:
            pass
    return _unique_path_objects(candidates)


def _unique_path_objects(paths: list) -> list:
    unique = []
    seen = set()
    for path in paths:
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def resolve_existing_results_path(path_text: str = EXISTING_RESULTS_PATH) -> Path:
    for candidate in _candidate_paths(path_text):
        if candidate.exists() and candidate.is_file():
            return candidate
    searched = '\n'.join(str(path) for path in _candidate_paths(path_text))
    raise FileNotFoundError(
        'Existing v2.3 10-seed results JSON not found. Set EXISTING_RESULTS_PATH '
        f'or keep the repo folder structure. Searched:\n{searched}'
    )


def resolve_merge_output_dir(path_text: str = J_MERGE_OUTPUT_DIR) -> Path:
    configured = Path(os.path.expanduser(str(path_text)))
    if configured.is_absolute():
        return configured
    try:
        script_dir = Path(__file__).resolve().parent
        repo_root = script_dir.parents[2]
        return (repo_root / configured).resolve()
    except NameError:
        cwd = Path.cwd()
        stripped = _strip_single_parent(configured)
        if configured.parts and configured.parts[0] == ".." and (cwd / "experiments" / "wsnds" / "codistillation").exists():
            return (cwd / stripped).resolve()
        return (cwd / configured).resolve()


def load_existing_results(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def validate_base_artifact_manifest(existing_json_path: Path) -> dict:
    base_dir = existing_json_path.parent.resolve()
    manifest_path = base_dir / 'artifact_manifest.json'
    if not manifest_path.is_file():
        raise RuntimeError('Completed v2 base artifact manifest is missing.')
    manifest = load_existing_results(manifest_path)
    if manifest.get('protocol_id') != BASE_PROTOCOL_ID or manifest.get('status') != 'complete':
        raise RuntimeError('Base artifact manifest protocol or completion status is invalid.')
    entries = manifest.get('files', [])
    if manifest.get('file_count_excluding_manifest') != len(entries):
        raise RuntimeError('Base artifact manifest file count is inconsistent.')
    seen = set()
    for entry in entries:
        relative = Path(entry['path'])
        candidate = (base_dir / relative).resolve()
        if base_dir != candidate and base_dir not in candidate.parents:
            raise RuntimeError(f'Unsafe path in base artifact manifest: {relative}')
        if relative.as_posix() in seen:
            raise RuntimeError(f'Duplicate path in base artifact manifest: {relative}')
        seen.add(relative.as_posix())
        if not candidate.is_file():
            raise RuntimeError(f'Base artifact is missing: {relative}')
        if candidate.stat().st_size != int(entry['size_bytes']):
            raise RuntimeError(f'Base artifact size mismatch: {relative}')
        if _sha256_file(candidate) != entry['sha256']:
            raise RuntimeError(f'Base artifact hash mismatch: {relative}')
    if existing_json_path.name not in seen:
        raise RuntimeError('Base result JSON is absent from its artifact manifest.')
    return manifest


def validate_existing_results_compatibility(existing: dict,
                                            expected_class_names: list,
                                            expected_feature_names: list,
                                            allow_existing_j: bool = ALLOW_EXISTING_J_IN_BASE) -> list:
    if existing.get('run_status') != 'complete':
        raise RuntimeError('Base result is not marked complete; refusing to merge Config J.')
    if existing.get('protocol_id') != BASE_PROTOCOL_ID:
        raise RuntimeError(
            f"Base protocol mismatch: expected {BASE_PROTOCOL_ID}, "
            f"found {existing.get('protocol_id')!r}"
        )
    preprocessing = existing.get('preprocessing_protocol', {})
    if preprocessing.get('protocol_id') != BASE_PROTOCOL_ID:
        raise RuntimeError('Base preprocessing protocol ID is missing or inconsistent.')
    completion = existing.get('completion_gate', {})
    if not completion.get('student_A_complete') or not completion.get('student_B_complete'):
        raise RuntimeError('Base result does not contain complete Student A and Student B runs.')
    grid = existing.get('kd_grid_search', {})
    if grid.get('mode') != 'controlled_single_seed_grid':
        raise RuntimeError('Base KD grid was not run in controlled single-seed mode.')
    if int(grid.get('tuning_seed', -1)) != GRID_TUNING_SEED:
        raise RuntimeError('Base KD grid tuning seed does not match the v2 protocol.')
    for block_name in RESULT_BLOCK_BY_STUDENT.values():
        if block_name not in existing:
            raise RuntimeError(f'Missing existing result block: {block_name}')
    if 'seeds' not in existing:
        raise RuntimeError('Missing seeds in existing result JSON')
    seeds = [int(seed) for seed in existing['seeds']]
    if seeds != SEEDS_PUBLICATION10:
        raise RuntimeError(f'Seed mismatch: expected {SEEDS_PUBLICATION10}, found {seeds}')
    if list(existing.get('class_names', [])) != list(expected_class_names):
        raise RuntimeError('Class names mismatch between existing results and current WSN-DS preprocessing')
    if list(existing.get('feature_names', [])) != list(expected_feature_names):
        raise RuntimeError('Feature names mismatch between existing results and current WSN-DS preprocessing')

    for student_name, block_name in RESULT_BLOCK_BY_STUDENT.items():
        block = existing[block_name]
        block_seeds = [int(seed) for seed in block.keys()]
        if block_seeds != seeds:
            raise RuntimeError(f'Seed mismatch in {block_name}: expected {seeds}, found {block_seeds}')
        for seed in seeds:
            seed_key = str(seed)
            if seed_key not in block:
                raise RuntimeError(f'Missing seed {seed_key} in {block_name}')
            for cfg in REQUIRED_EXISTING_CONFIGS:
                if cfg not in block[seed_key]:
                    raise RuntimeError(f'Required existing config missing: {student_name} seed {seed_key} {cfg}')
            if J_CONFIG_NAME in block[seed_key] and not allow_existing_j:
                raise RuntimeError(
                    f'{J_CONFIG_NAME} already exists in {block_name} seed {seed_key}; '
                    'use the original v2.3 results JSON, not an already-merged file.'
                )
    return seeds


existing_results_path = resolve_existing_results_path()
base_artifact_manifest = validate_base_artifact_manifest(existing_results_path)
existing_results = load_existing_results(existing_results_path)
base_preprocessing = existing_results.get('preprocessing_protocol', {})
if base_preprocessing.get('scaler_fit_partition') != 'train':
    raise RuntimeError(
        'The base result is not marked as a train-only-scaler rerun; refusing to merge J.'
    )
MERGE_SEEDS = validate_existing_results_compatibility(
    existing_results, CLASS_NAMES, FEATURE_NAMES
)
if J_ONLY_QUICK_MODE:
    MERGE_SEEDS = MERGE_SEEDS[:1]
    print('J_ONLY_QUICK_MODE=True: running one seed only. This is not final paper evidence.')

BEST_T = float(existing_results.get('kd_hyperparameters', {}).get('T', 2))
BEST_ALPHA = float(existing_results.get('kd_hyperparameters', {}).get('alpha', 0.5))
merge_output_dir = resolve_merge_output_dir()
merge_output_dir.mkdir(parents=True, exist_ok=True)

print(f'Existing results: {existing_results_path}')
print(f'Output dir: {merge_output_dir}')
print(f'J-only seeds: {MERGE_SEEDS}')
print(f'Using existing KD metadata: T={BEST_T}, alpha={BEST_ALPHA}')

# ============================================================================
# CELL 10: Seed-42 raw split and train-only scaling for J-only seed runner
# ============================================================================
X_trainval_raw, X_test_raw, y_trainval, y_test_np = train_test_split(
    X_all, y_all, test_size=0.15, random_state=42, stratify=y_all
)
X_train_raw, X_val_raw, y_train_np, y_val_np = train_test_split(
    X_trainval_raw, y_trainval, test_size=0.1765, random_state=42, stratify=y_trainval
)
scaler = StandardScaler()
X_train_np = scaler.fit_transform(X_train_raw).astype(np.float32, copy=False)
X_val_np = scaler.transform(X_val_raw).astype(np.float32, copy=False)
X_test_np = scaler.transform(X_test_raw).astype(np.float32, copy=False)

base_mean = np.asarray(base_preprocessing.get('scaler_mean', []), dtype=np.float64)
base_scale = np.asarray(base_preprocessing.get('scaler_scale', []), dtype=np.float64)
if not np.array_equal(scaler.mean_, base_mean) or not np.array_equal(scaler.scale_, base_scale):
    raise RuntimeError('J rerun scaler does not exactly match the train-only-scaler base result.')
dataset_sha256 = _sha256_file(Path(WSNDS_PATH))
split_hashes = {
    'train_raw_features_and_labels_sha256': _sha256_arrays(X_train_raw, y_train_np),
    'validation_raw_features_and_labels_sha256': _sha256_arrays(X_val_raw, y_val_np),
    'test_raw_features_and_labels_sha256': _sha256_arrays(X_test_raw, y_test_np),
}
scaler_sha256 = _sha256_arrays(scaler.mean_, scaler.scale_)
if base_preprocessing.get('dataset_sha256') != dataset_sha256:
    raise RuntimeError('Dataset hash does not match the completed v2 base run.')
if base_preprocessing.get('split_hashes') != split_hashes:
    raise RuntimeError('Raw split hashes do not match the completed v2 base run.')
if base_preprocessing.get('scaler_sha256') != scaler_sha256:
    raise RuntimeError('Scaler hash does not match the completed v2 base run.')
script_sha256 = _sha256_file(Path(__file__).resolve())
print(f"Train: {X_train_np.shape}, Val: {X_val_np.shape}, Test: {X_test_np.shape}")
print('Scaler fit partition: train only (exactly matched to base rerun)')


def _class_weights_tensor(y_np: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y_np, minlength=NUM_CLASSES)
    return torch.tensor(
        len(y_np) / (NUM_CLASSES * np.maximum(counts, 1)),
        dtype=torch.float32,
    )


def _load_validated_rf_cache(seed: int, expected_rows: int) -> np.ndarray:
    cache_path = RF_SOFT_CACHE_DIR / f'rf_soft_seed_{seed}.npy'
    manifest_path = RF_SOFT_CACHE_DIR / f'rf_soft_seed_{seed}.manifest.json'
    if not cache_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(
            f'Completed v2 RF cache and manifest are required for seed {seed}: '
            f'{cache_path.name}, {manifest_path.name}'
        )
    manifest = load_existing_results(manifest_path)
    expected_manifest = {
        'protocol_id': BASE_PROTOCOL_ID,
        'seed': int(seed),
        'dataset_sha256': dataset_sha256,
        'train_split_sha256': split_hashes['train_raw_features_and_labels_sha256'],
        'scaler_sha256': scaler_sha256,
        'class_names': CLASS_NAMES,
        'rf_calibration_config': RF_CALIBRATION_CONFIG,
        'shape': [expected_rows, NUM_CLASSES],
        'dtype': 'float32',
        'cache_file': cache_path.name,
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f'RF cache manifest mismatch for seed {seed}, key {key}: '
                f'expected {expected!r}, found {manifest.get(key)!r}'
            )
    base_sklearn = base_preprocessing.get('environment', {}).get('sklearn')
    if manifest.get('sklearn_version') != base_sklearn:
        raise RuntimeError(f'RF cache sklearn provenance mismatch for seed {seed}.')
    array = np.load(cache_path, allow_pickle=False)
    if array.shape != (expected_rows, NUM_CLASSES) or array.dtype != np.float32:
        raise RuntimeError(
            f'RF cache array contract mismatch for seed {seed}: '
            f'shape={array.shape}, dtype={array.dtype}'
        )
    if not np.isfinite(array).all() or np.any(array < 0.0):
        raise RuntimeError(f'RF cache contains invalid probabilities for seed {seed}.')
    if not np.allclose(array.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6):
        raise RuntimeError(f'RF cache rows do not sum to one for seed {seed}.')
    if _sha256_arrays(array) != manifest.get('probabilities_sha256'):
        raise RuntimeError(f'RF cache content hash mismatch for seed {seed}.')
    return array


def build_j_support_teachers(seed: int,
                             X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray,
                             class_weights: torch.Tensor,
                             verbose: bool = True) -> dict:
    """Train only the support objects Config J needs: RF + nominal-schedule CL teacher."""
    set_seed(seed)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    if verbose:
        print(f"Seed {seed}: calibrated RF soft labels for J")
    rf_soft_array = _load_validated_rf_cache(seed, len(X_train))
    rf_soft = torch.tensor(rf_soft_array, dtype=torch.float32)
    rf_time = float(
        existing_results['wsn_ds_multi_seed_student_A'][str(seed)]
        ['E_KD_from_RF'].get('rf_calibration_time_sec', 0.0)
    )

    if verbose:
        print(f"Seed {seed}: nominal-schedule CL teacher for J")
    cl_start = time.perf_counter()
    loss_order = compute_difficulty_loss_based(
        X_train_t, y_train_t, INPUT_DIM, NUM_CLASSES, seed=seed
    )
    teacher_c_fair = TeacherMLP(INPUT_DIM, NUM_CLASSES)
    teacher_c_fair, c_fair_curve = train_with_curriculum(
        teacher_c_fair, X_train_t, y_train_t, loss_order, X_val_t, y_val_t,
        stages=CL_STAGES_FAIR, class_weights=class_weights,
        return_loss_curve=True,
    )
    cl_time = time.perf_counter() - cl_start

    return {
        'rf_soft': rf_soft,
        'teacher_c_fair': teacher_c_fair,
        'teacher_c_curve': c_fair_curve,
        'rf_calibration_time_sec': float(rf_time),
        'rf_soft_reused_from_main_rerun': True,
        'cl_teacher_time_sec': float(cl_time),
    }


J_MODEL_ARTIFACT_DIR = merge_output_dir / 'model_artifacts'
J_MODEL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def _save_j_model_artifact(seed: int, student_name: str,
                           student_hidden: tuple, model: nn.Module,
                           metrics: dict) -> None:
    artifact_path = J_MODEL_ARTIFACT_DIR / f'{student_name}_seed_{seed}_config_J.pt'
    temporary = artifact_path.with_suffix('.pt.tmp')
    torch.save({
        'protocol_id': J_PROTOCOL_ID,
        'base_protocol_id': BASE_PROTOCOL_ID,
        'config_name': J_CONFIG_NAME,
        'student_name': student_name,
        'seed': int(seed),
        'input_dim': INPUT_DIM,
        'hidden_dims': list(student_hidden),
        'num_classes': NUM_CLASSES,
        'feature_names': FEATURE_NAMES,
        'class_names': CLASS_NAMES,
        'dataset_sha256': dataset_sha256,
        'split_hashes': split_hashes,
        'scaler_sha256': scaler_sha256,
        'kd_hyperparameters': {'T': BEST_T, 'alpha': BEST_ALPHA},
        'codistill_weights': {
            'ce': CODISTILL_CE_WEIGHT,
            'rf': CODISTILL_RF_WEIGHT,
            'cl': CODISTILL_CL_WEIGHT,
        },
        'metrics': {
            'accuracy': metrics['accuracy'],
            'macro_f1': metrics['macro_f1'],
            'per_class_f1': metrics['per_class_f1'],
            'confusion_matrix': metrics['confusion_matrix'],
        },
        'state_dict': {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
    }, temporary)
    temporary.replace(artifact_path)
    write_json(artifact_path.with_suffix('.manifest.json'), {
        'artifact_file': artifact_path.name,
        'artifact_sha256': _sha256_file(artifact_path),
        'protocol_id': J_PROTOCOL_ID,
        'base_protocol_id': BASE_PROTOCOL_ID,
        'config_name': J_CONFIG_NAME,
        'student_name': student_name,
        'seed': int(seed),
        'dataset_sha256': dataset_sha256,
        'scaler_sha256': scaler_sha256,
        'script_sha256': script_sha256,
    })


def run_j_for_student(seed: int,
                      student_name: str,
                      student_hidden: tuple,
                      support: dict,
                      X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      class_weights: torch.Tensor) -> dict:
    """Run only Config J for one student architecture and one seed."""
    set_seed(seed)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    print(f"Seed {seed}: {student_name} {J_CONFIG_NAME}")
    start = time.perf_counter()
    student_j = StudentMLP(INPUT_DIM, student_hidden, NUM_CLASSES)
    student_j, j_curve = train_codistill_rf_cl(
        student_j,
        support['rf_soft'],
        support['teacher_c_fair'],
        X_train_t, y_train_t,
        X_val_t, y_val_t,
        T=BEST_T,
        ce_weight=CODISTILL_CE_WEIGHT,
        rf_weight=CODISTILL_RF_WEIGHT,
        cl_weight=CODISTILL_CL_WEIGHT,
        class_weights=class_weights,
        epochs=CODISTILL_EPOCHS,
        batch_size=TRAIN_CONFIG['batch_size'],
        lr=CODISTILL_LR,
        weight_decay=TRAIN_CONFIG['weight_decay'],
        patience=CODISTILL_PATIENCE,
    )
    student_time = time.perf_counter() - start

    metrics = evaluate_model(student_j, X_test_t, y_test_t)
    metrics['params'] = count_params(student_j)
    metrics['model_size_kb'] = model_size_kb(student_j)
    metrics['model_size_kb_int8'] = model_size_kb(student_j, 1)
    metrics['flops'] = compute_flops_mlp(INPUT_DIM, student_hidden, NUM_CLASSES)
    metrics['ece'] = expected_calibration_error(_batched_probs(student_j, X_test_t), y_test)
    metrics['train_time_sec'] = float(student_time)
    metrics['rf_calibration_time_sec'] = support['rf_calibration_time_sec']
    metrics['rf_soft_reused_from_main_rerun'] = support['rf_soft_reused_from_main_rerun']
    metrics['cl_teacher_time_sec'] = support['cl_teacher_time_sec']
    metrics['total_support_plus_student_time_sec'] = float(
        support['rf_calibration_time_sec'] + support['cl_teacher_time_sec'] + student_time
    )
    metrics['loss_curve'] = j_curve
    _save_j_model_artifact(
        seed, student_name, student_hidden, student_j, metrics
    )
    return metrics


def run_j_only_seed(seed: int) -> dict:
    class_weights = _class_weights_tensor(y_train_np)
    support = build_j_support_teachers(
        seed, X_train_np, y_train_np, X_val_np, y_val_np, class_weights
    )
    seed_result = {}
    for student_name, hidden_dims in STUDENT_SPECS_FOR_J.items():
        seed_result[student_name] = {
            J_CONFIG_NAME: run_j_for_student(
                seed, student_name, hidden_dims, support,
                X_train_np, y_train_np,
                X_val_np, y_val_np,
                X_test_np, y_test_np,
                class_weights,
            )
        }
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return seed_result


def run_j_only_all_seeds(seeds: list) -> dict:
    j_results = {student_name: {} for student_name in STUDENT_SPECS_FOR_J}
    started = time.perf_counter()
    for seed in seeds:
        print("\n" + "=" * 60)
        print(f"J-only seed {seed}")
        print("=" * 60)
        seed_payload = run_j_only_seed(seed)
        for student_name in STUDENT_SPECS_FOR_J:
            j_results[student_name][str(seed)] = seed_payload[student_name][J_CONFIG_NAME]
        write_json(
            merge_output_dir / f'j_only_seed_{seed}_checkpoint.json',
            {'seed': int(seed), 'results': seed_payload},
        )
    j_results['_runtime_sec'] = float(time.perf_counter() - started)
    return j_results


def _validate_j_results(j_results: dict, seeds: list) -> None:
    expected_seed_keys = [str(seed) for seed in seeds]
    for student_name in STUDENT_SPECS_FOR_J:
        actual_seed_keys = list(j_results.get(student_name, {}).keys())
        if actual_seed_keys != expected_seed_keys:
            raise RuntimeError(
                f'{student_name} Config J seed coverage mismatch: '
                f'{actual_seed_keys} != {expected_seed_keys}'
            )
        for seed in seeds:
            metrics = j_results[student_name][str(seed)]
            for metric_name in ('accuracy', 'macro_f1'):
                value = float(metrics[metric_name])
                if not np.isfinite(value):
                    raise RuntimeError(
                        f'{student_name} seed {seed} has non-finite {metric_name}'
                    )
            artifact_path = (
                J_MODEL_ARTIFACT_DIR / f'{student_name}_seed_{seed}_config_J.pt'
            )
            manifest_path = artifact_path.with_suffix('.manifest.json')
            if not artifact_path.is_file() or not manifest_path.is_file():
                raise RuntimeError(
                    f'Missing Config J model artifact for {student_name}, seed {seed}'
                )
            manifest = load_existing_results(manifest_path)
            if manifest.get('artifact_sha256') != _sha256_file(artifact_path):
                raise RuntimeError(
                    f'Config J model artifact hash mismatch for {student_name}, seed {seed}'
                )


j_only_results = run_j_only_all_seeds(MERGE_SEEDS)
_validate_j_results(j_only_results, MERGE_SEEDS)
write_json(merge_output_dir / 'completion_gate.json', {
    'status': 'passed_full' if not J_ONLY_QUICK_MODE else 'passed_quick_sanity_only',
    'protocol_id': J_PROTOCOL_ID,
    'base_protocol_id': BASE_PROTOCOL_ID,
    'seeds': MERGE_SEEDS,
    'students': list(STUDENT_SPECS_FOR_J),
    'config': J_CONFIG_NAME,
    'publication_complete': not J_ONLY_QUICK_MODE,
})

# ============================================================================
# CELL 11: Merge J into existing 10-seed results and recompute statistics
# ============================================================================
def aggregate_multi_seed(seed_results: dict) -> pd.DataFrame:
    configs = set()
    for result_by_config in seed_results.values():
        configs.update(result_by_config.keys())
    configs = sorted(configs)

    rows = []
    for cfg in configs:
        accs = [seed_results[str(seed)][cfg]['accuracy'] for seed in MERGE_SEEDS
                if str(seed) in seed_results and cfg in seed_results[str(seed)]]
        f1s = [seed_results[str(seed)][cfg]['macro_f1'] for seed in MERGE_SEEDS
               if str(seed) in seed_results and cfg in seed_results[str(seed)]]
        per_class = [seed_results[str(seed)][cfg]['per_class_f1'] for seed in MERGE_SEEDS
                     if str(seed) in seed_results and cfg in seed_results[str(seed)]]
        if not accs:
            raise RuntimeError(f'Config has no seed results: {cfg}')
        per_class_arr = np.asarray(per_class, dtype=np.float64)
        row = {
            'Config': cfg,
            'Accuracy_mean': float(np.mean(accs)),
            'Accuracy_std': float(np.std(accs)),
            'MacroF1_mean': float(np.mean(f1s)),
            'MacroF1_std': float(np.std(f1s)),
            'n_seeds': int(len(accs)),
        }
        for idx, class_name in enumerate(CLASS_NAMES):
            row[f'{class_name}_F1_mean'] = float(per_class_arr[:, idx].mean())
            row[f'{class_name}_F1_std'] = float(per_class_arr[:, idx].std())
        first_seed = next(str(seed) for seed in MERGE_SEEDS
                          if str(seed) in seed_results and cfg in seed_results[str(seed)])
        first_hit = seed_results[first_seed][cfg]
        row['params'] = first_hit.get('params', None)
        row['size_kb'] = first_hit.get('model_size_kb', None)
        rows.append(row)
    return pd.DataFrame(rows)


def wilcoxon_test(seed_results: dict, cfg_a: str, cfg_b: str,
                  metric: str = 'macro_f1') -> dict:
    paired_seeds = [
        str(seed) for seed in MERGE_SEEDS
        if str(seed) in seed_results
        and cfg_a in seed_results[str(seed)]
        and cfg_b in seed_results[str(seed)]
    ]
    if len(paired_seeds) < 2:
        return {'skipped': True, 'reason': f'Insufficient paired data for {cfg_a} vs {cfg_b}'}
    vals_a = np.asarray([seed_results[seed][cfg_a][metric] for seed in paired_seeds], dtype=np.float64)
    vals_b = np.asarray([seed_results[seed][cfg_b][metric] for seed in paired_seeds], dtype=np.float64)
    diffs = vals_a - vals_b
    if np.all(diffs == 0):
        return {'stat': 0.0, 'p': 1.0, 'diff_mean': 0.0, 'n': len(paired_seeds), 'verdict': 'identical'}
    stat, p = wilcoxon(vals_a, vals_b, zero_method='wilcox')
    if p < 0.01:
        verdict = '** p<0.01'
    elif p < 0.05:
        verdict = '* p<0.05'
    else:
        verdict = 'not significant'
    return {
        'stat': float(stat),
        'p': float(p),
        'diff_mean': float(diffs.mean()),
        'n': len(paired_seeds),
        'verdict': verdict,
    }


def merge_j_results(existing: dict, j_results: dict) -> dict:
    merged = copy.deepcopy(existing)
    for student_name, block_name in RESULT_BLOCK_BY_STUDENT.items():
        for seed in MERGE_SEEDS:
            seed_key = str(seed)
            merged[block_name][seed_key][J_CONFIG_NAME] = copy.deepcopy(j_results[student_name][seed_key])
    merged['run_status'] = 'complete' if not J_ONLY_QUICK_MODE else 'quick_sanity_only'
    merged['protocol_id'] = J_PROTOCOL_ID
    merged['base_protocol_id'] = BASE_PROTOCOL_ID
    merged['j_only_metadata'] = {
        'source_existing_results': str(existing_results_path),
        'merge_output_dir': str(merge_output_dir),
        'seeds': MERGE_SEEDS,
        'students': list(STUDENT_SPECS_FOR_J.keys()),
        'config_added': J_CONFIG_NAME,
        'preprocessing': {
            'scaler_fit_partition': 'train',
            'split_random_state': 42,
            'base_preprocessing_artifact': str(existing_results_path),
            'scaler_exact_match_with_base': True,
        },
        'codistill_weights': {
            'ce': CODISTILL_CE_WEIGHT,
            'rf': CODISTILL_RF_WEIGHT,
            'cl': CODISTILL_CL_WEIGHT,
        },
        'codistill_schedule': {
            'epochs': CODISTILL_EPOCHS,
            'lr': CODISTILL_LR,
            'patience': CODISTILL_PATIENCE,
            'T': BEST_T,
        },
        'runtime_sec': j_results.get('_runtime_sec'),
        'note': 'Existing configs were loaded from the train-only-scaler 10-seed JSON; only Config J was newly trained.',
    }
    merged['result_scope'] = (
        'Config J co-distillation added to the completed v2 train-only-scaler '
        'run under the archived seed-42 stratified random-row split. Exact '
        'duplicate records may cross partitions.'
    )
    return merged


def build_j_wilcoxon_results(seed_results: dict) -> dict:
    comparisons = [
        (J_CONFIG_NAME, 'D_Small_MLP', 'J vs scratch student'),
        (J_CONFIG_NAME, 'E_KD_from_RF', 'J vs RF KD'),
        (J_CONFIG_NAME, 'E2_KD_from_MLP', 'J vs MLP KD'),
        (J_CONFIG_NAME, 'F_KD_from_CL_MLP_fair', 'J vs nominal-schedule CL KD'),
        (J_CONFIG_NAME, 'F_KD_from_CL_MLP', 'J vs CL KD alias'),
        (J_CONFIG_NAME, 'B_Full_MLP', 'J vs full MLP teacher'),
    ]
    out = {}
    for cfg_a, cfg_b, desc in comparisons:
        result = wilcoxon_test(seed_results, cfg_a, cfg_b)
        result['desc'] = desc
        result['a_config'] = cfg_a
        result['b_config'] = cfg_b
        out[f'{cfg_a}_vs_{cfg_b}'] = result
    return out


merged_results = merge_j_results(existing_results, j_only_results)
merged_A = merged_results['wsn_ds_multi_seed_student_A']
merged_B = merged_results['wsn_ds_multi_seed_student_B']
agg_A = aggregate_multi_seed(merged_A)
agg_B = aggregate_multi_seed(merged_B)

j_wilcoxon_results = {
    'student_A_32_16': build_j_wilcoxon_results(merged_A),
    'student_B_64_32': build_j_wilcoxon_results(merged_B),
}
merged_results['aggregate_student_A'] = agg_A.to_dict('records')
merged_results['aggregate_student_B'] = agg_B.to_dict('records')
merged_results['wilcoxon_results_with_J'] = j_wilcoxon_results
merged_results['wilcoxon_results'] = j_wilcoxon_results['student_A_32_16']

print("\nStudent A aggregate with J:")
print(agg_A[agg_A['Config'].isin(['D_Small_MLP', 'E_KD_from_RF', 'E2_KD_from_MLP',
                                  'F_KD_from_CL_MLP_fair', J_CONFIG_NAME])]
      [['Config', 'MacroF1_mean', 'MacroF1_std', 'Accuracy_mean', 'n_seeds']].to_string(index=False))
print("\nStudent B aggregate with J:")
print(agg_B[agg_B['Config'].isin(['D_Small_MLP', 'E_KD_from_RF', 'E2_KD_from_MLP',
                                  'F_KD_from_CL_MLP_fair', J_CONFIG_NAME])]
      [['Config', 'MacroF1_mean', 'MacroF1_std', 'Accuracy_mean', 'n_seeds']].to_string(index=False))

# ============================================================================
# CELL 12: Figures from merged results
# ============================================================================
def plot_per_class_f1(agg_df: pd.DataFrame, output_path: Path, title_suffix: str) -> None:
    configs_to_plot = [
        'B_Full_MLP',
        'D_Small_MLP',
        'E_KD_from_RF',
        'E2_KD_from_MLP',
        'F_KD_from_CL_MLP_fair',
        J_CONFIG_NAME,
    ]
    configs_present = [cfg for cfg in configs_to_plot if cfg in set(agg_df['Config'])]
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(NUM_CLASSES)
    width = max(0.10, min(0.14, 0.75 / max(len(configs_present), 1)))
    colors = plt.cm.tab10(np.linspace(0, 1, len(configs_present)))
    for idx, cfg in enumerate(configs_present):
        means = [agg_df[agg_df['Config'] == cfg][f'{name}_F1_mean'].iloc[0] for name in CLASS_NAMES]
        stds = [agg_df[agg_df['Config'] == cfg][f'{name}_F1_std'].iloc[0] for name in CLASS_NAMES]
        ax.bar(x_pos + idx * width, means, width, yerr=stds, label=cfg, color=colors[idx], capsize=2)
    ax.set_xticks(x_pos + width * (len(configs_present) - 1) / 2)
    ax.set_xticklabels(CLASS_NAMES, rotation=15)
    ax.set_ylabel('Per-class F1')
    ax.set_title(f'Per-class F1 across configurations ({title_suffix})')
    ax.legend(fontsize=8, loc='lower right', ncol=2)
    ax.set_ylim(0.80, 1.005)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_pareto_frontier(agg_a: pd.DataFrame, agg_b: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    def _plot(agg_df, marker, label):
        for _, row in agg_df.iterrows():
            cfg = row['Config']
            size_kb = row['size_kb'] if row['size_kb'] else 1.0
            f1 = row['MacroF1_mean']
            f1_std = row['MacroF1_std']
            ax.errorbar(size_kb, f1, yerr=f1_std, fmt=marker, markersize=7, capsize=3)
            if cfg in {'D_Small_MLP', 'E_KD_from_RF', 'E2_KD_from_MLP', 'F_KD_from_CL_MLP_fair', J_CONFIG_NAME}:
                ax.annotate(f'{cfg} ({label})', (size_kb, f1), xytext=(5, 4),
                            textcoords='offset points', fontsize=7)

    _plot(agg_a, 'o', 'A')
    _plot(agg_b, 's', 'B')
    ax.set_xscale('log')
    ax.set_xlabel('Model size (KB, fp32) - log scale')
    ax.set_ylabel('Macro F1')
    ax.set_title('Model size vs Macro F1 after adding Config J')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_j_confusion_matrix(seed_results: dict, student_name: str, output_path: Path) -> None:
    final_seed = str(MERGE_SEEDS[-1])
    cm = np.asarray(seed_results[final_seed][J_CONFIG_NAME]['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix - {J_CONFIG_NAME} - {student_name} - seed {final_seed}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_j_loss_curves(j_results: dict, output_path: Path) -> None:
    final_seed = str(MERGE_SEEDS[-1])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for student_name in STUDENT_SPECS_FOR_J:
        curve = j_results[student_name][final_seed]['loss_curve']
        axes[0].plot(curve['loss'], label=student_name)
        axes[1].plot(curve['val_f1'], label=student_name)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training loss')
    axes[0].set_title(f'Config J training loss - seed {final_seed}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation macro F1')
    axes[1].set_title(f'Config J validation F1 - seed {final_seed}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


plot_per_class_f1(agg_A, merge_output_dir / 'per_class_f1_student_A_with_J.png', 'Student A, 10 seeds with J')
plot_per_class_f1(agg_B, merge_output_dir / 'per_class_f1_student_B_with_J.png', 'Student B, 10 seeds with J')
plot_pareto_frontier(agg_A, agg_B, merge_output_dir / 'pareto_frontier_with_J.png')
plot_j_confusion_matrix(merged_A, 'student_A_32_16', merge_output_dir / 'confusion_matrix_J_student_A.png')
plot_j_confusion_matrix(merged_B, 'student_B_64_32', merge_output_dir / 'confusion_matrix_J_student_B.png')
plot_j_loss_curves(j_only_results, merge_output_dir / 'loss_curves_J_codistill.png')
print('Saved merged result figures.')

# ============================================================================
# CELL 13: Save merged JSON, CSVs, and report
# ============================================================================
def copy_existing_artifacts(existing_json_path: Path, output_dir: Path) -> None:
    source_dir = existing_json_path.parent
    for pattern in ['*.png']:
        for artifact in source_dir.glob(pattern):
            destination = output_dir / artifact.name
            if not destination.exists():
                shutil.copy2(artifact, destination)


merge_report = {
    'status': 'complete' if not J_ONLY_QUICK_MODE else 'quick_sanity_only',
    'protocol_id': J_PROTOCOL_ID,
    'base_protocol_id': BASE_PROTOCOL_ID,
    'source_existing_results': str(existing_results_path),
    'source_existing_results_sha256': _sha256_file(existing_results_path),
    'output_dir': str(merge_output_dir),
    'seeds': MERGE_SEEDS,
    'students': list(STUDENT_SPECS_FOR_J.keys()),
    'added_config': J_CONFIG_NAME,
    'preprocessing': {
        'scope': 'train-only StandardScaler after seed-42 raw split',
        'dataset_sha256': dataset_sha256,
        'split_hashes': split_hashes,
        'scaler_sha256': scaler_sha256,
        'duplicate_policy': 'archived random-row split; duplicate groups are not constrained',
    },
    'existing_configs_preserved': REQUIRED_EXISTING_CONFIGS,
    'j_runtime_sec': j_only_results.get('_runtime_sec'),
    'warning': 'Do not use quick-mode merged outputs for final evidence.',
}

copy_existing_artifacts(existing_results_path, merge_output_dir)
write_json(merge_output_dir / 'cukd_xai_results_with_J.json', merged_results)
write_json(merge_output_dir / 'j_only_results.json', j_only_results)
write_json(merge_output_dir / 'merge_report.json', merge_report)
agg_A.to_csv(merge_output_dir / 'wsnds_results_student_A.csv', index=False)
agg_B.to_csv(merge_output_dir / 'wsnds_results_student_B.csv', index=False)


def _write_artifact_manifest() -> None:
    manifest_path = merge_output_dir / 'artifact_manifest.json'
    files = []
    for path in sorted(merge_output_dir.rglob('*')):
        if not path.is_file():
            continue
        if path == manifest_path or path.name.endswith('.tmp'):
            continue
        files.append({
            'path': path.relative_to(merge_output_dir).as_posix(),
            'size_bytes': path.stat().st_size,
            'sha256': _sha256_file(path),
        })
    write_json(manifest_path, {
        'protocol_id': J_PROTOCOL_ID,
        'base_protocol_id': BASE_PROTOCOL_ID,
        'status': 'complete' if not J_ONLY_QUICK_MODE else 'quick_sanity_only',
        'script_sha256': script_sha256,
        'dataset_sha256': dataset_sha256,
        'split_hashes': split_hashes,
        'scaler_sha256': scaler_sha256,
        'file_count_excluding_manifest': len(files),
        'files': files,
    })


_write_artifact_manifest()

print(f"Saved merged results to: {merge_output_dir}")
print("  cukd_xai_results_with_J.json")
print("  j_only_results.json")
print("  merge_report.json")
print("  wsnds_results_student_A.csv")
print("  wsnds_results_student_B.csv")

# ============================================================================
# CELL 14: Final decision summary
# ============================================================================
def _row(agg_df: pd.DataFrame, cfg: str) -> pd.Series:
    return agg_df[agg_df['Config'] == cfg].iloc[0]


print("\n" + "=" * 60)
print("FINAL MERGED J-ONLY SUMMARY")
print("=" * 60)
for student_name, agg_df in [('student_A_32_16', agg_A), ('student_B_64_32', agg_B)]:
    j_row = _row(agg_df, J_CONFIG_NAME)
    d_row = _row(agg_df, 'D_Small_MLP')
    e_row = _row(agg_df, 'E_KD_from_RF')
    f_row = _row(agg_df, 'F_KD_from_CL_MLP_fair')
    print(f"\n{student_name}")
    print(f"  J Macro F1: {j_row['MacroF1_mean']:.4f} ± {j_row['MacroF1_std']:.4f}")
    print(f"  J - D:      {j_row['MacroF1_mean'] - d_row['MacroF1_mean']:+.4f}")
    print(f"  J - E:      {j_row['MacroF1_mean'] - e_row['MacroF1_mean']:+.4f}")
    print(f"  J - Ffair:  {j_row['MacroF1_mean'] - f_row['MacroF1_mean']:+.4f}")
print("\nUse the Wilcoxon entries in cukd_xai_results_with_J.json for final significance claims.")
