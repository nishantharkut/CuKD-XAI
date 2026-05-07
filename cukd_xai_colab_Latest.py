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

# ----------------------------------------------------------------------------
# EXPERIMENT CONFIGURATION
# ----------------------------------------------------------------------------
# v3.0 — 10-seed publication run (Workstream: tighter Wilcoxon CIs).
# Toggle RUN_FULL_10_SEEDS=False to fall back to the v2.3 5-seed list, or set
# QUICK_MODE=True below to run a single seed for debugging.
RUN_FULL_10_SEEDS = True
SEEDS_V2 = [42, 123, 456, 789, 1001]                                  # v2.3
SEEDS_V3 = [42, 123, 456, 789, 1001, 2024, 3141, 5678, 8192, 9999]    # v3.0
SEEDS    = SEEDS_V3 if RUN_FULL_10_SEEDS else SEEDS_V2
N_SEEDS  = len(SEEDS)                     # Set to 1 for quick debugging runs
QUICK_MODE = False                        # True = single seed, no grid search
RUN_CICIOT = False                        # True = also run CICIoT2023 (slow)
CICIOT_PATH = 'CICIoT2023.csv'            # Path if using CICIoT2023
WSNDS_PATH = 'WSN-DS.csv'                 # Path to WSN-DS

# ----------------------------------------------------------------------------
# v3.0 — Workstream A: Domain-informed feature engineering
# ----------------------------------------------------------------------------
# Adds engineered feature interactions targeting the diagnosed Blackhole/
# Grayhole confusion gap (ACCURACY_IMPROVEMENT_PLAN.md §1.2).
#   Tier 1: drop_rate, forward_ratio, data_drop_count    (3 features)
#   Tier 2: + control_to_data_ratio, adv_imbalance, join_imbalance  (6 total)
#   Tier 3: + ch_distance_normalized, energy_per_packet  (8 total)
USE_ENGINEERED_FEATURES = True
ENGINEERED_FEATURE_TIER = 2

# ----------------------------------------------------------------------------
# v3.0 — Workstream D1: Decoupled Knowledge Distillation (Zhao CVPR 2022)
# ----------------------------------------------------------------------------
# Decomposes the standard KD loss into TCKD (target class) + NCKD (non-target
# class). NCKD specifically attacks inter-class confusion — exactly our
# Blackhole/Grayhole failure mode.
#
# Total loss:  alpha * TCKD * T^2  +  beta * NCKD * T^2
# Zhao et al. recommend alpha=1.0, beta in {4..16}; we default to beta=8.
USE_DKD   = True
DKD_ALPHA = 1.0    # TCKD weight
DKD_BETA  = 8.0    # NCKD weight

# ----------------------------------------------------------------------------
# v3.0 — Workstream F: Training-procedure tricks
# ----------------------------------------------------------------------------
# Lightweight bundle: EMA (Polyak averaging) + label smoothing + grad clipping.
# Each is individually small (+0.1-0.5 pp) but they stack additively.
# Heavier tricks (SWA, OneCycleLR) intentionally deferred until baseline
# improvements stabilize.
USE_TRAINING_TRICKS   = True
TRICK_LABEL_SMOOTHING = 0.05    # CrossEntropyLoss label_smoothing
TRICK_EMA_DECAY       = 0.999   # exponential moving average of weights
TRICK_GRAD_CLIP       = 1.0     # max gradient norm (0 = disabled)

# ----------------------------------------------------------------------------
# v3.0 — Quantization-Aware Training for proper INT8 deployment
# ----------------------------------------------------------------------------
# PT dynamic quantization gave 19.78% size reduction with 2-12% F1 drops on
# tiny MLPs. QAT trains the model to be robust to int8 rounding from the
# start, typically retaining >99% of fp32 performance.
#   Cite: Jacob et al. CVPR 2018; Krishnamoorthi 2018 (arXiv:1806.08342).
# QAT runs on CPU because PyTorch quantization is most reliable there.
USE_QAT          = True
QAT_FT_EPOCHS    = 10           # fine-tuning epochs from a trained fp32 model
QAT_FT_LR        = 1e-4         # low LR for stable QAT fine-tuning

# ----------------------------------------------------------------------------
# v3.0 — CICIoT2023 generalization (replaces v2 single-CSV stub)
# ----------------------------------------------------------------------------
# v3 uses the official UNB CIC distribution: a directory of MERGED_CSV files
# (Merged01.csv ... Merged63.csv) each containing pre-merged attack types with
# a `Label` column. We stream them with a stratified per-class cap (50K) and
# fall back to all-available for the floor classes (Web ≈ 24K, BruteForce ≈ 13K).
CICIOT_MERGED_DIR    = '/content/MERGED_CSV'   # Colab default; set absolute path locally
CICIOT_CAP_PER_CLASS = 50_000
# (RUN_CICIOT and the legacy CICIOT_PATH from v2 above remain — v3 ignores
#  CICIOT_PATH and only triggers when RUN_CICIOT=True AND CICIOT_MERGED_DIR exists.)

# ----------------------------------------------------------------------------
# v3.0 — Workstream C: Architecture sweep (Pareto frontier across student sizes)
# ----------------------------------------------------------------------------
# Trains KD-from-RF students at multiple sizes to map the accuracy-vs-size
# trade-off. Answers "how much accuracy do you lose as you shrink the MLP?"
# (the third research question). Output drives an extended Pareto plot.
USE_ARCH_SWEEP = True
# (variant_name, hidden_dims) — XS / S / M / L / XL. S and M overlap with
# Student A and Student B; we re-train them inside the sweep for self-contained
# Pareto evidence (same seed list, fresh model instances).
ARCH_SWEEP_VARIANTS = [
    ('XS', (16, 8)),     # ultra-compact (~500 params, ~0.5 KB int8)
    ('S',  (32, 16)),    # current Student A (~1.2 KB int8)
    ('M',  (64, 32)),    # current Student B (~3.3 KB int8)
    ('L',  (96, 48)),    # larger (~7 KB int8 — tight on TelosB)
    ('XL', (128, 64)),   # too big for TelosB; informative ceiling
]

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
# CELL 4b (v3.0): Domain-informed feature engineering — Workstream A
# ============================================================================
# Toggle via USE_ENGINEERED_FEATURES in CELL 2. When True, this cell augments
# X_all (and X_all_std) with engineered features computed from the existing
# 17 raw columns. INPUT_DIM and FEATURE_NAMES are also updated so that the
# downstream model architectures (CELL 5) and run_all_configs (CELL 9) see
# the augmented feature set without any further changes.
# ============================================================================

if USE_ENGINEERED_FEATURES:
    print("\n" + "=" * 60)
    print(f"v3 FEATURE ENGINEERING — TIER {ENGINEERED_FEATURE_TIER}")
    print("=" * 60)

    # Required raw columns by tier (post str.strip() in CELL 3)
    _required_by_tier = {
        1: ['DATA_R', 'DATA_S', 'Data_Sent_To_BS'],
        2: ['DATA_R', 'DATA_S', 'Data_Sent_To_BS',
            'ADV_S', 'ADV_R', 'JOIN_S', 'JOIN_R'],
        3: ['DATA_R', 'DATA_S', 'Data_Sent_To_BS',
            'ADV_S', 'ADV_R', 'JOIN_S', 'JOIN_R',
            'Dist_To_CH', 'dist_CH_To_BS', 'Expaned Energy'],
    }
    _missing = [c for c in _required_by_tier[ENGINEERED_FEATURE_TIER]
                if c not in df.columns]
    if _missing:
        raise KeyError(f"Engineered features need columns {_missing}; "
                       f"got {list(df.columns)}")

    # Build augmented dataframe
    df_aug = df.copy()
    EPS = 1e-6

    # ----- Tier 1: drop-rate trio (targets Blackhole/Grayhole) -----
    df_aug['drop_rate'] = (1.0 - df_aug['Data_Sent_To_BS']
                                 / np.maximum(df_aug['DATA_R'], EPS))
    df_aug['forward_ratio'] = (df_aug['Data_Sent_To_BS']
                                / np.maximum(df_aug['DATA_S'], EPS))
    df_aug['data_drop_count'] = (df_aug['DATA_R'] - df_aug['Data_Sent_To_BS'])

    # ----- Tier 2: control-plane / Flooding sensitivity -----
    if ENGINEERED_FEATURE_TIER >= 2:
        df_aug['control_to_data_ratio'] = (
            (df_aug['ADV_S'] + df_aug['JOIN_S'])
            / np.maximum(df_aug['DATA_S'], EPS))
        df_aug['adv_imbalance'] = (
            (df_aug['ADV_S'] - df_aug['ADV_R']).abs()
            / np.maximum(df_aug['ADV_S'] + df_aug['ADV_R'], EPS))
        df_aug['join_imbalance'] = (
            (df_aug['JOIN_S'] - df_aug['JOIN_R']).abs()
            / np.maximum(df_aug['JOIN_S'] + df_aug['JOIN_R'], EPS))

    # ----- Tier 3: cluster-head topology context -----
    if ENGINEERED_FEATURE_TIER >= 3:
        df_aug['ch_distance_normalized'] = (
            df_aug['Dist_To_CH']
            / np.maximum(df_aug['dist_CH_To_BS'], EPS))
        df_aug['energy_per_packet'] = (
            df_aug['Expaned Energy']
            / np.maximum(df_aug['DATA_S'] + df_aug['DATA_R'], EPS))

    # Sanitize newly added columns (handle inf / nan from edge-case divisions)
    _new_cols = [c for c in df_aug.columns if c not in df.columns]
    for c in _new_cols:
        df_aug[c] = np.nan_to_num(df_aug[c].values.astype(np.float32),
                                   nan=0.0, posinf=1e6, neginf=-1e6)

    # Re-extract X_all, FEATURE_NAMES, INPUT_DIM with engineered features
    feature_cols_aug = FEATURE_NAMES + _new_cols
    X_all = df_aug[feature_cols_aug].values.astype(np.float32)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=1e6, neginf=-1e6)
    FEATURE_NAMES = feature_cols_aug
    INPUT_DIM = X_all.shape[1]

    # IMPORTANT: re-fit StandardScaler on the augmented X_all so downstream
    # cells see correctly normalized features.
    scaler = StandardScaler()
    X_all_std = scaler.fit_transform(X_all)

    print(f"  Engineered features added ({len(_new_cols)}): {_new_cols}")
    print(f"  New INPUT_DIM: {INPUT_DIM}")
    print(f"  New FEATURE_NAMES: {FEATURE_NAMES}")
else:
    print("\n[v3] Engineered features DISABLED — running v2.3 baseline.")

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


# ============================================================================
# v3.0 (Workstream D1): Decoupled Knowledge Distillation
#   Cite: Zhao, B. et al. 2022. "Decoupled Knowledge Distillation". CVPR 2022.
# ============================================================================
def _dkd_loss(s_logits: torch.Tensor,
              t_probs: torch.Tensor,
              labels: torch.Tensor,
              T: float = 4.0,
              alpha: float = 1.0,
              beta: float = 8.0,
              eps: float = 1e-8) -> torch.Tensor:
    """Decoupled KD loss from Zhao et al. 2022.

    Args:
      s_logits: [B, C] raw student logits.
      t_probs:  [B, C] teacher probabilities, ALREADY temperature-softened
                (matches train_kd's convention: softmax(teacher_logits / T)).
      labels:   [B] hard ground-truth labels.
      T:        temperature (must match the teacher_probs temperature).
      alpha:    TCKD weight.
      beta:     NCKD weight.

    Returns: scalar loss = (alpha * TCKD + beta * NCKD) * T^2
    """
    B, C = s_logits.shape

    # Student probabilities at temperature T
    s_probs = F.softmax(s_logits / T, dim=1)
    t_probs = t_probs.clamp(min=eps)

    # Boolean target mask
    gt_mask = torch.zeros_like(s_logits, dtype=torch.bool).scatter_(
        1, labels.unsqueeze(1), True)
    other_mask = ~gt_mask

    # Aggregate target / non-target probabilities (binary view)
    s_tc = s_probs.masked_select(gt_mask).view(B, 1)              # [B,1]
    s_nc = (s_probs * other_mask.float()).sum(dim=1, keepdim=True)
    t_tc = t_probs.masked_select(gt_mask).view(B, 1)
    t_nc = (t_probs * other_mask.float()).sum(dim=1, keepdim=True)

    # ----- TCKD: KL on the binary (target, non-target) distribution -----
    tckd = (
        t_tc * (torch.log(t_tc + eps) - torch.log(s_tc + eps))
        + t_nc * (torch.log(t_nc + eps) - torch.log(s_nc + eps))
    ).sum(dim=1).mean()

    # ----- NCKD: KL on the renormalized non-target distribution -----
    s_nc_renorm = (s_probs * other_mask.float()) / (s_nc + eps)
    t_nc_renorm = (t_probs * other_mask.float()) / (t_nc + eps)
    nckd = (
        t_nc_renorm
        * (torch.log(t_nc_renorm + eps) - torch.log(s_nc_renorm + eps))
        * other_mask.float()                  # zero-out target class
    ).sum(dim=1).mean()

    return (alpha * tckd + beta * nckd) * (T * T)


def train_kd_dkd(student: nn.Module,
                 teacher_source,
                 X_train: torch.Tensor, y_train: torch.Tensor,
                 X_val: torch.Tensor, y_val: torch.Tensor,
                 T: float = KD_T_DEFAULT,
                 alpha: float = 1.0,
                 beta: float = 8.0,
                 ce_weight: float = 0.0,
                 class_weights: torch.Tensor = None,
                 epochs: int = 30, batch_size: int = 256,
                 lr: float = 1e-3, weight_decay: float = 1e-3,
                 patience: int = 8, verbose: bool = False):
    """Knowledge distillation with Decoupled KD loss (Zhao CVPR 2022).

    Mirror of train_kd() but uses _dkd_loss instead of vanilla Hinton KD.

    teacher_source: same as train_kd — either a callable nn.Module or a raw
                    probability tensor (e.g., RF predict_proba).
    alpha:          TCKD weight (~1.0 standard).
    beta:           NCKD weight (4-16; Zhao default 8.0).
    ce_weight:      optional CE term added on top (set >0 if you want a hard-
                    label safety net; vanilla DKD does not include CE).
    """
    student = student.to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr,
                                   weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=epochs)

    if class_weights is not None:
        class_weights = class_weights.to(device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    X_d = X_train.to(device)
    y_d = y_train.to(device)
    X_val_d = X_val.to(device)
    y_val_np = y_val.cpu().numpy()

    # Precompute teacher probs at temperature T (matches train_kd convention)
    if callable(teacher_source):
        teacher_source.eval()
        soft_list = []
        with torch.no_grad():
            for i in range(0, len(X_d), 4096):
                logits = teacher_source(X_d[i:i + 4096])
                soft_list.append(F.softmax(logits / T, dim=1).detach())
        teacher_probs = torch.cat(soft_list, dim=0)
    else:
        raw = teacher_source.to(device)
        pseudo_logits = torch.log(raw.clamp(min=1e-8))
        teacher_probs = F.softmax(pseudo_logits / T, dim=1).detach()

    ds = TensorDataset(X_d, y_d, teacher_probs)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    best_val = 0.0
    best_state = None
    bad = 0

    for epoch in range(epochs):
        student.train()
        for xb, yb, tb in loader:
            optimizer.zero_grad()
            logits = student(xb)
            loss = _dkd_loss(logits, tb, yb, T=T, alpha=alpha, beta=beta)
            if ce_weight > 0:
                loss = loss + ce_weight * ce_loss(logits, yb)
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
            print(f"DKD epoch {epoch+1}: val_f1={val_f1:.4f}")

    if best_state is not None:
        student.load_state_dict(best_state)
    return student


# ============================================================================
# v3.0 (Workstream F): Training-procedure tricks
#   - EMA: Polyak averaging of weights (Tarvainen & Valpola 2017 style)
#   - Label smoothing: nn.CrossEntropyLoss(label_smoothing=...) since PyTorch 1.10
#   - Gradient clipping: torch.nn.utils.clip_grad_norm_
# ============================================================================
class EMAWrapper:
    """Exponential moving average over model parameters.

    Maintains a shadow set of weights with exponential decay. Apply shadow
    weights to the model for evaluation, then restore the live training
    weights for the next gradient step.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[n].mul_(self.decay).add_(p.detach(),
                                                  alpha=1.0 - self.decay)

    def apply_to(self, model: nn.Module):
        """Copy EMA shadow weights into model (in-place). Returns the
        previous live weights so caller can restore them after eval."""
        backup = {n: p.detach().clone() for n, p in model.named_parameters()
                  if p.requires_grad}
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n])
        return backup

    @staticmethod
    def restore(model: nn.Module, backup: dict):
        for n, p in model.named_parameters():
            if n in backup:
                p.data.copy_(backup[n])


def train_standard_v3(model: nn.Module,
                      X_train: torch.Tensor, y_train: torch.Tensor,
                      X_val: torch.Tensor, y_val: torch.Tensor,
                      class_weights: torch.Tensor = None,
                      epochs: int = 30, batch_size: int = 256, lr: float = 1e-3,
                      weight_decay: float = 1e-3, patience: int = 8,
                      label_smoothing: float = 0.0,
                      ema_decay: float = 0.0,
                      grad_clip_norm: float = 0.0,
                      return_loss_curve: bool = False, verbose: bool = False):
    """Drop-in replacement for train_standard with optional v3 tricks.

    Backwards-compatible defaults: with label_smoothing=0, ema_decay=0,
    grad_clip_norm=0 the function is functionally identical to train_standard.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=epochs)

    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights,
                                     label_smoothing=label_smoothing)

    train_ds = TensorDataset(X_train.to(device), y_train.to(device))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    X_val_d = X_val.to(device)
    y_val_np = y_val.cpu().numpy()

    ema = EMAWrapper(model, decay=ema_decay) if ema_decay > 0 else None

    loss_curve, val_curve = [], []
    best_val, best_state, bad = 0.0, None, 0

    for epoch in range(epochs):
        model.train()
        epoch_loss, nb = 0.0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                grad_clip_norm)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            epoch_loss += loss.item()
            nb += 1
        scheduler.step()
        epoch_loss /= max(nb, 1)
        loss_curve.append(epoch_loss)

        # Validation: use EMA weights if active
        if ema is not None:
            backup = ema.apply_to(model)
        preds = _batched_predict(model, X_val_d)
        val_f1 = f1_score(y_val_np, preds, average='macro')
        if ema is not None:
            EMAWrapper.restore(model, backup)
        val_curve.append(val_f1)

        if val_f1 > best_val:
            best_val = val_f1
            # Snapshot whichever weights were just evaluated (EMA or live)
            if ema is not None:
                backup = ema.apply_to(model)
                best_state = copy.deepcopy(model.state_dict())
                EMAWrapper.restore(model, backup)
            else:
                best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

        if verbose:
            print(f"v3 ep {epoch+1}: loss={epoch_loss:.4f} val_f1={val_f1:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    if return_loss_curve:
        return model, {'loss': loss_curve, 'val_f1': val_curve}
    return model


def train_kd_v3(student: nn.Module,
                teacher_source,
                X_train: torch.Tensor, y_train: torch.Tensor,
                X_val: torch.Tensor, y_val: torch.Tensor,
                T: float = KD_T_DEFAULT, alpha: float = KD_ALPHA_DEFAULT,
                class_weights: torch.Tensor = None,
                epochs: int = 30, batch_size: int = 256,
                lr: float = 1e-3, weight_decay: float = 1e-3,
                patience: int = 8,
                label_smoothing: float = 0.0,
                ema_decay: float = 0.0,
                grad_clip_norm: float = 0.0,
                verbose: bool = False):
    """KD (vanilla Hinton loss) with v3 tricks added (EMA + label smoothing
    on the CE term + gradient clipping). Backwards-compatible defaults."""
    student = student.to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr,
                                   weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=epochs)

    if class_weights is not None:
        class_weights = class_weights.to(device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights,
                                   label_smoothing=label_smoothing)

    X_d = X_train.to(device)
    y_d = y_train.to(device)
    X_val_d = X_val.to(device)
    y_val_np = y_val.cpu().numpy()

    if callable(teacher_source):
        teacher_source.eval()
        soft_list = []
        with torch.no_grad():
            for i in range(0, len(X_d), 4096):
                logits = teacher_source(X_d[i:i + 4096])
                soft_list.append(F.softmax(logits / T, dim=1).detach())
        soft_targets = torch.cat(soft_list, dim=0)
    else:
        raw = teacher_source.to(device)
        pseudo_logits = torch.log(raw.clamp(min=1e-8))
        soft_targets = F.softmax(pseudo_logits / T, dim=1).detach()

    ds = TensorDataset(X_d, y_d, soft_targets)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    ema = EMAWrapper(student, decay=ema_decay) if ema_decay > 0 else None

    best_val, best_state, bad = 0.0, None, 0
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
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(),
                                                grad_clip_norm)
            optimizer.step()
            if ema is not None:
                ema.update(student)
        scheduler.step()

        if ema is not None:
            backup = ema.apply_to(student)
        preds = _batched_predict(student, X_val_d)
        val_f1 = f1_score(y_val_np, preds, average='macro')
        if ema is not None:
            EMAWrapper.restore(student, backup)

        if val_f1 > best_val:
            best_val = val_f1
            if ema is not None:
                backup = ema.apply_to(student)
                best_state = copy.deepcopy(student.state_dict())
                EMAWrapper.restore(student, backup)
            else:
                best_state = copy.deepcopy(student.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose:
            print(f"KD-v3 ep {epoch+1}: val_f1={val_f1:.4f}")

    if best_state is not None:
        student.load_state_dict(best_state)
    return student


# ============================================================================
# v3.0 (Quantization-Aware Training): proper INT8 deployment
#   Cite: Jacob et al. 2018, CVPR; Krishnamoorthi 2018 whitepaper.
# ============================================================================
class QATStudentMLP(nn.Module):
    """StudentMLP wrapped with Quant/DeQuant stubs at the boundaries.

    Internal Sequential `self.net` is structurally identical to StudentMLP,
    so a state_dict from a trained StudentMLP can be loaded directly into
    `qat_model.net.load_state_dict(...)` to warm-start QAT fine-tuning.
    """
    def __init__(self, input_dim: int = 17, hidden_dims: tuple = (32, 16),
                 num_classes: int = 5):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.net(x)
        x = self.dequant(x)
        return x


def run_architecture_sweep(seed: int,
                            X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray,
                            variants=None) -> dict:
    """Workstream C — Train KD-from-RF students at multiple sizes to map
    the accuracy-vs-size Pareto trade-off.

    Re-uses one calibrated RF teacher across all variants (significantly
    cheaper than training one per variant). Returns dict keyed by variant
    name (e.g., 'arch_XS', 'arch_S', ...) with full metrics + size info.
    """
    if variants is None:
        variants = ARCH_SWEEP_VARIANTS
    set_seed(seed)
    results = {}

    # Tensorize once
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
    y_val_t   = torch.tensor(y_val,   dtype=torch.long)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_test_t  = torch.tensor(y_test,  dtype=torch.long)

    # Class weights (inverse frequency, same as run_all_configs)
    counts = np.bincount(y_train, minlength=NUM_CLASSES)
    class_weights = torch.tensor(
        len(y_train) / (NUM_CLASSES * np.maximum(counts, 1)),
        dtype=torch.float32)

    # Train RF teacher ONCE — reused across all student variants
    rf_calib_sweep = CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=500, max_depth=15,
                                random_state=seed, n_jobs=-1),
        method='isotonic', cv=3,
    )
    rf_calib_sweep.fit(X_train, y_train)
    rf_soft_sweep = torch.tensor(
        rf_calib_sweep.predict_proba(X_train), dtype=torch.float32)

    # Train each variant
    for name, hidden in variants:
        print(f"  [arch_{name}] hidden={hidden} ...")
        student = StudentMLP(INPUT_DIM, hidden, NUM_CLASSES)
        student = train_kd(
            student, rf_soft_sweep,
            X_train_t, y_train_t, X_val_t, y_val_t,
            T=BEST_T, alpha=BEST_ALPHA,
            class_weights=class_weights,
            epochs=TRAIN_CONFIG['epochs'],
            batch_size=TRAIN_CONFIG['batch_size'],
        )
        m = evaluate_model(student, X_test_t, y_test_t)
        m['hidden_dims']                 = list(hidden)
        m['params']                      = count_params(student)
        m['model_size_kb_fp32']          = model_size_kb(student)
        m['model_size_kb_int8_theoretical'] = m['params'] / 1024.0
        m['flops']                       = compute_flops_mlp(
            INPUT_DIM, hidden, NUM_CLASSES)
        m['fits_telosb_10kb']            = m['model_size_kb_int8_theoretical'] <= 10.0
        results[f'arch_{name}'] = m

    return results


def derive_student_hidden_dims(student_model: nn.Module) -> tuple:
    """Inspect a trained StudentMLP-shaped model and return its hidden_dims
    tuple. Robust to whatever architecture run_all_configs trained.
    """
    out_features = []
    for module in student_model.net:
        if isinstance(module, nn.Linear):
            out_features.append(module.out_features)
    # Last entry is num_classes (output layer); strip it
    return tuple(out_features[:-1])


def train_qat_from_pretrained(pretrained_student: nn.Module,
                              X_train: torch.Tensor, y_train: torch.Tensor,
                              X_val: torch.Tensor, y_val: torch.Tensor,
                              hidden_dims: tuple,
                              num_classes: int,
                              class_weights: torch.Tensor = None,
                              qat_epochs: int = 10,
                              batch_size: int = 256,
                              lr: float = 1e-4,
                              freeze_observer_after=None,
                              verbose: bool = False) -> nn.Module:
    """Take a trained fp32 StudentMLP, fine-tune with QAT for `qat_epochs`,
    return a converted int8 model.

    Notes:
      * Runs on CPU because PyTorch QAT/INT8 inference is most reliable there.
      * Initial weights are copied from `pretrained_student.net.state_dict()`.
      * Observers are frozen after `freeze_observer_after` epochs (default
        max(3, qat_epochs // 2)) so quantization parameters stabilize.
      * Returns the converted int8 model (calls `convert(...)` at the end).
    """
    qat_device = torch.device('cpu')
    if freeze_observer_after is None:
        freeze_observer_after = max(3, qat_epochs // 2)

    # Infer input_dim from the pretrained student
    input_dim = pretrained_student.net[0].in_features

    # Build QAT shell, copy fp32 weights, prepare for QAT
    qat_model = QATStudentMLP(input_dim, hidden_dims, num_classes)
    qat_model.net.load_state_dict(pretrained_student.net.state_dict())
    qat_model.to(qat_device).train()
    qat_model.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
    qat_model = torch.ao.quantization.prepare_qat(qat_model, inplace=False)

    # Loss
    cw = class_weights.to(qat_device) if class_weights is not None else None
    ce = nn.CrossEntropyLoss(weight=cw)

    X_d = X_train.to(qat_device); y_d = y_train.to(qat_device)
    X_v = X_val.to(qat_device);   y_v = y_val.to(qat_device)
    y_v_np = y_v.numpy()

    ds = TensorDataset(X_d, y_d)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(qat_model.parameters(), lr=lr,
                                   weight_decay=1e-4)

    best_val, best_state = 0.0, None
    for ep in range(qat_epochs):
        if ep == freeze_observer_after:
            qat_model.apply(torch.ao.quantization.disable_observer)

        qat_model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = qat_model(xb)
            loss = ce(logits, yb)
            loss.backward()
            optimizer.step()

        # Validation in eval mode (fake-quant still active for honest signal)
        qat_model.eval()
        with torch.no_grad():
            preds = qat_model(X_v).argmax(dim=1).numpy()
        val_f1 = f1_score(y_v_np, preds, average='macro')
        if val_f1 > best_val:
            best_val = val_f1
            best_state = copy.deepcopy(qat_model.state_dict())
        if verbose:
            print(f"  QAT ep {ep+1}: val_f1={val_f1:.4f}")

    if best_state is not None:
        qat_model.load_state_dict(best_state)

    # Convert to true int8
    qat_model.eval()
    int8_model = torch.ao.quantization.convert(qat_model.to(qat_device),
                                                inplace=False)
    return int8_model


def evaluate_int8_cpu(int8_model: nn.Module,
                      X_test: torch.Tensor, y_test: torch.Tensor) -> dict:
    """Evaluate a converted int8 QAT model on CPU. Returns standard metrics
    (uses precision_recall_fscore_support like the existing evaluate_model).
    """
    int8_model.eval().to('cpu')
    X_cpu = X_test.cpu(); y_cpu = y_test.cpu()
    with torch.no_grad():
        preds = int8_model(X_cpu).argmax(dim=1).numpy()
    y_np = y_cpu.numpy()
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_np, preds, average='macro', zero_division=0)
    pc_prec, pc_rec, pc_f1, _ = precision_recall_fscore_support(
        y_np, preds, average=None, zero_division=0)
    return {
        'accuracy':        float((preds == y_np).mean()),
        'macro_f1':        float(f1),
        'macro_precision': float(prec),
        'macro_recall':    float(rec),
        'per_class_f1':    pc_f1.tolist(),
        'per_class_precision': pc_prec.tolist(),
        'per_class_recall': pc_rec.tolist(),
        'confusion_matrix': confusion_matrix(y_np, preds).tolist(),
    }


def model_size_int8_theoretical(model: nn.Module) -> float:
    """Theoretical int8 size = sum(params) * 1 byte. This is the deployable
    floor (raw byte-array export); torch.save() bloats it with metadata."""
    total = sum(p.numel() for p in model.parameters())
    return total / 1024.0   # KB


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

    # ----- Config B_tricks: full MLP teacher + Workstream F training tricks -----
    if USE_TRAINING_TRICKS:
        if verbose: print("[B_tricks] Full MLP + EMA + label smoothing + grad clip ...")
        teacher_b_v3 = TeacherMLP(INPUT_DIM, NUM_CLASSES)
        teacher_b_v3 = train_standard_v3(
            teacher_b_v3, X_train_t, y_train_t, X_val_t, y_val_t,
            class_weights=class_weights,
            label_smoothing=TRICK_LABEL_SMOOTHING,
            ema_decay=TRICK_EMA_DECAY,
            grad_clip_norm=TRICK_GRAD_CLIP,
            **TRAIN_CONFIG
        )
        m_b_v3 = evaluate_model(teacher_b_v3, X_test_t, y_test_t)
        m_b_v3['params'] = count_params(teacher_b_v3)
        m_b_v3['model_size_kb'] = model_size_kb(teacher_b_v3)
        m_b_v3['ece'] = expected_calibration_error(
            _batched_probs(teacher_b_v3, X_test_t), y_test
        )
        results['B_Full_MLP_tricks'] = m_b_v3
        models['B_Full_MLP_tricks'] = teacher_b_v3
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

    # ----- Config D_tricks: scratch student + training tricks -----
    if USE_TRAINING_TRICKS:
        if verbose: print("[D_tricks] Scratch student + tricks ...")
        student_d_v3 = StudentMLP(INPUT_DIM, student_hidden, NUM_CLASSES)
        student_d_v3 = train_standard_v3(
            student_d_v3, X_train_t, y_train_t, X_val_t, y_val_t,
            class_weights=class_weights,
            label_smoothing=TRICK_LABEL_SMOOTHING,
            ema_decay=TRICK_EMA_DECAY,
            grad_clip_norm=TRICK_GRAD_CLIP,
            **TRAIN_CONFIG
        )
        m_d_v3 = evaluate_model(student_d_v3, X_test_t, y_test_t)
        m_d_v3['params'] = count_params(student_d_v3)
        m_d_v3['model_size_kb'] = model_size_kb(student_d_v3)
        m_d_v3['flops'] = compute_flops_mlp(INPUT_DIM, student_hidden, NUM_CLASSES)
        m_d_v3['ece'] = expected_calibration_error(
            _batched_probs(student_d_v3, X_test_t), y_test
        )
        results['D_Small_MLP_tricks'] = m_d_v3
        models['D_Small_MLP_tricks'] = student_d_v3
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

    # ----- Config E_dkd: same as E but with Decoupled KD loss (Workstream D1) -----
    if USE_DKD:
        if verbose: print("[E_dkd] DKD from calibrated RF teacher...")
        student_e_dkd = StudentMLP(INPUT_DIM, student_hidden, NUM_CLASSES)
        student_e_dkd = train_kd_dkd(
            student_e_dkd, rf_soft, X_train_t, y_train_t, X_val_t, y_val_t,
            T=kd_T, alpha=DKD_ALPHA, beta=DKD_BETA,
            class_weights=class_weights,
            epochs=TRAIN_CONFIG['epochs'], batch_size=TRAIN_CONFIG['batch_size']
        )
        m_e_dkd = evaluate_model(student_e_dkd, X_test_t, y_test_t)
        m_e_dkd['params'] = count_params(student_e_dkd)
        m_e_dkd['model_size_kb'] = model_size_kb(student_e_dkd)
        m_e_dkd['flops'] = compute_flops_mlp(INPUT_DIM, student_hidden, NUM_CLASSES)
        m_e_dkd['ece'] = expected_calibration_error(
            _batched_probs(student_e_dkd, X_test_t), y_test
        )
        results['E_KD_from_RF_dkd'] = m_e_dkd
        models['E_KD_from_RF_dkd'] = student_e_dkd

    # ----- Config E_tricks: KD from RF + training tricks (Workstream F) -----
    if USE_TRAINING_TRICKS:
        if verbose: print("[E_tricks] KD from RF + EMA + label smoothing + grad clip ...")
        student_e_v3 = StudentMLP(INPUT_DIM, student_hidden, NUM_CLASSES)
        student_e_v3 = train_kd_v3(
            student_e_v3, rf_soft, X_train_t, y_train_t, X_val_t, y_val_t,
            T=kd_T, alpha=kd_alpha, class_weights=class_weights,
            label_smoothing=TRICK_LABEL_SMOOTHING,
            ema_decay=TRICK_EMA_DECAY,
            grad_clip_norm=TRICK_GRAD_CLIP,
            epochs=TRAIN_CONFIG['epochs'], batch_size=TRAIN_CONFIG['batch_size']
        )
        m_e_v3 = evaluate_model(student_e_v3, X_test_t, y_test_t)
        m_e_v3['params'] = count_params(student_e_v3)
        m_e_v3['model_size_kb'] = model_size_kb(student_e_v3)
        m_e_v3['flops'] = compute_flops_mlp(INPUT_DIM, student_hidden, NUM_CLASSES)
        m_e_v3['ece'] = expected_calibration_error(
            _batched_probs(student_e_v3, X_test_t), y_test
        )
        results['E_KD_from_RF_tricks'] = m_e_v3
        models['E_KD_from_RF_tricks'] = student_e_v3

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

    # ----- Config E2_dkd: same as E2 but with Decoupled KD loss -----
    if USE_DKD:
        if verbose: print("[E2_dkd] DKD from standard MLP teacher...")
        student_e2_dkd = StudentMLP(INPUT_DIM, student_hidden, NUM_CLASSES)
        student_e2_dkd = train_kd_dkd(
            student_e2_dkd, teacher_b, X_train_t, y_train_t, X_val_t, y_val_t,
            T=kd_T, alpha=DKD_ALPHA, beta=DKD_BETA,
            class_weights=class_weights,
            epochs=TRAIN_CONFIG['epochs'], batch_size=TRAIN_CONFIG['batch_size']
        )
        m_e2_dkd = evaluate_model(student_e2_dkd, X_test_t, y_test_t)
        m_e2_dkd['params'] = count_params(student_e2_dkd)
        m_e2_dkd['model_size_kb'] = model_size_kb(student_e2_dkd)
        m_e2_dkd['flops'] = compute_flops_mlp(INPUT_DIM, student_hidden, NUM_CLASSES)
        m_e2_dkd['ece'] = expected_calibration_error(
            _batched_probs(student_e2_dkd, X_test_t), y_test
        )
        results['E2_KD_from_MLP_dkd'] = m_e2_dkd
        models['E2_KD_from_MLP_dkd'] = student_e2_dkd

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

    # ----- Config F_ext_dkd: KD from extended-budget CL-MLP with DKD loss -----
    if USE_DKD:
        if verbose: print("[F_ext_dkd] DKD from extended-budget CL-MLP...")
        student_f_ext_dkd = StudentMLP(INPUT_DIM, student_hidden, NUM_CLASSES)
        student_f_ext_dkd = train_kd_dkd(
            student_f_ext_dkd, teacher_c_ext,
            X_train_t, y_train_t, X_val_t, y_val_t,
            T=kd_T, alpha=DKD_ALPHA, beta=DKD_BETA,
            class_weights=class_weights,
            epochs=TRAIN_CONFIG['epochs'], batch_size=TRAIN_CONFIG['batch_size']
        )
        m_f_ext_dkd = evaluate_model(student_f_ext_dkd, X_test_t, y_test_t)
        m_f_ext_dkd['params'] = count_params(student_f_ext_dkd)
        m_f_ext_dkd['model_size_kb'] = model_size_kb(student_f_ext_dkd)
        m_f_ext_dkd['flops'] = compute_flops_mlp(INPUT_DIM, student_hidden, NUM_CLASSES)
        m_f_ext_dkd['ece'] = expected_calibration_error(
            _batched_probs(student_f_ext_dkd, X_test_t), y_test
        )
        results['F_KD_from_CL_MLP_ext_dkd'] = m_f_ext_dkd
        models['F_KD_from_CL_MLP_ext_dkd'] = student_f_ext_dkd

    # ----- Config F_ext_tricks: KD from CL_ext + training tricks -----
    if USE_TRAINING_TRICKS:
        if verbose: print("[F_ext_tricks] KD from CL_ext + tricks ...")
        student_f_ext_v3 = StudentMLP(INPUT_DIM, student_hidden, NUM_CLASSES)
        student_f_ext_v3 = train_kd_v3(
            student_f_ext_v3, teacher_c_ext,
            X_train_t, y_train_t, X_val_t, y_val_t,
            T=kd_T, alpha=kd_alpha, class_weights=class_weights,
            label_smoothing=TRICK_LABEL_SMOOTHING,
            ema_decay=TRICK_EMA_DECAY,
            grad_clip_norm=TRICK_GRAD_CLIP,
            epochs=TRAIN_CONFIG['epochs'], batch_size=TRAIN_CONFIG['batch_size']
        )
        m_f_ext_v3 = evaluate_model(student_f_ext_v3, X_test_t, y_test_t)
        m_f_ext_v3['params'] = count_params(student_f_ext_v3)
        m_f_ext_v3['model_size_kb'] = model_size_kb(student_f_ext_v3)
        m_f_ext_v3['flops'] = compute_flops_mlp(INPUT_DIM, student_hidden, NUM_CLASSES)
        m_f_ext_v3['ece'] = expected_calibration_error(
            _batched_probs(student_f_ext_v3, X_test_t), y_test
        )
        results['F_KD_from_CL_MLP_ext_tricks'] = m_f_ext_v3
        models['F_KD_from_CL_MLP_ext_tricks'] = student_f_ext_v3

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
# CELL 10b (v3.0): Architecture sweep — Workstream C (Pareto frontier)
# ============================================================================
# Trains KD-from-RF students at 5 sizes (XS/S/M/L/XL) for each seed.
# Output drives the extended Pareto plot in CELL 15 and answers
# "how much accuracy do we lose as we shrink the MLP?".
arch_sweep_results = {}
if USE_ARCH_SWEEP:
    print("\n" + "=" * 60)
    print(f"ARCHITECTURE SWEEP (Workstream C) — {len(SEEDS)} seeds × "
          f"{len(ARCH_SWEEP_VARIANTS)} variants")
    print("=" * 60)
    for seed in SEEDS:
        print(f"\n[Arch sweep seed {seed}]")
        arch_sweep_results[seed] = run_architecture_sweep(
            seed, X_train_np, y_train_np,
            X_val_np, y_val_np, X_test_np, y_test_np,
        )
        for cfg, m in arch_sweep_results[seed].items():
            telosb = '✓ fits' if m['fits_telosb_10kb'] else '✗ exceeds'
            print(f"    {cfg:10s} F1={m['macro_f1']:.4f}  "
                  f"acc={m['accuracy']:.4f}  "
                  f"params={m['params']:>6d}  "
                  f"int8={m['model_size_kb_int8_theoretical']:6.2f} KB  "
                  f"({telosb} TelosB)")

    # Aggregate across seeds for the Pareto plot
    print("\n--- ARCH SWEEP AGGREGATE ---")
    arch_sweep_aggregate = {}
    variant_names = [f'arch_{n}' for n, _ in ARCH_SWEEP_VARIANTS]
    for cfg in variant_names:
        f1s   = [arch_sweep_results[s][cfg]['macro_f1']  for s in SEEDS
                 if cfg in arch_sweep_results.get(s, {})]
        accs  = [arch_sweep_results[s][cfg]['accuracy']  for s in SEEDS
                 if cfg in arch_sweep_results.get(s, {})]
        params = arch_sweep_results[SEEDS[0]][cfg]['params']
        sz_int8 = arch_sweep_results[SEEDS[0]][cfg]['model_size_kb_int8_theoretical']
        sz_fp32 = arch_sweep_results[SEEDS[0]][cfg]['model_size_kb_fp32']
        fits   = arch_sweep_results[SEEDS[0]][cfg]['fits_telosb_10kb']
        if f1s:
            arch_sweep_aggregate[cfg] = {
                'macro_f1_mean': float(np.mean(f1s)),
                'macro_f1_std':  float(np.std(f1s)),
                'accuracy_mean': float(np.mean(accs)),
                'accuracy_std':  float(np.std(accs)),
                'params':        params,
                'size_kb_fp32':  sz_fp32,
                'size_kb_int8_theoretical': sz_int8,
                'fits_telosb_10kb': fits,
                'n_seeds':       len(f1s),
            }
            print(f"  {cfg:10s} F1={arch_sweep_aggregate[cfg]['macro_f1_mean']:.4f} "
                  f"± {arch_sweep_aggregate[cfg]['macro_f1_std']:.4f}  "
                  f"params={params:>6d}  int8={sz_int8:6.2f} KB  "
                  f"{'TelosB-fit' if fits else 'TelosB-exceed'}")
else:
    arch_sweep_aggregate = {}
    print("\n[v3] Architecture sweep DISABLED (USE_ARCH_SWEEP=False)")

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

# ============================================================================
# CELL 11b (v3.0): Enhanced Wilcoxon — bootstrap CIs + Holm-Bonferroni
# ============================================================================
# Adds two things over the standard wilcoxon_test in CELL 11:
#   1. Bootstrap 95% CI for the paired-difference effect size — addresses the
#      v2.3 issue that all p-values were > 0.05 with only 5 seeds.
#   2. Holm-Bonferroni correction across all v3 comparisons.
# Registers comparisons for the v3 configs (DKD, training tricks).
def bootstrap_paired_diff(seed_results: dict, cfg_a: str, cfg_b: str,
                           metric: str = 'macro_f1',
                           n_bootstrap: int = 5000,
                           seed: int = 42) -> dict:
    """Paired bootstrap of (cfg_a − cfg_b) for the chosen metric."""
    a_vals, b_vals = [], []
    for s, configs in seed_results.items():
        if cfg_a in configs and cfg_b in configs:
            ma = configs[cfg_a].get(metric)
            mb = configs[cfg_b].get(metric)
            if ma is not None and mb is not None:
                a_vals.append(ma); b_vals.append(mb)
    a_vals = np.asarray(a_vals); b_vals = np.asarray(b_vals)
    n = len(a_vals)
    if n < 3:
        return {'error': f'Need >=3 seeds, got {n}', 'n_seeds': n}
    diffs = a_vals - b_vals
    rng = np.random.RandomState(seed)
    boot_means = np.array([
        diffs[rng.randint(0, n, size=n)].mean() for _ in range(n_bootstrap)
    ])
    return {
        'cfg_a': cfg_a, 'cfg_b': cfg_b, 'metric': metric, 'n_seeds': n,
        'a_mean':       float(a_vals.mean()),
        'b_mean':       float(b_vals.mean()),
        'diff_mean':    float(diffs.mean()),
        'diff_std':     float(diffs.std(ddof=1)) if n > 1 else 0.0,
        'ci_95_low':    float(np.percentile(boot_means, 2.5)),
        'ci_95_high':   float(np.percentile(boot_means, 97.5)),
        'frac_positive': float((boot_means > 0).mean()),
    }


def holm_bonferroni(p_values: list, alpha: float = 0.05) -> list:
    """Holm-Bonferroni step-down correction.
    Returns list of (p_raw, p_adj, reject) in original input order."""
    n = len(p_values)
    if n == 0:
        return []
    order = list(np.argsort(p_values))
    p_sorted = np.asarray(p_values)[order]
    reject = [False] * n
    halt = False
    for i in range(n):
        threshold = alpha / (n - i)
        if not halt and p_sorted[i] < threshold:
            reject[order[i]] = True
        else:
            halt = True
    p_adj_sorted = np.minimum(1.0, p_sorted * (n - np.arange(n)))
    p_adj = [0.0] * n
    for i in range(n):
        p_adj[order[i]] = float(p_adj_sorted[i])
    return [(float(p_values[i]), p_adj[i], reject[i]) for i in range(n)]


# Pre-registered v3 comparisons. Each tuple = (cfg_a, cfg_b, alt, description).
# alt='greater'  → we hypothesize cfg_a > cfg_b
# alt='two-sided' → no prior direction
V3_COMPARISONS = [
    # ----- DKD lifts vanilla KD (Workstream D1) -----
    ('E_KD_from_RF_dkd',         'E_KD_from_RF',         'greater',
     'DKD lifts Config E (RF teacher)'),
    ('E2_KD_from_MLP_dkd',       'E2_KD_from_MLP',       'greater',
     'DKD lifts Config E2 (MLP teacher)'),
    ('F_KD_from_CL_MLP_ext_dkd', 'F_KD_from_CL_MLP_ext', 'greater',
     'DKD lifts Config F_ext (CL teacher)'),
    # ----- Training tricks lift baseline configs (Workstream F) -----
    ('B_Full_MLP_tricks',        'B_Full_MLP',           'greater',
     'Training tricks lift teacher MLP'),
    ('D_Small_MLP_tricks',       'D_Small_MLP',          'greater',
     'Training tricks lift scratch student'),
    ('E_KD_from_RF_tricks',      'E_KD_from_RF',         'greater',
     'Training tricks lift Config E'),
    ('F_KD_from_CL_MLP_ext_tricks', 'F_KD_from_CL_MLP_ext', 'greater',
     'Training tricks lift Config F_ext'),
    # ----- v2.3 questions, now at 10 seeds -----
    ('E_KD_from_RF',             'E2_KD_from_MLP',       'greater',
     'Tree teacher beats MLP teacher (10-seed)'),
    ('E_KD_from_RF',             'D_Small_MLP',          'greater',
     'KD beats scratch student (10-seed)'),
    ('C_CL_MLP_loss_ext',        'B_Full_MLP',           'two-sided',
     'CL helps teacher at extended budget (10-seed)'),
    ('F_KD_from_CL_MLP_ext',     'E_KD_from_RF',         'two-sided',
     'CL teacher KD vs RF teacher KD'),
]


def run_v3_wilcoxon_suite(seed_results: dict, label: str = '') -> dict:
    """Run all V3_COMPARISONS on `seed_results`, with bootstrap CI + Holm
    correction. Returns dict keyed by f'{cfg_a}_vs_{cfg_b}'."""
    out = {}
    p_values, keys_order = [], []
    for cfg_a, cfg_b, alt, desc in V3_COMPARISONS:
        a_vals, b_vals = [], []
        for s in seed_results:
            d = seed_results[s]
            if cfg_a in d and cfg_b in d:
                a_vals.append(d[cfg_a]['macro_f1'])
                b_vals.append(d[cfg_b]['macro_f1'])
        if len(a_vals) < 3:
            continue   # not enough seeds where both configs ran
        try:
            stat, p = wilcoxon(a_vals, b_vals, alternative=alt,
                                zero_method='wilcox')
        except Exception:
            stat, p = float('nan'), 1.0
        boot = bootstrap_paired_diff(seed_results, cfg_a, cfg_b)
        key = f'{cfg_a}_vs_{cfg_b}'
        out[key] = {
            'desc':                  desc,
            'a_config':              cfg_a,
            'b_config':              cfg_b,
            'alternative':           alt,
            'n_seeds':               len(a_vals),
            'a_macro_f1_mean':       float(np.mean(a_vals)),
            'b_macro_f1_mean':       float(np.mean(b_vals)),
            'diff_mean':             float(np.mean(a_vals) - np.mean(b_vals)),
            'wilcoxon_stat':         (float(stat) if not np.isnan(stat)
                                       else None),
            'wilcoxon_p_raw':        float(p),
            'bootstrap_ci_95':       [boot.get('ci_95_low'),
                                       boot.get('ci_95_high')],
            'bootstrap_frac_positive': boot.get('frac_positive'),
        }
        p_values.append(p); keys_order.append(key)

    if p_values:
        adjusted = holm_bonferroni(p_values, alpha=0.05)
        for key, (p_raw, p_adj, rej) in zip(keys_order, adjusted):
            out[key]['wilcoxon_p_holm_bonferroni'] = p_adj
            out[key]['significant_after_correction'] = rej
    return out


# ----- Run v3 Wilcoxon on Student A and Student B seed results -----
print("\n" + "=" * 70)
print(f"V3 ENHANCED WILCOXON SUITE — {len(SEEDS)} seeds, "
      "bootstrap CIs, Holm-Bonferroni")
print("=" * 70)

v3_wilcoxon_A = run_v3_wilcoxon_suite(all_seed_results_A, label='A')
print(f"\nStudent A — {len(v3_wilcoxon_A)} comparisons")
for key, r in v3_wilcoxon_A.items():
    sig = '✓ sig' if r.get('significant_after_correction') else '  not sig'
    cilo, cihi = r['bootstrap_ci_95']
    p_adj = r.get('wilcoxon_p_holm_bonferroni', float('nan'))
    print(f"  {key}")
    print(f"    {r['desc']}")
    print(f"    diff={r['diff_mean']:+.4f}  CI95=[{cilo:+.4f},{cihi:+.4f}]  "
          f"p_raw={r['wilcoxon_p_raw']:.4f}  p_adj={p_adj:.4f}  {sig}")

v3_wilcoxon_B = {}
if len(all_seed_results_B) > 0:
    v3_wilcoxon_B = run_v3_wilcoxon_suite(all_seed_results_B, label='B')
    print(f"\nStudent B — {len(v3_wilcoxon_B)} comparisons")
    for key, r in v3_wilcoxon_B.items():
        sig = '✓ sig' if r.get('significant_after_correction') else '  not sig'
        cilo, cihi = r['bootstrap_ci_95']
        p_adj = r.get('wilcoxon_p_holm_bonferroni', float('nan'))
        print(f"  {key}: diff={r['diff_mean']:+.4f}  "
              f"CI95=[{cilo:+.4f},{cihi:+.4f}]  p_adj={p_adj:.4f}  {sig}")

# Merge into the existing wilcoxon_results dict (CELL 11 built it)
wilcoxon_results['v3_student_A'] = v3_wilcoxon_A
wilcoxon_results['v3_student_B'] = v3_wilcoxon_B

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
    # ----- v3.0 additions (only included if their flags were True) -----
    'E_KD_from_RF_dkd',
    'E2_KD_from_MLP_dkd',
    'F_KD_from_CL_MLP_ext_dkd',
    'D_Small_MLP_tricks',
    'E_KD_from_RF_tricks',
    'F_KD_from_CL_MLP_ext_tricks',
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
# CELL 13b (v3.0): Quantization-Aware Training (QAT) for INT8 deployment
# ============================================================================
# Fine-tunes the trained fp32 students under fake-quant simulation, then
# converts to true int8. Reports macro-F1 delta vs fp32 — typically <1% drop
# vs the 2-12% drop of post-training dynamic quantization (see CELL 13).
qat_results = {}
if USE_QAT:
    print("\n" + "=" * 60)
    print("QUANTIZATION-AWARE TRAINING (QAT) — CELL 13b")
    print("=" * 60)

    # Recompute class weights for the QAT fine-tune CE loss (run_all_configs
    # computes this locally; we replicate it here at module scope).
    qat_counts = np.bincount(y_train_np, minlength=NUM_CLASSES)
    qat_class_weights = torch.tensor(
        len(y_train_np) / (NUM_CLASSES * np.maximum(qat_counts, 1)),
        dtype=torch.float32
    )

    # Tensorize splits (CPU — QAT is CPU-only here)
    X_train_qat = torch.tensor(X_train_np, dtype=torch.float32)
    y_train_qat = torch.tensor(y_train_np, dtype=torch.long)
    X_val_qat   = torch.tensor(X_val_np,   dtype=torch.float32)
    y_val_qat   = torch.tensor(y_val_np,   dtype=torch.long)
    X_test_qat  = torch.tensor(X_test_np,  dtype=torch.float32)
    y_test_qat  = torch.tensor(y_test_np,  dtype=torch.long)

    # Configs eligible for QAT (only student-architecture configs)
    QAT_TARGETS = [
        'E_KD_from_RF',
        'E_KD_from_RF_dkd',
        'E_KD_from_RF_tricks',
        'F_KD_from_CL_MLP_ext',
        'F_KD_from_CL_MLP_ext_dkd',
        'F_KD_from_CL_MLP_ext_tricks',
        'D_Small_MLP',
    ]

    for cfg in QAT_TARGETS:
        if cfg not in final_models:
            continue
        model_fp32 = final_models[cfg]
        # StudentMLP-shaped models have a Sequential `.net` whose first layer
        # is Linear. Skip anything else.
        if (not hasattr(model_fp32, 'net') or
            not isinstance(model_fp32.net, nn.Sequential) or
            not isinstance(model_fp32.net[0], nn.Linear)):
            continue

        # Derive hidden_dims from the trained model — robust to any architecture
        # (Student A 32-16, Student B 64-32, or future variants)
        hidden_dims_for_qat = derive_student_hidden_dims(model_fp32)
        print(f"\n[QAT] {cfg}  (derived hidden_dims={hidden_dims_for_qat})")
        try:
            int8_model = train_qat_from_pretrained(
                model_fp32,
                X_train_qat, y_train_qat,
                X_val_qat,   y_val_qat,
                hidden_dims=hidden_dims_for_qat,
                num_classes=NUM_CLASSES,
                class_weights=qat_class_weights,
                qat_epochs=QAT_FT_EPOCHS,
                lr=QAT_FT_LR,
                verbose=False,
            )
            int8_metrics = evaluate_int8_cpu(int8_model, X_test_qat, y_test_qat)
            theoretical_kb = model_size_int8_theoretical(int8_model)
            on_disk_kb     = model_size_on_disk_kb(int8_model)

            fp32_metrics = final_results.get(cfg, {})
            fp32_f1  = fp32_metrics.get('macro_f1')
            fp32_acc = fp32_metrics.get('accuracy')
            f1_delta = ((int8_metrics['macro_f1'] - fp32_f1) / fp32_f1 * 100
                        if fp32_f1 else None)

            qat_results[cfg] = {
                'qat_int8_accuracy':            int8_metrics['accuracy'],
                'qat_int8_macro_f1':            int8_metrics['macro_f1'],
                'qat_int8_per_class_f1':        int8_metrics['per_class_f1'],
                'qat_int8_size_kb_theoretical': theoretical_kb,
                'qat_int8_size_kb_on_disk':     on_disk_kb,
                'fp32_macro_f1':                fp32_f1,
                'fp32_accuracy':                fp32_acc,
                'f1_delta_pct':                 f1_delta,
            }
            if fp32_f1 is not None:
                print(f"  fp32 F1: {fp32_f1:.4f}  →  QAT int8 F1: "
                      f"{int8_metrics['macro_f1']:.4f}  (Δ {f1_delta:+.2f}%)")
            print(f"  Size: {theoretical_kb:.2f} KB theoretical / "
                  f"{on_disk_kb:.2f} KB on-disk")
        except Exception as ex:
            print(f"  QAT failed for {cfg}: {ex}")
            import traceback; traceback.print_exc()
            qat_results[cfg] = {'error': str(ex)}
else:
    print("\n[v3] QAT DISABLED (set USE_QAT=True in CELL 2 to enable).")

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

# v3.0 — overlay architecture-sweep points (XS/S/M/L/XL) as a connected curve
if 'arch_sweep_aggregate' in dir() and arch_sweep_aggregate:
    sweep_xs, sweep_ys, sweep_ye, sweep_lbls = [], [], [], []
    for name, _ in ARCH_SWEEP_VARIANTS:
        cfg_key = f'arch_{name}'
        a = arch_sweep_aggregate.get(cfg_key)
        if a is None:
            continue
        sweep_xs.append(a['size_kb_int8_theoretical'])
        sweep_ys.append(a['macro_f1_mean'])
        sweep_ye.append(a['macro_f1_std'])
        sweep_lbls.append(name)
    if sweep_xs:
        ax.errorbar(sweep_xs, sweep_ys, yerr=sweep_ye,
                    fmt='-D', color='red', linewidth=2, markersize=10,
                    capsize=3, alpha=0.85,
                    label='Architecture sweep (KD-from-RF, INT8 sizes)')
        for x, y, lbl in zip(sweep_xs, sweep_ys, sweep_lbls):
            ax.annotate(lbl, (x, y), xytext=(6, -10),
                        textcoords='offset points', fontsize=9,
                        color='red', fontweight='bold')
        # Shade the TelosB-deployable region
        ax.axvspan(0, 10.0, alpha=0.07, color='green',
                   label='TelosB 10 KB RAM budget')

ax.set_xscale('log')
ax.set_xlabel('Model size (KB) — log scale (INT8 theoretical for sweep, fp32 for run_all_configs)')
ax.set_ylabel('Macro F1 (test)')
ax.set_title('Model size vs. Macro F1 (Pareto frontier — Workstream C)')
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', fontsize=8)
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
# CELL 16 (v3.0): CICIoT2023 generalization — official UNB MERGED_CSV
# ============================================================================
# Multi-seed run with 8-class grouping, stratified cap-50K-per-class loader,
# and class-grounded CL ordering (literature-backed; see project memory).
# Configs: A_RF, B_MLP, C_lit, C_anti, C_random, D, E_KD_from_RF, E2, F.
# ============================================================================

# Literature-grounded class-difficulty tiers for CICIoT2023 — established by
# aggregating per-class F1 across Neto 2023, Susilo 2025, Fares 2025,
# Narayan 2023, Houichi 2025. These tiers drive the curriculum order.
CICIOT_CLASS_NAMES = ['Benign', 'DDoS', 'DoS', 'Mirai', 'Recon',
                      'Spoofing', 'Web', 'BruteForce']
CICIOT_NUM_CLASSES = 8
CICIOT_CL_STAGE_TIERS = {
    1: ['Mirai', 'DoS', 'DDoS'],            # F1 > 0.95 — easiest
    2: ['Benign', 'Recon', 'Spoofing'],     # F1 0.55-0.93
    3: ['BruteForce', 'Web'],               # F1 0.07-0.35 — hardest
}


def map_ciciot_label_to_8class(label: str):
    """Map fine-grained CICIoT2023 label (e.g. 'DDOS-ICMP_FLOOD') → 8-class
    group name (e.g. 'DDoS'). Returns None for unknown labels."""
    l = str(label).strip().upper()
    if l in ('BENIGN', 'BENIGN_FINAL', 'BENIGNTRAFFIC'):
        return 'Benign'
    if l.startswith('DDOS-'):
        return 'DDoS'
    if l.startswith('DOS-'):
        return 'DoS'
    if l.startswith('MIRAI-'):
        return 'Mirai'
    if l.startswith('RECON-') or l == 'VULNERABILITYSCAN':
        return 'Recon'
    if l in ('DNS_SPOOFING', 'MITM-ARPSPOOFING'):
        return 'Spoofing'
    if l in ('BACKDOOR_MALWARE', 'BROWSERHIJACKING', 'COMMANDINJECTION',
             'SQLINJECTION', 'UPLOADING_ATTACK', 'XSS'):
        return 'Web'
    if l == 'DICTIONARYBRUTEFORCE':
        return 'BruteForce'
    return None


def load_ciciot_balanced(merged_dir: str,
                          cap_per_class: int = 50_000,
                          chunk_size: int = 200_000,
                          verbose: bool = True):
    """Stream Merged*.csv files, accumulate up to cap_per_class rows per
    8-class group. Stops early once all groups are full.

    Returns (X, y_int, feature_names, group_counts).
    """
    import glob
    files = sorted(glob.glob(os.path.join(merged_dir, 'Merged*.csv')))
    if not files:
        raise FileNotFoundError(f"No Merged*.csv in {merged_dir}")

    cap = {g: cap_per_class for g in CICIOT_CLASS_NAMES}
    buckets = {g: [] for g in CICIOT_CLASS_NAMES}
    counts  = {g: 0 for g in CICIOT_CLASS_NAMES}
    feature_cols = None

    for fi, fpath in enumerate(files):
        if all(counts[g] >= cap[g] for g in CICIOT_CLASS_NAMES):
            if verbose:
                print(f"  All groups capped — stop at file {fi+1}/{len(files)}")
            break
        for chunk in pd.read_csv(fpath, chunksize=chunk_size):
            chunk.columns = chunk.columns.str.strip()
            if 'Label' not in chunk.columns:
                raise ValueError(f"'Label' column missing in {fpath}")
            if feature_cols is None:
                feature_cols = [c for c in chunk.columns if c != 'Label']
            chunk['_grp'] = chunk['Label'].map(map_ciciot_label_to_8class)
            chunk = chunk.dropna(subset=['_grp'])
            for g in CICIOT_CLASS_NAMES:
                if counts[g] >= cap[g]:
                    continue
                rows = chunk[chunk['_grp'] == g]
                need = cap[g] - counts[g]
                if len(rows) > need:
                    rows = rows.sample(need, random_state=42)
                if len(rows):
                    buckets[g].append(rows[feature_cols + ['_grp']])
                    counts[g] += len(rows)
        if verbose and (fi + 1) % 5 == 0:
            print(f"  After file {fi+1}/{len(files)}: {sum(counts.values()):,} "
                  f"rows; {counts}")

    parts = [pd.concat(buckets[g], ignore_index=True)
             for g in CICIOT_CLASS_NAMES if buckets[g]]
    if not parts:
        raise RuntimeError("No data loaded — check MERGED_CSV path / Label column")
    df_all = pd.concat(parts, ignore_index=True).sample(
        frac=1.0, random_state=42).reset_index(drop=True)

    name_to_idx = {n: i for i, n in enumerate(CICIOT_CLASS_NAMES)}
    y = df_all['_grp'].map(name_to_idx).values.astype(np.int64)
    X_df = df_all[feature_cols].select_dtypes(include=[np.number])
    feature_names = X_df.columns.tolist()
    X = X_df.values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    return X, y, feature_names, counts


def compute_engineered_features_ciciot(X: np.ndarray,
                                         feature_names: list,
                                         tier: int = 2):
    """Add CICIoT2023-specific engineered features (TCP flag ratios, packet
    stats interactions). CICIoT2023 has flow-level features unlike WSN-DS's
    LEACH telemetry, so the engineered features are different — but the same
    USE_ENGINEERED_FEATURES toggle controls both datasets.

    Returns (X_aug, feature_names_aug, new_feature_names).

    Tier 1 — TCP flag ratios (distinguish flood types: SYN, RST, FIN floods)
    Tier 2 — + packet-stat ratios (bytes/packet, IAT/AVG)
    Tier 3 — + protocol-mix features (TCP-vs-UDP fraction)
    """
    EPS = 1e-6
    name_to_idx = {n: i for i, n in enumerate(feature_names)}

    def col(name):
        i = name_to_idx.get(name)
        if i is None:
            return None
        return X[:, i]

    new_cols = []      # list of (name, np.ndarray)

    # ---- Tier 1: TCP flag ratios — useful for DDoS-SYN_Flood, RST/FIN floods
    number = col('Number')
    if number is not None:
        for flag in ('syn_count', 'rst_count', 'fin_count', 'ack_count'):
            c = col(flag)
            if c is not None:
                new_cols.append((f'{flag}_ratio', c / np.maximum(number, EPS)))

    # ---- Tier 2: Packet statistics interactions
    if tier >= 2:
        tot = col('Tot size')
        if tot is not None and number is not None:
            new_cols.append(('bytes_per_packet',
                             tot / np.maximum(number, EPS)))
        rate = col('Rate')
        if rate is not None and number is not None:
            new_cols.append(('rate_per_packet',
                             rate / np.maximum(number, EPS)))
        var = col('Variance')
        avg = col('AVG')
        if var is not None and avg is not None:
            new_cols.append(('size_variance_ratio',
                             var / np.maximum(np.abs(avg), EPS)))
        iat = col('IAT')
        if iat is not None and avg is not None:
            new_cols.append(('iat_normalized',
                             iat / np.maximum(np.abs(avg), EPS)))

    # ---- Tier 3: Protocol mix
    if tier >= 3:
        tcp = col('TCP'); udp = col('UDP')
        if tcp is not None and udp is not None:
            new_cols.append(('tcp_udp_fraction',
                             tcp / np.maximum(tcp + udp, EPS)))
        icmp = col('ICMP')
        if icmp is not None and number is not None:
            new_cols.append(('icmp_density',
                             icmp / np.maximum(number, EPS)))

    if not new_cols:
        return X, list(feature_names), []

    new_arr = np.stack([c for _, c in new_cols], axis=1).astype(np.float32)
    new_arr = np.nan_to_num(new_arr, nan=0.0, posinf=1e6, neginf=-1e6)
    X_aug = np.concatenate([X, new_arr], axis=1)
    new_names = [n for n, _ in new_cols]
    return X_aug, list(feature_names) + new_names, new_names


def compute_difficulty_class_lit_ciciot(y_train: np.ndarray,
                                         reverse: bool = False) -> np.ndarray:
    """Literature-grounded class-level CL ordering for CICIoT2023.
    reverse=True flips for anti-curriculum control."""
    name_to_tier = {}
    for tier, names in CICIOT_CL_STAGE_TIERS.items():
        for n in names:
            name_to_tier[n] = tier
    sample_tier = np.array([
        name_to_tier.get(CICIOT_CLASS_NAMES[int(lbl)], 2)
        for lbl in y_train
    ])
    if reverse:
        sample_tier = 4 - sample_tier
    rng = np.random.RandomState(42)
    return np.lexsort((rng.rand(len(sample_tier)), sample_tier))


def run_ciciot_one_seed(seed: int,
                         X_train_t, y_train_t, X_val_t, y_val_t,
                         X_test_t, y_test_t,
                         class_weights_c, input_dim_c: int) -> dict:
    """Mirror of run_all_configs() for CICIoT2023, 8-class, transfer-teacher CL."""
    set_seed(seed)
    res = {}

    # ----- A_RF: 500-tree RF with isotonic calibration -----
    print(f"  [A_RF] RF (500 trees) + isotonic calibration ...")
    rf_calib_c = CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=500, max_depth=15,
                                random_state=seed, n_jobs=-1),
        method='isotonic', cv=3,
    )
    rf_calib_c.fit(X_train_t.numpy(), y_train_t.numpy())
    rf_pred_c = rf_calib_c.predict(X_test_t.numpy())
    rf_proba_train_c = torch.tensor(
        rf_calib_c.predict_proba(X_train_t.numpy()), dtype=torch.float32)
    res['A_RF_500'] = {
        'accuracy': float((rf_pred_c == y_test_t.numpy()).mean()),
        'macro_f1': float(f1_score(y_test_t.numpy(), rf_pred_c, average='macro')),
        'per_class_f1': f1_score(y_test_t.numpy(), rf_pred_c,
                                  average=None).tolist(),
    }

    # ----- B_Full_MLP -----
    print(f"  [B] Full MLP teacher ...")
    t_b_c = TeacherMLP(input_dim_c, CICIOT_NUM_CLASSES)
    t_b_c = train_standard(t_b_c, X_train_t, y_train_t, X_val_t, y_val_t,
                            class_weights=class_weights_c, **TRAIN_CONFIG)
    res['B_Full_MLP'] = evaluate_model(t_b_c, X_test_t, y_test_t)

    # ----- C_lit, C_anti, C_random (CL teacher variants) -----
    for tag, kwargs in (
        ('C_CL_MLP_lit',   {'reverse': False}),
        ('C_CL_MLP_anti',  {'reverse': True}),
    ):
        print(f"  [{tag}] CL teacher ...")
        order_tier = compute_difficulty_class_lit_ciciot(
            y_train_t.numpy(), **kwargs)
        t_c_v = TeacherMLP(input_dim_c, CICIOT_NUM_CLASSES)
        t_c_v = train_with_curriculum(
            t_c_v, X_train_t, y_train_t, order_tier, X_val_t, y_val_t,
            stages=CL_STAGES_EXT, class_weights=class_weights_c)
        res[tag] = evaluate_model(t_c_v, X_test_t, y_test_t)
        if tag == 'C_CL_MLP_lit':
            t_c_lit_c = t_c_v

    print(f"  [C_random] random-order CL teacher ...")
    rng_r = np.random.RandomState(seed)
    rand_order = rng_r.permutation(len(y_train_t))
    t_c_rand_c = TeacherMLP(input_dim_c, CICIOT_NUM_CLASSES)
    t_c_rand_c = train_with_curriculum(
        t_c_rand_c, X_train_t, y_train_t, rand_order, X_val_t, y_val_t,
        stages=CL_STAGES_EXT, class_weights=class_weights_c)
    res['C_CL_MLP_random'] = evaluate_model(t_c_rand_c, X_test_t, y_test_t)

    # ----- D: small MLP from scratch -----
    print(f"  [D] Student scratch ({STUDENT_A_HIDDEN}) ...")
    s_d_c = StudentMLP(input_dim_c, STUDENT_A_HIDDEN, CICIOT_NUM_CLASSES)
    s_d_c = train_standard(s_d_c, X_train_t, y_train_t, X_val_t, y_val_t,
                            class_weights=class_weights_c, **TRAIN_CONFIG)
    res['D_Small_MLP'] = evaluate_model(s_d_c, X_test_t, y_test_t)

    # ----- E: KD from calibrated RF (THE key tree-to-neural config) -----
    print(f"  [E] KD from calibrated RF ...")
    s_e_c = StudentMLP(input_dim_c, STUDENT_A_HIDDEN, CICIOT_NUM_CLASSES)
    s_e_c = train_kd(s_e_c, rf_proba_train_c,
                      X_train_t, y_train_t, X_val_t, y_val_t,
                      T=BEST_T, alpha=BEST_ALPHA,
                      class_weights=class_weights_c,
                      epochs=TRAIN_CONFIG['epochs'],
                      batch_size=TRAIN_CONFIG['batch_size'])
    res['E_KD_from_RF'] = evaluate_model(s_e_c, X_test_t, y_test_t)

    # ----- E2: KD from MLP teacher -----
    print(f"  [E2] KD from standard MLP teacher ...")
    s_e2_c = StudentMLP(input_dim_c, STUDENT_A_HIDDEN, CICIOT_NUM_CLASSES)
    s_e2_c = train_kd(s_e2_c, t_b_c,
                       X_train_t, y_train_t, X_val_t, y_val_t,
                       T=BEST_T, alpha=BEST_ALPHA,
                       class_weights=class_weights_c,
                       epochs=TRAIN_CONFIG['epochs'],
                       batch_size=TRAIN_CONFIG['batch_size'])
    res['E2_KD_from_MLP'] = evaluate_model(s_e2_c, X_test_t, y_test_t)

    # ----- F: KD from lit-grounded CL teacher (CORE) -----
    print(f"  [F_lit] KD from lit-grounded CL-MLP ...")
    s_f_c = StudentMLP(input_dim_c, STUDENT_A_HIDDEN, CICIOT_NUM_CLASSES)
    s_f_c = train_kd(s_f_c, t_c_lit_c,
                      X_train_t, y_train_t, X_val_t, y_val_t,
                      T=BEST_T, alpha=BEST_ALPHA,
                      class_weights=class_weights_c,
                      epochs=TRAIN_CONFIG['epochs'],
                      batch_size=TRAIN_CONFIG['batch_size'])
    res['F_KD_from_CL_MLP_lit'] = evaluate_model(s_f_c, X_test_t, y_test_t)

    return res


# -----------------------------------------------------------------------------
# Main CICIoT2023 driver
# -----------------------------------------------------------------------------
ciciot_results = {}
if RUN_CICIOT and os.path.isdir(CICIOT_MERGED_DIR):
    print("\n" + "=" * 70)
    print("CICIoT2023 GENERALIZABILITY EXPERIMENT (v3)")
    print("=" * 70)
    try:
        print(f"[Loader] Streaming MERGED_CSV from {CICIOT_MERGED_DIR} ...")
        X_c, y_c, feat_names_c, group_counts_c = load_ciciot_balanced(
            CICIOT_MERGED_DIR, cap_per_class=CICIOT_CAP_PER_CLASS, verbose=True)
        print(f"  Loaded: {X_c.shape[0]:,} rows, {X_c.shape[1]} features")
        print(f"  Per-class: {group_counts_c}")

        # v3.0 — apply CICIoT-specific engineered features when toggle is on
        if USE_ENGINEERED_FEATURES:
            X_c, feat_names_c, new_eng = compute_engineered_features_ciciot(
                X_c, feat_names_c, tier=ENGINEERED_FEATURE_TIER)
            print(f"  CICIoT engineered features ({len(new_eng)} added): {new_eng}")
            print(f"  New feature dim: {X_c.shape[1]}")

        scaler_c = StandardScaler()
        X_c = scaler_c.fit_transform(X_c)

        X_ctr_full, X_cte_np, y_ctr_full, y_cte_np = train_test_split(
            X_c, y_c, test_size=0.15, random_state=42, stratify=y_c)
        X_ctr_np, X_cv_np, y_ctr_np, y_cv_np = train_test_split(
            X_ctr_full, y_ctr_full, test_size=0.1765,
            random_state=42, stratify=y_ctr_full)
        INPUT_DIM_C = X_ctr_np.shape[1]
        print(f"  Train: {X_ctr_np.shape}, Val: {X_cv_np.shape}, Test: {X_cte_np.shape}")

        Xtr_t_c = torch.tensor(X_ctr_np, dtype=torch.float32)
        ytr_t_c = torch.tensor(y_ctr_np, dtype=torch.long)
        Xv_t_c  = torch.tensor(X_cv_np,  dtype=torch.float32)
        yv_t_c  = torch.tensor(y_cv_np,  dtype=torch.long)
        Xte_t_c = torch.tensor(X_cte_np, dtype=torch.float32)
        yte_t_c = torch.tensor(y_cte_np, dtype=torch.long)
        cw_c = torch.tensor(
            len(y_ctr_np) / (CICIOT_NUM_CLASSES *
                              np.maximum(np.bincount(
                                  y_ctr_np, minlength=CICIOT_NUM_CLASSES), 1)),
            dtype=torch.float32)

        # ----- Multi-seed loop -----
        ciciot_per_seed = {}
        for seed in SEEDS:
            print(f"\n[CICIoT seed {seed}]")
            ciciot_per_seed[str(seed)] = run_ciciot_one_seed(
                seed, Xtr_t_c, ytr_t_c, Xv_t_c, yv_t_c, Xte_t_c, yte_t_c,
                cw_c, INPUT_DIM_C)

        # ----- Aggregate -----
        ciciot_agg = {}
        cfgs_seen = list(next(iter(ciciot_per_seed.values())).keys())
        for cfg in cfgs_seen:
            f1s  = [ciciot_per_seed[s][cfg]['macro_f1']  for s in ciciot_per_seed
                    if cfg in ciciot_per_seed[s]]
            accs = [ciciot_per_seed[s][cfg]['accuracy']  for s in ciciot_per_seed
                    if cfg in ciciot_per_seed[s]]
            ciciot_agg[cfg] = {
                'macro_f1_mean': float(np.mean(f1s)),
                'macro_f1_std':  float(np.std(f1s)),
                'accuracy_mean': float(np.mean(accs)),
                'accuracy_std':  float(np.std(accs)),
                'n_seeds': len(f1s),
            }

        ciciot_results = {
            'per_seed':       ciciot_per_seed,
            'aggregate':      ciciot_agg,
            'class_names':    CICIOT_CLASS_NAMES,
            'feature_names':  feat_names_c,
            'group_counts':   group_counts_c,
            'cl_stage_tiers': CICIOT_CL_STAGE_TIERS,
        }

        print("\n" + "=" * 70)
        print(f"CICIoT2023 AGGREGATE ({len(SEEDS)} seeds)")
        print("=" * 70)
        for cfg in cfgs_seen:
            a = ciciot_agg[cfg]
            print(f"  {cfg:30s} F1={a['macro_f1_mean']:.4f} ± "
                  f"{a['macro_f1_std']:.4f}  acc={a['accuracy_mean']:.4f}")

    except Exception as ex:
        print(f"CICIoT2023 pipeline failed: {ex}")
        import traceback; traceback.print_exc()
        ciciot_results = {}
else:
    if RUN_CICIOT:
        print(f"CICIoT2023 dir not found at {CICIOT_MERGED_DIR} — skipping")
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
    # v3.0 — also save Student B aggregate to JSON (was previously CSV-only)
    'aggregate_student_B': (aggregate_multi_seed(all_seed_results_B).to_dict('records')
                            if len(all_seed_results_B) > 0 else []),
    'kd_hyperparameters': {'T': BEST_T, 'alpha': BEST_ALPHA},
    'shap_results': shap_results,
    'quantization': quant_results,
    # v3.0 — QAT results from CELL 13b
    'qat_results': qat_results if 'qat_results' in dir() else {},
    # v3.0 — Workstream C: architecture sweep
    'arch_sweep_results':   arch_sweep_results   if 'arch_sweep_results'   in dir() else {},
    'arch_sweep_aggregate': arch_sweep_aggregate if 'arch_sweep_aggregate' in dir() else {},
    'inference_benchmarks': bench_results,
    'ciciot_results': ciciot_results,
    'wilcoxon_results': wilcoxon_results if 'wilcoxon_results' in dir() else {},
    # v3.0 — capture which v3 toggles were active for this run
    'v3_flags': {
        'USE_ENGINEERED_FEATURES': USE_ENGINEERED_FEATURES,
        'ENGINEERED_FEATURE_TIER': ENGINEERED_FEATURE_TIER,
        'USE_DKD':                  USE_DKD,
        'DKD_ALPHA':                DKD_ALPHA,
        'DKD_BETA':                 DKD_BETA,
        'USE_TRAINING_TRICKS':      USE_TRAINING_TRICKS,
        'TRICK_LABEL_SMOOTHING':    TRICK_LABEL_SMOOTHING,
        'TRICK_EMA_DECAY':          TRICK_EMA_DECAY,
        'TRICK_GRAD_CLIP':          TRICK_GRAD_CLIP,
        'USE_QAT':                  USE_QAT,
        'QAT_FT_EPOCHS':            QAT_FT_EPOCHS,
        'QAT_FT_LR':                QAT_FT_LR,
        'RUN_CICIOT':               RUN_CICIOT,
        'CICIOT_MERGED_DIR':        CICIOT_MERGED_DIR,
        'CICIOT_CAP_PER_CLASS':     CICIOT_CAP_PER_CLASS,
        'RUN_FULL_10_SEEDS':        RUN_FULL_10_SEEDS,
        'USE_ARCH_SWEEP':           USE_ARCH_SWEEP,
        'ARCH_SWEEP_VARIANTS':      [list(v) for v in ARCH_SWEEP_VARIANTS],
    },
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
