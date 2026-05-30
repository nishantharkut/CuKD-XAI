# ============================================================================
#  !!!! WARNING - READ BEFORE EDITING !!!!
#
#  This .py file is the Python source for the matching notebook. Make source
#  edits here, then regenerate the notebook with make_notebook_preserve_banner.py.
#
#  Edge-IIoT v2.3 generalization route:
#      python3 make_notebook_preserve_banner.py cukd_xai_edgeiiot_v23_generalization.py cukd_xai_edgeiiot_v23_generalization.ipynb
# ============================================================================

# ============================================================================
# CuKD-XAI Edge-IIoTset v2.3 Generalization Route
#
# Purpose:
# - Run the same v2.3 model/training/KD logic on Edge-IIoTset as secondary
#   dataset evidence.
# - Keep the Edge-IIoT changes isolated to a train-split-safe dataset adapter,
#   output aggregation, and run orchestration.
# - Exclude WSN-only explainability, statistical-test, and deployment extras
#   from this secondary-dataset generalization run.
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
import copy
import json
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              f1_score, classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

EDGEIIOT_ML_FILENAME = 'ML-EdgeIIoT-dataset.csv'
EDGEIIOT_ML_PATH = os.environ.get('EDGEIIOT_ML_PATH', EDGEIIOT_ML_FILENAME)
EDGEIIOT_TARGET_COL = 'Attack_type'
EDGEIIOT_OUTPUT_DIR = 'edgeiiot_v23_generalization_outputs'
EDGEIIOT_GUIDE_FILE = 'EDGEIIOT_V23_GENERALIZATION_GUIDE.md'

EDGEIIOT_SEEDS_FINAL5 = [42, 123, 456, 789, 1001]
EDGEIIOT_SEEDS_QUICK1 = [42]
EDGEIIOT_RUN_MODE = 'edgeiiot_final'  # 'edgeiiot_final' or 'edgeiiot_quick'
EDGEIIOT_SEEDS = EDGEIIOT_SEEDS_FINAL5 if EDGEIIOT_RUN_MODE == 'edgeiiot_final' else EDGEIIOT_SEEDS_QUICK1

EDGEIIOT_RF_TREES = 500
EDGEIIOT_RF_MAX_DEPTH = 15
EDGEIIOT_RF_CALIBRATION_METHOD = 'isotonic'
EDGEIIOT_RF_CALIBRATION_CV = 3

EDGEIIOT_MAX_CATEGORICAL_CARDINALITY = 64
EDGEIIOT_RARE_CATEGORY_MIN_COUNT = 10

KD_T_GRID = [2, 3, 4, 5]
KD_ALPHA_GRID = [0.5, 0.7, 0.9]
KD_T_DEFAULT = 4
KD_ALPHA_DEFAULT = 0.7
# Edge generalization uses the KD setting selected by the completed v2.3 WSN-DS
# non-quick runs, so the secondary dataset is not separately tuned.
KD_T_EDGE = 2
KD_ALPHA_EDGE = 0.5

CODISTILL_CE_WEIGHT = 0.30
CODISTILL_RF_WEIGHT = 0.40
CODISTILL_CL_WEIGHT = 0.30
CODISTILL_EPOCHS = 40
CODISTILL_LR = 7e-4
CODISTILL_PATIENCE = 10

TRAIN_CONFIG = {
    'epochs': 30,
    'batch_size': 256,
    'lr': 1e-3,
    'weight_decay': 1e-3,
    'patience': 8,
}

CL_STAGES_FAIR = [(0.33, 3), (0.66, 3), (1.0, 24)]
CL_STAGES_EXT  = [(0.33, 5), (0.66, 5), (1.0, 30)]
CL_STAGES = CL_STAGES_FAIR

STUDENT_A_HIDDEN = (32, 16)
STUDENT_B_HIDDEN = (64, 32)
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
    'class', 'Class', 'attack_type', 'Attack Type',
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = DEVICE

print('CuKD-XAI Edge-IIoTset v2.3 generalization route')
print(f'CSV path: {EDGEIIOT_ML_PATH}')
print(f'Target: {EDGEIIOT_TARGET_COL}')
print(f'Run mode: {EDGEIIOT_RUN_MODE}')
print(f'Seeds: {EDGEIIOT_SEEDS}')
print(f'Output dir: {EDGEIIOT_OUTPUT_DIR}')
print(f'Device: {device}')

# ============================================================================
# CELL 3: Edge-IIoT adapter
# ============================================================================
def _edge_json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _unique_paths(paths: list) -> list:
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


def resolve_edgeiiot_ml_path(csv_path: str = EDGEIIOT_ML_PATH) -> Path:
    configured = Path(os.path.expanduser(str(csv_path)))
    candidates = []
    if configured.is_absolute():
        candidates.append(configured)
    else:
        candidates.append(Path.cwd() / configured)
        try:
            candidates.append(Path(__file__).resolve().parent / configured)
        except NameError:
            pass

    relative_paths = [
        Path(EDGEIIOT_ML_FILENAME),
        Path('Selected dataset for ML and DL') / EDGEIIOT_ML_FILENAME,
        Path('Edge-IIoTset dataset') / 'Selected dataset for ML and DL' / EDGEIIOT_ML_FILENAME,
        Path('edgeiiot') / EDGEIIOT_ML_FILENAME,
        Path('edgeiiot') / 'extracted' / 'Edge-IIoTset dataset' / 'Selected dataset for ML and DL' / EDGEIIOT_ML_FILENAME,
        Path('datasets') / 'edgeiiot' / 'extracted' / 'Edge-IIoTset dataset' / 'Selected dataset for ML and DL' / EDGEIIOT_ML_FILENAME,
        Path('.cukd_xai_secret') / 'datasets' / 'edgeiiot' / 'extracted' / 'Edge-IIoTset dataset' / 'Selected dataset for ML and DL' / EDGEIIOT_ML_FILENAME,
    ]
    bases = [Path.cwd()]
    try:
        bases.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    for base in list(bases):
        bases.extend(list(base.parents)[:5])
    for base in _unique_paths(bases):
        for rel_path in relative_paths:
            candidates.append(base / rel_path)

    for candidate in _unique_paths(candidates):
        if candidate.exists() and candidate.is_file():
            return candidate

    searched_preview = '\n'.join(str(path) for path in _unique_paths(candidates)[:12])
    raise FileNotFoundError(
        'Edge-IIoT ML CSV not found. Put ML-EdgeIIoT-dataset.csv beside the notebook, '
        'keep the original extracted folder structure, or set EDGEIIOT_ML_PATH to the CSV path. '
        f'First searched paths:\n{searched_preview}'
    )


def write_edge_json(path: str, payload: dict) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(json.dumps(payload, indent=2, default=_edge_json_default), encoding='utf-8')


def edgeiiot_normalize_categorical(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().replace({'': '__MISSING__', 'nan': '__MISSING__', 'None': '__MISSING__'})


def fit_edgeiiot_category_policy(X_train_raw: pd.DataFrame, categorical_cols: list) -> dict:
    policy = {}
    for col in categorical_cols:
        counts = edgeiiot_normalize_categorical(X_train_raw[col]).value_counts(dropna=False)
        kept = counts[counts >= EDGEIIOT_RARE_CATEGORY_MIN_COUNT].index.tolist()
        kept = kept[:EDGEIIOT_MAX_CATEGORICAL_CARDINALITY]
        policy[col] = set(kept)
    return policy


def apply_edgeiiot_category_policy(X_raw: pd.DataFrame, categorical_cols: list, category_policy: dict) -> pd.DataFrame:
    X_cat = pd.DataFrame(index=X_raw.index)
    for col in categorical_cols:
        values = edgeiiot_normalize_categorical(X_raw[col])
        kept = category_policy.get(col, set())
        X_cat[col] = values.where(values.isin(kept), '__OTHER__')
    return X_cat


def edgeiiot_encode_split(X_raw: pd.DataFrame, categorical_cols: list, category_policy: dict) -> pd.DataFrame:
    if not categorical_cols:
        return pd.DataFrame(index=X_raw.index)
    X_cat = apply_edgeiiot_category_policy(X_raw, categorical_cols, category_policy)
    return pd.get_dummies(X_cat, columns=categorical_cols, dtype=np.float32)


def _numeric_column_report(df: pd.DataFrame, exclude_cols: set) -> tuple:
    numeric_cols = []
    categorical_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
        else:
            coerced = pd.to_numeric(series, errors='coerce')
            valid_ratio = float(coerced.notna().mean())
            if valid_ratio >= 0.98:
                df[col] = coerced
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
    return numeric_cols, categorical_cols


def prepare_edgeiiot_v23_arrays(csv_path: str = EDGEIIOT_ML_PATH,
                                target_col: str = EDGEIIOT_TARGET_COL) -> dict:
    csv_obj = resolve_edgeiiot_ml_path(csv_path)

    df = pd.read_csv(csv_obj, low_memory=False)
    df.columns = df.columns.str.strip()
    if target_col not in df.columns:
        raise ValueError(f'Target column {target_col!r} not found. Columns: {df.columns.tolist()}')

    df[target_col] = df[target_col].astype(str).str.strip()
    invalid_target_mask = df[target_col].isin(['', 'nan', 'None'])
    invalid_target_rows = int(invalid_target_mask.sum())
    if invalid_target_rows:
        df = df.loc[~invalid_target_mask].copy()

    drop_candidates = list(dict.fromkeys(EDGEIIOT_LEAKAGE_COLUMNS + EDGEIIOT_AUXILIARY_TARGET_COLUMNS))
    removed_leakage_columns = [col for col in drop_candidates if col in df.columns and col != target_col]
    if removed_leakage_columns:
        df = df.drop(columns=removed_leakage_columns)

    y_raw = df[target_col].astype(str).str.strip()
    X_df = df.drop(columns=[target_col]).copy()

    all_missing = [col for col in X_df.columns if X_df[col].isna().all()]
    if all_missing:
        X_df = X_df.drop(columns=all_missing)

    numeric_cols, categorical_cols = _numeric_column_report(X_df, exclude_cols=set())
    for col in numeric_cols:
        X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
    for col in categorical_cols:
        X_df[col] = edgeiiot_normalize_categorical(X_df[col])

    constant_cols = [col for col in X_df.columns if X_df[col].nunique(dropna=False) <= 1]
    if constant_cols:
        X_df = X_df.drop(columns=constant_cols)
        numeric_cols = [col for col in numeric_cols if col not in constant_cols]
        categorical_cols = [col for col in categorical_cols if col not in constant_cols]

    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(y_raw).astype(np.int64)
    class_names = label_encoder.classes_.tolist()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_df, y_all, test_size=0.15, random_state=42, stratify=y_all
    )
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_raw, y_train, test_size=0.1765, random_state=42, stratify=y_train
    )

    if numeric_cols:
        numeric_medians = X_train_raw[numeric_cols].median(numeric_only=True).fillna(0.0)
        X_train_raw.loc[:, numeric_cols] = X_train_raw[numeric_cols].fillna(numeric_medians).astype(np.float32)
        X_val_raw.loc[:, numeric_cols] = X_val_raw[numeric_cols].fillna(numeric_medians).astype(np.float32)
        X_test_raw.loc[:, numeric_cols] = X_test_raw[numeric_cols].fillna(numeric_medians).astype(np.float32)
    else:
        numeric_medians = pd.Series(dtype=np.float32)

    category_policy = fit_edgeiiot_category_policy(X_train_raw, categorical_cols)
    X_train_cat = edgeiiot_encode_split(X_train_raw, categorical_cols, category_policy)
    X_val_cat = edgeiiot_encode_split(X_val_raw, categorical_cols, category_policy)
    X_test_cat = edgeiiot_encode_split(X_test_raw, categorical_cols, category_policy)

    dummy_columns = X_train_cat.columns.tolist()
    X_val_cat = X_val_cat.reindex(columns=dummy_columns, fill_value=0.0)
    X_test_cat = X_test_cat.reindex(columns=dummy_columns, fill_value=0.0)

    X_train_num = X_train_raw[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=X_train_raw.index)
    X_val_num = X_val_raw[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=X_val_raw.index)
    X_test_num = X_test_raw[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=X_test_raw.index)

    X_train_encoded = pd.concat([X_train_num.reset_index(drop=True), X_train_cat.reset_index(drop=True)], axis=1)
    X_val_encoded = pd.concat([X_val_num.reset_index(drop=True), X_val_cat.reset_index(drop=True)], axis=1)
    X_test_encoded = pd.concat([X_test_num.reset_index(drop=True), X_test_cat.reset_index(drop=True)], axis=1)

    train_constant_encoded_cols = [col for col in X_train_encoded.columns if X_train_encoded[col].nunique(dropna=False) <= 1]
    if train_constant_encoded_cols:
        X_train_encoded = X_train_encoded.drop(columns=train_constant_encoded_cols)
        X_val_encoded = X_val_encoded.drop(columns=train_constant_encoded_cols, errors='ignore')
        X_test_encoded = X_test_encoded.drop(columns=train_constant_encoded_cols, errors='ignore')
        numeric_cols = [col for col in numeric_cols if col not in train_constant_encoded_cols]

    feature_names = X_train_encoded.columns.tolist()
    continuous_cols = [col for col in numeric_cols if col in X_train_encoded.columns]
    scaler = StandardScaler()
    if continuous_cols:
        X_train_encoded.loc[:, continuous_cols] = scaler.fit_transform(X_train_encoded.loc[:, continuous_cols])
        X_val_encoded.loc[:, continuous_cols] = scaler.transform(X_val_encoded.loc[:, continuous_cols])
        X_test_encoded.loc[:, continuous_cols] = scaler.transform(X_test_encoded.loc[:, continuous_cols])

    X_train_np = X_train_encoded.astype(np.float32).to_numpy()
    X_val_np = X_val_encoded.astype(np.float32).to_numpy()
    X_test_np = X_test_encoded.astype(np.float32).to_numpy()

    metadata = {
        'csv_path': str(csv_obj),
        'raw_shape_after_target_cleaning': [int(df.shape[0]), int(df.shape[1])],
        'invalid_target_rows_removed': invalid_target_rows,
        'removed_leakage_columns': removed_leakage_columns,
        'removed_all_missing_columns': all_missing,
        'removed_constant_columns': constant_cols,
        'removed_train_constant_encoded_columns': train_constant_encoded_cols,
        'categorical_cols': categorical_cols,
        'continuous_cols': continuous_cols,
        'numeric_medians': {str(key): float(value) for key, value in numeric_medians.to_dict().items()},
        'dummy_feature_count': len(dummy_columns),
        'input_dim': int(X_train_np.shape[1]),
        'num_classes': int(len(class_names)),
        'class_names': class_names,
        'class_distribution_all': dict(zip(class_names, np.bincount(y_all, minlength=len(class_names)).astype(int).tolist())),
        'class_distribution_train': dict(zip(class_names, np.bincount(y_train, minlength=len(class_names)).astype(int).tolist())),
        'class_distribution_val': dict(zip(class_names, np.bincount(y_val, minlength=len(class_names)).astype(int).tolist())),
        'class_distribution_test': dict(zip(class_names, np.bincount(y_test, minlength=len(class_names)).astype(int).tolist())),
        'feature_names': feature_names,
        'split_policy': '70/15/15 stratified using random_state=42, matching v2.3 fixed split intent',
        'category_policy': {col: sorted(list(values)) for col, values in category_policy.items()},
    }

    return {
        'X_train_np': X_train_np,
        'X_val_np': X_val_np,
        'X_test_np': X_test_np,
        'y_train_np': y_train.astype(np.int64),
        'y_val_np': y_val.astype(np.int64),
        'y_test_np': y_test.astype(np.int64),
        'class_names': class_names,
        'feature_names': feature_names,
        'input_dim': int(X_train_np.shape[1]),
        'num_classes': int(len(class_names)),
        'metadata': metadata,
        'label_encoder': label_encoder,
        'scaler': scaler,
    }

# ============================================================================
# CELL 4: Preserved v2.3 model, metric, training, KD, and quantization definitions
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
    return np.argsort(per_sample_loss)

# ============================================================================
# CELL 5: Edge-IIoT v2.3 run helpers
# ============================================================================
def class_weights_from_labels(y_np: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y_np, minlength=int(num_classes)).astype(np.float64)
    weights = len(y_np) / (int(num_classes) * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def rf_predict_proba_aligned(model, X_np: np.ndarray, num_classes: int) -> np.ndarray:
    probs = model.predict_proba(X_np)
    if isinstance(probs, list):
        probs = np.vstack([p[:, 1] for p in probs]).T
    probs = np.asarray(probs, dtype=np.float32)
    aligned = np.zeros((X_np.shape[0], int(num_classes)), dtype=np.float32)
    classes = getattr(model, 'classes_', np.arange(num_classes))
    for idx, cls in enumerate(classes):
        aligned[:, int(cls)] = probs[:, idx]
    row_sums = aligned.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    return aligned / row_sums


def sklearn_metrics_from_predictions(y_true: np.ndarray,
                                      y_pred: np.ndarray,
                                      probs: np.ndarray,
                                      class_names: list) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    per_precision, per_recall, per_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_precision': float(precision),
        'macro_recall': float(recall),
        'macro_f1': float(f1),
        'per_class_precision': per_precision.tolist(),
        'per_class_recall': per_recall.tolist(),
        'per_class_f1': per_f1.tolist(),
        'ece': float(expected_calibration_error(probs, y_true)) if probs is not None else None,
        'class_names': class_names,
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }


def torch_metrics_with_resources(model: nn.Module,
                                 X_test_t: torch.Tensor,
                                 y_test_t: torch.Tensor,
                                 class_names: list,
                                 input_dim: int,
                                 hidden_dims: tuple,
                                 num_classes: int) -> dict:
    metrics = evaluate_model(model, X_test_t, y_test_t)
    with torch.no_grad():
        probs = _batched_probs(model, X_test_t)
    y_test_np = y_test_t.cpu().numpy()
    metrics['ece'] = float(expected_calibration_error(probs, y_test_np))
    metrics['class_names'] = class_names
    metrics['params'] = int(count_params(model))
    metrics['size_kb'] = float(model_size_kb(model))
    metrics['flops_per_sample'] = int(compute_flops_mlp(input_dim, hidden_dims, num_classes))
    return metrics


def rf_model_size_kb(model) -> float:
    return len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)) / 1024.0


def fit_edgeiiot_v23_rf(seed: int, arrays: dict) -> tuple:
    start = time.time()
    rf = RandomForestClassifier(
        n_estimators=EDGEIIOT_RF_TREES,
        max_depth=EDGEIIOT_RF_MAX_DEPTH,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(arrays['X_train_np'], arrays['y_train_np'])
    train_time = time.time() - start
    probs = rf_predict_proba_aligned(rf, arrays['X_test_np'], arrays['num_classes'])
    preds = probs.argmax(axis=1)
    metrics = sklearn_metrics_from_predictions(arrays['y_test_np'], preds, probs, arrays['class_names'])
    metrics.update({
        'params': None,
        'size_kb': float(rf_model_size_kb(rf)),
        'flops_per_sample': None,
        'train_time_sec': float(train_time),
    })
    return rf, metrics


def fit_edgeiiot_v23_calibrated_rf(seed: int, arrays: dict):
    rf_base = RandomForestClassifier(
        n_estimators=EDGEIIOT_RF_TREES,
        max_depth=EDGEIIOT_RF_MAX_DEPTH,
        random_state=seed,
        n_jobs=-1,
    )
    calibrated = CalibratedClassifierCV(
        rf_base,
        method=EDGEIIOT_RF_CALIBRATION_METHOD,
        cv=EDGEIIOT_RF_CALIBRATION_CV,
    )
    calibrated.fit(arrays['X_train_np'], arrays['y_train_np'])
    return calibrated


def _train_teacher_standard(input_dim: int, num_classes: int, X_train_t: torch.Tensor,
                            y_train_t: torch.Tensor, X_val_t: torch.Tensor,
                            y_val_t: torch.Tensor, class_weights: torch.Tensor) -> tuple:
    teacher_b = TeacherMLP(input_dim=input_dim, num_classes=num_classes).to(device)
    start = time.time()
    teacher_b, teacher_history = train_standard(
        teacher_b, X_train_t, y_train_t, X_val_t, y_val_t,
        class_weights=class_weights, return_loss_curve=True, **TRAIN_CONFIG
    )
    return teacher_b, teacher_history, time.time() - start


def _train_teacher_curriculum(input_dim: int, num_classes: int, X_train_t: torch.Tensor,
                              y_train_t: torch.Tensor, X_val_t: torch.Tensor,
                              y_val_t: torch.Tensor, class_weights: torch.Tensor,
                              difficulty_order: torch.Tensor) -> tuple:
    teacher_c_fair = TeacherMLP(input_dim=input_dim, num_classes=num_classes).to(device)
    start = time.time()
    teacher_c_fair, teacher_history = train_with_curriculum(
        teacher_c_fair, X_train_t, y_train_t, difficulty_order, X_val_t, y_val_t,
        stages=CL_STAGES_FAIR,
        class_weights=class_weights,
        batch_size=TRAIN_CONFIG['batch_size'],
        lr=TRAIN_CONFIG['lr'],
        weight_decay=TRAIN_CONFIG['weight_decay'],
        patience=TRAIN_CONFIG['patience'],
        return_loss_curve=True,
    )
    return teacher_c_fair, teacher_history, time.time() - start


def _copy_teacher_metrics(metrics: dict, train_time_sec: float) -> dict:
    out = dict(metrics)
    out['train_time_sec'] = float(train_time_sec)
    return out

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
    """Train one student with hard labels plus RF and CL-teacher soft targets."""
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

    ds = TensorDataset(X_d, y_d, rf_soft_targets, cl_soft_targets)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
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
        for xb, yb, rf_sb, cl_sb in loader:
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
# CELL 6: Edge-IIoT v2.3 seed runner
# ============================================================================
def run_edgeiiot_v23_seed(arrays: dict, seed: int) -> dict:
    set_seed(seed)
    input_dim = arrays['input_dim']
    num_classes = arrays['num_classes']
    class_names = arrays['class_names']

    X_train_t = torch.tensor(arrays['X_train_np'], dtype=torch.float32, device=device)
    y_train_t = torch.tensor(arrays['y_train_np'], dtype=torch.long, device=device)
    X_val_t = torch.tensor(arrays['X_val_np'], dtype=torch.float32, device=device)
    y_val_t = torch.tensor(arrays['y_val_np'], dtype=torch.long, device=device)
    X_test_t = torch.tensor(arrays['X_test_np'], dtype=torch.float32, device=device)
    y_test_t = torch.tensor(arrays['y_test_np'], dtype=torch.long, device=device)
    class_weights = class_weights_from_labels(arrays['y_train_np'], num_classes).to(device)

    print(f'\nSeed {seed}: Config A_RF_500')
    rf_model, metrics_a = fit_edgeiiot_v23_rf(seed, arrays)

    print(f'Seed {seed}: calibrated RF soft labels for E_KD_from_RF')
    calibrated_rf = fit_edgeiiot_v23_calibrated_rf(seed, arrays)
    rf_soft_train = rf_predict_proba_aligned(calibrated_rf, arrays['X_train_np'], num_classes)
    rf_soft_train_t = torch.tensor(rf_soft_train, dtype=torch.float32, device=device)

    print(f'Seed {seed}: Config B_Full_MLP')
    teacher_b, hist_b, time_b = _train_teacher_standard(
        input_dim, num_classes, X_train_t, y_train_t, X_val_t, y_val_t, class_weights
    )
    metrics_b = torch_metrics_with_resources(
        teacher_b, X_test_t, y_test_t, class_names, input_dim, (128, 256, 128), num_classes
    )
    metrics_b = _copy_teacher_metrics(metrics_b, time_b)

    print(f'Seed {seed}: Config C_CL_MLP_loss_fair')
    difficulty_order = compute_difficulty_loss_based(
        X_train_t, y_train_t,
        input_dim=input_dim, num_classes=num_classes, seed=seed,
    )
    teacher_c_fair, hist_c, time_c = _train_teacher_curriculum(
        input_dim, num_classes, X_train_t, y_train_t, X_val_t, y_val_t, class_weights, difficulty_order
    )
    metrics_c = torch_metrics_with_resources(
        teacher_c_fair, X_test_t, y_test_t, class_names, input_dim, (128, 256, 128), num_classes
    )
    metrics_c = _copy_teacher_metrics(metrics_c, time_c)

    results = {
        'seed': int(seed),
        'teacher_metrics': {
            'A_RF_500': metrics_a,
            'B_Full_MLP': metrics_b,
            'C_CL_MLP_loss_fair': metrics_c,
        },
        'students': {},
    }

    for student_name, hidden_dims in EDGEIIOT_STUDENT_SPECS:
        print(f'Seed {seed}: {student_name} Config D_Small_MLP')
        student_d = StudentMLP(input_dim=input_dim, hidden_dims=hidden_dims, num_classes=num_classes).to(device)
        start = time.time()
        student_d, hist_d = train_standard(
            student_d, X_train_t, y_train_t, X_val_t, y_val_t,
            class_weights=class_weights, return_loss_curve=True, **TRAIN_CONFIG
        )
        metrics_d = torch_metrics_with_resources(
            student_d, X_test_t, y_test_t, class_names, input_dim, hidden_dims, num_classes
        )
        metrics_d['train_time_sec'] = float(time.time() - start)

        print(f'Seed {seed}: {student_name} Config E_KD_from_RF')
        student_e = StudentMLP(input_dim=input_dim, hidden_dims=hidden_dims, num_classes=num_classes).to(device)
        start = time.time()
        student_e = train_kd(
            student_e, rf_soft_train_t, X_train_t, y_train_t, X_val_t, y_val_t,
            class_weights=class_weights,
            T=KD_T_EDGE,
            alpha=KD_ALPHA_EDGE,
            **TRAIN_CONFIG,
        )
        hist_e = {'loss': [], 'val_f1': []}
        metrics_e = torch_metrics_with_resources(
            student_e, X_test_t, y_test_t, class_names, input_dim, hidden_dims, num_classes
        )
        metrics_e['train_time_sec'] = float(time.time() - start)

        print(f'Seed {seed}: {student_name} Config E2_KD_from_MLP')
        with torch.no_grad():
            teacher_b_train_probs = _batched_probs(teacher_b, X_train_t)
        teacher_b_train_t = torch.tensor(teacher_b_train_probs, dtype=torch.float32, device=device)
        student_e2 = StudentMLP(input_dim=input_dim, hidden_dims=hidden_dims, num_classes=num_classes).to(device)
        start = time.time()
        student_e2 = train_kd(
            student_e2, teacher_b_train_t, X_train_t, y_train_t, X_val_t, y_val_t,
            class_weights=class_weights,
            T=KD_T_EDGE,
            alpha=KD_ALPHA_EDGE,
            **TRAIN_CONFIG,
        )
        hist_e2 = {'loss': [], 'val_f1': []}
        metrics_e2 = torch_metrics_with_resources(
            student_e2, X_test_t, y_test_t, class_names, input_dim, hidden_dims, num_classes
        )
        metrics_e2['train_time_sec'] = float(time.time() - start)

        print(f'Seed {seed}: {student_name} Config F_KD_from_CL_MLP')
        with torch.no_grad():
            teacher_c_train_probs = _batched_probs(teacher_c_fair, X_train_t)
        teacher_c_train_t = torch.tensor(teacher_c_train_probs, dtype=torch.float32, device=device)
        student_f = StudentMLP(input_dim=input_dim, hidden_dims=hidden_dims, num_classes=num_classes).to(device)
        start = time.time()
        student_f = train_kd(
            student_f, teacher_c_train_t, X_train_t, y_train_t, X_val_t, y_val_t,
            class_weights=class_weights,
            T=KD_T_EDGE,
            alpha=KD_ALPHA_EDGE,
            **TRAIN_CONFIG,
        )
        hist_f = {'loss': [], 'val_f1': []}
        metrics_f = torch_metrics_with_resources(
            student_f, X_test_t, y_test_t, class_names, input_dim, hidden_dims, num_classes
        )
        metrics_f['train_time_sec'] = float(time.time() - start)

        print(f'Seed {seed}: {student_name} Config J_CoDistill_RF_CL')
        student_j = StudentMLP(input_dim=input_dim, hidden_dims=hidden_dims, num_classes=num_classes).to(device)
        start = time.time()
        student_j, hist_j = train_codistill_rf_cl(
            student_j, rf_soft_train_t, teacher_c_fair, X_train_t, y_train_t, X_val_t, y_val_t,
            T=KD_T_EDGE,
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
        metrics_j = torch_metrics_with_resources(
            student_j, X_test_t, y_test_t, class_names, input_dim, hidden_dims, num_classes
        )
        metrics_j['train_time_sec'] = float(time.time() - start)

        results['students'][student_name] = {
            'hidden_dims': list(hidden_dims),
            'configs': {
                'A_RF_500': metrics_a,
                'B_Full_MLP': metrics_b,
                'C_CL_MLP_loss_fair': metrics_c,
                'D_Small_MLP': metrics_d,
                'E_KD_from_RF': metrics_e,
                'E2_KD_from_MLP': metrics_e2,
                'F_KD_from_CL_MLP': metrics_f,
                'J_CoDistill_RF_CL': metrics_j,
            },
            'histories': {
                'B_Full_MLP': hist_b,
                'C_CL_MLP_loss_fair': hist_c,
                'D_Small_MLP': hist_d,
                'E_KD_from_RF': hist_e,
                'E2_KD_from_MLP': hist_e2,
                'F_KD_from_CL_MLP': hist_f,
                'J_CoDistill_RF_CL': hist_j,
            },
        }

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return results

# ============================================================================
# CELL 7: Aggregate and export Edge-IIoT results
# ============================================================================
def aggregate_edgeiiot_config(seed_results: list, student_name: str, config_name: str) -> dict:
    metric_rows = []
    per_class_f1 = []
    class_names = None
    for seed_result in seed_results:
        metrics = seed_result['students'][student_name]['configs'][config_name]
        class_names = metrics.get('class_names') or class_names
        metric_rows.append(metrics)
        per_class_f1.append(metrics['per_class_f1'])
    per_class_f1_arr = np.asarray(per_class_f1, dtype=np.float64)
    out = {
        'student_name': student_name,
        'Config': config_name,
        'n_seeds': int(len(seed_results)),
        'accuracy_mean': float(np.mean([row['accuracy'] for row in metric_rows])),
        'accuracy_std': float(np.std([row['accuracy'] for row in metric_rows], ddof=1)) if len(metric_rows) > 1 else 0.0,
        'macro_f1_mean': float(np.mean([row['macro_f1'] for row in metric_rows])),
        'macro_f1_std': float(np.std([row['macro_f1'] for row in metric_rows], ddof=1)) if len(metric_rows) > 1 else 0.0,
        'macro_precision_mean': float(np.mean([row['macro_precision'] for row in metric_rows])),
        'macro_recall_mean': float(np.mean([row['macro_recall'] for row in metric_rows])),
        'ece_mean': float(np.mean([row['ece'] for row in metric_rows if row['ece'] is not None])),
        'train_time_sec_mean': float(np.mean([row.get('train_time_sec', 0.0) for row in metric_rows])),
        'per_class_f1_mean': per_class_f1_arr.mean(axis=0).tolist(),
        'per_class_f1_std': per_class_f1_arr.std(axis=0, ddof=1).tolist() if len(metric_rows) > 1 else [0.0] * per_class_f1_arr.shape[1],
        'class_names': class_names,
        'params': metric_rows[0].get('params'),
        'size_kb': metric_rows[0].get('size_kb'),
        'flops_per_sample': metric_rows[0].get('flops_per_sample'),
    }
    return out


def _csv_safe_rows(rows: list) -> list:
    safe = []
    for row in rows:
        row_copy = dict(row)
        for key in ['per_class_f1_mean', 'per_class_f1_std', 'class_names']:
            row_copy[key] = json.dumps(row_copy[key])
        safe.append(row_copy)
    return safe


def run_edgeiiot_v23_generalization(seeds: list = None,
                                    output_dir: str = EDGEIIOT_OUTPUT_DIR) -> dict:
    seeds = list(EDGEIIOT_SEEDS if seeds is None else seeds)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    arrays = prepare_edgeiiot_v23_arrays(EDGEIIOT_ML_PATH, EDGEIIOT_TARGET_COL)
    print(f"Prepared Edge-IIoT: input_dim={arrays['input_dim']}, num_classes={arrays['num_classes']}")
    print(f"Classes: {arrays['class_names']}")
    print(f"Features: {len(arrays['feature_names'])}")

    seed_results = []
    started = time.time()
    for seed in seeds:
        seed_result = run_edgeiiot_v23_seed(arrays, seed)
        seed_results.append(seed_result)
        checkpoint_path = output_path / f'edgeiiot_v23_seed_{seed}_checkpoint.json'
        write_edge_json(str(checkpoint_path), seed_result)

    config_order = [
        'A_RF_500',
        'B_Full_MLP',
        'C_CL_MLP_loss_fair',
        'D_Small_MLP',
        'E_KD_from_RF',
        'E2_KD_from_MLP',
        'F_KD_from_CL_MLP',
        'J_CoDistill_RF_CL',
    ]
    aggregate_rows = []
    for student_name, _hidden_dims in EDGEIIOT_STUDENT_SPECS:
        for config_name in config_order:
            aggregate_rows.append(aggregate_edgeiiot_config(seed_results, student_name, config_name))

    ranking_rows = sorted(
        aggregate_rows,
        key=lambda row: (row['macro_f1_mean'], row['accuracy_mean']),
        reverse=True,
    )
    metadata = dict(arrays['metadata'])
    metadata.update({
        'run_mode': EDGEIIOT_RUN_MODE,
        'seeds': seeds,
        'n_seeds': len(seeds),
        'runtime_sec': float(time.time() - started),
        'configs': config_order,
        'students': [name for name, _hidden_dims in EDGEIIOT_STUDENT_SPECS],
        'kd_temperature': KD_T_EDGE,
        'kd_alpha': KD_ALPHA_EDGE,
        'rf_trees': EDGEIIOT_RF_TREES,
        'rf_max_depth': EDGEIIOT_RF_MAX_DEPTH,
        'rf_calibration_method': EDGEIIOT_RF_CALIBRATION_METHOD,
        'rf_calibration_cv': EDGEIIOT_RF_CALIBRATION_CV,
        'codistill_weights': {'ce': CODISTILL_CE_WEIGHT, 'rf': CODISTILL_RF_WEIGHT, 'cl': CODISTILL_CL_WEIGHT},
        'codistill_schedule': {'epochs': CODISTILL_EPOCHS, 'lr': CODISTILL_LR, 'patience': CODISTILL_PATIENCE},
        'guide': EDGEIIOT_GUIDE_FILE,
        'outputs': [
            'edgeiiot_v23_results.json',
            'edgeiiot_v23_results_student_A.csv',
            'edgeiiot_v23_results_student_B.csv',
            'edgeiiot_v23_metadata.json',
            'edgeiiot_v23_config_rankings.csv',
        ],
    })

    payload = {
        'metadata': metadata,
        'seed_results': seed_results,
        'aggregate_rows': aggregate_rows,
        'rankings': ranking_rows,
        'class_names': arrays['class_names'],
        'feature_names': arrays['feature_names'],
    }

    write_edge_json(str(output_path / 'edgeiiot_v23_results.json'), payload)
    write_edge_json(str(output_path / 'edgeiiot_v23_metadata.json'), metadata)
    pd.DataFrame(_csv_safe_rows(ranking_rows)).to_csv(output_path / 'edgeiiot_v23_config_rankings.csv', index=False)
    pd.DataFrame(_csv_safe_rows([row for row in aggregate_rows if row['student_name'] == 'student_A_32_16'])).to_csv(
        output_path / 'edgeiiot_v23_results_student_A.csv', index=False
    )
    pd.DataFrame(_csv_safe_rows([row for row in aggregate_rows if row['student_name'] == 'student_B_64_32'])).to_csv(
        output_path / 'edgeiiot_v23_results_student_B.csv', index=False
    )

    print('\nEdge-IIoT v2.3 generalization complete.')
    print(f"Results JSON: {output_path / 'edgeiiot_v23_results.json'}")
    print(f"Rankings CSV: {output_path / 'edgeiiot_v23_config_rankings.csv'}")
    return payload

# ============================================================================
# CELL 8: Run Edge-IIoT v2.3 generalization
# ============================================================================
edgeiiot_v23_outputs = run_edgeiiot_v23_generalization()
