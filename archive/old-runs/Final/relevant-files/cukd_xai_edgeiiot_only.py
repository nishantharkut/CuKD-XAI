# ============================================================================
# CuKD-XAI Edge-IIoT-only generalization runner
#
# This file is intentionally standalone. It does not execute the completed
# WSN route. Use it when the goal is only the Edge-IIoTset secondary-dataset
# evidence.
# ============================================================================

# ============================================================================
# CELL 1: Imports and configuration
# ============================================================================
import copy
import json
import os
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
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


EDGEIIOT_ML_PATH = '/home/ubuntu/nishn_workspce/oig-exclusion-testing/.cukd_xai_secret/datasets/edgeiiot/extracted/Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv'
EDGEIIOT_TARGET_COL = 'Attack_type'
EDGEIIOT_OUTPUT_DIR = 'edgeiiot_only_outputs'

EDGEIIOT_SEEDS_FINAL5 = [42, 123, 456, 789, 1001]
EDGEIIOT_SEEDS_QUICK1 = [42]
EDGEIIOT_RUN_MODE = 'edgeiiot_final'  # 'edgeiiot_final' or 'edgeiiot_quick'
EDGEIIOT_SEEDS = EDGEIIOT_SEEDS_FINAL5 if EDGEIIOT_RUN_MODE == 'edgeiiot_final' else EDGEIIOT_SEEDS_QUICK1

EDGEIIOT_RF_TREES = 300
EDGEIIOT_RF_MAX_DEPTH = 15
EDGEIIOT_RF_CLASS_WEIGHT = 'balanced_subsample'
EDGEIIOT_RF_CALIBRATION_METHOD = 'sigmoid'
EDGEIIOT_RF_CALIBRATION_CV = 3

EDGEIIOT_MAX_CATEGORICAL_CARDINALITY = 64
EDGEIIOT_RARE_CATEGORY_MIN_COUNT = 10

KD_T_DEFAULT = 2
KD_ALPHA_DEFAULT = 0.5
STUDENT_A_HIDDEN = (32, 16)
STUDENT_B_HIDDEN = (64, 32)
EDGEIIOT_STUDENT_SPECS = [
    ('student_A_32_16', STUDENT_A_HIDDEN),
    ('student_B_64_32', STUDENT_B_HIDDEN),
]

TRAIN_CONFIG = {
    'epochs': 30,
    'batch_size': 256,
    'lr': 1e-3,
    'weight_decay': 1e-3,
    'patience': 8,
}

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

print('Edge-IIoT-only runner')
print(f'CSV path: {EDGEIIOT_ML_PATH}')
print(f'Target: {EDGEIIOT_TARGET_COL}')
print(f'Seeds: {EDGEIIOT_SEEDS}')
print(f'Output dir: {EDGEIIOT_OUTPUT_DIR}')


# ============================================================================
# CELL 2: Model and metric helpers
# ============================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class StudentMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple, num_classes: int):
        super().__init__()
        layers = []
        prev = int(input_dim)
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev, int(hidden)))
            layers.append(nn.ReLU())
            prev = int(hidden)
        layers.append(nn.Linear(prev, int(num_classes)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_kb(model: nn.Module, bytes_per_param: int = 4) -> float:
    return count_params(model) * bytes_per_param / 1024.0


def compute_flops_mlp(input_dim: int, hidden_dims: tuple, num_classes: int) -> int:
    dims = [int(input_dim)] + [int(h) for h in hidden_dims] + [int(num_classes)]
    flops = 0
    for idx in range(len(dims) - 1):
        flops += 2 * dims[idx] * dims[idx + 1]
        if idx < len(dims) - 2:
            flops += dims[idx + 1]
    return int(flops)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def json_convert(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    return str(obj)


def write_json(path: str, payload: dict) -> None:
    ensure_dir(os.path.dirname(path) or '.')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, default=json_convert)


def class_weights_from_labels(y_np: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y_np, minlength=int(num_classes))
    weights = len(y_np) / (int(num_classes) * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, class_names: list) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    pc_precision, pc_recall, pc_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_precision': float(precision),
        'macro_recall': float(recall),
        'macro_f1': float(f1),
        'per_class_precision': pc_precision.tolist(),
        'per_class_recall': pc_recall.tolist(),
        'per_class_f1': pc_f1.tolist(),
        'class_names': class_names,
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }


# ============================================================================
# CELL 3: CPU-safe student training
# ============================================================================
def batched_predict_cpu(model: nn.Module, X, batch_size: int = 4096) -> np.ndarray:
    model_cpu = model.cpu().eval()
    X_cpu = X if torch.is_tensor(X) else torch.tensor(X, dtype=torch.float32)
    X_cpu = X_cpu.cpu().float()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X_cpu), batch_size):
            preds.append(model_cpu(X_cpu[start:start + batch_size]).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def evaluate_model_cpu(model: nn.Module, X, y, class_names: list, batch_size: int = 4096) -> dict:
    y_np = y.cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
    preds = batched_predict_cpu(model, X, batch_size=batch_size)
    return metrics_dict(y_np, preds, class_names)


def train_standard_cpu(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor,
                       X_val: torch.Tensor, y_val: torch.Tensor,
                       class_weights: torch.Tensor = None,
                       epochs: int = 30, batch_size: int = 256,
                       lr: float = 1e-3, weight_decay: float = 1e-3,
                       patience: int = 8) -> nn.Module:
    model = model.cpu()
    X_train = X_train.cpu().float()
    y_train = y_train.cpu().long()
    X_val = X_val.cpu().float()
    y_val_np = y_val.cpu().numpy()
    if class_weights is not None:
        class_weights = class_weights.cpu().float()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(epochs), 1))
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=int(batch_size), shuffle=True)
    best_val, best_state, bad = 0.0, None, 0
    for _ in range(int(epochs)):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()
        preds = batched_predict_cpu(model, X_val)
        val_f1 = f1_score(y_val_np, preds, average='macro')
        if val_f1 > best_val:
            best_val = val_f1
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= int(patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu().eval()


def train_kd_cpu(student: nn.Module, teacher_probs,
                 X_train: torch.Tensor, y_train: torch.Tensor,
                 X_val: torch.Tensor, y_val: torch.Tensor,
                 T: float = KD_T_DEFAULT, alpha: float = KD_ALPHA_DEFAULT,
                 class_weights: torch.Tensor = None,
                 epochs: int = 30, batch_size: int = 256,
                 lr: float = 1e-3, weight_decay: float = 1e-3,
                 patience: int = 8) -> nn.Module:
    student = student.cpu()
    X_train = X_train.cpu().float()
    y_train = y_train.cpu().long()
    X_val = X_val.cpu().float()
    y_val_np = y_val.cpu().numpy()
    probs = teacher_probs if torch.is_tensor(teacher_probs) else torch.tensor(teacher_probs, dtype=torch.float32)
    probs = probs.cpu().float()
    pseudo_logits = torch.log(probs.clamp(min=1e-8))
    soft_targets = F.softmax(pseudo_logits / float(T), dim=1).detach()
    if class_weights is not None:
        class_weights = class_weights.cpu().float()
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(epochs), 1))
    loader = DataLoader(TensorDataset(X_train, y_train, soft_targets), batch_size=int(batch_size), shuffle=True)
    best_val, best_state, bad = 0.0, None, 0
    for _ in range(int(epochs)):
        student.train()
        for xb, yb, sb in loader:
            optimizer.zero_grad()
            logits = student(xb)
            log_soft_s = F.log_softmax(logits / float(T), dim=1)
            kd_term = F.kl_div(log_soft_s, sb, reduction='batchmean') * (float(T) * float(T))
            ce_term = ce_loss(logits, yb)
            loss = float(alpha) * kd_term + (1.0 - float(alpha)) * ce_term
            loss.backward()
            optimizer.step()
        scheduler.step()
        preds = batched_predict_cpu(student, X_val)
        val_f1 = f1_score(y_val_np, preds, average='macro')
        if val_f1 > best_val:
            best_val = val_f1
            best_state = copy.deepcopy(student.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= int(patience):
                break
    if best_state is not None:
        student.load_state_dict(best_state)
    return student.cpu().eval()


# ============================================================================
# CELL 4: Edge-IIoTset preprocessing
# ============================================================================
def edgeiiot_normalize_categorical(series: pd.Series) -> pd.Series:
    values = series.where(series.notna(), '__MISSING__').astype(str).str.strip()
    return values.mask(values == '', '__EMPTY__')


def edgeiiot_fit_category_caps(X_train_raw: pd.DataFrame, categorical_cols: list) -> tuple:
    allowed_categories = {}
    category_stats = {}
    for col in categorical_cols:
        values = edgeiiot_normalize_categorical(X_train_raw[col])
        counts = values.value_counts(dropna=False)
        retained = counts[counts >= EDGEIIOT_RARE_CATEGORY_MIN_COUNT].head(
            EDGEIIOT_MAX_CATEGORICAL_CARDINALITY
        ).index.tolist()
        if not retained and len(counts) > 0:
            retained = counts.head(min(EDGEIIOT_MAX_CATEGORICAL_CARDINALITY, len(counts))).index.tolist()
        allowed_categories[col] = set(retained)
        category_stats[col] = {
            'raw_cardinality': int(len(counts)),
            'retained_categories': int(len(retained)),
            'other_rows_in_train': int((~values.isin(retained)).sum()),
        }
    return allowed_categories, category_stats


def edgeiiot_encode_split(X_raw: pd.DataFrame, numeric_cols: list, categorical_cols: list,
                          allowed_categories: dict, dummy_columns: list = None) -> pd.DataFrame:
    parts = []
    if numeric_cols:
        X_numeric = X_raw[numeric_cols].apply(pd.to_numeric, errors='coerce')
        X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
        parts.append(X_numeric)
    if categorical_cols:
        X_cat = pd.DataFrame(index=X_raw.index)
        for col in categorical_cols:
            values = edgeiiot_normalize_categorical(X_raw[col])
            retained = allowed_categories.get(col, set())
            X_cat[col] = values.where(values.isin(retained), '__OTHER__')
        X_dummy = pd.get_dummies(X_cat, columns=categorical_cols, dummy_na=False, dtype=np.float32)
        for col in categorical_cols:
            other_col = f'{col}___OTHER__'
            if other_col not in X_dummy.columns:
                X_dummy[other_col] = np.float32(0.0)
        if dummy_columns is not None:
            X_dummy = X_dummy.reindex(columns=dummy_columns, fill_value=0.0).astype(np.float32)
        parts.append(X_dummy)
    if not parts:
        raise ValueError('Edge-IIoTset preprocessing produced no usable features')
    return pd.concat(parts, axis=1)


def prepare_edgeiiot_ml(csv_path: str = EDGEIIOT_ML_PATH,
                        target_col: str = EDGEIIOT_TARGET_COL) -> dict:
    df_edge = pd.read_csv(csv_path, low_memory=False)
    df_edge.columns = df_edge.columns.str.strip()
    if target_col not in df_edge.columns:
        raise ValueError(f'{target_col!r} not found. Available columns: {df_edge.columns.tolist()}')
    rows_raw, cols_raw = df_edge.shape
    candidate_leakage = EDGEIIOT_LEAKAGE_COLUMNS + [
        col for col in EDGEIIOT_AUXILIARY_TARGET_COLUMNS if col != target_col
    ]
    removed_leakage = [col for col in dict.fromkeys(candidate_leakage) if col in df_edge.columns]
    df_edge = df_edge.drop(columns=removed_leakage)
    if target_col not in df_edge.columns:
        raise RuntimeError(f'Target column {target_col!r} was removed unexpectedly')
    target_raw = df_edge[target_col]
    target_missing = target_raw.isna()
    target_strings = target_raw.astype(str).str.strip()
    invalid_tokens = {'', 'nan', 'none', 'null', '__missing__'}
    valid_target_mask = (~target_missing) & (~target_strings.str.lower().isin(invalid_tokens))
    dropped_invalid_target_rows = int((~valid_target_mask).sum())
    df_edge = df_edge.loc[valid_target_mask].copy()
    df_edge[target_col] = target_strings.loc[valid_target_mask]
    df_edge = df_edge.drop_duplicates().reset_index(drop=True)
    y_labels = df_edge[target_col].astype(str).str.strip()
    X_df = df_edge.drop(columns=[target_col])
    zero_var = [col for col in X_df.columns if X_df[col].nunique(dropna=False) <= 1]
    X_df = X_df.drop(columns=zero_var)
    if X_df.empty:
        raise ValueError('No Edge-IIoTset feature columns remain after leakage and zero-variance removal')
    categorical_cols = X_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    numeric_cols = [col for col in X_df.columns if col not in categorical_cols]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_labels).astype(np.int64)
    class_counts = np.bincount(y)
    if len(class_counts) < 2:
        raise ValueError('Edge-IIoTset target must contain at least two classes')
    if int(class_counts.min()) < 2:
        raise ValueError(f'Every class needs at least 2 samples for stratified splitting; counts={class_counts.tolist()}')
    X_trainval_raw, X_test_raw, y_trainval, y_test = train_test_split(
        X_df, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_trainval_raw, y_trainval, test_size=0.1765, random_state=42, stratify=y_trainval
    )
    allowed_categories, category_stats = edgeiiot_fit_category_caps(X_train_raw, categorical_cols)
    X_train_encoded = edgeiiot_encode_split(X_train_raw, numeric_cols, categorical_cols, allowed_categories)
    dummy_columns = [col for col in X_train_encoded.columns if col not in numeric_cols]
    X_val_encoded = edgeiiot_encode_split(
        X_val_raw, numeric_cols, categorical_cols, allowed_categories, dummy_columns=dummy_columns
    )
    X_test_encoded = edgeiiot_encode_split(
        X_test_raw, numeric_cols, categorical_cols, allowed_categories, dummy_columns=dummy_columns
    )
    constant_encoded_cols = [
        col for col in X_train_encoded.columns
        if X_train_encoded[col].nunique(dropna=False) <= 1 and not col.endswith('___OTHER__')
    ]
    if constant_encoded_cols:
        X_train_encoded = X_train_encoded.drop(columns=constant_encoded_cols)
        X_val_encoded = X_val_encoded.drop(columns=constant_encoded_cols)
        X_test_encoded = X_test_encoded.drop(columns=constant_encoded_cols)
    continuous_cols = [col for col in numeric_cols if col in X_train_encoded.columns]
    scaler = StandardScaler()
    if continuous_cols:
        X_train_encoded.loc[:, continuous_cols] = scaler.fit_transform(X_train_encoded[continuous_cols]).astype(np.float32)
        X_val_encoded.loc[:, continuous_cols] = scaler.transform(X_val_encoded[continuous_cols]).astype(np.float32)
        X_test_encoded.loc[:, continuous_cols] = scaler.transform(X_test_encoded[continuous_cols]).astype(np.float32)
    X_train = X_train_encoded.to_numpy(dtype=np.float32, copy=True)
    X_val = X_val_encoded.to_numpy(dtype=np.float32, copy=True)
    X_test = X_test_encoded.to_numpy(dtype=np.float32, copy=True)
    metadata = {
        'rows_raw': int(rows_raw),
        'cols_raw': int(cols_raw),
        'rows_after_cleaning': int(len(df_edge)),
        'dropped_invalid_target_rows': dropped_invalid_target_rows,
        'removed_leakage_columns': removed_leakage,
        'removed_zero_variance_columns': zero_var,
        'removed_constant_encoded_columns': constant_encoded_cols,
        'categorical_columns': categorical_cols,
        'numeric_columns_scaled': continuous_cols,
        'categorical_cardinality_policy': {
            'max_categories_per_column': int(EDGEIIOT_MAX_CATEGORICAL_CARDINALITY),
            'rare_category_min_count': int(EDGEIIOT_RARE_CATEGORY_MIN_COUNT),
        },
        'categorical_cardinality_stats': category_stats,
        'n_features_after_encoding': int(X_train.shape[1]),
        'class_names': label_encoder.classes_.tolist(),
        'class_distribution': dict(zip(label_encoder.classes_.tolist(), np.bincount(y).astype(int).tolist())),
    }
    print('Edge-IIoTset preprocessing complete')
    print(f'Rows raw/clean: {rows_raw}/{len(df_edge)}')
    print(f'Features after encoding: {X_train.shape[1]}')
    print(f'Classes: {label_encoder.classes_.tolist()}')
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_names': X_train_encoded.columns.tolist(),
        'class_names': label_encoder.classes_.tolist(),
        'input_dim': int(X_train.shape[1]),
        'num_classes': int(len(label_encoder.classes_)),
        'metadata': metadata,
    }


# ============================================================================
# CELL 5: RF teacher and one-seed Edge-IIoT runs
# ============================================================================
def edgeiiot_predict_proba_aligned(model, X_np: np.ndarray, num_classes: int) -> np.ndarray:
    raw = model.predict_proba(X_np)
    classes = getattr(model, 'classes_', np.arange(raw.shape[1]))
    aligned = np.zeros((len(X_np), int(num_classes)), dtype=np.float32)
    for idx, cls in enumerate(classes):
        cls_idx = int(cls)
        if 0 <= cls_idx < int(num_classes):
            aligned[:, cls_idx] = raw[:, idx]
    row_sums = aligned.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0) or not np.all(np.isfinite(aligned)):
        raise RuntimeError('Edge-IIoTset RF teacher produced invalid probability rows')
    return aligned / row_sums


def fit_edgeiiot_rf(arrays: dict, seed: int, rf_trees: int = EDGEIIOT_RF_TREES):
    base_rf = RandomForestClassifier(
        n_estimators=int(rf_trees),
        max_depth=EDGEIIOT_RF_MAX_DEPTH,
        random_state=int(seed),
        n_jobs=-1,
        class_weight=EDGEIIOT_RF_CLASS_WEIGHT,
    )
    class_counts = np.bincount(arrays['y_train'], minlength=arrays['num_classes'])
    min_class_count = int(class_counts.min()) if len(class_counts) else 0
    cv = min(int(EDGEIIOT_RF_CALIBRATION_CV), min_class_count)
    if cv >= 2:
        rf = CalibratedClassifierCV(base_rf, method=EDGEIIOT_RF_CALIBRATION_METHOD, cv=cv)
        calibration_info = {
            'method': EDGEIIOT_RF_CALIBRATION_METHOD,
            'cv': int(cv),
            'min_train_class_count': min_class_count,
        }
    else:
        rf = base_rf
        calibration_info = {'method': 'none', 'cv': 0, 'min_train_class_count': min_class_count}
    start = time.perf_counter()
    rf.fit(arrays['X_train'], arrays['y_train'])
    train_time = time.perf_counter() - start
    rf_proba = edgeiiot_predict_proba_aligned(rf, arrays['X_test'], arrays['num_classes'])
    rf_pred = np.argmax(rf_proba, axis=1)
    rf_metrics = metrics_dict(arrays['y_test'], rf_pred, arrays['class_names'])
    rf_metrics['calibration'] = calibration_info
    rf_metrics['train_time_sec'] = float(train_time)
    rf_metrics['rf_trees'] = int(rf_trees)
    rf_metrics['rf_max_depth'] = EDGEIIOT_RF_MAX_DEPTH
    rf_metrics['rf_class_weight'] = EDGEIIOT_RF_CLASS_WEIGHT
    return rf, rf_metrics


def run_edgeiiot_one_seed(arrays: dict, seed: int, student_hidden: tuple,
                          rf_trees: int = EDGEIIOT_RF_TREES,
                          kd_T: float = KD_T_DEFAULT,
                          kd_alpha: float = KD_ALPHA_DEFAULT,
                          train_config: dict = None,
                          rf_model=None, rf_metrics: dict = None) -> dict:
    train_config = dict(TRAIN_CONFIG if train_config is None else train_config)
    set_seed(seed)
    X_train, y_train = arrays['X_train'], arrays['y_train']
    X_val, y_val = arrays['X_val'], arrays['y_val']
    X_test, y_test = arrays['X_test'], arrays['y_test']
    input_dim, num_classes = arrays['input_dim'], arrays['num_classes']
    class_names = arrays['class_names']
    if rf_model is None or rf_metrics is None:
        rf_model, rf_metrics = fit_edgeiiot_rf(arrays, seed, rf_trees=rf_trees)
    Xtr_t = torch.tensor(X_train, dtype=torch.float32)
    ytr_t = torch.tensor(y_train, dtype=torch.long)
    Xv_t = torch.tensor(X_val, dtype=torch.float32)
    yv_t = torch.tensor(y_val, dtype=torch.long)
    Xte_t = torch.tensor(X_test, dtype=torch.float32)
    yte_t = torch.tensor(y_test, dtype=torch.long)
    class_weights = class_weights_from_labels(y_train, num_classes)
    scratch = StudentMLP(input_dim, student_hidden, num_classes)
    scratch_start = time.perf_counter()
    scratch = train_standard_cpu(
        scratch, Xtr_t, ytr_t, Xv_t, yv_t,
        class_weights=class_weights,
        epochs=train_config['epochs'],
        batch_size=train_config['batch_size'],
        lr=train_config['lr'],
        weight_decay=train_config['weight_decay'],
        patience=train_config['patience'],
    )
    scratch_metrics = evaluate_model_cpu(scratch, Xte_t, yte_t, class_names)
    scratch_metrics.update({
        'params': count_params(scratch),
        'model_size_kb': model_size_kb(scratch),
        'model_size_kb_int8_theoretical': model_size_kb(scratch, 1),
        'flops_per_sample': compute_flops_mlp(input_dim, student_hidden, num_classes),
        'train_time_sec': float(time.perf_counter() - scratch_start),
    })
    rf_soft = edgeiiot_predict_proba_aligned(rf_model, X_train, num_classes)
    kd_student = StudentMLP(input_dim, student_hidden, num_classes)
    kd_start = time.perf_counter()
    kd_student = train_kd_cpu(
        kd_student, rf_soft, Xtr_t, ytr_t, Xv_t, yv_t,
        T=kd_T,
        alpha=kd_alpha,
        class_weights=class_weights,
        epochs=train_config['epochs'],
        batch_size=train_config['batch_size'],
        lr=train_config['lr'],
        weight_decay=train_config['weight_decay'],
        patience=train_config['patience'],
    )
    kd_metrics = evaluate_model_cpu(kd_student, Xte_t, yte_t, class_names)
    kd_metrics.update({
        'params': count_params(kd_student),
        'model_size_kb': model_size_kb(kd_student),
        'model_size_kb_int8_theoretical': model_size_kb(kd_student, 1),
        'flops_per_sample': compute_flops_mlp(input_dim, student_hidden, num_classes),
        'train_time_sec': float(time.perf_counter() - kd_start),
        'kd_T': float(kd_T),
        'kd_alpha': float(kd_alpha),
    })
    return {
        'A_RF_calibrated': copy.deepcopy(rf_metrics),
        'D_student_scratch': scratch_metrics,
        'E_KD_from_RF': kd_metrics,
    }


# ============================================================================
# CELL 6: Multi-seed Edge-IIoT runner and exports
# ============================================================================
def aggregate_seed_metrics(seed_results: dict, config_name: str) -> dict:
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
        'n_seeds': int(len(f1s)),
    }


def run_edgeiiot_generalization(csv_path: str = EDGEIIOT_ML_PATH,
                                output_dir: str = EDGEIIOT_OUTPUT_DIR,
                                seeds: list = None,
                                rf_trees: int = EDGEIIOT_RF_TREES,
                                train_config: dict = None) -> dict:
    seeds = list(EDGEIIOT_SEEDS if seeds is None else seeds)
    train_config = dict(TRAIN_CONFIG if train_config is None else train_config)
    out_dir = ensure_dir(output_dir)
    arrays = prepare_edgeiiot_ml(csv_path)
    payload = {
        'metadata': arrays['metadata'],
        'students': {},
        'seeds': seeds,
        'rf_trees': int(rf_trees),
        'kd_hyperparameters': {'T': float(KD_T_DEFAULT), 'alpha': float(KD_ALPHA_DEFAULT)},
        'train_config': train_config,
    }
    summary_rows = []
    student_seed_results = {student_name: {} for student_name, _ in EDGEIIOT_STUDENT_SPECS}
    for seed in seeds:
        print(f'[Edge-IIoTset] RF teacher seed {seed}')
        rf_model, rf_metrics = fit_edgeiiot_rf(arrays, seed, rf_trees=rf_trees)
        for student_name, hidden in EDGEIIOT_STUDENT_SPECS:
            print(f'[Edge-IIoTset] {student_name} seed {seed}')
            student_seed_results[student_name][seed] = run_edgeiiot_one_seed(
                arrays,
                seed,
                hidden,
                rf_trees=rf_trees,
                train_config=train_config,
                rf_model=rf_model,
                rf_metrics=copy.deepcopy(rf_metrics),
            )
    for student_name, _ in EDGEIIOT_STUDENT_SPECS:
        seed_results = student_seed_results[student_name]
        payload['students'][student_name] = seed_results
        for config_name in ['A_RF_calibrated', 'D_student_scratch', 'E_KD_from_RF']:
            row = {
                'student': student_name,
                'config': config_name,
                **aggregate_seed_metrics(seed_results, config_name),
            }
            summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, 'edgeiiot_generalization_summary.csv')
    results_path = os.path.join(out_dir, 'edgeiiot_generalization_results.json')
    summary_df.to_csv(summary_path, index=False)
    write_json(results_path, payload)
    print('\nEdge-IIoTset summary')
    print(summary_df.to_string(index=False))
    print(f'\nSaved: {summary_path}')
    print(f'Saved: {results_path}')
    return {'payload': payload, 'summary': summary_df, 'output_dir': out_dir}


# ============================================================================
# CELL 7: Execute Edge-IIoT-only run
# ============================================================================
if __name__ == '__main__':
    run_edgeiiot_generalization()
