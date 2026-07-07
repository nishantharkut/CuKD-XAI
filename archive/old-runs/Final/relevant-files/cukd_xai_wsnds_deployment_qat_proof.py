# ============================================================================
#  !!!! WARNING — READ BEFORE EDITING !!!!
#
#  This .py file is the Python SOURCE for the notebook. The canonical runnable
#  artifact is `cukd_xai_wsnds_deployment_qat_proof.ipynb`.
#
#  If you regenerate the .ipynb, do it from THIS file to keep the proof route
#  deterministic and aligned with the v2.3 base definitions.
# ============================================================================

# ============================================================================
# CuKD-XAI WSN-DS Deployment + QAT Proof Route (v2.3-derived)
#
# - This proof route is NOT a replacement for the completed 10-seed results.
# - It reuses v2.3 preprocessing, model definitions, and training helpers.
# - It runs a single deterministic seed for deployability, size, and latency.
#
# Author: Nishant Harkut (2023IMG-040), ABV-IIITM Gwalior
# ============================================================================

# ============================================================================
# CELL 1: Install dependencies
# ============================================================================
# !pip install -q scikit-learn pandas numpy torch joblib

# ============================================================================
# CELL 2: Imports and global config
# ============================================================================
import copy
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, confusion_matrix

try:
    import joblib
except Exception:
    joblib = None
import pickle

from torch.ao.quantization import (
    QuantStub,
    DeQuantStub,
    get_default_qat_qconfig,
    prepare_qat,
    convert,
    fuse_modules,
)
try:
    from torch.ao.quantization import fuse_modules_qat
except Exception:
    fuse_modules_qat = None

# ----------------------------------------------------------------------------
# PROOF CONFIGURATION
# ----------------------------------------------------------------------------
PROOF_SEED = 9999
PROOF_OUTPUT_DIR = "wsnds_deployment_qat_outputs"
WSNDS_PATH = "WSN-DS.csv"

KD_T_DEFAULT = 2
KD_ALPHA_DEFAULT = 0.5
KD_T_OVERRIDE = None
KD_ALPHA_OVERRIDE = None
KD_T = KD_T_OVERRIDE if KD_T_OVERRIDE is not None else KD_T_DEFAULT
KD_ALPHA = KD_ALPHA_OVERRIDE if KD_ALPHA_OVERRIDE is not None else KD_ALPHA_DEFAULT

TRAIN_CONFIG = {
    "epochs": 30,
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay": 1e-3,
    "patience": 8,
}

QAT_EPOCHS = 8
QAT_LR = 1e-4
QAT_PATIENCE = 4
QAT_BACKEND = "fbgemm"

LATENCY_WARMUP = 50
LATENCY_RUNS_B1 = 1000
LATENCY_RUNS_B64 = 300

STUDENT_A_HIDDEN = (32, 16)
STUDENT_B_HIDDEN = (64, 32)
TEACHER_HIDDEN = (128, 256, 128)

if QAT_BACKEND in torch.backends.quantized.supported_engines:
    torch.backends.quantized.engine = QAT_BACKEND

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("WSN-DS deployment/QAT proof route (v2.3-derived)")
print("This proof route is NOT a replacement for the completed 10-seed results.")
print(f"PROOF_SEED: {PROOF_SEED}")
print(f"KD defaults: T={KD_T}, alpha={KD_ALPHA}")
print(f"Device: {device}")
print(f"PyTorch: {torch.__version__}")

# ============================================================================
# CELL 3: Load WSN-DS dataset
# ============================================================================
# Upload WSN-DS.csv to the working directory before running this cell.

df = pd.read_csv(WSNDS_PATH)
print(f"Rows: {len(df)}, Cols: {len(df.columns)}")
print(f"Columns: {df.columns.tolist()}")
df.columns = df.columns.str.strip()
print(f"\nFirst row:\n{df.head(1).to_string()}")

# ============================================================================
# CELL 4: Preprocess WSN-DS (v2.3 exact)
# ============================================================================
# Identify target column (handle common variations)
target_candidates = ["Attack type", "Attack_Type", "attack_type", "Attack Type", "class"]
target_col = None
for cand in target_candidates:
    if cand in df.columns:
        target_col = cand
        break
if target_col is None:
    target_col = df.columns[-1]
print(f"Target column: {target_col}")

# Drop Id column (non-informative, can bias the model)
for id_col in ["id", "Id", "ID"]:
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
INPUT_DIM = len(X_all[0]) if len(X_all) > 0 else 0
print(f"Input dim: {INPUT_DIM}")
print(f"Class distribution: {dict(zip(CLASS_NAMES, np.bincount(y_all).tolist()))}")

# Standardize (fit on all data, then split — consistent with v2.3)
scaler = StandardScaler()
X_all_std = scaler.fit_transform(X_all)

# ============================================================================
# CELL 5: Model architectures (v2.3)
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
    """Compute FLOPs for a linear-ReLU MLP (one forward pass per sample)."""
    dims = [input_dim] + list(hidden_dims) + [num_classes]
    flops = 0
    for i in range(len(dims) - 1):
        flops += 2 * dims[i] * dims[i + 1]  # MAC operations
        flops += dims[i + 1]                # bias
        if i < len(dims) - 2:
            flops += dims[i + 1]            # ReLU
    return flops


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
# CELL 6: Training, quantization, and deployment utilities
# ============================================================================
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _batched_predict(model: nn.Module, X: torch.Tensor, batch_size: int = 4096,
                     device_override: torch.device = None):
    """Memory-safe batched inference returning predicted class indices."""
    model.eval()
    preds = []
    use_device = device_override if device_override is not None else device
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size].to(use_device)
            preds.append(model(batch).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def evaluate_model(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                   device_override: torch.device = None) -> dict:
    """Return dict of evaluation metrics."""
    preds = _batched_predict(model, X, device_override=device_override)
    y_np = y.cpu().numpy() if torch.is_tensor(y) else np.asarray(y)

    acc = accuracy_score(y_np, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_np, preds, average="macro", zero_division=0
    )
    per_class_prec, per_class_rec, per_class_f1_arr, _ = precision_recall_fscore_support(
        y_np, preds, average=None, zero_division=0
    )
    cm = confusion_matrix(y_np, preds)
    return {
        "accuracy": float(acc),
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1),
        "per_class_precision": per_class_prec.tolist(),
        "per_class_recall": per_class_rec.tolist(),
        "per_class_f1": per_class_f1_arr.tolist(),
        "confusion_matrix": cm.tolist(),
    }


def train_standard(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor,
                   X_val: torch.Tensor, y_val: torch.Tensor,
                   class_weights: torch.Tensor = None,
                   epochs: int = 30, batch_size: int = 256, lr: float = 1e-3,
                   weight_decay: float = 1e-3, patience: int = 8,
                   verbose: bool = False):
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

    best_val = 0.0
    best_state = None
    bad = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        preds = _batched_predict(model, X_val_d)
        val_f1 = f1_score(y_val_np, preds, average="macro")
        if val_f1 > best_val:
            best_val = val_f1
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

        if verbose:
            print(f"Epoch {epoch+1}: val_f1={val_f1:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_kd(student: nn.Module,
             teacher_source,
             X_train: torch.Tensor, y_train: torch.Tensor,
             X_val: torch.Tensor, y_val: torch.Tensor,
             T: float = KD_T, alpha: float = KD_ALPHA,
             class_weights: torch.Tensor = None,
             epochs: int = 30, batch_size: int = 256,
             lr: float = 1e-3, weight_decay: float = 1e-3,
             patience: int = 8, verbose: bool = False):
    """Knowledge distillation training (v2.3 KD formula)."""
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

    best_val = 0.0
    best_state = None
    bad = 0

    for epoch in range(epochs):
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

        preds = _batched_predict(student, X_val_d)
        val_f1 = f1_score(y_val_np, preds, average="macro")
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
    except Exception:
        try:
            from torch.ao.quantization import quantize_dynamic
            return quantize_dynamic(model_cpu, {nn.Linear}, dtype=torch.qint8)
        except Exception as exc:
            print(f"Dynamic INT8 quantization failed: {exc}")
            return model_cpu


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def model_state_size_kb(model: nn.Module, file_path: str) -> float:
    """Measure serialized model size by saving to disk (no temp dir)."""
    ensure_dir(os.path.dirname(file_path))
    torch.save(model.state_dict(), file_path)
    return os.path.getsize(file_path) / 1024


def serialize_rf_model(model, file_path: str) -> float:
    ensure_dir(os.path.dirname(file_path))
    if joblib is not None:
        joblib.dump(model, file_path)
    else:
        with open(file_path, "wb") as f:
            pickle.dump(model, f)
    return os.path.getsize(file_path) / 1024


def align_proba_to_class_order(proba: np.ndarray, classes, num_classes: int) -> np.ndarray:
    """Align RF predict_proba columns to the label-encoder class order."""
    class_list = list(classes)
    target_order = list(range(num_classes))
    if class_list == target_order:
        return proba
    order = [class_list.index(c) for c in target_order]
    return proba[:, order]


def get_first_linear_in_features(model: nn.Module):
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            return mod.in_features
    return None


def _latency_stats(times_ms: list, batch_size: int) -> dict:
    arr = np.asarray(times_ms, dtype=np.float64)
    mean_ms = float(arr.mean()) if arr.size else 0.0
    return {
        "mean_ms": mean_ms,
        "std_ms": float(arr.std()) if arr.size else 0.0,
        "p50_ms": float(np.percentile(arr, 50)) if arr.size else 0.0,
        "p95_ms": float(np.percentile(arr, 95)) if arr.size else 0.0,
        "p99_ms": float(np.percentile(arr, 99)) if arr.size else 0.0,
        "throughput_samples_per_s": float(batch_size * 1000.0 / mean_ms) if mean_ms > 0 else 0.0,
    }


def measure_latency_pytorch(model: nn.Module, X: torch.Tensor, batch_size: int,
                            warmup: int, runs: int) -> dict:
    model_cpu = copy.deepcopy(model).cpu().eval()
    input_dim = get_first_linear_in_features(model_cpu)
    if input_dim is not None and X.size(1) != input_dim:
        raise ValueError("Input dimension mismatch for latency benchmark")
    Xb = X[:batch_size].cpu()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model_cpu(Xb)
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            _ = model_cpu(Xb)
            times.append((time.perf_counter() - start) * 1000.0)
    return _latency_stats(times, batch_size)


def measure_latency_sklearn(model, X: np.ndarray, batch_size: int,
                            warmup: int, runs: int) -> dict:
    Xb = X[:batch_size]
    for _ in range(warmup):
        _ = model.predict_proba(Xb)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = model.predict_proba(Xb)
        times.append((time.perf_counter() - start) * 1000.0)
    return _latency_stats(times, batch_size)


def build_latency_block(stats_b1: dict, stats_b64: dict) -> dict:
    return {
        "latency_mean_ms_b1": stats_b1["mean_ms"],
        "latency_std_ms_b1": stats_b1["std_ms"],
        "latency_p50_ms_b1": stats_b1["p50_ms"],
        "latency_p95_ms_b1": stats_b1["p95_ms"],
        "latency_p99_ms_b1": stats_b1["p99_ms"],
        "throughput_samples_per_s_b1": stats_b1["throughput_samples_per_s"],
        "latency_mean_ms_b64": stats_b64["mean_ms"],
        "latency_std_ms_b64": stats_b64["std_ms"],
        "latency_p50_ms_b64": stats_b64["p50_ms"],
        "latency_p95_ms_b64": stats_b64["p95_ms"],
        "latency_p99_ms_b64": stats_b64["p99_ms"],
        "throughput_samples_per_s_b64": stats_b64["throughput_samples_per_s"],
    }


class QATStudentWrapper(nn.Module):
    def __init__(self, float_model: nn.Module):
        super().__init__()
        self.quant = QuantStub()
        self.model = copy.deepcopy(float_model)
        self.model.load_state_dict(float_model.state_dict())
        self.dequant = DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


def fuse_student_modules(wrapper: QATStudentWrapper) -> None:
    fusion = []
    layers = list(wrapper.model.net)
    for idx in range(len(layers) - 1):
        if isinstance(layers[idx], nn.Linear) and isinstance(layers[idx + 1], nn.ReLU):
            fusion.append([f"model.net.{idx}", f"model.net.{idx + 1}"])
    if fusion:
        if fuse_modules_qat is not None:
            fuse_modules_qat(wrapper, fusion, inplace=True)
        else:
            fuse_modules(wrapper, fusion, inplace=True)


def prepare_qat_student(float_student: nn.Module) -> QATStudentWrapper:
    wrapper = QATStudentWrapper(float_student)
    wrapper.qconfig = get_default_qat_qconfig(QAT_BACKEND)
    fuse_student_modules(wrapper)
    prepare_qat(wrapper, inplace=True)
    return wrapper


def train_qat(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor,
              X_val: torch.Tensor, y_val: torch.Tensor,
              class_weights: torch.Tensor,
              epochs: int, lr: float, patience: int) -> nn.Module:
    model = model.cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=TRAIN_CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    class_weights = class_weights.cpu()
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_ds = TensorDataset(X_train.cpu(), y_train.cpu())
    loader = DataLoader(train_ds, batch_size=TRAIN_CONFIG["batch_size"], shuffle=True)
    X_val_cpu = X_val.cpu()
    y_val_np = y_val.cpu().numpy()

    best_val = 0.0
    best_state = None
    bad = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        preds = _batched_predict(model, X_val_cpu, device_override=torch.device("cpu"))
        val_f1 = f1_score(y_val_np, preds, average="macro")
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
    return model


def environment_info() -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "proof_seed": PROOF_SEED,
        "qat_backend": QAT_BACKEND if QAT_BACKEND in torch.backends.quantized.supported_engines else None,
    }

# ============================================================================
# CELL 7: Deterministic deployment + QAT proof route
# ============================================================================
set_seed(PROOF_SEED)

output_dir = Path(PROOF_OUTPUT_DIR)
tmp_dir = output_dir / "tmp"
ensure_dir(str(output_dir))
ensure_dir(str(tmp_dir))

def _array_dims(arr: np.ndarray) -> tuple:
    rows = len(arr)
    cols = len(arr[0]) if rows > 0 else 0
    return rows, cols


X_trainval, X_test_np, y_trainval, y_test_np = train_test_split(
    X_all_std, y_all, test_size=0.15, random_state=42, stratify=y_all
)
X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
    X_trainval, y_trainval, test_size=0.1765, random_state=42, stratify=y_trainval
)
tr_dims = _array_dims(X_train_np)
val_dims = _array_dims(X_val_np)
te_dims = _array_dims(X_test_np)
print(f"Train: {tr_dims}, Val: {val_dims}, Test: {te_dims}")

X_train_t = torch.tensor(X_train_np, dtype=torch.float32)
y_train_t = torch.tensor(y_train_np, dtype=torch.long)
X_val_t = torch.tensor(X_val_np, dtype=torch.float32)
y_val_t = torch.tensor(y_val_np, dtype=torch.long)
X_test_t = torch.tensor(X_test_np, dtype=torch.float32)
y_test_t = torch.tensor(y_test_np, dtype=torch.long)

counts = np.bincount(y_train_np, minlength=NUM_CLASSES)
class_weights = torch.tensor(
    len(y_train_np) / (NUM_CLASSES * np.maximum(counts, 1)), dtype=torch.float32
)

# ---- A: RF teacher baseline ----
rf = RandomForestClassifier(
    n_estimators=500, max_depth=15, random_state=PROOF_SEED, n_jobs=-1
)
rf.fit(X_train_np, y_train_np)
rf_preds = rf.predict(X_test_np)
rf_probs_test = rf.predict_proba(X_test_np)
rf_acc = accuracy_score(y_test_np, rf_preds)
rf_prec, rf_rec, rf_f1, _ = precision_recall_fscore_support(
    y_test_np, rf_preds, average="macro", zero_division=0
)
rf_per_class_f1 = f1_score(y_test_np, rf_preds, average=None, zero_division=0)
rf_serialized_path = output_dir / "A_RF_500.pkl"
rf_serialized_kb = serialize_rf_model(rf, str(rf_serialized_path))

# Calibrated RF for KD (v2.3 Config E style)
rf_calib = CalibratedClassifierCV(
    RandomForestClassifier(
        n_estimators=500, max_depth=15, random_state=PROOF_SEED, n_jobs=-1
    ),
    method="isotonic",
    cv=3,
)
rf_calib.fit(X_train_np, y_train_np)
rf_soft = align_proba_to_class_order(
    rf_calib.predict_proba(X_train_np), rf_calib.classes_, NUM_CLASSES
)
rf_soft_t = torch.tensor(rf_soft, dtype=torch.float32)

# ---- B: Full MLP teacher ----
teacher_full = TeacherMLP(INPUT_DIM, NUM_CLASSES)
teacher_full = train_standard(
    teacher_full, X_train_t, y_train_t, X_val_t, y_val_t,
    class_weights=class_weights, **TRAIN_CONFIG
)

# ---- D/E: Students (A and B) ----
student_a_scratch = StudentMLP(INPUT_DIM, STUDENT_A_HIDDEN, NUM_CLASSES)
student_a_scratch = train_standard(
    student_a_scratch, X_train_t, y_train_t, X_val_t, y_val_t,
    class_weights=class_weights, **TRAIN_CONFIG
)

student_a_kd = StudentMLP(INPUT_DIM, STUDENT_A_HIDDEN, NUM_CLASSES)
student_a_kd = train_kd(
    student_a_kd, rf_soft_t, X_train_t, y_train_t, X_val_t, y_val_t,
    T=KD_T, alpha=KD_ALPHA, class_weights=class_weights,
    epochs=TRAIN_CONFIG["epochs"], batch_size=TRAIN_CONFIG["batch_size"]
)

student_b_scratch = StudentMLP(INPUT_DIM, STUDENT_B_HIDDEN, NUM_CLASSES)
student_b_scratch = train_standard(
    student_b_scratch, X_train_t, y_train_t, X_val_t, y_val_t,
    class_weights=class_weights, **TRAIN_CONFIG
)

student_b_kd = StudentMLP(INPUT_DIM, STUDENT_B_HIDDEN, NUM_CLASSES)
student_b_kd = train_kd(
    student_b_kd, rf_soft_t, X_train_t, y_train_t, X_val_t, y_val_t,
    T=KD_T, alpha=KD_ALPHA, class_weights=class_weights,
    epochs=TRAIN_CONFIG["epochs"], batch_size=TRAIN_CONFIG["batch_size"]
)

# ---- Metrics, sizes, quantization, and latency ----
fp32_models = {
    "B_Full_MLP": {
        "model": teacher_full,
        "arch": "MLP 128-256-128",
        "hidden": TEACHER_HIDDEN,
    },
    "D_student_A_scratch": {
        "model": student_a_scratch,
        "arch": "Student MLP (32, 16)",
        "hidden": STUDENT_A_HIDDEN,
    },
    "E_student_A_KD_from_RF": {
        "model": student_a_kd,
        "arch": "Student MLP (32, 16)",
        "hidden": STUDENT_A_HIDDEN,
    },
    "D_student_B_scratch": {
        "model": student_b_scratch,
        "arch": "Student MLP (64, 32)",
        "hidden": STUDENT_B_HIDDEN,
    },
    "E_student_B_KD_from_RF": {
        "model": student_b_kd,
        "arch": "Student MLP (64, 32)",
        "hidden": STUDENT_B_HIDDEN,
    },
}

fp32_metrics = {}
fp32_sizes = {}
fp32_macro_f1 = {}
summary_rows = []

# RF latency
rf_latency_b1 = measure_latency_sklearn(rf, X_test_np, 1, LATENCY_WARMUP, LATENCY_RUNS_B1)
rf_latency_b64 = measure_latency_sklearn(rf, X_test_np, 64, LATENCY_WARMUP, LATENCY_RUNS_B64)
rf_latency_block = build_latency_block(rf_latency_b1, rf_latency_b64)

# Record RF baseline row
summary_rows.append({
    "model_name": "A_RF_500",
    "architecture": "RandomForest 500 (max_depth=15)",
    "variant": "rf_sklearn",
    "accuracy": float(rf_acc),
    "macro_precision": float(rf_prec),
    "macro_recall": float(rf_rec),
    "macro_f1": float(rf_f1),
    "macro_f1_delta_vs_fp32": None,
    "per_class_f1": rf_per_class_f1.tolist(),
    "params": None,
    "flops_per_sample": None,
    "theoretical_fp32_size_kb": None,
    "serialized_size_kb": rf_serialized_kb,
    "compression_ratio_vs_rf": 1.0,
    "compression_ratio_vs_full_mlp_fp32": None,
    **rf_latency_block,
    "qat_backend": None,
    "proof_seed": PROOF_SEED,
})

# FP32 MLP variants
for name, info in fp32_models.items():
    metrics = evaluate_model(info["model"], X_test_t, y_test_t)
    fp32_metrics[name] = metrics
    fp32_macro_f1[name] = metrics["macro_f1"]

    params = count_params(info["model"])
    flops = compute_flops_mlp(INPUT_DIM, info["hidden"], NUM_CLASSES)
    theoretical_size = model_size_kb(info["model"])
    size_path = tmp_dir / f"{name}_fp32.pt"
    serialized_kb = model_state_size_kb(info["model"], str(size_path))
    fp32_sizes[name] = serialized_kb

    latency_b1 = measure_latency_pytorch(info["model"], X_test_t, 1, LATENCY_WARMUP, LATENCY_RUNS_B1)
    latency_b64 = measure_latency_pytorch(info["model"], X_test_t, 64, LATENCY_WARMUP, LATENCY_RUNS_B64)
    latency_block = build_latency_block(latency_b1, latency_b64)

    summary_rows.append({
        "model_name": name,
        "architecture": info["arch"],
        "variant": "fp32",
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "macro_f1_delta_vs_fp32": 0.0,
        "per_class_f1": metrics["per_class_f1"],
        "params": params,
        "flops_per_sample": flops,
        "theoretical_fp32_size_kb": theoretical_size,
        "serialized_size_kb": serialized_kb,
        "compression_ratio_vs_rf": rf_serialized_kb / serialized_kb if serialized_kb else None,
        "compression_ratio_vs_full_mlp_fp32": None,
        **latency_block,
        "qat_backend": None,
        "proof_seed": PROOF_SEED,
    })

full_mlp_fp32_size = fp32_sizes.get("B_Full_MLP", None)

# Dynamic INT8
for name, info in fp32_models.items():
    dyn_model = quantize_dynamic_int8(info["model"])
    metrics = evaluate_model(dyn_model, X_test_t.cpu(), y_test_t.cpu(), device_override=torch.device("cpu"))

    params = count_params(info["model"])
    flops = compute_flops_mlp(INPUT_DIM, info["hidden"], NUM_CLASSES)
    theoretical_size = model_size_kb(info["model"])
    size_path = tmp_dir / f"{name}_dynamic_int8.pt"
    serialized_kb = model_state_size_kb(dyn_model, str(size_path))

    latency_b1 = measure_latency_pytorch(dyn_model, X_test_t, 1, LATENCY_WARMUP, LATENCY_RUNS_B1)
    latency_b64 = measure_latency_pytorch(dyn_model, X_test_t, 64, LATENCY_WARMUP, LATENCY_RUNS_B64)
    latency_block = build_latency_block(latency_b1, latency_b64)

    summary_rows.append({
        "model_name": name,
        "architecture": info["arch"],
        "variant": "dynamic_int8",
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "macro_f1_delta_vs_fp32": metrics["macro_f1"] - fp32_macro_f1.get(name, metrics["macro_f1"]),
        "per_class_f1": metrics["per_class_f1"],
        "params": params,
        "flops_per_sample": flops,
        "theoretical_fp32_size_kb": theoretical_size,
        "serialized_size_kb": serialized_kb,
        "compression_ratio_vs_rf": rf_serialized_kb / serialized_kb if serialized_kb else None,
        "compression_ratio_vs_full_mlp_fp32": full_mlp_fp32_size / serialized_kb if full_mlp_fp32_size and serialized_kb else None,
        **latency_block,
        "qat_backend": None,
        "proof_seed": PROOF_SEED,
    })

# QAT (students only)
qat_students = {
    "D_student_A_scratch": student_a_scratch,
    "E_student_A_KD_from_RF": student_a_kd,
    "D_student_B_scratch": student_b_scratch,
    "E_student_B_KD_from_RF": student_b_kd,
}

for name, float_student in qat_students.items():
    qat_model = prepare_qat_student(float_student)
    qat_model = train_qat(
        qat_model, X_train_t, y_train_t, X_val_t, y_val_t,
        class_weights=class_weights,
        epochs=QAT_EPOCHS, lr=QAT_LR, patience=QAT_PATIENCE,
    )
    qat_int8 = convert(qat_model.eval(), inplace=False)

    info = fp32_models[name]
    metrics = evaluate_model(qat_int8, X_test_t.cpu(), y_test_t.cpu(), device_override=torch.device("cpu"))

    params = count_params(info["model"])
    flops = compute_flops_mlp(INPUT_DIM, info["hidden"], NUM_CLASSES)
    theoretical_size = model_size_kb(info["model"])
    size_path = tmp_dir / f"{name}_qat_int8.pt"
    serialized_kb = model_state_size_kb(qat_int8, str(size_path))

    latency_b1 = measure_latency_pytorch(qat_int8, X_test_t, 1, LATENCY_WARMUP, LATENCY_RUNS_B1)
    latency_b64 = measure_latency_pytorch(qat_int8, X_test_t, 64, LATENCY_WARMUP, LATENCY_RUNS_B64)
    latency_block = build_latency_block(latency_b1, latency_b64)

    summary_rows.append({
        "model_name": name,
        "architecture": info["arch"],
        "variant": "qat_int8",
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "macro_f1_delta_vs_fp32": metrics["macro_f1"] - fp32_macro_f1.get(name, metrics["macro_f1"]),
        "per_class_f1": metrics["per_class_f1"],
        "params": params,
        "flops_per_sample": flops,
        "theoretical_fp32_size_kb": theoretical_size,
        "serialized_size_kb": serialized_kb,
        "compression_ratio_vs_rf": rf_serialized_kb / serialized_kb if serialized_kb else None,
        "compression_ratio_vs_full_mlp_fp32": full_mlp_fp32_size / serialized_kb if full_mlp_fp32_size and serialized_kb else None,
        **latency_block,
        "qat_backend": QAT_BACKEND,
        "proof_seed": PROOF_SEED,
    })

# Fill in compression vs full MLP for fp32 rows once size is known
for row in summary_rows:
    if row["variant"] == "fp32" and row["model_name"] in fp32_sizes:
        if full_mlp_fp32_size and row["serialized_size_kb"]:
            row["compression_ratio_vs_full_mlp_fp32"] = full_mlp_fp32_size / row["serialized_size_kb"]

SUMMARY_COLUMNS = [
    "model_name",
    "architecture",
    "variant",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "macro_f1_delta_vs_fp32",
    "per_class_f1",
    "params",
    "flops_per_sample",
    "theoretical_fp32_size_kb",
    "serialized_size_kb",
    "compression_ratio_vs_rf",
    "compression_ratio_vs_full_mlp_fp32",
    "latency_p50_ms_b1",
    "latency_p95_ms_b1",
    "latency_p99_ms_b1",
    "latency_mean_ms_b1",
    "latency_std_ms_b1",
    "throughput_samples_per_s_b1",
    "latency_p50_ms_b64",
    "latency_p95_ms_b64",
    "latency_p99_ms_b64",
    "latency_mean_ms_b64",
    "latency_std_ms_b64",
    "throughput_samples_per_s_b64",
    "qat_backend",
    "proof_seed",
]

summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df[SUMMARY_COLUMNS]

summary_csv = output_dir / "wsnds_deployment_summary.csv"
summary_df.to_csv(summary_csv, index=False)
print(f"Saved {summary_csv}")

results_json = output_dir / "wsnds_deployment_results.json"
with open(results_json, "w") as f:
    json.dump(
        {
            "proof_seed": PROOF_SEED,
            "kd_hyperparameters": {"T": KD_T, "alpha": KD_ALPHA},
            "results": summary_rows,
            "environment": environment_info(),
        },
        f,
        indent=2,
    )
print(f"Saved {results_json}")

qat_df = summary_df[summary_df["variant"] == "qat_int8"]
qat_csv = output_dir / "wsnds_qat_summary.csv"
qat_df.to_csv(qat_csv, index=False)
print(f"Saved {qat_csv}")

latency_cols = [
    "model_name",
    "variant",
    "latency_p50_ms_b1",
    "latency_p95_ms_b1",
    "latency_p99_ms_b1",
    "latency_p50_ms_b64",
    "latency_p95_ms_b64",
    "latency_p99_ms_b64",
    "throughput_samples_per_s_b1",
    "throughput_samples_per_s_b64",
]
latency_df = summary_df[latency_cols]
latency_csv = output_dir / "wsnds_latency_summary.csv"
latency_df.to_csv(latency_csv, index=False)
print(f"Saved {latency_csv}")

env_json = output_dir / "wsnds_environment.json"
with open(env_json, "w") as f:
    json.dump(environment_info(), f, indent=2)
print(f"Saved {env_json}")

print("\nDeployment/QAT proof route complete.")
print("Outputs are in:", output_dir)
