# ============================================================================
#  !!!! WARNING - READ BEFORE EDITING !!!!
#
#  Runtime-only deployment benchmark source. The matching notebook is
#  `cukd_xai_wsnds_runtime_from_existing.ipynb`.
# ============================================================================

# ============================================================================
# CuKD-XAI WSN-DS Runtime Benchmarks From Existing Deployment Artifacts
#
# - Loads already-completed fp32 student `.pt` artifacts.
# - Does not retrain RF, KD, co-distillation, QAT, or any neural model.
# - Exports selected students to ONNX.
# - Benchmarks ONNX Runtime FP32, ONNX Runtime dynamic INT8, and OpenVINO CPU.
# ============================================================================

# ============================================================================
# CELL 1: Install dependencies
# ============================================================================
# Optional packages used by this runtime-only route:
# !pip install -q onnx onnxruntime openvino

# ============================================================================
# CELL 2: Imports and configuration
# ============================================================================
import importlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    REPO_ROOT = Path(__file__).resolve().parents[3]
except NameError:
    REPO_ROOT = Path.cwd()

EXISTING_DEPLOYMENT_OUTPUT_DIR = os.environ.get(
    "EXISTING_DEPLOYMENT_OUTPUT_DIR",
    str(REPO_ROOT / "results" / "runtime" / "onnx_openvino" / "wsnds"),
)
RUNTIME_OUTPUT_DIR = os.environ.get("RUNTIME_OUTPUT_DIR", "")
WSNDS_PATH = os.environ.get("WSNDS_PATH", str(REPO_ROOT / "data" / "wsnds" / "WSN-DS.csv"))
INSTALL_OPTIONAL_DEPLOYMENT_DEPS = True
ENABLE_ONNX_BENCHMARKS = True
ENABLE_OPENVINO_BENCHMARKS = True
ONNX_OPSET_VERSION = 17
LATENCY_WARMUP = 50
LATENCY_RUNS_B1 = 1000
LATENCY_RUNS_B64 = 300

STUDENT_A_HIDDEN = (32, 16)
STUDENT_B_HIDDEN = (64, 32)
NUM_CLASSES_EXPECTED = 5
INPUT_DIM_EXPECTED = 17

MODEL_ARTIFACTS = {
    "D_student_A_scratch": {
        "hidden": STUDENT_A_HIDDEN,
        "artifact": "D_student_A_scratch_fp32.pt",
    },
    "E_student_A_KD_from_RF": {
        "hidden": STUDENT_A_HIDDEN,
        "artifact": "E_student_A_KD_from_RF_fp32.pt",
    },
    "J_student_A_CoDistill_RF_CL": {
        "hidden": STUDENT_A_HIDDEN,
        "artifact": "J_student_A_CoDistill_RF_CL_fp32.pt",
    },
    "D_student_B_scratch": {
        "hidden": STUDENT_B_HIDDEN,
        "artifact": "D_student_B_scratch_fp32.pt",
    },
    "E_student_B_KD_from_RF": {
        "hidden": STUDENT_B_HIDDEN,
        "artifact": "E_student_B_KD_from_RF_fp32.pt",
    },
    "J_student_B_CoDistill_RF_CL": {
        "hidden": STUDENT_B_HIDDEN,
        "artifact": "J_student_B_CoDistill_RF_CL_fp32.pt",
    },
}
DEPLOYMENT_BENCHMARK_MODELS = list(MODEL_ARTIFACTS.keys())

print("Runtime-only benchmark from existing deployment artifacts")
print(f"Existing output dir setting: {EXISTING_DEPLOYMENT_OUTPUT_DIR}")
print(f"WSNDS_PATH setting: {WSNDS_PATH}")

# ============================================================================
# CELL 3: Model definition and path resolution
# ============================================================================
class StudentMLP(nn.Module):
    """v2.3 student MLP. No BatchNorm, Linear-ReLU stack."""
    def __init__(self, input_dim: int = 17, hidden_dims: tuple = (32, 16),
                 num_classes: int = 5):
        super().__init__()
        layers = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            prev = hidden
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _unique_paths(paths):
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


def _strip_single_parent(path: Path) -> Path:
    parts = path.parts
    if parts and parts[0] == "..":
        return Path(*parts[1:]) if len(parts) > 1 else Path(".")
    return path


def candidate_paths(path_text: str):
    configured = Path(os.path.expanduser(str(path_text)))
    if configured.is_absolute():
        return [configured]
    cwd = Path.cwd()
    stripped = _strip_single_parent(configured)
    candidates = [cwd / configured, cwd / stripped]
    try:
        script_dir = Path(__file__).resolve().parent
        repo_root = script_dir.parents[2]
        candidates.extend([
            script_dir / configured,
            script_dir / stripped,
            script_dir.parent / stripped,
            script_dir.parent / configured,
            repo_root / configured,
            repo_root / stripped,
        ])
    except NameError:
        pass
    candidates.extend([
        cwd / "results" / "runtime" / "onnx_openvino" / "wsnds",
        REPO_ROOT / "results" / "runtime" / "onnx_openvino" / "wsnds",
        cwd / "Final" / "wsnds_deployment_qat_outputs",
        cwd / "wsnds_deployment_qat_outputs",
    ])
    return _unique_paths(candidates)


def resolve_existing_output_dir(path_text: str = EXISTING_DEPLOYMENT_OUTPUT_DIR) -> Path:
    for candidate in candidate_paths(path_text):
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()
    searched = "\n".join(str(path) for path in candidate_paths(path_text))
    raise FileNotFoundError(
        "Existing deployment output folder not found. It should contain tmp/*_fp32.pt. "
        f"Set EXISTING_DEPLOYMENT_OUTPUT_DIR if needed. Searched:\n{searched}"
    )


def resolve_file(path_text: str, extra_candidates=None) -> Path:
    candidates = candidate_paths(path_text)
    if extra_candidates:
        candidates.extend(extra_candidates)
    for candidate in _unique_paths(candidates):
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    searched = "\n".join(str(path) for path in _unique_paths(candidates))
    raise FileNotFoundError(f"File not found. Searched:\n{searched}")


def validate_required_artifacts(tmp_dir: Path) -> None:
    missing = []
    for spec in MODEL_ARTIFACTS.values():
        artifact_path = tmp_dir / spec["artifact"]
        if not artifact_path.exists():
            missing.append(str(artifact_path))
    if missing:
        missing_text = "\n".join(missing)
        raise FileNotFoundError(
            "Missing required fp32 artifacts. Run or copy completed deployment/QAT outputs first:\n"
            f"{missing_text}"
        )


def runtime_output_dir(existing_output_dir: Path) -> Path:
    if RUNTIME_OUTPUT_DIR:
        out = Path(RUNTIME_OUTPUT_DIR)
        return out if out.is_absolute() else (Path.cwd() / out).resolve()
    return existing_output_dir / "runtime_from_existing_outputs"


existing_output_dir = resolve_existing_output_dir()
existing_tmp_dir = existing_output_dir / "tmp"
validate_required_artifacts(existing_tmp_dir)
output_dir = runtime_output_dir(existing_output_dir)
artifact_dir = output_dir / "deployable_runtime_artifacts"
output_dir.mkdir(parents=True, exist_ok=True)
artifact_dir.mkdir(parents=True, exist_ok=True)

print(f"Resolved existing deployment outputs: {existing_output_dir}")
print(f"Runtime-only outputs: {output_dir}")

# ============================================================================
# CELL 4: Load WSN-DS test split only
# ============================================================================
wsnds_file = resolve_file(
    WSNDS_PATH,
    extra_candidates=[
        REPO_ROOT / "data" / "wsnds" / "WSN-DS.csv",
        existing_output_dir.parent / "WSN-DS.csv",
        existing_output_dir.parent / "Relevant Files" / "WSN-DS.csv",
        Path.cwd() / "WSN-DS.csv",
    ],
)
print(f"Resolved WSN-DS CSV: {wsnds_file}")

df = pd.read_csv(wsnds_file)
df.columns = df.columns.str.strip()
target_candidates = ["Attack type", "Attack_Type", "attack_type", "Attack Type", "class"]
target_col = next((candidate for candidate in target_candidates if candidate in df.columns), df.columns[-1])
for id_col in ["id", "Id", "ID"]:
    if id_col in df.columns:
        df = df.drop(id_col, axis=1)
        break

df[target_col] = df[target_col].astype(str).str.strip()
label_encoder = LabelEncoder()
df[target_col] = label_encoder.fit_transform(df[target_col])
CLASS_NAMES = label_encoder.classes_.tolist()
NUM_CLASSES = len(CLASS_NAMES)
X_all = df.drop(target_col, axis=1).values.astype(np.float32)
y_all = df[target_col].values.astype(np.int64)
INPUT_DIM = X_all.shape[1]
if INPUT_DIM != INPUT_DIM_EXPECTED or NUM_CLASSES != NUM_CLASSES_EXPECTED:
    raise RuntimeError(
        f"Unexpected WSN-DS shape/classes: input_dim={INPUT_DIM}, num_classes={NUM_CLASSES}"
    )
scaler = StandardScaler()
X_all_std = scaler.fit_transform(X_all)
_, X_test_np, _, y_test_np = train_test_split(
    X_all_std, y_all, test_size=0.15, random_state=42, stratify=y_all
)
print(f"Test split: {X_test_np.shape}, classes={CLASS_NAMES}")

# ============================================================================
# CELL 5: Runtime benchmark utilities
# ============================================================================
def try_import_optional_deployment_modules() -> dict:
    optional_specs = {"onnx": "onnx", "onnxruntime": "onnxruntime", "openvino": "openvino"}
    status = {}
    modules = {}
    for module_name, package_name in optional_specs.items():
        try:
            modules[module_name] = importlib.import_module(module_name)
            status[f"{module_name}_available"] = True
            status[f"{module_name}_error"] = None
            continue
        except Exception as first_exc:
            if INSTALL_OPTIONAL_DEPLOYMENT_DEPS:
                try:
                    print(f"Installing optional deployment package: {package_name}")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package_name])
                    modules[module_name] = importlib.import_module(module_name)
                    status[f"{module_name}_available"] = True
                    status[f"{module_name}_error"] = None
                    continue
                except Exception as install_exc:
                    status[f"{module_name}_available"] = False
                    status[f"{module_name}_error"] = f"{type(install_exc).__name__}: {install_exc}"
            else:
                status[f"{module_name}_available"] = False
                status[f"{module_name}_error"] = f"{type(first_exc).__name__}: {first_exc}"
            modules[module_name] = None
    status["onnx_available"] = bool(status.get("onnx_available"))
    status["onnxruntime_available"] = bool(status.get("onnxruntime_available"))
    status["openvino_available"] = bool(status.get("openvino_available"))
    return {"modules": modules, "optional_dependency_status": status}


def load_existing_student_model(model_name: str) -> StudentMLP:
    spec = MODEL_ARTIFACTS[model_name]
    artifact_path = existing_tmp_dir / spec["artifact"]
    if not artifact_path.exists():
        raise FileNotFoundError(f"Missing existing fp32 artifact: {artifact_path}")
    model = StudentMLP(INPUT_DIM, spec["hidden"], NUM_CLASSES)
    try:
        state = torch.load(artifact_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(artifact_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate_numpy_predictions(preds: np.ndarray, y_true: np.ndarray) -> dict:
    acc = accuracy_score(y_true, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": float(acc),
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1),
    }


def latency_stats(times_ms: list, batch_size: int) -> dict:
    arr = np.asarray(times_ms, dtype=np.float64)
    mean_ms = float(arr.mean()) if arr.size else 0.0
    return {
        f"latency_mean_ms_b{batch_size}": mean_ms,
        f"latency_std_ms_b{batch_size}": float(arr.std()) if arr.size else 0.0,
        f"latency_p50_ms_b{batch_size}": float(np.percentile(arr, 50)) if arr.size else 0.0,
        f"latency_p95_ms_b{batch_size}": float(np.percentile(arr, 95)) if arr.size else 0.0,
        f"latency_p99_ms_b{batch_size}": float(np.percentile(arr, 99)) if arr.size else 0.0,
        f"throughput_samples_per_s_b{batch_size}": float(batch_size * 1000.0 / mean_ms) if mean_ms > 0 else 0.0,
    }


def export_student_to_onnx(model: nn.Module, output_path: Path) -> float:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_cpu = model.cpu().eval()
    dummy = torch.randn(1, INPUT_DIM, dtype=torch.float32)
    torch.onnx.export(
        model_cpu,
        dummy,
        str(output_path),
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=ONNX_OPSET_VERSION,
    )
    return output_path.stat().st_size / 1024


def quantize_onnx_dynamic(onnx_path: Path, quantized_path: Path) -> float:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantized_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(onnx_path), str(quantized_path), weight_type=QuantType.QInt8)
    return quantized_path.stat().st_size / 1024


def onnxruntime_predict(session, X_np: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    preds = []
    for start in range(0, len(X_np), batch_size):
        xb = X_np[start:start + batch_size].astype(np.float32, copy=False)
        logits = session.run(None, {input_name: xb})[0]
        preds.append(np.argmax(logits, axis=1))
    return np.concatenate(preds)


def measure_onnxruntime_latency(session, X_np: np.ndarray, batch_size: int,
                                warmup: int, runs: int) -> dict:
    input_name = session.get_inputs()[0].name
    Xb = X_np[:batch_size].astype(np.float32, copy=False)
    for _ in range(warmup):
        session.run(None, {input_name: Xb})
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, {input_name: Xb})
        times.append((time.perf_counter() - start) * 1000.0)
    return latency_stats(times, batch_size)


def benchmark_onnxruntime_model(onnx_path: Path, X_np: np.ndarray, y_np: np.ndarray) -> dict:
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    metrics = evaluate_numpy_predictions(onnxruntime_predict(session, X_np), y_np)
    return {
        **metrics,
        **measure_onnxruntime_latency(session, X_np, 1, LATENCY_WARMUP, LATENCY_RUNS_B1),
        **measure_onnxruntime_latency(session, X_np, 64, LATENCY_WARMUP, LATENCY_RUNS_B64),
    }


def openvino_output_array(infer_result):
    if isinstance(infer_result, dict):
        return next(iter(infer_result.values()))
    if isinstance(infer_result, (list, tuple)):
        return infer_result[0]
    return infer_result


def openvino_predict(compiled_model, X_np: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    preds = []
    for start in range(0, len(X_np), batch_size):
        xb = X_np[start:start + batch_size].astype(np.float32, copy=False)
        logits = openvino_output_array(compiled_model([xb]))
        preds.append(np.argmax(np.asarray(logits), axis=1))
    return np.concatenate(preds)


def measure_openvino_latency(compiled_model, X_np: np.ndarray, batch_size: int,
                             warmup: int, runs: int) -> dict:
    Xb = X_np[:batch_size].astype(np.float32, copy=False)
    for _ in range(warmup):
        compiled_model([Xb])
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        compiled_model([Xb])
        times.append((time.perf_counter() - start) * 1000.0)
    return latency_stats(times, batch_size)


def benchmark_openvino_model(onnx_path: Path, X_np: np.ndarray, y_np: np.ndarray) -> dict:
    import openvino as ov
    core = ov.Core()
    ov_model = core.read_model(str(onnx_path))
    compiled_model = core.compile_model(ov_model, "CPU")
    metrics = evaluate_numpy_predictions(openvino_predict(compiled_model, X_np), y_np)
    return {
        **metrics,
        **measure_openvino_latency(compiled_model, X_np, 1, LATENCY_WARMUP, LATENCY_RUNS_B1),
        **measure_openvino_latency(compiled_model, X_np, 64, LATENCY_WARMUP, LATENCY_RUNS_B64),
    }


def record_runtime_benchmark_skip(rows: list, model_name: str, runtime: str,
                                  variant: str, reason: str) -> None:
    rows.append({
        "model_name": model_name,
        "runtime": runtime,
        "variant": variant,
        "status": "skipped",
        "skip_reason": reason,
        "accuracy": None,
        "macro_precision": None,
        "macro_recall": None,
        "macro_f1": None,
        "serialized_size_kb": None,
        "latency_p50_ms_b1": None,
        "latency_p95_ms_b1": None,
        "latency_p99_ms_b1": None,
        "latency_p50_ms_b64": None,
        "latency_p95_ms_b64": None,
        "latency_p99_ms_b64": None,
        "throughput_samples_per_s_b1": None,
        "throughput_samples_per_s_b64": None,
        "artifact_path": None,
    })


def append_runtime_benchmark_row(rows: list, model_name: str, runtime: str,
                                 variant: str, metrics: dict,
                                 artifact_path: Path, serialized_size_kb: float) -> None:
    row = {
        "model_name": model_name,
        "runtime": runtime,
        "variant": variant,
        "status": "ok",
        "skip_reason": None,
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "serialized_size_kb": serialized_size_kb,
        "artifact_path": str(artifact_path),
    }
    for key, value in metrics.items():
        if key.startswith("latency_") or key.startswith("throughput_"):
            row[key] = value
    rows.append(row)

# ============================================================================
# CELL 6: Run ONNX Runtime and OpenVINO benchmarks from existing artifacts
# ============================================================================
def run_existing_artifact_runtime_benchmarks() -> tuple:
    optional = try_import_optional_deployment_modules()
    optional_dependency_status = optional["optional_dependency_status"]
    rows = []
    onnx_ready = ENABLE_ONNX_BENCHMARKS and optional_dependency_status.get("onnx_available")
    ort_ready = onnx_ready and optional_dependency_status.get("onnxruntime_available")
    ov_ready = ENABLE_OPENVINO_BENCHMARKS and onnx_ready and optional_dependency_status.get("openvino_available")

    for model_name in DEPLOYMENT_BENCHMARK_MODELS:
        try:
            model = load_existing_student_model(model_name)
        except Exception as exc:
            for runtime, variant in [
                ("onnxruntime", "onnx_fp32"),
                ("onnxruntime", "onnx_dynamic_int8"),
                ("openvino", "openvino_fp32_from_onnx"),
            ]:
                record_runtime_benchmark_skip(rows, model_name, runtime, variant,
                                              f"artifact_load_failed: {type(exc).__name__}: {exc}")
            continue

        onnx_path = artifact_dir / f"{model_name}.onnx"
        if not onnx_ready:
            for runtime, variant in [
                ("onnxruntime", "onnx_fp32"),
                ("onnxruntime", "onnx_dynamic_int8"),
                ("openvino", "openvino_fp32_from_onnx"),
            ]:
                record_runtime_benchmark_skip(rows, model_name, runtime, variant, "missing_dependency")
            continue

        try:
            onnx_size_kb = export_student_to_onnx(model, onnx_path)
        except Exception as exc:
            for runtime, variant in [
                ("onnxruntime", "onnx_fp32"),
                ("onnxruntime", "onnx_dynamic_int8"),
                ("openvino", "openvino_fp32_from_onnx"),
            ]:
                record_runtime_benchmark_skip(rows, model_name, runtime, variant,
                                              f"onnx_export_failed: {type(exc).__name__}: {exc}")
            continue

        if ort_ready:
            try:
                metrics = benchmark_onnxruntime_model(onnx_path, X_test_np, y_test_np)
                append_runtime_benchmark_row(rows, model_name, "onnxruntime", "onnx_fp32",
                                             metrics, onnx_path, onnx_size_kb)
            except Exception as exc:
                record_runtime_benchmark_skip(rows, model_name, "onnxruntime", "onnx_fp32",
                                              f"benchmark_failed: {type(exc).__name__}: {exc}")

            q_path = artifact_dir / f"{model_name}_dynamic_int8.onnx"
            try:
                q_size_kb = quantize_onnx_dynamic(onnx_path, q_path)
                metrics = benchmark_onnxruntime_model(q_path, X_test_np, y_test_np)
                append_runtime_benchmark_row(rows, model_name, "onnxruntime", "onnx_dynamic_int8",
                                             metrics, q_path, q_size_kb)
            except Exception as exc:
                record_runtime_benchmark_skip(rows, model_name, "onnxruntime", "onnx_dynamic_int8",
                                              f"quant_or_benchmark_failed: {type(exc).__name__}: {exc}")
        else:
            record_runtime_benchmark_skip(rows, model_name, "onnxruntime", "onnx_fp32", "missing_dependency")
            record_runtime_benchmark_skip(rows, model_name, "onnxruntime", "onnx_dynamic_int8", "missing_dependency")

        if ov_ready:
            try:
                metrics = benchmark_openvino_model(onnx_path, X_test_np, y_test_np)
                append_runtime_benchmark_row(rows, model_name, "openvino", "openvino_fp32_from_onnx",
                                             metrics, onnx_path, onnx_size_kb)
            except Exception as exc:
                record_runtime_benchmark_skip(rows, model_name, "openvino", "openvino_fp32_from_onnx",
                                              f"benchmark_failed: {type(exc).__name__}: {exc}")
        else:
            record_runtime_benchmark_skip(rows, model_name, "openvino", "openvino_fp32_from_onnx", "missing_dependency")
    return rows, optional_dependency_status


deployable_runtime_rows, optional_dependency_status = run_existing_artifact_runtime_benchmarks()
runtime_df = pd.DataFrame(deployable_runtime_rows)
runtime_csv = output_dir / "wsnds_existing_artifact_runtime_summary.csv"
runtime_df.to_csv(runtime_csv, index=False)

runtime_json = output_dir / "wsnds_existing_artifact_runtime_results.json"
with open(runtime_json, "w", encoding="utf-8") as handle:
    json.dump({
        "source_existing_deployment_output_dir": str(existing_output_dir),
        "model_artifacts": MODEL_ARTIFACTS,
        "optional_dependency_status": optional_dependency_status,
        "results": deployable_runtime_rows,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }, handle, indent=2)

print(f"Saved {runtime_csv}")
print(f"Saved {runtime_json}")
print(runtime_df[["model_name", "runtime", "variant", "status", "macro_f1", "latency_p50_ms_b1", "latency_p50_ms_b64"]].to_string(index=False))
