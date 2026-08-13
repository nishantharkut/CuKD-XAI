"""Refine the exact FG-DS seed-42 deployment models for frozen fixed point.

This is an additive, software-only experiment.  It starts from the two RF-KD
state dictionaries that were exported and hardware-replayed, but it never
modifies those dictionaries or any generated firmware artifact.  The fake
fixed forward pass uses the deployed integer contract:

* integer StandardScaler replay and signed int16 Q8 model inputs;
* int8 weights with the per-layer fractional bits frozen by the strict export;
* int32 biases and accumulator bounds;
* truncation toward zero, ReLU, and signed int16 clipping after every layer.

The straight-through estimator changes gradients only.  Forward values are
integer-valued and are checked against the existing NumPy integer simulator.
Model selection uses only exact fixed-point validation macro-F1. Previously
available source-model test evidence is verified during preflight; test metrics
for new refinement candidates are computed only after selection. Training
requires ``--confirm-training``; otherwise the command performs a read-only
preflight.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deployment.firmware_export.wsnds_rfkd_hil import (  # noqa: E402
    export_fgds_seed42_deployment as strict_export,
)
from experiments.wsnds.leakage_free_rerun import tier15_common as tier15_common_module  # noqa: E402
from experiments.wsnds.leakage_free_rerun.tier15_common import (  # noqa: E402
    CLASS_NAMES,
    KD_ALPHA,
    KD_T,
    STUDENT_SPECS,
    StudentMLP,
    artifact_manifest,
    atomic_save_npz,
    atomic_torch_save,
    atomic_write_json,
    class_weights,
    classification_metrics,
    set_seed,
    sha256_arrays,
    sha256_file,
)


PROTOCOL_ID = "wsnds_fgds_seed42_frozen_fixed_point_refinement_v1"
SOURCE_PROTOCOL_ID = "wsnds_feature_group_split_train_only_scaler_5seed_v1"
EXPECTED_DATASET_SHA256 = (
    "c65d05b983a85753bd62b6f76c5739fc52fe0c14cbb7644255cee4742f5ff7c9"
)
EXPECTED_SPLIT_INDICES_SHA256 = (
    "3d4061aa020122d4c5c5b2f7722de71e0c223c533869d3fdfa1f10784a0a0473"
)
EXPECTED_SCALER_SHA256 = (
    "5303fb570aeb82ffaf88e2d4cceda94a7611762f67c86761990e6a4f09af5dd6"
)
EXPECTED_RF_CONTENT_SHA256 = (
    "809755ca6ec3e8e317648e08947e41cc5f0fbcb1377a3e272d283a050888e452"
)
EXPECTED_INTEGER_PREPROCESS_SHA256 = (
    "fbb083b1cde16ae12d81012c7e946e19be720f22a6624e4e9e4b11178a1de246"
)
EXPECTED_SPLIT_SIZES = {"train": 262197, "validation": 56163, "test": 56301}

DEFAULT_DEPLOYMENT_ROOT = (
    REPO_ROOT
    / "results"
    / "wsnds"
    / "confirmation_runs_v2"
    / "remote_winterfell_feature_group_5seed_20260805"
    / "feature_group_5seed"
)
DEFAULT_DATASET = REPO_ROOT / "data" / "wsnds" / "WSN-DS.csv"
DEFAULT_GENERATED_ROOT = (
    REPO_ROOT / "deployment" / "firmware_export" / "wsnds_rfkd_hil"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results"
    / "wsnds"
    / "evidence_completion_20260811"
    / "fgds_fixed_point_refinement_seed42"
)

# These are the exact models and strict reports already used for deployment.
EXPECTED_STUDENTS: dict[str, dict[str, Any]] = {
    "student_A": {
        "letter": "A",
        "checkpoint_sha256": (
            "f1302fa76ee6739673cbb4a7b949c0addcab29337376993b9c3a7641b8caefa9"
        ),
        "strict_report_sha256": (
            "652b46071adb1921e932e97af999d9d91dbd80f18f3c7e55db30ab55da6987d0"
        ),
        "reference_sha256": (
            "8c37bbe70a4ebc9664518ec389fc5d8086ada897917b187e8f9a5bf161713740"
        ),
        "export_id": (
            "2b09ff0189b2c1db270c887fcb7cd86b3ba736481ef6cc4178a3f5d4374863a8"
        ),
        "layers": [
            {"source_prefix": "net.0", "input_frac": 8, "weight_frac": 5,
             "accum_frac": 13, "output_frac": 9, "output_shift": 4},
            {"source_prefix": "net.2", "input_frac": 9, "weight_frac": 6,
             "accum_frac": 15, "output_frac": 9, "output_shift": 6},
            {"source_prefix": "net.4", "input_frac": 9, "weight_frac": 5,
             "accum_frac": 14, "output_frac": 9, "output_shift": 5},
        ],
    },
    "student_B": {
        "letter": "B",
        "checkpoint_sha256": (
            "1109ce2621a406fc52cf4a7e9fe248981ffdf8b085d5b66dcb83d1eed510066f"
        ),
        "strict_report_sha256": (
            "c5e45b8c4196b493ac85e9a4e38f8a96feafbe501444f89c0715da46bff4ed60"
        ),
        "reference_sha256": (
            "06e5dc48f28f56f5e3a9e687ffaae3aa3c3bc0269a1a2d537cfd4c917a95cffb"
        ),
        "export_id": (
            "325064ed0260f138c42df206c2af1dbe873cd0ed35de1b966966e5de670b9485"
        ),
        "layers": [
            {"source_prefix": "net.0", "input_frac": 8, "weight_frac": 5,
             "accum_frac": 13, "output_frac": 9, "output_shift": 4},
            {"source_prefix": "net.2", "input_frac": 9, "weight_frac": 4,
             "accum_frac": 13, "output_frac": 9, "output_shift": 4},
            {"source_prefix": "net.4", "input_frac": 9, "weight_frac": 5,
             "accum_frac": 14, "output_frac": 9, "output_shift": 5},
        ],
    },
}

# This schedule is fixed before inspecting refinement outcomes and is not searched.
REFINEMENT_SEED = 42042
REFINEMENT_CONFIG = {
    "epochs": 20,
    "batch_size": 1024,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "patience": 5,
}
AUDIT_CHUNK_SIZE = 4096
INT16_MIN = -32768
INT16_MAX = 32767
INT8_MIN = -128
INT8_MAX = 127
INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1
FIRMWARE_MODEL_C = REPO_ROOT / "deployment/hardware_hil/firmware/common/cukd_model.c"
FIRMWARE_PREPROCESS_C = (
    REPO_ROOT / "deployment/hardware_hil/firmware/common/cukd_preprocess.c"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--deployment-root", type=Path, default=DEFAULT_DEPLOYMENT_ROOT)
    parser.add_argument("--generated-root", type=Path, default=DEFAULT_GENERATED_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Training device. The fake-fixed arithmetic itself uses float64.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Verify all inputs and numeric equivalence without writing or training.",
    )
    parser.add_argument(
        "--confirm-training",
        action="store_true",
        help="Required to create a new software-only refinement run.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def require_equal(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise RuntimeError(f"{label} mismatch: observed={observed!r}, expected={expected!r}")


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def numeric_source_records(legacy: Any) -> dict[str, dict[str, Any]]:
    paths = {
        "strict_export_module": Path(strict_export.__file__).resolve(),
        "legacy_numeric_exporter": Path(legacy.__file__).resolve(),
        "training_common_module": Path(tier15_common_module.__file__).resolve(),
        "firmware_model_c": FIRMWARE_MODEL_C.resolve(),
        "firmware_preprocess_c": FIRMWARE_PREPROCESS_C.resolve(),
    }
    records: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            display = path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            display = str(path)
        records[name] = {
            "path": display,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return records


def state_content_sha256(state: dict[str, torch.Tensor]) -> str:
    return sha256_arrays(*[
        state[key].detach().cpu().numpy() for key in sorted(state)
    ])


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def generated_dir(generated_root: Path, student_name: str) -> Path:
    return generated_root / f"generated_fgds_{student_name}_seed42"


def schedule_from_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    fields = [
        "source_prefix", "input_frac", "weight_frac", "accum_frac",
        "output_frac", "output_shift",
    ]
    return [
        {field: layer[field] for field in fields}
        for layer in report["fixed_point_calibration"]["layers"]
    ]


def verify_strict_inputs(
    context: dict[str, Any],
    generated_root: Path,
    student_name: str,
) -> dict[str, Any]:
    expected = EXPECTED_STUDENTS[student_name]
    output = generated_dir(generated_root.resolve(), student_name)
    report_path = output / "strict_export_report.json"
    metadata_path = output / "preprocess_int_metadata.json"
    reference_path = output / "hil_reference_predictions.csv"

    require_equal("source protocol", context["execution"]["protocol_id"], SOURCE_PROTOCOL_ID)
    require_equal("dataset SHA-256", context["dataset"]["dataset_sha256"], EXPECTED_DATASET_SHA256)
    require_equal(
        "split-index content SHA-256",
        context["preprocessing"]["split_indices_sha256"],
        EXPECTED_SPLIT_INDICES_SHA256,
    )
    require_equal("scaler content SHA-256", context["preprocessing"]["scaler_sha256"], EXPECTED_SCALER_SHA256)
    require_equal(
        "RF soft-target content SHA-256",
        context["teacher_soft_target_provenance"]["train_probability_content_sha256"],
        EXPECTED_RF_CONTENT_SHA256,
    )
    require_equal("deployment checkpoint SHA-256", sha256_file(context["model_path"]), expected["checkpoint_sha256"])
    require_equal("strict report SHA-256", sha256_file(report_path), expected["strict_report_sha256"])
    require_equal("integer preprocessing SHA-256", sha256_file(metadata_path), EXPECTED_INTEGER_PREPROCESS_SHA256)
    require_equal("fixed reference SHA-256", sha256_file(reference_path), expected["reference_sha256"])

    report = read_json(report_path)
    require_equal("strict report status", report.get("status"), "passed")
    require_equal("strict export ID", report.get("export_id"), expected["export_id"])
    require_equal("strict report student", report["provenance"]["student"], student_name)
    require_equal("strict report seed", report["provenance"]["seed"], 42)
    require_equal("strict report model hash", report["provenance"]["model_file_sha256"], expected["checkpoint_sha256"])
    require_equal("strict report dataset hash", report["provenance"]["dataset_sha256"], EXPECTED_DATASET_SHA256)
    require_equal("strict report scaler hash", report["provenance"]["scaler_sha256"], EXPECTED_SCALER_SHA256)
    require_equal(
        "strict report RF content hash",
        report["provenance"]["teacher_soft_target_provenance"]["train_probability_content_sha256"],
        EXPECTED_RF_CONTENT_SHA256,
    )
    require_equal("frozen fractional-bit schedule", schedule_from_report(report), expected["layers"])
    require_equal("fixed input fractional bits", report["fixed_point_calibration"]["input_frac"], 8)

    metadata = read_json(metadata_path)
    require_equal("integer preprocess output fraction", metadata["output_q_frac"], 8)
    if metadata.get("input_dim") != 17:
        raise RuntimeError("Integer preprocessing metadata is not the 17-feature contract")
    if context["preprocessing"].get("split_sizes") != EXPECTED_SPLIT_SIZES:
        raise RuntimeError("Deployment split sizes differ from the fixed refinement contract")
    if context["execution"].get("kd_hyperparameters") != {"T": 4.0, "alpha": 0.7}:
        raise RuntimeError("Deployment KD hyperparameters are not T=4, alpha=0.7")
    if KD_T != 4.0 or KD_ALPHA != 0.7:
        raise RuntimeError("Imported KD constants differ from the deployment contract")

    return {
        "generated_dir": output,
        "strict_report_path": report_path,
        "strict_report": report,
        "integer_metadata_path": metadata_path,
        "integer_metadata": metadata,
        "reference_path": reference_path,
    }


def load_plain_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"Checkpoint is not a non-empty state dictionary: {path}")
    for key, value in state.items():
        if not isinstance(key, str) or not torch.is_tensor(value) or not torch.isfinite(value).all():
            raise RuntimeError(f"Invalid checkpoint tensor: {key!r}")
    return state


def quantize_state_frozen(
    state: dict[str, torch.Tensor],
    schedule: list[dict[str, Any]],
    legacy: Any,
) -> tuple[list[tuple[str, Any, Any]], list[dict[str, Any]]]:
    layers = legacy.extract_linear_layers(state)
    if len(layers) != len(schedule):
        raise RuntimeError("Checkpoint layer count differs from the frozen schedule")
    quantized: list[dict[str, Any]] = []
    for index, ((prefix, weight_value, bias_value), spec) in enumerate(zip(layers, schedule)):
        require_equal(f"layer {index} source prefix", prefix, spec["source_prefix"])
        require_equal(
            f"layer {index} accumulator fraction",
            int(spec["input_frac"]) + int(spec["weight_frac"]),
            int(spec["accum_frac"]),
        )
        require_equal(
            f"layer {index} output shift",
            int(spec["accum_frac"]) - int(spec["output_frac"]),
            int(spec["output_shift"]),
        )
        weight = np.asarray(weight_value, dtype=np.float32)
        bias = np.asarray(bias_value, dtype=np.float32)
        weight_scale = float(1 << int(spec["weight_frac"]))
        bias_scale = float(1 << int(spec["accum_frac"]))
        q_weight = np.clip(np.rint(weight * weight_scale), INT8_MIN, INT8_MAX).astype(np.int8)
        q_bias = np.clip(np.rint(bias * bias_scale), INT32_MIN, INT32_MAX).astype(np.int32)
        quantized.append({
            **spec,
            "weight": q_weight,
            "bias": q_bias,
        })
    return layers, quantized


def quantize_raw_and_preprocess(
    raw_features: np.ndarray,
    metadata: dict[str, Any],
    legacy: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw_features = np.asarray(raw_features)
    scale = float(1 << int(metadata["raw_q_frac"]))
    unbounded_raw = np.rint(raw_features.astype(np.float64) * scale)
    raw_saturation = int(np.count_nonzero(
        (unbounded_raw < INT32_MIN) | (unbounded_raw > INT32_MAX)
    ))
    raw_q = np.clip(unbounded_raw, INT32_MIN, INT32_MAX).astype(np.int32)

    centered = raw_q.astype(np.int64) - np.asarray(metadata["scaler_mean_q"], dtype=np.int64)
    scaled = centered * np.asarray(metadata["scaler_inv_scale_q"], dtype=np.int64)
    pre_unclipped = strict_export.rescale_truncating_toward_zero(
        scaled, int(metadata["right_shift"])
    )
    preprocess_saturation = int(np.count_nonzero(
        (pre_unclipped < INT16_MIN) | (pre_unclipped > INT16_MAX)
    ))
    preprocessed = np.clip(pre_unclipped, INT16_MIN, INT16_MAX).astype(np.int16)
    legacy_preprocessed = legacy.simulate_integer_preprocess_q(raw_q, metadata)
    if not np.array_equal(preprocessed, legacy_preprocessed):
        raise RuntimeError("Local integer preprocessing differs from the strict simulator")
    return preprocessed, {
        "rows": int(len(raw_features)),
        "values": int(raw_features.size),
        "raw_input_saturation_count": raw_saturation,
        "integer_preprocess_saturation_count": preprocess_saturation,
        "minimum_preprocessed_q": int(np.min(preprocessed)),
        "maximum_preprocessed_q": int(np.max(preprocessed)),
    }


def parameter_saturation_audit(
    layers: list[tuple[str, Any, Any]],
    schedule: list[dict[str, Any]],
) -> dict[str, Any]:
    layer_results = []
    for index, ((prefix, weight_value, bias_value), spec) in enumerate(zip(layers, schedule)):
        weight = np.asarray(weight_value, dtype=np.float64)
        bias = np.asarray(bias_value, dtype=np.float64)
        unbounded_weight = np.rint(weight * float(1 << int(spec["weight_frac"])))
        unbounded_bias = np.rint(bias * float(1 << int(spec["accum_frac"])))
        layer_results.append({
            "layer": index,
            "source_prefix": prefix,
            "weight_saturation_count": int(np.count_nonzero(
                (unbounded_weight < INT8_MIN) | (unbounded_weight > INT8_MAX)
            )),
            "bias_saturation_count": int(np.count_nonzero(
                (unbounded_bias < INT32_MIN) | (unbounded_bias > INT32_MAX)
            )),
        })
    return {
        "layers": layer_results,
        "weight_saturation_count": int(sum(x["weight_saturation_count"] for x in layer_results)),
        "bias_saturation_count": int(sum(x["bias_saturation_count"] for x in layer_results)),
    }


def simulate_q_inputs(
    quantized_layers: list[dict[str, Any]],
    q_inputs: np.ndarray,
    *,
    collect_logits: bool,
    chunk_size: int = AUDIT_CHUNK_SIZE,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    layer_saturation = np.zeros(len(quantized_layers), dtype=np.int64)
    accumulator_overflow = np.zeros(len(quantized_layers), dtype=np.int64)
    layer_min = np.full(len(quantized_layers), np.iinfo(np.int64).max, dtype=np.int64)
    layer_max = np.full(len(quantized_layers), np.iinfo(np.int64).min, dtype=np.int64)
    accumulator_min = np.full(len(quantized_layers), np.iinfo(np.int64).max, dtype=np.int64)
    accumulator_max = np.full(len(quantized_layers), np.iinfo(np.int64).min, dtype=np.int64)
    logits_parts: list[np.ndarray] = []

    for start in range(0, len(q_inputs), chunk_size):
        activations = np.asarray(q_inputs[start:start + chunk_size], dtype=np.int64)
        for index, layer in enumerate(quantized_layers):
            accumulator = (
                activations @ np.asarray(layer["weight"], dtype=np.int64).T
                + np.asarray(layer["bias"], dtype=np.int64)
            )
            accumulator_min[index] = min(accumulator_min[index], int(accumulator.min()))
            accumulator_max[index] = max(accumulator_max[index], int(accumulator.max()))
            accumulator_overflow[index] += int(np.count_nonzero(
                (accumulator < INT32_MIN) | (accumulator > INT32_MAX)
            ))
            output = strict_export.rescale_truncating_toward_zero(
                accumulator, int(layer["output_shift"])
            )
            if index < len(quantized_layers) - 1:
                output = np.maximum(output, 0)
            layer_min[index] = min(layer_min[index], int(output.min()))
            layer_max[index] = max(layer_max[index], int(output.max()))
            layer_saturation[index] += int(np.count_nonzero(
                (output < INT16_MIN) | (output > INT16_MAX)
            ))
            activations = np.clip(output, INT16_MIN, INT16_MAX).astype(np.int16).astype(np.int64)
        if collect_logits:
            logits_parts.append(activations.astype(np.int16))

    layer_results = [
        {
            "layer": index,
            "accumulator_minimum": int(accumulator_min[index]),
            "accumulator_maximum": int(accumulator_max[index]),
            "accumulator_int32_overflow_count": int(accumulator_overflow[index]),
            "activation_minimum_before_clip": int(layer_min[index]),
            "activation_maximum_before_clip": int(layer_max[index]),
            "activation_saturation_count": int(layer_saturation[index]),
        }
        for index in range(len(quantized_layers))
    ]
    logits = np.concatenate(logits_parts) if collect_logits else None
    return logits, {
        "rows": int(len(q_inputs)),
        "layers": layer_results,
        "accumulator_int32_overflow_count": int(accumulator_overflow.sum()),
        "activation_saturation_count": int(layer_saturation.sum()),
    }


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(values)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def fixed_metrics(labels: np.ndarray, logits_q: np.ndarray, output_frac: int) -> dict[str, Any]:
    probabilities = softmax_numpy(np.asarray(logits_q, dtype=np.float64) / float(1 << output_frac))
    return classification_metrics(np.asarray(labels, dtype=np.int64), probabilities)


def batched_float_probabilities(
    model: nn.Module,
    values: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model = model.to(device).eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(values), 4096):
            batch = torch.from_numpy(values[start:start + 4096]).to(device=device, dtype=torch.float32)
            parts.append(F.softmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(parts)


def build_float_model(student_name: str, state: dict[str, torch.Tensor]) -> StudentMLP:
    model = StudentMLP(17, STUDENT_SPECS[student_name], len(CLASS_NAMES))
    model.load_state_dict(state, strict=True)
    return model


def ste_quantized_integer(
    values: torch.Tensor,
    fractional_bits: int,
    minimum: int,
    maximum: int,
) -> torch.Tensor:
    scaled = values * float(1 << fractional_bits)
    quantized = torch.clamp(torch.round(scaled), minimum, maximum)
    return scaled + (quantized - scaled).detach()


class FrozenScheduleFakeFixed(nn.Module):
    """Exact-value fixed forward with straight-through quantization gradients."""

    def __init__(self, float_model: StudentMLP, schedule: list[dict[str, Any]]):
        super().__init__()
        self.float_model = float_model
        self.schedule = copy.deepcopy(schedule)
        self.linear_layers = tuple(
            module for module in self.float_model.net if isinstance(module, nn.Linear)
        )
        if len(self.linear_layers) != len(self.schedule):
            raise RuntimeError("Float model and fixed schedule have different layer counts")

    def forward_q(self, inputs_q: torch.Tensor) -> torch.Tensor:
        # Float64 exactly represents all integer products and sums admitted by
        # the checked int32 accumulator contract.  It also remains differentiable.
        activations = inputs_q.to(dtype=torch.float64)
        for index, (linear, spec) in enumerate(zip(self.linear_layers, self.schedule)):
            weight_q = ste_quantized_integer(
                linear.weight, int(spec["weight_frac"]), INT8_MIN, INT8_MAX
            ).to(dtype=torch.float64)
            bias_q = ste_quantized_integer(
                linear.bias, int(spec["accum_frac"]), INT32_MIN, INT32_MAX
            ).to(dtype=torch.float64)
            accumulator = F.linear(activations, weight_q, bias_q)
            scaled = accumulator / float(1 << int(spec["output_shift"]))
            surrogate = F.relu(scaled) if index < len(self.linear_layers) - 1 else scaled
            exact = torch.trunc(scaled)
            if index < len(self.linear_layers) - 1:
                exact = F.relu(exact)
            exact = torch.clamp(exact, INT16_MIN, INT16_MAX)
            activations = surrogate + (exact - surrogate).detach()
        return activations

    def forward(self, inputs_q: torch.Tensor) -> torch.Tensor:
        output_frac = int(self.schedule[-1]["output_frac"])
        return self.forward_q(inputs_q) / float(1 << output_frac)


def project_parameters_to_frozen_ranges(
    model: StudentMLP,
    schedule: list[dict[str, Any]],
) -> None:
    linears = [module for module in model.net if isinstance(module, nn.Linear)]
    with torch.no_grad():
        for linear, spec in zip(linears, schedule):
            weight_scale = float(1 << int(spec["weight_frac"]))
            bias_scale = float(1 << int(spec["accum_frac"]))
            linear.weight.clamp_(INT8_MIN / weight_scale, INT8_MAX / weight_scale)
            linear.bias.clamp_(INT32_MIN / bias_scale, INT32_MAX / bias_scale)


def numeric_equivalence_assertions(
    state: dict[str, torch.Tensor],
    student_name: str,
    schedule: list[dict[str, Any]],
    q_validation: np.ndarray,
    y_validation: np.ndarray,
    legacy: Any,
    device: torch.device = torch.device("cpu"),
) -> dict[str, Any]:
    negative_values = np.asarray([-17, -16, -15, 15, 16, 17], dtype=np.int64)
    expected_shift = np.asarray([-1, -1, 0, 0, 1, 1], dtype=np.int64)
    observed_shift = strict_export.rescale_truncating_toward_zero(negative_values, 4)
    if not np.array_equal(observed_shift, expected_shift):
        raise RuntimeError("Signed rescaling does not truncate toward zero")

    ties = torch.tensor([0.5, 1.5, 2.5, -0.5, -1.5, -2.5], dtype=torch.float32)
    expected_ties = np.rint(ties.numpy()).astype(np.float32)
    if not np.array_equal(torch.round(ties).numpy(), expected_ties):
        raise RuntimeError("PyTorch and NumPy tie rounding differ")

    _, quantized = quantize_state_frozen(state, schedule, legacy)
    selected: set[int] = set()
    for class_index in range(len(CLASS_NAMES)):
        class_rows = np.flatnonzero(y_validation == class_index)
        selected.update(class_rows[: min(64, len(class_rows))].tolist())
    for feature_index in range(q_validation.shape[1]):
        selected.add(int(np.argmin(q_validation[:, feature_index])))
        selected.add(int(np.argmax(q_validation[:, feature_index])))
    sample_indices = np.asarray(sorted(selected), dtype=np.int64)
    sample = np.asarray(q_validation[sample_indices], dtype=np.int16)
    expected_logits, _ = legacy.simulate_fixed_point_inference(quantized, sample)
    float_model = build_float_model(student_name, state)
    fake_model = FrozenScheduleFakeFixed(float_model, schedule).to(device).eval()
    with torch.no_grad():
        observed_logits = fake_model.forward_q(
            torch.from_numpy(sample).to(device)
        ).cpu().numpy()
    if not np.array_equal(observed_logits.astype(np.int16), expected_logits):
        difference = np.max(np.abs(observed_logits.astype(np.int64) - expected_logits.astype(np.int64)))
        raise RuntimeError(f"Fake-fixed forward differs from integer simulator; max delta={difference}")
    if not np.array_equal(observed_logits, observed_logits.astype(np.int16).astype(np.float64)):
        raise RuntimeError("Fake-fixed forward did not produce exact integer-valued logits")
    return {
        "signed_shift_truncates_toward_zero": True,
        "torch_numpy_round_to_even_match": True,
        "fake_fixed_vs_numpy_integer_rows": int(len(sample)),
        "sample_selection": (
            "up to 64 validation rows per class plus per-feature minima and maxima"
        ),
        "classes_covered": np.unique(y_validation[sample_indices]).astype(int).tolist(),
        "device": str(device),
        "fake_fixed_vs_numpy_integer_logits_exact": True,
    }


def verify_preserved_baseline(
    context: dict[str, Any],
    strict_context: dict[str, Any],
    student_name: str,
    legacy: Any,
) -> dict[str, Any]:
    state = load_plain_state(context["model_path"])
    schedule = EXPECTED_STUDENTS[student_name]["layers"]
    layers, quantized = quantize_state_frozen(state, schedule, legacy)
    q_test, input_audit = quantize_raw_and_preprocess(
        context["split"]["X_test_raw"], strict_context["integer_metadata"], legacy
    )
    logits, activation_audit = simulate_q_inputs(quantized, q_test, collect_logits=True)
    if logits is None:
        raise RuntimeError("Baseline simulation did not produce logits")

    reference = pd.read_csv(strict_context["reference_path"])
    logit_columns = [f"fixed_logit_{index}" for index in range(len(CLASS_NAMES))]
    expected_columns = [
        "row_id", "source_row_index", "true_label", "fixed_pred", "fp32_pred",
        *logit_columns,
    ]
    require_equal("fixed reference columns", reference.columns.tolist(), expected_columns)
    require_equal("fixed reference row count", len(reference), EXPECTED_SPLIT_SIZES["test"])
    if not np.array_equal(
        reference["source_row_index"].to_numpy(dtype=np.int64),
        context["split"]["test_indices"],
    ):
        raise RuntimeError("Fixed reference source rows differ from the FG-DS test split")
    if not np.array_equal(
        reference["true_label"].to_numpy(dtype=np.int64), context["split"]["y_test"]
    ):
        raise RuntimeError("Fixed reference labels differ from the FG-DS test split")
    reference_logits = reference[logit_columns].to_numpy(dtype=np.int16)
    if not np.array_equal(logits, reference_logits):
        raise RuntimeError("Frozen-schedule simulator does not reproduce preserved fixed logits")
    predictions = logits.astype(np.int32).argmax(axis=1).astype(np.int64)
    if not np.array_equal(predictions, reference["fixed_pred"].to_numpy(dtype=np.int64)):
        raise RuntimeError("Frozen-schedule simulator does not reproduce preserved predictions")

    metrics = fixed_metrics(
        context["split"]["y_test"], logits, int(schedule[-1]["output_frac"])
    )
    report_metrics = strict_context["strict_report"]["fixed_metrics"]
    for key in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]:
        if not np.isclose(metrics[key], report_metrics[key], rtol=0.0, atol=1e-15):
            raise RuntimeError(f"Recomputed baseline {key} differs from the strict report")

    parameter_audit = parameter_saturation_audit(layers, schedule)
    numeric_tests = numeric_equivalence_assertions(
        state,
        student_name,
        schedule,
        quantize_raw_and_preprocess(
            context["split"]["X_validation_raw"], strict_context["integer_metadata"], legacy
        )[0],
        np.asarray(context["split"]["y_validation"], dtype=np.int64),
        legacy,
    )
    return {
        "checkpoint_path": str(context["model_path"]),
        "checkpoint_sha256": sha256_file(context["model_path"]),
        "state_content_sha256": state_content_sha256(state),
        "strict_export_report_path": str(strict_context["strict_report_path"]),
        "strict_export_report_sha256": sha256_file(strict_context["strict_report_path"]),
        "strict_export_id": strict_context["strict_report"]["export_id"],
        "fixed_metrics": metrics,
        "float_metrics": strict_context["strict_report"]["fp32_metrics"],
        "input_saturation_audit": input_audit,
        "parameter_saturation_audit": parameter_audit,
        "activation_and_accumulator_audit": activation_audit,
        "numeric_assertions": numeric_tests,
        "preserved_full_test_logits_exact": True,
        "preserved_full_test_predictions_exact": True,
    }


def preflight_student(
    deployment_root: Path,
    dataset_csv: Path,
    generated_root: Path,
    student_name: str,
    legacy: Any,
) -> dict[str, Any]:
    context = strict_export.load_verified_context(
        deployment_root, dataset_csv, EXPECTED_STUDENTS[student_name]["letter"]
    )
    strict_context = verify_strict_inputs(context, generated_root, student_name)
    baseline = verify_preserved_baseline(context, strict_context, student_name, legacy)
    return {
        "student": student_name,
        "hidden_dims": list(STUDENT_SPECS[student_name]),
        "frozen_schedule": EXPECTED_STUDENTS[student_name]["layers"],
        "source": baseline,
    }


def preflight(
    deployment_root: Path,
    dataset_csv: Path,
    generated_root: Path,
    legacy: Any,
) -> dict[str, Any]:
    students = {
        student_name: preflight_student(
            deployment_root, dataset_csv, generated_root, student_name, legacy
        )
        for student_name in EXPECTED_STUDENTS
    }
    return {
        "status": "passed",
        "mode": "read-only preflight",
        "protocol_id": PROTOCOL_ID,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "split_indices_sha256": EXPECTED_SPLIT_INDICES_SHA256,
        "scaler_sha256": EXPECTED_SCALER_SHA256,
        "rf_soft_target_content_sha256": EXPECTED_RF_CONTENT_SHA256,
        "numeric_implementation_sources": numeric_source_records(legacy),
        "kd_hyperparameters": {"T": KD_T, "alpha": KD_ALPHA},
        "selection_metric": "exact integer-simulator validation macro-F1 only",
        "test_chronology": (
            "Previously available source-model test evidence is verified during preflight. "
            "No new refinement candidate is evaluated on test data before validation-only "
            "checkpoint selection."
        ),
        "students": students,
        "claim_boundary": {
            "software_only": True,
            "hardware_replayed": False,
            "may_replace_existing_hardware_evidence": False,
            "model_state_scope": "one preserved training seed (42) per student",
        },
    }


def exact_validation_candidate(
    state: dict[str, torch.Tensor],
    schedule: list[dict[str, Any]],
    q_validation: np.ndarray,
    y_validation: np.ndarray,
    legacy: Any,
) -> dict[str, Any]:
    layers, quantized = quantize_state_frozen(state, schedule, legacy)
    logits, numeric_audit = simulate_q_inputs(quantized, q_validation, collect_logits=True)
    if logits is None:
        raise RuntimeError("Validation simulator did not produce logits")
    metrics = fixed_metrics(y_validation, logits, int(schedule[-1]["output_frac"]))
    parameter_audit = parameter_saturation_audit(layers, schedule)
    try:
        conservative_accumulator_bounds = strict_export.accumulator_bounds(quantized)
        conservative_accumulator_bound_passed = all(
            bool(item["passed"]) for item in conservative_accumulator_bounds
        )
        conservative_accumulator_bound_error = None
    except RuntimeError as exc:
        conservative_accumulator_bounds = []
        conservative_accumulator_bound_passed = False
        conservative_accumulator_bound_error = str(exc)
    admissible = (
        parameter_audit["weight_saturation_count"] == 0
        and parameter_audit["bias_saturation_count"] == 0
        and numeric_audit["accumulator_int32_overflow_count"] == 0
        and numeric_audit["activation_saturation_count"] == 0
        and conservative_accumulator_bound_passed
    )
    return {
        "fixed_validation_macro_f1": metrics["macro_f1"],
        "fixed_validation_metrics": metrics,
        "parameter_saturation_audit": parameter_audit,
        "activation_and_accumulator_audit": numeric_audit,
        "conservative_sequential_accumulator_bounds": conservative_accumulator_bounds,
        "conservative_sequential_accumulator_bound_passed": bool(
            conservative_accumulator_bound_passed
        ),
        "conservative_sequential_accumulator_bound_error": (
            conservative_accumulator_bound_error
        ),
        "numeric_admissible": bool(admissible),
    }


def train_one_student(
    deployment_root: Path,
    dataset_csv: Path,
    generated_root: Path,
    student_name: str,
    device: torch.device,
    student_output: Path,
    legacy: Any,
) -> dict[str, Any]:
    context = strict_export.load_verified_context(
        deployment_root, dataset_csv, EXPECTED_STUDENTS[student_name]["letter"]
    )
    strict_context = verify_strict_inputs(context, generated_root, student_name)
    source_state = load_plain_state(context["model_path"])
    schedule = EXPECTED_STUDENTS[student_name]["layers"]
    metadata = strict_context["integer_metadata"]

    q_train, train_input_audit = quantize_raw_and_preprocess(
        context["split"]["X_train_raw"], metadata, legacy
    )
    q_validation, validation_input_audit = quantize_raw_and_preprocess(
        context["split"]["X_validation_raw"], metadata, legacy
    )
    y_train = np.asarray(context["split"]["y_train"], dtype=np.int64)
    y_validation = np.asarray(context["split"]["y_validation"], dtype=np.int64)
    teacher_probabilities = np.load(context["teacher_probability_path"], allow_pickle=False)
    require_equal(
        "RF target content hash before refinement",
        sha256_arrays(teacher_probabilities),
        EXPECTED_RF_CONTENT_SHA256,
    )

    float_model = build_float_model(student_name, source_state).to(device)
    fake_model = FrozenScheduleFakeFixed(float_model, schedule).to(device)
    optimizer = torch.optim.AdamW(
        fake_model.parameters(),
        lr=REFINEMENT_CONFIG["learning_rate"],
        weight_decay=REFINEMENT_CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=REFINEMENT_CONFIG["epochs"]
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights(y_train).to(device))
    teacher_raw = torch.from_numpy(teacher_probabilities.astype(np.float32, copy=False))
    teacher_targets = F.softmax(
        torch.log(teacher_raw.clamp(min=1e-8)) / KD_T, dim=1
    )

    source_active_device_numeric_assertions = numeric_equivalence_assertions(
        source_state,
        student_name,
        schedule,
        q_validation,
        y_validation,
        legacy,
        device,
    )

    baseline_validation = exact_validation_candidate(
        source_state, schedule, q_validation, y_validation, legacy
    )
    if not baseline_validation["numeric_admissible"]:
        raise RuntimeError("The preserved PTQ baseline is not numerically admissible")
    best_state = copy.deepcopy(source_state)
    best_epoch = 0
    best_score = float(baseline_validation["fixed_validation_macro_f1"])
    best_validation = baseline_validation
    history = [{
        "epoch": 0,
        "source": "preserved PTQ checkpoint",
        **baseline_validation,
    }]
    stale = 0
    set_seed(REFINEMENT_SEED)
    started = time.perf_counter()

    q_train_tensor = torch.from_numpy(q_train)
    y_train_tensor = torch.from_numpy(y_train)
    for epoch in range(1, REFINEMENT_CONFIG["epochs"] + 1):
        fake_model.train()
        order = np.random.default_rng(REFINEMENT_SEED + epoch).permutation(len(y_train))
        total_loss = 0.0
        batches = 0
        for start in range(0, len(order), REFINEMENT_CONFIG["batch_size"]):
            index_np = order[start:start + REFINEMENT_CONFIG["batch_size"]]
            index = torch.from_numpy(index_np)
            input_batch = q_train_tensor[index].to(device)
            label_batch = y_train_tensor[index].to(device)
            teacher_batch = teacher_targets[index].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = fake_model(input_batch).to(dtype=torch.float32)
            kd_loss = F.kl_div(
                F.log_softmax(logits / KD_T, dim=1),
                teacher_batch,
                reduction="batchmean",
            ) * (KD_T * KD_T)
            ce_loss = criterion(logits, label_batch)
            loss = KD_ALPHA * kd_loss + (1.0 - KD_ALPHA) * ce_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite refinement loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            project_parameters_to_frozen_ranges(float_model, schedule)
            total_loss += float(loss.detach().cpu())
            batches += 1
        epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
        scheduler.step()

        candidate_state = copy.deepcopy({
            key: value.detach().cpu() for key, value in float_model.state_dict().items()
        })
        validation = exact_validation_candidate(
            candidate_state, schedule, q_validation, y_validation, legacy
        )
        history.append({
            "epoch": epoch,
            "mean_training_loss": total_loss / max(batches, 1),
            "learning_rate_used": epoch_learning_rate,
            **validation,
        })
        score = float(validation["fixed_validation_macro_f1"])
        # Eligibility is a numeric safety gate. Among eligible checkpoints the
        # only ranking quantity is exact fixed-point validation macro-F1.
        if validation["numeric_admissible"] and score > best_score + 1e-12:
            best_state = candidate_state
            best_epoch = epoch
            best_score = score
            best_validation = validation
            stale = 0
        else:
            stale += 1
            if stale >= REFINEMENT_CONFIG["patience"]:
                break

    elapsed = time.perf_counter() - started
    selected_active_device_numeric_assertions = numeric_equivalence_assertions(
        best_state, student_name, schedule, q_validation, y_validation, legacy, device
    )

    # New candidate test evaluation begins only after the validation-selected
    # state has been frozen. Preflight has already verified preserved source
    # checkpoint test evidence without using it for candidate selection.
    q_test, test_input_audit = quantize_raw_and_preprocess(
        context["split"]["X_test_raw"], metadata, legacy
    )
    best_layers, best_quantized = quantize_state_frozen(best_state, schedule, legacy)
    fixed_test_logits, fixed_test_audit = simulate_q_inputs(
        best_quantized, q_test, collect_logits=True
    )
    if fixed_test_logits is None:
        raise RuntimeError("Selected checkpoint test simulator did not produce logits")
    selected_float_model = build_float_model(student_name, best_state)
    float_test_probabilities = batched_float_probabilities(
        selected_float_model, context["scaled"]["X_test"], device
    )
    float_test_metrics = classification_metrics(context["split"]["y_test"], float_test_probabilities)
    fixed_test_metrics = fixed_metrics(
        context["split"]["y_test"], fixed_test_logits, int(schedule[-1]["output_frac"])
    )
    fixed_predictions = fixed_test_logits.astype(np.int32).argmax(axis=1).astype(np.int64)
    float_predictions = float_test_probabilities.argmax(axis=1).astype(np.int64)
    fixed_float_agreement = float(np.mean(fixed_predictions == float_predictions))

    baseline = verify_preserved_baseline(context, strict_context, student_name, legacy)
    train_layers, train_quantized = quantize_state_frozen(best_state, schedule, legacy)
    _, train_numeric_audit = simulate_q_inputs(train_quantized, q_train, collect_logits=False)
    parameter_audit = parameter_saturation_audit(train_layers, schedule)
    strict_zero_saturation = (
        train_input_audit["raw_input_saturation_count"] == 0
        and train_input_audit["integer_preprocess_saturation_count"] == 0
        and validation_input_audit["raw_input_saturation_count"] == 0
        and validation_input_audit["integer_preprocess_saturation_count"] == 0
        and test_input_audit["raw_input_saturation_count"] == 0
        and test_input_audit["integer_preprocess_saturation_count"] == 0
        and parameter_audit["weight_saturation_count"] == 0
        and parameter_audit["bias_saturation_count"] == 0
        and train_numeric_audit["activation_saturation_count"] == 0
        and best_validation["activation_and_accumulator_audit"]["activation_saturation_count"] == 0
        and fixed_test_audit["activation_saturation_count"] == 0
        and train_numeric_audit["accumulator_int32_overflow_count"] == 0
        and best_validation["activation_and_accumulator_audit"]["accumulator_int32_overflow_count"] == 0
        and fixed_test_audit["accumulator_int32_overflow_count"] == 0
        and best_validation["conservative_sequential_accumulator_bound_passed"]
    )
    if not strict_zero_saturation:
        raise RuntimeError(
            "Selected refinement checkpoint failed the strict saturation or "
            "sequential-accumulator safety gate"
        )

    student_output.mkdir(parents=True, exist_ok=False)
    plain_path = student_output / f"refined_{student_name}_fp32.pt"
    artifact_path = student_output / f"refined_{student_name}_artifact.pt"
    predictions_path = student_output / f"refined_{student_name}_test_predictions.npz"
    atomic_torch_save(plain_path, best_state)
    rich_artifact = {
        "protocol_id": PROTOCOL_ID,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "student": student_name,
        "seed": 42,
        "refinement_seed": REFINEMENT_SEED,
        "state_dict": best_state,
        "source_checkpoint_sha256": sha256_file(context["model_path"]),
        "source_state_content_sha256": state_content_sha256(source_state),
        "refined_state_content_sha256": state_content_sha256(best_state),
        "selected_epoch": best_epoch,
        "selection_metric": "exact integer-simulator validation macro-F1 only",
        "selected_fixed_validation_macro_f1": best_score,
        "frozen_schedule": schedule,
        "kd_hyperparameters": {"T": KD_T, "alpha": KD_ALPHA},
        "refinement_config": REFINEMENT_CONFIG,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "split_indices_sha256": EXPECTED_SPLIT_INDICES_SHA256,
        "scaler_sha256": EXPECTED_SCALER_SHA256,
        "rf_soft_target_content_sha256": EXPECTED_RF_CONTENT_SHA256,
        "software_only": True,
        "hardware_replayed": False,
    }
    atomic_torch_save(artifact_path, rich_artifact)
    atomic_save_npz(
        predictions_path,
        source_row_index=np.asarray(context["split"]["test_indices"], dtype=np.int64),
        true_label=np.asarray(context["split"]["y_test"], dtype=np.int64),
        float_probabilities=float_test_probabilities.astype(np.float32),
        float_prediction=float_predictions,
        fixed_logits=fixed_test_logits.astype(np.int16),
        fixed_prediction=fixed_predictions,
    )

    report = {
        "student": student_name,
        "status": "complete",
        "source_checkpoint": baseline,
        "refinement": {
            "selected_epoch": best_epoch,
            "changed_from_source": state_content_sha256(best_state) != state_content_sha256(source_state),
            "selected_fixed_validation_macro_f1": best_score,
            "baseline_fixed_validation_macro_f1": baseline_validation["fixed_validation_macro_f1"],
            "validation_macro_f1_delta": best_score - baseline_validation["fixed_validation_macro_f1"],
            "history": history,
            "source_checkpoint_active_device_numeric_assertions": (
                source_active_device_numeric_assertions
            ),
            "selected_checkpoint_active_device_numeric_assertions": (
                selected_active_device_numeric_assertions
            ),
            "wall_seconds": elapsed,
        },
        "test_evaluation_after_selection": {
            "float_metrics": float_test_metrics,
            "fixed_metrics": fixed_test_metrics,
            "fixed_vs_float_prediction_agreement": fixed_float_agreement,
            "fixed_macro_f1_minus_current_ptq": (
                fixed_test_metrics["macro_f1"] - baseline["fixed_metrics"]["macro_f1"]
            ),
            "float_macro_f1_minus_source": (
                float_test_metrics["macro_f1"] - baseline["float_metrics"]["macro_f1"]
            ),
        },
        "saturation_and_range_audit": {
            "train_input": train_input_audit,
            "validation_input": validation_input_audit,
            "test_input": test_input_audit,
            "parameters": parameter_audit,
            "train_activations_and_accumulators": train_numeric_audit,
            "validation_activations_and_accumulators": best_validation[
                "activation_and_accumulator_audit"
            ],
            "test_activations_and_accumulators": fixed_test_audit,
            "strict_zero_saturation_and_no_overflow_gate": bool(strict_zero_saturation),
        },
        "outputs": {
            "plain_state_dict": plain_path.name,
            "rich_artifact": artifact_path.name,
            "test_predictions": predictions_path.name,
        },
        "output_hashes": {
            "plain_state_dict_sha256": sha256_file(plain_path),
            "plain_state_content_sha256": state_content_sha256(best_state),
            "rich_artifact_sha256": sha256_file(artifact_path),
            "test_predictions_sha256": sha256_file(predictions_path),
        },
        "claim_boundary": {
            "software_only": True,
            "hardware_replayed": False,
            "firmware_exported": False,
            "board_timing_measured": False,
            "board_energy_measured": False,
            "may_replace_current_ptq_hardware_results": False,
        },
    }
    atomic_write_json(student_output / "refinement_result.json", report)
    return report


def prepare_staging(output_dir: Path) -> tuple[Path, Path]:
    final = output_dir.resolve()
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite output: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.parent / f".{final.name}.tmp.{os.getpid()}.{time.time_ns()}"
    staging.mkdir()
    return final, staging


def require_output_separate_from_inputs(
    output_dir: Path,
    protected_roots: list[Path],
) -> None:
    output = output_dir.resolve()
    for protected in protected_roots:
        protected = protected.resolve()
        try:
            output.relative_to(protected)
        except ValueError:
            continue
        raise RuntimeError(f"Output must not be inside protected input tree: {output}")


def environment_record(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "fake_fixed_compute_dtype": "float64 with exact integer-valued forward",
    }


def run_training(args: argparse.Namespace, preflight_report: dict[str, Any], legacy: Any) -> Path:
    deployment_root = args.deployment_root.resolve()
    dataset_csv = args.dataset_csv.resolve()
    generated_root = args.generated_root.resolve()
    final, staging = prepare_staging(args.output_dir)
    require_output_separate_from_inputs(
        final, [deployment_root, generated_root, dataset_csv.parent]
    )
    device = select_device(args.device)
    try:
        execution_contract = {
            "protocol_id": PROTOCOL_ID,
            "status": "running",
            "created_utc": utc_now(),
            "script_path": str(SCRIPT_PATH),
            "script_sha256": sha256_file(SCRIPT_PATH),
            "source_protocol_id": SOURCE_PROTOCOL_ID,
            "dataset_path_recorded": str(dataset_csv),
            "dataset_sha256": EXPECTED_DATASET_SHA256,
            "deployment_root_recorded": str(deployment_root),
            "generated_root_recorded": str(generated_root),
            "split_indices_sha256": EXPECTED_SPLIT_INDICES_SHA256,
            "scaler_sha256": EXPECTED_SCALER_SHA256,
            "rf_soft_target_content_sha256": EXPECTED_RF_CONTENT_SHA256,
            "numeric_implementation_sources": numeric_source_records(legacy),
            "source_checkpoint_sha256": {
                name: data["checkpoint_sha256"] for name, data in EXPECTED_STUDENTS.items()
            },
            "frozen_schedules": {
                name: data["layers"] for name, data in EXPECTED_STUDENTS.items()
            },
            "kd_hyperparameters": {"T": KD_T, "alpha": KD_ALPHA},
            "refinement_config": REFINEMENT_CONFIG,
            "selection_contract": {
                "eligible_checkpoint_requirements": (
                    "zero parameter saturation, zero activation saturation, and zero "
                    "int32 accumulator overflow"
                ),
                "ranking_metric": "exact integer-simulator validation macro-F1 only",
                "tie_policy": "retain the earlier checkpoint",
                "epoch_zero_candidate": "preserved current PTQ deployment checkpoint",
                "test_access": (
                    "Previously available source-model test evidence is verified before "
                    "training. New candidate test metrics are computed after selection only."
                ),
            },
            "environment": environment_record(device),
            "software_only": True,
            "hardware_replayed": False,
            "model_state_scope": "one preserved training seed (42) per student",
        }
        execution_contract["contract_sha256"] = canonical_json_sha256(execution_contract)
        atomic_write_json(staging / "execution_contract.json", execution_contract)
        atomic_write_json(staging / "preflight_report.json", preflight_report)

        student_results = {}
        for student_name in EXPECTED_STUDENTS:
            student_results[student_name] = train_one_student(
                deployment_root,
                dataset_csv,
                generated_root,
                student_name,
                device,
                staging / student_name,
                legacy,
            )
        final_report = {
            "protocol_id": PROTOCOL_ID,
            "status": "complete",
            "completed_utc": utc_now(),
            "students": student_results,
            "claim_boundary": {
                "software_only": True,
                "hardware_replayed": False,
                "existing_usb_and_wifi_hil_remain_bound_to_the_original_ptq_models": True,
                "new_hardware_claim_requires_new_strict_export_and_board_replay": True,
                "model_state_scope": "one preserved training seed (42) per student",
            },
        }
        atomic_write_json(staging / "refinement_report.json", final_report)
        execution_contract["status"] = "complete"
        execution_contract["completed_utc"] = final_report["completed_utc"]
        execution_contract.pop("contract_sha256", None)
        execution_contract["contract_sha256"] = canonical_json_sha256(execution_contract)
        atomic_write_json(staging / "execution_contract.json", execution_contract)
        atomic_write_json(staging / "artifact_manifest.json", artifact_manifest(
            staging, PROTOCOL_ID, "complete"
        ))
        os.replace(staging, final)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return final


def main() -> int:
    args = parse_args()
    if args.preflight_only and args.confirm_training:
        raise SystemExit("Choose either --preflight-only or --confirm-training, not both")
    legacy = strict_export.load_legacy_exporter()
    report = preflight(
        args.deployment_root.resolve(),
        args.dataset_csv.resolve(),
        args.generated_root.resolve(),
        legacy,
    )
    if not args.confirm_training:
        print(json.dumps(report, indent=2))
        return 0
    final = run_training(args, report, legacy)
    print(final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
