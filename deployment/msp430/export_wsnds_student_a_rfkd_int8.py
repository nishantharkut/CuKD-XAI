#!/usr/bin/env python3
"""Export a WSN-DS RF-KD student MLP to fixed-point C artifacts.

This is intentionally separate from the notebooks. It turns a trained
WSN-DS student FP32 state_dict into:

  - model_weights.h: int8 weights, int32 biases, calibrated int16 metadata
  - preprocess_int_metadata.h: integer StandardScaler constants
  - export_summary.json: byte counts, quantization settings, tensor errors

The generated C path is for hardware-facing feasibility evidence. It exports
an integer StandardScaler normalization contract, then expects the 17 WSN-DS
features to already exist as fixed-point raw feature values. It does not
implement WSN-DS feature extraction on a mote.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any


CLASS_NAMES = ["Blackhole", "Flooding", "Grayhole", "Normal", "TDMA"]


def _to_list(values: Any) -> list[Any]:
    """Convert NumPy, PyTorch, or list-like values to plain Python lists."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    if hasattr(values, "tolist"):
        values = values.tolist()
    return list(values)


def _to_matrix(values: Any) -> list[list[Any]]:
    rows = _to_list(values)
    return [list(row) for row in rows]


def _accuracy(labels: list[int], preds: list[int]) -> float | None:
    if not labels:
        return None
    return sum(int(y == p) for y, p in zip(labels, preds)) / len(labels)


def _agreement(left: list[int], right: list[int]) -> float | None:
    if not left:
        return None
    return sum(int(a == b) for a, b in zip(left, right)) / len(left)


def build_preprocessing_metadata(
    *,
    target_col: str,
    feature_names: list[str],
    class_names: list[str],
    scaler_mean: Any,
    scaler_scale: Any,
    split_sizes: dict[str, int],
) -> dict[str, Any]:
    """Build the reproducibility record for the v2.3 WSN-DS preprocessing contract."""
    return {
        "dataset": "WSN-DS",
        "preprocessing_contract": (
            "v2.3-compatible: strip column names, drop id/Id/ID, label-encode "
            "the target alphabetically, fit StandardScaler on all rows, then use "
            "the seed-42 stratified 70/15/15 split used by the deployment proof."
        ),
        "target_col": target_col,
        "feature_names": list(feature_names),
        "class_names": list(class_names),
        "input_dim": len(feature_names),
        "num_classes": len(class_names),
        "scaler": {
            "mean": [float(v) for v in _to_list(scaler_mean)],
            "scale": [float(v) for v in _to_list(scaler_scale)],
        },
        "split_sizes": {str(k): int(v) for k, v in split_sizes.items()},
        "limitations": [
            "Feature extraction is not implemented on TelosB in this artifact.",
            "Scaler constants are exported for reproducibility and host/gateway preprocessing.",
            "The fixed-point C core consumes already standardized and quantized feature vectors.",
        ],
    }


def build_equivalence_report(
    *,
    labels: Any,
    fp32_preds: Any,
    fixed_preds: Any,
    fixed_logits: Any,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Summarize FP32-vs-fixed behavior on generated representative vectors."""
    labels_l = [int(v) for v in _to_list(labels)]
    fp32_l = [int(v) for v in _to_list(fp32_preds)]
    fixed_l = [int(v) for v in _to_list(fixed_preds)]
    logits_m = [[int(v) for v in row] for row in _to_matrix(fixed_logits)]
    flat_logits = [v for row in logits_m for v in row]
    return {
        "metadata": metadata,
        "num_vectors": len(labels_l),
        "fp32_accuracy_on_vectors": _accuracy(labels_l, fp32_l),
        "fixed_accuracy_on_vectors": _accuracy(labels_l, fixed_l),
        "fixed_vs_fp32_agreement": _agreement(fp32_l, fixed_l),
        "fixed_logit_min": min(flat_logits) if flat_logits else None,
        "fixed_logit_max": max(flat_logits) if flat_logits else None,
        "interpretation": (
            "C self-test pass means the C integer kernel matches the generated "
            "fixed-point reference vectors. The fixed_vs_fp32_agreement field is "
            "the evidence needed before claiming that the fixed-point model "
            "preserves FP32 behavior."
        ),
    }


def _c_array_2d_int16(name: str, matrix: list[list[int]]) -> str:
    rows = []
    for row in matrix:
        rows.append("    {" + ", ".join(str(int(v)) for v in row) + "}")
    row_count = len(matrix)
    col_count = len(matrix[0]) if matrix else 0
    return f"static const int16_t {name}[{row_count}][{col_count}] = {{\n" + ",\n".join(rows) + "\n};\n"


def _c_array_2d_int16_dim(name: str, matrix: list[list[int]], col_count: int) -> str:
    rows = []
    for row in matrix:
        rows.append("    {" + ", ".join(str(int(v)) for v in row) + "}")
    return f"static const int16_t {name}[{len(matrix)}][{col_count}] = {{\n" + ",\n".join(rows) + "\n};\n"


def _c_array_2d_int32_dim(name: str, matrix: list[list[int]], col_count: int) -> str:
    rows = []
    for row in matrix:
        rows.append("    {" + ", ".join(str(int(v)) for v in row) + "}")
    return f"static const int32_t {name}[{len(matrix)}][{col_count}] = {{\n" + ",\n".join(rows) + "\n};\n"

def _c_array_1d_uint8(name: str, values: list[int]) -> str:
    return f"static const uint8_t {name}[{len(values)}] = {{" + ", ".join(str(int(v)) for v in values) + "};\n"


def write_test_vectors_header(
    output_path: Path,
    q_inputs: Any,
    labels: Any,
    fp32_preds: Any,
    fixed_preds: Any,
    fixed_logits: Any,
    raw_inputs_q: Any | None = None,
    expected_preprocessed_q: Any | None = None,
) -> dict[str, Any]:
    """Write generated representative vectors for host/C self-tests."""
    inputs_m = [[int(v) for v in row] for row in _to_matrix(q_inputs)]
    labels_l = [int(v) for v in _to_list(labels)]
    fp32_l = [int(v) for v in _to_list(fp32_preds)]
    fixed_l = [int(v) for v in _to_list(fixed_preds)]
    logits_m = [[int(v) for v in row] for row in _to_matrix(fixed_logits)]
    raw_m = [[int(v) for v in row] for row in _to_matrix(raw_inputs_q)] if raw_inputs_q is not None else []
    pre_m = [[int(v) for v in row] for row in _to_matrix(expected_preprocessed_q)] if expected_preprocessed_q is not None else []
    input_dim = len(inputs_m[0]) if inputs_m else 0
    output_dim = len(logits_m[0]) if logits_m else 0
    has_raw_preprocess = bool(raw_m and pre_m)

    lines = [
        "#ifndef CUKD_WSNDS_STUDENT_A_RFKD_TEST_VECTORS_H",
        "#define CUKD_WSNDS_STUDENT_A_RFKD_TEST_VECTORS_H",
        "",
        "#include <stdint.h>",
        "",
        f"#define CUKD_TEST_VECTOR_COUNT {len(inputs_m)}",
        f"#define CUKD_TEST_INPUT_DIM {input_dim}",
        f"#define CUKD_TEST_OUTPUT_DIM {output_dim}",
        f"#define CUKD_TEST_HAS_RAW_PREPROCESS {1 if has_raw_preprocess else 0}",
        "",
        _c_array_2d_int16("cukd_test_inputs_q15", inputs_m),
        *([
            _c_array_2d_int32_dim("cukd_test_raw_inputs_q", raw_m, input_dim),
            _c_array_2d_int16_dim("cukd_test_expected_preprocessed_q15", pre_m, input_dim),
        ] if has_raw_preprocess else []),
        _c_array_1d_uint8("cukd_test_labels", labels_l),
        _c_array_1d_uint8("cukd_test_fp32_pred", fp32_l),
        _c_array_1d_uint8("cukd_test_expected_fixed_pred", fixed_l),
        _c_array_2d_int16_dim("cukd_test_expected_fixed_logits", logits_m, output_dim),
        "",
        "#endif /* CUKD_WSNDS_STUDENT_A_RFKD_TEST_VECTORS_H */",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="ascii")

    return {
        "num_test_vectors": len(inputs_m),
        "input_dim": input_dim,
        "output_dim": output_dim,
        "fp32_accuracy_on_vectors": _accuracy(labels_l, fp32_l),
        "fixed_accuracy_on_vectors": _accuracy(labels_l, fixed_l),
        "fixed_vs_fp32_agreement": _agreement(fp32_l, fixed_l),
        "has_raw_preprocess_vectors": has_raw_preprocess,
    }



def write_hil_replay_csvs(
    *,
    vectors_path: Path,
    reference_path: Path,
    q_inputs: Any,
    labels: Any,
    fp32_preds: Any,
    fixed_preds: Any,
    raw_inputs_q: Any | None = None,
) -> dict[str, Any]:
    """Write host-streamable vectors and prediction references for HIL replay."""
    q_matrix = [[int(v) for v in row] for row in _to_matrix(q_inputs)]
    raw_matrix = [[int(v) for v in row] for row in _to_matrix(raw_inputs_q)] if raw_inputs_q is not None else []
    feature_matrix = raw_matrix if raw_matrix else q_matrix
    labels_l = [int(v) for v in _to_list(labels)]
    fp32_l = [int(v) for v in _to_list(fp32_preds)]
    fixed_l = [int(v) for v in _to_list(fixed_preds)]
    if not feature_matrix:
        raise ValueError("at least one vector is required")
    if not (len(feature_matrix) == len(labels_l) == len(fp32_l) == len(fixed_l)):
        raise ValueError("vectors, labels, fp32_preds, and fixed_preds must have matching lengths")

    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    with vectors_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["row_id", *[f"f{i}" for i in range(len(feature_matrix[0]))]]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_id, features in enumerate(feature_matrix):
            writer.writerow({"row_id": row_id, **{f"f{i}": int(v) for i, v in enumerate(features)}})

    reference_path.parent.mkdir(parents=True, exist_ok=True)
    with reference_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "true_label", "fixed_pred", "fp32_pred"])
        writer.writeheader()
        for row_id, (label, fixed_pred, fp32_pred) in enumerate(zip(labels_l, fixed_l, fp32_l)):
            writer.writerow({
                "row_id": row_id,
                "true_label": label,
                "fixed_pred": fixed_pred,
                "fp32_pred": fp32_pred,
            })

    return {
        "vectors_csv": str(vectors_path),
        "reference_csv": str(reference_path),
        "rows": len(feature_matrix),
        "input_dim": len(feature_matrix[0]),
        "feature_source": (
            "raw_fixed_point_preprocess_input" if raw_matrix else "standardized_fixed_point_model_input"
        ),
    }


def _c_string_array(name: str, values: list[str]) -> str:
    escaped = [str(v).replace("\\", "\\\\").replace('"', '\\"') for v in values]
    body = ", ".join(f'"{v}"' for v in escaped)
    return f"static const char *{name}[{len(values)}] = {{{body}}};\n"


def _c_array_1d_float(name: str, values: list[float]) -> str:
    body = ", ".join(f"{float(v):.9f}f" for v in values)
    return f"static const float {name}[{len(values)}] = {{{body}}};\n"


def write_preprocessing_header(output_path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """Write feature-order and StandardScaler metadata for host/gateway preprocessing."""
    feature_names = list(metadata["feature_names"])
    class_names = list(metadata["class_names"])
    scaler_mean = [float(v) for v in metadata["scaler"]["mean"]]
    scaler_scale = [float(v) for v in metadata["scaler"]["scale"]]
    if len(feature_names) != len(scaler_mean) or len(feature_names) != len(scaler_scale):
        raise ValueError("feature_names, scaler mean, and scaler scale must have the same length")

    lines = [
        "#ifndef CUKD_WSNDS_PREPROCESS_METADATA_H",
        "#define CUKD_WSNDS_PREPROCESS_METADATA_H",
        "",
        "/* Metadata for reproducing the Python WSN-DS v2.3 preprocessing contract.",
        " * These float scaler constants are for host/gateway preprocessing or audit.",
        " * The no-FPU integer inference core consumes standardized calibrated-int16 vectors.",
        " */",
        "",
        f"#define CUKD_PREPROCESS_INPUT_DIM {len(feature_names)}",
        f"#define CUKD_PREPROCESS_NUM_CLASSES {len(class_names)}",
        "",
        _c_array_1d_float("cukd_scaler_mean", scaler_mean),
        _c_array_1d_float("cukd_scaler_scale", scaler_scale),
        _c_string_array("cukd_feature_names", feature_names),
        _c_string_array("cukd_class_names", class_names),
        "",
        "#endif /* CUKD_WSNDS_PREPROCESS_METADATA_H */",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="ascii")
    return {
        "input_dim": len(feature_names),
        "num_classes": len(class_names),
        "has_scaler_constants": True,
    }


def _c_array_1d_int32_plain(name: str, values: list[int]) -> str:
    body = ", ".join(str(int(v)) for v in values)
    return f"static const int32_t {name}[{len(values)}] = {{{body}}};\n"


def build_integer_preprocessing_metadata(
    metadata: dict[str, Any],
    *,
    output_q_frac: int,
    raw_q_frac: int = 8,
    inv_scale_q_frac: int = 20,
) -> dict[str, Any]:
    """Convert StandardScaler constants to subtract/multiply/shift metadata.

    This covers only x_std = (x_raw - mean) / scale. It deliberately does not
    implement the WSN-DS feature extraction that produces the raw 17 features.
    """
    feature_names = list(metadata["feature_names"])
    means = [float(v) for v in metadata["scaler"]["mean"]]
    scales = [float(v) for v in metadata["scaler"]["scale"]]
    if len(feature_names) != len(means) or len(feature_names) != len(scales):
        raise ValueError("feature_names, scaler mean, and scaler scale must have the same length")
    if raw_q_frac < 0 or inv_scale_q_frac < 0 or output_q_frac < 0:
        raise ValueError("fixed-point fractional widths must be non-negative")

    right_shift = int(raw_q_frac) + int(inv_scale_q_frac) - int(output_q_frac)
    if right_shift < 0:
        raise ValueError("output_q_frac cannot exceed raw_q_frac + inv_scale_q_frac")

    scaler_mean_q = [int(round(v * (1 << int(raw_q_frac)))) for v in means]
    scaler_inv_scale_q: list[int] = []
    for idx, scale in enumerate(scales):
        if scale == 0.0 or not math.isfinite(scale):
            raise ValueError(f"Invalid scaler scale at feature {idx}: {scale!r}")
        scaler_inv_scale_q.append(int(round((1.0 / scale) * (1 << int(inv_scale_q_frac)))))

    n_features = len(feature_names)
    operation_counts = {
        "features": n_features,
        "subtracts": n_features,
        "multiplies": n_features,
        "shifts": n_features,
        "saturations": n_features,
    }
    return {
        "dataset": metadata.get("dataset", "WSN-DS"),
        "scope": "Integer metadata for StandardScaler normalization only; WSN-DS feature extraction is outside this artifact.",
        "formula": "standardized_q = ((raw_q - scaler_mean_q) * scaler_inv_scale_q) >> right_shift",
        "input_dim": n_features,
        "feature_names": feature_names,
        "raw_q_frac": int(raw_q_frac),
        "inv_scale_q_frac": int(inv_scale_q_frac),
        "output_q_frac": int(output_q_frac),
        "right_shift": int(right_shift),
        "scaler_mean_q": scaler_mean_q,
        "scaler_inv_scale_q": scaler_inv_scale_q,
        "operation_counts": operation_counts,
        "integer_preprocess_ops_per_sample": sum(operation_counts.values()) - operation_counts["features"],
        "limitations": [
            "This covers StandardScaler normalization only.",
            "It does not implement WSN-DS feature extraction on a mote.",
            "Raw feature inputs must already be represented with raw_q_frac fixed-point scaling.",
        ],
    }


def write_integer_preprocessing_header(output_path: Path, fixed: dict[str, Any]) -> dict[str, Any]:
    """Write integer StandardScaler constants for no-FPU preprocessing proof."""
    input_dim = int(fixed["input_dim"])
    mean_q = [int(v) for v in fixed["scaler_mean_q"]]
    inv_scale_q = [int(v) for v in fixed["scaler_inv_scale_q"]]
    if len(mean_q) != input_dim or len(inv_scale_q) != input_dim:
        raise ValueError("integer scaler arrays must match input_dim")

    op_counts = {str(k): int(v) for k, v in fixed.get("operation_counts", {}).items()}
    ops_per_sample = int(
        op_counts.get("subtracts", 0)
        + op_counts.get("multiplies", 0)
        + op_counts.get("shifts", 0)
        + op_counts.get("saturations", 0)
    )
    lines = [
        "#ifndef CUKD_WSNDS_PREPROCESS_INT_METADATA_H",
        "#define CUKD_WSNDS_PREPROCESS_INT_METADATA_H",
        "",
        "#include <stdint.h>",
        "",
        "/* Integer StandardScaler metadata for WSN-DS fixed-point preprocessing.",
        " * Feature extraction is handled before this normalization step.",
        " */",
        "",
        f"#define CUKD_PREPROCESS_INPUT_DIM {input_dim}",
        f"#define CUKD_PREPROCESS_RAW_Q_FRAC {int(fixed['raw_q_frac'])}",
        f"#define CUKD_PREPROCESS_INV_SCALE_Q_FRAC {int(fixed['inv_scale_q_frac'])}",
        f"#define CUKD_PREPROCESS_OUTPUT_Q_FRAC {int(fixed['output_q_frac'])}",
        f"#define CUKD_PREPROCESS_RIGHT_SHIFT {int(fixed['right_shift'])}",
        f"#define CUKD_PREPROCESS_OPS_PER_SAMPLE {ops_per_sample}",
        "",
        _c_array_1d_int32_plain("cukd_scaler_mean_q", mean_q),
        _c_array_1d_int32_plain("cukd_scaler_inv_scale_q", inv_scale_q),
        "",
        "#endif /* CUKD_WSNDS_PREPROCESS_INT_METADATA_H */",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="ascii")
    return {
        "input_dim": input_dim,
        "raw_q_frac": int(fixed["raw_q_frac"]),
        "inv_scale_q_frac": int(fixed["inv_scale_q_frac"]),
        "output_q_frac": int(fixed["output_q_frac"]),
        "right_shift": int(fixed["right_shift"]),
        "integer_preprocess_ops_per_sample": ops_per_sample,
        "has_integer_scaler_constants": True,
    }

def load_state_dict(path: str) -> dict[str, Any]:
    """Load a PyTorch state_dict from a local path or git object path."""
    import torch

    if Path(path).exists():
        obj = torch.load(path, map_location="cpu")
    elif ":" in path and path.startswith(("HEAD:", "origin/", "main:", "master:")):
        blob = subprocess.check_output(["git", "show", path])
        obj = torch.load(io.BytesIO(blob), map_location="cpu")
    else:
        raise FileNotFoundError(
            f"State dict not found: {path}. Use a local .pt path or a git object path."
        )

    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError(f"Expected state_dict-like object, got {type(obj)!r}")
    return obj


def as_numpy(tensor: Any):
    return tensor.detach().cpu().numpy()


def extract_linear_layers(state_dict: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    """Return [(prefix, weight, bias), ...] for Linear layers in order."""
    weight_keys = [
        key for key, value in state_dict.items()
        if key.endswith(".weight") and hasattr(value, "ndim") and value.ndim == 2
    ]

    def sort_key(key: str) -> tuple[int, str]:
        nums = [int(part) for part in key.split(".") if part.isdigit()]
        return (nums[0] if nums else 10_000, key)

    layers = []
    for weight_key in sorted(weight_keys, key=sort_key):
        prefix = weight_key[:-len(".weight")]
        bias_key = prefix + ".bias"
        if bias_key not in state_dict:
            raise KeyError(f"Missing bias tensor for {weight_key}")
        layers.append((prefix, as_numpy(state_dict[weight_key]), as_numpy(state_dict[bias_key])))
    return layers


def choose_weight_frac(max_abs: float) -> int:
    """Choose power-of-two weight scale for int8 weights."""
    if max_abs <= 0.0:
        return 7
    frac = math.floor(math.log2(127.0 / max_abs))
    return max(0, min(7, int(frac)))


def choose_q_frac_for_range(max_abs: float, max_frac: int = 15) -> int:
    """Choose the largest signed int16 fractional scale that avoids saturation."""
    if max_abs <= 0.0 or not math.isfinite(max_abs):
        return max_frac
    frac = math.floor(math.log2(32767.0 / max_abs))
    return max(0, min(max_frac, int(frac)))



def compute_fixed_point_scale_metadata(input_frac: int, weight_frac: int, output_frac: int) -> dict[str, int]:
    accum_frac = int(input_frac) + int(weight_frac)
    return {
        "input_frac": int(input_frac),
        "weight_frac": int(weight_frac),
        "output_frac": int(output_frac),
        "accum_frac": int(accum_frac),
        "output_shift": int(accum_frac - int(output_frac)),
    }

def quantize_layer(weight, bias, input_frac: int = 15, output_frac: int = 15) -> dict[str, Any]:
    import numpy as np

    weight = np.asarray(weight, dtype=np.float32)
    bias = np.asarray(bias, dtype=np.float32)
    max_abs = float(np.max(np.abs(weight))) if weight.size else 0.0
    weight_frac = choose_weight_frac(max_abs)
    weight_scale = float(1 << weight_frac)

    q_weight = np.rint(weight * weight_scale)
    q_weight = np.clip(q_weight, -128, 127).astype(np.int8)

    scale_meta = compute_fixed_point_scale_metadata(input_frac, weight_frac, output_frac)
    accum_frac = scale_meta["accum_frac"]
    output_shift = scale_meta["output_shift"]
    bias_scale = float(1 << accum_frac)
    q_bias = np.rint(bias * bias_scale)
    q_bias = np.clip(q_bias, -(2 ** 31), (2 ** 31) - 1).astype(np.int32)

    reconstructed_w = q_weight.astype("float32") / weight_scale
    reconstructed_b = q_bias.astype("float32") / bias_scale
    return {
        "input_frac": int(input_frac),
        "output_frac": int(output_frac),
        "weight_frac": int(weight_frac),
        "accum_frac": int(accum_frac),
        "output_shift": int(output_shift),
        "weight": q_weight,
        "bias": q_bias,
        "max_weight_abs_error": float(np.max(np.abs(weight - reconstructed_w))),
        "max_bias_abs_error": float(np.max(np.abs(bias - reconstructed_b))),
    }


def c_array_2d_int8(name: str, arr) -> str:
    rows = []
    for row in arr:
        values = ", ".join(str(int(v)) for v in row)
        rows.append(f"    {{{values}}}")
    return f"static const int8_t {name}[{arr.shape[0]}][{arr.shape[1]}] = {{\n" + ",\n".join(rows) + "\n};\n"


def c_array_1d_int32(name: str, arr) -> str:
    values = ", ".join(str(int(v)) for v in arr)
    return f"static const int32_t {name}[{arr.shape[0]}] = {{{values}}};\n"


def write_header(output_path: Path, quantized_layers: list[dict[str, Any]], source: str) -> dict[str, Any]:
    dims: list[int] = []
    for idx, layer in enumerate(quantized_layers):
        weight = layer["weight"]
        bias = layer["bias"]
        if len(weight.shape) != 2:
            raise ValueError(f"Layer {idx} weight must be 2D, got shape {weight.shape}")
        in_dim = int(weight.shape[1])
        out_dim = int(weight.shape[0])
        if int(bias.shape[0]) != out_dim:
            raise ValueError(
                f"Layer {idx} bias length {bias.shape[0]} does not match output dim {out_dim}"
            )
        if idx == 0:
            dims.append(in_dim)
        elif dims[-1] != in_dim:
            raise ValueError(
                f"Layer {idx} input dim {in_dim} does not match previous output dim {dims[-1]}"
            )
        dims.append(out_dim)

    if len(dims) != 4:
        raise ValueError(f"Expected 3 Linear layers for a 17->H1->H2->5 WSN-DS MLP, got dims {dims}")
    if dims[0] != 17 or dims[-1] != 5:
        raise ValueError(f"Expected WSN-DS dims [17, H1, H2, 5], got {dims}")

    weight_bytes = sum(int(layer["weight"].size) for layer in quantized_layers)
    bias_bytes = sum(int(layer["bias"].size) * 4 for layer in quantized_layers)
    activation_bytes = sum(dims) * 2
    param_bytes = weight_bytes + bias_bytes
    macs = sum(dims[i] * dims[i + 1] for i in range(len(dims) - 1))
    guard = "CUKD_WSNDS_RFKD_INT8_MODEL_H"

    lines = [
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "#include <stdint.h>",
        "",
        "/* Generated by deployment/msp430/export_wsnds_student_a_rfkd_int8.py.",
        f" * Source artifact: {source}",
        " * Numeric format: int8 weights, int32 biases, calibrated int16 activations.",
        " * Inputs must already be WSN-DS preprocessed/standardized and quantized",
        " * using CUKD_INPUT_Q_FRAC.",
        " */",
        "",
        f"#define CUKD_INPUT_DIM {dims[0]}",
        f"#define CUKD_H1_DIM {dims[1]}",
        f"#define CUKD_H2_DIM {dims[2]}",
        f"#define CUKD_OUTPUT_DIM {dims[3]}",
        f"#define CUKD_INPUT_Q_FRAC {quantized_layers[0]['input_frac']}",
        "#define CUKD_ACTIVATION_STORAGE_BITS 16",
        f"#define CUKD_WEIGHT_BYTES {weight_bytes}",
        f"#define CUKD_BIAS_BYTES {bias_bytes}",
        f"#define CUKD_PARAM_BYTES {param_bytes}",
        f"#define CUKD_ACTIVATION_BYTES_EST {activation_bytes}",
        f"#define CUKD_MACS_PER_INFERENCE {macs}",
        "",
    ]

    for idx, layer in enumerate(quantized_layers):
        weight = layer["weight"]
        bias = layer["bias"]
        lines.extend([
            f"#define CUKD_L{idx}_IN {weight.shape[1]}",
            f"#define CUKD_L{idx}_OUT {weight.shape[0]}",
            f"#define CUKD_L{idx}_IN_FRAC {layer['input_frac']}",
            f"#define CUKD_L{idx}_OUT_FRAC {layer['output_frac']}",
            f"#define CUKD_L{idx}_W_FRAC {layer['weight_frac']}",
            f"#define CUKD_L{idx}_SHIFT {layer['output_shift']}",
            c_array_2d_int8(f"cukd_l{idx}_weight", weight),
            c_array_1d_int32(f"cukd_l{idx}_bias", bias),
            "",
        ])

    for idx, name in enumerate(CLASS_NAMES):
        lines.append(f"#define CUKD_CLASS_{idx} \"{name}\"")

    lines.extend(["", f"#endif /* {guard} */", ""])
    output_path.write_text("\n".join(lines), encoding="ascii")

    return {
        "dims": dims,
        "weight_bytes": int(weight_bytes),
        "bias_bytes": int(bias_bytes),
        "param_bytes": int(param_bytes),
        "activation_bytes_est": int(activation_bytes),
        "macs_per_inference": int(macs),
        "input_q_frac": int(quantized_layers[0]["input_frac"]),
        "layer_q_fracs": [
            {
                "input_frac": int(layer["input_frac"]),
                "weight_frac": int(layer["weight_frac"]),
                "accum_frac": int(layer["accum_frac"]),
                "output_frac": int(layer["output_frac"]),
                "output_shift": int(layer["output_shift"]),
            }
            for layer in quantized_layers
        ],
    }


def prepare_wsnds_dataset_for_e2e(csv_path: Path) -> dict[str, Any]:
    """Load WSN-DS and reproduce the v2.3 preprocessing and split contract."""
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    target_candidates = ["Attack type", "Attack_Type", "attack_type", "Attack Type", "class"]
    target_col = next((cand for cand in target_candidates if cand in df.columns), df.columns[-1])

    for id_col in ["id", "Id", "ID"]:
        if id_col in df.columns:
            df = df.drop(id_col, axis=1)
            break

    df[target_col] = df[target_col].astype(str).str.strip()
    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(df[target_col]).astype(np.int64)
    class_names = label_encoder.classes_.tolist()

    feature_df = df.drop(target_col, axis=1)
    feature_names = feature_df.columns.tolist()
    x_all = feature_df.values.astype(np.float32)

    scaler = StandardScaler()
    x_all_std = scaler.fit_transform(x_all).astype(np.float32)

    x_trainval, x_test, x_trainval_raw, x_test_raw, y_trainval, y_test = train_test_split(
        x_all_std, x_all, y_all, test_size=0.15, random_state=42, stratify=y_all
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval, y_trainval, test_size=0.1765, random_state=42, stratify=y_trainval
    )

    metadata = build_preprocessing_metadata(
        target_col=target_col,
        feature_names=feature_names,
        class_names=class_names,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        split_sizes={"train": len(x_train), "val": len(x_val), "test": len(x_test)},
    )
    return {
        "metadata": metadata,
        "x_calibration": x_trainval,
        "y_calibration": y_trainval,
        "x_test": x_test,
        "x_test_raw": x_test_raw,
        "y_test": y_test,
        "x_train_shape": list(x_train.shape),
        "x_val_shape": list(x_val.shape),
        "x_test_shape": list(x_test.shape),
    }


def select_representative_indices(y_values: Any, count: int, seed: int):
    """Select deterministic class-covered indices from a label vector."""
    import numpy as np

    y_arr = np.asarray(y_values)
    if count <= 0 or len(y_arr) == 0:
        return np.asarray([], dtype=np.int64)
    if count >= len(y_arr):
        return np.arange(len(y_arr), dtype=np.int64)

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    classes = np.unique(y_arr)
    per_class = max(1, count // max(1, len(classes)))
    for cls in classes:
        cls_idx = np.where(y_arr == cls)[0]
        rng.shuffle(cls_idx)
        selected.extend(int(i) for i in cls_idx[:per_class])

    selected_set = set(selected)
    remaining = np.asarray([i for i in range(len(y_arr)) if i not in selected_set], dtype=np.int64)
    rng.shuffle(remaining)
    selected.extend(int(i) for i in remaining[:max(0, count - len(selected))])
    return np.asarray(selected[:count], dtype=np.int64)


def forward_numpy(layers: list[tuple[str, Any, Any]], x_values: Any):
    """Run the StudentMLP linear-ReLU-linear-ReLU-linear forward pass in NumPy."""
    import numpy as np

    out = np.asarray(x_values, dtype=np.float32)
    for idx, (_, weight, bias) in enumerate(layers):
        out = out @ np.asarray(weight, dtype=np.float32).T + np.asarray(bias, dtype=np.float32)
        if idx < len(layers) - 1:
            out = np.maximum(out, 0.0)
    return out


def collect_layer_outputs(layers: list[tuple[str, Any, Any]], x_values: Any) -> list[Any]:
    """Return post-activation outputs for each MLP layer on calibration data."""
    import numpy as np

    out = np.asarray(x_values, dtype=np.float32)
    outputs = []
    for idx, (_, weight, bias) in enumerate(layers):
        out = out @ np.asarray(weight, dtype=np.float32).T + np.asarray(bias, dtype=np.float32)
        if idx < len(layers) - 1:
            out = np.maximum(out, 0.0)
        outputs.append(out)
    return outputs


def calibrate_quantized_layers(layers: list[tuple[str, Any, Any]], calibration_x: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Quantize layers with calibrated int16 activation scales from train/val data."""
    import numpy as np

    calibration_x = np.asarray(calibration_x, dtype=np.float32)
    layer_outputs = collect_layer_outputs(layers, calibration_x)
    input_max_abs = float(np.max(np.abs(calibration_x))) if calibration_x.size else 0.0
    input_frac = choose_q_frac_for_range(input_max_abs)

    quantized_layers = []
    calibration = {
        "calibration_source": "WSN-DS trainval split after v2.3 StandardScaler",
        "input_max_abs": input_max_abs,
        "input_frac": int(input_frac),
        "layers": [],
    }
    for idx, ((prefix, weight, bias), layer_out) in enumerate(zip(layers, layer_outputs)):
        output_max_abs = float(np.max(np.abs(layer_out))) if layer_out.size else 0.0
        output_frac = choose_q_frac_for_range(output_max_abs)
        q = quantize_layer(weight, bias, input_frac=input_frac, output_frac=output_frac)
        quantized_layers.append(q)
        calibration["layers"].append({
            "index": idx,
            "source_prefix": prefix,
            "output_max_abs": output_max_abs,
            "input_frac": int(q["input_frac"]),
            "weight_frac": int(q["weight_frac"]),
            "accum_frac": int(q["accum_frac"]),
            "output_frac": int(q["output_frac"]),
            "output_shift": int(q["output_shift"]),
        })
        input_frac = output_frac
    return quantized_layers, calibration


def quantize_standardized_q15(x_values: Any, input_frac: int = 15) -> tuple[Any, dict[str, Any]]:
    """Quantize standardized features with the calibrated signed int16 input fraction."""
    import numpy as np

    raw = np.rint(np.asarray(x_values, dtype=np.float32) * float(1 << input_frac))
    clipped = np.clip(raw, -32768, 32767).astype(np.int16)
    saturation_count = int(np.sum(raw != clipped.astype(np.float32)))
    return clipped, {
        "q_format": f"signed Q{int(input_frac)} calibrated standardized features",
        "input_frac": int(input_frac),
        "saturation_count": saturation_count,
        "total_values": int(raw.size),
        "saturation_fraction": float(saturation_count / raw.size) if raw.size else 0.0,
        "input_min_before_clip": float(np.min(raw)) if raw.size else None,
        "input_max_before_clip": float(np.max(raw)) if raw.size else None,
    }


def _rescale_accumulator_np(acc, shift: int):
    import numpy as np

    if shift > 0:
        acc = np.asarray(acc, dtype=np.int64)
        return np.where(acc >= 0, acc >> shift, -((-acc) >> shift))
    if shift < 0:
        return np.asarray(acc, dtype=np.int64) << (-shift)
    return acc


def simulate_integer_preprocess_q(raw_inputs_q: Any, fixed: dict[str, Any]):
    """Python reference for wsnds_preprocess_int16.c."""
    import numpy as np

    raw = np.asarray(raw_inputs_q, dtype=np.int64)
    mean_q = np.asarray(fixed["scaler_mean_q"], dtype=np.int64)
    inv_scale_q = np.asarray(fixed["scaler_inv_scale_q"], dtype=np.int64)
    centered = raw - mean_q
    scaled = centered * inv_scale_q
    shifted = _rescale_accumulator_np(scaled, int(fixed["right_shift"]))
    return np.clip(shifted, -32768, 32767).astype(np.int16)

def quantize_raw_features_q(raw_values: Any, raw_q_frac: int):
    """Encode already extracted raw WSN-DS features into fixed-point int32."""
    import numpy as np

    raw = np.rint(np.asarray(raw_values, dtype=np.float64) * float(1 << int(raw_q_frac)))
    return np.clip(raw, -2147483648, 2147483647).astype(np.int32)

def simulate_fixed_point_inference(quantized_layers: list[dict[str, Any]], q_inputs: Any) -> tuple[Any, Any]:
    """Python reference for wsnds_student_a_rfkd_int8_inference.c."""
    import numpy as np

    activations = np.asarray(q_inputs, dtype=np.int64)
    for idx, layer in enumerate(quantized_layers):
        weight = layer["weight"].astype(np.int64)
        bias = layer["bias"].astype(np.int64)
        out = activations @ weight.T + bias
        out = _rescale_accumulator_np(out, int(layer["output_shift"]))
        if idx < len(quantized_layers) - 1:
            out = np.maximum(out, 0)
        activations = np.clip(out, -32768, 32767).astype(np.int16).astype(np.int64)
    logits = activations.astype(np.int16)
    preds = np.argmax(logits.astype(np.int32), axis=1).astype(np.int64)
    return logits, preds

def generate_e2e_artifacts(
    *,
    output_dir: Path,
    layers: list[tuple[str, Any, Any]],
    quantized_layers: list[dict[str, Any]],
    dataset: dict[str, Any],
    dataset_csv: Path,
    calibration_summary: dict[str, Any],
    num_test_vectors: int,
    test_vector_seed: int,
) -> dict[str, Any]:
    """Generate dataset-bound artifacts needed for an end-to-end export audit."""
    metadata = dataset["metadata"]
    x_test = dataset["x_test"]
    x_test_raw = dataset.get("x_test_raw")
    y_test = dataset["y_test"]

    if metadata["feature_names"] and len(metadata["feature_names"]) != 17:
        raise ValueError(f"Expected 17 WSN-DS features, found {len(metadata['feature_names'])}")
    if metadata["class_names"] != CLASS_NAMES:
        raise ValueError(f"Expected WSN-DS class order {CLASS_NAMES}, got {metadata['class_names']}")

    indices = select_representative_indices(y_test, num_test_vectors, test_vector_seed)
    x_sample = x_test[indices]
    x_sample_raw = x_test_raw[indices] if x_test_raw is not None else None
    y_sample = y_test[indices]

    fp32_logits = forward_numpy(layers, x_sample)
    fp32_preds = fp32_logits.argmax(axis=1)
    input_frac = int(quantized_layers[0]["input_frac"])
    direct_q_inputs, q_stats = quantize_standardized_q15(x_sample, input_frac=input_frac)

    preprocessing_header_summary = write_preprocessing_header(output_dir / "preprocess_metadata.h", metadata)
    (output_dir / "preprocess_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="ascii")

    integer_preprocess = build_integer_preprocessing_metadata(metadata, output_q_frac=input_frac)
    integer_preprocess_summary = write_integer_preprocessing_header(
        output_dir / "preprocess_int_metadata.h",
        integer_preprocess,
    )
    (output_dir / "preprocess_int_metadata.json").write_text(
        json.dumps(integer_preprocess, indent=2),
        encoding="ascii",
    )

    raw_inputs_q = None
    expected_preprocessed_q = None
    q_inputs_for_replay = direct_q_inputs
    preprocess_reference_delta = None
    if x_sample_raw is not None:
        import numpy as np

        raw_inputs_q = quantize_raw_features_q(x_sample_raw, integer_preprocess["raw_q_frac"])
        expected_preprocessed_q = simulate_integer_preprocess_q(raw_inputs_q, integer_preprocess)
        q_inputs_for_replay = expected_preprocessed_q
        delta = np.asarray(expected_preprocessed_q, dtype=np.int32) - np.asarray(direct_q_inputs, dtype=np.int32)
        preprocess_reference_delta = {
            "max_abs_delta_vs_direct_standardized_q": int(np.max(np.abs(delta))) if delta.size else 0,
            "exact_match_vs_direct_standardized_q": bool(np.all(delta == 0)) if delta.size else True,
        }

    fixed_logits, fixed_preds = simulate_fixed_point_inference(quantized_layers, q_inputs_for_replay)

    vector_summary = write_test_vectors_header(
        output_dir / "test_vectors.h",
        q_inputs_for_replay,
        y_sample,
        fp32_preds,
        fixed_preds,
        fixed_logits,
        raw_inputs_q=raw_inputs_q,
        expected_preprocessed_q=expected_preprocessed_q,
    )
    hil_csv_summary = write_hil_replay_csvs(
        vectors_path=output_dir / "hil_replay_vectors.csv",
        reference_path=output_dir / "hil_reference_predictions.csv",
        q_inputs=q_inputs_for_replay,
        labels=y_sample,
        fp32_preds=fp32_preds,
        fixed_preds=fixed_preds,
        raw_inputs_q=raw_inputs_q,
    )
    equivalence_report = build_equivalence_report(
        labels=y_sample,
        fp32_preds=fp32_preds,
        fixed_preds=fixed_preds,
        fixed_logits=fixed_logits,
        metadata={
            "dataset_csv": str(dataset_csv),
            "test_vector_seed": int(test_vector_seed),
            "selected_indices": [int(i) for i in _to_list(indices)],
            "split_shapes": {
                "train": dataset["x_train_shape"],
                "val": dataset["x_val_shape"],
                "test": dataset["x_test_shape"],
            },
        },
    )
    equivalence_report["fixed_point_calibration"] = calibration_summary
    equivalence_report["input_quantization"] = q_stats
    equivalence_report["hardware_replay_path"] = {
        "uses_raw_preprocess_vectors": raw_inputs_q is not None,
        "mcu_input_contract": (
            "already extracted WSN-DS raw tabular features encoded as fixed-point integers"
            if raw_inputs_q is not None
            else "standardized calibrated int16 feature vectors"
        ),
        "live_feature_extraction_included": False,
        "preprocess_reference_delta": preprocess_reference_delta,
    }
    (output_dir / "equivalence_report.json").write_text(json.dumps(equivalence_report, indent=2), encoding="ascii")

    return {
        "dataset_csv": str(dataset_csv),
        "preprocess_metadata_header": preprocessing_header_summary,
        "integer_preprocess_metadata_header": integer_preprocess_summary,
        "integer_preprocess_metadata": integer_preprocess,
        "fixed_point_calibration": calibration_summary,
        "test_vectors": vector_summary,
        "hil_replay_csvs": hil_csv_summary,
        "equivalence_report": equivalence_report,
        "artifacts": [
            "model_weights.h",
            "preprocess_metadata.h",
            "preprocess_metadata.json",
            "preprocess_int_metadata.h",
            "preprocess_int_metadata.json",
            "test_vectors.h",
            "hil_replay_vectors.csv",
            "hil_reference_predictions.csv",
            "equivalence_report.json",
        ],
    }


def infer_model_label(state_dict_ref: str) -> str:
    normalized = state_dict_ref.split(":", 1)[-1].replace("\\", "/")
    name = Path(normalized).name
    if name.endswith("_fp32.pt"):
        name = name[: -len("_fp32.pt")]
    elif name.endswith(".pt"):
        name = name[:-3]
    labels = {
        "E_student_A_KD_from_RF": "WSN-DS Student A E_KD_from_RF",
        "E_student_B_KD_from_RF": "WSN-DS Student B E_KD_from_RF",
        "J_student_A_CoDistill_RF_CL": "WSN-DS Student A J_CoDistill_RF_CL",
        "J_student_B_CoDistill_RF_CL": "WSN-DS Student B J_CoDistill_RF_CL",
    }
    return labels.get(name, f"WSN-DS {name or 'student model'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-dict",
        default="results/runtime/onnx_openvino/wsnds/tmp/E_student_A_KD_from_RF_fp32.pt",
        help="Local .pt path or git object path.",
    )
    parser.add_argument(
        "--model-label",
        default=None,
        help="Human-readable model label for export_summary.json. Inferred from --state-dict when omitted.",
    )
    parser.add_argument(
        "--output-dir",
        default="deployment/msp430/generated_student_a_rfkd",
        help="Directory for generated C artifacts.",
    )
    parser.add_argument(
        "--dataset-csv",
        default=None,
        help="Optional WSN-DS CSV path. When supplied, generate preprocessing metadata, test vectors, and equivalence report.",
    )
    parser.add_argument(
        "--num-test-vectors",
        type=int,
        default=256,
        help="Number of representative WSN-DS test vectors to export when --dataset-csv is supplied.",
    )
    parser.add_argument(
        "--test-vector-seed",
        type=int,
        default=42,
        help="Seed for deterministic class-covered test-vector selection.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_dict = load_state_dict(args.state_dict)
    layers = extract_linear_layers(state_dict)
    if len(layers) != 3:
        raise ValueError(f"Expected 3 Linear layers for WSN-DS MLP, found {len(layers)}")

    dataset = prepare_wsnds_dataset_for_e2e(Path(args.dataset_csv)) if args.dataset_csv else None
    if dataset is not None:
        quantized_layers, calibration_summary = calibrate_quantized_layers(layers, dataset["x_calibration"])
    else:
        quantized_layers = [quantize_layer(weight, bias) for _, weight, bias in layers]
        calibration_summary = {
            "calibration_source": "default Q15 fallback without dataset calibration",
            "input_frac": 15,
            "layers": [],
        }

    layer_summaries = []
    for idx, ((prefix, weight, bias), q) in enumerate(zip(layers, quantized_layers)):
        layer_summaries.append({
            "index": idx,
            "source_prefix": prefix,
            "weight_shape": list(weight.shape),
            "bias_shape": list(bias.shape),
            "input_frac": int(q["input_frac"]),
            "weight_frac": int(q["weight_frac"]),
            "accum_frac": int(q["accum_frac"]),
            "output_frac": int(q["output_frac"]),
            "output_shift": int(q["output_shift"]),
            "max_weight_abs_error": q["max_weight_abs_error"],
            "max_bias_abs_error": q["max_bias_abs_error"],
        })

    byte_summary = write_header(output_dir / "model_weights.h", quantized_layers, args.state_dict)
    e2e_summary = None
    if dataset is not None:
        e2e_summary = generate_e2e_artifacts(
            output_dir=output_dir,
            layers=layers,
            quantized_layers=quantized_layers,
            dataset=dataset,
            dataset_csv=Path(args.dataset_csv),
            calibration_summary=calibration_summary,
            num_test_vectors=args.num_test_vectors,
            test_vector_seed=args.test_vector_seed,
        )

    summary = {
        "source": args.state_dict,
        "model": args.model_label or infer_model_label(args.state_dict),
        "quantization": "int8 weights, int32 biases, calibrated int16 activations",
        "class_names": CLASS_NAMES,
        "layers": layer_summaries,
        **byte_summary,
        "e2e": e2e_summary,
        "limitations": [
            "Inputs must follow the Python pipeline's 17-feature order before fixed-point integer preprocessing.",
            "This exporter proves fixed-point model-core feasibility and integer StandardScaler replay, not live on-mote feature extraction from radio frames.",
            "Run host/device equivalence tests before claiming deployed accuracy.",
            "The current hardware replay path consumes already extracted WSN-DS tabular features encoded as fixed-point integers; live feature extraction remains outside this artifact.",
        ],
    }
    (output_dir / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="ascii")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

