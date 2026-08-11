"""Export and verify ONNX/OpenVINO for the exact FG-DS HIL checkpoints.

This software-only closure is deliberately bound to the preserved seed-42
feature-group-disjoint deployment artifacts. It refuses to run when the
dataset, split, scaler, model files, rich model artifacts, or strict firmware
export reports do not describe the same experimental lineage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import sklearn
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.wsnds.leakage_free_rerun.tier15_common import (  # noqa: E402
    CLASS_NAMES,
    STUDENT_SPECS,
    StudentMLP,
    load_wsnds,
    sha256_arrays,
)


PROTOCOL_ID = "wsnds_feature_group_split_train_only_scaler_5seed_v1"
RUNTIME_PROTOCOL_ID = "wsnds_fgds_seed42_exact_runtime_v1"
EXPECTED_SEED = 42
EXPECTED_TEST_ROWS = 56_301
DEFAULT_RUN_ROOT = (
    REPO_ROOT
    / "results"
    / "wsnds"
    / "confirmation_runs_v2"
    / "remote_winterfell_feature_group_5seed_20260805"
    / "feature_group_5seed"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results"
    / "runtime"
    / "onnx_openvino"
    / "wsnds"
    / "fgds_seed42_exact"
)
DEFAULT_EXPORTS = {
    "student_A": (
        REPO_ROOT
        / "deployment"
        / "firmware_export"
        / "wsnds_rfkd_hil"
        / "generated_fgds_student_A_seed42"
    ),
    "student_B": (
        REPO_ROOT
        / "deployment"
        / "firmware_export"
        / "wsnds_rfkd_hil"
        / "generated_fgds_student_B_seed42"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=REPO_ROOT / "data" / "wsnds" / "WSN-DS.csv",
    )
    parser.add_argument("--student-a-export", type=Path, default=DEFAULT_EXPORTS["student_A"])
    parser.add_argument("--student-b-export", type=Path, default=DEFAULT_EXPORTS["student_B"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--latency-warmup", type=int, default=50)
    parser.add_argument("--latency-iters", type=int, default=300)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_json(path: Path) -> dict[str, Any]:
    path = require_file(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def require_equal(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise RuntimeError(f"{label} mismatch: observed={observed!r}, expected={expected!r}")


def prepare_atomic_output(path: Path) -> tuple[Path, Path]:
    final_path = path.resolve()
    if final_path.exists() and any(final_path.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty runtime evidence: {final_path}. "
            "Choose a new --output-dir."
        )
    if final_path.exists():
        final_path.rmdir()
    work_path = final_path.with_name(f".{final_path.name}.tmp.{os.getpid()}")
    if work_path.exists():
        raise FileExistsError(f"Stale atomic work directory exists: {work_path}")
    work_path.mkdir(parents=True)
    return final_path, work_path


def load_stored_split_and_scaler(
    run_root: Path,
    dataset: dict[str, Any],
    preprocessing: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    split_path = require_file(run_root / "split_indices.npz")
    scaler_path = require_file(run_root / "scaler_parameters.npz")
    require_equal(
        "split indices file hash",
        sha256_file(split_path),
        preprocessing["split_indices_file_sha256"],
    )
    require_equal(
        "scaler parameters file hash",
        sha256_file(scaler_path),
        preprocessing["scaler_parameters_file_sha256"],
    )

    with np.load(split_path, allow_pickle=False) as split_npz:
        required_split_keys = {"train_indices", "validation_indices", "test_indices"}
        require_equal("split array keys", set(split_npz.files), required_split_keys)
        indices = {
            name: np.asarray(split_npz[f"{name}_indices"], dtype=np.int64)
            for name in ["train", "validation", "test"]
        }

    total_rows = len(dataset["labels"])
    assigned = np.concatenate(list(indices.values()))
    require_equal("split assigned row count", len(assigned), total_rows)
    require_equal("split unique row count", len(np.unique(assigned)), total_rows)
    if assigned.min(initial=0) < 0 or assigned.max(initial=-1) >= total_rows:
        raise RuntimeError("Stored split contains an out-of-range row index")

    split_hashes = {
        name: sha256_arrays(
            dataset["features"][index],
            dataset["labels"][index],
        )
        for name, index in indices.items()
    }
    require_equal("stored split content hashes", split_hashes, preprocessing["split_hashes"])

    with np.load(scaler_path, allow_pickle=False) as scaler_npz:
        required_scaler_keys = {"mean", "scale", "var", "n_samples_seen"}
        require_equal("scaler array keys", set(scaler_npz.files), required_scaler_keys)
        mean = np.asarray(scaler_npz["mean"], dtype=np.float64)
        scale = np.asarray(scaler_npz["scale"], dtype=np.float64)
        var = np.asarray(scaler_npz["var"], dtype=np.float64)
        n_samples_seen = int(np.asarray(scaler_npz["n_samples_seen"]).reshape(-1)[0])

    require_equal("scaler fitted row count", n_samples_seen, len(indices["train"]))
    scaler = StandardScaler()
    scaler.mean_ = mean
    scaler.scale_ = scale
    scaler.var_ = var
    scaler.n_features_in_ = len(mean)
    scaler.n_samples_seen_ = n_samples_seen

    transformed: dict[str, np.ndarray] = {}
    for name, index in indices.items():
        transformed[name] = scaler.transform(dataset["features"][index]).astype(
            np.float32, copy=False
        )
    transformed_hashes = {
        name: sha256_arrays(values) for name, values in transformed.items()
    }
    require_equal(
        "transformed split hashes",
        transformed_hashes,
        preprocessing["transformed_split_hashes"],
    )
    require_equal("test row count", len(indices["test"]), EXPECTED_TEST_ROWS)
    return (
        transformed["test"],
        dataset["labels"][indices["test"]],
        {
            "indices": indices,
            "split_file": split_path,
            "split_file_sha256": sha256_file(split_path),
            "scaler_file": scaler_path,
            "scaler_file_sha256": sha256_file(scaler_path),
            "split_hashes": split_hashes,
            "transformed_split_hashes": transformed_hashes,
        },
    )


def load_expected_predictions(
    path: Path,
    expected_indices: np.ndarray,
    expected_labels: np.ndarray,
) -> np.ndarray:
    predictions: list[int] = []
    source_indices: list[int] = []
    true_labels: list[int] = []
    with require_file(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"source_row_index", "true_label", "predicted_label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError(f"Invalid prediction CSV header: {path}")
        for row in reader:
            source_indices.append(int(row["source_row_index"]))
            true_labels.append(int(row["true_label"]))
            predictions.append(int(row["predicted_label"]))
    require_equal(
        "saved prediction source indices",
        np.asarray(source_indices, dtype=np.int64).tolist(),
        expected_indices.tolist(),
    )
    require_equal(
        "saved prediction true labels",
        np.asarray(true_labels, dtype=np.int64).tolist(),
        expected_labels.tolist(),
    )
    require_equal("saved prediction row count", len(predictions), EXPECTED_TEST_ROWS)
    return np.asarray(predictions, dtype=np.int64)


def load_and_verify_student(
    *,
    student: str,
    run_root: Path,
    export_dir: Path,
    execution: dict[str, Any],
    preprocessing: dict[str, Any],
    test_indices: np.ndarray,
    test_labels: np.ndarray,
) -> dict[str, Any]:
    suffix = student.removeprefix("student_")
    seed_dir = run_root / f"seed_{EXPECTED_SEED}"
    state_path = require_file(seed_dir / f"student_{suffix}_KD_from_RF_fp32.pt")
    artifact_path = require_file(seed_dir / f"student_{suffix}_KD_from_RF_artifact.pt")
    prediction_path = require_file(
        seed_dir / f"student_{suffix}_KD_from_RF_test_predictions.csv"
    )
    completion = read_json(seed_dir / "seed_completion.json")
    export_report = read_json(export_dir / "strict_export_report.json")
    export_manifest = read_json(export_dir / "strict_export_manifest.json")
    provenance = export_report.get("provenance") or {}

    route_key = f"student_{suffix}_rf_kd"
    recorded_route = completion["student_results"][route_key]
    require_equal("completion status", completion["status"], "complete")
    require_equal("completion protocol", completion["protocol_id"], PROTOCOL_ID)
    require_equal("completion seed", completion["seed"], EXPECTED_SEED)
    require_equal("state file hash", sha256_file(state_path), recorded_route["plain_state_dict_sha256"])
    require_equal("artifact file hash", sha256_file(artifact_path), recorded_route["rich_artifact_sha256"])
    require_equal("prediction file hash", sha256_file(prediction_path), recorded_route["test_predictions_sha256"])

    require_equal("strict export status", export_report["status"], "passed")
    require_equal("strict manifest status", export_manifest["status"], "passed")
    require_equal("strict export protocol", provenance["protocol_id"], PROTOCOL_ID)
    require_equal("strict export student", provenance["student"], student)
    require_equal("strict export seed", provenance["seed"], EXPECTED_SEED)
    require_equal("strict export model hash", provenance["model_file_sha256"], sha256_file(state_path))
    require_equal("strict export artifact hash", provenance["model_artifact_sha256"], sha256_file(artifact_path))
    require_equal("strict export dataset hash", provenance["dataset_sha256"], execution["dataset_sha256"])
    require_equal("strict export split hashes", provenance["split_hashes"], preprocessing["split_hashes"])
    require_equal("strict export scaler hash", provenance["scaler_sha256"], preprocessing["scaler_sha256"])
    require_equal("strict manifest export ID", export_manifest["export_id"], export_report["export_id"])

    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    for key, expected in {
        "protocol_id": PROTOCOL_ID,
        "seed": EXPECTED_SEED,
        "student": student,
        "route": "rf_kd",
        "dataset_sha256": execution["dataset_sha256"],
        "split_hashes": preprocessing["split_hashes"],
        "scaler_sha256": preprocessing["scaler_sha256"],
        "class_names": CLASS_NAMES,
    }.items():
        require_equal(f"rich artifact {key}", artifact[key], expected)

    hidden = tuple(int(value) for value in artifact["hidden_dims"])
    require_equal("student architecture", hidden, STUDENT_SPECS[student])
    model = StudentMLP(int(artifact["input_dim"]), hidden, int(artifact["num_classes"]))
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    expected_predictions = load_expected_predictions(
        prediction_path,
        test_indices,
        test_labels,
    )
    return {
        "model": model,
        "expected_predictions": expected_predictions,
        "metrics": recorded_route["metrics"],
        "state_path_recorded": str(state_path),
        "state_sha256": sha256_file(state_path),
        "artifact_path_recorded": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "prediction_path_recorded": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "export_report_path_recorded": str(export_dir / "strict_export_report.json"),
        "export_report_sha256": sha256_file(export_dir / "strict_export_report.json"),
        "export_manifest_path_recorded": str(export_dir / "strict_export_manifest.json"),
        "export_manifest_sha256": sha256_file(export_dir / "strict_export_manifest.json"),
        "export_id": export_report["export_id"],
    }


def torch_predictions(model: StudentMLP, values: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            logits = model(torch.from_numpy(values[start : start + batch_size]))
            rows.append(logits.argmax(dim=1).numpy())
    return np.concatenate(rows).astype(np.int64, copy=False)


def export_onnx(model: StudentMLP, output: Path) -> None:
    torch.onnx.export(
        model,
        torch.zeros((1, 17), dtype=torch.float32),
        str(output),
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    import onnx

    onnx.checker.check_model(onnx.load(str(output)))


def quantize_onnx(fp32_path: Path, output: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(fp32_path), str(output), weight_type=QuantType.QInt8)


def latency_summary(
    invoke: Callable[[], None], warmup: int, iterations: int
) -> dict[str, float]:
    if warmup < 0 or iterations < 2:
        raise ValueError("Latency requires non-negative warmup and at least two iterations")
    for _ in range(warmup):
        invoke()
    samples: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        invoke()
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return {
        "mean_ms_b1": statistics.fmean(samples),
        "sample_std_ms_b1": statistics.stdev(samples),
        "median_ms_b1": statistics.median(samples),
        "p95_ms_b1": float(np.percentile(samples, 95)),
        "p99_ms_b1": float(np.percentile(samples, 99)),
        "iterations": iterations,
        "warmup_iterations": warmup,
    }


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
    }


def ort_evaluate(
    path: Path,
    x_test: np.ndarray,
    y_test: np.ndarray,
    pytorch_predictions: np.ndarray,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    import onnxruntime as ort

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(path), session_options, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    outputs: list[np.ndarray] = []
    for start in range(0, len(x_test), 4096):
        logits = session.run(
            None, {input_name: x_test[start : start + 4096]}
        )[0]
        outputs.append(np.argmax(logits, axis=1))
    predictions = np.concatenate(outputs).astype(np.int64, copy=False)
    one = {input_name: x_test[:1]}
    return {
        **metrics(y_test, predictions),
        "agreement_vs_pytorch_fp32": float(np.mean(predictions == pytorch_predictions)),
        "artifact_path": f"artifacts/{path.name}",
        "artifact_sha256": sha256_file(path),
        "serialized_size_bytes": path.stat().st_size,
        "latency": latency_summary(
            lambda: session.run(None, one), warmup, iterations
        ),
    }


def openvino_evaluate(
    path: Path,
    x_test: np.ndarray,
    y_test: np.ndarray,
    pytorch_predictions: np.ndarray,
    ort_predictions_path: Path,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    import onnxruntime as ort
    import openvino as ov

    core = ov.Core()
    compiled = core.compile_model(core.read_model(str(path)), "CPU")
    input_port = compiled.inputs[0]
    output_port = compiled.outputs[0]
    request = compiled.create_infer_request()
    outputs: list[np.ndarray] = []
    for start in range(0, len(x_test), 4096):
        logits = request.infer({input_port: x_test[start : start + 4096]})[output_port]
        outputs.append(np.argmax(np.asarray(logits), axis=1))
    predictions = np.concatenate(outputs).astype(np.int64, copy=False)

    ort_session = ort.InferenceSession(str(ort_predictions_path), providers=["CPUExecutionProvider"])
    ort_name = ort_session.get_inputs()[0].name
    ort_outputs: list[np.ndarray] = []
    for start in range(0, len(x_test), 4096):
        logits = ort_session.run(None, {ort_name: x_test[start : start + 4096]})[0]
        ort_outputs.append(np.argmax(logits, axis=1))
    ort_predictions = np.concatenate(ort_outputs).astype(np.int64, copy=False)

    one = {input_port: x_test[:1]}
    return {
        **metrics(y_test, predictions),
        "agreement_vs_pytorch_fp32": float(np.mean(predictions == pytorch_predictions)),
        "agreement_vs_onnxruntime_fp32": float(np.mean(predictions == ort_predictions)),
        "source_onnx_path": f"artifacts/{path.name}",
        "source_onnx_sha256": sha256_file(path),
        "latency": latency_summary(
            lambda: request.infer(one), warmup, iterations
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "student",
        "runtime",
        "variant",
        "accuracy",
        "macro_f1",
        "agreement_vs_pytorch_fp32",
        "agreement_vs_onnxruntime_fp32",
        "serialized_size_bytes",
        "latency_mean_ms_b1",
        "latency_sample_std_ms_b1",
        "latency_median_ms_b1",
        "latency_p95_ms_b1",
        "latency_p99_ms_b1",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_manifest(output: Path, excluded: set[Path]) -> dict[str, Any]:
    files = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.resolve() in excluded:
            continue
        files.append(
            {
                "path": path.relative_to(output).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "status": "complete",
        "protocol_id": RUNTIME_PROTOCOL_ID,
        "file_count_excluding_manifest": len(files),
        "files": files,
    }


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    final_output, output = prepare_atomic_output(args.output_dir)
    artifacts = output / "artifacts"
    artifacts.mkdir()

    execution_path = require_file(run_root / "execution_contract.json")
    preprocessing_path = require_file(run_root / "preprocessing_contract.json")
    execution = read_json(execution_path)
    preprocessing = read_json(preprocessing_path)
    require_equal("execution protocol", execution["protocol_id"], PROTOCOL_ID)
    require_equal("preprocessing protocol", preprocessing["protocol_id"], PROTOCOL_ID)
    if EXPECTED_SEED not in execution["seeds"]:
        raise RuntimeError("Seed 42 is absent from the deployment-source execution contract")

    dataset_path = require_file(args.dataset_csv)
    dataset = load_wsnds(dataset_path)
    require_equal("dataset hash", dataset["dataset_sha256"], execution["dataset_sha256"])
    require_equal("preprocessing dataset hash", preprocessing["dataset_sha256"], execution["dataset_sha256"])
    x_test, y_test, split = load_stored_split_and_scaler(
        run_root, dataset, preprocessing
    )

    export_dirs = {
        "student_A": args.student_a_export.resolve(),
        "student_B": args.student_b_export.resolve(),
    }
    student_results: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    fp32_gate_values: list[float] = []

    for student in ["student_A", "student_B"]:
        verified = load_and_verify_student(
            student=student,
            run_root=run_root,
            export_dir=export_dirs[student],
            execution=execution,
            preprocessing=preprocessing,
            test_indices=split["indices"]["test"],
            test_labels=y_test,
        )
        model: StudentMLP = verified.pop("model")
        expected_predictions: np.ndarray = verified.pop("expected_predictions")
        pytorch_pred = torch_predictions(model, x_test)
        require_equal(
            f"{student} PyTorch predictions versus preserved CSV",
            pytorch_pred.tolist(),
            expected_predictions.tolist(),
        )
        pytorch_metrics = metrics(y_test, pytorch_pred)
        require_equal(
            f"{student} PyTorch accuracy",
            pytorch_metrics["accuracy"],
            float(verified["metrics"]["accuracy"]),
        )
        require_equal(
            f"{student} PyTorch macro-F1",
            pytorch_metrics["macro_f1"],
            float(verified["metrics"]["macro_f1"]),
        )

        stem = f"{student}_rf_kd_fgds_seed42"
        fp32_path = artifacts / f"{stem}.onnx"
        int8_path = artifacts / f"{stem}_dynamic_int8.onnx"
        export_onnx(model, fp32_path)
        quantize_onnx(fp32_path, int8_path)
        ort_fp32 = ort_evaluate(
            fp32_path,
            x_test,
            y_test,
            pytorch_pred,
            args.latency_warmup,
            args.latency_iters,
        )
        ort_int8 = ort_evaluate(
            int8_path,
            x_test,
            y_test,
            pytorch_pred,
            args.latency_warmup,
            args.latency_iters,
        )
        openvino_fp32 = openvino_evaluate(
            fp32_path,
            x_test,
            y_test,
            pytorch_pred,
            fp32_path,
            args.latency_warmup,
            args.latency_iters,
        )
        fp32_gate_values.extend(
            [
                ort_fp32["agreement_vs_pytorch_fp32"],
                openvino_fp32["agreement_vs_pytorch_fp32"],
                openvino_fp32["agreement_vs_onnxruntime_fp32"],
            ]
        )

        student_results[student] = {
            "lineage": verified,
            "pytorch_fp32": pytorch_metrics,
            "onnxruntime_fp32": ort_fp32,
            "onnxruntime_dynamic_int8_weights": ort_int8,
            "openvino_fp32_from_onnx": openvino_fp32,
        }
        for runtime, variant, row in [
            ("onnxruntime", "fp32", ort_fp32),
            ("onnxruntime", "dynamic_int8_weights", ort_int8),
            ("openvino", "fp32_from_onnx", openvino_fp32),
        ]:
            latency = row["latency"]
            csv_rows.append(
                {
                    "student": student,
                    "runtime": runtime,
                    "variant": variant,
                    "accuracy": row["accuracy"],
                    "macro_f1": row["macro_f1"],
                    "agreement_vs_pytorch_fp32": row["agreement_vs_pytorch_fp32"],
                    "agreement_vs_onnxruntime_fp32": row.get(
                        "agreement_vs_onnxruntime_fp32", ""
                    ),
                    "serialized_size_bytes": row.get("serialized_size_bytes", ""),
                    "latency_mean_ms_b1": latency["mean_ms_b1"],
                    "latency_sample_std_ms_b1": latency["sample_std_ms_b1"],
                    "latency_median_ms_b1": latency["median_ms_b1"],
                    "latency_p95_ms_b1": latency["p95_ms_b1"],
                    "latency_p99_ms_b1": latency["p99_ms_b1"],
                }
            )

    if any(value != 1.0 for value in fp32_gate_values):
        raise RuntimeError(f"FP32 conversion fidelity gate failed: {fp32_gate_values}")

    summary_path = output / "runtime_summary.csv"
    report_path = output / "runtime_report.json"
    manifest_path = output / "artifact_manifest.json"
    write_csv(summary_path, csv_rows)
    report = {
        "status": "passed",
        "protocol_id": RUNTIME_PROTOCOL_ID,
        "hardware_required": False,
        "deployment_source_protocol_id": PROTOCOL_ID,
        "deployment_seed": EXPECTED_SEED,
        "test_rows": EXPECTED_TEST_ROWS,
        "dataset": {
            "path_recorded": str(dataset_path),
            "sha256": dataset["dataset_sha256"],
        },
        "source_contracts": {
            "execution_contract": {
                "path_recorded": str(execution_path),
                "sha256": sha256_file(execution_path),
            },
            "preprocessing_contract": {
                "path_recorded": str(preprocessing_path),
                "sha256": sha256_file(preprocessing_path),
            },
            "split_indices": {
                "path_recorded": str(split["split_file"]),
                "sha256": split["split_file_sha256"],
            },
            "scaler_parameters": {
                "path_recorded": str(split["scaler_file"]),
                "sha256": split["scaler_file_sha256"],
            },
        },
        "students": student_results,
        "gates": {
            "pytorch_predictions_match_preserved_csv": True,
            "onnxruntime_fp32_prediction_agreement": 1.0,
            "openvino_fp32_prediction_agreement": 1.0,
            "openvino_vs_onnxruntime_fp32_prediction_agreement": 1.0,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "onnx": __import__("onnx").__version__,
            "onnxruntime": __import__("onnxruntime").__version__,
            "openvino": __import__("openvino").__version__,
            "cpu": platform.processor(),
        },
        "timing_boundary": (
            "Single-sample synchronous CPU inference in the recorded host environment. "
            "Values are descriptive runtime measurements and are not MCU latency."
        ),
        "quantization_boundary": (
            "ONNX dynamic quantization applies int8 weights to supported operators; "
            "it is distinct from the calibrated fixed-point MCU export."
        ),
        "script": {
            "path_recorded": str(SCRIPT_PATH),
            "sha256": sha256_file(SCRIPT_PATH),
        },
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    manifest = build_manifest(output, {manifest_path.resolve()})
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(output, final_output)
    print(json.dumps({
        "status": "passed",
        "output_dir": str(final_output),
        "test_rows": EXPECTED_TEST_ROWS,
        "fp32_conversion_agreement": 1.0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
