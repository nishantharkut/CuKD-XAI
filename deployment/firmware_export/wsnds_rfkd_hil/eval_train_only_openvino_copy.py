"""Evaluate train-only ONNX graphs under OpenVINO (software-only, copy package)."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

import sys

REPO = Path(__file__).resolve().parents[3]  # .../deployment/firmware_export/wsnds_rfkd_hil -> repo
sys.path.insert(0, str(REPO))

from experiments.wsnds.leakage_free_rerun.tier15_common import (  # noqa: E402
    STUDENT_SPECS,
    StudentMLP,
    apply_train_scaler,
    archived_random_split,
    load_wsnds,
    set_seed,
)

OUT = REPO / "results/runtime/onnx_openvino/wsnds/train_only_seed42_copy"
ONNX_DIR = OUT / "deployable_runtime_artifacts"
DEPLOY = REPO / "results/wsnds/confirmation_runs_v2/deployment_seed_42"


def load_model(pt: Path, hidden: tuple[int, int]) -> StudentMLP:
    model = StudentMLP(17, hidden, 5)
    state = torch.load(pt, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(k.startswith("net.") for k in state):
        model.load_state_dict(state)
    else:
        model.net.load_state_dict(state)
    model.eval()
    return model


def ov_predict(compiled, x: np.ndarray) -> np.ndarray:
    # OpenVINO 2024+ API
    infer = compiled.create_infer_request()
    # input name
    inp = compiled.inputs[0]
    logits = infer.infer({inp: x.astype(np.float32)})[compiled.outputs[0]]
    return np.argmax(np.asarray(logits), axis=1)


def measure_latency_ms(compiled, x_one: np.ndarray, warmup: int = 50, iters: int = 300) -> dict:
    infer = compiled.create_infer_request()
    inp = compiled.inputs[0]
    feed = {inp: x_one.astype(np.float32)}
    for _ in range(warmup):
        infer.infer(feed)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        infer.infer(feed)
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times, dtype=np.float64)
    return {
        "latency_mean_ms_b1": float(arr.mean()),
        "latency_p50_ms_b1": float(np.percentile(arr, 50)),
        "latency_p95_ms_b1": float(np.percentile(arr, 95)),
        "latency_p99_ms_b1": float(np.percentile(arr, 99)),
    }


def main() -> int:
    set_seed(42)
    from openvino import Core

    core = Core()
    dataset = load_wsnds(REPO / "data/wsnds/WSN-DS.csv")
    split = archived_random_split(dataset["features"], dataset["labels"])
    scaled, _ = apply_train_scaler(split)
    x_test = scaled["X_test"]
    y_test = split["y_test"]

    rows = []
    for stu, hidden in [("A", STUDENT_SPECS["student_A"]), ("B", STUDENT_SPECS["student_B"])]:
        name = f"E_student_{stu}_KD_from_RF_train_only"
        onnx_path = ONNX_DIR / f"{name}.onnx"
        model = load_model(
            DEPLOY / "seed_42" / f"student_{stu}_KD_from_RF_fp32.pt",
            hidden,
        )
        with torch.no_grad():
            torch_preds = np.argmax(
                model(torch.from_numpy(x_test.astype(np.float32))).numpy(), axis=1
            )

        compiled = core.compile_model(str(onnx_path), "CPU")
        preds = ov_predict(compiled, x_test)
        lat = measure_latency_ms(compiled, x_test[:1])
        row = {
            "model_name": name,
            "runtime": "openvino",
            "variant": "openvino_fp32_from_onnx",
            "status": "ok",
            "accuracy": float(accuracy_score(y_test, preds)),
            "macro_f1": float(f1_score(y_test, preds, average="macro")),
            "prediction_agreement_vs_pytorch_fp32": float((preds == torch_preds).mean()),
            "prediction_agreement_vs_onnx_runtime": None,  # filled below if ORT rows exist
            "serialized_size_kb": float(onnx_path.stat().st_size / 1024.0),
            "artifact_path": str(onnx_path),
            **lat,
        }
        rows.append(row)
        print(
            f"{name}: acc={row['accuracy']:.6f} f1={row['macro_f1']:.6f} "
            f"agree_pt={row['prediction_agreement_vs_pytorch_fp32']:.6f} "
            f"p50={row['latency_p50_ms_b1']:.4f}ms"
        )

    # compare to existing ORT summary if present
    ort_csv = OUT / "train_only_runtime_summary.csv"
    if ort_csv.is_file():
        import pandas as pd

        ort = pd.read_csv(ort_csv)
        for row in rows:
            match = ort[
                (ort.model_name == row["model_name"]) & (ort.variant == "onnx_fp32")
            ]
            if len(match) == 1:
                # re-run ORT preds for agreement if needed - use accuracy match only if 1.0
                # Prefer recomputing with ort
                try:
                    import onnxruntime as ort_rt

                    sess = ort_rt.InferenceSession(
                        row["artifact_path"], providers=["CPUExecutionProvider"]
                    )
                    in_name = sess.get_inputs()[0].name
                    ort_preds = np.argmax(
                        sess.run(None, {in_name: x_test.astype(np.float32)})[0], axis=1
                    )
                    # recompute OV preds for same order
                    compiled = core.compile_model(row["artifact_path"], "CPU")
                    ov_preds = ov_predict(compiled, x_test)
                    row["prediction_agreement_vs_onnx_runtime"] = float(
                        (ov_preds == ort_preds).mean()
                    )
                except Exception as exc:  # noqa: BLE001
                    row["prediction_agreement_vs_onnx_runtime_error"] = str(exc)

    out_csv = OUT / "train_only_openvino_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "protocol": "train_only_seed42_openvino_copy_v1",
        "openvino_version": __import__("openvino").__version__,
        "rows": rows,
        "note": "OpenVINO CPU plugin from train-only FP32 ONNX graphs; not a multi-seed table.",
    }
    (OUT / "train_only_openvino_results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print("Wrote", out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
