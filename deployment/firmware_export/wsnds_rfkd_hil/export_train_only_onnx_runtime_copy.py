"""Export and evaluate train-only seed-42 RF-KD ONNX runtime package (copy-only).

Software-only: no MCU / HIL hardware required.

Produces under results/runtime/onnx_openvino/wsnds/train_only_seed42_copy/:
  - E_student_{A,B}_KD_from_RF_train_only.onnx
  - E_student_{A,B}_KD_from_RF_train_only_dynamic_int8.onnx
  - train_only_runtime_summary.csv
  - train_only_runtime_results.json
  - train_only_tier15_completeness.json  (links HIL + QAT + ONNX evidence)

Does not modify archived runtime rows or original exporters.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.wsnds.leakage_free_rerun.tier15_common import (  # noqa: E402
    STUDENT_SPECS,
    StudentMLP,
    apply_train_scaler,
    archived_random_split,
    load_wsnds,
    set_seed,
)

DEFAULT_DEPLOYMENT = (
    REPO_ROOT / "results" / "wsnds" / "confirmation_runs_v2" / "deployment_seed_42"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results"
    / "runtime"
    / "onnx_openvino"
    / "wsnds"
    / "train_only_seed42_copy"
)
DEFAULT_QAT_REPORT = (
    REPO_ROOT
    / "results"
    / "wsnds"
    / "confirmation_runs_v2"
    / "deployment_seed_42_qat"
    / "qat_refinement_report.json"
)
DEFAULT_FOUR_PAIR = (
    REPO_ROOT
    / "results"
    / "hardware_hil"
    / "train_only_scaler_copy"
    / "four_pair_summary.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-root", type=Path, default=DEFAULT_DEPLOYMENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=REPO_ROOT / "data" / "wsnds" / "WSN-DS.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latency-warmup", type=int, default=20)
    parser.add_argument("--latency-iters", type=int, default=200)
    return parser.parse_args()


def load_fp32_student(pt_path: Path, hidden: tuple[int, int]) -> StudentMLP:
    model = StudentMLP(17, hidden, 5)
    state = torch.load(pt_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # tolerate net. prefix variants
    if any(k.startswith("net.") for k in state):
        model.load_state_dict(state)
    else:
        model.net.load_state_dict(state)
    model.eval()
    return model


def export_onnx(model: StudentMLP, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 17, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )


def quantize_dynamic_onnx(fp32_path: Path, int8_path: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )


def ort_predict(session, x: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: x.astype(np.float32)})[0]
    return np.argmax(logits, axis=1)


def measure_latency_ms(session, x_one: np.ndarray, warmup: int, iters: int) -> dict[str, float]:
    input_name = session.get_inputs()[0].name
    feed = {input_name: x_one.astype(np.float32)}
    for _ in range(warmup):
        session.run(None, feed)
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        session.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times, dtype=np.float64)
    return {
        "latency_mean_ms_b1": float(arr.mean()),
        "latency_std_ms_b1": float(arr.std(ddof=0)),
        "latency_p50_ms_b1": float(np.percentile(arr, 50)),
        "latency_p95_ms_b1": float(np.percentile(arr, 95)),
        "latency_p99_ms_b1": float(np.percentile(arr, 99)),
        "throughput_samples_per_s_b1": float(1000.0 / max(arr.mean(), 1e-12)),
    }


def eval_variant(
    *,
    model_name: str,
    runtime: str,
    variant: str,
    artifact: Path,
    x_test: np.ndarray,
    y_test: np.ndarray,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(artifact), providers=["CPUExecutionProvider"])
    preds = ort_predict(session, x_test)
    latency = measure_latency_ms(session, x_test[:1], warmup=warmup, iters=iters)
    return {
        "model_name": model_name,
        "runtime": runtime,
        "variant": variant,
        "status": "ok",
        "skip_reason": "",
        "accuracy": float(accuracy_score(y_test, preds)),
        "macro_f1": float(f1_score(y_test, preds, average="macro")),
        "serialized_size_kb": float(artifact.stat().st_size / 1024.0),
        "artifact_path": str(artifact),
        "artifact_sha256": sha256_file(artifact),
        **latency,
    }


def pytorch_eval(model: StudentMLP, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    with torch.no_grad():
        logits = model(torch.from_numpy(x_test.astype(np.float32))).numpy()
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "macro_f1": float(f1_score(y_test, preds, average="macro")),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    try:
        import onnx  # noqa: F401
        import onnxruntime as ort  # noqa: F401
        from onnxruntime.quantization import quantize_dynamic  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "onnx and onnxruntime are required. "
            "Install with: pip install onnx onnxruntime"
        ) from exc

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    artifacts_dir = out / "deployable_runtime_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_wsnds(args.dataset_csv)
    split = archived_random_split(dataset["features"], dataset["labels"])
    scaled, _scaler = apply_train_scaler(split)
    x_test = scaled["X_test"]
    y_test = split["y_test"]

    students = {
        "A": {
            "model_name": "E_student_A_KD_from_RF_train_only",
            "hidden": STUDENT_SPECS["student_A"],
            "pt": args.deployment_root
            / "seed_42"
            / "student_A_KD_from_RF_fp32.pt",
        },
        "B": {
            "model_name": "E_student_B_KD_from_RF_train_only",
            "hidden": STUDENT_SPECS["student_B"],
            "pt": args.deployment_root
            / "seed_42"
            / "student_B_KD_from_RF_fp32.pt",
        },
    }

    rows: list[dict[str, Any]] = []
    pytorch_baselines: dict[str, Any] = {}

    for student_key, meta in students.items():
        pt_path: Path = meta["pt"]
        if not pt_path.is_file():
            raise FileNotFoundError(pt_path)
        model = load_fp32_student(pt_path, meta["hidden"])
        pytorch_baselines[student_key] = {
            "pt_path": str(pt_path),
            "pt_sha256": sha256_file(pt_path),
            **pytorch_eval(model, x_test, y_test),
        }

        onnx_fp32 = artifacts_dir / f"{meta['model_name']}.onnx"
        onnx_int8 = artifacts_dir / f"{meta['model_name']}_dynamic_int8.onnx"
        export_onnx(model, onnx_fp32)
        quantize_dynamic_onnx(onnx_fp32, onnx_int8)

        for variant, path in (
            ("onnx_fp32", onnx_fp32),
            ("onnx_dynamic_int8", onnx_int8),
        ):
            row = eval_variant(
                model_name=meta["model_name"],
                runtime="onnxruntime",
                variant=variant,
                artifact=path,
                x_test=x_test,
                y_test=y_test,
                warmup=args.latency_warmup,
                iters=args.latency_iters,
            )
            # agreement vs pytorch fp32 preds
            import onnxruntime as ort

            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            preds = ort_predict(sess, x_test)
            with torch.no_grad():
                torch_preds = np.argmax(
                    model(torch.from_numpy(x_test.astype(np.float32))).numpy(),
                    axis=1,
                )
            row["prediction_agreement_vs_pytorch_fp32"] = float(
                (preds == torch_preds).mean()
            )
            row["macro_f1_delta_vs_pytorch_fp32"] = float(
                row["macro_f1"] - pytorch_baselines[student_key]["macro_f1"]
            )
            rows.append(row)
            print(
                f"{meta['model_name']} {variant}: "
                f"acc={row['accuracy']:.6f} f1={row['macro_f1']:.6f} "
                f"agree={row['prediction_agreement_vs_pytorch_fp32']:.6f} "
                f"size_kb={row['serialized_size_kb']:.3f}"
            )

    write_csv(out / "train_only_runtime_summary.csv", rows)

    qat_payload = None
    if DEFAULT_QAT_REPORT.is_file():
        qat_payload = json.loads(DEFAULT_QAT_REPORT.read_text(encoding="utf-8"))

    four_pair = None
    if DEFAULT_FOUR_PAIR.is_file():
        four_pair = json.loads(DEFAULT_FOUR_PAIR.read_text(encoding="utf-8"))

    results = {
        "protocol": "train_only_seed42_onnx_runtime_copy_v1",
        "hardware_required": False,
        "deployment_root": str(args.deployment_root),
        "deployment_root_sha256_execution_contract": (
            sha256_file(args.deployment_root / "execution_contract.json")
            if (args.deployment_root / "execution_contract.json").is_file()
            else None
        ),
        "dataset_csv": str(args.dataset_csv),
        "n_test": int(len(y_test)),
        "pytorch_fp32_baselines": pytorch_baselines,
        "onnx_runtime_rows": rows,
        "qat_refinement_report_path": str(DEFAULT_QAT_REPORT)
        if DEFAULT_QAT_REPORT.is_file()
        else None,
        "qat_note": (
            "QAT refine on train-only seed-42 lowered absolute macro-F1 for Student A "
            "despite tighter PTQ drop; HIL used direct PTQ of FP32 train-only weights "
            "with copy gate macro_drop<=0.03, not the QAT-refined weights."
        ),
        "qat_summary_excerpt": {
            "protocol": (qat_payload or {}).get("protocol"),
            "students": {
                k: {
                    "baseline_ptq_macro_f1_fp32": v.get("baseline_ptq", {}).get(
                        "macro_f1_fp32"
                    ),
                    "baseline_ptq_macro_f1_drop": v.get("baseline_ptq", {}).get(
                        "macro_f1_drop"
                    ),
                    "after_qat_macro_f1_fp32": v.get("after_qat_ptq", {}).get(
                        "macro_f1_fp32"
                    ),
                    "after_qat_macro_f1_drop": v.get("after_qat_ptq", {}).get(
                        "macro_f1_drop"
                    ),
                    "qat_best_val_macro_f1": v.get("qat", {}).get("best_val_macro_f1"),
                }
                for k, v in ((qat_payload or {}).get("students") or {}).items()
            },
        }
        if qat_payload
        else None,
    }
    (out / "train_only_runtime_results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )

    # Tier-1.5 completeness ledger (software + already-collected HIL)
    completeness = {
        "protocol": "train_only_seed42_tier15_completeness_v1",
        "scope": {
            "students": ["A", "B"],
            "boards": ["arduino_r4", "esp32c3"],
            "seed": 42,
            "scaler": "train_only",
            "split": "archived_random_state_42",
        },
        "software_runtime": {
            "status": "complete",
            "path": str(out / "train_only_runtime_results.json"),
            "onnx_fp32_and_dynamic_int8": True,
            "openvino": False,
            "openvino_note": "OpenVINO not re-run in this copy package; ONNX Runtime covers host conversion evidence.",
        },
        "qat_probe": {
            "status": "complete_probe_not_selected_for_hil",
            "path": str(DEFAULT_QAT_REPORT) if DEFAULT_QAT_REPORT.is_file() else None,
            "selected_for_mcu_export": False,
            "reason": "QAT reduced absolute Student A F1; PTQ of train-only FP32 used for HIL.",
        },
        "fixed_point_export": {
            "status": "complete",
            "paths": {
                "A": str(
                    REPO_ROOT
                    / "deployment/firmware_export/wsnds_rfkd_hil/generated_train_only_student_A_seed42_copy"
                ),
                "B": str(
                    REPO_ROOT
                    / "deployment/firmware_export/wsnds_rfkd_hil/generated_train_only_student_B_seed42_copy"
                ),
            },
        },
        "hardware_hil_four_pair": {
            "status": "complete" if four_pair and four_pair.get("matrix_complete") else "partial_or_missing",
            "path": str(DEFAULT_FOUR_PAIR) if DEFAULT_FOUR_PAIR.is_file() else None,
            "matrix_complete": bool(four_pair.get("matrix_complete")) if four_pair else False,
            "all_pairs_agree_1_0": False,
        },
        "claim_boundary": {
            "archived_10seed_tables": "still primary multi-seed predictive evidence (pre-split scaler lineage)",
            "train_only_seed42": (
                "deployment confirmation + conversion + four-board HIL under train-fitted scaler; "
                "single seed, not a replacement ten-seed distribution"
            ),
        },
    }
    if four_pair:
        pairs = []
        for board_key in ("arduino_r4", "esp32c3"):
            block = four_pair.get(board_key) or {}
            for p in block.get("pairs") or []:
                pairs.append(p)
        completeness["hardware_hil_four_pair"]["all_pairs_agree_1_0"] = all(
            float(p.get("mcu_vs_fixed_reference_agreement", 0)) == 1.0
            and p.get("all_status_ok") is True
            and int(p.get("n", 0)) == 56200
            for p in pairs
        ) and len(pairs) == 4
        completeness["hardware_hil_four_pair"]["pairs"] = [
            {
                "board": p.get("board") or ("arduino_r4" if "arduino" in str(p) else "esp32c3"),
                "student": p.get("student"),
                "agree": p.get("mcu_vs_fixed_reference_agreement"),
                "macro_f1": p.get("macro_f1"),
                "latency_us_mean": p.get("latency_us_mean"),
                "n": p.get("n"),
            }
            for p in pairs
        ]

    (out / "train_only_tier15_completeness.json").write_text(
        json.dumps(completeness, indent=2) + "\n", encoding="utf-8"
    )
    print("Wrote", out / "train_only_runtime_summary.csv")
    print("Wrote", out / "train_only_runtime_results.json")
    print("Wrote", out / "train_only_tier15_completeness.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
