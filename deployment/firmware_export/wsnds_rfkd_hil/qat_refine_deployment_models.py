"""QAT-refine train-only seed-42 RF-KD models before fixed-point export.

Grounded reason:
  Archived HIL exports cited weights under wsnds_deployment_qat_outputs/
  (export_summary.json source paths). Direct PTQ of raw FP32 deployment
  weights fails the train-only exporter's macro-F1-drop gate. This script
  reproduces the historical QAT warm-start step on the verified train-only
  deployment artifacts without modifying archived evidence.

Outputs under results/wsnds/confirmation_runs_v2/deployment_seed_42_qat/:
  - qat_refined_student_{A,B}_fp32.pt  (float net weights after QAT FT)
  - qat_refinement_report.json         (metrics + provenance hashes)
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
# .../deployment/firmware_export/wsnds_rfkd_hil -> parents[2] == repo root
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.wsnds.leakage_free_rerun.tier15_common import (  # noqa: E402
    CLASS_NAMES,
    STUDENT_SPECS,
    StudentMLP,
    apply_train_scaler,
    archived_random_split,
    atomic_torch_save,
    atomic_write_json,
    load_wsnds,
    set_seed,
    sha256_file,
)

LEGACY_PATH = SCRIPT_PATH.parent / "export_wsnds_student_a_rfkd_int8.py"
DEFAULT_DEPLOYMENT = (
    REPO_ROOT / "results" / "wsnds" / "confirmation_runs_v2" / "deployment_seed_42"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "results" / "wsnds" / "confirmation_runs_v2" / "deployment_seed_42_qat"
)


class QATStudentMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, int], num_classes: int):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], num_classes),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.quant(values)
        values = self.net(values)
        return self.dequant(values)


def load_legacy():
    spec = importlib.util.spec_from_file_location("legacy_exporter", LEGACY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {LEGACY_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-root", type=Path, default=DEFAULT_DEPLOYMENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-csv", type=Path, default=REPO_ROOT / "data" / "wsnds" / "WSN-DS.csv")
    parser.add_argument("--qat-epochs", type=int, default=10)
    parser.add_argument("--qat-lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def class_weights_from_labels(y_train: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    weights = len(y_train) / (num_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def train_qat(
    pretrained: StudentMLP,
    hidden: tuple[int, int],
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    class_weights: torch.Tensor,
    epochs: int,
    lr: float,
    batch_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Return float net.state_dict after best QAT epoch (before convert)."""
    device = torch.device("cpu")
    freeze_after = max(3, epochs // 2)
    model = QATStudentMLP(17, hidden, len(CLASS_NAMES))
    model.net.load_state_dict(pretrained.net.state_dict())
    model.to(device).train()
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig("fbgemm")
    model = torch.ao.quantization.prepare_qat(model, inplace=False)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    loader = DataLoader(
        TensorDataset(X_train.to(device), y_train.to(device)),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    y_val_np = y_val.numpy()
    best_val = -1.0
    best_state = None
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        if epoch == freeze_after:
            model.apply(torch.ao.quantization.disable_observer)
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            preds = model(X_val.to(device)).argmax(dim=1).cpu().numpy()
        val_f1 = float(f1_score(y_val_np, preds, average="macro"))
        history.append({"epoch": epoch + 1, "val_macro_f1": val_f1})
        if val_f1 > best_val:
            best_val = val_f1
            best_state = copy.deepcopy(model.state_dict())
        print(f"  QAT epoch {epoch + 1}/{epochs}: val_macro_f1={val_f1:.4f}")

    if best_state is None:
        raise RuntimeError("QAT produced no valid state")
    model.load_state_dict(best_state)
    # Extract float net weights (historical export used *_fp32.pt under qat_outputs)
    float_net = {
        key.replace("net.", "", 1) if key.startswith("net.") else key: value.detach().cpu().clone()
        for key, value in model.net.state_dict().items()
    }
    # Map Sequential keys to StudentMLP net.* keys
    student_state = {f"net.{k}" if not k.startswith("net.") else k: v for k, v in float_net.items()}
    # model.net state_dict keys are already 0.weight style without net. prefix when taken from .net
    # rebuild properly:
    student_state = {f"net.{k}": v.detach().cpu().clone() for k, v in model.net.state_dict().items()}
    meta = {
        "best_val_macro_f1": best_val,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "freeze_observer_after": freeze_after,
        "history": history,
        "qconfig": "fbgemm_default_qat",
    }
    return student_state, meta


def measure_fixed_drop(
    legacy: Any,
    state_dict: dict[str, torch.Tensor],
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    layers = legacy.extract_linear_layers(state_dict)
    quantized, calibration = legacy.calibrate_quantized_layers(layers, x_train)
    fp32_logits = legacy.forward_numpy(layers, x_test)
    pred_fp = fp32_logits.argmax(axis=1).astype(np.int64)
    direct_q, _ = legacy.quantize_standardized_q15(
        x_test, input_frac=int(quantized[0]["input_frac"])
    )
    _, pred_fixed = legacy.simulate_fixed_point_inference(quantized, direct_q)
    macro_fp32 = float(f1_score(y_test, pred_fp, average="macro"))
    macro_fixed = float(f1_score(y_test, pred_fixed, average="macro"))
    agree = float(np.mean(pred_fp == pred_fixed))
    return {
        "agreement": agree,
        "macro_f1_fp32": macro_fp32,
        "macro_f1_fixed": macro_fixed,
        "macro_f1_drop": macro_fp32 - macro_fixed,
        "per_class_f1_fp32": f1_score(y_test, pred_fp, average=None).tolist(),
        "per_class_f1_fixed": f1_score(y_test, pred_fixed, average=None).tolist(),
        "calibration_layer_max_abs": [layer["output_max_abs"] for layer in calibration["layers"]],
        "passes_agreement_0_99": agree >= 0.99,
        "passes_macro_drop_0_01": (macro_fp32 - macro_fixed) <= 0.01,
        # Historical Student B archived export measured drop ~0.0266 on hil_reference
        "passes_macro_drop_0_03": (macro_fp32 - macro_fixed) <= 0.03,
    }


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    legacy = load_legacy()
    deployment_root = args.deployment_root.resolve()
    seed_root = deployment_root / "seed_42"

    dataset = load_wsnds(args.dataset_csv)
    split = archived_random_split(dataset["features"], dataset["labels"])
    scaled, _scaler = apply_train_scaler(split)

    X_train = torch.tensor(scaled["X_train"], dtype=torch.float32)
    y_train = torch.tensor(split["y_train"], dtype=torch.long)
    X_val = torch.tensor(scaled["X_validation"], dtype=torch.float32)
    y_val = torch.tensor(split["y_validation"], dtype=torch.long)
    weights = class_weights_from_labels(split["y_train"], len(CLASS_NAMES))

    report: dict[str, Any] = {
        "protocol": "train_only_seed42_qat_refine_v1",
        "parent_deployment_root": str(deployment_root),
        "parent_execution_contract_sha256": sha256_file(
            deployment_root / "execution_contract.json"
        ),
        "dataset_sha256": dataset["dataset_sha256"],
        "qat_hyperparameters": {
            "epochs": args.qat_epochs,
            "lr": args.qat_lr,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "students": {},
        "historical_note": (
            "Archived HIL export_summary source paths pointed at "
            "wsnds_deployment_qat_outputs/*_fp32.pt. Archived Student A "
            "hil_reference macro drop was ~0.0035; Student B ~0.0266. "
            "Old exporter enforced agreement only, not a 0.01 macro-drop gate."
        ),
    }

    for letter, hidden in [("A", STUDENT_SPECS["student_A"]), ("B", STUDENT_SPECS["student_B"])]:
        plain_path = seed_root / f"student_{letter}_KD_from_RF_fp32.pt"
        if not plain_path.is_file():
            raise FileNotFoundError(plain_path)
        base_state = legacy.load_state_dict(str(plain_path))
        pretrained = StudentMLP(17, hidden, len(CLASS_NAMES))
        pretrained.load_state_dict(base_state)

        print(f"\n=== Student {letter} baseline PTQ on raw deployment FP32 ===")
        baseline = measure_fixed_drop(
            legacy, base_state, scaled["X_train"], scaled["X_test"], split["y_test"]
        )
        print(json.dumps(baseline, indent=2))

        print(f"=== Student {letter} QAT fine-tune ({args.qat_epochs} epochs, CPU) ===")
        refined_state, qat_meta = train_qat(
            pretrained,
            hidden,
            X_train,
            y_train,
            X_val,
            y_val,
            weights,
            args.qat_epochs,
            args.qat_lr,
            args.batch_size,
        )
        out_path = args.output_dir / f"qat_refined_student_{letter}_fp32.pt"
        atomic_torch_save(out_path, refined_state)

        print(f"=== Student {letter} PTQ after QAT-refined FP32 ===")
        after = measure_fixed_drop(
            legacy, refined_state, scaled["X_train"], scaled["X_test"], split["y_test"]
        )
        print(json.dumps(after, indent=2))

        report["students"][letter] = {
            "base_state_sha256": sha256_file(plain_path),
            "refined_state_path": str(out_path),
            "refined_state_sha256": sha256_file(out_path),
            "baseline_ptq": baseline,
            "qat": qat_meta,
            "after_qat_ptq": after,
        }

    atomic_write_json(args.output_dir / "qat_refinement_report.json", report)
    print(f"\nWrote {args.output_dir / 'qat_refinement_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
