#!/usr/bin/env python3
"""
Reuse completed Edge-IIoTset results to compute exact weighted/micro F1 and
literature-comparison gaps without retraining any model.

The completed literature-comparable run stores per-seed confusion matrices.
Those matrices are enough to compute macro, weighted, and micro metrics exactly.
This script reads the existing JSON artifact, writes a metric summary CSV, and
generates a single professor-facing literature comparison table.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACT_DIR = "Final/edgeiiot_v23_literature_comparable_selected_capacity_outputs"
DEFAULT_OUTPUT_DIR = "Edge-IIOT-run/edgeiiot_literature_metric_gap_outputs"
RESULTS_JSON_NAME = "edgeiiot_v23_results.json"


LITERATURE_ROWS = [
    {
        "paper_key": "ferrag_2022_dnn",
        "paper_work": "Ferrag et al., Edge-IIoTset",
        "year": 2022,
        "dataset_task": "Edge-IIoTset, 15-class",
        "paper_model": "DNN",
        "paper_accuracy": 94.67,
        "paper_precision": None,
        "paper_recall": None,
        "paper_f1": None,
        "paper_f1_type": "not_reported",
        "paper_footprint": "NR",
        "our_key": "student_c_rfkd",
        "interpretation": "We exceed the original 15-class DNN accuracy with a compressed student.",
    },
    {
        "paper_key": "ferrag_2022_rf",
        "paper_work": "Ferrag et al., Edge-IIoTset",
        "year": 2022,
        "dataset_task": "Edge-IIoTset, 15-class",
        "paper_model": "RF",
        "paper_accuracy": 80.83,
        "paper_precision": None,
        "paper_recall": None,
        "paper_f1": None,
        "paper_f1_type": "not_reported",
        "paper_footprint": "NR",
        "our_key": "student_c_rfkd",
        "interpretation": "Large accuracy improvement over the original RF baseline.",
    },
    {
        "paper_key": "ferrag_2022_svm",
        "paper_work": "Ferrag et al., Edge-IIoTset",
        "year": 2022,
        "dataset_task": "Edge-IIoTset, 15-class",
        "paper_model": "SVM",
        "paper_accuracy": 77.61,
        "paper_precision": None,
        "paper_recall": None,
        "paper_f1": None,
        "paper_f1_type": "not_reported",
        "paper_footprint": "NR",
        "our_key": "student_c_rfkd",
        "interpretation": "Large accuracy improvement over the original SVM baseline.",
    },
    {
        "paper_key": "ferrag_2022_knn",
        "paper_work": "Ferrag et al., Edge-IIoTset",
        "year": 2022,
        "dataset_task": "Edge-IIoTset, 15-class",
        "paper_model": "KNN",
        "paper_accuracy": 79.18,
        "paper_precision": None,
        "paper_recall": None,
        "paper_f1": None,
        "paper_f1_type": "not_reported",
        "paper_footprint": "NR",
        "our_key": "student_c_rfkd",
        "interpretation": "Large accuracy improvement over the original KNN baseline.",
    },
    {
        "paper_key": "ferrag_2022_dt",
        "paper_work": "Ferrag et al., Edge-IIoTset",
        "year": 2022,
        "dataset_task": "Edge-IIoTset, 15-class",
        "paper_model": "DT",
        "paper_accuracy": 67.11,
        "paper_precision": None,
        "paper_recall": None,
        "paper_f1": None,
        "paper_f1_type": "not_reported",
        "paper_footprint": "NR",
        "our_key": "student_c_rfkd",
        "interpretation": "Large accuracy improvement over the original DT baseline.",
    },
    {
        "paper_key": "diab_2025_lightgbm",
        "paper_work": "Diab et al., Hardware-Aware ML/DL",
        "year": 2025,
        "dataset_task": "Edge-IIoTset, 15-class",
        "paper_model": "LightGBM",
        "paper_accuracy": 95.25,
        "paper_precision": None,
        "paper_recall": None,
        "paper_f1": 94.74,
        "paper_f1_type": "unspecified",
        "paper_footprint": "74.93 KB flash, 1.13 KB RAM, 1.20K ops",
        "our_key": "student_c_rfkd",
        "interpretation": "Accuracy and storage are competitive; F1 type is unspecified, so the best-F1 reference is only optimistic context.",
    },
    {
        "paper_key": "diab_2025_xgboost",
        "paper_work": "Diab et al., Hardware-Aware ML/DL",
        "year": 2025,
        "dataset_task": "Edge-IIoTset, 15-class",
        "paper_model": "XGBoost",
        "paper_accuracy": 95.11,
        "paper_precision": None,
        "paper_recall": None,
        "paper_f1": 94.42,
        "paper_f1_type": "unspecified",
        "paper_footprint": "266.59 KB flash, 0.51 KB RAM, 4.27K ops",
        "our_key": "student_c_rfkd",
        "interpretation": "Accuracy is higher and storage is smaller; F1 type is unspecified, so the best-F1 reference is only optimistic context.",
    },
    {
        "paper_key": "diab_2025_rf",
        "paper_work": "Diab et al., Hardware-Aware ML/DL",
        "year": 2025,
        "dataset_task": "Edge-IIoTset, 15-class",
        "paper_model": "RF",
        "paper_accuracy": 94.12,
        "paper_precision": None,
        "paper_recall": None,
        "paper_f1": 93.21,
        "paper_f1_type": "unspecified",
        "paper_footprint": "211.22 KB flash, 4.61 KB RAM, 3.38K ops",
        "our_key": "student_c_rfkd",
        "interpretation": "Accuracy is higher and storage is smaller; F1 type is unspecified, so the best-F1 reference is only optimistic context.",
    },
    {
        "paper_key": "diab_2025_hwnas_cnn",
        "paper_work": "Diab et al., Hardware-Aware ML/DL",
        "year": 2025,
        "dataset_task": "Edge-IIoTset, 15-class",
        "paper_model": "HW-NAS 1D-CNN",
        "paper_accuracy": 96.73,
        "paper_precision": None,
        "paper_recall": None,
        "paper_f1": 97.24,
        "paper_f1_type": "unspecified",
        "paper_footprint": "190.34 KB flash, 6.89 KB RAM, 838.89K FLOPs",
        "our_key": "student_c_rfkd",
        "interpretation": "Accuracy is essentially matched with smaller storage; CNN F1 remains stronger.",
    },
    {
        "paper_key": "abualhassan_2025_tinydnn",
        "paper_work": "Abualhassan et al., IIoT-TinyDNN",
        "year": 2025,
        "dataset_task": "Edge-IIoTset",
        "paper_model": "TinyDNN",
        "paper_accuracy": 92.99,
        "paper_precision": None,
        "paper_recall": None,
        "paper_f1": None,
        "paper_f1_type": "not_reported",
        "paper_footprint": "2,255 params, 4.38K FLOPs",
        "our_key": "student_a_lgbmkd",
        "interpretation": "Similar parameter scale and higher accuracy.",
    },
    {
        "paper_key": "abualhassan_2025_tinycnn",
        "paper_work": "Abualhassan et al., IIoT-TinyDNN",
        "year": 2025,
        "dataset_task": "Edge-IIoTset",
        "paper_model": "TinyCNN",
        "paper_accuracy": 95.55,
        "paper_precision": None,
        "paper_recall": None,
        "paper_f1": None,
        "paper_f1_type": "not_reported",
        "paper_footprint": "5,967 params, 32.13K FLOPs",
        "our_key": "student_b_lgbmkd",
        "interpretation": "Similar parameter scale, higher accuracy, and lower FLOPs.",
    },
    {
        "paper_key": "hasan_2025_ae",
        "paper_work": "Hasan et al., Autoencoder Feature Learning",
        "year": 2025,
        "dataset_task": "Edge-IIoTset, multiclass",
        "paper_model": "AE + DT/XGB/LGBM/LDA/TabNet/LSTM",
        "paper_accuracy": 99.94,
        "paper_precision": None,
        "paper_recall": None,
        "paper_f1": 99.94,
        "paper_f1_type": "unspecified",
        "paper_footprint": "NR, Jetson Nano inference reported",
        "our_key": "student_c_rfkd",
        "interpretation": "Their metrics are higher; our defense is smaller compressed-student deployability.",
    },
    {
        "paper_key": "abdi_2025_cnn_15class",
        "paper_work": "Abdi et al., CNN Multiclass Attack Classification",
        "year": 2025,
        "dataset_task": "DNN-EdgeIIoT, 15-class",
        "paper_model": "CNN",
        "paper_accuracy": 95.50,
        "paper_precision": None,
        "paper_recall": None,
        "paper_f1": 94.60,
        "paper_f1_type": "unspecified",
        "paper_footprint": "NR",
        "our_key": "student_c_rfkd",
        "interpretation": "Our compressed student is higher in accuracy; their F1 type is unspecified and their CNN is not presented as an ultra-small deployment artifact.",
    },
    {
        "paper_key": "lens_xai_2025_student",
        "paper_work": "Yagiz and Goktas, LENS-XAI",
        "year": 2025,
        "dataset_task": "Edge-IIoTset, multiclass",
        "paper_model": "LENS-XAI Student",
        "paper_accuracy": 95.31,
        "paper_precision": 95.74,
        "paper_recall": 95.31,
        "paper_f1": 95.36,
        "paper_f1_type": "unspecified",
        "paper_footprint": "NR, KD/VAE/XAI with 10% training-data claim",
        "our_key": "student_c_rfkd",
        "interpretation": "Accuracy is higher; their F1 type is unspecified, while our macro-F1 remains lower because it stresses minority classes.",
    },
    {
        "paper_key": "salehiyan_2025_transformer_gan_ae",
        "paper_work": "Salehiyan et al., Transformer-GAN-AE",
        "year": 2025,
        "dataset_task": "Edge-IIoTset, 70/15/15 split",
        "paper_model": "Transformer-GAN-AE",
        "paper_accuracy": 98.63,
        "paper_precision": None,
        "paper_recall": 98.79,
        "paper_f1": None,
        "paper_f1_type": "not_reported",
        "paper_footprint": "RTX 3090 workstation; footprint NR",
        "our_key": "student_c_rfkd",
        "interpretation": "Their heavy hybrid DL model has higher accuracy; our contribution is much smaller compressed-student deployment.",
    },
    {
        "paper_key": "wo_xgb_2025",
        "paper_work": "WO-XGB Feature-Level Ensemble",
        "year": 2025,
        "dataset_task": "Edge-IIoTset",
        "paper_model": "WO-XGB",
        "paper_accuracy": 99.98,
        "paper_precision": 99.97,
        "paper_recall": 99.97,
        "paper_f1": 99.97,
        "paper_f1_type": "unspecified",
        "paper_footprint": "NR, XGBoost ensemble",
        "our_key": "student_c_rfkd",
        "interpretation": "Their ensemble is much stronger in raw metrics; not an ultra-small neural student comparison.",
    },
    {
        "paper_key": "cst_afnet_2025",
        "paper_work": "Ishtiaq et al., CST-AFNet",
        "year": 2025,
        "dataset_task": "Edge-IIoTset, 15 attack types + benign",
        "paper_model": "CST-AFNet CNN-BiGRU dual attention",
        "paper_accuracy": 99.97,
        "paper_precision": 99.30,
        "paper_recall": 99.30,
        "paper_f1": 99.30,
        "paper_f1_type": "unspecified",
        "paper_footprint": "NR, multi-scale CNN + BiGRU + dual attention",
        "our_key": "student_c_rfkd",
        "interpretation": "Their deep attention model is raw-metric superior but not comparable to a 61 KB compressed MLP target.",
    },
    {
        "paper_key": "neurosymbolic_2026",
        "paper_work": "Neuro-Symbolic Edge IDS",
        "year": 2026,
        "dataset_task": "Edge-IIoTset",
        "paper_model": "Neuro-symbolic KD framework",
        "paper_accuracy": 94.30,
        "paper_precision": None,
        "paper_recall": None,
        "paper_f1": 93.50,
        "paper_f1_type": "macro",
        "paper_footprint": "37% memory reduction, 54% latency reduction; absolute size NR",
        "our_key": "student_c_rfkd",
        "interpretation": "Our accuracy is higher; their explicitly reported macro-F1 and interpretability claims remain stronger.",
    },
    {
        "paper_key": "hybrid_llm_hgb_2026",
        "paper_work": "Hybrid LLM/HGB IDS",
        "year": 2026,
        "dataset_task": "Edge-IIoTset, leakage-safe hybrid features",
        "paper_model": "BERT embeddings + RF selection + HGB",
        "paper_accuracy": 98.19,
        "paper_precision": 98.21,
        "paper_recall": 98.19,
        "paper_f1": 98.19,
        "paper_f1_type": "weighted",
        "paper_footprint": "Frozen BERT + HGB; footprint NR",
        "our_key": "student_c_rfkd",
        "interpretation": "Their hybrid representation is stronger in macro/weighted metrics; our model is far smaller and standalone.",
    },
    {
        "paper_key": "abdulkareem_2024_fi_sel",
        "paper_work": "Abdulkareem et al., FI-SEL",
        "year": 2024,
        "dataset_task": "Edge-IIoTset, 8 selected features",
        "paper_model": "Feature-importance stacked ensemble",
        "paper_accuracy": 87.37,
        "paper_precision": 90.65,
        "paper_recall": 77.73,
        "paper_f1": 80.88,
        "paper_f1_type": "unspecified",
        "paper_footprint": "8 features; absolute model size NR",
        "our_key": "student_a_lgbmkd",
        "interpretation": "Our smallest student is much stronger in accuracy; F1 type is unspecified, so the best-F1 reference is only optimistic context.",
    },
    {
        "paper_key": "qathrady_2024_sacnn_ids",
        "paper_work": "Qathrady et al., SACNN-IDS",
        "year": 2024,
        "dataset_task": "Edge-IIoTset",
        "paper_model": "Self-attention CNN IDS",
        "paper_accuracy": 99.95,
        "paper_precision": 99.79,
        "paper_recall": 99.80,
        "paper_f1": 99.79,
        "paper_f1_type": "unspecified",
        "paper_footprint": "NR, deep self-attention CNN",
        "our_key": "student_c_rfkd",
        "interpretation": "Raw metrics are much higher; this is a heavy accuracy benchmark, not a compression-equivalent model.",
    },
    {
        "paper_key": "alshehri_2024_sadcnn",
        "paper_work": "Alshehri et al., SA-DCNN",
        "year": 2024,
        "dataset_task": "Edge-IIoTset",
        "paper_model": "Self-attention DCNN",
        "paper_accuracy": 99.96,
        "paper_precision": 99.83,
        "paper_recall": 99.79,
        "paper_f1": 99.81,
        "paper_f1_type": "unspecified",
        "paper_footprint": "NR, self-attention DCNN",
        "our_key": "student_c_rfkd",
        "interpretation": "Raw metrics are much higher; not comparable to our compressed 61 KB student objective.",
    },
    {
        "paper_key": "cao_2025_feddynst",
        "paper_work": "Cao et al., FedDynST",
        "year": 2025,
        "dataset_task": "Edge-IIoTset",
        "paper_model": "FedDynST",
        "paper_accuracy": 97.28,
        "paper_precision": 97.14,
        "paper_recall": 91.28,
        "paper_f1": 97.62,
        "paper_f1_type": "unspecified",
        "paper_footprint": "NR, federated/dynamic model",
        "our_key": "student_c_rfkd",
        "interpretation": "Accuracy is close but lower than their reported value; their model is not a tiny standalone student.",
    },
    {
        "paper_key": "gao_2026_tcn_selected",
        "paper_work": "Gao et al., Lightweight TCN",
        "year": 2026,
        "dataset_task": "Edge-IIoTset, multiclass, selected features",
        "paper_model": "TCN, 22 features",
        "paper_accuracy": 93.79,
        "paper_precision": 93.33,
        "paper_recall": 93.79,
        "paper_f1": 93.13,
        "paper_f1_type": "unspecified",
        "paper_footprint": "16.25 KB",
        "our_key": "student_a_lgbmkd",
        "interpretation": "Accuracy and size are stronger; F1 type is unspecified, so the best-F1 reference is only optimistic context.",
    },
    {
        "paper_key": "gao_2026_tcn_all",
        "paper_work": "Gao et al., Lightweight TCN",
        "year": 2026,
        "dataset_task": "Edge-IIoTset, multiclass, all features",
        "paper_model": "TCN, all features",
        "paper_accuracy": 94.24,
        "paper_precision": 93.77,
        "paper_recall": 94.24,
        "paper_f1": 93.71,
        "paper_f1_type": "unspecified",
        "paper_footprint": "25.37 KB",
        "our_key": "student_b_lgbmkd",
        "interpretation": "Accuracy and size are stronger; F1 type is unspecified, so the best-F1 reference is only optimistic context.",
    },
    {
        "paper_key": "seed_2026",
        "paper_work": "SEED: Edge Transformer to IoT Decisions",
        "year": 2026,
        "dataset_task": "Edge-IIoTset, offloaded edge/IoT split",
        "paper_model": "EdgeBERT + IoT classifier",
        "paper_accuracy": 99.99,
        "paper_precision": None,
        "paper_recall": None,
        "paper_f1": None,
        "paper_f1_type": "not_reported",
        "paper_footprint": "137 KB IoT classifier + 40.6 MB EdgeBERT",
        "our_key": "student_c_rfkd",
        "interpretation": "Not a direct standalone tiny-student comparison because their result uses edge offloading.",
    },
]


OUR_MODEL_SELECTIONS = {
    "student_a_lgbmkd": {
        "student_name": "student_A_32_16",
        "config": "E3_KD_from_LightGBM",
        "label": "Student A LightGBM-KD",
    },
    "student_b_lgbmkd": {
        "student_name": "student_B_64_32",
        "config": "E3_KD_from_LightGBM",
        "label": "Student B LightGBM-KD",
    },
    "student_c_rfkd": {
        "student_name": "student_C_128_64",
        "config": "E_KD_from_RF",
        "label": "Student C RF-KD",
    },
}


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def harmonic_mean(precision: float, recall: float) -> float:
    return safe_div(2.0 * precision * recall, precision + recall)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    value_mean = mean(values)
    assert value_mean is not None
    return math.sqrt(sum((value - value_mean) ** 2 for value in values) / (len(values) - 1))


def metrics_from_confusion_matrix(confusion_matrix: list[list[int]]) -> dict[str, Any]:
    """Compute macro, weighted, and micro metrics from a multiclass confusion matrix."""

    if not confusion_matrix or not all(len(row) == len(confusion_matrix) for row in confusion_matrix):
        raise ValueError("confusion_matrix must be a non-empty square matrix")

    num_classes = len(confusion_matrix)
    row_sums = [sum(row) for row in confusion_matrix]
    col_sums = [sum(confusion_matrix[row_idx][col_idx] for row_idx in range(num_classes)) for col_idx in range(num_classes)]
    diagonal = [confusion_matrix[idx][idx] for idx in range(num_classes)]
    support_total = sum(row_sums)
    correct_total = sum(diagonal)

    per_class_precision = [safe_div(diagonal[idx], col_sums[idx]) for idx in range(num_classes)]
    per_class_recall = [safe_div(diagonal[idx], row_sums[idx]) for idx in range(num_classes)]
    per_class_f1 = [harmonic_mean(per_class_precision[idx], per_class_recall[idx]) for idx in range(num_classes)]

    macro_precision = mean(per_class_precision) or 0.0
    macro_recall = mean(per_class_recall) or 0.0
    macro_f1 = mean(per_class_f1) or 0.0
    weighted_precision = safe_div(
        sum(per_class_precision[idx] * row_sums[idx] for idx in range(num_classes)),
        support_total,
    )
    weighted_recall = safe_div(
        sum(per_class_recall[idx] * row_sums[idx] for idx in range(num_classes)),
        support_total,
    )
    weighted_f1 = safe_div(
        sum(per_class_f1[idx] * row_sums[idx] for idx in range(num_classes)),
        support_total,
    )
    micro_precision = safe_div(correct_total, support_total)
    micro_recall = safe_div(correct_total, support_total)
    micro_f1 = harmonic_mean(micro_precision, micro_recall)

    return {
        "accuracy": safe_div(correct_total, support_total),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "support_total": support_total,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "class_support": row_sums,
    }


def add_sample(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    seed: int,
    student_name: str,
    config: str,
    payload: dict[str, Any],
) -> None:
    confusion_matrix = payload.get("confusion_matrix")
    if confusion_matrix is None:
        return
    metrics = metrics_from_confusion_matrix(confusion_matrix)
    sample = {
        "seed": seed,
        "student_name": student_name,
        "config": config,
        "params": payload.get("params"),
        "size_kb": payload.get("size_kb"),
        "flops_per_sample": payload.get("flops_per_sample"),
        **metrics,
    }
    grouped.setdefault((student_name, config), []).append(sample)


def aggregate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    first = samples[0]
    row: dict[str, Any] = {
        "student_name": first["student_name"],
        "config": first["config"],
        "n_seeds": len(samples),
        "params": first.get("params"),
        "size_kb": first.get("size_kb"),
        "flops_per_sample": first.get("flops_per_sample"),
        "support_total_per_seed": first.get("support_total"),
    }
    for metric_name in [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "micro_precision",
        "micro_recall",
        "micro_f1",
    ]:
        values = [float(sample[metric_name]) for sample in samples]
        row[f"{metric_name}_mean"] = mean(values)
        row[f"{metric_name}_std"] = std(values)
    return row


def collect_exact_metric_rows(result_data: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for seed_result in result_data.get("seed_results", []):
        seed = int(seed_result["seed"])
        for config, payload in seed_result.get("teacher_metrics", {}).items():
            add_sample(grouped, seed, "teacher", config, payload)
        for student_name, student_payload in seed_result.get("students", {}).items():
            for config, payload in student_payload.get("configs", {}).items():
                add_sample(grouped, seed, student_name, config, payload)

    rows = [aggregate_samples(samples) for samples in grouped.values()]
    return sorted(rows, key=lambda row: (row["student_name"], row["config"]))


def index_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["student_name"], row["config"]): row for row in rows}


def gap_points(our_fraction: float | None, paper_percent: float | None) -> float | None:
    if our_fraction is None or paper_percent is None:
        return None
    return round(our_fraction * 100.0 - paper_percent, 2)


def format_percent_fraction(value: float | None) -> str:
    if value is None:
        return "NR"
    return f"{value * 100.0:.2f}"


def format_percent_value(value: float | None) -> str:
    if value is None:
        return "NR"
    return f"{value:.2f}"


def format_number(value: Any) -> str:
    if value is None or value == "":
        return "NR"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


F1_AVERAGING_TYPES = {"not_reported", "unspecified", "macro", "weighted", "micro"}


def comparable_f1_for_paper(our_row: dict[str, Any], paper_f1_type: str) -> tuple[float | None, str]:
    """Return a same-basis F1 only when the paper states its F1 averaging type."""

    if paper_f1_type == "macro":
        return our_row.get("macro_f1_mean"), "macro-to-macro"
    if paper_f1_type == "weighted":
        return our_row.get("weighted_f1_mean"), "weighted-to-weighted"
    if paper_f1_type == "micro":
        return our_row.get("micro_f1_mean"), "micro-to-micro"
    if paper_f1_type == "not_reported":
        return None, "not comparable: paper F1 not reported"
    if paper_f1_type == "unspecified":
        return None, "not comparable: paper F1 averaging not specified"
    raise ValueError(f"Unsupported paper_f1_type: {paper_f1_type}")


def best_f1_reference_for_paper(
    our_row: dict[str, Any],
    paper_f1_percent: float | None,
    paper_f1_type: str,
) -> tuple[float | None, str | None, float | None, str]:
    """Return an explicitly non-apples-to-apples best-F1 reference for unspecified paper F1 rows."""

    if paper_f1_percent is None:
        return None, None, None, "not available: paper F1 not reported"
    if paper_f1_type != "unspecified":
        return None, None, None, "not needed: paper F1 averaging is specified"

    candidates = [
        ("macro", our_row.get("macro_f1_mean")),
        ("weighted", our_row.get("weighted_f1_mean")),
        ("micro", our_row.get("micro_f1_mean")),
    ]
    available = [(metric_type, value) for metric_type, value in candidates if value is not None]
    if not available:
        return None, None, None, "not available: our F1 values are missing"

    # Deterministic tie-breaking keeps output stable without changing the metric meaning.
    tie_order = {"macro": 0, "weighted": 1, "micro": 2}
    best_type, best_value = max(available, key=lambda item: (float(item[1]), tie_order[item[0]]))
    return (
        float(best_value),
        best_type,
        gap_points(float(best_value), paper_f1_percent),
        "non-apples-to-apples reference: paper F1 averaging not specified; compared against our best F1",
    )


def build_literature_comparison_rows(our_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_key = index_rows(our_rows)
    comparison_rows = []

    for paper_row in LITERATURE_ROWS:
        paper_f1_type = paper_row["paper_f1_type"]
        if paper_f1_type not in F1_AVERAGING_TYPES:
            raise ValueError(f"Unsupported paper_f1_type for {paper_row['paper_key']}: {paper_f1_type}")

        selection = OUR_MODEL_SELECTIONS[paper_row["our_key"]]
        our_row = rows_by_key.get((selection["student_name"], selection["config"]))
        if our_row is None:
            raise KeyError(f"Missing selected result row for {paper_row['our_key']}: {selection}")

        comparable_f1, f1_basis = comparable_f1_for_paper(our_row, paper_f1_type)
        best_f1, best_f1_type, best_f1_gap, best_f1_basis = best_f1_reference_for_paper(
            our_row,
            paper_row["paper_f1"],
            paper_f1_type,
        )

        comparison_rows.append(
            {
                "paper_key": paper_row["paper_key"],
                "paper_work": paper_row["paper_work"],
                "year": paper_row["year"],
                "dataset_task": paper_row["dataset_task"],
                "paper_model": paper_row["paper_model"],
                "paper_accuracy": paper_row["paper_accuracy"],
                "paper_precision": paper_row["paper_precision"],
                "paper_recall": paper_row["paper_recall"],
                "paper_f1": paper_row["paper_f1"],
                "paper_f1_type": paper_f1_type,
                "paper_footprint": paper_row["paper_footprint"],
                "our_model": selection["label"],
                "our_accuracy": our_row.get("accuracy_mean"),
                "our_macro_precision": our_row.get("macro_precision_mean"),
                "our_macro_recall": our_row.get("macro_recall_mean"),
                "our_macro_f1": our_row.get("macro_f1_mean"),
                "our_weighted_f1": our_row.get("weighted_f1_mean"),
                "our_micro_f1": our_row.get("micro_f1_mean"),
                "our_comparable_f1": comparable_f1,
                "our_best_f1_reference": best_f1,
                "our_best_f1_reference_type": best_f1_type,
                "our_size_kb": our_row.get("size_kb"),
                "our_params": our_row.get("params"),
                "our_flops_per_sample": our_row.get("flops_per_sample"),
                "accuracy_gap_points": gap_points(our_row.get("accuracy_mean"), paper_row["paper_accuracy"]),
                "comparable_f1_gap_points": gap_points(comparable_f1, paper_row["paper_f1"]),
                "best_f1_reference_gap_points": best_f1_gap,
                "f1_comparison_basis": f1_basis,
                "best_f1_reference_basis": best_f1_basis,
                "interpretation": paper_row["interpretation"],
            }
        )

    return comparison_rows


def read_artifact_text(repo_root: Path, artifact_dir: str, filename: str, git_ref: str | None) -> str:
    local_path = repo_root / artifact_dir / filename
    if local_path.exists():
        return local_path.read_text(encoding="utf-8")
    if not git_ref:
        raise FileNotFoundError(f"Missing local artifact and no git ref provided: {local_path}")
    artifact_path = f"{artifact_dir.rstrip('/')}/{filename}"
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "show", f"{git_ref}:{artifact_path}"],
        text=True,
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def metric_summary_csv_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output_rows = []
    for row in rows:
        output_rows.append(
            {
                "student_name": row["student_name"],
                "config": row["config"],
                "n_seeds": row["n_seeds"],
                "accuracy_percent": format_percent_fraction(row.get("accuracy_mean")),
                "accuracy_std_percent": format_percent_fraction(row.get("accuracy_std")),
                "macro_f1_percent": format_percent_fraction(row.get("macro_f1_mean")),
                "macro_f1_std_percent": format_percent_fraction(row.get("macro_f1_std")),
                "weighted_f1_percent": format_percent_fraction(row.get("weighted_f1_mean")),
                "weighted_f1_std_percent": format_percent_fraction(row.get("weighted_f1_std")),
                "micro_f1_percent": format_percent_fraction(row.get("micro_f1_mean")),
                "micro_f1_std_percent": format_percent_fraction(row.get("micro_f1_std")),
                "macro_precision_percent": format_percent_fraction(row.get("macro_precision_mean")),
                "macro_recall_percent": format_percent_fraction(row.get("macro_recall_mean")),
                "params": row.get("params"),
                "size_kb": row.get("size_kb"),
                "flops_per_sample": row.get("flops_per_sample"),
                "support_total_per_seed": row.get("support_total_per_seed"),
            }
        )
    return output_rows


def literature_csv_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output_rows = []
    for row in rows:
        output_rows.append(
            {
                "paper_work": row["paper_work"],
                "year": row["year"],
                "dataset_task": row["dataset_task"],
                "paper_model": row["paper_model"],
                "paper_accuracy_percent": format_percent_value(row["paper_accuracy"]),
                "paper_precision_percent": format_percent_value(row["paper_precision"]),
                "paper_recall_percent": format_percent_value(row["paper_recall"]),
                "paper_f1_percent": format_percent_value(row["paper_f1"]),
                "paper_f1_type": row["paper_f1_type"],
                "paper_footprint": row["paper_footprint"],
                "our_model": row["our_model"],
                "our_accuracy_percent": format_percent_fraction(row["our_accuracy"]),
                "our_macro_precision_percent": format_percent_fraction(row["our_macro_precision"]),
                "our_macro_recall_percent": format_percent_fraction(row["our_macro_recall"]),
                "our_macro_f1_percent": format_percent_fraction(row["our_macro_f1"]),
                "our_weighted_f1_percent": format_percent_fraction(row["our_weighted_f1"]),
                "our_micro_f1_percent": format_percent_fraction(row["our_micro_f1"]),
                "our_comparable_f1_percent": format_percent_fraction(row["our_comparable_f1"]),
                "our_best_f1_reference_percent": format_percent_fraction(row["our_best_f1_reference"]),
                "our_best_f1_reference_type": format_number(row["our_best_f1_reference_type"]),
                "our_size_kb": format_number(row["our_size_kb"]),
                "our_params": format_number(row["our_params"]),
                "our_flops_per_sample": format_number(row["our_flops_per_sample"]),
                "accuracy_gap_points": format_number(row["accuracy_gap_points"]),
                "comparable_f1_gap_points": format_number(row["comparable_f1_gap_points"]),
                "best_f1_reference_gap_points": format_number(row["best_f1_reference_gap_points"]),
                "f1_comparison_basis": row["f1_comparison_basis"],
                "best_f1_reference_basis": row["best_f1_reference_basis"],
                "interpretation": row["interpretation"],
            }
        )
    return output_rows


def markdown_table(rows: list[dict[str, str]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def write_markdown_report(path: Path, comparison_rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    csv_rows = literature_csv_rows(comparison_rows)
    display_columns = [
        ("paper_work", "Paper / Work"),
        ("year", "Year"),
        ("paper_model", "Paper Model"),
        ("paper_accuracy_percent", "Paper Acc. (%)"),
        ("paper_f1_percent", "Paper F1 (%)"),
        ("paper_f1_type", "Paper F1 Type"),
        ("paper_footprint", "Paper Footprint"),
        ("our_model", "Our Matched Model"),
        ("our_accuracy_percent", "Our Acc. (%)"),
        ("our_macro_f1_percent", "Our Macro-F1 (%)"),
        ("our_weighted_f1_percent", "Our Weighted-F1 (%)"),
        ("our_micro_f1_percent", "Our Micro-F1 (%)"),
        ("our_comparable_f1_percent", "Our Matched F1 (%)"),
        ("our_best_f1_reference_percent", "Our Best-F1 Ref. (%)"),
        ("our_best_f1_reference_type", "Best-F1 Type"),
        ("our_size_kb", "Our Size (KB)"),
        ("accuracy_gap_points", "Acc. Gap (pts)"),
        ("comparable_f1_gap_points", "Matched F1 Gap (pts)"),
        ("best_f1_reference_gap_points", "Best-F1 Ref. Gap (pts)"),
        ("f1_comparison_basis", "F1 Basis"),
        ("interpretation", "Interpretation"),
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Edge-IIoTset Literature Metric Comparison",
                "",
                "This report is generated from completed Edge-IIoTset confusion matrices. It does not retrain any model.",
                "",
                f"- Protocol: `{metadata.get('protocol', 'NR')}`",
                f"- Experiment scope: `{metadata.get('experiment_scope', 'NR')}`",
                f"- Seeds: `{metadata.get('seeds', 'NR')}`",
                f"- Classes: `{metadata.get('num_classes', 'NR')}`",
                f"- Input dimension: `{metadata.get('input_dim', 'NR')}`",
                "",
                "## Key Interpretation for Professor",
                "",
                "- F1 is compared apples-to-apples only when the paper states the averaging type: macro-to-macro, weighted-to-weighted, or micro-to-micro.",
                "- If a paper reports plain `F1-score` without stating macro/weighted/micro, the matched F1 gap is `NR`; the table separately shows an explicitly non-apples-to-apples best-F1 reference gap.",
                "- Use accuracy and deployment footprint for the broad literature comparison; use macro-F1 only for minority-class robustness and only against papers that explicitly report macro-F1.",
                "- No training rerun is needed for this metric recalculation because confusion matrices are already present in the completed results.",
                "",
                "## Literature Comparison",
                "",
                markdown_table(csv_rows, display_columns),
                "",
                "## Sources Used",
                "",
                "- Ferrag et al., Edge-IIoTset: https://doi.org/10.1109/ACCESS.2022.3165809",
                "- Diab et al., Hardware-Aware ML/DL: https://arxiv.org/abs/2512.02272",
                "- Abualhassan et al., IIoT-TinyDNN: https://doi.org/10.1109/CSCN67557.2025.11230733",
                "- Hasan et al., Autoencoder Feature Learning: https://doi.org/10.5220/0013203700003944",
                "- Abdi et al., CNN multiclass classification: https://doi.org/10.3390/fi17060230",
                "- Yagiz and Goktas, LENS-XAI: https://arxiv.org/abs/2501.00790",
                "- Salehiyan et al., Transformer-GAN-AE: https://doi.org/10.3390/fi17070279",
                "- WO-XGB feature-level ensemble: https://link.springer.com/article/10.1007/s43926-025-00185-7",
                "- CST-AFNet: https://doi.org/10.1016/j.array.2025.100501",
                "- Neuro-symbolic edge IDS: https://link.springer.com/article/10.1007/s44397-026-00047-z",
                "- Hybrid LLM/HGB IDS: https://www.mdpi.com/1424-8220/26/4/1231",
                "- Gao et al., lightweight TCN: https://doi.org/10.3390/electronics15050938",
                "- Rows for FI-SEL, SACNN-IDS, SA-DCNN, and FedDynST are included from the comparison table reported in the Hybrid LLM/HGB IDS paper above when primary metrics were not directly accessible in full text.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_action_plan(path: Path, comparison_rows: list[dict[str, Any]]) -> None:
    matched_f1_gaps = [
        row["comparable_f1_gap_points"]
        for row in comparison_rows
        if row.get("comparable_f1_gap_points") is not None
    ]
    best_reference_gaps = [
        row["best_f1_reference_gap_points"]
        for row in comparison_rows
        if row.get("best_f1_reference_gap_points") is not None
    ]
    payload = {
        "generated_from_existing_results": True,
        "training_required_for_this_analysis": False,
        "expected_runtime": "Under 1 minute on a normal laptop; no GPU needed.",
        "exact_macro_weighted_and_micro_f1_available": True,
        "f1_comparison_policy": (
            "Compute direct F1 gaps only when the paper states macro, weighted, or micro F1. "
            "For unspecified paper F1, report an explicitly non-apples-to-apples best-F1 reference gap."
        ),
        "f1_rows_with_comparable_gap": len(matched_f1_gaps),
        "f1_rows_with_best_reference_gap_only": len(best_reference_gaps),
        "comparable_f1_gap_points_range": [min(matched_f1_gaps), max(matched_f1_gaps)] if matched_f1_gaps else None,
        "best_f1_reference_gap_points_range": [min(best_reference_gaps), max(best_reference_gaps)] if best_reference_gaps else None,
        "recommended_next_step": (
            "Use accuracy and size for broad literature positioning. Use matched F1 gaps only where F1 averaging is explicit. "
            "When F1 averaging is unspecified, state that the best-F1 reference is optimistic and not apples-to-apples."
        ),
        "optional_training_only_if_professor_demands_improvement": [
            "Class-balanced KD or focal CE within Student A/B/C only.",
            "Teacher-family KD tuning within the existing Student A/B/C sizes.",
            "No Student D unless the compression-first goal is explicitly relaxed.",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Repository root containing the Edge-IIoT artifacts.")
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR, help="Directory containing edgeiiot_v23_results.json.")
    parser.add_argument("--git-ref", default="origin/main", help="Git ref used if artifacts are not present locally.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for generated CSV/Markdown outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = (repo_root / args.output_dir).resolve()

    result_text = read_artifact_text(repo_root, args.artifact_dir, RESULTS_JSON_NAME, args.git_ref)
    result_data = json.loads(result_text)
    exact_rows = collect_exact_metric_rows(result_data)
    comparison_rows = build_literature_comparison_rows(exact_rows)
    metadata = result_data.get("metadata", {})

    metric_summary_path = output_dir / "edgeiiot_exact_metric_summary.csv"
    comparison_csv_path = output_dir / "edgeiiot_literature_metric_comparison.csv"
    comparison_md_path = output_dir / "edgeiiot_literature_metric_comparison.md"
    action_plan_path = output_dir / "edgeiiot_metric_gap_action_plan.json"

    write_csv(
        metric_summary_path,
        metric_summary_csv_rows(exact_rows),
        [
            "student_name",
            "config",
            "n_seeds",
            "accuracy_percent",
            "accuracy_std_percent",
            "macro_f1_percent",
            "macro_f1_std_percent",
            "weighted_f1_percent",
            "weighted_f1_std_percent",
            "micro_f1_percent",
            "micro_f1_std_percent",
            "macro_precision_percent",
            "macro_recall_percent",
            "params",
            "size_kb",
            "flops_per_sample",
            "support_total_per_seed",
        ],
    )
    write_csv(
        comparison_csv_path,
        literature_csv_rows(comparison_rows),
        [
            "paper_work",
            "year",
            "dataset_task",
            "paper_model",
            "paper_accuracy_percent",
            "paper_precision_percent",
            "paper_recall_percent",
            "paper_f1_percent",
            "paper_f1_type",
            "paper_footprint",
            "our_model",
            "our_accuracy_percent",
            "our_macro_precision_percent",
            "our_macro_recall_percent",
            "our_macro_f1_percent",
            "our_weighted_f1_percent",
            "our_micro_f1_percent",
            "our_comparable_f1_percent",
            "our_best_f1_reference_percent",
            "our_best_f1_reference_type",
            "our_size_kb",
            "our_params",
            "our_flops_per_sample",
            "accuracy_gap_points",
            "comparable_f1_gap_points",
            "best_f1_reference_gap_points",
            "f1_comparison_basis",
            "best_f1_reference_basis",
            "interpretation",
        ],
    )
    write_markdown_report(comparison_md_path, comparison_rows, metadata)
    write_action_plan(action_plan_path, comparison_rows)

    print(
        json.dumps(
            {
                "status": "ok",
                "training_rerun_required": False,
                "output_dir": str(output_dir),
                "outputs": [
                    str(metric_summary_path),
                    str(comparison_csv_path),
                    str(comparison_md_path),
                    str(action_plan_path),
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
