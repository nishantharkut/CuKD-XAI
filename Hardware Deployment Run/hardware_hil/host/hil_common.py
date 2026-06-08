"""Common host-side helpers for CuKD-XAI hardware replay.

The protocol is intentionally simple and line-oriented so logs remain easy to
audit after a long MCU replay run.
"""

from __future__ import annotations

from collections import Counter
from math import ceil
from statistics import mean, median
from typing import Iterable


FEATURE_COUNT = 17
OUTPUT_COUNT = 5
REQUEST_PREFIX = "CUKD1"
RESPONSE_PREFIX = "CUKD1R"
STATUS_CODES = {
    "OK",
    "BAD_START",
    "BAD_LENGTH",
    "BAD_CHECKSUM",
    "BAD_FEATURE_RANGE",
    "INTERNAL_ERROR",
}


def crc16_ccitt(data: bytes, initial: int = 0xFFFF) -> int:
    """Return CRC-16-CCITT-FALSE for *data*."""
    crc = int(initial) & 0xFFFF
    for byte in data:
        crc ^= int(byte) << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc & 0xFFFF


def encode_request_line(row_id: int, features: Iterable[int]) -> str:
    """Encode one request row as CUKD1,row_id,17 features,crc."""
    values = [int(v) for v in features]
    if len(values) != FEATURE_COUNT:
        raise ValueError(f"expected {FEATURE_COUNT} features, got {len(values)}")
    if int(row_id) < 0:
        raise ValueError("row_id must be non-negative")

    body = ",".join([REQUEST_PREFIX, str(int(row_id)), *[str(v) for v in values]])
    crc = crc16_ccitt(body.encode("ascii"))
    return f"{body},{crc:04X}\n"


def _verify_crc(parts: list[str]) -> None:
    if len(parts) < 2:
        raise ValueError("packet is too short")
    body = ",".join(parts[:-1])
    try:
        supplied = int(parts[-1], 16)
    except ValueError as exc:
        raise ValueError("CRC is not hexadecimal") from exc
    expected = crc16_ccitt(body.encode("ascii"))
    if supplied != expected:
        raise ValueError(f"CRC mismatch: expected {expected:04X}, got {supplied:04X}")


def decode_response_line(line: str) -> dict[str, object]:
    """Decode and verify one MCU response line."""
    stripped = line.strip()
    parts = stripped.split(",")
    _verify_crc(parts)

    if parts[0] != RESPONSE_PREFIX:
        raise ValueError(f"bad response prefix: {parts[0]}")
    if len(parts) != 13:
        raise ValueError(f"expected 13 response fields including CRC, got {len(parts)}")

    status = parts[2]
    if status not in STATUS_CODES:
        raise ValueError(f"unknown status: {status}")

    logits = [int(v) for v in parts[4:9]]
    return {
        "row_id": int(parts[1]),
        "status": status,
        "predicted_class": int(parts[3]),
        "logits": logits,
        "preprocess_us": int(parts[9]),
        "inference_us": int(parts[10]),
        "total_us": int(parts[11]),
    }


def verify_response_sequence(
    expected_row_ids: Iterable[int],
    responses: Iterable[dict[str, object]],
) -> dict[str, object]:
    """Summarize missing, duplicate, and status counts for a replay."""
    expected = [int(v) for v in expected_row_ids]
    response_list = list(responses)
    seen = [int(r["row_id"]) for r in response_list]

    counts = Counter(seen)
    duplicates = sorted(row_id for row_id, count in counts.items() if count > 1)
    missing = sorted(set(expected) - set(seen))
    unexpected = sorted(set(seen) - set(expected))
    status_counts = Counter(str(r.get("status", "UNKNOWN")) for r in response_list)

    return {
        "expected": len(expected),
        "completed": len(response_list),
        "missing": missing,
        "duplicates": duplicates,
        "unexpected": unexpected,
        "status_counts": dict(sorted(status_counts.items())),
    }


def compute_classification_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    labels: Iterable[int],
) -> dict[str, object]:
    """Compute accuracy, per-class precision/recall/F1, macro-F1, weighted-F1."""
    true_values = [int(v) for v in y_true]
    pred_values = [int(v) for v in y_pred]
    label_values = [int(v) for v in labels]
    if len(true_values) != len(pred_values):
        raise ValueError("y_true and y_pred must have the same length")

    index = {label: i for i, label in enumerate(label_values)}
    matrix = [[0 for _ in label_values] for _ in label_values]
    for truth, pred in zip(true_values, pred_values):
        if truth in index and pred in index:
            matrix[index[truth]][index[pred]] += 1

    total = len(true_values)
    correct = sum(1 for truth, pred in zip(true_values, pred_values) if truth == pred)
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values = []
    weighted_sum = 0.0
    support_total = 0

    for label in label_values:
        i = index[label]
        tp = matrix[i][i]
        fp = sum(matrix[row][i] for row in range(len(label_values)) if row != i)
        fn = sum(matrix[i][col] for col in range(len(label_values)) if col != i)
        support = sum(matrix[i])
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[str(label)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1_values.append(f1)
        weighted_sum += f1 * support
        support_total += support

    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "weighted_f1": weighted_sum / support_total if support_total else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def _nearest_rank(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    idx = max(0, min(len(values) - 1, ceil((percentile / 100.0) * len(values)) - 1))
    return values[idx]


def summarize_latency(values: Iterable[int]) -> dict[str, float | int]:
    """Return min/mean/median/p95/p99/max/std-like latency summary."""
    vals = sorted(int(v) for v in values)
    if not vals:
        return {"count": 0, "min": 0, "mean": 0, "median": 0, "p95": 0, "p99": 0, "max": 0, "std": 0}

    avg = mean(vals)
    variance = sum((v - avg) ** 2 for v in vals) / len(vals)
    return {
        "count": len(vals),
        "min": vals[0],
        "mean": avg,
        "median": median(vals),
        "p95": _nearest_rank(vals, 95),
        "p99": _nearest_rank(vals, 99),
        "max": vals[-1],
        "std": variance ** 0.5,
    }

