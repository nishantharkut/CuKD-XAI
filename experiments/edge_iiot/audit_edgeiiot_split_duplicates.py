"""Audit Edge-IIoT exact feature duplicates across the archived split.

This is an audit-only script. It mirrors the Edge-IIoT v2.3 dataset adapter up
to model-input construction, then reports whether exact feature groups cross
the train/validation/test split. It does not train models or modify existing
results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


TARGET_COL = "Attack_type"
ML_FILENAME = "ML-EdgeIIoT-dataset.csv"
DNN_FILENAME = "DNN-EdgeIIoT-dataset.csv"

BASE_LEAKAGE_COLUMNS = [
    "frame.time",
    "ip.src_host",
    "ip.dst_host",
    "arp.src.proto_ipv4",
    "arp.dst.proto_ipv4",
    "http.file_data",
    "http.request.full_uri",
    "icmp.transmit_timestamp",
    "http.request.uri.query",
    "tcp.options",
    "tcp.payload",
    "mqtt.msg",
]
STRICT_EXTRA_LEAKAGE_COLUMNS = [
    "tcp.srcport",
    "tcp.dstport",
    "udp.port",
]
AUXILIARY_TARGET_COLUMNS = [
    "Attack_label",
    "attack_label",
    "Attack Label",
    "Label",
    "label",
    "class",
    "Class",
    "attack_type",
    "Attack Type",
]

MAX_CATEGORICAL_CARDINALITY = 64
RARE_CATEGORY_MIN_COUNT = 10


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def leakage_columns(protocol: str) -> list[str]:
    if protocol == "strict":
        return BASE_LEAKAGE_COLUMNS + STRICT_EXTRA_LEAKAGE_COLUMNS
    if protocol == "literature_comparable":
        return list(BASE_LEAKAGE_COLUMNS)
    raise ValueError(f"unsupported protocol: {protocol}")


def normalize_categorical(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .replace({"": "__MISSING__", "nan": "__MISSING__", "None": "__MISSING__"})
    )


def numeric_column_report(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
            continue
        coerced = pd.to_numeric(series, errors="coerce")
        if float(coerced.notna().mean()) >= 0.98:
            df[col] = coerced
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols


def fit_category_policy(X_train_raw: pd.DataFrame, categorical_cols: list[str]) -> dict[str, set[str]]:
    policy: dict[str, set[str]] = {}
    for col in categorical_cols:
        counts = normalize_categorical(X_train_raw[col]).value_counts(dropna=False)
        kept = counts[counts >= RARE_CATEGORY_MIN_COUNT].index.tolist()
        policy[col] = set(kept[:MAX_CATEGORICAL_CARDINALITY])
    return policy


def edge_encode_split(
    X_raw: pd.DataFrame,
    categorical_cols: list[str],
    category_policy: dict[str, set[str]],
) -> pd.DataFrame:
    if not categorical_cols:
        return pd.DataFrame(index=X_raw.index)
    X_cat = pd.DataFrame(index=X_raw.index)
    for col in categorical_cols:
        values = normalize_categorical(X_raw[col])
        kept = category_policy.get(col, set())
        X_cat[col] = values.where(values.isin(kept), "__OTHER__")
    return pd.get_dummies(X_cat, columns=categorical_cols, dtype=np.float32)


def row_hashes(df: pd.DataFrame) -> pd.Series:
    normalized = df.copy()
    for col in normalized.columns:
        if pd.api.types.is_float_dtype(normalized[col]):
            normalized[col] = normalized[col].astype(np.float32)
    return pd.util.hash_pandas_object(normalized, index=False, categorize=True).astype("uint64")


def collision_count(df: pd.DataFrame, hashes: pd.Series) -> int:
    collisions = 0
    duplicated_hashes = hashes[hashes.duplicated(keep=False)].unique()
    for value in duplicated_hashes:
        group = df.loc[hashes == value]
        if group.drop_duplicates().shape[0] > 1:
            collisions += 1
    return collisions


def audit_hashes(hashes: pd.Series, labels: np.ndarray, partitions: pd.Series) -> dict[str, Any]:
    frame = pd.DataFrame(
        {
            "hash": hashes.to_numpy(dtype=np.uint64),
            "label": labels.astype(np.int64),
            "partition": partitions.to_numpy(),
        }
    )
    group_sizes = frame.groupby("hash", sort=False).size()
    non_unique = group_sizes[group_sizes > 1]
    duplicate_rows_after_first = int((non_unique - 1).sum())
    rows_in_non_unique = int(non_unique.sum())

    part_sets = frame.groupby("hash", sort=False)["partition"].agg(lambda s: tuple(sorted(set(s))))
    crossing = part_sets[part_sets.map(len) > 1]

    label_sets = frame.groupby("hash", sort=False)["label"].agg(lambda s: tuple(sorted(set(s))))
    mixed_label = label_sets[label_sets.map(len) > 1]

    overlaps = {}
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        left_hash = set(frame.loc[frame["partition"] == left, "hash"])
        right_hash = set(frame.loc[frame["partition"] == right, "hash"])
        overlaps[f"{left}_{right}_feature_overlap"] = int(len(left_hash & right_hash))

    crossing_partition_counts = (
        frame[frame["hash"].isin(crossing.index)]
        .groupby("partition")
        .size()
        .reindex(["train", "validation", "test"], fill_value=0)
        .astype(int)
        .to_dict()
    )

    return {
        "total_rows": int(len(frame)),
        "unique_feature_rows": int(group_sizes.shape[0]),
        "duplicate_rows_excluding_first": duplicate_rows_after_first,
        "rows_in_non_unique_feature_groups": rows_in_non_unique,
        "non_unique_feature_groups": int(non_unique.shape[0]),
        "cross_partition_feature_groups": int(crossing.shape[0]),
        "rows_in_cross_partition_feature_groups_by_partition": crossing_partition_counts,
        "mixed_label_feature_groups": int(mixed_label.shape[0]),
        "overlaps": overlaps,
    }


def prepare_and_audit(csv_path: Path, protocol: str) -> dict[str, Any]:
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()
    if TARGET_COL not in df.columns:
        raise ValueError(f"{TARGET_COL!r} not found in {csv_path}")

    raw_shape_before = [int(df.shape[0]), int(df.shape[1])]
    df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()
    invalid_target_mask = df[TARGET_COL].isin(["", "nan", "None"])
    invalid_target_rows = int(invalid_target_mask.sum())
    if invalid_target_rows:
        df = df.loc[~invalid_target_mask].copy()

    drop_candidates = list(dict.fromkeys(leakage_columns(protocol) + AUXILIARY_TARGET_COLUMNS))
    removed_leakage_columns = [col for col in drop_candidates if col in df.columns and col != TARGET_COL]
    if removed_leakage_columns:
        df = df.drop(columns=removed_leakage_columns)

    y_raw = df[TARGET_COL].astype(str).str.strip()
    X_df = df.drop(columns=[TARGET_COL]).copy()

    all_missing = [col for col in X_df.columns if X_df[col].isna().all()]
    if all_missing:
        X_df = X_df.drop(columns=all_missing)

    numeric_cols, categorical_cols = numeric_column_report(X_df)
    for col in numeric_cols:
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce")
    for col in categorical_cols:
        X_df[col] = normalize_categorical(X_df[col])

    constant_cols = [col for col in X_df.columns if X_df[col].nunique(dropna=False) <= 1]
    if constant_cols:
        X_df = X_df.drop(columns=constant_cols)
        numeric_cols = [col for col in numeric_cols if col not in constant_cols]
        categorical_cols = [col for col in categorical_cols if col not in constant_cols]

    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(y_raw).astype(np.int64)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_df, y_all, test_size=0.15, random_state=42, stratify=y_all
    )
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_raw, y_train, test_size=0.1765, random_state=42, stratify=y_train
    )

    split_raw = pd.concat(
        [X_train_raw, X_val_raw, X_test_raw],
        axis=0,
    )
    split_labels = np.concatenate([y_train, y_val, y_test])
    split_partitions = pd.Series(
        ["train"] * len(X_train_raw) + ["validation"] * len(X_val_raw) + ["test"] * len(X_test_raw)
    )
    raw_hashes = row_hashes(split_raw.reset_index(drop=True))
    raw_audit = audit_hashes(raw_hashes, split_labels, split_partitions)
    raw_audit["hash_collisions_with_unequal_rows"] = collision_count(
        split_raw.reset_index(drop=True), raw_hashes
    )

    if numeric_cols:
        numeric_medians = X_train_raw[numeric_cols].median(numeric_only=True).fillna(0.0)
        X_train_raw.loc[:, numeric_cols] = X_train_raw[numeric_cols].fillna(numeric_medians).astype(np.float32)
        X_val_raw.loc[:, numeric_cols] = X_val_raw[numeric_cols].fillna(numeric_medians).astype(np.float32)
        X_test_raw.loc[:, numeric_cols] = X_test_raw[numeric_cols].fillna(numeric_medians).astype(np.float32)
    else:
        numeric_medians = pd.Series(dtype=np.float32)

    category_policy = fit_category_policy(X_train_raw, categorical_cols)
    X_train_cat = edge_encode_split(X_train_raw, categorical_cols, category_policy)
    X_val_cat = edge_encode_split(X_val_raw, categorical_cols, category_policy)
    X_test_cat = edge_encode_split(X_test_raw, categorical_cols, category_policy)
    dummy_columns = X_train_cat.columns.tolist()
    X_val_cat = X_val_cat.reindex(columns=dummy_columns, fill_value=0.0)
    X_test_cat = X_test_cat.reindex(columns=dummy_columns, fill_value=0.0)

    X_train_num = X_train_raw[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=X_train_raw.index)
    X_val_num = X_val_raw[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=X_val_raw.index)
    X_test_num = X_test_raw[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=X_test_raw.index)

    X_train_encoded = pd.concat([X_train_num.reset_index(drop=True), X_train_cat.reset_index(drop=True)], axis=1)
    X_val_encoded = pd.concat([X_val_num.reset_index(drop=True), X_val_cat.reset_index(drop=True)], axis=1)
    X_test_encoded = pd.concat([X_test_num.reset_index(drop=True), X_test_cat.reset_index(drop=True)], axis=1)

    train_constant_encoded_cols = [
        col for col in X_train_encoded.columns if X_train_encoded[col].nunique(dropna=False) <= 1
    ]
    if train_constant_encoded_cols:
        X_train_encoded = X_train_encoded.drop(columns=train_constant_encoded_cols)
        X_val_encoded = X_val_encoded.drop(columns=train_constant_encoded_cols, errors="ignore")
        X_test_encoded = X_test_encoded.drop(columns=train_constant_encoded_cols, errors="ignore")

    encoded_all = pd.concat([X_train_encoded, X_val_encoded, X_test_encoded], axis=0).reset_index(drop=True)
    encoded_hashes = row_hashes(encoded_all)
    encoded_audit = audit_hashes(encoded_hashes, split_labels, split_partitions)
    encoded_audit["hash_collisions_with_unequal_rows"] = collision_count(encoded_all, encoded_hashes)

    return {
        "protocol_id": "edgeiiot_split_duplicate_audit_v1",
        "protocol": protocol,
        "csv_path": str(csv_path),
        "csv_sha256": sha256_file(csv_path),
        "raw_shape_before_cleaning": raw_shape_before,
        "raw_shape_after_target_cleaning_and_drops": [int(df.shape[0]), int(df.shape[1])],
        "invalid_target_rows_removed": invalid_target_rows,
        "removed_leakage_columns": removed_leakage_columns,
        "removed_all_missing_columns": all_missing,
        "removed_constant_columns": constant_cols,
        "removed_train_constant_encoded_columns": train_constant_encoded_cols,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "numeric_medians_sha256": hashlib.sha256(
            json.dumps(
                {str(k): float(v) for k, v in numeric_medians.to_dict().items()},
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
        "dummy_feature_count": int(len(dummy_columns)),
        "input_dim": int(X_train_encoded.shape[1]),
        "class_names": label_encoder.classes_.tolist(),
        "class_distribution_all": dict(
            zip(label_encoder.classes_.tolist(), np.bincount(y_all).astype(int).tolist())
        ),
        "split_rows": {
            "train": int(len(X_train_encoded)),
            "validation": int(len(X_val_encoded)),
            "test": int(len(X_test_encoded)),
        },
        "split_policy": "train_test_split test_size=0.15 random_state=42 stratify=y_all; then validation test_size=0.1765 random_state=42 stratify=y_train",
        "raw_post_drop_feature_audit": raw_audit,
        "encoded_model_input_audit": encoded_audit,
        "interpretation": (
            "Cross-partition feature overlap in encoded_model_input_audit is the most relevant leakage indicator "
            "for the Edge-IIoT model-input table used by the archived scripts. This audit does not retrain models."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-csv", type=Path, required=True)
    parser.add_argument("--literature-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "strict": prepare_and_audit(args.strict_csv, "strict"),
        "literature_comparable": prepare_and_audit(args.literature_csv, "literature_comparable"),
    }
    for name, payload in outputs.items():
        (args.output_dir / f"{name}_duplicate_audit.json").write_text(
            json.dumps(payload, indent=2, default=json_default),
            encoding="utf-8",
        )
    summary = {
        "protocol_id": "edgeiiot_split_duplicate_audit_summary_v1",
        "status": "complete",
        "outputs": {
            name: {
                "csv_sha256": payload["csv_sha256"],
                "input_dim": payload["input_dim"],
                "split_rows": payload["split_rows"],
                "encoded_model_input_audit": payload["encoded_model_input_audit"],
            }
            for name, payload in outputs.items()
        },
    }
    (args.output_dir / "edgeiiot_duplicate_audit_summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default),
        encoding="utf-8",
    )
    print(args.output_dir / "edgeiiot_duplicate_audit_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
