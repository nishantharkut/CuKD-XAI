"""Build a sealed, class-balanced timing subset from a strict final FG-DS export.

This tool is additive. It reads an immutable WSN-DS CSV, the final ten-seed
split evidence, and one strict final model export. It never edits those inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROTOCOL_ID = "wsnds_feature_group_split_train_only_scaler_10seed_v2"
COHORT_PROTOCOL_ID = "wsnds_fgds_balanced_timing_cohort_1000_v1"
SELECTION_ALGORITHM = "sha256_seeded_canonical_f32_rank_v1"
SELECTION_SEED = 42
GROUPS_PER_CLASS = 200
EXPECTED_SEED = 42
EXPECTED_SEEDS = [42, 123, 456, 789, 1001, 2024, 3141, 5678, 8192, 9999]
EXPECTED_SPLIT_SIZES = {"train": 262_197, "validation": 56_163, "test": 56_301}
FEATURE_NAMES = [
    "Time",
    "Is_CH",
    "who CH",
    "Dist_To_CH",
    "ADV_S",
    "ADV_R",
    "JOIN_S",
    "JOIN_R",
    "SCH_S",
    "SCH_R",
    "Rank",
    "DATA_S",
    "DATA_R",
    "Data_Sent_To_BS",
    "dist_CH_To_BS",
    "send_code",
    "Expaned Energy",
]
CLASS_NAMES = ["Blackhole", "Flooding", "Grayhole", "Normal", "TDMA"]
REPLAY_NAME = "balanced_timing_replay_vectors.csv"
REFERENCE_NAME = "balanced_timing_reference_predictions.csv"
COHORT_NAME = "balanced_timing_cohort.json"
MANIFEST_NAME = "artifact_manifest.json"
GENERATOR_PATH = Path(__file__).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")


def safe_member(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Manifest path escapes its root: {relative!r}") from exc
    return candidate


def verify_manifest(
    root: Path,
    manifest_name: str,
    accepted_statuses: set[str],
) -> dict[str, Any]:
    manifest_path = root / manifest_name
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = read_json(manifest_path)
    if manifest.get("status") not in accepted_statuses:
        raise RuntimeError(
            f"Manifest {manifest_path} has unacceptable status {manifest.get('status')!r}"
        )
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError(f"Manifest has no file inventory: {manifest_path}")

    declared: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise RuntimeError(f"Malformed manifest item in {manifest_path}")
        relative = item.get("path")
        if not isinstance(relative, str) or not relative or relative in declared:
            raise RuntimeError(f"Invalid or duplicate manifest path: {relative!r}")
        declared.add(relative)
        member = safe_member(root, relative)
        if not member.is_file():
            raise FileNotFoundError(member)
        if member.stat().st_size != item.get("size_bytes"):
            raise RuntimeError(f"Manifest size mismatch: {relative}")
        if sha256_file(member) != item.get("sha256"):
            raise RuntimeError(f"Manifest SHA-256 mismatch: {relative}")

    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.resolve() != manifest_path.resolve()
    }
    if actual != declared:
        missing = sorted(actual - declared)
        stale = sorted(declared - actual)
        raise RuntimeError(
            f"Manifest inventory is not exact; unlisted={missing}, absent={stale}"
        )
    recorded_count = manifest.get("file_count_excluding_manifest")
    if recorded_count is not None and recorded_count != len(declared):
        raise RuntimeError("Manifest file count does not match its inventory")
    recorded_payload_hash = manifest.get("manifest_payload_sha256")
    if recorded_payload_hash is not None:
        payload = dict(manifest)
        payload.pop("manifest_payload_sha256")
        if recorded_payload_hash != canonical_json_sha256(payload):
            raise RuntimeError("Manifest payload SHA-256 is invalid")
    return manifest


def require_equal(label: str, values: Iterable[Any]) -> Any:
    observed = list(values)
    if not observed or any(value != observed[0] for value in observed[1:]):
        raise RuntimeError(f"Inconsistent {label}: {observed!r}")
    return observed[0]


def canonical_feature_bytes(row: np.ndarray) -> bytes:
    values = np.asarray(row, dtype=np.float32).copy()
    if values.shape != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
        raise RuntimeError("Feature row violates the finite 17-feature contract")
    values[values == 0.0] = 0.0
    return np.asarray(values, dtype="<f4").tobytes(order="C")


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    target = next(
        (
            name
            for name in ["Attack type", "Attack_Type", "attack_type", "Attack Type", "class"]
            if name in frame.columns
        ),
        None,
    )
    if target is None:
        raise RuntimeError("WSN-DS target column is missing")
    for candidate in ["id", "Id", "ID"]:
        if candidate in frame.columns:
            frame = frame.drop(columns=[candidate])
            break
    labels_text = frame[target].astype(str).str.strip()
    observed_classes = sorted(labels_text.unique().tolist())
    if observed_classes != CLASS_NAMES:
        raise RuntimeError(f"Unexpected WSN-DS classes: {observed_classes}")
    feature_frame = frame.drop(columns=[target])
    if feature_frame.columns.tolist() != FEATURE_NAMES:
        raise RuntimeError(
            f"Unexpected WSN-DS feature order: {feature_frame.columns.tolist()}"
        )
    features = feature_frame.to_numpy(dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError(f"Unexpected WSN-DS feature shape: {features.shape}")
    if not np.isfinite(features).all():
        raise RuntimeError("WSN-DS contains non-finite feature values")
    label_lookup = {name: index for index, name in enumerate(CLASS_NAMES)}
    labels = labels_text.map(label_lookup).to_numpy(dtype=np.int64)
    return features, labels, target


def load_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not fields:
        raise RuntimeError(f"CSV has no header: {path}")
    return fields, rows


def integer_column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    try:
        return np.asarray([int(row[name]) for row in rows], dtype=np.int64)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"CSV column {name!r} is missing or non-integral") from exc


def verify_execution_fingerprint(execution: dict[str, Any]) -> None:
    observed = execution.get("execution_fingerprint_sha256")
    payload = dict(execution)
    payload.pop("execution_fingerprint_sha256", None)
    if observed != canonical_json_sha256(payload):
        raise RuntimeError("Final execution-contract fingerprint is invalid")


def load_split(
    split_root: Path,
    dataset_rows: int,
    expected_split_sizes: dict[str, int],
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any], dict[str, Any]]:
    root_manifest = verify_manifest(split_root, MANIFEST_NAME, {"complete"})
    execution = read_json(split_root / "execution_contract.json")
    preprocessing = read_json(split_root / "preprocessing_contract.json")
    if root_manifest.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("Final split manifest has the wrong protocol")
    if execution.get("protocol_id") != PROTOCOL_ID or preprocessing.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("Final split contracts have the wrong protocol")
    if execution.get("seeds") != EXPECTED_SEEDS:
        raise RuntimeError("Final execution contract does not contain the frozen ten seeds")
    verify_execution_fingerprint(execution)

    indices_path = split_root / preprocessing.get("split_indices_file", "")
    scaler_path = split_root / preprocessing.get("scaler_parameters_file", "")
    if not indices_path.is_file() or not scaler_path.is_file():
        raise RuntimeError("Final split or scaler artifact is absent")
    if sha256_file(indices_path) != preprocessing.get("split_indices_file_sha256"):
        raise RuntimeError("Split-index file hash differs from preprocessing contract")
    if sha256_file(scaler_path) != preprocessing.get("scaler_parameters_file_sha256"):
        raise RuntimeError("Scaler file hash differs from preprocessing contract")

    with np.load(indices_path, allow_pickle=False) as archive:
        expected_keys = {"train_indices", "validation_indices", "test_indices"}
        if set(archive.files) != expected_keys:
            raise RuntimeError(f"Unexpected split-index members: {archive.files}")
        split = {name: np.asarray(archive[f"{name}_indices"]) for name in expected_split_sizes}
    for name, indices in split.items():
        if indices.dtype != np.int64 or indices.ndim != 1:
            raise RuntimeError(f"{name} indices are not a one-dimensional int64 array")
        if len(indices) != expected_split_sizes[name]:
            raise RuntimeError(f"{name} size differs from the frozen split")
        if np.any(indices < 0) or np.any(indices >= dataset_rows):
            raise RuntimeError(f"{name} indices are outside the dataset")
        if len(np.unique(indices)) != len(indices):
            raise RuntimeError(f"{name} indices contain duplicates")
    combined = np.concatenate([split["train"], split["validation"], split["test"]])
    if len(combined) != dataset_rows or not np.array_equal(
        np.sort(combined), np.arange(dataset_rows, dtype=np.int64)
    ):
        raise RuntimeError("Final split is not an exact partition of the dataset")

    logical_hash = sha256_arrays(split["train"], split["validation"], split["test"])
    require_equal(
        "split_indices_sha256",
        [logical_hash, execution.get("split_indices_sha256"), preprocessing.get("split_indices_sha256")],
    )
    if preprocessing.get("split_sizes") != expected_split_sizes:
        raise RuntimeError("Preprocessing contract has unexpected split sizes")

    with np.load(scaler_path, allow_pickle=False) as archive:
        if set(archive.files) != {"mean", "scale", "var", "n_samples_seen"}:
            raise RuntimeError(f"Unexpected scaler members: {archive.files}")
        mean = np.asarray(archive["mean"], dtype=np.float64)
        scale = np.asarray(archive["scale"], dtype=np.float64)
        variance = np.asarray(archive["var"], dtype=np.float64)
        n_samples_seen = np.asarray(archive["n_samples_seen"])
    if any(item.shape != (len(FEATURE_NAMES),) for item in [mean, scale, variance]):
        raise RuntimeError("Scaler parameters violate the 17-feature contract")
    if n_samples_seen.shape != (1,) or int(n_samples_seen[0]) != expected_split_sizes["train"]:
        raise RuntimeError("Scaler sample count differs from the final training partition")
    scaler_hash = sha256_arrays(mean, scale, variance)
    require_equal(
        "scaler_sha256",
        [scaler_hash, execution.get("scaler_sha256"), preprocessing.get("scaler_sha256")],
    )
    return split, execution, preprocessing, root_manifest


def identity_field(
    name: str,
    manifest: dict[str, Any],
    report: dict[str, Any],
) -> Any:
    provenance = report.get("provenance", {})
    identity_provenance = report.get("export_identity_payload", {}).get("provenance", {})
    values = [container[name] for container in [manifest, provenance, identity_provenance] if name in container]
    if not values:
        raise RuntimeError(f"Strict final export does not bind identity field {name!r}")
    return require_equal(f"export identity {name}", values)


def verify_export(
    export_dir: Path,
    dataset_sha256: str,
    execution: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[str], list[dict[str, str]], list[str], list[dict[str, str]]]:
    manifest = verify_manifest(export_dir, "strict_export_manifest.json", {"passed"})
    report = read_json(export_dir / "strict_export_report.json")
    if report.get("status") != "passed":
        raise RuntimeError("Strict final export report did not pass")
    if report.get("export_id") != manifest.get("export_id"):
        raise RuntimeError("Strict export ID differs between report and manifest")
    if manifest.get("export_identity_payload_sha256") != manifest.get("export_id"):
        raise RuntimeError("Strict manifest does not bind its export identity payload")
    identity_payload = report.get("export_identity_payload")
    if not isinstance(identity_payload, dict) or canonical_json_sha256(
        identity_payload
    ) != manifest.get("export_id"):
        raise RuntimeError("Strict export ID does not hash its recorded identity payload")

    expected_identity = {
        "protocol_id": PROTOCOL_ID,
        "seed": EXPECTED_SEED,
        "dataset_sha256": dataset_sha256,
        "split_indices_sha256": execution["split_indices_sha256"],
        "scaler_sha256": execution["scaler_sha256"],
    }
    for name, expected in expected_identity.items():
        if identity_field(name, manifest, report) != expected:
            raise RuntimeError(f"Strict final export has the wrong {name}")
    student = identity_field("student", manifest, report)
    route = identity_field("route", manifest, report)
    if student not in {"student_A", "student_B"}:
        raise RuntimeError(f"Unknown final export student: {student!r}")
    if route not in {"scratch", "rf_kd"}:
        raise RuntimeError(f"Unknown final export route: {route!r}")
    for required in ["model_file_sha256", "trained_state_sha256"]:
        value = identity_field(required, manifest, report)
        if not isinstance(value, str) or len(value) != 64:
            raise RuntimeError(f"Strict final export has invalid {required}")

    gates = report.get("gates", {})
    for gate in ["saved_test_rows_and_labels_exact", "saved_fp32_predictions_exact"]:
        if gates.get(gate) is not True:
            raise RuntimeError(f"Strict final export gate failed: {gate}")
    replay_fields, replay_rows = load_csv_rows(export_dir / "hil_replay_vectors.csv")
    reference_fields, reference_rows = load_csv_rows(
        export_dir / "hil_reference_predictions.csv"
    )
    return manifest, report, replay_fields, replay_rows, reference_fields, reference_rows


@dataclass
class GroupRecord:
    canonical: bytes
    label_mask: int = 0
    partition_mask: int = 0
    test_representative: int | None = None


def build_groups(
    features: np.ndarray,
    labels: np.ndarray,
    split: dict[str, np.ndarray],
) -> dict[bytes, GroupRecord]:
    partition = np.empty(len(labels), dtype=np.uint8)
    partition[split["train"]] = 1
    partition[split["validation"]] = 2
    partition[split["test"]] = 4
    groups: dict[bytes, GroupRecord] = {}
    for source_row_index, (row, label) in enumerate(zip(features, labels)):
        canonical = canonical_feature_bytes(row)
        record = groups.get(canonical)
        if record is None:
            record = GroupRecord(canonical=canonical)
            groups[canonical] = record
        record.label_mask |= 1 << int(label)
        record.partition_mask |= int(partition[source_row_index])
        if partition[source_row_index] == 4 and (
            record.test_representative is None
            or source_row_index < record.test_representative
        ):
            record.test_representative = source_row_index
    return groups


def select_groups(
    groups: dict[bytes, GroupRecord],
    groups_per_class: int,
    selection_seed: int,
) -> list[dict[str, Any]]:
    if groups_per_class <= 0 or selection_seed < 0:
        raise ValueError("Selection count and seed must be non-negative")
    seed_bytes = int(selection_seed).to_bytes(8, byteorder="little", signed=False)
    candidates: dict[int, list[tuple[bytes, int, bytes]]] = {
        label: [] for label in range(len(CLASS_NAMES))
    }
    for canonical, record in groups.items():
        if record.test_representative is None or record.partition_mask != 4:
            continue
        if record.label_mask <= 0 or record.label_mask & (record.label_mask - 1):
            continue
        label = record.label_mask.bit_length() - 1
        rank = hashlib.sha256(seed_bytes + canonical).digest()
        candidates[label].append((rank, record.test_representative, canonical))

    selected_by_class: dict[int, list[tuple[bytes, int, bytes]]] = {}
    for label, rows in candidates.items():
        rows.sort(key=lambda item: (item[0], item[1], item[2]))
        if len(rows) < groups_per_class:
            raise RuntimeError(
                f"Class {label} has only {len(rows)} eligible label-pure test groups; "
                f"{groups_per_class} are required"
            )
        selected_by_class[label] = rows[:groups_per_class]

    result: list[dict[str, Any]] = []
    for rank_within_class in range(groups_per_class):
        for label in range(len(CLASS_NAMES)):
            selection_rank, source_row_index, canonical = selected_by_class[label][
                rank_within_class
            ]
            result.append(
                {
                    "timing_row_id": len(result),
                    "source_row_index": int(source_row_index),
                    "true_label": label,
                    "class_name": CLASS_NAMES[label],
                    "class_rank": rank_within_class,
                    "selection_rank_sha256": selection_rank.hex(),
                    "feature_group_sha256": hashlib.sha256(canonical).hexdigest(),
                }
            )
    return result


def verify_full_export_rows(
    features: np.ndarray,
    labels: np.ndarray,
    split: dict[str, np.ndarray],
    report: dict[str, Any],
    replay_fields: list[str],
    replay_rows: list[dict[str, str]],
    reference_fields: list[str],
    reference_rows: list[dict[str, str]],
    export_dir: Path,
) -> None:
    required_replay = ["row_id", "source_row_index", *[f"f{i}" for i in range(17)]]
    required_reference = [
        "row_id",
        "source_row_index",
        "true_label",
        "fixed_pred",
        "fp32_pred",
        *[f"fixed_logit_{i}" for i in range(len(CLASS_NAMES))],
    ]
    if replay_fields != required_replay:
        raise RuntimeError("Full replay CSV violates the strict column contract")
    if reference_fields != required_reference:
        raise RuntimeError("Full reference CSV violates the strict column contract")
    test_indices = split["test"]
    if len(replay_rows) != len(test_indices) or len(reference_rows) != len(test_indices):
        raise RuntimeError("Strict export does not contain the complete final test partition")
    if report.get("gates", {}).get("full_test_rows") != len(test_indices):
        raise RuntimeError("Strict export full-test gate has the wrong row count")
    dense_ids = np.arange(len(test_indices), dtype=np.int64)
    replay_ids = integer_column(replay_rows, "row_id")
    reference_ids = integer_column(reference_rows, "row_id")
    if not np.array_equal(replay_ids, dense_ids) or not np.array_equal(reference_ids, dense_ids):
        raise RuntimeError("Strict full-export row IDs are not the dense full-test sequence")
    replay_source = integer_column(replay_rows, "source_row_index")
    reference_source = integer_column(reference_rows, "source_row_index")
    if not np.array_equal(replay_source, test_indices) or not np.array_equal(
        reference_source, test_indices
    ):
        raise RuntimeError("Strict full-export source rows differ from final test indices")
    if not np.array_equal(integer_column(reference_rows, "true_label"), labels[test_indices]):
        raise RuntimeError("Strict full-export labels differ from the immutable dataset")
    for name in ["fixed_pred", "fp32_pred"]:
        values = integer_column(reference_rows, name)
        if np.any(values < 0) or np.any(values >= len(CLASS_NAMES)):
            raise RuntimeError(f"Strict reference {name} is outside the class range")

    metadata = read_json(export_dir / "preprocess_int_metadata.json")
    if metadata.get("input_dim") != len(FEATURE_NAMES):
        raise RuntimeError("Integer preprocessing metadata has the wrong input dimension")
    raw_q_frac = metadata.get("raw_q_frac")
    if not isinstance(raw_q_frac, int) or raw_q_frac < 0 or raw_q_frac > 30:
        raise RuntimeError("Integer preprocessing metadata has an invalid raw_q_frac")
    expected_raw = np.rint(
        np.asarray(features[test_indices], dtype=np.float64) * float(1 << raw_q_frac)
    )
    int32 = np.iinfo(np.int32)
    if np.any(expected_raw < int32.min) or np.any(expected_raw > int32.max):
        raise RuntimeError("Immutable dataset values saturate the raw fixed-point contract")
    observed_raw = np.column_stack(
        [integer_column(replay_rows, f"f{index}") for index in range(len(FEATURE_NAMES))]
    )
    if not np.array_equal(observed_raw, expected_raw.astype(np.int64)):
        raise RuntimeError("Strict replay vectors do not encode the immutable dataset rows")


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def inventory(root: Path, excluded: set[str]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    ]


def build_timing_cohort(
    export_dir: Path,
    dataset_csv: Path,
    split_root: Path,
    output_dir: Path,
    *,
    selection_seed: int = SELECTION_SEED,
    groups_per_class: int = GROUPS_PER_CLASS,
    expected_split_sizes: dict[str, int] | None = None,
) -> Path:
    expected_sizes = expected_split_sizes or EXPECTED_SPLIT_SIZES
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    for path in [export_dir, split_root]:
        if not path.is_dir():
            raise NotADirectoryError(path)
    if not dataset_csv.is_file():
        raise FileNotFoundError(dataset_csv)

    features, labels, target_column = load_dataset(dataset_csv)
    dataset_hash = sha256_file(dataset_csv)
    split, execution, preprocessing, split_manifest = load_split(
        split_root, len(labels), expected_sizes
    )
    require_equal(
        "dataset_sha256",
        [dataset_hash, execution.get("dataset_sha256"), preprocessing.get("dataset_sha256")],
    )
    if preprocessing.get("dataset_shape") != list(features.shape):
        raise RuntimeError("Preprocessing contract has the wrong dataset shape")
    if preprocessing.get("feature_names") != FEATURE_NAMES:
        raise RuntimeError("Preprocessing contract has the wrong feature order")
    if preprocessing.get("class_names") != CLASS_NAMES:
        raise RuntimeError("Preprocessing contract has the wrong class order")

    (
        export_manifest,
        export_report,
        replay_fields,
        replay_rows,
        reference_fields,
        reference_rows,
    ) = verify_export(export_dir, dataset_hash, execution)
    verify_full_export_rows(
        features,
        labels,
        split,
        export_report,
        replay_fields,
        replay_rows,
        reference_fields,
        reference_rows,
        export_dir,
    )

    groups = build_groups(features, labels, split)
    cross_partition_groups = sum(
        1
        for record in groups.values()
        if record.partition_mask & (record.partition_mask - 1)
    )
    if cross_partition_groups != 0:
        raise RuntimeError(
            f"Final split contains {cross_partition_groups} cross-partition feature groups"
        )
    selected = select_groups(groups, groups_per_class, selection_seed)
    expected_rows = groups_per_class * len(CLASS_NAMES)
    if len(selected) != expected_rows:
        raise RuntimeError("Balanced selection produced the wrong number of rows")
    group_hashes = [row["feature_group_sha256"] for row in selected]
    if len(set(group_hashes)) != expected_rows:
        raise RuntimeError("Balanced selection contains duplicate feature groups")

    full_id_by_source = {
        int(row["source_row_index"]): int(row["row_id"]) for row in replay_rows
    }
    replay_by_id = {int(row["row_id"]): row for row in replay_rows}
    reference_by_id = {int(row["row_id"]): row for row in reference_rows}
    output_replay_rows: list[dict[str, Any]] = []
    output_reference_rows: list[dict[str, Any]] = []
    for selection in selected:
        source_row_index = selection["source_row_index"]
        original_id = full_id_by_source[source_row_index]
        timing_id = selection["timing_row_id"]
        selection["original_full_test_row_id"] = original_id
        replay = replay_by_id[original_id]
        reference = reference_by_id[original_id]
        common = {
            "row_id": timing_id,
            "timing_row_id": timing_id,
            "original_full_test_row_id": original_id,
            "source_row_index": source_row_index,
        }
        output_replay_rows.append(
            {**common, **{f"f{index}": replay[f"f{index}"] for index in range(17)}}
        )
        output_reference_rows.append(
            {
                **common,
                **{
                    name: reference[name]
                    for name in reference_fields
                    if name not in {"row_id", "source_row_index"}
                },
            }
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        replay_output = temporary_root / REPLAY_NAME
        reference_output = temporary_root / REFERENCE_NAME
        write_csv(
            replay_output,
            [
                "row_id",
                "timing_row_id",
                "original_full_test_row_id",
                "source_row_index",
                *[f"f{index}" for index in range(17)],
            ],
            output_replay_rows,
        )
        reference_extra = [
            name
            for name in reference_fields
            if name not in {"row_id", "source_row_index"}
        ]
        write_csv(
            reference_output,
            [
                "row_id",
                "timing_row_id",
                "original_full_test_row_id",
                "source_row_index",
                *reference_extra,
            ],
            output_reference_rows,
        )

        class_balance = {
            str(label): {
                "class_name": CLASS_NAMES[label],
                "rows": sum(row["true_label"] == label for row in selected),
                "unique_feature_groups": sum(
                    row["true_label"] == label for row in selected
                ),
            }
            for label in range(len(CLASS_NAMES))
        }
        cohort = {
            "status": "passed",
            "protocol_id": COHORT_PROTOCOL_ID,
            "source_protocol_id": PROTOCOL_ID,
            "generator_sha256": sha256_file(GENERATOR_PATH),
            "selection": {
                "seed": selection_seed,
                "algorithm": SELECTION_ALGORITHM,
                "groups_per_class": groups_per_class,
                "representative": "minimum zero-based immutable dataset source_row_index",
                "interleave_order": "class indices 0,1,2,3,4 repeated by within-class rank",
                "group_definition": (
                    "bit-exact little-endian float32 raw 17-feature row with signed zero canonicalized"
                ),
                "eligibility": (
                    "one label across the complete immutable dataset and membership only in the final test partition"
                ),
            },
            "row_identity": {
                "row_id": "protocol alias equal to timing_row_id",
                "timing_row_id": "dense zero-based ID in this 1,000-row timing artifact",
                "original_full_test_row_id": (
                    "dense row_id in the strict full-test export before subsetting"
                ),
                "source_row_index": (
                    "zero-based row index in the immutable WSN-DS CSV after header parsing"
                ),
            },
            "sources": {
                "dataset": {
                    "path_recorded": str(dataset_csv.resolve()),
                    "sha256": dataset_hash,
                    "shape": list(features.shape),
                    "target_column": target_column,
                },
                "split": {
                    "path_recorded": str(split_root.resolve()),
                    "manifest_sha256": sha256_file(split_root / MANIFEST_NAME),
                    "manifest_status": split_manifest["status"],
                    "split_indices_file_sha256": preprocessing[
                        "split_indices_file_sha256"
                    ],
                    "split_indices_sha256": execution["split_indices_sha256"],
                    "scaler_parameters_file_sha256": preprocessing[
                        "scaler_parameters_file_sha256"
                    ],
                    "scaler_sha256": execution["scaler_sha256"],
                    "split_sizes": expected_sizes,
                },
                "strict_export": {
                    "path_recorded": str(export_dir.resolve()),
                    "manifest_sha256": sha256_file(
                        export_dir / "strict_export_manifest.json"
                    ),
                    "report_sha256": sha256_file(export_dir / "strict_export_report.json"),
                    "export_id": export_manifest["export_id"],
                    "student": identity_field("student", export_manifest, export_report),
                    "route": identity_field("route", export_manifest, export_report),
                    "seed": identity_field("seed", export_manifest, export_report),
                    "model_file_sha256": identity_field(
                        "model_file_sha256", export_manifest, export_report
                    ),
                    "trained_state_sha256": identity_field(
                        "trained_state_sha256", export_manifest, export_report
                    ),
                    "full_replay_sha256": sha256_file(
                        export_dir / "hil_replay_vectors.csv"
                    ),
                    "full_reference_sha256": sha256_file(
                        export_dir / "hil_reference_predictions.csv"
                    ),
                },
            },
            "audit": {
                "rows": expected_rows,
                "class_balance": class_balance,
                "unique_feature_groups": len(set(group_hashes)),
                "duplicate_feature_groups": 0,
                "cross_partition_feature_groups_in_source_split": 0,
                "globally_conflicting_groups_excluded": sum(
                    1
                    for record in groups.values()
                    if record.label_mask & (record.label_mask - 1)
                ),
            },
            "outputs": {
                REPLAY_NAME: {
                    "rows": expected_rows,
                    "sha256": sha256_file(replay_output),
                },
                REFERENCE_NAME: {
                    "rows": expected_rows,
                    "sha256": sha256_file(reference_output),
                },
            },
            "rows": selected,
        }
        write_json(temporary_root / COHORT_NAME, cohort)
        files = inventory(temporary_root, {MANIFEST_NAME})
        manifest_payload = {
            "protocol_id": COHORT_PROTOCOL_ID,
            "source_protocol_id": PROTOCOL_ID,
            "status": "complete",
            "file_count_excluding_manifest": len(files),
            "files": files,
        }
        manifest_payload["manifest_payload_sha256"] = canonical_json_sha256(
            manifest_payload
        )
        write_json(temporary_root / MANIFEST_NAME, manifest_payload)

        verify_manifest(temporary_root, MANIFEST_NAME, {"complete"})
        written_cohort = read_json(temporary_root / COHORT_NAME)
        if written_cohort != cohort:
            raise RuntimeError("Written cohort JSON differs from its in-memory contract")
        os.replace(temporary_root, output_dir)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    return output_dir / MANIFEST_NAME


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--dataset-csv", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-seed", type=int, default=SELECTION_SEED)
    args = parser.parse_args()
    manifest = build_timing_cohort(
        export_dir=args.export_dir.resolve(),
        dataset_csv=args.dataset_csv.resolve(),
        split_root=args.split_root.resolve(),
        output_dir=args.output_dir.resolve(),
        selection_seed=args.selection_seed,
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
