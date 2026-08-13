import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from deployment.hardware_hil.host import generate_fgds_balanced_timing_cohort as cohort


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="ascii")


def inventory(root: Path, excluded: set[str]) -> list[dict]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": cohort.sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    ]


def seal(root: Path, name: str, status: str, protocol: str, **extra) -> None:
    files = inventory(root, {name})
    write_json(
        root / name,
        {
            "status": status,
            "protocol_id": protocol,
            "file_count_excluding_manifest": len(files),
            "files": files,
            **extra,
        },
    )


class CanonicalGroupingTests(unittest.TestCase):
    def test_signed_zero_is_canonical_but_distinct_float32_rows_are_not(self):
        left = np.arange(17, dtype=np.float32)
        right = left.copy()
        left[0] = -0.0
        right[0] = 0.0
        self.assertEqual(
            cohort.canonical_feature_bytes(left), cohort.canonical_feature_bytes(right)
        )
        right[1] = np.nextafter(right[1], np.float32(2.0), dtype=np.float32)
        self.assertNotEqual(
            cohort.canonical_feature_bytes(left), cohort.canonical_feature_bytes(right)
        )

    def test_selection_excludes_mixed_labels_and_interleaves_classes(self):
        groups = {}
        for label in range(5):
            for offset in range(3):
                raw = np.full(17, label * 10 + offset + 1, dtype=np.float32)
                key = cohort.canonical_feature_bytes(raw)
                groups[key] = cohort.GroupRecord(
                    canonical=key,
                    label_mask=1 << label,
                    partition_mask=4,
                    test_representative=label * 100 + offset,
                )
        mixed_raw = np.full(17, 999, dtype=np.float32)
        mixed_key = cohort.canonical_feature_bytes(mixed_raw)
        groups[mixed_key] = cohort.GroupRecord(
            canonical=mixed_key,
            label_mask=(1 << 0) | (1 << 1),
            partition_mask=4,
            test_representative=999,
        )
        selected = cohort.select_groups(groups, groups_per_class=2, selection_seed=42)
        self.assertEqual([row["true_label"] for row in selected], [0, 1, 2, 3, 4] * 2)
        self.assertEqual([row["timing_row_id"] for row in selected], list(range(10)))
        self.assertNotIn(999, [row["source_row_index"] for row in selected])
        self.assertEqual(len({row["feature_group_sha256"] for row in selected}), 10)

    def test_sha256_rank_is_deterministic_and_seed_bound(self):
        groups = {}
        for label in range(5):
            for offset in range(5):
                raw = np.zeros(17, dtype=np.float32)
                raw[0] = label
                raw[1] = offset
                key = cohort.canonical_feature_bytes(raw)
                groups[key] = cohort.GroupRecord(key, 1 << label, 4, label * 10 + offset)
        first = cohort.select_groups(groups, 3, 42)
        second = cohort.select_groups(groups, 3, 42)
        changed = cohort.select_groups(groups, 3, 123)
        self.assertEqual(first, second)
        self.assertNotEqual(
            [row["source_row_index"] for row in first],
            [row["source_row_index"] for row in changed],
        )


class SyntheticEndToEndTests(unittest.TestCase):
    def build_fixture(self, root: Path) -> tuple[Path, Path, Path, dict[str, int]]:
        dataset = root / "WSN-DS.csv"
        split_root = root / "split"
        export_root = root / "export"
        split_root.mkdir()
        export_root.mkdir()

        rows = []
        labels = []
        for label, class_name in enumerate(cohort.CLASS_NAMES):
            for group_index in range(202):
                values = [float((label + 1) * 100_000 + group_index * 100 + feature) for feature in range(17)]
                rows.append(values)
                labels.append(class_name)
        feature_frame = pd.DataFrame(rows, columns=cohort.FEATURE_NAMES)
        feature_frame["Attack type"] = labels
        feature_frame.to_csv(dataset, index=False)

        train = np.asarray([label * 202 for label in range(5)], dtype=np.int64)
        validation = np.asarray(
            [label * 202 + 1 for label in range(5)], dtype=np.int64
        )
        held_out = set(np.concatenate([train, validation]).tolist())
        test = np.asarray(
            [index for index in range(len(rows)) if index not in held_out],
            dtype=np.int64,
        )
        sizes = {"train": 5, "validation": 5, "test": len(test)}
        np.savez_compressed(
            split_root / "split_indices.npz",
            train_indices=train,
            validation_indices=validation,
            test_indices=test,
        )
        mean = np.zeros(17, dtype=np.float64)
        scale = np.ones(17, dtype=np.float64)
        variance = np.ones(17, dtype=np.float64)
        np.savez_compressed(
            split_root / "scaler_parameters.npz",
            mean=mean,
            scale=scale,
            var=variance,
            n_samples_seen=np.asarray([len(train)], dtype=np.int64),
        )
        dataset_hash = cohort.sha256_file(dataset)
        split_hash = cohort.sha256_arrays(train, validation, test)
        scaler_hash = cohort.sha256_arrays(mean, scale, variance)
        execution = {
            "protocol_id": cohort.PROTOCOL_ID,
            "dataset_sha256": dataset_hash,
            "split_indices_sha256": split_hash,
            "scaler_sha256": scaler_hash,
            "seeds": cohort.EXPECTED_SEEDS,
        }
        execution["execution_fingerprint_sha256"] = cohort.canonical_json_sha256(execution)
        write_json(split_root / "execution_contract.json", execution)
        write_json(
            split_root / "preprocessing_contract.json",
            {
                "protocol_id": cohort.PROTOCOL_ID,
                "dataset_sha256": dataset_hash,
                "dataset_shape": [len(rows), 17],
                "feature_names": cohort.FEATURE_NAMES,
                "class_names": cohort.CLASS_NAMES,
                "split_sizes": sizes,
                "split_indices_file": "split_indices.npz",
                "split_indices_file_sha256": cohort.sha256_file(
                    split_root / "split_indices.npz"
                ),
                "split_indices_sha256": split_hash,
                "scaler_parameters_file": "scaler_parameters.npz",
                "scaler_parameters_file_sha256": cohort.sha256_file(
                    split_root / "scaler_parameters.npz"
                ),
                "scaler_sha256": scaler_hash,
            },
        )
        seal(split_root, cohort.MANIFEST_NAME, "complete", cohort.PROTOCOL_ID)

        raw_q_frac = 8
        test_features = np.asarray(rows, dtype=np.float32)[test]
        replay_rows = []
        reference_rows = []
        for full_id, source_index in enumerate(test):
            fixed = np.rint(test_features[full_id].astype(np.float64) * (1 << raw_q_frac)).astype(np.int64)
            replay_rows.append(
                {
                    "row_id": full_id,
                    "source_row_index": int(source_index),
                    **{f"f{i}": int(fixed[i]) for i in range(17)},
                }
            )
            label = cohort.CLASS_NAMES.index(labels[source_index])
            reference_rows.append(
                {
                    "row_id": full_id,
                    "source_row_index": int(source_index),
                    "true_label": label,
                    "fixed_pred": label,
                    "fp32_pred": label,
                    **{f"fixed_logit_{i}": int(i == label) for i in range(5)},
                }
            )
        with (export_root / "hil_replay_vectors.csv").open("w", newline="", encoding="ascii") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(replay_rows[0]))
            writer.writeheader()
            writer.writerows(replay_rows)
        with (export_root / "hil_reference_predictions.csv").open("w", newline="", encoding="ascii") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(reference_rows[0]))
            writer.writeheader()
            writer.writerows(reference_rows)
        write_json(
            export_root / "preprocess_int_metadata.json",
            {"input_dim": 17, "raw_q_frac": raw_q_frac},
        )
        provenance = {
            "protocol_id": cohort.PROTOCOL_ID,
            "seed": 42,
            "student": "student_A",
            "route": "rf_kd",
            "dataset_sha256": dataset_hash,
            "split_indices_sha256": split_hash,
            "scaler_sha256": scaler_hash,
            "model_file_sha256": "1" * 64,
            "trained_state_sha256": "2" * 64,
        }
        identity_payload = {"provenance": provenance}
        export_id = cohort.canonical_json_sha256(identity_payload)
        write_json(
            export_root / "strict_export_report.json",
            {
                "status": "passed",
                "export_id": export_id,
                "provenance": provenance,
                "export_identity_payload": identity_payload,
                "gates": {
                    "full_test_rows": len(test),
                    "saved_test_rows_and_labels_exact": True,
                    "saved_fp32_predictions_exact": True,
                },
            },
        )
        seal(
            export_root,
            "strict_export_manifest.json",
            "passed",
            cohort.PROTOCOL_ID,
            export_id=export_id,
            export_identity_payload_sha256=export_id,
            seed=42,
            student="student_A",
            route="rf_kd",
            model_file_sha256="1" * 64,
            trained_state_sha256="2" * 64,
        )
        return dataset, split_root, export_root, sizes

    def test_end_to_end_emits_1000_balanced_rows_with_three_explicit_ids(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            dataset, split_root, export_root, sizes = self.build_fixture(root)
            output = root / "timing"
            manifest_path = cohort.build_timing_cohort(
                export_root,
                dataset,
                split_root,
                output,
                expected_split_sizes=sizes,
            )
            self.assertEqual(manifest_path, output / cohort.MANIFEST_NAME)
            metadata = cohort.read_json(output / cohort.COHORT_NAME)
            self.assertEqual(metadata["audit"]["rows"], 1000)
            self.assertEqual(metadata["audit"]["unique_feature_groups"], 1000)
            self.assertEqual(metadata["audit"]["duplicate_feature_groups"], 0)
            self.assertEqual(
                [row["true_label"] for row in metadata["rows"][:10]],
                [0, 1, 2, 3, 4] * 2,
            )
            for label in range(5):
                self.assertEqual(metadata["audit"]["class_balance"][str(label)]["rows"], 200)
            fields, rows = cohort.load_csv_rows(output / cohort.REPLAY_NAME)
            self.assertIn("timing_row_id", fields)
            self.assertIn("original_full_test_row_id", fields)
            self.assertIn("source_row_index", fields)
            self.assertEqual(integer_values(rows, "row_id"), list(range(1000)))
            cohort.verify_manifest(output, cohort.MANIFEST_NAME, {"complete"})

    def test_tampered_export_member_is_rejected_before_selection(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            dataset, split_root, export_root, sizes = self.build_fixture(root)
            with (export_root / "hil_reference_predictions.csv").open("a", encoding="ascii") as handle:
                handle.write("tampered\n")
            with self.assertRaisesRegex(RuntimeError, "Manifest size mismatch"):
                cohort.build_timing_cohort(
                    export_root,
                    dataset,
                    split_root,
                    root / "timing",
                    expected_split_sizes=sizes,
                )

    def test_historical_five_seed_protocol_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            dataset, split_root, export_root, sizes = self.build_fixture(root)
            report = cohort.read_json(export_root / "strict_export_report.json")
            historical = "wsnds_feature_group_split_train_only_scaler_5seed_v1"
            report["provenance"]["protocol_id"] = historical
            report["export_identity_payload"]["provenance"]["protocol_id"] = historical
            export_id = cohort.canonical_json_sha256(report["export_identity_payload"])
            report["export_id"] = export_id
            write_json(export_root / "strict_export_report.json", report)
            (export_root / "strict_export_manifest.json").unlink()
            seal(
                export_root,
                "strict_export_manifest.json",
                "passed",
                historical,
                export_id=export_id,
                export_identity_payload_sha256=export_id,
                seed=42,
                student="student_A",
                route="rf_kd",
                model_file_sha256="1" * 64,
                trained_state_sha256="2" * 64,
            )
            with self.assertRaisesRegex(RuntimeError, "wrong protocol_id"):
                cohort.build_timing_cohort(
                    export_root,
                    dataset,
                    split_root,
                    root / "timing",
                    expected_split_sizes=sizes,
                )

    def test_existing_output_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            dataset, split_root, export_root, sizes = self.build_fixture(root)
            output = root / "timing"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("unchanged", encoding="ascii")
            with self.assertRaises(FileExistsError):
                cohort.build_timing_cohort(
                    export_root,
                    dataset,
                    split_root,
                    output,
                    expected_split_sizes=sizes,
                )
            self.assertEqual(sentinel.read_text(encoding="ascii"), "unchanged")


def integer_values(rows: list[dict[str, str]], name: str) -> list[int]:
    return [int(row[name]) for row in rows]


class StaticContractTests(unittest.TestCase):
    def test_cli_is_fixed_to_final_protocol_and_exact_balance(self):
        source = (ROOT / "deployment/hardware_hil/host/generate_fgds_balanced_timing_cohort.py").read_text(
            encoding="utf-8"
        )
        for token in [
            'PROTOCOL_ID = "wsnds_feature_group_split_train_only_scaler_10seed_v2"',
            "GROUPS_PER_CLASS = 200",
            'SELECTION_ALGORITHM = "sha256_seeded_canonical_f32_rank_v1"',
            '"timing_row_id"',
            '"original_full_test_row_id"',
            '"source_row_index"',
            '"duplicate_feature_groups": 0',
            '"strict_export_manifest.json"',
            '"split_indices_sha256"',
        ]:
            self.assertIn(token, source)
        self.assertNotIn("stream_vectors", source)
        self.assertNotIn("serial.Serial", source)


if __name__ == "__main__":
    unittest.main()
