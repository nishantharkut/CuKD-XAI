import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deployment"))

from hardware_hil.host.stream_vectors import write_outputs
from hardware_hil.host.hil_common import (
    compute_classification_metrics,
    crc16_ccitt,
    decode_response_line,
    encode_request_line,
    summarize_latency,
    verify_response_sequence,
)


class HardwareHilHostTests(unittest.TestCase):
    def test_request_encoding_uses_row_id_17_features_and_crc(self):
        features = list(range(17))
        line = encode_request_line(row_id=42, features=features)

        self.assertTrue(line.startswith("CUKD1,42,"))
        self.assertTrue(line.endswith("\n"))

        parts = line.strip().split(",")
        self.assertEqual(parts[0], "CUKD1")
        self.assertEqual(int(parts[1]), 42)
        self.assertEqual([int(v) for v in parts[2:19]], features)
        expected_crc = crc16_ccitt(",".join(parts[:-1]).encode("ascii"))
        self.assertEqual(int(parts[-1], 16), expected_crc)

    def test_request_encoding_rejects_wrong_feature_count(self):
        with self.assertRaisesRegex(ValueError, "17 features"):
            encode_request_line(row_id=1, features=[1, 2, 3])

    def test_response_decoder_rejects_bad_crc(self):
        bad = "CUKD1R,7,OK,2,1,2,3,4,5,10,20,30,0000\n"

        with self.assertRaisesRegex(ValueError, "CRC"):
            decode_response_line(bad)

    def test_response_decoder_accepts_valid_packet(self):
        body = "CUKD1R,7,OK,2,1,2,3,4,5,10,20,30"
        crc = crc16_ccitt(body.encode("ascii"))
        packet = f"{body},{crc:04X}\n"

        decoded = decode_response_line(packet)

        self.assertEqual(decoded["row_id"], 7)
        self.assertEqual(decoded["status"], "OK")
        self.assertEqual(decoded["predicted_class"], 2)
        self.assertEqual(decoded["logits"], [1, 2, 3, 4, 5])
        self.assertEqual(decoded["preprocess_us"], 10)
        self.assertEqual(decoded["inference_us"], 20)
        self.assertEqual(decoded["total_us"], 30)

    def test_verify_response_sequence_detects_missing_and_duplicate_rows(self):
        responses = [
            {"row_id": 1, "status": "OK"},
            {"row_id": 1, "status": "OK"},
            {"row_id": 3, "status": "OK"},
        ]

        report = verify_response_sequence(expected_row_ids=[1, 2, 3], responses=responses)

        self.assertEqual(report["completed"], 3)
        self.assertEqual(report["duplicates"], [1])
        self.assertEqual(report["missing"], [2])
        self.assertEqual(report["status_counts"], {"OK": 3})

    def test_stream_write_outputs_preserves_partial_error_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_csv = tmp_path / "responses.csv"
            summary_json = tmp_path / "summary.json"

            write_outputs(
                responses=[{
                    "row_id": 0,
                    "status": "OK",
                    "predicted_class": 1,
                    "logits": [1, 2, 3, 4, 5],
                    "preprocess_us": 10,
                    "inference_us": 20,
                    "total_us": 30,
                }],
                output_csv=output_csv,
                summary_json=summary_json,
                expected_row_ids=[0, 1],
                error={"type": "TimeoutError", "message": "timeout waiting for row_id=1"},
            )

            summary = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertTrue(output_csv.exists())
            self.assertEqual(summary["completed"], 1)
            self.assertEqual(summary["missing"], [1])
            self.assertEqual(summary["error"]["type"], "TimeoutError")

    def test_classification_metrics_include_macro_and_weighted_f1(self):
        metrics = compute_classification_metrics(
            y_true=[0, 0, 1, 1, 2, 2],
            y_pred=[0, 1, 1, 1, 2, 0],
            labels=[0, 1, 2],
        )

        self.assertAlmostEqual(metrics["accuracy"], 4 / 6)
        self.assertIn("macro_f1", metrics)
        self.assertIn("weighted_f1", metrics)
        self.assertEqual(set(metrics["per_class"].keys()), {"0", "1", "2"})
        self.assertEqual(len(metrics["confusion_matrix"]), 3)

    def test_latency_summary_reports_percentiles(self):
        summary = summarize_latency([10, 20, 30, 40, 50])

        self.assertEqual(summary["min"], 10)
        self.assertEqual(summary["max"], 50)
        self.assertEqual(summary["mean"], 30)
        self.assertEqual(summary["median"], 30)
        self.assertEqual(summary["p95"], 50)
        self.assertEqual(summary["p99"], 50)


if __name__ == "__main__":
    unittest.main()
