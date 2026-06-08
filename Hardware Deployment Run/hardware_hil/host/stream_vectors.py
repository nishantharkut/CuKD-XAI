"""Stream fixed-point WSN-DS feature vectors to an MCU over USB serial."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

try:
    import serial
except Exception:  # pragma: no cover - handled at runtime on host
    serial = None

from hardware_hil.host.hil_common import (
    FEATURE_COUNT,
    decode_response_line,
    encode_request_line,
    verify_response_sequence,
)


def load_vectors(path: Path, limit: int | None = None) -> list[dict[str, object]]:
    """Load rows with row_id and f0..f16 columns."""
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = ["row_id", *[f"f{i}" for i in range(FEATURE_COUNT)]]
        missing = [col for col in required if col not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"missing required columns: {missing}")
        for row in reader:
            row_id = int(row["row_id"])
            features = [int(row[f"f{i}"]) for i in range(FEATURE_COUNT)]
            rows.append({"row_id": row_id, "features": features})
            if limit is not None and len(rows) >= limit:
                break
    return rows


class SerialReplay:
    def __init__(self, port: str, baud: int, timeout: float) -> None:
        if serial is None:
            raise RuntimeError("pyserial is required; install hardware_hil/host/requirements.txt")
        self.timeout = float(timeout)
        self.device = serial.Serial(port=port, baudrate=baud, timeout=timeout, write_timeout=timeout)

    def close(self) -> None:
        self.device.close()

    def drain_startup(self) -> None:
        if hasattr(self.device, "reset_input_buffer"):
            self.device.reset_input_buffer()
        if hasattr(self.device, "reset_output_buffer"):
            self.device.reset_output_buffer()

    def transact(self, row_id: int, features: list[int]) -> dict[str, object]:
        line = encode_request_line(row_id=row_id, features=features)
        self.device.write(line.encode("ascii"))
        self.device.flush()

        for _ in range(25):
            response = self.device.readline().decode("ascii", errors="replace")
            if not response:
                raise TimeoutError(f"timeout waiting for row_id={row_id}")
            if not response.startswith("CUKD1R,"):
                continue

            decoded = decode_response_line(response)
            if int(decoded["row_id"]) != int(row_id):
                raise ValueError(f"row mismatch: sent {row_id}, received {decoded['row_id']}")
            return decoded

        raise TimeoutError(f"too many non-protocol serial lines before row_id={row_id}")


def write_outputs(
    responses: list[dict[str, object]],
    output_csv: Path,
    summary_json: Path,
    expected_row_ids: list[int],
    error: dict[str, object] | None = None,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "row_id",
            "status",
            "predicted_class",
            "logits",
            "preprocess_us",
            "inference_us",
            "total_us",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for response in responses:
            row = dict(response)
            row["logits"] = " ".join(str(v) for v in row.get("logits", []))
            writer.writerow(row)

    summary = verify_response_sequence(expected_row_ids, responses)
    summary["output_csv"] = str(output_csv)
    if error is not None:
        summary["error"] = error
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--vectors-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    args = parser.parse_args()

    rows = load_vectors(Path(args.vectors_csv), limit=args.limit)
    replay = SerialReplay(args.port, args.baud, args.timeout)
    responses: list[dict[str, object]] = []
    replay_error: dict[str, object] | None = None
    try:
        time.sleep(args.settle_seconds)
        replay.drain_startup()
        for row in rows:
            responses.append(replay.transact(int(row["row_id"]), list(row["features"])))
    except Exception as exc:
        replay_error = {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "completed_before_error": len(responses),
        }
    finally:
        replay.close()

    write_outputs(
        responses=responses,
        output_csv=Path(args.output_csv),
        summary_json=Path(args.summary_json),
        expected_row_ids=[int(row["row_id"]) for row in rows],
        error=replay_error,
    )
    if replay_error is not None:
        print(json.dumps(replay_error, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

