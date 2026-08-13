from __future__ import annotations

from pathlib import Path

import pytest

from deployment.final_hil.report import (
    RESPONSE_FIELDS,
    _blocked_results,
    _model_macros,
    _repeat_summary,
    _validate_response_rows,
    verify_report,
)


def test_model_macros_rederive_architecture_and_storage(tmp_path: Path) -> None:
    header = tmp_path / "model_weights.h"
    header.write_text(
        "\n".join(
            [
                "#define CUKD_INPUT_DIM 17",
                "#define CUKD_H1_DIM 32",
                "#define CUKD_H2_DIM 16",
                "#define CUKD_OUTPUT_DIM 5",
                "#define CUKD_WEIGHT_BYTES 1136",
                "#define CUKD_BIAS_BYTES 212",
                "#define CUKD_PARAM_BYTES 1348",
                "#define CUKD_ACTIVATION_BYTES_EST 140",
                "#define CUKD_MACS_PER_INFERENCE 1136",
            ]
        )
        + "\n",
        encoding="ascii",
    )

    result = _model_macros(header)

    assert result == {
        "architecture": "17-32-16-5",
        "weight_bytes": 1136,
        "bias_bytes": 212,
        "parameter_bytes": 1348,
        "activation_bytes_estimate": 140,
        "macs_per_inference": 1136,
    }


def test_repeat_summary_uses_sample_sd_of_three_repeat_means() -> None:
    repeats = []
    for repeat, total in enumerate((10, 20, 30), start=1):
        repeats.append(
            {
                "stage": f"timing_1000_r{repeat}",
                "repeat": repeat,
                "preprocess_us": {"mean": 1},
                "inference_us": {"mean": total - 1},
                "total_us": {"mean": total},
                "_preprocess_us_values": [1, 1],
                "_inference_us_values": [total - 1, total - 1],
                "_total_us_values": [total, total],
                "host_observed_rtt_us": {"mean": total + 100},
                "transaction_elapsed_us": {"mean": total + 100},
                "_host_observed_rtt_us_values": [total + 100, total + 100],
                "_transaction_elapsed_us_values": [total + 100, total + 100],
            }
        )

    result = _repeat_summary(repeats)

    assert result["total_us"]["mean_of_repeat_means"] == 20
    assert result["total_us"]["sample_sd_of_repeat_means"] == 10
    assert result["total_us"]["row_level_descriptive_3000"]["count"] == 6


def _response(row_id: int) -> dict[str, str]:
    row = {field: "0" for field in RESPONSE_FIELDS}
    row.update(
        {
            "row_id": str(row_id),
            "status": "OK",
            "predicted_class": "3",
            "preprocess_us": "7",
            "inference_us": "20",
            "total_us": "27",
            "attempts": "1",
            "response_timeout_count": "0",
            "ignored_bad_crc_count": "0",
        }
    )
    return row


def test_response_validation_rejects_non_dense_rows() -> None:
    with pytest.raises(RuntimeError, match="dense sequence"):
        _validate_response_rows([_response(0), _response(2)], 2)


def test_response_validation_rejects_ignored_or_retried_responses() -> None:
    ignored = _response(0)
    ignored["ignored_bad_crc_count"] = "1"
    with pytest.raises(RuntimeError, match="ignored-response"):
        _validate_response_rows([ignored], 1)

    retried = _response(0)
    retried["attempts"] = "2"
    with pytest.raises(RuntimeError, match="retried"):
        _validate_response_rows([retried], 1)


def test_blocked_route_uses_top_level_audit_gate_fields(tmp_path: Path) -> None:
    audit = {
        "fixed_vs_fp32_agreement": 0.989,
        "absolute_macro_f1_drop": 0.02,
        "minimum_fixed_vs_fp32_agreement": 0.99,
        "maximum_absolute_macro_f1_drop": 0.015,
    }
    audit_path = tmp_path / "blocked.json"
    audit_path.write_text(__import__("json").dumps(audit), encoding="utf-8")
    semantic = {
        "models": {
            "student_B_scratch": {
                "status": "blocked",
                "blocked_audit_local_path": "blocked.json",
            }
        }
    }
    evidence = {
        "blocked_routes": [
            {"model_key": "student_B_scratch", "reason": "quality gates failed"}
        ],
        "excluded_combinations": [
            {
                "model_key": "student_B_scratch",
                "combination_id": "student_B_scratch__arduino_r4__usb_serial",
            }
        ],
    }

    result = _blocked_results(tmp_path, semantic, evidence)

    assert result[0]["fixed_vs_fp32_agreement"] == 0.989
    assert result[0]["absolute_macro_f1_drop"] == 0.02


def test_verify_report_rejects_changed_artifact(tmp_path: Path) -> None:
    import hashlib
    import json

    from deployment.final_hil.contracts import canonical_json_sha256

    report = {"status": "passed", "archive_id": "a" * 64}
    report["report_id"] = canonical_json_sha256(report)
    summary = tmp_path / "final_hil_summary.json"
    summary.write_text(json.dumps(report), encoding="utf-8")
    for name in (
        "final_hil_summary.md",
        "final_hil_results.csv",
        "final_hil_timing_repeats.csv",
    ):
        (tmp_path / name).write_text(name, encoding="utf-8")
    files = []
    for name in (
        "final_hil_summary.json",
        "final_hil_summary.md",
        "final_hil_results.csv",
        "final_hil_timing_repeats.csv",
    ):
        data = (tmp_path / name).read_bytes()
        files.append(
            {
                "path": name,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    manifest = {
        "schema": "cukd_final_hil_report_artifact_manifest_v1",
        "report_id": report["report_id"],
        "archive_id": report["archive_id"],
        "files": files,
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    (tmp_path / "report_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    assert verify_report(tmp_path)["artifact_count"] == 4
    (tmp_path / "final_hil_summary.md").write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact changed"):
        verify_report(tmp_path)
