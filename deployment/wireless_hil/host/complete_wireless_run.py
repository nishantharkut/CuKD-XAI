"""Validate and seal one three-stage FG-DS Wi-Fi UDP HIL result directory."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

try:
    from .wireless_common import (
        REQUIRED_WIRELESS_STAGES,
        WIRELESS_COMPLETION_PROTOCOL_ID,
        WIRELESS_PROTOCOL_ID,
        read_json,
        sha256_file,
        validate_connection_record,
        verify_export_for_wireless,
        verify_wireless_bundle,
    )
except ImportError:
    from wireless_common import (  # type: ignore
        REQUIRED_WIRELESS_STAGES,
        WIRELESS_COMPLETION_PROTOCOL_ID,
        WIRELESS_PROTOCOL_ID,
        read_json,
        sha256_file,
        validate_connection_record,
        verify_export_for_wireless,
        verify_wireless_bundle,
    )


def require_within(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Result artifact escapes its root: {path}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def verify_stage(
    *,
    root: Path,
    stage_name: str,
    contract: dict[str, int],
    connection_hash: str,
    run_script_hash: str,
    bundle: dict[str, Any],
) -> dict[str, Any]:
    mcu_path = require_within(root, root / f"{stage_name}_mcu.csv")
    sequence_path = require_within(root, root / f"{stage_name}_sequence.json")
    metrics_path = require_within(root, root / f"{stage_name}_metrics.json")
    sequence = read_json(sequence_path)
    metrics = read_json(metrics_path)
    expected_rows = int(contract["rows"])
    expected_ordinal = int(contract["ordinal"])
    stage_contract = {
        "name": stage_name,
        "stage_id": sequence.get("stage_contract", {}).get("stage_id"),
        "ordinal": expected_ordinal,
        "expected_rows": expected_rows,
    }
    if re.fullmatch(r"[0-9A-F]{16}", str(stage_contract["stage_id"])) is None:
        raise RuntimeError(f"Wireless stage ID is invalid: {stage_name}")
    if (
        sequence.get("status") != "passed"
        or sequence.get("protocol_id") != WIRELESS_PROTOCOL_ID
        or sequence.get("stage_contract") != stage_contract
        or sequence.get("expected") != expected_rows
        or sequence.get("completed") != expected_rows
        or sequence.get("missing") != []
        or sequence.get("duplicates") != []
        or sequence.get("unexpected") != []
        or sequence.get("status_counts") != {"OK": expected_rows}
        or sequence.get("error") is not None
        or sequence.get("output_csv_sha256") != sha256_file(mcu_path)
    ):
        raise RuntimeError(f"Wireless sequence stage is not complete: {stage_name}")
    if (
        metrics.get("status") != "passed"
        or metrics.get("protocol_id") != WIRELESS_PROTOCOL_ID
        or metrics.get("stage_contract") != stage_contract
        or metrics.get("completed_vectors") != expected_rows
        or metrics.get("mcu_vs_fixed_reference_agreement") != 1.0
        or metrics.get("exact_logit_agreement") != 1.0
        or metrics.get("non_ok_status_count") != 0
    ):
        raise RuntimeError(f"Wireless metric stage is not passed: {stage_name}")
    expected_identity = {
        "board": bundle["board"],
        "student": bundle["student"],
        "export_id": bundle["export_id"],
        "wireless_bundle_id": bundle["wireless_bundle_id"],
        "connection_json_sha256": connection_hash,
        "run_script_sha256": run_script_hash,
    }
    for key, expected in expected_identity.items():
        if sequence.get("provenance", {}).get(key) != expected:
            raise RuntimeError(f"Sequence identity differs for {stage_name}/{key}")
        if metrics.get("provenance", {}).get(key) != expected:
            raise RuntimeError(f"Metric identity differs for {stage_name}/{key}")
    current_scripts = {
        "stream_script_sha256": sha256_file(
            Path(__file__).with_name("stream_vectors_udp.py")
        ),
        "udp_session_sha256": sha256_file(Path(__file__).with_name("udp_session.py")),
        "wireless_common_sha256": sha256_file(
            Path(__file__).with_name("wireless_common.py")
        ),
    }
    for key, expected in current_scripts.items():
        if sequence.get("provenance", {}).get(key) != expected:
            raise RuntimeError(f"Sequence script identity differs for {stage_name}/{key}")
        if metrics.get("provenance", {}).get(key) != expected:
            raise RuntimeError(f"Metric script identity differs for {stage_name}/{key}")
    if metrics.get("provenance", {}).get("verification_script_sha256") != sha256_file(
        Path(__file__).with_name("verify_results_udp.py")
    ):
        raise RuntimeError(f"Metric verifier identity differs for {stage_name}")
    if (
        metrics.get("provenance", {}).get("mcu_csv_sha256") != sha256_file(mcu_path)
        or metrics.get("provenance", {}).get("sequence_json_sha256")
        != sha256_file(sequence_path)
    ):
        raise RuntimeError(f"Metric provenance differs from stage files: {stage_name}")
    return {
        "rows": expected_rows,
        "stage_id": stage_contract["stage_id"],
        "mcu_csv_sha256": sha256_file(mcu_path),
        "sequence_json_sha256": sha256_file(sequence_path),
        "metrics_json_sha256": sha256_file(metrics_path),
        "mcu_vs_fp32_agreement": metrics["mcu_vs_fp32_agreement"],
        "macro_f1": metrics["macro_f1"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--connection-json", type=Path, required=True)
    parser.add_argument("--run-script", type=Path, required=True)
    args = parser.parse_args()

    root = args.result_dir.resolve()
    generated_dir = args.generated_dir.resolve()
    bundle_dir = args.bundle_dir.resolve()
    connection_path = args.connection_json.resolve()
    run_script = args.run_script.resolve()
    output = root / "wireless_hil_completion_manifest.json"
    temporary = output.with_suffix(output.suffix + ".tmp")
    if not root.is_dir() or output.exists() or temporary.exists():
        raise FileExistsError(f"Cannot create completion manifest at {output}")
    if connection_path != (root / "connection.json").resolve():
        raise RuntimeError("Connection evidence must be result-dir/connection.json")
    require_within(root, connection_path)
    require_within(root, root / "preflight.json")
    if not run_script.is_file():
        raise FileNotFoundError(run_script)

    export_manifest = verify_export_for_wireless(generated_dir)
    bundle = verify_wireless_bundle(bundle_dir, export_manifest)
    connection = validate_connection_record(
        connection_path,
        bundle,
        generated_dir / "strict_export_manifest.json",
    )
    connection_hash = sha256_file(connection_path)
    run_script_hash = sha256_file(run_script)
    preflight_path = root / "preflight.json"
    preflight = read_json(preflight_path)
    if (
        preflight.get("status") != "passed"
        or preflight.get("protocol_id") != WIRELESS_PROTOCOL_ID
        or preflight.get("board") != bundle["board"]
        or preflight.get("student") != bundle["student"]
        or preflight.get("export_id") != bundle["export_id"]
        or preflight.get("wireless_bundle_id") != bundle["wireless_bundle_id"]
        or preflight.get("session_id") != connection["session_id"]
        or preflight.get("connection_json_sha256") != connection_hash
        or preflight.get("preflight_script_sha256")
        != sha256_file(Path(__file__).with_name("preflight_wireless_hil.py"))
        or preflight.get("udp_session_sha256")
        != sha256_file(Path(__file__).with_name("udp_session.py"))
        or preflight.get("wireless_common_sha256")
        != sha256_file(Path(__file__).with_name("wireless_common.py"))
    ):
        raise RuntimeError("Wireless preflight does not match the run identity")

    stage_evidence = {
        name: verify_stage(
            root=root,
            stage_name=name,
            contract=contract,
            connection_hash=connection_hash,
            run_script_hash=run_script_hash,
            bundle=bundle,
        )
        for name, contract in REQUIRED_WIRELESS_STAGES.items()
    }
    stage_ids = {evidence["stage_id"] for evidence in stage_evidence.values()}
    if len(stage_ids) != len(REQUIRED_WIRELESS_STAGES):
        raise RuntimeError("Wireless stages reused a stage nonce")

    expected_names = {"connection.json", "preflight.json"} | {
        f"{stage}_{suffix}"
        for stage in REQUIRED_WIRELESS_STAGES
        for suffix in ["mcu.csv", "sequence.json", "metrics.json"]
    }
    actual_names = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.resolve() != output.resolve()
    }
    if actual_names != expected_names:
        raise RuntimeError(
            "Wireless result inventory differs: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"unexpected={sorted(actual_names - expected_names)}"
        )
    files = [
        {
            "path": name,
            "size_bytes": (root / name).stat().st_size,
            "sha256": sha256_file(root / name),
        }
        for name in sorted(actual_names)
    ]
    payload = {
        "status": "complete",
        "protocol_id": WIRELESS_COMPLETION_PROTOCOL_ID,
        "wireless_application_protocol_id": WIRELESS_PROTOCOL_ID,
        "required_stages": REQUIRED_WIRELESS_STAGES,
        "board": bundle["board"],
        "student": bundle["student"],
        "export_id": bundle["export_id"],
        "wireless_bundle_id": bundle["wireless_bundle_id"],
        "session_id": connection["session_id"],
        "connection_json_sha256": connection_hash,
        "preflight_json_sha256": sha256_file(preflight_path),
        "run_script_sha256": run_script_hash,
        "completion_script_sha256": sha256_file(Path(__file__).resolve()),
        "stage_evidence": stage_evidence,
        "full_stage_inferences": REQUIRED_WIRELESS_STAGES["full_56301"]["rows"],
        "all_stage_inferences": sum(
            int(contract["rows"]) for contract in REQUIRED_WIRELESS_STAGES.values()
        ),
        "file_count_excluding_manifest": len(files),
        "files": files,
        "claim_boundary": (
            "Completion seals one provisioned board/model session containing smoke, "
            "validation, and full controlled-LAN Wi-Fi UDP replay stages."
        ),
    }
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
