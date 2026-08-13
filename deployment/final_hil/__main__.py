"""Command-line entry point for the additive final HIL campaign layer."""

from __future__ import annotations

import argparse
import getpass
import json
from pathlib import Path
from typing import Any

from .archive import create_campaign_archive, verify_campaign_archive
from .bundles import prepare_final_bundle
from .contracts import (
    FINAL_STAGES,
    MODEL_KEYS,
    TRANSPORTS,
    build_campaign_contract,
    generate_balanced_cohort,
)
from .evidence import (
    complete_six_stage_session,
    preflight_campaign,
    record_build_upload_provenance,
    verify_complete_campaign,
)
from .runtime import (
    configure_final_wifi,
    execute_usb_stage,
    execute_wifi_stage,
    verify_stage_attempt,
)


def _assignments(values: list[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        key, separator, path = value.partition("=")
        if not separator or not key or not path or key in result:
            raise ValueError(f"Invalid or duplicate {label} assignment: {value!r}")
        result[key] = Path(path)
    return result


def _contract_sources(exports: list[str], blocked: list[str]) -> dict[str, Any]:
    export_paths = _assignments(exports, "export")
    blocked_paths = _assignments(blocked, "blocked audit")
    overlap = set(export_paths) & set(blocked_paths)
    if overlap:
        raise ValueError(f"Models have both export and blocked audit: {sorted(overlap)}")
    unknown = (set(export_paths) | set(blocked_paths)) - set(MODEL_KEYS)
    if unknown:
        raise ValueError(f"Unknown model keys: {sorted(unknown)}")
    return {
        **{key: {"export_dir": path} for key, path in export_paths.items()},
        **{key: {"blocked_audit": path} for key, path in blocked_paths.items()},
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)

    contract = sub.add_parser("contract", help="build the fixed final campaign contract")
    contract.add_argument("--export", action="append", default=[], metavar="MODEL=DIR")
    contract.add_argument("--blocked", action="append", default=[], metavar="MODEL=JSON")
    contract.add_argument(
        "--transport",
        action="append",
        choices=TRANSPORTS,
        help="transport to include; omit to retain the full USB and Wi-Fi matrix",
    )
    contract.add_argument("--output", type=Path, required=True)

    cohort = sub.add_parser("cohort", help="generate the one shared balanced cohort")
    cohort.add_argument("--export", action="append", required=True, metavar="MODEL=DIR")
    cohort.add_argument("--dataset-csv", type=Path, required=True)
    cohort.add_argument("--split-root", type=Path, required=True)
    cohort.add_argument("--output-dir", type=Path, required=True)

    bundle = sub.add_parser("bundle", help="prepare one immutable final firmware bundle")
    bundle.add_argument("--export-dir", type=Path, required=True)
    bundle.add_argument("--output-dir", type=Path, required=True)
    bundle.add_argument("--board", choices=["esp32c3", "arduino_r4"], required=True)
    bundle.add_argument("--transport", choices=["usb_serial", "wifi_udp"], required=True)
    bundle.add_argument("--build-contract", type=Path, required=True)

    wifi = sub.add_parser("configure-wifi", help="provision credentials over local serial")
    wifi.add_argument("--export-dir", type=Path, required=True)
    wifi.add_argument("--bundle-dir", type=Path, required=True)
    wifi.add_argument("--port", required=True)
    wifi.add_argument("--physical-port-serial", required=True)
    wifi.add_argument("--output", type=Path, required=True)

    usb_run = sub.add_parser("run-usb-stage", help="execute one no-retry USB stage")
    for item in [usb_run]:
        item.add_argument("--export-dir", type=Path, required=True)
        item.add_argument("--cohort-dir", type=Path, required=True)
        item.add_argument("--bundle-dir", type=Path, required=True)
        item.add_argument(
            "--stage",
            choices=[stage["name"] for stage in FINAL_STAGES],
            required=True,
        )
        item.add_argument("--output-root", type=Path, required=True)
    usb_run.add_argument("--port", required=True)
    usb_run.add_argument("--physical-port-serial", required=True)
    usb_run.add_argument("--campaign-session-id", required=True)

    wifi_run = sub.add_parser("run-wifi-stage", help="execute one idempotent Wi-Fi stage")
    for item in [wifi_run]:
        item.add_argument("--export-dir", type=Path, required=True)
        item.add_argument("--cohort-dir", type=Path, required=True)
        item.add_argument("--bundle-dir", type=Path, required=True)
        item.add_argument(
            "--stage",
            choices=[stage["name"] for stage in FINAL_STAGES],
            required=True,
        )
        item.add_argument("--output-root", type=Path, required=True)
    wifi_run.add_argument("--connection-json", type=Path, required=True)
    wifi_run.add_argument("--campaign-session-id", required=True)
    wifi_run.add_argument("--timeout", type=float, default=1.0)
    wifi_run.add_argument("--max-attempts", type=int, default=3)

    verify = sub.add_parser("verify-stage", help="recompute one stage proof")
    verify.add_argument("--attempt-dir", type=Path, required=True)
    verify.add_argument("--export-dir", type=Path, required=True)
    verify.add_argument("--cohort-dir", type=Path, required=True)
    verify.add_argument("--bundle-dir", type=Path, required=True)

    provenance = sub.add_parser(
        "build-upload",
        help="execute and seal one compile/upload/direct-board-identity sequence",
    )
    provenance.add_argument("--export-dir", type=Path, required=True)
    provenance.add_argument("--bundle-dir", type=Path, required=True)
    provenance.add_argument("--physical-port", required=True)
    provenance.add_argument("--physical-port-serial", required=True)
    provenance.add_argument("--output-dir", type=Path, required=True)
    provenance.add_argument("--command-timeout", type=float, default=1800.0)

    session = sub.add_parser("complete-session", help="seal exactly six passed stages")
    session.add_argument("--attempt-dir", action="append", type=Path, required=True)
    session.add_argument("--export-dir", type=Path, required=True)
    session.add_argument("--cohort-dir", type=Path, required=True)
    session.add_argument("--bundle-dir", type=Path, required=True)
    session.add_argument("--provenance", type=Path, required=True)
    session.add_argument("--output", type=Path, required=True)

    preflight = sub.add_parser(
        "preflight", help="verify all intended and gate-eligible combinations"
    )
    preflight.add_argument("--contract", type=Path, required=True)
    preflight.add_argument("--cohort-dir", type=Path)
    preflight.add_argument("--export", action="append", default=[], metavar="MODEL=DIR")
    preflight.add_argument("--bundle", action="append", default=[], metavar="COMBINATION=DIR")
    preflight.add_argument("--output", type=Path, required=True)

    campaign = sub.add_parser(
        "complete-campaign", help="seal every gate-eligible completed session"
    )
    campaign.add_argument("--contract", type=Path, required=True)
    campaign.add_argument("--session", action="append", type=Path, required=True)
    campaign.add_argument("--output", type=Path, required=True)

    archive = sub.add_parser(
        "archive-campaign",
        help="create one portable, deeply verified final-HIL evidence archive",
    )
    archive.add_argument("--contract", type=Path, required=True)
    archive.add_argument("--campaign-evidence", type=Path)
    archive.add_argument("--cohort-dir", type=Path)
    archive.add_argument("--export", action="append", default=[], metavar="MODEL=DIR")
    archive.add_argument(
        "--blocked", action="append", default=[], metavar="MODEL=JSON"
    )
    archive.add_argument(
        "--source-root",
        action="append",
        default=[],
        metavar="RECORDED_ROOT=LOCAL_ROOT",
        help="relocate sealed historical paths without changing their evidence records",
    )
    archive.add_argument(
        "--host-source-root",
        type=Path,
        help="repository root containing the exact host-source ledger recorded by HIL",
    )
    archive.add_argument("--output-dir", type=Path, required=True)

    verify_archive = sub.add_parser(
        "verify-archive",
        help="deeply verify a final-HIL archive without original source paths",
    )
    verify_archive.add_argument("--archive-dir", type=Path, required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "contract":
        result = build_campaign_contract(
            _contract_sources(args.export, args.blocked),
            transports=args.transport or TRANSPORTS,
            output_path=args.output,
        )
        print(json.dumps({"status": result["status"], "contract_id": result["contract_id"]}))
        return 0 if result["status"].startswith("ready") else 2
    if args.command == "cohort":
        path = generate_balanced_cohort(
            _assignments(args.export, "export"),
            dataset_csv=args.dataset_csv,
            split_root=args.split_root,
            output_dir=args.output_dir,
        )
    elif args.command == "bundle":
        path = prepare_final_bundle(
            export_dir=args.export_dir,
            output_dir=args.output_dir,
            board=args.board,
            transport=args.transport,
            build_contract=args.build_contract,
        )
    elif args.command == "configure-wifi":
        path = configure_final_wifi(
            export_dir=args.export_dir,
            bundle_dir=args.bundle_dir,
            port=args.port,
            physical_port_serial=args.physical_port_serial,
            ssid=input("Wi-Fi SSID (not recorded): "),
            password=getpass.getpass("Wi-Fi password (not recorded): "),
            output_json=args.output,
        )
    elif args.command == "run-usb-stage":
        path = execute_usb_stage(
            export_dir=args.export_dir,
            cohort_dir=args.cohort_dir,
            bundle_dir=args.bundle_dir,
            stage_name=args.stage,
            port=args.port,
            physical_port_serial=args.physical_port_serial,
            output_root=args.output_root,
            campaign_session_id=args.campaign_session_id,
        )
    elif args.command == "run-wifi-stage":
        path = execute_wifi_stage(
            export_dir=args.export_dir,
            cohort_dir=args.cohort_dir,
            bundle_dir=args.bundle_dir,
            connection_json=args.connection_json,
            stage_name=args.stage,
            output_root=args.output_root,
            campaign_session_id=args.campaign_session_id,
            timeout_seconds=args.timeout,
            max_attempts=args.max_attempts,
        )
    elif args.command == "verify-stage":
        result = verify_stage_attempt(
            args.attempt_dir,
            export_dir=args.export_dir,
            cohort_dir=args.cohort_dir,
            bundle_dir=args.bundle_dir,
        )
        print(json.dumps({"status": result["status"], "attempt_id": result["attempt_id"]}))
        return 0
    elif args.command == "build-upload":
        path = record_build_upload_provenance(
            export_dir=args.export_dir,
            bundle_dir=args.bundle_dir,
            physical_port=args.physical_port,
            physical_port_serial=args.physical_port_serial,
            output_dir=args.output_dir,
            command_timeout_seconds=args.command_timeout,
        )
    elif args.command == "complete-session":
        path = complete_six_stage_session(
            attempt_dirs=args.attempt_dir,
            export_dir=args.export_dir,
            cohort_dir=args.cohort_dir,
            bundle_dir=args.bundle_dir,
            provenance_json=args.provenance,
            output_json=args.output,
        )
    elif args.command == "preflight":
        result = preflight_campaign(
            campaign_contract=args.contract,
            cohort_dir=args.cohort_dir,
            export_dirs=_assignments(args.export, "export"),
            bundle_dirs=_assignments(args.bundle, "bundle"),
            output_json=args.output,
        )
        print(json.dumps({"status": result["status"], "preflight_id": result["preflight_id"]}))
        return 0 if result["status"].startswith("ready") else 2
    elif args.command == "complete-campaign":
        result = verify_complete_campaign(
            campaign_contract=args.contract,
            session_jsons=args.session,
            output_json=args.output,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "campaign_evidence_id": result["campaign_evidence_id"],
                }
            )
        )
        return 0
    elif args.command == "archive-campaign":
        path = create_campaign_archive(
            campaign_contract=args.contract,
            campaign_evidence=args.campaign_evidence,
            cohort_dir=args.cohort_dir,
            export_dirs=_assignments(args.export, "export"),
            blocked_audits=_assignments(args.blocked, "blocked audit"),
            source_roots=_assignments(args.source_root, "source root"),
            host_source_root=args.host_source_root,
            output_dir=args.output_dir,
        )
    elif args.command == "verify-archive":
        result = verify_campaign_archive(args.archive_dir)
        print(json.dumps(result, sort_keys=True))
        return 0
    else:  # pragma: no cover - argparse enforces this
        raise AssertionError(args.command)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
