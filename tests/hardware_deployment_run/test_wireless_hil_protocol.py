import csv
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from deployment.hardware_hil.host.hil_common import (
    compute_classification_metrics,
    crc16_ccitt,
)
from deployment.wireless_hil.host.udp_session import (
    DeviceProtocolError,
    StrictUdpSession,
    parse_end_response,
)
from deployment.wireless_hil.host.verify_results_udp import (
    reconcile_full_metrics_with_strict_export,
)
from deployment.wireless_hil.host.wireless_common import (
    DEFAULT_DEVICE_UDP_PORT,
    DEFAULT_HOST_UDP_PORT,
    REQUEST_ENVELOPE_PREFIX,
    RESPONSE_ENVELOPE_PREFIX,
    WIRELESS_PROTOCOL_ID,
    EnvelopeError,
    decode_wifi_config_response,
    decode_wireless_envelope,
    encode_wifi_config_line,
    encode_wireless_envelope,
    expected_device_identity,
    read_compile_log_text,
    sha256_file,
    validate_compile_log_metadata,
    verify_export_for_wireless,
    verify_wireless_bundle,
)


SESSION = "A" * 32
STAGE = "B" * 16
TRANSACTION = "000000000000002A"


def free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as endpoint:
        endpoint.bind(("127.0.0.1", 0))
        return int(endpoint.getsockname()[1])


class WirelessEnvelopeTests(unittest.TestCase):
    def test_compile_log_decoder_accepts_utf8_and_bom_utf16_and_strips_ansi(self):
        line = (
            "Sketch uses 100 bytes (10%) of program storage space. "
            "Maximum is 1000 bytes.\n"
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            utf8 = root / "utf8.log"
            utf16 = root / "utf16.log"
            utf8.write_bytes(("\x1b[32m" + line + "\x1b[0m").encode("utf-8"))
            utf16.write_bytes(line.encode("utf-16"))
            self.assertEqual(read_compile_log_text(utf8), line)
            self.assertEqual(read_compile_log_text(utf16), line)

    def test_compile_log_decoder_rejects_ambiguous_or_binary_input(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            invalid_utf8 = root / "invalid.log"
            embedded_nul = root / "nul.log"
            invalid_utf8.write_bytes(b"compile\xfflog")
            embedded_nul.write_bytes(b"compile\x00log")
            with self.assertRaisesRegex(RuntimeError, "not valid"):
                read_compile_log_text(invalid_utf8)
            with self.assertRaisesRegex(RuntimeError, "embedded NUL"):
                read_compile_log_text(embedded_nul)

    def test_compile_log_metadata_requires_exact_provenance_markers(self):
        text = "\n".join(
            [
                "CUKD_FQBN=esp32:esp32:esp32c3",
                "CUKD_BOARD_CORE_VERSION=3.3.11",
                "CUKD_FRONTEND_VERSION=arduino-cli 1.5.1",
                "CUKD_TOOLCHAIN_VERSION=14.2.0",
            ]
        )
        validate_compile_log_metadata(
            text,
            fqbn="esp32:esp32:esp32c3",
            board_core_version="3.3.11",
            frontend_version="arduino-cli 1.5.1",
            toolchain_version="14.2.0",
        )
        with self.assertRaisesRegex(RuntimeError, "exact provenance"):
            validate_compile_log_metadata(
                text.replace("14.2.0", "14.2.1"),
                fqbn="esp32:esp32:esp32c3",
                board_core_version="3.3.11",
                frontend_version="arduino-cli 1.5.1",
                toolchain_version="14.2.0",
            )

    def test_envelope_round_trip_and_tamper_rejection(self):
        packet = encode_wireless_envelope(
            prefix=REQUEST_ENVELOPE_PREFIX,
            session_id=SESSION,
            stage_id=STAGE,
            transaction_id=TRANSACTION,
            attempt=1,
            inner_text="CUKD1,42,1",
        )
        decoded = decode_wireless_envelope(
            packet, expected_prefix=REQUEST_ENVELOPE_PREFIX
        )
        self.assertEqual(decoded.inner_text, "CUKD1,42,1")
        self.assertEqual(decoded.attempt, 1)
        tampered = bytearray(packet)
        tampered[-1] = ord("0") if tampered[-1] != ord("0") else ord("1")
        with self.assertRaises(EnvelopeError):
            decode_wireless_envelope(bytes(tampered))

    def test_envelope_rejects_noncanonical_or_binary_fields(self):
        valid = encode_wireless_envelope(
            prefix=REQUEST_ENVELOPE_PREFIX,
            session_id=SESSION,
            stage_id=STAGE,
            transaction_id=TRANSACTION,
            attempt=1,
            inner_text="PING",
        )
        with self.assertRaisesRegex(EnvelopeError, "NUL"):
            decode_wireless_envelope(valid[:10] + b"\x00" + valid[11:])
        with self.assertRaisesRegex(ValueError, "uppercase"):
            encode_wireless_envelope(
                prefix=REQUEST_ENVELOPE_PREFIX,
                session_id=SESSION.lower(),
                stage_id=STAGE,
                transaction_id=TRANSACTION,
                attempt=1,
                inner_text="PING",
            )
        with self.assertRaisesRegex(ValueError, "1..255"):
            encode_wireless_envelope(
                prefix=REQUEST_ENVELOPE_PREFIX,
                session_id=SESSION,
                stage_id=STAGE,
                transaction_id=TRANSACTION,
                attempt=0,
                inner_text="PING",
            )

    def test_end_response_requires_exact_completed_inference_count(self):
        text = (
            f"CUKDWENDR,{STAGE},10,10,12,0,0,0,0,0,0,0,0,1,0,10,1,OK"
        )
        parsed = parse_end_response(
            text, stage_id=STAGE, ordinal=1, expected_rows=10
        )
        self.assertEqual(parsed["duplicate_replays"], 1)
        with self.assertRaisesRegex(ValueError, "completed stage"):
            parse_end_response(
                text.replace(",10,1,OK", ",9,1,OK"),
                stage_id=STAGE,
                ordinal=1,
                expected_rows=10,
            )

    def test_wifi_config_response_is_canonical_and_ipv4_unicast(self):
        body = (
            f"CUKDWCFG2R,{SESSION},OK,192.168.137.50,42101,-42,312E302E30,"
            "AA:BB:CC:DD:EE:FF"
        )
        response = f"{body},{crc16_ccitt(body.encode('ascii')):04X}\r\n"
        parsed = decode_wifi_config_response(response)
        self.assertEqual(parsed["device_ip"], "192.168.137.50")
        self.assertEqual(parsed["connectivity_firmware"], "1.0.0")

        for invalid_body in [
            body.replace("312E302E30", "312e302e30"),
            body.replace("192.168.137.50", "224.0.0.1"),
        ]:
            invalid = (
                f"{invalid_body},"
                f"{crc16_ccitt(invalid_body.encode('ascii')):04X}\n"
            )
            with self.assertRaises(ValueError):
                decode_wifi_config_response(invalid)
        with self.assertRaisesRegex(ValueError, "framing"):
            decode_wifi_config_response(response + "\n")


class StrictUdpSessionTests(unittest.TestCase):
    def test_session_identity_is_validated_before_socket_creation(self):
        with self.assertRaisesRegex(ValueError, "32 uppercase"):
            StrictUdpSession(
                device_ip="127.0.0.1",
                device_port=free_udp_port(),
                host_port=free_udp_port(),
                session_id=SESSION.lower(),
                timeout_seconds=0.1,
                max_attempts=1,
            )

    def test_retry_filters_wrong_endpoint_and_stale_attempt(self):
        device_port = free_udp_port()
        host_port = free_udp_port()
        server_error: list[BaseException] = []

        def server() -> None:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as device:
                    device.bind(("127.0.0.1", device_port))
                    first, host = device.recvfrom(1024)
                    first_request = decode_wireless_envelope(
                        first, expected_prefix=REQUEST_ENVELOPE_PREFIX
                    )
                    self.assertEqual(first_request.attempt, 1)
                    second, host_again = device.recvfrom(1024)
                    self.assertEqual(host_again, host)
                    second_request = decode_wireless_envelope(
                        second, expected_prefix=REQUEST_ENVELOPE_PREFIX
                    )
                    self.assertEqual(second_request.attempt, 2)
                    rogue_payload = encode_wireless_envelope(
                        prefix=RESPONSE_ENVELOPE_PREFIX,
                        session_id=SESSION,
                        stage_id=STAGE,
                        transaction_id=TRANSACTION,
                        attempt=2,
                        inner_text="PONG",
                    )
                    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as rogue:
                        rogue.sendto(rogue_payload, host)
                    stale_payload = encode_wireless_envelope(
                        prefix=RESPONSE_ENVELOPE_PREFIX,
                        session_id=SESSION,
                        stage_id=STAGE,
                        transaction_id=TRANSACTION,
                        attempt=1,
                        inner_text="PONG",
                    )
                    device.sendto(stale_payload, host)
                    time.sleep(0.005)
                    matching = encode_wireless_envelope(
                        prefix=RESPONSE_ENVELOPE_PREFIX,
                        session_id=SESSION,
                        stage_id=STAGE,
                        transaction_id=TRANSACTION,
                        attempt=2,
                        inner_text="PONG",
                    )
                    device.sendto(matching, host)
            except BaseException as exc:  # surfaced in the test thread
                server_error.append(exc)

        thread = threading.Thread(target=server, daemon=True)
        thread.start()
        session = StrictUdpSession(
            device_ip="127.0.0.1",
            device_port=device_port,
            host_port=host_port,
            session_id=SESSION,
            timeout_seconds=0.05,
            max_attempts=3,
        )
        try:
            result = session.exchange(
                stage_id=STAGE,
                transaction_id=TRANSACTION,
                inner_text="PING",
            )
        finally:
            session.close()
        thread.join(timeout=1.0)
        if server_error:
            raise server_error[0]
        self.assertFalse(thread.is_alive())
        self.assertEqual(result.inner_text, "PONG")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.response_timeout_count, 1)
        self.assertEqual(result.ignored_by_category["wrong_endpoint"], 1)
        self.assertEqual(result.ignored_by_category["wrong_attempt"], 1)

    def test_future_attempt_response_is_fatal(self):
        device_port = free_udp_port()
        host_port = free_udp_port()

        def server() -> None:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as device:
                device.bind(("127.0.0.1", device_port))
                request_payload, host = device.recvfrom(1024)
                request = decode_wireless_envelope(
                    request_payload, expected_prefix=REQUEST_ENVELOPE_PREFIX
                )
                response = encode_wireless_envelope(
                    prefix=RESPONSE_ENVELOPE_PREFIX,
                    session_id=request.session_id,
                    stage_id=request.stage_id,
                    transaction_id=request.transaction_id,
                    attempt=request.attempt + 1,
                    inner_text="PONG",
                )
                device.sendto(response, host)

        thread = threading.Thread(target=server, daemon=True)
        thread.start()
        session = StrictUdpSession(
            device_ip="127.0.0.1",
            device_port=device_port,
            host_port=host_port,
            session_id=SESSION,
            timeout_seconds=0.05,
            max_attempts=3,
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "has not been sent"):
                session.exchange(
                    stage_id=STAGE,
                    transaction_id=TRANSACTION,
                    inner_text="PING",
                )
        finally:
            session.close()
        thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive())

    def test_matching_device_error_is_not_ignored(self):
        device_port = free_udp_port()
        host_port = free_udp_port()

        def server() -> None:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as device:
                device.bind(("127.0.0.1", device_port))
                request_payload, host = device.recvfrom(1024)
                request = decode_wireless_envelope(
                    request_payload, expected_prefix=REQUEST_ENVELOPE_PREFIX
                )
                response = encode_wireless_envelope(
                    prefix=RESPONSE_ENVELOPE_PREFIX,
                    session_id=request.session_id,
                    stage_id=request.stage_id,
                    transaction_id=request.transaction_id,
                    attempt=request.attempt,
                    inner_text="CUKDWERR,TEST_FAILURE",
                )
                device.sendto(response, host)

        thread = threading.Thread(target=server, daemon=True)
        thread.start()
        session = StrictUdpSession(
            device_ip="127.0.0.1",
            device_port=device_port,
            host_port=host_port,
            session_id=SESSION,
            timeout_seconds=0.2,
            max_attempts=1,
        )
        try:
            with self.assertRaisesRegex(DeviceProtocolError, "TEST_FAILURE"):
                session.exchange(
                    stage_id=STAGE,
                    transaction_id=TRANSACTION,
                    inner_text="PING",
                )
        finally:
            session.close()
        thread.join(timeout=1.0)


class NativeWirelessParserTests(unittest.TestCase):
    def test_python_and_c_protocols_have_identical_golden_frames(self):
        compiler = shutil.which("gcc") or shutil.which("cc")
        if compiler is None:
            self.skipTest("No native C compiler is available")
        config = encode_wifi_config_line(
            "CUKD-LAB", "temporary-passphrase", 42101, SESSION
        ).strip()
        malformed_body = f"CUKDWCFG2,{SESSION},042101,43554B442D4C4142,74656D706F726172792D70617373706872617365"
        malformed = (
            f"{malformed_body},{crc16_ccitt(malformed_body.encode('ascii')):04X}"
        )
        request = encode_wireless_envelope(
            prefix=REQUEST_ENVELOPE_PREFIX,
            session_id=SESSION,
            stage_id=STAGE,
            transaction_id=TRANSACTION,
            attempt=2,
            inner_text="CUKD1,42,1",
        ).decode("ascii")
        expected_response = encode_wireless_envelope(
            prefix=RESPONSE_ENVELOPE_PREFIX,
            session_id=SESSION,
            stage_id=STAGE,
            transaction_id=TRANSACTION,
            attempt=2,
            inner_text="CUKD1R,42,OK,0,1,0,0,0,0,1,2,3,ABCD",
        ).decode("ascii")
        harness = r'''
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "cukd_wifi_config.h"
#include "cukd_wifi_envelope.h"

int main(int argc, char **argv) {
    cukd_wifi_config_t config;
    cukd_wifi_envelope_t envelope;
    char response[CUKD_WIFI_DATAGRAM_MAX + 1u];
    const uint8_t config_nul[] = {'A', 0, 'B'};
    const uint8_t envelope_nul[] = {'A', 0, 'B'};
    if (argc != 4) return 90;
    memset(&config, 0, sizeof(config));
    if (cukd_parse_wifi_config_line((const uint8_t *)argv[1], strlen(argv[1]), &config) != CUKD_WIFI_CONFIG_OK) return 1;
    if (strcmp(config.session_id, "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA") != 0) return 2;
    if (strcmp(config.ssid, "CUKD-LAB") != 0) return 3;
    if (strcmp(config.password, "temporary-passphrase") != 0) return 4;
    if (config.udp_port != 42101u) return 5;
    if (cukd_parse_wifi_config_line((const uint8_t *)argv[3], strlen(argv[3]), &config) != CUKD_WIFI_CONFIG_BAD_PORT) return 6;
    if (config.session_id[0] != '\0' || config.ssid[0] != '\0' || config.password[0] != '\0' || config.udp_port != 0u) return 61;
    if (cukd_parse_wifi_config_line(config_nul, sizeof(config_nul), &config) != CUKD_WIFI_CONFIG_BAD_TEXT) return 7;
    if (cukd_parse_wifi_request_envelope((const uint8_t *)argv[2], strlen(argv[2]), &envelope) != CUKD_WIFI_ENVELOPE_OK) return 8;
    if (envelope.attempt != 2u || strcmp(envelope.inner_text, "CUKD1,42,1") != 0) return 9;
    if (cukd_parse_wifi_request_envelope(envelope_nul, sizeof(envelope_nul), &envelope) != CUKD_WIFI_ENVELOPE_BAD_TEXT) return 10;
    if (!cukd_format_wifi_response_envelope(response, sizeof(response), &envelope, "CUKD1R,42,OK,0,1,0,0,0,0,1,2,3,ABCD")) return 11;
    puts(response);
    cukd_clear_wifi_config(&config);
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary = Path(temporary_dir)
            harness_path = temporary / "wireless_harness.c"
            executable = temporary / ("wireless_harness.exe" if sys.platform == "win32" else "wireless_harness")
            harness_path.write_text(harness, encoding="ascii")
            common = ROOT / "deployment" / "wireless_hil" / "firmware" / "common"
            model_common = ROOT / "deployment" / "hardware_hil" / "firmware" / "common"
            compile_result = subprocess.run(
                [
                    compiler,
                    "-std=c11",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-pedantic",
                    f"-I{common}",
                    f"-I{model_common}",
                    str(harness_path),
                    str(common / "cukd_wifi_config.c"),
                    str(common / "cukd_wifi_envelope.c"),
                    str(model_common / "cukd_protocol.c"),
                    "-o",
                    str(executable),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            run_result = subprocess.run(
                [str(executable), config, request, malformed],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(run_result.returncode, 0, run_result.stderr)
            self.assertEqual(run_result.stdout.strip(), expected_response)


class WirelessHostPipelineIntegrationTests(unittest.TestCase):
    def test_full_metric_reconciliation_accepts_exports_and_rejects_tampering(self):
        for student in ["A", "B"]:
            generated = (
                ROOT
                / "deployment"
                / "firmware_export"
                / "wsnds_rfkd_hil"
                / f"generated_fgds_student_{student}_seed42"
            )
            report = json.loads(
                (generated / "strict_export_report.json").read_text(encoding="utf-8")
            )
            with (generated / "hil_reference_predictions.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                references = list(csv.DictReader(handle))
            self.assertEqual(len(references), 56301)
            metrics = compute_classification_metrics(
                [int(row["true_label"]) for row in references],
                [int(row["fixed_pred"]) for row in references],
                range(5),
            )
            metrics["mcu_vs_fp32_agreement"] = sum(
                int(row["fixed_pred"] == row["fp32_pred"]) for row in references
            ) / len(references)
            reconciled = reconcile_full_metrics_with_strict_export(
                metrics,
                report,
                student=f"student_{student}",
                export_id=report["export_id"],
            )
            self.assertEqual(reconciled["status"], "passed")

            tampered = dict(metrics)
            tampered["accuracy"] = float(metrics["accuracy"]) - 0.001
            with self.assertRaisesRegex(RuntimeError, "accuracy"):
                reconcile_full_metrics_with_strict_export(
                    tampered,
                    report,
                    student=f"student_{student}",
                    export_id=report["export_id"],
                )

    def test_real_export_smoke_stage_round_trip_and_verification(self):
        generated = (
            ROOT
            / "deployment"
            / "firmware_export"
            / "wsnds_rfkd_hil"
            / "generated_fgds_student_A_seed42"
        )
        bundle = (
            ROOT
            / "deployment"
            / "wireless_hil"
            / "build"
            / "fgds_student_A_seed42_esp32c3_wifi_udp"
        )
        if not generated.is_dir() or not bundle.is_dir():
            self.skipTest("Real FG-DS export and generated wireless bundle are required")

        export_manifest = verify_export_for_wireless(generated)
        bundle_manifest = verify_wireless_bundle(bundle, export_manifest)
        references: dict[int, dict[str, object]] = {}
        with (generated / "hil_reference_predictions.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            for row in csv.DictReader(handle):
                row_id = int(row["row_id"])
                if row_id >= 10:
                    break
                references[row_id] = {
                    "pred": int(row["fixed_pred"]),
                    "logits": [int(row[f"fixed_logit_{index}"]) for index in range(5)],
                }
        self.assertEqual(set(references), set(range(10)))

        ready = threading.Event()
        server_errors: list[BaseException] = []

        def server() -> None:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as endpoint:
                    endpoint.bind(("127.0.0.1", DEFAULT_DEVICE_UDP_PORT))
                    endpoint.settimeout(10.0)
                    ready.set()
                    active_stage = ""
                    completed = 0
                    received_datagrams = 0
                    while True:
                        payload, host = endpoint.recvfrom(1024)
                        request = decode_wireless_envelope(
                            payload, expected_prefix=REQUEST_ENVELOPE_PREFIX
                        )
                        inner = request.inner_text
                        if inner == "CUKDWID?":
                            response_inner = expected_device_identity(bundle_manifest)
                        elif inner.startswith("CUKDWBEGIN,"):
                            parts = inner.split(",")
                            self.assertEqual(parts[1:], [request.stage_id, "1", "10"])
                            active_stage = request.stage_id
                            completed = 0
                            received_datagrams = 1
                            response_inner = (
                                f"CUKDWBEGINR,{active_stage},1,10,OK"
                            )
                        elif inner.startswith("CUKD1,"):
                            self.assertEqual(request.stage_id, active_stage)
                            received_datagrams += 1
                            row_id = int(inner.split(",", 2)[1])
                            self.assertEqual(row_id, completed)
                            reference = references[row_id]
                            logits = reference["logits"]
                            body = ",".join(
                                [
                                    "CUKD1R",
                                    str(row_id),
                                    "OK",
                                    str(reference["pred"]),
                                    *[str(value) for value in logits],
                                    "7",
                                    "321",
                                    "328",
                                ]
                            )
                            response_inner = (
                                f"{body},{crc16_ccitt(body.encode('ascii')):04X}"
                            )
                            completed += 1
                        elif inner.startswith("CUKDWEND,"):
                            self.assertEqual(request.stage_id, active_stage)
                            self.assertEqual(completed, 10)
                            received_datagrams += 1
                            response_inner = (
                                f"CUKDWENDR,{active_stage},10,10,"
                                f"{received_datagrams},0,0,0,0,0,0,0,0,0,0,10,1,OK"
                            )
                        else:
                            raise AssertionError(f"Unexpected emulator request: {inner}")
                        response = encode_wireless_envelope(
                            prefix=RESPONSE_ENVELOPE_PREFIX,
                            session_id=request.session_id,
                            stage_id=request.stage_id,
                            transaction_id=request.transaction_id,
                            attempt=request.attempt,
                            inner_text=response_inner,
                        )
                        endpoint.sendto(response, host)
                        if inner.startswith("CUKDWEND,"):
                            return
            except BaseException as exc:
                server_errors.append(exc)
                ready.set()

        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary = Path(temporary_dir)
            connection_path = temporary / "connection.json"
            connection = {
                "status": "connected",
                "protocol_id": WIRELESS_PROTOCOL_ID,
                "transport": "wifi_udp",
                "board": bundle_manifest["board"],
                "student": bundle_manifest["student"],
                "device_ip": "127.0.0.1",
                "device_udp_port": DEFAULT_DEVICE_UDP_PORT,
                "host_udp_port": DEFAULT_HOST_UDP_PORT,
                "session_id": SESSION,
                "serial_closed_before_udp_replay": True,
                "device_identity": expected_device_identity(bundle_manifest),
                "export_id": bundle_manifest["export_id"],
                "wireless_bundle_id": bundle_manifest["wireless_bundle_id"],
                "strict_export_manifest_sha256": sha256_file(
                    generated / "strict_export_manifest.json"
                ),
                "wireless_bundle_manifest_sha256": bundle_manifest[
                    "_manifest_sha256"
                ],
                "provisioning_script_sha256": sha256_file(
                    ROOT
                    / "deployment"
                    / "wireless_hil"
                    / "host"
                    / "configure_wifi_serial.py"
                ),
            }
            connection_path.write_text(
                json.dumps(connection, indent=2) + "\n", encoding="utf-8"
            )
            thread = threading.Thread(target=server, daemon=True)
            thread.start()
            self.assertTrue(ready.wait(timeout=2.0))
            if server_errors:
                raise server_errors[0]
            output_csv = temporary / "smoke_10_mcu.csv"
            sequence_json = temporary / "smoke_10_sequence.json"
            metrics_json = temporary / "smoke_10_metrics.json"
            stream_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "deployment.wireless_hil.host.stream_vectors_udp",
                    "--generated-dir",
                    str(generated),
                    "--bundle-dir",
                    str(bundle),
                    "--vectors-csv",
                    str(generated / "hil_replay_vectors.csv"),
                    "--connection-json",
                    str(connection_path),
                    "--stage-name",
                    "smoke_10",
                    "--output-csv",
                    str(output_csv),
                    "--summary-json",
                    str(sequence_json),
                    "--timeout",
                    "0.2",
                    "--max-attempts",
                    "2",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(stream_result.returncode, 0, stream_result.stderr)
            thread.join(timeout=2.0)
            if server_errors:
                raise server_errors[0]
            self.assertFalse(thread.is_alive())
            verify_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "deployment.wireless_hil.host.verify_results_udp",
                    "--mcu-csv",
                    str(output_csv),
                    "--sequence-json",
                    str(sequence_json),
                    "--connection-json",
                    str(connection_path),
                    "--generated-dir",
                    str(generated),
                    "--bundle-dir",
                    str(bundle),
                    "--reference-csv",
                    str(generated / "hil_reference_predictions.csv"),
                    "--stage-name",
                    "smoke_10",
                    "--output-json",
                    str(metrics_json),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(verify_result.returncode, 0, verify_result.stderr)
            metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
            self.assertEqual(metrics["status"], "passed")
            self.assertEqual(metrics["completed_vectors"], 10)
            self.assertEqual(metrics["mcu_vs_fixed_reference_agreement"], 1.0)
            self.assertEqual(metrics["exact_logit_agreement"], 1.0)


class WirelessFirmwareStaticTests(unittest.TestCase):
    def test_timed_inference_includes_prediction_argmax(self):
        sketch = (
            ROOT
            / "deployment"
            / "wireless_hil"
            / "firmware"
            / "cukd_wireless_fgds"
            / "cukd_wireless_fgds.ino"
        ).read_text(encoding="utf-8")
        inference_body = sketch[
            sketch.index("cukd_forward_q15(input_q15, logits)") : sketch.index(
                "if (!cukd_format_response_line"
            )
        ]
        self.assertLess(
            inference_body.index("cukd_forward_q15(input_q15, logits)"),
            inference_body.index("cukd_argmax_logits(logits)"),
        )
        self.assertLess(
            inference_body.index("cukd_argmax_logits(logits)"),
            inference_body.index("const uint32_t inference_end = cukd_now_us()"),
        )

    def test_firmware_disables_esp_persistence_before_wifi_initialization(self):
        sketch = (
            ROOT
            / "deployment"
            / "wireless_hil"
            / "firmware"
            / "cukd_wireless_fgds"
            / "cukd_wireless_fgds.ino"
        ).read_text(encoding="utf-8")
        setup_body = sketch[sketch.index("void setup()") : sketch.index("void loop()")]
        persistence = setup_body.index("WiFi.persistent(false)")
        self.assertLess(persistence, setup_body.index("WiFi.mode(WIFI_STA)"))
        self.assertNotIn("WiFi.begin", setup_body)
        for forbidden in ["MQTT", "Bluetooth", "BLEDevice", "CUKD_WIFI_PASSWORD"]:
            self.assertNotIn(forbidden, sketch)
        for required in [
            "cukd_parse_wifi_request_envelope",
            "cukd_endpoint_bound",
            "cukd_cache_response",
            "cukd_is_unicast_ipv4",
            "CUKD_WIFI_DHCP_TIMEOUT_MS",
            '"DHCP_FAILED"',
            "cukd_secure_zero_buffer(cukd_serial_line",
            "cukd_aborted_stage_valid",
            "cukd_aborted_stage_response",
            "CUKDWBEGIN",
            "CUKDWEND",
            "CUKDWABORT",
            "duplicate_replays",
        ]:
            self.assertIn(required, sketch)

        connect_body = sketch[
            sketch.index("static bool cukd_connect_wifi") : sketch.index(
                "static void cukd_process_serial_line"
            )
        ]
        address_capture = connect_body.index(
            "IPAddress local_address = WiFi.localIP()"
        )
        dhcp_failure = connect_body.index('"DHCP_FAILED"')
        udp_bind = connect_body.index("cukd_udp.begin")
        self.assertLess(address_capture, dhcp_failure)
        self.assertLess(dhcp_failure, udp_bind)
        self.assertIn('"OK",\n        local_address,', connect_body)


if __name__ == "__main__":
    unittest.main()
