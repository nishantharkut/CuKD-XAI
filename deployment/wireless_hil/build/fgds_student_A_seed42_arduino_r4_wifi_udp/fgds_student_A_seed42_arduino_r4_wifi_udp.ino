#include <Arduino.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cukd_export_identity.h"
#include "cukd_wireless_bundle_identity.h"

#if defined(CUKD_WIRELESS_BOARD_ESP32C3)
#include <WiFi.h>
#include <WiFiUdp.h>
#elif defined(CUKD_WIRELESS_BOARD_ARDUINO_R4)
#include <WiFiS3.h>
#else
#error "A supported CUKD wireless board must be selected"
#endif

extern "C" {
#include "cukd_model.h"
#include "cukd_preprocess.h"
#include "cukd_protocol.h"
#include "cukd_wifi_config.h"
#include "cukd_wifi_envelope.h"
}

#define CUKD_WIFI_STAGE_ORDINAL_MAX 255u
#define CUKD_WIFI_STAGE_ROW_MAX 56301u
#define CUKD_WIFI_CONNECT_TIMEOUT_MS 30000u
#define CUKD_WIFI_DHCP_TIMEOUT_MS 15000u
#define CUKD_WIFI_IDENTITY_TX "FFFFFFFFFFFFFFFC"
#define CUKD_WIFI_ABORT_TX "FFFFFFFFFFFFFFFD"
#define CUKD_WIFI_BEGIN_TX "FFFFFFFFFFFFFFFE"
#define CUKD_WIFI_END_TX "FFFFFFFFFFFFFFFF"

typedef struct {
    uint32_t received_datagrams;
    uint32_t oversized_datagrams;
    uint32_t short_reads;
    uint32_t bad_envelopes;
    uint32_t wrong_sessions;
    uint32_t wrong_endpoints;
    uint32_t wrong_stages;
    uint32_t control_errors;
    uint32_t data_errors;
    uint32_t duplicate_replays;
    uint32_t stale_transactions;
    uint32_t inferences;
} cukd_stage_counters_t;

static WiFiUDP cukd_udp;
static bool cukd_udp_started = false;
static char cukd_session_id[CUKD_WIFI_SESSION_LENGTH + 1u];

static bool cukd_endpoint_bound = false;
static IPAddress cukd_bound_address;
static uint16_t cukd_bound_port = 0u;

static bool cukd_stage_active = false;
static char cukd_stage_id[CUKD_WIFI_STAGE_LENGTH + 1u];
static uint32_t cukd_stage_ordinal = 0u;
static uint32_t cukd_stage_expected = 0u;
static uint32_t cukd_next_row_id = 0u;
static cukd_stage_counters_t cukd_stage_counters;

static bool cukd_cache_valid = false;
static char cukd_cache_transaction[CUKD_WIFI_TRANSACTION_LENGTH + 1u];
static char cukd_cache_request[CUKD_HIL_LINE_MAX];
static char cukd_cache_response[CUKD_HIL_LINE_MAX];

static bool cukd_ended_stage_valid = false;
static char cukd_ended_stage_id[CUKD_WIFI_STAGE_LENGTH + 1u];
static char cukd_ended_stage_response[CUKD_HIL_LINE_MAX];

static bool cukd_aborted_stage_valid = false;
static char cukd_aborted_stage_id[CUKD_WIFI_STAGE_LENGTH + 1u];
static uint32_t cukd_aborted_stage_completed = 0u;
static char cukd_aborted_stage_response[CUKD_HIL_LINE_MAX];

static char cukd_serial_line[CUKD_WIFI_CONFIG_LINE_MAX];
static size_t cukd_serial_length = 0u;
static bool cukd_serial_overflow = false;
static uint8_t cukd_udp_datagram[CUKD_WIRELESS_MAX_DATAGRAM];
static char cukd_udp_response[CUKD_WIRELESS_MAX_DATAGRAM + 1u];

static uint32_t cukd_now_us() {
    return (uint32_t)micros();
}

static void cukd_secure_zero_buffer(void *address, size_t length) {
    volatile uint8_t *cursor = (volatile uint8_t *)address;
    while (length-- > 0u) {
        *cursor++ = 0u;
    }
}

static bool cukd_same_ip(const IPAddress &left, const IPAddress &right) {
    return left[0] == right[0] && left[1] == right[1] &&
           left[2] == right[2] && left[3] == right[3];
}

static uint8_t cukd_argmax_logits(const int16_t logits[CUKD_OUTPUT_DIM]) {
    uint8_t best = 0u;
    for (uint8_t index = 1u; index < CUKD_OUTPUT_DIM; ++index) {
        if (logits[index] > logits[best]) {
            best = index;
        }
    }
    return best;
}

static void cukd_write_crc_serial_response(const char *body) {
    const uint16_t checksum = cukd_crc16_ccitt(
        (const uint8_t *)body,
        strlen(body)
    );
    char suffix[7];
    Serial.print(body);
    if (snprintf(suffix, sizeof(suffix), ",%04X", (unsigned int)checksum) == 5) {
        Serial.println(suffix);
    }
}

static bool cukd_ascii_to_hex(
    const char *text,
    char *output,
    size_t output_size
) {
    static const char digits[] = "0123456789ABCDEF";
    const size_t length = text == NULL ? 0u : strlen(text);
    if (length == 0u || length > (output_size - 1u) / 2u) {
        return false;
    }
    for (size_t index = 0u; index < length; ++index) {
        const uint8_t value = (uint8_t)text[index];
        if (value < 0x20u || value > 0x7eu) {
            return false;
        }
        output[2u * index] = digits[value >> 4];
        output[2u * index + 1u] = digits[value & 0x0fu];
    }
    output[2u * length] = '\0';
    return true;
}

static const char *cukd_connectivity_firmware() {
#if defined(CUKD_WIRELESS_BOARD_ESP32C3)
    return ESP.getSdkVersion();
#else
    return WiFi.firmwareVersion();
#endif
}

static bool cukd_is_unicast_ipv4(const IPAddress &address) {
    const uint8_t first = address[0];
    const bool all_zero =
        first == 0u && address[1] == 0u && address[2] == 0u && address[3] == 0u;
    return !all_zero && first != 0u && first != 127u && first < 224u;
}

static void cukd_format_mac(char output[18]) {
    uint8_t mac[6] = {0u, 0u, 0u, 0u, 0u, 0u};
    WiFi.macAddress(mac);
    (void)snprintf(
        output,
        18u,
        "%02X:%02X:%02X:%02X:%02X:%02X",
        (unsigned int)mac[0],
        (unsigned int)mac[1],
        (unsigned int)mac[2],
        (unsigned int)mac[3],
        (unsigned int)mac[4],
        (unsigned int)mac[5]
    );
}

static void cukd_send_config_response(
    const char *session_id,
    const char *status,
    const IPAddress &address,
    uint16_t port,
    int32_t rssi_dbm
) {
    char body[CUKD_HIL_LINE_MAX];
    char firmware_hex[129];
    char mac[18];
    const char *firmware = cukd_connectivity_firmware();
    if (!cukd_ascii_to_hex(firmware, firmware_hex, sizeof(firmware_hex))) {
        (void)strcpy(firmware_hex, "4E41");
    }
    cukd_format_mac(mac);
    const int count = snprintf(
        body,
        sizeof(body),
        "CUKDWCFG2R,%s,%s,%u.%u.%u.%u,%u,%ld,%s,%s",
        session_id,
        status,
        (unsigned int)address[0],
        (unsigned int)address[1],
        (unsigned int)address[2],
        (unsigned int)address[3],
        (unsigned int)port,
        (long)rssi_dbm,
        firmware_hex,
        mac
    );
    if (count > 0 && (size_t)count < sizeof(body)) {
        cukd_write_crc_serial_response(body);
    }
}

static void cukd_send_build_identity_serial() {
    Serial.print("CUKDWBUILD,");
    Serial.print(CUKD_STUDENT_ID);
    Serial.print(",");
    Serial.print(CUKD_EXPORT_ID);
    Serial.print(",");
    Serial.print(CUKD_WIRELESS_BUNDLE_ID);
    Serial.print(",");
    Serial.print(CUKD_WIRELESS_BOARD_ID);
    Serial.print(",");
    Serial.println(CUKD_WIRELESS_PROTOCOL_ID);
}

static void cukd_reset_session_state() {
    cukd_endpoint_bound = false;
    cukd_bound_port = 0u;
    cukd_stage_active = false;
    cukd_stage_id[0] = '\0';
    cukd_stage_ordinal = 0u;
    cukd_stage_expected = 0u;
    cukd_next_row_id = 0u;
    memset(&cukd_stage_counters, 0, sizeof(cukd_stage_counters));
    cukd_cache_valid = false;
    cukd_ended_stage_valid = false;
    cukd_aborted_stage_valid = false;
}

static bool cukd_connect_wifi(cukd_wifi_config_t *config) {
    const IPAddress zero_address(0u, 0u, 0u, 0u);
    if (config->udp_port != CUKD_WIRELESS_UDP_PORT) {
        cukd_send_config_response(
            config->session_id,
            "BAD_PORT",
            zero_address,
            0u,
            0
        );
        return false;
    }
    if (cukd_udp_started) {
        cukd_udp.stop();
        cukd_udp_started = false;
    }
#if defined(CUKD_WIRELESS_BOARD_ESP32C3)
    WiFi.disconnect(true, true);
    delay(100);
#else
    WiFi.disconnect();
#endif
    (void)WiFi.begin(config->ssid, config->password);
    const uint32_t started = millis();
    while (
        WiFi.status() != WL_CONNECTED &&
        millis() - started < CUKD_WIFI_CONNECT_TIMEOUT_MS
    ) {
        delay(100);
    }
    if (WiFi.status() != WL_CONNECTED) {
        cukd_send_config_response(
            config->session_id,
            "CONNECT_FAILED",
            zero_address,
            0u,
            0
        );
        return false;
    }
    IPAddress local_address = WiFi.localIP();
    const uint32_t dhcp_started = millis();
    while (
        WiFi.status() == WL_CONNECTED &&
        !cukd_is_unicast_ipv4(local_address) &&
        millis() - dhcp_started < CUKD_WIFI_DHCP_TIMEOUT_MS
    ) {
        delay(100);
        local_address = WiFi.localIP();
    }
    if (WiFi.status() != WL_CONNECTED) {
        cukd_send_config_response(
            config->session_id,
            "CONNECT_FAILED",
            zero_address,
            0u,
            0
        );
        return false;
    }
    if (!cukd_is_unicast_ipv4(local_address)) {
        cukd_send_config_response(
            config->session_id,
            "DHCP_FAILED",
            zero_address,
            0u,
            (int32_t)WiFi.RSSI()
        );
        WiFi.disconnect();
        return false;
    }
    if (cukd_udp.begin(CUKD_WIRELESS_UDP_PORT) != 1u) {
        cukd_send_config_response(
            config->session_id,
            "UDP_BIND_FAILED",
            local_address,
            0u,
            (int32_t)WiFi.RSSI()
        );
        WiFi.disconnect();
        return false;
    }
    memcpy(
        cukd_session_id,
        config->session_id,
        CUKD_WIFI_SESSION_LENGTH + 1u
    );
    cukd_udp_started = true;
    cukd_reset_session_state();
    cukd_send_config_response(
        cukd_session_id,
        "OK",
        local_address,
        CUKD_WIRELESS_UDP_PORT,
        (int32_t)WiFi.RSSI()
    );
    return true;
}

static void cukd_process_serial_line(const uint8_t *data, size_t length) {
    static const char zero_session[] = "00000000000000000000000000000000";
    const IPAddress zero_address(0u, 0u, 0u, 0u);
    if (
        length == strlen("CUKDWID?") &&
        memcmp(data, "CUKDWID?", length) == 0
    ) {
        cukd_send_build_identity_serial();
        return;
    }
    cukd_wifi_config_t parsed;
    memset(&parsed, 0, sizeof(parsed));
    const cukd_wifi_config_status_t status = cukd_parse_wifi_config_line(
        data,
        length,
        &parsed
    );
    if (status != CUKD_WIFI_CONFIG_OK) {
        cukd_send_config_response(
            zero_session,
            cukd_wifi_config_status_name(status),
            zero_address,
            0u,
            0
        );
        cukd_clear_wifi_config(&parsed);
        return;
    }
    (void)cukd_connect_wifi(&parsed);
    cukd_clear_wifi_config(&parsed);
}

static void cukd_poll_serial() {
    while (Serial.available() > 0) {
        const int incoming = Serial.read();
        if (incoming < 0) {
            return;
        }
        const uint8_t value = (uint8_t)incoming;
        if (value == '\n') {
            if (cukd_serial_overflow) {
                static const char zero_session[] = "00000000000000000000000000000000";
                cukd_send_config_response(
                    zero_session,
                    "BAD_LENGTH",
                    IPAddress(0u, 0u, 0u, 0u),
                    0u,
                    0
                );
            } else if (cukd_serial_length > 0u) {
                cukd_process_serial_line(
                    (const uint8_t *)cukd_serial_line,
                    cukd_serial_length
                );
            }
            cukd_secure_zero_buffer(cukd_serial_line, sizeof(cukd_serial_line));
            cukd_serial_length = 0u;
            cukd_serial_overflow = false;
        } else if (value == '\r') {
            continue;
        } else if (value < 0x20u || value > 0x7eu) {
            cukd_secure_zero_buffer(cukd_serial_line, sizeof(cukd_serial_line));
            cukd_serial_length = 0u;
            cukd_serial_overflow = true;
        } else if (!cukd_serial_overflow) {
            if (cukd_serial_length < sizeof(cukd_serial_line)) {
                cukd_serial_line[cukd_serial_length++] = (char)value;
            } else {
                cukd_secure_zero_buffer(cukd_serial_line, sizeof(cukd_serial_line));
                cukd_serial_length = 0u;
                cukd_serial_overflow = true;
            }
        }
    }
}

static void cukd_drain_udp_packet() {
    while (cukd_udp.available() > 0) {
        (void)cukd_udp.read();
    }
}

static bool cukd_send_udp_inner(
    const IPAddress &address,
    uint16_t port,
    const cukd_wifi_envelope_t *request,
    const char *inner_response
) {
    if (!cukd_format_wifi_response_envelope(
            cukd_udp_response,
            sizeof(cukd_udp_response),
            request,
            inner_response)) {
        return false;
    }
    const size_t length = strlen(cukd_udp_response);
    if (cukd_udp.beginPacket(address, port) != 1) {
        return false;
    }
    const size_t written = cukd_udp.write(
        (const uint8_t *)cukd_udp_response,
        length
    );
    return written == length && cukd_udp.endPacket() == 1;
}

static void cukd_send_udp_error(
    const IPAddress &address,
    uint16_t port,
    const cukd_wifi_envelope_t *request,
    const char *code
) {
    char response[64];
    const int count = snprintf(response, sizeof(response), "CUKDWERR,%s", code);
    if (count > 0 && (size_t)count < sizeof(response)) {
        (void)cukd_send_udp_inner(address, port, request, response);
    }
}

static size_t cukd_split_fields(
    char *text,
    char **fields,
    size_t maximum_fields
) {
    size_t count = 0u;
    char *start = text;
    for (char *cursor = text;; ++cursor) {
        if (*cursor == ',' || *cursor == '\0') {
            if (cursor == start || count >= maximum_fields) {
                return 0u;
            }
            fields[count++] = start;
            if (*cursor == '\0') {
                return count;
            }
            *cursor = '\0';
            start = cursor + 1;
        }
    }
}

static bool cukd_parse_canonical_u32(
    const char *text,
    uint32_t minimum,
    uint32_t maximum,
    uint32_t *output
) {
    char *end = NULL;
    char canonical[11];
    const unsigned long value = strtoul(text, &end, 10);
    const int count = snprintf(canonical, sizeof(canonical), "%lu", value);
    if (
        end == text || *end != '\0' || value < minimum || value > maximum ||
        count <= 0 || (size_t)count >= sizeof(canonical) ||
        strcmp(text, canonical) != 0
    ) {
        return false;
    }
    *output = (uint32_t)value;
    return true;
}

static void cukd_send_identity(
    const IPAddress &address,
    uint16_t port,
    const cukd_wifi_envelope_t *request
) {
    char identity[CUKD_HIL_LINE_MAX];
    if (strcmp(request->transaction_id, CUKD_WIFI_IDENTITY_TX) != 0) {
        cukd_send_udp_error(address, port, request, "BAD_IDENTITY_TX");
        return;
    }
    const int count = snprintf(
        identity,
        sizeof(identity),
        "CUKDWBUILD,%s,%s,%s,%s,%s",
        CUKD_STUDENT_ID,
        CUKD_EXPORT_ID,
        CUKD_WIRELESS_BUNDLE_ID,
        CUKD_WIRELESS_BOARD_ID,
        CUKD_WIRELESS_PROTOCOL_ID
    );
    if (count > 0 && (size_t)count < sizeof(identity)) {
        (void)cukd_send_udp_inner(address, port, request, identity);
    }
}

static void cukd_begin_stage(
    const IPAddress &address,
    uint16_t port,
    const cukd_wifi_envelope_t *request
) {
    char buffer[CUKD_HIL_LINE_MAX];
    char *fields[4];
    uint32_t ordinal;
    uint32_t expected;
    if (strcmp(request->transaction_id, CUKD_WIFI_BEGIN_TX) != 0) {
        ++cukd_stage_counters.control_errors;
        cukd_send_udp_error(address, port, request, "BAD_BEGIN_TX");
        return;
    }
    (void)strcpy(buffer, request->inner_text);
    if (
        cukd_split_fields(buffer, fields, 4u) != 4u ||
        strcmp(fields[0], "CUKDWBEGIN") != 0 ||
        strcmp(fields[1], request->stage_id) != 0 ||
        !cukd_parse_canonical_u32(
            fields[2], 1u, CUKD_WIFI_STAGE_ORDINAL_MAX, &ordinal) ||
        !cukd_parse_canonical_u32(
            fields[3], 1u, CUKD_WIFI_STAGE_ROW_MAX, &expected)
    ) {
        ++cukd_stage_counters.control_errors;
        cukd_send_udp_error(address, port, request, "BAD_BEGIN");
        return;
    }
    if (cukd_stage_active) {
        if (
            strcmp(cukd_stage_id, request->stage_id) != 0 ||
            ordinal != cukd_stage_ordinal || expected != cukd_stage_expected
        ) {
            ++cukd_stage_counters.control_errors;
            cukd_send_udp_error(address, port, request, "STAGE_ACTIVE");
            return;
        }
    } else {
        if (ordinal <= cukd_stage_ordinal) {
            ++cukd_stage_counters.control_errors;
            cukd_send_udp_error(address, port, request, "STAGE_REPLAY");
            return;
        }
        memcpy(cukd_stage_id, request->stage_id, CUKD_WIFI_STAGE_LENGTH + 1u);
        cukd_stage_ordinal = ordinal;
        cukd_stage_expected = expected;
        cukd_next_row_id = 0u;
        cukd_stage_active = true;
        cukd_cache_valid = false;
        cukd_aborted_stage_valid = false;
        memset(&cukd_stage_counters, 0, sizeof(cukd_stage_counters));
        cukd_stage_counters.received_datagrams = 1u;
    }
    char response[96];
    const int count = snprintf(
        response,
        sizeof(response),
        "CUKDWBEGINR,%s,%lu,%lu,OK",
        cukd_stage_id,
        (unsigned long)cukd_stage_ordinal,
        (unsigned long)cukd_stage_expected
    );
    if (count > 0 && (size_t)count < sizeof(response)) {
        (void)cukd_send_udp_inner(address, port, request, response);
    }
}

static bool cukd_parse_stage_count_control(
    const char *inner_text,
    const char *prefix,
    const char *stage_id,
    uint32_t *completed
) {
    char buffer[CUKD_HIL_LINE_MAX];
    char *fields[3];
    (void)strcpy(buffer, inner_text);
    return cukd_split_fields(buffer, fields, 3u) == 3u &&
           strcmp(fields[0], prefix) == 0 &&
           strcmp(fields[1], stage_id) == 0 &&
           cukd_parse_canonical_u32(
               fields[2], 0u, CUKD_WIFI_STAGE_ROW_MAX, completed);
}

static void cukd_finish_stage(
    const IPAddress &address,
    uint16_t port,
    const cukd_wifi_envelope_t *request
) {
    uint32_t completed;
    if (
        !cukd_stage_active && cukd_ended_stage_valid &&
        strcmp(request->stage_id, cukd_ended_stage_id) == 0 &&
        strcmp(request->transaction_id, CUKD_WIFI_END_TX) == 0
    ) {
        (void)cukd_send_udp_inner(
            address,
            port,
            request,
            cukd_ended_stage_response
        );
        return;
    }
    if (
        strcmp(request->transaction_id, CUKD_WIFI_END_TX) != 0 ||
        !cukd_stage_active ||
        strcmp(request->stage_id, cukd_stage_id) != 0 ||
        !cukd_parse_stage_count_control(
            request->inner_text, "CUKDWEND", cukd_stage_id, &completed) ||
        completed != cukd_next_row_id ||
        completed != cukd_stage_expected ||
        cukd_stage_counters.inferences != cukd_stage_expected
    ) {
        ++cukd_stage_counters.control_errors;
        cukd_send_udp_error(address, port, request, "BAD_END");
        return;
    }
    const int count = snprintf(
        cukd_ended_stage_response,
        sizeof(cukd_ended_stage_response),
        "CUKDWENDR,%s,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,OK",
        cukd_stage_id,
        (unsigned long)completed,
        (unsigned long)cukd_stage_expected,
        (unsigned long)cukd_stage_counters.received_datagrams,
        (unsigned long)cukd_stage_counters.oversized_datagrams,
        (unsigned long)cukd_stage_counters.short_reads,
        (unsigned long)cukd_stage_counters.bad_envelopes,
        (unsigned long)cukd_stage_counters.wrong_sessions,
        (unsigned long)cukd_stage_counters.wrong_endpoints,
        (unsigned long)cukd_stage_counters.wrong_stages,
        (unsigned long)cukd_stage_counters.control_errors,
        (unsigned long)cukd_stage_counters.data_errors,
        (unsigned long)cukd_stage_counters.duplicate_replays,
        (unsigned long)cukd_stage_counters.stale_transactions,
        (unsigned long)cukd_stage_counters.inferences,
        (unsigned long)cukd_stage_ordinal
    );
    if (count <= 0 || (size_t)count >= sizeof(cukd_ended_stage_response)) {
        cukd_send_udp_error(address, port, request, "END_FORMAT");
        return;
    }
    memcpy(
        cukd_ended_stage_id,
        cukd_stage_id,
        CUKD_WIFI_STAGE_LENGTH + 1u
    );
    cukd_ended_stage_valid = true;
    cukd_stage_active = false;
    (void)cukd_send_udp_inner(
        address,
        port,
        request,
        cukd_ended_stage_response
    );
}

static void cukd_abort_stage(
    const IPAddress &address,
    uint16_t port,
    const cukd_wifi_envelope_t *request
) {
    uint32_t completed;
    if (
        !cukd_stage_active && cukd_aborted_stage_valid &&
        strcmp(request->stage_id, cukd_aborted_stage_id) == 0 &&
        strcmp(request->transaction_id, CUKD_WIFI_ABORT_TX) == 0 &&
        cukd_parse_stage_count_control(
            request->inner_text,
            "CUKDWABORT",
            cukd_aborted_stage_id,
            &completed) &&
        completed == cukd_aborted_stage_completed
    ) {
        (void)cukd_send_udp_inner(
            address,
            port,
            request,
            cukd_aborted_stage_response
        );
        return;
    }
    if (
        strcmp(request->transaction_id, CUKD_WIFI_ABORT_TX) != 0 ||
        !cukd_stage_active ||
        strcmp(request->stage_id, cukd_stage_id) != 0 ||
        !cukd_parse_stage_count_control(
            request->inner_text, "CUKDWABORT", cukd_stage_id, &completed) ||
        completed != cukd_next_row_id
    ) {
        ++cukd_stage_counters.control_errors;
        cukd_send_udp_error(address, port, request, "BAD_ABORT");
        return;
    }
    const int count = snprintf(
        cukd_aborted_stage_response,
        sizeof(cukd_aborted_stage_response),
        "CUKDWABORTR,%s,%lu,%lu,%lu,OK",
        cukd_stage_id,
        (unsigned long)completed,
        (unsigned long)cukd_stage_expected,
        (unsigned long)cukd_stage_counters.inferences
    );
    if (
        count <= 0 ||
        (size_t)count >= sizeof(cukd_aborted_stage_response)
    ) {
        cukd_send_udp_error(address, port, request, "ABORT_FORMAT");
        return;
    }
    memcpy(
        cukd_aborted_stage_id,
        cukd_stage_id,
        CUKD_WIFI_STAGE_LENGTH + 1u
    );
    cukd_aborted_stage_completed = completed;
    cukd_aborted_stage_valid = true;
    cukd_stage_active = false;
    cukd_cache_valid = false;
    (void)cukd_send_udp_inner(
        address,
        port,
        request,
        cukd_aborted_stage_response
    );
}

static void cukd_process_data_request(
    const IPAddress &address,
    uint16_t port,
    const cukd_wifi_envelope_t *envelope
) {
    if (!cukd_stage_active || strcmp(envelope->stage_id, cukd_stage_id) != 0) {
        ++cukd_stage_counters.wrong_stages;
        cukd_send_udp_error(address, port, envelope, "WRONG_STAGE");
        return;
    }
    if (cukd_cache_valid && strcmp(
            envelope->transaction_id,
            cukd_cache_transaction) == 0) {
        if (strcmp(envelope->inner_text, cukd_cache_request) != 0) {
            ++cukd_stage_counters.data_errors;
            cukd_send_udp_error(address, port, envelope, "TX_CONFLICT");
            return;
        }
        ++cukd_stage_counters.duplicate_replays;
        (void)cukd_send_udp_inner(
            address,
            port,
            envelope,
            cukd_cache_response
        );
        return;
    }

    cukd_request_t request;
    const cukd_status_t parse_status = cukd_parse_request_line(
        envelope->inner_text,
        &request
    );
    if (parse_status != CUKD_STATUS_OK) {
        ++cukd_stage_counters.data_errors;
        cukd_send_udp_error(address, port, envelope, cukd_status_name(parse_status));
        return;
    }
    char expected_transaction[CUKD_WIFI_TRANSACTION_LENGTH + 1u];
    (void)snprintf(
        expected_transaction,
        sizeof(expected_transaction),
        "%016lX",
        (unsigned long)request.row_id
    );
    if (strcmp(envelope->transaction_id, expected_transaction) != 0) {
        ++cukd_stage_counters.data_errors;
        cukd_send_udp_error(address, port, envelope, "ROW_TX_MISMATCH");
        return;
    }
    if (request.row_id < cukd_next_row_id) {
        ++cukd_stage_counters.stale_transactions;
        cukd_send_udp_error(address, port, envelope, "STALE_TRANSACTION");
        return;
    }
    if (
        request.row_id != cukd_next_row_id ||
        request.row_id >= cukd_stage_expected
    ) {
        ++cukd_stage_counters.data_errors;
        cukd_send_udp_error(address, port, envelope, "ROW_SEQUENCE");
        return;
    }

    int16_t input_q15[CUKD_INPUT_DIM];
    int16_t logits[CUKD_OUTPUT_DIM] = {0, 0, 0, 0, 0};
    const uint32_t preprocess_start = cukd_now_us();
    cukd_standardize_raw_q(request.features, input_q15);
    const uint32_t preprocess_end = cukd_now_us();
    cukd_forward_q15(input_q15, logits);
    const uint32_t inference_end = cukd_now_us();
    const uint8_t prediction = cukd_argmax_logits(logits);
    if (!cukd_format_response_line(
            cukd_cache_response,
            sizeof(cukd_cache_response),
            request.row_id,
            CUKD_STATUS_OK,
            (int16_t)prediction,
            logits,
            preprocess_end - preprocess_start,
            inference_end - preprocess_end,
            inference_end - preprocess_start)) {
        ++cukd_stage_counters.data_errors;
        cukd_send_udp_error(address, port, envelope, "INNER_FORMAT");
        return;
    }
    const size_t response_length = strlen(cukd_cache_response);
    if (
        response_length == 0u ||
        cukd_cache_response[response_length - 1u] != '\n'
    ) {
        ++cukd_stage_counters.data_errors;
        cukd_send_udp_error(address, port, envelope, "INNER_TERMINATOR");
        return;
    }
    cukd_cache_response[response_length - 1u] = '\0';
    (void)strcpy(cukd_cache_request, envelope->inner_text);
    memcpy(
        cukd_cache_transaction,
        envelope->transaction_id,
        CUKD_WIFI_TRANSACTION_LENGTH + 1u
    );
    cukd_cache_valid = true;
    ++cukd_next_row_id;
    ++cukd_stage_counters.inferences;
    (void)cukd_send_udp_inner(
        address,
        port,
        envelope,
        cukd_cache_response
    );
}

static void cukd_process_udp_packet() {
    const int packet_size = cukd_udp.parsePacket();
    if (packet_size <= 0) {
        return;
    }
    const IPAddress remote_address = cukd_udp.remoteIP();
    const uint16_t remote_port = cukd_udp.remotePort();
    if (cukd_stage_active) {
        ++cukd_stage_counters.received_datagrams;
    }
    if ((size_t)packet_size > sizeof(cukd_udp_datagram)) {
        if (cukd_stage_active) {
            ++cukd_stage_counters.oversized_datagrams;
        }
        cukd_drain_udp_packet();
        return;
    }
    const int received = cukd_udp.read(
        cukd_udp_datagram,
        (size_t)packet_size
    );
    if (received != packet_size || cukd_udp.available() > 0) {
        if (cukd_stage_active) {
            ++cukd_stage_counters.short_reads;
        }
        cukd_drain_udp_packet();
        return;
    }
    cukd_wifi_envelope_t envelope;
    const cukd_wifi_envelope_status_t envelope_status =
        cukd_parse_wifi_request_envelope(
            cukd_udp_datagram,
            (size_t)received,
            &envelope
        );
    if (envelope_status != CUKD_WIFI_ENVELOPE_OK) {
        if (cukd_stage_active) {
            ++cukd_stage_counters.bad_envelopes;
        }
        return;
    }
    if (strcmp(envelope.session_id, cukd_session_id) != 0) {
        if (cukd_stage_active) {
            ++cukd_stage_counters.wrong_sessions;
        }
        return;
    }
    if (!cukd_endpoint_bound) {
        cukd_bound_address = remote_address;
        cukd_bound_port = remote_port;
        cukd_endpoint_bound = true;
    } else if (
        !cukd_same_ip(remote_address, cukd_bound_address) ||
        remote_port != cukd_bound_port
    ) {
        if (cukd_stage_active) {
            ++cukd_stage_counters.wrong_endpoints;
        }
        return;
    }

    if (strcmp(envelope.inner_text, "CUKDWID?") == 0) {
        cukd_send_identity(remote_address, remote_port, &envelope);
    } else if (strncmp(envelope.inner_text, "CUKDWBEGIN,", 11u) == 0) {
        cukd_begin_stage(remote_address, remote_port, &envelope);
    } else if (strncmp(envelope.inner_text, "CUKDWEND,", 9u) == 0) {
        cukd_finish_stage(remote_address, remote_port, &envelope);
    } else if (strncmp(envelope.inner_text, "CUKDWABORT,", 11u) == 0) {
        cukd_abort_stage(remote_address, remote_port, &envelope);
    } else {
        cukd_process_data_request(remote_address, remote_port, &envelope);
    }
}

void setup() {
    Serial.begin(115200);
    cukd_serial_length = 0u;
    cukd_secure_zero_buffer(cukd_serial_line, sizeof(cukd_serial_line));
    cukd_session_id[0] = '\0';
#if defined(CUKD_WIRELESS_BOARD_ESP32C3)
    WiFi.persistent(false);
    WiFi.mode(WIFI_STA);
#endif
}

void loop() {
    cukd_poll_serial();
    if (cukd_udp_started) {
        cukd_process_udp_packet();
    }
}
