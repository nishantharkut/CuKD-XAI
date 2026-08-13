#include "cukd_wifi_config.h"

#include "cukd_protocol.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *cukd_wifi_config_status_name(cukd_wifi_config_status_t status) {
    switch (status) {
        case CUKD_WIFI_CONFIG_OK:
            return "OK";
        case CUKD_WIFI_CONFIG_BAD_PREFIX:
            return "BAD_PREFIX";
        case CUKD_WIFI_CONFIG_BAD_LENGTH:
            return "BAD_LENGTH";
        case CUKD_WIFI_CONFIG_BAD_TEXT:
            return "BAD_TEXT";
        case CUKD_WIFI_CONFIG_BAD_CHECKSUM:
            return "BAD_CHECKSUM";
        case CUKD_WIFI_CONFIG_BAD_SESSION:
            return "BAD_SESSION";
        case CUKD_WIFI_CONFIG_BAD_PORT:
            return "BAD_PORT";
        case CUKD_WIFI_CONFIG_BAD_SSID:
            return "BAD_SSID";
        case CUKD_WIFI_CONFIG_BAD_PASSWORD:
        default:
            return "BAD_PASSWORD";
    }
}

static int cukd_hex_nibble(char ch) {
    if (ch >= '0' && ch <= '9') {
        return ch - '0';
    }
    if (ch >= 'A' && ch <= 'F') {
        return ch - 'A' + 10;
    }
    return -1;
}

static int cukd_is_upper_hex(const char *text, size_t expected_length) {
    size_t index;
    if (text == NULL || strlen(text) != expected_length) {
        return 0;
    }
    for (index = 0; index < expected_length; ++index) {
        if (cukd_hex_nibble(text[index]) < 0) {
            return 0;
        }
    }
    return 1;
}

static int cukd_decode_printable_hex(
    const char *encoded,
    char *output,
    size_t output_size,
    size_t minimum_length,
    size_t maximum_length
) {
    size_t encoded_length;
    size_t decoded_length;
    size_t index;

    if (encoded == NULL || output == NULL || output_size == 0) {
        return 0;
    }
    encoded_length = strlen(encoded);
    if ((encoded_length & 1u) != 0u) {
        return 0;
    }
    decoded_length = encoded_length / 2u;
    if (
        decoded_length < minimum_length ||
        decoded_length > maximum_length ||
        decoded_length + 1u > output_size
    ) {
        return 0;
    }
    for (index = 0; index < decoded_length; ++index) {
        int high = cukd_hex_nibble(encoded[2u * index]);
        int low = cukd_hex_nibble(encoded[2u * index + 1u]);
        int value;
        if (high < 0 || low < 0) {
            return 0;
        }
        value = (high << 4) | low;
        if (value < 0x20 || value > 0x7e) {
            return 0;
        }
        output[index] = (char)value;
    }
    output[decoded_length] = '\0';
    return 1;
}

static void cukd_secure_zero(void *address, size_t length) {
    volatile uint8_t *cursor = (volatile uint8_t *)address;
    while (length-- > 0u) {
        *cursor++ = 0u;
    }
}

void cukd_clear_wifi_config(cukd_wifi_config_t *config) {
    if (config != NULL) {
        cukd_secure_zero(config, sizeof(*config));
    }
}

cukd_wifi_config_status_t cukd_parse_wifi_config_line(
    const uint8_t *data,
    size_t data_length,
    cukd_wifi_config_t *config
) {
    char buffer[CUKD_WIFI_CONFIG_LINE_MAX];
    char checksum_body[CUKD_WIFI_CONFIG_LINE_MAX];
    char *tokens[6];
    char *token;
    char *last_comma;
    char *end = NULL;
    unsigned long supplied_checksum;
    unsigned long port;
    char canonical_port[6];
    int canonical_port_length;
    uint16_t expected_checksum;
    size_t count = 0;
    size_t length;
    cukd_wifi_config_t parsed_config;

    size_t index;

    if (data == NULL || config == NULL) {
        return CUKD_WIFI_CONFIG_BAD_LENGTH;
    }
    cukd_clear_wifi_config(config);
    memset(&parsed_config, 0, sizeof(parsed_config));
    if (data_length == 0u || data_length >= sizeof(buffer)) {
        return CUKD_WIFI_CONFIG_BAD_LENGTH;
    }
    for (index = 0u; index < data_length; ++index) {
        if (data[index] < 0x20u || data[index] > 0x7eu) {
            return CUKD_WIFI_CONFIG_BAD_TEXT;
        }
    }
    length = data_length;
    memcpy(buffer, data, length);
    buffer[length] = '\0';
    memcpy(checksum_body, buffer, length + 1u);

    last_comma = strrchr(checksum_body, ',');
    if (last_comma == NULL) {
        return CUKD_WIFI_CONFIG_BAD_LENGTH;
    }
    supplied_checksum = strtoul(last_comma + 1, &end, 16);
    if (
        end == last_comma + 1 ||
        *end != '\0' ||
        strlen(last_comma + 1) != 4u ||
        !cukd_is_upper_hex(last_comma + 1, 4u) ||
        supplied_checksum > 0xffffu
    ) {
        return CUKD_WIFI_CONFIG_BAD_CHECKSUM;
    }
    *last_comma = '\0';
    expected_checksum = cukd_crc16_ccitt(
        (const uint8_t *)checksum_body,
        strlen(checksum_body)
    );
    if ((uint16_t)supplied_checksum != expected_checksum) {
        return CUKD_WIFI_CONFIG_BAD_CHECKSUM;
    }

    token = strtok(buffer, ",");
    while (token != NULL) {
        if (count >= sizeof(tokens) / sizeof(tokens[0])) {
            return CUKD_WIFI_CONFIG_BAD_LENGTH;
        }
        tokens[count++] = token;
        token = strtok(NULL, ",");
    }
    if (count != 6u) {
        return CUKD_WIFI_CONFIG_BAD_LENGTH;
    }
    if (strcmp(tokens[0], "CUKDWCFG2") != 0) {
        return CUKD_WIFI_CONFIG_BAD_PREFIX;
    }
    if (!cukd_is_upper_hex(tokens[1], CUKD_WIFI_SESSION_HEX_LENGTH)) {
        return CUKD_WIFI_CONFIG_BAD_SESSION;
    }
    memcpy(
        parsed_config.session_id,
        tokens[1],
        CUKD_WIFI_SESSION_HEX_LENGTH + 1u
    );
    port = strtoul(tokens[2], &end, 10);
    canonical_port_length = snprintf(
        canonical_port,
        sizeof(canonical_port),
        "%lu",
        port
    );
    if (
        end == tokens[2] ||
        *end != '\0' ||
        port == 0u ||
        port > 65535u ||
        canonical_port_length <= 0 ||
        (size_t)canonical_port_length >= sizeof(canonical_port) ||
        strcmp(tokens[2], canonical_port) != 0
    ) {
        return CUKD_WIFI_CONFIG_BAD_PORT;
    }
    if (!cukd_decode_printable_hex(
            tokens[3],
            parsed_config.ssid,
            sizeof(parsed_config.ssid),
            1u,
            CUKD_WIFI_SSID_MAX)) {
        cukd_clear_wifi_config(&parsed_config);
        return CUKD_WIFI_CONFIG_BAD_SSID;
    }
    if (!cukd_decode_printable_hex(
            tokens[4],
            parsed_config.password,
            sizeof(parsed_config.password),
            8u,
            CUKD_WIFI_PASSWORD_MAX)) {
        cukd_clear_wifi_config(&parsed_config);
        return CUKD_WIFI_CONFIG_BAD_PASSWORD;
    }
    parsed_config.udp_port = (uint16_t)port;
    memcpy(config, &parsed_config, sizeof(*config));
    cukd_clear_wifi_config(&parsed_config);
    return CUKD_WIFI_CONFIG_OK;
}
