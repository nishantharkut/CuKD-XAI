#include "cukd_wifi_envelope.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *cukd_wifi_envelope_status_name(cukd_wifi_envelope_status_t status) {
    switch (status) {
        case CUKD_WIFI_ENVELOPE_OK:
            return "OK";
        case CUKD_WIFI_ENVELOPE_BAD_LENGTH:
            return "BAD_LENGTH";
        case CUKD_WIFI_ENVELOPE_BAD_TEXT:
            return "BAD_TEXT";
        case CUKD_WIFI_ENVELOPE_BAD_PREFIX:
            return "BAD_PREFIX";
        case CUKD_WIFI_ENVELOPE_BAD_IDENTITY:
            return "BAD_IDENTITY";
        case CUKD_WIFI_ENVELOPE_BAD_ATTEMPT:
            return "BAD_ATTEMPT";
        case CUKD_WIFI_ENVELOPE_BAD_INNER_HEX:
            return "BAD_INNER_HEX";
        case CUKD_WIFI_ENVELOPE_BAD_CHECKSUM:
        default:
            return "BAD_CHECKSUM";
    }
}

static int cukd_hex_nibble(char value) {
    if (value >= '0' && value <= '9') {
        return value - '0';
    }
    if (value >= 'A' && value <= 'F') {
        return value - 'A' + 10;
    }
    return -1;
}

static int cukd_is_upper_hex(const char *text, size_t length) {
    size_t index;
    if (text == NULL || strlen(text) != length) {
        return 0;
    }
    for (index = 0; index < length; ++index) {
        if (cukd_hex_nibble(text[index]) < 0) {
            return 0;
        }
    }
    return 1;
}

static int cukd_parse_attempt(const char *text, uint8_t *attempt) {
    char *end = NULL;
    unsigned long value;
    char canonical[4];
    int count;
    if (text == NULL || attempt == NULL) {
        return 0;
    }
    value = strtoul(text, &end, 10);
    if (end == text || *end != '\0' || value == 0u || value > 255u) {
        return 0;
    }
    count = snprintf(canonical, sizeof(canonical), "%lu", value);
    if (count <= 0 || (size_t)count >= sizeof(canonical) || strcmp(text, canonical) != 0) {
        return 0;
    }
    *attempt = (uint8_t)value;
    return 1;
}

static int cukd_decode_inner_hex(const char *encoded, char *output, size_t output_size) {
    size_t encoded_length;
    size_t decoded_length;
    size_t index;
    if (encoded == NULL || output == NULL || output_size == 0u) {
        return 0;
    }
    encoded_length = strlen(encoded);
    if (encoded_length == 0u || (encoded_length & 1u) != 0u) {
        return 0;
    }
    decoded_length = encoded_length / 2u;
    if (decoded_length + 1u > output_size) {
        return 0;
    }
    for (index = 0; index < decoded_length; ++index) {
        const int high = cukd_hex_nibble(encoded[2u * index]);
        const int low = cukd_hex_nibble(encoded[2u * index + 1u]);
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

static int cukd_parse_supplied_crc(const char *text, uint16_t *crc) {
    int index;
    uint16_t value = 0u;
    if (!cukd_is_upper_hex(text, 4u) || crc == NULL) {
        return 0;
    }
    for (index = 0; index < 4; ++index) {
        value = (uint16_t)((value << 4) | (uint16_t)cukd_hex_nibble(text[index]));
    }
    *crc = value;
    return 1;
}

cukd_wifi_envelope_status_t cukd_parse_wifi_request_envelope(
    const uint8_t *data,
    size_t data_length,
    cukd_wifi_envelope_t *envelope
) {
    char buffer[CUKD_WIFI_DATAGRAM_MAX + 1u];
    char checksum_body[CUKD_WIFI_DATAGRAM_MAX + 1u];
    char *tokens[7];
    char *token;
    char *last_comma;
    size_t index;
    size_t token_count = 0u;
    uint16_t supplied_crc;
    uint16_t expected_crc;
    cukd_wifi_envelope_t parsed;

    if (
        data == NULL ||
        envelope == NULL ||
        data_length == 0u ||
        data_length > CUKD_WIFI_DATAGRAM_MAX
    ) {
        return CUKD_WIFI_ENVELOPE_BAD_LENGTH;
    }
    memset(&parsed, 0, sizeof(parsed));
    for (index = 0u; index < data_length; ++index) {
        if (data[index] < 0x20u || data[index] > 0x7eu) {
            return CUKD_WIFI_ENVELOPE_BAD_TEXT;
        }
    }
    memcpy(buffer, data, data_length);
    buffer[data_length] = '\0';
    memcpy(checksum_body, buffer, data_length + 1u);
    last_comma = strrchr(checksum_body, ',');
    if (last_comma == NULL) {
        return CUKD_WIFI_ENVELOPE_BAD_LENGTH;
    }
    if (!cukd_parse_supplied_crc(last_comma + 1, &supplied_crc)) {
        return CUKD_WIFI_ENVELOPE_BAD_CHECKSUM;
    }
    *last_comma = '\0';
    expected_crc = cukd_crc16_ccitt(
        (const uint8_t *)checksum_body,
        strlen(checksum_body)
    );
    if (supplied_crc != expected_crc) {
        return CUKD_WIFI_ENVELOPE_BAD_CHECKSUM;
    }

    token = strtok(buffer, ",");
    while (token != NULL) {
        if (token_count >= sizeof(tokens) / sizeof(tokens[0])) {
            return CUKD_WIFI_ENVELOPE_BAD_LENGTH;
        }
        tokens[token_count++] = token;
        token = strtok(NULL, ",");
    }
    if (token_count != 7u) {
        return CUKD_WIFI_ENVELOPE_BAD_LENGTH;
    }
    if (strcmp(tokens[0], CUKD_WIFI_REQUEST_PREFIX) != 0) {
        return CUKD_WIFI_ENVELOPE_BAD_PREFIX;
    }
    if (
        !cukd_is_upper_hex(tokens[1], CUKD_WIFI_SESSION_LENGTH) ||
        !cukd_is_upper_hex(tokens[2], CUKD_WIFI_STAGE_LENGTH) ||
        !cukd_is_upper_hex(tokens[3], CUKD_WIFI_TRANSACTION_LENGTH)
    ) {
        return CUKD_WIFI_ENVELOPE_BAD_IDENTITY;
    }
    if (!cukd_parse_attempt(tokens[4], &parsed.attempt)) {
        return CUKD_WIFI_ENVELOPE_BAD_ATTEMPT;
    }
    if (!cukd_decode_inner_hex(
            tokens[5],
            parsed.inner_text,
            sizeof(parsed.inner_text))) {
        return CUKD_WIFI_ENVELOPE_BAD_INNER_HEX;
    }
    memcpy(parsed.session_id, tokens[1], CUKD_WIFI_SESSION_LENGTH + 1u);
    memcpy(parsed.stage_id, tokens[2], CUKD_WIFI_STAGE_LENGTH + 1u);
    memcpy(
        parsed.transaction_id,
        tokens[3],
        CUKD_WIFI_TRANSACTION_LENGTH + 1u
    );
    memcpy(envelope, &parsed, sizeof(*envelope));
    return CUKD_WIFI_ENVELOPE_OK;
}

static int cukd_append_hex(
    char *output,
    size_t output_size,
    size_t *offset,
    const char *text
) {
    static const char digits[] = "0123456789ABCDEF";
    size_t index;
    const size_t length = strlen(text);
    if (*offset >= output_size || length > (output_size - *offset - 1u) / 2u) {
        return 0;
    }
    for (index = 0u; index < length; ++index) {
        const uint8_t value = (uint8_t)text[index];
        if (value < 0x20u || value > 0x7eu) {
            return 0;
        }
        output[(*offset)++] = digits[value >> 4];
        output[(*offset)++] = digits[value & 0x0fu];
    }
    output[*offset] = '\0';
    return 1;
}

int cukd_format_wifi_response_envelope(
    char *output,
    size_t output_size,
    const cukd_wifi_envelope_t *request,
    const char *inner_response
) {
    int count;
    size_t offset;
    uint16_t checksum;
    if (
        output == NULL ||
        request == NULL ||
        inner_response == NULL ||
        output_size == 0u ||
        output_size > CUKD_WIFI_DATAGRAM_MAX + 1u
    ) {
        return 0;
    }
    count = snprintf(
        output,
        output_size,
        "%s,%s,%s,%s,%u,",
        CUKD_WIFI_RESPONSE_PREFIX,
        request->session_id,
        request->stage_id,
        request->transaction_id,
        (unsigned int)request->attempt
    );
    if (count <= 0 || (size_t)count >= output_size) {
        return 0;
    }
    offset = (size_t)count;
    if (!cukd_append_hex(output, output_size, &offset, inner_response)) {
        return 0;
    }
    checksum = cukd_crc16_ccitt((const uint8_t *)output, offset);
    count = snprintf(
        output + offset,
        output_size - offset,
        ",%04X",
        (unsigned int)checksum
    );
    return count == 5 && offset + (size_t)count < output_size;
}
