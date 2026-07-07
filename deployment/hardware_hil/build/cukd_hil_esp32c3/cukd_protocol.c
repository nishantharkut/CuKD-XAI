#include "cukd_protocol.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

uint16_t cukd_crc16_ccitt(const uint8_t *data, size_t len) {
    uint16_t crc = 0xFFFFu;
    size_t i;
    for (i = 0; i < len; ++i) {
        uint8_t bit;
        crc ^= (uint16_t)data[i] << 8;
        for (bit = 0; bit < 8; ++bit) {
            if (crc & 0x8000u) {
                crc = (uint16_t)((crc << 1) ^ 0x1021u);
            } else {
                crc = (uint16_t)(crc << 1);
            }
        }
    }
    return crc;
}

const char *cukd_status_name(cukd_status_t status) {
    switch (status) {
        case CUKD_STATUS_OK:
            return "OK";
        case CUKD_STATUS_BAD_START:
            return "BAD_START";
        case CUKD_STATUS_BAD_LENGTH:
            return "BAD_LENGTH";
        case CUKD_STATUS_BAD_CHECKSUM:
            return "BAD_CHECKSUM";
        case CUKD_STATUS_BAD_FEATURE_RANGE:
            return "BAD_FEATURE_RANGE";
        case CUKD_STATUS_INTERNAL_ERROR:
        default:
            return "INTERNAL_ERROR";
    }
}

static int cukd_parse_i64(const char *text, int64_t *out) {
    char *end = NULL;
    long long value = strtoll(text, &end, 10);
    if (end == text || *end != '\0') {
        return 0;
    }
    *out = (int64_t)value;
    return 1;
}

static int cukd_parse_crc(const char *text, uint16_t *out) {
    char *end = NULL;
    unsigned long value = strtoul(text, &end, 16);
    if (end == text || *end != '\0' || value > 0xFFFFu) {
        return 0;
    }
    *out = (uint16_t)value;
    return 1;
}

cukd_status_t cukd_parse_request_line(const char *line, cukd_request_t *request) {
    char buffer[CUKD_HIL_LINE_MAX];
    char *tokens[CUKD_HIL_FEATURE_COUNT + 3];
    size_t len;
    size_t count = 0;
    char *token;
    uint16_t supplied_crc;
    uint16_t expected_crc;
    char *last_comma;
    int64_t parsed;
    int i;

    if (line == NULL || request == NULL) {
        return CUKD_STATUS_INTERNAL_ERROR;
    }
    len = strlen(line);
    if (len == 0 || len >= sizeof(buffer)) {
        return CUKD_STATUS_BAD_LENGTH;
    }
    memcpy(buffer, line, len + 1);
    while (len > 0 && (buffer[len - 1] == '\n' || buffer[len - 1] == '\r')) {
        buffer[len - 1] = '\0';
        len--;
    }

    last_comma = strrchr(buffer, ',');
    if (last_comma == NULL) {
        return CUKD_STATUS_BAD_LENGTH;
    }
    if (!cukd_parse_crc(last_comma + 1, &supplied_crc)) {
        return CUKD_STATUS_BAD_CHECKSUM;
    }
    *last_comma = '\0';
    expected_crc = cukd_crc16_ccitt((const uint8_t *)buffer, strlen(buffer));
    *last_comma = ',';
    if (supplied_crc != expected_crc) {
        return CUKD_STATUS_BAD_CHECKSUM;
    }

    token = strtok(buffer, ",");
    while (token != NULL) {
        if (count >= (sizeof(tokens) / sizeof(tokens[0]))) {
            return CUKD_STATUS_BAD_LENGTH;
        }
        tokens[count++] = token;
        token = strtok(NULL, ",");
    }
    if (count != CUKD_HIL_FEATURE_COUNT + 3) {
        return CUKD_STATUS_BAD_LENGTH;
    }
    if (strcmp(tokens[0], "CUKD1") != 0) {
        return CUKD_STATUS_BAD_START;
    }
    if (!cukd_parse_i64(tokens[1], &parsed) || parsed < 0) {
        return CUKD_STATUS_BAD_FEATURE_RANGE;
    }
    request->row_id = (uint32_t)parsed;

    for (i = 0; i < CUKD_HIL_FEATURE_COUNT; ++i) {
        if (!cukd_parse_i64(tokens[2 + i], &parsed)) {
            return CUKD_STATUS_BAD_FEATURE_RANGE;
        }
        if (parsed < INT32_MIN || parsed > INT32_MAX) {
            return CUKD_STATUS_BAD_FEATURE_RANGE;
        }
        request->features[i] = (int32_t)parsed;
    }
    return CUKD_STATUS_OK;
}

int cukd_format_response_line(
    char *out,
    size_t out_len,
    uint32_t row_id,
    cukd_status_t status,
    int16_t predicted_class,
    const int16_t logits[CUKD_HIL_OUTPUT_COUNT],
    uint32_t preprocess_us,
    uint32_t inference_us,
    uint32_t total_us
) {
    char body[CUKD_HIL_LINE_MAX];
    int n;
    uint16_t crc;
    int16_t zero_logits[CUKD_HIL_OUTPUT_COUNT] = {0, 0, 0, 0, 0};
    const int16_t *selected_logits = logits ? logits : zero_logits;

    n = snprintf(
        body,
        sizeof(body),
        "CUKD1R,%lu,%s,%d,%d,%d,%d,%d,%d,%lu,%lu,%lu",
        (unsigned long)row_id,
        cukd_status_name(status),
        (int)predicted_class,
        (int)selected_logits[0],
        (int)selected_logits[1],
        (int)selected_logits[2],
        (int)selected_logits[3],
        (int)selected_logits[4],
        (unsigned long)preprocess_us,
        (unsigned long)inference_us,
        (unsigned long)total_us
    );
    if (n < 0 || (size_t)n >= sizeof(body)) {
        return 0;
    }
    crc = cukd_crc16_ccitt((const uint8_t *)body, (size_t)n);
    n = snprintf(out, out_len, "%s,%04X\n", body, (unsigned int)crc);
    return n > 0 && (size_t)n < out_len;
}

