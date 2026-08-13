#ifndef CUKD_WIFI_ENVELOPE_H
#define CUKD_WIFI_ENVELOPE_H

#include <stddef.h>
#include <stdint.h>

#include "cukd_protocol.h"

#define CUKD_WIFI_ENVELOPE_PREFIX_LENGTH 7
#define CUKD_WIFI_SESSION_LENGTH 32
#define CUKD_WIFI_STAGE_LENGTH 16
#define CUKD_WIFI_TRANSACTION_LENGTH 16
#define CUKD_WIFI_DATAGRAM_MAX 768
#define CUKD_WIFI_REQUEST_PREFIX "CUKDW2Q"
#define CUKD_WIFI_RESPONSE_PREFIX "CUKDW2R"

typedef enum {
    CUKD_WIFI_ENVELOPE_OK = 0,
    CUKD_WIFI_ENVELOPE_BAD_LENGTH = 1,
    CUKD_WIFI_ENVELOPE_BAD_TEXT = 2,
    CUKD_WIFI_ENVELOPE_BAD_PREFIX = 3,
    CUKD_WIFI_ENVELOPE_BAD_IDENTITY = 4,
    CUKD_WIFI_ENVELOPE_BAD_ATTEMPT = 5,
    CUKD_WIFI_ENVELOPE_BAD_INNER_HEX = 6,
    CUKD_WIFI_ENVELOPE_BAD_CHECKSUM = 7
} cukd_wifi_envelope_status_t;

typedef struct {
    char session_id[CUKD_WIFI_SESSION_LENGTH + 1];
    char stage_id[CUKD_WIFI_STAGE_LENGTH + 1];
    char transaction_id[CUKD_WIFI_TRANSACTION_LENGTH + 1];
    uint8_t attempt;
    char inner_text[CUKD_HIL_LINE_MAX];
} cukd_wifi_envelope_t;

const char *cukd_wifi_envelope_status_name(cukd_wifi_envelope_status_t status);
cukd_wifi_envelope_status_t cukd_parse_wifi_request_envelope(
    const uint8_t *data,
    size_t data_length,
    cukd_wifi_envelope_t *envelope
);
int cukd_format_wifi_response_envelope(
    char *output,
    size_t output_size,
    const cukd_wifi_envelope_t *request,
    const char *inner_response
);

#endif
