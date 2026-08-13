#ifndef CUKD_HIL_PROTOCOL_H
#define CUKD_HIL_PROTOCOL_H

#include <stdint.h>
#include <stddef.h>

#define CUKD_HIL_FEATURE_COUNT 17
#define CUKD_HIL_OUTPUT_COUNT 5
#define CUKD_HIL_LINE_MAX 384

typedef enum {
    CUKD_STATUS_OK = 0,
    CUKD_STATUS_BAD_START = 1,
    CUKD_STATUS_BAD_LENGTH = 2,
    CUKD_STATUS_BAD_CHECKSUM = 3,
    CUKD_STATUS_BAD_FEATURE_RANGE = 4,
    CUKD_STATUS_INTERNAL_ERROR = 5
} cukd_status_t;

typedef struct {
    uint32_t row_id;
    int32_t features[CUKD_HIL_FEATURE_COUNT];
} cukd_request_t;

uint16_t cukd_crc16_ccitt(const uint8_t *data, size_t len);
const char *cukd_status_name(cukd_status_t status);
cukd_status_t cukd_parse_request_line(const char *line, cukd_request_t *request);
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
);

#endif

