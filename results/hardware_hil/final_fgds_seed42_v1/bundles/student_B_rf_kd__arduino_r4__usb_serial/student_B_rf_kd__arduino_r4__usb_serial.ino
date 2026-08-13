#include <Arduino.h>
#include <stdint.h>
#include <string.h>

#include "cukd_export_identity.h"
#include "cukd_final_bundle_identity.h"

extern "C" {
#include "cukd_model.h"
#include "cukd_preprocess.h"
#include "cukd_protocol.h"
}

// BAD_CHECKSUM is emitted by cukd_parse_request_line and returned through send_status.
static char rx_line[CUKD_HIL_LINE_MAX];
static size_t rx_len = 0;

static uint32_t cukd_now_us() {
    return (uint32_t)micros();
}

static void cukd_send_status(uint32_t row_id, cukd_status_t status) {
    char out[CUKD_HIL_LINE_MAX];
    int16_t logits[CUKD_HIL_OUTPUT_COUNT] = {0, 0, 0, 0, 0};
    if (cukd_format_response_line(out, sizeof(out), row_id, status, -1, logits, 0, 0, 0)) {
        Serial.print(out);
    }
}

static uint8_t cukd_argmax_logits(const int16_t logits[CUKD_OUTPUT_DIM]) {
    uint8_t best = 0;
    for (uint8_t i = 1; i < CUKD_OUTPUT_DIM; ++i) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    return best;
}

static void cukd_process_line(const char *line) {
    if (strcmp(line, "CUKDID?") == 0) {
        Serial.println(CUKD_FINAL_RUNTIME_IDENTITY);
        return;
    }
    cukd_request_t request;
    int16_t input_q15[CUKD_INPUT_DIM];
    int16_t logits[CUKD_OUTPUT_DIM];
    uint32_t start_us;
    uint32_t preprocess_start;
    uint32_t preprocess_end;
    uint32_t inference_end;
    uint32_t preprocess_us;
    uint32_t inference_us;
    uint32_t total_us;
    uint8_t pred;
    char out[CUKD_HIL_LINE_MAX];

    const cukd_status_t status = cukd_parse_request_line(line, &request);
    if (status != CUKD_STATUS_OK) {
        cukd_send_status(0, status);
        return;
    }

    start_us = cukd_now_us();
    preprocess_start = start_us;
    cukd_standardize_raw_q(request.features, input_q15);
    preprocess_end = cukd_now_us();
    cukd_forward_q15(input_q15, logits);
    pred = cukd_argmax_logits(logits);
#ifdef CUKD_HIL_VERIFY_PREDICT_WRAPPER
    if (cukd_predict_q15(input_q15) != pred) {
        cukd_send_status(request.row_id, CUKD_STATUS_INTERNAL_ERROR);
        return;
    }
#endif
    inference_end = cukd_now_us();
    preprocess_us = preprocess_end - preprocess_start;
    inference_us = inference_end - preprocess_end;
    total_us = inference_end - start_us;

    if (cukd_format_response_line(
            out,
            sizeof(out),
            request.row_id,
            CUKD_STATUS_OK,
            (int16_t)pred,
            logits,
            preprocess_us,
            inference_us,
            total_us)) {
        Serial.print(out);
    } else {
        cukd_send_status(request.row_id, CUKD_STATUS_INTERNAL_ERROR);
    }
}

void setup() {
    Serial.begin(115200);
    rx_len = 0;
}

void loop() {
    while (Serial.available() > 0) {
        const char ch = (char)Serial.read();
        if (ch == '\n') {
            rx_line[rx_len] = '\0';
            cukd_process_line(rx_line);
            rx_len = 0;
        } else if (ch != '\r') {
            if (rx_len + 1 < sizeof(rx_line)) {
                rx_line[rx_len++] = ch;
            } else {
                rx_len = 0;
                cukd_send_status(0, CUKD_STATUS_BAD_LENGTH);
            }
        }
    }
}

