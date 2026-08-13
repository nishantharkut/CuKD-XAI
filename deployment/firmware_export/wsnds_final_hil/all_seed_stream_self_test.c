/* Streamed C/Python equivalence test for the compact all-seed audit. */

#include <stdint.h>
#include <stdio.h>

#include "model_weights.h"
#include "preprocess_int_metadata.h"

#ifndef CUKD_AUDIT_ROWS
#error "CUKD_AUDIT_ROWS must be defined"
#endif

void cukd_standardize_raw_q(
    const int32_t raw_q[CUKD_PREPROCESS_INPUT_DIM],
    int16_t out_q[CUKD_PREPROCESS_INPUT_DIM]);
void cukd_forward_q15(
    const int16_t input_q15[CUKD_INPUT_DIM],
    int16_t logits_q15[CUKD_OUTPUT_DIM]);
uint8_t cukd_predict_q15(const int16_t input_q15[CUKD_INPUT_DIM]);

static uint8_t argmax_i16(const int16_t values[CUKD_OUTPUT_DIM]) {
    uint8_t best = 0;
    uint8_t index;
    for (index = 1; index < CUKD_OUTPUT_DIM; ++index) {
        if (values[index] > values[best]) {
            best = index;
        }
    }
    return best;
}

static int read_exact(FILE *handle, void *target, size_t size, size_t count) {
    return fread(target, size, count, handle) == count;
}

int main(int argc, char **argv) {
    FILE *handle;
    uint32_t row;
    if (argc != 2) {
        return 1;
    }
    if (CUKD_INPUT_DIM != 17 || CUKD_OUTPUT_DIM != 5 ||
        CUKD_PREPROCESS_INPUT_DIM != CUKD_INPUT_DIM ||
        CUKD_PREPROCESS_OUTPUT_Q_FRAC != CUKD_INPUT_Q_FRAC) {
        return 2;
    }
    handle = fopen(argv[1], "rb");
    if (handle == NULL) {
        return 3;
    }
    for (row = 0; row < (uint32_t)CUKD_AUDIT_ROWS; ++row) {
        int32_t raw[CUKD_INPUT_DIM];
        int16_t expected_preprocessed[CUKD_INPUT_DIM];
        int16_t expected_logits[CUKD_OUTPUT_DIM];
        int16_t observed_preprocessed[CUKD_INPUT_DIM];
        int16_t observed_logits[CUKD_OUTPUT_DIM];
        uint8_t expected_prediction;
        uint8_t index;

        if (!read_exact(handle, raw, sizeof(raw[0]), CUKD_INPUT_DIM) ||
            !read_exact(handle, expected_preprocessed,
                        sizeof(expected_preprocessed[0]), CUKD_INPUT_DIM) ||
            !read_exact(handle, expected_logits,
                        sizeof(expected_logits[0]), CUKD_OUTPUT_DIM) ||
            !read_exact(handle, &expected_prediction,
                        sizeof(expected_prediction), 1)) {
            fclose(handle);
            return 4;
        }
        cukd_standardize_raw_q(raw, observed_preprocessed);
        for (index = 0; index < CUKD_INPUT_DIM; ++index) {
            if (observed_preprocessed[index] != expected_preprocessed[index]) {
                fclose(handle);
                return 10;
            }
        }
        cukd_forward_q15(observed_preprocessed, observed_logits);
        for (index = 0; index < CUKD_OUTPUT_DIM; ++index) {
            if (observed_logits[index] != expected_logits[index]) {
                fclose(handle);
                return 11;
            }
        }
        if (argmax_i16(observed_logits) != expected_prediction ||
            cukd_predict_q15(observed_preprocessed) != expected_prediction) {
            fclose(handle);
            return 12;
        }
    }
    if (fgetc(handle) != EOF) {
        fclose(handle);
        return 5;
    }
    if (ferror(handle)) {
        fclose(handle);
        return 6;
    }
    fclose(handle);
    return 0;
}
