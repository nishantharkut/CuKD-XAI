/* Fail-closed host equivalence test for a train-only-scaler WSN-DS export. */

#include <stdint.h>

#include "model_weights.h"
#include "preprocess_int_metadata.h"
#include "test_vectors.h"

void cukd_standardize_raw_q(
    const int32_t raw_q[CUKD_PREPROCESS_INPUT_DIM],
    int16_t out_q[CUKD_PREPROCESS_INPUT_DIM]);
void cukd_forward_q15(
    const int16_t input_q15[CUKD_INPUT_DIM],
    int16_t logits_q15[CUKD_OUTPUT_DIM]);
uint8_t cukd_predict_q15(const int16_t input_q15[CUKD_INPUT_DIM]);

static uint8_t argmax_i16(const int16_t values[CUKD_OUTPUT_DIM]) {
    uint8_t best = 0;
    uint8_t i;
    for (i = 1; i < CUKD_OUTPUT_DIM; ++i) {
        if (values[i] > values[best]) {
            best = i;
        }
    }
    return best;
}

int main(void) {
    uint32_t row;
    if (CUKD_TEST_VECTOR_COUNT == 0) {
        return 1;
    }
    if (CUKD_TEST_HAS_RAW_PREPROCESS != 1) {
        return 2;
    }
    if (CUKD_TEST_INPUT_DIM != CUKD_INPUT_DIM ||
        CUKD_PREPROCESS_INPUT_DIM != CUKD_INPUT_DIM) {
        return 3;
    }
    if (CUKD_TEST_OUTPUT_DIM != CUKD_OUTPUT_DIM) {
        return 4;
    }
    if (CUKD_PREPROCESS_OUTPUT_Q_FRAC != CUKD_INPUT_Q_FRAC) {
        return 5;
    }

    for (row = 0; row < (uint32_t)CUKD_TEST_VECTOR_COUNT; ++row) {
        int16_t preprocessed[CUKD_INPUT_DIM];
        int16_t logits[CUKD_OUTPUT_DIM];
        uint8_t feature;
        uint8_t output;

        cukd_standardize_raw_q(cukd_test_raw_inputs_q[row], preprocessed);
        for (feature = 0; feature < CUKD_INPUT_DIM; ++feature) {
            if (preprocessed[feature] !=
                cukd_test_expected_preprocessed_q15[row][feature]) {
                return 10;
            }
        }

        cukd_forward_q15(preprocessed, logits);
        for (output = 0; output < CUKD_OUTPUT_DIM; ++output) {
            if (logits[output] != cukd_test_expected_fixed_logits[row][output]) {
                return 11;
            }
        }
        if (argmax_i16(logits) != cukd_test_expected_fixed_pred[row]) {
            return 12;
        }
        if (cukd_predict_q15(preprocessed) != cukd_test_expected_fixed_pred[row]) {
            return 13;
        }
    }
    return 0;
}
