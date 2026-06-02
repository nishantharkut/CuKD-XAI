/*
 * Host self-test for the generated CuKD-XAI WSN-DS Student A RF-KD
 * fixed-point export.
 *
 * Build with wsnds_student_a_rfkd_int8_inference.c and include the generated
 * directory containing model_weights.h and test_vectors.h.
 */

#include <stdint.h>
#include "model_weights.h"
#include "test_vectors.h"

void cukd_forward_q15(const int16_t input_q15[CUKD_INPUT_DIM],
                      int16_t logits_q15[CUKD_OUTPUT_DIM]);
uint8_t cukd_predict_q15(const int16_t input_q15[CUKD_INPUT_DIM]);

static uint8_t cukd_argmax_i16(const int16_t values[CUKD_OUTPUT_DIM]) {
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
#if CUKD_TEST_INPUT_DIM != CUKD_INPUT_DIM
    return 2;
#endif
#if CUKD_TEST_OUTPUT_DIM != CUKD_OUTPUT_DIM
    return 3;
#endif

    uint16_t t;
    uint16_t failures = 0;
    for (t = 0; t < CUKD_TEST_VECTOR_COUNT; ++t) {
        int16_t logits[CUKD_OUTPUT_DIM];
        uint8_t j;
        uint8_t pred_direct;
        uint8_t pred_from_logits;

        cukd_forward_q15(cukd_test_inputs_q15[t], logits);
        pred_direct = cukd_predict_q15(cukd_test_inputs_q15[t]);
        pred_from_logits = cukd_argmax_i16(logits);

        if (pred_direct != cukd_test_expected_fixed_pred[t]) {
            failures++;
        }
        if (pred_from_logits != cukd_test_expected_fixed_pred[t]) {
            failures++;
        }
        for (j = 0; j < CUKD_OUTPUT_DIM; ++j) {
            if (logits[j] != cukd_test_expected_fixed_logits[t][j]) {
                failures++;
                break;
            }
        }
    }

    return failures == 0 ? 0 : 1;
}
