#include <Arduino.h>
#include <stdint.h>

#include "model_weights.h"
#include "test_vectors.h"

extern "C" {
void cukd_forward_q15(const int16_t input_q15[CUKD_INPUT_DIM],
                      int16_t logits_q15[CUKD_OUTPUT_DIM]);
uint8_t cukd_predict_q15(const int16_t input_q15[CUKD_INPUT_DIM]);
}

#ifndef CUKD_HARDWARE_TEST_LIMIT
#define CUKD_HARDWARE_TEST_LIMIT CUKD_TEST_VECTOR_COUNT
#endif

static uint8_t cukd_argmax_i16_local(const int16_t values[CUKD_OUTPUT_DIM]) {
    uint8_t best = 0;
    for (uint8_t i = 1; i < CUKD_OUTPUT_DIM; ++i) {
        if (values[i] > values[best]) {
            best = i;
        }
    }
    return best;
}

static uint32_t cukd_vector_limit(void) {
    uint32_t limit = (uint32_t)CUKD_HARDWARE_TEST_LIMIT;
    if (limit > (uint32_t)CUKD_TEST_VECTOR_COUNT) {
        limit = (uint32_t)CUKD_TEST_VECTOR_COUNT;
    }
    return limit;
}

void setup() {
    Serial.begin(115200);
    delay(1500);

    Serial.println();
    Serial.println("CuKD-XAI Student A RF-KD fixed-point hardware self-test");

#if CUKD_TEST_INPUT_DIM != CUKD_INPUT_DIM
    Serial.println("status = failed");
    Serial.println("reason = input dimension mismatch");
    return;
#endif

#if CUKD_TEST_OUTPUT_DIM != CUKD_OUTPUT_DIM
    Serial.println("status = failed");
    Serial.println("reason = output dimension mismatch");
    return;
#endif

    const uint32_t limit = cukd_vector_limit();
    uint32_t prediction_failures = 0;
    uint32_t logit_failures = 0;
    uint32_t predict_wrapper_failures = 0;

    const uint32_t started_us = micros();
    for (uint32_t t = 0; t < limit; ++t) {
        int16_t logits[CUKD_OUTPUT_DIM];
        cukd_forward_q15(cukd_test_inputs_q15[t], logits);

        const uint8_t pred_from_logits = cukd_argmax_i16_local(logits);
        const uint8_t expected = cukd_test_expected_fixed_pred[t];

        if (pred_from_logits != expected) {
            prediction_failures++;
        }

        for (uint8_t j = 0; j < CUKD_OUTPUT_DIM; ++j) {
            if (logits[j] != cukd_test_expected_fixed_logits[t][j]) {
                logit_failures++;
                break;
            }
        }
    }
    const uint32_t elapsed_us = micros() - started_us;

    for (uint32_t t = 0; t < limit; ++t) {
        const uint8_t pred_direct = cukd_predict_q15(cukd_test_inputs_q15[t]);
        if (pred_direct != cukd_test_expected_fixed_pred[t]) {
            predict_wrapper_failures++;
        }
    }

    Serial.print("vectors = ");
    Serial.println(limit);
    Serial.print("prediction_failures = ");
    Serial.println(prediction_failures);
    Serial.print("logit_failures = ");
    Serial.println(logit_failures);
    Serial.print("predict_wrapper_failures = ");
    Serial.println(predict_wrapper_failures);
    Serial.print("elapsed_us = ");
    Serial.println(elapsed_us);
    Serial.print("avg_us_per_vector = ");
    if (limit > 0) {
        Serial.println(elapsed_us / limit);
    } else {
        Serial.println(0);
    }
    Serial.print("model_input_dim = ");
    Serial.println(CUKD_INPUT_DIM);
    Serial.print("model_output_dim = ");
    Serial.println(CUKD_OUTPUT_DIM);
    Serial.print("test_vector_count = ");
    Serial.println(CUKD_TEST_VECTOR_COUNT);

    if (prediction_failures == 0 && logit_failures == 0 && predict_wrapper_failures == 0) {
        Serial.println("status = passed");
    } else {
        Serial.println("status = failed");
    }
}

void loop() {
}
