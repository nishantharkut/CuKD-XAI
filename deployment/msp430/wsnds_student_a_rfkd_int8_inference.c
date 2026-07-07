/*
 * Dependency-free fixed-point inference core for the generated CuKD-XAI
 * WSN-DS Student A RF-KD model.
 *
 * Include path must contain generated model_weights.h from:
 *   python3 deployment/msp430/export_wsnds_student_a_rfkd_int8.py
 *
 * Numeric contract:
 *   - input_q15: 17 already preprocessed/standardized features in signed int16
 *     using CUKD_INPUT_Q_FRAC from the generated header
 *   - weights: int8, per-layer power-of-two scale from generated header
 *   - biases: int32 in Q(layer_input_frac + layer_weight_frac)
 *   - hidden activations/logits: signed int16 with generated output fractions
 */

#include <stdint.h>
#include "model_weights.h"

static int16_t cukd_sat_i16(int32_t x) {
    if (x > 32767) {
        return 32767;
    }
    if (x < -32768) {
        return -32768;
    }
    return (int16_t)x;
}

static int32_t cukd_rescale_acc(int32_t acc, int8_t shift) {
    if (shift > 0) {
        if (acc >= 0) {
            return acc >> shift;
        }
        return -((-acc) >> shift);
    }
    if (shift < 0) {
        return acc << (-shift);
    }
    return acc;
}

static void cukd_dense_i8_q15(
    const int16_t *input,
    int16_t *output,
    const int8_t *weights,
    const int32_t *biases,
    uint16_t in_dim,
    uint16_t out_dim,
    int8_t output_shift,
    uint8_t relu
) {
    uint16_t o;
    for (o = 0; o < out_dim; ++o) {
        int32_t acc = biases[o];
        uint16_t i;
        for (i = 0; i < in_dim; ++i) {
            acc += (int32_t)input[i] * (int32_t)weights[(o * in_dim) + i];
        }
        acc = cukd_rescale_acc(acc, output_shift);
        if (relu && acc < 0) {
            acc = 0;
        }
        output[o] = cukd_sat_i16(acc);
    }
}

void cukd_forward_q15(const int16_t input_q15[CUKD_INPUT_DIM],
                      int16_t logits_q15[CUKD_OUTPUT_DIM]) {
    int16_t h1[CUKD_H1_DIM];
    int16_t h2[CUKD_H2_DIM];

    cukd_dense_i8_q15(
        input_q15,
        h1,
        (const int8_t *)&cukd_l0_weight[0][0],
        cukd_l0_bias,
        CUKD_L0_IN,
        CUKD_L0_OUT,
        CUKD_L0_SHIFT,
        1
    );
    cukd_dense_i8_q15(
        h1,
        h2,
        (const int8_t *)&cukd_l1_weight[0][0],
        cukd_l1_bias,
        CUKD_L1_IN,
        CUKD_L1_OUT,
        CUKD_L1_SHIFT,
        1
    );
    cukd_dense_i8_q15(
        h2,
        logits_q15,
        (const int8_t *)&cukd_l2_weight[0][0],
        cukd_l2_bias,
        CUKD_L2_IN,
        CUKD_L2_OUT,
        CUKD_L2_SHIFT,
        0
    );
}

uint8_t cukd_predict_q15(const int16_t input_q15[CUKD_INPUT_DIM]) {
    int16_t logits[CUKD_OUTPUT_DIM];
    uint8_t best = 0;
    uint8_t i;

    cukd_forward_q15(input_q15, logits);
    for (i = 1; i < CUKD_OUTPUT_DIM; ++i) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    return best;
}

