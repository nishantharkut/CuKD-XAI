#ifndef CUKD_HIL_MODEL_H
#define CUKD_HIL_MODEL_H

#include <stdint.h>
#include "model_weights.h"

#ifdef __cplusplus
extern "C" {
#endif

void cukd_forward_q15(const int16_t input_q15[CUKD_INPUT_DIM],
                      int16_t logits_q15[CUKD_OUTPUT_DIM]);

uint8_t cukd_predict_q15(const int16_t input_q15[CUKD_INPUT_DIM]);

#ifdef __cplusplus
}
#endif

#endif

