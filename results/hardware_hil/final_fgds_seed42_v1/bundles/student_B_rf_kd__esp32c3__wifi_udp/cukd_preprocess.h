#ifndef CUKD_HIL_PREPROCESS_H
#define CUKD_HIL_PREPROCESS_H

#include <stdint.h>
#include "preprocess_int_metadata.h"

#ifdef __cplusplus
extern "C" {
#endif

void cukd_standardize_raw_q(const int32_t raw_q[CUKD_PREPROCESS_INPUT_DIM],
                            int16_t out_q[CUKD_PREPROCESS_INPUT_DIM]);

#ifdef __cplusplus
}
#endif

#endif

