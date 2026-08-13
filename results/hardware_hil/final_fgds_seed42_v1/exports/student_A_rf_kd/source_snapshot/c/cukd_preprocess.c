#include "cukd_preprocess.h"

static int16_t cukd_preprocess_sat_i16(int64_t x) {
    if (x > 32767) {
        return 32767;
    }
    if (x < -32768) {
        return -32768;
    }
    return (int16_t)x;
}

static int64_t cukd_preprocess_rescale(int64_t x) {
#if CUKD_PREPROCESS_RIGHT_SHIFT > 0
    if (x >= 0) {
        return x >> CUKD_PREPROCESS_RIGHT_SHIFT;
    }
    return -((-x) >> CUKD_PREPROCESS_RIGHT_SHIFT);
#elif CUKD_PREPROCESS_RIGHT_SHIFT < 0
    return x << (-CUKD_PREPROCESS_RIGHT_SHIFT);
#else
    return x;
#endif
}

void cukd_standardize_raw_q(const int32_t raw_q[CUKD_PREPROCESS_INPUT_DIM],
                            int16_t out_q[CUKD_PREPROCESS_INPUT_DIM]) {
    uint16_t i;
    for (i = 0; i < CUKD_PREPROCESS_INPUT_DIM; ++i) {
        int64_t centered = (int64_t)raw_q[i] - (int64_t)cukd_scaler_mean_q[i];
        int64_t scaled = centered * (int64_t)cukd_scaler_inv_scale_q[i];
        out_q[i] = cukd_preprocess_sat_i16(cukd_preprocess_rescale(scaled));
    }
}

