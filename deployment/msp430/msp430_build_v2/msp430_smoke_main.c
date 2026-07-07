#include <stdint.h>
#include "model_weights.h"
#include "preprocess_int_metadata.h"

void cukd_standardize_raw_q(const int32_t raw_q[CUKD_PREPROCESS_INPUT_DIM],
                            int16_t out_q[CUKD_PREPROCESS_INPUT_DIM]);
uint8_t cukd_predict_q15(const int16_t input_q15[CUKD_INPUT_DIM]);

volatile uint8_t cukd_sink;

int main(void) {
    int32_t raw_q[CUKD_PREPROCESS_INPUT_DIM] = {0};
    int16_t input_q[CUKD_INPUT_DIM] = {0};

    cukd_standardize_raw_q(raw_q, input_q);
    cukd_sink = cukd_predict_q15(input_q);

    while (1) {}
    return 0;
}
