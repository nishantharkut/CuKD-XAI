#include <stdint.h>

#include "cukd_export_identity.h"
#include "cukd_model.h"
#include "cukd_preprocess.h"

typedef char cukd_input_dimensions_must_match[
    (CUKD_INPUT_DIM == CUKD_PREPROCESS_INPUT_DIM) ? 1 : -1
];

volatile uint8_t cukd_msp430_prediction_sink;
volatile int16_t cukd_msp430_preprocess_sink;

int main(void) {
    int32_t raw_q[CUKD_PREPROCESS_INPUT_DIM] = {0};
    int16_t standardized_q[CUKD_PREPROCESS_INPUT_DIM] = {0};

    cukd_standardize_raw_q(raw_q, standardized_q);
    cukd_msp430_preprocess_sink = standardized_q[0];
    cukd_msp430_prediction_sink = cukd_predict_q15(standardized_q);

    for (;;) {
    }
}
