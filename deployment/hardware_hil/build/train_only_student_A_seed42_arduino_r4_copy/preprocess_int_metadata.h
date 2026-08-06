#ifndef CUKD_WSNDS_PREPROCESS_INT_METADATA_H
#define CUKD_WSNDS_PREPROCESS_INT_METADATA_H

#include <stdint.h>

/* Integer StandardScaler metadata for WSN-DS fixed-point preprocessing.
 * Feature extraction is handled before this normalization step.
 */

#define CUKD_PREPROCESS_INPUT_DIM 17
#define CUKD_PREPROCESS_RAW_Q_FRAC 8
#define CUKD_PREPROCESS_INV_SCALE_Q_FRAC 20
#define CUKD_PREPROCESS_OUTPUT_Q_FRAC 8
#define CUKD_PREPROCESS_RIGHT_SHIFT 20
#define CUKD_PREPROCESS_OPS_PER_SAMPLE 68

static const int32_t cukd_scaler_mean_q[17] = {272070, 30, 70201770, 5797, 69, 1775, 200, 189, 74, 192, 2491, 11486, 18882, 1162, 5751, 639, 78};

static const int32_t cukd_scaler_inv_scale_q[17] = {1167, 3277413, 3, 47745, 505066, 148866, 2532650, 222892, 378830, 2415510, 71196, 24624, 4556, 53436, 20900, 436179, 1607034};


#endif /* CUKD_WSNDS_PREPROCESS_INT_METADATA_H */
