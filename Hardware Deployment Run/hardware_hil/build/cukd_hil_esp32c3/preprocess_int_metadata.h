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

static const int32_t cukd_scaler_mean_q[17] = {272576, 30, 70394985, 5785, 69, 1777, 200, 189, 74, 191, 2480, 11484, 18916, 1170, 5776, 639, 78};

static const int32_t cukd_scaler_inv_scale_q[17] = {1166, 3277373, 3, 47759, 508735, 148854, 2530894, 223506, 380644, 2413438, 71420, 24629, 4554, 53284, 20862, 435576, 1566299};


#endif /* CUKD_WSNDS_PREPROCESS_INT_METADATA_H */
