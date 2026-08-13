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

static const int32_t cukd_scaler_mean_q[17] = {272441, 30, 70201885, 5787, 69, 1775, 200, 190, 74, 191, 2484, 11500, 18945, 1160, 5761, 641, 78};

static const int32_t cukd_scaler_inv_scale_q[17] = {1166, 3281864, 3, 47796, 504942, 149074, 2533930, 221754, 378140, 2414990, 71423, 24606, 4542, 53835, 20889, 435352, 1568278};


#endif /* CUKD_WSNDS_PREPROCESS_INT_METADATA_H */
