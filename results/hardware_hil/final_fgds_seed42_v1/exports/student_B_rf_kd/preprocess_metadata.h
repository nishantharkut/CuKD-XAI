#ifndef CUKD_WSNDS_PREPROCESS_METADATA_H
#define CUKD_WSNDS_PREPROCESS_METADATA_H

/* Metadata for reproducing the Python WSN-DS v2.3 preprocessing contract.
 * These float scaler constants are for host/gateway preprocessing or audit.
 * The no-FPU integer inference core consumes standardized calibrated-int16 vectors.
 */

#define CUKD_PREPROCESS_INPUT_DIM 17
#define CUKD_PREPROCESS_NUM_CLASSES 5

static const float cukd_scaler_mean[17] = {1064.224380904f, 0.115401778f, 274226.113941044f, 22.605620703f, 0.268397426f, 6.933980938f, 0.780638222f, 0.744116065f, 0.289911784f, 0.747941433f, 9.702551898f, 44.923652826f, 74.004225830f, 4.530890895f, 22.503704524f, 2.505471077f, 0.305023654f};

static const float cukd_scaler_scale[17] = {899.559534880f, 0.319506193f, 388132.161387414f, 21.938620872f, 2.076625208f, 7.033945599f, 0.413814195f, 4.728564913f, 2.772980202f, 0.434194709f, 14.681246650f, 42.615144771f, 230.867888950f, 19.477624823f, 50.197784612f, 2.408568170f, 0.668616218f};

static const char *cukd_feature_names[17] = {"Time", "Is_CH", "who CH", "Dist_To_CH", "ADV_S", "ADV_R", "JOIN_S", "JOIN_R", "SCH_S", "SCH_R", "Rank", "DATA_S", "DATA_R", "Data_Sent_To_BS", "dist_CH_To_BS", "send_code", "Expaned Energy"};

static const char *cukd_class_names[5] = {"Blackhole", "Flooding", "Grayhole", "Normal", "TDMA"};


#endif /* CUKD_WSNDS_PREPROCESS_METADATA_H */
