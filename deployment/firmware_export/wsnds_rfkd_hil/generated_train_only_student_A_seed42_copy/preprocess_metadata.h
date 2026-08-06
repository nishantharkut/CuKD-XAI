#ifndef CUKD_WSNDS_PREPROCESS_METADATA_H
#define CUKD_WSNDS_PREPROCESS_METADATA_H

/* Metadata for reproducing the Python WSN-DS v2.3 preprocessing contract.
 * These float scaler constants are for host/gateway preprocessing or audit.
 * The no-FPU integer inference core consumes standardized calibrated-int16 vectors.
 */

#define CUKD_PREPROCESS_INPUT_DIM 17
#define CUKD_PREPROCESS_NUM_CLASSES 5

static const float cukd_scaler_mean[17] = {1062.774034135f, 0.115762702f, 274225.662290469f, 22.643447998f, 0.268386895f, 6.934299834f, 0.780329607f, 0.736406205f, 0.289088358f, 0.748104876f, 9.732047801f, 44.866681665f, 73.758995165f, 4.540533533f, 22.466167168f, 2.497357503f, 0.305203347f};

static const float cukd_scaler_scale[17] = {898.464833316f, 0.319940148f, 388720.930764341f, 21.961827766f, 2.076118663f, 7.043776707f, 0.414023323f, 4.704404777f, 2.767934143f, 0.434101337f, 14.728095483f, 42.583218772f, 230.145325795f, 19.623041455f, 50.172224031f, 2.404003342f, 0.652491330f};

static const char *cukd_feature_names[17] = {"Time", "Is_CH", "who CH", "Dist_To_CH", "ADV_S", "ADV_R", "JOIN_S", "JOIN_R", "SCH_S", "SCH_R", "Rank", "DATA_S", "DATA_R", "Data_Sent_To_BS", "dist_CH_To_BS", "send_code", "Expaned Energy"};

static const char *cukd_class_names[5] = {"Blackhole", "Flooding", "Grayhole", "Normal", "TDMA"};


#endif /* CUKD_WSNDS_PREPROCESS_METADATA_H */
