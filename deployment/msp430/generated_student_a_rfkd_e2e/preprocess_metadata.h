#ifndef CUKD_WSNDS_PREPROCESS_METADATA_H
#define CUKD_WSNDS_PREPROCESS_METADATA_H

/* Metadata for reproducing the Python WSN-DS v2.3 preprocessing contract.
 * These float scaler constants are for host/gateway preprocessing or audit.
 * The no-FPU integer inference core consumes standardized Q15 vectors.
 */

#define CUKD_PREPROCESS_INPUT_DIM 17
#define CUKD_PREPROCESS_NUM_CLASSES 5

static const float cukd_scaler_mean[17] = {1064.748711502f, 0.115765986f, 274980.411107641f, 22.599380404f, 0.267697999f, 6.940562268f, 0.779905034f, 0.737493361f, 0.288983908f, 0.747451696f, 9.687103809f, 44.857924897f, 73.890044600f, 4.569448114f, 22.562735189f, 2.497956820f, 0.305660780f};

static const float cukd_scaler_scale[17] = {899.644963182f, 0.319944092f, 389910.701382128f, 21.955764807f, 2.061144757f, 7.044309541f, 0.414310478f, 4.691492053f, 2.754742316f, 0.434474002f, 14.681881734f, 42.574407148f, 230.246027848f, 19.679128852f, 50.261536735f, 2.407333394f, 0.669460868f};

static const char *cukd_feature_names[17] = {"Time", "Is_CH", "who CH", "Dist_To_CH", "ADV_S", "ADV_R", "JOIN_S", "JOIN_R", "SCH_S", "SCH_R", "Rank", "DATA_S", "DATA_R", "Data_Sent_To_BS", "dist_CH_To_BS", "send_code", "Expaned Energy"};

static const char *cukd_class_names[5] = {"Blackhole", "Flooding", "Grayhole", "Normal", "TDMA"};


#endif /* CUKD_WSNDS_PREPROCESS_METADATA_H */
