#ifndef CUKD_WIFI_CONFIG_H
#define CUKD_WIFI_CONFIG_H

#include <stddef.h>
#include <stdint.h>

#define CUKD_WIFI_SSID_MAX 32
#define CUKD_WIFI_PASSWORD_MAX 63
#define CUKD_WIFI_SESSION_HEX_LENGTH 32
#define CUKD_WIFI_CONFIG_LINE_MAX 256

typedef enum {
    CUKD_WIFI_CONFIG_OK = 0,
    CUKD_WIFI_CONFIG_BAD_PREFIX = 1,
    CUKD_WIFI_CONFIG_BAD_LENGTH = 2,
    CUKD_WIFI_CONFIG_BAD_TEXT = 3,
    CUKD_WIFI_CONFIG_BAD_CHECKSUM = 4,
    CUKD_WIFI_CONFIG_BAD_SESSION = 5,
    CUKD_WIFI_CONFIG_BAD_PORT = 6,
    CUKD_WIFI_CONFIG_BAD_SSID = 7,
    CUKD_WIFI_CONFIG_BAD_PASSWORD = 8
} cukd_wifi_config_status_t;

typedef struct {
    char session_id[CUKD_WIFI_SESSION_HEX_LENGTH + 1];
    char ssid[CUKD_WIFI_SSID_MAX + 1];
    char password[CUKD_WIFI_PASSWORD_MAX + 1];
    uint16_t udp_port;
} cukd_wifi_config_t;

const char *cukd_wifi_config_status_name(cukd_wifi_config_status_t status);
cukd_wifi_config_status_t cukd_parse_wifi_config_line(
    const uint8_t *data,
    size_t data_length,
    cukd_wifi_config_t *config
);
void cukd_clear_wifi_config(cukd_wifi_config_t *config);

#endif
