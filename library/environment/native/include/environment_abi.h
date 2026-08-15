#ifndef SEVERIAN_ENVIRONMENT_ABI_H
#define SEVERIAN_ENVIRONMENT_ABI_H

#ifndef SEVERIAN_C_V1_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
typedef struct { const uint8_t *data; size_t length; } sev_string_view_v1;
#endif

sev_string_view_v1 sev_abi_v1_environment_get(sev_string_view_v1 name, sev_string_view_v1 fallback);
bool sev_abi_v1_environment_set(sev_string_view_v1 name, sev_string_view_v1 value);
bool sev_abi_v1_environment_remove(sev_string_view_v1 name);

#endif
