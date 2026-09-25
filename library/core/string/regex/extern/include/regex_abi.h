#ifndef SEVERIAN_REGEX_ABI_H
#define SEVERIAN_REGEX_ABI_H

#ifndef SEVERIAN_C_V1_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
typedef struct { const uint8_t *data; size_t length; } sev_string_view_v1;
typedef struct { void *value; } sev_handle_v1;
#endif

bool sev_abi_v1_regex_matches(sev_string_view_v1 text, sev_string_view_v1 pattern);
int32_t sev_abi_v1_regex_find_all(
    sev_string_view_v1 text,
    sev_string_view_v1 pattern,
    sev_handle_v1 *values
);
int32_t sev_abi_v1_regex_split(
    sev_string_view_v1 text,
    sev_string_view_v1 pattern,
    sev_handle_v1 *values
);
size_t sev_abi_v1_regex_strings_length(sev_handle_v1 values);
sev_string_view_v1 sev_abi_v1_regex_strings_at(sev_handle_v1 values, size_t index);
void sev_abi_v1_regex_strings_release(sev_handle_v1 values);
int32_t sev_abi_v1_regex_substitute(
    sev_string_view_v1 text,
    sev_string_view_v1 pattern,
    sev_string_view_v1 replacement,
    sev_handle_v1 *value
);
sev_string_view_v1 sev_abi_v1_regex_text_value(sev_handle_v1 value);
void sev_abi_v1_regex_text_release(sev_handle_v1 value);

#endif
