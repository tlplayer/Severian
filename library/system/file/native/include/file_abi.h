#ifndef SEVERIAN_FILE_ABI_H
#define SEVERIAN_FILE_ABI_H

#ifndef SEVERIAN_C_V1_H
#include <stddef.h>
#include <stdint.h>
typedef struct { const uint8_t *data; size_t length; } sev_string_view_v1;
typedef struct { void *value; } sev_handle_v1;
typedef struct { int32_t code; sev_string_view_v1 message; } sev_error_v1;
#endif

int32_t sev_abi_v1_file_read_text(
    sev_string_view_v1 path,
    sev_handle_v1 *content,
    sev_error_v1 *error
);
sev_string_view_v1 sev_abi_v1_file_text_value(sev_handle_v1 content);
void sev_abi_v1_file_text_release(sev_handle_v1 content);

#endif
