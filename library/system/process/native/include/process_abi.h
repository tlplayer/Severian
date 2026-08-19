#ifndef SEVERIAN_PROCESS_ABI_H
#define SEVERIAN_PROCESS_ABI_H

#ifndef SEVERIAN_C_V1_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
typedef struct { const uint8_t *data; size_t length; } sev_string_view_v1;
typedef struct { void *value; } sev_handle_v1;
#endif

int64_t sev_abi_v1_process_run(sev_string_view_v1 command);
int64_t sev_abi_v1_process_spawn(sev_string_view_v1 command);
int64_t sev_abi_v1_process_wait(int64_t process);
bool sev_abi_v1_process_kill(int64_t process);
void sev_abi_v1_process_exit(int64_t status);
int32_t sev_abi_v1_process_arguments(sev_handle_v1 *output);
size_t sev_abi_v1_process_arguments_length(sev_handle_v1 arguments);
sev_string_view_v1 sev_abi_v1_process_arguments_at(sev_handle_v1 arguments, size_t index);
void sev_abi_v1_process_arguments_release(sev_handle_v1 arguments);

#endif
