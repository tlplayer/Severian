#include "environment_abi.h"

#include <stdlib.h>
#include <string.h>

static char *sev_copy_view(sev_string_view_v1 view) {
    if (!view.data && view.length) return NULL;
    char *copy = malloc(view.length + 1);
    if (!copy) return NULL;
    if (view.length) memcpy(copy, view.data, view.length);
    copy[view.length] = '\0';
    return copy;
}

static sev_string_view_v1 sev_view(const char *value) {
    sev_string_view_v1 view = {
        .data = (const uint8_t *)value,
        .length = value ? strlen(value) : 0,
    };
    return view;
}

sev_string_view_v1 sev_abi_v1_environment_get(
    sev_string_view_v1 name,
    sev_string_view_v1 fallback
) {
    char *name_text = sev_copy_view(name);
    if (!name_text) return fallback;
    const char *value = getenv(name_text);
    free(name_text);
    return value ? sev_view(value) : fallback;
}

bool sev_abi_v1_environment_set(sev_string_view_v1 name, sev_string_view_v1 value) {
    char *name_text = sev_copy_view(name);
    char *value_text = sev_copy_view(value);
    if (!name_text || !value_text) {
        free(name_text);
        free(value_text);
        return false;
    }
    bool success = setenv(name_text, value_text, 1) == 0;
    free(name_text);
    free(value_text);
    return success;
}

bool sev_abi_v1_environment_remove(sev_string_view_v1 name) {
    char *name_text = sev_copy_view(name);
    if (!name_text) return false;
    bool success = unsetenv(name_text) == 0;
    free(name_text);
    return success;
}
