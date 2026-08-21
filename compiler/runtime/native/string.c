#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    size_t length;
} sev_owned_string;

const char *__sev_string_concat(const char *left, const char *right) {
    size_t left_length = strlen(left);
    size_t right_length = strlen(right);
    if (right_length > SIZE_MAX - left_length - 1) abort();
    if (left_length + right_length + 1 > SIZE_MAX - sizeof(sev_owned_string)) abort();
    sev_owned_string *allocation =
        malloc(sizeof(sev_owned_string) + left_length + right_length + 1);
    if (allocation == NULL) abort();
    allocation->length = left_length + right_length;
    char *result = (char *)(allocation + 1);
    memcpy(result, left, left_length);
    memcpy(result + left_length, right, right_length + 1);
    return result;
}

int32_t __sev_string_compare(const char *left, const char *right) {
    return strcmp(left, right);
}

void __sev_string_release(const char *value) {
    if (value == NULL) return;
    free(((sev_owned_string *)value) - 1);
}
