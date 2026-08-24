#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    size_t length;
} sev_owned_string;

static _Thread_local char sev_conversion_buffer[128];

static char *sev_string_allocation(size_t length) {
    if (length + 1 > SIZE_MAX - sizeof(sev_owned_string)) abort();
    sev_owned_string *allocation = malloc(sizeof(sev_owned_string) + length + 1);
    if (allocation == NULL) abort();
    allocation->length = length;
    return (char *)(allocation + 1);
}

const char *__sev_string_identity(const char *value) {
    return value;
}

const char *__sev_string_from_int(int64_t value) {
    snprintf(sev_conversion_buffer, sizeof(sev_conversion_buffer), "%lld", (long long)value);
    return sev_conversion_buffer;
}

const char *__sev_string_from_float(double value) {
    snprintf(sev_conversion_buffer, sizeof(sev_conversion_buffer), "%.15g", value);
    return sev_conversion_buffer;
}

double __sev_float_from_string(const char *value) {
    return strtod(value, NULL);
}

const char *__sev_string_from_bool(_Bool value) {
    return value ? "true" : "false";
}

const char *__sev_string_from_char(uint32_t value) {
    if (value <= 0x7f) {
        sev_conversion_buffer[0] = (char)value;
        sev_conversion_buffer[1] = '\0';
    } else {
        sev_conversion_buffer[0] = '?';
        sev_conversion_buffer[1] = '\0';
    }
    return sev_conversion_buffer;
}

const char *__sev_string_from_usize(uintptr_t value) {
    snprintf(sev_conversion_buffer, sizeof(sev_conversion_buffer), "%llu", (unsigned long long)value);
    return sev_conversion_buffer;
}

const char *__sev_type_string(const char *value) {
    (void)value;
    return "string";
}

const char *__sev_type_int(int64_t value) {
    (void)value;
    return "int";
}

const char *__sev_type_float(double value) {
    (void)value;
    return "float";
}

const char *__sev_type_bool(_Bool value) {
    (void)value;
    return "bool";
}

const char *__sev_type_char(uint32_t value) {
    (void)value;
    return "char";
}

const char *__sev_type_usize(uintptr_t value) {
    (void)value;
    return "usize";
}

uintptr_t __sev_string_length(const char *value) {
    return strlen(value);
}

_Bool __sev_string_is_present(const char *value) {
    return value != NULL && value[0] != '\0';
}

_Bool __sev_string_is_empty(const char *value) {
    return value == NULL || value[0] == '\0';
}

const char *__sev_string_upper(const char *value) {
    size_t length = strlen(value);
    char *result = sev_string_allocation(length);
    for (size_t index = 0; index < length; ++index) {
        result[index] = (char)toupper((unsigned char)value[index]);
    }
    result[length] = '\0';
    return result;
}

_Bool __sev_string_contains(const char *value, const char *needle) {
    return strstr(value, needle) != NULL;
}

const char *__sev_string_slice(const char *value, int64_t start, int64_t end, int64_t step) {
    int64_t length = (int64_t)strlen(value);
    if (step == 0) abort();
    if (start < 0) start += length;
    if (end < 0) end += length;
    if (step > 0) {
        if (start < 0) start = 0;
        if (start > length) start = length;
        if (end < 0) end = 0;
        if (end > length) end = length;
    } else {
        if (start >= length) start = length - 1;
        if (end >= length) end = length - 1;
    }
    size_t result_length = 0;
    if (step > 0) {
        for (int64_t index = start; index < end; index += step) ++result_length;
    } else {
        for (int64_t index = start; index > end && index >= 0; index += step) ++result_length;
    }
    char *result = sev_string_allocation(result_length);
    size_t output = 0;
    if (step > 0) {
        for (int64_t index = start; index < end; index += step) result[output++] = value[index];
    } else {
        for (int64_t index = start; index > end && index >= 0; index += step) {
            result[output++] = value[index];
        }
    }
    result[output] = '\0';
    return result;
}

const char *__sev_string_concat(const char *left, const char *right) {
    size_t left_length = strlen(left);
    size_t right_length = strlen(right);
    if (right_length > SIZE_MAX - left_length - 1) abort();
    if (left_length + right_length + 1 > SIZE_MAX - sizeof(sev_owned_string)) abort();
    char *result = sev_string_allocation(left_length + right_length);
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
