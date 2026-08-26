#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    size_t length;
} sev_owned_string;

typedef struct {
    size_t length;
    size_t capacity;
    uintptr_t *values;
} sev_string_list;

static void sev_string_list_push(sev_string_list *list, const char *value);

static _Thread_local char sev_conversion_buffer[128];

static char *sev_string_allocation(size_t length) {
    if (length + 1 > SIZE_MAX - sizeof(sev_owned_string)) abort();
    sev_owned_string *allocation = malloc(sizeof(sev_owned_string) + length + 1);
    if (allocation == NULL) abort();
    allocation->length = length;
    return (char *)(allocation + 1);
}

static size_t sev_utf8_width(unsigned char byte) {
    if ((byte & 0x80) == 0) return 1;
    if ((byte & 0xe0) == 0xc0) return 2;
    if ((byte & 0xf0) == 0xe0) return 3;
    if ((byte & 0xf8) == 0xf0) return 4;
    return 1;
}

static size_t sev_utf8_length(const char *value) {
    size_t result = 0;
    for (size_t offset = 0; value[offset] != '\0';) {
        offset += sev_utf8_width((unsigned char)value[offset]);
        ++result;
    }
    return result;
}

static size_t sev_utf8_offset(const char *value, size_t character) {
    size_t offset = 0;
    while (character-- > 0 && value[offset] != '\0') {
        offset += sev_utf8_width((unsigned char)value[offset]);
    }
    return offset;
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

const char *__sev_string_from_pointer(const void *value) {
    snprintf(sev_conversion_buffer, sizeof(sev_conversion_buffer), "%p", value);
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
    return sev_utf8_length(value);
}

const char *__sev_string_index(const char *value, int64_t index) {
    int64_t length = (int64_t)sev_utf8_length(value);
    if (index < 0) index += length;
    if (index < 0 || index >= length) return "";
    size_t offset = sev_utf8_offset(value, (size_t)index);
    size_t width = sev_utf8_width((unsigned char)value[offset]);
    char *result = sev_string_allocation(width);
    memcpy(result, value + offset, width);
    result[width] = '\0';
    return result;
}

void *__sev_string_characters(const char *value) {
    sev_string_list *result = calloc(1, sizeof(sev_string_list));
    if (result == NULL) abort();
    for (size_t offset = 0; value[offset] != '\0';) {
        size_t width = sev_utf8_width((unsigned char)value[offset]);
        char *character = sev_string_allocation(width);
        memcpy(character, value + offset, width);
        character[width] = '\0';
        sev_string_list_push(result, character);
        offset += width;
    }
    return result;
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

const char *__sev_string_lower(const char *value) {
    size_t length = strlen(value);
    char *result = sev_string_allocation(length);
    for (size_t index = 0; index < length; ++index) {
        result[index] = (char)tolower((unsigned char)value[index]);
    }
    result[length] = '\0';
    return result;
}

const char *__sev_string_strip(const char *value) {
    const char *start = value;
    while (*start != '\0' && isspace((unsigned char)*start)) ++start;
    const char *end = value + strlen(value);
    while (end > start && isspace((unsigned char)end[-1])) --end;
    size_t length = (size_t)(end - start);
    char *result = sev_string_allocation(length);
    memcpy(result, start, length);
    result[length] = '\0';
    return result;
}

static void sev_string_list_push(sev_string_list *list, const char *value) {
    if (list->length == list->capacity) {
        size_t capacity = list->capacity == 0 ? 4 : list->capacity * 2;
        uintptr_t *values = realloc(list->values, capacity * sizeof(uintptr_t));
        if (values == NULL) abort();
        list->values = values;
        list->capacity = capacity;
    }
    list->values[list->length++] = (uintptr_t)value;
}

void *__sev_string_split(const char *value, const char *separator) {
    sev_string_list *result = calloc(1, sizeof(sev_string_list));
    if (result == NULL) abort();
    size_t separator_length = strlen(separator);
    if (separator_length == 0) return result;
    const char *start = value;
    for (;;) {
        const char *found = strstr(start, separator);
        const char *end = found == NULL ? value + strlen(value) : found;
        size_t length = (size_t)(end - start);
        char *piece = sev_string_allocation(length);
        memcpy(piece, start, length);
        piece[length] = '\0';
        sev_string_list_push(result, piece);
        if (found == NULL) break;
        start = found + separator_length;
    }
    return result;
}

const char *__sev_list_join(void *storage, const char *separator) {
    sev_string_list *list = storage;
    size_t separator_length = strlen(separator);
    size_t length = list->length > 0 ? (list->length - 1) * separator_length : 0;
    for (size_t index = 0; index < list->length; ++index) {
        const char *value = (const char *)list->values[index];
        if (value != NULL) length += strlen(value);
    }
    char *result = sev_string_allocation(length);
    size_t offset = 0;
    for (size_t index = 0; index < list->length; ++index) {
        if (index != 0) {
            memcpy(result + offset, separator, separator_length);
            offset += separator_length;
        }
        const char *value = (const char *)list->values[index];
        if (value != NULL) {
            size_t value_length = strlen(value);
            memcpy(result + offset, value, value_length);
            offset += value_length;
        }
    }
    result[offset] = '\0';
    return result;
}

const char *__sev_string_replace(const char *value, const char *needle, const char *replacement) {
    size_t needle_length = strlen(needle);
    if (needle_length == 0) return __sev_string_identity(value);
    size_t replacement_length = strlen(replacement);
    size_t count = 0;
    for (const char *cursor = value; (cursor = strstr(cursor, needle)) != NULL;
         cursor += needle_length) ++count;
    size_t value_length = strlen(value);
    size_t length = value_length + count * replacement_length - count * needle_length;
    char *result = sev_string_allocation(length);
    const char *cursor = value;
    size_t offset = 0;
    for (;;) {
        const char *found = strstr(cursor, needle);
        if (found == NULL) break;
        size_t prefix = (size_t)(found - cursor);
        memcpy(result + offset, cursor, prefix);
        offset += prefix;
        memcpy(result + offset, replacement, replacement_length);
        offset += replacement_length;
        cursor = found + needle_length;
    }
    strcpy(result + offset, cursor);
    return result;
}

_Bool __sev_string_starts_with(const char *value, const char *needle) {
    return strncmp(value, needle, strlen(needle)) == 0;
}

_Bool __sev_string_ends_with(const char *value, const char *needle) {
    size_t value_length = strlen(value);
    size_t needle_length = strlen(needle);
    return needle_length <= value_length
        && strcmp(value + value_length - needle_length, needle) == 0;
}

int64_t __sev_string_find(const char *value, const char *needle) {
    const char *found = strstr(value, needle);
    return found == NULL ? -1 : (int64_t)(found - value);
}

int64_t __sev_string_count(const char *value, const char *needle) {
    size_t needle_length = strlen(needle);
    if (needle_length == 0) return 0;
    int64_t count = 0;
    for (const char *cursor = value; (cursor = strstr(cursor, needle)) != NULL;
         cursor += needle_length) ++count;
    return count;
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

const char *__sev_string_slice_ex(
    const char *value,
    int64_t start,
    int64_t end,
    int64_t step,
    _Bool has_start,
    _Bool has_end,
    _Bool start_exclusive,
    _Bool end_inclusive
) {
    int64_t length = (int64_t)sev_utf8_length(value);
    if (step == 0) abort();
    if (!has_start) start = step > 0 ? 0 : length - 1;
    else if (start < 0) start += length;
    if (!has_end) end = step > 0 ? length : -1;
    else if (end < 0) end += length;
    if (start_exclusive) start += step > 0 ? 1 : -1;
    if (end_inclusive) end += step > 0 ? 1 : -1;
    if (step > 0) {
        if (start < 0) start = 0;
        if (start > length) start = length;
        if (end < 0) end = 0;
        if (end > length) end = length;
    } else {
        if (start >= length) start = length - 1;
        if (end >= length) end = length - 1;
    }
    size_t capacity = strlen(value);
    char *result = sev_string_allocation(capacity);
    size_t output = 0;
    if (step > 0) {
        for (int64_t index = start; index < end; index += step) {
            size_t offset = sev_utf8_offset(value, (size_t)index);
            size_t width = sev_utf8_width((unsigned char)value[offset]);
            memcpy(result + output, value + offset, width);
            output += width;
        }
    } else {
        for (int64_t index = start; index > end && index >= 0; index += step) {
            size_t offset = sev_utf8_offset(value, (size_t)index);
            size_t width = sev_utf8_width((unsigned char)value[offset]);
            memcpy(result + output, value + offset, width);
            output += width;
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
