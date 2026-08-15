#include "regex_abi.h"

#include <regex.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char **items;
    size_t length;
    size_t capacity;
} sev_regex_strings;

typedef struct {
    char *data;
    size_t length;
} sev_regex_text;

typedef struct {
    char *data;
    size_t length;
    size_t capacity;
} sev_regex_buffer;

static char *sev_regex_copy_view(sev_string_view_v1 view) {
    if (!view.data && view.length) return NULL;
    char *copy = malloc(view.length + 1);
    if (!copy) return NULL;
    if (view.length) memcpy(copy, view.data, view.length);
    copy[view.length] = '\0';
    return copy;
}

static sev_string_view_v1 sev_regex_view(const char *data, size_t length) {
    sev_string_view_v1 view = {
        .data = (const uint8_t *)data,
        .length = data ? length : 0,
    };
    return view;
}

static sev_regex_strings *sev_regex_strings_new(void) {
    return calloc(1, sizeof(sev_regex_strings));
}

static int sev_regex_strings_push(sev_regex_strings *values, const char *data, size_t length) {
    if (values->length == values->capacity) {
        size_t capacity = values->capacity ? values->capacity * 2 : 8;
        char **items = realloc(values->items, capacity * sizeof(*items));
        if (!items) return -1;
        values->items = items;
        values->capacity = capacity;
    }
    char *item = malloc(length + 1);
    if (!item) return -1;
    if (length) memcpy(item, data, length);
    item[length] = '\0';
    values->items[values->length++] = item;
    return 0;
}

static void sev_regex_strings_destroy(sev_regex_strings *values) {
    if (!values) return;
    for (size_t index = 0; index < values->length; ++index) free(values->items[index]);
    free(values->items);
    free(values);
}

static int sev_regex_buffer_reserve(sev_regex_buffer *buffer, size_t extra) {
    if (extra > SIZE_MAX - buffer->length - 1) return -1;
    size_t required = buffer->length + extra + 1;
    if (required <= buffer->capacity) return 0;
    size_t capacity = buffer->capacity ? buffer->capacity : 64;
    while (capacity < required) {
        if (capacity > SIZE_MAX / 2) {
            capacity = required;
            break;
        }
        capacity *= 2;
    }
    char *data = realloc(buffer->data, capacity);
    if (!data) return -1;
    buffer->data = data;
    buffer->capacity = capacity;
    return 0;
}

static int sev_regex_buffer_append(sev_regex_buffer *buffer, const char *data, size_t length) {
    if (sev_regex_buffer_reserve(buffer, length) != 0) return -1;
    if (length) memcpy(buffer->data + buffer->length, data, length);
    buffer->length += length;
    buffer->data[buffer->length] = '\0';
    return 0;
}

bool sev_abi_v1_regex_matches(sev_string_view_v1 text, sev_string_view_v1 pattern) {
    char *text_value = sev_regex_copy_view(text);
    char *pattern_value = sev_regex_copy_view(pattern);
    if (!text_value || !pattern_value) {
        free(text_value);
        free(pattern_value);
        return false;
    }
    regex_t expression;
    if (regcomp(&expression, pattern_value, REG_EXTENDED | REG_NOSUB) != 0) {
        free(text_value);
        free(pattern_value);
        return false;
    }
    bool matched = regexec(&expression, text_value, 0, NULL, 0) == 0;
    regfree(&expression);
    free(text_value);
    free(pattern_value);
    return matched;
}

int32_t sev_abi_v1_regex_find_all(
    sev_string_view_v1 text,
    sev_string_view_v1 pattern,
    sev_handle_v1 *output
) {
    if (!output) return -1;
    output->value = NULL;
    char *text_value = sev_regex_copy_view(text);
    char *pattern_value = sev_regex_copy_view(pattern);
    sev_regex_strings *values = sev_regex_strings_new();
    if (!text_value || !pattern_value || !values) {
        free(text_value);
        free(pattern_value);
        sev_regex_strings_destroy(values);
        return -1;
    }
    regex_t expression;
    if (regcomp(&expression, pattern_value, REG_EXTENDED) == 0) {
        size_t offset = 0;
        regmatch_t match;
        while (offset <= text.length && regexec(&expression, text_value + offset, 1, &match, 0) == 0) {
            size_t start = offset + (size_t)match.rm_so;
            size_t end = offset + (size_t)match.rm_eo;
            if (sev_regex_strings_push(values, text_value + start, end - start) != 0) {
                regfree(&expression);
                free(text_value);
                free(pattern_value);
                sev_regex_strings_destroy(values);
                return -1;
            }
            if (end > offset) {
                offset = end;
            } else if (offset < text.length) {
                offset += 1;
            } else {
                break;
            }
        }
        regfree(&expression);
    }
    free(text_value);
    free(pattern_value);
    output->value = values;
    return 0;
}

int32_t sev_abi_v1_regex_split(
    sev_string_view_v1 text,
    sev_string_view_v1 pattern,
    sev_handle_v1 *output
) {
    if (!output) return -1;
    output->value = NULL;
    char *text_value = sev_regex_copy_view(text);
    char *pattern_value = sev_regex_copy_view(pattern);
    sev_regex_strings *values = sev_regex_strings_new();
    if (!text_value || !pattern_value || !values) {
        free(text_value);
        free(pattern_value);
        sev_regex_strings_destroy(values);
        return -1;
    }
    regex_t expression;
    int compiled = regcomp(&expression, pattern_value, REG_EXTENDED);
    if (compiled != 0) {
        if (sev_regex_strings_push(values, text_value, text.length) != 0) {
            free(text_value);
            free(pattern_value);
            sev_regex_strings_destroy(values);
            return -1;
        }
    } else {
        size_t offset = 0;
        size_t segment = 0;
        regmatch_t match;
        while (offset <= text.length && regexec(&expression, text_value + offset, 1, &match, 0) == 0) {
            size_t start = offset + (size_t)match.rm_so;
            size_t end = offset + (size_t)match.rm_eo;
            if (sev_regex_strings_push(values, text_value + segment, start - segment) != 0) {
                regfree(&expression);
                free(text_value);
                free(pattern_value);
                sev_regex_strings_destroy(values);
                return -1;
            }
            segment = end;
            if (end > offset) {
                offset = end;
            } else if (offset < text.length) {
                offset += 1;
            } else {
                break;
            }
        }
        if (sev_regex_strings_push(values, text_value + segment, text.length - segment) != 0) {
            regfree(&expression);
            free(text_value);
            free(pattern_value);
            sev_regex_strings_destroy(values);
            return -1;
        }
        regfree(&expression);
    }
    free(text_value);
    free(pattern_value);
    output->value = values;
    return 0;
}

size_t sev_abi_v1_regex_strings_length(sev_handle_v1 handle) {
    const sev_regex_strings *values = handle.value;
    return values ? values->length : 0;
}

sev_string_view_v1 sev_abi_v1_regex_strings_at(sev_handle_v1 handle, size_t index) {
    const sev_regex_strings *values = handle.value;
    if (!values || index >= values->length) return sev_regex_view(NULL, 0);
    return sev_regex_view(values->items[index], strlen(values->items[index]));
}

void sev_abi_v1_regex_strings_release(sev_handle_v1 handle) {
    sev_regex_strings_destroy(handle.value);
}

int32_t sev_abi_v1_regex_substitute(
    sev_string_view_v1 text,
    sev_string_view_v1 pattern,
    sev_string_view_v1 replacement,
    sev_handle_v1 *output
) {
    if (!output) return -1;
    output->value = NULL;
    char *text_value = sev_regex_copy_view(text);
    char *pattern_value = sev_regex_copy_view(pattern);
    char *replacement_value = sev_regex_copy_view(replacement);
    sev_regex_text *value = calloc(1, sizeof(*value));
    if (!text_value || !pattern_value || !replacement_value || !value) {
        free(text_value);
        free(pattern_value);
        free(replacement_value);
        free(value);
        return -1;
    }

    sev_regex_buffer buffer = {0};
    regex_t expression;
    int compiled = regcomp(&expression, pattern_value, REG_EXTENDED);
    if (compiled != 0) {
        if (sev_regex_buffer_append(&buffer, text_value, text.length) != 0) goto failure;
    } else {
        size_t offset = 0;
        size_t segment = 0;
        regmatch_t match;
        while (offset <= text.length && regexec(&expression, text_value + offset, 1, &match, 0) == 0) {
            size_t start = offset + (size_t)match.rm_so;
            size_t end = offset + (size_t)match.rm_eo;
            if (sev_regex_buffer_append(&buffer, text_value + segment, start - segment) != 0 ||
                sev_regex_buffer_append(&buffer, replacement_value, replacement.length) != 0) {
                regfree(&expression);
                goto failure;
            }
            segment = end;
            if (end > offset) {
                offset = end;
            } else if (offset < text.length) {
                offset += 1;
            } else {
                break;
            }
        }
        if (sev_regex_buffer_append(&buffer, text_value + segment, text.length - segment) != 0) {
            regfree(&expression);
            goto failure;
        }
        regfree(&expression);
    }
    if (!buffer.data && sev_regex_buffer_append(&buffer, "", 0) != 0) goto failure;
    value->data = buffer.data;
    value->length = buffer.length;
    free(text_value);
    free(pattern_value);
    free(replacement_value);
    output->value = value;
    return 0;

failure:
    free(buffer.data);
    free(text_value);
    free(pattern_value);
    free(replacement_value);
    free(value);
    return -1;
}

sev_string_view_v1 sev_abi_v1_regex_text_value(sev_handle_v1 handle) {
    const sev_regex_text *value = handle.value;
    return value ? sev_regex_view(value->data, value->length) : sev_regex_view(NULL, 0);
}

void sev_abi_v1_regex_text_release(sev_handle_v1 handle) {
    sev_regex_text *value = handle.value;
    if (!value) return;
    free(value->data);
    free(value);
}
