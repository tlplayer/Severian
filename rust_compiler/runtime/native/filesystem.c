#include "../../../library/core/memory/native/memory.h"
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>

void *__sev_list_create(void);
void __sev_list_push_bool(void *storage, _Bool value);
void __sev_list_push_ptr(void *storage, const char *value);
void __sev_list_push_u8(void *storage, uint8_t value);
uintptr_t __sev_list_len(void *storage);
uint8_t __sev_list_index_u8(void *storage, int64_t index);

typedef struct sev_csv_any {
    int64_t tag;
    int64_t payload;
} sev_csv_any;

typedef struct sev_csv_list {
    void *storage;
} sev_csv_list;

void __sev_list_push_any(void *storage, sev_csv_any value);
void __sev_list_push_list(void *storage, sev_csv_list value);
void __sev_list_set_any(void *storage, int64_t index, sev_csv_any value);

extern sev_csv_any __sev_any_from_string(const char *value);

typedef struct {
    char *bytes;
    size_t length;
    size_t capacity;
} sev_csv_field;

static void sev_csv_field_push(sev_csv_field *field, char value) {
    if (field->length == field->capacity) {
        size_t capacity = field->capacity == 0 ? 64 : field->capacity * 2;
        if (capacity < field->capacity) abort();
        char *bytes = sev_memory_resize(field->bytes, capacity);
        if (bytes == NULL) abort();
        field->bytes = bytes;
        field->capacity = capacity;
    }
    field->bytes[field->length++] = value;
}

static char *sev_csv_field_copy(const sev_csv_field *field, _Bool trim) {
    size_t start = 0;
    size_t end = field->length;
    if (trim) {
        while (start < end && (field->bytes[start] == ' ' || field->bytes[start] == '\t')) ++start;
        while (end > start && (field->bytes[end - 1] == ' ' || field->bytes[end - 1] == '\t')) --end;
    }
    char *value = sev_memory_allocate(end - start + 1);
    if (value == NULL) abort();
    memcpy(value, field->bytes + start, end - start);
    value[end - start] = '\0';
    return value;
}

static void sev_csv_finish_field(
    void *columns,
    void *row,
    sev_csv_field *field,
    size_t record,
    size_t *field_count
) {
    char *value = sev_csv_field_copy(field, record == 0);
    if (record == 0) {
        if (columns != NULL) __sev_list_push_ptr(columns, value);
        else sev_memory_release(value);
    } else {
        __sev_list_push_any(row, __sev_any_from_string(value));
    }
    field->length = 0;
    *field_count += 1;
}

static void *sev_csv_parse(const char *source, _Bool columns_only) {
    void *result = __sev_list_create();
    void *row = NULL;
    sev_csv_field field = {0};
    size_t record = 0;
    size_t width = 0;
    size_t field_count = 0;
    _Bool quoted = 0;
    _Bool record_started = 0;
    const char *cursor = source == NULL ? "" : source;
    for (;;) {
        char character = *cursor;
        if (character == '"') {
            record_started = 1;
            if (quoted && cursor[1] == '"') {
                sev_csv_field_push(&field, '"');
                cursor += 2;
                continue;
            }
            quoted = !quoted;
            ++cursor;
            continue;
        }
        if (character == '\0' || (!quoted && (character == ',' || character == '\n'))) {
            if (character == ',' || record_started || field.length > 0 || field_count > 0) {
                if (record > 0 && row == NULL) row = __sev_list_create();
                sev_csv_finish_field(
                    columns_only ? result : NULL,
                    row,
                    &field,
                    record,
                    &field_count
                );
            }
            if (character == ',') {
                record_started = 1;
                ++cursor;
                continue;
            }
            if (field_count > 0) {
                if (record == 0) {
                    width = field_count;
                    if (columns_only) {
                        sev_memory_release(field.bytes);
                        return result;
                    }
                } else {
                    while (field_count < width) {
                        __sev_list_push_any(row, __sev_any_from_string(sev_memory_copy_text("")));
                        ++field_count;
                    }
                    sev_csv_list list = {row};
                    __sev_list_push_list(result, list);
                }
                ++record;
            }
            row = NULL;
            field_count = 0;
            record_started = 0;
            if (character == '\0') break;
            ++cursor;
            continue;
        }
        if (!quoted && character == '\r' && cursor[1] == '\n') {
            ++cursor;
            continue;
        }
        record_started = 1;
        sev_csv_field_push(&field, character);
        ++cursor;
    }
    sev_memory_release(field.bytes);
    return result;
}

void *__sev_csv_columns(const char *source) {
    return sev_csv_parse(source, 1);
}

void *__sev_csv_rows(const char *source) {
    return sev_csv_parse(source, 0);
}
