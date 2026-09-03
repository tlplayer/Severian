#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>

enum { SEV_PATH_CAPACITY = 4096 };

static _Thread_local char sev_join_buffer[SEV_PATH_CAPACITY];
static _Thread_local char sev_basename_buffer[SEV_PATH_CAPACITY];
static _Thread_local char sev_dirname_buffer[SEV_PATH_CAPACITY];
static _Thread_local char sev_extension_buffer[SEV_PATH_CAPACITY];
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
        char *bytes = realloc(field->bytes, capacity);
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
    char *value = malloc(end - start + 1);
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
        else free(value);
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
                        free(field.bytes);
                        return result;
                    }
                } else {
                    while (field_count < width) {
                        __sev_list_push_any(row, __sev_any_from_string(strdup("")));
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
    free(field.bytes);
    return result;
}

void *__sev_csv_columns(const char *source) {
    return sev_csv_parse(source, 1);
}

void *__sev_csv_rows(const char *source) {
    return sev_csv_parse(source, 0);
}

typedef struct {
    char **values;
    size_t length;
    size_t capacity;
} sev_json_columns;

static const char *sev_json_space(const char *cursor) {
    while (*cursor == ' ' || *cursor == '\t' || *cursor == '\r' || *cursor == '\n') ++cursor;
    return cursor;
}

static char *sev_json_string(const char **position) {
    const char *cursor = *position;
    if (*cursor != '"') return strdup("");
    ++cursor;
    sev_csv_field field = {0};
    while (*cursor != '\0' && *cursor != '"') {
        char value = *cursor++;
        if (value == '\\' && *cursor != '\0') {
            char escaped = *cursor++;
            switch (escaped) {
                case 'n': value = '\n'; break;
                case 'r': value = '\r'; break;
                case 't': value = '\t'; break;
                case 'b': value = '\b'; break;
                case 'f': value = '\f'; break;
                case '"': value = '"'; break;
                case '\\': value = '\\'; break;
                case '/': value = '/'; break;
                default: value = escaped; break;
            }
        }
        sev_csv_field_push(&field, value);
    }
    if (*cursor == '"') ++cursor;
    *position = cursor;
    char *result = sev_csv_field_copy(&field, 0);
    free(field.bytes);
    return result;
}

static char *sev_json_value(const char **position) {
    const char *cursor = sev_json_space(*position);
    if (*cursor == '"') {
        char *value = sev_json_string(&cursor);
        *position = cursor;
        return value;
    }
    const char *start = cursor;
    int depth = 0;
    _Bool quoted = 0;
    while (*cursor != '\0') {
        if (*cursor == '"' && (cursor == start || cursor[-1] != '\\')) quoted = !quoted;
        if (!quoted) {
            if (*cursor == '{' || *cursor == '[') ++depth;
            if (*cursor == '}' || *cursor == ']') {
                if (depth == 0) break;
                --depth;
            }
            if (depth == 0 && *cursor == ',') break;
        }
        ++cursor;
    }
    const char *end = cursor;
    while (end > start && (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\r' || end[-1] == '\n')) --end;
    char *value;
    if ((size_t)(end - start) == 4 && memcmp(start, "null", 4) == 0) {
        value = strdup("");
    } else {
        value = malloc((size_t)(end - start) + 1);
        if (value == NULL) abort();
        memcpy(value, start, (size_t)(end - start));
        value[end - start] = '\0';
    }
    *position = cursor;
    return value;
}

static size_t sev_json_column(sev_json_columns *columns, const char *key, _Bool insert) {
    for (size_t index = 0; index < columns->length; ++index) {
        if (strcmp(columns->values[index], key) == 0) return index;
    }
    if (!insert) return SIZE_MAX;
    if (columns->length == columns->capacity) {
        size_t capacity = columns->capacity == 0 ? 16 : columns->capacity * 2;
        char **values = realloc(columns->values, capacity * sizeof(char *));
        if (values == NULL) abort();
        columns->values = values;
        columns->capacity = capacity;
    }
    columns->values[columns->length] = strdup(key);
    if (columns->values[columns->length] == NULL) abort();
    return columns->length++;
}

static const char *sev_json_object(
    const char *cursor,
    sev_json_columns *columns,
    void *row,
    _Bool collect_columns
) {
    if (*cursor != '{') return cursor;
    ++cursor;
    for (;;) {
        cursor = sev_json_space(cursor);
        if (*cursor == '}' || *cursor == '\0') return *cursor == '}' ? cursor + 1 : cursor;
        if (*cursor != '"') {
            ++cursor;
            continue;
        }
        char *key = sev_json_string(&cursor);
        cursor = sev_json_space(cursor);
        if (*cursor == ':') ++cursor;
        char *value = sev_json_value(&cursor);
        size_t column = sev_json_column(columns, key, collect_columns);
        if (row != NULL && column != SIZE_MAX) {
            __sev_list_set_any(row, (int64_t)column, __sev_any_from_string(value));
        } else {
            free(value);
        }
        free(key);
        cursor = sev_json_space(cursor);
        if (*cursor == ',') ++cursor;
    }
}

static sev_json_columns sev_json_schema(const char *source) {
    sev_json_columns columns = {0};
    const char *cursor = source == NULL ? "" : source;
    while (*cursor != '\0') {
        cursor = sev_json_space(cursor);
        if (*cursor == '{') cursor = sev_json_object(cursor, &columns, NULL, 1);
        else ++cursor;
    }
    return columns;
}

void *__sev_json_columns(const char *source) {
    sev_json_columns columns = sev_json_schema(source);
    void *result = __sev_list_create();
    for (size_t index = 0; index < columns.length; ++index) {
        __sev_list_push_ptr(result, columns.values[index]);
    }
    free(columns.values);
    return result;
}

void *__sev_json_rows(const char *source) {
    sev_json_columns columns = sev_json_schema(source);
    void *result = __sev_list_create();
    const char *cursor = source == NULL ? "" : source;
    while (*cursor != '\0') {
        cursor = sev_json_space(cursor);
        if (*cursor != '{') {
            ++cursor;
            continue;
        }
        void *row = __sev_list_create();
        for (size_t index = 0; index < columns.length; ++index) {
            __sev_list_push_any(row, __sev_any_from_string(strdup("")));
        }
        cursor = sev_json_object(cursor, &columns, row, 0);
        sev_csv_list list = {row};
        __sev_list_push_list(result, list);
    }
    for (size_t index = 0; index < columns.length; ++index) free(columns.values[index]);
    free(columns.values);
    return result;
}

static const char *sev_copy_text(char *output, const char *start, size_t length) {
    if (length >= SEV_PATH_CAPACITY) length = SEV_PATH_CAPACITY - 1;
    memcpy(output, start, length);
    output[length] = '\0';
    return output;
}

const char *__sev_path_join(const char *left, const char *right) {
    size_t left_length = strlen(left);
    while (left_length > 1 && left[left_length - 1] == '/') --left_length;
    while (*right == '/') ++right;
    size_t right_length = strlen(right);
    if (left_length + 1 + right_length >= SEV_PATH_CAPACITY) return "";
    memcpy(sev_join_buffer, left, left_length);
    if (left_length != 0 && sev_join_buffer[left_length - 1] != '/') {
        sev_join_buffer[left_length++] = '/';
    }
    memcpy(sev_join_buffer + left_length, right, right_length + 1);
    return sev_join_buffer;
}

const char *__sev_path_basename(const char *value) {
    size_t length = strlen(value);
    while (length > 1 && value[length - 1] == '/') --length;
    size_t start = length;
    while (start > 0 && value[start - 1] != '/') --start;
    return sev_copy_text(sev_basename_buffer, value + start, length - start);
}

const char *__sev_path_dirname(const char *value) {
    size_t length = strlen(value);
    while (length > 1 && value[length - 1] == '/') --length;
    while (length > 0 && value[length - 1] != '/') --length;
    while (length > 1 && value[length - 1] == '/') --length;
    if (length == 0) return sev_copy_text(sev_dirname_buffer, ".", 1);
    return sev_copy_text(sev_dirname_buffer, value, length);
}

const char *__sev_path_extension(const char *value) {
    const char *base = strrchr(value, '/');
    base = base == NULL ? value : base + 1;
    const char *dot = strrchr(base, '.');
    if (dot == NULL || dot == base) return sev_copy_text(sev_extension_buffer, "", 0);
    return sev_copy_text(sev_extension_buffer, dot, strlen(dot));
}

_Bool __sev_path_exists(const char *value) {
    struct stat information;
    return stat(value, &information) == 0;
}

_Bool __sev_path_is_dir(const char *value) {
    struct stat information;
    return stat(value, &information) == 0 && S_ISDIR(information.st_mode);
}

_Bool __sev_os_is_file(const char *value) {
    struct stat information;
    return stat(value, &information) == 0 && S_ISREG(information.st_mode);
}

double __sev_os_file_size(const char *value) {
    struct stat information;
    return stat(value, &information) == 0 ? (double)information.st_size : -1.0;
}

_Bool __sev_os_make_directories(const char *value) {
    size_t length = strlen(value);
    if (length == 0 || length >= SEV_PATH_CAPACITY) return 0;
    char path[SEV_PATH_CAPACITY];
    memcpy(path, value, length + 1);
    while (length > 1 && path[length - 1] == '/') path[--length] = '\0';
    for (char *separator = path + 1; *separator != '\0'; ++separator) {
        if (*separator != '/') continue;
        *separator = '\0';
        if (mkdir(path, 0777) != 0 && errno != EEXIST) return 0;
        if (!__sev_path_is_dir(path)) return 0;
        *separator = '/';
    }
    if (mkdir(path, 0777) != 0 && errno != EEXIST) return 0;
    return __sev_path_is_dir(path);
}

int32_t __sev_file_write_text(const char *path, const char *contents) {
    FILE *file = fopen(path, "wb");
    if (file == NULL) return -1;
    size_t length = strlen(contents);
    int32_t result = fwrite(contents, 1, length, file) == length ? 0 : -1;
    if (fclose(file) != 0) result = -1;
    return result;
}

int32_t __sev_file_write_bytes(const char *path, void *storage) {
    FILE *file = fopen(path, "wb");
    if (file == NULL) return -1;
    uintptr_t length = __sev_list_len(storage);
    int32_t result = 0;
    for (uintptr_t index = 0; index < length; ++index) {
        uint8_t byte = __sev_list_index_u8(storage, (int64_t)index);
        if (fwrite(&byte, 1, 1, file) != 1) {
            result = -1;
            break;
        }
    }
    if (fclose(file) != 0) result = -1;
    return result;
}

const char *__sev_file_read_text(const char *path) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) return "";
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return "";
    }
    long end = ftell(file);
    if (end < 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return "";
    }
    size_t length = (size_t)end;
    char *contents = malloc(length + 1);
    if (contents == NULL) abort();
    size_t read_length = fread(contents, 1, length, file);
    contents[read_length] = '\0';
    fclose(file);
    return contents;
}

int64_t __sev_file_open(const char *path) {
    return open(path, O_RDONLY);
}

void *__sev_file_read_bytes(int64_t descriptor, double requested) {
    void *result = __sev_list_create();
    if (descriptor < 0 || requested <= 0.0) return result;
    uint64_t count = requested >= (double)UINT64_MAX ? UINT64_MAX : (uint64_t)requested;
    for (uint64_t index = 0; index < count; ++index) {
        uint8_t byte;
        ssize_t read_count = read((int)descriptor, &byte, 1);
        if (read_count <= 0) break;
        __sev_list_push_u8(result, byte);
    }
    return result;
}

void *__sev_file_map(const char *path) {
    void *result = __sev_list_create();
    FILE *file = fopen(path, "rb");
    if (file == NULL) return result;
    uint8_t buffer[8192];
    size_t count;
    while ((count = fread(buffer, 1, sizeof(buffer), file)) != 0) {
        for (size_t index = 0; index < count; ++index) {
            __sev_list_push_u8(result, buffer[index]);
        }
    }
    fclose(file);
    return result;
}

static void *sev_file_read_json_boole(const char *path, _Bool keys) {
    const char *cursor = __sev_file_read_text(path);
    void *result = __sev_list_create();
    while ((cursor = strchr(cursor, '"')) != NULL) {
        const char *key_start = ++cursor;
        const char *key_end = strchr(key_start, '"');
        if (key_end == NULL) break;
        cursor = key_end + 1;
        while (*cursor == ' ' || *cursor == '\t' || *cursor == '\n' || *cursor == '\r') ++cursor;
        if (*cursor++ != ':') continue;
        while (*cursor == ' ' || *cursor == '\t' || *cursor == '\n' || *cursor == '\r') ++cursor;
        _Bool value;
        if (strncmp(cursor, "true", 4) == 0) {
            value = 1;
            cursor += 4;
        } else if (strncmp(cursor, "false", 5) == 0) {
            value = 0;
            cursor += 5;
        } else {
            continue;
        }
        if (keys) {
            size_t length = (size_t)(key_end - key_start);
            char *key = malloc(length + 1);
            if (key == NULL) abort();
            memcpy(key, key_start, length);
            key[length] = '\0';
            __sev_list_push_ptr(result, key);
        } else {
            __sev_list_push_bool(result, value);
        }
    }
    return result;
}

void *__sev_file_read_json_bool_keys(const char *path) {
    return sev_file_read_json_boole(path, 1);
}

void *__sev_file_read_json_bool_values(const char *path) {
    return sev_file_read_json_boole(path, 0);
}

_Bool __sev_os_copy(const char *source, const char *destination) {
    FILE *input = fopen(source, "rb");
    if (input == NULL) return 0;
    FILE *output = fopen(destination, "wb");
    if (output == NULL) {
        fclose(input);
        return 0;
    }
    char buffer[8192];
    size_t count;
    _Bool success = 1;
    while ((count = fread(buffer, 1, sizeof(buffer), input)) != 0) {
        if (fwrite(buffer, 1, count, output) != count) {
            success = 0;
            break;
        }
    }
    if (ferror(input) || fclose(input) != 0 || fclose(output) != 0) success = 0;
    return success;
}

int32_t __sev_os_rename(const char *source, const char *destination) {
    return rename(source, destination);
}

int32_t __sev_os_remove(const char *path) {
    return remove(path);
}

int64_t __sev_file_lock(const char *path) {
    int descriptor = open(path, O_RDWR | O_CREAT, 0666);
    if (descriptor < 0) return -1;
    if (flock(descriptor, LOCK_EX) != 0) {
        close(descriptor);
        return -1;
    }
    return descriptor;
}

_Bool __sev_file_unlock(int64_t descriptor) {
    if (descriptor < 0) return 0;
    _Bool success = flock((int)descriptor, LOCK_UN) == 0;
    if (close((int)descriptor) != 0) success = 0;
    return success;
}
