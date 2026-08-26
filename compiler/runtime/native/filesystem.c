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
    if (mkdir(value, 0777) == 0 || errno == EEXIST) return __sev_path_is_dir(value);
    return 0;
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
