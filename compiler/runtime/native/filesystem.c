#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

enum { SEV_PATH_CAPACITY = 4096 };

static _Thread_local char sev_join_buffer[SEV_PATH_CAPACITY];
static _Thread_local char sev_basename_buffer[SEV_PATH_CAPACITY];
static _Thread_local char sev_dirname_buffer[SEV_PATH_CAPACITY];
static _Thread_local char sev_extension_buffer[SEV_PATH_CAPACITY];
static _Thread_local char sev_file_buffer[SEV_PATH_CAPACITY];

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

int64_t __sev_os_file_size(const char *value) {
    struct stat information;
    return stat(value, &information) == 0 ? (int64_t)information.st_size : -1;
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

const char *__sev_file_read_text(const char *path) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) return "";
    size_t length = fread(sev_file_buffer, 1, SEV_PATH_CAPACITY - 1, file);
    sev_file_buffer[length] = '\0';
    fclose(file);
    return sev_file_buffer;
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
