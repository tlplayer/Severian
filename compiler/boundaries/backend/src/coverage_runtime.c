#include <inttypes.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void __sev_coverage_hit(const char *key) {
    const char *path = getenv("SEV_COVERAGE_FILE");
    if (path == NULL) return;
    FILE *file = fopen(path, "a");
    if (file == NULL) return;
    fputs(key, file);
    fputc('\n', file);
    fclose(file);
}

const char *__sev_string_concat(const char *left, const char *right) {
    size_t left_length = strlen(left);
    size_t right_length = strlen(right);
    if (right_length > SIZE_MAX - left_length - 1) abort();
    char *result = malloc(left_length + right_length + 1);
    if (result == NULL) abort();
    memcpy(result, left, left_length);
    memcpy(result + left_length, right, right_length + 1);
    return result;
}

int32_t __sev_print_int(int64_t value) {
    return printf("%" PRId64 "\n", value);
}

int32_t __sev_print_float(double value) {
    return printf("%.15g\n", value);
}

int32_t __sev_print_i32(int32_t value) {
    return printf("%" PRId32 "\n", value);
}

int32_t __sev_print_i64(int64_t value) {
    return printf("%" PRId64 "\n", value);
}

int32_t __sev_print_f64(double value) {
    return printf("%.15g\n", value);
}

int32_t __sev_print_bool(_Bool value) {
    return puts(value ? "true" : "false");
}

int32_t __sev_print_char(uint32_t value) {
    unsigned char encoded[4];
    size_t length;
    if (value <= 0x7f) {
        encoded[0] = value;
        length = 1;
    } else if (value <= 0x7ff) {
        encoded[0] = 0xc0 | (value >> 6);
        encoded[1] = 0x80 | (value & 0x3f);
        length = 2;
    } else if (value <= 0xffff) {
        encoded[0] = 0xe0 | (value >> 12);
        encoded[1] = 0x80 | ((value >> 6) & 0x3f);
        encoded[2] = 0x80 | (value & 0x3f);
        length = 3;
    } else {
        encoded[0] = 0xf0 | (value >> 18);
        encoded[1] = 0x80 | ((value >> 12) & 0x3f);
        encoded[2] = 0x80 | ((value >> 6) & 0x3f);
        encoded[3] = 0x80 | (value & 0x3f);
        length = 4;
    }
    if (fwrite(encoded, 1, length, stdout) != length) return -1;
    return putchar('\n');
}
