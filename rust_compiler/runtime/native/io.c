#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern const char *__sev_string_from_i128(__int128 value);
extern const char *__sev_string_from_u128(unsigned __int128 value);
extern const char *__sev_string_from_char(uint32_t value);
extern const char *__sev_string_from_f128(__float128 value);

void __sev_assert(_Bool condition, const char *message) {
    if (condition) return;
    fputs(message, stderr);
    fputc('\n', stderr);
    exit(1);
}

void __sev_expect(_Bool condition, const char *message) {
    if (condition) return;
    fputs("expectation failed: ", stderr);
    fputs(message == NULL ? "condition was false" : message, stderr);
    fputc('\n', stderr);
}

int32_t __sev_print_string(const char *value) {
    return puts(value);
}

/* The bootstrap string ABI is NUL-terminated. Keep its adapter here, while
 * the source IO provider passes counted UTF-8 buffers directly. */
int32_t __sev_print_string_end(const char *value, const char *ending) {
    const char *parts[] = {value, ending};
    for (size_t part = 0; part < 2; ++part) {
        size_t remaining = strlen(parts[part]);
        const char *cursor = parts[part];
        while (remaining != 0) {
            size_t written = fwrite(cursor, 1, remaining, stdout);
            if (written == 0 || ferror(stdout)) return -1;
            cursor += written;
            remaining -= written;
        }
    }
    return 0;
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

int32_t __sev_print_i8(int8_t value) {
    return printf("%" PRId8 "\n", value);
}

int32_t __sev_print_i16(int16_t value) {
    return printf("%" PRId16 "\n", value);
}

int32_t __sev_print_i64(int64_t value) {
    return printf("%" PRId64 "\n", value);
}

int32_t __sev_print_i128(__int128 value) {
    return puts(__sev_string_from_i128(value));
}

int32_t __sev_print_isize(intptr_t value) {
    return printf("%" PRIdPTR "\n", value);
}

int32_t __sev_print_u8(uint8_t value) {
    return printf("%" PRIu8 "\n", value);
}

int32_t __sev_print_u16(uint16_t value) {
    return printf("%" PRIu16 "\n", value);
}

int32_t __sev_print_u32(uint32_t value) {
    return printf("%" PRIu32 "\n", value);
}

int32_t __sev_print_u64(uint64_t value) {
    return printf("%" PRIu64 "\n", value);
}

int32_t __sev_print_u128(unsigned __int128 value) {
    return puts(__sev_string_from_u128(value));
}

int32_t __sev_print_usize(uintptr_t value) {
    return printf("%" PRIuPTR "\n", value);
}

int32_t __sev_print_f32(float value) {
    return printf("%.7g\n", value);
}

int32_t __sev_print_f64(double value) {
    return printf("%.15g\n", value);
}

int32_t __sev_print_f128(__float128 value) {
    return puts(__sev_string_from_f128(value));
}

int32_t __sev_print_bool(_Bool value) {
    return puts(value ? "true" : "false");
}

int32_t __sev_print_char(uint32_t value) {
    return puts(__sev_string_from_char(value));
}
