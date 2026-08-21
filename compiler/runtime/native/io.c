#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

static int32_t __sev_print_u128_digits(unsigned __int128 value) {
    char digits[39];
    size_t length = 0;
    do {
        digits[length++] = (char)('0' + value % 10);
        value /= 10;
    } while (value != 0);
    for (size_t left = 0, right = length - 1; left < right; ++left, --right) {
        char swap = digits[left];
        digits[left] = digits[right];
        digits[right] = swap;
    }
    if (fwrite(digits, 1, length, stdout) != length) return -1;
    return putchar('\n');
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
    if (value >= 0) return __sev_print_u128_digits((unsigned __int128)value);
    if (putchar('-') == EOF) return -1;
    unsigned __int128 magnitude = (unsigned __int128)(-(value + 1)) + 1;
    return __sev_print_u128_digits(magnitude);
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
    return __sev_print_u128_digits(value);
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
