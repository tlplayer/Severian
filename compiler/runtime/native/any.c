#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int64_t tag;
    int64_t payload;
} sev_any;

extern const char *__sev_string_from_int(int64_t value);
extern const char *__sev_string_from_float(double value);
extern const char *__sev_string_from_bool(_Bool value);
extern const char *__sev_string_from_char(uint32_t value);
extern const char *__sev_string_from_uint(uint64_t value);
extern const char *__sev_string_from_i128(__int128 value);
extern const char *__sev_string_from_u128(unsigned __int128 value);
extern const char *__sev_string_from_f128(__float128 value);

sev_any __sev_any_from_string(const char *value) {
    sev_any result = {0, (int64_t)(intptr_t)value};
    return result;
}

sev_any __sev_any_from_int(int64_t value) {
    sev_any result = {1, value};
    return result;
}

sev_any __sev_any_from_float(double value) {
    int64_t payload;
    memcpy(&payload, &value, sizeof(payload));
    sev_any result = {2, payload};
    return result;
}

sev_any __sev_any_from_bool(_Bool value) {
    sev_any result = {3, value};
    return result;
}

sev_any __sev_any_from_char(uint32_t value) {
    sev_any result = {4, value};
    return result;
}

sev_any __sev_any_from_uint(uint64_t value) {
    sev_any result = {5, (int64_t)value};
    return result;
}

sev_any __sev_any_from_i128(__int128 value) {
    __int128 *payload = malloc(sizeof(value));
    if (payload == NULL) abort();
    *payload = value;
    sev_any result = {6, (int64_t)(intptr_t)payload};
    return result;
}

sev_any __sev_any_from_u128(unsigned __int128 value) {
    unsigned __int128 *payload = malloc(sizeof(value));
    if (payload == NULL) abort();
    *payload = value;
    sev_any result = {7, (int64_t)(intptr_t)payload};
    return result;
}

sev_any __sev_any_from_f128(__float128 value) {
    __float128 *payload = malloc(sizeof(value));
    if (payload == NULL) abort();
    *payload = value;
    sev_any result = {8, (int64_t)(intptr_t)payload};
    return result;
}

const char *__sev_any_string(sev_any value) {
    switch (value.tag) {
        case 0:
            return (const char *)(intptr_t)value.payload;
        case 1:
            return __sev_string_from_int(value.payload);
        case 2: {
            double number;
            memcpy(&number, &value.payload, sizeof(number));
            return __sev_string_from_float(number);
        }
        case 3:
            return __sev_string_from_bool((_Bool)value.payload);
        case 4:
            return __sev_string_from_char((uint32_t)value.payload);
        case 5:
            return __sev_string_from_uint((uint64_t)value.payload);
        case 6:
            return __sev_string_from_i128(*(__int128 *)(intptr_t)value.payload);
        case 7:
            return __sev_string_from_u128(*(unsigned __int128 *)(intptr_t)value.payload);
        case 8:
            return __sev_string_from_f128(*(__float128 *)(intptr_t)value.payload);
        default:
            return "";
    }
}

const char *__sev_any_kind(sev_any value) {
    switch (value.tag) {
        case 0:
            return "string";
        case 1:
            return "integer";
        case 2:
            return "float";
        case 3:
            return "boolean";
        case 4:
            return "character";
        case 5:
        case 7:
            return "unsigned integer";
        case 6:
            return "integer";
        case 8:
            return "float";
        default:
            return "null";
    }
}

_Bool __sev_any_is_null(sev_any value) {
    return value.tag < 0;
}

static int sev_any_compare(sev_any left, sev_any right) {
    if ((left.tag == 1 || left.tag == 2) &&
        (right.tag == 1 || right.tag == 2)) {
        double left_number;
        double right_number;
        if (left.tag == 1) {
            left_number = (double)left.payload;
        } else {
            memcpy(&left_number, &left.payload, sizeof(left_number));
        }
        if (right.tag == 1) {
            right_number = (double)right.payload;
        } else {
            memcpy(&right_number, &right.payload, sizeof(right_number));
        }
        return left_number < right_number ? -1 : left_number > right_number ? 1 : 0;
    }
    if (left.tag != right.tag) {
        return left.tag < right.tag ? -1 : 1;
    }
    if (left.tag == 0) {
        return strcmp((const char *)(intptr_t)left.payload,
                      (const char *)(intptr_t)right.payload);
    }
    if (left.tag == 5) {
        uint64_t left_value = (uint64_t)left.payload;
        uint64_t right_value = (uint64_t)right.payload;
        return left_value < right_value ? -1 : left_value > right_value ? 1 : 0;
    }
    if (left.tag == 6) {
        __int128 left_value = *(__int128 *)(intptr_t)left.payload;
        __int128 right_value = *(__int128 *)(intptr_t)right.payload;
        return left_value < right_value ? -1 : left_value > right_value ? 1 : 0;
    }
    if (left.tag == 7) {
        unsigned __int128 left_value = *(unsigned __int128 *)(intptr_t)left.payload;
        unsigned __int128 right_value = *(unsigned __int128 *)(intptr_t)right.payload;
        return left_value < right_value ? -1 : left_value > right_value ? 1 : 0;
    }
    if (left.tag == 8) {
        __float128 left_value = *(__float128 *)(intptr_t)left.payload;
        __float128 right_value = *(__float128 *)(intptr_t)right.payload;
        return left_value < right_value ? -1 : left_value > right_value ? 1 : 0;
    }
    return left.payload < right.payload ? -1 : left.payload > right.payload ? 1 : 0;
}

_Bool __sev_any_equal(sev_any left, sev_any right) {
    return sev_any_compare(left, right) == 0;
}

_Bool __sev_any_less(sev_any left, sev_any right) {
    return sev_any_compare(left, right) < 0;
}

_Bool __sev_any_less_equal(sev_any left, sev_any right) {
    return sev_any_compare(left, right) <= 0;
}

_Bool __sev_any_greater(sev_any left, sev_any right) {
    return sev_any_compare(left, right) > 0;
}

_Bool __sev_any_greater_equal(sev_any left, sev_any right) {
    return sev_any_compare(left, right) >= 0;
}
