#include <stdint.h>
#include <string.h>

typedef struct {
    int64_t tag;
    int64_t payload;
} sev_any;

extern const char *__sev_string_from_int(int64_t value);
extern const char *__sev_string_from_float(double value);
extern const char *__sev_string_from_bool(_Bool value);
extern const char *__sev_string_from_char(uint32_t value);

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
