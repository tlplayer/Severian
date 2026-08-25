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
