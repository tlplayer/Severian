#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "owned.h"

typedef struct {
    int64_t tag;
    int64_t payload;
} sev_any;

typedef struct {
    int64_t type;
    const char *traits;
    void *value;
} sev_any_record;

static void sev_any_record_destroy(void *storage) {
    sev_any_record *record = storage;
    __sev_storage_release(record->traits);
    __sev_storage_release(record->value);
}

int64_t __sev_any_box_aggregate(void *value, int64_t type, const char *traits) {
    sev_any_record *record = __sev_storage_new(sizeof(*record), sev_any_record_destroy);
    __sev_storage_retain(value);
    __sev_storage_retain(traits);
    *record = (sev_any_record){type, traits, value};
    return (int64_t)(intptr_t)record;
}

_Bool __sev_any_implements(sev_any value, const char *identity, _Bool primitive) {
    if (value.tag == 9) {
        const sev_any_record *record = (const void *)(intptr_t)value.payload;
        return strstr(record->traits, identity) != NULL;
    }
    return value.tag >= 0 && value.tag <= 8 && primitive;
}

_Bool __sev_any_record_is(sev_any value, int64_t type) {
    if (value.tag != 9) return 0;
    const sev_any_record *record = (const void *)(intptr_t)value.payload;
    return record->type == type;
}

void *__sev_any_record_read_aggregate(const sev_any *value, int64_t type) {
    if (!__sev_any_record_is(*value, type)) abort();
    const sev_any_record *record = (const void *)(intptr_t)value->payload;
    __sev_storage_retain(record->value);
    return record->value;
}

void __sev_any_retain(sev_any value) {
    if (value.tag == 0 || (value.tag >= 6 && value.tag <= 9))
        __sev_storage_retain((void *)(intptr_t)value.payload);
}

void __sev_any_release(sev_any value) {
    if (value.tag == 0 || (value.tag >= 6 && value.tag <= 9))
        __sev_storage_release((void *)(intptr_t)value.payload);
}

extern const char *__sev_string_from_int(int64_t value);
extern const char *__sev_string_from_float(double value);
extern const char *__sev_string_from_bool(_Bool value);
extern const char *__sev_string_from_char(uint32_t value);
extern const char *__sev_string_from_uint(uint64_t value);
extern const char *__sev_string_from_i128(__int128 value);
extern const char *__sev_string_from_u128(unsigned __int128 value);
extern const char *__sev_string_from_f128(__float128 value);

sev_any __sev_any_from_string(const char *value) {
    __sev_storage_retain(value);
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
    __int128 *payload = __sev_storage_new(sizeof(value), NULL);
    *payload = value;
    sev_any result = {6, (int64_t)(intptr_t)payload};
    return result;
}

sev_any __sev_any_from_u128(unsigned __int128 value) {
    unsigned __int128 *payload = __sev_storage_new(sizeof(value), NULL);
    *payload = value;
    sev_any result = {7, (int64_t)(intptr_t)payload};
    return result;
}

sev_any __sev_any_from_f128(__float128 value) {
    __float128 *payload = __sev_storage_new(sizeof(value), NULL);
    *payload = value;
    sev_any result = {8, (int64_t)(intptr_t)payload};
    return result;
}

const char *__sev_any_string(sev_any value) {
    switch (value.tag) {
        case 0:
            __sev_storage_retain((void *)(intptr_t)value.payload);
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
        case 9:
            return "<record>";
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
        case 9:
            return "record";
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

#ifdef SEVERIAN_ANY_TEST
#include <assert.h>
static int destroyed_records;
static void test_record_destroy(void *value) {
    assert(*(int64_t *)value == 42);
    destroyed_records++;
}

int main(void) {
    int64_t *record = __sev_storage_new(sizeof(*record), test_record_destroy);
    *record = 42;
    sev_any boxed = {9, __sev_any_box_aggregate(record, 23, "|abc:def:1||abc:def:2|")};
    __sev_storage_release(record);
    assert(__sev_any_implements(boxed, "|abc:def:1|", 0));
    assert(__sev_any_implements(boxed, "|abc:def:2|", 0));
    assert(!__sev_any_implements(boxed, "|abc:def:12|", 0));
    assert(!__sev_any_implements(boxed, "|another:def:1|", 1));
    assert(__sev_any_implements((sev_any){1, 4}, "|Copy|", 1));
    assert(!__sev_any_implements((sev_any){1, 4}, "|Marker|", 0));
    assert(!__sev_any_implements((sev_any){-1, 0}, "|Copy|", 1));
    assert(__sev_any_record_is(boxed, 23));
    assert(!__sev_any_record_is(boxed, 24));
    assert(!__sev_any_record_is((sev_any){1, 23}, 23));
    int64_t *read = __sev_any_record_read_aggregate(&boxed, 23);
    assert(*read == 42);
    __sev_any_retain(boxed);
    __sev_any_release(boxed);
    assert(destroyed_records == 0);
    __sev_any_release(boxed);
    assert(destroyed_records == 0);
    __sev_storage_release(read);
    assert(destroyed_records == 1);
    return 0;
}
#endif
