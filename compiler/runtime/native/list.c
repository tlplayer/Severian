#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    size_t length;
    size_t capacity;
    uintptr_t *values;
} sev_list;

typedef struct {
    int64_t first;
    int64_t second;
} sev_pair_i64;

typedef struct {
    void *storage;
} sev_list_value;

typedef struct {
    size_t length;
} sev_owned_string;

typedef struct {
    size_t size;
} sev_aggregate_box;

extern const char *__sev_any_string(sev_pair_i64 value);
extern _Bool __sev_any_equal(sev_pair_i64 left, sev_pair_i64 right);

static uintptr_t sev_f64_bits(double value) {
    uint64_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return (uintptr_t)bits;
}

static double sev_f64_from_bits(uintptr_t bits) {
    uint64_t raw = (uint64_t)bits;
    double value = 0.0;
    memcpy(&value, &raw, sizeof(value));
    return value;
}

void *__sev_aggregate_box(const void *value, int64_t size) {
    if (size < 0 || (uint64_t)size > SIZE_MAX - sizeof(sev_aggregate_box)) abort();
    sev_aggregate_box *box = malloc(sizeof(sev_aggregate_box) + (size_t)size);
    if (box == NULL) abort();
    box->size = (size_t)size;
    void *payload = box + 1;
    memcpy(payload, value, (size_t)size);
    return payload;
}

static _Bool sev_aggregate_equal(const void *left, const void *right) {
    if (left == right) return 1;
    if (left == NULL || right == NULL) return 0;
    const sev_aggregate_box *left_box = (const sev_aggregate_box *)left - 1;
    const sev_aggregate_box *right_box = (const sev_aggregate_box *)right - 1;
    return left_box->size == right_box->size
        && memcmp(left, right, left_box->size) == 0;
}

static char *sev_list_string_allocation(size_t length) {
    sev_owned_string *allocation = malloc(sizeof(sev_owned_string) + length + 1);
    if (allocation == NULL) abort();
    allocation->length = length;
    return (char *)(allocation + 1);
}

static void sev_list_reserve(sev_list *list) {
    if (list->length < list->capacity) return;
    size_t capacity = list->capacity == 0 ? 4 : list->capacity * 2;
    if (capacity < list->capacity || capacity > SIZE_MAX / sizeof(uintptr_t)) abort();
    uintptr_t *values = realloc(list->values, capacity * sizeof(uintptr_t));
    if (values == NULL) abort();
    list->values = values;
    list->capacity = capacity;
}

void *__sev_list_create(void) {
    sev_list *list = calloc(1, sizeof(sev_list));
    if (list == NULL) abort();
    return list;
}

void *__sev_string_bytes(const char *text) {
    sev_list *result = __sev_list_create();
    const unsigned char *cursor = (const unsigned char *)(text == NULL ? "" : text);
    while (*cursor != 0) {
        sev_list_reserve(result);
        result->values[result->length++] = (uintptr_t)*cursor++;
    }
    return result;
}

double __sev_io_write_all(int64_t stream, void *storage) {
    sev_list *bytes = storage;
    FILE *output = stream == 2 ? stderr : stdout;
    for (size_t index = 0; index < bytes->length; ++index) {
        if (fputc((unsigned char)bytes->values[index], output) == EOF) return -1.0;
    }
    if (fflush(output) != 0) return -1.0;
    return (double)bytes->length;
}

uintptr_t __sev_list_len(void *storage) {
    sev_list *list = storage;
    return list->length;
}

double __sev_list_bytes(void *storage) {
    sev_list *list = storage;
    return (double)list->length;
}

void *__sev_list_indices(void *storage) {
    sev_list *source = storage;
    sev_list *result = __sev_list_create();
    for (size_t index = 0; index < source->length; ++index) {
        sev_list_reserve(result);
        result->values[result->length++] = (uintptr_t)index;
    }
    return result;
}

void *__sev_range(int64_t start, int64_t end, int64_t step) {
    if (step == 0) abort();
    sev_list *result = __sev_list_create();
    if (step > 0) {
        for (int64_t value = start; value < end; value += step) {
            sev_list_reserve(result);
            result->values[result->length++] = (uintptr_t)value;
        }
    } else {
        for (int64_t value = start; value > end; value += step) {
            sev_list_reserve(result);
            result->values[result->length++] = (uintptr_t)value;
        }
    }
    return result;
}

static sev_list *sev_list_zip_side(const sev_list *left, const sev_list *right, _Bool take_left) {
    size_t length = left->length < right->length ? left->length : right->length;
    const sev_list *source = take_left ? left : right;
    sev_list *result = __sev_list_create();
    for (size_t index = 0; index < length; ++index) {
        sev_list_reserve(result);
        result->values[result->length++] = source->values[index];
    }
    return result;
}

void *__sev_list_zip_left(void *left, void *right) {
    return sev_list_zip_side(left, right, 1);
}

void *__sev_list_zip_right(void *left, void *right) {
    return sev_list_zip_side(left, right, 0);
}

void __sev_list_push_i64(void *storage, int64_t value) {
    sev_list *list = storage;
    sev_list_reserve(list);
    list->values[list->length++] = (uintptr_t)value;
}

void __sev_list_push_f64(void *storage, double value) {
    sev_list *list = storage;
    sev_list_reserve(list);
    list->values[list->length++] = sev_f64_bits(value);
}

void __sev_list_push_float(void *storage, double value) {
    __sev_list_push_f64(storage, value);
}

void __sev_list_push_u8(void *storage, uint8_t value) {
    sev_list *list = storage;
    sev_list_reserve(list);
    list->values[list->length++] = (uintptr_t)value;
}

void __sev_list_push_ptr(void *storage, const char *value) {
    sev_list *list = storage;
    sev_list_reserve(list);
    list->values[list->length++] = (uintptr_t)value;
}

void __sev_list_push_aggregate(void *storage, const void *value) {
    __sev_list_push_ptr(storage, value);
}

void __sev_list_push_bool(void *storage, _Bool value) {
    sev_list *list = storage;
    sev_list_reserve(list);
    list->values[list->length++] = (uintptr_t)value;
}

void __sev_list_clear(void *storage) {
    sev_list *list = storage;
    list->length = 0;
}

void *__sev_list_append_i64(void *storage, int64_t value) {
    __sev_list_push_i64(storage, value);
    return storage;
}

void *__sev_list_append_f64(void *storage, double value) {
    sev_list *list = storage;
    sev_list_reserve(list);
    list->values[list->length++] = sev_f64_bits(value);
    return storage;
}

void *__sev_list_append_float(void *storage, double value) {
    return __sev_list_append_f64(storage, value);
}

void *__sev_list_append_u8(void *storage, uint8_t value) {
    __sev_list_push_u8(storage, value);
    return storage;
}

void *__sev_list_append_ptr(void *storage, const char *value) {
    __sev_list_push_ptr(storage, value);
    return storage;
}

void *__sev_list_append_aggregate(void *storage, const void *value) {
    __sev_list_push_ptr(storage, value);
    return storage;
}

void *__sev_list_append_bool(void *storage, _Bool value) {
    __sev_list_push_bool(storage, value);
    return storage;
}

void *__sev_list_append_pair_i64(void *storage, sev_pair_i64 value) {
    sev_pair_i64 *copy = malloc(sizeof(sev_pair_i64));
    if (copy == NULL) abort();
    *copy = value;
    __sev_list_push_ptr(storage, (const char *)copy);
    return storage;
}

void __sev_list_push_pair_i64(void *storage, sev_pair_i64 value) {
    (void)__sev_list_append_pair_i64(storage, value);
}

void *__sev_list_append_any(void *storage, sev_pair_i64 value) {
    return __sev_list_append_pair_i64(storage, value);
}

void __sev_list_push_any(void *storage, sev_pair_i64 value) {
    __sev_list_push_pair_i64(storage, value);
}

void *__sev_list_append_list(void *storage, sev_list_value value) {
    __sev_list_push_ptr(storage, (const char *)value.storage);
    return storage;
}

void __sev_list_push_list(void *storage, sev_list_value value) {
    (void)__sev_list_append_list(storage, value);
}

static sev_list *sev_list_copy(const sev_list *source) {
    sev_list *copy = __sev_list_create();
    if (source->length == 0) return copy;
    copy->values = malloc(source->length * sizeof(uintptr_t));
    if (copy->values == NULL) abort();
    for (size_t index = 0; index < source->length; ++index) {
        copy->values[index] = source->values[index];
    }
    copy->length = source->length;
    copy->capacity = source->length;
    return copy;
}

void *__sev_list_copy_i64(void *storage) {
    return sev_list_copy(storage);
}

void *__sev_list_copy_ptr(void *storage) {
    return sev_list_copy(storage);
}

static int sev_list_compare_i64(const void *left, const void *right) {
    int64_t left_value = (int64_t)*(const uintptr_t *)left;
    int64_t right_value = (int64_t)*(const uintptr_t *)right;
    return (left_value > right_value) - (left_value < right_value);
}

static int sev_list_compare_ptr(const void *left, const void *right) {
    const char *left_value = (const char *)*(const uintptr_t *)left;
    const char *right_value = (const char *)*(const uintptr_t *)right;
    if (left_value == NULL || right_value == NULL) {
        return (left_value != NULL) - (right_value != NULL);
    }
    return strcmp(left_value, right_value);
}

void *__sev_list_sorted_i64(void *storage) {
    sev_list *copy = sev_list_copy(storage);
    if (copy->length > 1) {
        qsort(copy->values, copy->length, sizeof(uintptr_t), sev_list_compare_i64);
    }
    return copy;
}

void *__sev_list_sorted_ptr(void *storage) {
    sev_list *copy = sev_list_copy(storage);
    if (copy->length > 1) {
        qsort(copy->values, copy->length, sizeof(uintptr_t), sev_list_compare_ptr);
    }
    return copy;
}

void *__sev_list_sorted_order_i64(void *storage, _Bool descending) {
    sev_list *copy = __sev_list_sorted_i64(storage);
    if (descending) {
        for (size_t left = 0, right = copy->length == 0 ? 0 : copy->length - 1;
             left < right; ++left, --right) {
            uintptr_t value = copy->values[left];
            copy->values[left] = copy->values[right];
            copy->values[right] = value;
        }
    }
    return copy;
}

void *__sev_list_sorted_order_ptr(void *storage, _Bool descending) {
    sev_list *copy = __sev_list_sorted_ptr(storage);
    if (descending) {
        for (size_t left = 0, right = copy->length == 0 ? 0 : copy->length - 1;
             left < right; ++left, --right) {
            uintptr_t value = copy->values[left];
            copy->values[left] = copy->values[right];
            copy->values[right] = value;
        }
    }
    return copy;
}

const char *__sev_list_string_i64(void *storage) {
    sev_list *list = storage;
    size_t capacity = 3 + list->length * 24;
    char *result = sev_list_string_allocation(capacity);
    size_t offset = 0;
    result[offset++] = '[';
    for (size_t index = 0; index < list->length; ++index) {
        if (index != 0) {
            result[offset++] = ',';
            result[offset++] = ' ';
        }
        offset += (size_t)snprintf(
            result + offset,
            capacity + 1 - offset,
            "%lld",
            (long long)(int64_t)list->values[index]
        );
    }
    result[offset++] = ']';
    result[offset] = '\0';
    return result;
}

const char *__sev_list_string_ptr(void *storage) {
    sev_list *list = storage;
    size_t capacity = 3;
    for (size_t index = 0; index < list->length; ++index) {
        const char *value = (const char *)list->values[index];
        capacity += (value == NULL ? 4 : strlen(value)) + 2;
    }
    char *result = sev_list_string_allocation(capacity);
    size_t offset = 0;
    result[offset++] = '[';
    for (size_t index = 0; index < list->length; ++index) {
        if (index != 0) {
            result[offset++] = ',';
            result[offset++] = ' ';
        }
        const char *value = (const char *)list->values[index];
        if (value == NULL) value = "None";
        size_t length = strlen(value);
        memcpy(result + offset, value, length);
        offset += length;
    }
    result[offset++] = ']';
    result[offset] = '\0';
    return result;
}

const char *__sev_list_string_pair_i64(void *storage) {
    sev_list *list = storage;
    size_t capacity = 3;
    for (size_t index = 0; index < list->length; ++index) {
        sev_pair_i64 *value = (sev_pair_i64 *)list->values[index];
        const char *rendered = value == NULL ? "None" : __sev_any_string(*value);
        capacity += strlen(rendered) + 2;
    }
    char *result = sev_list_string_allocation(capacity);
    size_t offset = 0;
    result[offset++] = '[';
    for (size_t index = 0; index < list->length; ++index) {
        if (index != 0) {
            result[offset++] = ',';
            result[offset++] = ' ';
        }
        sev_pair_i64 *value = (sev_pair_i64 *)list->values[index];
        const char *rendered = value == NULL ? "None" : __sev_any_string(*value);
        size_t length = strlen(rendered);
        memcpy(result + offset, rendered, length);
        offset += length;
    }
    result[offset++] = ']';
    result[offset] = '\0';
    return result;
}

const char *__sev_list_string_any(void *storage) {
    return __sev_list_string_pair_i64(storage);
}

int64_t __sev_list_minimum_i64(void *storage) {
    sev_list *list = storage;
    if (list->length == 0) return 0;
    int64_t result = (int64_t)list->values[0];
    for (size_t index = 1; index < list->length; ++index) {
        int64_t value = (int64_t)list->values[index];
        if (value < result) result = value;
    }
    return result;
}

int64_t __sev_list_maximum_i64(void *storage) {
    sev_list *list = storage;
    if (list->length == 0) return 0;
    int64_t result = (int64_t)list->values[0];
    for (size_t index = 1; index < list->length; ++index) {
        int64_t value = (int64_t)list->values[index];
        if (value > result) result = value;
    }
    return result;
}

int64_t __sev_list_sum_i64(void *storage) {
    sev_list *list = storage;
    int64_t result = 0;
    for (size_t index = 0; index < list->length; ++index) {
        result += (int64_t)list->values[index];
    }
    return result;
}

int64_t __sev_list_last_i64(void *storage) {
    sev_list *list = storage;
    return list->length == 0 ? 0 : (int64_t)list->values[list->length - 1];
}

static sev_list *sev_frequency_keys(const sev_list *source, _Bool pointers) {
    sev_list *keys = __sev_list_create();
    for (size_t index = 0; index < source->length; ++index) {
        uintptr_t value = source->values[index];
        _Bool found = 0;
        for (size_t known = 0; known < keys->length; ++known) {
            if (pointers) {
                const char *left = (const char *)keys->values[known];
                const char *right = (const char *)value;
                found = left == right
                    || (left != NULL && right != NULL && strcmp(left, right) == 0);
            } else {
                found = keys->values[known] == value;
            }
            if (found) break;
        }
        if (!found) {
            sev_list_reserve(keys);
            keys->values[keys->length++] = value;
        }
    }
    return keys;
}

static sev_list *sev_frequency_values(const sev_list *source, _Bool pointers) {
    sev_list *keys = sev_frequency_keys(source, pointers);
    sev_list *values = __sev_list_create();
    for (size_t key = 0; key < keys->length; ++key) {
        int64_t count = 0;
        for (size_t index = 0; index < source->length; ++index) {
            if (pointers) {
                const char *left = (const char *)keys->values[key];
                const char *right = (const char *)source->values[index];
                if (left == right
                    || (left != NULL && right != NULL && strcmp(left, right) == 0)) ++count;
            } else if (keys->values[key] == source->values[index]) {
                ++count;
            }
        }
        sev_list_reserve(values);
        values->values[values->length++] = (uintptr_t)count;
    }
    return values;
}

void *__sev_list_frequency_keys_i64(void *storage) {
    return sev_frequency_keys(storage, 0);
}

void *__sev_list_frequency_values_i64(void *storage) {
    return sev_frequency_values(storage, 0);
}

void *__sev_list_frequency_keys_ptr(void *storage) {
    return sev_frequency_keys(storage, 1);
}

void *__sev_list_frequency_values_ptr(void *storage) {
    return sev_frequency_values(storage, 1);
}

const char *__sev_list_last_ptr(void *storage) {
    sev_list *list = storage;
    return list->length == 0 ? NULL : (const char *)list->values[list->length - 1];
}

_Bool __sev_list_identity(void *left, void *right) {
    return left == right;
}

_Bool __sev_list_equal_i64(void *left_storage, void *right_storage) {
    sev_list *left = left_storage;
    sev_list *right = right_storage;
    if (left->length != right->length) return 0;
    for (size_t index = 0; index < left->length; ++index) {
        if ((int64_t)left->values[index] != (int64_t)right->values[index]) return 0;
    }
    return 1;
}

_Bool __sev_list_equal_f64(void *left_storage, void *right_storage) {
    sev_list *left = left_storage;
    sev_list *right = right_storage;
    if (left->length != right->length) return 0;
    for (size_t index = 0; index < left->length; ++index) {
        if (sev_f64_from_bits(left->values[index]) !=
            sev_f64_from_bits(right->values[index])) return 0;
    }
    return 1;
}

_Bool __sev_list_equal_float(void *left_storage, void *right_storage) {
    return __sev_list_equal_f64(left_storage, right_storage);
}

_Bool __sev_list_equal_ptr(void *left_storage, void *right_storage) {
    sev_list *left = left_storage;
    sev_list *right = right_storage;
    if (left->length != right->length) return 0;
    for (size_t index = 0; index < left->length; ++index) {
        const char *left_value = (const char *)left->values[index];
        const char *right_value = (const char *)right->values[index];
        if (left_value == NULL || right_value == NULL) {
            if (left_value != right_value) return 0;
        } else if (strcmp(left_value, right_value) != 0) {
            return 0;
        }
    }
    return 1;
}

_Bool __sev_list_equal_bool(void *left_storage, void *right_storage) {
    return __sev_list_equal_i64(left_storage, right_storage);
}

_Bool __sev_list_equal_u8(void *left_storage, void *right_storage) {
    return __sev_list_equal_i64(left_storage, right_storage);
}

_Bool __sev_list_equal_any(void *left_storage, void *right_storage) {
    sev_list *left = left_storage;
    sev_list *right = right_storage;
    if (left == right) return 1;
    if (left == NULL || right == NULL || left->length != right->length) return 0;
    for (size_t index = 0; index < left->length; ++index) {
        const sev_pair_i64 *left_value = (const sev_pair_i64 *)left->values[index];
        const sev_pair_i64 *right_value = (const sev_pair_i64 *)right->values[index];
        if (left_value == NULL || right_value == NULL) {
            if (left_value != right_value) return 0;
        } else if (!__sev_any_equal(*left_value, *right_value)) {
            return 0;
        }
    }
    return 1;
}

_Bool __sev_list_equal_list(void *left_storage, void *right_storage) {
    sev_list *left = left_storage;
    sev_list *right = right_storage;
    if (left == right) return 1;
    if (left == NULL || right == NULL || left->length != right->length) return 0;
    for (size_t index = 0; index < left->length; ++index) {
        if (!__sev_list_equal_any(
                (void *)left->values[index],
                (void *)right->values[index])) return 0;
    }
    return 1;
}

_Bool __sev_list_contains_i64(void *storage, int64_t value) {
    sev_list *list = storage;
    for (size_t index = 0; index < list->length; ++index) {
        if ((int64_t)list->values[index] == value) return 1;
    }
    return 0;
}

_Bool __sev_list_contains_ptr(void *storage, const char *value) {
    sev_list *list = storage;
    for (size_t index = 0; index < list->length; ++index) {
        const char *known = (const char *)list->values[index];
        if (known == value || (known != NULL && value != NULL && strcmp(known, value) == 0)) {
            return 1;
        }
    }
    return 0;
}

_Bool __sev_list_contains_aggregate(void *storage, const void *value) {
    sev_list *list = storage;
    for (size_t index = 0; index < list->length; ++index) {
        if (sev_aggregate_equal((const void *)list->values[index], value)) return 1;
    }
    return 0;
}

_Bool __sev_list_any(void *storage) {
    sev_list *list = storage;
    for (size_t index = 0; index < list->length; ++index) {
        if (list->values[index] != 0) return 1;
    }
    return 0;
}

_Bool __sev_list_all(void *storage) {
    sev_list *list = storage;
    for (size_t index = 0; index < list->length; ++index) {
        if (list->values[index] == 0) return 0;
    }
    return 1;
}

int64_t __sev_abs_i64(int64_t value) { return value < 0 ? -value : value; }
int64_t __sev_min_i64(int64_t left, int64_t right) { return left < right ? left : right; }
int64_t __sev_max_i64(int64_t left, int64_t right) { return left > right ? left : right; }
int64_t __sev_div_i64(int64_t left, int64_t right) { return left / right; }
int64_t __sev_mod_i64(int64_t left, int64_t right) { return left % right; }

int64_t __sev_list_pop_i64(void *storage) {
    sev_list *list = storage;
    if (list->length == 0) return 0;
    return (int64_t)list->values[--list->length];
}

const char *__sev_list_pop_ptr(void *storage) {
    sev_list *list = storage;
    if (list->length == 0) return NULL;
    return (const char *)list->values[--list->length];
}

_Bool __sev_list_pop_bool(void *storage) {
    sev_list *list = storage;
    if (list->length == 0) return 0;
    return (_Bool)list->values[--list->length];
}

sev_pair_i64 __sev_list_pop_pair_i64(void *storage) {
    sev_pair_i64 empty = {0, 0};
    sev_pair_i64 *value = (sev_pair_i64 *)__sev_list_pop_ptr(storage);
    if (value == NULL) return empty;
    sev_pair_i64 result = *value;
    free(value);
    return result;
}

sev_pair_i64 __sev_list_pop_any(void *storage) {
    return __sev_list_pop_pair_i64(storage);
}

sev_list_value __sev_list_pop_list(void *storage) {
    sev_list_value result = {(void *)__sev_list_pop_ptr(storage)};
    return result;
}

int64_t __sev_list_get_i64(void *storage, int64_t index) {
    sev_list *list = storage;
    if (index < 0) index += (int64_t)list->length;
    if (index < 0 || (size_t)index >= list->length) return 0;
    return (int64_t)list->values[index];
}

double __sev_list_get_f64(void *storage, int64_t index) {
    sev_list *list = storage;
    if (index < 0) index += (int64_t)list->length;
    if (index < 0 || (size_t)index >= list->length) return 0.0;
    return sev_f64_from_bits(list->values[index]);
}

double __sev_list_get_float(void *storage, int64_t index) {
    return __sev_list_get_f64(storage, index);
}

const char *__sev_list_get_ptr(void *storage, int64_t index) {
    sev_list *list = storage;
    if (index < 0) index += (int64_t)list->length;
    if (index < 0 || (size_t)index >= list->length) return NULL;
    return (const char *)list->values[index];
}

const void *__sev_list_get_aggregate(void *storage, int64_t index) {
    return __sev_list_get_ptr(storage, index);
}

_Bool __sev_list_get_bool(void *storage, int64_t index) {
    return (_Bool)__sev_list_get_i64(storage, index);
}

int64_t __sev_list_index_i64(void *storage, int64_t index) {
    return __sev_list_get_i64(storage, index);
}

double __sev_list_index_f64(void *storage, int64_t index) {
    return __sev_list_get_f64(storage, index);
}

double __sev_list_index_float(void *storage, int64_t index) {
    return __sev_list_get_f64(storage, index);
}

uint8_t __sev_list_index_u8(void *storage, int64_t index) {
    return (uint8_t)__sev_list_get_i64(storage, index);
}

void *__sev_list_address(void *storage, int64_t index) {
    sev_list *list = storage;
    if (index < 0) index += (int64_t)list->length;
    if (index < 0 || (size_t)index >= list->length) return NULL;
    return list->values + index;
}

uint8_t __sev_pointer_index_u8(void *pointer, int64_t index) {
    return ((uint8_t *)pointer)[index];
}

uint8_t __sev_pointer_index_slot_u8(void *pointer, int64_t index) {
    return (uint8_t)((uintptr_t *)pointer)[index];
}

uint32_t __sev_pointer_index_u32(void *pointer, int64_t index) {
    return ((uint32_t *)pointer)[index];
}

int64_t __sev_pointer_index_i64(void *pointer, int64_t index) {
    return ((int64_t *)pointer)[index];
}

void __sev_pointer_set_u8(void *pointer, int64_t index, uint8_t value) {
    ((uint8_t *)pointer)[index] = value;
}

void __sev_pointer_set_slot_u8(void *pointer, int64_t index, uint8_t value) {
    ((uintptr_t *)pointer)[index] = (uintptr_t)value;
}

void __sev_pointer_set_u32(void *pointer, int64_t index, uint32_t value) {
    ((uint32_t *)pointer)[index] = value;
}

void __sev_pointer_set_i64(void *pointer, int64_t index, int64_t value) {
    ((int64_t *)pointer)[index] = value;
}

void *__sev_pointer_add_u8(void *pointer, int64_t offset) {
    return ((uint8_t *)pointer) + offset;
}

void *__sev_pointer_add_slot_u8(void *pointer, int64_t offset) {
    return ((uintptr_t *)pointer) + offset;
}

void *__sev_pointer_add_u32(void *pointer, int64_t offset) {
    return ((uint32_t *)pointer) + offset;
}

void *__sev_pointer_add_i64(void *pointer, int64_t offset) {
    return ((int64_t *)pointer) + offset;
}

void *__sev_pointer_subtract_u8(void *pointer, int64_t offset) {
    return ((uint8_t *)pointer) - offset;
}

void *__sev_pointer_subtract_slot_u8(void *pointer, int64_t offset) {
    return ((uintptr_t *)pointer) - offset;
}

void *__sev_pointer_subtract_u32(void *pointer, int64_t offset) {
    return ((uint32_t *)pointer) - offset;
}

void *__sev_pointer_subtract_i64(void *pointer, int64_t offset) {
    return ((int64_t *)pointer) - offset;
}

_Bool __sev_pointer_equal(void *left, void *right) {
    return left == right;
}

void *__sev_allocate(int64_t count) {
    if (count < 0 || (uint64_t)count > SIZE_MAX / sizeof(uintptr_t)) abort();
    void *allocation = calloc((size_t)count, sizeof(uintptr_t));
    if (allocation == NULL && count != 0) abort();
    return allocation;
}

void __sev_free(void *pointer) {
    free(pointer);
}

const char *__sev_list_index_ptr(void *storage, int64_t index) {
    return __sev_list_get_ptr(storage, index);
}

const void *__sev_list_index_aggregate(void *storage, int64_t index) {
    return __sev_list_get_ptr(storage, index);
}

_Bool __sev_list_index_bool(void *storage, int64_t index) {
    return (_Bool)__sev_list_get_i64(storage, index);
}

sev_pair_i64 __sev_list_get_pair_i64(void *storage, int64_t index) {
    sev_pair_i64 empty = {0, 0};
    sev_pair_i64 *value = (sev_pair_i64 *)__sev_list_get_ptr(storage, index);
    return value == NULL ? empty : *value;
}

sev_pair_i64 __sev_list_get_any(void *storage, int64_t index) {
    return __sev_list_get_pair_i64(storage, index);
}

sev_list_value __sev_list_get_list(void *storage, int64_t index) {
    sev_list_value result = {(void *)__sev_list_get_ptr(storage, index)};
    return result;
}

sev_pair_i64 __sev_list_index_pair_i64(void *storage, int64_t index) {
    return __sev_list_get_pair_i64(storage, index);
}

sev_pair_i64 __sev_list_index_any(void *storage, int64_t index) {
    return __sev_list_index_pair_i64(storage, index);
}

sev_list_value __sev_list_index_list(void *storage, int64_t index) {
    return __sev_list_get_list(storage, index);
}

void *__sev_list_slice(
    void *storage,
    int64_t start,
    int64_t end,
    int64_t step,
    _Bool has_start,
    _Bool has_end,
    _Bool start_exclusive,
    _Bool end_inclusive
) {
    sev_list *source = storage;
    sev_list *result = __sev_list_create();
    int64_t length = (int64_t)source->length;
    if (step == 0) abort();
    if (!has_start) start = step > 0 ? 0 : length - 1;
    else if (start < 0) start += length;
    if (!has_end) end = step > 0 ? length : -1;
    else if (end < 0) end += length;
    if (start_exclusive) start += step > 0 ? 1 : -1;
    if (end_inclusive) end += step > 0 ? 1 : -1;
    if (step > 0) {
        if (start < 0) start = 0;
        if (start > length) start = length;
        if (end < 0) end = 0;
        if (end > length) end = length;
        for (int64_t index = start; index < end; index += step) {
            sev_list_reserve(result);
            result->values[result->length++] = source->values[index];
        }
    } else {
        if (start >= length) start = length - 1;
        if (end >= length) end = length - 1;
        for (int64_t index = start; index > end && index >= 0; index += step) {
            sev_list_reserve(result);
            result->values[result->length++] = source->values[index];
        }
    }
    return result;
}

void __sev_list_set_i64(void *storage, int64_t index, int64_t value) {
    sev_list *list = storage;
    if (index < 0) index += (int64_t)list->length;
    if (index < 0 || (size_t)index >= list->length) return;
    list->values[index] = (uintptr_t)value;
}

void __sev_list_set_f64(void *storage, int64_t index, double value) {
    sev_list *list = storage;
    if (index < 0) index += (int64_t)list->length;
    if (index < 0 || (size_t)index >= list->length) return;
    list->values[index] = sev_f64_bits(value);
}

void __sev_list_set_float(void *storage, int64_t index, double value) {
    __sev_list_set_f64(storage, index, value);
}

void __sev_list_set_u8(void *storage, int64_t index, uint8_t value) {
    __sev_list_set_i64(storage, index, (int64_t)value);
}

void __sev_list_set_ptr(void *storage, int64_t index, const char *value) {
    sev_list *list = storage;
    if (index < 0) index += (int64_t)list->length;
    if (index < 0 || (size_t)index >= list->length) return;
    list->values[index] = (uintptr_t)value;
}

void __sev_list_set_aggregate(void *storage, int64_t index, const void *value) {
    __sev_list_set_ptr(storage, index, value);
}

void __sev_list_set_bool(void *storage, int64_t index, _Bool value) {
    __sev_list_set_i64(storage, index, (int64_t)value);
}

void __sev_list_set_pair_i64(void *storage, int64_t index, sev_pair_i64 value) {
    sev_pair_i64 *copy = malloc(sizeof(sev_pair_i64));
    if (copy == NULL) abort();
    *copy = value;
    __sev_list_set_ptr(storage, index, (const char *)copy);
}

void __sev_list_set_any(void *storage, int64_t index, sev_pair_i64 value) {
    __sev_list_set_pair_i64(storage, index, value);
}

static void sev_list_insert_raw(sev_list *list, int64_t index, uintptr_t value) {
    if (index < 0) index += (int64_t)list->length;
    if (index < 0) index = 0;
    if ((size_t)index > list->length) index = (int64_t)list->length;
    sev_list_reserve(list);
    memmove(
        list->values + index + 1,
        list->values + index,
        (list->length - (size_t)index) * sizeof(uintptr_t)
    );
    list->values[index] = value;
    ++list->length;
}

void __sev_list_appendleft_i64(void *storage, int64_t value) {
    sev_list_insert_raw(storage, 0, (uintptr_t)value);
}

void __sev_list_appendleft_ptr(void *storage, const char *value) {
    sev_list_insert_raw(storage, 0, (uintptr_t)value);
}

void __sev_list_extend(void *storage, void *other_storage) {
    sev_list *list = storage;
    sev_list *other = other_storage;
    for (size_t index = 0; index < other->length; ++index) {
        sev_list_reserve(list);
        list->values[list->length++] = other->values[index];
    }
}

static uintptr_t sev_list_pop_at_raw(sev_list *list, int64_t index) {
    if (index < 0) index += (int64_t)list->length;
    if (index < 0 || (size_t)index >= list->length) return 0;
    uintptr_t result = list->values[index];
    memmove(
        list->values + index,
        list->values + index + 1,
        (list->length - (size_t)index - 1) * sizeof(uintptr_t)
    );
    --list->length;
    return result;
}

int64_t __sev_list_popleft_i64(void *storage) {
    return (int64_t)sev_list_pop_at_raw(storage, 0);
}

const char *__sev_list_popleft_ptr(void *storage) {
    return (const char *)sev_list_pop_at_raw(storage, 0);
}

int64_t __sev_list_pop_at_i64(void *storage, int64_t index) {
    return (int64_t)sev_list_pop_at_raw(storage, index);
}

const char *__sev_list_pop_at_ptr(void *storage, int64_t index) {
    return (const char *)sev_list_pop_at_raw(storage, index);
}

void __sev_list_insert_i64(void *storage, int64_t index, int64_t value) {
    sev_list_insert_raw(storage, index, (uintptr_t)value);
}

void __sev_list_insert_ptr(void *storage, int64_t index, const char *value) {
    sev_list_insert_raw(storage, index, (uintptr_t)value);
}

void __sev_list_remove_i64(void *storage, int64_t value) {
    sev_list *list = storage;
    for (size_t index = 0; index < list->length; ++index) {
        if ((int64_t)list->values[index] == value) {
            (void)sev_list_pop_at_raw(list, (int64_t)index);
            return;
        }
    }
}

void __sev_list_remove_ptr(void *storage, const char *value) {
    sev_list *list = storage;
    for (size_t index = 0; index < list->length; ++index) {
        const char *known = (const char *)list->values[index];
        if (known == value || (known != NULL && value != NULL && strcmp(known, value) == 0)) {
            (void)sev_list_pop_at_raw(list, (int64_t)index);
            return;
        }
    }
}

void __sev_list_heap_push_i64(void *storage, int64_t value) {
    sev_list *heap = storage;
    sev_list_reserve(heap);
    size_t index = heap->length++;
    while (index > 0) {
        size_t parent = (index - 1) / 2;
        if ((int64_t)heap->values[parent] <= value) break;
        heap->values[index] = heap->values[parent];
        index = parent;
    }
    heap->values[index] = (uintptr_t)value;
}

int64_t __sev_list_heap_pop_i64(void *storage) {
    sev_list *heap = storage;
    if (heap->length == 0) return 0;
    int64_t result = (int64_t)heap->values[0];
    uintptr_t tail = heap->values[--heap->length];
    size_t index = 0;
    while (index * 2 + 1 < heap->length) {
        size_t child = index * 2 + 1;
        if (child + 1 < heap->length
            && (int64_t)heap->values[child + 1] < (int64_t)heap->values[child]) ++child;
        if ((int64_t)tail <= (int64_t)heap->values[child]) break;
        heap->values[index] = heap->values[child];
        index = child;
    }
    if (heap->length != 0) heap->values[index] = tail;
    return result;
}

void *__sev_set_append_i64(void *storage, int64_t value) {
    sev_list *set = storage;
    for (size_t index = 0; index < set->length; ++index) {
        if ((int64_t)set->values[index] == value) return storage;
    }
    __sev_list_push_i64(storage, value);
    return storage;
}

void __sev_set_add_i64(void *storage, int64_t value) {
    (void)__sev_set_append_i64(storage, value);
}

_Bool __sev_set_contains_i64(void *storage, int64_t value) {
    sev_list *set = storage;
    for (size_t index = 0; index < set->length; ++index) {
        if ((int64_t)set->values[index] == value) return 1;
    }
    return 0;
}

void *__sev_set_append_ptr(void *storage, const char *value) {
    sev_list *set = storage;
    for (size_t index = 0; index < set->length; ++index) {
        const char *known = (const char *)set->values[index];
        if (known == value || (known != NULL && value != NULL && strcmp(known, value) == 0)) {
            return storage;
        }
    }
    __sev_list_push_ptr(storage, value);
    return storage;
}

void *__sev_set_append_aggregate(void *storage, const void *value) {
    sev_list *set = storage;
    for (size_t index = 0; index < set->length; ++index) {
        if (sev_aggregate_equal((const void *)set->values[index], value)) return storage;
    }
    __sev_list_push_ptr(storage, value);
    return storage;
}

void __sev_set_add_aggregate(void *storage, const void *value) {
    (void)__sev_set_append_aggregate(storage, value);
}

_Bool __sev_set_contains_aggregate(void *storage, const void *value) {
    return __sev_list_contains_aggregate(storage, value);
}

void __sev_set_add_ptr(void *storage, const char *value) {
    (void)__sev_set_append_ptr(storage, value);
}

_Bool __sev_set_contains_ptr(void *storage, const char *value) {
    sev_list *set = storage;
    for (size_t index = 0; index < set->length; ++index) {
        const char *known = (const char *)set->values[index];
        if (known == value || (known != NULL && value != NULL && strcmp(known, value) == 0)) {
            return 1;
        }
    }
    return 0;
}

_Bool __sev_set_equal_i64(void *left_storage, void *right_storage) {
    sev_list *left = left_storage;
    sev_list *right = right_storage;
    if (left->length != right->length) return 0;
    for (size_t index = 0; index < left->length; ++index) {
        if (!__sev_set_contains_i64(right, (int64_t)left->values[index])) return 0;
    }
    return 1;
}

_Bool __sev_set_equal_ptr(void *left_storage, void *right_storage) {
    sev_list *left = left_storage;
    sev_list *right = right_storage;
    if (left->length != right->length) return 0;
    for (size_t index = 0; index < left->length; ++index) {
        if (!__sev_set_contains_ptr(right, (const char *)left->values[index])) return 0;
    }
    return 1;
}

void *__sev_set_union_i64(void *left_storage, void *right_storage) {
    sev_list *left = left_storage;
    sev_list *right = right_storage;
    sev_list *result = sev_list_copy(left);
    for (size_t index = 0; index < right->length; ++index) {
        (void)__sev_set_append_i64(result, (int64_t)right->values[index]);
    }
    return result;
}

void *__sev_set_intersection_i64(void *left_storage, void *right_storage) {
    sev_list *left = left_storage;
    sev_list *result = __sev_list_create();
    for (size_t index = 0; index < left->length; ++index) {
        int64_t value = (int64_t)left->values[index];
        if (__sev_set_contains_i64(right_storage, value)) {
            (void)__sev_set_append_i64(result, value);
        }
    }
    return result;
}

void *__sev_set_symmetric_difference_i64(void *left_storage, void *right_storage) {
    sev_list *left = left_storage;
    sev_list *right = right_storage;
    sev_list *result = __sev_list_create();
    for (size_t index = 0; index < left->length; ++index) {
        int64_t value = (int64_t)left->values[index];
        if (!__sev_set_contains_i64(right, value)) (void)__sev_set_append_i64(result, value);
    }
    for (size_t index = 0; index < right->length; ++index) {
        int64_t value = (int64_t)right->values[index];
        if (!__sev_set_contains_i64(left, value)) (void)__sev_set_append_i64(result, value);
    }
    return result;
}

int64_t __sev_map_get_i64_i64(void *keys_storage, void *values_storage, int64_t key) {
    sev_list *keys = keys_storage;
    sev_list *values = values_storage;
    for (size_t index = 0; index < keys->length; ++index) {
        if ((int64_t)keys->values[index] == key) return (int64_t)values->values[index];
    }
    return 0;
}

const char *__sev_map_get_i64_ptr(void *keys_storage, void *values_storage, int64_t key) {
    sev_list *keys = keys_storage;
    sev_list *values = values_storage;
    for (size_t index = 0; index < keys->length; ++index) {
        if ((int64_t)keys->values[index] == key) return (const char *)values->values[index];
    }
    return NULL;
}

int64_t __sev_map_get_ptr_i64(void *keys_storage, void *values_storage, const char *key) {
    sev_list *keys = keys_storage;
    sev_list *values = values_storage;
    for (size_t index = 0; index < keys->length; ++index) {
        const char *known = (const char *)keys->values[index];
        if (known == key || (known != NULL && key != NULL && strcmp(known, key) == 0)) {
            return (int64_t)values->values[index];
        }
    }
    return 0;
}

_Bool __sev_map_get_ptr_bool(void *keys_storage, void *values_storage, const char *key) {
    return (_Bool)__sev_map_get_ptr_i64(keys_storage, values_storage, key);
}

const char *__sev_map_get_ptr_ptr(void *keys_storage, void *values_storage, const char *key) {
    sev_list *keys = keys_storage;
    sev_list *values = values_storage;
    for (size_t index = 0; index < keys->length; ++index) {
        const char *known = (const char *)keys->values[index];
        if (known == key || (known != NULL && key != NULL && strcmp(known, key) == 0)) {
            return (const char *)values->values[index];
        }
    }
    return NULL;
}

void __sev_map_set_i64_i64(void *keys_storage, void *values_storage, int64_t key, int64_t value) {
    sev_list *keys = keys_storage;
    sev_list *values = values_storage;
    for (size_t index = 0; index < keys->length; ++index) {
        if ((int64_t)keys->values[index] == key) {
            values->values[index] = (uintptr_t)value;
            return;
        }
    }
    __sev_list_push_i64(keys, key);
    __sev_list_push_i64(values, value);
}

void __sev_map_set_i64_ptr(void *keys_storage, void *values_storage, int64_t key, const char *value) {
    sev_list *keys = keys_storage;
    sev_list *values = values_storage;
    for (size_t index = 0; index < keys->length; ++index) {
        if ((int64_t)keys->values[index] == key) {
            values->values[index] = (uintptr_t)value;
            return;
        }
    }
    __sev_list_push_i64(keys, key);
    __sev_list_push_ptr(values, value);
}

void __sev_map_set_ptr_i64(void *keys_storage, void *values_storage, const char *key, int64_t value) {
    sev_list *keys = keys_storage;
    sev_list *values = values_storage;
    for (size_t index = 0; index < keys->length; ++index) {
        const char *known = (const char *)keys->values[index];
        if (known == key || (known != NULL && key != NULL && strcmp(known, key) == 0)) {
            values->values[index] = (uintptr_t)value;
            return;
        }
    }
    __sev_list_push_ptr(keys, key);
    __sev_list_push_i64(values, value);
}

void __sev_map_set_ptr_ptr(void *keys_storage, void *values_storage, const char *key, const char *value) {
    sev_list *keys = keys_storage;
    sev_list *values = values_storage;
    for (size_t index = 0; index < keys->length; ++index) {
        const char *known = (const char *)keys->values[index];
        if (known == key || (known != NULL && key != NULL && strcmp(known, key) == 0)) {
            values->values[index] = (uintptr_t)value;
            return;
        }
    }
    __sev_list_push_ptr(keys, key);
    __sev_list_push_ptr(values, value);
}

int64_t __sev_map_get_default_i64_i64(
    void *keys_storage,
    void *values_storage,
    int64_t key,
    int64_t fallback
) {
    sev_list *keys = keys_storage;
    sev_list *values = values_storage;
    for (size_t index = 0; index < keys->length; ++index) {
        if ((int64_t)keys->values[index] == key) return (int64_t)values->values[index];
    }
    return fallback;
}

int64_t __sev_map_get_default_ptr_i64(
    void *keys_storage,
    void *values_storage,
    const char *key,
    int64_t fallback
) {
    sev_list *keys = keys_storage;
    sev_list *values = values_storage;
    for (size_t index = 0; index < keys->length; ++index) {
        const char *known = (const char *)keys->values[index];
        if (known == key || (known != NULL && key != NULL && strcmp(known, key) == 0)) {
            return (int64_t)values->values[index];
        }
    }
    return fallback;
}

int64_t __sev_map_set_default_ptr_i64(
    void *keys_storage,
    void *values_storage,
    const char *key,
    int64_t fallback
) {
    sev_list *keys = keys_storage;
    sev_list *values = values_storage;
    for (size_t index = 0; index < keys->length; ++index) {
        const char *known = (const char *)keys->values[index];
        if (known == key || (known != NULL && key != NULL && strcmp(known, key) == 0)) {
            return (int64_t)values->values[index];
        }
    }
    __sev_list_push_ptr(keys, key);
    __sev_list_push_i64(values, fallback);
    return fallback;
}

_Bool __sev_map_get_default_ptr_bool(
    void *keys_storage,
    void *values_storage,
    const char *key,
    _Bool fallback
) {
    return (_Bool)__sev_map_get_default_ptr_i64(
        keys_storage,
        values_storage,
        key,
        (int64_t)fallback
    );
}

void __sev_map_set_ptr_bool(
    void *keys_storage,
    void *values_storage,
    const char *key,
    _Bool value
) {
    __sev_map_set_ptr_i64(keys_storage, values_storage, key, (int64_t)value);
}

void __sev_map_set_ptr_pair_i64(
    void *keys_storage,
    void *values_storage,
    const char *key,
    sev_pair_i64 value
) {
    sev_list *keys = keys_storage;
    sev_list *values = values_storage;
    for (size_t index = 0; index < keys->length; ++index) {
        const char *known = (const char *)keys->values[index];
        if (known == key || (known != NULL && key != NULL && strcmp(known, key) == 0)) {
            sev_pair_i64 *copy = malloc(sizeof(sev_pair_i64));
            if (copy == NULL) abort();
            *copy = value;
            values->values[index] = (uintptr_t)copy;
            return;
        }
    }
    __sev_list_push_ptr(keys, key);
    __sev_list_push_pair_i64(values, value);
}

void __sev_map_set_ptr_any(
    void *keys_storage,
    void *values_storage,
    const char *key,
    sev_pair_i64 value
) {
    __sev_map_set_ptr_pair_i64(keys_storage, values_storage, key, value);
}

sev_pair_i64 __sev_map_get_default_ptr_pair_i64(
    void *keys_storage,
    void *values_storage,
    const char *key,
    sev_pair_i64 fallback
) {
    sev_list *keys = keys_storage;
    sev_list *values = values_storage;
    for (size_t index = 0; index < keys->length; ++index) {
        const char *known = (const char *)keys->values[index];
        if (known == key || (known != NULL && key != NULL && strcmp(known, key) == 0)) {
            sev_pair_i64 *value = (sev_pair_i64 *)values->values[index];
            return value == NULL ? fallback : *value;
        }
    }
    return fallback;
}

sev_pair_i64 __sev_map_get_default_ptr_any(
    void *keys_storage,
    void *values_storage,
    const char *key,
    sev_pair_i64 fallback
) {
    return __sev_map_get_default_ptr_pair_i64(keys_storage, values_storage, key, fallback);
}

sev_list_value __sev_map_get_default_ptr_list(
    void *keys_storage,
    void *values_storage,
    const char *key,
    sev_list_value fallback
) {
    sev_list *keys = keys_storage;
    sev_list *values = values_storage;
    for (size_t index = 0; index < keys->length; ++index) {
        const char *known = (const char *)keys->values[index];
        if (known == key || (known != NULL && key != NULL && strcmp(known, key) == 0)) {
            sev_list_value result = {(void *)values->values[index]};
            return result;
        }
    }
    return fallback;
}

void __sev_map_set_ptr_list(
    void *keys_storage,
    void *values_storage,
    const char *key,
    sev_list_value value
) {
    __sev_map_set_ptr_ptr(keys_storage, values_storage, key, (const char *)value.storage);
}

const void *__sev_map_get_default_ptr_aggregate(
    void *keys_storage,
    void *values_storage,
    const char *key,
    const void *fallback
) {
    const char *value = __sev_map_get_ptr_ptr(keys_storage, values_storage, key);
    return value == NULL ? fallback : value;
}

void __sev_map_set_ptr_aggregate(
    void *keys_storage,
    void *values_storage,
    const char *key,
    const void *value
) {
    __sev_map_set_ptr_ptr(keys_storage, values_storage, key, (const char *)value);
}

void __sev_set_add_pair_i64(void *storage, sev_pair_i64 value) {
    sev_list *set = storage;
    for (size_t index = 0; index < set->length; ++index) {
        sev_pair_i64 *known = (sev_pair_i64 *)set->values[index];
        if (known->first == value.first && known->second == value.second) return;
    }
    __sev_list_push_pair_i64(storage, value);
}

void __sev_set_add_any(void *storage, sev_pair_i64 value) {
    __sev_set_add_pair_i64(storage, value);
}

_Bool __sev_set_contains_pair_i64(void *storage, sev_pair_i64 value) {
    sev_list *set = storage;
    for (size_t index = 0; index < set->length; ++index) {
        sev_pair_i64 *known = (sev_pair_i64 *)set->values[index];
        if (known->first == value.first && known->second == value.second) return 1;
    }
    return 0;
}

_Bool __sev_set_contains_any(void *storage, sev_pair_i64 value) {
    return __sev_set_contains_pair_i64(storage, value);
}
