#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    size_t length;
    size_t capacity;
    uintptr_t *values;
} sev_list;

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

uintptr_t __sev_list_len(void *storage) {
    sev_list *list = storage;
    return list->length;
}

void __sev_list_push_i64(void *storage, int64_t value) {
    sev_list *list = storage;
    sev_list_reserve(list);
    list->values[list->length++] = (uintptr_t)value;
}

void __sev_list_push_ptr(void *storage, const char *value) {
    sev_list *list = storage;
    sev_list_reserve(list);
    list->values[list->length++] = (uintptr_t)value;
}

void *__sev_list_append_i64(void *storage, int64_t value) {
    __sev_list_push_i64(storage, value);
    return storage;
}

void *__sev_list_append_ptr(void *storage, const char *value) {
    __sev_list_push_ptr(storage, value);
    return storage;
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

int64_t __sev_list_get_i64(void *storage, int64_t index) {
    sev_list *list = storage;
    if (index < 0) index += (int64_t)list->length;
    if (index < 0 || (size_t)index >= list->length) return 0;
    return (int64_t)list->values[index];
}

const char *__sev_list_get_ptr(void *storage, int64_t index) {
    sev_list *list = storage;
    if (index < 0) index += (int64_t)list->length;
    if (index < 0 || (size_t)index >= list->length) return NULL;
    return (const char *)list->values[index];
}
