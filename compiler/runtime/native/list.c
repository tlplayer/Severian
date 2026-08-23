#include <stdint.h>
#include <stdlib.h>

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
