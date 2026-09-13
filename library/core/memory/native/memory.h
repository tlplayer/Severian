#ifndef SEVERIAN_CORE_MEMORY_H
#define SEVERIAN_CORE_MEMORY_H
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* The hosted physical allocation boundary. These functions do not own values
 * or run destructors: core.storage owns initialized contents and lifetimes.
 * Foreign owners must supply their own release callback instead of being freed
 * through this provider. A zero-byte request still has allocation identity. */
static inline void *sev_memory_allocate(size_t bytes) {
    return malloc(bytes == 0 ? 1 : bytes);
}
static inline void *sev_memory_zeroed(size_t count, size_t width) {
    if (width != 0 && count > SIZE_MAX / width) return NULL;
    if (count == 0 || width == 0) return calloc(1, 1);
    return calloc(count, width);
}
static inline void *sev_memory_resize(void *allocation, size_t bytes) {
    return realloc(allocation, bytes == 0 ? 1 : bytes);
}
static inline void sev_memory_release(void *allocation) { free(allocation); }

static inline char *sev_memory_copy_text_n(const char *text, size_t limit) {
    size_t length = 0;
    while (length < limit && text[length]) ++length;
    if (length == SIZE_MAX) return NULL;
    char *copy = sev_memory_allocate(length + 1);
    if (copy) { memcpy(copy, text, length); copy[length] = 0; }
    return copy;
}
static inline char *sev_memory_copy_text(const char *text) { return sev_memory_copy_text_n(text, strlen(text)); }

void *__sev_memory_allocate(size_t bytes);
void *__sev_memory_zeroed(size_t count, size_t width);
void __sev_memory_release(void *allocation);
#endif
