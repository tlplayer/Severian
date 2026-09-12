#ifndef SEVERIAN_OWNED_STORAGE_H
#define SEVERIAN_OWNED_STORAGE_H
#include <stddef.h>
#include <stdint.h>

typedef void (*sev_storage_destructor)(void *);
void *__sev_storage_new(uint64_t bytes, sev_storage_destructor destroy);
void __sev_storage_retain(const void *value);
void __sev_storage_release(const void *value);
void __sev_storage_set_destructor(void *value, sev_storage_destructor destroy);
uint64_t __sev_storage_size(const void *value);
uint64_t __sev_storage_live_bytes(void);
uint64_t __sev_storage_allocations(void);
uint64_t __sev_storage_allocated_bytes(void);
uint64_t __sev_storage_thread_allocations(void);
uint64_t __sev_storage_thread_allocated_bytes(void);
#endif
