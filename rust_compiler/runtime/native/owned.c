#include "owned.h"
#include <stdalign.h>
#include <stdatomic.h>
#include <stdlib.h>

/* Storage owns initialized contents and calls their destruction before the
 * memory provider releases the allocation. Copies retain storage; moves pass
 * their existing reference. Registry lookup makes literal/borrowed pointers
 * harmless without ever reading bytes before an unknown pointer. */
typedef union allocation allocation;
union allocation {
    struct {
        allocation *next;
        allocation **previous;
        uint64_t references, bytes;
        sev_storage_destructor destroy;
    } value;
    max_align_t alignment;
};

#define BUCKETS 8192
static allocation *allocations[BUCKETS];
static atomic_flag registry_lock = ATOMIC_FLAG_INIT;
static _Atomic uint64_t live_bytes, allocation_calls, allocated_bytes;
static _Thread_local uint64_t thread_calls, thread_bytes;

static void lock(void) { while (atomic_flag_test_and_set_explicit(&registry_lock, memory_order_acquire)) {} }
static void unlock(void) { atomic_flag_clear_explicit(&registry_lock, memory_order_release); }
static size_t bucket(uintptr_t address) {
    return ((address >> 4) ^ (address >> 17)) & (BUCKETS - 1);
}
static allocation *find(const void *value) {
    for (allocation *entry = allocations[bucket((uintptr_t)value)]; entry; entry = entry->value.next)
        if ((const void *)(entry + 1) == value) return entry;
    return NULL;
}

void *__sev_storage_new(uint64_t bytes, sev_storage_destructor destroy) {
    if (bytes > SIZE_MAX - sizeof(allocation)) abort();
    allocation *entry = malloc(sizeof(allocation) + (size_t)bytes);
    if (!entry) abort();
    entry->value.references = 1;
    entry->value.bytes = bytes;
    entry->value.destroy = destroy;
    void *data = entry + 1;
    size_t index = bucket((uintptr_t)data);
    lock();
    entry->value.next = allocations[index];
    entry->value.previous = &allocations[index];
    if (entry->value.next) entry->value.next->value.previous = &entry->value.next;
    allocations[index] = entry;
    unlock();
    atomic_fetch_add(&live_bytes, bytes);
    atomic_fetch_add(&allocation_calls, 1);
    atomic_fetch_add(&allocated_bytes, bytes);
    ++thread_calls;
    thread_bytes += bytes;
    return data;
}

void __sev_storage_retain(const void *value) {
    if (!value) return;
    lock();
    allocation *entry = find(value);
    if (entry) {
        if (entry->value.references == UINT64_MAX) abort();
        ++entry->value.references;
    }
    unlock();
}

void __sev_storage_release(const void *value) {
    if (!value) return;
    lock();
    allocation *entry = find(value);
    if (!entry || --entry->value.references) { unlock(); return; }
    *entry->value.previous = entry->value.next;
    if (entry->value.next) entry->value.next->value.previous = entry->value.previous;
    unlock();
    /* No registry lock is held during user/default recursive destruction. */
    if (entry->value.destroy) entry->value.destroy((void *)value);
    atomic_fetch_sub(&live_bytes, entry->value.bytes);
    free(entry);
}

void __sev_storage_set_destructor(void *value, sev_storage_destructor destroy) {
    lock();
    allocation *entry = find(value);
    if (!entry || (entry->value.destroy && entry->value.destroy != destroy)) abort();
    entry->value.destroy = destroy;
    unlock();
}

uint64_t __sev_storage_live_bytes(void) { return atomic_load(&live_bytes); }
uint64_t __sev_storage_allocations(void) { return atomic_load(&allocation_calls); }
uint64_t __sev_storage_allocated_bytes(void) { return atomic_load(&allocated_bytes); }

uint64_t __sev_storage_size(const void *value) {
    lock();
    allocation *entry = find(value);
    uint64_t bytes = entry ? entry->value.bytes : 0;
    unlock();
    return bytes;
}

uint64_t __sev_storage_thread_allocations(void) { return thread_calls; }
uint64_t __sev_storage_thread_allocated_bytes(void) { return thread_bytes; }
