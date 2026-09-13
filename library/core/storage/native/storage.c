#include "storage.h"
#include "../../memory/native/memory.h"
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

#define INITIAL_BUCKETS 8192
static allocation *initial_buckets[INITIAL_BUCKETS];
static allocation **allocations = initial_buckets;
static size_t bucket_count = INITIAL_BUCKETS, allocation_count;

static atomic_flag registry_lock = ATOMIC_FLAG_INIT;
static _Atomic uint64_t live_bytes, allocation_calls, allocated_bytes;
static _Thread_local uint64_t thread_calls, thread_bytes;

static void lock(void) { while (atomic_flag_test_and_set_explicit(&registry_lock, memory_order_acquire)) {} }
static void unlock(void) { atomic_flag_clear_explicit(&registry_lock, memory_order_release); }
static size_t bucket(uintptr_t address) {
    address ^= address >> 17;
    address *= (uintptr_t)0xed5ad4bbU;
    address ^= address >> 11;
    return address & (bucket_count - 1);
}
static allocation *find(const void *value) {
    for (allocation *entry = allocations[bucket((uintptr_t)value)]; entry; entry = entry->value.next)
        if ((const void *)(entry + 1) == value) return entry;
    return NULL;
}

/* The ownership registry must scale with live owners. Fixed buckets turn
 * compiler object graphs into a linear scan on every retain and release. */
static void grow_registry(void) {
    if (allocation_count < bucket_count * 2 || bucket_count > SIZE_MAX / 2 / sizeof(allocation *)) return;
    size_t previous_count = bucket_count;
    allocation **previous = allocations;
    allocation **grown = sev_memory_zeroed(bucket_count * 2, sizeof(allocation *));
    if (!grown) abort();
    bucket_count *= 2;
    allocations = grown;
    for (size_t index = 0; index < previous_count; ++index) {
        allocation *entry = previous[index];
        while (entry) {
            allocation *next = entry->value.next;
            size_t target = bucket((uintptr_t)(entry + 1));
            entry->value.next = allocations[target];
            entry->value.previous = &allocations[target];
            if (entry->value.next) entry->value.next->value.previous = &entry->value.next;
            allocations[target] = entry;
            entry = next;
        }
    }
    if (previous != initial_buckets) sev_memory_release(previous);
}

void *__sev_storage_new(uint64_t bytes, sev_storage_destructor destroy) {
    if (bytes > SIZE_MAX - sizeof(allocation)) abort();
    allocation *entry = sev_memory_allocate(sizeof(allocation) + (size_t)bytes);
    if (!entry) abort();
    entry->value.references = 1;
    entry->value.bytes = bytes;
    entry->value.destroy = destroy;
    void *data = entry + 1;
    lock();
    grow_registry();
    size_t index = bucket((uintptr_t)data);
    ++allocation_count;
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
    --allocation_count;
    *entry->value.previous = entry->value.next;
    if (entry->value.next) entry->value.next->value.previous = entry->value.previous;
    unlock();
    /* No registry lock is held during user/default recursive destruction. */
    if (entry->value.destroy) entry->value.destroy((void *)value);
    atomic_fetch_sub(&live_bytes, entry->value.bytes);
    sev_memory_release(entry);
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

/* The compiler's storage-owner projection passes only the allocation identity,
 * never the header or borrowed fields. These are physical ABI adapters. */
void __sev_storage_owner_retain_aggregate(void *owner) { __sev_storage_retain(owner); }
void __sev_storage_owner_release_aggregate(void *owner) { __sev_storage_release(owner); }
