#include "statistics.h"
#include "../../../memory/native/memory.h"

/* The reference-counted provider is embedded in the bootstrap runtime. Native
 * memrefs use the physical adapter instead. Both may coexist: storage_new uses
 * the inline allocator, not the measured memref adapter, so totals are disjoint.
 * Foreign provider allocations and ownership registry bookkeeping are excluded.
 */
extern uint64_t __sev_storage_thread_allocations(void) __attribute__((weak));
extern uint64_t __sev_storage_thread_allocated_bytes(void) __attribute__((weak));

uint64_t __sev_storage_measured_allocations(void) {
    return __sev_memory_thread_allocations() +
        (__sev_storage_thread_allocations ? __sev_storage_thread_allocations() : 0);
}

uint64_t __sev_storage_measured_bytes(void) {
    return __sev_memory_thread_allocated_bytes() +
        (__sev_storage_thread_allocated_bytes ? __sev_storage_thread_allocated_bytes() : 0);
}
