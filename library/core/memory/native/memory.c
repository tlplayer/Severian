#include "memory.h"

static _Thread_local uint64_t thread_allocations, thread_allocated_bytes;
#ifdef SEV_QUALITY_TRACK_ALLOCATIONS
extern void __sev_quality_allocation(void *pointer, size_t bytes);
extern void __sev_quality_release(void *pointer);
#endif

static void *measured_allocation(void *allocation, size_t bytes) {
    if (allocation) {
        ++thread_allocations;
        thread_allocated_bytes += bytes == 0 ? 1 : bytes;
#ifdef SEV_QUALITY_TRACK_ALLOCATIONS
        __sev_quality_allocation(allocation, bytes == 0 ? 1 : bytes);
#endif
    }
    return allocation;
}

uint64_t __sev_memory_thread_allocations(void) { return thread_allocations; }
uint64_t __sev_memory_thread_allocated_bytes(void) { return thread_allocated_bytes; }

void *__sev_memory_allocate(size_t bytes) {
    return measured_allocation(sev_memory_allocate(bytes), bytes);
}
void *__sev_memory_zeroed(size_t count, size_t width) {
    if (width != 0 && count > SIZE_MAX / width) return NULL;
    return measured_allocation(sev_memory_zeroed(count, width), count * width);
}
void __sev_memory_release(void *allocation) {
#ifdef SEV_QUALITY_TRACK_ALLOCATIONS
    __sev_quality_release(allocation);
#endif
    sev_memory_release(allocation);
}

/* MLIR's generic allocator ABI is a physical adapter to core.memory. The
 * ownership pass decides where releases occur; this adapter adds none. */
void *_mlir_memref_to_llvm_alloc(size_t bytes) { return __sev_memory_allocate(bytes); }
void _mlir_memref_to_llvm_free(void *allocation) { __sev_memory_release(allocation); }
