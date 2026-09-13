#include "memory.h"

static _Thread_local uint64_t thread_allocations, thread_allocated_bytes;

static void *measured_allocation(void *allocation, size_t bytes) {
    if (allocation) {
        ++thread_allocations;
        thread_allocated_bytes += bytes == 0 ? 1 : bytes;
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
void __sev_memory_release(void *allocation) { sev_memory_release(allocation); }

/* MLIR's generic allocator ABI is a physical adapter to core.memory. The
 * ownership pass decides where releases occur; this adapter adds none. */
void *_mlir_memref_to_llvm_alloc(size_t bytes) { return __sev_memory_allocate(bytes); }
void _mlir_memref_to_llvm_free(void *allocation) { sev_memory_release(allocation); }
