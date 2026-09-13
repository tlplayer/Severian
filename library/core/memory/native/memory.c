#include "memory.h"

void *__sev_memory_allocate(size_t bytes) { return sev_memory_allocate(bytes); }
void *__sev_memory_zeroed(size_t count, size_t width) { return sev_memory_zeroed(count, width); }
void __sev_memory_release(void *allocation) { sev_memory_release(allocation); }

/* MLIR's generic allocator ABI is a physical adapter to core.memory. The
 * ownership pass decides where releases occur; this adapter adds none. */
void *_mlir_memref_to_llvm_alloc(size_t bytes) { return sev_memory_allocate(bytes); }
void _mlir_memref_to_llvm_free(void *allocation) { sev_memory_release(allocation); }
