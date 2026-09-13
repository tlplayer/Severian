#include <assert.h>
#include "../../memory/native/memory.h"
#include "../native/statistics.h"
#ifdef SEV_TEST_OWNERSHIP
#include "../native/storage.h"
#endif
extern void *_mlir_memref_to_llvm_alloc(size_t);
extern void _mlir_memref_to_llvm_free(void *);
int main(void) {
    uint64_t before = __sev_memory_thread_allocations();
    uint64_t bytes = __sev_memory_thread_allocated_bytes();
    void *memory = _mlir_memref_to_llvm_alloc(256);
    assert(__sev_memory_thread_allocations() == before + 1);
    assert(__sev_memory_thread_allocated_bytes() == bytes + 256);
    _mlir_memref_to_llvm_free(memory);
    uint64_t measured_calls = __sev_storage_measured_allocations();
    uint64_t measured_bytes = __sev_storage_measured_bytes();
    memory = _mlir_memref_to_llvm_alloc(512);
    assert(__sev_storage_measured_allocations() == measured_calls + 1);
    assert(__sev_storage_measured_bytes() == measured_bytes + 512);
    _mlir_memref_to_llvm_free(memory);
    assert(__sev_storage_measured_bytes() == measured_bytes + 512);
#ifdef SEV_TEST_OWNERSHIP
    void *owner = __sev_storage_new(128, NULL);
    __sev_storage_retain(owner);
    assert(__sev_storage_measured_allocations() == measured_calls + 2);
    assert(__sev_storage_measured_bytes() == measured_bytes + 640);
    __sev_storage_release(owner);
    __sev_storage_release(owner);
    assert(__sev_storage_measured_bytes() == measured_bytes + 640);
#endif
}
