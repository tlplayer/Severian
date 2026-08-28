#ifndef SEVERIAN_TENSOR_JIT_H
#define SEVERIAN_TENSOR_JIT_H

#include <stdint.h>

#define SEV_STORAGE_VIEW_ABI_MAGIC UINT64_C(0x535653544f524147)
#define SEV_STORAGE_VIEW_ABI_VERSION UINT32_C(1)
#define SEV_TENSOR_JIT_ABI_MAGIC UINT64_C(0x5356544a49544142)
#define SEV_TENSOR_JIT_ABI_VERSION UINT32_C(1)

typedef struct {
    uint32_t abi_version;
    uint32_t byte_size;
    uint32_t kind;
    uint32_t bits;
    uint32_t float_format;
    uint32_t reserved;
} sev_jit_element_abi;

typedef struct {
    uint64_t magic;
    uint32_t abi_version;
    uint32_t byte_size;
    uint64_t flags;
    const uint8_t *data;
    uint64_t byte_length;
    uint64_t rank;
    const int64_t *dimensions;
    const int64_t *strides;
    int64_t offset;
    sev_jit_element_abi element;
    void *owner;
} sev_jit_storage_view_abi;

enum {
    SEV_TENSOR_JIT_VALUE_STORAGE = 1,
    SEV_TENSOR_JIT_VALUE_POINTER = 2,
    SEV_TENSOR_JIT_VALUE_SIGNED = 3,
    SEV_TENSOR_JIT_VALUE_UNSIGNED = 4,
    SEV_TENSOR_JIT_VALUE_FLOAT = 5,
};

typedef union {
    sev_jit_storage_view_abi *storage;
    void *pointer;
    int64_t signed_integer;
    uint64_t unsigned_integer;
    double floating;
} sev_tensor_jit_value_payload_abi;

typedef struct {
    uint32_t abi_version;
    uint32_t byte_size;
    uint32_t kind;
    uint32_t bits;
    sev_tensor_jit_value_payload_abi value;
} sev_tensor_jit_value_abi;

typedef struct {
    uint64_t magic;
    uint32_t abi_version;
    uint32_t byte_size;
    uint64_t graph_hash[4];
    /* Architecture, donor revision, and compiler options in canonical form. */
    uint64_t compiler_hash[4];
    /* Versioned structural FusionRegion bytecode owned by the executable. */
    const uint8_t *program;
    uint64_t program_size;
    uint32_t target;
    uint32_t input_count;
    uint32_t output_count;
    uint32_t reserved;
} sev_tensor_jit_region_abi;

typedef int32_t (*sev_tensor_jit_launch_fn)(
    void *instance,
    const sev_tensor_jit_value_abi *inputs,
    uint32_t input_count,
    sev_tensor_jit_value_abi *outputs,
    uint32_t output_count
);
typedef void (*sev_tensor_jit_destroy_fn)(void *instance);

typedef struct {
    uint32_t abi_version;
    uint32_t byte_size;
    void *instance;
    sev_tensor_jit_launch_fn launch;
    sev_tensor_jit_destroy_fn destroy;
} sev_tensor_jit_compiled_abi;

typedef int32_t (*sev_tensor_jit_compile_fn)(
    void *context,
    const sev_tensor_jit_region_abi *region,
    const sev_tensor_jit_value_abi *inputs,
    uint32_t input_count,
    sev_tensor_jit_compiled_abi *compiled
);

typedef struct {
    uint32_t abi_version;
    uint32_t byte_size;
    sev_tensor_jit_compile_fn compile;
    void *context;
} sev_tensor_jit_provider_abi;

enum {
    SEV_TENSOR_JIT_OK = 0,
    SEV_TENSOR_JIT_INVALID_ARGUMENT = 1,
    SEV_TENSOR_JIT_NO_COMPILER = 2,
    SEV_TENSOR_JIT_COMPILE_FAILED = 3,
    SEV_TENSOR_JIT_LAUNCH_FAILED = 4,
    SEV_TENSOR_JIT_OUT_OF_MEMORY = 5,
};

int32_t __sev_tensor_jit_install_v1(sev_tensor_jit_compile_fn compile, void *context);
int32_t __sev_tensor_jit_launch_v1(
    const sev_tensor_jit_region_abi *region,
    const sev_tensor_jit_value_abi *inputs,
    uint32_t input_count,
    sev_tensor_jit_value_abi *outputs,
    uint32_t output_count
);
uint64_t __sev_tensor_jit_cache_entries_v1(void);
void __sev_tensor_jit_shutdown_v1(void);

#endif
