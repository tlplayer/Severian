#ifndef SEVERIAN_TRITON_BRIDGE_H
#define SEVERIAN_TRITON_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SEV_TRITON_ABI_VERSION 2u

typedef struct { const uint8_t *data; size_t len; } sev_triton_bytes;
typedef struct { const uint32_t *data; size_t len; } sev_triton_u32_slice;
typedef struct { const int64_t *data; size_t len; } sev_triton_i64_slice;

typedef enum {
  SEV_TRITON_AMD_GPU = 1,
  SEV_TRITON_NVIDIA_GPU = 2,
} sev_triton_target;

typedef enum {
  SEV_TRITON_LLVM_IR = 1,
  SEV_TRITON_AMDGCN = 2,
  SEV_TRITON_HSACO = 3,
  SEV_TRITON_PTX = 4,
  SEV_TRITON_CUBIN = 5,
} sev_triton_kernel_format;

typedef enum {
  SEV_TRITON_OK = 0,
  SEV_TRITON_INVALID_ARGUMENT = 1,
  SEV_TRITON_PARSE_FAILURE = 2,
  SEV_TRITON_PASS_FAILURE = 3,
  SEV_TRITON_CODEGEN_FAILURE = 4,
  SEV_TRITON_UNSUPPORTED_TARGET = 5,
  SEV_TRITON_INTERNAL_FAILURE = 255,
} sev_triton_status;

typedef enum {
  SEV_TRITON_SIGNED_INTEGER = 1,
  SEV_TRITON_UNSIGNED_INTEGER = 2,
  SEV_TRITON_IEEE_FLOAT = 3,
  SEV_TRITON_BRAIN_FLOAT = 4,
  SEV_TRITON_FLOAT8_E4M3FN = 5,
  SEV_TRITON_FLOAT8_E5M2 = 6,
  SEV_TRITON_BOOLEAN = 7,
  SEV_TRITON_OPAQUE = 255,
} sev_triton_element_kind;

typedef struct {
  uint32_t id;
  uint32_t kind;
  sev_triton_bytes operation;
  sev_triton_i64_slice attributes;
  sev_triton_u32_slice inputs;
  sev_triton_i64_slice dimensions;
  sev_triton_element_kind element_kind;
  uint16_t element_bits;
  uint16_t reserved;
  uint64_t bytes_read;
  uint64_t bytes_written;
  uint64_t flops;
  uint64_t shared_memory_bytes;
  uint16_t unnested_reductions;
  uint8_t has_side_effects;
  uint8_t padding[5];
} sev_triton_node;

typedef struct {
  uint32_t abi_version;
  uint32_t region_id;
  const sev_triton_node *nodes;
  size_t node_count;
  sev_triton_u32_slice inputs;
  sev_triton_u32_slice outputs;
} sev_triton_fusion_region;

typedef struct {
  sev_triton_target target;
  sev_triton_bytes architecture;
  uint32_t num_warps;
  uint32_t warp_size;
  uint32_t num_ctas;
  uint32_t num_stages;
  sev_triton_kernel_format emit;
  uint8_t debug;
  uint8_t padding[7];
} sev_triton_compile_options;

typedef struct {
  uint32_t abi_version;
  const sev_triton_fusion_region *region;
  sev_triton_bytes ttir;
  const sev_triton_compile_options *options;
} sev_triton_compile_request;

typedef struct {
  uint32_t abi_version;
  sev_triton_kernel_format format;
  sev_triton_bytes entry_point;
  sev_triton_bytes code;
  sev_triton_bytes diagnostics;
  uint64_t shared_memory_bytes;
  void *owner;
} sev_triton_compiled_kernel;

sev_triton_status sev_triton_compile(
    const sev_triton_compile_request *request,
    sev_triton_compiled_kernel *output);
void sev_triton_destroy_kernel(sev_triton_compiled_kernel *kernel);

#ifdef __cplusplus
}
#endif

#endif
