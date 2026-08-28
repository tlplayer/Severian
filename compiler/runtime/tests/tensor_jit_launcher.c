#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../native/tensor_jit.c"

typedef struct { uint64_t rank; } mock_instance;
static uint64_t compile_count = 0;
static uint64_t launch_count = 0;

static int32_t mock_launch(void *opaque, const sev_jit_storage_view_abi *const *inputs,
                           uint32_t input_count, sev_jit_storage_view_abi **outputs,
                           uint32_t output_count) {
    mock_instance *instance = opaque;
    assert(input_count == 1 && output_count == 1 && instance->rank == inputs[0]->rank);
    ++launch_count;
    *outputs[0] = *inputs[0];
    outputs[0]->owner = instance;
    return SEV_TENSOR_JIT_OK;
}

static void mock_destroy(void *opaque) { free(opaque); }

static int32_t mock_compile(void *context, const sev_tensor_jit_region_abi *region,
                            const sev_jit_storage_view_abi *const *inputs,
                            uint32_t input_count, sev_tensor_jit_compiled_abi *compiled) {
    (void)context;
    assert(region->target == 1 && input_count == 1);
    mock_instance *instance = calloc(1, sizeof(*instance));
    assert(instance != NULL);
    instance->rank = inputs[0]->rank;
    ++compile_count;
    compiled->instance = instance;
    compiled->launch = mock_launch;
    compiled->destroy = mock_destroy;
    return SEV_TENSOR_JIT_OK;
}

static sev_jit_storage_view_abi view(const int64_t *dimensions, const int64_t *strides, uint64_t rank) {
    sev_jit_storage_view_abi result;
    memset(&result, 0, sizeof(result));
    result.magic = SEV_STORAGE_VIEW_ABI_MAGIC;
    result.abi_version = SEV_STORAGE_VIEW_ABI_VERSION;
    result.byte_size = sizeof(result);
    result.rank = rank;
    result.dimensions = dimensions;
    result.strides = strides;
    result.element.abi_version = 1;
    result.element.byte_size = sizeof(result.element);
    result.element.kind = 3;
    result.element.bits = 16;
    result.element.float_format = 2;
    return result;
}

int main(void) {
    sev_tensor_jit_region_abi region;
    memset(&region, 0, sizeof(region));
    region.magic = SEV_TENSOR_JIT_ABI_MAGIC;
    region.abi_version = SEV_TENSOR_JIT_ABI_VERSION;
    region.byte_size = sizeof(region);
    region.graph_hash[0] = UINT64_C(0xadd);
    region.compiler_hash[0] = UINT64_C(0x8957b9aa);
    static const uint8_t program[] = {1, 0, 0, 0, 1};
    region.program = program;
    region.program_size = sizeof(program);
    region.target = 1;
    region.input_count = 1;
    region.output_count = 1;

    int64_t rank2_dimensions[] = {2, 4}, rank2_strides[] = {4, 1};
    int64_t rank4_dimensions[] = {1, 2, 3, 4}, rank4_strides[] = {24, 12, 4, 1};
    sev_jit_storage_view_abi rank2 = view(rank2_dimensions, rank2_strides, 2);
    sev_jit_storage_view_abi rank4 = view(rank4_dimensions, rank4_strides, 4);
    const sev_jit_storage_view_abi *rank2_inputs[] = {&rank2};
    const sev_jit_storage_view_abi *rank4_inputs[] = {&rank4};
    sev_jit_storage_view_abi rank2_output, rank4_output;
    sev_jit_storage_view_abi *rank2_outputs[] = {&rank2_output};
    sev_jit_storage_view_abi *rank4_outputs[] = {&rank4_output};

    assert(__sev_tensor_jit_launch_v1(&region, rank2_inputs, 1, rank2_outputs, 1) == SEV_TENSOR_JIT_NO_COMPILER);
    assert(__sev_tensor_jit_install_v1(mock_compile, NULL) == SEV_TENSOR_JIT_OK);
    assert(__sev_tensor_jit_launch_v1(&region, rank2_inputs, 1, rank2_outputs, 1) == SEV_TENSOR_JIT_OK);
    assert(__sev_tensor_jit_launch_v1(&region, rank2_inputs, 1, rank2_outputs, 1) == SEV_TENSOR_JIT_OK);
    assert(__sev_tensor_jit_launch_v1(&region, rank4_inputs, 1, rank4_outputs, 1) == SEV_TENSOR_JIT_OK);
    assert(compile_count == 2 && launch_count == 3 && __sev_tensor_jit_cache_entries_v1() == 2);
    assert(rank2_output.rank == 2 && rank4_output.rank == 4);
    assert(rank2_output.element.bits == rank4_output.element.bits);
    __sev_tensor_jit_shutdown_v1();
    assert(__sev_tensor_jit_cache_entries_v1() == 0);
    return 0;
}
