#define _POSIX_C_SOURCE 200809L

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../native/tensor_jit.c"

typedef struct { uint32_t kind; uint64_t specialization; } mock_instance;
static uint64_t compile_count = 0;
static uint64_t launch_count = 0;

static int32_t mock_launch(void *opaque, const sev_tensor_jit_value_abi *inputs,
                           uint32_t input_count, sev_tensor_jit_value_abi *outputs,
                           uint32_t output_count) {
    mock_instance *instance = opaque;
    assert(input_count == 1 && instance->kind == inputs[0].kind);
    if (inputs[0].kind == SEV_TENSOR_JIT_VALUE_STORAGE) {
        assert(instance->specialization == inputs[0].value.storage->rank);
    } else {
        const sev_tensor_jit_i64_list *list = inputs[0].value.pointer;
        assert(inputs[0].kind == SEV_TENSOR_JIT_VALUE_LIST_I64 &&
               instance->specialization == list->length);
    }
    ++launch_count;
    if (output_count == 1) {
        outputs[0] = inputs[0];
        outputs[0].value.storage->owner = instance;
    }
    return SEV_TENSOR_JIT_OK;
}

static void mock_destroy(void *opaque) { free(opaque); }

static int32_t mock_compile(void *context, const sev_tensor_jit_region_abi *region,
                            const sev_tensor_jit_value_abi *inputs,
                            uint32_t input_count, sev_tensor_jit_compiled_abi *compiled) {
    (void)context;
    assert(region->target == 1 && input_count == 1);
    mock_instance *instance = calloc(1, sizeof(*instance));
    assert(instance != NULL);
    instance->kind = inputs[0].kind;
    instance->specialization = inputs[0].kind == SEV_TENSOR_JIT_VALUE_STORAGE
        ? inputs[0].value.storage->rank
        : ((const sev_tensor_jit_i64_list *)inputs[0].value.pointer)->length;
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
    sev_tensor_jit_value_abi rank2_inputs[] = {{SEV_TENSOR_JIT_ABI_VERSION, sizeof(sev_tensor_jit_value_abi), SEV_TENSOR_JIT_VALUE_STORAGE, 0, {.storage = &rank2}}};
    sev_tensor_jit_value_abi rank4_inputs[] = {{SEV_TENSOR_JIT_ABI_VERSION, sizeof(sev_tensor_jit_value_abi), SEV_TENSOR_JIT_VALUE_STORAGE, 0, {.storage = &rank4}}};
    sev_tensor_jit_value_abi rank2_outputs[1] = {0};
    sev_tensor_jit_value_abi rank4_outputs[1] = {0};

    assert(__sev_tensor_jit_launch_v1(&region, rank2_inputs, 1, rank2_outputs, 1) == SEV_TENSOR_JIT_NO_COMPILER);
    assert(__sev_tensor_jit_install_v1(mock_compile, NULL) == SEV_TENSOR_JIT_OK);
    assert(__sev_tensor_jit_launch_v1(&region, rank2_inputs, 1, rank2_outputs, 1) == SEV_TENSOR_JIT_OK);
    assert(__sev_tensor_jit_launch_v1(&region, rank2_inputs, 1, rank2_outputs, 1) == SEV_TENSOR_JIT_OK);
    assert(__sev_tensor_jit_launch_v1(&region, rank4_inputs, 1, rank4_outputs, 1) == SEV_TENSOR_JIT_OK);
    assert(compile_count == 2 && launch_count == 3 && __sev_tensor_jit_cache_entries_v1() == 2);
    assert(rank2_outputs[0].value.storage->rank == 2 && rank4_outputs[0].value.storage->rank == 4);
    assert(rank2_outputs[0].value.storage->element.bits == rank4_outputs[0].value.storage->element.bits);

    region.graph_hash[0] = UINT64_C(0x51a9e);
    region.output_count = 0;
    int64_t first_shape_values[] = {1, 16, 512, 128};
    int64_t equal_shape_values[] = {1, 16, 512, 128};
    int64_t changed_shape_values[] = {1, 16, 256, 128};
    sev_tensor_jit_i64_list first_shape = {4, 4, first_shape_values};
    sev_tensor_jit_i64_list equal_shape = {4, 4, equal_shape_values};
    sev_tensor_jit_i64_list changed_shape = {4, 4, changed_shape_values};
    sev_tensor_jit_value_abi first_shape_input[] = {{SEV_TENSOR_JIT_ABI_VERSION, sizeof(sev_tensor_jit_value_abi), SEV_TENSOR_JIT_VALUE_LIST_I64, 64, {.pointer = &first_shape}}};
    sev_tensor_jit_value_abi equal_shape_input[] = {{SEV_TENSOR_JIT_ABI_VERSION, sizeof(sev_tensor_jit_value_abi), SEV_TENSOR_JIT_VALUE_LIST_I64, 64, {.pointer = &equal_shape}}};
    sev_tensor_jit_value_abi changed_shape_input[] = {{SEV_TENSOR_JIT_ABI_VERSION, sizeof(sev_tensor_jit_value_abi), SEV_TENSOR_JIT_VALUE_LIST_I64, 64, {.pointer = &changed_shape}}};
    assert(__sev_tensor_jit_launch_v1(&region, first_shape_input, 1, NULL, 0) == SEV_TENSOR_JIT_OK);
    assert(__sev_tensor_jit_launch_v1(&region, equal_shape_input, 1, NULL, 0) == SEV_TENSOR_JIT_OK);
    assert(__sev_tensor_jit_launch_v1(&region, changed_shape_input, 1, NULL, 0) == SEV_TENSOR_JIT_OK);
    assert(compile_count == 4 && launch_count == 6 && __sev_tensor_jit_cache_entries_v1() == 4);
    __sev_tensor_jit_shutdown_v1();
    assert(__sev_tensor_jit_cache_entries_v1() == 0);
    return 0;
}
