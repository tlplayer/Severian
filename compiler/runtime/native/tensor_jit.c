#include <pthread.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "tensor_jit.h"

typedef struct sev_tensor_jit_cache_entry {
    uint64_t key[4];
    sev_tensor_jit_compiled_abi compiled;
    struct sev_tensor_jit_cache_entry *next;
} sev_tensor_jit_cache_entry;

static pthread_mutex_t sev_tensor_jit_mutex = PTHREAD_MUTEX_INITIALIZER;
static sev_tensor_jit_compile_fn sev_tensor_jit_compile = NULL;
static void *sev_tensor_jit_compile_context = NULL;
static sev_tensor_jit_cache_entry *sev_tensor_jit_cache = NULL;
static uint64_t sev_tensor_jit_cache_size = 0;
static void *sev_tensor_jit_library = NULL;

typedef const sev_tensor_jit_provider_abi *(*sev_tensor_jit_provider_fn)(void);

_Static_assert(sizeof(sev_jit_element_abi) == 24, "storage element ABI drift");
_Static_assert(sizeof(sev_jit_storage_view_abi) == 104, "storage view ABI drift");
_Static_assert(sizeof(sev_tensor_jit_region_abi) == 112, "tensor JIT region ABI drift");
_Static_assert(sizeof(sev_tensor_jit_compiled_abi) == 32, "tensor JIT compiled ABI drift");

static void sev_tensor_jit_load_provider_locked(void) {
    if (sev_tensor_jit_compile != NULL || sev_tensor_jit_library != NULL) return;
    const char *path = getenv("SEVERIAN_TENSOR_JIT_LIBRARY");
    if (path == NULL || *path == '\0') path = "libseverian_tensor_jit_provider.so";
    void *library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (library == NULL) return;
    sev_tensor_jit_provider_fn provider =
        (sev_tensor_jit_provider_fn)dlsym(library, "sev_tensor_jit_provider_v1");
    if (provider == NULL) {
        dlclose(library);
        return;
    }
    const sev_tensor_jit_provider_abi *selected = provider();
    if (selected == NULL || selected->abi_version != SEV_TENSOR_JIT_ABI_VERSION ||
        selected->byte_size != sizeof(*selected) || selected->compile == NULL) {
        dlclose(library);
        return;
    }
    sev_tensor_jit_library = library;
    sev_tensor_jit_compile = selected->compile;
    sev_tensor_jit_compile_context = selected->context;
}

static void sev_tensor_jit_hash_bytes(uint64_t key[4], const void *data, size_t size) {
    const uint8_t *bytes = data;
    static const uint64_t primes[4] = {
        UINT64_C(1099511628211),
        UINT64_C(14029467366897019727),
        UINT64_C(1609587929392839161),
        UINT64_C(9650029242287828579),
    };
    for (size_t byte = 0; byte < size; ++byte) {
        for (size_t lane = 0; lane < 4; ++lane) {
            key[lane] ^= (uint64_t)bytes[byte] + (uint64_t)(lane * 0x9d);
            key[lane] *= primes[lane];
        }
    }
}

static int32_t sev_tensor_jit_validate_view(const sev_jit_storage_view_abi *view) {
    if (view == NULL || view->magic != SEV_STORAGE_VIEW_ABI_MAGIC ||
        view->abi_version != SEV_STORAGE_VIEW_ABI_VERSION ||
        view->byte_size != sizeof(*view) || view->element.abi_version != 1 ||
        view->element.byte_size != sizeof(view->element) || view->element.bits == 0 ||
        view->rank > UINT32_MAX) {
        return SEV_TENSOR_JIT_INVALID_ARGUMENT;
    }
    if (view->rank != 0 && (view->dimensions == NULL || view->strides == NULL)) {
        return SEV_TENSOR_JIT_INVALID_ARGUMENT;
    }
    for (uint64_t axis = 0; axis < view->rank; ++axis) {
        if (view->dimensions[axis] < 0) return SEV_TENSOR_JIT_INVALID_ARGUMENT;
    }
    return SEV_TENSOR_JIT_OK;
}

static int32_t sev_tensor_jit_key(
    const sev_tensor_jit_region_abi *region,
    const sev_jit_storage_view_abi *const *inputs,
    uint32_t input_count,
    uint64_t key[4]
) {
    if (region == NULL || region->magic != SEV_TENSOR_JIT_ABI_MAGIC ||
        region->abi_version != SEV_TENSOR_JIT_ABI_VERSION ||
        region->byte_size != sizeof(*region) || region->input_count != input_count ||
        (region->program_size != 0 && region->program == NULL) ||
        (input_count != 0 && inputs == NULL)) {
        return SEV_TENSOR_JIT_INVALID_ARGUMENT;
    }
    key[0] = UINT64_C(1469598103934665603);
    key[1] = UINT64_C(7809847782465536322);
    key[2] = UINT64_C(9659303129496669493);
    key[3] = UINT64_C(2870177450012600261);
    sev_tensor_jit_hash_bytes(key, region->graph_hash, sizeof(region->graph_hash));
    sev_tensor_jit_hash_bytes(key, region->compiler_hash, sizeof(region->compiler_hash));
    sev_tensor_jit_hash_bytes(key, &region->target, sizeof(region->target));
    for (uint32_t input = 0; input < input_count; ++input) {
        const sev_jit_storage_view_abi *view = inputs[input];
        int32_t status = sev_tensor_jit_validate_view(view);
        if (status != SEV_TENSOR_JIT_OK) return status;
        sev_tensor_jit_hash_bytes(key, &view->element, sizeof(view->element));
        sev_tensor_jit_hash_bytes(key, &view->rank, sizeof(view->rank));
        sev_tensor_jit_hash_bytes(key, view->dimensions, (size_t)view->rank * sizeof(*view->dimensions));
        sev_tensor_jit_hash_bytes(key, view->strides, (size_t)view->rank * sizeof(*view->strides));
        sev_tensor_jit_hash_bytes(key, &view->offset, sizeof(view->offset));
    }
    return SEV_TENSOR_JIT_OK;
}

static void sev_tensor_jit_clear_locked(void) {
    while (sev_tensor_jit_cache != NULL) {
        sev_tensor_jit_cache_entry *entry = sev_tensor_jit_cache;
        sev_tensor_jit_cache = entry->next;
        if (entry->compiled.destroy != NULL) entry->compiled.destroy(entry->compiled.instance);
        free(entry);
    }
    sev_tensor_jit_cache_size = 0;
}

int32_t __sev_tensor_jit_install_v1(sev_tensor_jit_compile_fn compile, void *context) {
    if (compile == NULL) return SEV_TENSOR_JIT_INVALID_ARGUMENT;
    pthread_mutex_lock(&sev_tensor_jit_mutex);
    sev_tensor_jit_clear_locked();
    sev_tensor_jit_compile = compile;
    sev_tensor_jit_compile_context = context;
    pthread_mutex_unlock(&sev_tensor_jit_mutex);
    return SEV_TENSOR_JIT_OK;
}

int32_t __sev_tensor_jit_launch_v1(
    const sev_tensor_jit_region_abi *region,
    const sev_jit_storage_view_abi *const *inputs,
    uint32_t input_count,
    sev_jit_storage_view_abi **outputs,
    uint32_t output_count
) {
    uint64_t key[4];
    int32_t status = sev_tensor_jit_key(region, inputs, input_count, key);
    if (status != SEV_TENSOR_JIT_OK || region->output_count != output_count ||
        (output_count != 0 && outputs == NULL)) return SEV_TENSOR_JIT_INVALID_ARGUMENT;

    pthread_mutex_lock(&sev_tensor_jit_mutex);
    sev_tensor_jit_load_provider_locked();
    sev_tensor_jit_cache_entry *entry = sev_tensor_jit_cache;
    while (entry != NULL && memcmp(entry->key, key, sizeof(key)) != 0) entry = entry->next;
    if (entry == NULL) {
        if (sev_tensor_jit_compile == NULL) {
            pthread_mutex_unlock(&sev_tensor_jit_mutex);
            return SEV_TENSOR_JIT_NO_COMPILER;
        }
        entry = calloc(1, sizeof(*entry));
        if (entry == NULL) {
            pthread_mutex_unlock(&sev_tensor_jit_mutex);
            return SEV_TENSOR_JIT_OUT_OF_MEMORY;
        }
        entry->compiled.abi_version = SEV_TENSOR_JIT_ABI_VERSION;
        entry->compiled.byte_size = sizeof(entry->compiled);
        status = sev_tensor_jit_compile(sev_tensor_jit_compile_context, region, inputs, input_count, &entry->compiled);
        if (status != SEV_TENSOR_JIT_OK || entry->compiled.abi_version != SEV_TENSOR_JIT_ABI_VERSION ||
            entry->compiled.byte_size != sizeof(entry->compiled) || entry->compiled.launch == NULL) {
            if (entry->compiled.destroy != NULL) entry->compiled.destroy(entry->compiled.instance);
            free(entry);
            pthread_mutex_unlock(&sev_tensor_jit_mutex);
            return SEV_TENSOR_JIT_COMPILE_FAILED;
        }
        memcpy(entry->key, key, sizeof(key));
        entry->next = sev_tensor_jit_cache;
        sev_tensor_jit_cache = entry;
        ++sev_tensor_jit_cache_size;
    }
    status = entry->compiled.launch(entry->compiled.instance, inputs, input_count, outputs, output_count);
    pthread_mutex_unlock(&sev_tensor_jit_mutex);
    return status == SEV_TENSOR_JIT_OK ? SEV_TENSOR_JIT_OK : SEV_TENSOR_JIT_LAUNCH_FAILED;
}

uint64_t __sev_tensor_jit_cache_entries_v1(void) {
    pthread_mutex_lock(&sev_tensor_jit_mutex);
    uint64_t size = sev_tensor_jit_cache_size;
    pthread_mutex_unlock(&sev_tensor_jit_mutex);
    return size;
}

void __sev_tensor_jit_shutdown_v1(void) {
    pthread_mutex_lock(&sev_tensor_jit_mutex);
    sev_tensor_jit_clear_locked();
    sev_tensor_jit_compile = NULL;
    sev_tensor_jit_compile_context = NULL;
    pthread_mutex_unlock(&sev_tensor_jit_mutex);
}
