#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <pthread.h>
#include <dlfcn.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "tokenizer.h"

typedef struct {
    void *instance;
    sev_tokenizer_provider_abi provider;
} sev_tokenizer_handle;

typedef struct {
    sev_tokenizer_handle *tokenizer;
    int64_t *tokens;
    uint64_t count;
} sev_tokenizer_encoding;

static pthread_mutex_t sev_tokenizer_mutex = PTHREAD_MUTEX_INITIALIZER;
static sev_tokenizer_provider_abi sev_tokenizer_provider;
static void *sev_tokenizer_provider_context = NULL;
static void *sev_tokenizer_library = NULL;

typedef const sev_tokenizer_provider_abi *(*sev_tokenizer_provider_fn)(void);

static void sev_tokenizer_load_provider_locked(void) {
    if (sev_tokenizer_provider.open != NULL || sev_tokenizer_library != NULL) return;
    const char *path = getenv("SEVERIAN_TOKENIZER_LIBRARY");
    void *library = NULL;
    if (path != NULL && *path != '\0') {
        library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    } else {
        char executable[PATH_MAX];
        ssize_t length = readlink("/proc/self/exe", executable, sizeof(executable) - 1);
        if (length > 0) {
            executable[length] = '\0';
            char *separator = strrchr(executable, '/');
            if (separator != NULL) {
                separator[1] = '\0';
                const char provider[] = "libseverian_tokenizer_provider.so";
                if ((size_t)(separator + 1 - executable) + sizeof(provider) <= sizeof(executable)) {
                    memcpy(separator + 1, provider, sizeof(provider));
                    library = dlopen(executable, RTLD_NOW | RTLD_LOCAL);
                }
            }
        }
        if (library == NULL) {
            library = dlopen("libseverian_tokenizer_provider.so", RTLD_NOW | RTLD_LOCAL);
        }
    }
    if (library == NULL) return;
    sev_tokenizer_provider_fn provider =
        (sev_tokenizer_provider_fn)dlsym(library, "sev_tokenizer_provider_v1");
    if (provider == NULL) {
        dlclose(library);
        return;
    }
    const sev_tokenizer_provider_abi *selected = provider();
    if (selected == NULL || selected->abi_version != SEV_TOKENIZER_ABI_VERSION ||
        selected->byte_size != sizeof(*selected) || selected->open == NULL ||
        selected->encode == NULL || selected->release_tokens == NULL || selected->close == NULL) {
        dlclose(library);
        return;
    }
    sev_tokenizer_library = library;
    sev_tokenizer_provider = *selected;
}

int32_t __sev_tokenizer_install_v1(const sev_tokenizer_provider_abi *provider, void *context) {
    if (provider == NULL || provider->abi_version != SEV_TOKENIZER_ABI_VERSION ||
        provider->byte_size != sizeof(*provider) || provider->open == NULL ||
        provider->encode == NULL || provider->release_tokens == NULL || provider->close == NULL) {
        return SEV_TOKENIZER_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&sev_tokenizer_mutex);
    sev_tokenizer_provider = *provider;
    sev_tokenizer_provider_context = context;
    pthread_mutex_unlock(&sev_tokenizer_mutex);
    return SEV_TOKENIZER_OK;
}

int64_t __sev_tokenizer_open_v1(const char *path) {
    if (path == NULL) return 0;
    pthread_mutex_lock(&sev_tokenizer_mutex);
    sev_tokenizer_load_provider_locked();
    sev_tokenizer_provider_abi provider = sev_tokenizer_provider;
    void *context = sev_tokenizer_provider_context;
    pthread_mutex_unlock(&sev_tokenizer_mutex);
    if (provider.open == NULL) return 0;
    sev_tokenizer_handle *handle = calloc(1, sizeof(*handle));
    if (handle == NULL) return 0;
    handle->provider = provider;
    if (provider.open(context, path, &handle->instance) != SEV_TOKENIZER_OK ||
        handle->instance == NULL) {
        free(handle);
        return 0;
    }
    return (int64_t)(intptr_t)handle;
}

int64_t __sev_tokenizer_encode_v1(int64_t raw_handle, const char *text) {
    sev_tokenizer_handle *handle = (sev_tokenizer_handle *)(intptr_t)raw_handle;
    if (handle == NULL || text == NULL) return 0;
    sev_tokenizer_encoding *encoding = calloc(1, sizeof(*encoding));
    if (encoding == NULL) return 0;
    encoding->tokenizer = handle;
    if (handle->provider.encode(handle->instance, text, &encoding->tokens, &encoding->count) !=
        SEV_TOKENIZER_OK || (encoding->count != 0 && encoding->tokens == NULL)) {
        free(encoding);
        return 0;
    }
    return (int64_t)(intptr_t)encoding;
}

int64_t __sev_tokenizer_encoding_length_v1(int64_t raw_encoding) {
    sev_tokenizer_encoding *encoding = (sev_tokenizer_encoding *)(intptr_t)raw_encoding;
    return encoding == NULL || encoding->count > INT64_MAX ? -1 : (int64_t)encoding->count;
}

int64_t __sev_tokenizer_encoding_at_v1(int64_t raw_encoding, int64_t index) {
    sev_tokenizer_encoding *encoding = (sev_tokenizer_encoding *)(intptr_t)raw_encoding;
    if (encoding == NULL || index < 0 || (uint64_t)index >= encoding->count) return -1;
    return encoding->tokens[index];
}

int32_t __sev_tokenizer_encoding_release_v1(int64_t raw_encoding) {
    sev_tokenizer_encoding *encoding = (sev_tokenizer_encoding *)(intptr_t)raw_encoding;
    if (encoding == NULL) return SEV_TOKENIZER_INVALID_ARGUMENT;
    encoding->tokenizer->provider.release_tokens(
        encoding->tokenizer->instance,
        encoding->tokens,
        encoding->count
    );
    free(encoding);
    return SEV_TOKENIZER_OK;
}

int32_t __sev_tokenizer_close_v1(int64_t raw_handle) {
    sev_tokenizer_handle *handle = (sev_tokenizer_handle *)(intptr_t)raw_handle;
    if (handle == NULL) return SEV_TOKENIZER_INVALID_ARGUMENT;
    handle->provider.close(handle->instance);
    free(handle);
    return SEV_TOKENIZER_OK;
}
