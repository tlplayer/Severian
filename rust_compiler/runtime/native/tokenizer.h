#ifndef SEVERIAN_TOKENIZER_H
#define SEVERIAN_TOKENIZER_H

#include <stdint.h>

#define SEV_TOKENIZER_ABI_VERSION UINT32_C(1)

typedef int32_t (*sev_tokenizer_open_fn)(void *context, const char *path, void **instance);
typedef int32_t (*sev_tokenizer_encode_fn)(void *instance, const char *text, int64_t **tokens, uint64_t *count);
typedef void (*sev_tokenizer_release_tokens_fn)(void *instance, int64_t *tokens, uint64_t count);
typedef void (*sev_tokenizer_close_fn)(void *instance);

typedef struct {
    uint32_t abi_version;
    uint32_t byte_size;
    sev_tokenizer_open_fn open;
    sev_tokenizer_encode_fn encode;
    sev_tokenizer_release_tokens_fn release_tokens;
    sev_tokenizer_close_fn close;
} sev_tokenizer_provider_abi;

enum {
    SEV_TOKENIZER_OK = 0,
    SEV_TOKENIZER_INVALID_ARGUMENT = 1,
    SEV_TOKENIZER_NO_PROVIDER = 2,
    SEV_TOKENIZER_PROVIDER_FAILED = 3,
    SEV_TOKENIZER_OUT_OF_MEMORY = 4,
};

int32_t __sev_tokenizer_install_v1(const sev_tokenizer_provider_abi *provider, void *context);
int64_t __sev_tokenizer_open_v1(const char *path);
int64_t __sev_tokenizer_encode_v1(int64_t raw_handle, const char *text);
int64_t __sev_tokenizer_encoding_length_v1(int64_t raw_encoding);
int64_t __sev_tokenizer_encoding_at_v1(int64_t raw_encoding, int64_t index);
int32_t __sev_tokenizer_encoding_release_v1(int64_t raw_encoding);
int32_t __sev_tokenizer_close_v1(int64_t raw_handle);

#endif
