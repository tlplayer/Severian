#define _POSIX_C_SOURCE 200809L

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../native/tokenizer.c"

static int32_t mock_open(void *context, const char *path, void **instance) {
    (void)context;
    if (strcmp(path, "tokenizer.json") != 0) return SEV_TOKENIZER_PROVIDER_FAILED;
    *instance = malloc(1);
    return *instance == NULL ? SEV_TOKENIZER_OUT_OF_MEMORY : SEV_TOKENIZER_OK;
}

static int32_t mock_encode(void *instance, const char *text, int64_t **tokens, uint64_t *count) {
    assert(instance != NULL);
    *count = strlen(text);
    *tokens = calloc(*count, sizeof(**tokens));
    if (*tokens == NULL && *count != 0) return SEV_TOKENIZER_OUT_OF_MEMORY;
    for (uint64_t index = 0; index < *count; ++index) (*tokens)[index] = (uint8_t)text[index];
    return SEV_TOKENIZER_OK;
}

static void mock_release(void *instance, int64_t *tokens, uint64_t count) {
    (void)instance;
    (void)count;
    free(tokens);
}

static void mock_close(void *instance) { free(instance); }

int main(void) {
    assert(__sev_tokenizer_open_v1("tokenizer.json") == 0);
    sev_tokenizer_provider_abi provider = {
        SEV_TOKENIZER_ABI_VERSION,
        sizeof(provider),
        mock_open,
        mock_encode,
        mock_release,
        mock_close,
    };
    assert(__sev_tokenizer_install_v1(&provider, NULL) == SEV_TOKENIZER_OK);
    int64_t tokenizer = __sev_tokenizer_open_v1("tokenizer.json");
    assert(tokenizer != 0);
    int64_t encoding = __sev_tokenizer_encode_v1(tokenizer, "Hi");
    assert(encoding != 0);
    assert(__sev_tokenizer_encoding_length_v1(encoding) == 2);
    assert(__sev_tokenizer_encoding_at_v1(encoding, 0) == 'H');
    assert(__sev_tokenizer_encoding_at_v1(encoding, 1) == 'i');
    assert(__sev_tokenizer_encoding_at_v1(encoding, 2) == -1);
    assert(__sev_tokenizer_encoding_release_v1(encoding) == SEV_TOKENIZER_OK);
    assert(__sev_tokenizer_close_v1(tokenizer) == SEV_TOKENIZER_OK);
    return 0;
}
