#define _POSIX_C_SOURCE 200809L

#include <assert.h>
#include <stdint.h>

#include "../native/tokenizer.c"

int main(int argc, char **argv) {
    assert(argc == 2);
    static const int64_t expected[] = {
        151674, 785, 3974, 13876, 38835, 34208,
        916, 279, 15678, 5562, 13, 151675,
    };
    int64_t tokenizer = __sev_tokenizer_open_v1(argv[1]);
    assert(tokenizer != 0);
    int64_t encoding = __sev_tokenizer_encode_v1(
        tokenizer,
        "<|text_start|>The quick brown fox jumps over the lazy dog.<|text_end|>"
    );
    assert(encoding != 0);
    assert(__sev_tokenizer_encoding_length_v1(encoding) == (int64_t)(sizeof(expected) / sizeof(*expected)));
    for (int64_t index = 0; index < (int64_t)(sizeof(expected) / sizeof(*expected)); ++index) {
        assert(__sev_tokenizer_encoding_at_v1(encoding, index) == expected[index]);
    }
    assert(__sev_tokenizer_encoding_release_v1(encoding) == SEV_TOKENIZER_OK);
    assert(__sev_tokenizer_close_v1(tokenizer) == SEV_TOKENIZER_OK);
    return 0;
}
