#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "../../../core/memory/native/memory.h"
extern const char *__sev_file_read_text(const char *);
extern int32_t __sev_file_error(void);
#ifdef SEV_TEST_OWNERSHIP
#include "../../../core/storage/native/storage.h"
extern const char *__sev_file_read_owned_text(const char *);
#endif
int main(void) {
    char path[] = "/tmp/sev-file-native-XXXXXX";
    int fd = mkstemp(path);
    assert(fd >= 0);
    const char *empty = __sev_file_read_text(path);
    assert(__sev_file_error() == 0 && !strcmp(empty, ""));
    free((void *)empty);
#ifdef SEV_TEST_OWNERSHIP
    uint64_t live = __sev_storage_live_bytes();
    for (int iteration = 0; iteration < 128; ++iteration) {
        const char *owned = __sev_file_read_owned_text(path);
        assert(__sev_file_error() == 0 && !strcmp(owned, ""));
        __sev_storage_release(owned);
        assert(__sev_storage_live_bytes() == live);
    }
#endif
    for (size_t size = 1; size <= 32768; size *= 2) {
        assert(ftruncate(fd, 0) == 0 && lseek(fd, 0, SEEK_SET) == 0);
        char *data = malloc(size);
        memset(data, 'x', size);
        assert(write(fd, data, size) == (ssize_t)size);
        const char *read = __sev_file_read_text(path);
        assert(__sev_file_error() == 0 && strlen(read) == size && !memcmp(read, data, size));
        free((void *)read); free(data);
#ifdef SEV_TEST_OWNERSHIP
        const char *owned = __sev_file_read_owned_text(path);
        assert(__sev_file_error() == 0 && strlen(owned) == size);
        __sev_storage_release(owned);
        assert(__sev_storage_live_bytes() == live);
#endif
    }
    assert(lseek(fd, 0, SEEK_SET) == 0 && write(fd, "\0", 1) == 1);
    assert(!strcmp(__sev_file_read_text(path), "") && __sev_file_error() == EILSEQ);
    close(fd); unlink(path);
    __sev_file_read_text(path);
    assert(__sev_file_error() == ENOENT);
}
