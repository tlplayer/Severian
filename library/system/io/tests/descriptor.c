#define _GNU_SOURCE
#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
extern int64_t __sev_io_read(int64_t, void *, int64_t);
extern int64_t __sev_io_write(int64_t, const void *, int64_t);
extern int32_t __sev_io_error(void);
int main(void) {
    char path[] = "/tmp/sev-io-descriptor-XXXXXX";
    int descriptor = mkstemp(path);
    assert(descriptor >= 0);
    unlink(path);
    uint8_t data[33001], buffer[33001];
    for (int i = 0; i < 33001; ++i) data[i] = (uint8_t)i;
    int64_t offset = 0;
    while (offset < 33001) {
        int64_t count = __sev_io_write(descriptor, data + offset, 33001 - offset);
        assert(count > 0 && count <= 33001 - offset);
        offset += count;
    }
    for (int iteration = 0; iteration < 128; ++iteration) {
        assert(lseek(descriptor, 0, SEEK_SET) == 0);
        offset = 0;
        while (offset < 33001) {
            int64_t count = __sev_io_read(descriptor, buffer + offset, 33001 - offset);
            assert(count > 0 && count <= 33001 - offset);
            offset += count;
        }
        assert(memcmp(buffer, data, sizeof(data)) == 0);
        assert(__sev_io_read(descriptor, buffer, 1) == 0 && __sev_io_error() == 0);
    }
    assert(__sev_io_read(-1, buffer, 1) == -1 && __sev_io_error() != 0);
    assert(__sev_io_read(descriptor, buffer, -1) == -1 && __sev_io_error() == EINVAL);
    assert(__sev_io_write(-1, buffer, 1) == -1 && __sev_io_error() != 0);
    int ends[2];
    assert(pipe(ends) == 0);
    assert(__sev_io_write(ends[1], "a\0b", 3) == 3);
    assert(__sev_io_read(ends[0], buffer, sizeof(buffer)) == 3);
    assert(__sev_io_error() == 0 && memcmp(buffer, "a\0b", 3) == 0);
    close(ends[1]);
    assert(__sev_io_read(ends[0], buffer, sizeof(buffer)) == 0 && __sev_io_error() == 0);
    close(ends[0]);
    close(descriptor);
    puts("descriptor IO: file, pipe, partial read, EOF and errors passed");
}
