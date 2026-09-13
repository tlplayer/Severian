#define _GNU_SOURCE
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "../../../core/storage/native/storage.h"
extern int64_t __sev_io_read(int64_t, void *, int64_t);
extern int64_t __sev_io_write(int64_t, const void *, int64_t);
extern int32_t __sev_io_error(void);
extern void *__sev_io_read_chunk(int64_t, int64_t);
extern int64_t __sev_io_write_chunk(int64_t, void *, int64_t);
extern void *__sev_list_create(void);
extern void __sev_list_push_u8(void *, uint8_t);
extern uint8_t __sev_list_index_u8(void *, int64_t);
extern uintptr_t __sev_list_len(void *);
int main(void) {
    char path[] = "/tmp/sev-io-descriptor-XXXXXX";
    int descriptor = mkstemp(path);
    assert(descriptor >= 0);
    unlink(path);
    uint64_t live = __sev_storage_live_bytes();
    void *bytes = __sev_list_create();
    for (int i = 0; i < 33001; ++i) __sev_list_push_u8(bytes, (uint8_t)i);
    int64_t offset = 0;
    int writes = 0;
    while (offset < 33001) {
        int64_t count = __sev_io_write_chunk(descriptor, bytes, offset);
        assert(count > 0 && count <= 16384);
        offset += count;
        ++writes;
    }
    assert(writes == 3);
    __sev_storage_release(bytes);
    uint64_t calls = __sev_storage_thread_allocations();
    uint64_t allocated = __sev_storage_thread_allocated_bytes();
    struct timespec begin, end;
    clock_gettime(CLOCK_MONOTONIC, &begin);
    for (int iteration = 0; iteration < 128; ++iteration) {
        assert(lseek(descriptor, 0, SEEK_SET) == 0);
        offset = 0;
        do {
            bytes = __sev_io_read_chunk(descriptor, 16384);
            assert(__sev_io_error() == 0);
            uintptr_t count = __sev_list_len(bytes);
            for (uintptr_t i = 0; i < count; ++i)
                assert(__sev_list_index_u8(bytes, (int64_t)i) == (uint8_t)(offset + (int64_t)i));
            offset += (int64_t)count;
            __sev_storage_release(bytes);
            if (!count) break;
        } while (1);
        assert(offset == 33001);
        assert(__sev_storage_live_bytes() == live);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("{\"iterations\":128,\"nanoseconds\":%lld,\"allocations\":%llu,\"allocated_bytes\":%llu,\"retained_bytes\":%llu}\n",
        (long long)(end.tv_sec - begin.tv_sec) * 1000000000 + end.tv_nsec - begin.tv_nsec,
        (unsigned long long)(__sev_storage_thread_allocations() - calls),
        (unsigned long long)(__sev_storage_thread_allocated_bytes() - allocated),
        (unsigned long long)(__sev_storage_live_bytes() - live));
    char buffer[8];
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
    bytes = __sev_io_read_chunk(-1, 4);
    assert(__sev_io_error() != 0 && __sev_list_len(bytes) == 0);
    __sev_storage_release(bytes);
    bytes = __sev_io_read_chunk(descriptor, 0);
    assert(__sev_io_error() == EINVAL);
    __sev_storage_release(bytes);
    assert(__sev_storage_live_bytes() == live);
    close(descriptor);
}
