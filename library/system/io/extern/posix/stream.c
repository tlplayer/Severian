/* Descriptor IO primitives. Collection storage is owned by core.storage.
 * EOF is an empty successful read; failures use a separate error code. */
#include <errno.h>
#include <stdint.h>
#include <stddef.h>
#include <unistd.h>
#include <limits.h>
extern void *__sev_list_create(void);
extern void __sev_list_push_u8(void *, uint8_t);
extern uint8_t __sev_list_index_u8(void *, int64_t);
extern uintptr_t __sev_list_len(void *);
extern int64_t __sev_io_read(int64_t, void *, int64_t);
extern int64_t __sev_io_write(int64_t, const void *, int64_t);
void *__sev_io_read_chunk(int64_t descriptor, int64_t requested) {
    void *result = __sev_list_create();
    if (descriptor < 0 || descriptor > INT_MAX || requested <= 0) {
        __sev_io_read(-1, NULL, 0);
        return result;
    }
    uint8_t buffer[16384];
    size_t capacity = requested < (int64_t)sizeof(buffer) ? (size_t)requested : sizeof(buffer);
    int64_t count = __sev_io_read(descriptor, buffer, (int64_t)capacity);
    if (count < 0) return result;
    for (ssize_t index = 0; index < count; ++index) __sev_list_push_u8(result, buffer[index]);
    return result;
}
int64_t __sev_io_write_chunk(int64_t descriptor, void *storage, int64_t offset) {
    uintptr_t length = __sev_list_len(storage);
    if (descriptor < 0 || descriptor > INT_MAX || offset < 0 || (uint64_t)offset > length) {
        return __sev_io_write(-1, NULL, 0);
    }
    uint8_t buffer[16384];
    size_t count = length - (size_t)offset;
    if (count > sizeof(buffer)) count = sizeof(buffer);
    for (size_t index = 0; index < count; ++index)
        buffer[index] = __sev_list_index_u8(storage, offset + (int64_t)index);
    return __sev_io_write(descriptor, buffer, (int64_t)count);
}
