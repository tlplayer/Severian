#include <errno.h>
#include <stdint.h>
#include <unistd.h>
#include <limits.h>
#include <stddef.h>
static _Thread_local int stream_error;
int32_t __sev_io_error(void) { return stream_error; }
int64_t __sev_io_read(int64_t descriptor, void *buffer, int64_t count) {
    stream_error = 0;
    if (descriptor < 0 || descriptor > INT_MAX || count < 0) {
        stream_error = EINVAL;
        return -1;
    }
    ssize_t result;
    do { result = read((int)descriptor, buffer, (size_t)count); } while (result < 0 && errno == EINTR);
    if (result < 0) stream_error = errno;
    return result;
}
int64_t __sev_io_write(int64_t descriptor, const void *buffer, int64_t count) {
    stream_error = 0;
    if (descriptor < 0 || descriptor > INT_MAX || count < 0) {
        stream_error = EINVAL;
        return -1;
    }
    ssize_t result;
    do { result = write((int)descriptor, buffer, (size_t)count); } while (result < 0 && errno == EINTR);
    if (result < 0) stream_error = errno;
    return result;
}

/* XXI owns source-storage marshalling. Providers receive initialized byte views. */
#include "../../../../interop/xxi/extern/c/bytes.h"
int64_t __sev_io_read_view(int64_t descriptor, sev_xxi_bytes *buffer) {
    if (buffer->length > INT64_MAX) { stream_error = EOVERFLOW; return -1; }
    return __sev_io_read(descriptor, buffer->data, (int64_t)buffer->length);
}
int64_t __sev_io_write_view(int64_t descriptor, sev_xxi_bytes buffer, int64_t offset) {
    if (offset < 0 || (uint64_t)offset > buffer.length || buffer.length > INT64_MAX) {
        stream_error = EINVAL;
        return -1;
    }
    return __sev_io_write(descriptor, buffer.data + offset, (int64_t)(buffer.length - (uint64_t)offset));
}
