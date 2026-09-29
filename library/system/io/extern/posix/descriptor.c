#include <errno.h>
#include <stdint.h>
#include <unistd.h>
#include <limits.h>
#include <stddef.h>
#include <stdio.h>
static _Thread_local int stream_error;
int32_t __sev_io_error(void) { return stream_error; }

/* Counted output preserves embedded NUL bytes and shares stdio buffering
 * with native scalar print overloads. Return status, not a byte count. */
int32_t __sev_io_stdout_write(const void *buffer, int64_t count) {
    stream_error = 0;
    if (count < 0 || (uint64_t)count > SIZE_MAX || (count != 0 && buffer == NULL)) {
        stream_error = EINVAL;
        return -1;
    }
    const unsigned char *cursor = buffer;
    size_t remaining = (size_t)count;
    while (remaining != 0) {
        errno = 0;
        size_t written = fwrite(cursor, 1, remaining, stdout);
        if (written == 0 || ferror(stdout)) {
            stream_error = errno != 0 ? errno : EIO;
            return -1;
        }
        cursor += written;
        remaining -= written;
    }
    return 0;
}

int32_t __sev_io_stdout_flush(void) {
    stream_error = 0;
    errno = 0;
    if (fflush(stdout) == 0) return 0;
    stream_error = errno != 0 ? errno : EIO;
    return -1;
}
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
