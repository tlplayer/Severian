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
