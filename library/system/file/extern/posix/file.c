#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <sys/stat.h>
#include <unistd.h>
int32_t __sev_file_open_read(const char *path) {
    int handle;
    do { handle = open(path, O_RDONLY); } while (handle < 0 && errno == EINTR);
    return handle;
}
int32_t __sev_file_open_write(const char *path) {
    int handle;
    do { handle = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0666); } while (handle < 0 && errno == EINTR);
    return handle;
}
int32_t __sev_file_permissions(const char *path) {
    struct stat metadata;
    return stat(path, &metadata) == 0 ? (int32_t)(metadata.st_mode & 07777) : -1;
}
