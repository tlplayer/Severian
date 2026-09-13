#define _GNU_SOURCE
#include "../../../core/memory/native/memory.h"
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/syscall.h>

enum { SEV_PATH_CAPACITY = 4096 };

static _Thread_local char sev_cwd_buffer[SEV_PATH_CAPACITY];

const char *__sev_process_current_directory(void) {
    return getcwd(sev_cwd_buffer, sizeof(sev_cwd_buffer)) == NULL ? "" : sev_cwd_buffer;
}

_Bool __sev_path_exists(const char *value) {
    struct stat information;
    return stat(value, &information) == 0;
}

_Bool __sev_path_is_dir(const char *value) {
    struct stat information;
    return stat(value, &information) == 0 && S_ISDIR(information.st_mode);
}

_Bool __sev_os_is_file(const char *value) {
    struct stat information;
    return stat(value, &information) == 0 && S_ISREG(information.st_mode);
}

double __sev_os_file_size(const char *value) {
    struct stat information;
    return stat(value, &information) == 0 ? (double)information.st_size : -1.0;
}

_Bool __sev_os_make_directories(const char *value) {
    size_t length = strlen(value);
    if (length == 0 || length >= SEV_PATH_CAPACITY) return 0;
    char path[SEV_PATH_CAPACITY];
    memcpy(path, value, length + 1);
    while (length > 1 && path[length - 1] == '/') path[--length] = '\0';
    for (char *separator = path + 1; *separator != '\0'; ++separator) {
        if (*separator != '/') continue;
        *separator = '\0';
        if (mkdir(path, 0777) != 0 && errno != EEXIST) return 0;
        if (!__sev_path_is_dir(path)) return 0;
        *separator = '/';
    }
    if (mkdir(path, 0777) != 0 && errno != EEXIST) return 0;
    return __sev_path_is_dir(path);
}

_Bool __sev_os_copy(const char *source, const char *destination) {
    FILE *input = fopen(source, "rb");
    if (input == NULL) return 0;
    FILE *output = fopen(destination, "wb");
    if (output == NULL) {
        fclose(input);
        return 0;
    }
    char buffer[8192];
    size_t count;
    _Bool success = 1;
    while ((count = fread(buffer, 1, sizeof(buffer), input)) != 0) {
        if (fwrite(buffer, 1, count, output) != count) {
            success = 0;
            break;
        }
    }
    if (ferror(input)) success = 0;
    if (fclose(input) != 0) success = 0;
    if (fclose(output) != 0) success = 0;
    return success;
}

int32_t __sev_os_rename(const char *source, const char *destination) {
    return rename(source, destination);
}

int32_t __sev_os_remove(const char *path) {
    return remove(path);
}

int64_t __sev_file_lock(const char *path) {
    int descriptor = open(path, O_RDWR | O_CREAT, 0666);
    if (descriptor < 0) return -1;
    if (flock(descriptor, LOCK_EX) != 0) {
        close(descriptor);
        return -1;
    }
    return descriptor;
}

_Bool __sev_file_unlock(int64_t descriptor) {
    if (descriptor < 0) return 0;
    _Bool success = flock((int)descriptor, LOCK_UN) == 0;
    if (close((int)descriptor) != 0) success = 0;
    return success;
}

const char *__sev_path_canonical(const char *path) {
    char *resolved = realpath(path, NULL);
    return resolved == NULL ? "" : resolved;
}

_Bool __sev_os_is_symlink(const char *path) {
    struct stat info;
    return lstat(path, &info) == 0 && S_ISLNK(info.st_mode);
}

const char *__sev_os_temporary(const char *parent, const char *prefix) {
    size_t size = strlen(parent) + strlen(prefix) + 9;
    char *path = sev_memory_allocate(size);
    if (!path) abort();
    snprintf(path, size, "%s/%sXXXXXX", parent, prefix);
    if (mkdtemp(path) == NULL) { sev_memory_release(path); return ""; }
    return path;
}

// Handles avoid exposing dirent layouts or native collection storage to users.
int64_t __sev_os_directory_open(const char *path) {
    return (int64_t)(intptr_t)opendir(path);
}

static _Thread_local int sev_directory_error;
const char *__sev_os_directory_next(int64_t handle) {
    errno = 0;
    struct dirent *entry = readdir((DIR *)(intptr_t)handle);
    sev_directory_error = errno;
    return entry == NULL ? "" : entry->d_name;
}
int32_t __sev_os_directory_error(void) { return sev_directory_error; }
int32_t __sev_os_directory_close(int64_t handle) {
    return closedir((DIR *)(intptr_t)handle);
}

// Commit a directory snapshot without ever replacing an existing name,
// including dangling symlinks. Do not emulate this with check-then-rename.
int32_t __sev_os_publish_directory(const char *source, const char *destination) {
#if defined(__linux__) && defined(SYS_renameat2)
    if (syscall(SYS_renameat2, AT_FDCWD, source, AT_FDCWD, destination, 1) == 0) return 0;
    return errno;
#else
    (void)source; (void)destination;
    return ENOTSUP;
#endif
}

int32_t __sev_os_remove_tree(const char *path) {
    struct stat info;
    if (lstat(path, &info) != 0) return errno == ENOENT ? 0 : errno;
    if (!S_ISDIR(info.st_mode)) return unlink(path) == 0 ? 0 : errno;
    DIR *directory = opendir(path);
    if (!directory) return errno;
    int result = 0;
    for (;;) {
        errno = 0;
        struct dirent *entry = readdir(directory);
        if (!entry) { if (errno) result = errno; break; }
        if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) continue;
        size_t size = strlen(path) + strlen(entry->d_name) + 2;
        char *child = sev_memory_allocate(size);
        if (!child) abort();
        snprintf(child, size, "%s/%s", path, entry->d_name);
        result = __sev_os_remove_tree(child);
        sev_memory_release(child);
        if (result) break;
    }
    if (closedir(directory) != 0 && !result) result = errno;
    if (!result && rmdir(path) != 0) result = errno;
    return result;
}
