#define _GNU_SOURCE
#include "../../../core/memory/native/memory.h"
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <time.h>
#include <unistd.h>

extern uint64_t __sev_storage_thread_allocations(void) __attribute__((weak));
extern uint64_t __sev_storage_thread_allocated_bytes(void) __attribute__((weak));
extern uint64_t __sev_storage_live_bytes(void) __attribute__((weak));
extern const char *__sev_profile_source_path(int64_t source) __attribute__((weak));
extern const char *__sev_profile_source_text(int64_t source) __attribute__((weak));

typedef struct { int64_t source, start; const char *path; uint64_t line, column; } source_location;
static _Thread_local source_location locations[64];
static source_location location(int64_t source, int64_t start) {
    size_t slot = ((uint64_t)source ^ (uint64_t)start) % 64;
    source_location *known = &locations[slot];
    if (known->path && known->source == source && known->start == start) return *known;
    const char *path = __sev_profile_source_path ? __sev_profile_source_path(source) : NULL;
    const char *text = __sev_profile_source_text ? __sev_profile_source_text(source) : NULL;
    source_location result = { .source = source, .start = start, .path = path ? path : "" };
    if (text) {
        result.line = result.column = 1;
        int64_t index = 0;
        for (; index < start && text[index]; ++index) {
            unsigned char character = (unsigned char)text[index];
            if (character == '\n') { ++result.line; result.column = 1; }
            else if ((character & 0xc0) != 0x80) ++result.column;
        }
        if (index != start) result.line = result.column = 0;
    }
    *known = result;
    return result;
}

static char *escape(const char *text) {
    size_t length = strlen(text);
    if (length > (SIZE_MAX - 1) / 6) return NULL;
    char *result = sev_memory_allocate(length * 6 + 1);
    if (!result) return NULL;
    char *cursor = result;
    for (const unsigned char *p = (const unsigned char *)text; *p; ++p) {
        if (*p == '"' || *p == '\\') { *cursor++ = '\\'; *cursor++ = (char)*p; }
        else if (*p < 32) { snprintf(cursor, 7, "\\u%04x", *p); cursor += 6; }
        else *cursor++ = (char)*p;
    }
    *cursor = 0;
    return result;
}

static _Atomic uint64_t sequence;

static void allocation_snapshot(char *output, size_t capacity) {
    if (__sev_storage_thread_allocations && __sev_storage_thread_allocated_bytes && __sev_storage_live_bytes)
        snprintf(output, capacity, ",\"allocation_calls\":%" PRIu64 ",\"allocated_bytes\":%" PRIu64 ",\"process_live_bytes\":%" PRIu64,
            __sev_storage_thread_allocations(), __sev_storage_thread_allocated_bytes(), __sev_storage_live_bytes());
    else snprintf(output, capacity, ",\"allocation_calls\":null,\"allocated_bytes\":null,\"process_live_bytes\":null");
}

static int timestamp(uint64_t *value) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now)) return errno;
    *value = (uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec;
    return 0;
}

/* An emission is a complete record under an inter-process lock. No process or
 * thread keeps a descriptor, buffer or application reference after this call.
 * Abnormal termination leaves an unmatched with record, never a fake without. */
static int record(const char *path, const char *text, size_t length) {
    int fd = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
    if (fd < 0) return errno;
    int error = 0;
    while (flock(fd, LOCK_EX)) {
        if (errno != EINTR) { error = errno; break; }
    }
    size_t written = 0;
    while (!error && written < length) {
        ssize_t count = write(fd, text + written, length - written);
        if (count < 0) { if (errno != EINTR) error = errno; }
        else if (count == 0) error = EIO;
        else written += (size_t)count;
    }
    if (close(fd) && !error) error = errno;
    return error;
}

int64_t __sev_hook_record_with(const char *function, int64_t source,
                              int64_t start, int64_t end) {
    const char *path = getenv("SEVERIAN_HOOK_RECORDS");
    if (!path || !*path) return 0;
    if (!function || source < 0 || start < 0 || end < start) return -EINVAL;
    source_location source_at = location(source, start);
    char *escaped = escape(function);
    char *file = escape(source_at.path);
    if (!escaped || !file) { sev_memory_release(escaped); sev_memory_release(file); return -ENOMEM; }
    size_t length = strlen(escaped) + strlen(file) + 768;
    char *line = sev_memory_allocate(length);
    if (!line) { sev_memory_release(escaped); sev_memory_release(file); return -ENOMEM; }
    char costs[192];
    allocation_snapshot(costs, sizeof(costs));
    uint64_t now;
    int error = timestamp(&now);
    uint64_t id = atomic_fetch_add(&sequence, 1) + 1;
    if (!id || id > INT64_MAX) error = EOVERFLOW;
    if (!error) {
        int count = snprintf(line, length,
            "{\"event\":\"with\",\"pid\":%ld,\"thread\":%ld,\"id\":%" PRIu64 ",\"function\":\"%s\",\"source_id\":%" PRId64 ",\"start\":%" PRId64 ",\"end\":%" PRId64 ",\"time_ns\":%" PRIu64 ",\"file\":\"%s\",\"line\":%" PRIu64 ",\"column\":%" PRIu64 "%s}\n",
            (long)getpid(), (long)gettid(), id, escaped, source, start, end, now, file, source_at.line, source_at.column, costs);
        if (count < 0 || (size_t)count >= length) error = EOVERFLOW;
        else error = record(path, line, (size_t)count);
    }
    sev_memory_release(escaped); sev_memory_release(file); sev_memory_release(line);
    return error ? -(int64_t)error : (int64_t)id;
}

int64_t __sev_hook_record_without(int64_t id, double duration, _Bool failed) {
    if (!id) return 0;
    if (id < 0 || !isfinite(duration) || duration < 0) return EINVAL;
    const char *path = getenv("SEVERIAN_HOOK_RECORDS");
    if (!path || !*path) return ENOENT;
    char costs[192];
    allocation_snapshot(costs, sizeof(costs));
    uint64_t now;
    int error = timestamp(&now);
    if (error) return error;
    char line[512];
    int count = snprintf(line, sizeof(line),
        "{\"event\":\"without\",\"pid\":%ld,\"thread\":%ld,\"id\":%" PRId64 ",\"time_ns\":%" PRIu64 ",\"duration_seconds\":%.17g,\"error\":%s%s}\n",
        (long)getpid(), (long)gettid(), id, now, duration, failed ? "true" : "false", costs);
    if (count < 0 || (size_t)count >= sizeof(line)) return EOVERFLOW;
    return record(path, line, (size_t)count);
}
