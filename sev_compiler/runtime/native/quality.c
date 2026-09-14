#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <stdatomic.h>

static FILE *report;
static _Atomic bool active;
struct allocation { void *pointer; size_t bytes; uint64_t generation; struct allocation *next; };
static struct allocation *live;
static atomic_flag memory_lock = ATOMIC_FLAG_INIT;
static uint64_t generation, test_generation, allocated_bytes, test_bytes;

void __sev_quality_allocation(void *pointer, size_t bytes) {
    struct allocation *item = malloc(sizeof(*item));
    if (!item) { perror("quality allocation tracker"); exit(74); }
    while (atomic_flag_test_and_set(&memory_lock)) {}
    *item = (struct allocation){pointer, bytes, ++generation, live};
    allocated_bytes += bytes;
    live = item;
    atomic_flag_clear(&memory_lock);
}

void __sev_quality_release(void *pointer) {
    while (atomic_flag_test_and_set(&memory_lock)) {}
    struct allocation **cursor = &live;
    while (*cursor && (*cursor)->pointer != pointer) cursor = &(*cursor)->next;
    if (*cursor) {
        struct allocation *item = *cursor;
        *cursor = item->next;
        free(item);
    }
    atomic_flag_clear(&memory_lock);
}

static void write_record(char kind, uint64_t value) {
    if (!report) {
        const char *path = getenv("SEV_COVERAGE_FILE");
        if (!path || !*path) return;
        report = fopen(path, "a");
        if (!report) { perror("quality report"); exit(74); }
    }
    if (fprintf(report, "%c:%" PRIu64 "\n", kind, value) < 0 || fflush(report)) {
        perror("quality report"); exit(74);
    }
}

void __sev_quality_hit(uint64_t id) {
    if (active) write_record('H', id);
}

void __sev_quality_decision(uint64_t otherwise, uint64_t taken, bool value) {
    __sev_quality_hit(value ? taken : otherwise);
}

void __sev_quality_test_begin(uint64_t id) {
    write_record('B', id);
    while (atomic_flag_test_and_set(&memory_lock)) {}
    test_generation = generation;
    test_bytes = allocated_bytes;
    atomic_flag_clear(&memory_lock);
    active = true;
}

void __sev_quality_test_end(uint64_t id) {
    active = false;
    uint64_t retained = 0;
    while (atomic_flag_test_and_set(&memory_lock)) {}
    uint64_t count = generation - test_generation;
    uint64_t bytes = allocated_bytes - test_bytes;
    for (struct allocation *item = live; item; item = item->next)
        if (item->generation > test_generation) retained += item->bytes;
    atomic_flag_clear(&memory_lock);
    write_record('A', count);
    write_record('M', bytes);
    write_record('L', retained);
    write_record('E', id);
}
