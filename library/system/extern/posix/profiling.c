#define _POSIX_C_SOURCE 200809L
#include "../../../core/memory/native/memory.h"
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    char *name;
    uint64_t costs[6];
    uint64_t seen;
} profile_row;

typedef struct {
    profile_row *slots;
    size_t capacity, length;
    uint64_t generation;
} profile_table;

static uint64_t name_hash(const char *name) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (const unsigned char *p = (const unsigned char *)name; *p; ++p)
        hash = (hash ^ *p) * UINT64_C(1099511628211);
    return hash;
}

static int grow(profile_table *table) {
    size_t capacity = table->capacity ? table->capacity * 2 : 1024;
    if (capacity < table->capacity || capacity > SIZE_MAX / sizeof(profile_row)) return EOVERFLOW;
    profile_row *slots = sev_memory_zeroed(capacity, sizeof(*slots));
    if (!slots) return ENOMEM;
    for (size_t i = 0; i < table->capacity; ++i) {
        profile_row row = table->slots[i];
        if (!row.name) continue;
        size_t index = (size_t)name_hash(row.name) & (capacity - 1);
        while (slots[index].name) index = (index + 1) & (capacity - 1);
        slots[index] = row;
    }
    sev_memory_release(table->slots);
    table->slots = slots;
    table->capacity = capacity;
    return 0;
}

static int add_cost(profile_table *table, const char *name, uint64_t cost, int metric, int leaf) {
    if (!table->capacity || table->length >= table->capacity / 2) {
        int error = grow(table);
        if (error) return error;
    }
    size_t index = (size_t)name_hash(name) & (table->capacity - 1);
    while (table->slots[index].name && strcmp(table->slots[index].name, name))
        index = (index + 1) & (table->capacity - 1);
    profile_row *row = &table->slots[index];
    if (!row->name) {
        row->name = sev_memory_copy_text(name);
        if (!row->name) return ENOMEM;
        ++table->length;
    }
    if (row->seen != table->generation) {
        if (UINT64_MAX - row->costs[metric * 2 + 1] < cost) return EOVERFLOW;
        row->costs[metric * 2 + 1] += cost;
        row->seen = table->generation;
    }
    if (leaf) {
        if (UINT64_MAX - row->costs[metric * 2] < cost) return EOVERFLOW;
        row->costs[metric * 2] += cost;
    }
    return 0;
}

static int read_costs(profile_table *table, const char *path, int metric) {
    FILE *input = fopen(path, "r");
    if (!input) return errno;
    char *line = NULL;
    size_t capacity = 0;
    ssize_t length;
    int error = 0;
    while ((length = getline(&line, &capacity, input)) >= 0) {
        while (length && (line[length - 1] == '\n' || line[length - 1] == '\r')) line[--length] = 0;
        if (!length) continue;
        char *amount = strrchr(line, ' ');
        if (!amount || amount[1] < '0' || amount[1] > '9') { error = EINVAL; break; }
        *amount++ = 0;
        char *end;
        errno = 0;
        uint64_t cost = strtoull(amount, &end, 10);
        if (errno || *end) { error = EINVAL; break; }
        if (!cost) continue;
        if (++table->generation == 0) { error = EOVERFLOW; break; }
        /* Heaptrack leaves a trailing semicolon. Ignore empty frames. */
        size_t stack_length = strlen(line);
        while (stack_length && line[stack_length - 1] == ';') line[--stack_length] = 0;
        char *frame = line;
        while (*frame) {
            char *separator = strchr(frame, ';');
            if (separator) *separator = 0;
            if (*frame) error = add_cost(table, frame, cost, metric, separator == NULL);
            if (error || !separator) break;
            frame = separator + 1;
        }
        if (error) break;
    }
    if (!error && ferror(input)) error = EIO;
    sev_memory_release(line);
    if (fclose(input) && !error) error = EIO;
    return error;
}

static int ranked(const void *left, const void *right) {
    const profile_row *a = *(const profile_row *const *)left;
    const profile_row *b = *(const profile_row *const *)right;
    if (a->costs[1] != b->costs[1]) return a->costs[1] > b->costs[1] ? -1 : 1;
    return strcmp(a->name, b->name);
}

/* Streaming provider for large native traces. Only distinct function rows and
 * one input line remain live, independent of the trace's allocation count. */
int64_t __sev_profile_function_table(const char *allocations, const char *peak,
                                    const char *retained, const char *output) {
    profile_table table = {0};
    int error = read_costs(&table, allocations, 0);
    if (!error) error = read_costs(&table, peak, 1);
    if (!error) error = read_costs(&table, retained, 2);
    profile_row **rows = NULL;
    char *temporary = NULL;
    FILE *stream = NULL;
    if (!error) {
        rows = sev_memory_zeroed(table.length ? table.length : 1, sizeof(*rows));
        temporary = sev_memory_allocate(strlen(output) + 12);
        if (!rows || !temporary) error = ENOMEM;
    }
    if (!error) {
        size_t count = 0;
        for (size_t i = 0; i < table.capacity; ++i)
            if (table.slots[i].name) rows[count++] = &table.slots[i];
        qsort(rows, count, sizeof(*rows), ranked);
        sprintf(temporary, "%s.XXXXXX", output);
        int descriptor = mkstemp(temporary);
        if (descriptor < 0) error = errno;
        else {
            stream = fdopen(descriptor, "w");
            if (!stream) { error = errno; close(descriptor); }
        }
        if (!error) {
            fputs("function\tself_allocations\tinclusive_allocations\tself_peak_bytes\tinclusive_peak_bytes\tself_retained_bytes\tinclusive_retained_bytes\n", stream);
            for (size_t i = 0; i < count; ++i) {
                for (const char *p = rows[i]->name; *p; ++p) fputc(*p == '\t' ? ' ' : *p, stream);
                for (int metric = 0; metric < 6; ++metric) fprintf(stream, "\t%" PRIu64, rows[i]->costs[metric]);
                fputc('\n', stream);
            }
            if (ferror(stream)) error = EIO;
            if (fclose(stream) && !error) error = EIO;
            if (!error && rename(temporary, output)) error = errno;
        }
        if (error) unlink(temporary);
    }
    sev_memory_release(temporary);
    sev_memory_release(rows);
    for (size_t i = 0; i < table.capacity; ++i) sev_memory_release(table.slots[i].name);
    sev_memory_release(table.slots);
    return error;
}
