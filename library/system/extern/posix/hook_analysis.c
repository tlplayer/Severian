#define _GNU_SOURCE
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char *name, *file;
    uint64_t line, column;
    uint64_t source, start, end, calls, errors, incomplete;
    double inclusive, self;
    uint64_t measured, allocations, bytes, self_allocations, self_bytes;
} function_cost;
typedef struct invocation {
    uint64_t pid, thread, id;
    size_t function;
    double children;
    _Bool measured;
    uint64_t allocations, bytes, child_allocations, child_bytes;
    struct invocation *parent, *next;
} invocation;

static const char *field(const char *line, const char *key) {
    const char *value = strstr(line, key);
    return value ? value + strlen(key) : NULL;
}
static int number(const char *line, const char *key, uint64_t *result) {
    const char *value = field(line, key);
    if (!value || *value < '0' || *value > '9') return 0;
    char *end;
    errno = 0;
    *result = strtoull(value, &end, 10);
    return !errno && (*end == ',' || *end == '}');
}
static char *string_field(const char *line, const char *key) {
    const char *start = field(line, key);
    if (!start || *start != '"') return NULL;
    const char *end = start + 1;
    while (*end && *end != '"') {
        if (*end == '\\' && end[1]) ++end;
        ++end;
    }
    return *end == '"' ? strndup(start, (size_t)(end - start + 1)) : NULL;
}

static int compare_cost(const void *left, const void *right) {
    const function_cost *a = left, *b = right;
    if (a->inclusive != b->inclusive) return a->inclusive < b->inclusive ? 1 : -1;
    return strcmp(a->name, b->name);
}

/* Streaming aggregation of our versioned hook emission shape. Open calls are
 * kept only until their without event; incomplete calls remain explicit and
 * never receive fabricated durations after abnormal process termination. */
int64_t __sev_hook_function_table(const char *input, const char *output) {
    FILE *stream = fopen(input, "r");
    if (!stream) return errno;
    function_cost *functions = NULL;
    size_t count = 0, capacity = 0;
    invocation *active = NULL;
    char *line = NULL;
    size_t length = 0;
    int status = 0;
    while (getline(&line, &length, stream) >= 0) {
        uint64_t pid, thread, id;
        if (!number(line, "\"pid\":", &pid) || !number(line, "\"thread\":", &thread)
            || !number(line, "\"id\":", &id)) { status = EINVAL; break; }
        if (strstr(line, "\"event\":\"with\"")) {
            uint64_t source, start, end;
            if (!number(line, "\"source_id\":", &source) || !number(line, "\"start\":", &start)
                || !number(line, "\"end\":", &end) || start > end) { status = EINVAL; break; }
            const char *name = field(line, "\"function\":");
            if (!name || *name != '"') { status = EINVAL; break; }
            const char *finish = name + 1;
            while (*finish && *finish != '"') {
                if (*finish == '\\' && finish[1]) ++finish;
                ++finish;
            }
            if (*finish != '"') { status = EINVAL; break; }
            char *label = strndup(name, (size_t)(finish - name + 1));
            if (!label) { status = ENOMEM; break; }
            size_t index = 0;
            for (; index < count; ++index) {
                function_cost *cost = &functions[index];
                if (cost->source == source && cost->start == start && cost->end == end
                    && !strcmp(cost->name, label)) break;
            }
            if (index == count) {
                if (count == capacity) {
                    size_t next = capacity ? capacity * 2 : 16;
                    if (next < capacity || next > SIZE_MAX / sizeof(*functions)) { free(label); status = EOVERFLOW; break; }
                    void *storage = realloc(functions, next * sizeof(*functions));
                    if (!storage) { free(label); status = ENOMEM; break; }
                    functions = storage; capacity = next;
                }
                char *file = string_field(line, "\"file\":");
                if (!file) file = strdup("\"\"");
                if (!file) { free(label); status = ENOMEM; break; }
                uint64_t row = 0, column = 0;
                (void)number(line, "\"line\":", &row);
                (void)number(line, "\"column\":", &column);
                functions[count++] = (function_cost){ .name = label, .file = file, .line = row, .column = column,
                    .source = source, .start = start, .end = end };
            } else free(label);
            invocation *parent = NULL;
            for (invocation *call = active; call; call = call->next) {
                if (call->pid == pid && call->thread == thread) {
                    if (call->id == id) { status = EINVAL; break; }
                    if (!parent) parent = call;
                }
            }
            if (status) break;
            invocation *call = malloc(sizeof(*call));
            if (!call) { status = ENOMEM; break; }
            *call = (invocation){ .pid = pid, .thread = thread, .id = id,
                .function = index, .parent = parent, .next = active };
            call->measured = number(line, "\"allocation_calls\":", &call->allocations)
                && number(line, "\"allocated_bytes\":", &call->bytes);
            active = call;
            functions[index].calls++;
        } else if (strstr(line, "\"event\":\"without\"")) {
            invocation **position = &active;
            while (*position && ((*position)->pid != pid || (*position)->thread != thread || (*position)->id != id))
                position = &(*position)->next;
            if (!*position) { status = EINVAL; break; }
            invocation *call = *position;
            for (invocation *child = active; child != call; child = child->next) {
                if (child->pid == pid && child->thread == thread) { status = EINVAL; break; }
            }
            if (status) break;
            const char *elapsed = field(line, "\"duration_seconds\":");
            const char *error = field(line, "\"error\":");
            if (!elapsed || !error) { status = EINVAL; break; }
            char *end;
            errno = 0;
            double duration = strtod(elapsed, &end);
            if (errno || end == elapsed || *end != ',' || !isfinite(duration) || duration < 0
                || ((strncmp(error, "true", 4) || (error[4] != '}' && error[4] != ','))
                    && (strncmp(error, "false", 5) || (error[5] != '}' && error[5] != ',')))) { status = EINVAL; break; }
            function_cost *cost = &functions[call->function];
            cost->inclusive += duration;
            cost->self += duration > call->children ? duration - call->children : 0.0;
            cost->errors += *error == 't';
            uint64_t allocations, bytes;
            if (call->measured && number(line, "\"allocation_calls\":", &allocations)
                && number(line, "\"allocated_bytes\":", &bytes)) {
                if (allocations < call->allocations || bytes < call->bytes) { status = EINVAL; break; }
                allocations -= call->allocations; bytes -= call->bytes;
                if (allocations < call->child_allocations || bytes < call->child_bytes) { status = EINVAL; break; }
                cost->measured++;
                cost->allocations += allocations; cost->bytes += bytes;
                cost->self_allocations += allocations - call->child_allocations;
                cost->self_bytes += bytes - call->child_bytes;
                if (call->parent) { call->parent->child_allocations += allocations; call->parent->child_bytes += bytes; }
            }
            if (call->parent) call->parent->children += duration;
            *position = call->next;
            free(call);
        } else { status = EINVAL; break; }
    }
    if (ferror(stream) && !status) status = EIO;
    fclose(stream); free(line);
    while (active) {
        invocation *next = active->next;
        functions[active->function].incomplete++;
        free(active); active = next;
    }
    if (!status) {
        if (count > 1) qsort(functions, count, sizeof(*functions), compare_cost);
        FILE *report = fopen(output, "w");
        if (!report) status = errno;
        else {
            fputs("rank\tfunction\tsource_id\tstart\tend\tcalls\terrors\tincomplete\tself_seconds\tinclusive_seconds\tmeasured_calls\tself_allocations\tinclusive_allocations\tself_allocated_bytes\tinclusive_allocated_bytes\tfile\tline\tcolumn\n", report);
            for (size_t index = 0; index < count; ++index) {
                function_cost *cost = &functions[index];
                fprintf(report, "%zu\t%s\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%.9f\t%.9f\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%s\t%"PRIu64"\t%"PRIu64"\n",
                    index + 1, cost->name, cost->source, cost->start, cost->end,
                    cost->calls, cost->errors, cost->incomplete, cost->self, cost->inclusive,
                    cost->measured, cost->self_allocations, cost->allocations, cost->self_bytes, cost->bytes,
                    cost->file, cost->line, cost->column);
            }
            if (ferror(report)) status = EIO;
            if (fclose(report) && !status) status = errno;
        }
    }
    for (size_t index = 0; index < count; ++index) { free(functions[index].name); free(functions[index].file); }
    free(functions);
    return status;
}
