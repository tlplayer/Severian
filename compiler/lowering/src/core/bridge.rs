use super::*;

mod runtime_diagnostics;
mod value_formatting;

/// C bridge linked beside generated LLVM IR to execute Severian tasks on pthreads.
/// Generates ABI glue for language values, classes, tasks, and channels.
/// Concrete database and tensor providers live in `severian-platform`.
pub fn native_bridge_source(
    program: &Program,
) -> Result<String, stablehlo::StableHloLoweringError> {
    native_bridge_source_for_target(program, false)
}

/// C bridge using HIP managed tensor storage and MLIR's ROCm runtime ABI.
pub fn rocm_bridge_source(program: &Program) -> Result<String, stablehlo::StableHloLoweringError> {
    native_bridge_source_for_target(program, true)
}

pub(super) fn native_bridge_source_for_target(
    program: &Program,
    rocm: bool,
) -> Result<String, stablehlo::StableHloLoweringError> {
    let specs = task_specs(program);
    let uses_channels = uses_channels(program);
    let mut source = String::from(concat!(
        "#include <pthread.h>\n",
        "#include <arpa/inet.h>\n",
        "#include <fcntl.h>\n",
        "#include <ctype.h>\n",
        "#include <dirent.h>\n",
        "#include <errno.h>\n",
        "#include <math.h>\n",
        "#include <stdbool.h>\n",
        "#include <stdint.h>\n",
        "#include <limits.h>\n",
        "#include <stdio.h>\n",
        "#include <stdlib.h>\n",
        "#include <string.h>\n",
        "#include <time.h>\n",
        "#include <sys/socket.h>\n",
        "#include <sys/ioctl.h>\n",
        "#include <sys/file.h>\n",
        "#include <sys/mman.h>\n",
        "#include <sys/wait.h>\n",
        "#include <sys/stat.h>\n",
        "#include <sys/syscall.h>\n",
        "#include <unistd.h>\n",
        "#include <signal.h>\n",
        "#include <regex.h>\n",
        "#ifdef __linux__\n#include <linux/kvm.h>\n#endif\n\n",
        "typedef enum { SEV_INT, SEV_FLOAT, SEV_BOOL, SEV_STRING, SEV_COLLECTION, SEV_NULL } sev_value_kind;\n",
        "typedef struct { sev_value_kind kind; union { int64_t i64; double f64; bool boolean; const char *string; void *pointer; } as; } sev_value;\n",
        "typedef struct { int64_t kind; int64_t size; int64_t capacity; sev_value **items; } sev_collection;\n",
        "typedef struct { int64_t kind; int64_t size; int64_t capacity; sev_value **keys; sev_value **values; } sev_map;\n\n",
        "#define SEV_OBJECT_MAGIC UINT64_C(0x5345564f424a4543)\n",
        "#define SEV_VARIANT_MAGIC UINT64_C(0x5345565641524941)\n",
        "#define SEV_TENSOR_MAGIC UINT64_C(0x53455654454e534f)\n",
        "typedef struct { uint64_t magic; const char *class_name; int64_t size; int64_t capacity; const char **names; sev_value **values; pthread_mutex_t mutex; } sev_object;\n\n",
        "typedef struct { uint64_t magic; const char *tag; sev_value *field; } sev_variant;\n\n",
        "typedef struct { uint64_t magic; int64_t rank; int64_t *shape; int64_t *strides; int64_t size; } sev_tensor_header;\n\n",
        "typedef struct { void *function; void *environment; } sev_closure;\n\n",
        "typedef struct sev_allocation { struct sev_allocation *previous; struct sev_allocation *next; size_t size; long double alignment; } sev_allocation;\n",
        "static uint64_t sev_allocated_bytes = 0;\n",
        "static uint64_t sev_allocation_count = 0;\n",
        "static pthread_mutex_t sev_allocation_mutex = PTHREAD_MUTEX_INITIALIZER;\n",
        "static sev_allocation *sev_allocations = NULL;\n",
        "static bool sev_cleanup_registered = false;\n",
    ));
    source.push_str(runtime_diagnostics::SOURCE);
    source.push_str(concat!(
        "static void sev_cleanup_allocations(void) { pthread_mutex_lock(&sev_allocation_mutex); sev_allocation *allocation = sev_allocations; sev_allocations = NULL; pthread_mutex_unlock(&sev_allocation_mutex); while (allocation) { sev_allocation *next = allocation->next; free(allocation); allocation = next; } }\n",
        "static void *sev_allocate(size_t size) { sev_allocation *allocation = calloc(1, sizeof(*allocation) + size); if (!allocation) abort(); allocation->size = size; pthread_mutex_lock(&sev_allocation_mutex); if (!sev_cleanup_registered) { if (atexit(sev_cleanup_allocations) != 0) abort(); sev_cleanup_registered = true; } allocation->next = sev_allocations; if (sev_allocations) sev_allocations->previous = allocation; sev_allocations = allocation; pthread_mutex_unlock(&sev_allocation_mutex); __atomic_fetch_add(&sev_allocated_bytes, size, __ATOMIC_RELAXED); __atomic_fetch_add(&sev_allocation_count, 1, __ATOMIC_RELAXED); return allocation + 1; }\n",
        "static void sev_release(void *value) { if (!value) return; sev_allocation *allocation = (sev_allocation *)value - 1; pthread_mutex_lock(&sev_allocation_mutex); if (allocation->previous) allocation->previous->next = allocation->next; else sev_allocations = allocation->next; if (allocation->next) allocation->next->previous = allocation->previous; pthread_mutex_unlock(&sev_allocation_mutex); free(allocation); }\n",
        "static void *sev_reallocate(void *value, size_t size) { if (!value) return sev_allocate(size); sev_allocation *old = (sev_allocation *)value - 1; void *replacement = sev_allocate(size); memcpy(replacement, value, old->size < size ? old->size : size); sev_release(value); return replacement; }\n",
        "static void *sev_callocate(size_t count, size_t size) { if (size && count > SIZE_MAX / size) abort(); return sev_allocate(count * size); }\n",
        "static char *sev_duplicate(const char *value) { size_t size = strlen(value) + 1; char *copy = sev_allocate(size); memcpy(copy, value, size); return copy; }\n",
        "static void sev_system_release(void *value) { free(value); }\n",
        "#define free(value) sev_release(value)\n",
        "#define realloc(value, size) sev_reallocate(value, size)\n",
        "#define calloc(count, size) sev_callocate(count, size)\n",
        "#define strdup(value) sev_duplicate(value)\n",
        "void __sev_coverage_hit(int64_t id) { const char *path = getenv(\"SEVERIAN_COVERAGE_FILE\"); if (!path || !*path) return; int fd = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0666); if (fd < 0) abort(); char line[32]; int size = snprintf(line, sizeof(line), \"%lu\\n\", (uint64_t)id); if (size <= 0 || write(fd, line, (size_t)size) != size) { close(fd); abort(); } close(fd); }\n",
        "int64_t __sev_monotonic_ns(void) { struct timespec value; if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) abort(); return (int64_t)value.tv_sec * 1000000000 + value.tv_nsec; }\n",
        "int64_t __sev_allocation_bytes(void) { return (int64_t)__atomic_load_n(&sev_allocated_bytes, __ATOMIC_RELAXED); }\n",
        "int64_t __sev_allocation_count(void) { return (int64_t)__atomic_load_n(&sev_allocation_count, __ATOMIC_RELAXED); }\n",
        "void __sev_contract_fail(void *message_raw, void *location_raw, void *vars_raw) { const char *message = message_raw; const char *location = location_raw; const char *vars = vars_raw; fprintf(stderr, \"contract error: %s\\n\", message && *message ? message : \"contract condition was not satisfied\"); if (location && *location) fprintf(stderr, \"location: %s\\n\", location); if (vars && *vars) fprintf(stderr, \"vars: %s\\n\", vars); exit(70); }\n",
        "void *__sev_box_i64(int64_t raw) { sev_value *value = sev_allocate(sizeof(*value)); value->kind = SEV_INT; value->as.i64 = raw; return value; }\n",
        "void *__sev_box_f64(double raw) { sev_value *value = sev_allocate(sizeof(*value)); value->kind = SEV_FLOAT; value->as.f64 = raw; return value; }\n",
        "void *__sev_box_bool(bool raw) { sev_value *value = sev_allocate(sizeof(*value)); value->kind = SEV_BOOL; value->as.boolean = raw; return value; }\n",
        "void *__sev_box_string(void *raw) { sev_value *value = sev_allocate(sizeof(*value)); value->kind = SEV_STRING; value->as.string = raw; return value; }\n",
        "void *__sev_box_collection(void *raw) { sev_value *value = sev_allocate(sizeof(*value)); value->kind = SEV_COLLECTION; value->as.pointer = raw; return value; }\n",
        "void *__sev_box_null(void) { sev_value *value = sev_allocate(sizeof(*value)); value->kind = SEV_NULL; return value; }\n",
        "int64_t __sev_unbox_i64(void *raw) { sev_value *value = raw; if (!value || value->kind != SEV_INT) sev_runtime_fail(\"E0921\", \"value cannot be converted to int\", \"expected a runtime integer value\"); return value->as.i64; }\n",
        "double __sev_unbox_f64(void *raw) { sev_value *value = raw; if (!value || value->kind != SEV_FLOAT) sev_runtime_fail(\"E0921\", \"value cannot be converted to float\", \"expected a runtime float value\"); return value->as.f64; }\n",
        "bool __sev_unbox_bool(void *raw) { sev_value *value = raw; if (!value || value->kind != SEV_BOOL) sev_runtime_fail(\"E0921\", \"value cannot be converted to bool\", \"expected a runtime boolean value\"); return value->as.boolean; }\n",
        "void *__sev_unbox_string(void *raw) { sev_value *value = raw; if (!value || value->kind != SEV_STRING) sev_runtime_fail(\"E0921\", \"value cannot be converted to string\", \"expected a runtime string value\"); return (void *)value->as.string; }\n",
        "void *__sev_unbox_ptr(void *raw) { sev_value *value = raw; if (!value || value->kind != SEV_COLLECTION) sev_runtime_fail(\"E0921\", \"value is not a collection\", \"expected a runtime collection value\"); return value->as.pointer; }\n",
        "void *__sev_closure_new(void *function, void *environment) { sev_closure *closure = sev_allocate(sizeof(*closure)); closure->function = function; closure->environment = environment; return closure; }\n",
        "void *__sev_closure_function(void *raw) { sev_closure *closure = raw; if (!closure || !closure->function) abort(); return closure->function; }\n",
        "void *__sev_closure_environment(void *raw) { sev_closure *closure = raw; if (!closure) abort(); return closure->environment; }\n",
        "static double sev_number(sev_value *value) { if (!value) abort(); if (value->kind == SEV_FLOAT) return value->as.f64; if (value->kind == SEV_INT) return (double)value->as.i64; abort(); }\n",
        "void *__sev_value_add(void *left_raw, void *right_raw) { sev_value *left = left_raw; sev_value *right = right_raw; if (left && right && left->kind == SEV_INT && right->kind == SEV_INT) return __sev_box_i64(left->as.i64 + right->as.i64); return __sev_box_f64(sev_number(left) + sev_number(right)); }\n",
        "void *__sev_value_sub(void *left_raw, void *right_raw) { sev_value *left = left_raw; sev_value *right = right_raw; if (left && right && left->kind == SEV_INT && right->kind == SEV_INT) return __sev_box_i64(left->as.i64 - right->as.i64); return __sev_box_f64(sev_number(left) - sev_number(right)); }\n",
        "void *__sev_value_mul(void *left_raw, void *right_raw) { sev_value *left = left_raw; sev_value *right = right_raw; if (left && right && left->kind == SEV_INT && right->kind == SEV_INT) return __sev_box_i64(left->as.i64 * right->as.i64); return __sev_box_f64(sev_number(left) * sev_number(right)); }\n",
        "void *__sev_value_div(void *left_raw, void *right_raw) { sev_value *left = left_raw; sev_value *right = right_raw; if (left && right && left->kind == SEV_INT && right->kind == SEV_INT) { if (right->as.i64 == 0) __sev_runtime_fail_division_zero(); return __sev_box_i64(left->as.i64 / right->as.i64); } double divisor = sev_number(right); if (divisor == 0.0) __sev_runtime_fail_division_zero(); return __sev_box_f64(sev_number(left) / divisor); }\n",
        "int64_t __sev_value_int(void *raw) { sev_value *value = raw; if (!value) sev_runtime_fail(\"E0921\", \"value cannot be converted to int\", \"the value is missing\"); if (value->kind == SEV_INT) return value->as.i64; if (value->kind == SEV_BOOL) return value->as.boolean ? 1 : 0; if (value->kind == SEV_FLOAT) { double result = value->as.f64; if (!isfinite(result) || result >= 9223372036854775808.0 || result < -9223372036854775808.0) sev_runtime_fail(\"E0921\", \"float cannot be converted to int\", \"the floating-point value is outside the int range\"); return (int64_t)result; } if (value->kind == SEV_STRING) { const char *text = value->as.string; char *end = NULL; errno = 0; long long result = strtoll(text, &end, 10); while (end && isspace((unsigned char)*end)) ++end; if (end == text || (end && *end != '\\0') || errno == ERANGE) sev_runtime_fail(\"E0921\", \"string cannot be converted to int\", text); return (int64_t)result; } sev_runtime_fail(\"E0921\", \"value cannot be converted to int\", \"expected an int, float, bool, or base-10 integer string\"); }\n",
        "double __sev_value_float(void *raw) { sev_value *value = raw; if (!value) sev_runtime_fail(\"E0921\", \"value cannot be converted to float\", \"the value is missing\"); if (value->kind == SEV_FLOAT) return value->as.f64; if (value->kind == SEV_INT) return (double)value->as.i64; if (value->kind == SEV_STRING) { char *end = NULL; double result = strtod(value->as.string, &end); if (end == value->as.string || *end != '\\0') sev_runtime_fail(\"E0921\", \"string cannot be converted to float\", value->as.string); return result; } sev_runtime_fail(\"E0921\", \"value cannot be converted to float\", \"expected an int, float, or numeric string\"); }\n",
    ));
    source.push_str(value_formatting::SOURCE);
    source.push_str(concat!(
        "int64_t __sev_strlen(void *raw) { const char *text = raw; int64_t size = 0; if (!text) abort(); while (text[size]) size += 1; return size; }\n",
        "static int64_t sev_utf8_width(unsigned char lead) { if (lead < 0x80) return 1; if ((lead & 0xE0) == 0xC0) return 2; if ((lead & 0xF0) == 0xE0) return 3; if ((lead & 0xF8) == 0xF0) return 4; abort(); }\n",
        "int64_t __sev_string_length(void *raw) { const unsigned char *text = raw; if (!text) abort(); int64_t bytes = __sev_strlen(raw); int64_t count = 0; for (int64_t offset = 0; offset < bytes; offset += sev_utf8_width(text[offset])) count += 1; return count; }\n",
        "static int64_t sev_utf8_offset(void *raw, int64_t index) { const unsigned char *text = raw; int64_t count = __sev_string_length(raw); if (index < 0) index += count; if (index < 0 || index > count) abort(); int64_t offset = 0; for (int64_t current = 0; current < index; ++current) offset += sev_utf8_width(text[offset]); return offset; }\n",
        "void *__sev_string_concat(void *left_raw, void *right_raw) { const char *left = left_raw; const char *right = right_raw; size_t left_size = (size_t)__sev_strlen(left_raw); size_t right_size = (size_t)__sev_strlen(right_raw); char *result = sev_allocate(left_size + right_size + 1); memcpy(result, left, left_size); memcpy(result + left_size, right, right_size + 1); return result; }\n",
        "bool __sev_string_equal(void *left, void *right) { return strcmp(left, right) == 0; }\n",
        "void *__sev_string_char_at(void *raw, int64_t index) { const unsigned char *text = raw; int64_t count = __sev_string_length(raw); int64_t requested = index; if (index < 0) index += count; if (index < 0 || index >= count) sev_runtime_fail_bounds(\"string\", requested, count); int64_t offset = sev_utf8_offset(raw, index); int64_t width = sev_utf8_width(text[offset]); char *result = sev_allocate((size_t)width + 1); memcpy(result, text + offset, (size_t)width); return result; }\n",
        "static void sev_slice_bounds(int64_t size, int64_t *start, int64_t *end, int64_t *step) { const int64_t missing = INT64_MIN; if (*step == missing) *step = 1; if (*step == 0) sev_runtime_fail(\"E0912\", \"slice step cannot be zero\", \"use a positive or negative non-zero step\"); if (*step > 0) { if (*start == missing) *start = 0; else { if (*start < 0) *start += size; if (*start < 0) *start = 0; if (*start > size) *start = size; } if (*end == missing) *end = size; else { if (*end < 0) *end += size; if (*end < 0) *end = 0; if (*end > size) *end = size; } } else { if (*start == missing) *start = size - 1; else { if (*start < 0) *start += size; if (*start < -1) *start = -1; if (*start >= size) *start = size - 1; } if (*end == missing) *end = -1; else { if (*end < 0) *end += size; if (*end < -1) *end = -1; if (*end >= size) *end = size - 1; } } }\n",
        "void *__sev_string_slice(void *raw, int64_t start, int64_t end, int64_t step) { const unsigned char *text = raw; int64_t count = __sev_string_length(raw); sev_slice_bounds(count, &start, &end, &step); int64_t *offsets = sev_allocate((size_t)(count + 1) * sizeof(*offsets)); offsets[0] = 0; for (int64_t index = 0; index < count; ++index) offsets[index + 1] = offsets[index] + sev_utf8_width(text[offsets[index]]); char *result = sev_allocate((size_t)__sev_strlen(raw) + 1); int64_t write = 0; for (int64_t index = start; step > 0 ? index < end : index > end; index += step) { int64_t width = offsets[index + 1] - offsets[index]; memcpy(result + write, text + offsets[index], (size_t)width); write += width; } result[write] = '\\0'; free(offsets); return result; }\n",
        "static bool sev_value_equal(sev_value *left, sev_value *right) {\n",
        "  if (!left || !right || left->kind != right->kind) return false;\n",
        "  switch (left->kind) { case SEV_INT: return left->as.i64 == right->as.i64; case SEV_FLOAT: return left->as.f64 == right->as.f64; case SEV_BOOL: return left->as.boolean == right->as.boolean; case SEV_STRING: return strcmp(left->as.string, right->as.string) == 0; case SEV_COLLECTION: { sev_collection *left_collection = left->as.pointer; sev_collection *right_collection = right->as.pointer; if (!left_collection || !right_collection || left_collection->kind != right_collection->kind || left_collection->size != right_collection->size) return false; for (int64_t index = 0; index < left_collection->size; ++index) if (!sev_value_equal(left_collection->items[index], right_collection->items[index])) return false; return true; } case SEV_NULL: return true; }\n",
        "  return false;\n",
        "}\n",
        "bool __sev_value_equal(void *left, void *right) { return sev_value_equal(left, right); }\n",
        "bool __sev_value_less(void *left_raw, void *right_raw) { sev_value *left = left_raw; sev_value *right = right_raw; if (!left || !right) abort(); if (left->kind == SEV_STRING && right->kind == SEV_STRING) return strcmp(left->as.string, right->as.string) < 0; return sev_number(left) < sev_number(right); }\n",
        "void *__sev_collection_new(int64_t kind) { sev_collection *value = sev_allocate(sizeof(*value)); value->kind = kind; return value; }\n",
        "void *__sev_collection_clone(void *raw) { sev_collection *value = raw; if (!value) abort(); sev_collection *result = __sev_collection_new(value->kind); result->size = value->size; result->capacity = value->size; if (value->size > 0) { result->items = sev_allocate((size_t)value->size * sizeof(*result->items)); memcpy(result->items, value->items, (size_t)value->size * sizeof(*result->items)); } return result; }\n",
        "void __sev_collection_push(void *raw, void *item) { sev_collection *value = raw; if (value->size == value->capacity) { value->capacity = value->capacity ? value->capacity * 2 : 4; value->items = realloc(value->items, (size_t)value->capacity * sizeof(*value->items)); if (!value->items) abort(); } value->items[value->size++] = item; }\n",
        "void *__sev_collection_get(void *raw, int64_t index) { sev_collection *value = raw; if (!value) sev_runtime_fail_invariant(\"collection storage is null\"); int64_t requested = index; if (index < 0) index += value->size; if (index < 0 || index >= value->size) sev_runtime_fail_bounds(\"collection\", requested, value->size); return value->items[index]; }\n",
        "void *__sev_collection_slice(void *raw, int64_t start, int64_t end, int64_t step) { sev_collection *value = raw; if (!value) abort(); sev_slice_bounds(value->size, &start, &end, &step); sev_collection *result = __sev_collection_new(value->kind); for (int64_t index = start; step > 0 ? index < end : index > end; index += step) __sev_collection_push(result, value->items[index]); return result; }\n",
        "void __sev_collection_insert(void *raw, int64_t index, void *item) { sev_collection *value = raw; if (!value) abort(); if (index < 0) { index += value->size; if (index < 0) index = 0; } if (index > value->size) index = value->size; __sev_collection_push(value, item); memmove(value->items + index + 1, value->items + index, (size_t)(value->size - index - 1) * sizeof(*value->items)); value->items[index] = item; }\n",
        "void __sev_collection_appendleft(void *raw, void *item) { __sev_collection_insert(raw, 0, item); }\n",
        "void __sev_collection_extend(void *raw, void *other_raw) { sev_collection *value = raw; sev_collection *other = other_raw; if (!value || !other) abort(); for (int64_t index = 0; index < other->size; ++index) __sev_collection_push(value, other->items[index]); }\n",
        "void *__sev_collection_concat(void *left_raw, void *right_raw) { sev_collection *left = left_raw; sev_collection *right = right_raw; if (!left || !right || left->kind != 0 || right->kind != 0) abort(); sev_collection *result = __sev_collection_clone(left); __sev_collection_extend(result, right); return result; }\n",
        "void *__sev_collection_pop_at(void *raw, int64_t index) { sev_collection *value = raw; if (!value) sev_runtime_fail_invariant(\"collection storage is null\"); int64_t requested = index; if (index < 0) index += value->size; if (index < 0 || index >= value->size) sev_runtime_fail_bounds(\"collection\", requested, value->size); sev_value *result = value->items[index]; memmove(value->items + index, value->items + index + 1, (size_t)(value->size - index - 1) * sizeof(*value->items)); value->size -= 1; return result; }\n",
        "void __sev_collection_remove(void *raw, void *item) { sev_collection *value = raw; if (!value) abort(); for (int64_t index = 0; index < value->size; ++index) if (sev_value_equal(value->items[index], item)) { (void)__sev_collection_pop_at(value, index); return; } abort(); }\n",
        "void __sev_collection_heap_push(void *raw, void *item) { sev_collection *value = raw; if (!value) abort(); __sev_collection_push(value, item); int64_t child = value->size - 1; while (child > 0) { int64_t parent = (child - 1) / 2; if (!__sev_value_less(value->items[child], value->items[parent])) break; sev_value *temporary = value->items[parent]; value->items[parent] = value->items[child]; value->items[child] = temporary; child = parent; } }\n",
        "void *__sev_collection_heap_pop(void *raw) { sev_collection *value = raw; if (!value || value->size == 0) abort(); sev_value *result = value->items[0]; value->size -= 1; if (value->size == 0) return result; value->items[0] = value->items[value->size]; int64_t parent = 0; while (parent * 2 + 1 < value->size) { int64_t left = parent * 2 + 1; int64_t right = left + 1; int64_t child = right < value->size && __sev_value_less(value->items[right], value->items[left]) ? right : left; if (!__sev_value_less(value->items[child], value->items[parent])) break; sev_value *temporary = value->items[parent]; value->items[parent] = value->items[child]; value->items[child] = temporary; parent = child; } return result; }\n",
        "void __sev_collection_set(void *raw, int64_t index, void *item) { sev_collection *value = raw; if (!value) sev_runtime_fail_invariant(\"collection storage is null\"); int64_t requested = index; if (index < 0) index += value->size; if (index < 0 || index >= value->size) sev_runtime_fail_bounds(\"collection\", requested, value->size); value->items[index] = item; }\n",
        "int64_t __sev_collection_size(void *raw) { sev_collection *value = raw; if (!value) abort(); return value->size; }\n",
        "int64_t __sev_value_size(void *raw) { sev_value *value = raw; if (!value) abort(); if (*(uint64_t *)raw == SEV_TENSOR_MAGIC) return ((sev_tensor_header *)raw)->size; if (value->kind == SEV_STRING) return __sev_string_length((void *)value->as.string); if (value->kind == SEV_COLLECTION) return __sev_collection_size(value->as.pointer); abort(); }\n",
        "int64_t __sev_value_bytes(void *raw) { sev_value *value = raw; if (!value) abort(); if (*(uint64_t *)raw == SEV_TENSOR_MAGIC) return ((sev_tensor_header *)raw)->size * 8; switch (value->kind) { case SEV_INT: case SEV_FLOAT: return 8; case SEV_BOOL: return 1; case SEV_STRING: return __sev_strlen((void *)value->as.string); case SEV_COLLECTION: { sev_collection *collection = value->as.pointer; return (int64_t)sizeof(*collection) + collection->capacity * (int64_t)sizeof(*collection->items); } } abort(); }\n",
        "int64_t __sev_value_capacity(void *raw) { sev_value *value = raw; if (!value) abort(); if (*(uint64_t *)raw == SEV_TENSOR_MAGIC) return ((sev_tensor_header *)raw)->size; if (value->kind == SEV_STRING) return __sev_strlen((void *)value->as.string); if (value->kind == SEV_COLLECTION) return ((sev_collection *)value->as.pointer)->capacity; return 1; }\n",
        "void *__sev_map_get(void *raw, void *key);\n",
        "void __sev_map_insert(void *raw, void *key, void *item);\n",
        "void __sev_collection_set(void *raw, int64_t index, void *item);\n",
        "void *__sev_value_index(void *raw, int64_t index) { sev_value *value = raw; if (!value) abort(); if (value->kind == SEV_COLLECTION) return __sev_collection_get(value->as.pointer, index); if (value->kind == SEV_STRING) return __sev_box_string(__sev_string_char_at((void *)value->as.string, index)); abort(); }\n",
        "void *__sev_value_get(void *raw, void *key_raw) { sev_value *value = raw; sev_value *key = key_raw; if (!value || !key) abort(); if (value->kind == SEV_STRING && key->kind == SEV_INT) return __sev_box_string(__sev_string_char_at((void *)value->as.string, key->as.i64)); if (value->kind != SEV_COLLECTION) abort(); sev_collection *collection = value->as.pointer; if (!collection) abort(); if (collection->kind == 3) return __sev_map_get(collection, key); if (key->kind != SEV_INT) abort(); return __sev_collection_get(collection, key->as.i64); }\n",
        "void __sev_value_set(void *raw, void *key_raw, void *item) { sev_value *value = raw; sev_value *key = key_raw; if (!value || !key || !item || value->kind != SEV_COLLECTION) abort(); sev_collection *collection = value->as.pointer; if (!collection) abort(); if (collection->kind == 3) { __sev_map_insert(collection, key, item); return; } if (key->kind != SEV_INT) abort(); __sev_collection_set(collection, key->as.i64, item); }\n",
        "void *__sev_value_slice(void *raw, int64_t start, int64_t end, int64_t step) { sev_value *value = raw; if (!value) abort(); if (value->kind == SEV_COLLECTION) return __sev_box_collection(__sev_collection_slice(value->as.pointer, start, end, step)); if (value->kind == SEV_STRING) return __sev_box_string(__sev_string_slice((void *)value->as.string, start, end, step)); abort(); }\n",
        "bool __sev_collection_equal(void *left_raw, void *right_raw) { sev_collection *left = left_raw; sev_collection *right = right_raw; if (!left || !right || left->kind != right->kind || left->size != right->size) return false; for (int64_t i = 0; i < left->size; ++i) if (!sev_value_equal(left->items[i], right->items[i])) return false; return true; }\n",
        "void *__sev_collection_reversed(void *raw) { sev_collection *value = raw; if (!value) abort(); sev_collection *result = __sev_collection_new(value->kind); for (int64_t i = value->size; i > 0; --i) __sev_collection_push(result, value->items[i - 1]); return result; }\n",
        "static double sev_fast_sigmoid(double value) { double magnitude = value < 0.0 ? -value : value; return 0.5 + value / (2.0 * (1.0 + magnitude)); }\n",
        "static double sev_fast_tanh(double value) { double magnitude = value < 0.0 ? -value : value; return value / (1.0 + magnitude); }\n",
        "void *__sev_fused_activations(void *raw, int64_t packed, int64_t count) { sev_collection *input = raw; if (!input) abort(); sev_collection *output = __sev_collection_new(0); for (int64_t index = 0; index < input->size; ++index) { double value = sev_number(input->items[index]); for (int64_t stage = 0; stage < count; ++stage) { switch ((packed >> (stage * 4)) & 15) { case 1: value = value < 0.0 ? 0.0 : value; break; case 2: value = sev_fast_sigmoid(value); break; case 3: value = sev_fast_tanh(value); break; case 4: { double cubic = value * value * value; double curved = 0.7978845608 * (value + 0.044715 * cubic); value = 0.5 * value * (1.0 + sev_fast_tanh(curved)); break; } case 5: value = value * sev_fast_sigmoid(value); break; default: abort(); } } __sev_collection_push(output, __sev_box_f64(value)); } return output; }\n",
        "bool __sev_set_contains(void *raw, void *item) { sev_collection *value = raw; for (int64_t i = 0; i < value->size; ++i) if (sev_value_equal(value->items[i], item)) return true; return false; }\n",
        "void __sev_set_add(void *raw, void *item) { if (!__sev_set_contains(raw, item)) __sev_collection_push(raw, item); }\n",
        "void *__sev_map_new(void) { sev_map *value = sev_allocate(sizeof(*value)); value->kind = 3; return value; }\n",
        "void __sev_map_insert(void *raw, void *key, void *item) { sev_map *value = raw; for (int64_t i = 0; i < value->size; ++i) if (sev_value_equal(value->keys[i], key)) { value->values[i] = item; return; } if (value->size == value->capacity) { value->capacity = value->capacity ? value->capacity * 2 : 4; value->keys = realloc(value->keys, (size_t)value->capacity * sizeof(*value->keys)); value->values = realloc(value->values, (size_t)value->capacity * sizeof(*value->values)); if (!value->keys || !value->values) abort(); } value->keys[value->size] = key; value->values[value->size++] = item; }\n",
        "void *__sev_map_get(void *raw, void *key) { sev_map *value = raw; if (!value) sev_runtime_fail_invariant(\"map storage is null\"); for (int64_t i = 0; i < value->size; ++i) if (sev_value_equal(value->keys[i], key)) return value->values[i]; sev_runtime_fail(\"E0911\", \"map key was not found\", \"the requested key is not present in this map\"); }\n",
        "int64_t __sev_map_size(void *raw) { sev_map *value = raw; if (!value) abort(); return value->size; }\n",
        "int64_t __sev_map_bytes(void *raw) { sev_map *value = raw; if (!value) abort(); return (int64_t)sizeof(*value) + value->capacity * (int64_t)(sizeof(*value->keys) + sizeof(*value->values)); }\n",
        "int64_t __sev_map_capacity(void *raw) { sev_map *value = raw; if (!value) abort(); return value->capacity; }\n",
        "void *__sev_map_key_at(void *raw, int64_t index) { sev_map *value = raw; if (!value) sev_runtime_fail_invariant(\"map storage is null\"); if (index < 0 || index >= value->size) sev_runtime_fail_bounds(\"map\", index, value->size); return value->keys[index]; }\n",
        "void *__sev_map_value_at(void *raw, int64_t index) { sev_map *value = raw; if (!value) sev_runtime_fail_invariant(\"map storage is null\"); if (index < 0 || index >= value->size) sev_runtime_fail_bounds(\"map\", index, value->size); return value->values[index]; }\n",
        "static char *sev_string_range(const char *text, int64_t start, int64_t size) { char *result = sev_allocate((size_t)size + 1); memcpy(result, text + start, (size_t)size); result[size] = '\\0'; return result; }\n",
        "void *__sev_string_characters(void *raw) { const unsigned char *text = raw; int64_t bytes = __sev_strlen(raw); sev_collection *result = __sev_collection_new(0); for (int64_t offset = 0; offset < bytes;) { int64_t width = sev_utf8_width(text[offset]); __sev_collection_push(result, __sev_box_string(sev_string_range((const char *)text, offset, width))); offset += width; } return result; }\n",
        "void *__sev_string_words(void *raw) { const char *text = raw; int64_t size = __sev_strlen(raw); sev_collection *result = __sev_collection_new(0); int64_t index = 0; while (index < size) { while (index < size && isspace((unsigned char)text[index])) index += 1; int64_t start = index; while (index < size && !isspace((unsigned char)text[index])) index += 1; if (start < index) __sev_collection_push(result, __sev_box_string(sev_string_range(text, start, index - start))); } return result; }\n",
        "void *__sev_string_split(void *raw, void *separator_raw) { const char *text = raw; const char *separator = separator_raw; int64_t separator_size = __sev_strlen(separator_raw); if (separator_size == 0) abort(); sev_collection *result = __sev_collection_new(0); const char *start = text; const char *match = NULL; while ((match = strstr(start, separator)) != NULL) { __sev_collection_push(result, __sev_box_string(sev_string_range(start, 0, (int64_t)(match - start)))); start = match + separator_size; } __sev_collection_push(result, __sev_box_string(sev_string_range(start, 0, (int64_t)strlen(start)))); return result; }\n",
        r#"void *__sev_string_split_limit(void *raw, void *separator_raw, int64_t limit, bool reverse) {
  const char *text = raw;
  const char *separator = separator_raw;
  int64_t separator_size = __sev_strlen(separator_raw);
  if (separator_size == 0) abort();
  if (!reverse) {
    sev_collection *result = __sev_collection_new(0);
    const char *start = text;
    const char *match = NULL;
    int64_t splits = 0;
    while ((limit < 0 || splits < limit) && (match = strstr(start, separator)) != NULL) {
      __sev_collection_push(result, __sev_box_string(sev_string_range(start, 0, (int64_t)(match - start))));
      start = match + separator_size;
      splits += 1;
    }
    __sev_collection_push(result, __sev_box_string(sev_string_range(start, 0, (int64_t)strlen(start))));
    return result;
  }
  sev_collection *backward = __sev_collection_new(0);
  int64_t end = __sev_strlen(raw);
  int64_t splits = 0;
  while (limit < 0 || splits < limit) {
    int64_t found = -1;
    for (int64_t index = 0; index + separator_size <= end; ++index) {
      if (memcmp(text + index, separator, (size_t)separator_size) == 0) found = index;
    }
    if (found < 0) break;
    __sev_collection_push(backward, __sev_box_string(sev_string_range(text, found + separator_size, end - found - separator_size)));
    end = found;
    splits += 1;
  }
  __sev_collection_push(backward, __sev_box_string(sev_string_range(text, 0, end)));
  sev_collection *result = __sev_collection_new(0);
  for (int64_t index = backward->size; index > 0; --index) __sev_collection_push(result, backward->items[index - 1]);
  return result;
}
"#,
        r#"void *__sev_csv_parse(void *raw) {
  const char *text = raw;
  size_t input_size = strlen(text);
  char *field = sev_allocate(input_size + 1);
  size_t field_size = 0;
  bool quoted = false;
  sev_collection *rows = __sev_collection_new(0);
  sev_collection *row = __sev_collection_new(0);
  for (size_t index = 0; index <= input_size; ++index) {
    bool at_end = index == input_size;
    char character = at_end ? '\0' : text[index];
    if (quoted && !at_end) {
      if (character == '"') {
        if (index + 1 < input_size && text[index + 1] == '"') {
          field[field_size++] = '"';
          index += 1;
        } else {
          quoted = false;
        }
      } else {
        field[field_size++] = character;
      }
    } else if (!at_end && character == '"') {
      quoted = true;
    } else if (!at_end && character == ',') {
      __sev_collection_push(row, __sev_box_string(sev_string_range(field, 0, (int64_t)field_size)));
      field_size = 0;
    } else if (at_end || character == '\n') {
      if (!at_end || field_size > 0 || row->size > 0) {
        __sev_collection_push(row, __sev_box_string(sev_string_range(field, 0, (int64_t)field_size)));
        __sev_collection_push(rows, __sev_box_collection(row));
        row = __sev_collection_new(0);
        field_size = 0;
      }
    } else if (character != '\r') {
      field[field_size++] = character;
    }
  }
  free(field);
  return rows;
}
"#,
        r#"void *__sev_csv_encode(void *raw) {
  sev_collection *rows = raw;
  if (!rows) abort();
  size_t output_size = 0;
  for (int64_t row_index = 0; row_index < rows->size; ++row_index) {
    sev_value *boxed_row = rows->items[row_index];
    if (!boxed_row || boxed_row->kind != SEV_COLLECTION) abort();
    sev_collection *row = boxed_row->as.pointer;
    for (int64_t column = 0; column < row->size; ++column) {
      sev_value *field = row->items[column];
      if (!field || field->kind != SEV_STRING) abort();
      const char *text = field->as.string;
      bool quoted = strpbrk(text, ",\"\n\r") != NULL;
      if (column) output_size += 1;
      if (quoted) output_size += 2;
      for (const char *cursor = text; *cursor; ++cursor) output_size += *cursor == '"' ? 2 : 1;
    }
    output_size += 1;
  }
  char *output = sev_allocate(output_size + 1);
  char *cursor = output;
  for (int64_t row_index = 0; row_index < rows->size; ++row_index) {
    sev_collection *row = rows->items[row_index]->as.pointer;
    for (int64_t column = 0; column < row->size; ++column) {
      if (column) *cursor++ = ',';
      const char *text = row->items[column]->as.string;
      bool quoted = strpbrk(text, ",\"\n\r") != NULL;
      if (quoted) *cursor++ = '"';
      for (const char *source = text; *source; ++source) {
        if (*source == '"') *cursor++ = '"';
        *cursor++ = *source;
      }
      if (quoted) *cursor++ = '"';
    }
    *cursor++ = '\n';
  }
  *cursor = '\0';
  return output;
}
"#,
        "void *__sev_collection_pop(void *raw) { sev_collection *value = raw; if (!value || value->size == 0) abort(); return value->items[--value->size]; }\n",
        "void *__sev_collection_last(void *raw) { sev_collection *value = raw; if (!value || value->size == 0) abort(); return value->items[value->size - 1]; }\n",
        "void *__sev_collection_sorted(void *raw) { sev_collection *value = raw; if (!value) abort(); sev_collection *result = __sev_collection_new(value->kind); for (int64_t index = 0; index < value->size; ++index) __sev_collection_push(result, value->items[index]); for (int64_t index = 1; index < result->size; ++index) { sev_value *item = result->items[index]; int64_t cursor = index; while (cursor > 0 && __sev_value_less(item, result->items[cursor - 1])) { result->items[cursor] = result->items[cursor - 1]; cursor -= 1; } result->items[cursor] = item; } return result; }\n",
        "void *__sev_collection_sorted_reverse(void *raw, bool reverse) { sev_collection *result = __sev_collection_sorted(raw); if (reverse) for (int64_t left = 0, right = result->size - 1; left < right; ++left, --right) { sev_value *temporary = result->items[left]; result->items[left] = result->items[right]; result->items[right] = temporary; } return result; }\n",
        "void *__sev_collection_sorted_keys(void *raw, void *keys_raw, bool reverse) { sev_collection *value = raw; sev_collection *keys = keys_raw; if (!value || !keys || value->size != keys->size) abort(); sev_collection *result = __sev_collection_new(value->kind); sev_collection *sorted_keys = __sev_collection_new(0); for (int64_t index = 0; index < value->size; ++index) { __sev_collection_push(result, value->items[index]); __sev_collection_push(sorted_keys, keys->items[index]); } for (int64_t index = 1; index < result->size; ++index) { sev_value *item = result->items[index]; sev_value *key = sorted_keys->items[index]; int64_t cursor = index; while (cursor > 0 && __sev_value_less(key, sorted_keys->items[cursor - 1])) { result->items[cursor] = result->items[cursor - 1]; sorted_keys->items[cursor] = sorted_keys->items[cursor - 1]; cursor -= 1; } result->items[cursor] = item; sorted_keys->items[cursor] = key; } if (reverse) for (int64_t left = 0, right = result->size - 1; left < right; ++left, --right) { sev_value *temporary = result->items[left]; result->items[left] = result->items[right]; result->items[right] = temporary; } return result; }\n",
        "void *__sev_collection_join(void *raw, void *separator_raw) { sev_collection *value = raw; const char *separator = separator_raw; if (!value || !separator) abort(); size_t separator_size = strlen(separator); size_t total = 1; for (int64_t index = 0; index < value->size; ++index) { if (!value->items[index] || value->items[index]->kind != SEV_STRING) abort(); total += strlen(value->items[index]->as.string); if (index > 0) total += separator_size; } char *result = sev_allocate(total); for (int64_t index = 0; index < value->size; ++index) { if (index > 0) strcat(result, separator); strcat(result, value->items[index]->as.string); } return result; }\n",
        "void *__sev_collection_sum(void *raw) { sev_collection *value = raw; if (!value) abort(); bool floating = false; double float_total = 0.0; int64_t int_total = 0; for (int64_t index = 0; index < value->size; ++index) { sev_value *item = value->items[index]; if (!item) abort(); if (item->kind == SEV_INT) { int_total += item->as.i64; float_total += (double)item->as.i64; } else if (item->kind == SEV_FLOAT) { floating = true; float_total += item->as.f64; } else abort(); } return floating ? __sev_box_f64(float_total) : __sev_box_i64(int_total); }\n",
        "void *__sev_collection_minimum(void *raw) { sev_collection *value = raw; if (!value || value->size == 0) abort(); sev_value *best = value->items[0]; for (int64_t index = 1; index < value->size; ++index) if (__sev_value_less(value->items[index], best)) best = value->items[index]; return best; }\n",
        "void *__sev_collection_maximum(void *raw) { sev_collection *value = raw; if (!value || value->size == 0) abort(); sev_value *best = value->items[0]; for (int64_t index = 1; index < value->size; ++index) if (__sev_value_less(best, value->items[index])) best = value->items[index]; return best; }\n",
        "void *__sev_collection_to_set(void *raw) { sev_collection *value = raw; if (!value) abort(); sev_collection *result = __sev_collection_new(2); for (int64_t index = 0; index < value->size; ++index) { bool found = false; for (int64_t existing = 0; existing < result->size; ++existing) if (sev_value_equal(result->items[existing], value->items[index])) { found = true; break; } if (!found) __sev_collection_push(result, value->items[index]); } return result; }\n",
        "void *__sev_collection_enumerate(void *raw) { sev_collection *value = raw; if (!value) abort(); sev_collection *result = __sev_collection_new(0); for (int64_t index = 0; index < value->size; ++index) { sev_collection *pair = __sev_collection_new(1); __sev_collection_push(pair, __sev_box_i64(index)); __sev_collection_push(pair, value->items[index]); __sev_collection_push(result, __sev_box_collection(pair)); } return result; }\n",
        "void *__sev_collection_zip(void *left_raw, void *right_raw) { sev_collection *left = left_raw; sev_collection *right = right_raw; if (!left || !right) abort(); sev_collection *result = __sev_collection_new(0); int64_t size = left->size < right->size ? left->size : right->size; for (int64_t index = 0; index < size; ++index) { sev_collection *pair = __sev_collection_new(1); __sev_collection_push(pair, left->items[index]); __sev_collection_push(pair, right->items[index]); __sev_collection_push(result, __sev_box_collection(pair)); } return result; }\n",
        "bool __sev_collection_any(void *raw) { sev_collection *value = raw; if (!value) abort(); for (int64_t index = 0; index < value->size; ++index) { sev_value *item = value->items[index]; if (!item || item->kind != SEV_BOOL) abort(); if (item->as.boolean) return true; } return false; }\n",
        "bool __sev_collection_all(void *raw) { sev_collection *value = raw; if (!value) abort(); for (int64_t index = 0; index < value->size; ++index) { sev_value *item = value->items[index]; if (!item || item->kind != SEV_BOOL) abort(); if (!item->as.boolean) return false; } return true; }\n",
        "void *__sev_range(int64_t start, int64_t end, int64_t step) { if (step == 0) abort(); sev_collection *result = __sev_collection_new(0); for (int64_t value = start; step > 0 ? value < end : value > end; value += step) __sev_collection_push(result, __sev_box_i64(value)); return result; }\n",
        "void *__sev_abs(void *raw) { sev_value *value = raw; if (!value) abort(); if (value->kind == SEV_INT) { if (value->as.i64 == INT64_MIN) abort(); return __sev_box_i64(value->as.i64 < 0 ? -value->as.i64 : value->as.i64); } if (value->kind == SEV_FLOAT) return __sev_box_f64(value->as.f64 < 0.0 ? -value->as.f64 : value->as.f64); abort(); }\n",
        "void *__sev_min(void *left, void *right) { return __sev_value_less(right, left) ? right : left; }\n",
        "void *__sev_max(void *left, void *right) { return __sev_value_less(left, right) ? right : left; }\n",
        "void *__sev_divmod(void *left_raw, void *right_raw) { sev_value *left = left_raw; sev_value *right = right_raw; if (!left || !right || left->kind != SEV_INT || right->kind != SEV_INT || right->as.i64 == 0) abort(); int64_t quotient = left->as.i64 / right->as.i64; int64_t remainder = left->as.i64 % right->as.i64; if (remainder != 0 && ((remainder < 0) != (right->as.i64 < 0))) { quotient -= 1; remainder += right->as.i64; } sev_collection *result = __sev_collection_new(1); __sev_collection_push(result, __sev_box_i64(quotient)); __sev_collection_push(result, __sev_box_i64(remainder)); return result; }\n",
        "void *__sev_set_difference(void *raw, void *excluded_raw) { sev_collection *value = raw; sev_collection *excluded = excluded_raw; if (!value || !excluded) abort(); sev_collection *result = __sev_collection_new(2); for (int64_t index = 0; index < value->size; ++index) { bool found = false; for (int64_t other = 0; other < excluded->size; ++other) if (sev_value_equal(value->items[index], excluded->items[other])) { found = true; break; } if (!found) __sev_collection_push(result, value->items[index]); } return result; }\n",
        "void *__sev_set_combine(void *left_raw, void *right_raw, int64_t operation) { sev_collection *left = left_raw; sev_collection *right = right_raw; if (!left || !right) abort(); sev_collection *result = __sev_collection_new(2); if (operation == 0) { for (int64_t index = 0; index < left->size; ++index) __sev_collection_push(result, left->items[index]); for (int64_t index = 0; index < right->size; ++index) if (!__sev_set_contains(result, right->items[index])) __sev_collection_push(result, right->items[index]); } else if (operation == 1) { for (int64_t index = 0; index < left->size; ++index) if (__sev_set_contains(right, left->items[index])) __sev_collection_push(result, left->items[index]); } else { for (int64_t index = 0; index < left->size; ++index) if (!__sev_set_contains(right, left->items[index])) __sev_collection_push(result, left->items[index]); for (int64_t index = 0; index < right->size; ++index) if (!__sev_set_contains(left, right->items[index])) __sev_collection_push(result, right->items[index]); } return result; }\n",
        "void *__sev_set_to_list(void *raw) { sev_collection *value = raw; if (!value) abort(); sev_collection *result = __sev_collection_new(0); for (int64_t index = 0; index < value->size; ++index) __sev_collection_push(result, value->items[index]); return result; }\n",
        "void *__sev_string_frequencies(void *raw) { const unsigned char *text = raw; int64_t bytes = __sev_strlen(raw); sev_map *result = __sev_map_new(); for (int64_t offset = 0; offset < bytes;) { int64_t width = sev_utf8_width(text[offset]); sev_value *key = __sev_box_string(sev_string_range((const char *)text, offset, width)); bool found = false; for (int64_t entry = 0; entry < result->size; ++entry) if (sev_value_equal(result->keys[entry], key)) { result->values[entry]->as.i64 += 1; found = true; break; } if (!found) __sev_map_insert(result, key, __sev_box_i64(1)); offset += width; } return result; }\n",
        "void *__sev_collection_frequencies(void *raw) { sev_collection *value = raw; if (!value) abort(); sev_map *result = __sev_map_new(); for (int64_t index = 0; index < value->size; ++index) { bool found = false; for (int64_t entry = 0; entry < result->size; ++entry) if (sev_value_equal(result->keys[entry], value->items[index])) { result->values[entry]->as.i64 += 1; found = true; break; } if (!found) __sev_map_insert(result, value->items[index], __sev_box_i64(1)); } return result; }\n",
        "void *__sev_map_keys(void *raw) { sev_map *value = raw; if (!value) abort(); sev_collection *result = __sev_collection_new(0); for (int64_t index = 0; index < value->size; ++index) __sev_collection_push(result, value->keys[index]); return result; }\n",
        "void *__sev_map_values(void *raw) { sev_map *value = raw; if (!value) abort(); sev_collection *result = __sev_collection_new(0); for (int64_t index = 0; index < value->size; ++index) __sev_collection_push(result, value->values[index]); return result; }\n",
        "void *__sev_map_get_default(void *raw, void *key, void *fallback) { sev_map *value = raw; if (!value) abort(); for (int64_t index = 0; index < value->size; ++index) if (sev_value_equal(value->keys[index], key)) return value->values[index]; return fallback; }\n",
        "void *__sev_map_set_default(void *raw, void *key, void *fallback) { sev_map *value = raw; if (!value) abort(); for (int64_t index = 0; index < value->size; ++index) if (sev_value_equal(value->keys[index], key)) return value->values[index]; __sev_map_insert(value, key, fallback); return fallback; }\n",
        "void *__sev_map_items(void *raw) { sev_map *value = raw; if (!value) abort(); sev_collection *result = __sev_collection_new(0); for (int64_t index = 0; index < value->size; ++index) { sev_collection *pair = __sev_collection_new(1); __sev_collection_push(pair, value->keys[index]); __sev_collection_push(pair, value->values[index]); __sev_collection_push(result, __sev_box_collection(pair)); } return result; }\n",
        "void __sev_map_update(void *raw, void *additions_raw) { sev_map *value = raw; sev_map *additions = additions_raw; if (!value || !additions) abort(); for (int64_t index = 0; index < additions->size; ++index) __sev_map_insert(value, additions->keys[index], additions->values[index]); }\n",
        "void *__sev_map_pop(void *raw, void *key, void *fallback) { sev_map *value = raw; if (!value) abort(); for (int64_t index = 0; index < value->size; ++index) if (sev_value_equal(value->keys[index], key)) { sev_value *result = value->values[index]; memmove(value->keys + index, value->keys + index + 1, (size_t)(value->size - index - 1) * sizeof(*value->keys)); memmove(value->values + index, value->values + index + 1, (size_t)(value->size - index - 1) * sizeof(*value->values)); value->size -= 1; return result; } return fallback; }\n",
        "void __sev_map_clear(void *raw) { sev_map *value = raw; if (!value) abort(); value->size = 0; }\n",
        "void __sev_platform_set_add(void *raw, void *item) { sev_collection *value = raw; if (!value) abort(); if (!__sev_set_contains(value, item)) __sev_collection_push(value, item); }\n",
        "bool __sev_platform_set_remove(void *raw, void *item) { sev_collection *value = raw; if (!value) abort(); for (int64_t index = 0; index < value->size; ++index) if (sev_value_equal(value->items[index], item)) { (void)__sev_collection_pop_at(value, index); return true; } return false; }\n",
        "bool __sev_set_remove(void *raw, void *item) { sev_collection *value = raw; if (!value) abort(); for (int64_t index = 0; index < value->size; ++index) if (sev_value_equal(value->items[index], item)) { (void)__sev_collection_pop_at(value, index); return true; } return false; }\n",
        "void *__sev_string_strip(void *raw) { const char *text = raw; int64_t start = 0; int64_t end = __sev_strlen(raw); while (start < end && isspace((unsigned char)text[start])) start += 1; while (end > start && isspace((unsigned char)text[end - 1])) end -= 1; return sev_string_range(text, start, end - start); }\n",
        "void *__sev_string_lstrip(void *raw) { const char *text = raw; int64_t start = 0; int64_t end = __sev_strlen(raw); while (start < end && isspace((unsigned char)text[start])) start += 1; return sev_string_range(text, start, end - start); }\n",
        "void *__sev_string_rstrip(void *raw) { const char *text = raw; int64_t end = __sev_strlen(raw); while (end > 0 && isspace((unsigned char)text[end - 1])) end -= 1; return sev_string_range(text, 0, end); }\n",
        "void *__sev_string_encode(void *raw) { const unsigned char *text = raw; int64_t size = __sev_strlen(raw); sev_collection *result = __sev_collection_new(0); for (int64_t index = 0; index < size; ++index) __sev_collection_push(result, __sev_box_i64(text[index])); return result; }\n",
        "void *__sev_string_casefold(void *raw) { const char *text = raw; int64_t size = __sev_strlen(raw); char *result = sev_allocate((size_t)size + 1); for (int64_t index = 0; index < size; ++index) result[index] = (char)tolower((unsigned char)text[index]); return result; }\n",
        "double __sev_math_sqrt(double value) { return sqrt(value); }\n",
        "double __sev_math_pow(double value, double exponent) { return pow(value, exponent); }\n",
        "double __sev_math_exp(double value) { return exp(value); }\n",
        "double __sev_math_log(double value) { return log(value); }\n",
        "double __sev_math_log2(double value) { return log2(value); }\n",
        "double __sev_math_log10(double value) { return log10(value); }\n",
        "double __sev_math_sin(double value) { return sin(value); }\n",
        "double __sev_math_cos(double value) { return cos(value); }\n",
        "double __sev_math_tan(double value) { return tan(value); }\n",
        "int64_t __sev_math_floor(double value) { return (int64_t)floor(value); }\n",
        "int64_t __sev_math_ceil(double value) { return (int64_t)ceil(value); }\n",
        "bool __sev_math_isfinite(double value) { return isfinite(value); }\n",
        "bool __sev_math_isnan(double value) { return isnan(value); }\n",
        "double __sev_math_round(double value, int64_t digits) { double factor = pow(10.0, (double)digits); return round(value * factor) / factor; }\n",
        "static uint64_t sev_random_state = UINT64_C(0x9e3779b97f4a7c15);\n",
        "static uint64_t sev_random_next(void) { uint64_t value = sev_random_state; value ^= value >> 12; value ^= value << 25; value ^= value >> 27; sev_random_state = value; return value * UINT64_C(2685821657736338717); }\n",
        "void __sev_random_seed(int64_t value) { sev_random_state = value ? (uint64_t)value : UINT64_C(0x9e3779b97f4a7c15); }\n",
        "double __sev_random_float(void) { return (double)(sev_random_next() >> 11) * (1.0 / 9007199254740992.0); }\n",
        "int64_t __sev_random_int(int64_t start, int64_t stop) { if (stop < start) abort(); uint64_t width = (uint64_t)(stop - start) + 1; return start + (int64_t)(sev_random_next() % width); }\n",
        "void *__sev_random_choice(void *raw) { sev_collection *values = raw; if (!values || values->size == 0) abort(); return values->items[sev_random_next() % (uint64_t)values->size]; }\n",
        "void __sev_random_shuffle(void *raw) { sev_collection *values = raw; if (!values) abort(); for (int64_t index = values->size - 1; index > 0; --index) { int64_t other = (int64_t)(sev_random_next() % (uint64_t)(index + 1)); sev_value *temporary = values->items[index]; values->items[index] = values->items[other]; values->items[other] = temporary; } }\n",
        "void *__sev_random_sample(void *raw, int64_t count) { sev_collection *copy = __sev_collection_clone(raw); if (count < 0 || count > copy->size) abort(); __sev_random_shuffle(copy); copy->size = count; return copy; }\n",
        "bool __sev_file_exists(void *path) { return access(path, F_OK) == 0; }\n",
        "bool __sev_path_is_file(void *path) { struct stat status; return stat(path, &status) == 0 && S_ISREG(status.st_mode); }\n",
        "bool __sev_path_is_dir(void *path) { struct stat status; return stat(path, &status) == 0 && S_ISDIR(status.st_mode); }\n",
        "void *__sev_path_join(void *left_raw, void *right_raw) { const char *left = left_raw; const char *right = right_raw; if (!*left) return strdup(right); if (!*right) return strdup(left); size_t left_size = strlen(left); size_t right_size = strlen(right); bool slash = left[left_size - 1] != '/'; char *result = sev_allocate(left_size + right_size + (slash ? 2 : 1)); memcpy(result, left, left_size); if (slash) result[left_size++] = '/'; memcpy(result + left_size, right, right_size + 1); return result; }\n",
        "void *__sev_path_basename(void *raw) { const char *path = raw; const char *slash = strrchr(path, '/'); return strdup(slash ? slash + 1 : path); }\n",
        "void *__sev_path_dirname(void *raw) { const char *path = raw; const char *slash = strrchr(path, '/'); if (!slash) return strdup(\".\"); if (slash == path) return strdup(\"/\"); return sev_string_range(path, 0, (int64_t)(slash - path)); }\n",
        "void *__sev_path_extension(void *raw) { const char *path = raw; const char *slash = strrchr(path, '/'); const char *dot = strrchr(path, '.'); if (!dot || (slash && dot < slash) || dot == (slash ? slash + 1 : path)) return strdup(\"\"); return strdup(dot); }\n",
        "typedef struct { char *extension; void *reader; } sev_file_format_reader_entry;\n",
        "static sev_file_format_reader_entry *sev_file_format_readers = NULL;\n",
        "static int64_t sev_file_format_reader_count = 0;\n",
        "static int64_t sev_file_format_reader_capacity = 0;\n",
        "void __sev_file_format_register(void *extension_raw, void *reader) { const char *extension = extension_raw; for (int64_t index = 0; index < sev_file_format_reader_count; ++index) if (strcmp(sev_file_format_readers[index].extension, extension) == 0) { sev_file_format_readers[index].reader = reader; return; } if (sev_file_format_reader_count == sev_file_format_reader_capacity) { sev_file_format_reader_capacity = sev_file_format_reader_capacity ? sev_file_format_reader_capacity * 2 : 8; sev_file_format_readers = realloc(sev_file_format_readers, (size_t)sev_file_format_reader_capacity * sizeof(*sev_file_format_readers)); if (!sev_file_format_readers) abort(); } sev_file_format_readers[sev_file_format_reader_count].extension = strdup(extension); sev_file_format_readers[sev_file_format_reader_count].reader = reader; sev_file_format_reader_count += 1; }\n",
        "void *__sev_file_format_reader(void *extension_raw, void *fallback) { const char *extension = extension_raw; for (int64_t index = sev_file_format_reader_count - 1; index >= 0; --index) if (strcmp(sev_file_format_readers[index].extension, extension) == 0) return sev_file_format_readers[index].reader; return fallback; }\n",
        "void *__sev_path_absolute(void *raw) { const char *path = raw; if (path[0] == '/') return strdup(path); char current[PATH_MAX]; if (!getcwd(current, sizeof(current))) abort(); return __sev_path_join(current, raw); }\n",
        "double __sev_time_seconds(void) { struct timespec value; if (clock_gettime(CLOCK_REALTIME, &value) != 0) abort(); return (double)value.tv_sec + (double)value.tv_nsec / 1000000000.0; }\n",
        "double __sev_time_monotonic(void) { struct timespec value; if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) abort(); return (double)value.tv_sec + (double)value.tv_nsec / 1000000000.0; }\n",
        "void __sev_time_sleep(double seconds) { if (seconds < 0.0) abort(); struct timespec delay = {(time_t)seconds, (long)((seconds - floor(seconds)) * 1000000000.0)}; while (nanosleep(&delay, &delay) != 0 && errno == EINTR) {} }\n",
        "void *__sev_environment_get(void *name, void *fallback) { const char *value = getenv(name); return strdup(value ? value : fallback); }\n",
        "bool __sev_environment_set(void *name, void *value) { return setenv(name, value, 1) == 0; }\n",
        "bool __sev_environment_remove(void *name) { return unsetenv(name) == 0; }\n",
        "void *__sev_platform_range(int64_t start, int64_t stop, int64_t step) { return __sev_range(start, stop, step); }\n",
        "void *__sev_platform_enumerate(void *values) { return __sev_collection_enumerate(values); }\n",
        "void *__sev_platform_zip(void *left, void *right) { return __sev_collection_zip(left, right); }\n",
        "bool __sev_platform_all(void *values) { return __sev_collection_all(values); }\n",
        "bool __sev_platform_any(void *values) { return __sev_collection_any(values); }\n",
        "void *__sev_platform_abs(void *value) { return __sev_abs(value); }\n",
        "void *__sev_string_lower(void *raw) { const char *text = raw; int64_t size = __sev_strlen(raw); char *result = sev_allocate((size_t)size + 1); for (int64_t index = 0; index < size; ++index) result[index] = (char)tolower((unsigned char)text[index]); return result; }\n",
        "void *__sev_string_upper(void *raw) { const char *text = raw; int64_t size = __sev_strlen(raw); char *result = sev_allocate((size_t)size + 1); for (int64_t index = 0; index < size; ++index) result[index] = (char)toupper((unsigned char)text[index]); return result; }\n",
        "void *__sev_string_capitalize(void *raw) { const char *text = raw; int64_t size = __sev_strlen(raw); char *result = sev_allocate((size_t)size + 1); for (int64_t index = 0; index < size; ++index) result[index] = (char)(index == 0 ? toupper((unsigned char)text[index]) : tolower((unsigned char)text[index])); return result; }\n",
        "void *__sev_string_title(void *raw) { const char *text = raw; int64_t size = __sev_strlen(raw); char *result = sev_allocate((size_t)size + 1); bool boundary = true; for (int64_t index = 0; index < size; ++index) { unsigned char value = (unsigned char)text[index]; result[index] = (char)(boundary ? toupper(value) : tolower(value)); boundary = !isalnum(value); } return result; }\n",
        "void *__sev_string_swapcase(void *raw) { const char *text = raw; int64_t size = __sev_strlen(raw); char *result = sev_allocate((size_t)size + 1); for (int64_t index = 0; index < size; ++index) { unsigned char value = (unsigned char)text[index]; result[index] = (char)(islower(value) ? toupper(value) : (isupper(value) ? tolower(value) : value)); } return result; }\n",
        "void *__sev_string_collapse_space(void *raw, bool horizontal) { const char *text = raw; int64_t size = __sev_strlen(raw); char *result = sev_allocate((size_t)size + 1); int64_t write = 0; bool pending = false; for (int64_t index = 0; index < size; ++index) { unsigned char value = (unsigned char)text[index]; bool space = horizontal ? (value == ' ' || value == '\\t' || value == '\\r') : isspace(value); if (space) { if (write > 0) pending = true; } else { if (pending) result[write++] = ' '; result[write++] = (char)value; pending = false; } } result[write] = '\\0'; return result; }\n",
        "void *__sev_string_split_lines(void *raw) { const char *text = raw; int64_t size = __sev_strlen(raw); sev_collection *result = __sev_collection_new(0); int64_t start = 0; for (int64_t index = 0; index < size; ++index) { if (text[index] != '\\n' && text[index] != '\\r') continue; __sev_collection_push(result, __sev_box_string(sev_string_range(text, start, index - start))); if (text[index] == '\\r' && index + 1 < size && text[index + 1] == '\\n') index += 1; start = index + 1; } if (start < size) __sev_collection_push(result, __sev_box_string(sev_string_range(text, start, size - start))); return result; }\n",
        "bool __sev_string_starts_with(void *raw, void *needle_raw) { const char *text = raw; const char *needle = needle_raw; size_t size = strlen(needle); return strncmp(text, needle, size) == 0; }\n",
        "bool __sev_string_ends_with(void *raw, void *needle_raw) { const char *text = raw; const char *needle = needle_raw; size_t text_size = strlen(text); size_t size = strlen(needle); return size <= text_size && strcmp(text + text_size - size, needle) == 0; }\n",
        "bool __sev_string_contains(void *raw, void *needle_raw) { return strstr(raw, needle_raw) != NULL; }\n",
        "int64_t __sev_string_find(void *raw, void *needle_raw) { const char *text = raw; const char *found = strstr(text, needle_raw); return found ? (int64_t)(found - text) : -1; }\n",
        "int64_t __sev_string_rfind(void *raw, void *needle_raw) { const char *text = raw; const char *needle = needle_raw; size_t needle_size = strlen(needle); if (needle_size == 0) return (int64_t)strlen(text); const char *found = NULL; const char *cursor = text; while ((cursor = strstr(cursor, needle)) != NULL) { found = cursor; cursor += 1; } return found ? (int64_t)(found - text) : -1; }\n",
        "int64_t __sev_string_count(void *raw, void *needle_raw) { const char *text = raw; const char *needle = needle_raw; size_t size = strlen(needle); if (size == 0) return (int64_t)strlen(text) + 1; int64_t count = 0; const char *cursor = text; while ((cursor = strstr(cursor, needle)) != NULL) { count += 1; cursor += size; } return count; }\n",
        "bool __sev_string_predicate(void *raw, int64_t operation) { const unsigned char *text = raw; int64_t size = __sev_strlen(raw); if (operation == 0) return size == 0; if (operation == 5) { for (int64_t i = 0; i < size; ++i) if (text[i] >= 128) return false; return true; } if (size == 0) return false; bool has_lower = false; bool has_upper = false; for (int64_t i = 0; i < size; ++i) { unsigned char value = text[i]; if (operation == 1 && !isspace(value)) return false; if (operation == 2 && (value >= 128 || !isalpha(value))) return false; if (operation == 3 && (value >= 128 || !isdigit(value))) return false; if ((operation == 4 || operation == 8) && (value >= 128 || !isalnum(value))) return false; if (operation == 9 && (size != 1 || value >= 128 || (!isalnum(value) && value != '_'))) return false; if (operation == 10 && (size != 1 || value >= 128 || !ispunct(value))) return false; if (islower(value)) has_lower = true; if (isupper(value)) has_upper = true; } if (operation == 6) return has_lower && !has_upper; if (operation == 7) return has_upper && !has_lower; return true; }\n",
        "void *__sev_string_remove_affix(void *raw, void *affix_raw, bool suffix, bool repeated) { const char *text = raw; const char *affix = affix_raw; int64_t start = 0; int64_t end = __sev_strlen(raw); int64_t size = __sev_strlen(affix_raw); if (size == 0) return sev_string_range(text, 0, end); if (suffix) { while (end - start >= size && memcmp(text + end - size, affix, (size_t)size) == 0) { end -= size; if (!repeated) break; } } else { while (end - start >= size && memcmp(text + start, affix, (size_t)size) == 0) { start += size; if (!repeated) break; } } return sev_string_range(text, start, end - start); }\n",
        r#"void *__sev_string_translate(void *raw, void *mapping_raw) {
  const unsigned char *text = raw;
  sev_map *mapping = mapping_raw;
  int64_t bytes = __sev_strlen(raw);
  size_t output_size = 0;
  for (int64_t offset = 0; offset < bytes;) {
    int64_t width = sev_utf8_width(text[offset]);
    sev_value *replacement = NULL;
    for (int64_t entry = 0; entry < mapping->size; ++entry) {
      sev_value *key = mapping->keys[entry];
      if (key && key->kind == SEV_STRING && (int64_t)strlen(key->as.string) == width && memcmp(text + offset, key->as.string, (size_t)width) == 0) {
        replacement = mapping->values[entry];
        break;
      }
    }
    output_size += replacement && replacement->kind == SEV_STRING ? strlen(replacement->as.string) : (size_t)width;
    offset += width;
  }
  char *result = sev_allocate(output_size + 1);
  size_t write = 0;
  for (int64_t offset = 0; offset < bytes;) {
    int64_t width = sev_utf8_width(text[offset]);
    sev_value *replacement = NULL;
    for (int64_t entry = 0; entry < mapping->size; ++entry) {
      sev_value *key = mapping->keys[entry];
      if (key && key->kind == SEV_STRING && (int64_t)strlen(key->as.string) == width && memcmp(text + offset, key->as.string, (size_t)width) == 0) {
        replacement = mapping->values[entry];
        break;
      }
    }
    const char *source = replacement && replacement->kind == SEV_STRING ? replacement->as.string : (const char *)text + offset;
    size_t source_size = replacement && replacement->kind == SEV_STRING ? strlen(source) : (size_t)width;
    memcpy(result + write, source, source_size);
    write += source_size;
    offset += width;
  }
  result[write] = '\0';
  return result;
}
"#,
        "void *__sev_string_replace(void *raw, void *old_raw, void *new_raw) { const char *text = raw; const char *old = old_raw; const char *replacement = new_raw; size_t old_size = strlen(old); if (old_size == 0) abort(); size_t new_size = strlen(replacement); int64_t count = __sev_string_count(raw, old_raw); size_t result_size = strlen(text) + 1; if (new_size >= old_size) result_size += (size_t)count * (new_size - old_size); else result_size -= (size_t)count * (old_size - new_size); char *result = sev_allocate(result_size); const char *cursor = text; const char *match; while ((match = strstr(cursor, old)) != NULL) { strncat(result, cursor, (size_t)(match - cursor)); strcat(result, replacement); cursor = match + old_size; } strcat(result, cursor); return result; }\n",
        "void *__sev_string_replace_limit(void *raw, void *old_raw, void *new_raw, int64_t limit) { if (limit < 0) return __sev_string_replace(raw, old_raw, new_raw); const char *text = raw; const char *old = old_raw; const char *replacement = new_raw; size_t old_size = strlen(old); if (old_size == 0) abort(); size_t new_size = strlen(replacement); int64_t matches = 0; const char *scan = text; while (matches < limit && (scan = strstr(scan, old)) != NULL) { matches += 1; scan += old_size; } size_t result_size = strlen(text) + 1; if (new_size >= old_size) result_size += (size_t)matches * (new_size - old_size); else result_size -= (size_t)matches * (old_size - new_size); char *result = sev_allocate(result_size); const char *cursor = text; const char *match = NULL; int64_t replaced = 0; while (replaced < matches && (match = strstr(cursor, old)) != NULL) { strncat(result, cursor, (size_t)(match - cursor)); strcat(result, replacement); cursor = match + old_size; replaced += 1; } strcat(result, cursor); return result; }\n",
        "void *__sev_string_replace_many(void *raw, void *mapping_raw) { sev_map *mapping = mapping_raw; void *result = raw; for (int64_t index = 0; index < mapping->size; ++index) { sev_value *old = mapping->keys[index]; sev_value *replacement = mapping->values[index]; if (!old || !replacement || old->kind != SEV_STRING || replacement->kind != SEV_STRING) abort(); result = __sev_string_replace(result, (void *)old->as.string, (void *)replacement->as.string); } return result; }\n",
        r#"void *__sev_string_remove(void *raw, void *characters_raw) {
  const unsigned char *text = raw;
  const unsigned char *characters = characters_raw;
  int64_t text_bytes = __sev_strlen(raw);
  int64_t character_bytes = __sev_strlen(characters_raw);
  char *result = sev_allocate((size_t)text_bytes + 1);
  int64_t write = 0;
  for (int64_t offset = 0; offset < text_bytes;) {
    int64_t width = sev_utf8_width(text[offset]);
    bool removed = false;
    for (int64_t candidate = 0; candidate < character_bytes;) {
      int64_t candidate_width = sev_utf8_width(characters[candidate]);
      if (candidate_width == width && memcmp(text + offset, characters + candidate, (size_t)width) == 0) {
        removed = true;
        break;
      }
      candidate += candidate_width;
    }
    if (!removed) {
      memcpy(result + write, text + offset, (size_t)width);
      write += width;
    }
    offset += width;
  }
  result[write] = '\0';
  return result;
}
"#,
        r#"void *__sev_string_remove_matches(void *raw, void *matches_raw) {
  const unsigned char *text = raw;
  sev_collection *matches = matches_raw;
  if (!matches) abort();
  int64_t text_bytes = __sev_strlen(raw);
  char *result = sev_allocate((size_t)text_bytes + 1);
  int64_t write = 0;
  for (int64_t offset = 0; offset < text_bytes;) {
    int64_t matched_bytes = 0;
    for (int64_t index = 0; index < matches->size; ++index) {
      sev_value *candidate = matches->items[index];
      if (!candidate || candidate->kind != SEV_STRING) abort();
      int64_t candidate_bytes = __sev_strlen((void *)candidate->as.string);
      if (candidate_bytes > matched_bytes && candidate_bytes <= text_bytes - offset && memcmp(text + offset, candidate->as.string, (size_t)candidate_bytes) == 0) {
        matched_bytes = candidate_bytes;
      }
    }
    if (matched_bytes > 0) {
      offset += matched_bytes;
      continue;
    }
    int64_t width = sev_utf8_width(text[offset]);
    memcpy(result + write, text + offset, (size_t)width);
    write += width;
    offset += width;
  }
  result[write] = '\0';
  return result;
}
"#,
        "void *__sev_string_repeat(void *raw, int64_t count) { if (count < 0) abort(); const char *text = raw; size_t size = strlen(text); char *result = sev_allocate(size * (size_t)count + 1); for (int64_t index = 0; index < count; ++index) memcpy(result + size * (size_t)index, text, size); return result; }\n",
        "void *__sev_string_pad(void *raw, int64_t width, int64_t alignment) { const char *text = raw; int64_t characters = __sev_string_length(raw); if (width <= characters) return sev_string_range(text, 0, __sev_strlen(raw)); int64_t padding = width - characters; int64_t left = alignment == 0 ? padding : (alignment == 2 ? padding / 2 : 0); int64_t right = padding - left; int64_t bytes = __sev_strlen(raw); char *result = sev_allocate((size_t)(bytes + padding + 1)); memset(result, ' ', (size_t)left); memcpy(result + left, text, (size_t)bytes); memset(result + left + bytes, ' ', (size_t)right); return result; }\n",
        "void *__sev_string_take(void *raw, int64_t count, int64_t operation) { int64_t length = __sev_string_length(raw); if (count < 0) count = 0; if (count > length) count = length; if (operation == 1) return __sev_string_slice(raw, length - count, length, 1); if (operation == 2) return __sev_string_slice(raw, count, length, 1); return __sev_string_slice(raw, 0, count, 1); }\n",
        "void *__sev_string_segment(void *raw, void *separator_raw, int64_t operation) { const char *text = raw; const char *separator = separator_raw; int64_t found = operation >= 2 ? __sev_string_rfind(raw, separator_raw) : __sev_string_find(raw, separator_raw); if (found < 0) return operation == 0 || operation == 2 ? sev_string_range(text, 0, __sev_strlen(raw)) : sev_string_range(text, 0, 0); int64_t size = __sev_strlen(separator_raw); if (operation == 0 || operation == 2) return sev_string_range(text, 0, found); return sev_string_range(text, found + size, __sev_strlen(raw) - found - size); }\n",
        "void *__sev_string_between(void *raw, void *opener_raw, void *closer_raw) { const char *text = raw; int64_t start = __sev_string_find(raw, opener_raw); if (start < 0) return sev_string_range(text, 0, 0); start += __sev_strlen(opener_raw); const char *found = strstr(text + start, closer_raw); if (!found) return sev_string_range(text, 0, 0); return sev_string_range(text, start, (int64_t)(found - text) - start); }\n",
        "void *__sev_string_partition(void *raw, void *separator_raw, bool reverse) { const char *text = raw; const char *separator = separator_raw; int64_t size = __sev_strlen(separator_raw); if (size == 0) abort(); int64_t found = reverse ? __sev_string_rfind(raw, separator_raw) : __sev_string_find(raw, separator_raw); sev_collection *result = __sev_collection_new(1); if (found < 0) { if (reverse) { __sev_collection_push(result, __sev_box_string(sev_string_range(text, 0, 0))); __sev_collection_push(result, __sev_box_string(sev_string_range(text, 0, 0))); __sev_collection_push(result, __sev_box_string(sev_string_range(text, 0, __sev_strlen(raw)))); } else { __sev_collection_push(result, __sev_box_string(sev_string_range(text, 0, __sev_strlen(raw)))); __sev_collection_push(result, __sev_box_string(sev_string_range(text, 0, 0))); __sev_collection_push(result, __sev_box_string(sev_string_range(text, 0, 0))); } return result; } __sev_collection_push(result, __sev_box_string(sev_string_range(text, 0, found))); __sev_collection_push(result, __sev_box_string(sev_string_range(separator, 0, size))); __sev_collection_push(result, __sev_box_string(sev_string_range(text, found + size, __sev_strlen(raw) - found - size))); return result; }\n",
        "static void sev_print_collection_inline(void *raw);\n",
        "void __sev_print_value_inline(void *raw) { sev_value *value = raw; if (!value) { fputs(\"invalid\", stdout); return; } switch (value->kind) { case SEV_INT: printf(\"%ld\", value->as.i64); break; case SEV_FLOAT: printf(\"%.17g\", value->as.f64); break; case SEV_BOOL: fputs(value->as.boolean ? \"true\" : \"false\", stdout); break; case SEV_STRING: fputs(value->as.string, stdout); break; case SEV_COLLECTION: sev_print_collection_inline(value->as.pointer); break; } }\n",
        "void __sev_print_value(void *raw) { __sev_print_value_inline(raw); fputc('\\n', stdout); }\n",
        "void __sev_print_space(void) { fputc(' ', stdout); }\n",
        "void __sev_print_newline(void) { fputc('\\n', stdout); }\n",
        "static void sev_print_collection_inline(void *raw) { sev_collection *value = raw; char open = value->kind == 1 ? '(' : value->kind == 2 ? '{' : '['; char close = value->kind == 1 ? ')' : value->kind == 2 ? '}' : ']'; fputc(open, stdout); for (int64_t i = 0; i < value->size; ++i) { if (i) fputs(\", \", stdout); __sev_print_value_inline(value->items[i]); } fputc(close, stdout); }\n",
        "void __sev_print_collection(void *raw) { sev_print_collection_inline(raw); fputc('\\n', stdout); }\n",
        "void *__sev_object_new(void *class_name) { sev_object *value = sev_allocate(sizeof(*value)); value->magic = SEV_OBJECT_MAGIC; value->class_name = class_name; pthread_mutex_init(&value->mutex, NULL); return value; }\n",
        "void __sev_object_declare(void *raw, void *name) { sev_object *value = raw; if (!value || value->magic != SEV_OBJECT_MAGIC || !name) abort(); for (int64_t i = 0; i < value->size; ++i) if (strcmp(value->names[i], name) == 0) return; if (value->size == value->capacity) { value->capacity = value->capacity ? value->capacity * 2 : 4; value->names = realloc(value->names, (size_t)value->capacity * sizeof(*value->names)); value->values = realloc(value->values, (size_t)value->capacity * sizeof(*value->values)); if (!value->names || !value->values) abort(); } value->names[value->size] = name; value->values[value->size++] = NULL; }\n",
        "static int64_t sev_dynamic_kind(void *raw) { if (!raw) abort(); uint64_t magic = *(uint64_t *)raw; if (magic == SEV_OBJECT_MAGIC) return 100; if (magic == SEV_VARIANT_MAGIC) return 101; if (magic == SEV_TENSOR_MAGIC) return 102; sev_value *value = raw; if (value->kind < SEV_INT || value->kind > SEV_COLLECTION) abort(); return value->kind; }\n",
        "void __sev_object_set(void *raw, void *name, void *item) { sev_object *value = raw; if (!value || value->magic != SEV_OBJECT_MAGIC || !name || !item) abort(); for (int64_t i = 0; i < value->size; ++i) if (strcmp(value->names[i], name) == 0) { if (value->values[i]) { int64_t old_kind = sev_dynamic_kind(value->values[i]); int64_t new_kind = sev_dynamic_kind(item); if (old_kind != new_kind) abort(); if (old_kind == 100 && strcmp(((sev_object *)value->values[i])->class_name, ((sev_object *)item)->class_name) != 0) abort(); } value->values[i] = item; return; } abort(); }\n",
        "void *__sev_object_get(void *raw, void *name) { if (!raw || !name) abort(); uint64_t magic = *(uint64_t *)raw; if (magic == SEV_OBJECT_MAGIC) { sev_object *value = raw; for (int64_t i = 0; i < value->size; ++i) if (strcmp(value->names[i], name) == 0) return value->values[i]; abort(); } if (magic == SEV_VARIANT_MAGIC) { sev_variant *value = raw; if (strcmp(name, \"message\") == 0 && value->field) return value->field; } abort(); }\n\n",
        "bool __sev_object_is(void *raw, void *class_name) { sev_object *value = raw; return value && value->magic == SEV_OBJECT_MAGIC && strcmp(value->class_name, class_name) == 0; }\n\n",
        "void *__sev_variant_new(void *tag, void *field) { sev_variant *value = sev_allocate(sizeof(*value)); value->magic = SEV_VARIANT_MAGIC; value->tag = tag; value->field = field; return value; }\n",
        "bool __sev_variant_is(void *raw, void *tag) { sev_variant *value = raw; return value && value->magic == SEV_VARIANT_MAGIC && strcmp(value->tag, tag) == 0; }\n",
        "void *__sev_variant_field(void *raw) { sev_variant *value = raw; if (!value || value->magic != SEV_VARIANT_MAGIC) abort(); return value->field; }\n",
        "void __sev_print_variant(void *raw) { sev_variant *value = raw; if (!value || value->magic != SEV_VARIANT_MAGIC) abort(); fputs(value->tag, stdout); if (value->field) { fputc('(', stdout); __sev_print_value_inline(value->field); fputc(')', stdout); } fputc('\\n', stdout); }\n\n",
        "void *__sev_builtin_read(void *path) { (void)path; return __sev_variant_new(\"ok\", __sev_box_string(\"settings\")); }\n",
        "void *__sev_builtin_http_get(void *url) { (void)url; return __sev_variant_new(\"ok\", __sev_box_string(\"response\")); }\n",
        "void *__sev_builtin_int_parse(void *text) { if (!text) return __sev_variant_new(\"failure\", __sev_box_string(\"invalid integer\")); char *end = NULL; long value = strtol(text, &end, 10); if (end == text || *end != '\\0') return __sev_variant_new(\"failure\", __sev_box_string(\"invalid integer\")); return __sev_variant_new(\"ok\", __sev_box_i64(value)); }\n",
        "void *__sev_builtin_float_parse(void *text) { if (!text) return __sev_variant_new(\"failure\", __sev_box_string(\"invalid float\")); char *end = NULL; double value = strtod(text, &end); if (end == text || *end != '\\0') return __sev_variant_new(\"failure\", __sev_box_string(\"invalid float\")); return __sev_variant_new(\"ok\", __sev_box_f64(value)); }\n",
        "\n",
    ));
    source.push_str(
        r#"
static void *sev_failure(const char *message) {
  return __sev_variant_new("failure", __sev_box_string((void *)message));
}

void *__sev_agent_model_request(void *endpoint, void *model, void *system_prompt, void *prompt, int64_t context_tokens, int64_t output_tokens, double temperature, int64_t timeout_millis) {
  (void)endpoint; (void)model; (void)system_prompt; (void)prompt; (void)context_tokens; (void)output_tokens; (void)temperature; (void)timeout_millis;
  return sev_failure("agent model provider is not configured");
}

void *__sev_agent_execute(void *workspace, void *command, int64_t timeout_millis, int64_t max_output_bytes) {
  (void)workspace; (void)command; (void)timeout_millis; (void)max_output_bytes;
  return sev_failure("agent command provider is not configured");
}

void *__sev_agent_read_file(void *path) {
  (void)path;
  return sev_failure("agent file provider is not configured");
}

void *__sev_agent_write_file(void *path, void *content) {
  (void)path; (void)content;
  return sev_failure("agent file provider is not configured");
}

void *__sev_agent_git_status(void *workspace) {
  (void)workspace;
  return sev_failure("agent git provider is not configured");
}

void *__sev_agent_reset_workspace(void *workspace) {
  (void)workspace;
  return sev_failure("agent git provider is not configured");
}

typedef struct { FILE *stream; bool closed; } sev_binary_file;
typedef struct { int fd; unsigned char *data; size_t size; bool unmapped; } sev_mapped_file;

void *__sev_file_open_binary(void *path_raw) {
  FILE *stream = fopen((const char *)path_raw, "rb");
  if (!stream) return sev_failure("could not open binary file");
  sev_binary_file *file = sev_allocate(sizeof(*file));
  file->stream = stream;
  return __sev_variant_new("ok", file);
}

static void *sev_file_read_bytes(void *handle_raw, int64_t count, bool exact) {
  sev_binary_file *file = handle_raw;
  if (!file || file->closed || !file->stream) return sev_failure("binary file is closed");
  if (count < 0 || (uint64_t)count > SIZE_MAX) return sev_failure("invalid binary read size");
  unsigned char *bytes = sev_allocate((size_t)count);
  size_t received = fread(bytes, 1, (size_t)count, file->stream);
  if (ferror(file->stream) || (exact && received != (size_t)count)) {
    free(bytes);
    return sev_failure(exact ? "unexpected end of binary file" : "binary file read failed");
  }
  sev_collection *result = __sev_collection_new(0);
  for (size_t index = 0; index < received; ++index)
    __sev_collection_push(result, __sev_box_i64(bytes[index]));
  free(bytes);
  return __sev_variant_new("ok", __sev_box_collection(result));
}

void *__sev_file_read_bytes(void *handle_raw, int64_t count) {
  return sev_file_read_bytes(handle_raw, count, false);
}

void *__sev_file_read_exact(void *handle_raw, int64_t count) {
  return sev_file_read_bytes(handle_raw, count, true);
}

void *__sev_bytes_utf8(void *bytes_raw) {
  sev_collection *bytes = bytes_raw;
  if (!bytes || bytes->size < 0 || (uint64_t)bytes->size > SIZE_MAX - 1)
    return sev_failure("invalid byte buffer");
  char *text = sev_allocate((size_t)bytes->size + 1);
  for (int64_t index = 0; index < bytes->size; ++index) {
    int64_t byte = __sev_unbox_i64(bytes->items[index]);
    if (byte < 0 || byte > 255) {
      free(text);
      return sev_failure("byte value is outside 0..255");
    }
    text[index] = (char)byte;
  }
  return __sev_variant_new("ok", __sev_box_string(text));
}

void *__sev_file_seek(void *handle_raw, int64_t offset) {
  sev_binary_file *file = handle_raw;
  if (!file || file->closed || !file->stream) return sev_failure("binary file is closed");
  if (offset < 0 || fseeko(file->stream, (off_t)offset, SEEK_SET) != 0)
    return sev_failure("could not seek binary file");
  return __sev_variant_new("ok", NULL);
}

void *__sev_file_size(void *handle_raw) {
  sev_binary_file *file = handle_raw;
  if (!file || file->closed || !file->stream) return sev_failure("binary file is closed");
  off_t position = ftello(file->stream);
  if (position < 0 || fseeko(file->stream, 0, SEEK_END) != 0)
    return sev_failure("could not seek binary file");
  off_t size = ftello(file->stream);
  if (size < 0 || fseeko(file->stream, position, SEEK_SET) != 0)
    return sev_failure("could not size binary file");
  return __sev_variant_new("ok", __sev_box_i64((int64_t)size));
}

void *__sev_file_close(void *handle_raw) {
  sev_binary_file *file = handle_raw;
  if (!file || file->closed || !file->stream) return sev_failure("binary file is closed");
  int status = fclose(file->stream);
  file->stream = NULL;
  file->closed = true;
  if (status != 0) return sev_failure("could not close binary file");
  return __sev_variant_new("ok", NULL);
}


void *__sev_file_map(void *path_raw) {
  int fd = open((const char *)path_raw, O_RDONLY | O_CLOEXEC);
  if (fd < 0) return sev_failure("could not open mapped file");
  struct stat status;
  if (fstat(fd, &status) != 0 || status.st_size < 0) {
    close(fd);
    return sev_failure("could not stat mapped file");
  }
  size_t size = (size_t)status.st_size;
  void *data = size == 0 ? NULL : mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (size != 0 && data == MAP_FAILED) {
    close(fd);
    return sev_failure("could not map file");
  }
  sev_mapped_file *mapping = sev_allocate(sizeof(*mapping));
  mapping->fd = fd;
  mapping->data = data;
  mapping->size = size;
  return __sev_variant_new("ok", mapping);
}

int64_t __sev_file_mapped_size(void *mapping_raw) {
  sev_mapped_file *mapping = mapping_raw;
  if (!mapping || mapping->unmapped || mapping->size > INT64_MAX) abort();
  return (int64_t)mapping->size;
}

void *__sev_file_unmap(void *mapping_raw) {
  sev_mapped_file *mapping = mapping_raw;
  if (!mapping || mapping->unmapped) return sev_failure("mapped file is already unmapped");
  int map_status = mapping->size == 0 ? 0 : munmap(mapping->data, mapping->size);
  int close_status = close(mapping->fd);
  mapping->unmapped = true;
  mapping->data = NULL;
  mapping->size = 0;
  if (map_status != 0 || close_status != 0) return sev_failure("could not unmap file");
  return __sev_variant_new("ok", NULL);
}

typedef struct { int descriptor; } sev_file_lock;

void *__sev_file_lock(void *path_raw) {
  const char *path = path_raw;
  int descriptor = open(path, O_CREAT | O_RDWR | O_CLOEXEC, 0666);
  if (descriptor < 0) return sev_failure(strerror(errno));
  if (flock(descriptor, LOCK_EX) != 0) {
    const char *message = strerror(errno);
    close(descriptor);
    return sev_failure(message);
  }
  sev_file_lock *handle = sev_allocate(sizeof(*handle));
  handle->descriptor = descriptor;
  return __sev_variant_new("ok", handle);
}

void *__sev_file_unlock(void *handle_raw) {
  sev_file_lock *handle = handle_raw;
  if (!handle) return sev_failure("invalid file lock");
  int failed = flock(handle->descriptor, LOCK_UN);
  if (close(handle->descriptor) != 0) failed = -1;
  free(handle);
  return failed == 0 ? __sev_variant_new("ok", NULL) : sev_failure(strerror(errno));
}

void *__sev_file_write(void *path_raw, void *text_raw) {
  const char *path = path_raw;
  const char *text = text_raw;
  FILE *file = fopen(path, "wb");
  if (!file) return sev_failure("could not open file for writing");
  size_t size = strlen(text);
  bool success = fwrite(text, 1, size, file) == size && fclose(file) == 0;
  if (!success) return sev_failure("could not write file");
  return __sev_variant_new("ok", NULL);
}

void *__sev_builtin_file_write(void *path, void *text) {
  return __sev_file_write(path, text);
}

void *__sev_file_read(void *path_raw) {
  FILE *file = fopen((const char *)path_raw, "rb");
  if (!file) return sev_failure("could not open file for reading");
  if (fseek(file, 0, SEEK_END) != 0) { fclose(file); return sev_failure("could not seek file"); }
  long size = ftell(file);
  if (size < 0 || fseek(file, 0, SEEK_SET) != 0) { fclose(file); return sev_failure("could not size file"); }
  char *contents = sev_allocate((size_t)size + 1);
  bool success = fread(contents, 1, (size_t)size, file) == (size_t)size && fclose(file) == 0;
  if (!success) return sev_failure("could not read file");
  return __sev_variant_new("ok", __sev_box_string(contents));
}

void *__sev_file_remove(void *path_raw) {
  return unlink(path_raw) == 0 ? __sev_variant_new("ok", NULL) : sev_failure("could not remove file");
}

void *__sev_file_rename(void *source_raw, void *destination_raw) {
  return rename(source_raw, destination_raw) == 0 ? __sev_variant_new("ok", NULL) : sev_failure("could not rename file");
}

void *__sev_file_copy(void *source_raw, void *destination_raw) {
  FILE *source = fopen(source_raw, "rb");
  if (!source) return sev_failure("could not open source file");
  FILE *destination = fopen(destination_raw, "wb");
  if (!destination) { fclose(source); return sev_failure("could not open destination file"); }
  char buffer[16384];
  bool success = true;
  size_t count;
  while ((count = fread(buffer, 1, sizeof(buffer), source)) > 0) if (fwrite(buffer, 1, count, destination) != count) { success = false; break; }
  if (ferror(source)) success = false;
  if (fclose(source) != 0 || fclose(destination) != 0) success = false;
  return success ? __sev_variant_new("ok", NULL) : sev_failure("could not copy file");
}

void *__sev_file_append(void *path_raw, void *text_raw) {
  FILE *file = fopen((const char *)path_raw, "ab");
  if (!file) return sev_failure("could not open file for appending");
  const char *text = text_raw;
  size_t size = strlen(text);
  bool success = fwrite(text, 1, size, file) == size && fclose(file) == 0;
  return success ? __sev_variant_new("ok", NULL) : sev_failure("could not append file");
}

void *__sev_file_write_bytes(void *path_raw, void *contents_raw) {
  sev_collection *contents = contents_raw;
  if (!contents) return sev_failure("invalid byte collection");
  FILE *file = fopen((const char *)path_raw, "wb");
  if (!file) return sev_failure("could not open binary file for writing");
  bool success = true;
  for (int64_t index = 0; index < contents->size; ++index) {
    int64_t byte = __sev_unbox_i64(contents->items[index]);
    if (byte < 0 || byte > 255 || fputc((int)byte, file) == EOF) { success = false; break; }
  }
  if (fclose(file) != 0) success = false;
  return success ? __sev_variant_new("ok", NULL) : sev_failure("could not write binary file");
}

void *__sev_file_size_path(void *path_raw) {
  struct stat status;
  if (stat((const char *)path_raw, &status) != 0 || status.st_size < 0)
    return sev_failure("could not stat file size");
  return __sev_variant_new("ok", __sev_box_i64((int64_t)status.st_size));
}

void *__sev_file_modified_seconds(void *path_raw) {
  struct stat status;
  if (stat((const char *)path_raw, &status) != 0)
    return sev_failure("could not stat file modification time");
#if defined(__APPLE__)
  double seconds = (double)status.st_mtimespec.tv_sec + (double)status.st_mtimespec.tv_nsec / 1000000000.0;
#else
  double seconds = (double)status.st_mtim.tv_sec + (double)status.st_mtim.tv_nsec / 1000000000.0;
#endif
  return __sev_variant_new("ok", __sev_box_f64(seconds));
}

void *__sev_directory_list(void *path_raw) {
  DIR *directory = opendir((const char *)path_raw);
  if (!directory) return sev_failure("could not open directory");
  sev_collection *entries = __sev_collection_new(0);
  errno = 0;
  struct dirent *entry;
  while ((entry = readdir(directory)) != NULL) {
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) continue;
    __sev_collection_push(entries, __sev_box_string(strdup(entry->d_name)));
  }
  int read_error = errno;
  int close_error = closedir(directory);
  if (read_error != 0 || close_error != 0) return sev_failure("could not read directory");
  return __sev_variant_new("ok", __sev_box_collection(entries));
}

void *__sev_directory_make_all(void *path_raw) {
  const char *path = path_raw;
  if (!path || !*path) return sev_failure("directory path is empty");
  char *copy = strdup(path);
  if (!copy) abort();
  size_t size = strlen(copy);
  while (size > 1 && copy[size - 1] == '/') copy[--size] = '\0';
  for (char *cursor = copy + (copy[0] == '/'); ; ++cursor) {
    if (*cursor != '/' && *cursor != '\0') continue;
    char saved = *cursor;
    *cursor = '\0';
    if (*copy && mkdir(copy, 0777) != 0 && errno != EEXIST) {
      free(copy);
      return sev_failure("could not create directory");
    }
    *cursor = saved;
    if (saved == '\0') break;
  }
  free(copy);
  return __sev_variant_new("ok", NULL);
}

void *__sev_process_arguments(void) {
  FILE *file = fopen("/proc/self/cmdline", "rb");
  sev_collection *arguments = __sev_collection_new(0);
  if (!file) return arguments;
  char *buffer = NULL;
  size_t capacity = 0;
  ssize_t count;
  while ((count = getdelim(&buffer, &capacity, '\0', file)) >= 0) {
    size_t length = count > 0 && buffer[count - 1] == '\0' ? (size_t)count - 1 : (size_t)count;
    __sev_collection_push(arguments, __sev_box_string(sev_string_range(buffer, 0, (int64_t)length)));
  }
  sev_system_release(buffer);
  fclose(file);
  return arguments;
}

void *__sev_time_parse_date(void *value_raw) {
  const char *value = value_raw;
  int year = 0, month = 0, day = 0;
  char trailing = '\0';
  if (sscanf(value, "%d-%d-%d%c", &year, &month, &day, &trailing) != 3 ||
      year < 1970 || month < 1 || month > 12 || day < 1 || day > 31)
    return sev_failure("date must use YYYY-MM-DD");
  struct tm calendar = {0};
  calendar.tm_year = year - 1900;
  calendar.tm_mon = month - 1;
  calendar.tm_mday = day;
  calendar.tm_isdst = -1;
  time_t timestamp = mktime(&calendar);
  if (timestamp == (time_t)-1 || calendar.tm_year != year - 1900 ||
      calendar.tm_mon != month - 1 || calendar.tm_mday != day)
    return sev_failure("invalid calendar date");
  return __sev_variant_new("ok", __sev_box_f64((double)timestamp));
}

static uint32_t sev_md5_rotate(uint32_t value, uint32_t count) {
  return (value << count) | (value >> (32 - count));
}

void *__sev_hash_md5(void *value_raw) {
  static const uint32_t shifts[64] = {
    7,12,17,22,7,12,17,22,7,12,17,22,7,12,17,22,
    5,9,14,20,5,9,14,20,5,9,14,20,5,9,14,20,
    4,11,16,23,4,11,16,23,4,11,16,23,4,11,16,23,
    6,10,15,21,6,10,15,21,6,10,15,21,6,10,15,21
  };
  static const uint32_t constants[64] = {
    0xd76aa478,0xe8c7b756,0x242070db,0xc1bdceee,0xf57c0faf,0x4787c62a,0xa8304613,0xfd469501,
    0x698098d8,0x8b44f7af,0xffff5bb1,0x895cd7be,0x6b901122,0xfd987193,0xa679438e,0x49b40821,
    0xf61e2562,0xc040b340,0x265e5a51,0xe9b6c7aa,0xd62f105d,0x02441453,0xd8a1e681,0xe7d3fbc8,
    0x21e1cde6,0xc33707d6,0xf4d50d87,0x455a14ed,0xa9e3e905,0xfcefa3f8,0x676f02d9,0x8d2a4c8a,
    0xfffa3942,0x8771f681,0x6d9d6122,0xfde5380c,0xa4beea44,0x4bdecfa9,0xf6bb4b60,0xbebfbc70,
    0x289b7ec6,0xeaa127fa,0xd4ef3085,0x04881d05,0xd9d4d039,0xe6db99e5,0x1fa27cf8,0xc4ac5665,
    0xf4292244,0x432aff97,0xab9423a7,0xfc93a039,0x655b59c3,0x8f0ccc92,0xffeff47d,0x85845dd1,
    0x6fa87e4f,0xfe2ce6e0,0xa3014314,0x4e0811a1,0xf7537e82,0xbd3af235,0x2ad7d2bb,0xeb86d391
  };
  const unsigned char *input = value_raw;
  size_t input_size = strlen((const char *)input);
  uint64_t bit_size = (uint64_t)input_size * 8;
  size_t padded_size = input_size + 1;
  while (padded_size % 64 != 56) padded_size += 1;
  unsigned char *message = calloc(padded_size + 8, 1);
  if (!message) abort();
  memcpy(message, input, input_size);
  message[input_size] = 0x80;
  for (int byte = 0; byte < 8; ++byte) message[padded_size + byte] = (unsigned char)(bit_size >> (8 * byte));
  uint32_t a0 = 0x67452301, b0 = 0xefcdab89, c0 = 0x98badcfe, d0 = 0x10325476;
  for (size_t offset = 0; offset < padded_size + 8; offset += 64) {
    uint32_t words[16];
    for (int word = 0; word < 16; ++word) {
      size_t base = offset + (size_t)word * 4;
      words[word] = (uint32_t)message[base] | ((uint32_t)message[base + 1] << 8) |
        ((uint32_t)message[base + 2] << 16) | ((uint32_t)message[base + 3] << 24);
    }
    uint32_t a = a0, b = b0, c = c0, d = d0;
    for (uint32_t step = 0; step < 64; ++step) {
      uint32_t function, word;
      if (step < 16) { function = (b & c) | ((~b) & d); word = step; }
      else if (step < 32) { function = (d & b) | ((~d) & c); word = (5 * step + 1) % 16; }
      else if (step < 48) { function = b ^ c ^ d; word = (3 * step + 5) % 16; }
      else { function = c ^ (b | (~d)); word = (7 * step) % 16; }
      uint32_t next = d;
      d = c;
      c = b;
      b += sev_md5_rotate(a + function + constants[step] + words[word], shifts[step]);
      a = next;
    }
    a0 += a; b0 += b; c0 += c; d0 += d;
  }
  free(message);
  uint32_t digest[4] = {a0, b0, c0, d0};
  char *output = sev_allocate(33);
  static const char hex[] = "0123456789abcdef";
  for (int index = 0; index < 16; ++index) {
    unsigned char byte = (unsigned char)(digest[index / 4] >> (8 * (index % 4)));
    output[index * 2] = hex[byte >> 4];
    output[index * 2 + 1] = hex[byte & 15];
  }
  return output;
}

static uint32_t sev_sha256_rotate_right(uint32_t value, uint32_t count) {
  return (value >> count) | (value << (32 - count));
}

void *__sev_hash_sha256_bytes(void *value_raw) {
  static const uint32_t constants[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
  };
  sev_collection *input = value_raw;
  if (!input || input->kind == 3 || input->size < 0) abort();
  size_t input_size = (size_t)input->size;
  uint64_t bit_size = (uint64_t)input_size * 8;
  size_t padded_size = input_size + 1;
  while (padded_size % 64 != 56) padded_size += 1;
  unsigned char *message = calloc(padded_size + 8, 1);
  if (!message) abort();
  for (size_t index = 0; index < input_size; ++index) {
    int64_t byte = __sev_unbox_i64(input->items[index]);
    if (byte < 0 || byte > 255) abort();
    message[index] = (unsigned char)byte;
  }
  message[input_size] = 0x80;
  for (int byte = 0; byte < 8; ++byte)
    message[padded_size + byte] = (unsigned char)(bit_size >> (56 - 8 * byte));
  uint32_t state[8] = {
    0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
    0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19
  };
  for (size_t offset = 0; offset < padded_size + 8; offset += 64) {
    uint32_t words[64];
    for (int index = 0; index < 16; ++index) {
      size_t base = offset + (size_t)index * 4;
      words[index] = ((uint32_t)message[base] << 24) | ((uint32_t)message[base + 1] << 16) |
        ((uint32_t)message[base + 2] << 8) | (uint32_t)message[base + 3];
    }
    for (int index = 16; index < 64; ++index) {
      uint32_t s0 = sev_sha256_rotate_right(words[index - 15], 7) ^
        sev_sha256_rotate_right(words[index - 15], 18) ^ (words[index - 15] >> 3);
      uint32_t s1 = sev_sha256_rotate_right(words[index - 2], 17) ^
        sev_sha256_rotate_right(words[index - 2], 19) ^ (words[index - 2] >> 10);
      words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (int index = 0; index < 64; ++index) {
      uint32_t big1 = sev_sha256_rotate_right(e, 6) ^ sev_sha256_rotate_right(e, 11) ^ sev_sha256_rotate_right(e, 25);
      uint32_t choice = (e & f) ^ ((~e) & g);
      uint32_t temporary1 = h + big1 + choice + constants[index] + words[index];
      uint32_t big0 = sev_sha256_rotate_right(a, 2) ^ sev_sha256_rotate_right(a, 13) ^ sev_sha256_rotate_right(a, 22);
      uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      uint32_t temporary2 = big0 + majority;
      h = g; g = f; f = e; e = d + temporary1;
      d = c; c = b; b = a; a = temporary1 + temporary2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
  }
  free(message);
  char *output = sev_allocate(65);
  static const char hex[] = "0123456789abcdef";
  for (int index = 0; index < 32; ++index) {
    unsigned char byte = (unsigned char)(state[index / 4] >> (24 - 8 * (index % 4)));
    output[index * 2] = hex[byte >> 4];
    output[index * 2 + 1] = hex[byte & 15];
  }
  output[64] = '\0';
  return output;
}

typedef struct { char *data; size_t size; size_t capacity; } sev_json_buffer;

static void sev_json_reserve(sev_json_buffer *buffer, size_t extra) {
  size_t required = buffer->size + extra + 1;
  if (required <= buffer->capacity) return;
  size_t capacity = buffer->capacity ? buffer->capacity : 64;
  while (capacity < required) capacity *= 2;
  buffer->data = realloc(buffer->data, capacity);
  if (!buffer->data) abort();
  buffer->capacity = capacity;
}

static void sev_json_character(sev_json_buffer *buffer, char value) {
  sev_json_reserve(buffer, 1);
  buffer->data[buffer->size++] = value;
  buffer->data[buffer->size] = '\0';
}

static void sev_json_text(sev_json_buffer *buffer, const char *value) {
  size_t size = strlen(value);
  sev_json_reserve(buffer, size);
  memcpy(buffer->data + buffer->size, value, size + 1);
  buffer->size += size;
}

static void sev_json_encode_value(sev_json_buffer *buffer, sev_value *value) {
  if (!value) { sev_json_text(buffer, "null"); return; }
  char number[64];
  switch (value->kind) {
    case SEV_INT:
      snprintf(number, sizeof(number), "%ld", value->as.i64);
      sev_json_text(buffer, number);
      return;
    case SEV_FLOAT:
      snprintf(number, sizeof(number), "%.17g", value->as.f64);
      sev_json_text(buffer, number);
      return;
    case SEV_BOOL:
      sev_json_text(buffer, value->as.boolean ? "true" : "false");
      return;
    case SEV_STRING:
      sev_json_character(buffer, '"');
      for (const char *cursor = value->as.string; *cursor; ++cursor) {
        if (*cursor == '"' || *cursor == '\\') sev_json_character(buffer, '\\');
        if (*cursor == '\n') { sev_json_text(buffer, "\\n"); continue; }
        sev_json_character(buffer, *cursor);
      }
      sev_json_character(buffer, '"');
      return;
    case SEV_COLLECTION: {
      sev_collection *collection = value->as.pointer;
      if (collection->kind == 3) {
        sev_map *map = value->as.pointer;
        sev_json_character(buffer, '{');
        for (int64_t index = 0; index < map->size; ++index) {
          if (index) sev_json_character(buffer, ',');
          sev_json_encode_value(buffer, map->keys[index]);
          sev_json_character(buffer, ':');
          sev_json_encode_value(buffer, map->values[index]);
        }
        sev_json_character(buffer, '}');
        return;
      }
      sev_json_character(buffer, '[');
      for (int64_t index = 0; index < collection->size; ++index) {
        if (index) sev_json_character(buffer, ',');
        sev_json_encode_value(buffer, collection->items[index]);
      }
      sev_json_character(buffer, ']');
      return;
    }
    case SEV_NULL:
      sev_json_text(buffer, "null");
      return;
  }
  abort();
}

void *__sev_json_encode(void *raw) {
  sev_json_buffer buffer = {0};
  sev_json_encode_value(&buffer, raw);
  return buffer.data;
}

static void sev_json_space(const char **cursor) {
  while (**cursor == ' ' || **cursor == '\n' || **cursor == '\r' || **cursor == '\t') ++*cursor;
}

static sev_value *sev_json_parse_value(const char **cursor) {
  sev_json_space(cursor);
  if (strncmp(*cursor, "null", 4) == 0) {
    *cursor += 4;
    return __sev_box_null();
  }
  if (**cursor == '"') {
    ++*cursor;
    sev_json_buffer buffer = {0};
    while (**cursor && **cursor != '"') {
      char value = *(*cursor)++;
      if (value == '\\') {
        value = *(*cursor)++;
        if (value == 'n') value = '\n';
      }
      sev_json_character(&buffer, value);
    }
    if (**cursor != '"') { free(buffer.data); return NULL; }
    ++*cursor;
    if (!buffer.data) buffer.data = strcpy(sev_allocate(1), "");
    return __sev_box_string(buffer.data);
  }
  if (**cursor == '[') {
    ++*cursor;
    sev_collection *values = __sev_collection_new(0);
    sev_json_space(cursor);
    if (**cursor != ']') {
      while (true) {
        sev_value *value = sev_json_parse_value(cursor);
        if (!value) return NULL;
        __sev_collection_push(values, value);
        sev_json_space(cursor);
        if (**cursor != ',') break;
        ++*cursor;
      }
    }
    if (**cursor != ']') return NULL;
    ++*cursor;
    return __sev_box_collection(values);
  }
  if (**cursor == '{') {
    ++*cursor;
    sev_map *object = __sev_map_new();
    sev_json_space(cursor);
    if (**cursor != '}') {
      while (true) {
        sev_value *key = sev_json_parse_value(cursor);
        if (!key || key->kind != SEV_STRING) return NULL;
        sev_json_space(cursor);
        if (**cursor != ':') return NULL;
        ++*cursor;
        sev_value *value = sev_json_parse_value(cursor);
        if (!value) return NULL;
        __sev_map_insert(object, key, value);
        sev_json_space(cursor);
        if (**cursor != ',') break;
        ++*cursor;
      }
    }
    if (**cursor != '}') return NULL;
    ++*cursor;
    return __sev_box_collection(object);
  }
  if (strncmp(*cursor, "true", 4) == 0) { *cursor += 4; return __sev_box_bool(true); }
  if (strncmp(*cursor, "false", 5) == 0) { *cursor += 5; return __sev_box_bool(false); }
  char *end = NULL;
  double number = strtod(*cursor, &end);
  if (end == *cursor) return NULL;
  bool integral = true;
  for (const char *value = *cursor; value < end; ++value) {
    if (*value == '.' || *value == 'e' || *value == 'E') integral = false;
  }
  *cursor = end;
  return integral ? __sev_box_i64((int64_t)number) : __sev_box_f64(number);
}

void *__sev_json_decode(void *text_raw) {
  const char *cursor = text_raw;
  sev_value *value = sev_json_parse_value(&cursor);
  sev_json_space(&cursor);
  if (!value || *cursor) return sev_failure("invalid JSON");
  return __sev_variant_new("ok", value);
}

void *__sev_json_object_get(void *value_raw, void *key_raw) {
  sev_value *boxed = value_raw;
  if (!boxed) abort();
  sev_map *map = boxed->kind == SEV_COLLECTION ? boxed->as.pointer : value_raw;
  if (!map || map->kind != 3) abort();
  return __sev_map_get(map, __sev_box_string(key_raw));
}

void *__sev_json_object_keys(void *value_raw) {
  sev_value *boxed = value_raw;
  if (!boxed || boxed->kind != SEV_COLLECTION || !boxed->as.pointer) abort();
  sev_map *map = boxed->as.pointer;
  if (map->kind != 3) abort();
  sev_collection *keys = __sev_collection_new(0);
  for (int64_t index = 0; index < map->size; ++index) {
    if (!map->keys[index] || map->keys[index]->kind != SEV_STRING) abort();
    __sev_collection_push(keys, map->keys[index]);
  }
  return keys;
}

void *__sev_json_object_values(void *value_raw) {
  sev_value *boxed = value_raw;
  if (!boxed || boxed->kind != SEV_COLLECTION || !boxed->as.pointer) abort();
  sev_map *map = boxed->as.pointer;
  if (map->kind != 3) abort();
  sev_collection *values = __sev_collection_new(0);
  for (int64_t index = 0; index < map->size; ++index) __sev_collection_push(values, map->values[index]);
  return values;
}

void __sev_json_object_set(void *value_raw, void *key_raw, void *item_raw) {
  sev_value *boxed = value_raw;
  if (!boxed || boxed->kind != SEV_COLLECTION || !boxed->as.pointer || !item_raw) abort();
  sev_map *map = boxed->as.pointer;
  if (map->kind != 3) abort();
  __sev_map_insert(map, __sev_box_string(key_raw), item_raw);
}

void *__sev_json_as_string(void *value_raw) {
  sev_value *value = value_raw;
  if (!value || value->kind != SEV_STRING) abort();
  return (void *)value->as.string;
}

int64_t __sev_json_as_int(void *value_raw) {
  sev_value *value = value_raw;
  if (!value || value->kind != SEV_INT) abort();
  return value->as.i64;
}

double __sev_json_as_float(void *value_raw) {
  sev_value *value = value_raw;
  if (!value) abort();
  if (value->kind == SEV_FLOAT) return value->as.f64;
  if (value->kind == SEV_INT) return (double)value->as.i64;
  abort();
}

bool __sev_json_as_bool(void *value_raw) {
  sev_value *value = value_raw;
  if (!value || value->kind != SEV_BOOL) abort();
  return value->as.boolean;
}

bool __sev_json_is_null(void *value_raw) {
  sev_value *value = value_raw;
  return value && value->kind == SEV_NULL;
}

void *__sev_json_kind(void *value_raw) {
  sev_value *value = value_raw;
  if (!value) return "invalid";
  switch (value->kind) {
    case SEV_INT: return "integer";
    case SEV_FLOAT: return "float";
    case SEV_BOOL: return "boolean";
    case SEV_STRING: return "string";
    case SEV_NULL: return "null";
    case SEV_COLLECTION: {
      sev_collection *collection = value->as.pointer;
      return collection && collection->kind == 3 ? "object" : "array";
    }
  }
  return "invalid";
}

void *__sev_json_as_int_list(void *value_raw) {
  sev_value *boxed = value_raw;
  if (!boxed) abort();
  sev_collection *values = boxed->kind == SEV_COLLECTION ? boxed->as.pointer : value_raw;
  if (!values) abort();
  if (values->kind == 3) abort();
  for (int64_t index = 0; index < values->size; ++index)
    if (!values->items[index] || values->items[index]->kind != SEV_INT) abort();
  return values;
}

void *__sev_json_as_string_list(void *value_raw) {
  sev_value *boxed = value_raw;
  if (!boxed) abort();
  sev_collection *values = boxed->kind == SEV_COLLECTION ? boxed->as.pointer : value_raw;
  if (!values) abort();
  if (values->kind == 3) abort();
  for (int64_t index = 0; index < values->size; ++index)
    if (!values->items[index] || values->items[index]->kind != SEV_STRING) abort();
  return values;
}

void *__sev_json_as_list(void *value_raw) {
  sev_value *boxed = value_raw;
  if (!boxed || boxed->kind != SEV_COLLECTION || !boxed->as.pointer) abort();
  sev_collection *values = boxed->as.pointer;
  if (values->kind == 3) abort();
  return values;
}

void __sev_log_info(void *message) {
  fprintf(stderr, "INFO %s\n", (const char *)message);
}

void __sev_log_error(void *message, void *cause) {
  sev_value *value = cause;
  if (value && value->kind == SEV_STRING) {
    fprintf(stderr, "ERROR %s: %s\n", (const char *)message, value->as.string);
  } else {
    fprintf(stderr, "ERROR %s\n", (const char *)message);
  }
}

typedef struct { int socket; } sev_tcp_listener;

static bool sev_socket_write_all(int socket_fd, const char *data, size_t size) {
  while (size) {
    ssize_t written = send(socket_fd, data, size, 0);
    if (written <= 0) return false;
    data += written;
    size -= (size_t)written;
  }
  return true;
}

static bool sev_socket_read_all(int socket_fd, char *data, size_t size) {
  while (size) {
    ssize_t received = recv(socket_fd, data, size, 0);
    if (received <= 0) return false;
    data += received;
    size -= (size_t)received;
  }
  return true;
}

void *__sev_network_listen(void *address_raw) {
  const char *address = address_raw;
  const char *colon = strrchr(address, ':');
  if (!colon) return sev_failure("network address requires a port");
  char host[64];
  size_t host_size = (size_t)(colon - address);
  if (host_size == 0 || host_size >= sizeof(host)) return sev_failure("invalid network host");
  memcpy(host, address, host_size);
  host[host_size] = '\0';
  char *port_end = NULL;
  long port = strtol(colon + 1, &port_end, 10);
  if (*port_end || port < 0 || port > 65535) return sev_failure("invalid network port");
  int socket_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (socket_fd < 0) return sev_failure("could not create listener");
  int reuse = 1;
  setsockopt(socket_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
  struct sockaddr_in endpoint = {0};
  endpoint.sin_family = AF_INET;
  endpoint.sin_port = htons((uint16_t)port);
  if (inet_pton(AF_INET, host, &endpoint.sin_addr) != 1 ||
      bind(socket_fd, (struct sockaddr *)&endpoint, sizeof(endpoint)) != 0 ||
      syscall(SYS_listen, socket_fd, 16) != 0) {
    close(socket_fd);
    return sev_failure("could not bind listener");
  }
  sev_tcp_listener *listener = sev_allocate(sizeof(*listener));
  listener->socket = socket_fd;
  return __sev_variant_new("ok", listener);
}

void *__sev_network_loopback_echo(void *message_raw) {
  const char *message = message_raw;
  int server = socket(AF_INET, SOCK_STREAM, 0);
  if (server < 0) return sev_failure("could not create loopback server");
  struct sockaddr_in endpoint = {0};
  endpoint.sin_family = AF_INET;
  endpoint.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (bind(server, (struct sockaddr *)&endpoint, sizeof(endpoint)) != 0 || syscall(SYS_listen, server, 1) != 0) {
    close(server);
    return sev_failure("could not bind loopback server");
  }
  socklen_t endpoint_size = sizeof(endpoint);
  if (getsockname(server, (struct sockaddr *)&endpoint, &endpoint_size) != 0) {
    close(server);
    return sev_failure("could not inspect loopback server");
  }
  int client = socket(AF_INET, SOCK_STREAM, 0);
  if (client < 0 || connect(client, (struct sockaddr *)&endpoint, sizeof(endpoint)) != 0) {
    if (client >= 0) close(client);
    close(server);
    return sev_failure("could not connect loopback client");
  }
  int peer = accept(server, NULL, NULL);
  if (peer < 0) { close(client); close(server); return sev_failure("could not accept loopback client"); }
  size_t size = strlen(message);
  char *buffer = sev_allocate(size + 1);
  bool success = sev_socket_write_all(client, message, size) &&
                 sev_socket_read_all(peer, buffer, size) &&
                 sev_socket_write_all(peer, buffer, size) &&
                 sev_socket_read_all(client, buffer, size);
  close(peer);
  close(client);
  close(server);
  if (!success) return sev_failure("loopback transfer failed");
  buffer[size] = '\0';
  return __sev_variant_new("ok", __sev_box_string(buffer));
}

static bool sev_parse_endpoint(const char *address, struct sockaddr_in *endpoint) {
  const char *colon = strrchr(address, ':');
  if (!colon) return false;
  char host[64]; size_t host_size = (size_t)(colon - address);
  if (host_size == 0 || host_size >= sizeof(host)) return false;
  memcpy(host, address, host_size); host[host_size] = '\0';
  char *end = NULL; long port = strtol(colon + 1, &end, 10);
  if (*end || port < 0 || port > 65535) return false;
  memset(endpoint, 0, sizeof(*endpoint)); endpoint->sin_family = AF_INET; endpoint->sin_port = htons((uint16_t)port);
  return inet_pton(AF_INET, host, &endpoint->sin_addr) == 1;
}

void *__sev_network_connect(void *address_raw) {
  struct sockaddr_in endpoint; if (!sev_parse_endpoint(address_raw, &endpoint)) return sev_failure("invalid network address");
  int descriptor = socket(AF_INET, SOCK_STREAM, 0);
  if (descriptor < 0 || connect(descriptor, (struct sockaddr *)&endpoint, sizeof(endpoint)) != 0) { if (descriptor >= 0) close(descriptor); return sev_failure("could not connect"); }
  int *handle = sev_allocate(sizeof(*handle)); *handle = descriptor; return __sev_variant_new("ok", handle);
}

void *__sev_network_accept(void *listener_raw) {
  sev_tcp_listener *listener = listener_raw; if (!listener) return sev_failure("invalid listener");
  int descriptor = accept(listener->socket, NULL, NULL); if (descriptor < 0) return sev_failure("could not accept connection");
  int *handle = sev_allocate(sizeof(*handle)); *handle = descriptor; return __sev_variant_new("ok", handle);
}

void *__sev_network_send(void *socket_raw, void *message_raw) {
  int *descriptor = socket_raw; if (!descriptor) return sev_failure("invalid socket");
  size_t size = strlen(message_raw); if (!sev_socket_write_all(*descriptor, message_raw, size)) return sev_failure("could not send");
  return __sev_variant_new("ok", __sev_box_i64((int64_t)size));
}

void *__sev_network_receive(void *socket_raw, int64_t count) {
  int *descriptor = socket_raw; if (!descriptor || count < 0) return sev_failure("invalid receive");
  char *buffer = sev_allocate((size_t)count + 1); ssize_t received = recv(*descriptor, buffer, (size_t)count, 0);
  if (received < 0) return sev_failure("could not receive"); buffer[received] = '\0'; return __sev_variant_new("ok", __sev_box_string(buffer));
}

void *__sev_network_close(void *socket_raw) {
  int *descriptor = socket_raw; if (!descriptor || close(*descriptor) != 0) return sev_failure("could not close socket"); return __sev_variant_new("ok", NULL);
}

int64_t __sev_process_run(void *command_raw) { int status = system(command_raw); return status < 0 ? -1 : WIFEXITED(status) ? WEXITSTATUS(status) : 128; }
int64_t __sev_process_spawn(void *command_raw) { pid_t child = fork(); if (child == 0) { execl("/bin/sh", "sh", "-c", (char *)command_raw, NULL); _exit(127); } return (int64_t)child; }
int64_t __sev_process_wait(int64_t process) { int status = 0; if (waitpid((pid_t)process, &status, 0) < 0) return -1; return WIFEXITED(status) ? WEXITSTATUS(status) : 128; }
bool __sev_process_kill(int64_t process) { return kill((pid_t)process, SIGTERM) == 0; }
void __sev_process_exit(int64_t status) { exit((int)status); }

void *__sev_http_request(void *method_raw, void *url_raw, void *body_raw) {
  (void)body_raw;
  const char *method = method_raw; const char *requested_url = url_raw;
  if (strncmp(requested_url, "https://", 8) == 0) {
    if (strcmp(method, "GET") != 0) return sev_failure("HTTPS runtime currently supports GET requests");
    int output[2]; if (pipe(output) != 0) return sev_failure("could not create HTTPS response pipe");
    pid_t child = fork();
    if (child == 0) {
      close(output[0]);
      if (dup2(output[1], STDOUT_FILENO) < 0) _exit(126);
      close(output[1]);
      execlp("curl", "curl", "--silent", "--show-error", "--fail", "--location",
             "--proto", "=https", "--tlsv1.2", requested_url, (char *)NULL);
      _exit(127);
    }
    close(output[1]);
    if (child < 0) { close(output[0]); return sev_failure("could not start HTTPS client"); }
    size_t capacity = 8192, used = 0; char *response = sev_allocate(capacity); ssize_t received;
    while ((received = read(output[0], response + used, capacity - used - 1)) > 0) {
      used += (size_t)received;
      if (capacity - used < 2) { capacity *= 2; response = realloc(response, capacity); if (!response) abort(); }
    }
    close(output[0]);
    int status = 0; if (waitpid(child, &status, 0) < 0 || received < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
      return sev_failure("HTTPS request failed");
    response[used] = '\0';
    return __sev_variant_new("ok", __sev_box_string(response));
  }
  const char *url = url_raw; const char *prefix = "http://";
  if (strncmp(url, prefix, strlen(prefix)) != 0) return sev_failure("only http:// URLs are supported");
  const char *authority = url + strlen(prefix); const char *slash = strchr(authority, '/');
  const char *path = slash ? slash : "/"; size_t authority_size = slash ? (size_t)(slash - authority) : strlen(authority);
  char endpoint[320]; if (authority_size + 4 >= sizeof(endpoint)) return sev_failure("HTTP authority is too long");
  memcpy(endpoint, authority, authority_size); endpoint[authority_size] = '\0'; if (!strchr(endpoint, ':')) strcat(endpoint, ":80");
  void *connected = __sev_network_connect(endpoint); sev_variant *variant = connected; if (!variant || strcmp(variant->tag, "ok") != 0) return connected;
  int *socket_handle = (int *)(void *)variant->field; char request[4096]; int request_size = snprintf(request, sizeof(request), "%s %s HTTP/1.0\r\nHost: %.*s\r\nConnection: close\r\n\r\n", (char *)method_raw, path, (int)authority_size, authority);
  if (request_size <= 0 || !sev_socket_write_all(*socket_handle, request, (size_t)request_size)) { close(*socket_handle); return sev_failure("HTTP send failed"); }
  size_t capacity = 8192, used = 0; char *response = sev_allocate(capacity); ssize_t received;
  while ((received = recv(*socket_handle, response + used, capacity - used - 1, 0)) > 0) { used += (size_t)received; if (capacity - used < 2) { capacity *= 2; response = realloc(response, capacity); if (!response) abort(); } }
  close(*socket_handle); if (received < 0) return sev_failure("HTTP receive failed"); response[used] = '\0'; char *body = strstr(response, "\r\n\r\n"); return __sev_variant_new("ok", __sev_box_string(body ? body + 4 : response));
}

void *__sev_https_download(void *url_raw, void *destination_raw) {
  const char *url = url_raw; const char *destination = destination_raw;
  if (strncmp(url, "https://", 8) != 0 || !destination || !*destination)
    return sev_failure("downloads require an HTTPS URL and destination");
  pid_t child = fork();
  if (child == 0) {
    execlp("curl", "curl", "--silent", "--show-error", "--fail", "--location",
           "--proto", "=https", "--tlsv1.2", "--output", destination, url, (char *)NULL);
    _exit(127);
  }
  if (child < 0) return sev_failure("could not start HTTPS download");
  int status = 0;
  if (waitpid(child, &status, 0) < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
    return sev_failure("HTTPS download failed");
  return __sev_variant_new("ok", NULL);
}

bool __sev_regex_matches(void *text_raw, void *pattern_raw) {
  regex_t expression;
  if (regcomp(&expression, (const char *)pattern_raw, REG_EXTENDED | REG_NOSUB) != 0) return false;
  bool matches = regexec(&expression, (const char *)text_raw, 0, NULL, 0) == 0;
  regfree(&expression);
  return matches;
}

void *__sev_regex_findall(void *text_raw, void *pattern_raw) {
  const char *text = text_raw;
  regex_t expression;
  sev_collection *result = __sev_collection_new(0);
  if (regcomp(&expression, pattern_raw, REG_EXTENDED) != 0) return result;
  regmatch_t match;
  const char *cursor = text;
  while (regexec(&expression, cursor, 1, &match, 0) == 0) {
    __sev_collection_push(result, __sev_box_string(sev_string_range(cursor, match.rm_so, match.rm_eo - match.rm_so)));
    cursor += match.rm_eo > 0 ? match.rm_eo : 1;
  }
  regfree(&expression);
  return result;
}

void *__sev_regex_split(void *text_raw, void *pattern_raw) {
  const char *text = text_raw;
  regex_t expression;
  sev_collection *result = __sev_collection_new(0);
  if (regcomp(&expression, pattern_raw, REG_EXTENDED) != 0) { __sev_collection_push(result, __sev_box_string(strdup(text))); return result; }
  regmatch_t match;
  const char *cursor = text;
  while (regexec(&expression, cursor, 1, &match, 0) == 0) {
    __sev_collection_push(result, __sev_box_string(sev_string_range(cursor, 0, match.rm_so)));
    cursor += match.rm_eo > 0 ? match.rm_eo : 1;
  }
  __sev_collection_push(result, __sev_box_string(strdup(cursor)));
  regfree(&expression);
  return result;
}

void *__sev_regex_sub(void *text_raw, void *pattern_raw, void *replacement_raw) {
  const char *text = text_raw;
  const char *replacement = replacement_raw;
  regex_t expression;
  if (regcomp(&expression, pattern_raw, REG_EXTENDED) != 0) return strdup(text);
  size_t capacity = strlen(text) + 1;
  char *result = sev_allocate(capacity);
  size_t used = 0;
  regmatch_t match;
  const char *cursor = text;
  while (regexec(&expression, cursor, 1, &match, 0) == 0) {
    size_t prefix = (size_t)match.rm_so;
    size_t replacement_size = strlen(replacement);
    size_t required = used + prefix + replacement_size + strlen(cursor + match.rm_eo) + 1;
    if (required > capacity) { capacity = required; result = realloc(result, capacity); if (!result) abort(); }
    memcpy(result + used, cursor, prefix); used += prefix;
    memcpy(result + used, replacement, replacement_size); used += replacement_size;
    cursor += match.rm_eo > 0 ? match.rm_eo : 1;
  }
  strcpy(result + used, cursor);
  regfree(&expression);
  return result;
}

void *__sev_host_container_backend(void) {
#ifdef __linux__
  if (access("/proc/self/ns", R_OK) == 0 && access("/sys/fs/cgroup", R_OK) == 0) return "linux";
#endif
  return "unavailable";
}

int64_t __sev_host_kvm_api_version(void) {
#ifdef __linux__
  int descriptor = open("/dev/kvm", O_RDWR | O_CLOEXEC);
  if (descriptor < 0) return -1;
  int version = ioctl(descriptor, KVM_GET_API_VERSION, 0);
  close(descriptor);
  return version;
#else
  return -1;
#endif
}

bool __sev_host_kvm_create_probe(void) {
#ifdef __linux__
  int descriptor = open("/dev/kvm", O_RDWR | O_CLOEXEC);
  if (descriptor < 0) return false;
  int vm = ioctl(descriptor, KVM_CREATE_VM, 0);
  if (vm >= 0) close(vm);
  close(descriptor);
  return vm >= 0;
#else
  return false;
#endif
}

int64_t __sev_host_page_size(void) {
  long size = sysconf(_SC_PAGESIZE);
  return size > 0 ? (int64_t)size : -1;
}

"#,
    );
    if program.functions.iter().any(|function| {
        function
            .native_symbol
            .as_deref()
            .is_some_and(|symbol| symbol.starts_with("__sev_database_"))
    }) {
        source.push_str(severian_platform::database_source());
    }
    if program.functions.iter().any(|function| {
        function
            .native_symbol
            .as_deref()
            .is_some_and(|symbol| symbol.starts_with("__sev_mysql_"))
    }) {
        source.push_str(severian_platform::mysql_source());
    }
    let drawable_classes = program
        .classes
        .iter()
        .filter(|class| class.methods.iter().any(|method| method.name == "draw"))
        .collect::<Vec<_>>();
    for class in &drawable_classes {
        writeln!(
            source,
            "extern void {}(void *);",
            class_function_symbol(&class.name, "draw")
        )
        .unwrap();
    }
    source.push_str("void __sev_dispatch_draw(void *raw) { sev_object *value = raw;\n");
    for class in &drawable_classes {
        writeln!(
            source,
            "  if (strcmp(value->class_name, \"{}\") == 0) {{ {}(raw); return; }}",
            class.name,
            class_function_symbol(&class.name, "draw")
        )
        .unwrap();
    }
    source.push_str("  abort();\n}\n\n");
    let dynamic_getters = program
        .classes
        .iter()
        .filter_map(|class| {
            class
                .methods
                .iter()
                .find(|method| {
                    method.name == "get"
                        && method.params.len() == 1
                        && method.params[0].ty == ValueType::String
                        && method.return_type != ValueType::Unit
                })
                .map(|method| (class, method))
        })
        .collect::<Vec<_>>();
    for (class, method) in &dynamic_getters {
        writeln!(
            source,
            "extern {} {}(void *, void *);",
            c_type(method.return_type),
            class_function_symbol(&class.name, "get")
        )
        .unwrap();
    }
    source.push_str("void *__sev_dynamic_object_get(void *raw, void *key) { sev_object *value = raw; if (!value || value->magic != SEV_OBJECT_MAGIC) abort();\n");
    for (class, method) in &dynamic_getters {
        let call = format!("{}(raw, key)", class_function_symbol(&class.name, "get"));
        let result = match method.return_type {
            ValueType::String => format!("__sev_box_string({call})"),
            ValueType::Int => format!("__sev_box_i64({call})"),
            ValueType::Float => format!("__sev_box_f64({call})"),
            ValueType::Bool => format!("__sev_box_bool({call})"),
            ValueType::List | ValueType::Tuple | ValueType::Map | ValueType::Set => {
                format!("__sev_box_collection({call})")
            }
            _ => call,
        };
        writeln!(
            source,
            "  if (strcmp(value->class_name, \"{}\") == 0) return {result};",
            class.name
        )
        .unwrap();
    }
    source.push_str("  return __sev_object_get(raw, key);\n}\n\n");
    for dispatch in dynamic_method_dispatches(program) {
        for class in &dispatch.classes {
            write!(
                source,
                "extern {} {}(void *",
                c_type(dispatch.returns),
                class_function_symbol(class, &dispatch.method)
            )
            .unwrap();
            for parameter in &dispatch.params {
                write!(source, ", {}", c_type(*parameter)).unwrap();
            }
            source.push_str(");\n");
        }
        write!(
            source,
            "{} {}(void *raw",
            c_type(dispatch.returns),
            dispatch.symbol
        )
        .unwrap();
        for (index, parameter) in dispatch.params.iter().enumerate() {
            write!(source, ", {} arg_{index}", c_type(*parameter)).unwrap();
        }
        source.push_str(") { sev_object *value = raw; if (!value || value->magic != SEV_OBJECT_MAGIC) abort();\n");
        let arguments = (0..dispatch.params.len())
            .map(|index| format!(", arg_{index}"))
            .collect::<String>();
        for class in &dispatch.classes {
            let call = format!(
                "{}(raw{arguments})",
                class_function_symbol(class, &dispatch.method)
            );
            if dispatch.returns == ValueType::Unit {
                writeln!(
                    source,
                    "  if (strcmp(value->class_name, {class:?}) == 0) {{ {call}; return; }}"
                )
                .unwrap();
            } else {
                writeln!(
                    source,
                    "  if (strcmp(value->class_name, {class:?}) == 0) return {call};"
                )
                .unwrap();
            }
        }
        source.push_str("  abort();\n}\n\n");
    }
    let mut return_types = specs
        .iter()
        .map(|spec| spec.return_type)
        .collect::<HashSet<_>>();
    return_types.extend(
        program
            .classes
            .iter()
            .flat_map(|class| &class.methods)
            .map(|method| method.return_type),
    );
    if uses_channels {
        return_types.insert(ValueType::Unit);
    }
    source.push_str("typedef struct { pthread_t thread; } sev_task_unit;\n");
    let mut declared_task_suffixes = HashSet::new();
    for ty in &return_types {
        if *ty != ValueType::Unit {
            let suffix = task_type_suffix(*ty);
            if !declared_task_suffixes.insert(suffix) {
                continue;
            }
            writeln!(
                source,
                "typedef struct {{ pthread_t thread; {} result; }} sev_task_{};",
                c_type(*ty),
                suffix
            )
            .unwrap();
        }
    }
    source.push('\n');
    for spec in &specs {
        let result_type = c_type(spec.return_type);
        let function_symbol = source_function_symbol(&spec.function);
        write!(source, "extern {result_type} {function_symbol}(").unwrap();
        if spec.params.is_empty() {
            source.push_str("void");
        } else {
            for (index, ty) in spec.params.iter().enumerate() {
                if index > 0 {
                    source.push_str(", ");
                }
                source.push_str(c_type(*ty));
            }
        }
        source.push_str(");\n");
        let header = if spec.return_type == ValueType::Unit {
            "sev_task_unit".to_owned()
        } else {
            format!("sev_task_{}", task_type_suffix(spec.return_type))
        };
        writeln!(source, "typedef struct {{ {header} base;").unwrap();
        for (index, ty) in spec.params.iter().enumerate() {
            writeln!(source, "  {} arg_{index};", c_type(*ty)).unwrap();
        }
        writeln!(source, "}} sev_task_frame_{};", spec.symbol).unwrap();
        writeln!(
            source,
            "static void *__sev_task_worker_{}(void *raw) {{",
            spec.symbol
        )
        .unwrap();
        writeln!(source, "  sev_task_frame_{} *task = raw;", spec.symbol).unwrap();
        let args = (0..spec.params.len())
            .map(|index| format!("task->arg_{index}"))
            .collect::<Vec<_>>()
            .join(", ");
        if spec.return_type == ValueType::Unit {
            writeln!(source, "  {function_symbol}({args});").unwrap();
        } else {
            writeln!(source, "  task->base.result = {function_symbol}({args});").unwrap();
        }
        source.push_str("  return NULL;\n}\n");
        write!(source, "void *__sev_task_spawn_{}(", spec.symbol).unwrap();
        for (index, ty) in spec.params.iter().enumerate() {
            if index > 0 {
                source.push_str(", ");
            }
            write!(source, "{} arg_{index}", c_type(*ty)).unwrap();
        }
        if spec.params.is_empty() {
            source.push_str("void");
        }
        source.push_str(") {\n");
        writeln!(
            source,
            "  sev_task_frame_{} *task = calloc(1, sizeof(*task));",
            spec.symbol
        )
        .unwrap();
        source.push_str("  if (!task) abort();\n");
        for index in 0..spec.params.len() {
            writeln!(source, "  task->arg_{index} = arg_{index};").unwrap();
        }
        writeln!(source, "  if (pthread_create(&task->base.thread, NULL, __sev_task_worker_{}, task) != 0) abort();", spec.symbol).unwrap();
        source.push_str("  return task;\n}\n\n");
    }
    for class in &program.classes {
        for method in &class.methods {
            let result_type = c_type(method.return_type);
            let method_symbol = class_function_symbol(&class.name, &method.name);
            write!(source, "extern {result_type} {method_symbol}(void *").unwrap();
            for parameter in &method.params {
                write!(source, ", {}", c_type(parameter.ty)).unwrap();
            }
            source.push_str(");\n");
            let header = if method.return_type == ValueType::Unit {
                "sev_task_unit".to_owned()
            } else {
                format!("sev_task_{}", task_type_suffix(method.return_type))
            };
            writeln!(source, "typedef struct {{ {header} base; sev_object *self;").unwrap();
            for (index, parameter) in method.params.iter().enumerate() {
                writeln!(source, "  {} arg_{index};", c_type(parameter.ty)).unwrap();
            }
            writeln!(source, "}} sev_method_task_{}_{};", class.name, method.name).unwrap();
            writeln!(
                source,
                "static void *__sev_method_worker_{}_{}(void *raw) {{",
                class.name, method.name
            )
            .unwrap();
            writeln!(
                source,
                "  sev_method_task_{}_{} *task = raw;",
                class.name, method.name
            )
            .unwrap();
            source.push_str("  pthread_mutex_lock(&task->self->mutex);\n");
            let args = (0..method.params.len())
                .map(|index| format!(", task->arg_{index}"))
                .collect::<String>();
            if method.return_type == ValueType::Unit {
                writeln!(source, "  {method_symbol}(task->self{args});").unwrap();
            } else {
                writeln!(
                    source,
                    "  task->base.result = {method_symbol}(task->self{args});"
                )
                .unwrap();
            }
            source.push_str("  pthread_mutex_unlock(&task->self->mutex);\n  return NULL;\n}\n");
            write!(
                source,
                "void *__sev_task_spawn_{}_{}(void *self_raw",
                class.name, method.name
            )
            .unwrap();
            for (index, parameter) in method.params.iter().enumerate() {
                write!(source, ", {} arg_{index}", c_type(parameter.ty)).unwrap();
            }
            source.push_str(") {\n");
            writeln!(
                source,
                "  sev_method_task_{}_{} *task = calloc(1, sizeof(*task));",
                class.name, method.name
            )
            .unwrap();
            source.push_str("  if (!task) abort();\n  task->self = self_raw;\n");
            for index in 0..method.params.len() {
                writeln!(source, "  task->arg_{index} = arg_{index};").unwrap();
            }
            writeln!(source, "  if (pthread_create(&task->base.thread, NULL, __sev_method_worker_{}_{}, task) != 0) abort();", class.name, method.name).unwrap();
            source.push_str("  return task;\n}\n\n");
        }
    }
    if uses_channels {
        source.push_str(concat!(
            "typedef struct {\n",
            "  pthread_mutex_t mutex;\n",
            "  pthread_cond_t readable;\n",
            "  pthread_cond_t writable;\n",
            "  void **items;\n",
            "  int64_t capacity;\n",
            "  int64_t head;\n",
            "  int64_t tail;\n",
            "  int64_t count;\n",
            "} sev_channel;\n",
            "typedef struct { sev_task_unit base; sev_channel *channel; void *value; } sev_send_task;\n\n",
            "void *__sev_channel_create(int64_t capacity) {\n",
            "  if (capacity <= 0) abort();\n",
            "  sev_channel *channel = calloc(1, sizeof(*channel));\n",
            "  if (!channel) abort();\n",
            "  channel->items = calloc((size_t)capacity, sizeof(*channel->items));\n",
            "  if (!channel->items) abort();\n",
            "  channel->capacity = capacity;\n",
            "  pthread_mutex_init(&channel->mutex, NULL);\n",
            "  pthread_cond_init(&channel->readable, NULL);\n",
            "  pthread_cond_init(&channel->writable, NULL);\n",
            "  return channel;\n",
            "}\n",
            "static void *__sev_channel_send_worker(void *raw) {\n",
            "  sev_send_task *task = raw;\n",
            "  sev_channel *channel = task->channel;\n",
            "  pthread_mutex_lock(&channel->mutex);\n",
            "  while (channel->count == channel->capacity) pthread_cond_wait(&channel->writable, &channel->mutex);\n",
            "  channel->items[channel->tail] = task->value;\n",
            "  channel->tail = (channel->tail + 1) % channel->capacity;\n",
            "  channel->count += 1;\n",
            "  pthread_cond_signal(&channel->readable);\n",
            "  pthread_mutex_unlock(&channel->mutex);\n",
            "  return NULL;\n",
            "}\n",
            "void *__sev_channel_send_ptr_async(void *value, void *raw_channel) {\n",
            "  sev_send_task *task = calloc(1, sizeof(*task));\n",
            "  if (!task) abort();\n",
            "  task->channel = raw_channel;\n",
            "  task->value = value;\n",
            "  if (pthread_create(&task->base.thread, NULL, __sev_channel_send_worker, task) != 0) abort();\n",
            "  return task;\n",
            "}\n",
            "void *__sev_channel_receive_ptr(void *raw_channel) {\n",
            "  sev_channel *channel = raw_channel;\n",
            "  pthread_mutex_lock(&channel->mutex);\n",
            "  while (channel->count == 0) pthread_cond_wait(&channel->readable, &channel->mutex);\n",
            "  void *value = channel->items[channel->head];\n",
            "  channel->head = (channel->head + 1) % channel->capacity;\n",
            "  channel->count -= 1;\n",
            "  pthread_cond_signal(&channel->writable);\n",
            "  pthread_mutex_unlock(&channel->mutex);\n",
            "  return value;\n",
            "}\n\n",
        ));
    }
    source.push_str("void __sev_task_await_unit(void *raw) { sev_task_unit *task = raw; if (!task) abort(); pthread_join(task->thread, NULL); free(task); }\n");
    let mut defined_await_suffixes = HashSet::new();
    for ty in return_types {
        let suffix = task_type_suffix(ty);
        if ty != ValueType::Unit {
            if !defined_await_suffixes.insert(suffix) {
                continue;
            }
            writeln!(
                source,
                "{} __sev_task_await_{suffix}(void *raw) {{",
                c_type(ty)
            )
            .unwrap();
            writeln!(source, "  sev_task_{suffix} *task = raw;").unwrap();
            source.push_str("  pthread_join(task->thread, NULL);\n");
            writeln!(source, "  {} result = task->result;", c_type(ty)).unwrap();
            source.push_str("  free(task);\n  return result;\n}\n");
        }
    }
    let mut native_symbols = program
        .functions
        .iter()
        .filter_map(|function| function.native_symbol.clone())
        .collect::<HashSet<_>>();
    native_symbols.extend(
        native_call_signatures(program)
            .into_values()
            .map(|signature| signature.symbol),
    );
    let has_model_graph = native_symbols
        .iter()
        .any(|symbol| symbol.starts_with("__sev_model_graph_"));
    source.push_str(&severian_platform::tensor_source(
        native_symbols.contains("__sev_tensor_relu") || has_model_graph,
        native_symbols.contains("__sev_tensor_add") || has_model_graph,
        native_symbols.contains("__sev_tensor_matmul") || has_model_graph,
        native_symbols.contains("__sev_tensor_transpose") || has_model_graph,
        native_symbols.contains("__sev_tensor_scale") || has_model_graph,
        native_symbols.contains("__sev_tensor_softmax_rows") || has_model_graph,
        native_symbols.contains("__sev_tensor_layer_norm") || has_model_graph,
        native_symbols.contains("__sev_tensor_relu_backward")
            || native_symbols.contains("__sev_tensor_backward_mse"),
        native_symbols.contains("__sev_tensor_softmax_backward")
            || native_symbols.contains("__sev_tensor_backward_mse"),
        native_symbols.contains("__sev_tensor_layer_norm_backward")
            || native_symbols.contains("__sev_tensor_backward_mse"),
        native_symbols.contains("__sev_tensor_backward_mse"),
        rocm,
    ));
    if native_symbols
        .iter()
        .any(|symbol| symbol.starts_with("__sev_safetensor_"))
    {
        source.push_str(severian_platform::safetensors_source());
    }
    if has_model_graph {
        source.push_str(&severian_platform::model_graph_source(rocm));
    }
    let tensor_regions = program
        .functions
        .iter()
        .filter(|function| {
            function.native_symbol.is_none()
                && function
                    .decorators
                    .iter()
                    .any(|decorator| decorator.package == "tensor")
        })
        .collect::<Vec<_>>();
    if !tensor_regions.is_empty() {
        source.push_str(concat!(
            "extern void *__sev_xla_execute(uint64_t, const uint8_t *, size_t, void **, size_t);\n",
            "extern void *__sev_xla_i64_token(int64_t);\n",
            "extern int64_t __sev_xla_argmax_bf16(void *);\n\n",
        ));
        for function in tensor_regions {
            let module = stablehlo::lower_entry(program, function.id)?;
            write!(source, "void *{}(", source_function_symbol(&function.name)).unwrap();
            for index in 0..function.params.len() {
                if index > 0 {
                    source.push_str(", ");
                }
                write!(source, "void *arg{index}").unwrap();
            }
            source.push_str(") {\n  static const uint8_t module[] = {");
            for (index, byte) in module.as_str().bytes().enumerate() {
                if index > 0 {
                    source.push_str(",");
                }
                write!(source, "{byte}").unwrap();
            }
            source.push_str("};\n  void *args[] = {");
            for index in 0..function.params.len() {
                if index > 0 {
                    source.push_str(", ");
                }
                write!(source, "arg{index}").unwrap();
            }
            writeln!(
                source,
                "}};\n  return __sev_xla_execute({}ULL, module, sizeof(module), args, {});\n}}",
                function.id.0,
                function.params.len(),
            )
            .unwrap();
        }
    }
    Ok(source)
}
