#include <float.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
    size_t length;
    size_t capacity;
    uintptr_t *values;
} sev_list;

/* An MLIR unranked memref points at a ranked descriptor owned by its current
 * function. Aggregate tensor fields need a stable copy that survives return. */
typedef struct {
    int64_t rank;
    uintptr_t descriptor[];
} sev_unranked_tensor_box;

void *__sev_tensor_box_unranked(int64_t rank, const void *descriptor) {
    if (rank < 0 || rank > 64 || descriptor == NULL) abort();
    size_t words = 3u + 2u * (size_t)rank;
    if (words > (SIZE_MAX - sizeof(sev_unranked_tensor_box)) / sizeof(uintptr_t)) abort();
    sev_unranked_tensor_box *box = malloc(
        sizeof(sev_unranked_tensor_box) + words * sizeof(uintptr_t));
    if (box == NULL) abort();
    box->rank = rank;
    memcpy(box->descriptor, descriptor, words * sizeof(uintptr_t));
    return box;
}

int64_t __sev_tensor_box_rank(const void *value) {
    if (value == NULL) abort();
    return ((const sev_unranked_tensor_box *)value)->rank;
}

void *__sev_tensor_box_descriptor(void *value) {
    if (value == NULL) abort();
    return ((sev_unranked_tensor_box *)value)->descriptor;
}

typedef struct {
    unsigned __int128 bits;
} sev_tensor_cell;

typedef struct sev_tensor {
    size_t rank;
    size_t count;
    int32_t dtype;
    sev_tensor_cell *values;
    const uint8_t *mapped_values;
    size_t mapped_byte_width;
    int64_t *shape;
    int64_t *strides;
    int64_t offset;
    _Bool owns_values;
    sev_tensor_cell *gradient;
    struct sev_tensor *parent;
    char operation;
} sev_tensor;

#define SEV_STORAGE_VIEW_ABI_MAGIC UINT64_C(0x535653544f524147)
#define SEV_STORAGE_VIEW_ABI_VERSION UINT32_C(1)
#define SEV_STORAGE_VIEW_READ_ONLY UINT64_C(1)
#define SEV_STORAGE_VIEW_CONTIGUOUS UINT64_C(2)

typedef enum {
    SEV_STORAGE_ELEMENT_SIGNED_INTEGER = 1,
    SEV_STORAGE_ELEMENT_UNSIGNED_INTEGER = 2,
    SEV_STORAGE_ELEMENT_FLOAT = 3,
} sev_storage_element_kind;

typedef enum {
    SEV_STORAGE_FLOAT_NONE = 0,
    SEV_STORAGE_FLOAT_IEEE = 1,
    SEV_STORAGE_FLOAT_BRAIN = 2,
    SEV_STORAGE_FLOAT8_E4M3_FN = 3,
    SEV_STORAGE_FLOAT8_E5M2 = 4,
} sev_storage_float_format;

typedef struct {
    uint32_t abi_version;
    uint32_t byte_size;
    uint32_t kind;
    uint32_t bits;
    uint32_t float_format;
    uint32_t reserved;
} sev_storage_element_representation_abi;

typedef struct {
    uint64_t magic;
    uint32_t abi_version;
    uint32_t byte_size;
    uint64_t flags;
    const uint8_t *data;
    uint64_t byte_length;
    uint64_t rank;
    const int64_t *dimensions;
    const int64_t *strides;
    int64_t offset;
    sev_storage_element_representation_abi element;
    void *owner;
} sev_storage_view_abi;

_Static_assert(sizeof(sev_storage_element_representation_abi) == 24, "storage element ABI drift");
_Static_assert(sizeof(sev_storage_view_abi) == 104, "storage view ABI drift");

typedef struct {
    int descriptor;
    size_t length;
    const uint8_t *bytes;
    uint64_t header_length;
} sev_safetensor;

static sev_tensor *sev_tensor_new(size_t rank, const int64_t *shape, int32_t dtype);
static void *sev_tensor_wrap(sev_tensor *tensor);
static unsigned sev_tensor_dtype_bits(int32_t dtype);
static _Bool sev_tensor_dtype_signed(int32_t dtype);
static _Bool sev_tensor_dtype_unsigned(int32_t dtype);

static void sev_tensor_abort_if(_Bool condition) {
    if (condition) abort();
}

static sev_list *sev_tensor_list(void) {
    sev_list *list = calloc(1, sizeof(*list));
    sev_tensor_abort_if(list == NULL);
    return list;
}

static void sev_tensor_list_push(sev_list *list, uintptr_t value) {
    if (list->length == list->capacity) {
        size_t capacity = list->capacity == 0 ? 4 : list->capacity * 2;
        sev_tensor_abort_if(capacity < list->capacity);
        uintptr_t *values = realloc(list->values, capacity * sizeof(*values));
        sev_tensor_abort_if(values == NULL);
        list->values = values;
        list->capacity = capacity;
    }
    list->values[list->length++] = value;
}

static uint64_t sev_read_u64(const uint8_t *bytes) {
    uint64_t value = 0;
    for (unsigned index = 0; index < 8; ++index) {
        value |= (uint64_t)bytes[index] << (index * 8);
    }
    return value;
}

static unsigned __int128 sev_read_u128(const uint8_t *bytes, size_t width) {
    unsigned __int128 value = 0;
    for (size_t index = 0; index < width; ++index) {
        value |= (unsigned __int128)bytes[index] << (index * 8);
    }
    return value;
}

static const uint8_t *sev_find_bytes(
    const uint8_t *start,
    size_t length,
    const char *needle
) {
    size_t needle_length = strlen(needle);
    if (needle_length == 0 || needle_length > length) return NULL;
    for (size_t index = 0; index <= length - needle_length; ++index) {
        if (memcmp(start + index, needle, needle_length) == 0) return start + index;
    }
    return NULL;
}

static const uint8_t *sev_skip_json_space(const uint8_t *cursor, const uint8_t *end) {
    while (cursor < end && (*cursor == ' ' || *cursor == '\t' || *cursor == '\r' || *cursor == '\n')) {
        ++cursor;
    }
    return cursor;
}

static _Bool sev_safetensor_object(
    const sev_safetensor *store,
    const char *name,
    const uint8_t **object,
    const uint8_t **object_end
) {
    size_t name_length = strlen(name);
    char *quoted = malloc(name_length + 3);
    sev_tensor_abort_if(quoted == NULL);
    quoted[0] = '"';
    memcpy(quoted + 1, name, name_length);
    quoted[name_length + 1] = '"';
    quoted[name_length + 2] = '\0';
    const uint8_t *header = store->bytes + 8;
    const uint8_t *header_end = header + store->header_length;
    const uint8_t *cursor = sev_find_bytes(header, store->header_length, quoted);
    free(quoted);
    if (cursor == NULL) return 0;
    cursor += name_length + 2;
    cursor = sev_skip_json_space(cursor, header_end);
    if (cursor == header_end || *cursor++ != ':') return 0;
    cursor = sev_skip_json_space(cursor, header_end);
    if (cursor == header_end || *cursor++ != '{') return 0;
    const uint8_t *end = cursor;
    while (end < header_end && *end != '}') ++end;
    if (end == header_end) return 0;
    *object = cursor;
    *object_end = end;
    return 1;
}

static _Bool sev_safetensor_string_field(
    const uint8_t *object,
    const uint8_t *object_end,
    const char *field,
    char *output,
    size_t capacity
) {
    const uint8_t *cursor = sev_find_bytes(object, (size_t)(object_end - object), field);
    if (cursor == NULL) return 0;
    cursor += strlen(field);
    cursor = sev_skip_json_space(cursor, object_end);
    if (cursor == object_end || *cursor++ != ':') return 0;
    cursor = sev_skip_json_space(cursor, object_end);
    if (cursor == object_end || *cursor++ != '"') return 0;
    const uint8_t *end = cursor;
    while (end < object_end && *end != '"') ++end;
    if (end == object_end || (size_t)(end - cursor) >= capacity) return 0;
    memcpy(output, cursor, (size_t)(end - cursor));
    output[end - cursor] = '\0';
    return 1;
}

static _Bool sev_parse_u64(
    const uint8_t **cursor,
    const uint8_t *end,
    uint64_t *value
) {
    *cursor = sev_skip_json_space(*cursor, end);
    if (*cursor == end || **cursor < '0' || **cursor > '9') return 0;
    uint64_t parsed = 0;
    while (*cursor < end && **cursor >= '0' && **cursor <= '9') {
        uint64_t digit = (uint64_t)(**cursor - '0');
        if (parsed > (UINT64_MAX - digit) / 10) return 0;
        parsed = parsed * 10 + digit;
        ++*cursor;
    }
    *value = parsed;
    return 1;
}

static const uint8_t *sev_safetensor_array_field(
    const uint8_t *object,
    const uint8_t *object_end,
    const char *field
) {
    const uint8_t *cursor = sev_find_bytes(object, (size_t)(object_end - object), field);
    if (cursor == NULL) return NULL;
    cursor += strlen(field);
    cursor = sev_skip_json_space(cursor, object_end);
    if (cursor == object_end || *cursor++ != ':') return NULL;
    cursor = sev_skip_json_space(cursor, object_end);
    if (cursor == object_end || *cursor++ != '[') return NULL;
    return cursor;
}

static _Bool sev_safetensor_offsets(
    const sev_safetensor *store,
    const char *name,
    uint64_t *start,
    uint64_t *end
) {
    const uint8_t *object;
    const uint8_t *object_end;
    if (!sev_safetensor_object(store, name, &object, &object_end)) return 0;
    const uint8_t *cursor = sev_safetensor_array_field(object, object_end, "\"data_offsets\"");
    if (cursor == NULL || !sev_parse_u64(&cursor, object_end, start)) return 0;
    cursor = sev_skip_json_space(cursor, object_end);
    if (cursor == object_end || *cursor++ != ',') return 0;
    return sev_parse_u64(&cursor, object_end, end) && *end >= *start;
}

int64_t __sev_safetensor_open(const char *path) {
    int descriptor = open(path, O_RDONLY);
    if (descriptor < 0) return 0;
    struct stat information;
    if (fstat(descriptor, &information) != 0 || information.st_size < 8) {
        close(descriptor);
        return 0;
    }
    size_t length = (size_t)information.st_size;
    const uint8_t *bytes = mmap(NULL, length, PROT_READ, MAP_PRIVATE, descriptor, 0);
    if (bytes == MAP_FAILED) {
        close(descriptor);
        return 0;
    }
    uint64_t header_length = sev_read_u64(bytes);
    if (header_length > length - 8) {
        munmap((void *)bytes, length);
        close(descriptor);
        return 0;
    }
    sev_safetensor *store = malloc(sizeof(*store));
    if (store == NULL) abort();
    *store = (sev_safetensor){descriptor, length, bytes, header_length};
    return (int64_t)(intptr_t)store;
}

int32_t __sev_safetensor_close(int64_t handle) {
    sev_safetensor *store = (sev_safetensor *)(intptr_t)handle;
    if (store == NULL) return -1;
    int result = munmap((void *)store->bytes, store->length);
    if (close(store->descriptor) != 0) result = -1;
    free(store);
    return result;
}

const char *__sev_safetensor_dtype(int64_t handle, const char *name) {
    static _Thread_local char dtype[24];
    sev_safetensor *store = (sev_safetensor *)(intptr_t)handle;
    const uint8_t *object;
    const uint8_t *object_end;
    if (store == NULL || !sev_safetensor_object(store, name, &object, &object_end)
        || !sev_safetensor_string_field(object, object_end, "\"dtype\"", dtype, sizeof(dtype))) {
        dtype[0] = '\0';
    }
    return dtype;
}

void *__sev_safetensor_shape(int64_t handle, const char *name) {
    sev_list *shape = sev_tensor_list();
    sev_safetensor *store = (sev_safetensor *)(intptr_t)handle;
    const uint8_t *object;
    const uint8_t *object_end;
    if (store == NULL || !sev_safetensor_object(store, name, &object, &object_end)) return shape;
    const uint8_t *cursor = sev_safetensor_array_field(object, object_end, "\"shape\"");
    if (cursor == NULL) return shape;
    cursor = sev_skip_json_space(cursor, object_end);
    while (cursor < object_end && *cursor != ']') {
        uint64_t dimension;
        if (!sev_parse_u64(&cursor, object_end, &dimension) || dimension > INT64_MAX) return shape;
        sev_tensor_list_push(shape, (uintptr_t)dimension);
        cursor = sev_skip_json_space(cursor, object_end);
        if (cursor < object_end && *cursor == ',') ++cursor;
        cursor = sev_skip_json_space(cursor, object_end);
    }
    return shape;
}

int64_t __sev_safetensor_byte_start(int64_t handle, const char *name) {
    sev_safetensor *store = (sev_safetensor *)(intptr_t)handle;
    uint64_t start;
    uint64_t end;
    if (store == NULL || !sev_safetensor_offsets(store, name, &start, &end)) return -1;
    uint64_t data_start = 8 + store->header_length;
    if (start > INT64_MAX - data_start || data_start + end > store->length) return -1;
    return (int64_t)(data_start + start);
}

int64_t __sev_safetensor_byte_end(int64_t handle, const char *name) {
    sev_safetensor *store = (sev_safetensor *)(intptr_t)handle;
    uint64_t start;
    uint64_t end;
    if (store == NULL || !sev_safetensor_offsets(store, name, &start, &end)) return -1;
    uint64_t data_start = 8 + store->header_length;
    if (end > INT64_MAX - data_start || data_start + end > store->length) return -1;
    return (int64_t)(data_start + end);
}

static int32_t sev_safetensor_dtype_tag(const char *dtype) {
    const char *names[] = {
        "I8", "I16", "I32", "I64", "I128", "U8", "U16", "U32", "U64", "U128",
        "F8_E4M3FN", "F8_E5M2", "F16", "BF16", "F32", "F64", "F128", "F80"
    };
    for (int32_t index = 0; index < 18; ++index) {
        if (strcmp(dtype, names[index]) == 0) return index;
    }
    if (strcmp(dtype, "F8_E4M3") == 0) return 10;
    return -1;
}

static long double sev_decode_binary_float(
    const uint8_t *bytes,
    unsigned exponent_bits,
    unsigned fraction_bits,
    int bias
) {
    unsigned total_bits = 1 + exponent_bits + fraction_bits;
    unsigned __int128 raw = 0;
    for (unsigned index = 0; index < total_bits / 8; ++index) {
        raw |= (unsigned __int128)bytes[index] << (index * 8);
    }
    _Bool negative = ((raw >> (total_bits - 1)) & 1) != 0;
    unsigned __int128 exponent_mask = ((unsigned __int128)1 << exponent_bits) - 1;
    unsigned exponent = (unsigned)((raw >> fraction_bits) & exponent_mask);
    unsigned __int128 fraction_mask = ((unsigned __int128)1 << fraction_bits) - 1;
    unsigned __int128 fraction = raw & fraction_mask;
    if (exponent == (unsigned)exponent_mask) {
        if (fraction != 0) return NAN;
        return negative ? -INFINITY : INFINITY;
    }
    long double significand = exponent == 0 ? 0.0L : 1.0L;
    for (unsigned bit = 0; bit < fraction_bits; ++bit) {
        if (((fraction >> bit) & 1) != 0) {
            significand += ldexpl(1.0L, (int)bit - (int)fraction_bits);
        }
    }
    int power = exponent == 0 ? 1 - bias : (int)exponent - bias;
    long double value = ldexpl(significand, power);
    return negative ? -value : value;
}

sev_storage_view_abi *__sev_safetensor_view(int64_t handle, const char *name) {
    sev_safetensor *store = (sev_safetensor *)(intptr_t)handle;
    sev_tensor_abort_if(store == NULL);
    char dtype[24];
    const uint8_t *object;
    const uint8_t *object_end;
    sev_tensor_abort_if(!sev_safetensor_object(store, name, &object, &object_end));
    sev_tensor_abort_if(!sev_safetensor_string_field(
        object, object_end, "\"dtype\"", dtype, sizeof(dtype)
    ));
    int32_t storage_dtype = sev_safetensor_dtype_tag(dtype);
    sev_tensor_abort_if(storage_dtype < 0);
    const uint8_t *cursor = sev_safetensor_array_field(object, object_end, "\"shape\"");
    sev_tensor_abort_if(cursor == NULL);
    size_t rank = 0;
    const uint8_t *shape_cursor = cursor;
    shape_cursor = sev_skip_json_space(shape_cursor, object_end);
    while (shape_cursor < object_end && *shape_cursor != ']') {
        uint64_t ignored;
        sev_tensor_abort_if(!sev_parse_u64(&shape_cursor, object_end, &ignored));
        ++rank;
        shape_cursor = sev_skip_json_space(shape_cursor, object_end);
        if (shape_cursor < object_end && *shape_cursor == ',') ++shape_cursor;
        shape_cursor = sev_skip_json_space(shape_cursor, object_end);
    }
    int64_t *shape = calloc(rank == 0 ? 1 : rank, sizeof(*shape));
    sev_tensor_abort_if(shape == NULL);
    for (size_t axis = 0; axis < rank; ++axis) {
        uint64_t dimension;
        sev_tensor_abort_if(!sev_parse_u64(&cursor, object_end, &dimension) || dimension > INT64_MAX);
        shape[axis] = (int64_t)dimension;
        cursor = sev_skip_json_space(cursor, object_end);
        if (cursor < object_end && *cursor == ',') ++cursor;
    }
    uint64_t relative_start;
    uint64_t relative_end;
    sev_tensor_abort_if(!sev_safetensor_offsets(store, name, &relative_start, &relative_end));
    uint64_t data_start = 8 + store->header_length;
    sev_tensor_abort_if(data_start + relative_end > store->length);
    sev_tensor *tensor = sev_tensor_new(rank, shape, storage_dtype);
    free(shape);
    size_t byte_width = storage_dtype == 4 || storage_dtype == 9 || storage_dtype == 16
        ? 16
        : storage_dtype == 17
            ? 10
        : storage_dtype == 3 || storage_dtype == 8 || storage_dtype == 15
            ? 8
            : storage_dtype == 2 || storage_dtype == 7 || storage_dtype == 14
                ? 4
                : storage_dtype == 1 || storage_dtype == 6 || storage_dtype == 12 || storage_dtype == 13
                    ? 2
                    : 1;
    sev_tensor_abort_if(relative_end - relative_start != tensor->count * byte_width);
    const uint8_t *data = store->bytes + data_start + relative_start;
    free(tensor->values);
    tensor->values = NULL;
    tensor->owns_values = 0;
    tensor->mapped_values = data;
    tensor->mapped_byte_width = byte_width;
    sev_storage_view_abi *view = calloc(1, sizeof(*view));
    sev_tensor_abort_if(view == NULL);
    view->magic = SEV_STORAGE_VIEW_ABI_MAGIC;
    view->abi_version = SEV_STORAGE_VIEW_ABI_VERSION;
    view->byte_size = (uint32_t)sizeof(*view);
    view->flags = SEV_STORAGE_VIEW_READ_ONLY | SEV_STORAGE_VIEW_CONTIGUOUS;
    view->data = data;
    view->byte_length = relative_end - relative_start;
    view->rank = rank;
    view->dimensions = tensor->shape;
    view->strides = tensor->strides;
    view->offset = tensor->offset;
    view->element.abi_version = SEV_STORAGE_VIEW_ABI_VERSION;
    view->element.byte_size = (uint32_t)sizeof(view->element);
    view->element.bits = sev_tensor_dtype_bits(storage_dtype);
    view->element.kind = sev_tensor_dtype_signed(storage_dtype)
        ? SEV_STORAGE_ELEMENT_SIGNED_INTEGER
        : sev_tensor_dtype_unsigned(storage_dtype)
            ? SEV_STORAGE_ELEMENT_UNSIGNED_INTEGER
            : SEV_STORAGE_ELEMENT_FLOAT;
    view->element.float_format = storage_dtype == 13
        ? SEV_STORAGE_FLOAT_BRAIN
        : storage_dtype == 10
            ? SEV_STORAGE_FLOAT8_E4M3_FN
            : storage_dtype == 11
                ? SEV_STORAGE_FLOAT8_E5M2
                : view->element.kind == SEV_STORAGE_ELEMENT_FLOAT
                    ? SEV_STORAGE_FLOAT_IEEE
                    : SEV_STORAGE_FLOAT_NONE;
    view->owner = tensor;
    return view;
}

int32_t __sev_storage_view_validate(
    const void *storage,
    uint32_t expected_kind,
    uint32_t expected_bits,
    uint32_t expected_float_format,
    uint64_t expected_rank
) {
    const sev_storage_view_abi *view = storage;
    if (view == NULL
        || view->magic != SEV_STORAGE_VIEW_ABI_MAGIC
        || view->abi_version != SEV_STORAGE_VIEW_ABI_VERSION
        || view->byte_size < sizeof(*view)
        || view->element.abi_version != SEV_STORAGE_VIEW_ABI_VERSION
        || view->element.byte_size < sizeof(view->element)) {
        return 0;
    }
    return view->element.kind == expected_kind
        && view->element.bits == expected_bits
        && view->element.float_format == expected_float_format
        && view->rank == expected_rank;
}

void *__sev_storage_view_data(const void *storage) {
    const sev_storage_view_abi *view = storage;
    return view == NULL ? NULL : (void *)view->data;
}

int64_t __sev_storage_view_dimension(const void *storage, uint64_t axis) {
    const sev_storage_view_abi *view = storage;
    sev_tensor_abort_if(view == NULL || axis >= view->rank);
    return view->dimensions[axis];
}

int64_t __sev_storage_view_stride(const void *storage, uint64_t axis) {
    const sev_storage_view_abi *view = storage;
    sev_tensor_abort_if(view == NULL || axis >= view->rank);
    return view->strides[axis];
}

int64_t __sev_storage_view_offset(const void *storage) {
    const sev_storage_view_abi *view = storage;
    sev_tensor_abort_if(view == NULL);
    return view->offset;
}

static int32_t sev_tensor_test_dtype(uint32_t kind, uint32_t bits, uint32_t format) {
    if (kind == SEV_STORAGE_ELEMENT_SIGNED_INTEGER && format == SEV_STORAGE_FLOAT_NONE) {
        if (bits == 8) return 0;
        if (bits == 16) return 1;
        if (bits == 32) return 2;
        if (bits == 64) return 3;
        if (bits == 128) return 4;
    }
    if (kind == SEV_STORAGE_ELEMENT_UNSIGNED_INTEGER && format == SEV_STORAGE_FLOAT_NONE) {
        if (bits == 8) return 5;
        if (bits == 16) return 6;
        if (bits == 32) return 7;
        if (bits == 64) return 8;
        if (bits == 128) return 9;
    }
    if (kind == SEV_STORAGE_ELEMENT_FLOAT) {
        if (bits == 8 && format == SEV_STORAGE_FLOAT8_E4M3_FN) return 10;
        if (bits == 8 && format == SEV_STORAGE_FLOAT8_E5M2) return 11;
        if (bits == 16 && format == SEV_STORAGE_FLOAT_IEEE) return 12;
        if (bits == 16 && format == SEV_STORAGE_FLOAT_BRAIN) return 13;
        if (bits == 32 && format == SEV_STORAGE_FLOAT_IEEE) return 14;
        if (bits == 64 && format == SEV_STORAGE_FLOAT_IEEE) return 15;
        if (bits == 128 && format == SEV_STORAGE_FLOAT_IEEE) return 16;
        if (bits == 80 && format == SEV_STORAGE_FLOAT_IEEE) return 17;
    }
    return -1;
}

/*
 * One data-driven descriptor fixture for the source-level conformance test.
 * It deliberately uses the production StorageView ABI and never creates
 * dtype- or rank-named symbols.
 */
void *__sev_tensor_test_storage_view(int32_t kind, int32_t bits, int32_t format) {
    int32_t dtype = sev_tensor_test_dtype((uint32_t)kind, (uint32_t)bits, (uint32_t)format);
    sev_tensor_abort_if(dtype < 0);
    const int64_t shape[] = {2, 2};
    sev_tensor *tensor = sev_tensor_new(2, shape, dtype);
    size_t byte_width = ((size_t)bits + 7) / 8;
    uint8_t *data = calloc(tensor->count == 0 ? 1 : tensor->count, byte_width);
    sev_tensor_abort_if(data == NULL);
    free(tensor->values);
    tensor->values = NULL;
    tensor->mapped_values = data;
    tensor->mapped_byte_width = byte_width;
    tensor->owns_values = 0;

    sev_storage_view_abi *view = calloc(1, sizeof(*view));
    sev_tensor_abort_if(view == NULL);
    view->magic = SEV_STORAGE_VIEW_ABI_MAGIC;
    view->abi_version = SEV_STORAGE_VIEW_ABI_VERSION;
    view->byte_size = (uint32_t)sizeof(*view);
    view->flags = SEV_STORAGE_VIEW_CONTIGUOUS;
    view->data = data;
    view->byte_length = tensor->count * byte_width;
    view->rank = tensor->rank;
    view->dimensions = tensor->shape;
    view->strides = tensor->strides;
    view->offset = tensor->offset;
    view->element.abi_version = SEV_STORAGE_VIEW_ABI_VERSION;
    view->element.byte_size = (uint32_t)sizeof(view->element);
    view->element.kind = (uint32_t)kind;
    view->element.bits = (uint32_t)bits;
    view->element.float_format = (uint32_t)format;
    view->owner = tensor;
    return view;
}

static uintptr_t sev_tensor_f64_bits(double value) {
    uint64_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return (uintptr_t)bits;
}

static double sev_tensor_f64_from_bits(uintptr_t bits) {
    uint64_t raw = (uint64_t)bits;
    double value = 0.0;
    memcpy(&value, &raw, sizeof(value));
    return value;
}

static unsigned sev_tensor_dtype_bits(int32_t dtype) {
    static const unsigned widths[] = {
        8, 16, 32, 64, 128, 8, 16, 32, 64, 128, 8, 8, 16, 16, 32, 64, 128, 80
    };
    sev_tensor_abort_if(dtype < 0 || dtype >= 18);
    return widths[dtype];
}

static _Bool sev_tensor_dtype_signed(int32_t dtype) {
    return dtype >= 0 && dtype <= 4;
}

static _Bool sev_tensor_dtype_unsigned(int32_t dtype) {
    return dtype >= 5 && dtype <= 9;
}

static int32_t sev_tensor_accumulation_dtype(int32_t dtype) {
    if (dtype >= 0 && dtype <= 2) return 3;
    if (dtype >= 5 && dtype <= 7) return 8;
    if (dtype >= 10 && dtype <= 13) return 14;
    return dtype;
}

static unsigned __int128 sev_tensor_mask(unsigned bits) {
    return bits == 128 ? ~(unsigned __int128)0 : ((unsigned __int128)1 << bits) - 1;
}

static __int128 sev_tensor_signed(sev_tensor_cell cell, int32_t dtype) {
    unsigned bits = sev_tensor_dtype_bits(dtype);
    unsigned __int128 raw = cell.bits & sev_tensor_mask(bits);
    if (bits < 128 && ((raw >> (bits - 1)) & 1) != 0) {
        raw |= ~sev_tensor_mask(bits);
    }
    return (__int128)raw;
}

static unsigned __int128 sev_tensor_unsigned(sev_tensor_cell cell, int32_t dtype) {
    return cell.bits & sev_tensor_mask(sev_tensor_dtype_bits(dtype));
}

static long double sev_decode_fp8(uint8_t raw, int32_t dtype) {
    if (dtype == 11) return sev_decode_binary_float(&raw, 5, 2, 15);
    _Bool negative = (raw & 0x80) != 0;
    unsigned exponent = (raw >> 3) & 0x0f;
    unsigned fraction = raw & 0x07;
    if (exponent == 0x0f && fraction == 0x07) return NAN;
    long double significand = exponent == 0
        ? (long double)fraction / 8.0L
        : 1.0L + (long double)fraction / 8.0L;
    int power = exponent == 0 ? -6 : (int)exponent - 7;
    long double value = ldexpl(significand, power);
    return negative ? -value : value;
}

static __float128 sev_tensor_float(sev_tensor_cell cell, int32_t dtype) {
    uint8_t bytes[16] = {0};
    unsigned width = sev_tensor_dtype_bits(dtype) / 8;
    for (unsigned index = 0; index < width; ++index) {
        bytes[index] = (uint8_t)(cell.bits >> (index * 8));
    }
    switch (dtype) {
        case 10:
        case 11: return (__float128)sev_decode_fp8(bytes[0], dtype);
        case 12: return (__float128)sev_decode_binary_float(bytes, 5, 10, 15);
        case 13: return (__float128)sev_decode_binary_float(bytes, 8, 7, 127);
        case 14: {
            uint32_t raw = (uint32_t)cell.bits;
            float value;
            memcpy(&value, &raw, sizeof(value));
            return (__float128)value;
        }
        case 15: {
            uint64_t raw = (uint64_t)cell.bits;
            double value;
            memcpy(&value, &raw, sizeof(value));
            return (__float128)value;
        }
        case 16: {
            __float128 value;
            memcpy(&value, &cell.bits, sizeof(value));
            return value;
        }
        case 17: {
            long double value = 0.0L;
            memcpy(&value, &cell.bits, 10);
            return (__float128)value;
        }
        default: abort();
    }
}

static uint16_t sev_f32_to_f16(float value) {
    uint32_t raw;
    memcpy(&raw, &value, sizeof(raw));
    uint32_t sign = (raw >> 16) & 0x8000;
    uint32_t exponent = (raw >> 23) & 0xff;
    uint32_t fraction = raw & 0x7fffff;
    if (exponent == 0xff) {
        return (uint16_t)(sign | (fraction == 0 ? 0x7c00 : 0x7e00));
    }
    int32_t half_exponent = (int32_t)exponent - 127 + 15;
    if (half_exponent >= 31) return (uint16_t)(sign | 0x7c00);
    if (half_exponent <= 0) {
        if (half_exponent < -10) return (uint16_t)sign;
        fraction |= 0x800000;
        unsigned shift = (unsigned)(14 - half_exponent);
        uint32_t result = fraction >> shift;
        uint32_t remainder = fraction & (((uint32_t)1 << shift) - 1);
        uint32_t halfway = (uint32_t)1 << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (result & 1))) ++result;
        return (uint16_t)(sign | result);
    }
    uint32_t result = (uint32_t)half_exponent << 10 | fraction >> 13;
    uint32_t remainder = fraction & 0x1fff;
    if (remainder > 0x1000 || (remainder == 0x1000 && (result & 1))) ++result;
    return (uint16_t)(sign | result);
}

static uint16_t sev_f32_to_bf16(float value) {
    uint32_t raw;
    memcpy(&raw, &value, sizeof(raw));
    if ((raw & 0x7f800000) == 0x7f800000 && (raw & 0x007fffff) != 0) {
        return (uint16_t)((raw >> 16) | 0x0040);
    }
    raw += 0x7fff + ((raw >> 16) & 1);
    return (uint16_t)(raw >> 16);
}

static uint8_t sev_float_to_fp8(__float128 value, int32_t dtype) {
    long double requested = (long double)value;
    if (isnan(requested)) return dtype == 10 ? 0x7f : 0x7d;
    if (isinf(requested)) {
        uint8_t sign = signbit(requested) ? 0x80 : 0;
        return dtype == 10 ? (uint8_t)(sign | 0x7e) : (uint8_t)(sign | 0x7c);
    }
    uint8_t selected = 0;
    long double distance = INFINITY;
    for (unsigned raw = 0; raw <= UINT8_MAX; ++raw) {
        long double candidate = sev_decode_fp8((uint8_t)raw, dtype);
        if (isnan(candidate)) continue;
        long double difference = fabsl(candidate - requested);
        if (difference < distance || (difference == distance && (raw & 1) == 0)) {
            distance = difference;
            selected = (uint8_t)raw;
        }
    }
    return selected;
}

float __sev_float_decode(uint8_t value, int32_t format) {
    sev_tensor_abort_if(format != 3 && format != 4);
    return (float)sev_decode_fp8(value, format == 3 ? 10 : 11);
}

uint8_t __sev_float_encode(float value, int32_t format) {
    sev_tensor_abort_if(format != 3 && format != 4);
    return sev_float_to_fp8((__float128)value, format == 3 ? 10 : 11);
}

static sev_tensor_cell sev_tensor_from_signed(__int128 value, int32_t dtype) {
    return (sev_tensor_cell){(unsigned __int128)value & sev_tensor_mask(sev_tensor_dtype_bits(dtype))};
}

static sev_tensor_cell sev_tensor_from_unsigned(unsigned __int128 value, int32_t dtype) {
    return (sev_tensor_cell){value & sev_tensor_mask(sev_tensor_dtype_bits(dtype))};
}

static sev_tensor_cell sev_tensor_from_float(__float128 value, int32_t dtype) {
    sev_tensor_cell result = {0};
    if (dtype == 10 || dtype == 11) {
        result.bits = sev_float_to_fp8(value, dtype);
    } else if (dtype == 12) {
        result.bits = sev_f32_to_f16((float)value);
    } else if (dtype == 13) {
        result.bits = sev_f32_to_bf16((float)value);
    } else if (dtype == 14) {
        float narrowed = (float)value;
        uint32_t raw;
        memcpy(&raw, &narrowed, sizeof(raw));
        result.bits = raw;
    } else if (dtype == 15) {
        double narrowed = (double)value;
        uint64_t raw;
        memcpy(&raw, &narrowed, sizeof(raw));
        result.bits = raw;
    } else if (dtype == 16) {
        memcpy(&result.bits, &value, sizeof(value));
    } else if (dtype == 17) {
        long double narrowed = (long double)value;
        memcpy(&result.bits, &narrowed, 10);
    } else {
        abort();
    }
    return result;
}

static sev_tensor_cell sev_tensor_convert_cell(
    sev_tensor_cell value,
    int32_t source,
    int32_t target
) {
    if (source == target) return value;
    if (sev_tensor_dtype_signed(target)) {
        __int128 converted = sev_tensor_dtype_signed(source)
            ? sev_tensor_signed(value, source)
            : sev_tensor_dtype_unsigned(source)
                ? (__int128)sev_tensor_unsigned(value, source)
                : (__int128)sev_tensor_float(value, source);
        return sev_tensor_from_signed(converted, target);
    }
    if (sev_tensor_dtype_unsigned(target)) {
        unsigned __int128 converted = sev_tensor_dtype_signed(source)
            ? (unsigned __int128)sev_tensor_signed(value, source)
            : sev_tensor_dtype_unsigned(source)
                ? sev_tensor_unsigned(value, source)
                : (unsigned __int128)sev_tensor_float(value, source);
        return sev_tensor_from_unsigned(converted, target);
    }
    __float128 converted = sev_tensor_dtype_signed(source)
        ? (__float128)sev_tensor_signed(value, source)
        : sev_tensor_dtype_unsigned(source)
            ? (__float128)sev_tensor_unsigned(value, source)
            : sev_tensor_float(value, source);
    return sev_tensor_from_float(converted, target);
}

static double sev_tensor_as_f64(sev_tensor_cell value, int32_t dtype) {
    if (sev_tensor_dtype_signed(dtype)) return (double)sev_tensor_signed(value, dtype);
    if (sev_tensor_dtype_unsigned(dtype)) return (double)sev_tensor_unsigned(value, dtype);
    return (double)sev_tensor_float(value, dtype);
}

static size_t sev_tensor_element_count(size_t rank, const int64_t *shape) {
    size_t count = 1;
    for (size_t axis = 0; axis < rank; ++axis) {
        sev_tensor_abort_if(shape[axis] < 0);
        sev_tensor_abort_if(shape[axis] != 0 && count > SIZE_MAX / (size_t)shape[axis]);
        count *= (size_t)shape[axis];
    }
    return count;
}

static int64_t *sev_tensor_contiguous_strides(size_t rank, const int64_t *shape) {
    int64_t *strides = calloc(rank == 0 ? 1 : rank, sizeof(*strides));
    sev_tensor_abort_if(strides == NULL);
    int64_t stride = 1;
    for (size_t axis = rank; axis > 0; --axis) {
        strides[axis - 1] = stride;
        sev_tensor_abort_if(shape[axis - 1] != 0 && stride > INT64_MAX / shape[axis - 1]);
        stride *= shape[axis - 1];
    }
    return strides;
}

static sev_tensor *sev_tensor_new(size_t rank, const int64_t *shape, int32_t dtype) {
    sev_tensor *tensor = calloc(1, sizeof(*tensor));
    sev_tensor_abort_if(tensor == NULL);
    tensor->rank = rank;
    tensor->dtype = dtype;
    tensor->shape = calloc(rank == 0 ? 1 : rank, sizeof(*tensor->shape));
    sev_tensor_abort_if(tensor->shape == NULL);
    memcpy(tensor->shape, shape, rank * sizeof(*shape));
    tensor->strides = sev_tensor_contiguous_strides(rank, shape);
    tensor->count = sev_tensor_element_count(rank, shape);
    tensor->values = calloc(tensor->count == 0 ? 1 : tensor->count, sizeof(*tensor->values));
    sev_tensor_abort_if(tensor->values == NULL);
    tensor->owns_values = 1;
    return tensor;
}

static sev_tensor *sev_tensor_get(void *value) {
    sev_tensor_abort_if(value == NULL);
    sev_storage_view_abi *view = value;
    if (view->magic == SEV_STORAGE_VIEW_ABI_MAGIC) {
        sev_tensor_abort_if(
            view->abi_version != SEV_STORAGE_VIEW_ABI_VERSION
                || view->byte_size < sizeof(*view)
                || view->owner == NULL
        );
        return view->owner;
    }
    return value;
}

static void *sev_tensor_wrap(sev_tensor *tensor) {
    return tensor;
}

static size_t sev_tensor_physical_index(const sev_tensor *tensor, size_t logical) {
    int64_t physical = tensor->offset;
    for (size_t axis = tensor->rank; axis > 0; --axis) {
        size_t dimension = (size_t)tensor->shape[axis - 1];
        size_t coordinate = dimension == 0 ? 0 : logical % dimension;
        logical = dimension == 0 ? 0 : logical / dimension;
        physical += (int64_t)coordinate * tensor->strides[axis - 1];
    }
    sev_tensor_abort_if(physical < 0);
    return (size_t)physical;
}

static sev_tensor_cell sev_tensor_value(const sev_tensor *tensor, size_t physical) {
    if (tensor->mapped_values != NULL) {
        return (sev_tensor_cell){
            sev_read_u128(
                tensor->mapped_values + physical * tensor->mapped_byte_width,
                tensor->mapped_byte_width
            )
        };
    }
    return tensor->values[physical];
}

void *__sev_tensor_from_elements(void *values_storage, void *shape_storage) {
    sev_list *values = values_storage;
    sev_list *shape = shape_storage;
    int64_t *dimensions = calloc(shape->length == 0 ? 1 : shape->length, sizeof(*dimensions));
    sev_tensor_abort_if(dimensions == NULL);
    for (size_t axis = 0; axis < shape->length; ++axis) {
        dimensions[axis] = (int64_t)shape->values[axis];
    }
    sev_tensor *tensor = sev_tensor_new(shape->length, dimensions, 15);
    free(dimensions);
    sev_tensor_abort_if(tensor->count != values->length);
    for (size_t index = 0; index < tensor->count; ++index) {
        tensor->values[index] = sev_tensor_from_float(
            (__float128)sev_tensor_f64_from_bits(values->values[index]),
            15
        );
    }
    return sev_tensor_wrap(tensor);
}

void *__sev_tensor_shape(void *value) {
    sev_tensor *tensor = sev_tensor_get(value);
    sev_list *result = sev_tensor_list();
    for (size_t axis = 0; axis < tensor->rank; ++axis) {
        sev_tensor_list_push(result, (uintptr_t)tensor->shape[axis]);
    }
    return result;
}

void *__sev_tensor_strides(void *value) {
    sev_tensor *tensor = sev_tensor_get(value);
    sev_list *result = sev_tensor_list();
    for (size_t axis = 0; axis < tensor->rank; ++axis) {
        sev_tensor_list_push(result, (uintptr_t)tensor->strides[axis]);
    }
    return result;
}

void *__sev_tensor_values(void *value) {
    sev_tensor *tensor = sev_tensor_get(value);
    sev_list *result = sev_tensor_list();
    for (size_t index = 0; index < tensor->count; ++index) {
        double element = sev_tensor_as_f64(
            sev_tensor_value(tensor, sev_tensor_physical_index(tensor, index)),
            tensor->dtype
        );
        sev_tensor_list_push(result, sev_tensor_f64_bits(element));
    }
    return result;
}

void *__sev_tensor_materialize(void *value) {
    sev_tensor *source = sev_tensor_get(value);
    sev_tensor *result = sev_tensor_new(source->rank, source->shape, source->dtype);
    for (size_t index = 0; index < source->count; ++index) {
        result->values[index] = sev_tensor_value(
            source,
            sev_tensor_physical_index(source, index)
        );
    }
    return sev_tensor_wrap(result);
}

void *__sev_tensor_convert(void *value, int32_t dtype) {
    void *materialized = __sev_tensor_materialize(value);
    sev_tensor *result = sev_tensor_get(materialized);
    int32_t source = result->dtype;
    result->dtype = dtype;
    for (size_t index = 0; index < result->count; ++index) {
        result->values[index] = sev_tensor_convert_cell(result->values[index], source, dtype);
    }
    return materialized;
}

static void sev_tensor_broadcast_shape(
    const sev_tensor *left,
    const sev_tensor *right,
    size_t *rank,
    int64_t **shape
) {
    *rank = left->rank > right->rank ? left->rank : right->rank;
    *shape = calloc(*rank == 0 ? 1 : *rank, sizeof(**shape));
    sev_tensor_abort_if(*shape == NULL);
    for (size_t offset = 0; offset < *rank; ++offset) {
        int64_t l = offset < left->rank ? left->shape[left->rank - 1 - offset] : 1;
        int64_t r = offset < right->rank ? right->shape[right->rank - 1 - offset] : 1;
        sev_tensor_abort_if(l != r && l != 1 && r != 1);
        (*shape)[*rank - 1 - offset] = l > r ? l : r;
    }
}

static size_t sev_tensor_broadcast_index(
    const sev_tensor *source,
    size_t result_rank,
    const int64_t *result_shape,
    size_t logical
) {
    int64_t physical = source->offset;
    for (size_t offset = 0; offset < result_rank; ++offset) {
        size_t axis = result_rank - 1 - offset;
        size_t dimension = (size_t)result_shape[axis];
        size_t coordinate = dimension == 0 ? 0 : logical % dimension;
        logical = dimension == 0 ? 0 : logical / dimension;
        if (offset < source->rank) {
            size_t source_axis = source->rank - 1 - offset;
            if (source->shape[source_axis] != 1) {
                physical += (int64_t)coordinate * source->strides[source_axis];
            }
        }
    }
    sev_tensor_abort_if(physical < 0);
    return (size_t)physical;
}

static sev_tensor_cell sev_tensor_binary_cell(
    sev_tensor_cell left,
    sev_tensor_cell right,
    int32_t dtype,
    char operation
) {
    if (sev_tensor_dtype_signed(dtype)) {
        unsigned bits = sev_tensor_dtype_bits(dtype);
        unsigned __int128 mask = sev_tensor_mask(bits);
        unsigned __int128 l = left.bits & mask;
        unsigned __int128 r = right.bits & mask;
        if (operation == '+') return (sev_tensor_cell){(l + r) & mask};
        if (operation == '-') return (sev_tensor_cell){(l - r) & mask};
        if (operation == '*') return (sev_tensor_cell){(l * r) & mask};
        __int128 signed_left = sev_tensor_signed(left, dtype);
        __int128 signed_right = sev_tensor_signed(right, dtype);
        sev_tensor_abort_if(signed_right == 0);
        unsigned __int128 minimum = (unsigned __int128)1 << (bits - 1);
        if (l == minimum && signed_right == -1) return (sev_tensor_cell){minimum};
        return sev_tensor_from_signed(signed_left / signed_right, dtype);
    }
    if (sev_tensor_dtype_unsigned(dtype)) {
        unsigned __int128 mask = sev_tensor_mask(sev_tensor_dtype_bits(dtype));
        unsigned __int128 l = left.bits & mask;
        unsigned __int128 r = right.bits & mask;
        if (operation == '+') return (sev_tensor_cell){(l + r) & mask};
        if (operation == '-') return (sev_tensor_cell){(l - r) & mask};
        if (operation == '*') return (sev_tensor_cell){(l * r) & mask};
        sev_tensor_abort_if(r == 0);
        return sev_tensor_from_unsigned(l / r, dtype);
    }
    __float128 l = sev_tensor_float(left, dtype);
    __float128 r = sev_tensor_float(right, dtype);
    __float128 value = operation == '+' ? l + r
        : operation == '-' ? l - r
        : operation == '*' ? l * r
        : l / r;
    return sev_tensor_from_float(value, dtype);
}

static void *sev_tensor_binary(
    void *left_value,
    void *right_value,
    char operation
) {
    sev_tensor *left = sev_tensor_get(left_value);
    sev_tensor *right = sev_tensor_get(right_value);
    sev_tensor_abort_if(left->dtype != right->dtype);
    size_t rank = 0;
    int64_t *shape = NULL;
    sev_tensor_broadcast_shape(left, right, &rank, &shape);
    sev_tensor *result = sev_tensor_new(rank, shape, left->dtype);
    free(shape);
    for (size_t index = 0; index < result->count; ++index) {
        sev_tensor_cell l = sev_tensor_value(
            left,
            sev_tensor_broadcast_index(left, rank, result->shape, index)
        );
        sev_tensor_cell r = sev_tensor_value(
            right,
            sev_tensor_broadcast_index(right, rank, result->shape, index)
        );
        result->values[index] = sev_tensor_binary_cell(l, r, result->dtype, operation);
    }
    return sev_tensor_wrap(result);
}

void *__sev_tensor_add(void *left, void *right) {
    return sev_tensor_binary(left, right, '+');
}

void *__sev_tensor_subtract(void *left, void *right) {
    return sev_tensor_binary(left, right, '-');
}

void *__sev_tensor_multiply(void *left, void *right) {
    return sev_tensor_binary(left, right, '*');
}

void *__sev_tensor_divide(void *left, void *right) {
    return sev_tensor_binary(left, right, '/');
}

void *__sev_tensor_sum(void *value) {
    sev_tensor *source = sev_tensor_get(value);
    int64_t shape = 1;
    sev_tensor *result = sev_tensor_new(1, &shape, source->dtype);
    int32_t accumulation_dtype = sev_tensor_accumulation_dtype(source->dtype);
    sev_tensor_cell total = {0};
    for (size_t index = 0; index < source->count; ++index) {
        sev_tensor_cell element = sev_tensor_convert_cell(
            sev_tensor_value(source, sev_tensor_physical_index(source, index)),
            source->dtype,
            accumulation_dtype
        );
        total = sev_tensor_binary_cell(
            total,
            element,
            accumulation_dtype,
            '+'
        );
    }
    result->values[0] = sev_tensor_convert_cell(total, accumulation_dtype, source->dtype);
    return sev_tensor_wrap(result);
}

void *__sev_tensor_sum_axis(void *value, int64_t requested_axis) {
    sev_tensor *source = sev_tensor_get(value);
    sev_tensor_abort_if(source->rank == 0);
    int64_t normalized_axis = requested_axis < 0
        ? requested_axis + (int64_t)source->rank
        : requested_axis;
    sev_tensor_abort_if(normalized_axis < 0 || normalized_axis >= (int64_t)source->rank);
    size_t axis = (size_t)normalized_axis;
    size_t rank = source->rank - 1;
    int64_t *shape = calloc(rank == 0 ? 1 : rank, sizeof(*shape));
    sev_tensor_abort_if(shape == NULL);
    for (size_t source_axis = 0, result_axis = 0; source_axis < source->rank; ++source_axis) {
        if (source_axis != axis) shape[result_axis++] = source->shape[source_axis];
    }
    sev_tensor *result = sev_tensor_new(rank, shape, source->dtype);
    free(shape);
    int32_t accumulation_dtype = sev_tensor_accumulation_dtype(source->dtype);
    for (size_t output = 0; output < result->count; ++output) {
        size_t coordinates = output;
        int64_t base = source->offset;
        for (size_t result_axis = rank; result_axis > 0; --result_axis) {
            size_t current = result_axis - 1;
            size_t dimension = (size_t)result->shape[current];
            size_t coordinate = dimension == 0 ? 0 : coordinates % dimension;
            coordinates = dimension == 0 ? 0 : coordinates / dimension;
            size_t source_axis = current < axis ? current : current + 1;
            base += (int64_t)coordinate * source->strides[source_axis];
        }
        sev_tensor_cell total = {0};
        for (int64_t coordinate = 0; coordinate < source->shape[axis]; ++coordinate) {
            int64_t physical = base + coordinate * source->strides[axis];
            sev_tensor_abort_if(physical < 0);
            sev_tensor_cell element = sev_tensor_convert_cell(
                sev_tensor_value(source, (size_t)physical),
                source->dtype,
                accumulation_dtype
            );
            total = sev_tensor_binary_cell(total, element, accumulation_dtype, '+');
        }
        result->values[output] = sev_tensor_convert_cell(
            total,
            accumulation_dtype,
            source->dtype
        );
    }
    return sev_tensor_wrap(result);
}

void *__sev_tensor_matmul(void *left_value, void *right_value) {
    sev_tensor *left = sev_tensor_get(left_value);
    sev_tensor *right = sev_tensor_get(right_value);
    sev_tensor_abort_if(left->rank < 2 || right->rank < 2);
    sev_tensor_abort_if(
        left->shape[left->rank - 1] != right->shape[right->rank - 2]
        || left->dtype != right->dtype
    );
    size_t batch_rank = (left->rank - 2) > (right->rank - 2)
        ? left->rank - 2
        : right->rank - 2;
    size_t rank = batch_rank + 2;
    int64_t *shape = calloc(rank, sizeof(*shape));
    sev_tensor_abort_if(shape == NULL);
    for (size_t offset = 0; offset < batch_rank; ++offset) {
        int64_t l = offset < left->rank - 2
            ? left->shape[left->rank - 3 - offset]
            : 1;
        int64_t r = offset < right->rank - 2
            ? right->shape[right->rank - 3 - offset]
            : 1;
        sev_tensor_abort_if(l != r && l != 1 && r != 1);
        shape[batch_rank - 1 - offset] = l > r ? l : r;
    }
    shape[rank - 2] = left->shape[left->rank - 2];
    shape[rank - 1] = right->shape[right->rank - 1];
    sev_tensor *result = sev_tensor_new(rank, shape, left->dtype);
    free(shape);
    int32_t accumulation_dtype = sev_tensor_accumulation_dtype(left->dtype);
    size_t batch_count = 1;
    for (size_t axis = 0; axis < batch_rank; ++axis) {
        batch_count *= (size_t)result->shape[axis];
    }
    int64_t rows = result->shape[rank - 2];
    int64_t columns = result->shape[rank - 1];
    int64_t inner_size = left->shape[left->rank - 1];
    for (size_t batch = 0; batch < batch_count; ++batch) {
        int64_t left_base = left->offset;
        int64_t right_base = right->offset;
        size_t remaining = batch;
        for (size_t axis = batch_rank; axis > 0; --axis) {
            size_t result_axis = axis - 1;
            size_t dimension = (size_t)result->shape[result_axis];
            size_t coordinate = dimension == 0 ? 0 : remaining % dimension;
            remaining = dimension == 0 ? 0 : remaining / dimension;
            if (result_axis + 2 >= batch_rank + 2 - left->rank
                && left->rank > 2) {
                size_t source_axis = result_axis + (left->rank - 2) - batch_rank;
                if (source_axis < left->rank - 2 && left->shape[source_axis] != 1) {
                    left_base += (int64_t)coordinate * left->strides[source_axis];
                }
            }
            if (result_axis + 2 >= batch_rank + 2 - right->rank
                && right->rank > 2) {
                size_t source_axis = result_axis + (right->rank - 2) - batch_rank;
                if (source_axis < right->rank - 2 && right->shape[source_axis] != 1) {
                    right_base += (int64_t)coordinate * right->strides[source_axis];
                }
            }
        }
        for (int64_t row = 0; row < rows; ++row) {
            for (int64_t column = 0; column < columns; ++column) {
            sev_tensor_cell total = {0};
            for (int64_t inner = 0; inner < inner_size; ++inner) {
                size_t l = (size_t)(
                    left_base
                    + row * left->strides[left->rank - 2]
                    + inner * left->strides[left->rank - 1]
                );
                size_t r = (size_t)(
                    right_base
                    + inner * right->strides[right->rank - 2]
                    + column * right->strides[right->rank - 1]
                );
                sev_tensor_cell left_element = sev_tensor_convert_cell(
                    sev_tensor_value(left, l), left->dtype, accumulation_dtype
                );
                sev_tensor_cell right_element = sev_tensor_convert_cell(
                    sev_tensor_value(right, r), right->dtype, accumulation_dtype
                );
                sev_tensor_cell product = sev_tensor_binary_cell(
                    left_element,
                    right_element,
                    accumulation_dtype,
                    '*'
                );
                total = sev_tensor_binary_cell(total, product, accumulation_dtype, '+');
            }
            size_t output = (batch * (size_t)rows + (size_t)row) * (size_t)columns
                + (size_t)column;
            result->values[output] = sev_tensor_convert_cell(
                total, accumulation_dtype, result->dtype
            );
            }
        }
    }
    return sev_tensor_wrap(result);
}

void *__sev_tensor_transpose(void *value) {
    sev_tensor *source = sev_tensor_get(value);
    sev_tensor_abort_if(source->rank != 2);
    sev_tensor *result = calloc(1, sizeof(*result));
    sev_tensor_abort_if(result == NULL);
    *result = *source;
    result->shape = calloc(2, sizeof(*result->shape));
    result->strides = calloc(2, sizeof(*result->strides));
    sev_tensor_abort_if(result->shape == NULL || result->strides == NULL);
    result->shape[0] = source->shape[1];
    result->shape[1] = source->shape[0];
    result->strides[0] = source->strides[1];
    result->strides[1] = source->strides[0];
    result->owns_values = 0;
    result->gradient = NULL;
    return sev_tensor_wrap(result);
}

void *__sev_tensor_slice(
    void *value,
    void *starts_storage,
    void *ends_storage,
    void *steps_storage
) {
    sev_tensor *source = sev_tensor_get(value);
    sev_list *starts = starts_storage;
    sev_list *ends = ends_storage;
    sev_list *steps = steps_storage;
    sev_tensor_abort_if(starts->length != source->rank || ends->length != source->rank
        || steps->length != source->rank);
    sev_tensor *result = calloc(1, sizeof(*result));
    sev_tensor_abort_if(result == NULL);
    *result = *source;
    result->shape = calloc(source->rank == 0 ? 1 : source->rank, sizeof(*result->shape));
    result->strides = calloc(source->rank == 0 ? 1 : source->rank, sizeof(*result->strides));
    sev_tensor_abort_if(result->shape == NULL || result->strides == NULL);
    result->count = 1;
    result->owns_values = 0;
    result->gradient = NULL;
    for (size_t axis = 0; axis < source->rank; ++axis) {
        int64_t start = (int64_t)starts->values[axis];
        int64_t end = (int64_t)ends->values[axis];
        int64_t step = (int64_t)steps->values[axis];
        sev_tensor_abort_if(step <= 0 || start < 0 || end < start || end > source->shape[axis]);
        result->offset += start * source->strides[axis];
        result->shape[axis] = (end - start + step - 1) / step;
        result->strides[axis] = source->strides[axis] * step;
        result->count *= (size_t)result->shape[axis];
    }
    return sev_tensor_wrap(result);
}

void *__sev_tensor_reshape(void *value, void *shape_storage) {
    sev_tensor *source = sev_tensor_get(value);
    sev_list *shape = shape_storage;
    int64_t *dimensions = calloc(shape->length == 0 ? 1 : shape->length, sizeof(*dimensions));
    sev_tensor_abort_if(dimensions == NULL);
    int64_t inferred_axis = -1;
    size_t known = 1;
    for (size_t axis = 0; axis < shape->length; ++axis) {
        int64_t dimension = (int64_t)shape->values[axis];
        if (dimension == -1) {
            sev_tensor_abort_if(inferred_axis >= 0);
            inferred_axis = (int64_t)axis;
            dimensions[axis] = 1;
        } else {
            sev_tensor_abort_if(dimension < 0);
            dimensions[axis] = dimension;
            sev_tensor_abort_if(dimension != 0 && known > SIZE_MAX / (size_t)dimension);
            known *= (size_t)dimension;
        }
    }
    if (inferred_axis >= 0) {
        sev_tensor_abort_if(known == 0 || source->count % known != 0);
        dimensions[inferred_axis] = (int64_t)(source->count / known);
    }
    sev_tensor_abort_if(sev_tensor_element_count(shape->length, dimensions) != source->count);
    sev_tensor *materialized = source;
    _Bool contiguous = source->offset == 0;
    int64_t expected = 1;
    for (size_t axis = source->rank; axis > 0; --axis) {
        if (source->strides[axis - 1] != expected) contiguous = 0;
        expected *= source->shape[axis - 1];
    }
    if (!contiguous) {
        materialized = sev_tensor_get(__sev_tensor_materialize(source));
        free(materialized->shape);
        free(materialized->strides);
        materialized->rank = shape->length;
        materialized->shape = dimensions;
        materialized->strides = sev_tensor_contiguous_strides(
            materialized->rank, materialized->shape
        );
        materialized->offset = 0;
        return sev_tensor_wrap(materialized);
    }
    sev_tensor *result = calloc(1, sizeof(*result));
    sev_tensor_abort_if(result == NULL);
    *result = *source;
    result->rank = shape->length;
    result->shape = dimensions;
    result->strides = sev_tensor_contiguous_strides(result->rank, result->shape);
    result->offset = 0;
    result->owns_values = 0;
    result->gradient = NULL;
    return sev_tensor_wrap(result);
}

void *__sev_tensor_permute(void *value, void *axes_storage) {
    sev_tensor *source = sev_tensor_get(value);
    sev_list *axes = axes_storage;
    sev_tensor_abort_if(axes->length != source->rank);
    sev_tensor *result = calloc(1, sizeof(*result));
    sev_tensor_abort_if(result == NULL);
    *result = *source;
    result->shape = calloc(source->rank == 0 ? 1 : source->rank, sizeof(*result->shape));
    result->strides = calloc(source->rank == 0 ? 1 : source->rank, sizeof(*result->strides));
    _Bool *seen = calloc(source->rank == 0 ? 1 : source->rank, sizeof(*seen));
    sev_tensor_abort_if(result->shape == NULL || result->strides == NULL || seen == NULL);
    for (size_t axis = 0; axis < source->rank; ++axis) {
        size_t selected = (size_t)axes->values[axis];
        sev_tensor_abort_if(selected >= source->rank || seen[selected]);
        seen[selected] = 1;
        result->shape[axis] = source->shape[selected];
        result->strides[axis] = source->strides[selected];
    }
    free(seen);
    result->owns_values = 0;
    result->gradient = NULL;
    return sev_tensor_wrap(result);
}

static void *sev_tensor_unary_float(void *value, char operation) {
    sev_tensor *source = sev_tensor_get(value);
    sev_tensor_abort_if(sev_tensor_dtype_signed(source->dtype) || sev_tensor_dtype_unsigned(source->dtype));
    sev_tensor *result = sev_tensor_new(source->rank, source->shape, source->dtype);
    for (size_t index = 0; index < source->count; ++index) {
        __float128 input = sev_tensor_float(
            sev_tensor_value(source, sev_tensor_physical_index(source, index)), source->dtype
        );
        long double narrowed = (long double)input;
        __float128 output = operation == 'r' ? (__float128)(1.0L / sqrtl(narrowed))
            : operation == 'e' ? (__float128)expl(narrowed)
            : operation == 'l' ? (__float128)logl(narrowed)
            : operation == 't' ? (__float128)tanhl(narrowed)
            : input / (1.0Q + (__float128)expl(-narrowed));
        result->values[index] = sev_tensor_from_float(output, source->dtype);
    }
    return sev_tensor_wrap(result);
}

void *__sev_tensor_rsqrt(void *value) { return sev_tensor_unary_float(value, 'r'); }
void *__sev_tensor_exp(void *value) { return sev_tensor_unary_float(value, 'e'); }
void *__sev_tensor_log(void *value) { return sev_tensor_unary_float(value, 'l'); }
void *__sev_tensor_tanh(void *value) { return sev_tensor_unary_float(value, 't'); }
void *__sev_tensor_silu(void *value) { return sev_tensor_unary_float(value, 's'); }

void *__sev_tensor_relu(void *value) {
    sev_tensor *source = sev_tensor_get(value);
    sev_tensor *result = sev_tensor_new(source->rank, source->shape, source->dtype);
    for (size_t index = 0; index < source->count; ++index) {
        sev_tensor_cell current = sev_tensor_value(
            source,
            sev_tensor_physical_index(source, index)
        );
        _Bool positive = sev_tensor_dtype_signed(source->dtype)
            ? sev_tensor_signed(current, source->dtype) > 0
            : sev_tensor_dtype_unsigned(source->dtype)
                ? sev_tensor_unsigned(current, source->dtype) > 0
                : sev_tensor_float(current, source->dtype) > 0.0Q;
        result->values[index] = positive ? current : (sev_tensor_cell){0};
    }
    result->parent = source;
    result->operation = 'r';
    return sev_tensor_wrap(result);
}

void *__sev_tensor_scale(void *value, double scale) {
    sev_tensor *source = sev_tensor_get(value);
    sev_tensor *result = sev_tensor_new(source->rank, source->shape, source->dtype);
    for (size_t index = 0; index < source->count; ++index) {
        sev_tensor_cell current = sev_tensor_value(
            source,
            sev_tensor_physical_index(source, index)
        );
        if (sev_tensor_dtype_signed(source->dtype)) {
            result->values[index] = sev_tensor_from_signed(
                (__int128)((double)sev_tensor_signed(current, source->dtype) * scale),
                source->dtype
            );
        } else if (sev_tensor_dtype_unsigned(source->dtype)) {
            result->values[index] = sev_tensor_from_unsigned(
                (unsigned __int128)((double)sev_tensor_unsigned(current, source->dtype) * scale),
                source->dtype
            );
        } else {
            result->values[index] = sev_tensor_from_float(
                sev_tensor_float(current, source->dtype) * (__float128)scale,
                source->dtype
            );
        }
    }
    return sev_tensor_wrap(result);
}

void *__sev_tensor_add_scalar(void *value, float scalar) {
    sev_tensor *source = sev_tensor_get(value);
    sev_tensor_abort_if(sev_tensor_dtype_signed(source->dtype) || sev_tensor_dtype_unsigned(source->dtype));
    sev_tensor *result = sev_tensor_new(source->rank, source->shape, source->dtype);
    for (size_t index = 0; index < source->count; ++index) {
        __float128 current = sev_tensor_float(
            sev_tensor_value(source, sev_tensor_physical_index(source, index)), source->dtype
        );
        result->values[index] = sev_tensor_from_float(
            current + (__float128)scalar, source->dtype
        );
    }
    return sev_tensor_wrap(result);
}

void *__sev_tensor_layer_norm(void *value, double epsilon) {
    sev_tensor *source = sev_tensor_get(value);
    sev_tensor_abort_if(source->rank == 0 || sev_tensor_dtype_signed(source->dtype)
        || sev_tensor_dtype_unsigned(source->dtype));
    int64_t width = source->shape[source->rank - 1];
    sev_tensor *result = sev_tensor_new(source->rank, source->shape, source->dtype);
    size_t rows = width == 0 ? 0 : source->count / (size_t)width;
    for (size_t row = 0; row < rows; ++row) {
        __float128 mean = 0.0Q;
        for (int64_t column = 0; column < width; ++column) {
            mean += sev_tensor_float(
                sev_tensor_value(
                    source,
                    sev_tensor_physical_index(source, row * (size_t)width + (size_t)column)
                ),
                source->dtype
            );
        }
        mean /= (__float128)width;
        __float128 variance = 0.0Q;
        for (int64_t column = 0; column < width; ++column) {
            __float128 current = sev_tensor_float(
                sev_tensor_value(
                    source,
                    sev_tensor_physical_index(source, row * (size_t)width + (size_t)column)
                ),
                source->dtype
            );
            __float128 centered = current - mean;
            variance += centered * centered;
        }
        variance /= (__float128)width;
        __float128 inverse = (__float128)(1.0L / sqrtl((long double)(variance + epsilon)));
        for (int64_t column = 0; column < width; ++column) {
            size_t logical = row * (size_t)width + (size_t)column;
            __float128 current = sev_tensor_float(
                sev_tensor_value(source, sev_tensor_physical_index(source, logical)), source->dtype
            );
            result->values[logical] = sev_tensor_from_float(
                (current - mean) * inverse, source->dtype
            );
        }
    }
    return sev_tensor_wrap(result);
}

void *__sev_tensor_relu_backward(void *input_value, void *upstream_value) {
    sev_tensor *input = sev_tensor_get(input_value);
    sev_tensor *upstream = sev_tensor_get(upstream_value);
    sev_tensor_abort_if(input->count != upstream->count || input->dtype != upstream->dtype);
    sev_tensor *result = sev_tensor_new(input->rank, input->shape, input->dtype);
    for (size_t index = 0; index < input->count; ++index) {
        sev_tensor_cell current = sev_tensor_value(
            input,
            sev_tensor_physical_index(input, index)
        );
        _Bool positive = sev_tensor_dtype_signed(input->dtype)
            ? sev_tensor_signed(current, input->dtype) > 0
            : sev_tensor_dtype_unsigned(input->dtype)
                ? sev_tensor_unsigned(current, input->dtype) > 0
                : sev_tensor_float(current, input->dtype) > 0.0Q;
        result->values[index] = positive
            ? sev_tensor_value(upstream, sev_tensor_physical_index(upstream, index))
            : (sev_tensor_cell){0};
    }
    return sev_tensor_wrap(result);
}

void *__sev_tensor_softmax_backward(void *output_value, void *upstream_value) {
    sev_tensor *output = sev_tensor_get(output_value);
    sev_tensor *upstream = sev_tensor_get(upstream_value);
    sev_tensor_abort_if(output->rank == 0 || output->count != upstream->count
        || output->dtype != upstream->dtype);
    int64_t width = output->shape[output->rank - 1];
    sev_tensor *result = sev_tensor_new(output->rank, output->shape, output->dtype);
    size_t rows = width == 0 ? 0 : output->count / (size_t)width;
    for (size_t row = 0; row < rows; ++row) {
        __float128 dot = 0.0Q;
        for (int64_t column = 0; column < width; ++column) {
            size_t logical = row * (size_t)width + (size_t)column;
            dot += sev_tensor_float(
                sev_tensor_value(output, sev_tensor_physical_index(output, logical)), output->dtype
            ) * sev_tensor_float(
                sev_tensor_value(upstream, sev_tensor_physical_index(upstream, logical)), upstream->dtype
            );
        }
        for (int64_t column = 0; column < width; ++column) {
            size_t logical = row * (size_t)width + (size_t)column;
            __float128 probability = sev_tensor_float(
                sev_tensor_value(output, sev_tensor_physical_index(output, logical)), output->dtype
            );
            __float128 gradient = sev_tensor_float(
                sev_tensor_value(upstream, sev_tensor_physical_index(upstream, logical)), upstream->dtype
            );
            result->values[logical] = sev_tensor_from_float(
                probability * (gradient - dot), output->dtype
            );
        }
    }
    return sev_tensor_wrap(result);
}

void *__sev_tensor_layer_norm_backward(
    void *input_value,
    void *upstream_value,
    double epsilon
) {
    sev_tensor *input = sev_tensor_get(input_value);
    sev_tensor *upstream = sev_tensor_get(upstream_value);
    sev_tensor_abort_if(input->rank == 0 || input->count != upstream->count
        || input->dtype != upstream->dtype);
    int64_t width = input->shape[input->rank - 1];
    sev_tensor *result = sev_tensor_new(input->rank, input->shape, input->dtype);
    size_t rows = width == 0 ? 0 : input->count / (size_t)width;
    for (size_t row = 0; row < rows; ++row) {
        __float128 mean = 0.0Q;
        for (int64_t column = 0; column < width; ++column) {
            mean += sev_tensor_float(
                sev_tensor_value(
                    input,
                    sev_tensor_physical_index(input, row * (size_t)width + (size_t)column)
                ),
                input->dtype
            );
        }
        mean /= (__float128)width;
        __float128 variance = 0.0Q;
        for (int64_t column = 0; column < width; ++column) {
            __float128 centered = sev_tensor_float(
                sev_tensor_value(
                    input,
                    sev_tensor_physical_index(input, row * (size_t)width + (size_t)column)
                ),
                input->dtype
            ) - mean;
            variance += centered * centered;
        }
        variance /= (__float128)width;
        __float128 inverse = (__float128)(1.0L / sqrtl((long double)(variance + epsilon)));
        __float128 sum_gradient = 0.0Q;
        __float128 sum_gradient_normalized = 0.0Q;
        for (int64_t column = 0; column < width; ++column) {
            size_t logical = row * (size_t)width + (size_t)column;
            __float128 gradient = sev_tensor_float(
                sev_tensor_value(upstream, sev_tensor_physical_index(upstream, logical)), upstream->dtype
            );
            __float128 normalized = (
                sev_tensor_float(
                    sev_tensor_value(input, sev_tensor_physical_index(input, logical)), input->dtype
                ) - mean
            ) * inverse;
            sum_gradient += gradient;
            sum_gradient_normalized += gradient * normalized;
        }
        for (int64_t column = 0; column < width; ++column) {
            size_t logical = row * (size_t)width + (size_t)column;
            __float128 gradient = sev_tensor_float(
                sev_tensor_value(upstream, sev_tensor_physical_index(upstream, logical)), upstream->dtype
            );
            __float128 normalized = (
                sev_tensor_float(
                    sev_tensor_value(input, sev_tensor_physical_index(input, logical)), input->dtype
                ) - mean
            ) * inverse;
            result->values[logical] = sev_tensor_from_float(
                inverse * (
                    (__float128)width * gradient - sum_gradient
                    - normalized * sum_gradient_normalized
                ) / (__float128)width,
                input->dtype
            );
        }
    }
    return sev_tensor_wrap(result);
}

void *__sev_tensor_backward_mse(void *output_value) {
    sev_tensor *output = sev_tensor_get(output_value);
    free(output->gradient);
    output->gradient = calloc(output->count == 0 ? 1 : output->count, sizeof(*output->gradient));
    sev_tensor_abort_if(output->gradient == NULL);
    for (size_t index = 0; index < output->count; ++index) {
        output->gradient[index] = sev_tensor_value(
            output,
            sev_tensor_physical_index(output, index)
        );
    }
    if (output->parent != NULL && output->operation == 'r') {
        sev_tensor *parent = output->parent;
        free(parent->gradient);
        parent->gradient = calloc(parent->count == 0 ? 1 : parent->count, sizeof(*parent->gradient));
        sev_tensor_abort_if(parent->gradient == NULL);
        for (size_t index = 0; index < parent->count; ++index) {
            sev_tensor_cell input = sev_tensor_value(
                parent,
                sev_tensor_physical_index(parent, index)
            );
            _Bool positive = sev_tensor_dtype_signed(parent->dtype)
                ? sev_tensor_signed(input, parent->dtype) > 0
                : sev_tensor_dtype_unsigned(parent->dtype)
                    ? sev_tensor_unsigned(input, parent->dtype) > 0
                    : sev_tensor_float(input, parent->dtype) > 0.0Q;
            parent->gradient[index] = positive ? output->gradient[index] : (sev_tensor_cell){0};
        }
    }
    return output_value;
}

void *__sev_tensor_gradient(void *value) {
    sev_tensor *source = sev_tensor_get(value);
    sev_tensor *result = sev_tensor_new(source->rank, source->shape, source->dtype);
    if (source->gradient != NULL) {
        memcpy(result->values, source->gradient, source->count * sizeof(*source->gradient));
    }
    return sev_tensor_wrap(result);
}

void *__sev_tensor_sgd(void *value, double learning_rate) {
    sev_tensor *source = sev_tensor_get(value);
    sev_tensor *result = sev_tensor_new(source->rank, source->shape, source->dtype);
    for (size_t index = 0; index < source->count; ++index) {
        __float128 current = sev_tensor_float(
            sev_tensor_value(source, sev_tensor_physical_index(source, index)), source->dtype
        );
        __float128 gradient = source->gradient == NULL
            ? 0.0Q
            : sev_tensor_float(source->gradient[index], source->dtype);
        result->values[index] = sev_tensor_from_float(
            current - (__float128)learning_rate * gradient, source->dtype
        );
    }
    return sev_tensor_wrap(result);
}

void *__sev_tensor_mean_last(void *value) {
    sev_tensor *source = sev_tensor_get(value);
    sev_tensor_abort_if(source->rank == 0);
    int64_t *shape = calloc(source->rank, sizeof(*shape));
    sev_tensor_abort_if(shape == NULL);
    memcpy(shape, source->shape, source->rank * sizeof(*shape));
    int64_t width = shape[source->rank - 1];
    shape[source->rank - 1] = 1;
    sev_tensor *result = sev_tensor_new(source->rank, shape, source->dtype);
    free(shape);
    for (size_t outer = 0; outer < result->count; ++outer) {
        __float128 total = 0.0Q;
        for (int64_t column = 0; column < width; ++column) {
            size_t logical = outer * (size_t)width + (size_t)column;
            total += sev_tensor_float(
                sev_tensor_value(source, sev_tensor_physical_index(source, logical)), source->dtype
            );
        }
        result->values[outer] = sev_tensor_from_float(total / (__float128)width, source->dtype);
    }
    return sev_tensor_wrap(result);
}

void *__sev_tensor_softmax_last(void *value) {
    sev_tensor *source = sev_tensor_get(value);
    sev_tensor_abort_if(source->rank == 0);
    int64_t width = source->shape[source->rank - 1];
    sev_tensor *result = sev_tensor_new(source->rank, source->shape, source->dtype);
    size_t rows = width == 0 ? 0 : source->count / (size_t)width;
    for (size_t row = 0; row < rows; ++row) {
        __float128 maximum = -(__float128)HUGE_VALL;
        for (int64_t column = 0; column < width; ++column) {
            size_t logical = row * (size_t)width + (size_t)column;
            __float128 current = sev_tensor_float(
                sev_tensor_value(source, sev_tensor_physical_index(source, logical)), source->dtype
            );
            if (current > maximum) maximum = current;
        }
        __float128 total = 0.0Q;
        for (int64_t column = 0; column < width; ++column) {
            size_t logical = row * (size_t)width + (size_t)column;
            __float128 current = sev_tensor_float(
                sev_tensor_value(source, sev_tensor_physical_index(source, logical)), source->dtype
            );
            __float128 exponent = (__float128)expl((long double)(current - maximum));
            result->values[logical] = sev_tensor_from_float(exponent, source->dtype);
            total += exponent;
        }
        for (int64_t column = 0; column < width; ++column) {
            size_t logical = row * (size_t)width + (size_t)column;
            __float128 exponent = sev_tensor_float(
                sev_tensor_value(result, logical),
                source->dtype
            );
            result->values[logical] = sev_tensor_from_float(exponent / total, source->dtype);
        }
    }
    return sev_tensor_wrap(result);
}

void *__sev_tensor_gather(void *value, void *indices_value) {
    sev_tensor *source = sev_tensor_get(value);
    sev_tensor *indices = sev_tensor_get(indices_value);
    sev_tensor_abort_if(source->rank == 0);
    size_t rank = indices->rank + source->rank - 1;
    int64_t *shape = calloc(rank == 0 ? 1 : rank, sizeof(*shape));
    sev_tensor_abort_if(shape == NULL);
    memcpy(shape, indices->shape, indices->rank * sizeof(*shape));
    memcpy(shape + indices->rank, source->shape + 1, (source->rank - 1) * sizeof(*shape));
    sev_tensor *result = sev_tensor_new(rank, shape, source->dtype);
    free(shape);
    size_t row_width = source->shape[0] == 0 ? 0 : source->count / (size_t)source->shape[0];
    for (size_t index = 0; index < indices->count; ++index) {
        sev_tensor_cell cell = sev_tensor_value(
            indices,
            sev_tensor_physical_index(indices, index)
        );
        int64_t selected = sev_tensor_dtype_signed(indices->dtype)
            ? (int64_t)sev_tensor_signed(cell, indices->dtype)
            : (int64_t)sev_tensor_unsigned(cell, indices->dtype);
        sev_tensor_abort_if(selected < 0 || selected >= source->shape[0]);
        for (size_t column = 0; column < row_width; ++column) {
            size_t source_logical = (size_t)selected * row_width + column;
            result->values[index * row_width + column] = sev_tensor_value(
                source,
                sev_tensor_physical_index(source, source_logical)
            );
        }
    }
    return sev_tensor_wrap(result);
}

void *__sev_tensor_concatenate(void *left_value, void *right_value, void *axis_storage) {
    sev_tensor *left = sev_tensor_get(left_value);
    sev_tensor *right = sev_tensor_get(right_value);
    sev_list *axis_list = axis_storage;
    sev_tensor_abort_if(axis_list->length != 1 || left->rank != right->rank || left->dtype != right->dtype);
    size_t axis = (size_t)axis_list->values[0];
    sev_tensor_abort_if(axis >= left->rank);
    int64_t *shape = calloc(left->rank == 0 ? 1 : left->rank, sizeof(*shape));
    sev_tensor_abort_if(shape == NULL);
    memcpy(shape, left->shape, left->rank * sizeof(*shape));
    for (size_t current = 0; current < left->rank; ++current) {
        if (current != axis) sev_tensor_abort_if(left->shape[current] != right->shape[current]);
    }
    shape[axis] += right->shape[axis];
    sev_tensor *result = sev_tensor_new(left->rank, shape, left->dtype);
    free(shape);
    size_t inner = 1;
    for (size_t current = axis + 1; current < left->rank; ++current) inner *= (size_t)left->shape[current];
    size_t outer = left->count / ((size_t)left->shape[axis] * inner);
    size_t left_block = (size_t)left->shape[axis] * inner;
    size_t right_block = (size_t)right->shape[axis] * inner;
    for (size_t block = 0; block < outer; ++block) {
        for (size_t item = 0; item < left_block; ++item) {
            result->values[block * (left_block + right_block) + item] = sev_tensor_value(
                left,
                sev_tensor_physical_index(left, block * left_block + item)
            );
        }
        for (size_t item = 0; item < right_block; ++item) {
            result->values[block * (left_block + right_block) + left_block + item] =
                sev_tensor_value(
                    right,
                    sev_tensor_physical_index(right, block * right_block + item)
                );
        }
    }
    return sev_tensor_wrap(result);
}

void *__sev_tensor_repeat(void *value, void *spec_storage) {
    sev_tensor *source = sev_tensor_get(value);
    sev_list *spec = spec_storage;
    sev_tensor_abort_if(spec->length != 2);
    size_t axis = (size_t)spec->values[0];
    size_t repeats = (size_t)spec->values[1];
    sev_tensor_abort_if(axis >= source->rank || repeats == 0);
    int64_t *shape = calloc(source->rank == 0 ? 1 : source->rank, sizeof(*shape));
    sev_tensor_abort_if(shape == NULL);
    memcpy(shape, source->shape, source->rank * sizeof(*shape));
    shape[axis] *= (int64_t)repeats;
    sev_tensor *result = sev_tensor_new(source->rank, shape, source->dtype);
    free(shape);
    size_t inner = 1;
    for (size_t current = axis + 1; current < source->rank; ++current) inner *= (size_t)source->shape[current];
    size_t axis_width = (size_t)source->shape[axis];
    size_t outer = source->count / (axis_width * inner);
    for (size_t block = 0; block < outer; ++block) {
        for (size_t selected = 0; selected < axis_width; ++selected) {
            for (size_t copy = 0; copy < repeats; ++copy) {
                for (size_t item = 0; item < inner; ++item) {
                    size_t input = (block * axis_width + selected) * inner + item;
                    size_t output = ((block * axis_width + selected) * repeats + copy) * inner + item;
                    result->values[output] = sev_tensor_value(
                        source,
                        sev_tensor_physical_index(source, input)
                    );
                }
            }
        }
    }
    return sev_tensor_wrap(result);
}

void *__sev_tensor_rope(void *value, void *configuration_value) {
    sev_tensor *source = sev_tensor_get(value);
    sev_tensor *configuration = sev_tensor_get(configuration_value);
    sev_tensor_abort_if(source->rank < 2 || configuration->count < 2);
    int64_t sequence = source->shape[source->rank - 2];
    int64_t width = source->shape[source->rank - 1];
    sev_tensor_abort_if(width % 2 != 0);
    double theta = sev_tensor_as_f64(sev_tensor_value(configuration, 0), configuration->dtype);
    double offset = sev_tensor_as_f64(sev_tensor_value(configuration, 1), configuration->dtype);
    sev_tensor *result = sev_tensor_new(source->rank, source->shape, source->dtype);
    size_t matrices = source->count / ((size_t)sequence * (size_t)width);
    int64_t half = width / 2;
    for (size_t matrix = 0; matrix < matrices; ++matrix) {
        for (int64_t position = 0; position < sequence; ++position) {
            for (int64_t column = 0; column < half; ++column) {
                double frequency = pow(theta, -(double)column / (double)half);
                double angle = ((double)position + offset) * frequency;
                size_t base = (matrix * (size_t)sequence + (size_t)position) * (size_t)width;
                __float128 left = sev_tensor_float(
                    sev_tensor_value(
                        source,
                        sev_tensor_physical_index(source, base + (size_t)column)
                    ),
                    source->dtype
                );
                __float128 right = sev_tensor_float(
                    sev_tensor_value(
                        source,
                        sev_tensor_physical_index(
                            source,
                            base + (size_t)(column + half)
                        )
                    ),
                    source->dtype
                );
                __float128 cosine = (__float128)cosl((long double)angle);
                __float128 sine = (__float128)sinl((long double)angle);
                result->values[base + (size_t)column] = sev_tensor_from_float(
                    left * cosine - right * sine, source->dtype
                );
                result->values[base + (size_t)(column + half)] = sev_tensor_from_float(
                    right * cosine + left * sine, source->dtype
                );
            }
        }
    }
    return sev_tensor_wrap(result);
}

int32_t __sev_tensor_release(void *value) {
    sev_tensor *tensor = value;
    if (tensor == NULL) return -1;
    if (tensor->owns_values) free(tensor->values);
    free(tensor->gradient);
    free(tensor->shape);
    free(tensor->strides);
    free(tensor);
    return 0;
}
