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

typedef struct {
    unsigned __int128 bits;
} sev_tensor_cell;

typedef struct {
    size_t rank;
    size_t count;
    int32_t dtype;
    sev_tensor_cell *values;
    int64_t *shape;
    int64_t *strides;
    int64_t offset;
} sev_tensor;

typedef struct {
    int descriptor;
    size_t length;
    const uint8_t *bytes;
    uint64_t header_length;
} sev_safetensor;

static sev_tensor *sev_tensor_new(size_t rank, const int64_t *shape, int32_t dtype);
static void *sev_tensor_wrap(sev_tensor *tensor);

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
        "F8_E4M3FN", "F8_E5M2", "F16", "BF16", "F32", "F64", "F128"
    };
    for (int32_t index = 0; index < 17; ++index) {
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

static void *sev_safetensor_view(int64_t handle, const char *name, int32_t requested_dtype) {
    sev_safetensor *store = (sev_safetensor *)(intptr_t)handle;
    sev_tensor_abort_if(store == NULL);
    char dtype[24];
    const uint8_t *object;
    const uint8_t *object_end;
    sev_tensor_abort_if(!sev_safetensor_object(store, name, &object, &object_end));
    sev_tensor_abort_if(!sev_safetensor_string_field(
        object, object_end, "\"dtype\"", dtype, sizeof(dtype)
    ));
    sev_tensor_abort_if(sev_safetensor_dtype_tag(dtype) != requested_dtype);
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
    sev_tensor *tensor = sev_tensor_new(rank, shape, requested_dtype);
    free(shape);
    size_t byte_width = requested_dtype == 4 || requested_dtype == 9 || requested_dtype == 16
        ? 16
        : requested_dtype == 3 || requested_dtype == 8 || requested_dtype == 15
            ? 8
            : requested_dtype == 2 || requested_dtype == 7 || requested_dtype == 14
                ? 4
                : requested_dtype == 1 || requested_dtype == 6 || requested_dtype == 12 || requested_dtype == 13
                    ? 2
                    : 1;
    sev_tensor_abort_if(relative_end - relative_start != tensor->count * byte_width);
    const uint8_t *data = store->bytes + data_start + relative_start;
    for (size_t index = 0; index < tensor->count; ++index) {
        tensor->values[index].bits = sev_read_u128(data + index * byte_width, byte_width);
    }
    return sev_tensor_wrap(tensor);
}

#define SEV_SAFETENSOR_VIEW(name, tag) \
    void *__sev_safetensor_##name##_view(int64_t handle, const char *tensor_name) { \
        return sev_safetensor_view(handle, tensor_name, tag); \
    }

SEV_SAFETENSOR_VIEW(i8, 0)
SEV_SAFETENSOR_VIEW(i16, 1)
SEV_SAFETENSOR_VIEW(i32, 2)
SEV_SAFETENSOR_VIEW(i64, 3)
SEV_SAFETENSOR_VIEW(i128, 4)
SEV_SAFETENSOR_VIEW(u8, 5)
SEV_SAFETENSOR_VIEW(u16, 6)
SEV_SAFETENSOR_VIEW(u32, 7)
SEV_SAFETENSOR_VIEW(u64, 8)
SEV_SAFETENSOR_VIEW(u128, 9)
SEV_SAFETENSOR_VIEW(f8e4m3fn, 10)
SEV_SAFETENSOR_VIEW(f8e5m2, 11)
SEV_SAFETENSOR_VIEW(f16, 12)
SEV_SAFETENSOR_VIEW(bf16, 13)
SEV_SAFETENSOR_VIEW(f32, 14)
SEV_SAFETENSOR_VIEW(f64, 15)
SEV_SAFETENSOR_VIEW(f128, 16)

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
        8, 16, 32, 64, 128, 8, 16, 32, 64, 128, 8, 8, 16, 16, 32, 64, 128
    };
    sev_tensor_abort_if(dtype < 0 || dtype >= 17);
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
    return tensor;
}

static sev_tensor *sev_tensor_get(void *value) {
    sev_tensor_abort_if(value == NULL);
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
            tensor->values[sev_tensor_physical_index(tensor, index)],
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
        result->values[index] = source->values[sev_tensor_physical_index(source, index)];
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
        sev_tensor_cell l = left->values[
            sev_tensor_broadcast_index(left, rank, result->shape, index)
        ];
        sev_tensor_cell r = right->values[
            sev_tensor_broadcast_index(right, rank, result->shape, index)
        ];
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
            source->values[sev_tensor_physical_index(source, index)],
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

void *__sev_tensor_matmul(void *left_value, void *right_value) {
    sev_tensor *left = sev_tensor_get(left_value);
    sev_tensor *right = sev_tensor_get(right_value);
    sev_tensor_abort_if(left->rank != 2 || right->rank != 2);
    sev_tensor_abort_if(left->shape[1] != right->shape[0] || left->dtype != right->dtype);
    int64_t shape[2] = {left->shape[0], right->shape[1]};
    sev_tensor *result = sev_tensor_new(2, shape, left->dtype);
    int32_t accumulation_dtype = sev_tensor_accumulation_dtype(left->dtype);
    for (int64_t row = 0; row < shape[0]; ++row) {
        for (int64_t column = 0; column < shape[1]; ++column) {
            sev_tensor_cell total = {0};
            for (int64_t inner = 0; inner < left->shape[1]; ++inner) {
                size_t l = (size_t)(left->offset + row * left->strides[0] + inner * left->strides[1]);
                size_t r = (size_t)(right->offset + inner * right->strides[0] + column * right->strides[1]);
                sev_tensor_cell left_element = sev_tensor_convert_cell(
                    left->values[l], left->dtype, accumulation_dtype
                );
                sev_tensor_cell right_element = sev_tensor_convert_cell(
                    right->values[r], right->dtype, accumulation_dtype
                );
                sev_tensor_cell product = sev_tensor_binary_cell(
                    left_element,
                    right_element,
                    accumulation_dtype,
                    '*'
                );
                total = sev_tensor_binary_cell(total, product, accumulation_dtype, '+');
            }
            result->values[(size_t)(row * shape[1] + column)] = sev_tensor_convert_cell(
                total, accumulation_dtype, result->dtype
            );
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
