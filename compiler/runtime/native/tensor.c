#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    size_t length;
    size_t capacity;
    uintptr_t *values;
} sev_list;

typedef struct {
    size_t rank;
    size_t count;
    int32_t dtype;
    long double *values;
    int64_t *shape;
    int64_t *strides;
    int64_t offset;
} sev_tensor;

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

static long double sev_tensor_quantize_float(long double value, int exponent_bits, int precision) {
    if (!isfinite(value) || value == 0.0L) return value;
    int exponent = 0;
    long double fraction = frexpl(value, &exponent);
    int maximum_exponent = (1 << (exponent_bits - 1)) - 1;
    int minimum_exponent = 2 - (1 << (exponent_bits - 1));
    if (exponent > maximum_exponent) return copysignl(INFINITY, value);
    if (exponent < minimum_exponent - precision) return copysignl(0.0L, value);
    long double scale = ldexpl(1.0L, precision - 1);
    fraction = roundl(fraction * scale) / scale;
    return ldexpl(fraction, exponent);
}

static long double sev_tensor_quantize(long double value, int32_t dtype) {
    switch (dtype) {
        case 0: return (int8_t)value;
        case 1: return (int16_t)value;
        case 2: return (int32_t)value;
        case 3: return (int64_t)value;
        case 4: return truncl(value);
        case 5: return (uint8_t)value;
        case 6: return (uint16_t)value;
        case 7: return (uint32_t)value;
        case 8: return (uint64_t)value;
        case 9: return value < 0.0L ? 0.0L : truncl(value);
        case 10: return sev_tensor_quantize_float(value, 4, 4);
        case 11: return sev_tensor_quantize_float(value, 5, 3);
        case 12: return sev_tensor_quantize_float(value, 5, 11);
        case 13: return sev_tensor_quantize_float(value, 8, 8);
        case 14: return (float)value;
        case 15: return (double)value;
        case 16: return value;
        default: abort();
    }
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
        tensor->values[index] = sev_tensor_f64_from_bits(values->values[index]);
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
        double element = (double)tensor->values[sev_tensor_physical_index(tensor, index)];
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
    result->dtype = dtype;
    for (size_t index = 0; index < result->count; ++index) {
        result->values[index] = sev_tensor_quantize(result->values[index], dtype);
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
        long double l = left->values[sev_tensor_broadcast_index(left, rank, result->shape, index)];
        long double r = right->values[sev_tensor_broadcast_index(right, rank, result->shape, index)];
        long double value = operation == '+' ? l + r
            : operation == '-' ? l - r
            : operation == '*' ? l * r
            : l / r;
        result->values[index] = sev_tensor_quantize(value, result->dtype);
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
    long double total = 0.0L;
    for (size_t index = 0; index < source->count; ++index) {
        total += source->values[sev_tensor_physical_index(source, index)];
    }
    result->values[0] = sev_tensor_quantize(total, result->dtype);
    return sev_tensor_wrap(result);
}

void *__sev_tensor_matmul(void *left_value, void *right_value) {
    sev_tensor *left = sev_tensor_get(left_value);
    sev_tensor *right = sev_tensor_get(right_value);
    sev_tensor_abort_if(left->rank != 2 || right->rank != 2);
    sev_tensor_abort_if(left->shape[1] != right->shape[0] || left->dtype != right->dtype);
    int64_t shape[2] = {left->shape[0], right->shape[1]};
    sev_tensor *result = sev_tensor_new(2, shape, left->dtype);
    for (int64_t row = 0; row < shape[0]; ++row) {
        for (int64_t column = 0; column < shape[1]; ++column) {
            long double total = 0.0L;
            for (int64_t inner = 0; inner < left->shape[1]; ++inner) {
                size_t l = (size_t)(left->offset + row * left->strides[0] + inner * left->strides[1]);
                size_t r = (size_t)(right->offset + inner * right->strides[0] + column * right->strides[1]);
                total += left->values[l] * right->values[r];
            }
            result->values[(size_t)(row * shape[1] + column)] =
                sev_tensor_quantize(total, result->dtype);
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
