#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "../native/tensor.c"

static void exercise_integer(int32_t dtype) {
    sev_tensor_cell one = sev_tensor_from_unsigned(1, dtype);
    sev_tensor_cell two = sev_tensor_binary_cell(one, one, dtype, '+');
    assert(sev_tensor_as_f64(two, dtype) == 2.0);
    sev_tensor_cell six = sev_tensor_binary_cell(
        sev_tensor_from_unsigned(3, dtype),
        sev_tensor_from_unsigned(2, dtype),
        dtype,
        '*'
    );
    assert(sev_tensor_as_f64(six, dtype) == 6.0);
}

static void exercise_float(int32_t dtype) {
    sev_tensor_cell left = sev_tensor_from_float(1.5Q, dtype);
    sev_tensor_cell right = sev_tensor_from_float(2.0Q, dtype);
    sev_tensor_cell result = sev_tensor_binary_cell(left, right, dtype, '+');
    assert(sev_tensor_as_f64(result, dtype) == 3.5);
}

static void exercise_safetensor_view(int32_t dtype, const char *name) {
    size_t width = sev_tensor_dtype_bits(dtype) / 8;
    char header[160];
    int rendered = snprintf(
        header,
        sizeof(header),
        "{\"x\":{\"dtype\":\"%s\",\"shape\":[1],\"data_offsets\":[0,%zu]}}",
        name,
        width
    );
    assert(rendered > 0 && (size_t)rendered < sizeof(header));
    size_t header_length = (size_t)rendered;
    size_t length = 8 + header_length + width;
    uint8_t *bytes = calloc(length, 1);
    assert(bytes != NULL);
    uint64_t encoded_header_length = (uint64_t)header_length;
    memcpy(bytes, &encoded_header_length, 8);
    memcpy(bytes + 8, header, header_length);
    for (size_t index = 0; index < width; ++index) {
        bytes[8 + header_length + index] = (uint8_t)(index + 1);
    }
    sev_safetensor store = {-1, length, bytes, header_length};
    sev_tensor *tensor = sev_safetensor_view((int64_t)(intptr_t)&store, "x", dtype);
    assert(tensor->dtype == dtype);
    assert(tensor->rank == 1 && tensor->shape[0] == 1 && tensor->count == 1);
    assert(tensor->values[0].bits == sev_read_u128(bytes + 8 + header_length, width));
    free(tensor->values);
    free(tensor->shape);
    free(tensor->strides);
    free(tensor);
    free(bytes);
}

static sev_tensor_cell scalar_cell(double value, int32_t dtype) {
    if (sev_tensor_dtype_signed(dtype)) return sev_tensor_from_signed((__int128)value, dtype);
    if (sev_tensor_dtype_unsigned(dtype)) {
        return sev_tensor_from_unsigned((unsigned __int128)value, dtype);
    }
    return sev_tensor_from_float((__float128)value, dtype);
}

static void exercise_kernels(int32_t dtype) {
    int64_t left_shape[2] = {1, 2};
    int64_t right_shape[2] = {2, 1};
    sev_tensor *left = sev_tensor_new(2, left_shape, dtype);
    sev_tensor *right = sev_tensor_new(2, right_shape, dtype);
    left->values[0] = scalar_cell(1.0, dtype);
    left->values[1] = scalar_cell(2.0, dtype);
    right->values[0] = scalar_cell(3.0, dtype);
    right->values[1] = scalar_cell(4.0, dtype);
    sev_tensor *product = __sev_tensor_matmul(left, right);
    assert(product->dtype == dtype);
    double product_value = sev_tensor_as_f64(product->values[0], dtype);
    double stored_expected = dtype == 11 ? 12.0 : 11.0;
    if (product_value != stored_expected) {
        fprintf(stderr, "dtype %d matmul produced %g\n", dtype, product_value);
    }
    assert(product_value == stored_expected);
    sev_tensor *total = __sev_tensor_sum(left);
    assert(total->dtype == dtype);
    assert(sev_tensor_as_f64(total->values[0], dtype) == 3.0);
}

int main(void) {
    for (int32_t dtype = 0; dtype <= 9; ++dtype) exercise_integer(dtype);
    for (int32_t dtype = 10; dtype <= 16; ++dtype) exercise_float(dtype);
    const char *safetensor_names[] = {
        "I8", "I16", "I32", "I64", "I128", "U8", "U16", "U32", "U64", "U128",
        "F8_E4M3FN", "F8_E5M2", "F16", "BF16", "F32", "F64", "F128"
    };
    for (int32_t dtype = 0; dtype < 17; ++dtype) {
        exercise_safetensor_view(dtype, safetensor_names[dtype]);
        exercise_kernels(dtype);
    }

    unsigned __int128 high = (unsigned __int128)1 << 100;
    sev_tensor_cell high_u128 = sev_tensor_from_unsigned(high, 9);
    sev_tensor_cell doubled_u128 = sev_tensor_binary_cell(high_u128, high_u128, 9, '+');
    assert(sev_tensor_unsigned(doubled_u128, 9) == (unsigned __int128)1 << 101);

    __int128 negative = -((__int128)1 << 100);
    sev_tensor_cell high_i128 = sev_tensor_from_signed(negative, 4);
    assert(sev_tensor_signed(high_i128, 4) == negative);

    __float128 precise = 1.0Q + 0x1p-100Q;
    sev_tensor_cell f128 = sev_tensor_from_float(precise, 16);
    assert(sev_tensor_float(f128, 16) == precise);
    sev_tensor_cell f128_sum = sev_tensor_binary_cell(f128, f128, 16, '+');
    assert(sev_tensor_float(f128_sum, 16) == precise + precise);

    for (int32_t dtype = 0; dtype < 17; ++dtype) {
        int32_t accumulation = sev_tensor_accumulation_dtype(dtype);
        assert(accumulation >= 0 && accumulation < 17);
    }
    return 0;
}
