pub(crate) fn source() -> &'static str {
    r#"
typedef enum {
  SEV_DTYPE_BOOL,
  SEV_DTYPE_I8,
  SEV_DTYPE_I16,
  SEV_DTYPE_I32,
  SEV_DTYPE_I64,
  SEV_DTYPE_U8,
  SEV_DTYPE_U16,
  SEV_DTYPE_U32,
  SEV_DTYPE_U64,
  SEV_DTYPE_F8E4M3FN,
  SEV_DTYPE_F8E5M2,
  SEV_DTYPE_F16,
  SEV_DTYPE_BF16,
  SEV_DTYPE_F32,
  SEV_DTYPE_F64,
  SEV_DTYPE_C64,
  SEV_DTYPE_C128
} sev_tensor_dtype;

#define SEV_MAPPED_TENSOR_MAGIC UINT64_C(0x53455654454E534F)

typedef struct {
  uint64_t magic;
  int32_t dtype;
  int32_t element_bytes;
  int64_t rank;
  int64_t *shape;
  int64_t elements;
  const void *data;
  sev_mapped_file *mapping;
  size_t byte_offset;
} sev_mapped_tensor;

static sev_mapped_tensor *sev_safetensor_view(void *mapping_raw,
                                              int64_t byte_offset,
                                              void *shape_raw,
                                              sev_tensor_dtype dtype,
                                              int32_t element_bytes) {
  sev_mapped_file *mapping = mapping_raw;
  sev_collection *shape_values = shape_raw;
  if (!mapping || mapping->unmapped || !shape_values || shape_values->size <= 0) abort();
  if (byte_offset < 0 || element_bytes <= 0 || byte_offset % element_bytes != 0) abort();
  sev_mapped_tensor *tensor = sev_allocate(sizeof(*tensor));
  tensor->magic = SEV_MAPPED_TENSOR_MAGIC;
  tensor->dtype = (int32_t)dtype;
  tensor->element_bytes = element_bytes;
  tensor->rank = shape_values->size;
  tensor->shape = sev_allocate((size_t)tensor->rank * sizeof(*tensor->shape));
  tensor->elements = 1;
  for (int64_t axis = 0; axis < tensor->rank; ++axis) {
    int64_t dimension = __sev_unbox_i64(shape_values->items[axis]);
    if (dimension < 0 || (dimension != 0 && tensor->elements > INT64_MAX / dimension)) abort();
    tensor->shape[axis] = dimension;
    tensor->elements *= dimension;
  }
  if ((uint64_t)tensor->elements > SIZE_MAX / (size_t)element_bytes) abort();
  size_t bytes = (size_t)tensor->elements * (size_t)element_bytes;
  if ((uint64_t)byte_offset > mapping->size || bytes > mapping->size - (size_t)byte_offset) abort();
  tensor->data = mapping->data + byte_offset;
  tensor->mapping = mapping;
  tensor->byte_offset = (size_t)byte_offset;
  return tensor;
}

void *__sev_safetensor_shape(void *tensor_raw) {
  sev_mapped_tensor *tensor = tensor_raw;
  if (!tensor || tensor->magic != SEV_MAPPED_TENSOR_MAGIC || !tensor->mapping || tensor->mapping->unmapped) abort();
  sev_collection *shape = __sev_collection_new(0);
  for (int64_t axis = 0; axis < tensor->rank; ++axis)
    __sev_collection_push(shape, __sev_box_i64(tensor->shape[axis]));
  return shape;
}

void *__sev_safetensor_bf16_view(void *mapping_raw, int64_t byte_offset,
                                 void *shape_raw) {
  return sev_safetensor_view(mapping_raw, byte_offset, shape_raw, SEV_DTYPE_BF16, 2);
}

void *__sev_safetensor_bf16_shape(void *tensor_raw) {
  return __sev_safetensor_shape(tensor_raw);
}

void *__sev_safetensor_f32_view(void *mapping_raw, int64_t byte_offset,
                                void *shape_raw) {
  return sev_safetensor_view(mapping_raw, byte_offset, shape_raw, SEV_DTYPE_F32, 4);
}

void *__sev_safetensor_f32_shape(void *tensor_raw) {
  return __sev_safetensor_shape(tensor_raw);
}

void *__sev_safetensor_f8e4m3fn_view(void *mapping_raw, int64_t byte_offset,
                                     void *shape_raw) {
  return sev_safetensor_view(mapping_raw, byte_offset, shape_raw, SEV_DTYPE_F8E4M3FN, 1);
}

void *__sev_safetensor_f8e5m2_view(void *mapping_raw, int64_t byte_offset,
                                   void *shape_raw) {
  return sev_safetensor_view(mapping_raw, byte_offset, shape_raw, SEV_DTYPE_F8E5M2, 1);
}

void *__sev_safetensor_f16_view(void *mapping_raw, int64_t byte_offset,
                                void *shape_raw) {
  return sev_safetensor_view(mapping_raw, byte_offset, shape_raw, SEV_DTYPE_F16, 2);
}
"#
}
