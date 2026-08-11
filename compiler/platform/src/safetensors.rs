pub(crate) fn source() -> &'static str {
    r#"
typedef struct {
  int64_t rank;
  int64_t *shape;
  int64_t elements;
  const uint16_t *data;
  sev_mapped_file *mapping;
  size_t byte_offset;
} sev_mapped_bf16_tensor;

void *__sev_safetensor_bf16_view(void *mapping_raw, int64_t byte_offset,
                                 void *shape_raw) {
  sev_mapped_file *mapping = mapping_raw;
  sev_collection *shape_values = shape_raw;
  if (!mapping || mapping->unmapped || !shape_values || shape_values->size <= 0) abort();
  if (byte_offset < 0 || (byte_offset & 1) != 0) abort();
  sev_mapped_bf16_tensor *tensor = sev_allocate(sizeof(*tensor));
  tensor->rank = shape_values->size;
  tensor->shape = sev_allocate((size_t)tensor->rank * sizeof(*tensor->shape));
  tensor->elements = 1;
  for (int64_t axis = 0; axis < tensor->rank; ++axis) {
    int64_t dimension = __sev_unbox_i64(shape_values->items[axis]);
    if (dimension < 0 || (dimension != 0 && tensor->elements > INT64_MAX / dimension)) abort();
    tensor->shape[axis] = dimension;
    tensor->elements *= dimension;
  }
  size_t bytes = 0;
  if ((uint64_t)tensor->elements > SIZE_MAX / sizeof(uint16_t)) abort();
  bytes = (size_t)tensor->elements * sizeof(uint16_t);
  if ((uint64_t)byte_offset > mapping->size || bytes > mapping->size - (size_t)byte_offset) abort();
  tensor->data = (const uint16_t *)(mapping->data + byte_offset);
  tensor->mapping = mapping;
  tensor->byte_offset = (size_t)byte_offset;
  return tensor;
}

void *__sev_safetensor_bf16_shape(void *tensor_raw) {
  sev_mapped_bf16_tensor *tensor = tensor_raw;
  if (!tensor || !tensor->mapping || tensor->mapping->unmapped) abort();
  sev_collection *shape = __sev_collection_new(0);
  for (int64_t axis = 0; axis < tensor->rank; ++axis)
    __sev_collection_push(shape, __sev_box_i64(tensor->shape[axis]));
  return shape;
}
"#
}
