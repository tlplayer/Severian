pub(crate) fn source(relu: bool, add: bool, matmul: bool) -> String {
    if !relu && !add && !matmul {
        return String::new();
    }
    let mut source = String::from(
        r#"
typedef struct { int64_t rank; int64_t *shape; int64_t size; double *data; } sev_tensor;
typedef struct { double *allocated; double *aligned; int64_t offset; int64_t sizes[1]; int64_t strides[1]; } sev_memref_1d_f64;
typedef struct { double *allocated; double *aligned; int64_t offset; int64_t sizes[2]; int64_t strides[2]; } sev_memref_2d_f64;

static sev_tensor *sev_tensor_allocate(int64_t rank, const int64_t *shape) {
  if (rank <= 0) abort();
  sev_tensor *tensor = sev_allocate(sizeof(*tensor));
  tensor->rank = rank;
  tensor->shape = sev_allocate((size_t)rank * sizeof(*tensor->shape));
  tensor->size = 1;
  for (int64_t axis = 0; axis < rank; ++axis) {
    if (shape[axis] < 0 || (shape[axis] != 0 && tensor->size > INT64_MAX / shape[axis])) abort();
    tensor->shape[axis] = shape[axis];
    tensor->size *= shape[axis];
  }
  tensor->data = sev_allocate((size_t)tensor->size * sizeof(*tensor->data));
  return tensor;
}

void *__sev_tensor_from_list(void *values_raw, void *shape_raw) {
  sev_collection *values = values_raw;
  sev_collection *shape_values = shape_raw;
  int64_t *shape = sev_allocate((size_t)shape_values->size * sizeof(*shape));
  for (int64_t axis = 0; axis < shape_values->size; ++axis) shape[axis] = __sev_unbox_i64(shape_values->items[axis]);
  sev_tensor *tensor = sev_tensor_allocate(shape_values->size, shape);
  free(shape);
  if (tensor->size != values->size) abort();
  for (int64_t index = 0; index < tensor->size; ++index) tensor->data[index] = sev_number(values->items[index]);
  return tensor;
}

void *__sev_tensor_to_list(void *tensor_raw) {
  sev_tensor *tensor = tensor_raw;
  sev_collection *values = __sev_collection_new(0);
  for (int64_t index = 0; index < tensor->size; ++index) __sev_collection_push(values, __sev_box_f64(tensor->data[index]));
  return values;
}

void *__sev_tensor_shape(void *tensor_raw) {
  sev_tensor *tensor = tensor_raw;
  sev_collection *shape = __sev_collection_new(0);
  for (int64_t axis = 0; axis < tensor->rank; ++axis) __sev_collection_push(shape, __sev_box_i64(tensor->shape[axis]));
  return shape;
}

static sev_memref_1d_f64 sev_tensor_memref_1d(sev_tensor *tensor) {
  sev_memref_1d_f64 value = {tensor->data, tensor->data, 0, {tensor->size}, {1}};
  return value;
}

static sev_memref_2d_f64 sev_tensor_memref_2d(sev_tensor *tensor) {
  if (tensor->rank != 2) abort();
  sev_memref_2d_f64 value = {tensor->data, tensor->data, 0, {tensor->shape[0], tensor->shape[1]}, {tensor->shape[1], 1}};
  return value;
}
"#,
    );
    if relu {
        source.push_str(
            r#"
extern void _mlir_ciface___sev_linalg_relu(sev_memref_1d_f64 *, sev_memref_1d_f64 *);
void *__sev_tensor_relu(void *input_raw) {
  sev_tensor *input = input_raw;
  sev_tensor *output = sev_tensor_allocate(input->rank, input->shape);
  sev_memref_1d_f64 input_memref = sev_tensor_memref_1d(input);
  sev_memref_1d_f64 output_memref = sev_tensor_memref_1d(output);
  _mlir_ciface___sev_linalg_relu(&input_memref, &output_memref);
  return output;
}
"#,
        );
    }
    if add {
        source.push_str(
            r#"
extern void _mlir_ciface___sev_linalg_add(sev_memref_1d_f64 *, sev_memref_1d_f64 *, sev_memref_1d_f64 *);
void *__sev_tensor_add(void *left_raw, void *right_raw) {
  sev_tensor *left = left_raw;
  sev_tensor *right = right_raw;
  if (left->rank != right->rank || left->size != right->size) abort();
  for (int64_t axis = 0; axis < left->rank; ++axis) if (left->shape[axis] != right->shape[axis]) abort();
  sev_tensor *output = sev_tensor_allocate(left->rank, left->shape);
  sev_memref_1d_f64 left_memref = sev_tensor_memref_1d(left);
  sev_memref_1d_f64 right_memref = sev_tensor_memref_1d(right);
  sev_memref_1d_f64 output_memref = sev_tensor_memref_1d(output);
  _mlir_ciface___sev_linalg_add(&left_memref, &right_memref, &output_memref);
  return output;
}
"#,
        );
    }
    if matmul {
        source.push_str(
            r#"
extern void _mlir_ciface___sev_linalg_matmul(sev_memref_2d_f64 *, sev_memref_2d_f64 *, sev_memref_2d_f64 *);
void *__sev_tensor_matmul(void *left_raw, void *right_raw) {
  sev_tensor *left = left_raw;
  sev_tensor *right = right_raw;
  if (left->rank != 2 || right->rank != 2 || left->shape[1] != right->shape[0]) abort();
  int64_t output_shape[2] = {left->shape[0], right->shape[1]};
  sev_tensor *output = sev_tensor_allocate(2, output_shape);
  sev_memref_2d_f64 left_memref = sev_tensor_memref_2d(left);
  sev_memref_2d_f64 right_memref = sev_tensor_memref_2d(right);
  sev_memref_2d_f64 output_memref = sev_tensor_memref_2d(output);
  _mlir_ciface___sev_linalg_matmul(&left_memref, &right_memref, &output_memref);
  return output;
}
"#,
        );
    }
    source
}
