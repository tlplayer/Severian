pub(crate) fn source(
    relu: bool,
    add: bool,
    matmul: bool,
    transpose: bool,
    scale: bool,
    softmax_rows: bool,
    layer_norm: bool,
    relu_backward: bool,
    softmax_backward: bool,
    layer_norm_backward: bool,
    autodiff: bool,
    rocm: bool,
) -> String {
    if !relu
        && !add
        && !matmul
        && !transpose
        && !scale
        && !softmax_rows
        && !layer_norm
        && !relu_backward
        && !softmax_backward
        && !layer_norm_backward
        && !autodiff
    {
        return String::new();
    }
    let mut source = String::new();
    if rocm {
        source.push_str(ROCM_RUNTIME_SOURCE);
    } else {
        source.push_str(
            "static void sev_tensor_graph_begin(void) {}\n\
             static void sev_tensor_graph_end(void) {}\n\
             static void *sev_tensor_data_allocate(size_t size) { return sev_allocate(size); }\n",
        );
    }
    source.push_str(
        r#"
typedef enum {
  SEV_TENSOR_LEAF,
  SEV_TENSOR_RELU,
  SEV_TENSOR_ADD,
  SEV_TENSOR_MATMUL,
  SEV_TENSOR_TRANSPOSE,
  SEV_TENSOR_SCALE,
  SEV_TENSOR_SOFTMAX_ROWS,
  SEV_TENSOR_LAYER_NORM
} sev_tensor_operation;
typedef struct sev_tensor {
  int64_t rank;
  int64_t *shape;
  int64_t *strides;
  int64_t size;
  double *data;
  double *allocation;
  bool is_view;
  sev_tensor_operation operation;
  struct sev_tensor *left;
  struct sev_tensor *right;
  struct sev_tensor *gradient;
  double scalar;
} sev_tensor;
typedef struct { double *allocated; double *aligned; int64_t offset; int64_t sizes[1]; int64_t strides[1]; } sev_memref_1d_f64;
typedef struct { double *allocated; double *aligned; int64_t offset; int64_t sizes[2]; int64_t strides[2]; } sev_memref_2d_f64;

static sev_tensor *sev_tensor_allocate(int64_t rank, const int64_t *shape) {
  if (rank <= 0) abort();
  sev_tensor *tensor = sev_allocate(sizeof(*tensor));
  tensor->rank = rank;
  tensor->shape = sev_allocate((size_t)rank * sizeof(*tensor->shape));
  tensor->strides = sev_allocate((size_t)rank * sizeof(*tensor->strides));
  tensor->size = 1;
  for (int64_t axis = rank - 1; axis >= 0; --axis) {
    if (shape[axis] < 0 || (shape[axis] != 0 && tensor->size > INT64_MAX / shape[axis])) abort();
    tensor->shape[axis] = shape[axis];
    tensor->strides[axis] = tensor->size;
    tensor->size *= shape[axis];
  }
  tensor->data = sev_tensor_data_allocate((size_t)tensor->size * sizeof(*tensor->data));
  tensor->allocation = tensor->data;
  tensor->is_view = false;
  return tensor;
}

static int64_t sev_tensor_offset(const sev_tensor *tensor, int64_t linear) {
  int64_t offset = 0;
  for (int64_t axis = tensor->rank - 1; axis >= 0; --axis) {
    int64_t coordinate = tensor->shape[axis] == 0 ? 0 : linear % tensor->shape[axis];
    if (tensor->shape[axis] != 0) linear /= tensor->shape[axis];
    offset += coordinate * tensor->strides[axis];
  }
  return offset;
}

static bool sev_tensor_is_contiguous(const sev_tensor *tensor) {
  int64_t stride = 1;
  for (int64_t axis = tensor->rank - 1; axis >= 0; --axis) {
    if (tensor->shape[axis] > 1 && tensor->strides[axis] != stride) return false;
    stride *= tensor->shape[axis];
  }
  return true;
}

static sev_tensor *sev_tensor_materialize(sev_tensor *input) {
  if (!input) abort();
  sev_tensor *output = sev_tensor_allocate(input->rank, input->shape);
  for (int64_t index = 0; index < input->size; ++index)
    output->data[index] = input->data[sev_tensor_offset(input, index)];
  return output;
}

static sev_tensor *sev_tensor_contiguous(sev_tensor *input) {
  return sev_tensor_is_contiguous(input) ? input : sev_tensor_materialize(input);
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
  for (int64_t index = 0; index < tensor->size; ++index)
    __sev_collection_push(values, __sev_box_f64(tensor->data[sev_tensor_offset(tensor, index)]));
  return values;
}

void *__sev_tensor_shape(void *tensor_raw) {
  sev_tensor *tensor = tensor_raw;
  sev_collection *shape = __sev_collection_new(0);
  for (int64_t axis = 0; axis < tensor->rank; ++axis) __sev_collection_push(shape, __sev_box_i64(tensor->shape[axis]));
  return shape;
}

void *__sev_tensor_strides(void *tensor_raw) {
  sev_tensor *tensor = tensor_raw;
  sev_collection *strides = __sev_collection_new(0);
  for (int64_t axis = 0; axis < tensor->rank; ++axis)
    __sev_collection_push(strides, __sev_box_i64(tensor->strides[axis]));
  return strides;
}

void *__sev_tensor_slice(void *tensor_raw, void *starts_raw, void *ends_raw, void *steps_raw) {
  sev_tensor *input = tensor_raw;
  sev_collection *starts = starts_raw;
  sev_collection *ends = ends_raw;
  sev_collection *steps = steps_raw;
  if (!input || starts->size != input->rank || ends->size != input->rank || steps->size != input->rank) abort();
  sev_tensor *view = sev_allocate(sizeof(*view));
  *view = *input;
  view->shape = sev_allocate((size_t)input->rank * sizeof(*view->shape));
  view->strides = sev_allocate((size_t)input->rank * sizeof(*view->strides));
  view->size = 1;
  int64_t offset = 0;
  for (int64_t axis = 0; axis < input->rank; ++axis) {
    int64_t start = __sev_unbox_i64(starts->items[axis]);
    int64_t end = __sev_unbox_i64(ends->items[axis]);
    int64_t step = __sev_unbox_i64(steps->items[axis]);
    if (start < 0) start += input->shape[axis];
    if (end < 0) end += input->shape[axis];
    if (step <= 0 || start < 0 || end < start || end > input->shape[axis]) abort();
    int64_t extent = (end - start + step - 1) / step;
    if (extent != 0 && view->size > INT64_MAX / extent) abort();
    view->shape[axis] = extent;
    view->strides[axis] = input->strides[axis] * step;
    view->size *= extent;
    offset += start * input->strides[axis];
  }
  view->data = input->data + offset;
  view->allocation = input->allocation;
  view->is_view = true;
  view->operation = SEV_TENSOR_LEAF;
  view->left = NULL;
  view->right = NULL;
  view->gradient = NULL;
  return view;
}

void *__sev_tensor_materialize(void *tensor_raw) {
  return sev_tensor_materialize(tensor_raw);
}

static sev_memref_1d_f64 sev_tensor_memref_1d(sev_tensor *tensor) {
  if (!sev_tensor_is_contiguous(tensor)) abort();
  sev_memref_1d_f64 value = {tensor->data, tensor->data, 0, {tensor->size}, {1}};
  return value;
}

static sev_memref_2d_f64 sev_tensor_memref_2d(sev_tensor *tensor) {
  if (tensor->rank != 2) abort();
  sev_memref_2d_f64 value = {tensor->allocation, tensor->data, 0, {tensor->shape[0], tensor->shape[1]}, {tensor->strides[0], tensor->strides[1]}};
  return value;
}

extern void _mlir_ciface___sev_linalg_sum(sev_memref_1d_f64 *, sev_memref_1d_f64 *);
void *__sev_tensor_sum(void *input_raw) {
  sev_tensor *input = sev_tensor_contiguous(input_raw);
  int64_t output_shape[1] = {1};
  sev_tensor *output = sev_tensor_allocate(1, output_shape);
  sev_memref_1d_f64 input_memref = sev_tensor_memref_1d(input);
  sev_memref_1d_f64 output_memref = sev_tensor_memref_1d(output);
  _mlir_ciface___sev_linalg_sum(&input_memref, &output_memref);
  return output;
}

static int64_t *sev_tensor_broadcast_shape(const sev_tensor *left, const sev_tensor *right, int64_t *rank) {
  *rank = left->rank > right->rank ? left->rank : right->rank;
  int64_t *shape = sev_allocate((size_t)*rank * sizeof(*shape));
  for (int64_t output_axis = *rank - 1; output_axis >= 0; --output_axis) {
    int64_t left_axis = output_axis - (*rank - left->rank);
    int64_t right_axis = output_axis - (*rank - right->rank);
    int64_t left_size = left_axis < 0 ? 1 : left->shape[left_axis];
    int64_t right_size = right_axis < 0 ? 1 : right->shape[right_axis];
    if (left_size != right_size && left_size != 1 && right_size != 1) abort();
    shape[output_axis] = left_size > right_size ? left_size : right_size;
  }
  return shape;
}

static int64_t sev_tensor_broadcast_offset(const sev_tensor *input, const sev_tensor *output, int64_t linear) {
  int64_t offset = 0;
  for (int64_t output_axis = output->rank - 1; output_axis >= 0; --output_axis) {
    int64_t coordinate = output->shape[output_axis] == 0 ? 0 : linear % output->shape[output_axis];
    if (output->shape[output_axis] != 0) linear /= output->shape[output_axis];
    int64_t input_axis = output_axis - (output->rank - input->rank);
    if (input_axis >= 0 && input->shape[input_axis] != 1)
      offset += coordinate * input->strides[input_axis];
  }
  return offset;
}
"#,
    );
    if relu {
        source.push_str(
            r#"
extern void _mlir_ciface___sev_linalg_relu(sev_memref_1d_f64 *, sev_memref_1d_f64 *);
void *__sev_tensor_relu(void *input_raw) {
  sev_tensor *input = sev_tensor_contiguous(input_raw);
  sev_tensor *output = sev_tensor_allocate(input->rank, input->shape);
  sev_memref_1d_f64 input_memref = sev_tensor_memref_1d(input);
  sev_memref_1d_f64 output_memref = sev_tensor_memref_1d(output);
  _mlir_ciface___sev_linalg_relu(&input_memref, &output_memref);
  output->operation = SEV_TENSOR_RELU;
  output->left = input;
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
  int64_t output_rank = 0;
  int64_t *output_shape = sev_tensor_broadcast_shape(left, right, &output_rank);
  sev_tensor *output = sev_tensor_allocate(output_rank, output_shape);
  free(output_shape);
  bool identical = left->rank == right->rank && left->size == right->size;
  for (int64_t axis = 0; identical && axis < left->rank; ++axis)
    identical = left->shape[axis] == right->shape[axis];
  if (!identical || !sev_tensor_is_contiguous(left) || !sev_tensor_is_contiguous(right)) {
    for (int64_t index = 0; index < output->size; ++index)
      output->data[index] = left->data[sev_tensor_broadcast_offset(left, output, index)]
                          + right->data[sev_tensor_broadcast_offset(right, output, index)];
    output->operation = SEV_TENSOR_ADD;
    output->left = left;
    output->right = right;
    return output;
  }
  sev_memref_1d_f64 left_memref = sev_tensor_memref_1d(left);
  sev_memref_1d_f64 right_memref = sev_tensor_memref_1d(right);
  sev_memref_1d_f64 output_memref = sev_tensor_memref_1d(output);
  _mlir_ciface___sev_linalg_add(&left_memref, &right_memref, &output_memref);
  output->operation = SEV_TENSOR_ADD;
  output->left = left;
  output->right = right;
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
  output->operation = SEV_TENSOR_MATMUL;
  output->left = left;
  output->right = right;
  return output;
}
"#,
        );
    }
    if transpose {
        source.push_str(
            r#"
extern void _mlir_ciface___sev_linalg_transpose(sev_memref_2d_f64 *, sev_memref_2d_f64 *);
void *__sev_tensor_transpose(void *input_raw) {
  sev_tensor *input = input_raw;
  if (input->rank != 2) abort();
  int64_t output_shape[2] = {input->shape[1], input->shape[0]};
  sev_tensor *output = sev_tensor_allocate(2, output_shape);
  sev_memref_2d_f64 input_memref = sev_tensor_memref_2d(input);
  sev_memref_2d_f64 output_memref = sev_tensor_memref_2d(output);
  _mlir_ciface___sev_linalg_transpose(&input_memref, &output_memref);
  output->operation = SEV_TENSOR_TRANSPOSE;
  output->left = input;
  return output;
}
"#,
        );
    }
    if scale {
        source.push_str(
            r#"
extern void _mlir_ciface___sev_linalg_scale(sev_memref_1d_f64 *, double, sev_memref_1d_f64 *);
void *__sev_tensor_scale(void *input_raw, double scale) {
  sev_tensor *input = sev_tensor_contiguous(input_raw);
  sev_tensor *output = sev_tensor_allocate(input->rank, input->shape);
  sev_memref_1d_f64 input_memref = sev_tensor_memref_1d(input);
  sev_memref_1d_f64 output_memref = sev_tensor_memref_1d(output);
  _mlir_ciface___sev_linalg_scale(&input_memref, scale, &output_memref);
  output->operation = SEV_TENSOR_SCALE;
  output->left = input;
  output->scalar = scale;
  return output;
}
"#,
        );
    }
    if softmax_rows {
        source.push_str(
            r#"
extern void _mlir_ciface___sev_linalg_softmax_rows(sev_memref_2d_f64 *, sev_memref_2d_f64 *);
void *__sev_tensor_softmax_rows(void *input_raw) {
  sev_tensor *input = input_raw;
  if (input->rank != 2) abort();
  sev_tensor *output = sev_tensor_allocate(2, input->shape);
  sev_memref_2d_f64 input_memref = sev_tensor_memref_2d(input);
  sev_memref_2d_f64 output_memref = sev_tensor_memref_2d(output);
  _mlir_ciface___sev_linalg_softmax_rows(&input_memref, &output_memref);
  output->operation = SEV_TENSOR_SOFTMAX_ROWS;
  output->left = input;
  return output;
}
"#,
        );
    }
    if layer_norm {
        source.push_str(
            r#"
extern void _mlir_ciface___sev_linalg_layer_norm(sev_memref_2d_f64 *, double, sev_memref_2d_f64 *);
void *__sev_tensor_layer_norm(void *input_raw, double epsilon) {
  sev_tensor *input = input_raw;
  if (input->rank != 2 || epsilon <= 0.0) abort();
  sev_tensor *output = sev_tensor_allocate(2, input->shape);
  sev_memref_2d_f64 input_memref = sev_tensor_memref_2d(input);
  sev_memref_2d_f64 output_memref = sev_tensor_memref_2d(output);
  _mlir_ciface___sev_linalg_layer_norm(&input_memref, epsilon, &output_memref);
  output->operation = SEV_TENSOR_LAYER_NORM;
  output->left = input;
  output->scalar = epsilon;
  return output;
}
"#,
        );
    }
    if relu_backward {
        source.push_str(
            r#"
extern void _mlir_ciface___sev_linalg_relu_backward(sev_memref_1d_f64 *, sev_memref_1d_f64 *, sev_memref_1d_f64 *);
void *__sev_tensor_relu_backward(void *input_raw, void *upstream_raw) {
  sev_tensor *input = input_raw;
  sev_tensor *upstream = upstream_raw;
  if (input->size != upstream->size) abort();
  sev_tensor *output = sev_tensor_allocate(input->rank, input->shape);
  sev_memref_1d_f64 input_memref = sev_tensor_memref_1d(input);
  sev_memref_1d_f64 upstream_memref = sev_tensor_memref_1d(upstream);
  sev_memref_1d_f64 output_memref = sev_tensor_memref_1d(output);
  _mlir_ciface___sev_linalg_relu_backward(&input_memref, &upstream_memref, &output_memref);
  return output;
}
"#,
        );
    }
    if softmax_backward {
        source.push_str(
            r#"
extern void _mlir_ciface___sev_linalg_softmax_backward(sev_memref_2d_f64 *, sev_memref_2d_f64 *, sev_memref_2d_f64 *);
void *__sev_tensor_softmax_backward(void *softmax_raw, void *upstream_raw) {
  sev_tensor *softmax = softmax_raw;
  sev_tensor *upstream = upstream_raw;
  if (softmax->rank != 2 || upstream->rank != 2 || softmax->size != upstream->size) abort();
  sev_tensor *output = sev_tensor_allocate(2, softmax->shape);
  sev_memref_2d_f64 softmax_memref = sev_tensor_memref_2d(softmax);
  sev_memref_2d_f64 upstream_memref = sev_tensor_memref_2d(upstream);
  sev_memref_2d_f64 output_memref = sev_tensor_memref_2d(output);
  _mlir_ciface___sev_linalg_softmax_backward(&softmax_memref, &upstream_memref, &output_memref);
  return output;
}
"#,
        );
    }
    if layer_norm_backward {
        source.push_str(
            r#"
extern void _mlir_ciface___sev_linalg_layer_norm_backward(sev_memref_2d_f64 *, sev_memref_2d_f64 *, double, sev_memref_2d_f64 *);
void *__sev_tensor_layer_norm_backward(void *input_raw, void *upstream_raw, double epsilon) {
  sev_tensor *input = input_raw;
  sev_tensor *upstream = upstream_raw;
  if (input->rank != 2 || upstream->rank != 2 || input->size != upstream->size || epsilon <= 0.0) abort();
  sev_tensor *output = sev_tensor_allocate(2, input->shape);
  sev_memref_2d_f64 input_memref = sev_tensor_memref_2d(input);
  sev_memref_2d_f64 upstream_memref = sev_tensor_memref_2d(upstream);
  sev_memref_2d_f64 output_memref = sev_tensor_memref_2d(output);
  _mlir_ciface___sev_linalg_layer_norm_backward(&input_memref, &upstream_memref, epsilon, &output_memref);
  return output;
}
"#,
        );
    }
    if autodiff {
        source.push_str(
            r#"
static void sev_tensor_detach(sev_tensor *value) {
  value->operation = SEV_TENSOR_LEAF;
  value->left = NULL;
  value->right = NULL;
}

static void sev_tensor_accumulate_gradient(sev_tensor *value, sev_tensor *gradient) {
  sev_tensor_detach(gradient);
  if (!value->gradient) {
    value->gradient = gradient;
    return;
  }
  value->gradient = __sev_tensor_add(value->gradient, gradient);
  sev_tensor_detach(value->gradient);
}

static void sev_tensor_backward(sev_tensor *value, sev_tensor *upstream) {
  sev_tensor_accumulate_gradient(value, upstream);
  switch (value->operation) {
    case SEV_TENSOR_LEAF:
      return;
    case SEV_TENSOR_RELU:
      sev_tensor_backward(value->left, __sev_tensor_relu_backward(value->left, upstream));
      return;
    case SEV_TENSOR_ADD:
      sev_tensor_backward(value->left, upstream);
      sev_tensor_backward(value->right, upstream);
      return;
    case SEV_TENSOR_MATMUL: {
      sev_tensor *right_transpose = __sev_tensor_transpose(value->right);
      sev_tensor *left_transpose = __sev_tensor_transpose(value->left);
      sev_tensor_backward(value->left, __sev_tensor_matmul(upstream, right_transpose));
      sev_tensor_backward(value->right, __sev_tensor_matmul(left_transpose, upstream));
      return;
    }
    case SEV_TENSOR_TRANSPOSE:
      sev_tensor_backward(value->left, __sev_tensor_transpose(upstream));
      return;
    case SEV_TENSOR_SCALE:
      sev_tensor_backward(value->left, __sev_tensor_scale(upstream, value->scalar));
      return;
    case SEV_TENSOR_SOFTMAX_ROWS:
      sev_tensor_backward(value->left, __sev_tensor_softmax_backward(value, upstream));
      return;
    case SEV_TENSOR_LAYER_NORM:
      sev_tensor_backward(value->left, __sev_tensor_layer_norm_backward(value->left, upstream, value->scalar));
      return;
  }
  abort();
}

void __sev_tensor_backward_mse(void *output_raw) {
  sev_tensor *output = output_raw;
  if (!output || output->size <= 0) abort();
  sev_tensor *seed = __sev_tensor_scale(output, 2.0 / (double)output->size);
  sev_tensor_detach(seed);
  sev_tensor_backward(output, seed);
}

void *__sev_tensor_gradient(void *value_raw) {
  sev_tensor *value = value_raw;
  if (!value || !value->gradient) abort();
  return value->gradient;
}

void *__sev_tensor_sgd(void *value_raw, double learning_rate) {
  sev_tensor *value = value_raw;
  if (!value || !value->gradient || learning_rate < 0.0) abort();
  sev_tensor *step = __sev_tensor_scale(value->gradient, -learning_rate);
  sev_tensor *updated = __sev_tensor_add(value, step);
  sev_tensor_detach(updated);
  return updated;
}
"#,
        );
    }
    source
}

const ROCM_RUNTIME_SOURCE: &str = r#"
typedef int hipError_t;
typedef void *hipModule_t;
typedef void *hipFunction_t;
typedef void *hipStream_t;

extern hipError_t hipInit(unsigned int);
extern hipError_t hipMallocManaged(void **, size_t, unsigned int);
extern hipError_t hipModuleLoadData(hipModule_t *, const void *);
extern hipError_t hipModuleUnload(hipModule_t);
extern hipError_t hipModuleGetFunction(hipFunction_t *, hipModule_t, const char *);
extern hipError_t hipModuleLaunchKernel(hipFunction_t, unsigned int, unsigned int, unsigned int,
                                       unsigned int, unsigned int, unsigned int, unsigned int,
                                       hipStream_t, void **, void **);
extern hipError_t hipStreamCreate(hipStream_t *);
extern hipError_t hipStreamSynchronize(hipStream_t);
extern hipError_t hipStreamDestroy(hipStream_t);
extern hipError_t hipMemsetAsync(void *, int, size_t, hipStream_t);
extern const char *hipGetErrorString(hipError_t);

static int sev_tensor_graph_depth = 0;
static hipStream_t sev_tensor_graph_stream = NULL;

static void sev_hip_check(hipError_t result, const char *operation) {
  if (result == 0) return;
  const char *message = hipGetErrorString(result);
  fprintf(stderr, "Severian ROCm failure in %s: %s (%d)\n", operation,
          message ? message : "unknown HIP error", result);
  abort();
}

static bool sev_rocm_trace_enabled(void) {
  const char *value = getenv("SEVERIAN_ROCM_TRACE");
  return value && value[0] && strcmp(value, "0") != 0;
}

static void sev_tensor_graph_begin(void) {
  if (sev_tensor_graph_depth++ == 0) {
    sev_hip_check(hipStreamCreate(&sev_tensor_graph_stream), "hipStreamCreate(graph)");
    if (sev_rocm_trace_enabled()) fprintf(stderr, "severian-rocm: begin optimized model graph\n");
  }
}

static void sev_tensor_graph_end(void) {
  if (sev_tensor_graph_depth <= 0) abort();
  if (--sev_tensor_graph_depth == 0) {
    sev_hip_check(hipStreamSynchronize(sev_tensor_graph_stream), "hipStreamSynchronize(graph)");
    sev_hip_check(hipStreamDestroy(sev_tensor_graph_stream), "hipStreamDestroy(graph)");
    sev_tensor_graph_stream = NULL;
    if (sev_rocm_trace_enabled()) fprintf(stderr, "severian-rocm: end optimized model graph\n");
  }
}

static void *sev_tensor_data_allocate(size_t size) {
  void *value = NULL;
  sev_hip_check(hipInit(0), "hipInit");
  sev_hip_check(hipMallocManaged(&value, size ? size : 1, 1), "hipMallocManaged");
  if (sev_tensor_graph_depth > 0)
    sev_hip_check(hipMemsetAsync(value, 0, size, sev_tensor_graph_stream), "hipMemsetAsync");
  else
    memset(value, 0, size);
  return value;
}

void *mgpuModuleLoad(void *data, size_t size) {
  hipModule_t module = NULL;
  sev_hip_check(hipInit(0), "hipInit");
  sev_hip_check(hipModuleLoadData(&module, data), "hipModuleLoadData");
  if (sev_rocm_trace_enabled()) fprintf(stderr, "severian-rocm: loaded code object (%zu bytes)\n", size);
  return module;
}

void mgpuModuleUnload(hipModule_t module) {
  sev_hip_check(hipModuleUnload(module), "hipModuleUnload");
}

void *mgpuModuleGetFunction(hipModule_t module, const char *name) {
  hipFunction_t function = NULL;
  sev_hip_check(hipModuleGetFunction(&function, module, name), "hipModuleGetFunction");
  return function;
}

void mgpuLaunchKernel(hipFunction_t function, intptr_t grid_x, intptr_t grid_y,
                      intptr_t grid_z, intptr_t block_x, intptr_t block_y,
                      intptr_t block_z, int32_t shared_memory, hipStream_t stream,
                      void **parameters, void **extra, size_t parameter_count) {
  if (sev_rocm_trace_enabled())
    fprintf(stderr, "severian-rocm: launch grid=(%ld,%ld,%ld) block=(%ld,%ld,%ld) args=%zu\n",
            grid_x, grid_y, grid_z, block_x, block_y, block_z, parameter_count);
  sev_hip_check(hipModuleLaunchKernel(function, (unsigned int)grid_x, (unsigned int)grid_y,
                                     (unsigned int)grid_z, (unsigned int)block_x,
                                     (unsigned int)block_y, (unsigned int)block_z,
                                     (unsigned int)shared_memory, stream, parameters, extra),
                "hipModuleLaunchKernel");
}

void *mgpuStreamCreate(void) {
  if (sev_tensor_graph_depth > 0) return sev_tensor_graph_stream;
  hipStream_t stream = NULL;
  sev_hip_check(hipStreamCreate(&stream), "hipStreamCreate");
  return stream;
}

void mgpuStreamSynchronize(hipStream_t stream) {
  if (sev_tensor_graph_depth > 0 && stream == sev_tensor_graph_stream) return;
  sev_hip_check(hipStreamSynchronize(stream), "hipStreamSynchronize");
}

void mgpuStreamDestroy(hipStream_t stream) {
  if (sev_tensor_graph_depth > 0 && stream == sev_tensor_graph_stream) return;
  sev_hip_check(hipStreamDestroy(stream), "hipStreamDestroy");
}
"#;
