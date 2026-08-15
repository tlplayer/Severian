pub(super) const SOURCE: &str = r#"
/* Native reference execution stores every numeric tensor as doubles. Typed
 * entry points preserve the source-level dtype while sharing that storage. */
void *__sev_tensor_to_f32(void *tensor_raw) { return tensor_raw; }
void *__sev_tensor_to_f64(void *tensor_raw) { return tensor_raw; }
void *__sev_tensor_to_i64(void *tensor_raw) { return tensor_raw; }
void *__sev_tensor_to_bf16(void *tensor_raw) { return tensor_raw; }
void *__sev_tensor_bf16_to_f32(void *tensor_raw) { return tensor_raw; }
void *__sev_tensor_f32_to_bf16(void *tensor_raw) { return tensor_raw; }
void *__sev_tensor_bf16_shape(void *tensor_raw) { return __sev_tensor_shape(tensor_raw); }

static sev_tensor *sev_tensor_typed_unary(sev_tensor *input, int operation) {
  sev_tensor *output = sev_tensor_allocate(input->rank, input->shape);
  for (int64_t index = 0; index < input->size; ++index) {
    double value = input->data[sev_tensor_offset(input, index)];
    output->data[index] = operation == 0 ? cos(value) : operation == 1 ? sin(value) : exp(value);
  }
  return output;
}

void *__sev_tensor_f32_cosine(void *input_raw) { return sev_tensor_typed_unary(input_raw, 0); }
void *__sev_tensor_f32_sine(void *input_raw) { return sev_tensor_typed_unary(input_raw, 1); }
void *__sev_tensor_f32_exp(void *input_raw) { return sev_tensor_typed_unary(input_raw, 2); }

static sev_tensor *sev_tensor_typed_binary(sev_tensor *left, sev_tensor *right, int operation) {
  int64_t rank = 0;
  int64_t *shape = sev_tensor_broadcast_shape(left, right, &rank);
  sev_tensor *output = sev_tensor_allocate(rank, shape);
  free(shape);
  for (int64_t index = 0; index < output->size; ++index) {
    double left_value = left->data[sev_tensor_broadcast_offset(left, output, index)];
    double right_value = right->data[sev_tensor_broadcast_offset(right, output, index)];
    output->data[index] = operation == 0 ? left_value * right_value
                        : operation == 1 ? left_value - right_value
                                         : left_value / right_value;
  }
  return output;
}

void *__sev_tensor_f32_multiply(void *left, void *right) {
  return sev_tensor_typed_binary(left, right, 0);
}
void *__sev_tensor_f32_subtract(void *left, void *right) {
  return sev_tensor_typed_binary(left, right, 1);
}
void *__sev_tensor_f32_divide(void *left, void *right) {
  return sev_tensor_typed_binary(left, right, 2);
}

void *__sev_tensor_f32_scale(void *input_raw, double scale) {
  sev_tensor *input = input_raw;
  sev_tensor *output = sev_tensor_allocate(input->rank, input->shape);
  for (int64_t index = 0; index < input->size; ++index)
    output->data[index] = input->data[sev_tensor_offset(input, index)] * scale;
  return output;
}

static sev_tensor *sev_tensor_typed_broadcast(sev_tensor *input, const int64_t *shape, int64_t rank) {
  sev_tensor *output = sev_tensor_allocate(rank, shape);
  for (int64_t axis = 0; axis < input->rank; ++axis) {
    int64_t output_axis = rank - input->rank + axis;
    if (output_axis < 0 || (input->shape[axis] != 1 && input->shape[axis] != shape[output_axis])) abort();
  }
  for (int64_t index = 0; index < output->size; ++index)
    output->data[index] = input->data[sev_tensor_broadcast_offset(input, output, index)];
  return output;
}

void *__sev_tensor_f32_broadcast_like(void *input_raw, void *target_raw) {
  sev_tensor *target = target_raw;
  return sev_tensor_typed_broadcast(input_raw, target->shape, target->rank);
}

void *__sev_tensor_bf16_broadcast(void *input_raw, void *shape_raw) {
  sev_collection *values = shape_raw;
  int64_t *shape = sev_allocate((size_t)values->size * sizeof(*shape));
  for (int64_t axis = 0; axis < values->size; ++axis)
    shape[axis] = __sev_unbox_i64(values->items[axis]);
  sev_tensor *output = sev_tensor_typed_broadcast(input_raw, shape, values->size);
  free(shape);
  return output;
}

void *__sev_tensor_f32_slice(void *input_raw, void *starts_raw, void *limits_raw, void *strides_raw) {
  sev_tensor *input = input_raw;
  sev_collection *limits = limits_raw;
  if (limits->size != input->rank) abort();
  sev_collection *normalized = __sev_collection_new(0);
  for (int64_t axis = 0; axis < limits->size; ++axis) {
    int64_t limit = __sev_unbox_i64(limits->items[axis]);
    __sev_collection_push(normalized, __sev_box_i64(limit < 0 ? input->shape[axis] : limit));
  }
  return __sev_tensor_slice(input_raw, starts_raw, normalized, strides_raw);
}

void *__sev_tensor_f32_concatenate(void *values_raw, int64_t axis) {
  sev_collection *values = values_raw;
  if (!values || values->size == 0) abort();
  sev_tensor *first = (sev_tensor *)values->items[0];
  if (axis < 0) axis += first->rank;
  if (axis < 0 || axis >= first->rank) abort();
  int64_t *shape = sev_allocate((size_t)first->rank * sizeof(*shape));
  memcpy(shape, first->shape, (size_t)first->rank * sizeof(*shape));
  shape[axis] = 0;
  for (int64_t item = 0; item < values->size; ++item) {
    sev_tensor *input = (sev_tensor *)values->items[item];
    if (!input || input->rank != first->rank) abort();
    for (int64_t dimension = 0; dimension < input->rank; ++dimension)
      if (dimension != axis && input->shape[dimension] != first->shape[dimension]) abort();
    shape[axis] += input->shape[axis];
  }
  sev_tensor *output = sev_tensor_allocate(first->rank, shape);
  free(shape);
  int64_t axis_offset = 0;
  for (int64_t item = 0; item < values->size; ++item) {
    sev_tensor *input = (sev_tensor *)values->items[item];
    for (int64_t linear = 0; linear < input->size; ++linear) {
      int64_t remainder = linear;
      int64_t output_offset = 0;
      for (int64_t dimension = input->rank - 1; dimension >= 0; --dimension) {
        int64_t coordinate = remainder % input->shape[dimension];
        remainder /= input->shape[dimension];
        if (dimension == axis) coordinate += axis_offset;
        output_offset += coordinate * output->strides[dimension];
      }
      output->data[output_offset] = input->data[sev_tensor_offset(input, linear)];
    }
    axis_offset += input->shape[axis];
  }
  return output;
}

static sev_tensor *sev_tensor_typed_reduce_last(sev_tensor *input, bool maximum) {
  if (!input || input->rank < 2 || input->shape[input->rank - 1] <= 0) abort();
  sev_tensor *output = sev_tensor_allocate(input->rank - 1, input->shape);
  int64_t width = input->shape[input->rank - 1];
  for (int64_t outer = 0; outer < output->size; ++outer) {
    double value = maximum ? -INFINITY : 0.0;
    for (int64_t column = 0; column < width; ++column) {
      double candidate = input->data[sev_tensor_offset(input, outer * width + column)];
      value = maximum ? (candidate > value ? candidate : value) : value + candidate;
    }
    output->data[outer] = value;
  }
  return output;
}

void *__sev_tensor_f32_max_last(void *input) { return sev_tensor_typed_reduce_last(input, true); }
void *__sev_tensor_f32_sum_last(void *input) { return sev_tensor_typed_reduce_last(input, false); }

void *__sev_tensor_bf16_transpose(void *input_raw, void *axes_raw) {
  sev_tensor *input = input_raw;
  sev_collection *axes = axes_raw;
  if (!input || axes->size != input->rank) abort();
  int64_t *shape = sev_allocate((size_t)input->rank * sizeof(*shape));
  for (int64_t axis = 0; axis < input->rank; ++axis) {
    int64_t source_axis = __sev_unbox_i64(axes->items[axis]);
    if (source_axis < 0 || source_axis >= input->rank) abort();
    shape[axis] = input->shape[source_axis];
  }
  sev_tensor *output = sev_tensor_allocate(input->rank, shape);
  free(shape);
  for (int64_t linear = 0; linear < output->size; ++linear) {
    int64_t remainder = linear;
    int64_t input_offset = 0;
    for (int64_t axis = output->rank - 1; axis >= 0; --axis) {
      int64_t coordinate = remainder % output->shape[axis];
      remainder /= output->shape[axis];
      int64_t source_axis = __sev_unbox_i64(axes->items[axis]);
      input_offset += coordinate * input->strides[source_axis];
    }
    output->data[linear] = input->data[input_offset];
  }
  return output;
}

void *__sev_tensor_f32_batched_matmul(void *left_raw, void *right_raw) {
  sev_tensor *left = left_raw;
  sev_tensor *right = right_raw;
  if (!left || !right || left->rank != right->rank || left->rank < 2) abort();
  int64_t rank = left->rank;
  for (int64_t axis = 0; axis < rank - 2; ++axis)
    if (left->shape[axis] != right->shape[axis]) abort();
  int64_t rows = left->shape[rank - 2];
  int64_t shared = left->shape[rank - 1];
  int64_t columns = right->shape[rank - 1];
  if (shared != right->shape[rank - 2]) abort();
  int64_t *shape = sev_allocate((size_t)rank * sizeof(*shape));
  memcpy(shape, left->shape, (size_t)rank * sizeof(*shape));
  shape[rank - 1] = columns;
  sev_tensor *output = sev_tensor_allocate(rank, shape);
  free(shape);
  int64_t batches = left->size / (rows * shared);
  for (int64_t batch = 0; batch < batches; ++batch)
    for (int64_t row = 0; row < rows; ++row)
      for (int64_t column = 0; column < columns; ++column) {
        double total = 0.0;
        for (int64_t inner = 0; inner < shared; ++inner) {
          int64_t left_index = batch * rows * shared + row * shared + inner;
          int64_t right_index = batch * shared * columns + inner * columns + column;
          total += left->data[sev_tensor_offset(left, left_index)]
                 * right->data[sev_tensor_offset(right, right_index)];
        }
        output->data[batch * rows * columns + row * columns + column] = total;
      }
  return output;
}

void *__sev_tensor_f32_matmul(void *left_raw, void *right_raw) {
  sev_tensor *left = left_raw;
  sev_tensor *right = right_raw;
  if (!left || !right || left->rank < 2 || right->rank < 2) abort();
  int64_t left_batch_rank = left->rank - 2;
  int64_t right_batch_rank = right->rank - 2;
  int64_t batch_rank = left_batch_rank > right_batch_rank ? left_batch_rank : right_batch_rank;
  int64_t rank = batch_rank + 2;
  int64_t *shape = sev_allocate((size_t)rank * sizeof(*shape));
  for (int64_t axis = 0; axis < batch_rank; ++axis) {
    int64_t left_axis = axis - (batch_rank - left_batch_rank);
    int64_t right_axis = axis - (batch_rank - right_batch_rank);
    int64_t left_size = left_axis < 0 ? 1 : left->shape[left_axis];
    int64_t right_size = right_axis < 0 ? 1 : right->shape[right_axis];
    if (left_size != right_size && left_size != 1 && right_size != 1) abort();
    shape[axis] = left_size > right_size ? left_size : right_size;
  }
  int64_t rows = left->shape[left->rank - 2];
  int64_t shared = left->shape[left->rank - 1];
  int64_t columns = right->shape[right->rank - 1];
  if (shared != right->shape[right->rank - 2]) abort();
  shape[rank - 2] = rows;
  shape[rank - 1] = columns;
  sev_tensor *output = sev_tensor_allocate(rank, shape);
  free(shape);
  int64_t *coordinates = sev_allocate((size_t)rank * sizeof(*coordinates));
  for (int64_t linear = 0; linear < output->size; ++linear) {
    int64_t remainder = linear;
    for (int64_t axis = rank - 1; axis >= 0; --axis) {
      coordinates[axis] = remainder % output->shape[axis];
      remainder /= output->shape[axis];
    }
    double total = 0.0;
    for (int64_t inner = 0; inner < shared; ++inner) {
      int64_t left_offset = coordinates[rank - 2] * left->strides[left->rank - 2]
                          + inner * left->strides[left->rank - 1];
      for (int64_t axis = 0; axis < left_batch_rank; ++axis) {
        int64_t output_axis = batch_rank - left_batch_rank + axis;
        int64_t coordinate = left->shape[axis] == 1 ? 0 : coordinates[output_axis];
        left_offset += coordinate * left->strides[axis];
      }
      int64_t right_offset = inner * right->strides[right->rank - 2]
                           + coordinates[rank - 1] * right->strides[right->rank - 1];
      for (int64_t axis = 0; axis < right_batch_rank; ++axis) {
        int64_t output_axis = batch_rank - right_batch_rank + axis;
        int64_t coordinate = right->shape[axis] == 1 ? 0 : coordinates[output_axis];
        right_offset += coordinate * right->strides[axis];
      }
      total += left->data[left_offset] * right->data[right_offset];
    }
    output->data[linear] = total;
  }
  free(coordinates);
  return output;
}

void *__sev_tensor_bf16_dynamic_update_slice(void *input_raw, void *update_raw, void *starts_raw) {
  sev_tensor *input = input_raw;
  sev_tensor *update = update_raw;
  sev_collection *starts = starts_raw;
  if (!input || !update || input->rank != update->rank || starts->size != input->rank) abort();
  sev_tensor *output = sev_tensor_materialize(input);
  for (int64_t linear = 0; linear < update->size; ++linear) {
    int64_t remainder = linear;
    int64_t output_offset = 0;
    for (int64_t axis = update->rank - 1; axis >= 0; --axis) {
      int64_t coordinate = remainder % update->shape[axis];
      remainder /= update->shape[axis];
      coordinate += __sev_unbox_i64(starts->items[axis]);
      if (coordinate < 0 || coordinate >= input->shape[axis]) abort();
      output_offset += coordinate * output->strides[axis];
    }
    output->data[output_offset] = update->data[sev_tensor_offset(update, linear)];
  }
  return output;
}
"#;
