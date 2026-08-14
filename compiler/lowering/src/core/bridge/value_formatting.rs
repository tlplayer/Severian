pub(super) const SOURCE: &str = r#"typedef struct { char *data; size_t size; size_t capacity; } sev_format_buffer;
typedef struct { uint64_t magic; int64_t rank; int64_t *shape; int64_t *strides; int64_t size; double *data; } sev_formattable_tensor;

static void sev_format_reserve(sev_format_buffer *buffer, size_t additional) {
  size_t needed = buffer->size + additional + 1;
  if (needed <= buffer->capacity) return;
  size_t capacity = buffer->capacity ? buffer->capacity : 64;
  while (capacity < needed) capacity *= 2;
  buffer->data = realloc(buffer->data, capacity);
  if (!buffer->data) sev_runtime_fail_invariant("could not allocate value formatting buffer");
  buffer->capacity = capacity;
}

static void sev_format_bytes(sev_format_buffer *buffer, const char *text, size_t size) {
  sev_format_reserve(buffer, size);
  memcpy(buffer->data + buffer->size, text, size);
  buffer->size += size;
  buffer->data[buffer->size] = '\0';
}

static void sev_format_text(sev_format_buffer *buffer, const char *text) {
  sev_format_bytes(buffer, text, strlen(text));
}

static void sev_format_float(sev_format_buffer *buffer, double value) {
  char number[64];
  snprintf(number, sizeof(number), "%.15g", value);
  sev_format_text(buffer, number);
  if (isfinite(value) && !strchr(number, '.') && !strchr(number, 'e') && !strchr(number, 'E')) sev_format_text(buffer, ".0");
}

static void sev_format_string(sev_format_buffer *buffer, const char *text, bool quoted) {
  if (!quoted) { sev_format_text(buffer, text); return; }
  sev_format_text(buffer, "\"");
  for (const unsigned char *cursor = (const unsigned char *)text; *cursor; ++cursor) {
    switch (*cursor) {
      case '\\': sev_format_text(buffer, "\\\\"); break;
      case '"': sev_format_text(buffer, "\\\""); break;
      case '\n': sev_format_text(buffer, "\\n"); break;
      case '\r': sev_format_text(buffer, "\\r"); break;
      case '\t': sev_format_text(buffer, "\\t"); break;
      default: sev_format_bytes(buffer, (const char *)cursor, 1); break;
    }
  }
  sev_format_text(buffer, "\"");
}

static void sev_format_raw(sev_format_buffer *buffer, void *raw, bool nested, int depth);

static void sev_format_collection(sev_format_buffer *buffer, sev_collection *value, int depth) {
  if (!value) { sev_format_text(buffer, "null"); return; }
  if (value->kind == 3) {
    sev_map *map = (sev_map *)value;
    sev_format_text(buffer, "{");
    for (int64_t index = 0; index < map->size; ++index) {
      if (index) sev_format_text(buffer, ", ");
      sev_format_raw(buffer, map->keys[index], true, depth + 1);
      sev_format_text(buffer, ": ");
      sev_format_raw(buffer, map->values[index], true, depth + 1);
    }
    sev_format_text(buffer, "}");
    return;
  }
  const char *open = value->kind == 1 ? "(" : value->kind == 2 ? "{" : "[";
  const char *close = value->kind == 1 ? ")" : value->kind == 2 ? "}" : "]";
  if (value->kind == 2 && value->size == 0) { sev_format_text(buffer, "set()"); return; }
  sev_format_text(buffer, open);
  for (int64_t index = 0; index < value->size; ++index) {
    if (index) sev_format_text(buffer, ", ");
    sev_format_raw(buffer, value->items[index], true, depth + 1);
  }
  if (value->kind == 1 && value->size == 1) sev_format_text(buffer, ",");
  sev_format_text(buffer, close);
}

static void sev_format_object(sev_format_buffer *buffer, sev_object *value, int depth) {
  sev_format_text(buffer, value->class_name && *value->class_name ? value->class_name : "Object");
  sev_format_text(buffer, "(");
  for (int64_t index = 0; index < value->size; ++index) {
    if (index) sev_format_text(buffer, ", ");
    sev_format_text(buffer, value->names[index]);
    sev_format_text(buffer, "=");
    sev_format_raw(buffer, value->values[index], true, depth + 1);
  }
  sev_format_text(buffer, ")");
}

static int64_t sev_format_tensor_offset(const sev_formattable_tensor *tensor, int64_t linear) {
  int64_t offset = 0;
  for (int64_t axis = tensor->rank - 1; axis >= 0; --axis) {
    int64_t coordinate = tensor->shape[axis] ? linear % tensor->shape[axis] : 0;
    if (tensor->shape[axis]) linear /= tensor->shape[axis];
    offset += coordinate * tensor->strides[axis];
  }
  return offset;
}

static void sev_format_tensor(sev_format_buffer *buffer, sev_formattable_tensor *tensor) {
  char number[64];
  sev_format_text(buffer, "Tensor(shape=[");
  for (int64_t axis = 0; axis < tensor->rank; ++axis) {
    if (axis) sev_format_text(buffer, ", ");
    snprintf(number, sizeof(number), "%ld", tensor->shape[axis]);
    sev_format_text(buffer, number);
  }
  sev_format_text(buffer, "], values=[");
  for (int64_t index = 0; index < tensor->size; ++index) {
    if (index) sev_format_text(buffer, ", ");
    sev_format_float(buffer, tensor->data[sev_format_tensor_offset(tensor, index)]);
  }
  sev_format_text(buffer, "])");
}

static void sev_format_raw(sev_format_buffer *buffer, void *raw, bool nested, int depth) {
  if (!raw) { sev_format_text(buffer, "null"); return; }
  if (depth > 12) { sev_format_text(buffer, "..."); return; }
  uint64_t magic = *(uint64_t *)raw;
  if (magic == SEV_OBJECT_MAGIC) { sev_format_object(buffer, (sev_object *)raw, depth); return; }
  if (magic == SEV_TENSOR_MAGIC) { sev_format_tensor(buffer, (sev_formattable_tensor *)raw); return; }
  if (magic == SEV_VARIANT_MAGIC) {
    sev_variant *value = raw;
    sev_format_text(buffer, value->tag ? value->tag : "variant");
    if (value->field) { sev_format_text(buffer, "("); sev_format_raw(buffer, value->field, true, depth + 1); sev_format_text(buffer, ")"); }
    return;
  }
  sev_value *value = raw;
  char number[64];
  switch (value->kind) {
    case SEV_INT: snprintf(number, sizeof(number), "%ld", value->as.i64); sev_format_text(buffer, number); return;
    case SEV_FLOAT: sev_format_float(buffer, value->as.f64); return;
    case SEV_BOOL: sev_format_text(buffer, value->as.boolean ? "true" : "false"); return;
    case SEV_STRING: sev_format_string(buffer, value->as.string, nested); return;
    case SEV_COLLECTION: sev_format_collection(buffer, value->as.pointer, depth); return;
    case SEV_NULL: sev_format_text(buffer, "null"); return;
  }
  sev_runtime_fail("E000921", "value cannot be converted to string", "the runtime value has an unknown representation");
}

void *__sev_value_string(void *raw) {
  if (raw && *(uint64_t *)raw != SEV_OBJECT_MAGIC && *(uint64_t *)raw != SEV_TENSOR_MAGIC && *(uint64_t *)raw != SEV_VARIANT_MAGIC) {
    sev_value *value = raw;
    if (value->kind == SEV_STRING) return (void *)value->as.string;
  }
  sev_format_buffer buffer = {0};
  sev_format_raw(&buffer, raw, false, 0);
  return buffer.data;
}
"#;
