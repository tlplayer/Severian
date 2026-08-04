pub(crate) fn source(rocm: bool) -> String {
    let mut source = String::new();
    if rocm {
        source.push_str("#define SEV_MODEL_GRAPH_ROCM 1\n");
    }
    source.push_str(MODEL_GRAPH_SOURCE);
    source
}

const MODEL_GRAPH_SOURCE: &str = r#"
typedef enum {
  SEV_GRAPH_INPUT,
  SEV_GRAPH_RELU,
  SEV_GRAPH_ADD,
  SEV_GRAPH_MATMUL,
  SEV_GRAPH_TRANSPOSE,
  SEV_GRAPH_SCALE,
  SEV_GRAPH_SOFTMAX_ROWS,
  SEV_GRAPH_LAYER_NORM
} sev_graph_operation;

typedef struct sev_graph_node {
  uint64_t magic;
  sev_graph_operation operation;
  struct sev_graph_node *left;
  struct sev_graph_node *right;
  void *input;
  void *result;
  double scalar;
  uint64_t result_epoch;
  uint64_t active_epoch;
} sev_graph_node;

#define SEV_GRAPH_MAGIC UINT64_C(0x5345564752415048)

static uint64_t sev_graph_epoch = 0;

static sev_graph_node *sev_graph_require(void *raw) {
  sev_graph_node *node = raw;
  if (!node || node->magic != SEV_GRAPH_MAGIC) abort();
  return node;
}

static sev_graph_node *sev_graph_new(sev_graph_operation operation,
                                     sev_graph_node *left,
                                     sev_graph_node *right,
                                     double scalar) {
  sev_graph_node *node = sev_allocate(sizeof(*node));
  node->magic = SEV_GRAPH_MAGIC;
  node->operation = operation;
  node->left = left;
  node->right = right;
  node->scalar = scalar;
  return node;
}

void *__sev_model_graph_input(void *value) {
  sev_graph_node *node = sev_graph_new(SEV_GRAPH_INPUT, NULL, NULL, 0.0);
  node->input = value;
  return node;
}

void *__sev_model_graph_relu(void *value) {
  return sev_graph_new(SEV_GRAPH_RELU, sev_graph_require(value), NULL, 0.0);
}

void *__sev_model_graph_add(void *left, void *right) {
  return sev_graph_new(SEV_GRAPH_ADD, sev_graph_require(left), sev_graph_require(right), 0.0);
}

void *__sev_model_graph_matmul(void *left, void *right) {
  return sev_graph_new(SEV_GRAPH_MATMUL, sev_graph_require(left), sev_graph_require(right), 0.0);
}

void *__sev_model_graph_transpose(void *value) {
  sev_graph_node *input = sev_graph_require(value);
  if (input->operation == SEV_GRAPH_TRANSPOSE) return input->left;
  return sev_graph_new(SEV_GRAPH_TRANSPOSE, input, NULL, 0.0);
}

void *__sev_model_graph_scale(void *value, double scale) {
  sev_graph_node *input = sev_graph_require(value);
  if (scale == 1.0) return input;
  if (input->operation == SEV_GRAPH_SCALE)
    return sev_graph_new(SEV_GRAPH_SCALE, input->left, NULL, input->scalar * scale);
  return sev_graph_new(SEV_GRAPH_SCALE, input, NULL, scale);
}

void *__sev_model_graph_softmax_rows(void *value) {
  return sev_graph_new(SEV_GRAPH_SOFTMAX_ROWS, sev_graph_require(value), NULL, 0.0);
}

void *__sev_model_graph_layer_norm(void *value, double epsilon) {
  if (epsilon <= 0.0) abort();
  return sev_graph_new(SEV_GRAPH_LAYER_NORM, sev_graph_require(value), NULL, epsilon);
}

static void *sev_graph_execute(sev_graph_node *node, uint64_t epoch) {
  if (node->result_epoch == epoch) return node->result;
  if (node->active_epoch == epoch) abort();
  node->active_epoch = epoch;
  void *left = node->left ? sev_graph_execute(node->left, epoch) : NULL;
  void *right = node->right ? sev_graph_execute(node->right, epoch) : NULL;
  switch (node->operation) {
    case SEV_GRAPH_INPUT: node->result = node->input; break;
    case SEV_GRAPH_RELU: node->result = __sev_tensor_relu(left); break;
    case SEV_GRAPH_ADD: node->result = __sev_tensor_add(left, right); break;
    case SEV_GRAPH_MATMUL: node->result = __sev_tensor_matmul(left, right); break;
    case SEV_GRAPH_TRANSPOSE: node->result = __sev_tensor_transpose(left); break;
    case SEV_GRAPH_SCALE: node->result = __sev_tensor_scale(left, node->scalar); break;
    case SEV_GRAPH_SOFTMAX_ROWS: node->result = __sev_tensor_softmax_rows(left); break;
    case SEV_GRAPH_LAYER_NORM: node->result = __sev_tensor_layer_norm(left, node->scalar); break;
  }
  node->result_epoch = epoch;
  node->active_epoch = 0;
  return node->result;
}

void *__sev_model_graph_run(void *output) {
  sev_graph_node *node = sev_graph_require(output);
  uint64_t epoch = ++sev_graph_epoch;
  if (!epoch) epoch = ++sev_graph_epoch;
  sev_tensor_graph_begin();
  void *result = sev_graph_execute(node, epoch);
  sev_tensor_graph_end();
  return result;
}
"#;
