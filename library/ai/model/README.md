# model

`model` is the single author-facing model namespace. `model.load(...)`
constructs a network from a local artifact or a verified cached model, while
`file.read(...)` remains the lower-level way to inspect a file. Imported model
formats are converted into Severian metadata and NeuralNet IR; they are never
treated as runtime Python, PyTorch, TensorFlow, or Hugging Face dependencies.

```sev
import model
from model import neuralnet as nn

X = model.tensor([1.0, 2.0], [1, 2])
linear = nn.Linear(model.tensor([1.0, 0.0, 0.0, 1.0], [2, 2]), model.tensor([0.0, 0.0], [2]))
Y = linear.forward(X)
```

Checkpoint models own the matching tokenizer. Encoding produces a typed
`TokenBatch`; `forward` returns owned logits and the predicted next token, while
`generate` provides the ordinary high-level text path:

```sev
net = model.load("hf://Qwen/Qwen2.5-3B")
tokens = net.tokenizer.encode("Hello")
output = net.forward(tokens)
print(output.next_token)
output.close()

text = net.generate("Hello", maximum_new_tokens=32)
```

Token batches remain host-side until execution. The current checkpoint-backed
Qwen2.5 program materializes a `[1, 256]` token tensor and matching causal
attention mask at `forward`/`session` time. `forward_kernel` remains available
for architecture work that supplies KV caches, rotary values, and masks
directly.

The loader implements native importers for Hugging Face/safetensors, ONNX
`ModelProto`, Keras `.keras` archives, restricted PyTorch state dictionaries,
and native `.sevmodel` artifacts. It includes ZIP/DEFLATE and protobuf readers,
an allowlisted weights-only pickle machine, graph validation, and a Severian
graph interpreter. Keras HDF5 weight bytes remain inert data associated with
the imported layer graph; no custom Keras object code is executed.

`hf://owner/model@revision` resolves a revision to a full immutable commit,
checks each LFS SHA-256, and caches by commit. Direct HTTPS loading requires an
explicit `sha256` argument. Model repositories never execute Python, PyTorch,
TensorFlow, Keras, or Hugging Face runtime code.

The package also exposes the migration corpus directly: tensor operations,
neural layers, losses, optimizers and schedules, datasets/model selection,
preprocessing, metrics, regression/classification, trees and ensembles,
clustering, PCA/SVD/NMF, and anomaly estimators. These are Severian
implementations rather than framework adapters.

Native masked-audio models can import `model.speech` and reuse the
OmniVoice-compatible speech helpers
instead of reimplementing generation in each application. The surface includes
time-shifted unmask schedules, classifier-free guidance, Gumbel selection,
codebook-layer penalties, reference-rate duration estimates, voice-clone prompt
data, and the token-selection primitives used by iterative masked decoding.

Compiler lowering of imported `ModelGraph` nodes to MLIR, Triton, and XLA is a
separate pass. The importers deliberately produce a compiler-owned IR now so
that later lowering does not change the public API or add framework runtimes.

The lower-level `tensor` and `models` packages remain available for compiler
fixtures, graph construction, autodiff experiments, and backend work.

## Examples

- `examples/transformer_classifier.sev` is a native translation of a PyTorch
  transformer sequence-classification workload. It imports only `model` and
  includes synthetic data generation, sinusoidal positions, transformer
  blocks, clipped classifier-head training, warmup/cosine scheduling,
  validation, checkpoint restore, attention inspection, and inference.
