# ONNX gold-model comparison

This benchmark trains a small `4 -> 12 -> 3` ReLU classifier on the real
[UCI Iris dataset](https://archive.ics.uci.edu/dataset/53/iris) (CC BY 4.0;
DOI `10.24432/C56C76`), exports it as ONNX, checks the graph, and generates
equivalent Severian source from the ONNX initializers. The graph deliberately
uses the currently executable intersection of both stacks:
`Gemm -> Relu -> Gemm`.

It repeats the 150 observations to run 60,000 inferences. Native Severian uses
four `with self and local:` shards. PyTorch and ONNX Runtime use the same ONNX
weights and normalized inputs. Before accepting timing samples, the runner
checks output shape, exact predicted-class counts, and floating-point logit
checksums within a documented tolerance.

The report separates fresh-process time from warm PyTorch/ONNX Runtime model
calls. Native Severian currently reports its complete executable time because
the language does not yet expose a monotonic benchmark clock.

A sequential Severian executable provides a control for the four-shard local
version. Their ratio shows whether task distribution helps before comparing the
scalar list kernel with batched framework GEMMs.

## Clarity delta

PyTorch expresses the mathematical model most directly:

```python
model = torch.nn.Sequential(
    torch.nn.Linear(4, 12),
    torch.nn.ReLU(),
    torch.nn.Linear(12, 3),
)
logits = model(features)
```

Severian now gives the ONNX activation its model-domain spelling while making
execution topology and lifetime visible:

```sev
@models(Relu)
def inferChunk(...):
    hidden = Relu(add(matVec(...), hiddenBias))

with self and local:
    first = async inferChunk(...)
    second = async inferChunk(...)
    firstValues = await first
    secondValues = await second
```

Today the PyTorch version is clearer for model algebra and automatic batching.
The Severian version is clearer about ownership, placement, deterministic shard
boundaries, and join order. An ONNX graph importer plus first-class batched
tensor operations should remove most of the handwritten Severian algebra
without hiding its execution contract.

Prepare and run with a Python environment containing PyTorch, ONNX, ONNX
Runtime, and NumPy:

```sh
python3 -m venv --system-site-packages /tmp/severian-onnx-venv
/tmp/severian-onnx-venv/bin/pip install -r bench/onnx-gold/requirements.txt
/tmp/severian-onnx-venv/bin/python bench/onnx-gold/prepare.py
/tmp/severian-onnx-venv/bin/python bench/onnx-gold/run.py \
  --torch-python /tmp/severian-onnx-venv/bin/python
```

Generated data, ONNX, and Severian source remain under the ignored
`bench/onnx-gold/generated` directory. This is an import/code-generation
prototype, not an in-language protobuf reader. Convolution, attention, dynamic
shapes, and ONNX operator coverage remain separate compiler/runtime work.
