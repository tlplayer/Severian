# Tensor and neural-network lowering example

This package imports `tensor` and `neuralnet` as path dependencies and runs the
same dense/ReLU computation through kernels annotated for SIMD and GPU policy.
Both annotations currently execute the portable CPU implementation. Their HIR
metadata defines where future MLIR vector and GPU/NVVM lowering selects a
backend without changing the neural-network API.

The example also runs a task-parallel activation and prints a flattened input
Jacobian. Run it with:

```sh
cargo run -p severian-driver --bin sev -- compile \
  docs/examples/18-tensor-neuralnet/main.sev \
  -o bin/examples/18-tensor-neuralnet/main
bin/examples/18-tensor-neuralnet/main
```

Inspect the backend selection points with:

```sh
cargo run -p severian-driver --bin sev -- emit-mlir \
  docs/examples/18-tensor-neuralnet/main.sev | rg severian_tensor_policy
```
