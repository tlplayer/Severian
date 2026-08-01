# Tensor and neural-network lowering example

This package imports `tensor` and `neuralnet` as path dependencies and runs a
dense/ReLU computation through the portable CPU implementation.

The example also runs a task-parallel activation and prints a flattened input
Jacobian. Run it with:

```sh
cargo run -p severian-driver --bin sev -- compile \
  docs/examples/18-tensor-neuralnet/main.sev \
  -o bin/examples/18-tensor-neuralnet/main
bin/examples/18-tensor-neuralnet/main
```

SIMD and GPU selection are not claimed here yet. Severian decorators import
domain syntax rather than wrapping functions or selecting an execution backend;
backend selection needs an explicit operation-local language construct.
