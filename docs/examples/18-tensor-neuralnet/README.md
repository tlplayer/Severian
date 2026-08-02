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

Severian decorators import domain syntax rather than wrapping functions or
selecting an execution backend. The later `21-parallel-kernels` example shows
the matrix/model library stack and automatic activation fusion. Internal
`simd`, `simt`, and `gpu` candidates currently retain an explicit CPU fallback
rather than claiming device speedup.
