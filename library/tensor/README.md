# tensor

`tensor` provides portable, shape-oriented kernels used by numerical and
machine-learning packages. It now exposes first-class `Tensor[f64]` values
with contiguous storage and runtime rank/shape metadata. Ranked ReLU, addition,
and matrix multiplication lower to executable MLIR `linalg` kernels. Flat
`list[float]` algorithms remain available as a portable reference layer.

The public operations include ReLU, leaky ReLU, fast sigmoid, fast tanh, GELU,
Swish, backward kernels, activation Jacobians, task-parallel ReLU,
matrix-vector multiplication, and vector addition. The sigmoid and tanh
implementations are inexpensive rational approximations suitable for showing
portable lowering; exact transcendental variants will use math intrinsics.

## Lowering direction

Ranked kernels cross a typed native ABI into MLIR memref descriptors and execute
lowered `linalg.generic` or `linalg.matmul` code on the CPU. Severian decorators
import a package's syntax symbols; they are not Python-style wrappers or
execution-policy annotations. The `parallel` package enables task-local `simd`,
`simt`, and `gpu` requests for library kernels. Existing compatible list-based
activation chains fuse automatically; ranked-kernel fusion, vectorization, and
GPU lowering are the next compiler stages.

The API keeps mathematical behavior independent of execution placement, so
`neuralnet` can reuse the same activation and Jacobian definitions as those
backends are added.
