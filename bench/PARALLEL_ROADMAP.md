# Tensor, fusion, and parallel-backend roadmap

The first automatic activation-fusion experiment is measured, not
hypothetical. On a 262,144-value pipeline, compiling the nested model expression
`Swish(FastTanh(Relu(X)))` to one traversal reduced median fresh-process time
from 37.574 ms to 22.357 ms, a 1.681x speedup. Both native executables produced
the exact same checked stdout fixture.

## What fuses today

Fusion is a compiler decision, not a task capability and not syntax the model
author requests. The `models` and `tensor` manifests declare which operations
share a native elementwise pipeline. A generic compiler pass consumes those
rules and creates one opaque `FusedPipeline` HIR operation. Native lowering
traverses the input once and applies each scalar stage before storing the
output:

```sev
@models(Relu, FastTanh, Swish)
def forward(X: list[float]) -> list[float]:
    return Swish(FastTanh(Relu(X)))
```

Explicit bindings are materialization boundaries and therefore make a useful
unfused control. Calls with side effects, incompatible arity, or unknown
functions are not fused. User-written `fuse` placement is rejected; `local`,
`gpu`, `simd`, and `simt` remain execution-placement contracts for library and
compiler kernels.

The emitted operation currently records `simd`, `simt`, and `gpu` as lowering
candidates but executes the CPU fallback. This is honest intent metadata, not a
claim that hardware-specific lowering exists already.

## The library boundary

`tensor` owns ranked storage, contraction, and elementwise operations, while
`model` supplies domain names
such as `Relu` and `Swish`. Model code composes those operations; package
metadata registers their resolved identities with generic compiler passes,
which choose fusion and backend lowering. Hardware decisions do not leak into
ordinary model source, and new aliases do not require driver edits.

Ranked tensors use contiguous storage with explicit shapes and strides. This is
the sole numerical representation used by compiler kernels and model fusion.

## Why warm PyTorch is still faster

In the ONNX gold test, warm PyTorch measured 1.892 ms and ONNX Runtime measured
0.314 ms per model call. Severian measured 156.449 ms for a complete fresh
four-shard executable, so that is not a direct kernel-only comparison. The
backend gap is nevertheless real:

1. PyTorch and ONNX Runtime execute contiguous batched matrix multiplications;
   Severian performs one small matrix-vector operation per sample.
2. Severian `list[float]` elements are dynamically boxed, so indexed arithmetic
   crosses runtime helpers and allocates scalar results.
3. Mature frameworks dispatch tuned vectorized kernels and persistent worker
   pools; Severian currently emits scalar LLVM loops and pthread tasks.
4. Framework autodiff tracks tensor operations and invokes specialized backward
   kernels rather than interpreting Python scalar arithmetic.
5. Severian has no persistent model/session benchmark API yet.

Autograd itself is not the source of PyTorch's speed. The advantage comes from
tensor-granularity graphs and optimized forward/backward kernels.

## Backend sequence

1. Use ranked contiguous tensor storage with explicit shape,
   strides, and unboxed element access.
2. Make tensor operations produce typed graph nodes and lower matrix algebra to
   MLIR `tensor`/`linalg` operations.
3. Generalize activation fusion with legality checks for shapes, aliasing, side
   effects, broadcasts, reductions, and materialization boundaries.
4. Add whole expressions such as affine + bias + activation to the same graph
   fusion system, driven by library operation identities rather than user hints.
5. Lower CPU candidates through MLIR vectorization and target-specific SIMD,
   including scalar remainder loops.
6. Outline GPU/SIMT kernels with explicit transfer, logical-lane mapping,
   synchronization, and async completion semantics.
7. Build reverse-mode autodiff over tensor HIR, then fuse backward graphs such
   as activation masks plus transposed matrix products.
8. Add persistent benchmark sessions for direct warm-kernel comparisons.
