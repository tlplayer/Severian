# Parallel and automatic-differentiation performance roadmap

The first automatic fusion experiment is now measured, not hypothetical. On
the 60,000-row Iris ONNX workload, rewriting
`Relu(add(matVec(...), bias))` to `fusedDenseRelu(...)` reduced the four-shard
median from 147.618 ms to 117.499 ms, or 20.4%. Output shape, class counts, and
logit checksums remained equal to PyTorch and ONNX Runtime.

## What `fuse` does today

`with self and local and fuse:` marks the spawned function as a fusion target.
The compiler walks that function's HIR and recognizes this graph:

```text
matVec(weights, rows, columns, X) -> add(..., bias) -> Relu(...)
```

It rewrites the graph to the `parallel.fusedDenseRelu` reference kernel. The
unfused benchmark differs only by omitting `fuse`, which makes it a useful
control. Other graphs are left unchanged; this is a safe first pattern, not a
general-purpose fusion engine.

## Why warm PyTorch is still faster

Warm PyTorch measured 1.664 ms for the same model call, but its measurement
starts after the framework and worker pools are initialized. Severian's
117.499 ms is a complete fresh executable. Startup does not explain the whole
gap, however. The important backend differences are:

1. PyTorch executes a contiguous batched matrix multiplication. Severian runs
   one small matrix-vector operation per sample.
2. Severian's `list[float]` elements are dynamically boxed. Indexed arithmetic
   crosses boxing/unboxing helpers and frequently allocates scalar results.
3. PyTorch dispatches tuned, vectorized native kernels with mature thread-pool
   scheduling. Severian currently emits scalar LLVM loops and pthread tasks.
4. PyTorch autograd builds a graph of tensor operations and invokes specialized
   backward kernels. It does not interpret one Python operation per scalar.
5. Severian has no persistent model/session API yet, so its reported timing
   includes process and runtime initialization.

Autograd itself is not the source of the speedup. The advantage comes from
tracking derivatives at tensor-operation granularity and lowering those
operations to batched forward and backward kernels.

## Backend sequence

The practical implementation order is:

1. Add ranked, contiguous `Tensor[element, shape]` storage with unboxed element
   access and explicit strides.
2. Lower tensor algebra to MLIR `linalg`/`tensor` operations instead of scalar
   boxed-list runtime calls.
3. Generalize fusion into a graph pass with legality checks for shape,
   aliasing, side effects, and reduction boundaries.
4. Lower `simd` tasks through MLIR vectorization and target-specific LLVM
   vector types, with scalar remainder loops.
5. Lower `gpu`/`simt` tasks by outlining kernels, mapping logical lanes, and
   making host/device transfer and async completion explicit.
6. Build reverse-mode autodiff over tensor HIR, then fuse patterns such as
   ReLU-mask plus transposed GEMM in the backward graph.
7. Add persistent benchmark sessions so Severian kernel time can be compared
   directly with warm PyTorch and ONNX Runtime calls.

Until steps 4 and 5 land, generated MLIR records `simd`, `simt`, and `gpu`
placement but deliberately labels execution as a CPU fallback.
