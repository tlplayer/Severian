# Tensor kernels and automatic model fusion

This example has no user-directed optimization syntax. `tensor` owns ranked
storage and numerical kernels; `model` imports activation names. The model is
written as ordinary composition:

```sev
return Swish(FastTanh(Relu(X)))
```

The `model` package declares these functions as compatible members of the
tensor elementwise pipeline in its package metadata. The generic compiler pass
therefore replaces three list traversals with one opaque `FusedPipeline` HIR
operation. Native lowering emits one runtime traversal with automatic `simd`,
`simt`, and `gpu` candidates plus an explicit CPU fallback. Model callers do not
select a backend or request fusion, while the compiler contains no table of
model function names.

The example uses `tensor.ranked` for storage and `tensor.rankedMatmul` for
contraction, so the compiler has one numerical representation to lower.

The directory name is retained to avoid renumbering the example inventory; its
content now demonstrates the corrected library-driven parallel architecture.
