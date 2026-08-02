# Matrix notation and automatic model fusion

This example has no user-directed optimization syntax. `matrix` imports the
linear-algebra symbols `X`, `^`, `I`, and `J`; `models` imports activation
names. The model is written as ordinary composition:

```sev
return Swish(FastTanh(Relu(X)))
```

The `models` package declares these functions as compatible members of the
tensor elementwise pipeline in its package metadata. The generic compiler pass
therefore replaces three list traversals with one opaque `FusedPipeline` HIR
operation. Native lowering emits one runtime traversal with automatic `simd`,
`simt`, and `gpu` candidates plus an explicit CPU fallback. Model callers do not
select a backend or request fusion, while the compiler contains no table of
model function names.

The example also uses the explicit constructors requested by the packages:
`from matrix import matrix` and `from tensor import tensor`.

The directory name is retained to avoid renumbering the example inventory; its
content now demonstrates the corrected library-driven parallel architecture.
