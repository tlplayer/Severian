# Severian GPU MLIR slice

This example is the first direct GPU compiler path:

```text
.sev ranked tensor expression
  -> structural Tensor IR
  -> FusionRegion
  -> Severian blocked layout and warp schedule
  -> masked gpu.launch + memref MLIR
  -> MLIR GPU target lowering
```

The layout representation and default blocked policy are translated from the
pinned Triton donor, but the compiler does not load, invoke, or link Triton.
The Rust bootstrap compiler consumes the same structural policy implemented in
`sev_compiler/transforms/fusion/src/lib.sev` while the compiler is being moved
into Severian source.
