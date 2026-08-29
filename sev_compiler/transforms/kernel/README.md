# Severian kernel IR

This package owns the object representation used between tensor fusion and
target MLIR lowering:

```text
LogicalKernel
  -> layout/scheduling transformations
ScheduledKernel + IndexOperation SSA graph
  -> GPU MLIR object lowering
GpuMlirModule/GpuMlirOperation
  -> terminal MLIR serialization
```

Kernel transformations operate on `.sev` objects. Dtype, rank, dimensions,
launch geometry, masks, and scalar operations remain data. The package does
not construct TTIR, call Triton, or encode dtype/rank in operation names.
