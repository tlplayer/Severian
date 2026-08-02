# parallel

`parallel` owns execution-placement contracts used by numerical and systems
libraries. Library implementations may place internal tasks with `gpu`, `simd`,
or `simt`:

```sev
import parallel

with self and simd:
    result = async internalKernel(values)
```

Ordinary model callers should not need these controls. `matrix`, `tensor`, and
`models` expose algebraic operations; the compiler selects legal backend
candidates after fusion and shape analysis.

- `simd` means vector lanes executing one instruction over multiple values.
- `simt` means many logical lanes with independent control state.
- `gpu` means a device-kernel placement, normally using SIMT lowering.

These placements are retained in HIR and MLIR. The current native backend
labels and executes a CPU fallback rather than claiming device acceleration.

Kernel fusion is not a placement and is not requested with `with`. Compatible
model operations are fused automatically. Writing `fuse` in a task context is
therefore rejected with a diagnostic explaining that optimization belongs to
the model/tensor pipeline.

Data, model/tensor, and pipeline parallelism describe partitioning across tasks
and devices rather than execution of one kernel. Those future plans require
explicit shard shapes, transfer ownership, collectives, barriers, and failure
semantics.
