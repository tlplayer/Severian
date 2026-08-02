# parallel

`parallel` owns operation-local execution placement and kernel-fusion
contracts. Importing it enables `gpu`, `simd`, `simt`, and `fuse` in an async
task context:

```sev
import parallel

with self and gpu and fuse:
    result = async fusedDenseRelu(weights, rows, columns, bias, X)

single = async fusedDenseRelu(weights, rows, columns, bias, X) with gpu and fuse
```

- `simd` requests vector lanes executing one instruction over multiple values.
- `simt` requests many logical lanes with independent control state.
- `gpu` requests a device kernel and will normally lower through SIMT-oriented
  GPU dialects.
- `fuse` asks the optimizer to keep compatible producer/consumer operations in
  one kernel.

The compiler preserves these choices as attributes on the task-spawn operation.
The current executable backend intentionally records
`severian_device_fallback = "cpu"` and runs the task through the existing
pthread runtime. This makes programs executable without claiming that a GPU or
vector backend exists yet.

The package also contains manually fused reference kernels. They establish the
semantics and provide a performance control for later automatic fusion passes.
`fusedDenseRelu` combines matrix-vector multiplication, bias addition, and ReLU
without allocating intermediate lists. `fusedDenseReluBackwardInput` combines
the ReLU mask with the transposed matrix-vector product used for input
gradients.

## Other parallel paradigms

`simd`, `simt`, and `gpu` describe how one task executes. Data parallelism,
model/tensor parallelism, pipeline parallelism, work stealing, and remote
collectives describe how many tasks and tensors are partitioned. They should be
higher-level plans in this package rather than additional device-placement
words. Their implementation needs explicit shard shapes, transfer ownership,
barriers, reductions, failure behavior, and collectives such as broadcast and
all-reduce; none are silently treated as working placements yet.
