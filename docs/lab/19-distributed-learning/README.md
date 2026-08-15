# Distributed learning pass

This example constructs 65,536 values, splits them deterministically across
four native tasks, runs a ReLU forward pass, and distributes its backward pass
over the same shards. Results are joined in worker order and reduced to stable
checksums.

Each fan-out uses a `with self and local:` block. `self` owns the enclosed
structured task lifetimes, while `local` selects pthread-backed placement once
for the group. The bare `async` expressions inherit that context. Inputs are
shared read-only.

`REMOTE` is intentionally not accepted as an executable placement yet. It will
require a transport/runtime contract covering tensor serialization, worker
failure, cancellation, retry, and gradient reduction.
