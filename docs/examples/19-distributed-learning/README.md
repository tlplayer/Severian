# Distributed learning pass

This example constructs 65,536 values, splits them deterministically across
four native tasks, runs a ReLU forward pass, and distributes its backward pass
over the same shards. Results are joined in worker order and reduced to stable
checksums.

Each shard is launched with `async ... with self and local`. `self` owns the
structured task lifetime, while `local` selects the pthread-backed local
placement at the point where work is distributed. Inputs are shared read-only.

`REMOTE` is intentionally not accepted as an executable placement yet. It will
require a transport/runtime contract covering tensor serialization, worker
failure, cancellation, retry, and gradient reduction.
