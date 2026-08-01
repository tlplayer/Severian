# distributed

`distributed` owns deterministic shard planning independently of tensor or
model code. The current backend maps workers to native Severian tasks in one
process. Future placement implementations can map the same worker indices to
remote processes, GPU workgroups, or VM-isolated workers.

Placement belongs on the task launch, alongside its lifetime owner:

```sev
task = async processShard(values) with self and local
```

`self` gives the task a structured lifetime. `local` selects the native
pthread-backed scheduler at the exact point where work fans out, and lowering
preserves it as a `severian_distribution = "local"` call attribute. The
placement symbol requires `import distributed`.

Remote execution still requires serialization, transport, cancellation, and
retry semantics rather than a fake networking stub, so `REMOTE` is not exposed
as an executable placement yet.
