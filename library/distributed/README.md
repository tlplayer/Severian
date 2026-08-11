# distributed

`distributed` owns deterministic shard planning independently of tensor or
model code. The current backend maps workers to native Severian tasks in one
process. Future placement implementations can map the same worker indices to
remote processes, GPU workgroups, or VM-isolated workers.

Placement can be scoped across a group of task launches alongside their
lifetime owner:

```sev
with self and local:
    first = async process_shard(first_values)
    second = async process_shard(second_values)
    first_result = await first
    second_result = await second
```

`self` gives every enclosed task a structured lifetime. `local` selects the
native pthread-backed scheduler for the scoped fan-out, and lowering preserves
it as a `severian_distribution = "local"` attribute on each spawn call. The
placement symbol requires `import distributed`. A one-off task may still write
`async process_shard(values) with self and local` directly.

Remote execution still requires serialization, transport, cancellation, and
retry semantics rather than a fake networking stub, so `REMOTE` is not exposed
as an executable placement yet.
