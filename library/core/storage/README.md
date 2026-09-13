# Core storage

`core.storage` depends on `core.memory` and provides checked owning buffers.
`allocate[T](count)` validates the allocation size. `free(buffer)` consumes the
buffer; dropping it or leaving its scope also releases the allocation through
the existing compiler ownership lowering.

`core.memory` supplies the allocation boundary. This package does not implement
another allocator or MLIR backend. Collections own their length, capacity and
growth policies above this layer; raw memory remains available separately.

The agreed [owned destruction contract](DESTRUCTION.md) defines default field
cleanup and the source compiler’s `operator drop(move self) -> unit` hook: custom cleanup
runs first, followed by automatic destruction of remaining owned fields and
storage. General collection and bootstrap object-graph destruction remain incomplete;
see the implementation status and tested limits in that contract.

The [ownership boundary](OWNERSHIP.md) describes the checked compiler plan and
the canonical native implementation in [`native/storage.c`](native/storage.c),
including shared enum payload ownership and its tested limits.
