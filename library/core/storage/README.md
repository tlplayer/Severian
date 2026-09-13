# Core storage

`core.storage` depends on `core.memory` and provides checked owning buffers.
`allocate[T](count)` validates the allocation size. `free(buffer)` consumes the
buffer; dropping it or leaving its scope also releases the allocation through
the existing compiler ownership lowering.

`copy(view)` returns independent initialized storage through `memref.copy`.
Allocation and copying stay visible to the shared MLIR ownership pipeline.

`core.memory` supplies the allocation boundary. This package does not implement
another allocator or MLIR backend. Collections own their length, capacity and
growth policies above this layer; raw memory remains available separately.

The agreed [owned destruction contract](DESTRUCTION.md) defines default field
cleanup and the source compiler’s `operator drop(move self) -> unit` hook: custom cleanup
runs first, followed by automatic destruction of remaining owned fields and
storage. General collection and bootstrap object-graph destruction remain incomplete;
see the implementation status and tested limits in that contract.

The [ownership boundary](OWNERSHIP.md) describes the checked compiler plan and
the preserved native compatibility implementation in [`native/storage.c`](native/storage.c),
including shared enum payload ownership and its tested limits. The
[ownership library](../../../sev_compiler/frontend/ownership/README.md) owns the
contract; the C registry is not the authority for source buffer lifetimes.
