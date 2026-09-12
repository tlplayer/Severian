# Core storage

`core.storage` depends on `core.memory` and provides checked owning buffers.
`allocate[T](count)` validates the allocation size. `free(buffer)` consumes the
buffer; dropping it or leaving its scope also releases the allocation through
the existing compiler ownership lowering.

`core.memory` supplies the allocation boundary. This package does not implement
another allocator or MLIR backend. Collections own their length, capacity and
growth policies above this layer; raw memory remains available separately.
