# Core memory

`core.memory` owns the unsafe allocation boundary used by source collections.
It follows the raw-memory model demonstrated in `docs/examples/07-systems/02-memory`:
allocation is explicit, pointer access is confined to `unsafe`, values move out
before storage is released, and `drop` frees the allocation.

`core.storage` depends on this package for checked owning buffers. Collections
use that safe interface and retain their own length, capacity and growth
policies. Both interfaces use the existing memory primitives and compiler
lowering, rather than separate collection allocation backends.

The existing `Allocation[T]` interface remains available to its current callers.

The hosted native provider is [`native/memory.h`](native/memory.h), with external
and MLIR allocator adapters in [`native/memory.c`](native/memory.c). Native
runtime and system helpers use this boundary for raw bytes. Initialization,
views, transfers, and destructor callbacks belong to the
[ownership and storage contract](../storage/OWNERSHIP.md).

## Source MLIR operations

Raw memory operations are declared in `src/raw_intrinsics.sev` and use the
`!llvm.ptr` representation shared by the pointer primitive and C boundaries.
At the bootstrap boundary, bare `pointer` denotes the erased byte-address
contract (`pointer[u8]`); `pointer[T]` retains its pointee type in C signatures.
Assertion diagnostics and test reports call `__raw_write_bytes`, which checks
the byte range and holds a storage view across address extraction and the
synchronous write. Callers do not extract or retain a detached address.

`zeroed_bytes(count)` returns a zero-initialized `array[u8]`.
`resized_bytes(view, count)` returns an independent buffer, preserves the common
prefix and initializes new bytes to zero. `allocate_buffer` and `copy_buffer`
use `memref.alloc` and `memref.copy`. Their descriptors remain visible to the
ownership library's MLIR pipeline until physical lowering.

The hosted C provider is preserved for raw and foreign adapters. It does not
replace source ownership analysis. See the
[ownership audit](../../../sev_compiler/frontend/ownership/AUDIT.md).
