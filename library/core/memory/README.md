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
