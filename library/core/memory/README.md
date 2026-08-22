# Core memory

`core.memory` owns the unsafe allocation boundary used by source collections.
It follows the raw-memory model demonstrated in `docs/examples/07-systems/02-memory`:
allocation is explicit, pointer access is confined to `unsafe`, values move out
before storage is released, and `drop` frees the allocation.

Public collections depend on safe storage packages, never on allocator,
runtime, platform, pointer, or FFI APIs directly.
