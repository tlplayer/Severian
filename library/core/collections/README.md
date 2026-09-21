# Collections

Each collection owns a package.json and a src/lib.sev entry point. Depend on an
individual package, or on the collections facade for all public collections.
Source is canonical inside those package folders; generated package.pkg output
is not checked in as a second implementation.

Collections delegate allocation and initialized-element destruction to
core.storage, which depends on core.memory. No collection package declares a C
provider, foreign ABI, or direct allocator call. Optional native storage
instrumentation is isolated in core.storage.statistics and is not a dependency.

The list package supplies owned lists and borrowed slice regions. Array, deque,
stack and heap share core.storage. BTree and Hash own explicit list-package
sequences. Maps and sets select a provider; dictionaries use Hash and preserve
insertion order. Count builds on dictionaries with checked cardinality updates.

vector[T, N] is growable with initial capacity N. numeric_vector[T, N] preserves
the former fixed-lane arithmetic operations under a distinct name. Neither
representation requires a C implementation.

The canonical package APIs use optional absence results. List pop/remove return
owned elements, indexing borrows, and slice regions retain a borrow of the owner.
The old loose bootstrap files with Copy + Default and tuple absence results have
been consolidated into these package APIs; callers must use the package API.

Example dependency (SIP-0003):

```json
{"dependencies":{"lists":{"package":"list","path":"../Severian/library/core/collections/list","version":"0.1.0"}}}
```

```sev
import list from lists
values := list[int]()
values.append(1)
```

Build the facade with `sev build` from this directory. The package manager owns
lock resolution, interfaces, and reusable dependency artifacts per SIP-0003.
