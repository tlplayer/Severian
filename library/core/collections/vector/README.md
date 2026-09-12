# Vector

`vector[T, N = 256]` owns a typed allocation directly. `N` is the initial
capacity; the vector starts empty and grows geometrically when needed.

`append`, `reserve`, `get`, `len`, `capacity`, `clear` and `copy` are implemented
in Severian. `core.storage` checks allocation requests and delegates them to
`core.memory`; replacing or dropping the buffer releases it through ordinary
ownership. The vector owns its length and growth policy. `clear` retains
capacity, while `copy` allocates independent storage.

Native tests live alongside the implementation in `src/vector.sev`.
