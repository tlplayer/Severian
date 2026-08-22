# Collections

`library/collections` is an organizational namespace, not a package. There is
deliberately no `library/collections/package.toml`.

Public packages are independently compiled and imported:

```text
core.primitives
      |
collections.traits
      |
      +-- list
      +-- set
      +-- map
      +-- deque -- queue
      +-- heap
```

Concrete collections do not depend on one another unless their public
semantics fundamentally require it. Shared implementation machinery belongs in
`collections/internal`; those packages are private and are never imported by
application code.

The intended concrete dependency edges are:

- `list -> internal.contiguous_storage`
- `set -> internal.hash_table`
- `map -> internal.hash_table`
- `deque ->` its own segmented or ring storage
- `queue -> deque`
- `heap ->` its own storage or `internal.contiguous_storage`

Set never wraps the public map package, and map never wraps the public set
package. Their eventual hash-table sharing stays below both public APIs.

The golden path is traits, then a complete source-owned `list[T]`, followed by
set, map, deque, queue, and heap. Literal syntax remains compiler-owned, but
literal meaning resolves to these library types. Storage, growth, iteration,
hashing, equality, removal, and algorithms remain library-owned.

The list literal boundary is the source-owned `list.builder[T](capacity)` API:
the compiler creates a builder, pushes literal elements in source order, and
finishes it into `list[T]`. That is a lowering convention, not a second list
implementation.

Compiler literal lowering resolves `[a, b]`, `{a, b}`, and `{key: value}` to
the public `list`, `set`, and `map` package types respectively. The compiler may
select and call their builder entry points; it must not retain an independent
Rust container implementation.

Class implementation headers use one terminating newline:

```sev
class Example: FirstTrait + SecondTrait
    value: int
```

There is no trailing colon after the final implemented trait.
