# Pointers

Pointers are a systems type, not a second reference system. Safe borrows remain
the normal way to share values. A `*[T]` is used for foreign memory, allocated
raw storage, memory-mapped regions, and other addresses whose validity requires
an explicit invariant.

These examples freeze the following rules:

- forming or indexing a raw pointer requires `unsafe`;
- pointer indexing is spelled `pointer[index]`; Severian has no ambiguous unary
  dereference syntax;
- pointer arithmetic is measured in elements of `T` and is only available for
  raw pointers; `pointer + 1` advances one `T`, while an actual byte offset must
  be written with a data unit such as `1B`;
- nullability is written `*[T] | None` rather than hidden inside every pointer;
- casts must state both source and destination pointer types and preserve
  alignment requirements.
