# `isize`

API ID: `primitive.signed_integer`; Universal path:
`universal.primitive.isize`.

## Representation

`isize` uses `PrimitiveRepresentation.PointerInteger { signed: true }`. Its
width comes from the target pointer layout, not `IntegerWidth::Machine`, so it
is semantically distinct from `int`.

## Source semantics

The integer operator family is registered. `isize` is appropriate for signed
pointer-relative offsets and native handles whose ABI contract uses pointer
width; serialized data should use a fixed width instead.

```sev
def offset(base: isize, delta: isize) -> isize:
    return base + delta
```

## ABI and lowering

FFI selects `pointer.size * 8` with signed operations. Conversions to fixed
widths depend on the concrete target and may be lossy.

## Tensor

Pointer-width integers are not accepted by current `TensorElementKind`; tensor
index buffers should use an explicit fixed width such as `i64`.

## Current weakness

Cross-compilation tests do not yet exercise `isize` on every supported pointer
width and calling convention.
