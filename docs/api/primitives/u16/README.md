# `u16`

API ID: `primitive.unsigned_integer`; Universal path: `universal.primitive.u16`.

## Representation

`u16` is an unsigned fixed-width 16-bit integer with range 0 through 65,535.
It is selected by context or explicit conversion.

## Source semantics

All integer arithmetic, bitwise, equality, and ordered comparison identities
are registered with unsigned interpretation where relevant.

```sev
def flags(left: u16, right: u16) -> u16:
    return left | right
```

## ABI and lowering

FFI uses unsigned 16-bit classification. MLIR integer types omit signedness, so
the selected operation must retain unsigned comparison/division semantics.

## Tensor

`Tensor[u16, S...]` is `UnsignedInteger(16)` with current `u64` accumulation.

## Current weakness

Backend conformance does not yet cover every unsigned narrow-vector operation.
