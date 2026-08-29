# `u8`

API ID: `primitive.unsigned_integer`; Universal path: `universal.primitive.u8`.

## Representation

`u8` is an unsigned fixed-width 8-bit integer with range 0 through 255. It is
not a default literal target.

## Source semantics

The full integer operator family is registered. Ordered comparisons, division,
remainder, and extension use unsigned semantics.

```sev
def mask(value: u8, bits: u8) -> u8:
    return value & bits
```

## ABI and lowering

FFI uses an unsigned 8-bit scalar. `u8` is a byte-sized number; it does not
inherit the ownership/view semantics of `bytes`.

## Tensor

`Tensor[u8, S...]` is `UnsignedInteger(8)` and currently accumulates into
unsigned 64-bit values.

## Current weakness

Saturating versus checked conversion and all GPU byte-vector paths are not yet
exhaustively specified.
