# `u32`

API ID: `primitive.unsigned_integer`; Universal path: `universal.primitive.u32`.

## Representation

`u32` is an unsigned fixed-width 32-bit integer and is not the default integer
literal representation.

## Source semantics

It supports all integer operator identities. Unsigned ordering differs from
`i32` for values with the high bit set even though both lower to MLIR `i32`.

```sev
def advance(value: u32, amount: u32) -> u32:
    return value + amount
```

## ABI and lowering

FFI classifies it as unsigned 32-bit. Widening into `u64` is a promotion;
conversion into signed `i32` is not generally lossless.

## Tensor

`Tensor[u32, S...]` is `UnsignedInteger(32)` and currently accumulates to `u64`.

## Current weakness

The tensor/backend matrix for unsigned divide, remainder, power, and reductions
is incomplete.
