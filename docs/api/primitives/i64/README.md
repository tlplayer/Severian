# `i64`

API ID: `primitive.signed_integer`; Universal path: `universal.primitive.i64`.

## Representation

`i64` is a signed fixed-width 64-bit integer. It is distinct from target
machine `int` even when both currently lower to 64 bits on a host.

## Source semantics

The full integer operator family is registered. `i64` is commonly used for
tensor indices and serialized integer values when a stable width is required.

```sev
def square(value: i64) -> i64:
    return value * value
```

## ABI and lowering

FFI uses a signed 64-bit scalar. Conversions to `isize` depend on target pointer
width and are not universally lossless.

## Tensor

`Tensor[i64, S...]` is `SignedInteger(64)`. It is already the current integer
accumulator target, so accumulation does not widen beyond 64 bits.

## Current weakness

Overflow policy for large reductions and optimized GPU `i64` arithmetic needs
explicit target coverage.
