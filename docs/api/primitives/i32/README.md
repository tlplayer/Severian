# `i32`

API ID: `primitive.signed_integer`; Universal path: `universal.primitive.i32`.

## Representation

`i32` is a signed fixed-width 32-bit integer and is not a default literal type.
It is a common model index/storage representation but does not replace `isize`
for pointer arithmetic.

## Source semantics

All integer operations and ordered comparisons are registered. Literal operands
resolve against an exact `i32` signature when the other operand supplies context.

```sev
def step(value: i32, delta: i32) -> i32:
    return value + delta
```

## ABI and lowering

FFI uses a signed 32-bit scalar. Widening to `i64` is a promotion; narrowing
from wider types is potentially lossy and must remain explicit.

## Tensor

`Tensor[i32, S...]` uses `SignedInteger(32)` and currently widens accumulation
to signed 64-bit.

## Current weakness

The complete CPU/GPU matrix for integer division, power, and reductions is not
yet recorded.
