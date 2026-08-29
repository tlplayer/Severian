# `int`

API ID: `primitive.signed_integer`; Universal path: `universal.primitive.int`.

## Representation

`int` is `PrimitiveRepresentation.Integer { bits: Machine, signed: true }` and
is the default integer-literal type. Machine width comes from the target data
layout; it is not promised to equal pointer width or `i64` on every target.

## Source semantics

Unary sign, arithmetic, remainder, power, bitwise operations, equality, and
ordering are registered. Checked overflow and division by zero remain observable.

```sev
def mix(left: int, right: int) -> int:
    return (left + right) ^ (left - right)
```

## ABI and lowering

FFI chooses the target's machine-integer bit width and signed operations.
Conversion to a fixed width is explicit when portability depends on that width.

## Tensor

Tensor element legalization currently accepts fixed-width integers, not the
target-dependent `Machine` width. Model/storage contracts should use `i32` or
`i64` explicitly.

## Current weakness

The language target specification does not yet publish a complete per-target
machine-integer table in this API.
