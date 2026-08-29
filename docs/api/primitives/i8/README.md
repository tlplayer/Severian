# `i8`

API ID: `primitive.signed_integer`; Universal path: `universal.primitive.i8`.

## Representation

`i8` is a signed fixed-width 8-bit integer. It is not the default literal type;
context or `i8(...)` must select it. Its mathematical range is −128 through 127.

## Source semantics

All integer unary, arithmetic, remainder, power, bitwise, equality, and ordered
comparison identities are registered with `i8` operands/results.

```sev
def masked(value: i8, mask: i8) -> i8:
    return value & mask
```

## ABI and lowering

FFI uses a signed 8-bit scalar. Promotion to wider signed integers is lossless;
truncation and signed/unsigned conversions require range-aware conversion policy.

## Tensor

`Tensor[i8, S...]` is legal with element kind `SignedInteger(8)`. Accumulation
widens to signed 64-bit in the current tensor policy.

## Current weakness

GPU execution and conversion coverage is not exhaustive for every `i8` tensor
operation, layout, and saturation policy.
