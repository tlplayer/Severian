# `f64`

API ID: `primitive.float`; Universal path: `universal.primitive.f64`.

## Representation

`f64` is `FloatFormat::Ieee(64)`, with eleven exponent bits and fifty-three
bits of significand precision. It is stable-width binary64 and is distinct from
machine `float` despite their current conversion relationship.

## Source semantics

The full floating operator family is registered. `f64` is the physical
representation currently used for measured primitives, but those types retain
distinct semantic identities.

```sev
def ratio(left: f64, right: f64) -> f64:
    return left / right
```

## ABI and lowering

FFI/MLIR use direct `f64`. Narrowing to f32 or lower formats is lossy; integer
conversion is also classified as lossy.

## Tensor

`Tensor[f64, S...]` uses `IeeeFloat(64)` and retains f64 accumulation.

## Current weakness

High-performance GPU execution and tolerance-defined symmetry tests are not
complete for every f64 tensor operation.
