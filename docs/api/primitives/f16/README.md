# `f16`

API ID: `primitive.float`; Universal path: `universal.primitive.f16`.

## Representation

`f16` is `FloatFormat::Ieee(16)`, with five exponent bits and eleven bits of
significand precision including the implicit bit. It is not the default
floating-literal target.

## Source semantics

All floating operator identities and ordered comparisons are registered.
Literals receive `f16` through annotation or operand context.

```sev
def scale(value: f16, factor: f16) -> f16:
    return value * factor
```

## ABI and lowering

FFI uses a direct 16-bit IEEE float ABI type where supported. Promotion to
`f32`/`f64` is lossless by format containment; narrowing requires rounding.

## Tensor

`Tensor[f16, S...]` uses `IeeeFloat(16)` with `f32` accumulation for reductions
and contractions.

## Current weakness

Targets without native half scalar operations need explicit legal widening
paths and conformance results.
