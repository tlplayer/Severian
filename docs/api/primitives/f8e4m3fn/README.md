# `f8e4m3fn`

API ID: `primitive.float`; Universal path: `universal.primitive.f8e4m3fn`.

## Representation

`f8e4m3fn` uses `FloatFormat::Float8E4M3Fn`: four exponent bits and four bits
of significand precision including the implicit bit. The `fn` format has finite
number semantics distinct from IEEE binary interchange formats. It is never a
default literal target.

## Source semantics

The floating operator family is registered, but source availability does not
promise native FP8 scalar arithmetic. Operations may widen, compute, round, and
store according to an explicit backend policy.

```sev
def preserve(value: f8e4m3fn) -> f8e4m3fn:
    return +value
```

## ABI and lowering

ABI records preserve the exact E4M3FN format. Conversion analysis recognizes
its `(exponent=4, precision=4)` shape; promotion to a containing format is
lossless while reverse conversion is lossy.

## Tensor

`Tensor[f8e4m3fn, S...]` uses `Float8E4M3Fn` with 8-bit storage and `f32`
accumulation. Operation identity remains unchanged.

## Current weakness

Rounding, saturation, exceptional-value behavior, and AMD/NVIDIA execution are
not exhaustively specified for every tensor operation.
