# `percentage`

API ID: `primitive.measured`; Universal path:
`universal.primitive.percentage`.

## Representation

`percentage` is a measured semantic type represented as f64 canonical ratio.
The `pct` suffix divides the source magnitude by 100, so `80pct` stores `0.8`.

## Source semantics

Same-type sign, addition/subtraction, equality, and ordering are registered.
Equality/inequality against machine `float` is also explicitly registered;
other mixed arithmetic is not.

```sev
def saturated(value: percentage) -> bool:
    return value >= 100pct
```

## ABI and lowering

Lowering uses f64 ratio values, not whole percent points. External APIs that
expect `0..100` must convert explicitly.

## Tensor

`percentage` is not a tensor element. Probability tensors conventionally use
explicit floats with range constraints.

## Current weakness

The valid range is not currently restricted to 0–100%, and mixed comparison
with fixed `f32`/`f64` is not exhaustively specified.
