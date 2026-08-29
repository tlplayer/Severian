# `power`

API ID: `primitive.measured`; Universal path:
`universal.primitive.power`.

## Representation

`power` is an f64-represented measured type in canonical watts. The `W` suffix
maps directly to that canonical value.

## Source semantics

Unary sign, same-type addition/subtraction, equality, and ordering are
registered. Power is currently constructed by literals or typed/native values,
not by a registered voltage/current product.

```sev
def within_budget(value: power) -> bool:
    return value <= 220W
```

## ABI and lowering

The scalar lowers as f64 watts. Integer milliwatt device APIs require explicit
unit conversion and rounding.

## Tensor

`power` is not a tensor element. Power time series use explicit numeric tensor
storage and schema-level unit metadata.

## Current weakness

Electrical derivation, energy integration, scaled literals, and device ABI
conversions are not yet part of the dimensional algebra.
