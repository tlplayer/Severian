# `float`

API ID: `primitive.float`; Universal path: `universal.primitive.float`.

## Representation

`float` is `PrimitiveRepresentation.Float { format: Machine }` and is the
default floating-literal type. The target data layout selects its physical
width; the universal conversion model currently treats it canonically like
binary64 for promotion analysis.

## Source semantics

Unary sign, arithmetic, remainder, power, equality, and ordered comparisons are
registered. Floating NaN/rounding behavior must come from the selected target
policy, not from the source name alone.

```sev
def average(left: float, right: float) -> float:
    return (left + right) / 2.0
```

## ABI and lowering

FFI selects the target machine-float width. Code that requires stable storage or
cross-target reproducibility should use `f32`, `f64`, or another fixed format.

## Tensor

Machine-format floats are not accepted by current `TensorElementKind`; tensor
element contracts use explicit float formats.

## Current weakness

The per-target machine-float table and exact default rounding/NaN contract are
not yet published in the API.
