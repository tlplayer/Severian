# `f8e5m2`

API ID: `primitive.float`; Universal path: `universal.primitive.f8e5m2`.

## Representation

`f8e5m2` uses `FloatFormat::Float8E5M2`, with five exponent bits and three bits
of significand precision including the implicit bit. It trades precision for a
wider exponent range than E4M3FN and is not a default literal target.

## Source semantics

Floating operators are structurally registered. A legal implementation may
widen the operands and round the result back to E5M2; it must not silently treat
the bits as E4M3FN.

```sev
def preserve(value: f8e5m2) -> f8e5m2:
    return +value
```

## ABI and lowering

The ABI carries an explicit E5M2 format tag. Conversion analysis uses
`(exponent=5, precision=3)`, so range and precision containment are evaluated
separately.

## Tensor

`Tensor[f8e5m2, S...]` has 8-bit E5M2 storage and `f32` accumulation.

## Current weakness

Target capability, exceptional values, rounding, and storage-conversion
symmetry tests are incomplete.
