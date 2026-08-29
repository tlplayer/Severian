# `f80`

API ID: `primitive.float`; Universal path: `universal.primitive.f80`.

## Representation

`f80` is `FloatFormat::Ieee(80)` in Universal and lowers to MLIR `f80`. It is an
explicit extended format, not a machine-float alias and not a default literal
target.

## Source semantics

Floating operators are registered. The universal format-containment table does
not currently assign an exponent/precision pair to 80-bit format, so automatic
promotion relationships involving f80 are deliberately not inferred.

```sev
def preserve(value: f80) -> f80:
    return +value
```

## ABI and lowering

FFI requests an 80-bit float. Physical padding and calling-convention behavior
are target ABI concerns; unsupported targets must reject the boundary.

## Tensor

`Tensor[f80, S...]` is structurally `IeeeFloat(80)` and retains f80
accumulation. Backend execution support is target-dependent.

## Current weakness

The format's exact precision model, cross-target ABI, conversion lattice, and
GPU legality are incomplete.
