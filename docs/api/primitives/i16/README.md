# `i16`

API ID: `primitive.signed_integer`; Universal path: `universal.primitive.i16`.

## Representation

`i16` is a signed fixed-width 16-bit integer and is not a default literal
target. Its range is −32,768 through 32,767.

## Source semantics

It supports the complete integer operator family. Operations preserve `i16`
unless an explicit conversion or a surrounding generic signature selects a
different result.

```sev
def combine(left: i16, right: i16) -> i16:
    return left | right
```

## ABI and lowering

FFI and scalar MLIR use a 16-bit integer with signed selection for comparisons,
division, remainder, and extension. Signedness is never encoded in MLIR `i16`
alone; the operation carries it.

## Tensor

`Tensor[i16, S...]` is `SignedInteger(16)` and currently accumulates into signed
64-bit values for reductions/contractions.

## Current weakness

Narrow integer overflow and target-specific vector legality need exhaustive
execution tests.
