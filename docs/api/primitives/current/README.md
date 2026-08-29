# `current`

API ID: `primitive.measured`; Universal path:
`universal.primitive.current`.

## Representation

`current` is an f64-represented measured type in canonical amperes. `A` is
direct and `mA` divides by 1,000.

## Source semantics

Unary sign, same-type addition/subtraction, equality, and ordering are
registered. Cross-dimensional electrical operators are absent.

```sev
def overcurrent(value: current) -> bool:
    return value > 2A
```

## ABI and lowering

The scalar lowers as f64 amperes. Native monitoring APIs using integer
milliamperes require explicit conversion and checked range handling.

## Tensor

`current` is not directly a tensor element; sampled-current tensors use an
explicit float/integer representation plus unit metadata.

## Current weakness

The API lacks `voltage * current -> power`, uncertainty/calibration, and
device-integer conversion contracts.
