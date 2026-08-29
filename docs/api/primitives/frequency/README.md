# `frequency`

API ID: `primitive.measured`; Universal path:
`universal.primitive.frequency`.

## Representation

`frequency` is an f64-represented measured type in canonical hertz. `Hz`,
`kHz`, `MHz`, and `GHz` literals normalize to cycles per second.

## Source semantics

Same-type sign, addition/subtraction, equality, and ordering are registered.
`float / duration` produces `frequency`; multiplication by duration is not yet
part of the dimensional operator table.

```sev
def inverse(period: duration) -> frequency:
    return 1.0 / period
```

## ABI and lowering

Scalar lowering is f64 after normalization. A device API using integer clock
ticks needs an explicit conversion and clock-domain contract.

## Tensor

`frequency` is not directly a tensor element. Spectral tensors use an explicit
float dtype and axis-unit metadata.

## Current weakness

Zero/negative frequency policy, clock-domain semantics, and complete inverse
unit algebra remain underspecified.
