# `temperature`

API ID: `primitive.measured`; Universal path:
`universal.primitive.temperature`.

## Representation

`temperature` is an f64-represented measured type in canonical degrees Celsius.
`C` is direct, `F` applies `(value−32)*5/9`, and `K` subtracts 273.15.

## Source semantics

Same-type sign, addition/subtraction, equality, and ordering are currently
registered. Ordering is meaningful for absolute temperatures; addition of two
absolute temperatures exposes the absence of a separate temperature-delta type.

```sev
def safe(value: temperature) -> bool:
    return value < 90C
```

## ABI and lowering

After literal normalization the scalar lowers as f64 Celsius. External sensors
using integer millidegrees or Kelvin require explicit scale/offset conversion.

## Tensor

`temperature` is not a tensor element. Sensor tensors use numeric storage with
unit metadata and calibration records.

## Current weakness

Absolute temperature and temperature difference are not distinct types;
therefore the current same-type add/subtract algebra is physically incomplete.
