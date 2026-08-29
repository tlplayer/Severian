# `voltage`

API ID: `primitive.measured`; Universal path:
`universal.primitive.voltage`.

## Representation

`voltage` is an f64-represented measured type in canonical volts. `V` is direct
and `mV` divides the magnitude by 1,000.

## Source semantics

Unary sign, same-type addition/subtraction, equality, and ordering are
registered. Electrical dimensional relations are not yet registered.

```sev
def undervoltage(value: voltage) -> bool:
    return value < 900mV
```

## ABI and lowering

Lowering uses f64 volts. Hardware APIs often use integer micro/millivolts and
require explicit scale, range, and rounding behavior.

## Tensor

`voltage` is not a tensor element; telemetry tensors need numeric dtype plus
voltage-unit metadata.

## Current weakness

`voltage * current -> power`, precision policy, and device-scale conversions
are missing from the dimensional API.
