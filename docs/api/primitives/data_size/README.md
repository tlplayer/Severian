# `data_size`

API ID: `primitive.measured`; Universal path:
`universal.primitive.data_size`.

## Representation

`data_size` is a distinct `PrimitiveCategory.Measured` value represented
physically as `f64`. Canonical storage is bytes. Literal suffixes normalize
bits, decimal bytes (`KB`…`TB`), and binary bytes (`KiB`…`TiB`) into that value.

## Source semantics

Unary sign, same-type addition/subtraction, equality, and ordering are
registered. Dividing by another `data_size` returns `float`; dividing by
`duration` returns `data_rate`.

```sev
def blocks(total: data_size, block: data_size) -> float:
    return total / block
```

## ABI and lowering

Scalar lowering uses f64 operations while semantic analysis preserves the
dimension. A foreign ABI that expects a byte count integer needs an explicit
conversion; it is not implied by physical f64 storage.

## Tensor

Measured types are not accepted by `TensorElementKind`. Numeric tensor storage
must choose an explicit scalar representation and carry units separately.

## Current weakness

Fractional-byte policy, integer byte-count conversion, and overflow/precision
behavior above 2^53 bytes need exact contracts.
