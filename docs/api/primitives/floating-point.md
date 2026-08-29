# Floating-point types

API ID: `primitive.float`

The family contains:

- machine `float`;
- `f8e4m3fn` and `f8e5m2`;
- IEEE `f16`, `f32`, `f64`, `f80`, and `f128`;
- `bf16`.

Format and bit width are fields of `PrimitiveRepresentation.Float`. They stay
fields through tensor IR, specialization, and ABI argument packing.

```sev
trait Numeric:
    def add(other: Self) -> Self
    def multiply(other: Self) -> Self

def affine[T: Numeric](value: T, scale: T, bias: T) -> T:
    return value * scale + bias

test "ordinary f32 arithmetic":
    assert(affine[f32](4.0, 0.5, 1.0) == 3.0)
```

An operation may use a wider accumulator without changing its source result
type or operation ID. FP8 formats in particular require explicit accumulator
and rounding policies.

Current weakness: not every target supports native arithmetic for every
registered format. Legal widening sequences and unsupported pairs must be
recorded by backend capability data; they must not become `add_bf16`-style
functions.
