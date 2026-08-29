# `usize`

API ID: `primitive.unsigned_integer`; Universal path:
`universal.primitive.usize`.

## Representation

`usize` uses `PrimitiveRepresentation.PointerInteger { signed: false }`. Its
width equals the target pointer size and is independent of machine `int`.

## Source semantics

It supports the integer operator family and is the natural result for sizes and
collection lengths. It should not be used as a stable serialized width.

```sev
def total(left: usize, right: usize) -> usize:
    return left + right
```

## ABI and lowering

FFI selects an unsigned integer with `pointer.size * 8` bits. Cross-target
conversion to fixed widths must account for the concrete pointer layout.

## Tensor

`usize` is not accepted by current tensor element legalization. Shape operands
lower to target index/pointer-width scalars separately from tensor element type.

## Current weakness

The API needs cross-compilation conformance for 32- and 64-bit pointer targets
and clearer overflow rules for collection-size arithmetic.
