# `f32`

API ID: `primitive.float`; Universal path: `universal.primitive.f32`.

## Representation

`f32` is `FloatFormat::Ieee(32)`, with eight exponent bits and twenty-four bits
of significand precision including the implicit bit. It provides stable-width
binary32 semantics independent of machine `float`.

## Source semantics

All floating operators and ordered comparisons are registered. It is the
current accumulation target for FP8, f16, and bf16 tensor operations.

```sev
def affine(value: f32, scale: f32, bias: f32) -> f32:
    return value * scale + bias
```

## ABI and lowering

FFI and MLIR use direct `f32`. Promotion from the supported 8/16-bit formats is
lossless by the universal format-shape model.

## Tensor

`Tensor[f32, S...]` uses `IeeeFloat(32)` and retains f32 accumulation unless an
algorithm explicitly requests a wider accumulator.

## Current weakness

Although this is the broadest working tensor format, the complete GPU structural
operation and dynamic-layout matrix is not yet executable.
