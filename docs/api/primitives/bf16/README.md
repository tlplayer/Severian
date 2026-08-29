# `bf16`

API ID: `primitive.float`; Universal path: `universal.primitive.bf16`.

## Representation

`bf16` is `FloatFormat::BrainFloat16`, with eight exponent bits and eight bits
of significand precision including the implicit bit. It shares binary32's
exponent range but not its precision and is not a default literal target.

## Source semantics

The floating operator family is registered. Compiler and runtime carry the
BrainFloat format explicitly; the type must never be inferred from a 16-bit
width alone.

```sev
def residual(left: bf16, right: bf16) -> bf16:
    return left + right
```

## ABI and lowering

FFI uses an explicit bfloat16 ABI type. Conversion to `f32` is a promotion;
conversion back selects a declared rounding policy.

## Tensor

`Tensor[bf16, S...]` uses `BrainFloat16`, 16-bit storage, and `f32`
accumulation. It shares `Elementwise`, `Reduce`, and `Matmul` operation IDs with
other formats.

## Current weakness

End-to-end GPU lowering, cached launch, and model-weight execution remain
incomplete for the full bf16 operation matrix.
