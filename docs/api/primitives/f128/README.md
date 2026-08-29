# `f128`

API ID: `primitive.float`; Universal path: `universal.primitive.f128`.

## Representation

`f128` is `FloatFormat::Ieee(128)` with fifteen exponent bits and 113 bits of
significand precision. It is explicit binary128 and is never the default
floating-literal representation.

## Source semantics

All floating operators are registered. Lowering may use native instructions or
runtime helpers; source registration does not promise hardware binary128.

```sev
def preserve(value: f128) -> f128:
    return +value
```

## ABI and lowering

FFI and MLIR preserve `f128`. Some LLVM operations lower through helper or
integer paths; targets lacking a legal ABI must report that fact.

## Tensor

`Tensor[f128, S...]` is `IeeeFloat(128)` and retains f128 accumulation.

## Current weakness

Portable ABI behavior, runtime helper coverage, GPU support, printing, and
numeric symmetry are not exhaustive.
