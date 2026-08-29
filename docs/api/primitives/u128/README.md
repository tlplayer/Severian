# `u128`

API ID: `primitive.unsigned_integer`; Universal path: `universal.primitive.u128`.

## Representation

`u128` is an unsigned fixed-width 128-bit integer with range 0 through 2^128−1.
It is always selected explicitly or by type context.

## Source semantics

Every integer operator identity is registered. Backend implementation may use
multiword arithmetic; operator availability at the type level does not promise
a native 128-bit instruction.

```sev
def preserve(value: u128) -> u128:
    return +value
```

## ABI and lowering

FFI requests unsigned 128-bit classification. Targets that cannot represent
that ABI must reject it rather than truncate to pointer or machine width.

## Tensor

`Tensor[u128, S...]` is structurally `UnsignedInteger(128)`, but current GPU and
vector execution coverage is incomplete.

## Current weakness

Portable ABI classification, runtime helpers, vectorization, and GPU lowering
for `u128` need explicit target tests.
