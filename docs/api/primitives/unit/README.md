# `unit`

API ID: `primitive.text_and_storage`; Universal path:
`universal.primitive.unit`.

## Representation

`unit` uses `PrimitiveRepresentation.Unit` and represents successful completion
without a payload. It is the default unit-literal/result type. Equality and
inequality are defined.

## Source semantics

Functions that perform effects without producing a value return `unit`, either
explicitly or through omitted result syntax. `unit | Error` distinguishes
successful completion from failure without manufacturing a dummy integer.

```sev
def completed() -> unit:
    return
```

## ABI and lowering

A standalone unit result lowers to void. Unit cannot be passed as a foreign
argument. Calls still preserve their effects and sequencing even though no SSA
payload is produced.

## Tensor

`unit` is not a tensor element. Effectful tensor operations use effect tokens or
region sequencing, not unit-valued tensor cells.

## Current weakness

The effect system does not yet expose a complete API-level table showing when
unit-returning calls may be reordered or eliminated.
