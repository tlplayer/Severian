# `None`

API ID: `primitive.text_and_storage`; Universal path:
`universal.primitive.None`.

## Representation

`None` uses `PrimitiveRepresentation.None`, is the default target for the
absence literal, and carries no payload. It is semantically distinct from
`unit`: absence is an active union member, while unit is a successful no-value
result.

## Source semantics

`T | None` describes an optional value. Equality and inequality are defined.
Flow analysis may refine the active member after a presence check, but it must
not invent a default representation when the surrounding union is unknown.

```sev
def lookup(found: bool) -> string | None:
    if found:
        return "value"
    return None
```

## ABI and lowering

As a result by itself, `None` lowers to void. It cannot be passed as a standalone
FFI argument. Within a union, the union ABI supplies a tag and any payload.

## Tensor

`None` is not a tensor element. Optional tensors are unions around whole tensor
values, not tensors with absent scalar cells.

## Current weakness

The canonical external tagged-union ABI for `T | None` needs a dedicated
versioned record specification.
