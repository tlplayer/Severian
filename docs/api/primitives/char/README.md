# `char`

API ID: `primitive.text_and_storage`; Universal path:
`universal.primitive.char`.

## Representation

`char` uses `PrimitiveRepresentation.Character`, is the default target for a
character literal, and is Copy. The foreign ABI lowers it directly as an
unsigned 32-bit integer, which is a scalar code-point representation rather
than a UTF-8 byte.

## Source semantics

Character literals use single quotes. Equality and total ordering are defined.
Concatenation is not a `char` operator; conversion to or construction of
`string` is a separate API decision.

```sev
def before(left: char, right: char) -> bool:
    return left < right
```

## ABI and lowering

Calls pass an unsigned 32-bit scalar. The frontend must reject malformed or
multi-character literals before ABI lowering. Encoding a character into UTF-8
storage is not a no-op cast.

## Tensor

`char` is not accepted by `TensorElementKind`; tensor kernels therefore reject
it during legalization rather than treating it as `u32`.

## Current weakness

The public API does not yet specify Unicode scalar validation, escape coverage,
or normalization policy exhaustively.
