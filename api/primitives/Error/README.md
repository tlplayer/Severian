# `Error`

API ID: `primitive.text_and_storage`; Universal path:
`universal.primitive.Error`.

## Representation

`Error` is a distinct semantic primitive whose current physical representation
is `PrimitiveRepresentation.String`. Representation sharing does not make it
assignable to `string`. It is not a default string-literal target.

## Source semantics

`Error(message)` constructs the root error value for `throw`, contracts, and
error unions. A function returning `T | Error` propagates the active error
member through ordinary result semantics; `?=` captures it without erasing its
type.

```sev
def require(found: bool) -> string | Error:
    if not found:
        throw Error("not found")
    return "value"
```

## ABI and lowering

The current string-like representation is an implementation fact. Error union
tagging, message ownership, and call-stack metadata belong to ABI lowering and
must remain valid if the physical record grows.

## Tensor

`Error` is never a tensor element. Tensor compilation failures are diagnostics
or result errors at graph/launcher boundaries, not values inside compute tensors.

## Current weakness

The stable external error ABI and the relationship between message storage and
structured error subclasses are not fully specified.
