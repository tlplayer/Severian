# `string`

API ID: `primitive.text_and_storage`; Universal path:
`universal.primitive.string`.

## Representation

`string` uses `PrimitiveRepresentation.String` and is the default string-literal
target. It is pointer-bearing storage, not Copy by representation.

The versioned owned boundary is `StringAbiV1 { data, length, capacity }`; the
borrowed boundary is `StringViewAbiV1 { data, length }`. Both lengths are UTF-8
byte counts. A canonical empty owned value has a null pointer, zero length, and
zero capacity.

## Source semantics

Equality, inequality, ordering, and `+` concatenation are registered. Double,
triple, and formatted strings produce the same value type. Ownership on calls
comes from the parameter contract; a raw pointer to bytes is not a `string`.

```sev
def greeting(name: string) -> string:
    return "hello " + name
```

## ABI and lowering

The view length is explicit, so embedded zero bytes are not a language-level
terminator. ABI conversion owns any temporary encoding/lifetime work. Native
callees must not retain borrowed views beyond their declared lifetime.

Severian library behavior is lowered through checked-in or compiler-generated
MLIR. The initial `core.text.string` MLIR library owns concat, compare, and
release compatibility exports. The compiler imports a definition only when the
ordinary module contains its unresolved declaration, then verifies the combined
operation graph. C-compatible signatures may exist at an external boundary, but
new String semantics are not implemented in C.

The compatibility exports still consume the old NUL-terminated pointer value.
They are a transition mechanism, not the promised final representation. Moving
ordinary source lowering to `StringAbiV1` is required before embedded NUL bytes
work end to end.

## Tensor

`string` is not a tensor element. Tables or tokenizers may store strings in
host collections, but compute tensor IR must receive numeric/boolean storage.

## Current weaknesses

- Ordinary source lowering still uses the pointer compatibility representation.
- Remaining conversion and formatting helpers still need migration from the
  legacy native runtime into `.sev` or MLIR.
- Normalization, indexing-by-code-point versus byte, and lifetime guarantees for
  retained foreign views need dedicated contracts and tests.
