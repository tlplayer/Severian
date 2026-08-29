# `string`

API ID: `primitive.text_and_storage`; Universal path:
`universal.primitive.string`.

## Representation

`string` uses `PrimitiveRepresentation.String` and is the default string-literal
target. It is pointer-bearing storage, not Copy by representation. FFI exposes
a C-layout `StringView { data: *u8, length: usize }` with UTF-8 view conversion.

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

## Tensor

`string` is not a tensor element. Tables or tokenizers may store strings in
host collections, but compute tensor IR must receive numeric/boolean storage.

## Current weakness

Normalization, indexing-by-code-point versus byte, and lifetime guarantees for
retained foreign views need dedicated pages and tests.
