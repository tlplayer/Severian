# `bytes`

API ID: `primitive.text_and_storage`; Universal path:
`universal.primitive.bytes`.

## Representation

`bytes` uses `PrimitiveRepresentation.Bytes`. It is byte storage with a
`BytesView { data: *u8, length: usize }` foreign ABI conversion. It is not
implicitly UTF-8 and is not assumed Copy.

## Source semantics

Equality, inequality, and `+` concatenation are registered. Ordering is not.
Byte literals resolve here when the lexer produces `LiteralKind.Bytes`;
ordinary strings do not implicitly convert.

```sev
def combine(left: bytes, right: bytes) -> bytes:
    return left + right
```

## ABI and lowering

Length is explicit and zero bytes are ordinary data. Borrowed view lifetime and
ownership are selected at the call boundary. A bytes view cannot be returned as
a builtin MLIR tensor without a storage descriptor and specialization step.

## Tensor

`bytes` itself is not a tensor element. A `Tensor[u8, S...]` is typed numeric
storage and has different shape/stride/alias semantics.

## Current weakness

Literal spelling and owned-versus-borrowed foreign return policy need
standalone conformance coverage beyond view conversion.
