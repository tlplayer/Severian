# `i128`

API ID: `primitive.signed_integer`; Universal path: `universal.primitive.i128`.

## Representation

`i128` is a signed fixed-width 128-bit integer and is never implied by an
unannotated integer literal. Its range is −2^127 through 2^127−1.

## Source semantics

All integer operator identities are registered at the type level. Registration
does not guarantee a single native instruction; lowering may require multiple
machine words or runtime helpers.

```sev
def preserve(value: i128) -> i128:
    return +value
```

## ABI and lowering

FFI models a signed 128-bit scalar where the target ABI supports it. Calling
convention classification and helper selection are target responsibilities.

## Tensor

`Tensor[i128, S...]` is structurally representable as `SignedInteger(128)`.
Current accumulation remains 128-bit, but backend legality is incomplete.

## Current weakness

Native CPU ABI, vector lowering, and GPU execution for `i128` are not exhaustive
and may be unavailable on individual targets.
