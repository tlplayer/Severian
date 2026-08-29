# Unsigned integers

API ID: `primitive.unsigned_integer`

`u8`, `u16`, `u32`, `u64`, and `u128` are fixed-width unsigned integers;
`usize` has pointer width. Unsignedness controls comparison, division,
remainder, extension, and conversion semantics while preserving the same
structural operation identity.

```sev
def byte_count(blocks: u128, block_size: u128) -> u128:
    return blocks * block_size

test "unsigned arithmetic":
    assert(byte_count(3, 16) == 48)
```

Negative literals do not implicitly inhabit unsigned types. Conversions that
cannot represent their input report an invalid conversion.

Current weakness: as with signed integers, frontend coverage is broader than
the exhaustive backend execution matrix.
