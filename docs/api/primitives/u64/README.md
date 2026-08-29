# `u64`

API ID: `primitive.unsigned_integer`; Universal path: `universal.primitive.u64`.

## Representation

`u64` is an unsigned fixed-width 64-bit integer. It is distinct from `usize`
even on a 64-bit host because its serialized width is stable.

## Source semantics

All integer operators and unsigned ordering are registered. It is suitable for
stable counters, masks, and serialized identifiers within its range.

```sev
def combine(left: u64, right: u64) -> u64:
    return left ^ right
```

## ABI and lowering

FFI uses unsigned 64-bit classification. Conversion to signed `i64` can be
lossy for values above 2^63−1.

## Tensor

`Tensor[u64, S...]` uses `UnsignedInteger(64)` and does not widen accumulation.

## Current weakness

Some GPU targets have limited native `u64` throughput; target-specific
capability/cost data is not yet surfaced here.
