# `data_rate`

API ID: `primitive.measured`; Universal path:
`universal.primitive.data_rate`.

## Representation

`data_rate` is a measured semantic type with physical f64 representation. Its
canonical meaning is bytes per second. It has no direct literal suffix in the
current lexer and is produced by `data_size / duration` or typed values.

## Source semantics

Unary sign, same-type addition/subtraction, equality, and ordering are
registered. Cross-dimensional multiplication back into `data_size` is not yet
registered.

```sev
def throughput(transferred: data_size, elapsed: duration) -> data_rate:
    return transferred / elapsed
```

## ABI and lowering

The value lowers as f64, but an ABI must preserve or explicitly document its
bytes-per-second unit. Zero-duration division reports the numeric division error.

## Tensor

`data_rate` is not a legal tensor element; numeric telemetry tensors require
separate schema metadata.

## Current weakness

No direct rate literal syntax exists, and the dimensional algebra lacks
`data_rate * duration -> data_size`.
