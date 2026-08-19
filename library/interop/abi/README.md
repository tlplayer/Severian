# ABI

`abi` describes how typed values cross a foreign language boundary. It owns C
calling conventions, layouts, pointer and buffer shapes, nullability, and
ownership vocabulary. It does not locate or invoke symbols; that is `ffi`'s
job.

The initial stable boundary is `abi.c()` (`c-v1`). Provider declarations use
fixed-width scalars or explicit wrappers such as `abi.StringView`,
`abi.BytesView`, `abi.Handle`, and output parameters. Dynamic `Any` and integer
pointer escape hatches are rejected.

The descriptor surface is intentionally source-owned:

```sev
signature = abi.c().function(
    "strlen",
    [abi.Type("string-view", abi.borrowed(), false)],
    abi.Type("usize", abi.copy(), false),
)
```

The compiler retains only the closed generic ABI representation needed for
validation and lowering. Rust ABI descriptors can later reuse that foreign-call
representation without creating a second compiler call pipeline.
