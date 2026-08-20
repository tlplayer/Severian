# XXI Boundary

Source-level external-language integration. XXI resolves languages, declarations, attributes, and symbols without deciding FFI safety or ABI layout.

```severian
import "c:libc" as libc

@c(repr = "opaque")
type FILE

@c(symbol = "fwrite")
def write(data: borrowed[bytes], output: borrowed[FILE]) -> usize

@rust
def rust_entry(value: i32) -> i32
```

Supported contract wrappers are `borrowed[T]`, `owned[T]`, `transferred[T]`,
`out[T]`, `inout[T]`, `nullable[T]`, `ptr[T]`, and `mut_ptr[T]`. Unknown
language attributes remain representable and default to the target system ABI;
an explicit `abi = "..."` selects a supported concrete convention.
