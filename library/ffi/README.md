# Foreign-function interface

`ffi` identifies foreign libraries and symbols. `abi` separately owns calling
conventions, layouts, and ownership. Domain packages keep provider declarations
private; safe public APIs expose domain values rather than raw handles or output
parameters.

The compiler recognizes these wrappers at a C v1 boundary and generates the
conversion shim. Providers receive only stable scalars, views, handles, and
output structures—not compiler-owned Severian values.

```sev
import ffi
import abi

libc = ffi.library("c")
strlen = libc.symbol(
    "strlen",
    abi.c().function(
        "strlen",
        [abi.Type("string-view", abi.borrowed(), false)],
        abi.Type("usize", abi.copy(), false),
    ),
)
```

`Library` and `Symbol` are declarative identities, not runtime `dlopen` escape
hatches. Package manifests resolve providers, and the compiler lowers the same
typed `ForeignCall` regardless of which domain package requested it. The legacy
wrapper classes remain temporarily for packages migrating to `abi.*`.
