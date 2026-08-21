# Foreign interfaces

Foreign declarations remain normal source declarations annotated with a
language boundary. There is no separate `extern def` grammar.

```sev
import c from xxi

@c(symbol = "native_add")
def add(left: i32, right: i32) -> i32
```

XXI resolves `c` to a provider identity. FFI validates ownership and performs
declared value conversions. ABI derives physical layout and calling
classification from the selected target. Backends consume that already-lowered
contract and do not look Severian types up again.

Direct foreign calls are unsafe unless a library wraps the declaration with a
safe contract. A safe wrapper is responsible for validating pointers, lengths,
nullability, ownership transfer, and returned lifetimes.
