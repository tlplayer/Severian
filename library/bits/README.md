# bits

`bits` owns Severian's bitwise integer algebra and the generic `Bits[T]`
contract. Integer operands resolve `|`, `&`, and `^` automatically when there
is no ambiguity. A declaration can use `@bits(...)` to select and limit the
symbolic vocabulary explicitly. The named `bit_or`, `bit_and`, and `bit_xor`
functions are also available when symbolic notation is undesirable.

```sev
import bits

@bits(|, &, ^)
def combine(left: int, right: int) -> int:
    return (left | right) ^ (left & right)
```

Boolean logic remains part of the language's default algebra and uses the
short-circuiting `and`, `or`, and `not` keywords. It does not require an import
or decorator.

The package contract is composable like any other trait:

```sev
trait Flags[T]:
    bits.Bits[T]
    def enabled(flag: T) -> bool

trait Register[T]:
    Flags[T]
    def read() -> T
    def write(value: T)
```

An implementation of `Register[int]` must therefore provide `Register` and
`Flags` methods, while the compiler verifies that `int` satisfies the inherited
`Bits[int]` operator contract.
