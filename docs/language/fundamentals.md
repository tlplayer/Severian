# Language Fundamentals

Sevarian's surface syntax is intentionally familiar to Python programmers, but
the compiler treats the program as a statically checked systems language.

## Bindings

Local bindings are stable by default. `:=` creates a changeable local binding
that can be assigned again.

```sev
name = "Ada"
count := 0

count += 1
```

Use `:=` when the binding itself changes over time: counters, builders,
accumulators, state machines, and buffers. It does not mean dynamically typed;
types are still inferred or checked statically.

Plain `=` bindings are already constant in the everyday local sense, so Sevarian
does not need `const` just to prevent reassignment.

`const` is reserved for named compile-time values, usually at module scope.

```sev
const MaxRetries: int = 3
const Pi: float = 3.1415926
```

Explicit types are available where they clarify public APIs or interop.
Sevarian uses annotation syntax so the name stays first and richer type
information can grow to the right.

```sev
width: int = 1920
height: int = 1080
```

## Control Flow

`while` keeps the condition next to the keyword. A scoped setup clause can follow
the condition with `with`.

```sev
while count < 3 with count := 0:
    print(count)
    count += 1
```

The `with` setup runs once before the first condition check. Names introduced by
the setup live only inside the loop condition and body.

## Functions

Functions use Python-like `def` syntax with optional return annotations.
Parameters use the same `name: Type` annotation style as bindings.

```sev
def add(a: int, b: int) -> int:
    return a + b
```

## Imports

Sevarian uses Python-style imports.

```sev
import math
import io as console

from math import sqrt
from io import print as write
```

## Classes And Traits

Classes are value types by default. Traits describe capabilities, not inheritance
hierarchies.

```sev
trait Drawable:
    draw()

class Point: Drawable
    x: float
    y: float

    def draw():
        print(x, y)
```

## Ownership

The compiler infers borrows, moves, and copies whenever it can. Explicit
ownership operations remain available when the program needs to say what it
means.

```sev
numbers = [1, 2, 3]

view = numbers.borrow()
copy = numbers.clone()
owned = numbers.move()
```

## Errors

Recoverable errors are values. `?` propagates an error from a `Result`.

```sev
def load(path: Path) -> Result[String, IOError]:
    data = read(path)?
    return Ok(data)
```

## Concurrency

Concurrent work starts with `spawn` and is joined with `await`.

```sev
worker = spawn fetch(url)
body = await worker
```

The compiler is responsible for proving safe access to shared state.

## Fixtures

The examples in `docs/examples` are source fixtures. As the parser and driver are
implemented, every fixture should move from "documented syntax" to "compiled by
tests".

The folders are ordered so the compiler can grow in passes:

1. `00-getting-started` through `03-collections-iteration` cover the Python-like
   core: indentation, bindings, calls, control flow, and built-in collections.
2. `04-classes-traits` through `07-generics-constraints` introduce Rust-flavored
   structure: value classes, traits, ownership, results, patterns, and generic
   constraints.
3. `08-concurrency` through `10-numerics-mlir` layer in Go-style concurrency,
   systems boundaries, and MLIR-oriented numeric kernels.
