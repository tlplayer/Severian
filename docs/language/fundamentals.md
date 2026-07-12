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

Plain `=` bindings are stable and cannot be reassigned. `:=` is the explicit
form for changeable bindings.

```sev
int MaxRetries = 3
float Pi = 3.1415926
```

Explicit types are available where they clarify public APIs or interop.
Valued declarations use one concrete type before the name.

```sev
int width = 1920
int height = 1080
```

Uninitialized fields use `name: Type`, because class schemas tend to evolve and
the name is the stable part of the declaration.

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
Parameters use `name: Type`, which keeps the parameter name fixed while richer
accepted input types grow to the right.

```sev
def add(a: int, b: int) -> int:
    return a + b

test:
    assert(add(1, 2) == 3)
```

Use `|` for accepted type alternatives.

```sev
def parse(value: string | int | float) -> float:
    return float(value)
```

Tests can be attached directly to functions. They compile with the function and
can call it without extra ceremony.

```sev
def x() -> int:
    return 0

test:
    assert(x() == 0)
```

A `test:` block attaches to the immediately preceding function or constructor at
the same indentation level. Inside a class, an indented `test:` block attaches to
the constructor or method before it.

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

    def Point(px: float, py: float):
        x = px
        y = py

    def draw():
        print(x, y)
```

Constructors are class-scoped functions with the same name as the class. A class
may define more than one constructor when the signatures are distinct.

```sev
class X:
    value: int

    def X(x: int, y: int):
        value = x + y

    def X(x: int):
        value = x
```

Inside a constructor, assigning to a declared field initializes that field on the
new instance. This keeps construction compact without requiring `self.x = x`
boilerplate.

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

## Optional Values

Optional values represent presence or absence without null. A function returning
`Option[T]` returns either `present(value)` or `absent`.

```sev
def find_name(id: int) -> Option[string]:
    if id == 1:
        return present("ada")

    return absent

switch find_name(1):
    present name:
        print(name)

    absent:
        print("missing")
```

## Errors

Recoverable errors are values. A fallible function returns a `Result[T, E]`,
which is either `ok(value)` or `failure(error)`.

```sev
def load(path: Path) -> Result[string, IOError]:
    data = read(path)?
    return ok(data)
```

`?` unwraps the `ok(...)` value and returns early from the current function when
it sees a `failure(...)` outcome.

```sev
switch result:
    ok body:
        print(body)

    failure error:
        print(error)
```

Sevarian uses `switch` for structural branching. The word `match` is reserved
for domain syntax, such as regex helpers imported by a decorator.

## Concurrency

Calls block by default. `async` starts work without blocking the current task and
returns a handle that can be joined with `await`.

```sev
worker = async fetch(url)
body = await worker
```

The compiler is responsible for proving safe access to shared state for async
work. Code that intentionally bypasses those guards must live inside an explicit
`unsafe:` block.

```sev
unsafe:
    worker = async raw_driver_call()
    result = await worker
```

## Math Mode

Most functions use ordinary expression syntax. A function can opt into reserved
domain symbols with decorators.

```sev
import math

@math(X)
def transform(a: Matrix[f32], b: Matrix[f32]) -> Matrix[f32]:
    return a X b
```

Decorator arguments name the symbols being imported into that function's syntax.
For example, `@math(X)` imports only the math meaning of `X`, while
`@math(X, *)` imports both `X` and math-specific `*` behavior.

Multiple decorators can compose isolated symbol packs.

```sev
import math
import probability

@math(X)
@probability(P)
def expected(weights: Matrix[f32], samples: Matrix[f32]) -> f32:
    projected = weights X samples
    return P(projected > 0.5)
```

Outside decorated functions, those spellings are not silently reinterpreted. Each
decorator gives the compiler a link to the library or domain that owns the
symbols, their type rules, and their lowering behavior.

The same idea can reserve words for non-math domains.

```sev
import regex

@regex(match)
def has_slug(text: string) -> bool:
    return match text with r"[a-z]+-[0-9]+"
```

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
