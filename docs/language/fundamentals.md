# Language Fundamentals

Severian's surface syntax is intentionally familiar to Python programmers, but
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

Class-like types use PascalCase, including `Result`, `Option`, `Channel`, and
`Buffer`. Ubiquitous primitives such as `int`, `float`, and `string` remain
lowercase. Parameterized types follow Python's square-bracket convention.
Parentheses are reserved for calls and runtime construction.

### Naming

Severian uses casing to make a name's role visible without extra punctuation.

- Classes, traits, and enums use `UpperCamelCase`: `ChatEvent`, `TcpConnection`.
- Enum variants use `UpperCamelCase`: `Join`, `Say`, `Leave`.
- Functions and methods use `lowerCamelCase`: `runHub`, `readLine`.
- Variables, parameters, and fields use `snake_case`: `client_id`, `next_job_id`.
- Primitive types remain lowercase: `int`, `float`, `string`.

The linter reports names that do not follow the convention. A variant arm may
omit its field list; the fields declared by that variant are then bound under
their declared names for the arm's scope.

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

Specialized tests place their modes before the optional name. `test` remains the
only test declaration.

```sev
test with property "reverse twice preserves values":
    values = [1, 2, 3]
    assert(values.reversed().reversed() == values)

test with bench "parser throughput":
    assert(parse("42") == 42)

test with chaos "read failures":
    assert(read() != absent)

test with property and chaos "generated input failures":
    assert(validate([1, 2, 3]))
```

The property runner controls case generation, random seeds, distributions, and
shrinking. The chaos runner derives a function's complete reachable failure
surface from the call graph, including failures introduced by callees. Tests add
returned values with `chaos.add(when function return value)` and thrown
exceptions with `chaos.add(when function throw error)`. This injection pattern
is valid inside any test and forbidden outside test scope. The runner injects
one event at a time by default, and handled events remain in the transitive
catalog. Compatible modes compose explicitly with `and`; commas do not combine
test modes.

## Imports

Severian uses Python-style imports.

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
new instance. Methods and constructors access their current object's fields by
name without an explicit receiver parameter. `self` names the current execution
context, not a class instance.

## Counts, Bytes, And Midpoints

`size(values)` returns the number of elements in a collection. `values.size()`
returns the number of bytes in the object. Severian does not provide `.len()`.

```sev
values = [10, 20, 30]

count = size(values)
bytes = values.size()
middle = values.mid()
```

`values.mid()` is the collection's midpoint primitive.

### Shape-Safety Hypothesis

Index-based iteration borrows the collection's shape for the loop. Safe code may
replace elements, but it cannot resize the collection while that shape is live.

```sev
for index in indices(values):
    values[index] += 1
```

Operations such as `pop`, `remove`, `clear`, and resizing are rejected inside
that loop. An `unsafe` region may override the shape restriction, but indexing
remains bounds-checked. Removing a bounds check is a separate unsafe operation.

Frozen collections preserve both their contents and shape. Fixed arrays preserve
their shape while allowing element mutation. Resizable collections retain runtime
bounds checks whenever the compiler cannot prove an index belongs to their
current `indices(values)` set.

## Ownership

The compiler infers borrows, moves, and copies whenever it can. The reserved
prefix keywords `view`, `borrow`, `clone`, and `move` make an ownership operation
explicit when the program needs to say what it means. `view` creates a shared
read-only borrow, `borrow` creates an exclusive mutable borrow, `clone` creates
an independent owner, and `move` transfers ownership.

```sev
numbers := [1, 2, 3]

values_view = view numbers
print(values_view[0])

writable = borrow numbers
writable.push(4)

copy = clone numbers
owned = move copy
```

Parameter declarations contain names and types, not ownership modes. Parameters
are viewed by default. A call may use `view`, `borrow`, `clone`, or `move` on an
argument when the ownership operation must be explicit.

```sev
def update(values: list[int]):
    values.push(4)

update(borrow numbers)
```

## Optional Values

Optional values represent presence or absence without null. A function returning
`Option[type]` returns either `present(value)` or `absent`.

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

Recoverable errors are values. A fallible function returns a
`Result[type, exception]`, which contains either a successful value or a failure
exception.

```sev
def load(path: Path) -> Result[string, IOError]:
    data ?= read(path)
    return data
```

`?=` requires a binding name. It binds the successful value and returns early
from the current function when it receives a failure outcome. It is invalid to
write `?=` without storing that value. Return an exact `Result` directly when no
successful value needs to be stored.

Inside a function returning `Result[type, exception]`, returning a value of
`type` produces the successful result. Returning an expression that already has
the exact declared `Result` type forwards it unchanged. A bare `return` produces
a successful `unit` result when the declared success type is `unit`.

```sev
switch result:
    ok body:
        print(body)

    failure error:
        print(error)
```

Severian uses `switch` for structural branching. The word `match` is reserved
for domain syntax, such as regex helpers imported by a decorator.

## Function Contracts

A function may declare entry requirements in a `with { ... }` suffix. Within
those contract braces only, every comma-separated requirement must hold, so a
comma is equivalent to `and`. `and` may be written explicitly. There is no
contract shorthand for `or`; alternatives must use the `or` keyword. This rule
does not change the meaning of commas in calls, tuples, collection literals, or
any other Severian construct.

```sev
def runJob(job_id: int, connection: network.TcpConnection) with {
    0 <= job_id <= 1000,
    connection != invalid,
    with connection,
}:
    process(job_id, connection)
```

This contract is equivalent to requiring the first two expressions with
`and`, plus the `connection` capability. A caller must supply that capability
explicitly with `runJob(job_id, connection) with connection`. A missing or
incorrect capability is a compile-time error. A value requirement that can be
proved false is also a compile-time error; a requirement depending on runtime
data is checked once at function entry.

The capability belongs in the function contract and call suffix. Wrapping the
function's entire body in `with connection:` when the contract already requires
that capability is a compile-time error.

## Concurrency

Calls block by default. `async` starts work without blocking the current task and
returns a handle that can be joined with `await`.

```sev
worker = async fetch(url) with self
body = await worker
```

Channels use the PascalCase `Channel` class and an explicit `Buffer` policy.
Receiving is an ordinary `await` on the channel.

```sev
messages = Channel[string] with Buffer(16)
producer = async send "hello" with messages
message = await messages
```

Use `switch` when one task must receive from whichever of several channels is
ready. Exactly one ready arm commits; the other channel receives remain
untouched. The word after `from` names the source channel. An uppercase pattern
such as `Job from jobs:` destructures the received value and binds its declared
fields; a lowercase pattern such as `message from messages:` binds the entire
value under that name.

The optional `while` condition repeats selection without adding another indented
block. Its `with` setup runs once and remains scoped to the switch.

```sev
switch messages and commands while received < 2 with received := 0:
    command from commands:
        await handle(command) with runtime and lock
        received += 1

    message from messages:
        process(message)
        received += 1

    fail error:
        panic("Channels collapsed", error)
```

Every task names its lifetime owner. A task declared `with self` cannot outlive
the current execution. A task declared `with runtime` is runtime-owned and must
be created inside an explicit `unsafe:` block.

Arguments passed to an async call are frozen by default. The child may read
them, but it cannot mutate the caller's values. Frozen arguments need no lock.
Code requests scoped access to a captured binding by naming it after the task
owner. The parent cannot perform a conflicting operation on that binding until
the child completes.

```sev
task = async do(x) with self and x
```

Here `x` remains owned by the surrounding scope, `self` owns the task, and the
borrow checker keeps the task's access to `x` within both lifetimes. Explicit
`clone x` and `move x` arguments remain available when the child needs an
independent value or permanent ownership transfer.

`with self and lock` transfers the lock capability to the child for the call.
The parent does not retain the lock while it waits. When several children need
the same mutable value, the lock serializes their access.

```sev
class Account:
    balance: int
    status: string

    def Account():
        balance = 0
        status = "surplus"

    def increment(amount: int):
        balance += amount
        status = "debt" if balance < 0 else "surplus"

    def decrement(amount: int):
        balance -= amount
        status = "debt" if balance < 0 else "surplus"

def main():
    account := Account()
    credit = async account.increment(10) with self and lock
    debit = async account.decrement(15) with self and lock

    await credit, debit
```

The lock protects the relationship between `balance` and `status`, not the
integer operations alone. Each child completes both field updates before the
other child may mutate the account. Calling either mutable method asynchronously
with only `with self` is rejected.

Use a lexical lock when several synchronous operations must form one exclusive
critical section:

```sev
with lock:
    increment(10)
    record_transaction("credit")
```

Mutable raw values otherwise cannot cross a task boundary. Frozen values permit
shared reads. Atomic values permit synchronized scalar mutation. Mutex locks
guard larger mutable state.

```sev
counter := atomic int 0
left = async counter += 1 with self
right = async counter += 1 with self

await left
await right
```

```sev
unsafe:
    worker = async raw_driver_call() with runtime

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
4. `12-enums-aliases` onward cover evolving features: enums, aliases, method
   mutation contracts, Cargo-like packaging, specialized tests, and
   compiler-stage fixture organization.
