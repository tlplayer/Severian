# State transitions and typestate

Severian uses one transition graph for runtime enum states and compile-time
typestate. A transition-aware enum lists legal successors directly:

```sev
enum Download:
    Pending -> Connecting,
    Connecting -> Receiving | Failed,
    Receiving -> Complete | Failed,
    Complete,
    Failed,
```

An enum becomes transition-aware when any variant declares `->`. Variants with
no successors are terminal. Trailing commas are optional.

State changes use ordinary assignment to a changeable binding:

```sev
state := Pending
state = Connecting
state = Receiving
```

If both variants are statically known, an edge missing from the declaration is
a compile-time `E000213`. When a function or dynamic boundary hides the current
variant, the compiler inserts a runtime transition check; an invalid edge fails
with the same diagnostic and includes both states. A dynamic path therefore
cannot weaken the state machine.

## Typestate

A transition enum's variants can also be phantom arguments of a generic class.
An existing function contract gates a method to a state, so typestate adds no
method-specific keyword:

```sev
enum SocketState:
    Closed -> Connected
    Connected -> Closed

class Socket[State]:
    descriptor: int

    def connect(address: string) -> Socket[Connected] with { State == Closed }:
        return Socket[Connected](descriptor)

    def send(data: string) -> int with { State == Connected }:
        return size(data)
```

During generic specialization, a satisfied state clause is removed and an
unsatisfied method is omitted. Consequently, `Socket[Closed].send(...)` is a
compile-time `E000214`, not a runtime precondition.

State availability clauses may combine equality with `and`/`or`, or use
`State in [Connecting, Receiving]` when one method is legal in several states.

Flow-sensitive rebinding follows the enum graph:

```sev
current := socket             # Socket[Closed]
current = current.connect(host)
current.send(payload)         # current is Socket[Connected]
```

Rebinding between two specializations of the same class is accepted only when
their state arguments share a transition enum and that enum declares the edge.
This model applies equally to `File[Open]`, `Tensor[GPU]`,
`Model[Compiled]`, and `Transaction[Committed]`.

## Junctions

Severian does not introduce Raku-style junction values. Membership remains the
canonical finite-choice spelling:

```sev
if value in [1, 2, 3]:
    ...
```

`all()` and `any()` remain reductions over boolean collections. Tensor-wide
conditions should be supplied by tensor operations returning boolean
collections or scalar predicates, rather than by changing equality and
ordering into context-sensitive junction operations.

## Module functors

Severian does not add module-to-module function syntax. Generic traits and
classes already carry the useful capability:

```sev
trait Storage:
    def get(key: string) -> Any
    def set(key: string, value: Any)

class Cache[S: Storage]:
    storage: S
```

This preserves ordinary type checking, specialization, and trait registry
behavior without a second parameterized-module system.
