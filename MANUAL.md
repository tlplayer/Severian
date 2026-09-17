# Severian: Manual
# Author: Timothy Player
# Date 2026-09-14
# Status: Draft

## Introduction

Severian is a programming language that deals with these cyclical problems:
- Code should have quality through tests, naming, standards, types, and simplicity
- Code should be fast and flexible
- Rewriting code is a waste, extending code and improving one place should apply broadly to many places


## Prelude

The prelude is what every sev function gets without imports

```
my_list = []

```

## Classes

Classes attempt to do a couple things, standardize set,get methods. 

'''
Objects expose generic get() and set() operations to avoid getX()/setX()
methods for every field.

Field constraints may return Errors directly or delegate to functions.
Builders use the same set() path, so construction and later mutation obey
the same validation rules.

Tests use:
    assert(...)      hard requirement
    expect(...)      record failure and continue
    throws(...)      expected Error
    mock(...)        test-scoped function behavior
'''


## Testing

Tests are seperated cleanly from code. this provides core benefits. 
- Compilation can be isolated from testing.
- No external libraries are needed to bake into the project
- Function naming can be a string to avoid test_foo_bar() notation which becomes word soup
- different imports can live solely in the test
- Functionality ignores security since it's isolated from code/published artifacts MOSTLY (besides shadow)

```sev

def foo():
    return 1

test "Foo is a valid function and returns 2":
    assert(foo() == 1)

test "Foo is a valid function and returns 2":
    assert(foo() == 1)
```


## Interfaces and Implementers

Interfaces are handy because they can handle different data while keeping the same facade. 
Traits are nice because they allow flattening the inheritance from OOP. 
Classes apply implementations on those traits and traits can dispatch to their implementers
For example the `+` operator has a nice interface

```sev
a = 0 
b = 1.0

c = a+b

# + essentially took left/right operands the interface looks like:

trait Add:
    operator +(left:T,right:T):


BIG_NUMBER = 2_000_000
OTHER_BIG_NUMBER = 2_000_000

class BigAdder: Add
    operator +(left: int, right: int) -> int with {
        left > 10_000,
        right > 10_000,
    }:
        return add(left, right)

```

So overload resolution becomes:

- Match operand types.
- Match compile-time-known value constraints.
- Select the most specific valid Add.
- If values are runtime-only, either emit a guarded dispatch or fall back to the unconstrained Add.

More concretely:

Yes. That is cleaner: type resolution narrows the candidate set, then value resolution must produce exactly one match.

```sev
def resolve(name: Symbol, values: tuple[V]) -> F | Error:
    type_key = tuple(type(value) for value in values)

    type_functions = dispatch[name].get(type_key)

    if type_functions is None:
        return Error(
            "Dispatch method for {name} not implemented for type {type_key}"
        )

    candidate: F | Error = Error(
        "Dispatch method for {name} not implemented for value {values}"
    )

    for function in type_functions:
        if function.conditions(values):
            if candidate is not Error:
                return Error(
                    "Ambiguous dispatch for {name}: multiple implementations match {values}"
                )

            candidate = function

    return candidate
```

Semantically:

```text
resolve TYPE
    ↓
dispatch[name][argument types]
    ↓
candidate implementations

resolve VALUE
    ↓
evaluate each implementation's constraints
    ↓
0 matches  -> not implemented for these values
1 match    -> dispatch
2+ matches -> ambiguity error
```

For `Add`:

```sev
class PositiveAdd: Add:
    operator +(left: int, right: int) -> int with {
        left > 0,
        right > 0,
    }

class BigAdd: Add:
    operator +(left: int, right: int) -> int with {
        left > 10_000,
        right > 10_000,
    }
```

Then:

```sev
1 + 2
```

matches only `PositiveAdd`.

But:

```sev
20_000 + 30_000
```

matches both, so compilation/runtime dispatch fails:

```text
E: ambiguous dispatch for +(int, int)

matching implementations:
    PositiveAdd.+
    BigAdd.+

values:
    left  = 20000
    right = 30000
```

I prefer this over specificity ordering. The language does not guess author intent. If domains overlap, the programmer must make them disjoint:

```sev
class PositiveAdd: Add:
    operator +(left: int, right: int) -> int with {
        0 < left <= 10_000,
        0 < right <= 10_000,
    }
```
```text
For every dispatch point:

|matching implementations| == 1
```

Anything else is an error.

You can also represent the resolver as essentially:

```text
Dictionary[
    Operation,
    Dictionary[
        TypeTuple,
        List[ConstrainedImplementation]
    ]
]
```

So you avoid testing irrelevant implementations entirely. Type dispatch is indexing; value dispatch is predicate evaluation over an already-small bucket. This fits the grammar/operation model well because types identify the dispatch family while `with { ... }` partitions that family's value space.
