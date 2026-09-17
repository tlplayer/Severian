# Severian: Manual
# Author: Timothy Player
# Date 2026-09-14
# Status: Draft

## Introduction

Severian is a programming language that deals with these cyclical problems:
- Code should have quality through tests, naming, standards, types, and simplicity
- Code should be fast and flexible
- Rewriting code is a waste, extending code and improving one place should apply broadly to many places

## Scopes

Severian has the following scopes:

- global 
- main
- test

By isolating global certain setup work can occur before all else
By isolating test into it's own block we can break some rules for safety/behavior while not polluting the file
We also get to keep tests proximally close to what they test 


```sev
#Constant global executes

X = "value"

def main():
    assert()
    def foobar():
        return 2
    test "foobar() returns 2 always":
        assert(foobar() == 2)

    return true

test "Testing if value is set":
    assert(X == "value")

'''
Scoping the test's functionality with allows finer grain test scoping and pipelining in the package
`sev test --integ` will run the below while sev test --unit only runs the smaller scopes
This allows extension and tiering of tests to follow a pyramid pattern
'''
test with integ "testing main":
    assert(main() == true) 
```

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

## Memory & Ownership



```sev
view # Const reference to an object, you can view it but it might be changed/altered by another process
copy # copy an object 
move # Take ownership of the object
borrow # Temporarily take ownership of the object and return it potentially changed
mirror # Copy on write equivalent but flat, and cannot mirror a mirror to a T
```

Severian have some core methods for owning objects

```sev

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
    operator +(left: int, right: int) -> int with 
    {
        left > 0,
        right > 0,
    }

class BigAdd: Add:
    operator +(left: int, right: int) -> int with 
    {
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



 NEVER REMOVE CODE, BEFORE IMPLEMENTING THE REPLACEMENT!
For an agent friendly representation use to see the call graph/agent friendly IR 
```
sev_rust # Rust compiler
sev # sev_compiler 
sev update # Updates to latest version  

#See agent IR
sev build --emit agent-ir

#test the current package
sev test
```


Type
    What a value is.
    int, bool, string, list[T], User, union, T | Error

Primitive
    Compiler-known fundamental types.
    bool, integers, floats, char, etc.

Operator
    Semantic operations represented by syntax.
    +, -, *, ==, [], =

Expression
    Produces a value.
    x + 1
    foo()
    Point(1, 2)

Statement
    Performs an action/control-flow operation.
    return
    break
    continue
    assignment

Declaration
    Introduces something into the program.
    def
    class
    trait
    enum
    variable

Symbol
    The resolved identity of a declaration.
    "x" in source eventually becomes Symbol/DefId(x)

Literal
    Source representation of a direct value.
    10
    1.5
    true
    "abc"

Pattern
    Destructures/tests values.
    match/case
    union variants
    tuple destructuring

Conversion
    Relationship between source and target types.
    i8 -> i32
    f64 -> f32
    string -> Path

Callable
    Functions, methods, constructors, operators.
    Parameters + result + effects.

Constraint
    Semantic requirement that must hold.
    T implements Ordered
    x: int {x > 0}
    type equality

Fixity With [W]
    def interface() with condition
    def interface() with 
    foo(a) with a>5:
    for i in 0..5 with j := 1:
        if i > j:...
    def increment_by_ten(x,view y) with 
    {
        prefix fix x > 0, # x enters the function and must be greater than 0
        suffix x >= 10, # At the end of the function x must be greater than or equal to 10
        fix x < 100, # Throughout the function x cannot be greater than 99
        defer unchanged(y) # If y changes throughout the function fail, defer is an alias for fix
    }

Effect
    What computation does beyond returning a value.
    throw
    mutation
    borrow/move
    IO
    async


| Symbol | Meaning |
| ------ | ------- |
| `T` | Type |
| `V` | Value |
| `S` | Shape |
| `N` | Number of elements |
| `E` | Error |
| `Ex` | Expression |
| `M` | Macro → operations |
| `L` | Literal |
| `O` | Operation |
| `B` | Block |
| `R` | Result |
| `F` | Callable |
| `W` | With-clause operations (including constraints) |
| `C` | Container |
| `Y` | Symbol |
| `X` | Any compiler term |
| `G` | Grammar |
| `To` | Token — classified parser input; includes identifiers, indentation and end-of-file, not just literals or symbols. |
| `Lx` | Lexeme — exact matched source text and span, before interpretation. It is not a literal value. |

This is the canonical compiler-term vocabulary. Symbols describe semantic roles;
they do not reserve generic parameter names or erase concrete type contracts.
Statement and instruction terms use `O`; arguments use `V`; type-kind metadata
uses `T`. Declarations, patterns, and nodes use `X` with their specific contracts.
`S` always means Shape, `N` always means Number of elements, and `C` always means
Container in this vocabulary. Constraints belong to `W`.

Modules use `X` with an explicit module role; `M` is reserved for macros.
Ordinary local variable names, descriptive type names, and established format
names such as AST, HIR, MIR, CFG, and MLIR are not compiler-term abbreviations.



# Prelude functions

API ID: `prelude.function.assert`


### Async

| Function  | Purpose                                         |
| --------- | ----------------------------------------------- |
| `async` | Call function asynchronously                    |
| `await` | Await a task created by asyn |

### Attributes and Properties

| Function     | Purpose                           |
| ------------ | --------------------------------- |
| `borrow`  | borrow an object                  |
| `clone`  | copy an object             |
| `view`  | only view but not modify object                  |
| `drop`  | drop memory of an object                  |
| `mirror`  | similar to COw cannot mirror a mirror'd item                  |

### Collections

| Function       | Purpose                           |
| -------------- | --------------------------------- |
| `array()`  | Create an immutable byte array       |
| `vector()`  | Create a mutable byte array       |
| `dict()`       | Create a dictionary               |
| `map()`       | Create a map               |
| `list()`       | Create a list                     |
| `set()`        | Create a set                      |
| `tuple()`      | Create a tuple                    |

### Compilation and Execution

| Function       | Purpose                |
| -------------- | ---------------------- |
| `import()` | Import a module        |
| `compile()`    | Compile source         |
| `eval()`       | Evaluate an expression |
| `exe()`       | Execute code           |
| `parse()`       | Parse code to ast/IR           |

### Conversion and Representation

| Function    | Purpose                                          |
| ----------- | ------------------------------------------------ |
| `ascii()`   | Produce an ASCII representation                  |
| `bin()`     | Convert an integer to binary representation      |
| `bool()`    | Convert to boolean                               |
| `char()`     | Convert an integer to a character                |
| `float()`   | Convert to floating point                        |
| `format()`  | Format a value                                   |
| `f""`       | Format a string                                   |
| `hex()`     | Convert an integer to hexadecimal representation |
| `int()`     | Convert to integer                               |
| `numeric()`     | Convert to numeric                               |
| `string()`     | Convert to string                                |

### Functional

| Function   | Purpose                        |
| ---------- | ------------------------------ |
| `filter()` | Filter values using a function |
| `stream()` | Creates a stream of items |
| `apply()`    | Apply a function over values   |

### Input, Output, and Debugging

| Function       | Purpose                |
| -------------- | ---------------------- |
| `breakpoint()` | Enter the debugger     |
| `help()`       | Display help           |
| `input()`      | Read input             |
| `open()`       | Open a file or stream  |
| `print()`      | Write formatted output |

### Introspection and Reflection

| Function       | Purpose                             |
| -------------- | ----------------------------------- |
| `hash()`       | Get a hash value                    |
| `type(T) -> T`       | Get or construct a type             |

### Iteration and Sequences

| Function      | Purpose                                  |
| ------------- | ---------------------------------------- |
| `all()`       | Test whether all values are true         |
| `any()`       | Test whether any value is true           |
| `enumerate()` | Iterate with indexes                     |
| `iterator()`  | Get an iterator                          |
| `len()`       | Get the number of items                  |
| `size()`      | Get the number of items                  |
| `bytes()`     | Get the number of bytes owned by item   |
| `next()`      | Get the next iterator item               |
| `range()`     | Create an integer range                  |
| `0..5`        | Create an integer range                  |
| `slice()`     | Create a slice                           |
| `sort()`      | Return values in sorted order            |
| `zip()`       | Iterate over multiple sequences together |

### Numeric

| Function   | Purpose                    |
| ---------- | -------------------------- |
| `abs()`    | Get the absolute value     |
| `max(T...)`    | Get the maximum value of a list can take containers      |
| `min(T...)`    | Get the minimum value of a list can take containers      |
| `**`    | Raise a value to a power   |
| `round()`  | Round a numeric value      |
| `sum(T...)`    | Sum values                 |

