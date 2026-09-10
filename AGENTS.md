
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
    int, bool, string, list[T], User, T | Error

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
| `complex()` | Convert to a complex number                      |
| `float()`   | Convert to floating point                        |
| `format()`  | Format a value                                   |
| `hex()`     | Convert an integer to hexadecimal representation |
| `int()`     | Convert to integer                               |
| `string()`     | Convert to string                                |

### Functional

| Function   | Purpose                        |
| ---------- | ------------------------------ |
| `filter()` | Filter values using a function |
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
| `type()`       | Get or construct a type             |

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
| `slice()`     | Create a slice                           |
| `sorted()`    | Return values in sorted order            |
| `zip()`       | Iterate over multiple sequences together |

### Numeric

| Function   | Purpose                    |
| ---------- | -------------------------- |
| `abs()`    | Get the absolute value     |
| `max()`    | Get the maximum value      |
| `min()`    | Get the minimum value      |
| `pow()`    | Raise a value to a power   |
| `round()`  | Round a numeric value      |
| `sum()`    | Sum values                 |

