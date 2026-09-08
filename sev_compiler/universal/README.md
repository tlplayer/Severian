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
|`To`| Token Classified parser input; includes identifiers, indentation and end-of-file, not just literals or symbols. |
|`Lx` | Lexeme Exact matched source text and span, before interpretation. It is not a literal value. |

This is the canonical compiler-term vocabulary. Symbols describe semantic roles;
they do not reserve generic parameter names or erase concrete type contracts.
Statement and instruction terms use `O`; arguments use `V`; type-kind metadata
uses `T`. Declarations, patterns, and nodes use `X` with their specific contracts.
`S` always means Shape, `N` always means Number of elements, and `C` always means
Container in this vocabulary. Constraints belong to `W`.

Modules use `X` with an explicit module role; `M` is reserved for macros.
Ordinary local variable names, descriptive type names, and established format
names such as AST, HIR, MIR, CFG, and MLIR are not compiler-term abbreviations.
