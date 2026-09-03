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


Symbols
T  Type
V  Value

E  Error
Ex Expression
M  Macro
S  Statement
D  Declaration
P  Pattern
L  Literal

O  Operation
I  Instruction
B  Block

A  Argument
R  Result
F  Callable

C  Constraint
K  Kind
Y  Symbol
N  Node
X  Any compiler term ex: X: E | S | D | P | T...