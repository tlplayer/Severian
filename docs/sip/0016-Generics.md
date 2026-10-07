SIP-0016: Generics

Status: Draft | Accepted | Implementing | Implemented | Rejected | Superseded
Type: Language | Compiler | Runtime | Tooling | Package | Interop | Process
Authors:
Created: YYYY-MM-DD
Target:
Supersedes:
Superseded by:

## Appendix

These are implemented in syntax/generics and exported to prelude

- G grammer used to specify syntax, lexeme, and symbol of object
- T type generic, generates different type implementations 
- V value type string, int, etc. 
- K implements hashable and comparison operators
- C container type: list, array, vector, set, etc. 
- W constraint with conditions and operations sentence if W: B, else...: if ] 
- S sentence 
- Y symbol lexeme + token: int, x, async etc. :, @, etc. which contain meaning for other operations or themselves convey meaning the the compiler
- B block of code, if, else, def, class, trait, test etc. these encompass the other generics and other blocks themselves
- M macro, useful for generating lots of code denoted with -> 
- F function, could be expression, callable, etc.
- R result type, includes T | E | absent | None | ...
- E Error type. Needs to express enough information and pointers to where the error happened and how it went wrong
- L literal used to attach 0 -> int, 0.0 -> float, 
- N number of elements

## Context

Why have generics? They allow better communication/contracts between the code and the compiler in order to avoid custom fit solutions throughout the lowering/compilation stack. Without them we tumble towards type checking and other. Concretely,
trait Container:
...

Container as C
that way we lock C as what container is and then collections pulls that in. 

## Problem(s)

## Examples

These examples specify intended generic resolution and lowering.

### Type specialization

```sev
def identity[T = int](value: T) -> T:
    return value
```

```text
Source
    → parse declaration and template parameters
    → resolve explicit or inferred arguments
    → apply defaults for remaining parameters
    → check the resolved contracts
    → specialize the declaration
    → generate typed MIR
    → check ownership and insert cleanup
    → lower through LIR to MLIR
```

Ownership comes from the resolved type’s object contract. A specialization retains parameter and result lifetime relationships.

### Integer

```sev
original: int = 42
result = identity(original)
```

```text
Resolution:
    T = int
    specialization = identity[int]

Ownership:
    int declares copy
    argument and result use value copies

MIR:
    call identity[int](copy original) -> int

LIR:
    parameter: i64
    result: i64
```

```mlir
func.func @identity_int(%value: i64) -> i64 {
    func.return %value : i64
}
```

### String

```sev
original: string = "hello"
result = identity(original)
```

```text
Resolution:
    T = string
    specialization = identity[string]

Ownership:
    string declares view
    result views original's storage
    result cannot outlive that storage

MIR:
    call identity[string](view original) -> view string
    result lifetime depends on original

LIR:
    forward the string descriptor
    do not copy the underlying bytes
```

```mlir
func.func @identity_string(%value: memref<?xi8>) -> memref<?xi8> {
    func.return %value : memref<?xi8>
}
```

### Custom object

```sev
class Point:
    x: int
    y: int

original := Point(10, 20)
result = identity(original)
```

For this example, `Point` declares `view` as its passing contract and uses an address as its value representation.

```text
Resolution:
    T = Point
    specialization = identity[Point]

Ownership:
    original retains ownership
    result views original

MIR:
    call identity[Point](view original) -> view Point

LIR:
    Point supplies its layout and address representation
    forward the address
```

```mlir
func.func @identity_Point(%value: !llvm.ptr) -> !llvm.ptr {
    func.return %value : !llvm.ptr
}
```

```text
Point declares copy:
    → invoke Point's copy implementation
    → result owns the copied object

Point declares move:
    → transfer ownership into the function
    → transfer ownership into the result
    → original becomes unavailable
```

### Dynamic

```sev
original: dynamic = 42
result = identity(original)
```

Example representation:

```text
dynamic:
    descriptor: address of concrete type metadata and operations
    payload: address of concrete value storage
```

```text
Resolution:
    T = dynamic
    specialization = identity[dynamic]
    concrete payload type remains runtime information

Construction:
    create integer payload
    associate its int descriptor

Ownership:
    dynamic declares view in this example
    result retains a lifetime dependency on original

MIR:
    call identity[dynamic](view original) -> view dynamic

LIR:
    forward descriptor and payload addresses
```

```mlir
func.func @identity_dynamic(
    %value: !llvm.struct<(ptr, ptr)>
) -> !llvm.struct<(ptr, ptr)> {
    func.return %value : !llvm.struct<(ptr, ptr)>
}
```

Forwarding a dynamic value requires no payload operation dispatch. Applying an operation to its payload resolves that operation through the runtime descriptor.

### Default type argument

```sev
def reserve[T = int](count: int) -> array[T]:
    unsafe:
        return allocate[T](count)

integers = reserve(128)
smaller = reserve[i32](128)
```

```text
reserve(128)
    → count does not determine the element type
    → default T = int
    → specialize reserve[int]
    → allocate array[int]

reserve[i32](128)
    → explicit T = i32
    → specialize reserve[i32]
    → allocate array[i32]
```

### Default callable argument

```sev
def some_function_default[T](value: T) -> T:
    return value

def increment(value: int) -> int:
    return value + 1

def apply[T = int, F = some_function_default](value: T) -> T:
    return F(value)

a = apply(10)
b = apply[int, increment](10)
c = apply("hello")
```

```text
apply(10)
    → infer T = int
    → default F = some_function_default
    → resolve F[int]
    → result = 10

apply[int, increment](10)
    → explicit T = int
    → explicit F = increment
    → verify argument, result, ownership, and effect contracts
    → result = 11

apply("hello")
    → infer T = string
    → default F = some_function_default
    → resolve F[string]
    → result views the input string
```

```text
Argument selection:
    explicit argument → inferred argument → declared default

Specialization identity:
    declaration identity + resolved template arguments

Different resolved F arguments:
    → different specialization identities

Defaults:
    → resolve in the declaration's scope
    → may depend on earlier template parameters
    → must satisfy the same contracts as explicit arguments
```

## Testing


