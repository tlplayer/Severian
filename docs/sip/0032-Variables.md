SIP-0032: Variables

Status: Draft | Accepted | Implementing | Implemented | Rejected | Superseded
Type: Language | Compiler | Runtime | Tooling | Package | Interop | Process
Authors:
Created: YYYY-MM-DD
Target:
Supersedes:
Superseded by:

## Appendix

primtives:    int | char | float | bool | byte | pointer | type | infer | None | none | absent | atom
unit:         unit | duration | frequency | memory (byte) 
complex:      class | trait | generics | object (root of objects) | enum | union  | with | callable (F)
generic:      T | V | K | F | B | G | Y | N | C | W | L | E | R | M
collections:  list | map | dict | stack | heap | queue | chain | iterator
lifetime:     static 
scope:        local | global | self 
ownership:    move | borrow | view | copy | mirror | drop | _ (infer ownership)
mutability:   := | =
dynamic:      dynamic | any | numeric 
concurrency:  atomic

## Goal(s)

1. compile time type checking that prevents bugs
2. runtime flexibility to allow programs that don't have bugs to run
3. ability to optimize concurrent/memory optimized programs
4. prevent issues in other languages (int indices into memory etc.)

## Examples

```
# assignment to literal int
x = 1

# constant binding to of y to x, y cannot be reassigned
y := x

# We increment x
x += 1

#  y is equal to 1 because default assignment is not a reference to ints, it's a copy
assert(y == 1)

# So the := operator for int/int means constant binding to copy of value
y: int := copy x

# However, := on a complex object becomes whatever the default of the object permits
y: File := view file.open("file.txt")
s: Server := share server()
z: Server := s
a: Server := s

# The lifecycle of s stays until all the shares are dropped, similar to reference counters or atomic reference counters
drop(s)
drop(z)
drop(a)
```

```
#An atom is the smallest compilable unit that the compiler needs to get to. It's the boundary of Severian and IR/binary .o/.so files. For example

a = 1

# The int class will match the grammar of assignment and emit an atom to the compiler's LIR that is directly lowerable
```

```
type is the wrapper around types

type(0) -> int
type(0.0) -> float
type('c') -> char
type("") -> string

# integer conversion is hassle free when no bugs can occur and catch bugs if they pop up

test[Float in {f16, bf16, f32, f64, f128},
     Integer in {i8, i16, i32, i64, i128, u8, u16, u32, u64, u128}]
"float x int conversion and arithmetic":
    f: Float = Float(6.0)
    i: Integer = Integer(2)

    # explicit conversion edges exist both ways
    assert((i as F) == F(2.0))
    assert((f as I) == I(6))

    # safe round trips
    assert(((i as F) as I) == i)
    assert(((f as I) as F) == f)

    # mixed arithmetic promotes integer -> concrete float
    assert(f + i == F(8.0))
    assert(i + f == F(8.0))

    assert(f - i == F(4.0))
    assert(i - f == F(-4.0))

    assert(f * i == F(12.0))
    assert(i * f == F(12.0))

    assert(f / i == F(3.0))
    assert(i / f == F(1.0 / 3.0))

    assert(f // i == F(3.0))
    assert(f % i == F(0.0))

    # comparison accepts the conversion edge
    assert(f > i)
    assert(i < f)
    assert(f != i)
    assert(F(2.0) == i)

    def float_function(a:f32)

def f(x: i64):
    return x

f(10)                # Literal realizes as i64

x: i32 = 10
f(x)                 # Implicit lossless conversion to i64

y: number = 10
f(y)                 # Runtime conversion/check if y's type
                     # and range cannot be proven statically
```

Types are defaulted to infer, meaning static types are inferred. If the type cannot be inferred this is a compiler error. 
This is different from the runtime `any` which is to say `any` value can be used which is too permissive and is supposed to be used
for dynamic runtime values.

```
class X:
    p: pointer

a:infer = x().p

```

## Conversion

Type conversion works as a graph problem. If there is a strong link between 2 types, many types can be used to traverse the type graph.
for example

```sev
# Example
class dint:



```

## iterator

```sev
for x in [1,2,3]:
    
```

## absent and None

```
def foo() -> absent:
    return

foo()          # Valid: executes, produces no value.
value = foo()  # Compile error: cannot bind an absent result.
def missing() -> None:
    return None

value = missing()  # Valid: None is a value.
foobar(42)        # Argument absent → return None.
foobar(42, None)  # Argument supplied, containing None → return 42.
foobar(42, 7)     # Argument supplied → return 42.

'''
is absent checks whether an argument was supplied, independently of its value.
The Python None in my patch was an internal “no result” marker, not Severian’s None type. Omitted-argument handling is still unimplemented.
'''

```
### union

the union type is handy when 1 or more types can be used in a return, a function or a type.

```
def add[T:float|int](a:T,b:T) -> T:
    return a+b

add(0,0)
add(0.0,1.0)
add(0.1,1) # invalid exception T cannot be inferred from T int and T2 float

def fail() -> Error | int:
    return 0 if true else throw Error("Failed to be true")

#unions can comprise many other types
union numeric:
    i8
    f8
    ...
    f128
    i128
    u128
```

### Enum

The primary use case is destructuring enum variants:

```sev
enum Shape:
    Circle(radius: float)
    Rectangle(width: float, height: float)


def area(shape: Shape) -> float:
    match shape:
        case Circle(radius):
            return 3.1415926 * radius ** 2

        case Rectangle(width, height):
            return width * height
```

Unlike Python structural matching, Severian does not attempt to inspect arbitrary objects, invoke user-defined matching protocols, or infer object structure at runtime.

A pattern may only inspect structure already present in the static type.

Enums in match statements are exhaustive and cannoted default to _ they must all be enumerated. The lint step can replace _ with all the other cases for convenience and ease of future compilation but then the failure will propogate with a new enum value (THIS IS GOOD) because unaccounted for values in the match could lead to bugs. adding V1 |...Vn | _ would result in the same expansion 
of appending Vn+1 to the end as new enums come and replace the _
## Problem(s)

- Conversion of types how to avoid making a new conversion when the conversion path can be inferred?

```
class A:
    value: int

class B:
    value: string
    
    -> A:

class C:
    value: float

class D:
    value: float

    => D:
        return self
    -> B:
        return string(value)
    ~> A | Error:
        return A(string(value))

d:D = C(0) # C automatically gets converted to D type and assigned to d.
b:B = B(d) # because d can be represented as d they can be explicitly converted 
a:A = d    # Sometimes, d and a are a perfect match whole numbers in the floats, etc. so we pass them however they can cause an error.


The conversion operators have distinct guarantees:

| Operator | Meaning | Example |
|---|---|---|
| `A => B` | Conversion is always valid | `i32 => i64` |
| `A -> B` | Conversion is permitted but can fail | `i64 -> i32` |
| `A ~> B` | Conversion may lose information or precision | `i64 ~> f32` |

**Directionality and validity are separate concepts.** `i32 => i64` does not imply `i64 => i32`.

### 1. Conversion graph

Consider:

```text
A -> B => C -> D
 \        ^
  ------> C
```

Severian can derive conversion paths without defining every possible pair.

For example:

```text
A -> B
B => C
C -> D

A -> C
```

The compiler discovers:

| Source | Target | Path | Guarantee |
|---|---|---|---|
| A | B | A → B | May fail |
| B | C | B → C | Always valid |
| A | C | A → C | May fail |
| A | D | A → C → D | May fail |
| B | D | B → C → D | May fail |

Importantly, **the guarantee of a composed conversion depends on every edge**.

```text
A => B => C    produces A => C

A -> B => C    produces A -> C

A => B -> C    produces A -> C

A ~> B => C    produces an approximate A-to-C conversion
```

These are derived conversion paths, not new conversion declarations.

### 2. Graph resolution in HIR

I would make conversion resolution part of HIR type checking.

```text
Source
  ↓
Lexer
  ↓
Parser
  ↓
HIR
  ├── Resolve types
  ├── Construct conversion graph
  ├── Resolve conversion paths
  ├── Validate conversion guarantees
  └── Select conversion implementations
  ↓
MIR
  ↓
LIR
  ↓
MLIR
```

The graph should contain conversion metadata:

```text
ConversionEdge:
    source: TypeId
    target: TypeId
    kind: Always | Permitted | Approximate
    implementation: Callable
    cost: Complexity
```

A conversion from `i32` to `i64` becomes:

```text
ConversionEdge(
    source=i32,
    target=i64,
    kind=Always,
    implementation=integer_extend
)
```

HIR can retain the selected conversion path. MIR can then inline or optimize it.

### 3. Choosing between multiple paths

Suppose:

```text
A => B => D
A -> C => D
A -> D
```

Three paths reach `D`.

I would resolve them in this order:

1. Exact type match.
2. Direct `=>` conversion.
3. Chained `=>` conversions.
4. Permitted `->` conversions.
5. Approximate `~>` conversions, according to the conversion policy.

Within each category, prefer a direct conversion, then evaluate conversion costs.

However, user-defined conversions may have different semantics, even when their destination types match. If two paths are equally preferred and potentially produce different values, report an ambiguity instead of choosing arbitrarily.

### 4. Compile-time versus runtime errors

```python
def f(x: i8):
    return x

a: i16 = 100
b: i16 = 1000

f(a)  # i16 -> i8: permitted; value fits
f(b)  # i16 -> i8: compiler error if range is known

def g(x: i16):
    f(x)  # Runtime range check if value is unknown
```

A `->` edge permits the conversion operation. It does not guarantee that every value can be converted or all operations are valid.

An `=>` edge requires no runtime validity check.

You can apply the same model to `string -> int`:

```python
a: i32 = "123"    # Valid conversion static check
b: i32 = "hello"  # Compile-time error if evaluated statically
a: i32 = "123" + 1 # Compile time error 
parse("123")
def parse(s: string) -> i32:
    return i32(s) # runtime conversion attempt could error

#inferred types don't impose this restriction
a = "123"
a = 1
a = 1.0
a = 'c'
print(a) 
#Since a is an inferred type the aboves is valid so far as we can tell what type a is reassigned to 
```

Whether fallible conversions are allowed implicitly in assignments is a separate policy decision.

### 5. One distinction worth preserving

**Always valid does not necessarily mean lossless.**

For example, converting `i64` to `f32` can produce a floating-point value while losing integer precision.
```

### Collections

The most common use of the implementers for the C trait is the `list` type which looks like this:


```
values = [1,2,3,4,5]
# indexing can use slices, and ranges, 
values[0..2] -> [1,2,3]
# Indexing can pull many values
values[0,2] -> [1,3]
values[0..2) -> [1,2]
values(0..2) -> [2]
values(0) -> rejected by compiler

# .. can be subbed with : for the standard python syntax.
# values[0:2] note, [] means inclusive on a range, (..2) means exclusive on the range.  python uses [0,8) for [0:8] which isn't as mathematically consisten accurate and requires special knowledge to infer...



```