# SIP-00XX: Typed Pattern Matching

Status: Proposal
Feature: `match` / `case`
Scope: enums, literals, tuples, unions

## Summary

Add a small, statically typed pattern-matching system centered on Severian's type system.

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

---

## Motivation

Severian needs a concise way to consume:

* enum variants
* tagged unions
* tuples
* literals
* optional/result-like types

without requiring chains such as:

```sev
if shape is Circle:
    radius = shape.radius
    ...

else if shape is Rectangle:
    width = shape.width
    height = shape.height
    ...
```

Python's `match` provides useful syntax but supports substantially more runtime structural behavior than Severian needs.

Severian should keep:

* `match value:`
* ordered `case` clauses
* destructuring
* literals
* `_`
* `|`
* guards

and reject the rest.

---

# 1. Enums are tagged unions

An enum variant may contain values:

```sev
enum Shape:
    Circle(radius: float)
    Rectangle(width: float, height: float)
    Point
```

Conceptually:

```text
Shape =
    Circle(float)
  | Rectangle(float, float)
  | Point
```

Each variant has:

1. a compile-time discriminant
2. a statically known payload
3. statically known field types

No runtime reflection is required.

---

# 2. Variant matching

Payloads are bound explicitly:

```sev
match shape:
    case Circle(radius):
        print(radius)

    case Rectangle(width, height):
        print(width * height)

    case Point:
        print("point")
```

The names in a pattern are new case-local bindings.

Therefore:

```sev
case Circle(r):
```

is valid even though the enum field was declared as:

```sev
Circle(radius: float)
```

`r` receives the `radius` payload.

There is no Python-style `__match_args__`. The payload order is exactly the order declared by the enum variant.

---

# 3. Ignoring values

`_` ignores a value:

```sev
match shape:
    case Circle(_):
        print("circle")

    case Rectangle(width, _):
        print(width)

    case Point:
        pass
```

A bare variant may also ignore its complete payload:

```sev
match shape:
    case Circle:
        print("some circle")

    case Rectangle:
        print("some rectangle")

    case Point:
        pass
```

Thus:

```sev
case Circle:
```

means:

```sev
case Circle(_):
```

for a one-field variant.

For multiple fields:

```sev
case Rectangle:
```

means:

```sev
case Rectangle(_, _):
```

This makes tag-only matching cheap to express.

---

# 4. Literal matching

Primitive literals may be matched directly:

```sev
match status:
    case 200:
        return "ok"

    case 404:
        return "missing"

    case _:
        return "other"
```

Supported literal patterns initially:

```text
integer
float
bool
char
string
None
enum constant
```

Literal matching uses the normal equality semantics of that type.

It does not invoke a separate pattern-matching protocol.

---

# 5. OR patterns

Cases with identical behavior may be combined:

```sev
match status:
    case 401 | 403 | 404:
        return "denied"

    case _:
        return "other"
```

For the initial implementation, `|` patterns may not introduce bindings.

Rejected:

```sev
case Circle(radius) | Rectangle(radius, _):
```

This restriction avoids Python's rule requiring alternatives to produce compatible binding sets.

It can be relaxed later if a real use case appears.

---

# 6. Guards

Cases may have a boolean guard:

```sev
match shape:
    case Circle(radius) if radius == 0.0:
        return 0.0

    case Circle(radius):
        return 3.1415926 * radius ** 2

    case Rectangle(width, height):
        return width * height
```

Evaluation is strictly:

```text
match variant
    ↓
bind payload
    ↓
evaluate guard
    ↓
execute case
```

If the guard is false, all bindings from that case are discarded and matching continues.

No partially bound variables escape.

---

# 7. Bindings are case scoped

Bindings exist only inside their case:

```sev
match shape:
    case Circle(radius):
        print(radius)

print(radius)
# TypeError: radius is not defined here
```

This is intentionally different from Python.

A failed pattern can never modify surrounding scope.

---

# 8. Exhaustiveness

Matching a closed type is checked at compile time.

This is valid:

```sev
enum Result[T, E]:
    Ok(T)
    Err(E)


match result:
    case Ok(value):
        use(value)

    case Err(error):
        handle(error)
```

This is rejected:

```sev
match result:
    case Ok(value):
        use(value)
```

Diagnostic:

```text
MatchError: non-exhaustive match

missing:
    Err(E)
```

A wildcard closes the match:

```sev
match result:
    case Ok(value):
        use(value)

    case _:
        handle_other()
```

This allows adding new enum variants without requiring every caller to change when the caller intentionally does not care about the distinction.

---

# 9. Unreachable cases

The compiler diagnoses cases that can never execute:

```sev
match shape:
    case Circle:
        pass

    case Circle(radius):
        use(radius)
```

Error:

```text
MatchError: unreachable case

Circle is completely matched by an earlier case
```

Likewise:

```sev
match value:
    case _:
        pass

    case 10:
        pass
```

is invalid.

---

# 10. Tuple destructuring

Fixed structural types may be destructured:

```sev
point: tuple[int, int] = (10, 20)

match point:
    case (0, 0):
        print("origin")

    case (x, 0):
        print(x)

    case (x, y):
        print(x, y)
```

Tuple matching is statically known.

There is no general "sequence protocol."

Therefore this does not automatically apply to:

```text
list
deque
vector
array with unknown shape
arbitrary iterable
```

Those types may define normal APIs instead.

---

# 11. Nested patterns

Patterns compose only where the underlying types compose.

```sev
enum Option[T]:
    Some(T)
    None


value: Option[Shape]

match value:
    case Some(Circle(radius)):
        print(radius)

    case Some(Rectangle(width, height)):
        print(width * height)

    case Some(Point):
        pass

    case None:
        pass
```

This remains purely type-directed.

---

# 12. Match expressions

`match` should eventually also produce a value:

```sev
def area(shape: Shape) -> float:
    return match shape:
        case Circle(radius):
            3.1415926 * radius ** 2

        case Rectangle(width, height):
            width * height

        case Point:
            0.0
```

Every reachable case must produce a compatible type.

For:

```sev
case Circle(radius):
    float

case Rectangle(width, height):
    float
```

the result type is:

```sev
float
```

This should reuse normal Severian type-unification rules rather than adding match-specific conversion rules.

---

# 13. No arbitrary object patterns

Severian deliberately rejects Python-style matching such as:

```python
case Click(position=(x, y))
```

unless `Click` is explicitly represented by a destructurable Severian type such as an enum or tuple.

There is no:

```text
__match_args__
match protocol
runtime attribute lookup
reflection
implicit isinstance chain
mapping protocol
sequence protocol
```

A class is not automatically a pattern.

If destructuring a class becomes useful, it should be added through an explicit language-level declaration rather than runtime convention.

---

# 14. No mapping patterns

Not included:

```python
case {"name": name, "age": age}:
```

Use ordinary dictionary operations:

```sev
if "name" in value and "age" in value:
    ...
```

or convert unstructured data into a typed value first:

```sev
person = Person(value)

match person:
    ...
```

Pattern matching should operate on known types, not double as schema validation.

---

# 15. No `as` pattern

Python permits constructs similar to:

```python
case Circle(radius) as circle:
```

Severian does not need this.

The original matched value already exists:

```sev
match shape:
    case Circle(radius):
        use(shape)
        use(radius)
```

No second binding syntax is necessary.

---

# 16. No bare-name ambiguity

A bare lowercase identifier in a pattern is always a binding:

```sev
case value:
```

However this pattern is irrefutable and should normally be written:

```sev
case _:
```

Variant names and constants are resolved through their static symbol category.

For example:

```sev
case Circle(radius):
case Direction.North:
case 42:
case _:
```

The compiler never guesses whether `Circle` means "capture a variable named Circle" or "match the Circle variant."

Symbols already know what they are.

---

# 17. Pure matching semantics

Testing whether a pattern matches must not execute arbitrary user code.

For:

```sev
case Rectangle(width, height):
```

the compiler performs only:

```text
read discriminant
compare discriminant
project payload field 0
project payload field 1
```

Pattern matching itself cannot invoke:

```text
getattr
len
getitem
user equality callbacks
reflection
dynamic matching hooks
```

Guards remain normal expressions and may perform normal operations.

This gives `match` deterministic lowering and makes optimization straightforward.

---

# 18. Ownership

Pattern inspection does not implicitly consume the matched value.

```sev
match value:
    case Some(item):
        ...
```

`item` follows the same ownership rules as an ordinary projection from `value`.

`match` introduces no separate move/copy/reference model.

The ownership checker remains the source of truth.

---

# 19. Proposed grammar

Initial grammar:

```text
match_statement
    := "match" expression ":" case+

case
    := "case" pattern guard? ":" block

guard
    := "if" expression

pattern
    := wildcard
     | literal
     | variant
     | tuple
     | or_pattern

wildcard
    := "_"

variant
    := VariantName
     | VariantName "(" pattern_list? ")"

tuple
    := "(" pattern_list ")"

or_pattern
    := closed_pattern ("|" closed_pattern)+
```

No separate grammar exists for:

```text
class_pattern
mapping_pattern
sequence_protocol_pattern
as_pattern
capture_pattern
value_pattern
```

A name's meaning comes from normal Severian symbol resolution.

---

# 20. Compiler lowering

Given:

```sev
match shape:
    case Circle(radius):
        circle(radius)

    case Rectangle(width, height):
        rectangle(width, height)
```

semantic analysis resolves:

```text
match Shape

Circle:
    discriminant = Shape.Circle
    bindings:
        radius: float <- payload[0]

Rectangle:
    discriminant = Shape.Rectangle
    bindings:
        width: float  <- payload[0]
        height: float <- payload[1]
```

MIR can lower this directly into:

```text
%tag = enum.tag %shape

switch %tag
    Shape.Circle -> ^circle
    Shape.Rectangle -> ^rectangle

^circle:
    %radius = enum.field %shape[0]
    ...

^rectangle:
    %width = enum.field %shape[0]
    %height = enum.field %shape[1]
    ...
```

After this point, `match` no longer needs to exist.

It becomes ordinary CFG.

---

# 21. Exhaustiveness belongs in semantic analysis

The frontend should produce a resolved pattern representation such as:

```text
Pattern
    Wildcard
    Literal(Value)
    Variant {
        type
        discriminant
        fields: [Pattern]
    }
    Tuple([Pattern])
    Or([Pattern])
```

Semantic analysis owns:

```text
variant resolution
binding types
arity checking
exhaustiveness
unreachable-case detection
guard typing
```

MIR should not rediscover any of this.

MIR receives resolved discriminants and projections.

---

# 22. Diagnostics

Wrong variant:

```sev
match shape:
    case Ok(value):
```

```text
TypeError: Shape has no variant `Ok`
```

Wrong arity:

```sev
case Rectangle(width):
```

```text
PatternError: Rectangle expects 2 fields, got 1

Rectangle(width: float, height: float)
```

Wrong tuple size:

```sev
value: tuple[int, int]

case (x, y, z):
```

```text
PatternError: tuple[int, int] cannot match a 3-element pattern
```

Missing cases:

```text
MatchError: non-exhaustive match

missing:
    Rectangle(float, float)
```

Unreachable case:

```text
MatchError: unreachable case

Circle was completely matched at line 14
```

---

# 23. Initial implementation phases

## Phase 1 — enums

Implement only:

```sev
case Variant
case Variant(...)
case _
```

Tests:

```sev
enum Shape:
    Circle(radius: float)
    Rectangle(width: float, height: float)
```

Verify:

* discriminant selection
* payload binding
* wildcard
* exhaustiveness
* unreachable cases
* wrong variant
* wrong payload arity

## Phase 2 — literals and guards

Add:

```sev
case 0:
case 0.0:
case true:
case "foo":
case 'c':
```

Verify guards cannot leak bindings.
## Phase 3 — guards

Add:

```sev
case Circle(radius) if radius > 0.0:
case Circle(radius) with radius > 0.0:
case Circle(radius) with 
{
    radius > 0.0,
    radius < 10.0,
    odd(radius) -> error("radius must be even") 
}:
```

## Phase 4 — match expressions

Add:

```sev
value = match input:
    case A:
        1
    case B:
        2
```

Use existing type unification for result typing.

## Phase 5 — OR patterns

Add simple non-binding alternatives:

```sev
case 401 | 403 | 404:
```

Do not initially allow bound OR patterns.

---

# 24. End-to-end test

```sev
enum Shape:
    Circle(radius: float)
    Rectangle(width: float, height: float)
    Trapezoid(side1: float, side2: float, height: float)
    Oblong(width: float, height: float)
    Point


def area(shape: Shape) -> float:
    return match shape:
        case Circle(radius):
            3.1415926 * radius ** 2

        case Rectangle(width, height):
            width * height

        case trapezoid: Trapezoid | Oblong:
            trapezoid.width * trapezoid.height

        case Point:
            0.0

test "matches enum variants and destructures payloads":
    assert(area(Circle(2.0)) > 12.5)
    assert(area(Rectangle(3.0, 4.0)) == 12.0)
    assert(area(Point) == 0.0)
```

Expected compiler pipeline:

```text
parse
  ↓
resolve Shape
  ↓
resolve variant discriminants
  ↓
type payload bindings
  ↓
check exhaustiveness
  ↓
Pattern MIR
  ↓
CFG switch + payload projections
  ↓
MLIR
```

---

# 25. Design rule

The core rule is:

> `match` may destructure what the compiler already knows structurally from the type.

It does not discover structure.

That keeps matching as a type-system and CFG feature rather than creating a second dynamic object protocol.

ADDENDUM:
Yes. I would add this as a distinct pattern form: a typed capture pattern.

The important semantic rule should be:

```sev
case shape: T1 | T2:
```

means:

1. Match if the value is `T1` or `T2`.
2. Narrow `shape` to `T1 | T2`.
3. Bind the whole variant, not its payload fields.
4. Permit `shape.member` only when that member is valid across the narrowed union.

So your example becomes:

```sev
enum Shape:
    Circle(radius: float)
    Rectangle(width: float, height: float)
    Trapezoid(side1: float, side2: float, height: float)
    Oblong(width: float, height: float)
    Point


def area(shape: Shape) -> float:
    return match shape:
        case Circle(radius):
            3.1415926 * radius ** 2

        case Rectangle(width, height):
            width * height

        case trapezoid: Trapezoid | Oblong:
            trapezoid.width * trapezoid.height

        case Point:
            0.0
```

However, that specific `width` access would only compile if both `Trapezoid` and `Oblong` expose `width`. If they don't, the compiler should reject it.

I would add this section to the SIP:

# Typed Capture Patterns

A case may match one or more types while retaining the complete matched value.

```sev
case value: T:
```

or:

```sev
case value: T1 | T2 | T3:
```

Example:

```sev
enum Shape:
    Circle(radius: float)
    Rectangle(width: float, height: float)
    Trapezoid(
        width: float,
        height: float,
        side1: float,
        side2: float,
    )
    Oblong(width: float, height: float)
    Point


def area(shape: Shape) -> float:
    return match shape:
        case Circle(radius):
            3.1415926 * radius ** 2

        case Rectangle(width, height):
            width * height

        case quadrilateral: Trapezoid | Oblong:
            quadrilateral.width * quadrilateral.height

        case Point:
            0.0
```

Within:

```sev
case quadrilateral: Trapezoid | Oblong:
```

the compiler narrows:

```text
quadrilateral: Trapezoid | Oblong
```

The original variable remains:

```text
shape: Shape
```

The new binding is scoped to that case.

## Variant Types

Enum variants are usable as narrowed types.

Given:

```sev
enum Shape:
    Circle(radius: float)
    Rectangle(width: float, height: float)
```

the compiler recognizes:

```text
Shape
├── Shape.Circle
└── Shape.Rectangle
```

Within the enum's natural scope, these may be written:

```sev
Circle
Rectangle
```

Therefore:

```sev
case circle: Circle:
```

binds:

```text
circle: Circle
```

rather than merely:

```text
circle: Shape
```

Similarly:

```sev
case quadrilateral: Trapezoid | Oblong:
```

binds:

```text
quadrilateral: Trapezoid | Oblong
```

## Member Access Through Narrowed Unions

Member access on a narrowed union is allowed when every possible member of the union exposes that member with a compatible type.

Given:

```sev
Trapezoid(
    width: float,
    height: float,
    side1: float,
    side2: float,
)

Oblong(
    width: float,
    height: float,
)
```

this is valid:

```sev
case shape: Trapezoid | Oblong:
    shape.width
    shape.height
```

because:

```text
Trapezoid.width: float
Oblong.width:    float

Trapezoid.height: float
Oblong.height:    float
```

The effective narrowed interface is:

```text
Trapezoid | Oblong

common:
    width: float
    height: float
```

This is rejected:

```sev
case shape: Trapezoid | Oblong:
    shape.side1
```

because `Oblong` has no `side1`.

Diagnostic:

```text
TypeError: `side1` is not available on every member of
Trapezoid | Oblong

available on:
    Trapezoid

missing on:
    Oblong
```

The programmer can narrow again:

```sev
case shape: Trapezoid | Oblong:
    match shape:
        case trapezoid: Trapezoid:
            use(trapezoid.side1)

        case oblong: Oblong:
            ...
```

## Compatible Member Types

Members do not necessarily need identical types if Severian's normal type system can determine a common usable type.

For:

```text
T1.value: i32
T2.value: i64
```

access through:

```sev
case value: T1 | T2:
    value.value
```

uses the normal union/member type rules.

Conceptually:

```text
value.value: i32 | i64
```

No special match-specific conversion occurs.

## Typed Capture Versus Destructuring

These forms have intentionally different meanings.

Destructure:

```sev
case Circle(radius):
```

produces:

```text
radius: float
```

Typed capture:

```sev
case circle: Circle:
```

produces:

```text
circle: Circle
```

Multiple typed capture:

```sev
case shape: Circle | Rectangle:
```

produces:

```text
shape: Circle | Rectangle
```

Wildcard type match:

```sev
case _: Circle | Rectangle:
```

tests the types without creating a binding.

This gives four simple forms:

```sev
case Circle(radius):
    # destructure

case circle: Circle:
    # capture whole variant

case shape: Circle | Rectangle:
    # capture narrowed union

case _:
    # everything else
```

## Exhaustiveness

A multi-type case contributes every listed type to exhaustiveness analysis.

Given:

```sev
enum Shape:
    Circle(radius: float)
    Rectangle(width: float, height: float)
    Trapezoid(width: float, height: float)
    Oblong(width: float, height: float)
    Point
```

then:

```sev
match shape:
    case Circle(radius):
        ...

    case Rectangle(width, height):
        ...

    case quad: Trapezoid | Oblong:
        ...

    case Point:
        ...
```

covers:

```text
Circle
Rectangle
Trapezoid
Oblong
Point
```

and is exhaustive.

Adding:

```sev
Triangle(...)
```

would produce:

```text
MatchError: non-exhaustive match

missing:
    Triangle
```

## Overlap Detection

Cases are ordered, but the compiler should diagnose completely unreachable type alternatives.

Rejected:

```sev
match shape:
    case trapezoid: Trapezoid:
        ...

    case quad: Trapezoid | Oblong:
        ...
```

The second case is still partially reachable because of `Oblong`, so the compiler should report:

```text
MatchWarning: partially redundant pattern

Trapezoid was already matched by an earlier case.

remaining:
    Oblong
```

Completely unreachable:

```sev
match shape:
    case quad: Trapezoid | Oblong:
        ...

    case trapezoid: Trapezoid:
        ...
```

produces:

```text
MatchError: unreachable case

Trapezoid was completely matched by an earlier case
```

## Grammar

Extend patterns with:

```text
pattern
    := destructure_pattern
     | typed_capture
     | literal
     | tuple
     | wildcard

typed_capture
    := identifier ":" type_pattern

type_pattern
    := type
     | type ("|" type)+
```

Therefore:

```sev
case shape: Trapezoid | Oblong:
```

does not need a separate OR-pattern mechanism.

It is simply a binding followed by a union type pattern.

## Lowering

Given:

```sev
case quad: Trapezoid | Oblong:
    use(quad.width)
```

semantic analysis produces approximately:

```text
Pattern:
    TypedCapture {
        binding: quad
        type: Trapezoid | Oblong
    }
```

CFG lowering becomes:

```text
%tag = enum.tag %shape

switch %tag:
    Trapezoid -> ^quad
    Oblong    -> ^quad
    ...
```

Both edges enter:

```text
^quad:
    %quad = %shape narrowed to Trapezoid | Oblong
    %width = union.project.common %quad, width
    ...
```

No runtime type discovery is required.

The discriminant already determines which variant is present.

## Design Rule

A typed capture pattern:

```sev
case value: T1 | T2 | ...:
```

means:

> Match any listed type and expose the complete value under their statically known union.

It is not destructuring, reflection, or a runtime pattern protocol.

I prefer this over allowing:

```sev
case Trapezoid | Oblong:
```

alone because the binding is useful. It also gives Severian a general narrowing construct that works beyond enums:

```sev
value: int | float | string

match value:
    case number: int | float:
        calculate(number)

    case text: string:
        parse(text)
```

That makes `case x: T1 | T2` essentially the match equivalent of a statically checked type narrowing operation, which fits Severian's existing union model well.
