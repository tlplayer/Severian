# bool

`bool` represents a logical value.

A boolean has one of two values:

```sev
true
false
```

## Declaration

```sev
enabled: bool = true
disabled: bool = false
```

When the type can be inferred:

```sev
enabled = true
disabled = false
```

## Type information

| Property           | Value                    |
| ------------------ | ------------------------ |
| Type               | `bool`                   |
| Category           | Boolean primitive        |
| Values             | `true`, `false`          |
| Size               | `1b` logically           |
| Default value      | `false`                  |
| Equality           | Supported                |
| Ordering           | Not supported            |
| Logical operations | Supported                |
| Bitwise operations | Not implicitly supported |

`bool` implements the primitive interfaces required for logical operations and equality.

## Literals

Boolean literals are:

```sev
true
false
```

Their inferred type is `bool`.

```sev
value = true

assert(type(value) == bool)
```

## Logical operators

### `and`

Returns `true` when both operands are `true`.

```sev
true and true    # true
true and false   # false
false and true   # false
false and false  # false
```

```sev
authenticated = has_token and token_valid
```

`and` uses short-circuit evaluation. The right operand is evaluated only when the left operand is `true`.

```sev
ready = object != None and object.ready()
```

### `or`

Returns `true` when at least one operand is `true`.

```sev
true or true     # true
true or false    # true
false or true    # true
false or false   # false
```

`or` uses short-circuit evaluation. The right operand is evaluated only when the left operand is `false`.

```sev
available = cached or load()
```

### `not`

Inverts a boolean value.

```sev
not true     # false
not false    # true
```

```sev
if not connected:
    reconnect()
```

## Equality

Boolean values support equality and inequality.

```sev
true == true      # true
true == false     # false

true != false     # true
false != false    # false
```

Normally a boolean should be used directly:

```sev
if enabled:
    run()
```

rather than:

```sev
if enabled == true:
    run()
```

## Conditions

Expressions used as boolean conditions must evaluate to `bool`.

```sev
enabled = true

if enabled:
    run()
```

Non-boolean values can be treated as implicitly truthy/falsy see [truth](../../../../../docs/api/types/primitives/truth/truth.md)

```sev
count = 3

if count:
    print("hello")
```

Use an explicit condition instead:

```sev
if count > 0:
    ...
```

This applies to conditional constructs such as:

```sev
if condition:
    ...

while condition:
    ...

assert(condition)
```

## Boolean expressions

Comparison operations produce `bool`.

```sev
x = 10
y = 20

less = x < y
equal = x == y
different = x != y

assert(type(less) == bool)
```

Boolean expressions can be composed:

```sev
valid = x >= 0 and x < 100
```

```sev
allowed = is_admin or (authenticated and has_permission)
```

## Assignment

A variable declared as `bool` accepts boolean values.

```sev
active: bool = true
active = false
```

Assigning an incompatible type is rejected.

```sev
active: bool = 1
```

## Conversion

Conversions to `bool` can be explicit

```sev
bool(value)
```


## Functions

`bool` can be used anywhere another type can appear.

```sev
def is_valid(value: int) -> bool:
    return value >= 0
```

```sev
def set_enabled(enabled: bool):
    ...
```

Function calls are type checked:

```sev
set_enabled(true)
set_enabled(false)
```

An incompatible argument is rejected:

```sev
set_enabled(1)
```

## Collections

Boolean values may be stored in typed collections.

```sev
flags: list[bool] = [true, false, true]

if any(flags):
    print("This prints")
```

```sev

```

where supported by the corresponding collection or tensor API.

## Operators

`bool` exposes the following language operators:

| Operator | Signature       | Result |
| -------- | --------------- | ------ |
| `and`    | `bool and bool` | `bool` |
| `or`     | `bool or bool`  | `bool` |
| `not`    | `not bool`      | `bool` |
| `==`     | `bool == bool`  | `bool` |
| `!=`     | `bool != bool`  | `bool` |

Conceptually:

```sev
operator and(right: bool) -> bool
operator or(right: bool) -> bool
operator not() -> bool

operator ==(right: bool) -> bool
operator !=(right: bool) -> bool
```

## Unsupported arithmetic

Boolean values are not integers.

Arithmetic on `bool` is rejected:

```sev
true + false
true * 2
true - false
```

Use an explicit conversion when numeric behavior is intended.

```sev
int(true) -> 1
float(true) -> 1.0
```

if the numeric type supports conversion from `bool`.

## Bitwise operations

Logical boolean operations use:

```sev
and
or
not
```

Bitwise operators such as:

```sev
&
|
^
~
!
```

are separate operations and are not aliases for boolean logic unless explicitly provided by the relevant type/interface.

This distinction prevents logical conditions from being coupled to integer or bit-level representations.

## Examples

```sev
def can_access(
    authenticated: bool,
    is_admin: bool,
    has_permission: bool,
) -> bool:
    return is_admin or (authenticated and has_permission)


test "bool logical operations":
    assert(true and true)
    assert(not (true and false))

    assert(true or false)
    assert(not false)

    assert(true == true)
    assert(true != false)
```

