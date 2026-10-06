SIP-0032: Variables

Status: Draft | Accepted | Implementing | Implemented | Rejected | Superseded
Type: Language | Compiler | Runtime | Tooling | Package | Interop | Process
Authors:
Created: YYYY-MM-DD
Target:
Supersedes:
Superseded by:

## Appendix

primtives:    int | char | float | bool | byte | pointer | type | None| none | absent | atom
unit:         unit | duration | frequency | memory (byte) 
complex:      class | trait | generics | enum | union | iterator | with
generic:      T | V | K | F | B | G | Y | N | C | W | L | E | R | M
collections:  list | map | dict | stack | heap | queue | chain 
lifetime:     static 
scope:        local | global | self 
ownership:    move | borrow | view | copy | mirror | drop 
mutability:   := | =
dynamic:      dynamic | any | numeric 
concurrency:  atomic

## Context

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

#  y is equal to 1
assert(y == 1)

# So the := operator for int/int means constant binding to copy of value
y: int := copy x

# However, := on a complex object becomes a view of that object
y: File := view file.open("file.txt")


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
```

## Responsibilities

## Data Models

## API

## Structure

## Problem(s)

## Testing


