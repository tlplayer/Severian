Function

Status: Draft | Accepted | Implementing | Implemented | Rejected | Superseded
Type: Language | Compiler | Runtime | Tooling | Package | Interop | Process
Authors:
Created: YYYY-MM-DD
Target:
Supersedes:
Superseded by:

## Appendix

-- all terms, file formats, variables, classes, functions, traits in a table

## Goals

1. Functions should be clean and extensible building blocks
2. Functions should be able to fulfill contracts from traits
3. Functions should have complexity definitions for time/space built into the function
`F.complexity.time -> enum of complexities` sorting the functions should return the least complex first. 
This is to allow complexity sorting 
4. Functions should allow compact operations square = lambda x: x*2 \n square([1,2,3])

## Structure

```sev

enum BigO:
    constant(0) {0}
    linear(1)
    quadratic(2)
    undefined(3)
    exponential(4)
    combinatorial(5)

class FunctionComplexity:
    time: BigO = undefined
    space: BigO = undefined



trait F: B
    complexity: FunctionComplexity | absent
    params: list[]

#Example of a well formed function

def foobar(a:int) -> int | Error with
{
    fix 0 <= a <= 100 
    foobar.complexity.space := BigO.constant
    foobar.complexity.time := BigO.constant
}:
    return a

```

Why have the complexity on the function's with block? The API contract needs to be visible to the developers/callers. If it's hidden in a doc/transient you aren't promising what's to occur in the function
you also can catch things like calling a constant F in a loop and flag that the runtime/space is out of contract. This is also handy for dispatch checking

```
trait Bar:
    def foobar(a:T) with 
    {
        conditions...,
    }



def foobar(a:int) -> int | Error with
{
    fix 0 <= a <= 100,
    foobar.complexity.space := BigO.constant,
    foobar.complexity.time := BigO.constant,
}:
    return a

```

## Problem(s)

- Initializing and evaluating expressions statically vs. dynamically
- How to parse dynamic runtimes based on dominating or potentially dominating structure
- 

## Examples

### F describes the callable
```py

# Proposed explicit F contract:
func: F[(int) -> int] = lambda x: x + 1

# Existing signature notation can be shorthand:
func: (int) -> int = lambda x: x + 1

test:
    assert(func(3) == 4)

# Rejected: incompatible result type.
bad: F[(int) -> int] = lambda x: "hello"

# Variadic params
def foobar(params...: int):
    return params

test:
    assert(foobar() == ())
    assert(foobar(1) == (1,))
    assert(foobar(1, 2, 3) == (1, 2, 3))

    a, b, c = foobar(1, 2, 3)
    assert(a == 1)
    assert(b == 2)
    assert(c == 3)

# Either an integer argument pack OR one list.
def first(params: int... | list[int]):
    return params

# Each argument may independently be an integer OR a list.
def second(params...: int | list[int]):
    return params

test:
    assert(second(1, [2, 3]) == (1, [2, 3]))

def foobar(params: int... | list[int]):
    return params

test:
    assert(foobar(1, 2, 3) == (1, 2, 3))
    assert(foobar([1, 2, 3]) == (1, 2, 3))
    assert(foobar() == ())
    assert(foobar([]) == ())
```

## Testing


