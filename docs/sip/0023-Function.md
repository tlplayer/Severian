SIP-0000: Title

Status: Draft | Accepted | Implementing | Implemented | Rejected | Superseded
Type: Language | Compiler | Runtime | Tooling | Package | Interop | Process
Authors:
Created: YYYY-MM-DD
Target:
Supersedes:
Superseded by:

## Appendix

-- all terms, file formats, variables, classes, functions, traits in a table

## Context


## Goals

1. Functions should be clean and extensible building blocks
2. Functions should be able to fulfill contracts from traits
3. Functions should have complexity definitions for time/space built into the function
`F.complexity.time -> enum of complexities` sorting the functions should return the least complex first. 
This is to allow complexity sorting 
4. 

## Structure

```sev

enum BigO:
    constant
    linear
    quadratic
    exponential
    combinatorial
    undefined

class FunctionComplexity:
    time: BigO = undefined
    space: BigO = undefined



trait F: B
    complexity: FunctionComplexity | absent


#Example of a well formed function

def foobar(a:int) -> int | Error with
{
    fix 0 <= a <= 100 
    F.complexity.space := BigO.constant
    F.complexity.time := BigO.constant
}:
    return a

```

Why have the complexity on the function's with block? The API contract needs to be visible to the developers/callers. If it's hidden in a doc/transient you aren't promising what's to occur in the function
you also can catch things like calling a constant F in a loop and flag that the runtime/space is out of contract. This is also handy for dispatch checking

```
trait Bar:
    def foobar(a:T) with 
    {
        conditions...
    }



def foobar(a:int) -> int | Error with
{
    fix 0 <= a <= 100 
    F.complexity.space := BigO.constant
    F.complexity.time := BigO.constant
}:
    return a

```

## Problem(s)

## Examples

## Testing


