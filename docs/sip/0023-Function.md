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


```


## Problem(s)

## Examples

## Testing


