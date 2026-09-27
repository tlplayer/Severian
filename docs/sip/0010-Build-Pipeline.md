SIP-0010: Build Pipeline

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
Current compilers follow a narrowing funnel for simplicity. This is good for processing speed but is bad for ensuring good software gets through.
Severian will needs to define it's syntax within itself. Usually this is hidden in IR in other languages and frankly results in a desync between
interface/implementation details of the language. For example, str/string/byte-strings are all represented in hidden c whereas if they were written in the same structure that classes etc. were written in it would enable easier fixing/modifications and shows a better compiler pipeline.

### Phases of the Compiler

### Syntax

Defining primitives, generic types, modules, and general compiler structures and processes 

example:
```sev
if x == y:
    return 0
```

Here we have a block which looks at two variables, unknown types, which must 
implemnet operator/sentence sentence equal[self,"==": Y,comparator: string|int|float|char] 

The block is an if block (these can be dynamically defined and are all automatically registered without tweaking the whole compiler stack)
inside we have an operation that interacts with the CFG and returns to the caller block return 0 which looks at the current block and returns to the caller block's 
previous cursor with the Result[R] type with value 0 which could've been an error

### Frontend

Source: Loads the source text and other imports into collections 
Parsing: groups items into blocks 


## Problem(s)

## Examples

## Testing




