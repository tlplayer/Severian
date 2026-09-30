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

The lexer takes source's output and syntax and greedily attempts to convert the raw string to the corresponding:

- blocks (classes, traits, enums, if, else, try, catch, elif, )
- sentences (for x in 0..10: while x with x := 0: etc.)
- operators (+,-,=,<=, >=, ->, etc.) 
- type
- variables (a)
- literals (0,0.0, -1)

## Problem(s)

## Examples

## Testing


