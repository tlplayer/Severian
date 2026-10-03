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

Errors need to help the development process

## Goals

1. Errors are understandable, point to where they occur, inform the root cuase 
2. Errors should give help and 
3. Errors are standardized provide, callstacks, files which compilation caused the issue
4. Only Errors are throwable
5. Errors should propose regression tests against their input (ie x caused error Y maybe this should not occur here's the x values and a test block to construct it)
6. Errors should be strict and force good practice, otherwise you give your program's runtime over to bad/sloppy code

## Problem(s)

- Categorizing errors 
- Formatting errors
- Callstacks
- information relevant to errors
- helpful information to solve the errors
## Examples

## Testing


