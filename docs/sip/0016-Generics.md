SIP-0000: Title

Status: Draft | Accepted | Implementing | Implemented | Rejected | Superseded
Type: Language | Compiler | Runtime | Tooling | Package | Interop | Process
Authors:
Created: YYYY-MM-DD
Target:
Supersedes:
Superseded by:

## Appendix

These are implemented in syntax/generics and exported to prelude

- G grammer used to specify syntax, lexeme, and symbol of object
- T type generic, generates different type implementations 
- V value type string, int, etc. 
- K implements hashable and comparison operators
- C container type: list, array, vector, set, etc. 
- W constraint with conditions and operations sentence if W: B, else...: if ] 
- S sentence 
- Y symbol lexeme + token: int, x, async etc. :, @, etc. which contain meaning for other operations or themselves convey meaning the the compiler
- B block of code, if, else, def, class, trait, test etc. these encompass the other generics and other blocks themselves
- M macro, useful for generating lots of code denoted with -> 
- F function, could be expression, callable, etc.
- R result type, includes T | E | absent | None | ...
- E Error type. Needs to express enough information and pointers to where the error happened and how it went wrong
- L literal used to attach 0 -> int, 0.0 -> float, 
- N number of elements

## Context

Why have generics? They allow better communication/contracts between the code and the compiler in order to avoid custom fit solutions throughout the lowering/compilation stack. Without them we tumble towards type checking and other. Concretely,
trait Container:
...

Container as C
that way we lock C as what container is and then collections pulls that in. 

## Problem(s)

## Examples

## Testing


