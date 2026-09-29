SIP-0000: Submodule

Status: Draft | Accepted | Implementing | Implemented | Rejected | Superseded
Type: Language | Compiler | Runtime | Tooling | Package | Interop | Process
Authors: Timothy Player
Created: YYYY-MM-DD
Target:
Supersedes:
Superseded by:

## Appendix

-- all terms, file formats, variables, classes, functions, traits in a table

## Summary

A submodule is the grouping of strongly connected components in the HIR block graph. If two distinct grphs are within one file
list and chain those two would produce similarly named .o or .so files which would have names like list.len.o or chain.len.o etc. 
They don't always map directly to compiled units but they establish a loose linkage between compiled units and submodules vs a module which is a collection of submodules. This is to allow better file compilation lookup for dependencies. Say 

```sev
class list[T]:
    storage: vector[T]

class chain[T]:
    storage[T]: pointer[T]
    def len(self):
        return 
```


## Examples

## Testing


