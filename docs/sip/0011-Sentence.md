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

The sentence is a more complex way to express semantic meaning while expanding to LIR/MIR level information. 
Operator is a basic sentence which essentially blends compiler responsibility with symbols/operations. 
Sentence is the bridge between scalar operations/well defined single variables ops and more complex operations 
similar to blocks of code. The barrier between them is blocks have their own scope whereas sentences operate within blocks.

The goal is to explain in enough hard simple details what the complex operation means in the language for better self expression. This should allow avoiding if type == array and symbol... throughout the compiler. We should be able to expand the elegant syntax while maintaining expressiveness in the language

## Examples

```
#Array length ten with 0 initialized
a:array[int] = array[int,10](0)

#below the compiler needs to either KNOW what [2] following a means at all layers or parse the meaning
a[2] = 1+2

#simplified class example
class array[T,N]:
    sentence assign[self,'[',index:int,']',V]():
        self.pointer[index] = V

```


## Problem(s)

1. Order of sentence resolution in the parser/source
2. exact meaning of syntax


## Testing

3 examples of sentences which apply complex operations

```sev
#
task = async F with self
```

