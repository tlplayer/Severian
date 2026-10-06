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

- mock
- throws
- assert
- when
- fix
- suffix
- prefix
- with


## Context



## Problem(s)

## Examples

```
#Basic function test
test "foobar":
    foobar()

# With allows sev test --integ to only call the functions within that scope inclusive 
test with integ "Print outputs X":
    assert("X" in stdout)

test with compiler "MIR error":
    reject:
        x = view 1
        #We would be modifying a view which the compiler rejects even though it's a literal
        x += 1
    accept:
        x = borrow 1
        x += 1

test with profile "Operation takes less than 10s" with
{
    fix profile.time < 10s,#If we ever go over 10s clip the test
    suffix profile.time < 5s, # If we go over 5s by the end we've failed
}
    foobar()
```


## Testing


