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

- mock: when this occurs it should return this instead
- throws: the following operation should throw something
- assert: if this is 
- when: if a condition happens these mocked values and or values should equal this 
- fix: during the execution of this block this condition is *always true (if we leave the block we can't promise no change)
- suffix: at the end of this test, this should be true
- prefix: at the start of this block, this should be true
- with: conditions for the test as well as packed imports test with profile does many imports/exposes variables and tests for devices. 



## Examples & Problem(s) 

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


