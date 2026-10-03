SIP-0012: Blocks

Status: Draft | Accepted | Implementing | Implemented | Rejected | Superseded
Type: Language | Compiler | Runtime | Tooling | Package | Interop | Process
Authors: Timothy Player
Created: YYYY-MM-DD
Target:
Supersedes:
Superseded by:

## Appendix

-- all terms, file formats, variables, classes, functions, traits in a table

## Examples

### Global scope blocks
The global block holds global variables and other blocks
```sev
i = 1
block
...
```
### Main Block

Main is a sub block of global 

```sev

def main():
    return 0

```

### Function Block

```sev
def foobar():
    return 0
```
### Class blocks

```sev
class Point:
    x = 0
    y = 0

    def coordinates() -> tuple(int,int):
        return (x,y)

```

Scenario
If a class has a variable, it's assumed always to be self.x essentially. If it conflicts with global, 
if needs to index into the global block for the variable

```sev
x = 1

class Point:
    x = 0
    y = 0

    def coordinates() -> tuple(int,int):
        return (x,y)
    
    def global_x() -> int:
        return global.x

```

### Trait Blocks

### Conditional blocks
if blocks can have chain/sibling blocks
```
if condition:
    block
elif condition:
    block
else condition:
    block
else:
    block
```

## Interface


## Data Model

## Testing


