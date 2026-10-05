SIP-0012: Blocks

Status: Draft | Accepted | Implementing | Implemented | Rejected | Superseded
Type: Language | Compiler | Runtime | Tooling | Package | Interop | Process
Authors: Timothy Player
Created: YYYY-MM-DD
Target:
Supersedes:
Superseded by:

## Appendix

| Block |
|---|
| `global` |
| `main` |
| `def` |
| `operator` |
| `class` |
| `trait` |
| `enum` |
| `extend` |
| `test` |
| `if` |
| `elif` |
| `else` |
| `for` |
| `while` |
| `match` |
| `switch` |
| `case` |
| `try` |
| `catch` |
| `with` |
| `unsafe` |

## Block

A block implements B and establishes a semantic region. Its implementing declaration defines its grammar, contents, bindings, context, and applicable execution behavior. HIR preserves the resolved block and its relationships. 

## Examples

Blocks compose other blocks

```
class A:
    class B:
        value: int = 0
    def B_value():
        return B().value

#Useful in tests
test with compiler "Needs custom import":
    class X:
        value: int = 10
        _read_only: bool = true 
    x = X() 
    assert(x.value == 10)
    reject:
        x._read_only = false

test "Needs custom import":
    import profile
    start = profile.time()
    A().B_value()
    stop = profile.time()
    assert(stop-start < 5)

test with profile "golden path profile" with 
{
    suffix time < 5s
}:
    A().B_Value()
```

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

traits are both properties of classes and the interface of the language. 

```sev
trait Interface:
    def foobar()
    def default_functionality():
        return "default implementation that would be annoying to remake on all implementers to fulfill interface"
        

class Implementation: Interface
    def foobar():
        return 1

test "Traits are callable to their dispatch":
    assert(interface.foobar())

test "Implementations get default behavior":
    assert(Implementation().default_functionality() is string)
```

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


## With

With blocks have access to some really nice hooking/properties of the block

prefix: happens before the block. This is the default behavior in a with block
suffix: happens after the block is done
fix: this makes sure during the block a behavior is observed. A Promise to entry/inside/exit

```
def positive(x) with
{
    x > 0 -> Error("x must be positive")
}:
    return x

def negative(x) with
{
    suffix x < 0 -> Error("x must be negative on exit")
}:
    return -x

def zero(x) with
{
    fix x == 0 -> Error("x must be 0")
}:
    return x

```

It's also useful for initializing variables and fields and scope them out when unneeded.
For example

```
#historical method

i = 0
while i < 0:
    print(i)
    i+=1

#The above has a couple problems, i is in global scope so say you have two loops

i = 0
while i < 0:
    print(i)
    i+=1

#You could forget to change this to a different param, lints sometimes say use a different variable etc. 
i = 0
while i < 0:
    print(i)
    #Could forget to increment
    i+=1

#The prefered method is to make it part of the scope

while i < 0 with i := 0
    print(i)
    i += 1

#This is better, but forgetting i += 1 is going to get/bite you

while i < 0 with 
{
    i := 0,
    defer i += 1 # defer is an alias for suffix 
}:
    print(i)


#We've removed some pinch points but there are more useful scenarios, grid search for one
grid = [[0]*10]
grid[(rand(0..9),rand(0..9))] = 1
while with
{
    i,j := 0,0, #initialization happens once := /=
    0 <= i < 10 -> continue,
    0 <= j < 10 -> continue,
    suffix grid[i][j] == goal -> return i,j,
}


```

## Data Model


## Testing

