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

The class is the MOA for programs in modern programming languages. They encapsolate data and methods and a good class structure can greatly speed up development 

## Goal(s)

1. Classes are easy to use and interoperate on
2. Classes implement traits when needed but have a default set of traits they have
3. Class construction and use is easy
4. Classes are easy to extend without inheritance
5. Severian's class structure is flat and composed of traits and member variable classes

## Examples

See docs/examples/01-types/02-classes

```
#Classes must start with uppercase letter, they can sometimes use lowercase for important items 
class A:
    x
    def value():
        return x

#Classes allow for default construction in a vareity of useful ways

value = A(x=1).value()
assert(value == 1)

value = A({"x":1}).value()
assert(value == 1)


#Pipelined builders are also valid

value = A.builder().x(3).build().value()
assert(value == 3)

#They have a default constructor and when an explicity destructor is needed they allow for implementing ownership specifications on the type
x = A()

drop x
#print(x) would give a compiler error

```


```
#Classes can implement interfaces of traits

trait Print:
    def print()

class Printable: Print
    def print():
        print("{}")

#Call the trait to route to an implementer
assert(Print.print() == "{}")
#If no route is found it throws unimplemented as an error

#Fields in classes can have with conditions
class User:
    name: string 
    {
        len(name) == 0 -> Error("name cannot be empty"),
    }

    age: int 
    {
        age < 0 -> Error("age cannot be negative"),
        age > 130 -> invalid_age(age),
        age < 18 and restricted_account() -> Error("restricted accounts require an adult"),
    }

# User("",-1) would cause an error
# 
```

There are optional storage policies:

```
def allocate[self]() #Gives unitialized storage for the user to handle in rare cases mostly handled by the language

```


Traits allow for API contracts to be compiler verified, intelligent routing without nexus classes
and 
## Responsibilities

## Data Models

## API

## Structure

## Problem(s)

## Testing


