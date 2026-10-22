SIP-0000: Title

Status: Draft | Accepted | Implementing | Implemented | Rejected | Superseded
Type: Language | Compiler | Runtime | Tooling | Package | Interop | Process
Tags: types | containers | blocks | ... (topics relevant to the document)
## Appendix

 all terms in a list
 - term 1: description
 -- subterms
 - term 2: description

## Goal(s)

1. Prevent bugs from linting errors
2. Improve code quality
3. enforce standard format automatically
4. If it cannot be enforced automatically it should atleast error to enforce proper code.
5. Linting and formatting are the same step! there's not point in an fmt/format seperation. The two are coupled so tightly it's pointless to seperate them.


## Examples & Problems

### Naming convention
variables start with lower case and do snake case a_b_c 

functions are snake case foo_bar()

user classes are capitalized

trait LanguageTrait:
...

class Language: LanguageTrait
    value: string = "French"


abbreviations ARE capitalized 
```
class MLIROperation:
...
```

language level operations can ignore these rules and prefer lower case
```
document = http.get("url")
values = async foobar() with self
```

No word mangaling, spell out the whole word, unless the simple form is DEAD simple (min,max) not frmt.
```
frmt -> format 

```


### Line width

Column length should be no greater than 80 characters

```
#valid
if condition:
    block...
#Invalid: the linter should wrap the conditions 
else condition and condition condition and condition condition and condition condition and condition 
->
else condition
    and condition
    and condition
    and condition:
```

---

Parameters cannot be greater than 3 this leads to over parameterization. 
Unless they are explicit
```
foo(a,b,c,d,e,f,g,h) # What happens if e/f are swapped -> BUG
foo(a,b,c) # This is a lot easier to reason over
foo(
    A=a,
    B=b,
    C=c,
    D=d,
    ...) # By making explicit what is mapping fields->params it prevents uncertainty +readability

#Exception is variadic params

print(a,b,c,d,e,f,g) # variadic there is not explicit mapping so it's ok. 
```

---
Blocks of severian code to show the desired behavior

Examples of corner cases

### Classes

```
# Classes construction is free form
class User:
    name: string
    age: int
    active: bool

    def active(status:bool) -> User | Error:
        if status and other_flag:
            self.active = status
        elif not other_flag:
            throw Error("Other flag was false")
        return User




user1 = User("Alice",36,true)
user2 =User{name:"Alice",age:36,active:true}
other_flag = true
user3 = User.builder().name("Alice").age(36).active(true)
```

## Tests

Blocks of severian code to test behavior and problems to resolve/reject certain behaviors
