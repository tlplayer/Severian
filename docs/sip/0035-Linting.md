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
else condition\
    and condition\
    and condition\
    and condition:
```

---

Parameters cannot be greater than 8 this leads to over parameterization. 

```

```

---
Blocks of severian code to show the desired behavior

Examples of corner cases

## Tests

Blocks of severian code to test behavior and problems to resolve/reject certain behaviors
