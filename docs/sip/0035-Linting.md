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
6. Linting is enforced as at the frontend head of compilation. The language has the format the user does not. 


## Examples & Problems

### Compilation and Automatic formatting

To reiterate, formatting and linting are the same thing as far as the language is concerned.

Pre Source -> Lexer we need to lint the source files to adhere to the format. Thus we modify they files 
to adhere to our lint rules. 

Formatting is automatically applied and rewrites/changes the source file to correct. Then proceeds with compilation.

### Naming convention
variables start with lower case and do snake case a_b_c 

Global constants are all caps snaked MODULE_CONFIG

Global variables are lower snakecase but require module.snake_case keywords  this is auto added for the user. 

functions are snake case foo_bar()

user classes are capitalized

trait LanguageTrait:
...

class Language: LanguageTrait
    value: string = "French"


Acronyms ARE capitalized when combined with other words
```
class MLIROperation:
...
```

Standalone acronyms are lowercase
language level operations can ignore these rules and prefer lower case
```
document = http.get("url")
values = async foobar() with self
```

No word mangaling, spell out the whole word, unless the simple form is DEAD simple (min,max) not frmt.
```
frmt -> format 

```

Id -> ID
Identity stays the same or is lowercased when not a class/trait


### Line width

Column length should be no greater than 80 characters

```
#valid
if condition:
    block...
#Invalid: the linter should wrap the conditions once past the third condition
else condition and condition and condition and condition and condition and condition condition and condition 
->
else condition
    and condition
    and condition
    and condition:

if (condition and other_condition) or (condition and condition and condition and condition and condition and condition condition and condition)
-> 

if ( condition and condition)
    or ( condition 
        and condition...
    ):
    ...
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
# Not this formatting takes place for the user because it's a pain to remember. But easy for the lint
# To map 

foo(a,b,c,d) -> 
foo(
    A=a,
    B=b,
    C=c,
    D=d, #Note, a trailing , is ok, otherwise this flags a two line change in git. 
)

# Before
foo(a, b, c, d)

# After during compilation step linting is applied automatically to the source file
foo(first=a, second=b, third=c, fourth=d)


# If one field has an explicit for 3 variables
foo(a,b,bar=c)

# It's odd we only did the correct mapping for one item and not them all. This leads to a feeling of incompletion. Below is markidly more readable.
foo(
    b1=a,
    b2=b,
    bar=c,
)

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



# These are all valid forms
user1 = User("Alice",36,true)
user2 =User{name:"Alice",age:36,active:true}
other_flag = true
user3 = User.builder().name("Alice").age(36).active(true)
```

# Comments

Comnments are shifted aboe the line
```sev
config.apply(new_config) # networking config
-> 

#networking config
config.apply(new_config)
```

Comments with multiple lines above transform to block comments

```sev
# this is a multi line comment
# this is a multi line comment
->
'''
 this is a multi line comment
 this is a multi line comment
'''
```
# With constraints

with can have a single one line suffix

```sev
call_task = async http.get("url") with local
```

with can also have two clauses


```sev
# we override the local's default async behavior with thread
call_task = async http.get("url") with local, thread 
```


## Tests

Blocks of severian code to test behavior and problems to resolve/reject certain behaviors

### Frontend lint

```text
General policy: lint/sev.grammar
    columns -> 80
    indentation -> 4 spaces
    more than 3 conditions -> parenthesized continuation
    more than 3 nonvariadic arguments -> every argument named
    acronym alone -> lowercase
    acronym in a compound type name -> uppercase acronym
    known abbreviation map -> frmt becomes format

Y/G implementers -> lint_rules()
    source phase -> lexical forms and source fixes
    syntax phase -> declaration/reference spelling
    resolved phase -> checked callable parameter names

Finding with replacement -> apply the fix to its source snapshot
Finding without replacement -> diagnostic; compilation fails
Overlapping/conflicting/cyclic fixes -> diagnostic; compilation fails
```

```sev
# Problem: positional arguments hide the parameter mapping.
def foo(first: int, second: int, third: int, fourth: int) -> int:
    return first + second + third + fourth

def work() -> int:
    return foo(1, 2, 3, 4)

# Expected source after automatic lint:
# return foo(first = 1, second = 2, third = 3, fourth = 4)
# A mixed call gets the same fix; variadic calls retain positional values.
```

```sev
# Problem: a condition has more than three terms.
# Before: if first and second or third and fourth:
# Expected source:
if (
    first
    and second
    or third
    and fourth
):
    work()
```

```text
Required behavior -> adjacent implementation tests:
    MlirBuilder -> MLIRBuilder; HTTP -> http; frmt -> format
    rename declaration -> rename its scoped references, including interpolation
    T in a type annotation -> retain its type contract when a value shadows T
    rename would capture another symbol -> diagnostic, no guessed rename
    long string -> parenthesized adjacent pieces, identical value and bytes
    long default/global/test-label/skip-reason -> retain its own grammar
    inline comment -> preceding comment; adjacent comments -> block comment
    overload -> parameter labels from the callable selected by the checker
    constructor -> labels from constructor parameters or declared fields
    literal spread -> named literal arguments when expansion preserves meaning
    opaque callable/nontrivial spread -> diagnostic when names cannot be fixed
    synthetic interval constructor -> do not insert labels into 0..3
    custom Y/G provider -> inherit general rules and contribute local rules
    lint twice -> identical source on the second pass
    rewritten file -> preserve permissions; refuse stale conflicting content
    source import -> lint the imported file and record it in cache inputs
    imported helper changes -> rebuild dependency and executable
    unchanged canonical source -> reuse the checked executable
```

```bash
python3 -m unittest \
  py_compiler.frontend.lint.rules \
  py_compiler.frontend.lint.layout \
  py_compiler.frontend.lint.names \
  py_compiler.frontend.lint.lint
```

```text
Validated after implementation:
    70 combined lint/frontend/package/executable tests -> pass
    native MLIR build: named fixes preserve evaluation order -> pass
    native MLIR build: wrapped conditions preserve precedence -> pass
    native MLIR build: split strings retain exact bytes -> pass
    dependency cache: helper edit changes exit status 42 -> 43 -> pass
    Python syntax compilation -> pass
```
