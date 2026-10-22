# SIP-0007: Interfaces with dispatch methods

Status: Draft | Accepted | Implementing | Implemented | Rejected | Superseded
Type: Language | Compiler | Runtime | Tooling | Package | Interop | Process
Authors:
Created: YYYY-MM-DD
Target:
Supersedes:
Superseded by:


## Summary

Interfaces allow the swapping of implementation without breaking user contracts. If a.add(a,b) swaps MLIR/Assembly lowering
The callers need not change their code in order to hadnle changes. In traditional programming languages, the type is what
is used to route to implementers. For example, a:int b:int a+b routes to a function through the addition operator that takes
int and int operator +(int,int) -> int if it was a string the compiler would route to a string + string F etc. This is 
handy when there is a delta in the type of operators. Otherwise you'd need a chain of ifs/match statements to route within values for example: def language("bonjour") inside of the language function, there essentially needs to be a nexus of 
string comparison, if else, and maybe even ML to group language probabilities. You could construct sub types for each language and do building that way with language adapters but then the function needs type semantics around a simple problem:

What language does this string or can this language contain? That's where `with` comes in

By establishing `with` on the interface we can dispatch to methods with a more complex contract
In the following order:
1. Type dispatch: Find implementations compatible with the argument types.
2. Value dispatch: Evaluate each implementation's with predicates.
3. Selection: If multiple implementations match, apply the interface's selection policy using implementation metadata.

## Appendix


- Prelude traits: 
-- Generics: (T,C,N,G, Y, K, V, N)
-- Primitives: Equatable, Comparable, Formattable, Convert[T], Default, Arithmetic, BitArithmetic
-- Util traits: Hash, Sortable, Ownership, Printable, Numeric, Formattable, Serde, Share
-- Callable
prelude/
├── generic/
│   ├── T, C, N, G, Y, K, V, W
│   └── constraints
│
├── comparison/
│   ├── Equal
│   ├── Compare
│   └── Hash
│
├── numeric/
│   ├── Numeric
│   ├── Integer
│   ├── Floating
│   ├── Signed
│   └── BitArithmitic
│
├── collection/
│   ├── Iterable
│   ├── Sized
│   ├── Indexable
│   ├── Contains
│   ├── Sortable
│   └── MutableCollection
│
├── conversion/
│   ├── Convertible
│   ├── Printable
│   ├── Formattable
│   └── Parseable
│
├── ownership/
│   ├── Ownership (Copy,Move,Borrow,Drop, abbreviation)
│   ├── Copy
│   ├── View
│   ├── Borrow
│   ├── Mirror
│   ├── Move
│   ├── Drop
│   └── Share
│
├── callable/
│   ├── Call
│   ├── Await
│   └── Generator
│
├── Error/
│   ├── Diagnostic
│   ├── Warning
├── io/
│   ├── Read
│   ├── Write
│   ├── Seek
│   └── Close
│
└── introspection/
    ├── Reflectable
    └── Error

## Examples
Examples
Type dispatch

Traditional interface dispatch first narrows candidates by argument type.

trait Add:

    operator +(left: T, right: T) -> T

Implementers may provide different behavior for different types:

class IntAdd: Add

    operator +(left: int, right: int) -> int:
        return intrinsic.add(left, right)


class StringAdd: Add

    operator +(left: string, right: string) -> string:
        return string.concat(left, right)

The compiler resolves:

1 + 2

as:

Add
-> +(int, int)
-> IntAdd.+

while:

"hello" + " world"

resolves as:

Add
-> +(string, string)
-> StringAdd.+

No value predicates are evaluated until type dispatch has reduced the candidate set.

Value dispatch

`with` allows implementations sharing the same type signature to partition the value space.

trait Add:

    operator +(left: T, right: T) -> T

class SmallAdd: Add

    operator +(left: int, right: int) -> int with {
        left <= 10_000,
        right <= 10_000,
    }:
        return intrinsic.add(left, right)


class BigAdd: Add

    operator +(left: int, right: int) -> int with {
        left > 10_000,
        right > 10_000,
    }:
        return big_integer.add(left, right)

For:

a = 20
b = 30
c = a + b

resolution is:

resolve Add
-> resolve types (int, int)
-> [SmallAdd.+, BigAdd.+]
-> evaluate value predicates
-> SmallAdd.+ = true
-> BigAdd.+   = false
-> select SmallAdd.+

For:

a = 2_000_000
b = 3_000_000
c = a + b

resolution selects BigAdd.+.

Ambiguous dispatch

Value domains may not overlap.

class PositiveAdd: Add

    operator +(left: int, right: int) -> int with {
        left > 0,
        right > 0,
    }


class BigAdd: Add

    operator +(left: int, right: int) -> int with {
        left > 10_000,
        right > 10_000,
    }

The expression:

20_000 + 30_000

satisfies both contracts.

The compiler must not select one based on declaration order, number of predicates, package order, or perceived specificity.

It produces:

error: ambiguous dispatch for +(int, int)

values satisfy multiple implementations:

    PositiveAdd.+
        left > 0
        right > 0

    BigAdd.+
        left > 10000
        right > 10000

The programmer must refine the domains:

class PositiveAdd: Add

    operator +(left: int, right: int) -> int with {
        0 < left <= 10_000,
        0 < right <= 10_000,
    }

The dispatch invariant is:

number of matching implementations == 1

Zero matches is an error unless the trait provides a default body. Multiple
matches remain an error; a default body never resolves ambiguity.

File dispatch with a reusable default

```sev
# Proposed trait-default semantics. Sections below belong to separate modules.
# Reading a file does not require an extension. Extensions are one possible
# condition for choosing a parser, not a requirement of the underlying reader.

# In system.file: text.read owns the lower-level text read through IO streams.
trait File:
    @file
    def read[T = string](path: string) -> T:
        # This body supplies the string fallback, not an arbitrary T conversion.
        # T comes from the selected implementation; only the fallback defaults
        # to string. This body is NOT a candidate in with-clause dispatch.
        return text.read(path)

# In json: importing this module makes its File implementation available.
def read(path: string) -> Data with { path.suffix(".json") }: File
    # Call the trait's original body directly, without conditional redispatch.
    # Read the SAME .json path as text; do not rename it or pretend it ends .txt.
    source = file.read.default(path)
    return json.parse(source)

# In the application:
import file
import json

document = file.read("settings.json")  # JSON implementation -> Data
notes = file.read("notes.txt")        # No matching implementation -> string
license = file.read("LICENSE")        # No extension -> default reader -> string
custom = parse_custom(file.read("settings.custom"))
# An unknown extension also falls back to text; apply application parsing above.

# The application can instead register its own conditional implementation.
# parse_config is the application's parser returning its Config type.
def read(path: string) -> Config with { path.suffix(".cfg") }: File
    return parse_config(file.read.default(path))

config = file.read("settings.cfg")    # Local implementation -> Config
raw = file.read.default("settings.json")  # Explicit default -> string

# Selection: one matching imported/local implementation -> call it;
# zero matches -> call the default; unresolved multiple matches -> error.
# A selected implementation's read/parse failure propagates; it does not trigger
# fallback. The default is reusable behavior, not error recovery.
# Known selection -> its concrete result type. A runtime-only selection retains
# the union of possible result types, including string when fallback is possible.
```

Language detection

Value dispatch is useful when the type contains insufficient information.

Without constrained dispatch:

def language(text: string) -> string:

    if contains_french_words(text):
        return french(text)

    if contains_spanish_words(text):
        return spanish(text)

    if contains_german_words(text):
        return german(text)

    ...

The routing logic becomes a nexus that owns knowledge about every implementation.

With interface dispatch:

trait Language:

    def language(text: string) -> string with {
        (text) -> bool,
    }

Implementers define their own acceptance contract:

class French: Language

    def language(text: string) -> string with {
        contains(text, "bonjour")
        or contains(text, "merci")
        or contains(text, "oui")
    }:
        return "french"


class Spanish: Language

    def language(text: string) -> string with {
        contains(text, "hola")
        or contains(text, "gracias")
        or contains(text, "sí")
    }:
        return "spanish"

Calling:

language("bonjour tout le monde")

becomes:

resolve Language.language
-> resolve type string
-> evaluate French condition
-> evaluate Spanish condition
-> exactly one match
-> French.language

Adding another language does not require modifying the central language function.

Multiple arguments

Predicates may depend on any dispatched argument.

trait Transfer:

    def transfer(source: Account, destination: Account, amount: int) -> Result
class SmallTransfer: Transfer

    def transfer(
        source: Account,
        destination: Account,
        amount: int,
    ) -> Result with {
        amount > 0,
        amount <= 10_000,
    }:
        ...


class LargeTransfer: Transfer

    def transfer(
        source: Account,
        destination: Account,
        amount: int,
    ) -> Result with {
        amount > 10_000,
    }:
        ...

The type tuple narrows the candidate bucket first:

(Account, Account, int)

The value predicates are evaluated only within that bucket.

Variadic conditions

with may contain multiple conditions.

class BigAdder: Add

    operator +(left: int, right: int) -> int with {
        left > 10_000,
        right > 10_000,
        left < int.max,
        right < int.max,
    }:
        return intrinsic.add(left, right)

All conditions must evaluate to true for the implementation to match.

Equivalent resolution logic is:

matches = true

for condition in implementation.conditions:
    if not condition(arguments):
        matches = false
        break
## Problem(s)

Dispatching to methods of implementers requires custom typing ontop of simple primitives in most scenarios leading to match/if statements and constriction on modification/addition of existing interfaces. 

- Overlap of implementers, traditional type dispatch avoids this with rigid non type overlap, we do this by erroring on conditions which pass for both dispatchers
- Routing is type top level then value level 
```
trait Foo:
    def bar(x: Primitive) -> int

def bar(x:int) -> int: Foo
    return 1

def bar(x:string) -> int: Foo
    return 
```

- It progresses by complexity of checks using the function's declared complexity (see the Function sip)

- It would be handy to sort the implementers if there is overlap, this allows a couple things
When matches are ambiguous but meta params determine which one is optimal dynamic routing would be very nice

```
class NaiveMatMul: MatMul

    def multiply(a: tensor, b: tensor) -> tensor with {
        multiply.complexity.time = complexity.Cubic
        multiply.complexity.space = complexity.Quadratic
        a.ndim == 2,
        b.ndim == 2,
        a.shape[1] == b.shape[0]
    }: MatMul
        ...


class BlockedMatMul: MatMul
    def multiply(a: tensor, b: tensor) -> tensor with {
        multiply.complexity.time = complexity.Cubic
        multiply.complexity.space = complexity.Linear
        a.ndim == 2,
        b.ndim == 2,
        a.shape[1] == b.shape[0]
    }: MatMul
        ...

```

### Dispatch Diagram
```
MatMul.multiply
    |
    +-- Type matching
    |      NaiveMatMul     ✓
    |      BlockedMatMul   ✓
    |
    +-- Value predicates
    |      NaiveMatMul     ✓
    |      BlockedMatMul   ✓
    |
    +-- Interface selection
           min(F.complexity.space)
                 |
                 +-- NaiveMatMul    O(n²)
                 +-- BlockedMatMul  O(n)
                         |
                         v
                    BlockedMatMul
```
### static and dynamic costs

```
trait Foo:
    def bar() -> int with {
        min(F.complexity.space)
    }
```

works when complexity metadata is comparable.

However:
Implementation A: O(n)
Implementation B: O(n log n)

requires a shared definition of n and valid bounds on the relevant input domain. For different variables or multivariate bounds, a total ordering may not exist.
For instance:
A: O(n²)
B: O(n*m)

Neither universally dominates without knowing the relationship between n and m.
Severian could handle this by evaluating symbolic cost expressions against the argument shapes.
trait Foo:
    def bar(a: array[array[int]]) -> int with {
        min(F.complexity.space(a.shape))
    }

Now the compiler can specialize the selection based on known shapes.
If shapes are dynamic, the same selector can be evaluated at runtime, provided it is a valid runtime expression.
```
```
## Testing




## Performance

## Structured annotation phases: example, problem, test

Function annotations can supply ordered callback phases without declaring a special `HookContext` class or `with context` / `without context` bodies.

```sev
@monitor_time with {
    prefix timer_started(F.name),
    suffix report_time(F.name, context.duration),
}
@monitor_error with {
    prefix started(F.name),
    prefix search_arguments(F.args...),
    suffix report_error(F.name, context.error),
    suffix report_result(R),
}
def search(items: view list[int], item: int) -> int | Error:
    for value in items:
        if value == item:
            return item
    throw Error("Failed to find item in items")
```

- `F.name`: the decorated function's name.
- `F.args...`: argument values in parameter order, checked using ordinary ownership rules.
- `F.params[index]`: compile-time parameter metadata with `.name`, `.type`, `.ownership`, and `.default` string fields. The index must be constant; empty default source means no default. Select a field rather than materializing the metadata as runtime storage.
- `R` / `context.result`: optional typed result, available only in suffix callbacks.
- `context.error`: optional typed error, preserving its fields; success supplies `None`.
- `context.duration`: elapsed seconds, finalized before the first suffix callback. It includes prefixes and the function body, excludes suffixes.
- Callback arguments are read-only. Callbacks must not propagate errors; panic remains a terminal failure without an unwinding guarantee.
- Nested calls in callback arguments are rejected; put that work inside a nonthrowing callback so argument evaluation cannot bypass the phase checks.
- Annotation names label explicit phase groups. Multiple annotations enter top-to-bottom and exit bottom-to-top. Callbacks within a group keep their written order.
- Error propagation uses the existing declared error-result contract (`int | Error` above). A locally caught error does not exit the decorated function.

```text
success -> time.prefix -> error.prefix -> search
        -> error.suffix(result=10, error=None)
        -> time.suffix -> return 10

throw -> time.prefix -> error.prefix -> search
      -> error.suffix(result=None, error=original Error)
      -> time.suffix -> propagate original Error

prefix reads R -> compile error
suffix consumes or mutates a borrowed argument -> compile error
suffix reads an argument already moved by the function -> compile error
callback propagates Error -> compile error
function hook reads T.fields -> compile error: wrong target
phase annotation on a type -> compile error: execution phase not defined
panic -> suffix execution is not guaranteed
```

Problem: a raw decorator wrapper can erase argument ownership, change an error into a string, or skip an outer hook when an inner callback throws. Here phase calls are checked and lowered inside the decorated function's control flow. Recoverable error exits execute suffix callbacks before returning the error. Observer callbacks receive typed optional values and cannot replace the pending outcome.

```sev
test with integ "suffixes run before the caller catches the error":
    try:
        result = search([10, 20, 30], 99)
    catch error: Error:
        assert(error.message == "Failed to find item in items")
        print("caught")
    assert("error: search Failed to find item in items\ntimer duration: search\ncaught" in stdout)
    assert("result:" not in stdout)
```

The executable callbacks and tests are in `docs/examples/01-types/07-traits/05-trait-hook.sev`. Parser and binding rejection tests follow the implementation in `py_compiler/syntax/block/hooks.py`. Type-target lifecycle phases and trait-level reusable phase-group declarations need their own execution semantics; this implementation handles explicit function annotations.
