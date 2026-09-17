SIP-0000: Interfaces with dispatch methods

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

## Appendix

-- all terms, file formats, variables, classes, functions, traits in a table

## Context

## Problem(s)

Dispatching to methods of implementers requires custom typing ontop of simple primitives in most scenarios leading to match/if statements and constriction on modification/addition of existing interfaces. 

- Overlap of implementers, traditional type dispatch avoids this with rigid non type overlap we do this by erroring on conditions which pass for both dispatchers
- Routing is type top level then value level
- 

## Examples
Examples
Type dispatch

Traditional interface dispatch first narrows candidates by argument type.

trait Add:

    operator +(left: T, right: T) -> T

Implementers may provide different behavior for different types:

class IntAdd: Add:

    operator +(left: int, right: int) -> int:
        return intrinsic.add(left, right)


class StringAdd: Add:

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

with allows implementations sharing the same type signature to partition the value space.

trait Add:

    operator +(left: T, right: T) -> T
class SmallAdd: Add:

    operator +(left: int, right: int) -> int with {
        left <= 10_000,
        right <= 10_000,
    }:
        return intrinsic.add(left, right)


class BigAdd: Add:

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

class PositiveAdd: Add:

    operator +(left: int, right: int) -> int with {
        left > 0,
        right > 0,
    }


class BigAdd: Add:

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

class PositiveAdd: Add:

    operator +(left: int, right: int) -> int with {
        0 < left <= 10_000,
        0 < right <= 10_000,
    }

The dispatch invariant is:

number of matching implementations == 1

Zero matches and multiple matches are both errors.

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

class French: Language:

    def language(text: string) -> string with {
        contains(text, "bonjour")
        or contains(text, "merci")
        or contains(text, "oui")
    }:
        return "french"


class Spanish: Language:

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
class SmallTransfer: Transfer:

    def transfer(
        source: Account,
        destination: Account,
        amount: int,
    ) -> Result with {
        amount > 0,
        amount <= 10_000,
    }:
        ...


class LargeTransfer: Transfer:

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

class BigAdder: Add:

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

## Testing




## Performance


