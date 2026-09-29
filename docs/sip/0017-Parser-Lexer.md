# SIP-0017: Parser/Lexer Span Dispatch

Status: Draft

## Contract

Parsing dispatches on the value of a source span. Language declarations implement
the existing generic and trait contracts and declare acceptance conditions through
`with`. The lexer/parser applies those contracts instead of acquiring a special
case whenever the language gains syntax.

The intended interface is schematically:

```sev
trait Lexeme:
    def parse(span: view string) -> T with {
        ...
    }
```

Implementers supply concrete result contracts and acceptance predicates. `Y`
connects lexemes and tokens to symbols; `S` describes sentences; `B` describes
blocks; `G` describes their grammar; `W` supplies constraints. These retain their
existing meanings and the traits required of their implementers.

## Source windows

The frontend starts with the source window, establishes its blocks, and recursively
establishes windows for their nested blocks and sentence regions. Enclosing
structure is recognized before its captured regions are resolved. Delimiters,
indentation, and the selected grammar establish the boundaries; quoted contents
and nested regions retain their own boundaries. Every window retains its original
source location for diagnostics.

Within a window, candidates come from implementers of the required contract.
Their `with` conditions progressively exclude incompatible implementations.
Resolving a captured window can establish facts required by an enclosing candidate.
A class declaration therefore fails the assignment contract through its declared
conditions, rather than through a parser exception for `class`.

For dictionary construction, the class declares a sentence recognizing braces
around repeated key/value captures separated by commas. That sentence invokes
the dictionary's construction behavior. Braces alone do not establish a block.

## Resolution

Resolution follows the existing interface dispatch scheme:

1. Restrict candidates by the required trait and concrete type contracts.
2. Evaluate ready `with` constraints and exclude refuted candidates.
3. Resolve captured regions needed by remaining constraints; unknown candidates
   remain eligible.
4. Invoke the parse body only when exactly one implementation is admissible.

Zero matches is a syntax failure for the required contract. Multiple surviving
matches is ambiguity. Declaration order, priority, or cost does not choose a
winner. Failure in the selected body is a diagnostic, not a dispatch retry.
Constraint evaluation follows SIP-0007's existing safety and dependency rules.

## Example

```sev
if 1 == 0.0:
    return false
```

`if` is a `B` and controls CFG manipulation. Its declaration captures a condition
window and a body window. The condition resolves its operands through their literal
contracts and `==` through the applicable operation contract. The body resolves
`return` and the boolean literal `false` through their declarations.

The integer spelling contract accepts `0`; floating spelling accepts `0.0` and
`.0`. Literal recognition and construction belong to the implementing types.
The frontend applies their declared conditions rather than hiding those decisions
inside parser branches.

Adding or refining syntax therefore changes the implementing declarations and
their `with` conditions, using the same frontend dispatch machinery.
