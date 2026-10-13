SIP-0011: Universal Grammar

Status: Draft
Type: Compiler
Authors: Timothy Player
Created: 2026-09-30
Target: Syntax contracts and frontend lexer/parser
Supersedes: None
Superseded by: None

## Appendix

Proposed forms and interfaces describe the design; they do not claim existing compiler support.

| Term or interface | Kind | Meaning |
|---|---|---|
| `.sev` | File format | Severian source text containing language and grammar declarations. |
| `grammar` | Proposed declaration | Defines how source is recognized and given meaning through the common grammar contract. |
| `G` | Trait / compiler term | Universal syntax-definition contract, from atomic symbols through complex blocks. |
| `Y` | Trait / compiler term | Symbol or atomic recognized term, with source and semantic identity. |
| `B` | Trait / compiler term | Block structure and scope contract. |
| `T` | Compiler term | Type required or produced by a declaration or capture. |
| `V` | Compiler term | Value. |
| `F` | Compiler term | Callable, expression, or operation, retaining its concrete contract. |
| `W` | Compiler term | With-clause operations, including acceptance constraints. |
| `X` | Compiler term | Any compiler term, retaining its specific contract. |
| `N` | Compiler term | Number of elements. |
| `Operator` | Trait | Operation and expression-binding capabilities associated with a grammar. |
| `Lexeme` | Class | Exact matched spelling and its source span. |
| `TokenTerm` | Trait | Classified parser input with a lexeme and span. |
| `Span` | Class | Original source identity and half-open Unicode scalar offsets. |
| `SourceWindow` | Source view | Bounded source text with its original location. |
| Capture | Grammar term | Named source region or compiler term with a concrete required contract. |
| Grammar provider | Interface role | Exposes grammar before an instance of the recognized language object exists. |
| Syntax master list | Registry | Ordered grammar providers used during the stopgap registration phase. |
| `recognition_order()` | Registry function | Intended access point for ordered providers; file/name locators alone do not satisfy the grammar interface. |
| Recognition | Compiler operation | Determines a match, its consumed range, captures, and remaining requirements. |
| Construction | Compiler operation | Produces the selected grammar's compiler representation after applicable requirements are satisfied. |
| CFG | Representation | Control-flow graph, manipulated by the applicable semantic/lowering behavior. |
| `Grammar` | Existing class | Grammar-specific representation of literal fields, captures, and repetitions. |
| `GrammarDeclaration` | Existing class | Stores grammar declarations, bodies, and a program of lowering instructions. |
| `OperatorSyntax` | Existing class | Stores operator spelling and binding metadata alongside additional grammar fields. |
| `grammar()`, `resolve(captures)` | Existing sentence methods | Expose sentence structure and its resolution behavior. |
| `valid`, `semantic` | Existing grammar methods | Express acceptance and semantic behavior through signatures currently shaped around operands. |
| `left`, `right` | Capture names | Ordinary named operands in a grammar declaration. |
| `operation`, `owner` | Capture names | Operation and ownership-context captures in the asynchronous sentence example. |
| `input` | Parameter | Source window supplied to a procedural grammar. |

## Context

Symbols, operators, sentences, and blocks all need to explain how source is recognized and what it means. They currently expose several overlapping formats to the lexer and parser.

The original distinctions describe useful levels of composition:

- `Y` covers simple symbols such as `!` and `|`.
- Operators provide lightweight grammar, including operands conventionally named `left` and `right`.
- Grammars describe arrangements such as `async foobar() with self` within a block having n-ary operators in that composition.
- Grammar `G` can describe complex forms, including blocks and behavior that eventually manipulates the CFG allowing n-ary language decomposition.

These forms should share one syntax-definition interface. Their complexity determines how much grammar they supply. Symbols, sentences, operators, and blocks retain meaningful result and semantic contracts.

The common concept is a grammar that recognizes input and gives it meaning. `G` should express that contract. A grammar declaration can use a bracket arrangement or a procedural implementation; both forms satisfy the same interface.

The bracket form supplies an arrangement from which recognition behavior is derived. The procedural form supplies recognition behavior directly using bounded source views and ordinary operations. Both use the same capture, constraint, diagnostic, and construction machinery.

This draft keeps the distinctions explicit for discussion. Their representation may be collapsed further after the common contract is established.

## Problem(s)

### One required grammar contract

| Capability | Required contract |
|---|---|
| Recognition | Examine a bounded source window and report a match, no match, or diagnostic. |
| Consumption | Identify exactly which original source range matched. |
| Captures | Return named regions or terms with their concrete required contracts. |
| Composition | Request recognition of a capture through another grammar or required result contract. |
| Acceptance | Supply `with` requirements, including dependencies on unresolved captures. |
| Construction | Produce the declared result once the applicable implementation is selected. |
| Expression binding | Supply precedence and associativity when participating in expression composition. |

`left` and `right` are ordinary named captures. Expression binding remains necessary, but grammars that do not use operands or precedence should not be required to supply them.

Implementers must supply their applicable contracts. Empty defaults and duplicated spelling tables cannot substitute for a missing recognition interface.

### Recognition and execution

Trying a grammar candidate must not execute its language behavior. Trying the grammar for `async` cannot launch a task; trying a conditional cannot mutate the CFG.

Recognition establishes source boundaries, captures, and eligibility. Capture resolution establishes facts needed by acceptance constraints. Construction runs after selection and produces a compiler representation. That representation can subsequently supply evaluation or CFG lowering behavior.

A result annotation such as `-> int` can describe the resulting language expression's type. Recognizing `a + b` preserves an expression referring to `a` and `b`; recognition does not require their runtime integer values.

### Procedural grammar and source ownership

A procedural grammar receives a `SourceWindow` containing a view of the text and its original location. It can use string operations to recognize structure, capture nested regions, and delegate their recognition through the common interface.

Every result retains its consumed range and original source identity. An ordinary detached string does not carry enough information to preserve those guarantees by itself.

Declarative and procedural forms obey the same matching contract. A procedural form must preserve ambiguity handling, capture contracts, source bounds, and progress requirements.

### Registration and semantic roles

The master list registers providers implementing `G`. Providers expose grammar before block bodies, conditions, parents, or other matched object fields have been constructed. The concrete mechanism for declaration-level access remains to be specified.

Registry order controls candidate visitation during the stopgap. Applicable contracts determine eligibility; ordering alone does not resolve an ambiguity between admissible implementations.

The roles remain:

- `G`: universal syntax-definition contract.
- `Y`: symbol or atomic recognized term.
- `B`: block structure and scope contract.
- Operator: optional operation and expression-binding capabilities on a grammar.

This avoids expanding `Y` into the interface for every level of syntactic complexity. Grammars and operators can remain useful declaration conveniences while sharing the same underlying grammar machinery.

## Examples

These forms are schematic proposals. Their exact syntax and concrete interface signatures remain to be established.

### Atomic symbol

Note, the grammar implements Y's contract meaning it provides the function needed for Y("!") -> Y of BANG type. 
```sev
grammar bang["!"]() -> Y:Y
    ...
```

The same grammar interface handles a single spelling and larger arrangements. The recognized symbol preserves its lexeme and location.

### Infix operator

grammar default implements the G on the function so G("left+right") would tokenize it it then attempt to match the grammar on the resolved Types. 
```sev
grammar +[left: int, "+", right: int]() -> int:
    ...
```

At times, it's convenient to not expose the full arity functionality so for arity<3 we provide operators

```
grammar operator[left:Numeric,"+",right:Numeric]
```

`left` and `right` are named captures. Their type requirements participate in resolution. This grammar supplies expression-binding properties and the behavior associated with the resulting addition expression.

### Grammar

```sev
grammar launch["async", operation: F, "with", owner: V]() -> F:
    ...
```

For input such as `async foobar() with self`, the grammar captures the operation and ownership context. Recognition preserves the call as a compiler term; it does not execute it.

### Indexed assignment

```sev
class array[T, N]:
    grammar assign[self, "[", index: int, "]", "=", value: T]():
        ...
```

The array supplies the arrangement and meaning of indexed assignment. The frontend applies its grammar when recognizing `a[2] = 1 + 2`, and delegates the value expression to the appropriate grammar contract.

### Procedural block

```sev
grammar block(input: SourceWindow) -> B:
    ...
```

This implementation can inspect indentation, quoted regions, and delimiters; establish header and body captures; and delegate recognition of captured windows. Its recognition behavior uses the same result contract as the bracket forms.

CFG behavior belongs to the selected construct's semantic/lowering contract and occurs at the appropriate compiler stage.

### Place

**The grammar should own the syntax and its meaning. `Place` should be a reusable semantic facility that a grammar can choose to use.**

The SIP already establishes that `left` and `right` are ordinary capture names, and recognition preserves expressions without evaluating them. That gives us the separation we need. [SIP-0011](/home/tplayer/Documents/Severian/docs/sip/0011-Grammar.md)

Example

```sev
class Collection[T]:
    storage: array[T]

    grammar assign[self, "[", index: int, "]", "=", value: T]():
        self.storage[index] = value
```

Here:

```text
Recognition
    self  → receiver expression
    index → expression required to produce int
    value → expression required to produce T

Selected meaning
    self.storage[index] = value

Lowering
    apply storage's grammar
    eventually produce checked storage operations
```

`index: int` constrains the expression’s resulting type. It does not require recognition to compute the index. Likewise, the captures need not become eagerly evaluated function arguments.

Different syntax can express the same operation:

```sev
grammar put["put", value: T, "into", self, "at", index: int]():
    self.storage[index] = value
```

Or introduce different semantics:

```sev
grammar wrapped[self, "[", index: int, "]", "<-", value: T]():
    self.storage[index % len(self.storage)] = value
```

Both can eventually use the same checked element-storage implementation. The compiler does not need a branch recognizing `put` or `<-`.

**Evaluation order belongs to the selected semantics, rather than the textual order of captures.**

For the first example, the body delegates to the backing array’s assignment semantics. If those specify RHS-first evaluation, the conceptual lowering is:

```python
# Compiler pseudocode; captures remain expressions until lowered.
value = lower_value(captures.value)
receiver = lower_value(captures.self)
index = lower_value(captures.index)

destination = element_place(receiver.storage, index)
assign(destination, value)
```

A custom control-flow grammar may evaluate a capture conditionally or repeatedly. Therefore the common grammar interface must preserve captures as compiler terms until the selected implementation determines their evaluation.

The shared facilities would be:

```text
Grammar
    recognition, captures, constraints, construction

Selected semantic implementation
    evaluation order, repetition, control flow, effects

Reusable operations
    resolve symbol
    project field
    checked indexing
    read / borrow / move / initialize / replace / destroy

MIR
    typed operations and explicit control flow
    ownership and lifetime verification

LIR / MLIR
    target lowering
```

Ownership remains enforced across custom grammars:

```text
custom syntax replaces storage with a live pointer
    → reject

custom syntax reads through a view
    → allowed

custom syntax duplicates a unique owner without copying
    → reject
```

For your nested example, each indexing expression is recognized through its applicable grammar; the outer collection’s assignment grammar supplies the final meaning:

```sev
a[b[c[0]]] = c[b[a[0]]]
```

A useful design acceptance test is:

```text
Register a new collection grammar
    → nested expressions compose
    → its semantics determine evaluation
    → ownership checks still apply
    → no changes to the assignment dispatcher
```

That preserves the SIP’s flexibility while giving grammar authors common, safe operations to build upon.

## Testing

Establish the common contract using one symbol, one infix operator, one sentence, and one procedural block before expanding the lexer implementation.

| Scenario | Required result |
|---|---|
| Atomic symbol | The supplied grammar recognizes the spelling and preserves its source span. |
| Infix expression | Named operands are captured, and declared precedence/associativity establish expression structure. |
| Unknown runtime operands | Parsing `a + b` preserves references without evaluating their runtime values. |
| Async sentence | The operation and owner have distinct captures; recognition launches no task. |
| Indexed assignment | The array grammar supplies the arrangement without an array-specific frontend branch. |
| Procedural block | Header and body windows retain their boundaries, nesting, and original source locations. |
| Declarative/procedural equivalence | Equivalent implementations satisfy the same recognition and capture contracts. |
| Grammar composition | Captures delegate through concrete required contracts and retain nested results. |
| Incomplete provider | Missing required grammar capabilities are diagnosed instead of silently registering an empty rule. |
| Provider discovery | Grammar is available without constructing a matched block or other language object. |
| Ambiguous implementations | Multiple admissible implementations produce ambiguity independently of registry order. |
| Deferred constraints | Requirements remain unresolved until their capture dependencies are available. |
| Selected implementation failure | Construction failure is reported without retrying another candidate. |
| Malformed source | Failure identifies the original source span and compiler step. |
| Progress and bounds | Recursive recognition cannot repeatedly consume the same unchanged region or escape its source window. |
| Syntax extension | Adding a provider requires no new language-specific lexer/parser branch or duplicate spelling table. |

Place implementation unit tests after the code they exercise. Use package integration tests for interactions across Source, syntax providers, and the frontend.
