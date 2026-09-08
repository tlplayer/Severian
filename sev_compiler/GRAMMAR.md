# Symbols, operators, grammar, and CFG ownership

This is the target architecture for the source compiler. Source trait/operator
requirements, concrete specialization and an extensible source lexical registry
are executable today. Compiler-semantic CFG execution and grammar-only CFG
capabilities are not implemented yet.

The combined traits/types and grammar migration is one task. Its compiler
vocabulary is contextual: metadata declarations (`effects`, `evaluation`,
`conversions`, `overflow`, and related terms), operand protocols (`Place`,
`Value`, `Lazy`), and compiler directives (`read`, `write`, `mutate`, `borrow`,
`consume`, `evaluate_once`, and CFG construction) are available in the compiler
`G` trait and declarations implementing that resolved trait. Their spelling does
not reserve ordinary variable, field, or function names globally.

An operator binding such as `operator +=[G:AddAssign]` refers to that contract;
it does not make the concrete type or the operator body an implementation of
the compiler `G` trait. In particular, it does not grant direct CFG access.
Indirect trait inheritance must retain the compiler context, and an unrelated
user trait merely named `G` must not grant it.

Use `absent` for contract fields that do not apply or need no specification.
Optional fields have the type `FieldType | absent` and default to `absent`.
For example, unary operations can have `right = absent`, logical operations
can have `overflow = absent`, and operations without a parsing role can have
`precedence = absent`. Preserve absence through AST, registration, and typed
definitions. Do not replace it with zero, an empty string, `false`, or an
invented enum member. An explicit `short_circuit = false`, `effects = Pure`,
or forbidden conversion remains distinct from an unspecified field.

Registration validates applicability: `absent` is not permission to omit
information needed to parse or execute the chosen contract. Required operand,
parsing, and semantic information must resolve before instantiation. There is
no implicit arithmetic policy, conversion permission, or compiler capability
when its declaration is absent.

`trait AddAssign: G` defines its place/value operands, evaluation order, effects,
validation, and semantic expansion. Assignment behavior comes from that
contract. Do not introduce a handwritten mapping from `"+="` to an `"Assign"`
behavior, derive an operator by appending `"="`, or infer mutation from a
compiler naming convention. A binding such as
`operator %=[G:RemainderAssign]` must resolve an actual source contract.

## Definition hierarchy

`Y` identifies an interned source symbol. `O` identifies a value operation and
`G` identifies a grammar operation. Both refer to symbols; neither owns a second
spelling table. A spelling can participate in multiple forms: prefix `+` and
infix `+` have different operations and share the same symbol.

```text
Y / source symbols
├── O / value operations and their type-specific implementations
└── G / grammar operations and their lowering definitions
       └── exclusive access to CFG construction
```

The existing `universal/symbol/symbol.sev` models linker symbols. Keep linker
identities distinct from lexical `Y` identities.

Source operator descriptors own spelling, syntactic form, precedence, and
associativity. Trait requirements and implementations own signatures and
behavior. A method named `add` does not declare `+`; an explicit operator
requirement does. Do not infer an implementation from a descriptor's name.

`TokenKind: string` can be a declaration-facing name, but it must resolve to
validated compiler metadata during registration. Adding an operator must not
require adding an enum variant or a lexer/parser switch. Tokens carry a symbol
identity and source span; the parser consults the registered operation/form.

Grammar descriptors additionally declare their operand forms and lowering
definition. The grammar's syntactic arity and an operand protocol's runtime
arity are distinct: an `if` grammar accepts a condition and blocks, whereas its
truth operator accepts one value.

## Grammar operand protocols

The following syntax describes the target source contract:

```sev
class Counter:
    value: int

    operator if[G:If](self) -> bool:
        return value != 0
```

`[G:If]` follows the same generic-parameter/constraint syntax as `[T:Numeric]`:
`G` is the parameter, and `If` names its constraint. The parameter can be
renamed. Registration resolves the constraint and its inherited contracts to
determine that it belongs to the compiler grammar domain; the parser does not
infer that domain from a parameter's spelling. This constraint does not grant
the operator body CFG capability. Overload resolution includes the resolved
domain and contract identity in its key.

For `if counter:`, semantic analysis resolves `operator if[G:If](Counter)` and
checks its result against the grammar's required `bool` result. The typed call
retains the implementation `DefId`; `G.If` uses that value to build the branch.
Types supply their own truth bodies. There is no compiler truth table for
integers, strings, or collections. Boolean identity behavior should also have
an explicit source contract.

Generic truth requirements follow the existing explicit-operator pipeline:
requirement definition, symbolic generic call, concrete substitution,
implementation call. An ordinary method or a value-domain operator with the
same spelling cannot satisfy a grammar-domain requirement.

Other grammar operations can request specific protocols, such as an iterator
from `operator for[G:For]`. `else`, `return`, `break`, and `continue` need no
type-specific operator. `throw` remains closed to the language's `Error`
contract; arbitrary conversion hooks must not make unrelated values throwable.

## Capability boundary

Only a grammar lowering definition may create blocks or issue CFG primitives:
`branch`, `goto`, `switch`, `return`, `throw`, and `unreachable`. The CFG builder
is compiler context, not an importable runtime type or a value that can escape
into a field, function argument, closure, or operator body.

Ordinary functions and operator bodies may contain normal `if`, loops, calls,
and returns. Those constructs lower through their registered grammars. They
cannot directly call CFG primitives. A grammar-constrained operator computes only the value
or protocol requested by grammar.

Enforce this on resolved compiler primitive identities and their effects, not
on a variable named `cfg`. Renaming or aliasing a primitive must not bypass the
check. Compile-time helpers that mutate CFG must carry the same restricted
effect; a call from a grammar does not grant an ordinary helper that effect.

Grammar lowering must respect an already terminated block. An early return in
an `if` arm cannot acquire a second `goto merge`. A loop maintains scoped
header/exit targets so nested `break` and `continue` choose the correct loop.

## One executable control-flow representation

The current executable path is:

```text
frontend/semantic/src/callable.sev
  → transforms/mir/src/callable.sev
  → structured Operation.If / Loop / Choose
  → transforms/mlir/src/emit/callable.sev
```

`universal/cfg/cfg.sev` already defines `CfgBody`, `BasicBlock`, and
`Terminator`. The separate `cfg_from_hir` in `transforms/mir/src/build/mod.sev`
does not implement complete expression/control-flow lowering and is not the
executable path. Merely adding CFG metadata alongside structured operations
would leave two competing representations.

Migrate the executable lowering and its consumers to the shared CFG. Retain
structured syntax in AST/HIR; CFG becomes the authority for executable edges.
Ownership, reachability, return-path checks, emission, and Agent IR must consume
that same body. Backend structuring may derive structured regions from it when
needed by the MLIR ownership pipeline.

Record grammar definition identity and source origin on generated terminators.
Implicit returns, short-circuit Boolean evaluation, conditional expressions,
call continuation/unwind paths, and cleanup must also have grammar origins.
Short-circuit `and`/`or` cannot become eager value operations merely because
their spelling appears in the operator catalog.

The checkable invariant is: every source-language CFG edge has a grammar
origin. Backend passes can split or synthesize edges, but must retain derived
provenance rather than pretending those edges appeared in source syntax.

## Implementation sequence and acceptance

1. **Register source symbols and operation descriptors.** Replace
   `frontend/lexer/src/scanner/mod.sev:punctuation` and the parser's operator
   switch with shared registered metadata from `universal/operator/kind.sev`
   and grammar declarations. Keep a small bootstrap lexer for declaration
   headers, identifiers, literals, comments, indentation, and delimiters.
   Discover imported descriptors before lexing affected bodies; registration
   must not depend on declaration order or execute arbitrary source bodies.
   Match the longest registered symbol and preserve identifier boundaries for
   word symbols. Diagnose conflicting syntax definitions and unknown symbols.
   Test a new multi-character operator without editing either switch, imported
   descriptors, prefix overlap, word boundaries, and conflicting declarations.

2. **Resolve grammar definitions and typed operand protocols.** Add grammar
   declarations to the universal AST and shared semantic registry. Represent
   `[G:Contract]` explicitly in operator signatures. Register grammar operand/result
   contracts and check restricted compiler effects before lowering. Exercise
   `if` on a user-defined type and through an explicit generic requirement;
   reject a non-`bool` result, a missing protocol, a same-spelled ordinary
   operator, and CFG access from functions/operators, including indirect access.
   Existing arithmetic generic tests must keep resolving source definitions.

3. **Execute grammar lowering into the shared CFG.** Move block and terminator
   construction behind the grammar capability, connect the executable driver
   and MLIR emission to that CFG, then retire the competing MIR topology.
   Start with branches, loops, and returns and migrate every supported construct
   before claiming the invariant. Verify nested branches/loops, early returns,
   break/continue targets, short-circuit side effects, and predecessor/argument
   consistency. Extend to iteration/error/cleanup protocols as their value
   representations become executable. Test CFG provenance and inspect
   `sev build --emit agent-ir` alongside native behavior.

Moving a fixed punctuation table to another file is not completion of step 1.
Parsing a grammar without executing its definition is not completion of step 2
or 3. The acceptance tests must demonstrate that editing source declarations
changes the compiler's behavior through registration and resolved identities.
