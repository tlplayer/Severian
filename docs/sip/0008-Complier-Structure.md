# SIP-0008: Semantic Compilation Hierarchy and Constraint Graph

Status: Draft

Repository review (2026-09-22): see [V2 implementation and retirement plan](0008-v2-implementation-review.md)
for proposed semantic corrections, concrete files to edit/add, ordered migration
gates, and the conditional purge list. The draft below remains the proposal;
the review identifies claims that need revision before implementation.

Type: Language | Compiler | Package

Authors:

Created: 2026-09-22

Target:

Supersedes:

Superseded by:

## Normative migration contract (2026-09-22)

The following decisions supersede conflicting illustrative examples below.
Implementation status and retirement gates are tracked in
[the implementation review](0008-v2-implementation-review.md); the proposal is
not a statement that those gates have passed.

- Dependency graphs and CFGs may be cyclic. Callable recursion is legal; schedule
  strongly connected components through their acyclic condensation graph.
  Containment, inheritance, initialization and predicate prerequisites reject
  cycles where their respective contracts forbid them.
- A declaration has one semantic owner. Uses and imported aliases reference it;
  they do not duplicate its identity. A semantic owner key is independent of
  a source path. Default file membership remains a compatibility adapter until
  explicit package/module/submodule membership is available. Moving files only
  preserves identity when the declared semantic owner and declaration key stay
  the same. Anonymous definitions have revision-local identity until an explicit
  matching policy is implemented. Initializers retain declared source order.
- Package resolution and dependencies at the module/library level belong to
  `library/package`. Compiler semantic dependency graphs are scoped within
  submodules. Cross-submodule references are projected through package-owned
  membership and planning APIs, rather than a compiler-owned package resolver.
- Lexical regions, source sentences, CFG blocks and tasks have separate typed
  identities. A sentence can own a nested region whose children are sentences.
  The active `Module.cfg_bodies` tables remain the sole executable topology.
- Internal lexer/parser spans use half-open Unicode scalar indices. Tooling
  interfaces serialize half-open UTF-8 byte offsets through an explicit source
  conversion. Synthetic declarations have absent provenance plus a reason;
  they must not claim to originate from source offset zero.
- Dispatch selects exactly one implementation from a closed candidate set.
  Zero matches is a no-match diagnostic/failure; multiple matches is an ambiguity
  diagnostic/failure. Costs do not break ties. Static proofs may eliminate work;
  dynamic dispatch must establish uniqueness before executing an implementation.
- Reordered/shared predicates must be pure, non-throwing, terminating and have
  stable reads. Unknown effect/termination information does not grant these
  properties. Prerequisites (including shape bounds before indexing) dominate
  their checks. Other attached operations retain control and effect ordering.
- Proof outcomes are Proven, Refuted and Unknown. Exhausting a budget yields
  Unknown or a diagnostic, never Proven. Static-only Unknown diagnoses; eligible
  dynamic obligations may generate checks. Value refinements attach to SSA or
  memory versions and are invalidated by affected writes and calls. Joins must
  be sound; loop widening is used only where convergence requires it.
- Evaluation stage (static, specialization, runtime) is independent of check
  site (prefix, invariant, defer/suffix). `fix` is established on prefix and checked after
  relevant writes/calls, on loop edges, and on region defer/suffix. Suspension/re-prefix
  needs its own policy before support. It does not continuously poll memory.
  The successful integer increment example requires `0 < x and x < 90` before
  `x += 10` under `fix x < 100`; `x = 95` is a negative test.
- Ordinary `defer` retains delayed cleanup on scope defer/suffix. It is not an alias for
  `fix`. Cleanup and defer/suffix checks execute exactly once on each supported normal,
  early-return, error and loop-defer/suffix path crossing their region. Abort does not
  promise cleanup. A failed invariant never retries another implementation.
- Constraint payloads distinguish setup declarations, guards, contracts,
  ownership and execution context. Resolved interfaces do not grant CFG
  capabilities or arbitrary effects. `sorted` constructs a sequence, not a
  Boolean predicate; floating comparisons must account for NaN.
- Interfaces include typed generic/grammar/predicate bodies where clients need
  them, or explicitly reject unsupported source-free use. Query dependencies
  distinguish signatures, bodies, candidate sets and realization inputs.
  Semantic identity is not a content fingerprint. ObjectUnit fingerprints include
  compiler/schema, target/ABI, options and specialization inputs.
- An ObjectUnit is a tagged realization boundary, with many-to-many submodule
  membership. Initialization order, symbol ownership and incompatible device
  targets are checked before emission. Existing provider APIs stay available
  until their replacements pass their gates.

## Summary

Severian should organize source semantics using:

```text
Library
  └─ Module
      └─ Submodule
          └─ Block
              └─ Sentence
                  └─ Symbol / Operator / Operation
              └─ Block
                  └─ ...
```

This hierarchy describes containment, not the complete compiler representation.

The complete representation is a typed Universal graph containing these entities and relationships between them.

```text
                       Universal Graph

   containment       dependencies       constraints
       │                  │                  │
       ▼                  ▼                  ▼
Library→Module       Operation DAG        with DAG
      →Submodule          │                  │
      →Block              │                  │
      →Sentence           ▼                  ▼
                     scheduling          refinement
                          │                  │
                          └──────┬───────────┘
                                 ▼
                              lowering
                                 │
                                MIR
                                 │
                               MLIR
```

The hierarchy answers:

```text
where does this belong?
```

Graph edges answer:

```text
what does this depend on?
what does it require?
what does it know?
what does it own?
what does it produce?
what implementation applies?
```

The levels have distinct responsibilities:

```text
Library
    Package/library boundary.

Module
    Namespace and major dependency boundary.

Submodule
    Semantic compilation unit and default object-generation boundary.

Block
    Scope, control-flow, ownership, and lifetime region.

Sentence
    Complete grammar-resolved unit.

Symbol / Operator / Operation
    Semantic components of a sentence.
```

`Sentence` and `Block` remain separate.

A sentence may contain complex syntax and operations without introducing a nested scope.

```sev
task = async foobar() with self
```

is one sentence.

A block is introduced when execution creates a nested region:

```sev
if ready:
    task = async foobar() with self
```

which becomes:

```text
Sentence If
├─ ready
└─ Block
   └─ Sentence Assignment
```

The distinction is:

```text
Sentence
    grammar and evaluation boundary

Block
    scope, lifetime, CFG, and ownership boundary
```

`with` is a first-class semantic interface.

It associates constraints, conditions, declarations, invariants, ownership requirements, and dispatch requirements with any compatible graph node.

For example:

```sev
def increment(x, view y) with {
    prefix x > 0
    fix x < 100
    suffix x >= 10
    defer unchanged(y)
}:
```

where:

```text
prefix
    checked when entering the region

fix
    invariant maintained across the region

defer/suffix
    checked when leaving the region

defer
    alias/convenience form of fix
```

These conditions should not remain arbitrary syntax attached to a function.

They become nodes in the constraint graph:

```text
                   Function increment
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
      x > 0            x < 100          x >= 10
       prefix              fix             defer/suffix
          │               │               │
          └───────────────┼───────────────┘
                          ▼
                     Function CFG
```

The compiler can then construct a dependency DAG for evaluating constraints.

Checks should be ordered by dependency and cost rather than merely by declaration order.

For dispatch this may produce:

```text
Type
 ↓
Trait / subtype
 ↓
Static shape
 ↓
Static value
 ↓
Cheap runtime condition
 ↓
Expensive runtime condition
 ↓
Implementation
```

This allows Severian to cheaply eliminate candidates before performing expensive checks.

Submodules are the boundary between language/package semantics and backend realization.

The normal mapping is:

```text
Submodule
    ↓
semantic resolution
    ↓
with / constraint resolution
    ↓
specialization
    ↓
ObjectUnit
    ↓
.o
```

but:

```text
Submodule != .o
```

A submodule may generate several ObjectUnits, and multiple submodules may be merged into one ObjectUnit.

`ObjectUnit` is therefore the backend emission boundary.

Files remain source-storage boundaries.

```text
File
    path
    text
    source positions
```

They are not required to correspond to modules, submodules, or object files.

---

## Appendix

| Term          | Kind                | Definition                                                                              |
| ------------- | ------------------- | --------------------------------------------------------------------------------------- |
| `Library`     | package node        | Published collection of modules.                                                        |
| `Module`      | namespace node      | Public namespace containing submodules.                                                 |
| `Submodule`   | compilation node    | Semantic compilation and dependency unit.                                               |
| `Block`       | execution node      | Scope, CFG, ownership, and lifetime region.                                             |
| `Sentence`    | grammar node        | Complete grammar-resolved operation or statement.                                       |
| `Symbol`      | semantic node       | Named, literal, type, value, or other language symbol.                                  |
| `Operator`    | semantic node       | Grammar-defined relation or operation over symbols.                                     |
| `Operation`   | execution node      | Resolved computation produced from a sentence.                                          |
| `With`        | interface           | Attaches declarations, constraints, invariants, refinements, or requirements to a node. |
| `Constraint`  | graph node          | Predicate required for validity, dispatch, refinement, or execution.                    |
| `Entry`       | constraint phase    | Constraint evaluated when entering a region.                                            |
| `Fix`         | constraint phase    | Constraint maintained throughout a region.                                              |
| `Exit`        | constraint phase    | Constraint evaluated when leaving a region.                                             |
| `Defer`       | constraint form     | Alias/convenience form for a fixed invariant.                                           |
| `ValueDomain` | analysis node       | Known subset/range/properties of possible values.                                       |
| `Universal`   | graph IR            | Typed semantic graph containing compiler-visible entities.                              |
| `ObjectUnit`  | backend node        | Unit selected for native/backend emission.                                              |
| `File`        | source object       | Physical source text and source positions.                                              |
| `Contains`    | graph edge          | Hierarchical containment.                                                               |
| `Depends`     | graph edge          | One node depends on another.                                                            |
| `Requires`    | graph edge          | Node requires a constraint/fact.                                                        |
| `Refines`     | graph edge          | Constraint narrows a type/value domain.                                                 |
| `TypeOf`      | graph edge          | Value or symbol has a type.                                                             |
| `Reads`       | graph edge          | Operation reads a value.                                                                |
| `Writes`      | graph edge          | Operation writes a value.                                                               |
| `Produces`    | graph edge          | Operation produces a value.                                                             |
| `Owns`        | graph edge          | Region owns a value/resource.                                                           |
| `Borrows`     | graph edge          | Region temporarily accesses a value/resource.                                           |
| `ControlFlow` | graph edge          | Runtime CFG transition.                                                                 |
| `Implements`  | graph edge          | Implementation satisfies an interface/trait.                                            |
| `Realizes`    | graph edge          | Semantic entity maps to compiled realization.                                           |
| `Compldefer/suffixy`  | constraint metadata | Asymptotic or relative evaluation cost.                                                 |
| `Phase`       | constraint metadata | Static, specialization, runtime, prefix, fix, or defer/suffix.                                   |
| `.sev`        | source format       | Severian source file.                                                                   |
| `.sevi`       | interface format    | Serialized package/compiler semantic interface.                                         |
| `.o`          | artifact            | Relocatable object file.                                                                |
| `.a`          | artifact            | Static object archive.                                                                  |
| `.so`         | artifact            | ELF shared library.                                                                     |

Conceptual interfaces:

```sev
trait Node:
    id: Symbol

trait Container: Node:
    children: list[Node]

class Library: Container:
    modules: list[Module]

class Module: Container:
    submodules: list[Submodule]

class Submodule: Container:
    blocks: list[Block]
    imports: list[Submodule]
    exports: list[Symbol]
    dependencies: list[Node]

class Block: Container:
    sentences: list[Sentence]
    symbols: list[Symbol]
    blocks: list[Block]

class Sentence: Node:
    symbols: list[Symbol]
    operators: list[Operator]
    operations: list[Operation]

trait With:
    requirements: list[Constraint]

trait Constraint: Node:
    dependencies: list[Node]
    phase: ConstraintPhase
    compldefer/suffixy: Compldefer/suffixy
    cost: number | unknown

enum ConstraintPhase:
    Static
    Specialization
    Entry
    Fix
    Runtime
    Exit

class ObjectUnit:
    realizations: list[Node]
    target: Target
```

The syntax is illustrative rather than normative.

---

## Context

Severian already has concepts that conventional AST/HIR pipelines usually introduce later:

```text
symbols
grammar
operators
sentences
constraints
ownership
CFG
Universal
MIR
```

A conventional:

```text
AST → HIR → MIR
```

pipeline would duplicate some of those responsibilities.

However, Severian still needs stable structural units for:

```text
scope
dependency tracking
ownership
CFG construction
incremental compilation
package interfaces
artifact generation
```

The hierarchy supplies those structural boundaries.

The graph supplies cross-hierarchy relationships.

For example:

```text
Library
  Module
    Submodule Tensor
      Block matmul
        Sentence result = lhs @ rhs
```

may simultaneously have:

```text
lhs ─TypeOf→ Tensor[bf16]
rhs ─TypeOf→ Tensor[bf16]

matmul ─Requires→ rank(lhs) == 2
matmul ─Requires→ rank(rhs) == 2
matmul ─Requires→ lhs.cols == rhs.rows

matmul ─Depends→ AMDGPU implementation
matmul ─Realizes→ matmul-amdgpu.o
```

Containment is therefore only one projection.

The compiler should be able to derive graph views such as:

```text
Universal.subgraph(Contains)
    → source/package hierarchy

Universal.subgraph(Depends)
    → compilation DAG

Universal.subgraph(Requires)
    → with/constraint DAG

Universal.subgraph(ControlFlow)
    → CFG

Universal.subgraph(Owns, Borrows)
    → memory graph

Universal.subgraph(TypeOf, Refines)
    → type/value-domain graph

Universal.subgraph(Realizes)
    → package/artifact graph
```

### `with` as an interface

`with` should provide a common mechanism for associating additional semantic information with a construct.

Examples:

```sev
def foo(x) with x > 0:
```

```sev
for x in values with i := 0:
```

```sev
foo(x) with x > 5:
```

```sev
task = async foobar() with self
```

These do not necessarily mean the same thing operationally.

The grammar/interface implementing the construct determines how the attached `with` information is interpreted.

Conceptually:

```text
Sentence
   │
   ├─ primary grammar
   │
   └─ With interface
        │
        ├─ declaration
        ├─ constraint
        ├─ invariant
        ├─ execution context
        └─ ownership/context requirement
```

This keeps `with` extensible rather than hard-coding every possible use into parser/compiler logic.

---

## Problem(s)

### Hierarchy alone is insufficient

A simple hierarchy:

```text
Library
→ Module
→ Submodule
→ Block
→ Sentence
```

cannot represent cross-cutting relationships such as:

```text
operation dependency
type refinement
ownership
implementation selection
constraints
artifact realization
```

These require graph edges.

### A single undifferentiated graph is also insufficient

Containment, CFG, dependency resolution, and constraint evaluation have different invariants.

The solution should therefore be:

```text
one typed Universal graph
+
multiple graph/DAG projections
```

rather than completely independent IRs.

### `with` constraints need ordering

Consider candidate dispatch requiring:

```text
type(x) == Tensor
dtype(x) == bf16
rank(x) == 2
len(x) > 1024
sorted(x)
arbitrary_runtime_test(x)
```

These checks do not have equal cost.

The compiler should avoid:

```text
expensive predicate
↓
discover wrong type
```

and instead schedule:

```text
type
↓
dtype
↓
rank
↓
length
↓
sorted
↓
expensive predicate
```

### Constraints depend on other constraints

A constraint may only become meaningful after another fact is established.

For example:

```text
x has Tensor type
    ↓
rank(x) is defined
    ↓
shape(x)[1] is defined
    ↓
shape(x)[1] == y.shape[0]
```

The constraints therefore naturally form a DAG.

### Conditions refine values

Given:

```sev
if x > 0:
    foo(x)
```

the true branch knows more than merely `x: int`.

```text
before:
    x : int

true branch:
    x : int
    domain = Positive

false branch:
    x : int
    domain = NonPositive
```

The compiler should record this through `Refines` relationships.

### Files are not semantic units

A physical file should not determine package or incremental compilation boundaries.

### Submodules must not equal object files

Submodules are target-independent semantic objects.

`.o` files are target-specific linker artifacts.

Specialization can require:

```text
Submodule List
    ├─ list-base.o
    ├─ list-int.o
    └─ list-string.o
```

### Sentences and blocks cannot be fully collapsed

This:

```sev
task = async foobar() with self
```

is structurally complex but introduces no child execution region.

A `Sentence` is therefore required independently of `Block`.

---

## Examples

### Containment

```text
Library collections
└─ Module sequence
   └─ Submodule list
      ├─ Block List
      ├─ Block Iterator
      └─ Block algorithms
         ├─ Sentence sort(...)
         └─ Sentence reverse(...)
```

### Complex sentence

```sev
task = async foobar() with self
```

may form:

```text
Sentence Assignment
├─ Symbol task
├─ Operator =
└─ Operation AsyncCall
   ├─ Operator async
   ├─ Symbol foobar
   ├─ Operator call
   └─ With
      └─ Symbol self
```

No nested block is required.

### Block-producing sentence

```sev
if ready:
    foo()
```

becomes:

```text
Sentence If
├─ ready
└─ Block
   └─ Sentence Call(foo)
```

### `with` contract

```sev
def increment(x, view y) with {
    prefix x > 0
    fix x < 100
    defer/suffix x >= 10
    defer unchanged(y)
}:
    x += 10
    return x
```

Constraint graph:

```text
                    increment
                        │
         ┌──────────────┼──────────────┐
         │              │              │
         ▼              ▼              ▼
      x > 0          x < 100        x >= 10
       prefix            fix            defer/suffix
                         │
                  unchanged(y)
                       fix
```

These constraints then connect to the CFG locations at which they must hold.

### Constraint DAG

Suppose an implementation requires:

```text
x : Tensor
x.dtype == bf16
x.rank == 2
x.contiguous
x.shape[1] == y.shape[0]
```

Dependencies become:

```text
Tensor(x)
   │
   ├──→ dtype(x)
   │      └──→ bf16?
   │
   └──→ shape(x)
          │
          ├──→ rank(x) == 2
          │
          └──→ x.shape[1]
                    │
                    └──→ compare y.shape[0]

Tensor(x)
   └──→ contiguous(x)
```

The compiler topologically orders the required checks.

### Cost-aware dispatch

Consider implementations:

```sev
def process(x: Tensor) with x.rank == 2:
    ...

def process(x: Tensor) with expensive_check(x):
    ...
```

The dispatch DAG should favor cheap discrimination:

```text
input x
  │
  ▼
Tensor?
  │
  ▼
rank == 2?
  │
  ├─ matching candidate
  │
  └─ unresolved
       │
       ▼
 expensive_check(x)
```

Each constraint can expose:

```text
phase
dependencies
compldefer/suffixy
estimated cost
```

For example:

```text
type(x) == Tensor
    static
    O(1)

rank(x) == 2
    shape/runtime
    O(1)

len(x) > 1000
    runtime
    O(1)

sorted(x)
    runtime
    O(n)
```

### Value refinement

```sev
if x > 0:
    foo(x)
else:
    bar(x)
```

produces:

```text
                       x : int
                          │
                        x > 0
                       /     \
                    true     false
                     │         │
                     ▼         ▼
                 Positive   NonPositive
                     │         │
                   foo()     bar()
                     \         /
                      \       /
                         join
                          │
                          ▼
                        int
```

The branch constraint refines the value domain.

### Type → value → operation selection

The same mechanism can dispatch from coarse facts toward increasingly specific facts:

```text
Type
 ↓
Subtype / Trait
 ↓
Shape
 ↓
Value domain
 ↓
Exact value
 ↓
Runtime predicate
 ↓
Operation implementation
```

For example:

```text
number
 ├─ int
 │   ├─ Negative
 │   ├─ Zero
 │   ├─ Range(1..255)
 │   └─ Dynamic
 │
 └─ float
```

The compiler does not create nodes for every possible integer.

It records useful abstract value domains.

### Package compilation

```text
Library
  ↓
Module
  ↓
Submodule
  ↓
dependency DAG
  ↓
constraint resolution
  ↓
specialization
  ↓
ObjectUnit
  ↓
.o
```

Example:

```text
Submodule tensor.matmul
    │
    ├─ requires Tensor
    ├─ requires bf16
    ├─ requires rank=2
    ├─ requires AMDGPU
    │
    └─ realization
          ↓
      matmul-amdgpu.o
```

### Object splitting

```text
Submodule list
├─ ObjectUnit list.core
│  └─ list.core.o
├─ ObjectUnit List[int]
│  └─ list.int.o
└─ ObjectUnit List[string]
   └─ list.string.o
```

### Package consumer

```sev
import collections.sequence.list
```

can resolve:

```text
Library collections
    ↓
Module sequence
    ↓
Submodule list
    ↓
required symbols
    ↓
required constraints
    ↓
required realization
    ↓
ObjectUnit
```

without loading the original source hierarchy.

---

## Testing

### Containment tests

Validate:

```text
Library → Module
Module → Submodule
Submodule → Block
Block → Sentence
Sentence → Symbol / Operator / Operation
```

### Sentence tests

Ensure the following remain single sentences:

```sev
x = 1
x = a + b * c
task = async foobar() with self
foo(x) with x > 5
```

### Block tests

Ensure nested execution creates blocks:

```sev
if x:
    ...

for x in y:
    ...

while x:
    ...

def foo():
    ...
```

### `with` parsing tests

Test:

```sev
foo(a) with a > 5:
```

```sev
for i in values with j := 1:
```

```sev
task = async foo() with self
```

```sev
def foo(x) with {
    prefix x > 0
    fix x < 10
    defer/suffix x >= 1
}:
```

Verify that grammar-specific implementations receive the appropriate attached `With` structure.

### Constraint DAG tests

Construct constraints with explicit dependencies.

Verify:

```text
acyclic graph accepted
cycle diagnosed
topological ordering deterministic
```

### Constraint cost tests

Given:

```text
TypeCheck cost=1
ShapeCheck cost=2
LinearScan cost=n
```

verify that independent predicates are scheduled by appropriate cost while preserving dependency ordering.

### Refinement tests

Compile:

```sev
if x > 0:
    foo(x)
```

Verify that the true branch receives a refined positive domain.

At the CFG join, verify correct domain widening.

### Dispatch tests

Provide several implementations differentiated by:

```text
type
trait
shape
value
runtime predicate
```

Verify that cheap/static discrimination occurs before expensive runtime predicates.

### Ambiguity tests

If multiple implementations remain valid after all constraints:

```text
candidate A matches
candidate B matches
```

the compiler must diagnose ambiguity rather than select by incidental ordering.

### Entry/fix/defer/suffix tests

Verify:

```text
prefix
    checked at region prefix

fix
    maintained at relevant mutation/control boundaries

defer/suffix
    checked for each valid defer/suffix

defer
    behaves as fix
```

### Ownership integration

A `with` constraint referencing ownership state must be represented in the same graph as:

```text
Owns
Borrows
Moves
```

Verify constraints remain valid across CFG edges.

### Incremental compilation

Changing an implementation without changing a submodule interface should not invalidate unrelated dependents.

Changing exported constraints must invalidate consumers that depend on those constraints.

### Source independence

Move a declaration between physical files while retaining semantic identity.

Verify no unnecessary package-level invalidation.

### Object realization

Test:

```text
one Submodule → one ObjectUnit
one Submodule → multiple ObjectUnits
multiple Submodules → one ObjectUnit
```

### Package interface round trip

Compile a package and consume it using only:

```text
.sevi
object artifacts
metadata
```

Verify that constraints and `with` interfaces required for downstream dispatch remain available.

### Universal graph projections

Verify consistency between:

```text
Contains
Depends
Requires
Refines
TypeOf
Owns
Borrows
ControlFlow
Realizes
```

---

## Performance

The graph representation should improve compilation and runtime dispatch by exposing dependency structure explicitly.

### Constraint scheduling

Constraints should carry enough metadata to permit ordering by:

```text
dependency
phase
cost
compldefer/suffixy
selectivity where known
```

Dependency ordering always takes priority.

Within independent constraints, cheaper checks should generally occur first.

Instead of:

```text
O(n) predicate
↓
type check
↓
shape check
```

the compiler can produce:

```text
type check       O(1)
↓
shape check      O(1)
↓
runtime predicate O(n)
```

This is particularly important for overloaded operators and multiple implementers.

### Static elimination

Constraints resolvable during compilation should be removed from runtime execution.

```text
Static
    ↓ resolved during compile

Specialization
    ↓ resolved when realization is selected

Runtime
    ↓ emitted only when still unknown
```

Therefore:

```text
with x: int
```

should not generate runtime checking if type resolution already guarantees it.

### Refinement reuse

Once a condition establishes:

```text
x ∈ Range(1..255)
```

downstream operations should reuse that fact rather than re-evaluate equivalent constraints.

The value-domain graph acts as cached semantic knowledge.

### Dispatch DAG reuse

A dispatch family such as `+` should construct a reusable decision DAG rather than linearly testing every implementation.

Instead of:

```text
candidate1?
candidate2?
candidate3?
...
candidateN?
```

use shared prefixes:

```text
                   Type
                 /      \
               int      tensor
              /            \
           value           dtype
           /   \             |
        small large         bf16
```

Multiple implementations can reuse the same type, trait, shape, and value checks.

### Incremental compilation

Submodules should maintain separate hashes where useful:

```text
source hash
interface hash
constraint hash
implementation hash
ObjectUnit hash
```

A local source change should therefore not automatically invalidate every consumer.

### Package-level DAG

Compilation follows the changed dependency closure:

```text
changed Submodule
      │
      ▼
changed interface?
   /          \
 no            yes
 │              │
 ▼              ▼
local       dependent
ObjectUnit  submodules
```

### Object generation

The default remains:

```text
Submodule → ObjectUnit → .o
```

but backend optimization may split or merge ObjectUnits according to:

```text
target
generic specialization
device
linkage
visibility
LTO
code size
```

without affecting semantic identity.

### Overall architecture

The resulting compiler model is:

```text
                         Universal
                            │
      ┌─────────────┬───────┼────────┬─────────────┐
      │             │       │        │             │
      ▼             ▼       ▼        ▼             ▼
 containment    dependency  with   ownership       CFG
 hierarchy         DAG      DAG      graph          graph
      │             │       │        │             │
      └─────────────┴───────┼────────┴─────────────┘
                            ▼
                        resolution
                            │
                            ▼
                           MIR
                            │
                            ▼
                          MLIR
                            │
                            ▼
                       ObjectUnits
                            │
                            ▼
                      .o / .a / .so
```

The key invariants are:

```text
Hierarchy defines where a semantic entity belongs.

Grammar defines how symbols form sentences.

Sentence defines a complete grammar/evaluation unit.

Block defines nested execution, scope, CFG, and lifetime.

with defines requirements and refinements over semantic entities.

The constraint DAG defines what must be known and in what order.

Submodule defines the semantic compilation/package unit.

ObjectUnit defines what is emitted together.

Universal connects all of these through typed graph relationships.
```
