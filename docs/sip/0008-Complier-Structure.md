SIP-0000: Semantic Compilation Hierarchy

Status: Draft

Type: Language | Compiler | Package

Authors:

Created: 2026-09-22

Target:

Supersedes:

Superseded by:

**## Summary**

Severian should organize source and compilation semantics using the following hierarchy:

```text
Library
  └─ Module
      └─ Submodule
          └─ Block
              └─ Sentence
                  └─ Symbol / Operator / Operation
```

These levels describe semantic organization rather than filesystem organization.

The hierarchy has distinct responsibilities:

```text
Library
    Published package or standard-library boundary.

Module
    Public namespace and dependency boundary.

Submodule
    Semantic compilation unit and default object-generation boundary.

Block
    Scope, control-flow, ownership, and lifetime region.

Sentence
    Complete grammar-resolved expression or statement.

Symbol / Operator / Operation
    Atomic semantic components used to construct sentences.
```

The compiler should not treat this structure as a strict syntax tree. It is the containment projection of Severian's Universal graph.

Other graph edges represent:

```text
dependency
type
value refinement
control flow
ownership
borrow
constraint
implementation
artifact realization
```

A submodule is the primary boundary between package semantics and backend compilation.

A submodule should commonly produce one object file:

```text
Submodule
    ↓ resolution
    ↓ specialization
    ↓ lowering
ObjectUnit
    ↓ serialization
.o
```

However:

```text
Submodule != .o
```

The compiler may split one submodule into several object units or merge multiple submodules into one object unit.

Examples include:

```text
list
  ├─ list.core.o
  ├─ list.int.o
  └─ list.string.o
```

or:

```text
small_a ─┐
small_b ─┼→ utilities.o
small_c ─┘
```

This keeps semantic organization independent from backend and linker requirements.

Files remain source-storage objects rather than compiler semantic boundaries.

```text
File
    path
    text
    source ranges

Submodule
    semantic identity
    dependencies
    exports
    implementations
    compilation state
```

One file may provide several submodules, and multiple files may contribute to one submodule.

A `Sentence` is retained separately from `Block`.

A sentence represents a complete grammar-resolved unit that may contain complex internal structure without introducing a new scope.

For example:

```sev
task = async foobar() with self
```

is one sentence.

Conceptually:

```text
Sentence
├─ assignment
│  └─ task
└─ async call
   ├─ foobar
   └─ with
      └─ self
```

By contrast:

```sev
if ready:
    task = async foobar() with self
```

contains a nested block:

```text
Sentence: if
├─ condition: ready
└─ Block
   └─ Sentence
      └─ task = async foobar() with self
```

The defining distinction is:

```text
Sentence
    grammatical and evaluation boundary

Block
    nested execution, scope, CFG, and ownership boundary
```

The hierarchy therefore gives Severian structural organization normally associated with AST/HIR systems without requiring a conventional AST/HIR pipeline.

The expected pipeline is:

```text
source
  ↓
symbols
  ↓
grammar matching
  ↓
sentences
  ↓
blocks
  ↓
submodules
  ↓
Universal graph
  ↓
semantic / constraint / ownership / CFG projections
  ↓
MIR
  ↓
MLIR
  ↓
ObjectUnit
  ↓
.o / .a / .so
```

The package system operates primarily on modules and submodules rather than files.

Compiled package interfaces can therefore expose semantic units independently of original source layout.

---

**## Appendix**

| Term           | Kind                   | Definition                                                                            |
| -------------- | ---------------------- | ------------------------------------------------------------------------------------- |
| `Library`      | semantic/package unit  | Published collection of modules.                                                      |
| `Module`       | semantic/package unit  | Public namespace and dependency grouping within a library.                            |
| `Submodule`    | semantic/compiler unit | Smallest package-addressable compilation unit and default object-generation boundary. |
| `Block`        | semantic region        | Scope, CFG, lifetime, ownership, and nested execution region.                         |
| `Sentence`     | grammar unit           | Complete grammar-resolved expression or statement.                                    |
| `Symbol`       | semantic atom          | Named or literal entity participating in a sentence.                                  |
| `Operator`     | grammar/semantic atom  | Operation syntax used to relate symbols or other operations.                          |
| `Operation`    | semantic atom          | Resolved computation produced from symbols/operators.                                 |
| `Universal`    | graph IR               | Shared semantic graph containing language entities and typed relationships.           |
| `ObjectUnit`   | backend unit           | Group of lowered definitions selected for emission into an object artifact.           |
| `File`         | source unit            | Physical source text and source-location information.                                 |
| `LibraryId`    | identifier             | Stable identity for a library.                                                        |
| `ModuleId`     | identifier             | Stable identity for a module.                                                         |
| `SubmoduleId`  | identifier             | Stable semantic identity independent of file path.                                    |
| `BlockId`      | identifier             | Stable identity for a block within a submodule.                                       |
| `SentenceId`   | identifier             | Stable identity for a grammar-resolved sentence.                                      |
| `SymbolId`     | identifier             | Identity of a symbol in the Universal graph.                                          |
| `ObjectUnitId` | identifier             | Identity of an emitted backend partition.                                             |
| `Contains`     | graph edge             | Hierarchical containment relationship.                                                |
| `Depends`      | graph edge             | Semantic or compilation dependency.                                                   |
| `TypeOf`       | graph edge             | Associates a value or symbol with a type.                                             |
| `Requires`     | graph edge             | Associates an entity with a constraint.                                               |
| `Refines`      | graph edge             | Records value/type-domain refinement.                                                 |
| `Reads`        | graph edge             | Operation reads a value.                                                              |
| `Writes`       | graph edge             | Operation mutates a value.                                                            |
| `Produces`     | graph edge             | Operation produces a value.                                                           |
| `Owns`         | graph edge             | Block/value ownership relationship.                                                   |
| `Borrows`      | graph edge             | Temporary ownership/access relationship.                                              |
| `ControlFlow`  | graph edge             | CFG transition between operations or regions.                                         |
| `Realizes`     | graph edge             | Semantic entity maps to a backend implementation/artifact.                            |
| `.sev`         | source file            | Human-authored Severian source.                                                       |
| `.sevi`        | interface artifact     | Serialized semantic/package interface required by consumers.                          |
| `.o`           | object artifact        | Relocatable native object emitted from one ObjectUnit.                                |
| `.a`           | archive artifact       | Static archive containing object files.                                               |
| `.so`          | shared artifact        | Dynamically linkable shared library on ELF platforms.                                 |
| `.dll`         | shared artifact        | Dynamically linkable library on Windows.                                              |
| `.dylib`       | shared artifact        | Dynamically linkable library on macOS.                                                |

Proposed conceptual interfaces:

```sev
trait SemanticNode:
    id: Symbol

trait ContainerNode: SemanticNode:
    children: list[SemanticNode]

class Library: ContainerNode:
    modules: list[Module]

class Module: ContainerNode:
    submodules: list[Submodule]

class Submodule: ContainerNode:
    blocks: list[Block]
    imports: list[Submodule]
    exports: list[Symbol]
    dependencies: list[Submodule]

class Block: ContainerNode:
    sentences: list[Sentence]
    symbols: list[Symbol]
    children: list[Block]

class Sentence: SemanticNode:
    symbols: list[Symbol]
    operators: list[Operator]
    operations: list[Operation]

class ObjectUnit:
    source: list[Submodule | Block | Operation]
    target: Target
    symbols: list[Symbol]
```

These interfaces are conceptual and do not prescribe final syntax.

---

**## Context**

Severian does not need to reproduce the conventional compiler structure:

```text
parser
  ↓
AST
  ↓
HIR
  ↓
MIR
```

The language already has higher-level semantic concepts:

```text
grammar
symbols
operators
sentences
constraints
ownership
CFG
Universal
MIR
```

A conventional AST would duplicate much of the information already represented by grammar matching and Universal lowering.

However, Severian still requires explicit answers to several structural questions:

```text
What constitutes a namespace?

What constitutes a compilation unit?

Where does scope begin and end?

Where does ownership analysis operate?

Where does CFG construction occur?

What unit is cached?

What unit is invalidated when source changes?

What unit is exposed through a package interface?

What unit maps to native artifacts?
```

Using files for these responsibilities couples filesystem layout to compiler architecture.

For example:

```text
src/list.sev
```

should not imply:

```text
namespace == compilation unit == cache unit == object file
```

The language needs semantic boundaries independent of storage layout.

The hierarchy provides those boundaries.

```text
Library
    package distribution

Module
    namespace

Submodule
    semantic compilation

Block
    scope and execution region

Sentence
    grammar-resolved computation

Symbol/Operator
    atomic semantics
```

The hierarchy is also only one view of Universal.

Containment forms:

```text
Library → Module → Submodule → Block → Sentence
```

while semantic relationships may cross that hierarchy:

```text
Submodule A ─depends→ Submodule B

Operation X ─reads→ Value Y

Value Y ─type→ Tensor

Sentence S ─requires→ Constraint C

Block F ─owns→ Value Y
```

This allows containment to remain simple while dependency resolution, ownership, CFG construction, and dispatch use graph/DAG representations.

---

**## Problem(s)**

### Files are not semantic compilation boundaries

A physical file exists for editing and storage.

It should not determine:

```text
incremental invalidation
package visibility
linker partitioning
semantic identity
```

Moving a declaration between source files should not necessarily change its semantic identity or invalidate consumers.

### Modules are too large as object-generation units

A module may contain many independent implementations.

Generating one object file for an entire module can increase:

```text
recompilation
linking work
artifact size
dependency fan-out
```

A smaller semantic boundary is needed.

`Submodule` fills this role.

### Object files are too low-level as language units

An object file contains target-specific linker information.

A `.o` cannot adequately represent:

```text
generic definitions
grammar definitions
traits
constraints
ownership contracts
unrealized implementations
target-independent types
```

Therefore `.o` must remain below the semantic model.

### Blocks and sentences have different responsibilities

Collapsing `Sentence` into `Block` makes simple expressions appear to create scopes.

For example:

```sev
task = async foobar() with self
```

contains several operators and semantic relationships but no nested scope.

Calling this a block would make block semantics ambiguous.

Conversely:

```sev
if ready:
    foo()
```

does introduce a nested execution region.

The language therefore needs both concepts.

### Package artifacts must not depend on source layout

Published packages should expose:

```text
modules
submodules
symbols
types
implementations
artifacts
```

rather than requiring consumers to understand:

```text
src/foo.sev
src/internal/bar.sev
```

The package interface should point from semantic definitions to realizations.

### Incremental compilation requires stable units

The compiler must distinguish:

```text
source changed

implementation changed

semantic interface changed

target realization changed
```

A stable `SubmoduleId` provides a useful unit for semantic hashing and invalidation.

### Backend partitioning must remain flexible

A one-to-one mapping:

```text
submodule → .o
```

would fail for generics, target specialization, GPU kernels, or linker optimization.

The compiler therefore requires an explicit `ObjectUnit` boundary.

---

**## Examples**

### Basic hierarchy

```text
library collections
│
└─ module sequence
   │
   ├─ submodule list
   │  │
   │  ├─ block List
   │  │
   │  ├─ block Iterator
   │  │
   │  └─ block algorithms
   │  │     ├─ sentence sort(...)
   │  │     └─ sentence reverse(...)
   │  │
   │  └─ object realization
   │        └─ list.o
   │
   └─ submodule vector
      └─ ...
```

### Sentence without nested block

```sev
task = async foobar() with self
```

Representation:

```text
Sentence Assignment
│
├─ Symbol task
├─ Operator =
└─ Operation AsyncCall
   ├─ Symbol foobar
   ├─ Operator async
   ├─ Operator call
   └─ Operator with
      └─ Symbol self
```

No nested scope is created.

### Sentence containing a block

```sev
if ready:
    task = async foobar() with self
```

Representation:

```text
Sentence If
├─ Symbol ready
└─ Block
   └─ Sentence Assignment
      ├─ task
      └─ AsyncCall
```

### Function block

```sev
def increment(x: int) -> int:
    y = x + 1
    return y
```

Representation:

```text
Block Function increment
│
├─ symbols
│  ├─ x:int
│  └─ y:int
│
├─ Sentence Assignment
│  └─ y = x + 1
│
└─ Sentence Return
   └─ y
```

The block owns:

```text
scope
CFG
ownership state
lifetime state
effects
```

The sentences describe the computations contributing to that CFG.

### One file containing several submodules

```text
math.sev
```

may produce:

```text
Module math
├─ Submodule vector
├─ Submodule matrix
└─ Submodule scalar
```

Changing `matrix` does not automatically invalidate `vector`.

### Multiple files contributing to one submodule

```text
vector.sev
vector-arithmetic.sev
vector-conversions.sev
```

may all contribute to:

```text
Submodule vector
```

especially when extensions are used.

### Default object mapping

```text
Submodule vector
    ↓
ObjectUnit vector/native/x86_64
    ↓
vector.o
```

### Split object mapping

Generic or specialized code may require:

```text
Submodule list
│
├─ ObjectUnit list.core
│    └─ list.core.o
│
├─ ObjectUnit list[int]
│    └─ list.int.o
│
└─ ObjectUnit list[string]
     └─ list.string.o
```

### Target-dependent mapping

```text
Submodule tensor.matmul
│
├─ ObjectUnit host
│    └─ matmul-host.o
│
├─ ObjectUnit AMDGPU
│    └─ matmul-amdgpu.o
│
└─ MLIR/XLA realization
     └─ matmul.mlir
```

### Merged object mapping

Small submodules may be combined:

```text
Submodule math.constants ─┐
Submodule math.scalar    ─┼→ math-support.o
Submodule math.compare   ─┘
```

The semantic hierarchy remains unchanged.

### Package consumption

A consumer writes:

```sev
import collections.sequence.list
```

Resolution becomes:

```text
Library collections
    ↓
Module sequence
    ↓
Submodule list
    ↓
exported symbol List
    ↓
required realizations
    ↓
ObjectUnit(s)
    ↓
.o / .a / .so
```

The consumer does not need the original `.sev` source layout.

---

**## Testing**

Testing should validate structural semantics, incremental compilation, graph integrity, and artifact realization.

### Hierarchy tests

Verify:

```text
Library contains Module
Module contains Submodule
Submodule contains Block
Block contains Sentence
Sentence references Symbols/Operators
```

Reject illegal ownership or containment relationships.

### Sentence tests

Test sentences with no nested blocks:

```sev
x = 1
x = a + b * c
task = async foobar() with self
result = foo(x) with x > 0
```

Verify they remain single sentences despite nested expression structure.

### Block tests

Test structures introducing execution regions:

```sev
if x:
    ...

for x in values:
    ...

while x:
    ...

def foo():
    ...

class Foo:
    ...
```

Verify correct block creation, parentage, scope, and CFG ownership.

### Source-layout independence

Compile:

```text
a.sev
```

then move an unchanged declaration to:

```text
b.sev
```

while preserving semantic identity.

Verify that semantic hashes and dependent submodules do not change solely because the source path changed.

### Multi-file submodule

Provide declarations across multiple source files that contribute to one submodule.

Verify:

```text
one semantic Submodule
multiple source origins
correct source diagnostics
correct final artifact
```

### Multi-submodule file

Place multiple submodules in one `.sev` file.

Change one.

Verify unrelated submodules remain cached.

### Incremental invalidation

Given:

```text
A → B → C
D
```

change B's implementation without changing its interface.

Expected:

```text
rebuild B ObjectUnit
preserve dependent semantic state where valid
do not rebuild D
```

Then change B's exported interface.

Expected:

```text
invalidate A
invalidate relevant dependents
do not invalidate unrelated D
```

### Object splitting

Compile one submodule into several ObjectUnits.

Verify all emitted symbols resolve correctly during linking.

### Object merging

Compile several submodules into one ObjectUnit.

Verify semantic dependency information remains available even though backend artifacts were merged.

### Generic realization

Compile:

```sev
List[int]
List[string]
```

Verify:

```text
shared generic semantic definition
distinct realization when required
no unnecessary recompilation
```

### Ownership tests

For every executable block verify:

```text
values owned on entry
borrows introduced
moves performed
values live on CFG edges
drops generated at exits
```

Nested block exits must preserve parent-block ownership invariants.

### CFG tests

Verify sentences contribute operations to the containing block's CFG rather than constructing independent CFGs unnecessarily.

### Package interface round-trip

Compile a library.

Delete or hide its source.

Compile a consumer using only:

```text
.sevi
.o/.a/.so
metadata
```

Verify the consumer can resolve:

```text
types
symbols
implementations
ownership contracts
ABI
required artifacts
```

### Determinism

Identical semantic inputs must produce identical:

```text
SubmoduleId
semantic hash
dependency graph
ObjectUnit partition before nondeterministic linker metadata
```

### Graph integrity

Validate that projections of Universal remain consistent:

```text
Contains
Depends
TypeOf
Requires
Owns
Borrows
ControlFlow
Realizes
```

A semantic node must not become unreachable solely because it appears in a different source file.

---

**## Performance**

The hierarchy is intended to reduce compilation work rather than add a new heavyweight IR.

The primary performance unit is the `Submodule`.

Each submodule should support a semantic hash derived from information that affects consumers:

```text
exports
types
constraints
layouts
ownership contracts
ABI
implementation requirements
```

Implementation-only information should be hashed separately where possible.

This allows the compiler to distinguish:

```text
source hash
semantic/interface hash
implementation hash
ObjectUnit hash
```

A source edit therefore does not necessarily imply recompilation of every dependent unit.

Expected invalidation model:

```text
source changed
    ↓
recompute affected sentence/block
    ↓
recompute affected submodule
    ↓
interface unchanged?
    ├─ yes → preserve dependents
    └─ no  → invalidate dependent submodules
```

Compilation complexity should primarily scale with the changed dependency closure rather than the complete package.

The package dependency graph operates at submodule granularity:

```text
Submodule A ─depends→ Submodule B
```

rather than merely:

```text
Module A ─depends→ Module B
```

This allows unused submodules to remain uncompiled and unlinked.

Backend partitioning remains independent.

The compiler may choose ObjectUnits based on:

```text
target
generic realization
linkage
visibility
optimization
device
code size
LTO policy
```

without modifying semantic organization.

The default should remain simple:

```text
one used Submodule
    ↓
one ObjectUnit
    ↓
one .o
```

and only split or merge when required.

`Sentence` and `Block` should also avoid unnecessary compiler-node expansion.

Expression structure belongs within a sentence unless execution introduces a nested region.

This prevents simple code such as:

```sev
x = a + b * c
```

from producing artificial block structures while still preserving its operator dependency graph.

The intended result is:

```text
stable semantic units
+
fine-grained dependency invalidation
+
package-driven artifact selection
+
flexible backend partitioning
+
no dependency on physical file organization
```

The compiler's containment hierarchy remains simple while Universal provides the graph relationships required by dispatch, ownership, CFG analysis, constraints, and package linking.
