# Severian Compiler

The Severian compiler transforms source code and package interfaces into typed,
ownership-checked, backend-ready programs.

The compiler is built around one declaration-backed type model shared across
interfaces, generics, primitives, collections, `CompileType`, XXI, FFI, ABI,
HIR, MIR, and lowering.

No compiler stage may invent a second representation for the same language
concept.

## Golden Path

```text
resolved package graph
        |
        v
package interfaces
        |
        +--> core.primitives
        +--> imported packages
        +--> external declarations
        |
        v
source files
        |
        v
lexer
        |
        v
parser
        |
        v
AST
        |
        v
semantic analysis
        |
        +--> declaration resolution
        +--> generic constraint solving
        +--> trait and implementation resolution
        +--> CompileType instantiation
        +--> XXI external-language resolution
        |
        v
typed HIR
        |
        v
ownership and effect analysis
        |
        v
MIR construction
        |
        +--> generic instantiation
        +--> typed operation normalization
        +--> external-call normalization
        |
        v
lowering
        |
        +--> ordinary language operations
        +--> CompileType handlers
        +--> FFI and ABI calls
        |
        v
backend
        |
        +--> MLIR / LLVM
        +--> StableHLO / XLA
        +--> Triton
        +--> platform-specific output
        |
        v
compiler output
```

The compiler receives a resolved package graph from the build tools. It does
not resolve package versions, read registries, or decide workspace build order.

## Directory Shape

```text
compiler/
├── README.md
│
├── source/
│   └── source files, locations, spans, and source maps
│
├── diagnostics/
│   └── structured diagnostics, error codes, labels, and rendering
│
├── boundaries/
│   ├── README.md
│   ├── interface/
│   │   └── portable package and declaration contracts
│   ├── xxi/
│   │   └── source-level external-language integration
│   ├── ffi/
│   │   └── foreign-function safety and representation contracts
│   ├── abi/
│   │   └── concrete calling conventions and data layouts
│   ├── backend/
│   │   └── backend capabilities and code-emission adapters
│   └── driver/
│       └── compilation-session orchestration
│
├── frontend/
│   ├── README.md
│   ├── lexer/
│   │   └── source text to tokens
│   ├── parser/
│   │   └── tokens to syntax
│   ├── ast/
│   │   └── syntax-level program model
│   ├── semantic/
│   │   └── names, types, generics, traits, contracts, and calls
│   ├── hir/
│   │   └── resolved and typed program model
│   └── ownership/
│       └── moves, borrows, resources, effects, and lifetime validity
│
└── transforms/
    ├── README.md
    ├── mir/
    │   └── execution-oriented typed IR
    └── lowering/
        └── conversion from MIR to backend operations
```

The repository-level `tools/` directory is outside `compiler/`.

Tools invoke the compiler through the driver API. Compiler crates must not
depend on build, package, test, or CLI tools.

## Canonical Identity Model

The compiler distinguishes stable declaration identity from session-local
compiler identity.

### Stable identities

Stable identities may appear in package interfaces and compiler artifacts.

```rust
pub struct PackageId(/* stable package identity */);
pub struct ModuleId(/* stable module identity */);
pub struct DeclarationId(/* stable declaration identity */);
pub struct GenericParamId(/* declaration-local parameter identity */);
pub struct TraitId(pub DeclarationId);
pub struct ImplementationId(/* stable implementation identity */);
pub struct CompileTypeId(/* stable CompileType contract identity */);
pub struct CompileHandlerId(/* stable lowering-handler identity */);
pub struct ExternalSymbolId(/* stable external declaration identity */);
```

A stable identity must derive from the registered declaration or interface
record. Compiler stages must not reconstruct identity by concatenating source
names.

### Session-local identities

Session-local identities exist only inside one compilation.

```rust
pub struct SourceId(pub u32);
pub struct HirId(pub u32);
pub struct TypeId(pub u32);
pub struct MirValueId(pub u32);
pub struct CompilationUnitId(pub u32);
```

`TypeId` is an interned reference into the current compilation's type table. It
must not be serialized into `.pkg` or interface files.

## Package Interface Model

`boundaries/interface` owns the portable representation of package contracts.

A package interface contains enough information to type-check a consuming
package without reading the dependency's implementation source.

```rust
pub struct PackageInterface {
    pub package: PackageIdentity,
    pub modules: Vec<ModuleInterface>,
    pub declarations: DeclarationTable,
    pub implementations: Vec<ImplementationInterface>,
    pub compile_types: Vec<CompileTypeContract>,
    pub external_symbols: Vec<ExternalSymbolContract>,
}
```

A declaration interface has one stable identity and one generic contract.

```rust
pub struct DeclarationInterface {
    pub id: DeclarationId,
    pub path: QualifiedName,
    pub visibility: Visibility,
    pub kind: DeclarationKind,
    pub generics: GenericSignature,
    pub declared_type: Option<InterfaceType>,
    pub constraints: Vec<InterfaceConstraint>,
    pub attributes: Vec<InterfaceAttribute>,
}
```

The interface model owns data representation and validation. It does not perform
source-level inference or expression analysis.

## Generic Model

Generics are structural. They are not encoded in strings such as
`"Tensor[f16, 32]"`.

A generic parameter has a stable identity within its declaration.

```rust
pub struct GenericSignature {
    pub parameters: Vec<GenericParameter>,
    pub constraints: Vec<GenericConstraint>,
}

pub struct GenericParameter {
    pub id: GenericParamId,
    pub name: String,
    pub kind: GenericParameterKind,
    pub default: Option<InterfaceGenericArgument>,
}
```

The initial parameter kinds are:

```rust
pub enum GenericParameterKind {
    Type,
    Const {
        value_type: InterfaceType,
    },
    Shape,
    Effect,
}
```

Generic arguments retain their category:

```rust
pub enum GenericArgument<T> {
    Type(T),
    Const(ConstValue),
    Shape(ShapeExpression),
    Effect(EffectSet),
}
```

A generic substitution records the result of solving a generic application:

```rust
pub struct GenericSubstitution<T> {
    pub bindings: BTreeMap<GenericParamId, GenericArgument<T>>,
}
```

For example:

```sev
Tensor[f16, Shape[8, 1024, 4096]]
```

is represented structurally:

```text
constructor:
    DeclarationId(core.tensor.Tensor)

arguments:
    Type(f16)
    Shape([8, 1024, 4096])
```

It must never be represented as one type-name string.

## Interface Type Model

Portable interface types use stable declaration identities.

```rust
pub enum InterfaceType {
    Declared(DeclarationId),

    Applied {
        constructor: DeclarationId,
        arguments: Vec<InterfaceGenericArgument>,
    },

    Parameter(GenericParamId),

    Tuple(Vec<InterfaceType>),
    Union(Vec<InterfaceType>),

    Function {
        parameters: Vec<InterfaceType>,
        result: Box<InterfaceType>,
        effects: EffectSet,
    },

    Reference {
        kind: ReferenceKind,
        inner: Box<InterfaceType>,
    },
}
```

Interfaces must not contain session-local `TypeId` values.

## Resolved Type Model

Semantic analysis converts interface and AST types into interned resolved
types.

```rust
pub struct ResolvedType {
    pub kind: TypeKind,
    pub compile_type: Option<ResolvedCompileType>,
}

pub enum TypeKind {
    Primitive(PrimitiveId),

    Declared(DeclarationId),

    Applied {
        constructor: DeclarationId,
        arguments: Vec<ResolvedGenericArgument>,
    },

    Parameter(GenericParamId),

    Tuple(Vec<TypeId>),
    Union(Vec<TypeId>),

    Function(FunctionType),

    Reference {
        kind: ReferenceKind,
        inner: TypeId,
    },

    Error,
}
```

There must not be parallel variants such as:

```rust
TypeKind::Primitive(i32_id)
TypeKind::Int
```

There must not be compiler-enumerated collection variants such as:

```rust
TypeKind::List
TypeKind::Set
TypeKind::Map
```

Collections are declaration-backed applied types:

```text
List[int]
    =
Applied {
    constructor: DeclarationId(core.collections.List),
    arguments: [Type(int)]
}
```

## Primitive Bootstrap

`core.primitives` is a mandatory package interface loaded before ordinary user
semantic analysis.

The primitive package owns:

* primitive declarations
* primitive categories
* widths and signedness
* literal defaults
* primitive capabilities
* primitive compatibility policy
* primitive arithmetic policy

The compiler owns:

* loading the primitive package interface
* binding primitive declarations to `PrimitiveId`
* assigning primitive types to expressions
* carrying resolved identities through HIR and MIR
* lowering resolved primitive operations

The compiler must not fabricate fallback declarations.

An embedded primitive package is allowed only when it is generated from the
same canonical `library/core/primitives` source or package artifact.

HIR must reference primitive identities. HIR must not own the language-level
primitive definitions.

## CompileType Model

A `CompileType` is not a second generic type representation.

Every source type first resolves through the normal declaration-backed type
model. A type declaration may additionally expose a `CompileTypeContract`.

```rust
pub struct CompileTypeContract {
    pub id: CompileTypeId,
    pub owner: DeclarationId,
    pub handler: CompileHandlerId,
    pub properties: Vec<CompilePropertyBinding>,
    pub operations: Vec<CompileOperationContract>,
    pub capabilities: CompileCapabilitySet,
}
```

When semantic analysis resolves an application of that declaration, it creates
a resolved CompileType instance:

```rust
pub struct ResolvedCompileType {
    pub contract: CompileTypeId,
    pub handler: CompileHandlerId,
    pub substitution: GenericSubstitution<TypeId>,
    pub properties: CompilePropertyValues,
}
```

For example, a tensor remains an applied type:

```text
Applied {
    constructor: DeclarationId(core.tensor.Tensor),
    arguments: [
        Type(f16),
        Shape([8, 1024, 4096]),
        Type(Device[gpu]),
    ]
}
```

Its resolved type record may also contain:

```text
compile_type:
    handler = tensor
    dtype = f16
    shape = [8, 1024, 4096]
    device = gpu
```

Ordinary collections generally have no CompileType contract:

```text
List[int]
    kind = Applied(...)
    compile_type = None
```

Compiler stages must not detect CompileTypes using source-name checks:

```rust
if name == "Tensor" { ... } // prohibited
```

They query resolved metadata:

```rust
types.compile_type(type_id)
```

## CompileType Flow

```text
interface declaration
        |
        v
CompileTypeContract
        |
        v
generic application
        |
        v
generic substitution
        |
        v
ResolvedCompileType
        |
        v
typed HIR operation
        |
        v
MIR operation
        |
        v
CompileHandlerId
        |
        v
lowering handler
        |
        +--> MLIR tensor/linalg
        +--> StableHLO/XLA
        +--> Triton
        +--> another registered backend
```

Semantic analysis determines what the operation means.

MIR records the resolved operation and types.

Lowering decides how the resolved operation maps to a backend.

## HIR Contract

HIR is the resolved source program.

Every semantically relevant node stores identities rather than source spelling.

Conceptually:

```rust
pub struct HirExpression {
    pub id: HirId,
    pub type_id: TypeId,
    pub kind: HirExpressionKind,
}

pub struct HirCall {
    pub declaration: DeclarationId,
    pub generic_substitution: GenericSubstitution<TypeId>,
    pub arguments: Vec<HirExpression>,
}
```

HIR preserves:

* resolved declarations
* resolved types
* generic substitutions
* selected traits and implementations
* effects
* ownership-relevant operations
* CompileType instances
* external symbol identities

HIR does not:

* parse source
* inspect type names
* define primitive policy
* decide ABI layouts
* emit backend operations

## Generic Instantiation

Generic definitions remain generic in HIR.

Call sites retain the solved substitution:

```text
identity[T](value: T) -> T

call:
    declaration = identity
    substitution = {
        T -> i32
    }
```

MIR construction decides whether to:

* instantiate a concrete MIR unit
* share a generic implementation
* pass witness or capability metadata

This implementation choice must not alter source-level generic semantics.

The first implementation may use monomorphization, but the data model must not
assume monomorphization is the only possible strategy.

## Trait and Constraint Resolution

Traits and constraints are declaration-backed.

A solved operation records the selected implementation:

```rust
pub struct ResolvedImplementation {
    pub implementation: ImplementationId,
    pub substitution: GenericSubstitution<TypeId>,
}
```

Operator analysis resolves through traits or primitive capability policy. It
must not grow operator-specific primitive name tables in expression analysis.

Static constraints include:

```rust
pub enum GenericConstraint {
    Trait {
        subject: InterfaceType,
        trait_id: TraitId,
        arguments: Vec<InterfaceType>,
    },

    SameType {
        left: InterfaceType,
        right: InterfaceType,
    },

    ConstRelation(ConstConstraint),
    ShapeRelation(ShapeConstraint),
    Capability(CompileCapabilityConstraint),
}
```

General `with` predicates that cannot be proven statically remain explicit
contract predicates rather than being silently treated as type constraints.

## XXI, FFI, and ABI Flow

External-language integration has three separate stages.

```text
source declaration
        |
        v
XXI
        |
        v
FFI contract
        |
        v
ABI contract
        |
        v
lowered external call
```

### XXI

XXI owns the user-facing external-language model.

Example:

```sev
import c from xxi

@c
def external_function(...)
```

XXI resolves:

* external language
* symbol identity
* source-level external attributes
* language-specific declaration conventions

XXI does not choose register layouts or emit calls.

### FFI

FFI validates the semantic boundary:

* which Severian types may cross
* ownership transfer
* borrowing
* mutability
* nullability
* error behavior
* marshalling requirements
* resource lifetime

FFI operates on resolved `TypeId` and declaration identities.

### ABI

ABI determines concrete machine representation:

* calling convention
* parameter placement
* return placement
* aggregate layout
* alignment
* padding
* symbol mangling
* platform-specific conventions

ABI must not perform source type inference.

## External Symbol Model

```rust
pub struct ExternalSymbolContract {
    pub id: ExternalSymbolId,
    pub declaration: DeclarationId,
    pub language: ExternalLanguageId,
    pub symbol: ExternalSymbolName,
    pub function: InterfaceFunctionType,
    pub ffi: FfiContract,
    pub abi: AbiSelector,
}
```

HIR stores `ExternalSymbolId`.

MIR stores a normalized external call.

Lowering asks ABI for the concrete call representation.

No downstream stage reparses decorators or external-language source syntax.

## MIR Contract

MIR is execution-oriented and fully typed.

MIR operations contain resolved `TypeId` values and stable declaration,
implementation, CompileType, or external symbol identities.

Conceptually:

```rust
pub struct MirValue {
    pub id: MirValueId,
    pub type_id: TypeId,
}

pub enum MirOperation {
    Call {
        declaration: DeclarationId,
        substitution: GenericSubstitution<TypeId>,
        arguments: Vec<MirValueId>,
        result: Option<MirValueId>,
    },

    Compile {
        handler: CompileHandlerId,
        operation: CompileOperationId,
        arguments: Vec<MirValueId>,
        result: Option<MirValueId>,
    },

    ExternalCall {
        symbol: ExternalSymbolId,
        arguments: Vec<MirValueId>,
        result: Option<MirValueId>,
    },
}
```

MIR must not:

* resolve source names
* infer generic arguments
* inspect `"Tensor"`, `"i32"`, or other source spellings
* recreate interface declarations
* determine package dependencies

## Lowering Contract

Lowering consumes resolved MIR.

It routes operations by identity and capability:

```text
ordinary MIR operation
    -> ordinary lowering

CompileHandlerId
    -> registered CompileType lowering

ExternalSymbolId
    -> FFI marshalling
    -> ABI lowering
```

Lowering owns backend conversion.

It does not own:

* source type compatibility
* primitive arithmetic policy
* generic constraint solving
* package interfaces
* trait selection

Backend mappings such as:

```text
32-bit declared floating representation
    -> MLIR f32
```

belong here or in the backend adapter.

## Backend Contract

A backend reports capabilities and accepts a backend-neutral lowered module.

Backends may include:

* MLIR/LLVM native compilation
* StableHLO/XLA tensor compilation
* Triton kernels
* platform-specific system backends

The backend boundary must allow one program to use more than one lowering path.

XLA is an optional tensor backend, not the entire language backend.

## Driver Contract

The driver orchestrates one compilation session.

Input:

```rust
pub struct CompilationRequest {
    pub package: ResolvedPackageInput,
    pub interfaces: Vec<PackageInterface>,
    pub sources: Vec<SourceInput>,
    pub target: TargetConfiguration,
    pub emit: EmitRequest,
}
```

Output:

```rust
pub struct CompilationOutput {
    pub diagnostics: Vec<Diagnostic>,
    pub interfaces: Vec<PackageInterface>,
    pub artifacts: Vec<CompilerArtifact>,
}
```

The driver sequences compiler stages. It does not implement their policies.

## Diagnostic Contract

All stages emit structured diagnostics:

```rust
pub struct Diagnostic {
    pub code: DiagnosticCode,
    pub severity: Severity,
    pub message: String,
    pub primary: SourceLabel,
    pub related: Vec<SourceLabel>,
    pub notes: Vec<String>,
}
```

Diagnostics may display source names such as `i32` or `Tensor`.

Diagnostic formatting is not semantic type identity.

## Dependency Direction

The compiler dependency graph must remain acyclic.

```text
source
  ↑
diagnostics

interface
  ↑
semantic
  ↑
HIR
  ↑
ownership
  ↑
MIR
  ↑
lowering
  ↑
backend

XXI
  ↓
FFI
  ↓
ABI
  ↓
lowering

driver
  ↓
all compiler stage APIs

tools
  ↓
driver
```

More precisely:

* interface must not depend on semantic, HIR, MIR, or lowering
* AST must not depend on semantic or HIR
* HIR must not own primitive, interface, FFI, or ABI policy
* MIR must not depend on AST or parser
* lowering must not depend on parser or source syntax
* compiler crates must not depend on repository-level tools

## Prohibited Architecture

The following patterns are not allowed:

```rust
ValueType::Int
ValueType::Float
ValueType::TensorAny
TypeKind::List
TypeKind::Map
TypeKind::Set
```

when they duplicate declaration-backed type identity.

Also prohibited:

```rust
if type_name == "Tensor" { ... }
if type_name == "f32" { ... }
```

outside parsing, interface decoding, diagnostics, or explicit registry setup.

Do not:

* maintain parallel old and new type systems
* convert an exact type to `Any` for compatibility
* store generic applications as strings
* reconstruct declaration IDs from display names after registration
* define language primitives inside HIR
* let MIR redo semantic analysis
* let lowering select source-level trait implementations
* fabricate core declarations when bootstrap fails
* introduce adapters without a named deletion point

## First Vertical Slices

The rebuilt compiler should proceed through complete vertical slices rather than
creating every abstraction in advance.

### Primitive identity

```sev
x: i32 = 1
```

Must prove:

```text
core.primitives.i32 declaration
    -> PrimitiveId
    -> TypeId
    -> HIR expression
    -> MIR value
    -> backend representation
```

The annotation and literal must resolve to the same type identity.

### Generic identity

```sev
def identity[T](value: T) -> T:
    return value

x: i32 = identity(1)
```

Must prove:

```text
T -> i32 substitution
    -> HIR call
    -> MIR instantiation
    -> i32 result
```

### Collection identity

```sev
values: List[i32] = [1, 2, 3]
```

Must prove that `List[i32]` is an applied library declaration, not a compiler
builtin.

### CompileType identity

```sev
x: Tensor[f16, Shape[2, 4]]
```

Must prove:

```text
applied declaration
    -> generic substitution
    -> ResolvedCompileType
    -> HIR
    -> MIR Compile operation
    -> registered lowering handler
```

No stage may match the source name `Tensor`.

### External identity

```sev
import c from xxi

@c
def external_add(left: i32, right: i32) -> i32
```

Must prove:

```text
XXI declaration
    -> FFI validation
    -> ExternalSymbolId
    -> MIR external call
    -> ABI lowering
```

## Completion Criteria

The compiler architecture is in its intended shape when:

1. Every source type resolves through one declaration-backed type model.
2. Generic applications retain structured arguments and substitutions.
3. CompileType is attached metadata, not a competing type system.
4. HIR stores resolved identities and no language policy.
5. MIR consumes HIR metadata without inspecting source names.
6. Lowering dispatches through operation, handler, and symbol identities.
7. Primitive declarations come only from `core.primitives`.
8. Collections are ordinary library-defined applied types.
9. XXI, FFI, and ABI have separate contracts.
10. Package and build tools interact only through compiler interfaces and the
    driver.
11. Adding `f128` does not require edits to unrelated expression, call, HIR, or
    MIR code.
12. Adding a new CompileType does not require adding a new compiler-wide enum
    variant.
