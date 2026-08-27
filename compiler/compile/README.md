# CompileType routing

`severian-compile` is the opt-in escape hatch between typed MIR and the shared
MLIR/target pipeline. It does not own type semantics and it does not introduce
specialized type or compiler names.

Universal records one route for each semantic type:

```text
CompileRoute::Standard
CompileRoute::Compiler(CompilerId)
```

`CompilerId` is derived from stable declaration identity. A compiler route can
only reference a declaration already registered in the universal type context.

The planner assigns each MIR operation by collecting the routes of all operand
and result types. No custom routes means standard lowering, one compiler means
that compiler, and multiple compiler IDs are an error. Adjacent operations with
the same route form a region.

The bootstrap implementation intentionally supports only straight-line
regions. Every custom region has explicit typed inputs and outputs and an
explicit effect summary. Arbitrary control-flow edges cannot cross a custom
region boundary. MIR produced by the planner is not valid planner input.

A registered `CompileHandler` receives the typed region and neutral
`TargetSpec`, and returns one `MlirArtifact`. The registry verifies handler
identity, region arity, the declared MLIR signature, operation validity, and
allowed target dialects through MLIR's parser and verifier. The custom
operations are then replaced by a typed `ArtifactCall`, ordinary MIR resumes
through LIR and MLIR, and the driver composes verified custom functions through
MLIR's symbol table. Handlers provide only local entry symbols; composition
assigns final artifact symbols centrally from `ArtifactId`.

`severian-compile` does not depend on LIR or ordinary lowering. The driver owns
the fork and join between custom compilation and the standard lowering path.
`TypeContextBuilder` is the only place routes can be registered; the resulting
`TypeContext` exposes read-only routing, including inheritance from a generic
constructor to its instantiated types.

Every source compilation is planned. A plan containing only standard segments
uses the direct LIR backend path. A plan containing a custom segment is
compiled, resumed, lowered, composed, and sent through the MLIR/LLVM target
toolchain. The C emitter continues to reject `ArtifactCall`; it never infers or
implements CompileType behavior.

CompileType is orthogonal to external-call boundaries. ABI and FFI are not
CompileType phases and normal compilation does not run through them.

Source declarations make that separation explicit: `@c`/`@rust` select a
foreign ABI boundary, while `@mlir`, `@xla`, and
`@compile(mlir, stablehlo, xla)` select compiler lowering targets. Compile
policy attributes are carried on CompileOps and are never interpreted by XXI.
