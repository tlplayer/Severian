# Semantic analysis

Semantic analysis connects parsed syntax to the universal language model.

## Inputs and output

```text
analyze_package(module_graph, universal_context) -> TypedProgram | Diagnostic
```

`TypedProgram` contains the package-wide `ProgramIndex` and HIR. Analysis is
two-phase: every module-level declaration receives a stable `DefId` and import
scopes are resolved first; function bodies are then checked against that
complete namespace. The single-AST `analyze` function remains a convenience
for isolated compiler tests and in-memory fragments.

Generic declarations are indexed without compiling their bodies eagerly. A
reachable concrete call creates the body specialization used by HIR/MIR. The
bootstrap currently permits one concrete specialization per generic
definition and diagnoses a second distinct instance explicitly; representing
multiple instances is the next `InstanceId`-level extension, not an import or
name-resolution rewrite.

The analyzer owns:

- Lexical and module scopes.
- Name resolution.
- Binding identities.
- Expected-type propagation.
- Applying universal resolution results to HIR.
- Source diagnostics and spans.

The analyzer does not own:

- Primitive definitions.
- Literal default tables.
- Operator support tables.
- Numeric promotion or coercion rules.
- Physical representations.
- Target pointer width.

## Required calls

Literal analysis delegates to:

```rust
universal.resolve_literal(literal, expected_type)
```

Operator analysis delegates to:

```rust
universal.resolve_binary(operator, left_type, right_type)
```

Resolution must be symmetric. `literal + value` and `value + literal` are solved as one constraint problem rather than by concretizing the left operand first.

## Diagnostics

Universal returns typed errors. Semantic adds source context:

```text
UnknownType
NoLiteralDefault
InvalidLiteralForType
NoMatchingOperator
AmbiguousOperator
ConstraintFailure
```

Error codes and spans remain frontend responsibilities.

## Typed MLIR declarations

`mlir.rs` lowers bodyless `@mlir("dialect.operation")` declarations into typed
HIR operation bodies. They follow the ordinary MIR, LIR and MLIR pipeline,
including zero-result operations such as `cf.assert` and `vector.print`.
Named string arguments become MLIR attributes; dialect attributes beginning
with `#` retain their MLIR spelling.

`@mlir("func.call", callee="symbol")` binds a typed declaration to an MLIR
library symbol using `CallType::Mlir`. Registered library composition supplies
the implementation before native lowering. This does not create a C boundary.
`Compiler::compile_library_object` retains the composed `.mlir`, relocatable
`.o`, and `.symbols.json` in the package build output, alongside its shared
library. Unsupported lowering requirements are diagnosed explicitly.

## Automatic wildcard narrowing

Normal builds interpret `import * from X` as a request for the names actually
used by the importing module. `analyze_package_with_context` collects local
declarations, builds an `ImportPlan`, and installs that plan before type checking.
It does not construct the complete wildcard-expanded export index first.

`package/imports/uses.rs` walks the parsed AST, including interpolation,
annotations, constraints, defaults, and nested lexical scopes. The resolver in
`package/imports.rs` follows individual name requests through import edges;
its work list handles cycles, diamonds, overloads and namespace aliases. Files
and their initialization order remain in the graph. Trait registry namespaces
are tracked independently of imported callable stubs.
Bare enum-variant references retain their enum declaration through the same
import edges, including reexports.

Wildcard resolution always selects names used by the file, including types,
constraints, interpolations, test bodies, and downstream re-export requirements.
After all selected targets build successfully, the Rust CLI writes those names
back: `import a, b, * from "module.sev"`. The trailing wildcard remains extensible.
`language.explicit-imports = true` writes `import a, b from "module.sev"` instead;
the setting defaults to `false` and affects source spelling, not visibility.
`import * from "module.sev" as namespace` and qualified uses remain unchanged.

The optional `sev build --explicit-imports` command forces the closed form.
Ordinary `check` and in-process semantic analysis do not rewrite source. Edits
are limited to the build root, excluding generated files and nested dependency
packages. Imports with no named uses remain for initialization/extension effects;
internal comments are preserved. Source edits invalidate the previous byte-based
cache entry, so the next build may rebuild once with the normalized source.

`import_plan` exposes selections and counts for inspection. `import_index`
continues to provide the complete public surface for refactoring tools that
explicitly request it. Compare counts without compiling function bodies:

```sh
cargo run -p severian-driver --example import_counts -- \
  sev_compiler/package.json sev_compiler/sev_compiler/src/main.sev
```

`SEVERIAN_PROFILE_ACTIVE=1` also reports request, binding and export counts during
ordinary builds. Export entries measure symbol exposure; they are not a count
of parsed files or proof of a proportional wall-time improvement.

The source compiler's `frontend/modules/src/imports.sev` performs selection
before appending imported declarations to a flattened module. It retains
implementation dependencies, records, traits and operators. When an AST form's
implicit requirements are not covered by its visitor, it conservatively retains
the dependency instead of dropping potentially required declarations.
