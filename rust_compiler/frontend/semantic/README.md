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
