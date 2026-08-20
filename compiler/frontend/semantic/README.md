# Semantic analysis

Semantic analysis connects parsed syntax to the universal language model.

## Inputs and output

```text
analyze(ast, universal_context) -> HIR | Diagnostic
```

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
