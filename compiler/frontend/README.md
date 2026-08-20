# Frontend

The frontend converts source syntax into typed program meaning. It does not own the language-wide type, literal, or operator catalog.

## Responsibilities

```text
source -> lexer -> parser -> AST -> semantic -> HIR -> ownership
```

- Lexer: token boundaries, escapes, and source spans.
- Parser: syntax and AST construction.
- AST: what the user wrote, including raw literal spelling where needed.
- Semantic: names, constraints, type/operator/literal resolution through `UniversalContext`.
- HIR: typed expressions and declarations referencing universal IDs.
- Ownership: ownership and effect validation over typed HIR.

## Representation rule

Different representations are justified only by different invariants:

- AST may preserve `0xFF`, suffixes, quoting, and source spans.
- Universal `ConstantValue` stores the canonical value after validation.
- HIR and MIR should reference the universal operator and canonical constant types unless they require additional stage-specific information.

If two enums convert one-to-one without loss and neither has a stage-specific invariant, they should not both exist.

## Forbidden frontend knowledge

Frontend code must not:

- Build a primitive catalog.
- Maintain a list such as `integer`, `float`, `boolean`, `text`, `absence`, or `unit`.
- Check `primitive.supports("+")` directly.
- Match primitive representations.
- Reload `library/core/primitives`.
- Assign primitive IDs from declaration order.

The frontend asks `UniversalContext` to resolve a literal or operator and translates the returned typed error into a source diagnostic.
