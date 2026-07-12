# Sevarian AST

The AST models source syntax after parsing and before semantic analysis. It keeps
the shape of the user's program intact: names, spans, indentation-derived blocks,
patterns, explicit type annotations, and expressions such as `spawn`, `await`,
`?`, `.borrow()`, `.clone()`, and `.move()`.

## Principles

- Every AST node that can produce diagnostics carries a `Span`.
- The AST records syntax, not inferred meaning. Ownership, lifetimes, overload
  resolution, and type inference belong in later compiler phases.
- Type annotations use `name: Type` surface syntax for bindings, parameters,
  fields, constants, and extern declarations.
- Python-like syntax should remain visible as blocks, declarations, calls,
  members, and collection literals.
- Rust-like safety hooks are represented explicitly through result types,
  patterns, unsafe blocks, and ownership operations.
- Go-like concurrency starts as simple `spawn` and `await` expressions.

## Current Coverage

- Modules, `import`, `from ... import ...`, functions, classes, traits, fields,
  and trait methods.
- Statements for stable `=` bindings, changeable `:=` bindings, assignment,
  returns, loops, `while condition with setup` clauses, matches, unsafe blocks,
  break, continue, and expression statements.
- Expressions for literals, identifiers, calls, members, collections, indexing,
  conditionals, matches, lambdas, concurrency, ownership, and error propagation.
- Patterns for wildcard, literals, identifiers, tuples, lists, constructors, and
  alternatives.
- Types for named paths, collections, functions, results, options, futures, and
  references.

## Validation

Until this directory becomes a Cargo crate, the node definitions can be checked
directly:

```sh
rustc --crate-type lib compiler/ast/nodes.rs -o /tmp/sevarian_ast_nodes.rlib
```
