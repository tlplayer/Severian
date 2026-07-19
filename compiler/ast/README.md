# Severian AST

The AST models source syntax after parsing and before semantic analysis. It keeps
the shape of the user's program intact: names, spans, indentation-derived blocks,
patterns, explicit type annotations, and statements such as `?=`, plus
expressions such as `async`, `await`, `.borrow()`, `.clone()`, and `.move()`.

## Principles

- Every AST node that can produce diagnostics carries a `Span`.
- The AST records syntax, not inferred meaning. Ownership, lifetimes, overload
  resolution, and type inference belong in later compiler phases.
- Valued declarations use one concrete prefix type, such as `int count = 0`.
  Uninitialized fields and parameters use `name: Type`; parameters can accept
  alternatives with union types such as `value: string | int | float`.
- Decorators such as `@math(X)` are recorded on declarations so later phases can
  opt functions into domain-specific symbol packs, syntax, and checks.
- Python-like syntax should remain visible as blocks, declarations, calls,
  members, and collection literals.
- Rust-like safety hooks are represented explicitly through result types,
  patterns, unsafe blocks, and ownership operations.
- Concurrent calls are explicit: ordinary calls block, `async` starts work
  without blocking, and `await` joins a task handle.

## Current Coverage

- Modules, `import`, `from ... import ...`, functions, classes, constructors,
  traits, fields, and trait methods.
- Statements for stable `=` bindings, changeable `:=` bindings, assignment,
  assertions, returns, loops, `while condition with setup` clauses, switches,
  unsafe blocks, break, continue, and expression statements.
- Function and constructor declarations can carry attached `test:` blocks.
- Expressions for literals, identifiers, calls, members, collections, indexing,
  conditionals, switches, lambdas, math operators, concurrency, and ownership.
  `name ?= expression` is represented as a try-bind statement for error
  propagation; the binding name is required.
- Patterns for wildcard, literals, identifiers, tuples, lists, constructors, and
  alternatives.
- Types for named paths, collections, functions, results, options, futures, and
  references.

## Validation

Until this directory becomes a Cargo crate, the node definitions can be checked
directly:

```sh
rustc --crate-type lib compiler/ast/nodes.rs -o /tmp/severian_ast_nodes.rlib
```
