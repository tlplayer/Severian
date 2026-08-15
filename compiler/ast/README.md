# Severian AST

The AST models source syntax after parsing and before semantic analysis. It keeps
the shape of the user's program intact: names, spans, indentation-derived blocks,
patterns, explicit type annotations, and statements such as `?=`, plus
expressions such as `async`, `await`, channel creation and sending, `view`,
`borrow`, `clone`, and `move`.

## Principles

- Every AST node that can produce diagnostics carries a `Span`.
- The AST records syntax, not inferred meaning. Ownership, lifetimes, overload
  resolution, and type inference belong in later compiler phases.
- Valued declarations use one concrete prefix type, such as `int count = 0`.
  Uninitialized fields and parameters use `name: Type`; parameters can accept
  alternatives with union types such as `value: string | int | float`.
- Decorators such as `@math(X, *)` are recorded as a package path plus an
  explicit symbol pack; named policies such as `@tensor(backend = auto)` retain
  their key and value separately. Traits may declare decorators to own a
  semantic namespace. Activation and provider resolution belong to HIR.
- Python-like syntax should remain visible as blocks, declarations, calls,
  members, and collection literals.
- Rust-like safety hooks are represented explicitly through result types,
  patterns, unsafe blocks, and prefix-keyword ownership operations.
- Concurrent calls are explicit: ordinary calls block, `async` starts work
  without blocking, and `await` joins a task handle.

## Current Coverage

- Modules, `import`, `from ... import ...`, functions, classes, constructors,
  traits, fields, trait methods, operator contracts, trait-owned semantic
  decorators, and direct trait composition requirements in either body or
  `Trait: First + Second` header form.
- Statements for stable `=` bindings, changeable `:=` bindings, safe `?=`
  result capture, assignment,
  assertions, returns, loops, `while condition with setup` clauses, ordinary
  switches, repeating multi-channel switches, unsafe blocks, break, continue,
  and expression statements.
- Function and constructor declarations can carry attached `test:` blocks.
- Expressions for literals, identifiers, calls, members, collections, indexing,
  conditionals, switches, lambdas, math operators, concurrency, and ownership.
  A `Result` on the right of `=` or `:=` is propagated by semantic lowering;
  `name ?= expression` remains a distinct statement that stores the complete
  result for local handling.
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
