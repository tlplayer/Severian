# Generic compilation gates

Run from the checkout root after building the Rust seed:

```sh
cargo build -p severian-driver --bin sev
(cd sev_compiler && ../target/debug/sev build)
SEVERIAN_SKIP_BUILD=1 python3 tests/sev_compiler/generics.py
```

The runner executes native programs and separately checks rejection diagnostics.
It loads the real checkout prelude with `--sysroot` while running the source
compiler outside the checkout.

Implemented and covered:

- Direct enumerative macros (`-> int_lists[T: int]():`) bind `T` in their bodies.
  Multiple parameters can enumerate the same family. Duplicate bindings and
  unknown families are errors. The older `with` spelling remains compatible.
- Source generic records specialize fields and ordinary methods, including
  `Self`, aliases, nested records, and recursive function argument inference.
  `check`, `test`, and `build` share the subject initializer namespace.
  Specializations retain declaration identity and ordered type bindings in the
  shared definition environment. Recursive inline storage, incorrect arity,
  conflicting inference, and owning string fields are rejected.
- The seed infers constructor-local type and `usize` dimension parameters after
  class substitution. Member-local parameter names shadow enclosing names.
  Array dimensions stay symbolic until the constructor is selected for a call.
- Seed ownership validation includes parameter types alongside local bindings;
  parameter array writes still respect active slice loans. Missing binding
  metadata is diagnosed instead of panicking.
- Aggregate boxing uses LLVM's target layout, installed before LLVM translation,
  and the native payload header preserves scalar alignment. The native regression
  checks trailing fields after `u128` identities stored in lists.
- A canonical `list[i32]` executes append, length, and indexed reads through the
  seed. This is distinct from the source prelude's `buffer`-based list protocols.

Still required to complete the broader migration:

- General constant-generic class fields in both compilers (beyond the seed array
  specialization), source constant-generic syntax, structural interner integration,
  generic methods/operators and record trait implementations, borrowed receivers,
  and complete capability checking
  of generic class bodies. Current record specialization is a type-argument subset.
- General seed constructor overload resolution and whole-body lowering. In
  particular, `list[i32](array[i32, 3](...))` passes dimension inference but still
  fails on its local `requested` binding: the seed extracts field initializers
  instead of executing the complete constructor body.
- Arbitrary owned collection elements, initialized-element tracking, move/drop
  behavior during growth and removal, and loans that prevent invalidating mutation.
- Migration of compiler consumers to the canonical collection implementations.

The new focused semantic/ownership tests and native gates do not imply completion
of these remaining stages. The broader semantic suite currently has ten failures
that also reproduce at the unchanged baseline; run with `RUST_MIN_STACK=33554432`
to avoid its existing test-thread stack overflow.
