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
- Seed field-initializing constructors execute complete callable bodies: locals,
  branches, loops, assertions, method calls, recursive construction, and early
  returns. Both `field = value` and `field := value` initialize declared fields.
  Every successful return must initialize all fields; reads and escaping `self`
  before initialization are rejected. Loop bodies are conservatively allowed to
  execute zero times when checking initialization.
- Constructor overloads use parameter types and conversion ranks, prefer concrete
  overloads over equally ranked generic fallbacks, and diagnose ambiguity and
  missing matches. Named/default arguments retain source evaluation order, and
  supplied arguments execute once. Keyword conversion ranks are compared in
  source argument order even when overloads declare parameters in different orders.
- Seed constructors support explicit `Self` returns and `Self | Error` results,
  including bare returns, fallthrough, factory returns, `throw`, and explicit
  `return error(...)`. Successful explicit results retain field constraints and
  are evaluated once; error exits do not require complete initialization.
  Constraint failures retain typed error values through the catch boundary.
- Seed specialization discovery resolves explicit type arguments before using
  them as inference evidence. Constructor expected-type checking uses structural
  assignability and rejects substitution of a different nominal specialization.
- Seed ownership validation includes parameter types alongside local bindings;
  parameter array writes still respect active slice loans. Missing binding
  metadata is diagnosed instead of panicking.
- Aggregate boxing uses LLVM's target layout, installed before LLVM translation,
  and the native payload header preserves scalar alignment. The native regression
  checks trailing fields after `u128` identities stored in lists.
- A canonical `list[i32]` executes empty, array, and slice construction, append,
  growth, length, capacity, and indexed reads through the seed. Array constructor
  coverage includes dimensions zero, three, and six. This is distinct from the
  source prelude's `buffer`-based list protocols.
- Typed integer `min`/`max` preserve unsigned `usize` comparisons and evaluate
  operands once. Inlined index operators bind their own receiver fields.
  Embedded `throws(...)` tests do not change production function return types.

The seed constructor gates can also run independently after building `sev`:

```sh
python3 tests/sev_compiler/constructors.py
```

Still required to complete the broader migration:

- General constant-generic class fields in both compilers (beyond the seed array
  specialization), source constant-generic syntax, structural interner integration,
  generic methods/operators and record trait implementations, borrowed receivers,
  and complete capability checking
  of generic class bodies. Current record specialization is a type-argument subset.
- Source-compiler constructor bodies and fallible constructor execution, together
  with the general generic-body capability checking described above. Seed support
  does not establish these capabilities in the source compiler.
- Arbitrary owned collection elements, initialized-element tracking, move/drop
  behavior during growth and removal, and loans that prevent invalidating mutation.
- Migration of compiler consumers to the canonical collection implementations.

These gates do not imply completion of the remaining stages. Seven of the ten
baseline Rust semantic failures are fixed: multiline contracts, bodyless hook
signatures, union overload ranking, unresolved explicit type arguments, and the
two structural tensor constructor cases. The three remaining tests disagree with
the current integer-to-float conversion policy (`lossy` versus implicit
`promote`); that language-policy decision is pending. Run semantic tests with
`RUST_MIN_STACK=33554432` to avoid test-thread stack overflow.
