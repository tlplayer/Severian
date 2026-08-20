# Core primitives

This package is the source-level definition of Severian primitive types. It is compiler input, not a second Rust implementation of the language.

## Contents

The package declares primitive identities, literal defaults, typed representations, and full operator signatures in `.sev` source.

Examples of information declared here:

- Stable qualified identity.
- Primitive category.
- Fixed-width, pointer-width, float, text, bytes, absence, or unit representation.
- Whether a literal kind defaults to the primitive.
- Operator parameter and result types.
- Constraints inherited from numeric or other traits.

Operator metadata must retain signatures. A set containing only `"+"` is insufficient because operand types, result types, constraints, and coercions are part of the contract.

## Compiler access

```text
this package
  -> compiler/bootstrap
  -> UniversalContext
  -> all remaining compiler phases by reference
```

No semantic, lowering, MIR, backend, or interface crate reads this directory directly.

## Prohibited implementation

This package must not contain a Rust parser that scans `.sev` text with string operations. The real Severian lexer and parser are the only source parser.

The end state is a source-only Severian package, not a Rust workspace crate. Any transitional Rust loader is frozen and deleted after `compiler/bootstrap` supplies the same functionality.

## Compatibility tests

- Reordering declarations preserves stable IDs.
- Formatting and comments do not change definitions.
- Exactly one default exists per literal kind that requires a default.
- Operator signatures resolve through universal tests.
- Every declared representation can be lowered or produces a target-specific unsupported error.
