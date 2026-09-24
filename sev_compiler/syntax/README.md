# Syntax contracts

This dependency-free package owns source coordinates, lexemes, token roles,
primitive/type contracts, operator syntax, sentence composition and block origin
contracts. `S` means Sentence; write `Shape` in full.

`source` owns text loading and source maps. `primitives` implements the primitive
contracts and supplies literal recognition/defaults. Compiler passes consume
these definitions; syntax does not import the compiler, CFG or diagnostics.

Some old universal and frontend files remain as compatibility exports. See
[the migration status](../PIPELINE_MIGRATION.md) for the remaining model split.
