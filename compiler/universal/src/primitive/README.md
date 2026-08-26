# Universal primitives

This module is the sole owner of Severian primitive identities, categories,
representations, literal defaults, capabilities, and operator signatures.

Primitive facts are compiler axioms. Bootstrap installs this catalog before it
loads extensible source protocols, so compiling the standard library never
participates in defining the types required to compile that library.

The catalog owns:

- Boolean, character, integer, floating-point, text, bytes, absence, unit,
  argument, error, and measured primitive definitions.
- Fixed, machine, pointer, and floating-point representations.
- Literal-default selection.
- Primitive capabilities and complete operator signatures.
- Stable identities rooted at `universal.primitive`.

Higher-level numeric, string, collection, formatting, parsing, and conversion
algorithms belong in `library/core`; they consume this schema but do not define
it.

Semantic compatibility remains structural. It derives widening and literal
compatibility from category, signedness, width, and representation rather than
reconstructing primitive rules from names. HIR, MIR, and lowering preserve the
resolved universal IDs.
