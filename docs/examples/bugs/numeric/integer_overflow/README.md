# Integer overflow

Arithmetic is checked by default: constants that cannot fit are rejected and
dynamic overflow produces `OverflowError`. Wrapping, saturating, and
overflow-reporting behavior must be selected explicitly.

Mitigation: **compile-time** for constant expressions, **runtime** for dynamic
checked arithmetic, and **library-level** for explicit alternatives.
