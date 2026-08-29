# Core libraries

Stable IDs: `library.core*` and `library.collections`.

Core contains the prelude-adjacent foundations: bits, collections, compile
protocols, errors, hashing, math, memory, randomness, regex, size, text, and
time. The prelude may re-export selected names; that does not make the prelude
their semantic owner. The package declaring a symbol owns its behavior, while
Universal owns primitive identity and representation.

Important relationships:

- `core.compile` defines ordinary source protocols; Rust bootstrap only loads
  them into stable compiler identities.
- `core.memory` and ABI pointer/storage views are different levels. Memory APIs
  manipulate values; ABI descriptors cross external boundaries.
- `core.size` and measured primitives express units; tensor dimensions are
  dimension expressions and are not byte sizes.
- `core.random` is observable nondeterminism and must not influence compiler
  artifact ordering unless a seed is explicit.

Per-symbol export validation is complete for collections and remains an audit
item for the other core packages.
