# Bug-Mitigation Fixtures

These fixtures specify how Sevarian prevents common C and Rust footguns while
keeping CPU, GPU, and task-parallel work explicit and optimizable. They are
compiler contracts, not runnable programs as a set: each bug directory contains
an intentionally rejected `invalid.sev`, a runnable `fixed.sev`, and the
diagnostic text the compiler must eventually produce.

## Fixture contract

`expected-error.txt` is deliberately small and stable:

- line 1 is the diagnostic code;
- line 2 is the exact `file:line:column` start of the highlighted span;
- line 3 is the highlighted source spelling;
- line 4 is the invariant being enforced.

The checker validates the first three fields once a `sev` driver exists. The
invariant line is human-facing documentation and must remain one sentence.

## First safety baseline

| Area | Fixture | Mitigation |
| --- | --- | --- |
| Ownership | `use_after_move` | compile-time |
| Ownership | `aliasing_mutation` | compile-time |
| Indexing | `array_out_of_bounds` | compile-time and runtime |
| Indexing | `off_by_one_range` | compile-time and runtime |
| Numeric | `integer_overflow` | runtime by default; explicit library operations |
| Threads | `data_race` | compile-time |
| Threads | `async_mutation_without_lock` | compile-time |
| Errors | `ignored_error` | compile-time |

The syntax in these fixtures is a language commitment. If implementation needs
to revise a spelling, update the invalid source, fix, expected diagnostic, and
language reference in the same change.
