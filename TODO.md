# Bug Prevention TODO

Work through one fixture at a time. Remove its directory only after the language rule and compiler test replace it.

## 1. Ignored error

Fixture: `docs/examples/bugs/errors/ignored_error/invalid.sev`

- [ ] Decide how `Result` values must be consumed.
- [x] Propagate a `Result[unit, exception]` with a direct return.
- [ ] Specify and test the compiler diagnostic.
- [ ] Move the rule into the error language documentation.
- [ ] Remove `docs/examples/bugs/errors/ignored_error`.

End result: every `Result` is handled, propagated, or explicitly discarded.

## 2. Integer overflow

Fixture: `docs/examples/bugs/numeric/integer_overflow/invalid.sev`

- [ ] Decide integer literal coercion and promotion rules.
- [ ] Decide default overflow behavior.
- [ ] Define checked, wrapping, saturating, and overflow-reporting operations.
- [ ] Specify and test compile-time and runtime behavior.
- [ ] Move the rule into the numeric language documentation.
- [ ] Remove `docs/examples/bugs/numeric/integer_overflow`.

End result: integer overflow never wraps silently.

## 3. Array out of bounds

Fixture: `docs/examples/bugs/indexing/array_out_of_bounds/invalid.sev`

- [ ] Distinguish fixed arrays from dynamic lists.
- [ ] Define checked indexing and optional lookup.
- [ ] Specify and test static and runtime bounds checks.
- [ ] Move the rule into the collection language documentation.
- [ ] Remove `docs/examples/bugs/indexing/array_out_of_bounds`.

End result: provably invalid indices are rejected and dynamic indices are checked.

## 4. Off-by-one range

Fixture: `docs/examples/bugs/indexing/off_by_one_range/invalid.sev`

- [ ] Choose the final exclusive and inclusive range syntax.
- [ ] Define collection-derived index iteration.
- [ ] Specify and test invalid collection-bound ranges.
- [ ] Move the rule into the range language documentation.
- [ ] Remove `docs/examples/bugs/indexing/off_by_one_range`.

End result: the normal collection index range cannot include `len()`.

## 5. Use after move

Fixture: `docs/examples/bugs/ownership/use_after_move/invalid.sev`

- [ ] Define borrowed, mutable, and consuming parameter modes.
- [ ] Decide when moves are inferred and when they are explicit.
- [ ] Specify and test moved-binding diagnostics.
- [ ] Move the rule into the ownership language documentation.
- [ ] Remove `docs/examples/bugs/ownership/use_after_move`.

End result: a moved value cannot be used unless it is reassigned.

## 6. Aliasing mutation

Fixture: `docs/examples/bugs/ownership/aliasing_mutation/invalid.sev`

- [ ] Define shared and exclusive borrow rules.
- [ ] Define when a borrow ends.
- [ ] Specify and test conflicting-mutation diagnostics.
- [ ] Move the rule into the ownership language documentation.
- [ ] Remove `docs/examples/bugs/ownership/aliasing_mutation`.

End result: mutation cannot overlap a live conflicting borrow.

## 7. Data race

Fixture: `docs/examples/bugs/threads/data_race/invalid.sev`

- [ ] Define task capture access rules.
- [ ] Define which types are safe to share across tasks.
- [ ] Define standard atomic and mutex capabilities.
- [ ] Specify and test shared-mutation diagnostics.
- [ ] Merge the rule into ownership and concurrency documentation.
- [ ] Remove `docs/examples/bugs/threads/data_race`.

End result: concurrent mutable state is uniquely owned or synchronized.
