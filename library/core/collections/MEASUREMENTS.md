# Collection compiler measurements

Generated with `python3 tests/sev_compiler/collection_capabilities.py --record-only`.

All 12 algorithm-adapter tests pass separately. They do not establish native support.

| File | seed_parse | seed_check | seed_tests | source_check | source_tests | agent_ir | native |
| --- | --- | --- | --- | --- | --- | --- | --- |
| btree.sev | pass | pass | fail | fail | fail | fail | blocked |
| count.sev | pass | pass | fail | fail | fail | fail | blocked |
| deque.sev | pass | pass | fail | fail | fail | fail | blocked |
| dict.sev | pass | pass | fail | fail | fail | fail | blocked |
| heap.sev | pass | pass | fail | fail | fail | fail | blocked |
| list.sev | pass | pass | fail | fail | fail | fail | blocked |
| map.sev | pass | pass | pass | fail | fail | fail | blocked |
| set.sev | pass | fail | fail | fail | fail | fail | blocked |
| storage.sev | pass | pass | pass | fail | fail | fail | blocked |
| vector.sev | pass | pass | fail | fail | fail | fail | blocked |

`seed_tests` uses the seed to compile and execute tests; `native` is execution
through the source compiler. The storage seed run passes its overflow/growth
test and 22 imported scalar tests. The map-only seed pass is not concrete
collection coverage. Full commands, source/compiler hashes, parsed method
inventories and unmodified diagnostics are in
`sev_compiler/target/collection-ledger/results.json` and adjacent artifacts.

Representative first diagnostics:

- Source checking: `error: expected macro identifier`.
- `btree.sev`: `seed_tests`: E000204: array length must be a compile-time `usize` value.
- `count.sev`: `seed_tests`: E000204: array length must be a compile-time `usize` value.
- `deque.sev`: `seed_tests`: E000201: unknown binding `physical`.
- `dict.sev`: `seed_tests`: E000204: array length must be a compile-time `usize` value.
- `heap.sev`: `seed_tests`: E000204: array length must be a compile-time `usize` value.
- `list.sev`: `seed_tests`: seed ownership validator panicked: no entry found for key (exit 101).
- `set.sev`: `seed_check`: error: E000211: class method is unknown or cannot be used as a value; `seed_tests`: E000204: array length must be a compile-time `usize` value.
- `vector.sev`: `seed_tests`: E000204: this source type form is not yet supported by universal resolution.

The compiler package Agent IR build succeeds with
`../target/debug/sev build --emit agent-ir` from `sev_compiler`.
It still uses the existing primitive buffer provider. Canonical collection
native execution, owned-element cleanup and compiler migration remain open.
