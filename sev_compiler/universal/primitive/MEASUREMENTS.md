# Primitive measurement checkpoint

Observed on 2026-09-08 using the checkout seed and rebuilt source compiler.

Regenerate with `python3 tests/sev_compiler/primitives.py --snapshot tests/sev_compiler/fixtures/primitives/checkpoint.json`.

The [checkpoint JSON](../../../tests/sev_compiler/fixtures/primitives/checkpoint.json) preserves compiler/source hashes, exact commands, exits and diagnostics. See [CAPABILITIES.md](CAPABILITIES.md) for interpretation and outstanding dependencies. API counts are parsed declaration nodes, including requirements and metadata; they do not measure native operation coverage. A dash means seed parsing failed. Generated source definitions remain available in Agent IR.

| File | AST API nodes | Seed parse | Seed check | Source check | Source tests | Agent IR | Native |
| --- | --- | --- | --- | --- | --- | --- | --- |
| array.sev | 25 | pass | pass | fail | fail | fail | blocked |
| bool.sev | 49 | pass | pass | fail | fail | fail | blocked |
| char/encoding.sev | 5 | pass | pass | pass | pass | pass | pass |
| char/utf8.sev | 1 | pass | pass | pass | pass | pass | pass |
| char.sev | 23 | pass | pass | fail | fail | fail | blocked |
| collections.sev | — | fail | fail | pass | pass | pass | pass |
| float.sev | 170 | pass | pass | fail | fail | fail | blocked |
| int.sev | 218 | pass | pass | fail | fail | fail | blocked |
| numeric/conversion.sev | — | fail | fail | pass | pass | pass | pass |
| numeric/operators.sev | — | fail | fail | pass | pass | pass | pass |
| pointer.sev | 23 | pass | pass | fail | fail | fail | blocked |
| primitive.sev | 40 | pass | pass | fail | fail | fail | blocked |
| slice.sev | 27 | pass | pass | fail | fail | fail | blocked |
| string/core.sev | 10 | pass | fail | pass | pass | pass | pass |
| string/format.sev | 10 | pass | fail | pass | pass | pass | pass |
| string.sev | 57 | pass | fail | fail | fail | fail | blocked |

Seven of sixteen primitive files pass source checking, source test emission, Agent IR and native execution. The other nine retain visible failures. This does not complete P0 or establish coverage of all public operations.

| Capability fixture | Source check | First diagnostic |
| --- | --- | --- |
| constant_parameter | fail | error: E000120: expected closing subscript bracket |
| constructor_metadata | fail | error: E000120: expected an expression |
| conversion_relation | fail | error: E000120: expected operator parameters |
| documentation | pass | — |
| enum | fail | error: E000120: expected a type name |
| fallible_outcome | fail | error: E000200: unknown callable error |
| float_storage | fail | error: E000205: unknown type f32 |
| generic_character | fail | error: E000200: unknown callable .codepoint |
| generic_record | fail | error: E000200: generic record layout is not implemented |
| owned_field | fail | error: E000200: record string fields require aggregate ownership lowering |
| wide_integer | fail | error: E000205: unknown type i128 |

Validation: native compiler build; compiler Agent IR build; 117 bootstrap acceptance checks; eight primitive behavior tests (including exceptional arithmetic, invalid UTF-8, unchanged-binary source edits and specialization budget); eight source-contract tests; nine lexical-rule tests; six diagnostic-location checks; callable native/negative regressions.

The bootstrap runner had stale rejection checks for the truth-protocol diagnostic and module-level compound assignment. The diagnostic expectation now matches the unchanged truth checker; compound-assignment rejections run inside a function so they reach semantic checking. The former character-equality rejection is preserved as a positive native match test now that character comparisons have source implementations.
