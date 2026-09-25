# Text formatting

`src/format.sev` forwards to the source formatting implementation in
[`core.string.format`](../string/format/src/lib.sev), used by the native
prelude. Integer formatting allocates one output buffer and fills its digits in
place. Float formatting reuses a format buffer and output buffer while finding
the shortest significant-digit precision that round-trips. Printing iterates
over the resulting bytes without recursion.

`quoted(text)` escapes JSON string content. `object_text((name, value), ...)`
uses one append builder for typed fields, including strings, integers, booleans,
floats, characters, and None. Nonfinite numbers are rejected for JSON output.
A class's string conversion can call this builder; `string(value)` and
`print(value)` then use the same conversion. This is an explicit conversion
protocol, not automatic reflection over arbitrary class fields.

Tests live in `tests/format.sev` and `tests/profile.sev`. Profile budgets cover
1000 signed integer conversions and 100 full-precision float conversions; results
and allocation measurements are written under `package.pkg/debug/profile/` by
`test/validation/performance/libraries.sh`.
