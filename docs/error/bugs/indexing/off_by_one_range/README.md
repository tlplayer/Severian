# Off-by-one ranges

Ranges state endpoint ownership in the delimiters: `[a:b]` is inclusive,
`[a:b)` excludes the end, `(a:b]` excludes the start, and `(a:b)` excludes both.
Index-based collection iteration uses `indices(values)` so the collection owns
the bounds.

Mitigation: **compile-time** when bounds are proved and **runtime** otherwise.
