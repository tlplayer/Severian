# Off-by-one ranges

Ranges state endpoint ownership in the delimiters: `[a:b]` is inclusive,
`[a:b)` excludes the end, `(a:b]` excludes the start, and `(a:b)` excludes both.
Iteration over collection indices normally uses `[0:length)`; it is the same
half-open interval used by binary-search bounds.

Mitigation: **compile-time** when bounds are proved and **runtime** otherwise.
