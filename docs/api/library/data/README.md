# Data libraries

Stable IDs: `library.data`, `library.data.csv`, `library.data.database`,
`library.data.json`, `library.data.pql`, `library.data.sql`,
`library.data.storage`, and `library.data.yaml`.

Format packages translate bytes/text and structured values. `database` owns
connection/query abstractions; SQL and PQL own query syntax/builders; `storage`
owns persistence abstractions. A parser returning a value does not transfer
ownership of an underlying external connection unless its signature says so.

Errors must separate malformed input, schema/type mismatch, transport failure,
and storage failure. The current package catalogue proves every manifest has a
stable API ID, but exact exported-symbol and error taxonomies remain partial.
