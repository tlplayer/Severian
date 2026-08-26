# storage

`storage` is a backend-neutral connection and logical-plan layer. Every read
returns the canonical `data.Data` type from `library/data/src`; storage does
not define a competing dataframe representation.

Provider packages implement the `storage.Storage` dispatch contract for their
locator scheme. Relational scans, document operations, and key/value access
remain distinct inspectable plans while sharing connection and transaction
infrastructure. The in-memory provider is available for tests and harness
context.
