# database

`database` is the executable database layer. It owns real SQLite connections,
DDL and mutation execution, iterable query rows, persistence, transactions, and
a TCP loopback database server with client connections. SQL errors are returned
as `DatabaseError` failures rather than represented as plan strings.

PQL validates relational query structure before SQL reaches this package.
`storage` remains the provider-neutral plan layer used by SQL, document,
key/value, Dynamo-style, and object-storage adapters; it is not the database
implementation.
