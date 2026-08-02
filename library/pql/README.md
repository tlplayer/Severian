# PQL (Prequel)

PQL is Severian's typed prestructured query layer. A query is checked against a
schema before an adapter emits SQL or a deterministic fixture executes it.
Validation covers relation names, projected fields, filter parameter types,
grouping rules, join compatibility, nullability metadata, and result shape.

This baseline deliberately separates structural proof from database behavior.
It can reject invalid query structure without contacting a database; vendor
locking, indexes, query planners, and transaction isolation still require real
adapter integration tests.
