# storage

`storage` is a backend-neutral logical-plan layer. Relational scans, document
operations, key/value access, and Dynamo-style partition queries use the same
inspectable `StoragePlan` representation. Transactions and migrations are
validated before an adapter receives a plan.

The package does not pretend every data model is relational. Adapters preserve
model-specific operations while sharing validation, inspection, transaction,
and migration infrastructure.
