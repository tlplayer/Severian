# Data race

Tasks may run on CPU worker threads or accelerator submission threads; scheduling
does not weaken ownership. A mutable capture is rejected unless it is transferred
to one task or wrapped in a capability such as `atomic` or `mutex`.

Mitigation: **compile-time** through task-sendability and ownership checks;
**library-level** synchronization provides the permitted shared-state shape.
