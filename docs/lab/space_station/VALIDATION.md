# Validation record

The laboratory has four acceptance layers:

| Layer | Native evidence |
| --- | --- |
| simulation | 96 stable tiles, one door, one fire, one breach |
| concurrency | three workers process four typed jobs and replay ids `1..4` with values `[82, 86, 88, 2]` |
| graphics | an 840×380 SVG frame contains tiles, crew, hazards, telemetry, and alerts |
| networking | two concurrent observers receive byte-identical snapshots through independent native TCP loopback sockets |

Run the deterministic tests with `sev test main.sev`. Run the application and
compare its output to `main.stdout` for end-to-end acceptance. Native networking
requires an execution environment that permits local IPv4 sockets; a restricted
sandbox produces explicit `network-error` observer records instead of aborting.

During implementation the lab exposed these compiler/runtime behaviors:

1. `await` expressions inside list literals currently generate an empty MLIR
   collection operand. Appending each awaited value in a separate statement is
   the valid lowering path.
2. Integration tests are parsed but skipped by the native test compiler, and the
   CLI currently has no `--integration` selector.
3. Socket denial must remain a typed network failure. Consuming each `Result`
   inside its observer task and sending a plain snapshot/error record keeps task
   ownership explicit and prevents the coordinator from conflating environment
   policy with a station failure.
