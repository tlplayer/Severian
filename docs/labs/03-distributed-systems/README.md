# MIT 6.5840 distributed-systems labs in Severian

This directory is an executable language-validation suite derived from
[`tlplayer/distributed_systems`](https://github.com/tlplayer/distributed_systems),
fetched at commit `d23e16e80c7bcc1103be34f77f823b4bc6cddc78`.

The exact Go input is preserved in [`go_source`](go_source). The upstream tree
contains complete infrastructure and MapReduce code, but most consensus and
replicated-service files are intentionally incomplete course assignments. The
Severian side therefore has two kinds of port:

- implemented Go helpers are translated directly;
- assignment skeletons are represented by small, deterministic implementations
  of the behavior described by their public API and tests.

These are native compiler/runtime tests. They do not use a source interpreter or
simulate Severian syntax in another frontend.

## Labs

| Directory | Covered behavior |
| --- | --- |
| `01_serialization` | RPC/persistence schema checks, detached wire values, default receivers |
| `02_rpc_network` | endpoints, connections, server dispatch, drops, deletion, counters |
| `03_map_reduce` | map/reduce scheduling, retries, word count, index generation |
| `04_key_value_lock` | versioned writes, ambiguous retries, optimistic distributed lock |
| `05_raft` | elections, log matching, majority commit, partitions, snapshots, restart |
| `06_replicated_state_machine` | leader routing, command application, duplicate suppression |
| `07_shard_configuration` | deterministic key sharding and balanced join/leave/move |
| `08_sharded_key_value` | ownership checks and shard migration between replica groups |
| `09_test_harness` | persistent-state copies and model-based history validation |

Run every file with:

```sh
./run_labs.sh
```

The runner performs `sev check`, native `sev test`, and native `sev run` for
each source. Build artifacts use each standalone lab's ignored `target`
directory.

[`PORTING_MANIFEST.md`](PORTING_MANIFEST.md) maps every upstream Go file to its
native Severian coverage and records the translation boundary for Go-specific
harness mechanics.

[`VALIDATION.md`](VALIDATION.md) records the upstream baseline, native test
counts, full-workspace result, and compiler defects found by the ports.

## Scope notes

The upstream `main` directory imports four packages that are not present in the
repository (`diskv`, `lockservice`, `pbservice`, and `viewservice`). Those
launchers are retained as provenance, but there is no missing Go implementation
to translate. The corresponding fault-tolerance concepts are exercised by the
Raft, replicated-state-machine, and sharded-key-value labs here.

The ports intentionally use deterministic fault schedules. Random timing is a
poor executable specification; deterministic loss, partition, and retry cases
exercise the same state transitions and produce reproducible compiler failures.
