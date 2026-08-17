# Porting manifest

This manifest maps every upstream Go source file to its executable Severian
coverage. Course test files map to native `test` blocks; they are specifications,
not production modules to copy line-for-line.

| Go source | Severian coverage |
| --- | --- |
| `labgob/labgob.go`, `labgob/test_test.go` | `01_serialization`: public-field schema validation, default receivers, detached round trips |
| `labrpc/labrpc.go`, `labrpc/test_test.go` | `02_rpc_network`: endpoint enable/connect, dispatch, deterministic loss, server deletion, request/byte counters |
| `mr/*.go`, `mrapps/*.go` | `03_map_reduce`: task phases, leases/retry, word count, inverted index, deterministic application output |
| `kvsrv1/*.go`, `kvsrv1/rpc/*.go`, `kvsrv1/lock/*.go` | `04_key_value_lock`: Get/Put versions, all public error cases, ambiguous retry, lock exclusion/release |
| `kvtest1/*.go`, `models1/kv.go` | `09_test_harness`: detached persistence and model validation; KV semantics also run in `04_key_value_lock` |
| `raftapi/raftapi.go`, `raft1/*.go` | `05_raft`: election, voting freshness, AppendEntries matching/repair, majority commit, partitions, snapshot/restart |
| `kvraft1/rsm/*.go`, `kvraft1/client.go`, `kvraft1/server.go`, `kvraft1/test.go`, `kvraft1/kvraft_test.go` | `06_replicated_state_machine`: leader routing, replicated application, duplicate suppression, leader change |
| `shardkv1/shardcfg/*.go`, `shardkv1/shardctrler/shardctrler.go` | `07_shard_configuration`: deterministic key placement and numbered join/leave/move configurations |
| `shardkv1/shardgrp/*.go`, `shardkv1/shardgrp/shardrpc/*.go`, `shardkv1/client.go`, `shardkv1/test.go`, `shardkv1/shardkv_test.go` | `08_sharded_key_value`: group ownership, wrong-group handling, versioned reconfiguration and shard transfer |
| `tester1/*.go`, `tester1/tester_test.go` | `02_rpc_network`, `05_raft`, and `09_test_harness`: partitions, server lifetime, persistence copies, operation history checks |
| `main/mrcoordinator.go`, `main/mrsequential.go`, `main/mrworker.go` | Executable MapReduce behavior in `03_map_reduce` |
| `main/diskvd.go`, `main/lockc.go`, `main/lockd.go`, `main/pbc.go`, `main/pbd.go`, `main/viewd.go` | Launcher intent is covered by labs 04–08; their imported Go packages are absent upstream, so there is no implementation body to translate |

## Translation boundary

The Go repository is a course workspace, not a completed reference solution.
Files containing `Your code here` are retained exactly under `go_source`; their
Severian counterparts implement the API behavior exercised by the upstream
tests. Go-only mechanics such as reflection, `plugin.Open`, Unix-domain RPC,
goroutine scheduling, and Porcupine visualization are represented by typed
state transitions and deterministic fault schedules. This keeps the labs useful
as repeatable tests of Severian rather than tests of a host-language adapter.

