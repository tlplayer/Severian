#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
compiler="${SEVERIAN_COMPILER:-${repository_root}/target/debug/sev}"
laboratory_root="${repository_root}/docs/lab/distributed_systems"

if [[ ! -x "${compiler}" ]]; then
    echo "Severian compiler is not executable: ${compiler}" >&2
    exit 1
fi

laboratories=(
    01_serialization
    02_rpc_network
    03_map_reduce
    04_key_value_lock
    05_raft
    06_replicated_state_machine
    07_shard_configuration
    08_sharded_key_value
    09_test_harness
)

for laboratory in "${laboratories[@]}"; do
    source_path="${laboratory_root}/${laboratory}/main.sev"
    echo "==> ${laboratory}: check"
    "${compiler}" check "${source_path}"
    echo "==> ${laboratory}: test"
    "${compiler}" test "${source_path}"
    echo "==> ${laboratory}: run"
    "${compiler}" run "${source_path}"
done

echo "All ${#laboratories[@]} distributed-systems labs passed."

