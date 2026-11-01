#!/usr/bin/env bash
# Run adjacent network and chat tests; all socket integration uses loopback port 0.
set -euo pipefail
lab_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "$lab_root/../../.." && pwd)"
cd -- "$repository_root"
python3 -m py_compiler test library/network/linux_x86_64 --timeout 10 "$@"
python3 -m py_compiler test docs/labs/17-chat --timeout 10 "$@"
