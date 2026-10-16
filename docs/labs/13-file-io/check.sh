#!/usr/bin/env bash
# Python compiler only. MLIR artifacts remain under each package's build output.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd -- "$repo_root"
python3 -m py_compiler build library/system/io/linux_x86_64
python3 -m py_compiler build docs/labs/13-file-io/consumer
python3 -m py_compiler test library/system/io/linux_x86_64 "$@"
python3 -m py_compiler test docs/labs/13-file-io/consumer "$@"
