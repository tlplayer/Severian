#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd -- "$repo_root"
python3 -m py_compiler build library/system/io/linux_x86_64
python3 -m py_compiler build library/system/file/linux_x86_64
python3 -m py_compiler build docs/labs/14-file-dispatch/format
python3 -m py_compiler build docs/labs/14-file-dispatch/consumer
python3 -m py_compiler test docs/labs/14-file-dispatch/consumer "$@"
