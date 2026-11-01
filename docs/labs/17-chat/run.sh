#!/usr/bin/env bash
# From any directory: run.sh [PORT=7777] [HOST=127.0.0.1] [SECONDS=0]
# Example: run.sh 7777; in two other terminals: nc 127.0.0.1 7777
# /nick alice -> choose a name; /who -> list clients; /quit -> leave.
set -euo pipefail
lab_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "$lab_root/../../.." && pwd)"
export PYTHONPATH="$repository_root${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p -- "$lab_root/package.pkg"
if ! python3 -m py_compiler build "$lab_root" -o "$lab_root/package.pkg/chat" > "$lab_root/package.pkg/chat-build.log" 2>&1; then
    cat -- "$lab_root/package.pkg/chat-build.log" >&2
    exit 1
fi
exec "$lab_root/package.pkg/chat" "$@"
