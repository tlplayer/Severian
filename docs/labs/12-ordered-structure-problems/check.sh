#!/usr/bin/env bash
# Build/test the independent labs with the Python compiler.
# Optional arguments: --sanitize address --backtrace.
set -euo pipefail
labs_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd -- "$labs_dir/../.." && pwd)"
cd -- "$repo_root"
packages=(02-utils/utils 08-heap/heap 09-stack/stack 10-btree/btree 11-disjoint-sets/disjoint_sets 12-ordered-structure-problems/consumer)
for package in "${packages[@]}"; do
  python3 -m py_compiler build "$labs_dir/$package"
done
for package in "${packages[@]}"; do
  python3 -m py_compiler test "$labs_dir/$package" "$@"
done
