#!/usr/bin/env bash
# Run from any directory. Optional test arguments: --sanitize address --backtrace.
# Production builds must precede consumer imports; test artifacts are separate.
set -euo pipefail
labs_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd -- "$labs_dir/../.." && pwd)"
cd -- "$repo_root"
packages=(
  04-linked-list/linked_list
  05-binary-search-tree/bst
  06-hash-dictionary/dictionary
)
for package in "${packages[@]}"; do
  python3 -m py_compiler build "$labs_dir/$package"
done
for package in "${packages[@]}" 07-data-structure-problems/consumer; do
  python3 -m py_compiler test "$labs_dir/$package" "$@"
done
