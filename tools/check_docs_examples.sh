#!/usr/bin/env bash
set -euo pipefail

compiler="${SEV:-sev}"

if ! command -v "$compiler" >/dev/null 2>&1; then
    echo "error: '$compiler' was not found. Set SEV=/path/to/sev once the compiler driver exists." >&2
    exit 127
fi

find docs/examples -type f -name '*.sev' | sort | while IFS= read -r example; do
    "$compiler" check "$example"
done
