#!/usr/bin/env bash
set -euo pipefail

compiler="${SEV:-sev}"

if ! command -v "$compiler" >/dev/null 2>&1; then
    echo "error: '$compiler' was not found. Set SEV=/path/to/sev once the compiler driver exists." >&2
    exit 127
fi

find docs/examples -type f -name '*.sev' ! -name 'invalid.sev' | sort | while IFS= read -r example; do
    "$compiler" check "$example"
done

find docs/examples/bugs -type f -name 'invalid.sev' 2>/dev/null | sort | while IFS= read -r example; do
    expected="$(dirname "$example")/expected-error.txt"

    if [[ ! -f "$expected" ]]; then
        echo "error: missing expected diagnostic for '$example'" >&2
        exit 1
    fi

    output="$(mktemp)"
    if "$compiler" check "$example" >"$output" 2>&1; then
        echo "error: invalid fixture unexpectedly compiled: '$example'" >&2
        rm -f "$output"
        exit 1
    fi

    code="$(sed -n '1p' "$expected")"
    span="$(sed -n '2p' "$expected")"
    text="$(sed -n '3p' "$expected")"
    if ! grep -Fq "$code" "$output" || ! grep -Fq "$span" "$output" || ! grep -Fq "$text" "$output"; then
        echo "error: diagnostic for '$example' did not match '$expected'" >&2
        cat "$output" >&2
        rm -f "$output"
        exit 1
    fi
    rm -f "$output"
done
