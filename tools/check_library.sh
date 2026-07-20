#!/usr/bin/env bash
set -u

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"

compiler="$repository_root/target/debug/sev"
if [[ ! -x "$compiler" ]]; then
    cargo build -p severian-driver --bin sev || exit 1
fi

checked=0
passed=0
failed=0

while IFS= read -r source; do
    package="${source#library/}"
    package="${package%%/*}"
    status="$(sed -n 's/^status = "\([^"]*\)"/\1/p' "library/$package/Severian.toml")"

    if [[ "$status" == "interface-pending" ]]; then
        printf 'SKIP  %-16s interface pending\n' "$package"
        continue
    fi

    checked=$((checked + 1))
    if "$compiler" check "$source" && "$compiler" test "$source"; then
        printf 'PASS  %s\n' "$package"
        passed=$((passed + 1))
    else
        printf 'FAIL  %s\n' "$package"
        failed=$((failed + 1))
    fi
done < <(find library -mindepth 3 -maxdepth 3 -path '*/src/lib.sev' -print | sort)

printf '\n%d checked, %d passed, %d failed\n' "$checked" "$passed" "$failed"
((failed == 0))

