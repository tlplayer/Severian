#!/usr/bin/env bash
set -u

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"

compiler="$repository_root/target/debug/sev"
if [[ ! -x "$compiler" ]]; then
    cargo build -p severian-driver --bin sev || exit 1
fi

temporary_dir="$(mktemp -d /tmp/severian-library.XXXXXX)"
trap 'rm -rf -- "$temporary_dir"' EXIT

checked=0
passed=0
failed=0

while IFS= read -r source; do
    package_root="${source%/src/lib.sev}"
    package="$(sed -n 's/^name = "\([^"]*\)"/\1/p' "$package_root/Severian.toml" | head -n 1)"
    status="$(sed -n 's/^status = "\([^"]*\)"/\1/p' "$package_root/Severian.toml")"

    if [[ "$status" != "experimental" && "$status" != "stable" ]]; then
        printf 'SKIP  %-16s %s\n' "$package" "$status"
        continue
    fi

    checked=$((checked + 1))
    executable="$temporary_dir/$package"
    actual_stdout="$temporary_dir/$package.stdout"
    actual_stderr="$temporary_dir/$package.stderr"
    expected_stdout="$package_root/src/lib.stdout"
    expected_stderr="$package_root/src/lib.stderr"
    if "$compiler" check "$source" \
        && "$compiler" compile-tests "$source" -o "$executable" \
        && "$executable" >"$actual_stdout" 2>"$actual_stderr" \
        && cmp -s "$expected_stdout" "$actual_stdout" \
        && cmp -s "$expected_stderr" "$actual_stderr"; then
        printf 'PASS  %s\n' "$package"
        passed=$((passed + 1))
    else
        printf 'FAIL  %s\n' "$package"
        if [[ ! -f "$expected_stdout" || ! -f "$expected_stderr" ]]; then
            printf '      missing native output fixture\n'
        else
            diff -u "$expected_stdout" "$actual_stdout" || true
            diff -u "$expected_stderr" "$actual_stderr" || true
        fi
        failed=$((failed + 1))
    fi
done < <(find library -mindepth 3 -path '*/src/lib.sev' -print | sort)

printf '\n%d checked, %d passed, %d failed\n' "$checked" "$passed" "$failed"
((failed == 0))
