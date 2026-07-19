#!/usr/bin/env bash
set -uo pipefail

usage() {
    cat <<'EOF'
usage: tools/check_docs_examples.sh [--native]

Compile every docs/examples/**/*.sev file in lexical order and run its attached
Severian tests. Files named invalid.sev must fail with the diagnostic described
by the adjacent expected-error.txt file.

Options:
  --native  Compile and run files with main(), compare adjacent .stdout files,
            and publish verified executables under bin/examples/.

Set SEV=/path/to/sev to use a specific compiler. Without SEV, the script builds
and uses target/debug/sev from this workspace.
EOF
}

native=false
case "${1:-}" in
    "") ;;
    --native) native=true ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "$script_dir/.." && pwd)"
cd "$repository_root" || exit 1

if [[ -n "${SEV:-}" ]]; then
    compiler="$SEV"
else
    compiler="$repository_root/target/debug/sev"
    if ! cargo build --quiet -p severian-driver --bin sev; then
        echo "error: failed to build the Severian compiler" >&2
        exit 1
    fi
fi

if [[ "$compiler" == */* ]]; then
    if [[ ! -x "$compiler" ]]; then
        echo "error: compiler is not executable: $compiler" >&2
        exit 127
    fi
elif ! command -v "$compiler" >/dev/null 2>&1; then
    echo "error: compiler was not found: $compiler" >&2
    exit 127
fi

temporary_dir="$(mktemp -d /tmp/severian-examples.XXXXXX)"
trap 'rm -rf -- "$temporary_dir"' EXIT

mapfile -d '' examples < <(
    find docs/examples -type f -name '*.sev' -print0 | sort -z
)

total=${#examples[@]}
compile_passed=0
compile_failed=0
test_passed=0
test_failed=0
expected_error_passed=0
expected_error_failed=0
native_passed=0
native_failed=0
native_skipped=0
output_passed=0
output_failed=0

print_failure() {
    local output=$1
    sed 's/^/      /' "$output" >&2
}

for index in "${!examples[@]}"; do
    example=${examples[$index]}
    number=$((index + 1))
    printf '[%03d/%03d] %s\n' "$number" "$total" "$example"

    compile_output="$temporary_dir/compile-$number.txt"
    if [[ $(basename -- "$example") == invalid.sev ]]; then
        expected="$(dirname -- "$example")/expected-error.txt"
        if [[ ! -f "$expected" ]]; then
            echo "    EXPECTED ERROR FAIL (missing $expected)" >&2
            expected_error_failed=$((expected_error_failed + 1))
            continue
        fi

        if "$compiler" check "$example" >"$compile_output" 2>&1; then
            echo "    EXPECTED ERROR FAIL (file compiled successfully)" >&2
            expected_error_failed=$((expected_error_failed + 1))
            continue
        fi

        code=$(sed -n '1p' "$expected")
        span=$(sed -n '2p' "$expected")
        message=$(sed -n '3p' "$expected")
        if grep -Fq "$code" "$compile_output" \
            && grep -Fq "$span" "$compile_output" \
            && grep -Fq "$message" "$compile_output"; then
            echo "    EXPECTED ERROR PASS"
            expected_error_passed=$((expected_error_passed + 1))
        else
            echo "    EXPECTED ERROR FAIL (diagnostic mismatch)" >&2
            print_failure "$compile_output"
            expected_error_failed=$((expected_error_failed + 1))
        fi
        continue
    fi

    if ! "$compiler" check "$example" >"$compile_output" 2>&1; then
        echo "    COMPILE FAIL" >&2
        print_failure "$compile_output"
        compile_failed=$((compile_failed + 1))
        continue
    fi
    echo "    COMPILE PASS"
    compile_passed=$((compile_passed + 1))

    test_output="$temporary_dir/test-$number.txt"
    if "$compiler" test "$example" >"$test_output" 2>&1; then
        result=$(tail -n 1 "$test_output")
        echo "    TEST PASS ($result)"
        test_passed=$((test_passed + 1))
    else
        echo "    TEST FAIL" >&2
        print_failure "$test_output"
        test_failed=$((test_failed + 1))
    fi

    if [[ "$native" == true ]]; then
        if ! grep -Eq '^def[[:space:]]+main\(' "$example"; then
            echo "    NATIVE SKIP (no main function)"
            native_skipped=$((native_skipped + 1))
            continue
        fi

        relative_path=${example#docs/examples/}
        executable="bin/examples/${relative_path%.sev}"
        mkdir -p -- "$(dirname -- "$executable")"
        rm -f -- "$executable"
        temporary_executable="$temporary_dir/example-$number"
        native_output="$temporary_dir/native-$number.txt"
        if "$compiler" compile "$example" -o "$temporary_executable" >"$native_output" 2>&1; then
            echo "    NATIVE COMPILE PASS"
            native_passed=$((native_passed + 1))
        else
            echo "    NATIVE COMPILE FAIL" >&2
            print_failure "$native_output"
            native_failed=$((native_failed + 1))
            continue
        fi

        expected_output="${example%.sev}.stdout"
        if [[ ! -f "$expected_output" ]]; then
            echo "    OUTPUT FAIL (missing $expected_output)" >&2
            output_failed=$((output_failed + 1))
            continue
        fi

        actual_output="$temporary_dir/stdout-$number.txt"
        actual_error="$temporary_dir/stderr-$number.txt"
        timeout "${SEV_EXAMPLE_TIMEOUT:-5}" "$temporary_executable" \
            >"$actual_output" 2>"$actual_error"
        status=$?
        if (( status != 0 )); then
            echo "    OUTPUT FAIL (exit $status or timeout)" >&2
            print_failure "$actual_error"
            output_failed=$((output_failed + 1))
            continue
        fi
        if [[ -s "$actual_error" ]]; then
            echo "    OUTPUT FAIL (unexpected stderr)" >&2
            print_failure "$actual_error"
            output_failed=$((output_failed + 1))
            continue
        fi
        if ! cmp -s "$expected_output" "$actual_output"; then
            echo "    OUTPUT FAIL (stdout mismatch)" >&2
            diff -u "$expected_output" "$actual_output" | sed 's/^/      /' >&2 || true
            output_failed=$((output_failed + 1))
            continue
        fi
        mv -- "$temporary_executable" "$executable"
        echo "    OUTPUT PASS ($executable)"
        output_passed=$((output_passed + 1))
    fi
done

echo
echo "Example summary"
echo "  Files:                  $total"
echo "  Compile:                $compile_passed passed, $compile_failed failed"
echo "  Test commands:          $test_passed passed, $test_failed failed"
echo "  Expected diagnostics:   $expected_error_passed passed, $expected_error_failed failed"
if [[ "$native" == true ]]; then
    echo "  Native compilation:     $native_passed passed, $native_failed failed, $native_skipped skipped"
    echo "  Native output:          $output_passed passed, $output_failed failed"
fi

failures=$((compile_failed + test_failed + expected_error_failed + native_failed + output_failed))
if (( failures > 0 )); then
    exit 1
fi
