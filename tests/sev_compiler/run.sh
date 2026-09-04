#!/usr/bin/env bash
set -u

repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
severian_bin=${SEVERIAN_BIN:-"$repository_root/target/debug/sev"}
timeout_seconds=${SEVERIAN_TEST_TIMEOUT_SECONDS:-60}
parallel_jobs=${SEVERIAN_TEST_JOBS:-4}
show_output=${SEVERIAN_TEST_OUTPUT:-0}

if [[ ! -x "$severian_bin" ]]; then
    echo "FAILED: Severian compiler is not executable: $severian_bin" >&2
    exit 1
fi

declare -a roots
if (( $# == 0 )); then
    roots=("$repository_root/sev_compiler")
else
    roots=()
    for requested in "$@"; do
        if [[ "$requested" = /* ]]; then
            roots+=("$requested")
        else
            roots+=("$repository_root/$requested")
        fi
    done
fi

temporary_root=$(mktemp -d)
trap 'rm -rf -- "$temporary_root"' EXIT

declare -A selected_targets=()
for root in "${roots[@]}"; do
    if [[ -f "$root" ]]; then
        selected_targets["$root"]=1
        continue
    fi
    if [[ -f "$root/package.toml" ]]; then
        selected_targets["$root"]=1
    else
        while IFS= read -r -d '' manifest; do
            selected_targets["${manifest%/package.toml}"]=1
        done < <(find "$root" -type f -name package.toml -print0)
    fi
    while IFS= read -r test_source; do
        selected_targets["$test_source"]=1
    done < <(rg -l --glob '*.sev' '^[[:space:]]*test([[:space:]]|:)' "$root" || true)
done
mapfile -t targets < <(printf '%s\n' "${!selected_targets[@]}" | sort)
total=${#targets[@]}
if (( total == 0 )); then
    echo "FAILED: no Severian packages or test-bearing .sev files found" >&2
    exit 1
fi

completed=0
failed=0
timed_out=0

run_one() {
    local index=$1
    local file=$2
    local log="$temporary_root/$index.log"
    local status_file="$temporary_root/$index.status"

    local status
    if (( show_output != 0 )); then
        timeout --verbose "${timeout_seconds}s" "$severian_bin" test "$file" 2>&1 | tee "$log"
        status=${PIPESTATUS[0]}
    else
        timeout --verbose "${timeout_seconds}s" "$severian_bin" test "$file" >"$log" 2>&1
        status=$?
    fi
    printf '%s\n' "$status" >"$status_file"
}

echo "sev compiler source tests: $total package/scenario targets, $parallel_jobs jobs, ${timeout_seconds}s per target"

batch_start=0
while (( batch_start < total )); do
    batch_end=$((batch_start + parallel_jobs))
    if (( batch_end > total )); then
        batch_end=$total
    fi

    index=$batch_start
    while (( index < batch_end )); do
        relative=${targets[$index]#"$repository_root/"}
        echo "starting $((index + 1))/$total: $relative"
        run_one "$index" "${targets[$index]}" &
        index=$((index + 1))
    done
    wait

    index=$batch_start
    while (( index < batch_end )); do
        status=$(<"$temporary_root/$index.status")
        if (( status != 0 )); then
            failed=$((failed + 1))
            if (( status == 124 )); then
                timed_out=$((timed_out + 1))
            fi
        fi
        index=$((index + 1))
    done

    completed=$batch_end
    echo "checked $completed/$total; failures $failed"
    batch_start=$batch_end
done

if (( failed != 0 )); then
    echo
    echo "failure details"
    index=0
    while (( index < total )); do
        status=$(<"$temporary_root/$index.status")
        if (( status != 0 )); then
            relative=${targets[$index]#"$repository_root/"}
            if (( status == 124 )); then
                echo "FAILED (timeout after ${timeout_seconds}s): $relative"
                echo "  investigate frontend cost with:"
                echo "    /usr/bin/time $severian_bin build --emit hir -o /tmp/slow.hir $relative"
                echo "  narrow imports or split scenarios so each test proves one semantic layer."
            else
                echo "FAILED (exit $status): $relative"
            fi
            sed 's/^/  /' "$temporary_root/$index.log"
        fi
        index=$((index + 1))
    done
fi

passed=$((total - failed))
echo
echo "test result: $passed passed; $failed failed; $timed_out timed out"
(( failed == 0 ))
