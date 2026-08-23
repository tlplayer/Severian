fail=0
timeout_seconds=10

while IFS= read -r -d '' f; do
    echo "=== $f ==="

    timeout "${timeout_seconds}s" sev test "$f"
    status=$?

    if [ "$status" -eq 124 ]; then
        echo "FAILED (timeout after ${timeout_seconds}s): $f"
        fail=1
    elif [ "$status" -ne 0 ]; then
        fail=1
    fi
done < <(find docs/examples -type f -name '*.sev' -print0 | sort -z)

exit "$fail"