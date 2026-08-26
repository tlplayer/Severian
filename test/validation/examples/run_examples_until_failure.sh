ROOT="docs/examples/"

find "$ROOT" -type f -name '*.sev' -printf '%h\0' | sort -zu | while IFS= read -r -d '' dir; do
    echo "=== $dir ==="
    (
        cd "$dir" || exit 1
        sev test
        sev test --mutate
    )
done