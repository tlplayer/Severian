find docs/examples -type f -name '*.sev' -print0 | sort -z | while IFS= read -r -d '' f; do
    echo "=== $f ==="
    sev test "$f" || break
done